"""The three components of the hybrid:

1. LSTMForecaster  - PyTorch LSTM that predicts the next-day log return (magnitude).
2. ResidualARIMA   - Auto-ARIMA fitted to the LSTM's residuals, rolled forward one step
                     at a time to correct any linear structure the LSTM missed.
3. MetaLearner     - Classifier stacked on top of the LSTM + ARIMA ensemble: it learns P(up)
                     from their out-of-sample outputs plus technical and market context, and
                     decides the final direction (with a validation gate, see the class).
"""
import random

import numpy as np
import torch
import torch.nn as nn
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

TARGET_SCALE = 100.0  # train on percent returns: better-conditioned than raw log returns


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_windows(X, n):
    """Window ending at row i (rows i-n+1 .. i) for every i >= n-1."""
    return np.stack([X[i - n + 1:i + 1] for i in range(n - 1, len(X))])


class LSTMNet(nn.Module):
    def __init__(self, n_features, hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):            # x: (batch, n, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # last time step -> (batch, 1)


class LSTMForecaster:
    def __init__(self, n, hidden=50, dropout=0.2, lr=1e-3, epochs=50, batch_size=32,
                 patience=8, val_frac=0.15, weight_decay=1e-4):
        self.n, self.hidden, self.dropout = n, hidden, dropout
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.patience, self.val_frac, self.weight_decay = patience, val_frac, weight_decay
        self.history = {"train": [], "val": []}

    def fit(self, X, y):
        """X: (rows, features) raw features; y: (rows,) next-day log returns."""
        self.scaler = StandardScaler().fit(X)  # fitted on training rows only
        W = make_windows(self.scaler.transform(X), self.n)
        t = y[self.n - 1:] * TARGET_SCALE

        split = int(len(W) * (1 - self.val_frac))  # chronological validation split
        Xtr, ytr = torch.tensor(W[:split], dtype=torch.float32), torch.tensor(t[:split], dtype=torch.float32)
        Xva, yva = torch.tensor(W[split:], dtype=torch.float32), torch.tensor(t[split:], dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xtr, ytr.view(-1, 1)), batch_size=self.batch_size, shuffle=True)

        self.model = LSTMNet(X.shape[1], self.hidden, self.dropout)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        best, best_state, bad = np.inf, None, 0

        for _ in range(self.epochs):
            self.model.train()
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(xb)
            self.model.eval()
            with torch.no_grad():
                val = loss_fn(self.model(Xva), yva.view(-1, 1)).item()
            self.history["train"].append(total / len(Xtr))
            self.history["val"].append(val)
            if val < best - 1e-6:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, X):
        """Prediction for every row of X (NaN for the first n-1 rows, which lack history)."""
        W = make_windows(self.scaler.transform(X), self.n)
        with torch.no_grad():
            p = self.model(torch.tensor(W, dtype=torch.float32)).numpy().ravel() / TARGET_SCALE
        return np.concatenate([np.full(self.n - 1, np.nan), p])


class ResidualARIMA:
    """Auto-ARIMA on LSTM residuals, forecast one step ahead and updated as each day closes."""

    def fit(self, residuals):
        self.model = auto_arima(residuals * TARGET_SCALE, seasonal=False, stepwise=True,
                                suppress_warnings=True, error_action="ignore")
        return self

    @property
    def order(self):
        return self.model.order

    def forecast(self):
        return float(np.asarray(self.model.predict(n_periods=1))[0]) / TARGET_SCALE

    def update(self, residual):
        self.model.update([residual * TARGET_SCALE])


META_FEATURES = ["lstm", "arima", "hybrid", "rsi", "macd_norm", "sma10_gap", "sma50_gap",
                 "vol_20", "ret_1", "ret_5"]


def make_classifier(kind, seed):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_leaf=20,
                                      random_state=seed, n_jobs=-1)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))


class MetaLearner:
    """Stacked meta-learner on top of the LSTM + ARIMA ensemble.

    1. Learns P(next-day return > 0) from out-of-sample base-model outputs plus context.
    2. Chooses its 'up' threshold on an inner chronological split of the meta window.
    3. Gate: if, on that inner split, it does not beat the majority-class rule, it falls back
       to the majority direction (no validated edge -> follow the base rate).
    Steps 2-3 use the meta window only, never the test window.
    """

    THRESHOLDS = (0.50, 0.45, 0.40, 0.35)

    def __init__(self, kind="logreg", seed=42, inner_frac=0.7):
        self.kind, self.seed, self.inner_frac = kind, seed, inner_frac

    def _fit_with_threshold(self, M, y):
        cut = int(len(M) * self.inner_frac)
        inner = make_classifier(self.kind, self.seed).fit(M[:cut], y[:cut])
        p = inner.predict_proba(M[cut:])[:, 1]
        best_t, best_acc = 0.5, -1.0
        for t in self.THRESHOLDS:
            acc = np.mean((p >= t).astype(int) == y[cut:])
            if acc > best_acc + 1e-9:
                best_t, best_acc = t, acc
        return make_classifier(self.kind, self.seed).fit(M, y), best_t

    def fit(self, M, y_up):
        cut = int(len(M) * self.inner_frac)
        inner_model, t = self._fit_with_threshold(M[:cut], y_up[:cut])
        model_acc = np.mean((inner_model.predict_proba(M[cut:])[:, 1] >= t).astype(int) == y_up[cut:])
        prior_acc = np.mean(y_up[cut:] == int(y_up[:cut].mean() >= 0.5))
        self.gated = model_acc <= prior_acc
        self.majority = int(y_up.mean() >= 0.5)
        self.inner_acc, self.inner_prior_acc = float(model_acc), float(prior_acc)
        if not self.gated:
            self.model, self.threshold = self._fit_with_threshold(M, y_up)
        return self

    def direction(self, M):
        if self.gated:
            return np.full(len(M), 1 if self.majority else -1)
        return np.where(self.model.predict_proba(M)[:, 1] >= self.threshold, 1, -1)

    def p_up(self, M):
        if self.gated:
            return np.full(len(M), float(self.majority))
        return self.model.predict_proba(M)[:, 1]
