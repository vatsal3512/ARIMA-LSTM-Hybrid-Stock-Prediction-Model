"""ARIMA-LSTM hybrid with a stacked meta-learner, evaluated by walk-forward validation.

Per fold (each fold = one 30-trading-day test window):
  base rows  [0, meta_start)        -> train the LSTM (early stopping on its last 15%)
                                       and fit Auto-ARIMA on its in-sample residuals
  meta rows  [meta_start, t0)       -> out-of-sample LSTM + ARIMA outputs train the meta-learner
  test rows  [t0, t0 + TEST_DAYS)   -> final predictions, never seen by any component

At the close of day t the pipeline predicts the log return from t to t+1.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from statsmodels.graphics.tsaplots import plot_acf

import evaluation as ev
from features import FEATURES, MARKET_FEATURES, add_market_features, build_features, load_ohlcv
from models import META_FEATURES, LSTMForecaster, MetaLearner, ResidualARIMA, set_seed

load_dotenv()


def env(key, default, cast=str):
    value = os.getenv(key)
    return cast(value) if value not in (None, "") else default


DATA_PATH = env("FILE_ADDRESS", "AAPL_historical_data.csv")
MARKET_PATH = env("MARKET_ADDRESS", "market_data.csv")
META_MODEL = env("META_MODEL", "logreg")
OUT_DIR = env("OUTPUT_ADDRESS", "./")
N_LAGS = env("NN_LAGS", 60, int)
HIDDEN = env("NUMBER_NODES", 50, int)
EPOCHS = env("EPOCHS", 50, int)
LR = env("LEARNING_RATE", 0.001, float)
BATCH = env("BATCH_SIZE", 32, int)
TEST_DAYS = env("Prediction_days", 30, int)
ACF_LAGS = env("LAG", 30, int)
FOLDS = env("WF_FOLDS", 12, int)
META_DAYS = env("META_DAYS", 250, int)
PATIENCE = env("PATIENCE", 8, int)
DROPOUT = env("DROPOUT", 0.2, float)
SEED = env("SEED", 42, int)
COST_BPS = env("COST_BPS", 0.0, float)
RESULTS_DIR = os.path.join(OUT_DIR, "results")


def run_fold(data, t0):
    X, y = data[FEATURES].values, data["target"].values
    meta_start, end = t0 - META_DAYS, t0 + TEST_DAYS

    set_seed(SEED)
    lstm = LSTMForecaster(N_LAGS, HIDDEN, DROPOUT, LR, EPOCHS, BATCH, PATIENCE)
    lstm.fit(X[:meta_start], y[:meta_start])
    pred = lstm.predict(X[:end])  # each prediction uses features up to its own row only

    base = np.arange(N_LAGS - 1, meta_start)
    residuals = y[base] - pred[base]
    arima = ResidualARIMA().fit(residuals)

    correction = np.full(end, np.nan)
    for i in range(meta_start, end):
        correction[i] = arima.forecast()       # decided at the close of day i
        arima.update(y[i] - pred[i])           # known at the close of day i+1

    frame = data.iloc[:end].copy()
    frame["lstm"], frame["arima"] = pred, correction
    frame["hybrid"] = frame["lstm"] + frame["arima"]

    meta_rows, test_rows = frame.iloc[meta_start:t0], frame.iloc[t0:end]
    cols = META_FEATURES + [c for c in MARKET_FEATURES if c in frame.columns]
    meta = MetaLearner(META_MODEL, SEED).fit(meta_rows[cols].values,
                                             (meta_rows["target"] > 0).astype(int).values)
    p_up = meta.p_up(test_rows[cols].values)

    out = test_rows[["close", "target", "ret_1", "lstm", "arima", "hybrid"]].copy()
    out["p_up"] = p_up
    out["direction"] = meta.direction(test_rows[cols].values)      # meta-learner decides direction
    out["final"] = out["direction"] * out["hybrid"].abs()        # hybrid decides magnitude
    info = {"arima_order": arima.order, "epochs": len(lstm.history["val"]), "gated": meta.gated,
            "train_rows": meta_start, "residuals": residuals, "history": lstm.history}
    return out, info


def fold_metrics(f):
    up = f["direction"].values == 1
    strat = ev.equity_curve(f["target"].values, up, cost_bps=COST_BPS)
    hold = ev.equity_curve(f["target"].values, np.ones(len(f), bool))
    return {
        "start": f.index[0].date(), "end": f.index[-1].date(),
        "acc_meta": ev.directional_accuracy(f["target"].values, f["direction"].values),
        "acc_hybrid_sign": ev.directional_accuracy(f["target"].values, np.sign(f["hybrid"].values)),
        "acc_always_up": ev.directional_accuracy(f["target"].values, np.ones(len(f))),
        "mdd_strategy": ev.max_drawdown(strat), "mdd_buy_hold": ev.max_drawdown(hold),
        "return_strategy": (strat[-1] / 10_000 - 1) * 100, "return_buy_hold": (hold[-1] / 10_000 - 1) * 100,
    }


def make_dashboard(all_preds, fm, last, last_info, next_dates, path):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    a = ax[0, 0]
    actual_next = last["close"] * np.exp(last["target"])
    pred_next = last["close"] * np.exp(last["final"])
    a.plot(next_dates, actual_next, "o-", color="#1f3a5f", lw=1.8, ms=3, label="Actual close")
    a.plot(next_dates, pred_next, "s-", color="#e09f3e", lw=1.4, ms=3, label="Hybrid + meta-learner")
    a.plot(next_dates, last["close"].values, "--", color="#8a94a6", lw=1, label="Naive: tomorrow = today")
    a.set_title("Latest 30-day test window"); a.set_ylabel("USD"); a.legend(fontsize=8)
    a.tick_params(axis="x", rotation=30)

    a = ax[0, 1]
    x = np.arange(len(fm))
    a.bar(x - 0.2, fm["acc_meta"], 0.4, color="#1b998b", label="Meta-learner")
    a.bar(x + 0.2, fm["acc_always_up"], 0.4, color="#c9d0db", label="Always up")
    a.axhline(50, color="#c0392b", ls="--", lw=1)
    a.set_xticks(x); a.set_xticklabels([str(s)[:7] for s in fm["start"]], rotation=45, fontsize=7)
    a.set_ylabel("Directional accuracy (%)"); a.set_title("Accuracy per 30-day walk-forward window")
    a.legend(fontsize=8)

    a = ax[1, 0]
    up = all_preds["direction"].values == 1
    a.plot(all_preds.index, ev.equity_curve(all_preds["target"].values, up, cost_bps=COST_BPS),
           color="#1b998b", lw=1.8, label="Strategy (long when meta-learner says up)")
    a.plot(all_preds.index, ev.equity_curve(all_preds["target"].values, np.ones(len(up), bool)),
           color="#8a94a6", lw=1.5, label="Buy and hold")
    a.set_title("Walk-forward equity curve ($10,000 start)"); a.legend(fontsize=8)

    plot_acf(last_info["residuals"], lags=ACF_LAGS, ax=ax[1, 1])
    ax[1, 1].set_title(f"ACF of LSTM residuals (Auto-ARIMA order {last_info['arima_order']})")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    t_start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    prices = load_ohlcv(DATA_PATH)
    data = build_features(prices)
    if os.path.exists(MARKET_PATH):
        data = add_market_features(data, MARKET_PATH)
    n = len(data)
    starts = [n - TEST_DAYS * k for k in range(FOLDS, 0, -1)]
    print(f"Rows after feature engineering: {n}  ({data.index[0].date()} -> {data.index[-1].date()})")

    preds, metrics, infos = [], [], []
    for k, t0 in enumerate(starts, 1):
        f, info = run_fold(data, t0)
        f["fold"] = k
        preds.append(f); infos.append(info); metrics.append(fold_metrics(f))
        m = metrics[-1]
        print(f"Fold {k:2d} {m['start']} -> {m['end']}: acc {m['acc_meta']:.1f}%  "
              f"(always-up {m['acc_always_up']:.1f}%)  MDD {m['mdd_strategy']:.2f}%  "
              f"ARIMA{info['arima_order']}  epochs {info['epochs']}  meta {'gated->majority' if info['gated'] else 'active'}")

    all_preds = pd.concat(preds)
    fm = pd.DataFrame(metrics)
    actual, direction = all_preds["target"].values, all_preds["direction"].values
    nonzero = np.sign(actual) != 0
    correct = int(np.sum(np.sign(actual)[nonzero] == direction[nonzero]))
    total = int(nonzero.sum())
    up = direction == 1
    strat = ev.equity_curve(actual, up, cost_bps=COST_BPS)
    hold = ev.equity_curve(actual, np.ones(len(actual), bool))
    strat_daily = np.diff(np.concatenate([[10_000], strat])) / np.concatenate([[10_000], strat[:-1]])
    hold_daily = np.expm1(actual)

    next_close = all_preds["close"] * np.exp(actual)
    pred_close = all_preds["close"] * np.exp(all_preds["final"])
    last, last_info = preds[-1], infos[-1]
    last_m = metrics[-1]
    pos = prices.index.get_indexer(last.index)
    next_dates = prices.index[pos + 1]

    summary = {
        "walk_forward_days": total,
        "acc_meta": correct / total * 100,
        "binomial_p": ev.binomial_p(correct, total),
        "acc_hybrid_sign": ev.directional_accuracy(actual, np.sign(all_preds["hybrid"].values)),
        "acc_always_up": ev.directional_accuracy(actual, np.ones(len(actual))),
        "acc_momentum": ev.directional_accuracy(actual, np.sign(all_preds["ret_1"].values)),
        "mdd_strategy": ev.max_drawdown(strat), "mdd_buy_hold": ev.max_drawdown(hold),
        "sharpe_strategy": ev.sharpe(strat_daily), "sharpe_buy_hold": ev.sharpe(hold_daily),
        "return_strategy": (strat[-1] / 10_000 - 1) * 100, "return_buy_hold": (hold[-1] / 10_000 - 1) * 100,
        "rmse_ret_model": ev.rmse(actual, all_preds["final"]) * 100,
        "rmse_ret_zero": ev.rmse(actual, np.zeros(len(actual))) * 100,
        "rmse_price_model": ev.rmse(next_close, pred_close),
        "rmse_price_naive": ev.rmse(next_close, all_preds["close"]),
    }

    all_preds.to_csv(os.path.join(RESULTS_DIR, "walk_forward_predictions.csv"))
    fm.to_csv(os.path.join(RESULTS_DIR, "fold_metrics.csv"), index=False)
    make_dashboard(all_preds, fm, last, last_info, next_dates, os.path.join(OUT_DIR, "Final_Results_Dashboard.jpg"))

    s = summary
    lines = [
        "ARIMA-LSTM HYBRID WITH STACKED META-LEARNER - WALK-FORWARD RESULTS",
        "=" * 68,
        f"Data: {DATA_PATH}  ({data.index[0].date()} -> {data.index[-1].date()}, {n} rows)",
        f"Target: next-day log return | Features: {', '.join(FEATURES)}",
        f"Walk-forward: {FOLDS} folds x {TEST_DAYS} trading days | meta window {META_DAYS} days | seed {SEED}",
        "",
        f"LATEST 30-DAY WINDOW ({last_m['start']} -> {last_m['end']})",
        f"  Directional accuracy (meta-learner): {last_m['acc_meta']:.2f}%   (always-up baseline {last_m['acc_always_up']:.2f}%)",
        f"  Maximum drawdown (strategy):         {last_m['mdd_strategy']:.2f}%   (buy-and-hold {last_m['mdd_buy_hold']:.2f}%)",
        f"  Return (strategy):                   {last_m['return_strategy']:.2f}%   (buy-and-hold {last_m['return_buy_hold']:.2f}%)",
        "",
        f"ALL {FOLDS} WINDOWS POOLED ({s['walk_forward_days']} out-of-sample days)",
        f"  Directional accuracy, meta-learner:  {s['acc_meta']:.2f}%   (binomial p vs coin = {s['binomial_p']:.3f})",
        f"  Directional accuracy, hybrid sign:   {s['acc_hybrid_sign']:.2f}%",
        f"  Baseline, always up:                 {s['acc_always_up']:.2f}%",
        f"  Baseline, yesterday's direction:     {s['acc_momentum']:.2f}%",
        f"  Maximum drawdown:  strategy {s['mdd_strategy']:.2f}%  | buy-and-hold {s['mdd_buy_hold']:.2f}%",
        f"  Total return:      strategy {s['return_strategy']:.2f}%  | buy-and-hold {s['return_buy_hold']:.2f}%",
        f"  Sharpe (annual):   strategy {s['sharpe_strategy']:.2f}   | buy-and-hold {s['sharpe_buy_hold']:.2f}",
        f"  RMSE next-day return (%): model {s['rmse_ret_model']:.3f} | predict-zero {s['rmse_ret_zero']:.3f}",
        f"  RMSE next-day price ($):  model {s['rmse_price_model']:.3f} | naive {s['rmse_price_naive']:.3f}",
        "",
        "PER-FOLD RESULTS",
        fm.round(2).to_string(index=False),
        "",
        f"Auto-ARIMA orders on LSTM residuals: {[i['arima_order'] for i in infos]}",
        f"Meta-learner ({META_MODEL}) gated to majority direction in {sum(i['gated'] for i in infos)} of {FOLDS} folds",
        f"Run time: {time.time() - t_start:.1f} s",
    ]
    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "output.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
