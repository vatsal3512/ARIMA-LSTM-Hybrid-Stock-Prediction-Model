"""Evaluation metrics. All inputs are aligned per decision day t:
actual_ret[t] = realised next-day log return, direction[t] = +1 (long) / -1 (flat or short call).
"""
import numpy as np
from scipy.stats import binomtest


def directional_accuracy(actual_ret, predicted_dir):
    actual_dir = np.sign(actual_ret)
    mask = actual_dir != 0
    return float(np.mean(actual_dir[mask] == np.sign(predicted_dir)[mask]) * 100)


def equity_curve(actual_ret, long_mask, start=10_000.0, cost_bps=0.0):
    """Long-only: hold the stock on days the model predicts up, cash otherwise."""
    daily = np.where(long_mask, np.expm1(actual_ret), 0.0)
    trades = np.abs(np.diff(np.concatenate([[0], long_mask.astype(int)])))
    daily = daily - trades * cost_bps / 10_000
    return start * np.cumprod(1 + daily)


def max_drawdown(equity, start=10_000.0):
    equity = np.concatenate([[start], equity])  # include the starting capital as the first peak
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak) * 100)


def sharpe(daily_simple_returns):
    sd = np.std(daily_simple_returns)
    return float(np.sqrt(252) * np.mean(daily_simple_returns) / sd) if sd > 0 else 0.0


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def binomial_p(correct, total):
    """One-sided p-value: probability a coin (p = 0.5) does at least this well."""
    return float(binomtest(int(correct), int(total), 0.5, alternative="greater").pvalue)
