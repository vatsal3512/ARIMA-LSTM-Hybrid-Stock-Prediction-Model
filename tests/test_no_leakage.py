"""Guards against look-ahead bias. Run with:  python -m pytest tests"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import FEATURES, build_features, load_ohlcv  # noqa: E402
from models import make_windows  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AAPL_historical_data.csv")


def test_features_do_not_use_future_rows():
    prices = load_ohlcv(DATA)
    cut = len(prices) - 100
    full = build_features(prices)
    truncated = build_features(prices.iloc[:cut])
    common = truncated.index[:-1]  # last truncated row loses its target, by design
    np.testing.assert_allclose(full.loc[common, FEATURES].values,
                               truncated.loc[common, FEATURES].values, rtol=1e-10)


def test_target_is_next_day_log_return():
    prices = load_ohlcv(DATA)
    f = build_features(prices)
    t = f.index[100]
    nxt = prices.index[prices.index.get_loc(t) + 1]
    expected = np.log(prices.loc[nxt, "Close"] / prices.loc[t, "Close"])
    assert abs(f.loc[t, "target"] - expected) < 1e-12


def test_window_ends_at_its_own_row():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    W = make_windows(X, 3)
    assert W.shape == (8, 3, 1)
    assert W[0, -1, 0] == 2 and W[-1, -1, 0] == 9  # window i ends at row i, never beyond
