"""Data loading and stationary feature engineering.

Every feature at row t uses only information available at the close of day t.
The target at row t is the NEXT day's log return, ln(C[t+1] / C[t]).
"""
import numpy as np
import pandas as pd

FEATURES = [
    "ret_1",      # today's log return
    "ret_5",      # 5-day log return (weekly momentum)
    "oc",         # intraday move: ln(Close / Open)
    "hl",         # daily range: ln(High / Low)
    "vol_z",      # volume z-score vs. its 20-day history
    "sma10_gap",  # Close / SMA-10 - 1
    "sma50_gap",  # Close / SMA-50 - 1
    "rsi",        # Wilder RSI-14, scaled to 0..1
    "macd_norm",  # (EMA-12 - EMA-26) / Close
    "vol_20",     # 20-day realised volatility of log returns
]


def load_ohlcv(path):
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna()
    df = df[df["Volume"] > 0].sort_index()
    # Drop a partial last session (downloaded while the market was open).
    if len(df) > 21 and df["Volume"].iloc[-1] < 0.5 * df["Volume"].iloc[-21:-1].median():
        df = df.iloc[:-1]
    return df


def wilder_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def build_features(df):
    c = df["Close"]
    log_c = np.log(c)
    log_v = np.log(df["Volume"])
    out = pd.DataFrame(index=df.index)

    out["ret_1"] = log_c.diff()
    out["ret_5"] = log_c.diff(5)
    out["oc"] = np.log(c / df["Open"])
    out["hl"] = np.log(df["High"] / df["Low"])
    out["vol_z"] = (log_v - log_v.rolling(20).mean()) / log_v.rolling(20).std()
    out["sma10_gap"] = c / c.rolling(10).mean() - 1
    out["sma50_gap"] = c / c.rolling(50).mean() - 1
    out["rsi"] = wilder_rsi(c) / 100
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["macd_norm"] = macd / c
    out["vol_20"] = out["ret_1"].rolling(20).std()

    out["close"] = c
    out["target"] = out["ret_1"].shift(-1)  # next-day log return
    return out.dropna()


MARKET_FEATURES = ["sp_ret_1", "sp_ret_5", "vix_log", "vix_chg", "rel_1"]


def add_market_features(feats, market_path):
    """S&P 500 and VIX context (columns SP500, VIX), aligned to AAPL dates."""
    mk = pd.read_csv(market_path, index_col="Date", parse_dates=True).sort_index().ffill()
    m = pd.DataFrame(index=mk.index)
    m["sp_ret_1"] = np.log(mk["SP500"]).diff()
    m["sp_ret_5"] = np.log(mk["SP500"]).diff(5)
    m["vix_log"] = np.log(mk["VIX"])
    m["vix_chg"] = np.log(mk["VIX"]).diff()
    out = feats.join(m.reindex(feats.index, method="ffill"))
    out["rel_1"] = out["ret_1"] - out["sp_ret_1"]  # AAPL's move relative to the market
    return out.dropna()
