"""Download the data the model needs:
  AAPL_historical_data.csv  - AAPL daily OHLCV (adjusted), last 10 years
  market_data.csv           - S&P 500 and VIX daily closes (meta-learner context)
"""
import datetime

import pandas as pd
import yfinance as yf

TICKER = "AAPL"
END = datetime.datetime.today()
START = END - datetime.timedelta(days=365 * 10)

stock = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
if isinstance(stock.columns, pd.MultiIndex):
    stock.columns = stock.columns.droplevel("Ticker")
stock[["Close", "High", "Low", "Open", "Volume"]].to_csv("AAPL_historical_data.csv")

market = yf.download(["^GSPC", "^VIX"], start=START, end=END, auto_adjust=True, progress=False)["Close"]
market = market.rename(columns={"^GSPC": "SP500", "^VIX": "VIX"})[["SP500", "VIX"]]
market.index.name = "Date"
market.to_csv("market_data.csv")

print(f"{TICKER}: {len(stock)} rows, market: {len(market)} rows")
