# triton/scripts/test_fetch.py

import yfinance as yf

ticker = "AAPL"
print(f"📥 Fetching {ticker}...")
df = yf.download(ticker, period="1y", interval="1d", progress=False)

print(df.head())
print(f"\nColumns: {df.columns}")
