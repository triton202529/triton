import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

tickers = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX", "AMD",
    "SPY", "QQQ", "DIA", "IWM", "ARKK", "VTI", "VOO",
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLRE",
    "GLD", "SLV", "USO", "UNG", "DBA",
    "GBTC", "BITO"
]

start_date = (datetime.today() - timedelta(days=1825)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

for ticker in tickers:
    try:
        df = yf.download(ticker, start=start_date, end=end_date)

        if df.empty:
            print(f"⚠️ No data for {ticker}")
            continue

        df.reset_index(inplace=True)

        # Safely select available columns
        columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        df = df[[col for col in columns_to_keep if col in df.columns]]

        # Ensure numeric data
        for col in df.columns:
            if col != 'Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        file_name = f"{ticker}_{start_date}_to_{end_date}.csv"
        df.to_csv(os.path.join(output_dir, file_name), index=False)
        print(f"✅ Saved {ticker} to {file_name}")
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
