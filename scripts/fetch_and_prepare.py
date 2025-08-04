import os
import sys
import time
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf

# Allow importing from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.feature_generator import add_technical_indicators

# Tickers to fetch
TICKERS = [
    "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "NFLX", "AMD", "INTC",
    "SPY", "QQQ", "DIA", "VTI", "ARKK", "XLF", "XLE", "XLY", "XLV", "XLI", "XLK",
    "XLP", "XLU", "XLRE", "XLB", "GLD", "SLV", "BITO", "GBTC", "USO", "UNG", "DBA",
    "^GSPC", "^IXIC", "^DJI", "^VIX", "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW",
    "BLK", "BRK-B", "GE", "UNH", "JNJ", "PG", "V", "MA", "PEP", "KO", "CVX", "XOM",
    "WMT", "HD", "DIS", "T", "PFE", "ABBV", "MRK"
]

OUTPUT_FILE = "data/processed/stock_data.parquet"
FAILED_LOG = "data/logs/failed_tickers.txt"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
os.makedirs(os.path.dirname(FAILED_LOG), exist_ok=True)

def fetch_data(ticker, retries=3, wait=2):
    for attempt in range(1, retries + 1):
        try:
            print(f"\n📥 Fetching {ticker} (Attempt {attempt})...")
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period="1y", interval="1d", auto_adjust=False)

            if df.empty or df.isna().all().all():
                raise ValueError("Empty or invalid DataFrame")

            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            if "close" not in df.columns:
                raise ValueError("Missing 'close' column")

            df["ticker"] = ticker
            return df

        except Exception as e:
            print(f"⚠️ Error fetching {ticker}: {e}")
            if attempt < retries:
                print(f"🔁 Retrying {ticker} in {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"❌ Failed to fetch {ticker} after {retries} attempts.")
                with open(FAILED_LOG, "a") as log:
                    log.write(f"{ticker}\n")
                return None

def main():
    all_data = []

    print(f"📊 Fetching data for {len(TICKERS)} tickers...")
    for ticker in tqdm(TICKERS):
        df = fetch_data(ticker)
        if df is not None:
            all_data.append(df)
        time.sleep(random.uniform(0.5, 2.5))  # Delay to avoid rate limits

    if not all_data:
        print("❌ No data fetched. Aborting.")
        return

    print("✨ Combining and adding features...")
    full_df = pd.concat(all_data).dropna(subset=["close"])
    spy_df = full_df[full_df["ticker"] == "SPY"]

    enhanced_frames = []
    for ticker in full_df["ticker"].unique():
        df = full_df[full_df["ticker"] == ticker].copy()
        try:
            df = add_technical_indicators(df, spy_df)
            enhanced_frames.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {ticker} due to indicator error: {e}")

    if not enhanced_frames:
        print("❌ All feature generation failed. Nothing to save.")
        return

    final_df = pd.concat(enhanced_frames).dropna()
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Saved processed data to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
