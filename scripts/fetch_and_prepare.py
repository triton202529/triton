import os
import sys
import time
import random
import pandas as pd
from tqdm import tqdm
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import shutil

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

PROCESSED_FILE = "data/processed/stock_data.parquet"
FAILED_LOG = "data/logs/failed_tickers.txt"
RESULTS_DIR = "data/results"

# Ensure folders exist
os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
os.makedirs(os.path.dirname(FAILED_LOG), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 🔹 Clear old files in data/results/
for file in os.listdir(RESULTS_DIR):
    file_path = os.path.join(RESULTS_DIR, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
print("🧹 Cleared old files from data/results/")

# Clear old failed tickers log
with open(FAILED_LOG, "w") as log:
    log.write("")

def fetch_data(ticker, retries=3, wait=2):
    for attempt in range(1, retries + 1):
        try:
            print(f"\n📥 Fetching {ticker} (Attempt {attempt})...")
            ticker_obj = yf.Ticker(ticker)
            # 🔹 Fetch 10 years of daily data
            df = ticker_obj.history(period="10y", interval="1d", auto_adjust=False)

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

            # Save each ticker's file to data/results/
            output_path = os.path.join(RESULTS_DIR, f"{ticker}.parquet")
            df.to_parquet(output_path, index=False)
        except Exception as e:
            print(f"⚠️ Skipping {ticker} due to indicator error: {e}")
            with open(FAILED_LOG, "a") as log:
                log.write(f"{ticker} (indicator error)\n")

    if not enhanced_frames:
        print("❌ All feature generation failed. Nothing to save.")
        return

    final_df = pd.concat(enhanced_frames).dropna()
    final_df.to_parquet(PROCESSED_FILE, index=False)
    print(f"✅ Saved merged dataset to: {PROCESSED_FILE}")
    print(f"✅ Saved individual ticker files to: {RESULTS_DIR}")

    print("\n📄 Failed tickers log saved to:", FAILED_LOG)

if __name__ == "__main__":
    main()
