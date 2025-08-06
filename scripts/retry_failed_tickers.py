import os
import sys
import time
import random
import pandas as pd
from tqdm import tqdm
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

# Allow importing from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.feature_generator import add_technical_indicators

# Paths
FAILED_INPUT_FILE = "data/logs/failed_tickers.unique.txt"
PROCESSED_FILE = "data/processed/retried_stock_data.parquet"
RETRY_FAILED_LOG = "data/logs/failed_tickers_retry.txt"
RESULTS_DIR = "data/results"

# Ensure folders exist
os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
os.makedirs(os.path.dirname(RETRY_FAILED_LOG), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load tickers
with open(FAILED_INPUT_FILE, "r") as file:
    tickers = sorted(set(line.strip() for line in file if line.strip()))

print(f"🔁 Retrying {len(tickers)} failed tickers...")

def fetch_data(ticker, retries=3, wait=2):
    for attempt in range(1, retries + 1):
        try:
            print(f"\n📥 Fetching {ticker} (Attempt {attempt})...")
            ticker_obj = yf.Ticker(ticker)
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
                print(f"⏳ Retrying in {wait} sec...")
                time.sleep(wait)
            else:
                print(f"❌ Failed to fetch {ticker}")
                with open(RETRY_FAILED_LOG, "a") as log:
                    log.write(f"{ticker}\n")
                return None

def main():
    all_data = []

    for ticker in tqdm(tickers):
        df = fetch_data(ticker)
        if df is not None:
            all_data.append(df)
        time.sleep(random.uniform(0.5, 2.5))  # Anti-rate-limiting

    if not all_data:
        print("❌ No data fetched. Exiting.")
        return

    print("✨ Adding features to retried data...")
    full_df = pd.concat(all_data).dropna(subset=["close"])
    spy_df = full_df[full_df["ticker"] == "SPY"]

    enhanced_frames = []
    for ticker in full_df["ticker"].unique():
        df = full_df[full_df["ticker"] == ticker].copy()
        try:
            df = add_technical_indicators(df, spy_df)
            enhanced_frames.append(df)

            # Save to data/results/
            output_path = os.path.join(RESULTS_DIR, f"{ticker}.parquet")
            df.to_parquet(output_path, index=False)
        except Exception as e:
            print(f"⚠️ Skipping {ticker} (indicator error): {e}")
            with open(RETRY_FAILED_LOG, "a") as log:
                log.write(f"{ticker} (indicator error)\n")

    if not enhanced_frames:
        print("❌ Feature generation failed for all. Nothing saved.")
        return

    final_df = pd.concat(enhanced_frames).dropna()
    final_df.to_parquet(PROCESSED_FILE, index=False)
    print(f"✅ Retrained data saved to: {PROCESSED_FILE}")

if __name__ == "__main__":
    main()
