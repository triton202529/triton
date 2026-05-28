import os
import sys
import time
import random
import pandas as pd
from tqdm import tqdm
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

PROCESSED_FILE = "data/processed/stock_data.parquet"
FAILED_LOG = "data/logs/failed_tickers.txt"
RESULTS_DIR = "data/results"

# Ensure folders exist
os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
os.makedirs(os.path.dirname(FAILED_LOG), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Start with empty failed tickers set
failed_tickers = set()

def clear_generated_ticker_files(results_dir=RESULTS_DIR):
    """Remove stale per-ticker parquet outputs without deleting CSV reports."""
    for file in os.listdir(results_dir):
        file_path = os.path.join(results_dir, file)
        if os.path.isfile(file_path) and file.endswith(".parquet"):
            os.remove(file_path)

def log_failed_ticker(ticker, reason="fetch error"):
    """Logs a failed ticker without duplicates"""
    if ticker not in failed_tickers:
        failed_tickers.add(ticker)
        with open(FAILED_LOG, "a") as log:
            log.write(f"{ticker} ({reason})\n")

def fetch_data(ticker, retries=3, wait=2):
    for attempt in range(1, retries + 1):
        try:
            print(f"\n📥 Fetching {ticker} (Attempt {attempt})...")
            df = yf.Ticker(ticker).history(period="10y", interval="1d", auto_adjust=False)

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
                log_failed_ticker(ticker, "fetch error")
                return None

def main():
    all_data = []

    print(f"📊 Fetching data for {len(TICKERS)} tickers...")
    for ticker in tqdm(TICKERS):
        df = fetch_data(ticker)
        if df is not None:
            all_data.append(df)
        time.sleep(random.uniform(0.5, 2.5))  # Anti-rate-limit delay

    if not all_data:
        print("❌ No data fetched. Aborting.")
        return

    print("✨ Combining and adding features...")
    full_df = pd.concat(all_data).dropna(subset=["close"])
    spy_df = full_df[full_df["ticker"] == "SPY"]

    enhanced_frames = []
    enhanced_by_ticker = []
    for ticker in full_df["ticker"].unique():
        df = full_df[full_df["ticker"] == ticker].copy()
        try:
            df = add_technical_indicators(df, spy_df)
            enhanced_frames.append(df)
            enhanced_by_ticker.append((ticker, df))
        except Exception as e:
            print(f"⚠️ Skipping {ticker} due to indicator error: {e}")
            log_failed_ticker(ticker, "indicator error")

    if not enhanced_frames:
        print("❌ All feature generation failed. Nothing to save.")
        return

    clear_generated_ticker_files()
    print("🧹 Cleared old ticker parquet files from data/results/")

    for ticker, df in enhanced_by_ticker:
        output_path = os.path.join(RESULTS_DIR, f"{ticker}.parquet")
        df.to_parquet(output_path, index=False)

    final_df = pd.concat(enhanced_frames).dropna()
    final_df.to_parquet(PROCESSED_FILE, index=False)

    print(f"✅ Saved merged dataset to: {PROCESSED_FILE}")
    print(f"✅ Saved individual ticker files to: {RESULTS_DIR}")
    print(f"\n📄 Failed tickers log saved to: {FAILED_LOG}")

if __name__ == "__main__":
    main()
