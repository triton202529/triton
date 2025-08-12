# scripts/retry_failed_tickers.py
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

# yfinance primary fetch with caching to reduce JSONDecode/ratelimit issues
import requests_cache
import yfinance as yf

# Optional Stooq fallback (pip install pandas-datareader)
try:
    from pandas_datareader import data as pdr
    HAS_STOOQ = True
except Exception:
    HAS_STOOQ = False

# Allow importing from the parent directory for your feature generator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.feature_generator import add_technical_indicators

# ---------------- Paths ----------------
FAILED_INPUT_FILE = "data/logs/failed_tickers_unique.txt"
PROCESSED_FILE = "data/processed/retried_stock_data.parquet"
RETRY_FAILED_LOG = "data/logs/failed_tickers_retry.txt"
RESULTS_DIR = "data/results"

# Ensure folders exist
os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
os.makedirs(os.path.dirname(RETRY_FAILED_LOG), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- yfinance session ----------------
# Cached session reduces repeat calls and tames Yahoo’s hiccups
session = requests_cache.CachedSession(
    cache_name=".yfcache",
    backend="sqlite",
    expire_after=10 * 60,  # 10 minutes
)
session.headers["User-Agent"] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121 Safari/537.36"
)

# ---------------- Helpers ----------------
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.lower().strip().replace(" ", "_") for c in df.columns})

def fetch_yf_download(ticker: str, period="10y", interval="1d",
                      max_retries=3, sleep=(1.2, 3.0)) -> pd.DataFrame | None:
    """
    More robust than Ticker().history(); handles multi-index and empties.
    """
    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=False,
                timeout=20,
                session=session,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                # If df is for a single ticker, columns are standard OHLCV
                # If multi-level, select that ticker
                if isinstance(df.columns, pd.MultiIndex):
                    # select the first level matching our ticker if present
                    top = [lvl for lvl in df.columns.levels[0] if str(lvl).upper() == ticker.upper()]
                    if top:
                        df = df[top[0]]
                df = df.reset_index()
                df = _norm_cols(df)
                # Ensure we have 'close'
                if "close" not in df.columns and "adj_close" in df.columns:
                    df["close"] = df["adj_close"]
                if "close" in df.columns and not df.empty:
                    df["ticker"] = ticker
                    return df
            last_err = "empty dataframe"
        except Exception as e:
            last_err = str(e)
        if attempt < max_retries:
            time.sleep(random.uniform(*sleep))
    # print last error once
    if last_err:
        print(f"⚠️ yfinance failed for {ticker}: {last_err}")
    return None

def fetch_stooq(ticker: str) -> pd.DataFrame | None:
    """
    Fallback: Stooq daily bars. Skips index-style symbols (^GSPC, etc.).
    For equities/ETFs like AAPL/TSLA/SPY, Stooq usually works.
    """
    if not HAS_STOOQ:
        return None
    if ticker.startswith("^"):
        return None  # skip indices here (mapping varies by provider)
    try:
        df = pdr.DataReader(ticker, "stooq")
        if df is None or df.empty:
            return None
        df = df.sort_index().reset_index()
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        # Stooq may not have volume; ok for most features
        df["ticker"] = ticker
        return df
    except Exception as e:
        print(f"⚠️ Stooq failed for {ticker}: {e}")
        return None

def save_parquet_basic(df: pd.DataFrame, ticker: str) -> bool:
    # Minimal OHLC check
    needed = {"date", "open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        return False
    out_path = os.path.join(RESULTS_DIR, f"{ticker}.parquet")
    df.to_parquet(out_path, index=False)
    return True

def fetch_data(ticker: str) -> pd.DataFrame | None:
    """
    Try yfinance (cached) then Stooq.
    Returns raw OHLCV with ['date','open','high','low','close','volume','ticker'].
    """
    print(f"\n📥 Fetching {ticker} …")
    df = fetch_yf_download(ticker)
    if df is None:
        df = fetch_stooq(ticker)
    if df is None or df.empty or "close" not in df.columns:
        with open(RETRY_FAILED_LOG, "a") as log:
            log.write(f"{ticker}\n")
        return None
    return df

# ---------------- Main ----------------
def main():
    # Load tickers
    if not os.path.exists(FAILED_INPUT_FILE):
        print(f"❌ {FAILED_INPUT_FILE} not found.")
        return
    with open(FAILED_INPUT_FILE, "r") as file:
        tickers = sorted(set(line.strip().upper() for line in file if line.strip()))

    print(f"🔁 Retrying {len(tickers)} failed tickers...")

    all_raw = []
    for ticker in tqdm(tickers):
        df = fetch_data(ticker)
        if df is not None:
            all_raw.append(df)
        # Gentle pacing (helps avoid throttle)
        time.sleep(random.uniform(0.6, 1.6))

    if not all_raw:
        print("❌ No data fetched. Exiting.")
        return

    print("✨ Adding features to retried data…")
    full_df = pd.concat(all_raw, ignore_index=True)
    full_df = full_df.dropna(subset=["close"])

    # Build SPY reference for indicators if available in newly fetched data
    spy_df = full_df[full_df["ticker"] == "SPY"].copy()

    enhanced_frames = []
    # Compute indicators per ticker and save per‑ticker parquet
    for ticker in sorted(full_df["ticker"].unique()):
        df_t = full_df[full_df["ticker"] == ticker].copy()
        try:
            out = add_technical_indicators(df_t, spy_df)
            enhanced_frames.append(out)
            out_path = os.path.join(RESULTS_DIR, f"{ticker}.parquet")
            out.to_parquet(out_path, index=False)
        except Exception as e:
            print(f"⚠️ Skipping {ticker} (indicator error): {e}")
            with open(RETRY_FAILED_LOG, "a") as log:
                log.write(f"{ticker} (indicator error)\n")

    if not enhanced_frames:
        print("❌ Feature generation failed for all. Nothing saved.")
        return

    final_df = pd.concat(enhanced_frames, ignore_index=True).dropna(subset=["close"])
    final_df.to_parquet(PROCESSED_FILE, index=False)
    print(f"✅ Retrained data saved to: {PROCESSED_FILE}")

if __name__ == "__main__":
    main()
