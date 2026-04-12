# scripts/fetch_and_prepare.py
"""
Fetch + prepare OHLCV data from yfinance into per-ticker parquet files.

Fixes:
- Session-aware AS_OF using NYSE last completed session (market_calendar.py)
- AS_OF coercion: handles str / Timestamp / datetime / date reliably
- Column normalization handles tuple/MultiIndex columns (yfinance sometimes returns these)
- ✅ End-date exclusivity fix: end = AS_OF + 1 day so AS_OF bar can appear
- ✅ Hard filters all rows to date <= AS_OF (after tz-safe normalization)
- ✅ “Actual AS_OF” guard: if data max date < calendar AS_OF, we downshift AS_OF to observed_max
- Safe results clearing: do NOT delete existing results unless we actually fetched data

Outputs:
- data/processed/{TICKER}.parquet   (preferred, per-ticker)
- data/processed/stock_data.parquet (optional merged)
- data/logs/failed_tickers.txt
"""

import os
import sys
import time
import random
from datetime import date, datetime, timedelta
from typing import Optional, List, Any

import pandas as pd
from tqdm import tqdm
import yfinance as yf

# Allow importing from parent directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.feature_generator import add_technical_indicators
from services.market_calendar import last_completed_session_date


# -----------------------------
# Config
# -----------------------------
TICKERS = [
    "AAPL",
    "TSLA",
    "GOOGL",
    "MSFT",
    "AMZN",
    "META",
    "NVDA",
    "NFLX",
    "AMD",
    "INTC",
    # ETFs / broad market
    "SPY",
    "QQQ",
    "DIA",
    "VTI",
    "VOO",
    "IWM",
    "ARKK",
    # Sector ETFs
    "XLF",
    "XLE",
    "XLY",
    "XLV",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLRE",
    "XLB",
    # Metals / crypto proxies / commodities
    "GLD",
    "SLV",
    "BITO",
    "GBTC",
    "USO",
    "UNG",
    "DBA",
    # Indexes (may be flaky on yfinance; kept for completeness)
    "^GSPC",
    "^IXIC",
    "^DJI",
    "^VIX",
    # Mega-cap / financials / blue chips
    "JPM",
    "BAC",
    "WFC",
    "C",
    "GS",
    "MS",
    "SCHW",
    "BLK",
    "BRK-B",
    "GE",
    "UNH",
    "JNJ",
    "PG",
    "V",
    "MA",
    "PEP",
    "KO",
    "CVX",
    "XOM",
    "WMT",
    "HD",
    "DIS",
    "T",
    "PFE",
    "ABBV",
    "MRK",
]

PROCESSED_DIR = os.path.join("data", "processed")
MERGED_PARQUET = os.path.join(PROCESSED_DIR, "stock_data.parquet")
FAILED_LOG = os.path.join("data", "logs", "failed_tickers.txt")

# We do NOT clear data/results here anymore (dangerous) — this script is Step 1.
CLEAR_RESULTS_DIR = False
RESULTS_DIR = os.path.join("data", "results")

# Fetch window
YEARS_BACK = 10
RETRIES = 3
WAIT_SEC = 2


# -----------------------------
# Setup folders
# -----------------------------
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAILED_LOG), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------
# Failure log
# -----------------------------
failed_tickers = set()


def log_failed_ticker(ticker: str, reason: str = "fetch error") -> None:
    if ticker not in failed_tickers:
        failed_tickers.add(ticker)
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ticker} ({reason})\n")


# -----------------------------
# AS_OF coercion (critical fix)
# -----------------------------
def coerce_as_of_date(x: Any) -> date:
    """
    Accepts date/datetime/pd.Timestamp or YYYY-MM-DD strings and returns datetime.date.
    This hardens the pipeline so market_calendar return type can vary safely.
    """
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()

    if isinstance(x, pd.Timestamp):
        try:
            return x.to_pydatetime().date()
        except Exception:
            return x.date()

    d = pd.to_datetime(x, errors="coerce")
    if pd.isna(d):
        raise ValueError(f"AS_OF is not parseable as a date: {repr(x)}")
    return d.date()


# -----------------------------
# Column normalization (tuple/MultiIndex safe)
# -----------------------------
def _flatten_col(c) -> str:
    if isinstance(c, tuple):
        parts = [str(x) for x in c if x is not None and str(x).strip() != ""]
        return "_".join(parts) if parts else ""
    return str(c)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [tuple(x) for x in df.columns.to_list()]
    except Exception:
        pass

    cols = [_flatten_col(c) for c in df.columns]
    cols = [c.strip() for c in cols]
    cols = [c.lower().replace(" ", "_") for c in cols]
    df.columns = cols
    return df


# -----------------------------
# Date normalization (tz-safe)
# -----------------------------
def normalize_date_column(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """
    Convert df[col] to timezone-naive pandas datetime, then create df["_d"] as python date.
    This avoids “today becomes yesterday” issues due to tz.
    """
    df[col] = pd.to_datetime(df[col], errors="coerce")

    # If tz-aware, drop tz. If tz-naive, this is harmless.
    try:
        if getattr(df[col].dt, "tz", None) is not None:
            df[col] = df[col].dt.tz_convert(None)
    except Exception:
        try:
            df[col] = df[col].dt.tz_localize(None)
        except Exception:
            pass

    df["_d"] = df[col].dt.date
    return df


# -----------------------------
# Fetch
# -----------------------------
def fetch_data(
    ticker: str,
    as_of: date,
    retries: int = RETRIES,
    wait: int = WAIT_SEC,
) -> Optional[pd.DataFrame]:
    """
    Fetches up to YEARS_BACK of daily bars, then hard-filters to date <= AS_OF.

    ✅ Permanent fix for "missing today's date after close":
    - yfinance 'end' behaves like an exclusive boundary in practice.
    - So we set end = AS_OF + 1 day, then filter back down to <= AS_OF.
    """
    as_of = coerce_as_of_date(as_of)
    start = as_of - timedelta(days=365 * YEARS_BACK)
    end = as_of + timedelta(days=1)  # ✅ critical

    for attempt in range(1, retries + 1):
        try:
            print(f"\n📥 Fetching {ticker} (Attempt {attempt})... AS_OF={as_of}")

            df = yf.Ticker(ticker).history(
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                actions=False,
            )

            if df is None or df.empty or df.isna().all().all():
                raise ValueError("Empty or invalid DataFrame")

            # Normalize columns BEFORE reset_index to avoid tuple/MultiIndex lower() errors
            df = normalize_columns(df)

            # Bring index out as a column
            df = df.reset_index(drop=False)
            df = normalize_columns(df)

            # Normalize date column name
            if "date" not in df.columns:
                if "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "date"})
                elif "index" in df.columns:
                    df = df.rename(columns={"index": "date"})

            if "date" not in df.columns:
                raise ValueError("Missing date column after normalization")

            for col in ("open", "high", "low", "close", "volume"):
                if col not in df.columns:
                    raise ValueError(f"Missing '{col}' column")

            # tz-safe normalize, then filter
            df = normalize_date_column(df, "date")
            df = df.dropna(subset=["_d", "close"])
            df = df[df["_d"] <= as_of].copy()
            df = df.drop(columns=["_d"], errors="ignore")

            if df.empty:
                raise ValueError("No rows remain after AS_OF filter")

            df["ticker"] = str(ticker).upper().strip()
            return df

        except Exception as e:
            print(f"⚠️ Error fetching {ticker}: {e}")
            if attempt < retries:
                print(f"🔁 Retrying {ticker} in {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"❌ Failed to fetch {ticker} after {retries} attempts.")
                log_failed_ticker(ticker, f"fetch error: {e}")
                return None


def main() -> None:
    raw_as_of = last_completed_session_date()
    as_of = coerce_as_of_date(raw_as_of)

    print(f"🧭 AS_OF (last completed NYSE session, ET): {as_of}  (raw={repr(raw_as_of)})")

    cleared_results = False
    all_data: List[pd.DataFrame] = []

    print(f"📊 Fetching data for {len(TICKERS)} tickers...")
    for ticker in tqdm(TICKERS):
        df = fetch_data(ticker, as_of=as_of)
        if df is not None and not df.empty:
            all_data.append(df)

            # Only clear results after first success, and only if configured
            if CLEAR_RESULTS_DIR and (not cleared_results):
                for fn in os.listdir(RESULTS_DIR):
                    fp = os.path.join(RESULTS_DIR, fn)
                    if os.path.isfile(fp):
                        os.remove(fp)
                print("🧹 Cleared old files from data/results/ (after first successful fetch)")
                cleared_results = True

        time.sleep(random.uniform(0.5, 2.5))

    if not all_data:
        print("❌ No data fetched. Aborting.")
        return

    print("✨ Combining and adding features...")

    full_df = pd.concat(all_data, ignore_index=True)
    full_df["ticker"] = full_df["ticker"].astype(str).str.upper().str.strip()

    # Normalize date tz-safe, create _d
    full_df = normalize_date_column(full_df, "date")
    full_df = full_df.dropna(subset=["_d", "close"])

    # ✅ “Actual AS_OF” guard — never claim today if data doesn't actually contain it
    observed_max = full_df["_d"].max()
    if observed_max is None or pd.isna(observed_max):
        observed_max = as_of

    actual_as_of = min(as_of, observed_max)
    if actual_as_of != as_of:
        print(
            f"⚠️ Calendar AS_OF={as_of} but data max date={observed_max}. Using actual_as_of={actual_as_of}."
        )

    # Enforce actual_as_of across everything downstream
    full_df = full_df[full_df["_d"] <= actual_as_of].copy()

    # Build SPY frame for indicators (fallback to first ticker if SPY missing)
    spy_df = full_df[full_df["ticker"] == "SPY"].copy()
    if spy_df.empty:
        spy_df = full_df[full_df["ticker"] == full_df["ticker"].iloc[0]].copy()

    enhanced_frames: List[pd.DataFrame] = []
    for ticker in sorted(full_df["ticker"].unique()):
        df_t = full_df[full_df["ticker"] == ticker].copy()

        try:
            # add_technical_indicators expects a datetime-like date column
            df_t["date"] = pd.to_datetime(df_t["date"], errors="coerce")
            spy_df_use = spy_df.copy()
            spy_df_use["date"] = pd.to_datetime(spy_df_use["date"], errors="coerce")

            df_t = add_technical_indicators(df_t, spy_df_use)

            # Final hard clamp again (belt + suspenders)
            if "date" in df_t.columns:
                df_t = normalize_date_column(df_t, "date")
                df_t = df_t[df_t["_d"] <= actual_as_of].copy()
                df_t = df_t.drop(columns=["_d"], errors="ignore")

            enhanced_frames.append(df_t)

            out_path = os.path.join(PROCESSED_DIR, f"{ticker}.parquet")
            df_t.to_parquet(out_path, index=False)

        except Exception as e:
            print(f"⚠️ Skipping {ticker} due to indicator error: {e}")
            log_failed_ticker(ticker, f"indicator error: {e}")

    if not enhanced_frames:
        print("❌ All feature generation failed. Nothing to save.")
        return

    final_df = pd.concat(enhanced_frames, ignore_index=True)

    # Drop rows with null close/date as final hygiene
    if "date" in final_df.columns:
        final_df["date"] = pd.to_datetime(final_df["date"], errors="coerce")
        final_df = final_df.dropna(subset=["date", "close"])
    else:
        final_df = final_df.dropna(subset=["close"])

    final_df.to_parquet(MERGED_PARQUET, index=False)

    print(f"✅ Saved merged dataset to: {MERGED_PARQUET}")
    print(f"✅ Saved per-ticker processed files to: {PROCESSED_DIR}")
    print(f"📄 Failed tickers log saved to: {FAILED_LOG}")
    print(f"🧾 FINAL AS_OF used (actual): {actual_as_of} (calendar: {as_of})")


if __name__ == "__main__":
    main()
