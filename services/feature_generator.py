# services/feature_generator.py

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Where your locally saved price files may live (Stooq/YF outputs)
SEARCH_DIRS = [
    "data/parquet",
    "data/raw",
    "data/processed",
    "data/results",
]

# Preferred benchmark symbols to try locally (we'll pick the first found)
BENCHMARK_CANDIDATES = ["SPY", "^GSPC", "^SPX", "^SP500"]

def _find_local_file(symbol: str) -> Path | None:
    """Return the first existing parquet/csv path for a symbol across SEARCH_DIRS."""
    for folder in SEARCH_DIRS:
        p_parquet = Path(folder) / f"{symbol}.parquet"
        if p_parquet.exists():
            return p_parquet
        p_csv = Path(folder) / f"{symbol}.csv"
        if p_csv.exists():
            return p_csv
    return None

def _read_price_df(path: Path) -> pd.DataFrame:
    """Read a local parquet/csv and return a DataFrame with ['date','close'] sorted by date."""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Normalize columns (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = cols_lower.get("date", None)
    if date_col is None:
        # try some common alternates
        for cand in ["Datetime", "Time", "timestamp", "Date"]:
            if cand in df.columns:
                date_col = cand
                break
    if date_col is None:
        # as last resort try first column
        date_col = df.columns[0]

    # find close-like column
    close_col = None
    for cand in ["close", "Close", "adj_close", "Adj Close", "Adj_Close"]:
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        raise ValueError(f"No close/price column found in {path}")

    out = (
        df[[date_col, close_col]]
        .rename(columns={date_col: "date", close_col: "close"})
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce"))
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    # Ensure numeric
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["close"]).reset_index(drop=True)
    return out

def load_benchmark_df() -> pd.DataFrame | None:
    """Try to load a local benchmark (SPY preferred). Returns ['date','spy_close'] or None."""
    for sym in BENCHMARK_CANDIDATES:
        p = _find_local_file(sym)
        if p is None:
            continue
        try:
            df = _read_price_df(p)
            return df.rename(columns={"close": "spy_close"})[["date", "spy_close"]]
        except Exception:
            # try next candidate if file malformed
            continue
    return None

def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI with EWM for stability."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_technical_indicators(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Adds robust technical features to a single-ticker DataFrame.
    Expects at minimum: ['date','close'] (case-insensitive OK).

    If spy_df is omitted, we attempt to load a local benchmark (SPY preferred) and align by date.
    No external/network fetches occur.
    """
    if df is None or df.empty:
        return df

    # Normalize column names presence
    cols_lower = {c.lower(): c for c in df.columns}
    if "date" not in cols_lower or "close" not in cols_lower:
        raise ValueError(f"DataFrame must include 'date' and 'close' columns. Got: {df.columns.tolist()}")

    date_col = cols_lower["date"]
    close_col = cols_lower["close"]

    # Basic cleanup / sorting
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")

    # === Core indicators ===
    # Returns with explicit fill_method=None to avoid FutureWarning
    df["returns"] = df[close_col].pct_change(fill_method=None)

    # Simple MAs with min_periods
    df["ma7"]  = df[close_col].rolling(window=7,  min_periods=3).mean()
    df["ma21"] = df[close_col].rolling(window=21, min_periods=7).mean()

    # EMAs
    df["ema12"] = df[close_col].ewm(span=12, adjust=False, min_periods=12).mean()
    df["ema26"] = df[close_col].ewm(span=26, adjust=False, min_periods=26).mean()

    # RSI(14) (Wilder)
    df["rsi14"] = _rsi_wilder(df[close_col], 14)

    # MACD (12/26 EMA) + signal(9) + hist
    macd = df["ema12"] - df["ema26"]
    df["macd"] = macd
    df["macd_signal"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Volatility = rolling std of returns (10d) with min_periods
    df["vol10"] = df["returns"].rolling(window=10, min_periods=5).std()

    # === Benchmark alignment (optional) ===
    if spy_df is None:
        spy_df = load_benchmark_df()

    if spy_df is not None and not spy_df.empty:
        spy = spy_df.copy()
        # normalize incoming spy_df
        if "date" not in spy.columns:
            raise ValueError("spy_df must include a 'date' column.")
        if "spy_close" not in spy.columns:
            # allow passing a generic close column
            if "close" in spy.columns:
                spy = spy.rename(columns={"close": "spy_close"})
            else:
                raise ValueError("spy_df must include 'spy_close' (or 'close') column.")

        spy["date"] = pd.to_datetime(spy["date"], errors="coerce")
        spy = spy.dropna(subset=["date"]).sort_values("date")
        spy = spy[["date", "spy_close"]]

        df = df.merge(spy, left_on=date_col, right_on="date", how="left", suffixes=("", "_bench"))
        if "date_bench" in df.columns:
            df = df.drop(columns=["date_bench"])
        # Benchmark returns with explicit fill_method=None
        df["spy_returns"] = df["spy_close"].pct_change(fill_method=None)
    else:
        # Graceful fallback to a neutral market feature
        df["spy_close"] = np.nan
        df["spy_returns"] = 0.0

    # === Final tidy ===
    # Fill engineered columns only (leave raw OHLC as-is if present)
    engineered = ["returns", "ma7", "ma21", "ema12", "ema26", "rsi14",
                  "macd", "macd_signal", "macd_hist", "vol10", "spy_returns"]
    for c in engineered:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Keep all rows; do NOT dropna() globally
    return df
