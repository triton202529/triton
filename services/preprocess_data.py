# services/preprocess_data.py
from __future__ import annotations

import re
import pandas as pd


def _coalesce_base_from_suffix(df: pd.DataFrame, base: str) -> pd.DataFrame:
    """
    If df[base] is missing or mostly NaN, try to fill it from any suffixed
    columns like f"{base}_aapl" that may have been introduced by yfinance
    multiindex flattening / merges.

    We prefer:
      1) existing base column values
      2) first suffixed column with non-null values
    """
    # Find suffix columns like close_aapl, close_msft, etc.
    suffix_cols = [c for c in df.columns if c.startswith(base + "_")]
    if not suffix_cols:
        return df

    if base not in df.columns:
        df[base] = pd.NA

    # Fill base NaNs from suffix columns (first non-null wins)
    for c in suffix_cols:
        df[base] = df[base].where(df[base].notna(), df[c])

    return df


def preprocess_stock_csv(file_path: str) -> pd.DataFrame:
    """
    Preprocess raw OHLCV CSV into canonical columns:
      date, open, high, low, close, volume, ticker

    Key behavior:
    - DO NOT clamp to filename date range.
    - If yfinance appended rows into suffixed columns (close_aapl, etc),
      we coalesce those into the base columns.
    """
    df = pd.read_csv(file_path)
    if df.empty:
        return pd.DataFrame()

    # normalize columns
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # ensure date exists
    if "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    if "date" not in df.columns:
        return pd.DataFrame()

    # ensure ticker exists (try infer from file name if missing)
    if "ticker" not in df.columns:
        m = re.search(
            r"(?:^|\\\\|/)([a-zA-Z0-9]+)_\\d{4}-\\d{2}-\\d{2}_to_\\d{4}-\\d{2}-\\d{2}\\.csv$",
            file_path.replace("\\\\", "/"),
        )
        df["ticker"] = (m.group(1).upper() if m else "").strip()
    else:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # Coalesce base OHLCV from suffixed columns if needed
    for base in ["open", "high", "low", "close", "volume", "adj_close"]:
        df = _coalesce_base_from_suffix(df, base)

    # Require at least OHLCV base columns after coalesce
    required = ["open", "high", "low", "close", "volume"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    # Parse types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing essentials
    df = df.dropna(subset=["date", "close"])

    # Sort + dedupe by date (keep last)
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")

    # Canonical output
    out = df[["date", "open", "high", "low", "close", "volume", "ticker"]].reset_index(drop=True)
    return out
