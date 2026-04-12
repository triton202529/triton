#!/usr/bin/env python3
"""
Normalize price CSV to: date, close[, volume]

Handles messy inputs:
- No date column? Use --infer-dates-from <csv-with-dates> OR --start-date YYYY-MM-DD
- Price strings like "$1,234.56" or "1.234,56" are cleaned robustly
- Auto-detect delimiter/header
- Falls back to open/high/low if close is unusable
- NEW:
  --force-price-col <name>   pick a specific column as price (e.g., open/high/low/close)
  --allow-zero               keep rows where price == 0 after cleaning
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import re

DATE_CANDIDATES = [
    "date",
    "Date",
    "DATE",
    "datetime",
    "Datetime",
    "timestamp",
    "Timestamp",
    "time",
    "Time",
    "TradeDate",
    "Index",
]
PRICE_PRIORITY = [
    "Adj Close",
    "Adj close",
    "Adj_Close",
    "AdjClose",
    "adjusted_close",
    "Adjusted Close",
    "Close",
    "close",
    "CLOSE",
    "last",
    "Last",
    "Price",
    "price",
    "close_price",
    "Close Price",
    "Open",
    "open",
    "High",
    "high",
    "Low",
    "low",
]
VOLUME_CANDIDATES = [
    "volume",
    "Volume",
    "VOL",
    "vol",
    "Total Volume",
    "TotalVolume",
    "shares",
    "Shares",
]


# ---------- IO helpers ----------
def _read_with_sniff(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if df.shape[1] > 0:
            return df
    except Exception:
        pass
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 0:
            return df
    except Exception:
        pass
    df = pd.read_csv(path, header=None)
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    return df


def _pick_date_column(df: pd.DataFrame) -> str | None:
    for c in DATE_CANDIDATES:
        if c in df.columns:
            return c
    best, score = None, -1.0
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            s = parsed.notna().mean()
            if s > score and s > 0.5:
                best, score = col, s
        except Exception:
            continue
    return best


# ---------- numeric cleaning ----------
_CURRENCY = re.compile(r"[^\d,\.\-\s]")  # keep digits, comma, dot, minus, space


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "fi":
        return s.astype(float)
    x = s.astype(str).str.strip()
    x = x.str.replace(_CURRENCY, "", regex=True)
    x = x.str.replace("\u00a0", " ", regex=False)  # nbsp
    x = x.str.replace(" ", "", regex=False)  # remove spaces

    def _coerce_one(v: str) -> float:
        if v == "" or v.lower() in ("nan", "none", "null", "-"):
            return np.nan
        if "," in v and "." in v:
            if v.rfind(",") > v.rfind("."):
                v2 = v.replace(".", "").replace(",", ".")
            else:
                v2 = v.replace(",", "")
        elif "," in v and "." not in v:
            v2 = v.replace(",", ".")
        else:
            v2 = v
        try:
            return float(v2)
        except Exception:
            return np.nan

    return x.apply(_coerce_one)


def _pick_price_column(df: pd.DataFrame) -> str | None:
    for c in PRICE_PRIORITY:
        if c in df.columns:
            return c
    # fallback: numeric-ish with variance
    best_col, best_score = None, -1.0
    for col in df.columns:
        vals = _clean_numeric_series(df[col])
        score = vals.notna().mean() * float(vals.std(skipna=True) or 0)
        if score > best_score:
            best_col, best_score = col, score
    return best_col


def _pick_volume_column(df: pd.DataFrame) -> str | None:
    for c in VOLUME_CANDIDATES:
        if c in df.columns:
            return c
    best_col, best_score = None, -1.0
    for col in df.columns:
        vals = _clean_numeric_series(df[col])
        frac_big = (vals > 1000).mean()
        if vals.notna().mean() > 0.7 and frac_big > best_score:
            best_col, best_score = col, frac_big
    return best_col


# ---------- dates ----------
def _get_dates_from_other(file: Path, target_len: int) -> pd.Series:
    other = _read_with_sniff(file)
    dcol = _pick_date_column(other)
    if dcol is None:
        if other.index.name and other.index.name in DATE_CANDIDATES:
            other = other.reset_index()
            dcol = other.columns[0]
        else:
            raise ValueError(f"No date-like column in {file}")
    dates = pd.to_datetime(other[dcol], errors="coerce").dropna()
    if dates.empty:
        raise ValueError(f"No parseable dates in {file}")
    if len(dates) >= target_len:
        dates = dates.iloc[-target_len:].reset_index(drop=True)
    else:
        add = target_len - len(dates)
        extra = pd.bdate_range(start=dates.iloc[-1], periods=add + 1, inclusive="neither")
        dates = pd.concat([dates, pd.Series(extra)], ignore_index=True)
    return dates


def _synthesize_dates(start_date: str, periods: int) -> pd.Series:
    start = pd.to_datetime(start_date, errors="raise")
    rng = pd.bdate_range(start=start, periods=periods)
    return pd.Series(rng)


# ---------- normalize ----------
def normalize(
    in_file: Path,
    out_file: Path,
    preview: bool = False,
    infer_from: Path | None = None,
    start_date: str | None = None,
    force_price_col: str | None = None,
    allow_zero: bool = False,
) -> None:
    df = _read_with_sniff(in_file)

    dcol = _pick_date_column(df)
    pcol = (
        force_price_col
        if (force_price_col and force_price_col in df.columns)
        else _pick_price_column(df)
    )
    vcol = _pick_volume_column(df)

    if preview:
        print("Columns:", list(df.columns))
        print("Detected date col:", dcol)
        print("Detected price col:", pcol)
        print("Detected volume col:", vcol)
        return

    # dates
    if dcol is not None:
        dates = pd.to_datetime(df[dcol], errors="coerce")
    elif infer_from is not None:
        dates = _get_dates_from_other(infer_from, len(df))
    elif start_date is not None:
        dates = _synthesize_dates(start_date, len(df))
    else:
        raise ValueError(
            "No date-like column. Provide --infer-dates-from <csv> or --start-date YYYY-MM-DD."
        )

    if pcol is None:
        for alt in ("close", "Close", "open", "Open", "high", "High", "low", "Low"):
            if alt in df.columns:
                pcol = alt
                break
    if pcol is None:
        raise ValueError("No price-like column found (looked for close/adj close/open/high/low).")

    price = _clean_numeric_series(df[pcol])

    # If everything is NaN, try combining OHLC if present
    if price.notna().sum() == 0:
        o = _clean_numeric_series(df["open"]) if "open" in df.columns else None
        h = _clean_numeric_series(df["high"]) if "high" in df.columns else None
        l = _clean_numeric_series(df["low"]) if "low" in df.columns else None
        if o is not None and h is not None and l is not None:
            price = (o + h + l) / 3.0

    vol = _clean_numeric_series(df[vcol]) if vcol is not None else None

    out = pd.DataFrame({"date": dates, "close": price})
    if vol is not None:
        out["volume"] = vol

    # drop NaN dates; allow zeros if requested
    out = out.dropna(subset=["date"])
    if not allow_zero:
        out = out.dropna(subset=["close"])
    else:
        out = out[~out["close"].isna()]

    out = out.sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid rows after normalization (price/date both invalid).")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False)
    print(f"✅ Wrote {out_file} (rows={len(out)})  [price from '{pcol}']")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Normalize price CSV to date, close[, volume]")
    ap.add_argument("input", help="Input CSV file")
    ap.add_argument("output", nargs="?", help="Output CSV (default: <input>.fixed.csv)")
    ap.add_argument(
        "--preview", action="store_true", help="Preview detected columns without writing"
    )
    ap.add_argument(
        "--infer-dates-from", help="CSV to borrow dates from when input has no date column"
    )
    ap.add_argument(
        "--start-date", help="YYYY-MM-DD to synthesize business-day dates for the input length"
    )
    ap.add_argument(
        "--force-price-col", help="Force a specific column as price (e.g., open/high/low/close)"
    )
    ap.add_argument(
        "--allow-zero", action="store_true", help="Keep rows where price == 0 after cleaning"
    )
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output) if args.output else inp.with_suffix(".fixed.csv")
    infer_from = Path(args.infer_dates_from) if args.infer_dates_from else None

    normalize(
        inp,
        outp,
        preview=args.preview,
        infer_from=infer_from,
        start_date=args.start_date,
        force_price_col=args.force_price_col,
        allow_zero=args.allow_zero,
    )


if __name__ == "__main__":
    main()
