#!/usr/bin/env python3
"""
services/refresh_raw_inplace_yf.py

Updates existing data/raw/*.csv files IN PLACE by appending new yfinance rows.

Works with your current naming convention:
  TICKER_2020-07-08_to_2025-07-07.csv

It:
- extracts ticker from filename (prefix before first "_")
- reads last usable date from the file
- downloads from last_date+1 to today (end exclusive)
- appends + dedupes by date
- writes back to the SAME filename
"""

from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit(f"yfinance not installed/working: {e}")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TICKER_RE = re.compile(r"^([A-Za-z0-9.\-]+)_\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}\.csv$")


def infer_ticker(name: str) -> str | None:
    m = TICKER_RE.match(name)
    if not m:
        return None
    return m.group(1).upper()


def last_date_in_file(p: Path) -> pd.Timestamp | None:
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    c = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
    if not c:
        return None
    mx = pd.to_datetime(df[c], errors="coerce").max()
    if pd.isna(mx):
        return None
    return pd.Timestamp(mx).normalize()


def download(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    # normalize
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    return df


def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        print(f"🚫 No CSVs found in {RAW_DIR}")
        return

    today = pd.Timestamp.today().normalize()
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # exclusive

    updated = 0
    for p in files:
        tkr = infer_ticker(p.name)
        if not tkr:
            # skip unexpected filenames
            continue

        last = last_date_in_file(p)
        if last is None:
            print(f"⚠️ {p.name}: cannot read last date, skipping")
            continue

        if last >= today:
            print(f"✅ {tkr}: already up to date (last={last.date()})")
            continue

        start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"⬇️  {tkr}: {start} → {end} (file={p.name})")

        new = download(tkr, start=start, end=end)
        if new.empty:
            print(f"⚠️  {tkr}: no new rows returned")
            continue

        old = pd.read_csv(p)
        if "Date" in old.columns and "date" not in old.columns:
            old = old.rename(columns={"Date": "date"})
        old.columns = [c.strip().lower().replace(" ", "_") for c in old.columns]
        old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.normalize()

        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.dropna(subset=["date"]).sort_values("date")
        merged = merged.drop_duplicates(subset=["date"], keep="last")

        merged.to_csv(p, index=False)
        updated += 1
        print(f"✔ {tkr}: wrote {len(merged)} rows (latest={merged['date'].max().date()})")

    print(f"\n🏁 Done. Updated {updated} raw files.")


if __name__ == "__main__":
    main()
