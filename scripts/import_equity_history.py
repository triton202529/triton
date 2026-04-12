#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge-import historical equity into TRITON portfolio_history.csv

Accepted input shapes (per source CSV)
--------------------------------------
A) equity-only:
   - columns:  date[, cash], equity
   - equity wins; cash defaults to 0 if missing

B) cash + market value:
   - columns:  date, cash, market_value
   - also accepted aliases for market_value: marketValue, portfolio_value, nav

Date column aliases accepted: date, timestamp, dt, time
If a row's time component is missing (midnight), we inject default-close time (16:00:00 ET).

Examples
--------
# 1) Single broker CSV (equity column)
python scripts/import_equity_history.py --source broker_history.csv

# 2) Cash + MV input (portfolio_value or marketValue or nav also OK)
python scripts/import_equity_history.py --source broker_cash_mv.csv

# 3) Multiple sources, only 2024+ rows, preview only
python scripts/import_equity_history.py --source s1.csv --source s2.csv --min-date 2024-01-01 --dry-run

# 4) Force backups and prefer existing rows on timestamp collisions
python scripts/import_equity_history.py --source s.csv --backup --prefer-existing
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
from typing import List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

DEFAULT_OUT = "data/results/portfolio_history.csv"
NY = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Import/merge historical equity snapshots into portfolio_history.csv"
    )
    p.add_argument(
        "--source",
        action="append",
        required=True,
        help=(
            "Path to a CSV file (may repeat). Requires either "
            "[date, equity(, cash)] OR [date, cash, market_value/portfolio_value/marketValue/nav]."
        ),
    )
    p.add_argument("--out", default=DEFAULT_OUT, help=f"Output CSV (default: {DEFAULT_OUT})")
    p.add_argument("--min-date", help="Only import rows with date >= this (YYYY-MM-DD)")
    p.add_argument("--max-date", help="Only import rows with date <= this (YYYY-MM-DD)")
    p.add_argument(
        "--prefer-existing",
        action="store_true",
        help="On same timestamp, keep existing row instead of replacing with imported row.",
    )
    p.add_argument("--dry-run", action="store_true", help="Parse/merge but do not write.")
    p.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak timestamped copy of the existing out file.",
    )
    p.add_argument(
        "--default-close",
        default="16:00:00",
        help="Time (HH:MM:SS) to inject when source dates have no time (default 16:00:00 ET).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# IO + Normalization
# ---------------------------------------------------------------------
def _read_source(path: str) -> pd.DataFrame:
    """Read one source CSV and normalize it to columns: date, cash, market_value."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["date", "cash", "market_value"])

    # Coalesce date column
    date_col = next((c for c in ["date", "timestamp", "dt", "time"] if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"{path}: no date/timestamp column found")
    df = df.rename(columns={date_col: "date"})

    # Normalize numeric columns (accept common aliases)
    numeric_candidates = ["equity", "cash", "market_value", "marketValue", "portfolio_value", "nav"]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derive cash/market_value when only equity given
    if "equity" in df.columns and not any(
        c in df.columns for c in ["market_value", "marketValue", "portfolio_value", "nav"]
    ):
        if "cash" not in df.columns:
            df["cash"] = 0.0
        df["market_value"] = df["equity"] - df["cash"].fillna(0.0)
    else:
        # Prefer standard 'market_value' naming; accept aliases
        if "market_value" not in df.columns:
            if "marketValue" in df.columns:
                df = df.rename(columns={"marketValue": "market_value"})
            elif "portfolio_value" in df.columns:
                df = df.rename(columns={"portfolio_value": "market_value"})
            elif "nav" in df.columns:
                df = df.rename(columns={"nav": "market_value"})

        if "cash" not in df.columns or "market_value" not in df.columns:
            raise ValueError(
                f"{path}: need either 'equity' (optionally 'cash') OR both 'cash' and "
                "'market_value' (aliases: portfolio_value, marketValue, nav)"
            )

    # Keep only required columns
    keep = ["date", "cash", "market_value"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns after normalization: {missing}")

    return df[keep].copy()


def _ensure_time_component(dt_series: pd.Series, default_close: str) -> pd.Series:
    """
    Convert to NY-localized timestamps; if time missing (midnight), inject default_close.
    Returns naive local strings 'YYYY-MM-DD HH:MM:SS' (NY), matching analyzer expectations.

    This implementation avoids chained-assignment warnings.
    """
    parsed = pd.to_datetime(dt_series, errors="coerce")

    # Localize or convert to America/New_York
    tz_attr = getattr(parsed.dt, "tz", None)
    if tz_attr is None:
        parsed = parsed.dt.tz_localize(NY, nonexistent="NaT", ambiguous="NaT")
    else:
        parsed = parsed.dt.tz_convert(NY)

    hh, mm, ss = (int(x) for x in default_close.split(":"))
    is_midnight = (parsed.dt.hour == 0) & (parsed.dt.minute == 0) & (parsed.dt.second == 0)
    adjusted = parsed.where(
        ~is_midnight, parsed.dt.floor("D") + pd.Timedelta(hours=hh, minutes=mm, seconds=ss)
    )

    return adjusted.dt.strftime("%Y-%m-%d %H:%M:%S")


def _filter_date_window(
    df: pd.DataFrame, min_date: Optional[str], max_date: Optional[str]
) -> pd.DataFrame:
    if not min_date and not max_date:
        return df
    dt = pd.to_datetime(df["date"], errors="coerce")
    if min_date:
        dt_min = pd.to_datetime(min_date)
        df = df[dt >= dt_min]
    if max_date:
        # inclusive day range by pushing max to end-of-day
        dt_max = pd.to_datetime(max_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[dt <= dt_max]
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    # Read/normalize all sources
    frames: List[pd.DataFrame] = []
    for src in args.source:
        try:
            df = _read_source(src)
        except Exception as e:
            print(f"[ERR] {e}", file=sys.stderr)
            sys.exit(1)
        df["date"] = _ensure_time_component(df["date"], args.default_close)
        frames.append(df)

    if not frames:
        print("[ERR] No data parsed.", file=sys.stderr)
        sys.exit(1)

    incoming = pd.concat(frames, ignore_index=True)
    incoming = incoming.dropna(subset=["date"])
    incoming = _filter_date_window(incoming, args.min_date, args.max_date)

    # Coerce numeric
    for c in ["cash", "market_value"]:
        incoming[c] = pd.to_numeric(incoming[c], errors="coerce")
    incoming = incoming.dropna(subset=["cash", "market_value"])

    # Load existing
    if os.path.exists(args.out):
        existing = pd.read_csv(args.out)
        if "date" not in existing.columns:
            existing["date"] = ""
        for c in ["cash", "market_value"]:
            if c in existing.columns:
                existing[c] = pd.to_numeric(existing[c], errors="coerce")
    else:
        existing = pd.DataFrame(columns=["date", "cash", "market_value"])

    # Standardize existing dates to same format
    try:
        existing_dt = pd.to_datetime(existing["date"], errors="coerce")
        # Assume existing is local ET; localize then format
        existing_dt = existing_dt.dt.tz_localize(NY, nonexistent="NaT", ambiguous="NaT")
        existing["date"] = existing_dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        existing["date"] = existing["date"].astype(str)

    # Merge (incoming wins by default)
    if args.prefer_existing:
        mask_dupe = incoming["date"].isin(existing["date"])
        replaced = int(mask_dupe.sum())
        incoming_use = incoming.loc[~mask_dupe]
        merged = pd.concat([existing, incoming_use], ignore_index=True)
        action = f"kept existing on {replaced} timestamp(s)"
    else:
        existing_use = existing[~existing["date"].isin(incoming["date"])]
        replaced = len(existing) - len(existing_use)
        merged = pd.concat([existing_use, incoming], ignore_index=True)
        action = f"replaced {replaced} timestamp(s) with incoming"

    # Sort and clean
    try:
        mdt = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.loc[mdt.notna()].copy()
        merged = merged.assign(_dt=mdt).sort_values("_dt").drop(columns="_dt")
        merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    before = len(existing)
    after = len(merged)
    added = max(0, after - before)
    print(
        f"[INFO] Incoming rows: {len(incoming)} | Existing: {before} | After merge: {after} | Added: {added}; {action}"
    )

    if args.dry_run:
        print(f"[DRY] Would write: {args.out}")
        return

    # Backup
    if args.backup and os.path.exists(args.out):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = f"{args.out}.{ts}.bak"
        os.makedirs(os.path.dirname(bak) or ".", exist_ok=True)
        shutil.copy2(args.out, bak)
        print(f"[OK] Backup written: {bak}")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    merged.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] Wrote merged history → {args.out}")


if __name__ == "__main__":
    main()
