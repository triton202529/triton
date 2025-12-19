#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds data/orders/orders_today.csv from signals_with_rationale.csv

Columns emitted:
ticker,side,qty,close,target_notional,time

Usage:
  python services/build_orders_from_signals.py
  python services/build_orders_from_signals.py --date 2025-11-04
"""

import argparse
from pathlib import Path
import pandas as pd

SIG_PATH = Path("data/results/signals_with_rationale.csv")
OUT_DIR  = Path("data/orders")
OUT_CSV  = OUT_DIR / "orders_today.csv"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD; default = latest in file")
    ap.add_argument("--qty", type=float, default=1.0, help="Qty per signal row (default 1)")
    ap.add_argument("--only", choices=["BUY","SELL"], help="Filter to only BUY or SELL")
    args = ap.parse_args()

    if not SIG_PATH.exists():
        raise SystemExit(f"Missing {SIG_PATH}")

    df = pd.read_csv(SIG_PATH)
    if "date" not in df.columns or "ticker" not in df.columns or "signal" not in df.columns:
        raise SystemExit("signals_with_rationale.csv must contain date,ticker,signal columns.")

    # Normalize/parse
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["signal"] = df["signal"].astype(str).str.upper()
    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    else:
        df["close"] = pd.NA

    if args.date:
        day = pd.Timestamp(args.date)
    else:
        day = df["date"].max()

    today = df[df["date"] == day].copy()
    if today.empty:
        raise SystemExit(f"No rows for {day.date()} in signals file.")

    keep = today[today["signal"].isin(["BUY","SELL"])].copy()
    if args.only:
        keep = keep[keep["signal"] == args.only]

    if keep.empty:
        raise SystemExit("No BUY/SELL rows to emit for chosen date/filters.")

    # Build orders
    out = pd.DataFrame({
        "ticker": keep["ticker"].astype(str).str.upper(),
        "side": keep["signal"],
        "qty": args.qty,
        "close": pd.to_numeric(keep["close"], errors="coerce"),
        "time": ""  # optional column used by executor
    })

    # Fill close if missing -> drop those rows (executor needs a price to set limit padding)
    out = out.dropna(subset=["close"])

    out["target_notional"] = out["qty"] * out["close"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out[["ticker","side","qty","close","target_notional","time"]].to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows -> {OUT_CSV}")

if __name__ == "__main__":
    main()
