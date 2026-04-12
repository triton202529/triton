#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Append/Update a daily equity snapshot for TRITON.

Usage examples:
  # simplest: pass equity directly
  python scripts/append_equity_snapshot.py --equity 100000

  # cash + market value
  python scripts/append_equity_snapshot.py --cash 10000 --market-value 90000

  # pull from a CSV (must have columns: cash,market_value[,date])
  python scripts/append_equity_snapshot.py --source-csv data/results/account_snapshot.csv

  # pick a specific date/time (ISO or 'YYYY-MM-DD HH:MM:SS')
  python scripts/append_equity_snapshot.py --equity 100000 --date "2025-10-19 16:00:00"
"""

from __future__ import annotations
import argparse, os, sys
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

DEFAULT_OUT = "data/results/portfolio_history.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append/update a daily equity snapshot for TRITON.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--equity", type=float, help="Total equity (cash + market_value).")
    p.add_argument("--cash", type=float, help="Cash component.")
    p.add_argument("--market-value", type=float, help="Market value component.")
    src.add_argument(
        "--source-csv",
        help="CSV with columns: cash,market_value[,date]. Uses last row if multiple.",
    )
    p.add_argument(
        "--date", help="Timestamp to record (defaults to market close 16:00 America/New_York)."
    )
    p.add_argument("--out", default=DEFAULT_OUT, help=f"Output CSV path (default: {DEFAULT_OUT})")
    return p.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[datetime, float, float, float]:
    # Date
    if args.date:
        try:
            ts = pd.to_datetime(args.date).to_pydatetime()
        except Exception:
            print(f"[ERR] Could not parse --date: {args.date}", file=sys.stderr)
            sys.exit(1)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=ZoneInfo("America/New_York"))
    else:
        # default to today's 16:00 America/New_York
        now = datetime.now(ZoneInfo("America/New_York"))
        ts = datetime.combine(now.date(), time(16, 0), tzinfo=ZoneInfo("America/New_York"))

    # Inputs
    cash = None
    mv = None
    eq = None

    if args.equity is not None:
        eq = float(args.equity)
        cash = 0.0 if args.cash is None else float(args.cash)
        mv = eq - cash
    elif args.cash is not None and args.market_value is not None:
        cash = float(args.cash)
        mv = float(args.market_value)
        eq = cash + mv
    elif args.source_csv:
        if not os.path.exists(args.source_csv):
            print(f"[ERR] --source-csv not found: {args.source_csv}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(args.source_csv)
        if "cash" not in df.columns or (
            "market_value" not in df.columns and "marketValue" not in df.columns
        ):
            print("[ERR] source CSV must have columns: cash, market_value", file=sys.stderr)
            sys.exit(1)
        mv_col = "market_value" if "market_value" in df.columns else "marketValue"
        row = df.tail(1).iloc[0]
        cash = float(row["cash"])
        mv = float(row[mv_col])
        eq = cash + mv
        # Optional: borrow date from source if present
        if args.date is None and "date" in df.columns and pd.notna(row["date"]):
            try:
                ts = pd.to_datetime(row["date"]).to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=ZoneInfo("America/New_York"))
            except Exception:
                pass
    else:
        print(
            "[ERR] Provide either --equity OR both --cash and --market-value OR --source-csv.",
            file=sys.stderr,
        )
        sys.exit(1)

    return ts, cash, mv, eq


def main():
    args = parse_args()
    ts, cash, mv, eq = resolve_inputs(args)

    outdir = os.path.dirname(args.out)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    # Normalize to naive local string (the analyzer parses to datetime anyway)
    ts_str = ts.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")

    # Load or create
    if os.path.exists(args.out):
        df = pd.read_csv(args.out)
    else:
        df = pd.DataFrame(columns=["date", "cash", "market_value"])

    # Update if exact timestamp exists; else append
    if "date" not in df.columns:
        df["date"] = ""

    mask = df["date"] == ts_str
    if mask.any():
        idx = df.index[mask][0]
        df.loc[idx, "cash"] = cash
        df.loc[idx, "market_value"] = mv
    else:
        df = pd.concat(
            [df, pd.DataFrame([{"date": ts_str, "cash": cash, "market_value": mv}])],
            ignore_index=True,
        )

    # Sort by date and save
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        pass
    df = df.sort_values("date")
    # Ensure CSV stores date as string with seconds
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] Snapshot recorded: {ts_str} | equity={eq:.2f} (cash={cash:.2f}, mv={mv:.2f})")
    print(f"[OK] Wrote: {args.out}")


if __name__ == "__main__":
    main()
