#!/usr/bin/env python3
"""
generate_naive_orders.py

Builds naive BUY orders from pipeline weights.

Inputs (auto-detected unless overridden):
- weights: data/results/institutional/weights.csv  (columns: ticker,weight or Unnamed: 0,weight)
- prices : predictions/latest_predictions.csv       (columns: ticker,close)
- portfolio value:
    - CLI --portfolio-value, OR
    - data/results/risk/portfolio_value.csv (total_value), OR
    - data/broker_cash_mv.csv (cash + market_value)

Outputs:
- data/results/institutional/orders_naive.csv  (ticker, qty, close, target_notional, side, time)

Guards:
- Only positive weights
- Floors to whole-share qty
- Optional max spend from broker_cash_mv.csv cash (if --respect-cash)
- Optional top-N by target notional
"""

from __future__ import annotations

import argparse, math, sys, json
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
WEIGHTS_DEFAULT = ROOT / "data" / "results" / "institutional" / "weights.csv"
PRICES_DEFAULT = ROOT / "predictions" / "latest_predictions.csv"
PV_CSV_DEFAULT = ROOT / "data" / "results" / "risk" / "portfolio_value.csv"
BROKER_CASH_MV = ROOT / "data" / "broker_cash_mv.csv"
OUT_ORDERS = ROOT / "data" / "results" / "institutional" / "orders_naive.csv"


def load_portfolio_value(cli_value: Optional[float]) -> Optional[float]:
    if cli_value is not None:
        try:
            return float(cli_value)
        except Exception:
            pass
    # Try total_value first
    if PV_CSV_DEFAULT.exists():
        try:
            df = pd.read_csv(PV_CSV_DEFAULT)
            if "total_value" in df.columns and len(df):
                return float(df["total_value"].iloc[-1])
        except Exception:
            pass
    # Try cash + market_value fallback
    if BROKER_CASH_MV.exists():
        try:
            df = pd.read_csv(BROKER_CASH_MV)
            if {"cash", "market_value"}.issubset(df.columns) and len(df):
                return float(df["cash"].iloc[-1]) + float(df["market_value"].iloc[-1])
        except Exception:
            pass
    return None


def load_cash_available() -> Optional[float]:
    if BROKER_CASH_MV.exists():
        try:
            df = pd.read_csv(BROKER_CASH_MV)
            if "cash" in df.columns and len(df):
                return float(df["cash"].iloc[-1])
        except Exception:
            return None
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate naive BUY orders from weights")
    ap.add_argument("--weights", type=str, default=str(WEIGHTS_DEFAULT), help="Path to weights.csv")
    ap.add_argument(
        "--prices",
        type=str,
        default=str(PRICES_DEFAULT),
        help="Path to latest_predictions.csv (ticker,close)",
    )
    ap.add_argument(
        "--portfolio-value", type=float, default=None, help="Override total portfolio value"
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Keep only top-N by target notional (after weighting)",
    )
    ap.add_argument(
        "--respect-cash",
        action="store_true",
        help="Scale orders to not exceed available cash in broker_cash_mv.csv",
    )
    ap.add_argument(
        "--min-qty",
        type=int,
        default=1,
        help="Minimum integer shares per order (post-scaling). Default 1.",
    )
    ap.add_argument("--out", type=str, default=str(OUT_ORDERS), help="Output CSV path")
    args = ap.parse_args(argv)

    weights_p = Path(args.weights)
    prices_p = Path(args.prices)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Load weights
    if not weights_p.exists():
        print(f"ERROR: weights file not found: {weights_p}")
        return 2
    w = pd.read_csv(weights_p)
    # Normalize columns
    if "ticker" not in w.columns and "Unnamed: 0" in w.columns:
        w = w.rename(columns={"Unnamed: 0": "ticker"})
    if "ticker" not in w.columns or "weight" not in w.columns:
        print("ERROR: weights.csv must have columns [ticker, weight]")
        return 2
    w["ticker"] = w["ticker"].astype(str).str.upper()
    w = w[["ticker", "weight"]].dropna()
    w = w[w["weight"] > 0]

    # Load prices
    if not prices_p.exists():
        print(f"ERROR: prices file not found: {prices_p}")
        return 2
    px = pd.read_csv(prices_p)
    if not {"ticker", "close"}.issubset(px.columns):
        print("ERROR: prices CSV must have columns [ticker, close]")
        return 2
    px["ticker"] = px["ticker"].astype(str).str.upper()
    px = px[["ticker", "close"]].dropna()

    # Portfolio value
    total_val = load_portfolio_value(args.portfolio_value)
    if total_val is None:
        print(
            "ERROR: unable to determine portfolio value. Provide --portfolio-value or create data/results/risk/portfolio_value.csv"
        )
        return 2

    # Merge + compute target notionals
    df = w.merge(px, on="ticker", how="left").dropna(subset=["close"])
    if df.empty:
        print("ERROR: no overlap between weights and prices.")
        return 2
    df["target_notional"] = df["weight"] * float(total_val)

    # Sort by target_notional desc
    df = df.sort_values("target_notional", ascending=False).reset_index(drop=True)

    # Top-N filter (optional)
    if args.top_n and args.top_n > 0:
        df = df.head(args.top_n).copy()

    # Convert to integer qty
    df["qty"] = (df["target_notional"] / df["close"]).apply(lambda x: max(0, math.floor(x)))

    # Respect cash: scale down proportionally if sum(cost) > available cash
    cash_avail = None
    if args.respect_cash:
        cash_avail = load_cash_available()
        if cash_avail is None:
            print(
                "⚠️  --respect-cash was set but no cash found in data/broker_cash_mv.csv; skipping cash guard."
            )
        else:
            # Compute gross cost at floor shares
            cost = (df["qty"] * df["close"]).sum()
            if cost > cash_avail and cost > 0:
                scale = cash_avail / cost
                # Recompute qty with scale (still floor and >= min-qty if possible)
                df["qty"] = ((df["target_notional"] * scale) / df["close"]).apply(
                    lambda x: max(0, math.floor(x))
                )

    # Enforce min-qty
    if args.min_qty > 1:
        df.loc[df["qty"] > 0, "qty"] = df.loc[df["qty"] > 0, "qty"].clip(lower=args.min_qty)

    # Build orders
    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    orders = df.loc[df["qty"] > 0, ["ticker", "qty", "close", "target_notional"]].copy()
    orders["side"] = "BUY"
    orders["time"] = now_utc

    # Save
    orders.sort_values("target_notional", ascending=False).to_csv(outp, index=False)

    # Console summary
    lines = len(orders)
    gross = float((orders["qty"] * orders["close"]).sum()) if lines else 0.0
    print(f"✅ Wrote {outp} ({lines} lines)")
    if cash_avail is not None:
        print(f"   Cash guard: spent ~${gross:,.2f} of available ${cash_avail:,.2f}")
    print("\nTop 10 preview:")
    if lines:
        print(orders.head(10).to_string(index=False))
    else:
        print("(no orders)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
