# scripts/reprice_to_marketable.py
"""
Reprice orders_today.csv into orders_today_marketable.csv using fresh market quotes.

WHY THIS EXISTS
- Turns "close" prices into *marketable* limit prices at the current bid/ask.
- Adds/updates a `limit_price` column.
- Handles input schemas that use `ticker` (your case) OR `symbol` OR `sym`.

SAFETY
- By default, refuses to reprice when market is CLOSED (use --allow-closed to override).
- Requires quote timestamp freshness (max age in minutes via --max-age-min).
- Skips rows with missing symbol/side/qty, missing quotes, stale quotes, or bad bid/ask.

OUTPUT
- Writes: data/live/orders_today_marketable.csv by default
- Keeps existing columns and appends `limit_price`.
- Ensures BOTH `ticker` and `symbol` columns exist (so downstream tools don’t break).

USAGE (PowerShell)
  python .\\scripts\\reprice_to_marketable.py --mode paper --max-age-min 5 --verbose

NOTES
- "marketable" here means:
    BUY  -> limit_price = ask
    SELL -> limit_price = bid
  (If ask/bid missing, we skip.)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

import pandas as pd

# ---- path bootstrap (repo root) ----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.broker_alpaca import AlpacaBroker  # noqa: E402


DEFAULT_IN = os.path.join("data", "live", "orders_today.csv")
DEFAULT_OUT = os.path.join("data", "live", "orders_today_marketable.csv")


def _iso_to_dt(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _minutes_since(ts: str) -> Optional[float]:
    dt = _iso_to_dt(ts)
    if not dt:
        return None
    now = datetime.now(timezone.utc)
    return (now - dt).total_seconds() / 60.0


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    def has(col: str) -> bool:
        return col in lower_map

    def colname(col: str) -> str:
        return lower_map[col]

    # Normalize symbol/ticker/sym -> ensure BOTH symbol and ticker columns exist
    if not has("symbol"):
        if has("ticker"):
            df["symbol"] = df[colname("ticker")]
        elif has("sym"):
            df["symbol"] = df[colname("sym")]
        else:
            df["symbol"] = ""

    if not has("ticker"):
        df["ticker"] = df["symbol"]

    # Side
    if has("side"):
        df[colname("side")] = df[colname("side")].astype(str).str.strip().str.upper()
    else:
        df["side"] = ""

    # Qty
    if has("qty"):
        df[colname("qty")] = pd.to_numeric(df[colname("qty")], errors="coerce").fillna(0.0)
    else:
        df["qty"] = 0.0

    # Ensure limit_price exists (do not auto-fill here)
    if not has("limit_price"):
        df["limit_price"] = ""

    return df


def _row_symbol(df: pd.DataFrame, idx: int) -> str:
    s = str(df.at[idx, "symbol"] if "symbol" in df.columns else "").strip().upper()
    if not s and "ticker" in df.columns:
        s = str(df.at[idx, "ticker"]).strip().upper()
    return s


def _row_side(df: pd.DataFrame, idx: int) -> str:
    return str(df.at[idx, "side"] if "side" in df.columns else "").strip().upper()


def _row_qty(df: pd.DataFrame, idx: int) -> float:
    try:
        return float(df.at[idx, "qty"])
    except Exception:
        return 0.0


def _quote_ok(q: Dict, max_age_min: float) -> Tuple[bool, str]:
    if not isinstance(q, dict):
        return False, "quote_not_dict"

    ts = str(q.get("ts") or "").strip()
    age = _minutes_since(ts)
    if age is None:
        return False, "quote_missing_ts"
    if age > max_age_min:
        return False, f"quote_stale_{age:.1f}m"

    bid = float(q.get("bid") or 0) if q.get("bid") is not None else 0.0
    ask = float(q.get("ask") or 0) if q.get("ask") is not None else 0.0
    if bid <= 0 and ask <= 0:
        return False, "quote_missing_bid_ask"

    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reprice orders to marketable limits using fresh bid/ask quotes."
    )
    ap.add_argument(
        "--mode", default="paper", choices=["paper", "live"], help="Broker mode for quotes/clock."
    )
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN, help="Input orders CSV.")
    ap.add_argument(
        "--out", dest="out_path", default=DEFAULT_OUT, help="Output CSV with limit_price."
    )
    ap.add_argument(
        "--max-age-min", type=float, default=5.0, help="Max quote age (minutes) to accept."
    )
    ap.add_argument(
        "--allow-closed", action="store_true", help="Allow repricing even if market is closed."
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Print skip reasons and per-row actions."
    )
    args = ap.parse_args()

    in_path = args.in_path
    out_path = args.out_path

    if not os.path.exists(in_path):
        raise SystemExit(f"[BLOCK] Input file not found: {in_path}")

    try:
        df = pd.read_csv(in_path)
    except Exception:
        df = pd.read_csv(in_path, engine="python", on_bad_lines="skip")

    df = _norm_cols(df)

    broker = AlpacaBroker(mode=args.mode)

    clock = broker.get_clock()
    is_open = bool(clock.get("is_open"))

    if not is_open and not args.allow_closed:
        if args.verbose:
            print("[BLOCK] Market is CLOSED. Not repricing (use --allow-closed to override).")
            print("clock:", clock)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"[OK] Input:  {in_path}")
        print(f"[OK] Output: {out_path}")
        print(f"[OK] Repriced rows: 0 | Skipped: {len(df)}")
        print("[INFO] Market closed block engaged (output written without repricing).")
        return

    repriced = 0
    skipped = 0
    skip_reasons: Dict[str, int] = {}

    for i in range(len(df)):
        sym = _row_symbol(df, i)
        side = _row_side(df, i)
        qty = _row_qty(df, i)

        if not sym or side not in ("BUY", "SELL") or qty <= 0:
            skipped += 1
            r = "bad_row"
            skip_reasons[r] = skip_reasons.get(r, 0) + 1
            if args.verbose:
                print(f"[SKIP] {r} sym='{sym}' side='{side.lower() if side else ''}' qty={qty}")
            continue

        try:
            q = broker.get_latest_quote(sym)
        except Exception:
            skipped += 1
            r = "quote_fetch_error"
            skip_reasons[r] = skip_reasons.get(r, 0) + 1
            if args.verbose:
                print(f"[SKIP] {r} {sym}")
            continue

        ok, reason = _quote_ok(q, max_age_min=float(args.max_age_min))
        if not ok:
            skipped += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            if args.verbose:
                print(
                    f"[SKIP] {sym} {reason} ts={q.get('ts')} bid={q.get('bid')} ask={q.get('ask')}"
                )
            continue

        bid = float(q.get("bid") or 0)
        ask = float(q.get("ask") or 0)

        if side == "BUY":
            if ask <= 0:
                skipped += 1
                r = "ask_missing"
                skip_reasons[r] = skip_reasons.get(r, 0) + 1
                if args.verbose:
                    print(f"[SKIP] {sym} {r} bid={bid} ask={ask}")
                continue
            lp = ask
        else:
            if bid <= 0:
                skipped += 1
                r = "bid_missing"
                skip_reasons[r] = skip_reasons.get(r, 0) + 1
                if args.verbose:
                    print(f"[SKIP] {sym} {r} bid={bid} ask={ask}")
                continue
            lp = bid

        # write the repriced value (float -> keeps CSV clean)
        df.at[i, "limit_price"] = float(lp)
        df.at[i, "symbol"] = sym
        df.at[i, "ticker"] = sym

        repriced += 1
        if args.verbose:
            print(f"[REPRICE] {sym} {side} qty={qty} -> limit_price={lp} (ts={q.get('ts')})")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"[OK] Input:  {in_path}")
    print(f"[OK] Output: {out_path}")
    print(f"[OK] Repriced rows: {repriced} | Skipped: {skipped}")

    if args.verbose and skip_reasons:
        items = sorted(skip_reasons.items(), key=lambda x: (-x[1], x[0]))
        print("[INFO] Skip reasons:", ", ".join([f"{k}={v}" for k, v in items]))


if __name__ == "__main__":
    main()
