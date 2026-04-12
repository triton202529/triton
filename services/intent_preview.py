# services/intent_preview.py
"""
TRITON — Intent Preview (Position-Aware Preflight)

Reads orders CSV (orders_today.csv) and current broker positions, then prints a preflight report:
- current position qty
- planned side/qty/limit
- projected qty after order
- flags: illegal sells, duplicates, oversized adds (soft)

Supports your repo's actual schema:
  ticker,side,qty,close,target_notional,time

Also supports:
  symbol/sym, limit_price/limit/close/price, etc.

Exit codes:
- 0: OK (or warnings only)
- 2: Strict failure (illegal sells / duplicates / parse failures)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_ORDERS_PATH = Path("orders_today.csv")


@dataclass
class PlannedOrder:
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    limit_price: Optional[float] = None


def _to_float(x: str, default: float = 0.0) -> float:
    try:
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _norm_side(x: str) -> str:
    s = (x or "").strip().lower()
    if s in ("buy", "b"):
        return "buy"
    if s in ("sell", "s"):
        return "sell"
    return s


def _clean_key(k: str) -> str:
    """
    Normalize header keys:
    - strip whitespace
    - lowercase
    - remove BOM if present
    """
    if k is None:
        return ""
    k2 = k.strip().lower()
    # Remove common UTF-8 BOM artifacts if present
    k2 = k2.lstrip("\ufeff").lstrip("ï»¿")
    return k2


def _normalize_row_keys(row: Dict[str, str]) -> Dict[str, str]:
    return {_clean_key(k): v for k, v in row.items() if k is not None}


def load_orders_csv(path: Path) -> Tuple[List[PlannedOrder], List[str]]:
    errors: List[str] = []
    orders: List[PlannedOrder] = []

    if not path.exists():
        return [], [f"[MISSING] orders file not found: {path.resolve()}"]

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return [], [f"[INVALID] empty header in {path.name}"]

            # normalize fieldnames once
            fieldnames = [_clean_key(x) for x in (reader.fieldnames or [])]
            # csv.DictReader keeps original keys; we normalize per-row below.

            for i, raw_row in enumerate(reader, start=2):
                # handle blank lines (DictReader gives {None:[...]} or all empty)
                if raw_row is None:
                    continue

                row = _normalize_row_keys(raw_row)

                # If this is a blank row, skip (common at EOF)
                if all((str(v).strip() == "" for v in row.values())):
                    continue

                # Support symbol/sym/ticker columns
                sym = (
                    (
                        row.get("symbol")
                        or row.get("sym")
                        or row.get("ticker")
                        or row.get("asset")
                        or row.get("security")
                        or ""
                    )
                    .strip()
                    .upper()
                )

                side = _norm_side(
                    row.get("side") or row.get("action") or row.get("order_side") or ""
                )
                qty = _to_float(row.get("qty") or row.get("quantity") or "0")

                # Support close->limit_price, or limit/price columns
                lp_raw = (
                    row.get("limit_price")
                    or row.get("limit")
                    or row.get("close")
                    or row.get("price")
                    or row.get("limitprice")
                    or ""
                )
                limit_price = None
                if str(lp_raw).strip() != "":
                    val = _to_float(str(lp_raw), default=0.0)
                    limit_price = val if val > 0 else None

                if not sym:
                    errors.append(f"[ROW {i}] missing symbol/ticker")
                    continue
                if side not in ("buy", "sell"):
                    errors.append(f"[ROW {i}] invalid side={row.get('side')!r}")
                    continue
                if qty <= 0:
                    errors.append(f"[ROW {i}] invalid qty={row.get('qty')!r}")
                    continue

                orders.append(PlannedOrder(symbol=sym, side=side, qty=qty, limit_price=limit_price))

    except Exception as e:
        return [], [f"[READ_FAIL] {path.name}: {e!r}"]

    return orders, errors


def get_positions(mode: str) -> Dict[str, float]:
    from services.broker_alpaca import AlpacaBroker  # type: ignore

    b = AlpacaBroker(mode=mode)
    ps = b.list_positions() or []
    out: Dict[str, float] = {}
    for p in ps:
        sym = (p.get("symbol") or "").strip().upper()
        qty = _to_float(p.get("qty") or "0")
        if sym:
            out[sym] = qty
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="TRITON intent preview (orders vs positions).")
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--orders", default=str(DEFAULT_ORDERS_PATH))
    ap.add_argument(
        "--warn-large-qty",
        type=float,
        default=10.0,
        help="Warn if abs(current position qty) >= this and we are adding more (soft warning).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail (exit 2) if illegal sells, duplicates, or CSV errors are found.",
    )
    ap.add_argument(
        "--top", type=int, default=30, help="Max rows to print (sorted by abs(current_qty) desc)."
    )
    args = ap.parse_args()

    orders_path = Path(args.orders)
    orders, csv_errors = load_orders_csv(orders_path)

    if csv_errors:
        print("❌ Orders CSV issues:")
        for e in csv_errors:
            print("  ", e)

    if not orders:
        print("No valid orders to preview.")
        return 2 if args.strict else 0

    # Detect duplicates (same symbol appears multiple times)
    counts: Dict[str, int] = {}
    for o in orders:
        counts[o.symbol] = counts.get(o.symbol, 0) + 1
    duplicates = [sym for sym, n in counts.items() if n > 1]

    positions = get_positions(args.mode)

    rows = []
    illegal_sells: List[str] = []
    large_add_warnings: List[str] = []

    for o in orders:
        cur = positions.get(o.symbol, 0.0)
        projected = cur + o.qty if o.side == "buy" else cur - o.qty

        flags: List[str] = []

        # illegal sell (no shorts): selling more than you hold long
        if o.side == "sell" and o.qty > cur + 1e-9:
            flags.append("ILLEGAL_SELL")
            illegal_sells.append(o.symbol)

        # large add soft warning (only for buys)
        if o.side == "buy" and abs(cur) >= float(args.warn_large_qty):
            flags.append("LARGE_POS_ADD")
            large_add_warnings.append(o.symbol)

        if o.symbol in duplicates:
            flags.append("DUPLICATE_SYMBOL")

        rows.append((o.symbol, cur, o.side, o.qty, o.limit_price, projected, ",".join(flags)))

    # Sort by abs(current position) desc, then symbol
    rows.sort(key=lambda r: (abs(r[1]), r[0]), reverse=True)

    print("\nTRITON — Intent Preview")
    print(f"Mode           : {args.mode}")
    print(f"Orders file     : {orders_path.resolve()}")
    print(f"Positions loaded: {len(positions)}")
    print(f"Orders planned  : {len(orders)}\n")

    header = f"{'Symbol':<6} {'PosQty':>10} {'Side':>6} {'OrdQty':>10} {'Limit':>10} {'ProjQty':>10} {'Flags':>18}"
    print(header)
    print("-" * len(header))

    shown = 0
    for sym, cur, side, qty, lp, proj, flags in rows:
        if shown >= int(args.top):
            break
        lp_str = "" if lp is None else f"{lp:.2f}"
        print(
            f"{sym:<6} {cur:>10.4f} {side:>6} {qty:>10.4f} {lp_str:>10} {proj:>10.4f} {flags:>18}"
        )
        shown += 1

    print("\nSummary")
    print(
        f"- duplicates: {len(duplicates)}" + (f" → {sorted(set(duplicates))}" if duplicates else "")
    )
    print(
        f"- illegal sells: {len(set(illegal_sells))}"
        + (f" → {sorted(set(illegal_sells))}" if illegal_sells else "")
    )
    print(
        f"- large-pos adds (soft): {len(set(large_add_warnings))}"
        + (f" → {sorted(set(large_add_warnings))}" if large_add_warnings else "")
    )

    hard_fail = bool(csv_errors) or bool(duplicates) or bool(illegal_sells)
    if args.strict and hard_fail:
        print("\n❌ STRICT FAIL: Fix issues above before placement.")
        return 2

    if hard_fail:
        print("\n⚠️  WARN: Issues found (run with --strict to hard-fail).")
    else:
        print("\n✅ OK: No hard issues detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
