# scripts/postfill_attach_oco.py
# Attach OCO exits (TP/SL) to today's BUY fills that don't already have exits.
# Usage examples:
#   python scripts/postfill_attach_oco.py --tp 0.08 --sl 0.05 --qty 1
#   python scripts/postfill_attach_oco.py --since "2025-11-05T10:00:00Z" --tp 0.08 --sl 0.05
#   python scripts/postfill_attach_oco.py --symbols NVDA,AMZN --tp 0.08 --sl 0.05 --qty 1

import argparse
import datetime as dt
import os
import sys
import time
from typing import Dict, List, Optional

import requests


def alpaca_base() -> str:
    return os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")


def alpaca_headers() -> Dict[str, str]:
    key = os.getenv("APCA_API_KEY_ID")
    sec = os.getenv("APCA_API_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def get_fill_activities_since(base: str, hdrs: Dict[str, str], after_iso: str) -> List[Dict]:
    """Fetch account fill activities since an ISO timestamp (UTC)."""
    r = requests.get(
        f"{base}/v2/account/activities/FILL",
        params={"after": after_iso, "direction": "asc"},
        headers=hdrs,
        timeout=30,
    )
    r.raise_for_status()
    acts = r.json()
    if isinstance(acts, dict):
        acts = [acts]
    return acts


def get_position(base: str, hdrs: Dict[str, str], sym: str) -> Optional[Dict]:
    r = requests.get(f"{base}/v2/positions/{sym}", headers=hdrs, timeout=15)
    return r.json() if r.status_code == 200 else None


def has_open_sell(base: str, hdrs: Dict[str, str], sym: str) -> bool:
    """True if ANY open sell order exists for the symbol (parent or leg)."""
    r = requests.get(
        f"{base}/v2/orders",
        params={"status": "open", "nested": "true", "symbols": sym, "limit": 500},
        headers=hdrs,
        timeout=30,
    )
    r.raise_for_status()
    for o in r.json():
        if str(o.get("side")).lower() == "sell":
            return True
        # also scan nested legs if present
        for leg in o.get("legs") or []:
            if str(leg.get("side")).lower() == "sell":
                return True
    return False


def attach_oco_for_symbol(
    base: str,
    hdrs: Dict[str, str],
    sym: str,
    qty: int,
    tp_pct: float,
    sl_pct: float,
    tif: str = "gtc",
    dry_run: bool = False,
) -> Dict:
    pos = get_position(base, hdrs, sym)
    if not pos:
        return {"symbol": sym, "attached": False, "reason": "no position"}

    if has_open_sell(base, hdrs, sym):
        return {"symbol": sym, "attached": False, "reason": "sell leg already open"}

    try:
        avg = float(pos["avg_entry_price"])
    except Exception:
        return {"symbol": sym, "attached": False, "reason": "avg_entry_price missing"}

    tp = round(avg * (1 + tp_pct), 2)
    sl = round(avg * (1 - sl_pct), 2)

    payload = {
        "symbol": sym,
        "qty": str(int(qty)),
        "side": "sell",
        "type": "limit",  # REQUIRED for OCO parent
        "time_in_force": tif,
        "order_class": "oco",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss": {"stop_price": f"{sl:.2f}"},
    }

    if dry_run:
        return {"symbol": sym, "attached": True, "dry_run": True, "payload": payload}

    r = requests.post(f"{base}/v2/orders", headers=hdrs, json=payload, timeout=30)
    ok = r.status_code in (200, 201)
    return {
        "symbol": sym,
        "attached": ok,
        "status": r.status_code,
        "resp": r.text[:300],
        "tp": tp,
        "sl": sl,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attach OCO exits to today's BUY fills.")
    p.add_argument("--tp", type=float, default=0.08, help="Take-profit pct (e.g. 0.08)")
    p.add_argument("--sl", type=float, default=0.05, help="Stop-loss pct (e.g. 0.05)")
    p.add_argument("--qty", type=int, default=1, help="Exit quantity per symbol")
    p.add_argument("--tif", default="gtc", help="time_in_force for exits (default: gtc)")
    p.add_argument("--since", default=None, help="ISO start time (default: today 00:00Z)")
    p.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated whitelist of symbols to consider (e.g. NVDA,AMZN)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print intended actions only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = alpaca_base()
    hdrs = alpaca_headers()

    # Determine starting timestamp
    if args.since:
        after = args.since
    else:
        after = dt.datetime.now(dt.timezone.utc).date().isoformat() + "T00:00:00Z"

    # Optional user-provided filter
    user_syms = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    )

    # Gather BUY fills
    acts = get_fill_activities_since(base, hdrs, after)
    buy_syms: List[str] = []
    for a in acts:
        # Expect keys: type (fill/partial_fill), side (buy/sell), symbol
        if (
            str(a.get("type")).lower() in ("fill", "partial_fill")
            and str(a.get("side")).lower() == "buy"
        ):
            sym = (a.get("symbol") or "").upper()
            if not sym:
                continue
            if user_syms and sym not in user_syms:
                continue
            buy_syms.append(sym)

    # De-dupe in order
    seen = set()
    queue: List[str] = []
    for s in buy_syms:
        if s not in seen:
            queue.append(s)
            seen.add(s)

    if not queue:
        print("No qualifying BUY fills found since", after)
        return

    results = []
    for sym in queue:
        res = attach_oco_for_symbol(
            base,
            hdrs,
            sym,
            qty=args.qty,
            tp_pct=args.tp,
            sl_pct=args.sl,
            tif=args.tif,
            dry_run=args.dry_run,
        )
        results.append(res)
        print(res)
        # small spacing to be polite to the API
        time.sleep(0.15)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", repr(e))
        sys.exit(1)
