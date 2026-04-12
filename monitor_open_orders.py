#!/usr/bin/env python3
"""
monitor_open_orders.py
Triton Order Hygiene / Risk Steward

What it does:
- Pull all open orders from Alpaca.
- Decide which ones are "stale" (time or price drift).
- Cancel those safely (mostly unfilled BUYs).
- Log everything to data/results/live_orders.csv for audit/compliance.

Why this matters:
- Triton is now placing bracketed limit BUY orders with stop-loss/take-profit.
- If price runs away and we never actually get filled, those sit in limbo.
- We do NOT want the system thinking we already deployed risk when in reality
  nothing filled. This script closes that loop.

You should run this periodically (ex: every N minutes via Task Scheduler or cron).
"""

import os
import csv
import math
import time
import datetime as dt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import requests


# -------- CONFIG / CONSTANTS -------------------------------------------------

AUDIT_LOG_PATH = Path("data/results/live_orders.csv")

# If an order is older than this, we consider it stale even if price is close
STALE_MINUTES = 30

# If current market price has drifted away from our limit this much (%),
# assume it's unlikely to fill without chasing and mark it stale.
# e.g. 1.0 means 1% away from our limit price
MAX_DRIFT_PCT = 1.0

# How long to sleep between Alpaca API calls that mutate state (just polite)
API_THROTTLE_SEC = 0.15


# -------- UTILITIES ----------------------------------------------------------


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def now_utc_iso() -> str:
    return now_utc().isoformat()


def get_alpaca_headers() -> Dict[str, str]:
    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET")
    if not key or not sec:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in environment.")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": sec,
    }


def get_alpaca_base() -> str:
    # We'll respect ALPACA_ENDPOINT if you've pointed at paper/live
    base = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets").rstrip("/")
    return base


def parse_alpaca_time(ts: str) -> Optional[dt.datetime]:
    """
    Alpaca returns RFC3339-ish timestamps like "2025-10-31T14:23:05.123456Z".
    We convert that to aware UTC datetime.
    """
    if not ts:
        return None
    try:
        # Handle both ...Z and ...+00:00
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return dt.datetime.fromisoformat(ts).astimezone(dt.timezone.utc)
    except Exception:
        return None


def minutes_old(created_at: Optional[dt.datetime]) -> Optional[float]:
    if created_at is None:
        return None
    delta = now_utc() - created_at
    return delta.total_seconds() / 60.0


def fetch_open_orders() -> List[Dict]:
    """
    GET /v2/orders?status=open
    Returns a list of open (unfilled or partially filled) orders.
    We'll keep only the essential stuff we need.
    """
    H = get_alpaca_headers()
    B = get_alpaca_base()
    url = f"{B}/v2/orders?status=open&nested=true"
    resp = requests.get(url, headers=H, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Alpaca /v2/orders error {resp.status_code}: {resp.text[:300]}")

    raw = resp.json()
    out = []
    for order in raw:
        try:
            out.append(
                {
                    "id": order.get("id"),
                    "symbol": order.get("symbol", "").upper(),
                    "side": order.get("side", "").upper(),  # BUY / SELL
                    "qty": float(order.get("qty", 0)),
                    "filled_qty": float(order.get("filled_qty", 0)),
                    "type": order.get("type", "").upper(),  # LIMIT / MARKET / STOP / ...
                    "time_in_force": order.get("time_in_force", ""),
                    "limit_price": safe_float(order.get("limit_price")),
                    "stop_price": safe_float(order.get("stop_price")),
                    "created_at": parse_alpaca_time(order.get("created_at")),
                    "status": order.get("status", ""),
                    "order_class": order.get("order_class", ""),  # 'bracket' etc.
                }
            )
        except Exception:
            # fail-soft on a single row
            continue

    return out


def fetch_latest_quote(symbol: str) -> Optional[float]:
    """
    Pull last trade/quote price so we can see drift vs our limit.
    We'll try /v2/stocks/{symbol}/quotes/latest or /trades/latest.
    Using last trade price as 'current'.
    """
    H = get_alpaca_headers()
    B = os.getenv("ALPACA_MARKETDATA_ENDPOINT", "https://data.alpaca.markets").rstrip("/")

    # Try /v2/stocks/{symbol}/trades/latest
    url = f"{B}/v2/stocks/{symbol}/trades/latest"
    r = requests.get(url, headers=H, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    try:
        return float(data["trade"]["p"])
    except Exception:
        return None


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def drift_pct(limit_price: Optional[float], current_price: Optional[float]) -> Optional[float]:
    """
    For a BUY limit:
    - if current_price >> limit_price (price ran away above us), then drift is big
    Because we're trying to buy cheaper than or equal to limit.
    We define drift as abs(current - limit)/limit * 100%.
    """
    if limit_price is None or current_price is None:
        return None
    if limit_price == 0:
        return None
    return abs((current_price - limit_price) / limit_price) * 100.0


def should_cancel(order: Dict) -> Tuple[bool, str]:
    """
    Decide if we should cancel this open order.

    Rules:
    - We NEVER cancel if it's a SELL. Selling reduces risk, so leave it.
    - We NEVER cancel if it's partially filled (filled_qty > 0 but not fully filled).
      (Later we can get fancier, but for now leave partials alone.)
    - If it's a BUY limit and:
        * older than STALE_MINUTES
        OR
        * price drift > MAX_DRIFT_PCT
      => cancel it.

    Returns (bool,reason)
    """
    side = order["side"]
    filled_qty = order["filled_qty"]
    qty = order["qty"]
    created_at = order["created_at"]
    otype = order["type"]
    limit_price = order["limit_price"]

    # Sell orders are "risk off" or cleanup → don't block
    if side == "SELL":
        return (False, "SELL order, skip")

    # If we somehow got a MARKET order still open, that's weird. Leave it alone for now.
    if otype != "LIMIT":
        return (False, f"Non-LIMIT type={otype}, skip")

    # If we already got some fill but not done, let Alpaca finish
    # (We could handle partials in v2, but it's safer to not interrupt fills right now.)
    if filled_qty > 0 and filled_qty < qty:
        return (False, "PARTIAL fill in progress, skip")

    age_min = minutes_old(created_at)
    age_reason = ""
    old_enough = False
    if age_min is not None and age_min >= STALE_MINUTES:
        old_enough = True
        age_reason = f"age={age_min:.1f}m>=STALE_MINUTES({STALE_MINUTES})"

    # Drift check
    cur_px = fetch_latest_quote(order["symbol"])
    d_pct = drift_pct(limit_price, cur_px)
    drift_reason = ""
    drift_bad = False
    if d_pct is not None and d_pct >= MAX_DRIFT_PCT:
        drift_bad = True
        drift_reason = f"drift={d_pct:.2f}%>=MAX_DRIFT_PCT({MAX_DRIFT_PCT}%)"

    if old_enough or drift_bad:
        combined = ", ".join(x for x in [age_reason, drift_reason] if x)
        if not combined:
            combined = "stale_condition"
        return (True, combined)

    return (False, "fresh enough")


def cancel_order(order_id: str) -> Tuple[bool, str]:
    """
    Cancel an order in Alpaca:
    DELETE /v2/orders/{order_id}
    """
    H = get_alpaca_headers()
    B = get_alpaca_base()
    url = f"{B}/v2/orders/{order_id}"
    r = requests.delete(url, headers=H, timeout=10)

    # Alpaca returns 204 No Content on success typically
    if r.status_code in (200, 204):
        return (True, "cancelled")
    else:
        return (False, f"{r.status_code}: {r.text[:200]}")


def write_audit_row(row: Dict):
    """
    Append a single audit row to live_orders.csv
    We'll keep same columns used by place_orders_from_csv.py so compliance is consistent:

    timestamp, symbol, side, qty, order_type, tif,
    limit_price, stop_price, take_profit, status, note
    """
    is_new = not AUDIT_LOG_PATH.exists()
    with AUDIT_LOG_PATH.open("a", newline="") as f:
        fieldnames = [
            "timestamp",
            "symbol",
            "side",
            "qty",
            "order_type",
            "tif",
            "limit_price",
            "stop_price",
            "take_profit",
            "status",
            "note",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        w.writerow(row)


# -------- MAIN LOGIC ---------------------------------------------------------


def main():
    print(f"[INFO] monitor_open_orders.py start {now_utc_iso()}")
    print(f"[INFO] STALE_MINUTES={STALE_MINUTES}, MAX_DRIFT_PCT={MAX_DRIFT_PCT}%")

    # 1. Fetch all open orders
    try:
        open_orders = fetch_open_orders()
    except Exception as e:
        print(f"[ERR ] Failed to fetch open orders: {e}")
        return

    if not open_orders:
        print("[INFO] No open orders.")
        return

    print(f"[INFO] {len(open_orders)} open orders found.")

    cancel_attempts = 0
    cancel_success = 0
    cancel_fail = 0

    # 2. For each open order, decide if we should cancel
    for od in open_orders:
        oid = od["id"]
        sym = od["symbol"]
        side = od["side"]
        qty = od["qty"]
        tif = od["time_in_force"]
        limit_px = od["limit_price"]
        stop_px = od.get("stop_price")
        order_type = od["type"]
        created_at = od["created_at"]
        filled_qty = od["filled_qty"]
        ord_class = od["order_class"]

        created_iso = created_at.isoformat() if created_at else "?"
        print(
            f"[DBG] {oid} {sym} {side} qty={qty} filled={filled_qty} "
            f"type={order_type} tif={tif} limit={limit_px} class={ord_class} "
            f"created={created_iso}"
        )

        do_cancel, reason = should_cancel(od)
        if not do_cancel:
            print(f"[KEEP] {sym} ({side}) -> {reason}")
            continue

        # 3. Cancel
        cancel_attempts += 1
        ok, note = cancel_order(oid)
        status_str = "CANCELLED" if ok else "CANCEL_ERR"
        if ok:
            cancel_success += 1
            print(f"[CANCEL-OK ] {sym} ({side}) qty={qty} reason={reason}")
        else:
            cancel_fail += 1
            print(f"[CANCEL-ERR] {sym} ({side}) qty={qty} reason={reason} err={note}")

        # 4. Audit append
        audit_row = {
            "timestamp": now_utc_iso(),
            "symbol": sym,
            "side": side,
            "qty": f"{qty}",
            "order_type": order_type.lower(),
            "tif": tif,
            "limit_price": f"{limit_px:.2f}" if limit_px is not None else "",
            "stop_price": f"{stop_px:.2f}" if stop_px is not None else "",
            "take_profit": "",  # we don't re-log bracket legs here; this is cleanup
            "status": status_str,
            "note": f"{reason}; {note}",
        }
        write_audit_row(audit_row)

        time.sleep(API_THROTTLE_SEC)

    print(
        f"[INFO] Done. Attempts={cancel_attempts}, "
        f"success={cancel_success}, fail={cancel_fail}."
    )
    print(f"[INFO] monitor_open_orders.py end {now_utc_iso()}")


if __name__ == "__main__":
    main()
