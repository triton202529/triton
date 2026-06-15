# scripts/chase_limit.py
import argparse
import decimal as D
import os
import sys
import time
from pathlib import Path

import requests


def _deprecation_guard() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from services.execution_path_guard import exit_if_deprecated

    exit_if_deprecated(__file__)


def base():
    return os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")


def hdrs():
    return {
        "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID"),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY"),
    }


def latest_open_limit_for(b, h, sym):
    r = requests.get(
        f"{b}/v2/orders",
        params={"status": "open", "nested": "true", "symbols": sym, "limit": 200},
        headers=h,
        timeout=15,
    )
    r.raise_for_status()
    cands = [o for o in r.json() if o["side"] == "buy" and o["type"] == "limit"]
    return max(cands, key=lambda x: x.get("created_at", "")) if cands else None


def main():
    _deprecation_guard()
    p = argparse.ArgumentParser(description="Bump a BUY limit order up to a ceiling.")
    p.add_argument("--symbol", required=True)
    p.add_argument("--step", type=str, default="0.05", help="increment per bump")
    p.add_argument("--ceil", type=str, required=True, help="max limit price to allow")
    p.add_argument("--sleep", type=int, default=10, help="seconds between bumps")
    p.add_argument("--max-bumps", type=int, default=20)
    args = p.parse_args()

    b, h = base(), hdrs()
    step = D.Decimal(args.step)
    ceil = D.Decimal(args.ceil)

    for i in range(args.max_bumps):
        o = latest_open_limit_for(b, h, args.symbol.upper())
        if not o:
            print("No open limit found — done.")
            return
        cur = D.Decimal(o["limit_price"])
        if cur >= ceil:
            print("Reached ceiling — stopping.")
            return

        # cancel & re-place with +step
        requests.delete(f"{b}/v2/orders/{o['id']}", headers=h, timeout=10)
        new_lim = cur + step
        payload = {
            "symbol": o["symbol"],
            "qty": o["qty"],
            "side": "buy",
            "type": "limit",
            "limit_price": str(new_lim),
            "time_in_force": "gtc",
        }
        rr = requests.post(f"{b}/v2/orders", headers=h, json=payload, timeout=15)
        print(f"bump {i+1}: {o['symbol']} -> {new_lim} ({rr.status_code})")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
