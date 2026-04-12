"""CLI gate self-test.

Place at: services/check_live_gate.py

Usage (from TRITON root):
  python -m services.check_live_gate --mode live --ttl 10 --fresh 180 --with-broker

Exit codes:
  0 = ready
  1 = blocked
"""

from __future__ import annotations

import argparse
import sys

from services.live_gate import compute_live_gates


def print_summary(live_ready, gates):
    print("LIVE_READY:", "YES" if live_ready else "NO")
    for g in gates:
        ok = bool(g.get("ok"))
        name = g.get("name", "?")
        detail = g.get("detail", "")
        meta = g.get("meta") or {}
        flag = "OK" if ok else "BLOCK"
        if meta:
            print(f"  [{flag}] {name}: {detail} | {meta}")
        else:
            print(f"  [{flag}] {name}: {detail}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="live", help="live|paper (live evaluates live gates)")
    ap.add_argument("--ttl", type=int, default=10, help="confirm TTL minutes")
    ap.add_argument("--fresh", type=int, default=180, help="freshness max age minutes")
    ap.add_argument("--no-market", action="store_true", help="skip market-open requirement")
    ap.add_argument(
        "--with-broker",
        action="store_true",
        help="instantiate AlpacaBroker to verify MARKET via clock",
    )
    args = ap.parse_args(argv)

    broker = None
    if args.with_broker:
        try:
            from services.broker_alpaca import AlpacaBroker  # type: ignore

            broker = AlpacaBroker(mode=args.mode)
        except Exception as e:
            print(f"WARN: could not initialize broker for MARKET check: {e}")
            broker = None

    live_ready, gates = compute_live_gates(
        confirm_ttl_minutes=args.ttl,
        freshness_max_age_minutes=args.fresh,
        require_market_open=not args.no_market,
        broker=broker,
    )
    print_summary(live_ready, gates)
    return 0 if live_ready else 1


if __name__ == "__main__":
    sys.exit(main())
