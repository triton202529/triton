# services/preflight_place.py
"""
TRITON — Preflight + Place (Safe Wrapper)

Flow:
  1) intent_preview --strict
  2) place_live_orders --dry-run (truth pass)
  3) place_live_orders (real)

Notes:
- With --cancel-duplicates, dry-run cannot cancel broker orders, so it may show a
  smaller planned count than the real run. Real run can cancel and then place.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_ORDERS = Path("data/live/orders_today.csv")


def run(cmd: List[str], verbose: bool = False) -> int:
    if verbose:
        print("CMD:", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="TRITON preflight intent preview then place orders safely."
    )
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--orders", default=str(DEFAULT_ORDERS))
    ap.add_argument(
        "--session", required=True, help="Placement session tag (client_order_id uniqueness)."
    )
    ap.add_argument("--log-session", required=True, help="Stable grouping tag for logs.")
    ap.add_argument("--warn-large-qty", type=float, default=10.0)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--verbose", action="store_true")

    # Placement guardrails
    ap.add_argument("--cancel-duplicates", action="store_true", default=True)
    ap.add_argument("--block-on-duplicates", action="store_true", default=False)
    ap.add_argument("--drop-illegal-sells", action="store_true", default=True)
    ap.add_argument("--allow-shorts", action="store_true", default=False)

    ap.add_argument("--require-marketdata", action="store_true", default=False)
    ap.add_argument("--max-limit-deviation-pct", type=float, default=None)
    ap.add_argument("--open-buffer-pct", type=float, default=None)
    ap.add_argument("--time-in-force", default=None)

    # ✅ NEW: explicit batch notional cap
    ap.add_argument("--max-batch-notional", type=float, default=None)

    ap.add_argument("--ignore-market-closed", action="store_true", default=False)

    args = ap.parse_args()

    orders_path = Path(args.orders)
    if not orders_path.exists():
        print(f"❌ Orders file missing: {orders_path.resolve()}")
        return 2

    py = sys.executable

    # 1) STRICT intent preview gate
    preview_cmd = [
        py,
        "-m",
        "services.intent_preview",
        "--mode",
        args.mode,
        "--orders",
        str(orders_path),
        "--top",
        str(args.top),
        "--warn-large-qty",
        str(args.warn_large_qty),
        "--strict",
    ]
    rc = run(preview_cmd, verbose=args.verbose)
    if rc != 0:
        print("\n🚫 PRECHECK BLOCK: intent_preview strict failed — placement aborted.")
        return rc

    # Build base place_live_orders command
    base_place = [
        py,
        "-m",
        "services.place_live_orders",
        "--mode",
        args.mode,
        "--orders",
        str(orders_path),
        "--session",
        args.session,
        "--log-session",
        args.log_session,
        "--verbose",
    ]

    if args.cancel_duplicates:
        base_place.append("--cancel-duplicates")
    if args.block_on_duplicates:
        base_place.append("--block-on-duplicates")
    if args.drop_illegal_sells:
        base_place.append("--drop-illegal-sells")
    if args.allow_shorts:
        base_place.append("--allow-shorts")

    if args.require_marketdata:
        base_place.append("--require-marketdata")
    if args.ignore_market_closed:
        base_place.append("--ignore-market-closed")

    if args.max_batch_notional is not None:
        base_place += ["--max-batch-notional", str(args.max_batch_notional)]
    if args.max_limit_deviation_pct is not None:
        base_place += ["--max-limit-deviation-pct", str(args.max_limit_deviation_pct)]
    if args.open_buffer_pct is not None:
        base_place += ["--open-buffer-pct", str(args.open_buffer_pct)]
    if args.time_in_force is not None:
        base_place += ["--time-in-force", str(args.time_in_force)]

    # 2) Truth pass: DRY-RUN
    dry_cmd = base_place + ["--dry-run"]
    print("\n🔎 DRY-RUN (truth pass): what would actually be submitted?\n")
    rc_dry = run(dry_cmd, verbose=args.verbose)
    if rc_dry != 0:
        print("\n🚫 DRY-RUN BLOCK: placement aborted.")
        return rc_dry

    # 3) Real placement (must override place_live_orders default dry-run)
    print("\n✅ PRECHECK PASS: placing orders…\n")
    return run(base_place + ["--no-dry-run"], verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
