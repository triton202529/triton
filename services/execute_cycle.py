# services/execute_cycle.py
"""
TRITON — Market Open Cycle Runner (paper/live)

This is the "market open cycle" runner. It is NOT the modular orchestrator.
It performs:

1) List open orders (broker)
2) Optional cancel open orders (--cancel-open)
3) Check for pending_cancel / pending_replace (broker)
4) Place orders using services.place_live_orders (with refresh-orders option)
5) Poll using services.poll_order_status for N rounds with sleep

CRITICAL FIX (2026-01-29):
- Removes accidental recursion: this runner must NEVER call `python -m services.execute_cycle`
  from inside itself (that caused infinite loops).

PATCH (2026-02-02):
- Add MARKET-CLOSED GUARD:
  - Block cancel-open when market is closed (unless --ignore-market-closed)
  - Block placement when market is closed (unless --ignore-market-closed)
  - Polling/listing remain allowed while closed (safe read-only)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

_EXECUTE_CYCLE_ROOT = Path(__file__).resolve().parents[1]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_order_line(o: Dict[str, Any]) -> None:
    sym = (o.get("symbol") or "").strip()
    side = (o.get("side") or "").strip().lower()
    otype = (o.get("type") or o.get("order_type") or "").strip().lower()
    status = (o.get("status") or "").strip().lower()
    oid = (o.get("id") or o.get("order_id") or "").strip()
    tif = (o.get("time_in_force") or "").strip().lower()
    lp = o.get("limit_price", "")
    try:
        lp = float(lp) if lp not in ("", None) else ""
    except Exception:
        pass
    if lp != "":
        print(f"  {sym} {side} {otype} {status} id={oid} limit={lp} tif={tif}")
    else:
        print(f"  {sym} {side} {otype} {status} id={oid} tif={tif}")


def list_open_orders(broker, mode: str, limit: int) -> List[Dict[str, Any]]:
    try:
        oo = broker.list_orders(status="open", nested=True, limit=limit) or []
        print(f"open={len(oo)} (mode={mode})")
        for o in oo[:25]:
            _print_order_line(o)
        if len(oo) > 25:
            print(f"  ... +{len(oo) - 25} more")
        return oo
    except Exception as e:
        print(f"[WARN] list_open_orders failed: {e}")
        return []


def cancel_open_orders(broker, open_orders: List[Dict[str, Any]]) -> Tuple[int, int]:
    cancelled = 0
    failed = 0
    for o in open_orders:
        oid = (o.get("id") or o.get("order_id") or "").strip()
        if not oid:
            continue
        try:
            broker.cancel_order(oid)
            cancelled += 1
        except Exception:
            failed += 1
    return cancelled, failed


def check_pending_cancel_replace(broker, limit: int = 500) -> int:
    """
    Broker-based check for stuck orders in pending_cancel / pending_replace.
    """
    pending = 0
    try:
        all_orders = broker.list_orders(status="all", nested=True, limit=limit) or []
    except Exception:
        # If broker doesn't support status="all", fall back to open only.
        all_orders = broker.list_orders(status="open", nested=True, limit=limit) or []

    for o in all_orders:
        st = (o.get("status") or "").strip().lower()
        if st in ("pending_cancel", "pending_replace"):
            pending += 1
    return pending


def run_cmd(cmd: List[str]) -> int:
    return subprocess.call(cmd)


def _get_clock_safe(broker) -> Dict[str, Any]:
    """
    Best-effort broker clock. If unavailable, return 'unknown' clock.
    """
    try:
        c = broker.get_clock()
        if isinstance(c, dict):
            return c
    except Exception:
        pass
    return {
        "is_open": None,
        "next_open": None,
        "next_close": None,
        "timestamp": None,
    }


def _boolish(v: Any) -> Optional[bool]:
    if v is True or v is False:
        return v
    if v is None:
        return None
    # Sometimes APIs return strings
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y"):
            return True
        if s in ("false", "0", "no", "n"):
            return False
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="TRITON Market Open Cycle (place + poll) with safe defaults",
        allow_abbrev=False,  # ✅ prevents --poll ambiguity forever
    )
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")

    # Keep your existing CLI from the usage output:
    ap.add_argument("--cancel-open", action="store_true")
    ap.add_argument("--poll-rounds", type=int, default=6)
    ap.add_argument("--poll-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--session", type=str, default=None)
    ap.add_argument("--refresh-orders", action="store_true")
    ap.add_argument("--no-refresh-orders", action="store_true")
    ap.add_argument("--drop-illegal-sells", action="store_true")
    ap.add_argument("--no-drop-illegal-sells", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    # ✅ New: market-closed guard override
    ap.add_argument(
        "--ignore-market-closed",
        action="store_true",
        help="Override safety: allow cancel/place when market is closed.",
    )

    args = ap.parse_args()

    mode = args.mode
    base_session = (args.session or f"OPEN_{mode}_{utc_stamp()}").strip()
    placement_session = base_session

    # If refresh-orders is enabled, create a unique placement session to avoid collisions,
    # but keep log grouping stable via --log-session.
    use_refresh = bool(args.refresh_orders) and (not bool(args.no_refresh_orders))
    if use_refresh:
        placement_session = f"{base_session}_R{utc_stamp().split('_')[-1]}"

    drop_illegal = bool(args.drop_illegal_sells) and (not bool(args.no_drop_illegal_sells))

    print("")
    print("TRITON — Market Open Cycle")
    print(f"Mode          : {mode}")
    print(f"Base session  : {base_session}")
    print(f"As of (UTC)   : {now_iso()}")
    print("")

    # Broker init
    try:
        from services.broker_alpaca import AlpacaBroker  # type: ignore

        broker = AlpacaBroker(mode=mode)
    except Exception as e:
        print(f"[BLOCK] Broker init failed: {e}")
        raise SystemExit(2)

    # Clock (best-effort)
    clock = _get_clock_safe(broker)
    is_open = _boolish(clock.get("is_open"))
    if is_open is True:
        print("[CLOCK] Market: OPEN")
    elif is_open is False:
        print(
            f"[CLOCK] Market: CLOSED | next_open={clock.get('next_open')} next_close={clock.get('next_close')}"
        )
    else:
        print("[CLOCK] Market: UNKNOWN (clock unavailable)")
    print("")

    # 1) List open orders
    print("=== 1) List Open Orders (broker) ===")
    open_orders = list_open_orders(broker, mode=mode, limit=args.limit)
    print("")

    # MARKET-CLOSED GUARD: cancel-open
    if args.cancel_open:
        if is_open is False and not args.ignore_market_closed:
            print("[BLOCK] --cancel-open requested but market is CLOSED.")
            print("        Refusing to cancel overnight (prevents thrash).")
            print("        Use --ignore-market-closed to override intentionally.")
            raise SystemExit(3)

    # 2) Optional cancel open orders
    if args.cancel_open and open_orders:
        print("=== 2) Cancel Open Orders (broker) ===")
        cancelled, failed = cancel_open_orders(broker, open_orders)
        print(f"found open={len(open_orders)}")
        print(f"cancelled={cancelled} failed={failed}")
        print("")
        # small pause + relist
        time.sleep(1.0)
        print("=== 1) List Open Orders (broker) ===")
        open_orders = list_open_orders(broker, mode=mode, limit=args.limit)
        print("")

    # 3) Check pending cancel/replace
    print("=== 3) Check Pending Cancel/Replace (broker) ===")
    pending = check_pending_cancel_replace(broker, limit=500)
    if pending == 0:
        print("pending=0 (good)")
    else:
        print(f"pending={pending} (WARN)")
    print(f"Placement session (expected): {placement_session}")
    print("")

    # MARKET-CLOSED GUARD: placement
    # If market is known CLOSED and user didn't explicitly override, block placement.
    # Polling/listing are safe and remain allowed.
    if is_open is False and not args.ignore_market_closed:
        print("[BLOCK] Market is CLOSED — placement disabled by default.")
        print("        This prevents overnight cancel→re-place loops.")
        print("        Use --ignore-market-closed to force placement intentionally.")
        print("")
    else:
        # 4) Place (via place_live_orders) — IMPORTANT: do NOT call this module again.
        print("=== 4) Place Orders (place_live_orders) ===")
        if not args.dry_run:
            try:
                from services.master_execution_gate import (
                    MasterExecutionGate,
                    append_gate_log_csv,
                    write_snapshot,
                )

                _dec = MasterExecutionGate(project_root=_EXECUTE_CYCLE_ROOT).evaluate(
                    mode=mode,
                    broker=broker,
                    verbose=args.verbose,
                    require_market_open=(
                        False if (mode == "live" and args.ignore_market_closed) else None
                    ),
                )
                write_snapshot(_dec)
                append_gate_log_csv(_dec)
                if not _dec.ok:
                    print(f"[MASTER_GATE_BLOCK] {_dec.summary}")
                    for _r in _dec.reasons:
                        print(f"  reason={_r}")
                    raise SystemExit(2)
            except SystemExit:
                raise
            except Exception as _e:
                print(f"[MASTER_GATE] evaluation error: {_e}")
                raise SystemExit(1)

        cmd = [
            sys.executable,
            "-m",
            "services.place_live_orders",
            "--mode",
            mode,
            "--session",
            placement_session,  # unique if refresh-orders
            "--log-session",
            base_session,  # stable grouping for live_orders_log and summaries
        ]
        if use_refresh:
            cmd.append("--cancel-duplicates")
        if drop_illegal:
            cmd.append("--drop-illegal-sells")
        if args.dry_run:
            cmd.append("--dry-run")
        else:
            cmd.append("--no-dry-run")
        if args.verbose:
            cmd.append("--verbose")

        if args.verbose:
            print("CMD:", " ".join(cmd))

        rc = run_cmd(cmd)
        if rc != 0:
            print(f"[WARN] place_live_orders rc={rc} (continuing to poll anyway).")
        print("")

    # 5) Poll loop
    rounds = max(0, int(args.poll_rounds))
    every = max(1, int(args.poll_every))

    if rounds == 0:
        print("[INFO] poll-rounds=0 -> skipping poll.")
        print("")
    else:
        for i in range(rounds):
            print(f"=== 5) Poll Orders ({i+1}/{rounds}) ===")
            poll_cmd = [
                sys.executable,
                "-m",
                "services.poll_order_status",
                "--mode",
                mode,
                "--session",
                base_session,
                "--refresh",
            ]
            if args.verbose:
                print("CMD:", " ".join(poll_cmd))
            prc = run_cmd(poll_cmd)
            if prc != 0:
                print(f"[WARN] poll_order_status rc={prc}")

            # Optional early-exit: if nothing open and nothing pending, stop polling.
            oo = []
            try:
                oo = broker.list_orders(status="open", nested=True, limit=50) or []
            except Exception:
                oo = []
            pend2 = 0
            try:
                pend2 = check_pending_cancel_replace(broker, limit=200)
            except Exception:
                pend2 = 0

            if len(oo) == 0 and pend2 == 0:
                # 2 consecutive quiet polls is safer; but keep it simple:
                # if we're already beyond the first poll and still quiet, exit early.
                if i >= 1:
                    print("[EARLY_EXIT] open=0 and pending=0 -> stopping poll loop early.")
                    print("")
                    break

            if i < rounds - 1:
                time.sleep(every)
            print("")

    print("========================")
    print("Market Open Cycle Summary")
    print("========================")
    print(f"[OK ] Completed cycle for session={base_session}")
    print("")


if __name__ == "__main__":
    main()
