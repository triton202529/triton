# services/live_cycle.py
"""
Phase 1.5 Live Cycle Runner

Runs a single "health cycle" suitable for scheduler or manual invocation:
- heartbeat -> cycle_start -> runtime_verify -> cycle_end
- writes runtime_state.json + open_orders_verify.json
- optional cleanup policies (stale, dupes, non-gtc, orphans)

This is the control-plane orchestrator.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict

from services.runtime_state import (
    write_heartbeat,
    cycle_start,
    cycle_end,
)
from services.runtime_verify import verify_open_orders


def run_live_cycle(**kwargs) -> Dict[str, Any]:
    """
    One atomic "cycle" that updates runtime_state.json before/after verification.

    Control-plane args:
      - mode

    Execution-plane args:
      - everything else (passed to verify_open_orders)
    """
    mode = kwargs.get("mode", "NORMAL")

    # Always heartbeat first so UI sees liveness even if verify fails
    write_heartbeat(
        automation_enabled=True,
        mode=mode,
        note="live_cycle starting",
    )

    ctx = cycle_start(
        automation_enabled=True,
        mode=mode,
        reason="health_cycle",
        extra={"stage": "start"},
    )

    try:
        # Strip control-plane args before verify
        verify_kwargs = dict(kwargs)
        verify_kwargs.pop("mode", None)

        rep = verify_open_orders(**verify_kwargs)

        cycle_end(
            ctx,
            success=True,
            reason=f"verify_completed status={rep.get('status')}",
            extra={
                "stage": "end",
                "verify_status": rep.get("status"),
                "summary": rep.get("summary"),
            },
        )
        return rep

    except Exception as e:
        tb = traceback.format_exc()
        cycle_end(
            ctx,
            success=False,
            reason=f"exception: {e}",
            extra={
                "traceback": tb[:2000],
            },
        )
        raise


# ──────────────────────────────
# CLI ENTRYPOINT
# ──────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Triton Phase 1.5 live health cycle runner")

    # ── mirror runtime_verify flags ──
    p.add_argument("--expect-tif", default="gtc", help="Expected time_in_force. Use '' to disable.")
    p.add_argument("--cancel-non-gtc", action="store_true")
    p.add_argument("--cancel-dupes", action="store_true")
    p.add_argument("--cancel-orphans", action="store_true")
    p.add_argument("--cancel-stale", action="store_true")
    p.add_argument("--stale-minutes", type=int, default=240)
    p.add_argument("--stale-only-day", action="store_true")
    p.add_argument("--really-cancel", action="store_true")
    p.add_argument("--no-nested", action="store_true")
    p.add_argument("--write-report", action="store_true")
    p.add_argument("--report-path", default="data/runtime/open_orders_verify.json")

    # ── control-plane ──
    p.add_argument(
        "--mode",
        default="NORMAL",
        help="Runtime mode: NORMAL / DEFENSIVE / LOCKDOWN",
    )

    args = p.parse_args()

    rep = run_live_cycle(
        expect_tif=(args.expect_tif or None),
        cancel_non_gtc=args.cancel_non_gtc,
        cancel_dupes=args.cancel_dupes,
        cancel_orphans=args.cancel_orphans,
        cancel_stale=args.cancel_stale,
        stale_minutes=args.stale_minutes,
        stale_only_day=args.stale_only_day,
        dry_run=(not args.really_cancel),
        nested=(not args.no_nested),
        write_report=args.write_report,
        report_path=args.report_path,
        mode=args.mode,
    )

    print(f"[live_cycle] done status={rep.get('status')} summary={rep.get('summary')}")
