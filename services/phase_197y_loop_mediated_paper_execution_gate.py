"""
phase_197y_loop_mediated_paper_execution_gate.py
Phase 197Y — Loop-mediated paper execution authorization gate.
Read-only evaluation; does not submit broker orders or create authorization artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUTPUT_JSON = RESULTS / "phase_197y_loop_mediated_paper_execution_gate.json"

GATE_BLOCKED_EXECUTION_AUTH_REQUIRED = "PAPER_EXECUTION_GATE_BLOCKED_EXECUTION_AUTH_REQUIRED"
GATE_BLOCKED_197W_APPROVAL_EXPIRED = "PAPER_EXECUTION_GATE_BLOCKED_197W_APPROVAL_EXPIRED"
GATE_BLOCKED_197X_DRY_RUN_INVALID = "PAPER_EXECUTION_GATE_BLOCKED_197X_DRY_RUN_INVALID"
GATE_BLOCKED_BATCH_MISMATCH = "PAPER_EXECUTION_GATE_BLOCKED_BATCH_MISMATCH"
GATE_BLOCKED_LIVE_NOT_BLOCKED = "PAPER_EXECUTION_GATE_BLOCKED_LIVE_NOT_BLOCKED"
GATE_BLOCKED_OPEN_ORDERS_PRESENT = "PAPER_EXECUTION_GATE_BLOCKED_OPEN_ORDERS_PRESENT"
GATE_READY_LOOP_MEDIATED = "PAPER_EXECUTION_GATE_READY_LOOP_MEDIATED"

SAFE_BLOCKED_GATE_STATES = frozenset(
    {
        GATE_BLOCKED_EXECUTION_AUTH_REQUIRED,
        GATE_BLOCKED_197W_APPROVAL_EXPIRED,
        GATE_BLOCKED_197X_DRY_RUN_INVALID,
        GATE_BLOCKED_BATCH_MISMATCH,
        GATE_BLOCKED_LIVE_NOT_BLOCKED,
        GATE_BLOCKED_OPEN_ORDERS_PRESENT,
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def map_gate_state_to_dispatch(gate_state: str) -> Tuple[str, str]:
    """Map 197Y gate final_state to dispatch interlock state and reason."""
    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_GOVERNANCE,
        DISPATCH_BLOCKED_LIVE_BLOCK,
        DISPATCH_BLOCKED_OPEN_ORDERS,
        EXPLICIT_BROKER_EXECUTION_REASON,
    )

    gate_s = str(gate_state or "")
    if gate_s == GATE_BLOCKED_EXECUTION_AUTH_REQUIRED:
        return DISPATCH_BLOCKED_GOVERNANCE, EXPLICIT_BROKER_EXECUTION_REASON
    if gate_s == GATE_BLOCKED_197W_APPROVAL_EXPIRED:
        return DISPATCH_BLOCKED_GOVERNANCE, "197w_approval_expired"
    if gate_s == GATE_BLOCKED_197X_DRY_RUN_INVALID:
        return DISPATCH_BLOCKED_GOVERNANCE, "197x_dry_run_invalid"
    if gate_s == GATE_BLOCKED_BATCH_MISMATCH:
        return DISPATCH_BLOCKED_GOVERNANCE, "197x_batch_mismatch"
    if gate_s == GATE_BLOCKED_LIVE_NOT_BLOCKED:
        return DISPATCH_BLOCKED_LIVE_BLOCK, "live_blocked=false"
    if gate_s == GATE_BLOCKED_OPEN_ORDERS_PRESENT:
        return DISPATCH_BLOCKED_OPEN_ORDERS, "open_orders_present"
    return DISPATCH_BLOCKED_GOVERNANCE, f"paper_execution_gate_{gate_s.lower()}"


def evaluate_loop_mediated_paper_execution_gate(
    *,
    broker_snapshot_ok: bool = True,
    open_orders_count: int = 0,
    live_blocked: Optional[bool] = None,
    market_open: Optional[bool] = None,
    write_output: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate the full 197Y prerequisite chain for loop-mediated paper broker execution.
    Does not submit orders.
    """
    from services.defensive_exit_governance_review_criteria import (
        collect_governance_review_evidence,
    )
    from services.live_block_classifier import evaluate_live_block_status
    from services.paper_broker_execution_authorization_state import (
        PAPER_BROKER_EXECUTION_AUTH_197W_APPROVAL_MISSING,
        PAPER_BROKER_EXECUTION_AUTH_197X_DRY_RUN_MISSING,
        PAPER_BROKER_EXECUTION_AUTH_ABSENT,
        PAPER_BROKER_EXECUTION_AUTH_BATCH_MISMATCH,
        PAPER_BROKER_EXECUTION_AUTH_EXPIRED,
        PAPER_BROKER_EXECUTION_AUTH_LIVE_INVALID,
        PAPER_BROKER_EXECUTION_AUTH_OPEN_ORDERS_PRESENT,
        PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT,
        classify_paper_broker_execution_authorization,
        load_197x_dry_run_authority,
    )
    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_EXPIRED,
        PAPER_APPROVAL_VALID_CURRENT,
        classify_paper_operator_approval_artifact,
    )

    if live_blocked is None:
        live_blocked = bool(evaluate_live_block_status(paper_mode=True).get("live_blocked"))
    if market_open is None:
        evidence = collect_governance_review_evidence(broker_snapshot_ok=broker_snapshot_ok)
        market_open = bool(evidence.get("market_open"))
        if open_orders_count == 0:
            open_orders_count = int(evidence.get("open_orders_count") or 0)

    ap_state, ap_reason, ap_details = classify_paper_operator_approval_artifact(
        broker_snapshot_ok=broker_snapshot_ok
    )
    x197 = load_197x_dry_run_authority()
    auth_state, auth_reason, auth_details = classify_paper_broker_execution_authorization(
        open_orders_count=open_orders_count,
        broker_snapshot_ok=broker_snapshot_ok,
    )

    final_state = GATE_BLOCKED_EXECUTION_AUTH_REQUIRED
    gate_reason = "execution_authorization_absent"
    eligible_for_execution = False
    operator_execution_required = True
    explicit_broker_execution_phase_required = True

    if not live_blocked:
        final_state = GATE_BLOCKED_LIVE_NOT_BLOCKED
        gate_reason = "live_blocked=false"
    elif open_orders_count > 0:
        final_state = GATE_BLOCKED_OPEN_ORDERS_PRESENT
        gate_reason = f"open_orders_count={open_orders_count}"
    elif auth_state == PAPER_BROKER_EXECUTION_AUTH_ABSENT:
        final_state = GATE_BLOCKED_EXECUTION_AUTH_REQUIRED
        gate_reason = "paper_broker_execution_authorization_absent"
    elif auth_state == PAPER_BROKER_EXECUTION_AUTH_EXPIRED:
        final_state = GATE_BLOCKED_EXECUTION_AUTH_REQUIRED
        gate_reason = "paper_broker_execution_authorization_expired"
    elif (
        ap_state == PAPER_APPROVAL_EXPIRED
        or auth_state == PAPER_BROKER_EXECUTION_AUTH_197W_APPROVAL_MISSING
    ):
        if ap_state == PAPER_APPROVAL_EXPIRED:
            final_state = GATE_BLOCKED_197W_APPROVAL_EXPIRED
            gate_reason = ap_reason
        elif auth_state == PAPER_BROKER_EXECUTION_AUTH_197W_APPROVAL_MISSING:
            final_state = GATE_BLOCKED_197W_APPROVAL_EXPIRED
            gate_reason = auth_reason
    elif auth_state in (PAPER_BROKER_EXECUTION_AUTH_197X_DRY_RUN_MISSING,) or not x197.get(
        "dry_run_valid"
    ):
        final_state = GATE_BLOCKED_197X_DRY_RUN_INVALID
        gate_reason = (
            auth_reason
            if auth_state == PAPER_BROKER_EXECUTION_AUTH_197X_DRY_RUN_MISSING
            else "197x_dry_run_invalid"
        )
    elif auth_state == PAPER_BROKER_EXECUTION_AUTH_BATCH_MISMATCH:
        final_state = GATE_BLOCKED_BATCH_MISMATCH
        gate_reason = auth_reason
    elif auth_state == PAPER_BROKER_EXECUTION_AUTH_LIVE_INVALID:
        final_state = GATE_BLOCKED_LIVE_NOT_BLOCKED
        gate_reason = auth_reason
    elif auth_state == PAPER_BROKER_EXECUTION_AUTH_OPEN_ORDERS_PRESENT:
        final_state = GATE_BLOCKED_OPEN_ORDERS_PRESENT
        gate_reason = auth_reason
    elif auth_state == PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT:
        projected_symbols = list(x197.get("projected_symbols") or [])
        projected_rows = list(x197.get("projected_orders") or [])
        checks_ok = (
            ap_state == PAPER_APPROVAL_VALID_CURRENT
            and x197.get("dry_run_valid")
            and len(projected_rows) > 0
            and live_blocked
            and open_orders_count == 0
            and market_open
        )
        for row in projected_rows:
            if str(row.get("broker_order_submitted") or "").lower() in ("true", "1", "yes"):
                checks_ok = False
                final_state = GATE_BLOCKED_BATCH_MISMATCH
                gate_reason = "projected_order_already_submitted"
                break
        else:
            if checks_ok:
                final_state = GATE_READY_LOOP_MEDIATED
                gate_reason = "all_paper_execution_gate_prerequisites_satisfied"
                eligible_for_execution = True
                operator_execution_required = False
                explicit_broker_execution_phase_required = False
            elif not market_open:
                final_state = GATE_BLOCKED_197X_DRY_RUN_INVALID
                gate_reason = "market_closed"
            else:
                final_state = GATE_BLOCKED_197X_DRY_RUN_INVALID
                gate_reason = "execution_prerequisites_incomplete"
    else:
        final_state = GATE_BLOCKED_EXECUTION_AUTH_REQUIRED
        gate_reason = f"execution_authorization_invalid:{auth_state}:{auth_reason}"

    payload: Dict[str, Any] = {
        "phase": "197Y",
        "generated_at": _utc_now_iso(),
        "final_state": final_state,
        "gate_reason": gate_reason,
        "paper_approval_state": ap_state,
        "paper_approval_reason": ap_reason,
        "execution_authorization_state": auth_state,
        "execution_authorization_reason": auth_reason,
        "197x_dry_run_valid": bool(x197.get("dry_run_valid")),
        "197x_projected_order_count": int(x197.get("projected_order_count") or 0),
        "197x_projected_symbols": x197.get("projected_symbols") or [],
        "market_open": bool(market_open),
        "live_blocked": bool(live_blocked),
        "open_orders_count": open_orders_count,
        "broker_orders_submitted": False,
        "exits_executed": 0,
        "trims_executed": 0,
        "operator_execution_required": operator_execution_required,
        "explicit_broker_execution_phase_required": explicit_broker_execution_phase_required,
        "eligible_for_execution": eligible_for_execution,
        "execution_must_use_loop": True,
        "execution_must_use_approved_scope_only": True,
        "safe_blocked": final_state in SAFE_BLOCKED_GATE_STATES,
        "paper_approval_details": {
            "artifact_path": ap_details.get("artifact_path"),
            "artifact_exists": ap_details.get("artifact_exists"),
        },
        "execution_authorization_details": {
            "artifact_path": auth_details.get("artifact_path"),
            "artifact_exists": auth_details.get("artifact_exists"),
        },
        "197x_authority": {
            "dry_run_status": x197.get("dry_run_status"),
            "dry_run_final_state": x197.get("dry_run_final_state"),
            "dry_run_artifact_hash": x197.get("dry_run_artifact_hash"),
            "projected_orders_csv_hash": x197.get("projected_orders_csv_hash"),
        },
    }

    if write_output:
        _atomic_write_json(OUTPUT_JSON, payload)
    return payload


def run_gate(*, verbose: bool = True, write_output: bool = True) -> Dict[str, Any]:
    result = evaluate_loop_mediated_paper_execution_gate(write_output=False)
    if write_output:
        _atomic_write_json(OUTPUT_JSON, result)
    if verbose:
        print(f"[197Y] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197Y] final_state={result.get('final_state')} "
            f"eligible_for_execution={result.get('eligible_for_execution')}"
        )
    return result


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197Y loop-mediated paper execution gate")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    run_gate(verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
