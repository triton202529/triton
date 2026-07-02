"""
phase_197z_readiness_bridge_validation.py
Phase 197Z-RB — Validate 197W/197Z/197Y readiness bridge (read-only; no broker submission).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_197z_readiness_bridge_validation.json"
EXEC_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_PHASE = "197Z"
PREVIOUS_COMMIT = "58713deb"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _run_snapshot() -> Tuple[int, Dict[str, Any]]:
    proc = subprocess.run(
        [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    snapshot: Dict[str, Any] = {"exit_code": proc.returncode}
    jstart = text.find("{")
    if jstart >= 0:
        try:
            snapshot.update(json.loads(text[jstart:]))
        except Exception:
            pass
    return proc.returncode, snapshot


def run_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_READINESS,
        DISPATCH_READY,
        evaluate_defensive_exit_dispatch_interlock,
        load_paper_approved_execution_scope,
    )
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.live_block_classifier import evaluate_live_block_status
    from services.master_execution_gate import MasterExecutionGate
    from services.paper_broker_execution_authorization_state import (
        PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT,
        classify_paper_broker_execution_authorization,
    )
    from services.paper_loop_mediated_execution_readiness_bridge import (
        evaluate_paper_loop_mediated_execution_readiness_bridge,
    )
    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_VALID_CURRENT,
        classify_paper_operator_approval_artifact,
    )
    from services.phase_197y_loop_mediated_paper_execution_gate import (
        GATE_READY_LOOP_MEDIATED,
        evaluate_loop_mediated_paper_execution_gate,
    )

    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=True)
    live = evaluate_live_block_status(paper_mode=True)
    ap_state, _, _ = classify_paper_operator_approval_artifact()
    auth_state, _, _ = classify_paper_broker_execution_authorization()
    gate = evaluate_loop_mediated_paper_execution_gate(write_output=False)
    bridge = evaluate_paper_loop_mediated_execution_readiness_bridge(mode="paper")
    interlock = evaluate_defensive_exit_dispatch_interlock(
        mode="paper",
        broker_snapshot_ok=True,
        require_market_open=True,
        planned_count=10,
    )
    master = MasterExecutionGate(project_root=ROOT).evaluate(mode="paper", verbose=False)
    scope_symbols, scope_source, scope_valid = load_paper_approved_execution_scope()
    snap_rc, snapshot = _run_snapshot()

    bridge_active = bool(bridge.get("bridge_active"))
    dispatch_state = str(interlock.get("dispatch_interlock_state") or "")
    dispatch_reason = str(interlock.get("dispatch_interlock_reason") or "")
    baseline_readiness = str(criteria.get("readiness_status") or "UNKNOWN")

    status = "PASS_SAFE_BLOCKED"
    if bridge_active:
        status = "PASS"
        if dispatch_state == DISPATCH_BLOCKED_READINESS:
            status = "FAIL"
            notes.append("dispatch_still_blocked_by_readiness")
        if not interlock.get("dispatch_permitted") and dispatch_state != DISPATCH_READY:
            if dispatch_state not in (DISPATCH_READY,):
                notes.append(f"dispatch_state={dispatch_state} reason={dispatch_reason}")
        if master.status != "READY" or not master.ok:
            status = "FAIL"
            notes.append(f"master_gate_status={master.status} reasons={master.reasons}")
        if not gate.get("eligible_for_execution"):
            status = "FAIL"
            notes.append("197y_eligible_for_execution=false")
    else:
        notes.append(f"bridge_inactive reasons={bridge.get('block_reasons')}")

    if not live.get("live_blocked"):
        status = "FAIL"
        notes.append("live_blocked=false")

    if int(criteria.get("open_orders_count") or 0) != 0:
        status = "FAIL"
        notes.append("open_orders!=0")

    if snap_rc != 0:
        notes.append(f"snapshot exit_code={snap_rc}")

    payload: Dict[str, Any] = {
        "phase": "197Z-RB",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "bridge_active": bridge_active,
        "bridge_state": bridge.get("bridge_state"),
        "baseline_readiness_status": baseline_readiness,
        "effective_readiness_status": bridge.get("effective_readiness_status"),
        "execution_readiness_bridged": bool(bridge.get("execution_readiness_bridged")),
        "paper_approval_present": APPROVAL_PATH.is_file(),
        "paper_approval_state": ap_state,
        "paper_approval_valid_current": ap_state == PAPER_APPROVAL_VALID_CURRENT,
        "execution_authorization_present": EXEC_AUTH_PATH.is_file(),
        "execution_authorization_state": auth_state,
        "execution_authorization_valid_current": auth_state
        == PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT,
        "197y_gate_final_state": gate.get("final_state"),
        "197y_eligible_for_execution": bool(gate.get("eligible_for_execution")),
        "197y_gate_ready": str(gate.get("final_state") or "") == GATE_READY_LOOP_MEDIATED,
        "master_execution_gate_status": master.status,
        "master_execution_gate_ok": master.ok,
        "master_execution_gate_reasons": master.reasons,
        "dispatch_interlock_state": dispatch_state,
        "dispatch_interlock_reason": dispatch_reason,
        "dispatch_permitted": bool(interlock.get("dispatch_permitted")),
        "dispatch_blocked_readiness": dispatch_state == DISPATCH_BLOCKED_READINESS,
        "approved_scope_symbols": scope_symbols,
        "approved_scope_source": scope_source,
        "approved_scope_valid": scope_valid,
        "broker_orders_submitted": False,
        "exits_executed": 0,
        "trims_executed": 0,
        "live_blocked": bool(live.get("live_blocked")),
        "open_orders": int(criteria.get("open_orders_count") or 0),
        "open_orders_written": int(
            snapshot.get("open_orders_written") or snapshot.get("open_orders") or 0
        ),
        "positions_written": int(snapshot.get("positions_written") or 0),
        "snapshot_errors": snapshot.get("errors") or [],
        "bridge": bridge,
        "interlock": {
            "dispatch_interlock_state": dispatch_state,
            "dispatch_interlock_reason": dispatch_reason,
            "dispatch_permitted": interlock.get("dispatch_permitted"),
            "execution_readiness_bridged": interlock.get("execution_readiness_bridged"),
            "effective_readiness_status": interlock.get("effective_readiness_status"),
        },
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197Z-RB] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197Z-RB] status={status} bridge_active={bridge_active} "
            f"dispatch={dispatch_state} master={master.status} "
            f"baseline_readiness={baseline_readiness}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197Z-RB readiness bridge validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_validation(verbose=not args.quiet)
    return 0 if result.get("status") in ("PASS", "PASS_SAFE_BLOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
