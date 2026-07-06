"""
phase_197z_batch_notional_scope_validation.py
Phase 197Z-CAP — Validate batch notional cap uses approved 197X scope only.
Read-only; does not submit broker orders.
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
OUTPUT_JSON = RESULTS / "phase_197z_batch_notional_scope_validation.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
PROJECTED_ORDERS_CSV = RESULTS / "phase_197x_projected_paper_orders.csv"
PREVIOUS_PHASE = "197Z-RB"
PREVIOUS_COMMIT = "e2acadf4"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


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
        DISPATCH_BLOCKED_BATCH_NOTIONAL_CAP,
        DISPATCH_READY,
        compute_dispatch_batch_notional_cap_context,
        evaluate_defensive_exit_dispatch_interlock,
        load_paper_approved_execution_scope,
    )
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.live_block_classifier import evaluate_live_block_status
    from services.paper_loop_mediated_execution_readiness_bridge import (
        evaluate_paper_loop_mediated_execution_readiness_bridge,
    )
    from services.phase_197y_loop_mediated_paper_execution_gate import (
        GATE_READY_LOOP_MEDIATED,
        evaluate_loop_mediated_paper_execution_gate,
    )

    preview_doc = _read_json(PREVIEW_JSON)
    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=True)
    live = evaluate_live_block_status(paper_mode=True)
    gate = evaluate_loop_mediated_paper_execution_gate(write_output=False)
    bridge = evaluate_paper_loop_mediated_execution_readiness_bridge(mode="paper")
    interlock = evaluate_defensive_exit_dispatch_interlock(
        mode="paper",
        broker_snapshot_ok=True,
        require_market_open=True,
        planned_count=10,
    )
    scope_symbols, scope_source, scope_valid = load_paper_approved_execution_scope()
    cap_context = compute_dispatch_batch_notional_cap_context(
        preview_doc=preview_doc,
        use_approved_197x_scope=True,
    )
    cap_context_full = compute_dispatch_batch_notional_cap_context(
        preview_doc=preview_doc,
        use_approved_197x_scope=False,
    )
    snap_rc, snapshot = _run_snapshot()

    bridge_active = bool(bridge.get("bridge_active"))
    gate_ready = str(gate.get("final_state") or "") == GATE_READY_LOOP_MEDIATED
    dispatch_state = str(interlock.get("dispatch_interlock_state") or "")
    dispatch_reason = str(interlock.get("dispatch_interlock_reason") or "")
    interlock_cap = interlock.get("batch_notional_cap_diagnostics") or cap_context
    final_notional = float(interlock_cap.get("final_cap_checked_notional") or 0.0)
    cap_limit = float(interlock_cap.get("batch_notional_cap") or 50000.0)
    pre_scope = float(interlock_cap.get("pre_scope_planned_notional") or 0.0)
    approved_notional = float(interlock_cap.get("approved_scope_notional") or 0.0)
    excluded_notional = float(interlock_cap.get("excluded_outside_197x_scope_notional") or 0.0)
    cap_passed = bool(interlock_cap.get("cap_passed"))

    status = "PASS_SAFE_BLOCKED"
    if bridge_active and gate_ready:
        status = "PASS"
        if dispatch_state == DISPATCH_BLOCKED_BATCH_NOTIONAL_CAP and cap_passed:
            status = "FAIL"
            notes.append("dispatch_blocked_by_cap_despite_scope_cap_passed")
        if not cap_passed and final_notional <= cap_limit:
            status = "FAIL"
            notes.append("cap_passed_false_but_scope_notional_within_cap")
        if interlock_cap.get("cap_basis") != "phase_197x_projected_paper_orders":
            status = "FAIL"
            notes.append(f"unexpected_cap_basis={interlock_cap.get('cap_basis')}")
        if pre_scope > approved_notional and excluded_notional <= 0:
            notes.append("expected_positive_excluded_outside_scope_notional")
        if not scope_valid:
            status = "FAIL"
            notes.append("approved_scope_invalid")
        if dispatch_state == DISPATCH_READY:
            notes.append("dispatch_ready_observed")
    else:
        notes.append(f"bridge_active={bridge_active} gate_ready={gate_ready}")

    if not live.get("live_blocked"):
        status = "FAIL"
        notes.append("live_blocked=false")

    if int(criteria.get("open_orders_count") or 0) != 0:
        status = "FAIL"
        notes.append("open_orders!=0")

    if snap_rc != 0:
        notes.append(f"snapshot exit_code={snap_rc}")

    payload: Dict[str, Any] = {
        "phase": "197Z-CAP",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "bridge_active": bridge_active,
        "197y_gate_ready": gate_ready,
        "197y_eligible_for_execution": bool(gate.get("eligible_for_execution")),
        "approved_scope_symbols": scope_symbols,
        "approved_scope_source": scope_source,
        "approved_scope_valid": scope_valid,
        "pre_scope_planned_notional": pre_scope,
        "excluded_outside_197x_scope_notional": excluded_notional,
        "approved_scope_notional": approved_notional,
        "final_cap_checked_notional": final_notional,
        "batch_notional_cap": cap_limit,
        "cap_passed": cap_passed,
        "excluded_symbols": interlock_cap.get("excluded_symbols") or [],
        "cap_basis": interlock_cap.get("cap_basis"),
        "full_preview_cap_context": cap_context_full,
        "scoped_cap_context": cap_context,
        "dispatch_interlock_state": dispatch_state,
        "dispatch_interlock_reason": dispatch_reason,
        "dispatch_permitted": bool(interlock.get("dispatch_permitted")),
        "dispatch_blocked_batch_notional_cap": dispatch_state
        == DISPATCH_BLOCKED_BATCH_NOTIONAL_CAP,
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
        "projected_orders_csv_exists": PROJECTED_ORDERS_CSV.is_file(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197Z-CAP] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197Z-CAP] status={status} dispatch={dispatch_state} "
            f"final_notional={final_notional} cap={cap_limit} cap_passed={cap_passed}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197Z-CAP batch notional scope validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_validation(verbose=not args.quiet)
    return 0 if result.get("status") in ("PASS", "PASS_SAFE_BLOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
