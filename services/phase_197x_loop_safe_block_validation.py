"""
phase_197x_loop_safe_block_validation.py
Phase 197X-LOOP-FIX — Validate manage_positions safe-blocks on valid paper approval.
Read-only validation; does not submit broker orders.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUTPUT_JSON = RESULTS / "phase_197x_loop_safe_block_validation.json"
PREVIOUS_COMMIT = "32f0e33b"
PREVIOUS_PHASE = "197X"

EXPLICIT_REASON = "paper_operator_approval_valid_broker_execution_separate"
SAFE_BLOCK_STATE = "SAFE_BLOCK_EXPLICIT_BROKER_EXECUTION_PHASE_REQUIRED"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _run_manage_positions() -> Tuple[int, str]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "services.manage_positions",
            "--mode",
            "paper",
            "--use-performance-risk-overlay",
            "--max-rotation-exits",
            "3",
            "--verbose",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")


def _run_paper_entry_dry_run() -> Tuple[str, str]:
    env = os.environ.copy()
    env["TRITON_ENABLE_PAPER_EXECUTION"] = "1"
    env["TRITON_ENABLE_AUTO_PAPER_ENTRIES"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "services.paper_entry_loop_stage", "--verbose", "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        env=env,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    gate = "UNKNOWN"
    reason = ""
    for line in text.splitlines():
        if "[PAPER_ENTRY_GATE]" in line:
            if "status=" in line:
                gate = line.split("status=", 1)[-1].split()[0]
            if "reason=" in line:
                reason = line.split("reason=", 1)[-1].strip()
    if reason.endswith(" (SAFE_BLOCK)"):
        reason = reason[: -len(" (SAFE_BLOCK)")]
    return gate, reason


def run_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_GOVERNANCE,
        EXPLICIT_BROKER_EXECUTION_REASON,
        load_paper_approved_execution_scope,
    )
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_VALID_CURRENT,
        classify_paper_operator_approval_artifact,
    )

    ap_state, _, _ = classify_paper_operator_approval_artifact()
    approval_valid = ap_state == PAPER_APPROVAL_VALID_CURRENT
    scope_symbols, scope_source, scope_valid = load_paper_approved_execution_scope()

    rc, manage_text = _run_manage_positions()
    paper_gate, paper_reason = _run_paper_entry_dry_run()
    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=True)

    safe_block_marker = "[SAFE_BLOCK]" in manage_text
    safe_block_state = None
    m = re.search(r"\[SAFE_BLOCK\] state=(\S+)", manage_text)
    if m:
        safe_block_state = m.group(1)
    dispatch_state = None
    m2 = re.search(r"dispatch_interlock_state=(\S+)", manage_text)
    if m2:
        dispatch_state = m2.group(1)
    if "DISPATCH_BLOCKED_GOVERNANCE" in manage_text:
        dispatch_state = dispatch_state or DISPATCH_BLOCKED_GOVERNANCE
    scope_excluded = "outside approved 197W/197X batch" in manage_text

    manage_exit_ok = rc == 0
    explicit_safe_block = (
        safe_block_marker
        and safe_block_state == SAFE_BLOCK_STATE
        and EXPLICIT_REASON in manage_text
    )
    governance_block = dispatch_state == DISPATCH_BLOCKED_GOVERNANCE

    status = "PASS"
    if approval_valid:
        if not manage_exit_ok:
            status = "FAIL"
            notes.append(f"manage_positions exit_code={rc} expected 0")
        if not explicit_safe_block and governance_block:
            status = "FAIL"
            notes.append("expected explicit broker execution safe-block marker")
        if not scope_valid:
            status = "FAIL"
            notes.append("approval valid but scope not loaded")
    elif ap_state.endswith("EXPIRED"):
        status = "PASS_SAFE_BLOCKED"
        notes.append("approval_expired_observation_only")
    else:
        status = "PASS_SAFE_BLOCKED"
        notes.append(f"approval_state={ap_state}")

    if criteria.get("eligible_for_execution"):
        status = "FAIL"
        notes.append("eligible_for_execution must be false")
    if not criteria.get("live_blocked"):
        status = "FAIL"
        notes.append("live_blocked=false")
    if int(criteria.get("open_orders_count") or 0) != 0:
        status = "FAIL"
        notes.append("open_orders!=0")

    expected_paper_reason = (
        "explicit_broker_execution_phase_required"
        if approval_valid
        else ("approval_expired" if "EXPIRED" in ap_state else paper_reason)
    )
    if approval_valid and paper_reason != expected_paper_reason:
        notes.append(f"paper_entry reason={paper_reason} expected={expected_paper_reason}")

    payload: Dict[str, Any] = {
        "phase": "197X-LOOP-FIX",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "approval_artifact_state": ap_state,
        "approval_artifact_valid": approval_valid,
        "paper_scope_source": scope_source,
        "paper_scope_symbols": scope_symbols,
        "manage_positions_exit_code": rc,
        "manage_positions_exit_ok": manage_exit_ok,
        "dispatch_interlock_state": dispatch_state,
        "dispatch_interlock_reason": EXPLICIT_BROKER_EXECUTION_REASON if governance_block else None,
        "safe_block": bool(explicit_safe_block or safe_block_marker),
        "safe_block_state": safe_block_state,
        "safe_block_marker_present": safe_block_marker,
        "scope_excluded_symbols_logged": scope_excluded,
        "broker_orders_submitted": False,
        "exits_executed": 0,
        "trims_executed": 0,
        "operator_execution_required": True,
        "explicit_broker_execution_phase_required": True,
        "cycle_should_continue": manage_exit_ok,
        "open_orders": int(criteria.get("open_orders_count") or 0),
        "live_blocked": bool(criteria.get("live_blocked")),
        "paper_entry_dry_run_status": paper_gate,
        "paper_entry_dry_run_reason": paper_reason,
        "manage_output_tail": manage_text[-3000:] if len(manage_text) > 3000 else manage_text,
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197X-LOOP-FIX] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197X-LOOP-FIX] status={status} manage_rc={rc} safe_block={payload['safe_block']} "
            f"dispatch={dispatch_state}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197X loop safe-block validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_validation(verbose=not args.quiet)
    return 0 if result.get("status") in ("PASS", "PASS_SAFE_BLOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
