"""
phase_197y_loop_mediated_paper_execution_validation.py
Phase 197Y — Validate loop-mediated paper execution gate (no broker submission).
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
OUTPUT_JSON = RESULTS / "phase_197y_loop_mediated_paper_execution_validation.json"
PREVIOUS_PHASE = "197X-LOOP-FIX"
PREVIOUS_COMMIT = "b7031748"

GATE_BLOCKED_EXECUTION_AUTH_REQUIRED = "PAPER_EXECUTION_GATE_BLOCKED_EXECUTION_AUTH_REQUIRED"
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
        DISPATCH_BLOCKED_GOVERNANCE,
        EXPLICIT_BROKER_EXECUTION_REASON,
    )
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.paper_broker_execution_authorization_state import (
        EXEC_AUTH_PATH,
        PAPER_BROKER_EXECUTION_AUTH_ABSENT,
        classify_paper_broker_execution_authorization,
    )
    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_VALID_CURRENT,
        classify_paper_operator_approval_artifact,
    )
    from services.phase_197y_loop_mediated_paper_execution_gate import run_gate

    exec_auth_exists = EXEC_AUTH_PATH.is_file()
    exec_state, _, _ = classify_paper_broker_execution_authorization()
    ap_state, _, _ = classify_paper_operator_approval_artifact()

    gate = run_gate(verbose=False, write_output=True)
    gate_state = str(gate.get("final_state") or "")

    rc, manage_text = _run_manage_positions()
    paper_gate, paper_reason = _run_paper_entry_dry_run()
    snap_rc, snapshot = _run_snapshot()
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

    open_orders_written = int(
        snapshot.get("open_orders_written") or snapshot.get("open_orders") or 0
    )
    positions_written = int(snapshot.get("positions_written") or 0)
    snapshot_errors = snapshot.get("errors") or []

    status = "PASS_SAFE_BLOCKED"
    if not exec_auth_exists and exec_state != PAPER_BROKER_EXECUTION_AUTH_ABSENT:
        status = "FAIL"
        notes.append(f"unexpected exec_auth_state={exec_state} without artifact")

    if gate_state != GATE_BLOCKED_EXECUTION_AUTH_REQUIRED:
        if exec_auth_exists:
            status = "FAIL" if gate_state != GATE_BLOCKED_EXECUTION_AUTH_REQUIRED else status
            notes.append(f"gate_state={gate_state}")
        elif ap_state == PAPER_APPROVAL_VALID_CURRENT:
            status = "FAIL"
            notes.append(f"expected {GATE_BLOCKED_EXECUTION_AUTH_REQUIRED} got {gate_state}")

    if gate.get("broker_orders_submitted"):
        status = "FAIL"
        notes.append("broker_orders_submitted must be false")
    if gate.get("exits_executed"):
        status = "FAIL"
        notes.append("exits_executed must be 0")
    if gate.get("eligible_for_execution"):
        status = "FAIL"
        notes.append("eligible_for_execution must be false without execution auth")

    manage_exit_ok = rc == 0
    if ap_state == PAPER_APPROVAL_VALID_CURRENT and not manage_exit_ok:
        status = "FAIL"
        notes.append(f"manage_positions exit_code={rc} expected 0")

    if int(criteria.get("open_orders_count") or 0) != 0:
        status = "FAIL"
        notes.append("open_orders!=0")
    if not criteria.get("live_blocked"):
        status = "FAIL"
        notes.append("live_blocked=false")

    if snap_rc != 0:
        notes.append(f"snapshot exit_code={snap_rc}")

    payload: Dict[str, Any] = {
        "phase": "197Y",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "final_state": gate_state,
        "gate_reason": gate.get("gate_reason"),
        "execution_authorization_artifact_exists": exec_auth_exists,
        "execution_authorization_state": exec_state,
        "paper_approval_state": ap_state,
        "manage_positions_exit_code": rc,
        "manage_positions_exit_ok": manage_exit_ok,
        "dispatch_interlock_state": dispatch_state,
        "dispatch_interlock_reason": (
            EXPLICIT_BROKER_EXECUTION_REASON
            if dispatch_state == DISPATCH_BLOCKED_GOVERNANCE
            else None
        ),
        "safe_block": safe_block_marker,
        "safe_block_state": safe_block_state,
        "broker_orders_submitted": False,
        "exits_executed": 0,
        "trims_executed": 0,
        "operator_execution_required": bool(gate.get("operator_execution_required")),
        "explicit_broker_execution_phase_required": bool(
            gate.get("explicit_broker_execution_phase_required")
        ),
        "eligible_for_execution": bool(gate.get("eligible_for_execution")),
        "open_orders": int(criteria.get("open_orders_count") or 0),
        "open_orders_written": open_orders_written,
        "positions_written": positions_written,
        "snapshot_errors": snapshot_errors,
        "live_blocked": bool(criteria.get("live_blocked")),
        "paper_entry_dry_run_status": paper_gate,
        "paper_entry_dry_run_reason": paper_reason,
        "197y_gate": gate,
        "manage_output_tail": manage_text[-3000:] if len(manage_text) > 3000 else manage_text,
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197Y-VALIDATION] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197Y-VALIDATION] status={status} final_state={gate_state} "
            f"manage_rc={rc} open_orders={payload['open_orders']}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197Y loop-mediated paper execution validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_validation(verbose=not args.quiet)
    return 0 if result.get("status") in ("PASS", "PASS_SAFE_BLOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
