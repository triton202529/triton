"""
phase_197z_paper_broker_execution_authorization_validation.py
Phase 197Z — Validate paper broker execution authorization tooling (no broker submission).
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
OUTPUT_JSON = RESULTS / "phase_197z_paper_broker_execution_authorization_validation.json"
EXEC_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
PREVIOUS_PHASE = "197Y"
PREVIOUS_COMMIT = "e362576e"

BLOCKED_197W = "PAPER_BROKER_EXECUTION_AUTH_CREATOR_BLOCKED_197W_APPROVAL_REQUIRED"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


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
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.paper_broker_execution_authorization_state import (
        PAPER_BROKER_EXECUTION_AUTH_ABSENT,
        classify_paper_broker_execution_authorization,
    )
    from services.paper_operator_approval_artifact_state import (
        classify_paper_operator_approval_artifact,
    )
    from services.phase_197y_loop_mediated_paper_execution_gate import (
        GATE_BLOCKED_EXECUTION_AUTH_REQUIRED,
        evaluate_loop_mediated_paper_execution_gate,
    )
    from services.phase_197z_paper_broker_execution_authorization_creator import run_creator

    creator = run_creator(create=False, verbose=False)
    creator_state = str(creator.get("final_state") or "")
    ap_state, _, _ = classify_paper_operator_approval_artifact()
    exec_state, _, _ = classify_paper_broker_execution_authorization()
    gate = evaluate_loop_mediated_paper_execution_gate(write_output=False)
    gate_state = str(gate.get("final_state") or "")

    paper_gate, paper_reason = _run_paper_entry_dry_run()
    snap_rc, snapshot = _run_snapshot()
    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=True)

    exec_auth_exists = EXEC_AUTH_PATH.is_file()
    status = "PASS_SAFE_BLOCKED"

    if exec_auth_exists:
        status = "FAIL"
        notes.append("execution_authorization_artifact_must_not_exist")

    if creator.get("real_authorization_created"):
        status = "FAIL"
        notes.append("creator_must_not_create_real_authorization_in_validation")

    if creator_state != BLOCKED_197W and ap_state.endswith("ABSENT"):
        status = "FAIL"
        notes.append(f"expected creator_state={BLOCKED_197W} got {creator_state}")

    if (
        creator.get("broker_orders_submitted")
        or creator.get("exits_executed")
        or creator.get("trims_executed")
    ):
        status = "FAIL"
        notes.append("creator_must_not_report_execution")

    if exec_state != PAPER_BROKER_EXECUTION_AUTH_ABSENT and not exec_auth_exists:
        notes.append(f"exec_auth_state={exec_state}")

    if gate.get("eligible_for_execution"):
        status = "FAIL"
        notes.append("197y_gate_eligible_for_execution_must_be_false")

    if not criteria.get("live_blocked"):
        status = "FAIL"
        notes.append("live_blocked=false")

    if int(criteria.get("open_orders_count") or 0) != 0:
        status = "FAIL"
        notes.append("open_orders!=0")

    if snap_rc != 0:
        notes.append(f"snapshot exit_code={snap_rc}")

    payload: Dict[str, Any] = {
        "phase": "197Z",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "final_state": creator_state,
        "creator_status": creator.get("status"),
        "would_create_authorization": bool(creator.get("would_create_authorization")),
        "real_authorization_created": bool(creator.get("real_authorization_created")),
        "execution_authorization_artifact_exists": exec_auth_exists,
        "execution_authorization_state": exec_state,
        "paper_approval_state": ap_state,
        "197y_gate_final_state": gate_state,
        "197y_gate_blocked_execution_auth_required": gate_state
        == GATE_BLOCKED_EXECUTION_AUTH_REQUIRED,
        "broker_orders_submitted": False,
        "exits_executed": 0,
        "trims_executed": 0,
        "live_blocked": bool(criteria.get("live_blocked")),
        "open_orders": int(criteria.get("open_orders_count") or 0),
        "open_orders_written": int(
            snapshot.get("open_orders_written") or snapshot.get("open_orders") or 0
        ),
        "positions_written": int(snapshot.get("positions_written") or 0),
        "snapshot_errors": snapshot.get("errors") or [],
        "paper_entry_dry_run_status": paper_gate,
        "paper_entry_dry_run_reason": paper_reason,
        "creator": creator,
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197Z-VALIDATION] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197Z-VALIDATION] status={status} final_state={creator_state} "
            f"exec_auth_exists={exec_auth_exists}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 197Z paper broker execution authorization validation"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_validation(verbose=not args.quiet)
    return 0 if result.get("status") in ("PASS", "PASS_SAFE_BLOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
