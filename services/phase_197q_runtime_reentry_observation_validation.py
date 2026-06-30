"""
phase_197q_runtime_reentry_observation_validation.py
Phase 197Q — Validate runtime re-entry observation gate (read-only).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_197q_runtime_reentry_observation_validation.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "f8a6dd63"
PREVIOUS_PHASE = "197P"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _git_branch() -> str:
    proc = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (proc.stdout or "").strip()


def _git_clean() -> bool:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return not (proc.stdout or "").strip()


def run_phase_197q_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    status = "PASS"
    git_clean_before = _git_clean()

    from services.live_block_classifier import validator_live_block_check
    from services.order_guard_snapshot_authority import (
        ORDER_GUARD_FRESH,
        ORDER_GUARD_STALE_BUT_BROKER_ZERO_OPEN_ORDERS,
    )
    from services.runtime_reentry_observation_gate import (
        DO_NOT_RESTART_CONTINUOUS,
        RESTART_CONTINUOUS_FOR_OBSERVATION_ONLY,
        RUNTIME_REENTRY_BLOCKED_BY_LIVE_BLOCK_REGRESSION,
        RUNTIME_REENTRY_BLOCKED_BY_OPEN_ORDERS,
        RUNTIME_REENTRY_BLOCKED_BY_REAL_APPROVAL_ARTIFACT,
        RUNTIME_REENTRY_READY_FOR_OBSERVATION,
        RUNTIME_REENTRY_WAITING_FOR_MARKET_OPEN,
        run_runtime_reentry_observation,
    )

    snapshot_ok = False
    try:
        snap = subprocess.run(
            [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        jstart = (snap.stdout or "").find("{")
        if jstart >= 0:
            snapshot_ok = bool(json.loads(snap.stdout[jstart:]).get("ok"))
    except Exception as exc:
        notes.append(f"snapshot_live_orders failed: {exc}")

    observation = run_runtime_reentry_observation(
        broker_snapshot_ok=snapshot_ok,
        broker_orders_submitted=False,
    )
    live_block_ok, _ = validator_live_block_check(paper_mode=True)

    state = str(observation.get("runtime_reentry_state") or "")
    safe_states = {
        RUNTIME_REENTRY_WAITING_FOR_MARKET_OPEN,
        RUNTIME_REENTRY_READY_FOR_OBSERVATION,
    }
    safe_guard = observation.get("order_guard_authority_state") in (
        ORDER_GUARD_FRESH,
        ORDER_GUARD_STALE_BUT_BROKER_ZERO_OPEN_ORDERS,
    )

    if not live_block_ok or not observation.get("live_blocked"):
        status = "FAIL"
        notes.append("FAIL_LIVE_BLOCK_REGRESSION")
    if state == RUNTIME_REENTRY_BLOCKED_BY_LIVE_BLOCK_REGRESSION:
        status = "FAIL"
        notes.append("runtime re-entry live block regression")
    if state == RUNTIME_REENTRY_BLOCKED_BY_OPEN_ORDERS:
        status = "FAIL"
        notes.append("open orders present")
    if state == RUNTIME_REENTRY_BLOCKED_BY_REAL_APPROVAL_ARTIFACT:
        status = "FAIL"
        notes.append("real approval artifact exists")
    if REAL_APPROVAL_PATH.is_file():
        status = "FAIL"
        notes.append("real approval artifact on disk")
    if not safe_guard:
        status = "FAIL"
        notes.append(f"order guard unsafe: {observation.get('order_guard_authority_state')}")
    if int(observation.get("open_orders_count") or 0) != 0:
        status = "FAIL"
        notes.append("open_orders_count must be 0")
    if observation.get("eligible_for_execution"):
        status = "FAIL"
        notes.append("eligible_for_execution must be false")
    if observation.get("broker_orders_submitted"):
        status = "FAIL"
        notes.append("broker_orders_submitted must be false")
    if state not in safe_states:
        status = "FAIL"
        notes.append(f"unexpected runtime_reentry_state {state}")
    if observation.get("restart_recommendation") != RESTART_CONTINUOUS_FOR_OBSERVATION_ONLY:
        status = "FAIL"
        notes.append(
            f"restart recommendation must be {RESTART_CONTINUOUS_FOR_OBSERVATION_ONLY} "
            f"got {observation.get('restart_recommendation')}"
        )
    if not snapshot_ok:
        status = "FAIL"
        notes.append("broker snapshot not ok")

    if not observation.get("watchdog_loop_running"):
        notes.append("watchdog_loop_running=false (reported, not a failure)")

    payload: Dict[str, Any] = {
        "phase": "197Q",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "Defensive exit governance hardening completed but continuous runtime had no "
            "integrated re-entry observation gate before supervisor restart."
        ),
        "fix": (
            "runtime_reentry_observation_gate aggregates live block, order guard authority, "
            "197O/197N rechecks, paper entry gate, and loop status for observation-only restart."
        ),
        "runtime_reentry_state": state,
        "runtime_reentry_reason": observation.get("runtime_reentry_reason"),
        "continuous_loop_running": observation.get("continuous_loop_running"),
        "watchdog_loop_running": observation.get("watchdog_loop_running"),
        "continuous_lock_exists": observation.get("continuous_lock_exists"),
        "watchdog_lock_exists": observation.get("watchdog_lock_exists"),
        "detected_continuous_process_ids": observation.get("detected_continuous_process_ids"),
        "detected_watchdog_process_ids": observation.get("detected_watchdog_process_ids"),
        "continuous_runtime_process_lock_state": observation.get(
            "continuous_runtime_process_lock_state"
        ),
        "process_detection_confidence": observation.get("process_detection_confidence"),
        "process_detection_warnings": observation.get("process_detection_warnings"),
        "live_blocked": observation.get("live_blocked"),
        "live_block_reason": observation.get("live_block_reason"),
        "order_guard_authority_state": observation.get("order_guard_authority_state"),
        "broker_snapshot_ok": snapshot_ok,
        "open_orders_count": observation.get("open_orders_count"),
        "positions_count": observation.get("positions_count"),
        "market_open": observation.get("market_open"),
        "market_hours_recheck_state": observation.get("market_hours_recheck_state"),
        "governance_review_state": observation.get("governance_review_state"),
        "governance_authorized": observation.get("governance_authorized"),
        "readiness_status": observation.get("readiness_status"),
        "paper_entry_gate_status": observation.get("paper_entry_gate_status"),
        "paper_entry_gate_reason": observation.get("paper_entry_gate_reason"),
        "real_approval_artifact_exists": observation.get("real_approval_artifact_exists"),
        "operator_approval_status": observation.get("operator_approval_status"),
        "capped_preview_exists": observation.get("capped_preview_exists"),
        "capped_symbols": observation.get("capped_symbols"),
        "deferred_symbols": observation.get("deferred_symbols"),
        "eligible_for_operator_decision": observation.get("eligible_for_operator_decision"),
        "eligible_for_execution": observation.get("eligible_for_execution"),
        "broker_orders_submitted": False,
        "restart_recommendation": observation.get("restart_recommendation"),
        "restart_recommendation_reason": observation.get("restart_recommendation_reason"),
        "restart_command": observation.get("restart_command"),
        "observation": observation,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197Q-RUNTIME-REENTRY] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197Q-RUNTIME-REENTRY] status={status} state={state} "
            f"watchdog={observation.get('watchdog_loop_running')} "
            f"continuous={observation.get('continuous_loop_running')} "
            f"restart={observation.get('restart_recommendation')}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197Q runtime re-entry observation validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197q_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
