"""
phase_197q_fix_runtime_process_and_env_authority_validation.py
Phase 197Q-FIX — Validate runtime process detection and CLI paper env authority.
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
OUTPUT_JSON = RESULTS / "phase_197q_fix_runtime_process_and_env_authority_validation.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "bf00f5a5"
PREVIOUS_PHASE = "197Q"


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


def run_phase_197q_fix_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    status = "PASS"
    git_clean_before = _git_clean()

    from services.live_block_classifier import (
        defensive_exit_phase_validator_gates,
        validator_live_block_check,
    )
    from services.phase_197m_fix_live_block_detection_validation import (
        run_phase_197m_fix_validation,
    )
    from services.phase_197q_runtime_reentry_observation_validation import (
        run_phase_197q_validation,
    )
    from services.runtime_process_detector import (
        RUNTIME_PROCESS_STOPPED_LOCK_ABSENT,
        detect_runtime_loop_processes,
    )
    from services.runtime_reentry_observation_gate import (
        RESTART_CONTINUOUS_FOR_OBSERVATION_ONLY,
        run_runtime_reentry_observation,
    )
    from services.triton_env_loader import load_cli_validation_env

    env_authority = load_cli_validation_env()
    processes = detect_runtime_loop_processes()
    gates = defensive_exit_phase_validator_gates(verbose=False)
    live_block_ok, live_payload = validator_live_block_check(paper_mode=True)

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
    q197 = run_phase_197q_validation(verbose=False)
    m197 = run_phase_197m_fix_validation(verbose=False)

    continuous_running = bool(processes.get("continuous_loop_running"))
    watchdog_running = bool(processes.get("watchdog_loop_running"))
    continuous_lock = bool(processes.get("continuous_lock_exists"))
    cl_state = str(processes.get("continuous_runtime_process_lock_state") or "")

    if continuous_running and not continuous_lock:
        notes.append("continuous running without lock (reported)")
    if not continuous_running and continuous_lock:
        notes.append("continuous lock without process (may be stale)")

    if continuous_running and not watchdog_running:
        status = "FAIL"
        notes.append("continuous reported without watchdog")
    if (
        not continuous_running
        and not continuous_lock
        and cl_state != RUNTIME_PROCESS_STOPPED_LOCK_ABSENT
    ):
        status = "FAIL"
        notes.append(f"expected {RUNTIME_PROCESS_STOPPED_LOCK_ABSENT} got {cl_state}")
    if observation.get("continuous_loop_running") != continuous_running:
        status = "FAIL"
        notes.append("197Q observation continuous_running mismatch with detector")
    if q197.get("status") != "PASS":
        status = "FAIL"
        notes.append(f"197Q status={q197.get('status')}")
    if m197.get("status") != "PASS":
        status = "FAIL"
        notes.append(f"197M-FIX status={m197.get('status')}")
    if not live_block_ok or not live_payload.get("live_blocked"):
        status = "FAIL"
        notes.append("live block regression")
    if env_authority.get("live_execution_enabled") or env_authority.get("live_force_enabled"):
        status = "FAIL"
        notes.append("live execution env enabled")
    if REAL_APPROVAL_PATH.is_file():
        status = "FAIL"
        notes.append("real approval artifact exists")
    if int(observation.get("open_orders_count") or 0) != 0:
        status = "FAIL"
        notes.append("open orders must be 0")
    if observation.get("eligible_for_execution"):
        status = "FAIL"
        notes.append("eligible_for_execution must be false")
    if observation.get("restart_recommendation") != RESTART_CONTINUOUS_FOR_OBSERVATION_ONLY:
        status = "FAIL"
        notes.append("restart recommendation must be observation-only")

    payload: Dict[str, Any] = {
        "phase": "197Q-FIX",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "Process detection matched its own PowerShell query command line containing "
            "run_continuous_loop.ps1; CLI validators failed when paper env unset outside supervisor."
        ),
        "fix": (
            "runtime_process_detector excludes self-match and requires -File script invocation; "
            "triton_env_loader reports paper env authority; validators accept SAFE_BLOCK_PAPER_ENV_NOT_ARMED."
        ),
        "continuous_loop_running": continuous_running,
        "watchdog_loop_running": watchdog_running,
        "detected_continuous_process_ids": processes.get("detected_continuous_process_ids"),
        "detected_watchdog_process_ids": processes.get("detected_watchdog_process_ids"),
        "continuous_lock_exists": continuous_lock,
        "watchdog_lock_exists": processes.get("watchdog_lock_exists"),
        "continuous_runtime_process_lock_state": cl_state,
        "watchdog_runtime_process_lock_state": processes.get("watchdog_runtime_process_lock_state"),
        "process_detection_confidence": processes.get("process_detection_confidence"),
        "process_detection_warnings": processes.get("process_detection_warnings"),
        "env_authority_state": env_authority.get("env_authority_state"),
        "env_source": env_authority.get("env_source"),
        "paper_execution_enabled": env_authority.get("paper_execution_enabled"),
        "auto_paper_entries_enabled": env_authority.get("auto_paper_entries_enabled"),
        "auto_paper_approval_enabled": env_authority.get("auto_paper_approval_enabled"),
        "live_execution_enabled": env_authority.get("live_execution_enabled"),
        "live_force_enabled": env_authority.get("live_force_enabled"),
        "broker_paper_env_present": env_authority.get("broker_paper_env_present"),
        "broker_live_env_present_or_unknown": env_authority.get(
            "broker_live_env_present_or_unknown"
        ),
        "secrets_masked": env_authority.get("secrets_masked"),
        "cli_validation_env_loaded": env_authority.get("cli_validation_env_loaded"),
        "paper_entry_gate_status": gates.get("gate_classification"),
        "paper_entry_gate_reason": gates.get("gate_reason"),
        "phase_197m_status_after_env_load": m197.get("status"),
        "phase_197q_status_after_process_fix": q197.get("status"),
        "live_blocked": live_payload.get("live_blocked"),
        "live_block_reason": live_payload.get("live_block_reason"),
        "open_orders_count": observation.get("open_orders_count"),
        "real_approval_artifact_exists": REAL_APPROVAL_PATH.is_file(),
        "broker_orders_submitted": False,
        "eligible_for_execution": False,
        "restart_recommendation": observation.get("restart_recommendation"),
        "runtime_reentry_state": observation.get("runtime_reentry_state"),
        "processes": processes,
        "env_authority": env_authority,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197Q-FIX] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197Q-FIX] status={status} continuous={continuous_running} "
            f"watchdog={watchdog_running} cl_state={cl_state} "
            f"env={env_authority.get('env_authority_state')} "
            f"197M={m197.get('status')} 197Q={q197.get('status')}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 197Q-FIX runtime process and env authority validation"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197q_fix_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
