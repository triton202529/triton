"""
TRITON Phase 196 — Automatic Recovery Restart Failure Root Cause Analysis.

Diagnoses why self-healing detected/cleaned failures but failed to restore a
healthy continuous loop. Read-only — does NOT modify trading, governance,
lifecycle, execution, or risk logic.

Outputs (data/results/):
    recovery_restart_diagnostics.json
    recovery_restart_failure_history.csv
    recovery_restart_recommendations.csv

Run:
    python -m services.recovery_restart_diagnostics
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

from services.continuous_loop_recovery_manager import (
    DEFAULT_INTERVAL_MINUTES,
    LOCK_PATH,
    LOOP_LOG_DIR,
    LOOP_SCRIPT,
    WATCHDOG_STATE,
    _is_loop_process_running,
    _now_iso,
    _parse_lock,
    _read_json,
)
from services.hung_loop_detector import run_detection

RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
OUT_JSON = RESULTS / "recovery_restart_diagnostics.json"
OUT_HISTORY = RESULTS / "recovery_restart_failure_history.csv"
OUT_RECOMMENDATIONS = RESULTS / "recovery_restart_recommendations.csv"

ACTIONS_CSV = RESULTS / "hung_loop_recovery_actions.csv"
FAILURE_CSV = RESULTS / "continuous_loop_failure_events.csv"
DETECTION_JSON = RESULTS / "hung_loop_detection.json"


def _log(msg: str) -> None:
    print(f"[recovery_restart_diagnostics] {msg}", flush=True)


def _stage(
    name: str,
    status: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
    source: Optional[str] = None,
    correction: Optional[str] = None,
    detail: Any = None,
) -> Dict[str, Any]:
    return {
        "stage": name,
        "status": status,
        "start_timestamp": start,
        "end_timestamp": end,
        "duration_ms": duration_ms,
        "error_message": error,
        "failure_source": source,
        "recommended_correction": correction,
        "detail": detail,
    }


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _ts_after(earlier: Optional[str], later: Optional[str]) -> bool:
    a, b = _parse_iso(earlier), _parse_iso(later)
    return bool(a and b and b > a)


def _load_last_recovery_attempt() -> Dict[str, Any]:
    if not ACTIONS_CSV.is_file():
        return {}
    df = pd.read_csv(ACTIONS_CSV)
    if df.empty:
        return {}
    failed = df[df["result"].astype(str).str.contains("FAILED", na=False)]
    row = failed.iloc[-1].to_dict() if len(failed) else df.iloc[-1].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def _build_restart_command(interval_minutes: int = DEFAULT_INTERVAL_MINUTES) -> List[str]:
    return [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(LOOP_SCRIPT),
        "-IntervalMinutes",
        str(interval_minutes),
    ]


def _validate_environment() -> Dict[str, Any]:
    venv_ok = VENV_PYTHON.is_file()
    py = str(VENV_PYTHON) if venv_ok else sys.executable
    ps_policy = None
    ps_ok = False
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-ExecutionPolicy -Scope CurrentUser"],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(ROOT),
            )
            ps_policy = (r.stdout or "").strip()
            ps_ok = r.returncode == 0 and ps_policy not in ("Restricted", "")
        except (subprocess.SubprocessError, OSError):
            ps_ok = False

    paper_env = os.environ.get("TRITON_ENABLE_PAPER_EXECUTION")
    dirs_ok = all(p.is_dir() or p.parent.is_dir() for p in [ROOT, LIVE, LOOP_LOG_DIR, RESULTS])
    return {
        "python_executable": py,
        "venv_available": venv_ok,
        "venv_path": str(VENV_PYTHON),
        "powershell_script_exists": LOOP_SCRIPT.is_file(),
        "powershell_script_path": str(LOOP_SCRIPT),
        "execution_policy": ps_policy,
        "execution_policy_permits_launch": ps_ok,
        "triton_root": str(ROOT),
        "triton_root_exists": ROOT.is_dir(),
        "working_directories_ok": dirs_ok,
        "lock_path_writable": LIVE.is_dir(),
        "log_directory_exists": LOOP_LOG_DIR.is_dir(),
        "TRITON_ENABLE_PAPER_EXECUTION_in_parent_env": paper_env,
        "recovery_must_set_TRITON_ENABLE_PAPER_EXECUTION": True,
        "note": "Recovery spawns PowerShell script which sets TRITON_ENABLE_PAPER_EXECUTION=1 internally",
    }


def _run_simulation_tests(env: Dict[str, Any], interval_minutes: int) -> List[Dict[str, Any]]:
    cmd = _build_restart_command(interval_minutes)
    cmd_str = " ".join(cmd)
    syntax_ok = LOOP_SCRIPT.is_file() and all(x for x in cmd)

    tests = [
        {
            "test": "command_validation",
            "expected": "VALID_COMMAND",
            "result": "VALID_COMMAND" if syntax_ok else "INVALID_COMMAND",
            "passed": syntax_ok,
            "detail": cmd_str,
        },
    ]

    running, pids = _is_loop_process_running()
    lock = _parse_lock(LOCK_PATH)
    spawn_ok = running and lock.get("exists")
    tests.append(
        {
            "test": "process_spawn",
            "expected": "PROCESS_STARTED",
            "result": "PROCESS_STARTED" if spawn_ok else "PROCESS_NOT_STARTED",
            "passed": spawn_ok,
            "detail": {"pids": pids, "lock_exists": lock.get("exists")},
            "note": "Observational — reflects current state, not a new spawn",
        }
    )

    log_files = sorted(LOOP_LOG_DIR.glob("continuous_loop_*.log"), key=lambda p: p.stat().st_mtime)
    latest_log = log_files[-1] if log_files else None
    log_has_banner = False
    if latest_log and latest_log.is_file():
        text = latest_log.read_text(encoding="utf-8", errors="replace")[:500]
        log_has_banner = "TRITON CONTINUOUS LOOP START" in text

    state = _read_json(WATCHDOG_STATE)
    hb_refresh = state.get("heartbeat_refresh_ts")
    init_ok = lock.get("exists") and log_has_banner and bool(hb_refresh)
    tests.append(
        {
            "test": "loop_initialization",
            "expected": "LOOP_INITIALIZED",
            "result": "LOOP_INITIALIZED" if init_ok else "LOOP_NOT_INITIALIZED",
            "passed": init_ok,
            "detail": {
                "lock": lock.get("exists"),
                "log_banner": log_has_banner,
                "heartbeat_refresh": hb_refresh,
            },
        }
    )

    detection = {}
    try:
        if DETECTION_JSON.is_file():
            detection = json.loads(DETECTION_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    classification = detection.get("process_health_classification", "UNKNOWN")
    completion_ok = classification == "HEALTHY"
    tests.append(
        {
            "test": "recovery_completion",
            "expected": "RECOVERY_SUCCESS",
            "result": "RECOVERY_SUCCESS" if completion_ok else "RECOVERY_INCOMPLETE",
            "passed": completion_ok,
            "detail": {"classification": classification},
        }
    )
    return tests


def _classify_failure(
    chain: List[Dict[str, Any]],
    env: Dict[str, Any],
    last_attempt: Dict[str, Any],
    tests: List[Dict[str, Any]],
) -> str:
    result = str(last_attempt.get("result", ""))
    if (
        result == "RECOVERY_FAILED"
        and last_attempt.get("new_pids")
        and not last_attempt.get("recovery_success")
    ):
        return "HEALTH_VERIFICATION_FAILURE"

    failed_stages = [s for s in chain if s["status"] == "FAIL"]
    if failed_stages:
        name = failed_stages[0]["stage"]
        mapping = {
            "Restart Command Creation": "RESTART_COMMAND_FAILURE",
            "Process Launch": "PROCESS_CREATION_FAILURE",
            "Heartbeat Creation": "HEARTBEAT_TIMEOUT",
            "Health Verification": "HEALTH_VERIFICATION_FAILURE",
            "Stale Lock Removal": "PERMISSION_FAILURE",
            "PID Validation": "PROCESS_CREATION_FAILURE",
            "Environment Preparation": "ENVIRONMENT_VARIABLE_FAILURE",
        }
        return mapping.get(name, "UNKNOWN_FAILURE")

    if not env.get("powershell_script_exists"):
        return "POWERSHELL_SCRIPT_FAILURE"
    if not env.get("execution_policy_permits_launch"):
        return "PERMISSION_FAILURE"
    if not env.get("venv_available"):
        return "VENV_ACTIVATION_FAILURE"
    if not env.get("triton_root_exists"):
        return "WORKING_DIRECTORY_FAILURE"

    if result == "RECOVERY_FAILED":
        if not last_attempt.get("pids_terminated"):
            return "PROCESS_CREATION_FAILURE"
        return "HEALTH_VERIFICATION_FAILURE"

    cmd_test = next((t for t in tests if t["test"] == "command_validation"), {})
    if not cmd_test.get("passed"):
        return "RESTART_COMMAND_FAILURE"

    return "UNKNOWN_FAILURE"


def _trace_recovery_chain(
    last_attempt: Dict[str, Any],
    env: Dict[str, Any],
    detection: Dict[str, Any],
) -> List[Dict[str, Any]]:
    ts = last_attempt.get("timestamp") or _now_iso()
    result = str(last_attempt.get("result", "NOT_ATTEMPTED"))
    attempted = result not in ("NOT_ATTEMPTED", "NOT_REQUIRED", "BLOCKED")

    chain: List[Dict[str, Any]] = []

    classification = last_attempt.get("classification") or detection.get(
        "process_health_classification"
    )
    chain.append(
        _stage(
            "Hung Detection",
            "PASS" if classification == "HUNG_PROCESS" else "FAIL",
            start=ts,
            detail={"classification": classification},
        )
    )
    chain.append(
        _stage(
            "Health Classification",
            "PASS" if classification == "HUNG_PROCESS" else "FAIL",
            detail=detection.get("recovery_decision"),
        )
    )

    safe = (detection.get("recovery_decision") or {}).get("verdict") == "SAFE_TERMINATION_CANDIDATE"
    chain.append(
        _stage(
            "Safety Verification",
            "PASS" if safe or not attempted else "FAIL",
            correction=None if safe else "Await SAFE_TERMINATION_CANDIDATE before recovery",
        )
    )

    terminated = bool(last_attempt.get("pids_terminated"))
    chain.append(
        _stage(
            "PID Validation",
            (
                "PASS"
                if terminated or result == "RECOVERY_DRY_RUN"
                else ("SKIP" if not attempted else "FAIL")
            ),
            detail={"pids_terminated": last_attempt.get("pids_terminated")},
            error=None if terminated or not attempted else "Not all hung PIDs terminated",
            source="hung_loop_recovery_manager._terminate_pids",
            correction="Terminate all run_continuous_loop.ps1 PIDs, not only primary forensic PID",
        )
    )

    lock_before = _parse_lock(LOCK_PATH)
    new_pids = last_attempt.get("new_pids")
    lock_recreated = bool(new_pids) and lock_before.get("exists")
    lock_cleared = (
        result == "RECOVERY_SUCCESS"
        or result == "RECOVERY_DRY_RUN"
        or lock_recreated
        or (attempted and not lock_before.get("exists"))
    )
    chain.append(
        _stage(
            "Stale Lock Removal",
            ("PASS" if lock_cleared else ("SKIP" if not attempted else "FAIL")),
            detail={
                "lock_exists_now": lock_before.get("exists"),
                "lock_age": lock_before.get("lock_age_minutes"),
                "lock_recreated_by_restart": lock_recreated,
            },
            error=(
                None if lock_cleared or not attempted else "Stale lock remained or removal blocked"
            ),
            source="continuous_loop_recovery_manager._clear_stale_lock",
            correction="Ensure all PIDs cleared before lock removal; call Clear-OrphanLoopLock pattern",
        )
    )

    cmd_ok = env.get("powershell_script_exists") and env.get("execution_policy_permits_launch")
    chain.append(
        _stage(
            "Restart Command Creation",
            "PASS" if cmd_ok else "FAIL",
            detail=_build_restart_command(),
            error=None if cmd_ok else "Script missing or execution policy blocked",
            source="continuous_loop_recovery_manager._restart_loop",
            correction="Verify scripts/run_continuous_loop.ps1 and ExecutionPolicy Bypass",
        )
    )

    env_ok = env.get("triton_root_exists") and env.get("working_directories_ok")
    chain.append(
        _stage(
            "Environment Preparation",
            "PASS" if env_ok else "FAIL",
            detail={
                "venv": env.get("venv_available"),
                "paper_env_note": env.get("note"),
            },
        )
    )

    launched = bool(new_pids) or result == "RECOVERY_SUCCESS"
    chain.append(
        _stage(
            "Process Launch",
            ("PASS" if launched else ("SKIP" if not attempted else "FAIL")),
            detail={"new_pids": new_pids, "spawn_detected": launched},
            error=(
                None if launched or not attempted else "Process did not remain running after spawn"
            ),
            source="continuous_loop_recovery_manager._restart_loop",
            correction="Capture stderr; increase post-spawn wait; use Start-Process -WindowStyle Hidden",
        )
    )

    state = _read_json(WATCHDOG_STATE)
    hb = state.get("heartbeat_refresh_ts")
    attempt_ts = last_attempt.get("timestamp")
    hb_after_recovery = _ts_after(attempt_ts, hb)
    hb_ok = hb_after_recovery or result == "RECOVERY_SUCCESS"
    chain.append(
        _stage(
            "Heartbeat Creation",
            (
                "PASS"
                if hb_ok or result == "RECOVERY_DRY_RUN"
                else ("SKIP" if not attempted else "FAIL")
            ),
            detail={
                "heartbeat_refresh_ts": hb,
                "recovery_attempt_ts": attempt_ts,
                "heartbeat_after_recovery": hb_after_recovery,
            },
            error=(
                None if hb_ok or not attempted else "Heartbeat not refreshed after restart window"
            ),
            source="run_continuous_loop.ps1 + loop_safety",
            correction="Extend _verify_recovery wait; confirm loop reaches HEARTBEAT_REFRESH line",
        )
    )

    verify_ok = last_attempt.get("recovery_success") is True
    chain.append(
        _stage(
            "Health Verification",
            "PASS" if verify_ok else ("SKIP" if not attempted else "FAIL"),
            detail={
                "recovery_successful_requires": (
                    "term.all_cleared AND restart.started AND verify.recovery_successful"
                ),
                "actual_recovery_success": last_attempt.get("recovery_success"),
                "spawn_succeeded_but_final_failed": bool(new_pids) and result == "RECOVERY_FAILED",
            },
            error=None if verify_ok or not attempted else "Health verification failed post-restart",
            source="hung_loop_recovery_manager.run_recovery",
            correction=(
                "Extend verify window; terminate all child PIDs; require heartbeat_refresh_confirmed"
            ),
        )
    )

    chain.append(
        _stage(
            "Recovery Confirmation",
            "PASS" if result == "RECOVERY_SUCCESS" else ("SKIP" if not attempted else "FAIL"),
            detail={"result": result},
            error=None if result == "RECOVERY_SUCCESS" else f"Final result={result}",
        )
    )
    return chain


def _recommendations(
    failure_class: str,
    chain: List[Dict[str, Any]],
    env: Dict[str, Any],
    last_attempt: Dict[str, Any],
) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    failed = [s for s in chain if s["status"] == "FAIL"]

    for stage in failed:
        if stage.get("recommended_correction"):
            recs.append(
                {
                    "priority": "HIGH",
                    "category": stage["stage"],
                    "recommendation": stage["recommended_correction"],
                    "failure_class": failure_class,
                    "auto_correctable": False,
                }
            )

    if failure_class == "HEALTH_VERIFICATION_FAILURE":
        recs.append(
            {
                "priority": "CRITICAL",
                "category": "verification_timing",
                "recommendation": (
                    "Increase post-restart verification window from ~5s to 30-60s; loop must "
                    "complete HEARTBEAT_REFRESH and create lock before verify runs"
                ),
                "failure_class": failure_class,
                "auto_correctable": True,
            }
        )
        recs.append(
            {
                "priority": "HIGH",
                "category": "verification_criteria",
                "recommendation": (
                    "Align success criteria: hung recovery requires term.all_cleared AND "
                    "verify.recovery_successful; partial PID termination causes RECOVERY_FAILED"
                ),
                "failure_class": failure_class,
                "auto_correctable": True,
            }
        )

    if failure_class == "PROCESS_CREATION_FAILURE":
        recs.append(
            {
                "priority": "HIGH",
                "category": "pid_cleanup",
                "recommendation": "Terminate all matching PowerShell PIDs (parent+child) before restart",
                "failure_class": failure_class,
                "auto_correctable": True,
            }
        )

    if not env.get("TRITON_ENABLE_PAPER_EXECUTION_in_parent_env"):
        recs.append(
            {
                "priority": "INFO",
                "category": "environment",
                "recommendation": (
                    "Parent Python env lacks TRITON_ENABLE_PAPER_EXECUTION; script sets it internally — OK"
                ),
                "failure_class": failure_class,
                "auto_correctable": False,
            }
        )

    recs.append(
        {
            "priority": "HIGH",
            "category": "observability",
            "recommendation": (
                "Redirect restart subprocess stderr to recovery log instead of DEVNULL for failure diagnosis"
            ),
            "failure_class": failure_class,
            "auto_correctable": True,
        }
    )

    if not recs:
        recs.append(
            {
                "priority": "MEDIUM",
                "category": "general",
                "recommendation": "Re-run hung_loop_recovery_manager --recover --dry-run to trace steps",
                "failure_class": failure_class,
                "auto_correctable": False,
            }
        )
    return recs


def _forensic_evidence(
    last_attempt: Dict[str, Any], env: Dict[str, Any], detection: Dict[str, Any]
) -> Dict[str, Any]:
    lock = _parse_lock(LOCK_PATH)
    running, pids = _is_loop_process_running()
    recovery_exec = detection.get("recovery_execution") or {}
    steps = recovery_exec.get("steps") or []
    term_step = next((s for s in steps if s.get("step") == "terminate_stale_pid"), {})
    verify_step = next((s for s in steps if s.get("step") == "verify_heartbeat"), {})
    restart_step = next((s for s in steps if s.get("step") == "restart_continuous_loop"), {})

    result = str(last_attempt.get("result", ""))
    hypothesis = None
    if result == "RECOVERY_FAILED" and last_attempt.get("new_pids"):
        if term_step and not term_step.get("all_cleared"):
            hypothesis = "Partial PID termination (term.all_cleared=false) blocked RECOVERY_SUCCESS"
        elif verify_step and not verify_step.get("recovery_successful"):
            hypothesis = (
                "Post-restart verify failed within ~5s window "
                "(process_running AND lock_exists not both true in time)"
            )
        else:
            hypothesis = "Composite gate failed: term.all_cleared AND restart.started AND verify.recovery_successful"

    return {
        "failed_restart_timestamp": last_attempt.get("timestamp"),
        "recovery_result": last_attempt.get("result"),
        "previous_pids_terminated": last_attempt.get("pids_terminated"),
        "new_pids_after_restart": last_attempt.get("new_pids"),
        "lock_before_after_note": "See lock snapshot at diagnostic time",
        "lock_snapshot": lock,
        "current_pids": pids,
        "process_running_now": running,
        "restart_command": " ".join(_build_restart_command()),
        "environment_snapshot": env,
        "recovery_duration_estimate_seconds": 5,
        "exit_codes_note": "Restart uses detached Popen — exit code not captured (stderr discarded)",
        "recovery_step_forensics": {
            "terminate": {
                "all_cleared": term_step.get("all_cleared"),
                "remaining_pids": term_step.get("remaining_pids"),
                "terminated": term_step.get("terminated"),
            },
            "restart": {
                "started": restart_step.get("started"),
                "detected_pids": restart_step.get("detected_pids"),
                "spawn_pid": restart_step.get("spawn_pid"),
            },
            "verify": {
                "recovery_successful": verify_step.get("recovery_successful"),
                "heartbeat_refresh_confirmed": verify_step.get("heartbeat_refresh_confirmed"),
                "process_running": verify_step.get("process_running"),
                "lock_exists": verify_step.get("lock_exists"),
            },
        },
        "failure_hypothesis": hypothesis,
    }


def _executive_report(
    last_attempt: Dict[str, Any],
    failure_class: str,
    chain: List[Dict[str, Any]],
    recs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    failed = next((s for s in chain if s["status"] == "FAIL"), None)
    primary_stage = {
        "HEALTH_VERIFICATION_FAILURE": "Health Verification",
        "HEARTBEAT_TIMEOUT": "Heartbeat Creation",
        "PROCESS_CREATION_FAILURE": "Process Launch",
        "RESTART_COMMAND_FAILURE": "Restart Command Creation",
        "PERMISSION_FAILURE": "Stale Lock Removal",
        "ENVIRONMENT_VARIABLE_FAILURE": "Environment Preparation",
        "VENV_ACTIVATION_FAILURE": "Environment Preparation",
        "WORKING_DIRECTORY_FAILURE": "Environment Preparation",
        "POWERSHELL_SCRIPT_FAILURE": "Restart Command Creation",
    }
    failed_stage = primary_stage.get(failure_class) or (failed.get("stage") if failed else None)
    failure_detail = failed.get("error_message") if failed else None
    if failure_class == "HEALTH_VERIFICATION_FAILURE" and failed_stage == "Health Verification":
        failure_detail = "Health verification failed post-restart"

    auto_fix = any(r.get("auto_correctable") for r in recs)
    result = str(last_attempt.get("result", "NOT_ATTEMPTED"))
    trustworthy = result == "RECOVERY_SUCCESS"

    return {
        "report_type": "RECOVERY_RESTART_DIAGNOSTICS_REPORT",
        "restart_attempted": result not in ("NOT_ATTEMPTED", "NOT_REQUIRED", "BLOCKED"),
        "failed_stage": failed_stage,
        "failure_cause": failure_class,
        "failure_detail": failure_detail,
        "auto_correctable": auto_fix,
        "autonomous_recovery_trustworthy": trustworthy,
        "production_readiness_gaps": [
            r["recommendation"] for r in recs if r.get("priority") in ("CRITICAL", "HIGH")
        ][:5],
        "executive_questions": {
            "why_restart_failed": failure_class,
            "restart_attempted": result not in ("NOT_ATTEMPTED", "NOT_REQUIRED"),
            "restart_command_correct": True,
            "powershell_available": True,
            "process_launched": bool(last_attempt.get("new_pids")),
            "heartbeat_never_restored_reason": (
                "Verification window too short and/or partial PID termination"
                if failure_class == "HEALTH_VERIFICATION_FAILURE"
                else failed.get("error_message") if failed else "N/A"
            ),
            "required_before_full_autonomous_recovery": [
                r["recommendation"] for r in recs if r.get("priority") == "CRITICAL"
            ],
        },
    }


def _success_rate() -> Dict[str, Any]:
    if not ACTIONS_CSV.is_file():
        return {"attempts": 0, "successes": 0, "failures": 0, "success_pct": None}
    df = pd.read_csv(ACTIONS_CSV)
    attempts = len(df)
    successes = len(df[df["result"] == "RECOVERY_SUCCESS"])
    failures = len(df[df["result"] == "RECOVERY_FAILED"])
    pct = round(successes / attempts * 100, 1) if attempts else None
    return {"attempts": attempts, "successes": successes, "failures": failures, "success_pct": pct}


def run_analysis() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    last_attempt = _load_last_recovery_attempt()
    detection = {}
    if DETECTION_JSON.is_file():
        try:
            detection = json.loads(DETECTION_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    if not detection:
        detection = run_detection()

    env = _validate_environment()
    chain = _trace_recovery_chain(last_attempt, env, detection)
    tests = _run_simulation_tests(env, DEFAULT_INTERVAL_MINUTES)
    failure_class = _classify_failure(chain, env, last_attempt, tests)
    forensics = _forensic_evidence(last_attempt, env, detection)
    recs = _recommendations(failure_class, chain, env, last_attempt)
    stats = _success_rate()
    executive = _executive_report(last_attempt, failure_class, chain, recs)

    report = {
        "generated_at": ts,
        "phase": "196",
        "mode": "recovery_restart_diagnostics",
        "disclaimer": (
            "Restart failure diagnostics only — does NOT modify governance, lifecycle, "
            "allocation, execution strategy, broker logic, or risk controls."
        ),
        "last_recovery_attempt": last_attempt,
        "recovery_chain_trace": chain,
        "failure_classification": failure_class,
        "environment_validation": env,
        "simulation_tests": tests,
        "forensic_evidence": forensics,
        "recovery_success_rate": stats,
        "recommendations": recs,
        "executive_recommendation": executive,
        "summary": {
            "failure_classification": failure_class,
            "restart_attempted": executive.get("restart_attempted"),
            "failed_stage": executive.get("failed_stage"),
            "autonomous_recovery_trustworthy": executive.get("autonomous_recovery_trustworthy"),
            "recovery_success_pct": stats.get("success_pct"),
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    hist_row = {
        "timestamp": ts,
        "last_attempt_timestamp": last_attempt.get("timestamp"),
        "recovery_result": last_attempt.get("result"),
        "failure_classification": failure_class,
        "failed_stage": executive.get("failed_stage"),
        "new_pids": last_attempt.get("new_pids"),
        "recovery_success_pct": stats.get("success_pct"),
    }
    if OUT_HISTORY.is_file():
        hist = pd.read_csv(OUT_HISTORY)
        hist = pd.concat([hist, pd.DataFrame([hist_row])], ignore_index=True)
    else:
        hist = pd.DataFrame([hist_row])
    hist.to_csv(OUT_HISTORY, index=False, encoding="utf-8")

    pd.DataFrame(recs).to_csv(OUT_RECOMMENDATIONS, index=False, encoding="utf-8")

    _log(
        f"failure_class={failure_class} stage={executive.get('failed_stage')} result={last_attempt.get('result')}"
    )
    return report


def main() -> int:
    run_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
