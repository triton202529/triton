"""
TRITON Phase 194 — Continuous Loop Self-Healing & Recovery Manager.

Monitors 45-minute continuous loop health, classifies failures, and optionally
performs safe recovery (stale lock clear + loop restart). Does NOT modify
governance, lifecycle, allocation, or execution strategy logic.

Outputs (data/results/):
    continuous_loop_health.json
    continuous_loop_recovery_history.csv  (appended)
    continuous_loop_failure_events.csv    (appended on failures/recoveries)

Run:
    python -m services.continuous_loop_recovery_manager
    python -m services.continuous_loop_recovery_manager --recover
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
LOOP_LOG_DIR = RESULTS / "continuous_loop"
LOCK_PATH = LIVE / "continuous_loop.lock"
LOOP_SCRIPT = ROOT / "scripts" / "run_continuous_loop.ps1"
WATCHDOG_STATUS = RESULTS / "watchdog_status.json"
WATCHDOG_ALERTS = RESULTS / "watchdog_alerts.json"
WATCHDOG_STATE = LIVE / "watchdog_alert_state.json"
PIPELINE_HEARTBEAT = RESULTS / "heartbeat.json"

OUT_JSON = RESULTS / "continuous_loop_health.json"
OUT_HISTORY = RESULTS / "continuous_loop_recovery_history.csv"
OUT_FAILURES = RESULTS / "continuous_loop_failure_events.csv"

DEFAULT_INTERVAL_MINUTES = 45
WARNING_GRACE_MINUTES = 55.0
FAILURE_GRACE_MINUTES = 60.0
LOOP_SCRIPT_NAME = "run_continuous_loop.ps1"


def _log(msg: str) -> None:
    print(f"[continuous_loop_recovery_manager] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _minutes_since(dt: Optional[datetime], now: Optional[datetime] = None) -> Optional[float]:
    if not dt:
        return None
    ref = now or datetime.now(timezone.utc)
    return (ref - dt).total_seconds() / 60.0


def _is_loop_process_running() -> Tuple[bool, List[int]]:
    """Return (running, pids) for run_continuous_loop.ps1 on Windows."""
    if sys.platform != "win32":
        return False, []
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name = 'powershell.exe' OR Name = 'pwsh.exe'\" "
        f"| Where-Object {{ $_.CommandLine -like '*{LOOP_SCRIPT_NAME}*' }} "
        "| Select-Object -ExpandProperty ProcessId"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        pids = []
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return len(pids) > 0, pids
    except (subprocess.SubprocessError, OSError):
        return False, []


def _parse_lock(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    info: Dict[str, Any] = {"exists": True, "raw": text}
    m = re.search(r"started_at=([^\r\n]+)", text)
    if m:
        info["started_at"] = m.group(1).strip()
        info["started_at_dt"] = _parse_ts(info["started_at"])
    m_pid = re.search(r"pid=(\d+)", text, re.I)
    if m_pid:
        info["pid"] = int(m_pid.group(1))
    age = _minutes_since(info.get("started_at_dt"))
    if age is not None:
        info["lock_age_minutes"] = round(age, 1)
    return info


def _latest_loop_log() -> Optional[Path]:
    if not LOOP_LOG_DIR.is_dir():
        return None
    logs = sorted(LOOP_LOG_DIR.glob("continuous_loop_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _parse_loop_log(path: Optional[Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "log_path": str(path) if path else None,
        "last_cycle_start": None,
        "last_cycle_ok": None,
        "last_cycle_fail": None,
        "cycle_count": 0,
        "consecutive_ok": 0,
    }
    if not path or not path.is_file():
        return out

    last_start = last_ok = last_fail = None
    cycle_count = 0
    consecutive_ok = 0
    temp_ok = 0

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[CYCLE_START]" in line:
            cycle_count += 1
            ts_part = line.split()[0] if line else None
            last_start = ts_part
            temp_ok = 0
        elif "[CYCLE_OK]" in line:
            ts_part = line.split()[0] if line else None
            last_ok = ts_part
            temp_ok += 1
            consecutive_ok = temp_ok
        elif "[CYCLE_FAIL]" in line:
            ts_part = line.split()[0] if line else None
            last_fail = ts_part
            consecutive_ok = 0

    out.update(
        {
            "last_cycle_start": last_start,
            "last_cycle_ok": last_ok,
            "last_cycle_fail": last_fail,
            "cycle_count": cycle_count,
            "consecutive_ok": consecutive_ok,
            "last_cycle_ok_dt": _parse_ts(last_ok.replace(" ", "T") + "Z") if last_ok else None,
            "last_cycle_start_dt": (
                _parse_ts(last_start.replace(" ", "T") + "Z") if last_start else None
            ),
        }
    )
    return out


def _heartbeat_snapshot() -> Dict[str, Any]:
    wd = _read_json(WATCHDOG_STATUS)
    state = _read_json(WATCHDOG_STATE)
    pipe = _read_json(PIPELINE_HEARTBEAT)
    hb = wd.get("heartbeat") if isinstance(wd.get("heartbeat"), dict) else {}
    wd_ts = _parse_ts(hb.get("timestamp") or wd.get("timestamp"))
    refresh_ts = _parse_ts(state.get("heartbeat_refresh_ts"))
    last_cycle_ts = _parse_ts(state.get("last_cycle_ts"))
    now = datetime.now(timezone.utc)

    stale_alerts = []
    alerts_doc = _read_json(WATCHDOG_ALERTS)
    for alert in alerts_doc.get("active_alerts") or []:
        if isinstance(alert, dict) and alert.get("alert_type") == "STALE_HEARTBEAT":
            stale_alerts.append(alert)

    return {
        "watchdog_heartbeat_ts": hb.get("timestamp") or wd.get("timestamp"),
        "watchdog_heartbeat_age_minutes": _minutes_since(wd_ts, now),
        "continuous_loop_refresh_ts": state.get("heartbeat_refresh_ts"),
        "continuous_loop_refresh_age_minutes": _minutes_since(refresh_ts, now),
        "refresh_source": state.get("heartbeat_refresh_source"),
        "last_cycle_ts": state.get("last_cycle_ts"),
        "last_cycle_age_minutes": _minutes_since(last_cycle_ts, now),
        "pipeline_heartbeat_ts": pipe.get("timestamp"),
        "stale_heartbeat_alerts_active": len(stale_alerts),
        "grace_until": state.get("heartbeat_grace_until"),
    }


def _classify_failure(
    process_running: bool,
    lock: Dict[str, Any],
    log_info: Dict[str, Any],
    hb: Dict[str, Any],
    health: str,
) -> Optional[str]:
    if health != "FAILED":
        return None
    if lock.get("exists") and not process_running:
        return "LOCK_FAILURE"
    if hb.get("stale_heartbeat_alerts_active", 0) > 0:
        return "HEARTBEAT_FAILURE"
    if not process_running and not lock.get("exists"):
        return "PROCESS_FAILURE"
    if process_running and log_info.get("last_cycle_ok_dt"):
        age = _minutes_since(log_info["last_cycle_ok_dt"])
        if age and age > FAILURE_GRACE_MINUTES:
            return "HEARTBEAT_FAILURE"
    if process_running:
        return "UNEXPECTED_EXIT"
    return "UNKNOWN_FAILURE"


def _investigate_root_cause(
    failure_class: Optional[str],
    process_running: bool,
    lock: Dict[str, Any],
    log_info: Dict[str, Any],
) -> str:
    if failure_class == "LOCK_FAILURE":
        return "Stale lock file — process terminated without finally block cleanup"
    if failure_class == "PROCESS_FAILURE":
        return "Process terminated unexpectedly — no active loop or lock"
    if failure_class == "HEARTBEAT_FAILURE":
        return "Heartbeat or cycle completion exceeded maximum grace period"
    if failure_class == "UNEXPECTED_EXIT":
        return "Process exists but cycle completion stale — possible hung process"
    if not process_running and log_info.get("last_cycle_fail"):
        return "Application exception — last cycle recorded as failed"
    return "Operating system interruption or manual termination (unconfirmed)"


def _assess_health(
    interval_minutes: float,
) -> Tuple[str, Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    process_running, pids = _is_loop_process_running()
    lock = _parse_lock(LOCK_PATH)
    log_info = _parse_loop_log(_latest_loop_log())
    hb = _heartbeat_snapshot()

    last_ok_dt = log_info.get("last_cycle_ok_dt")
    last_ok_age = _minutes_since(last_ok_dt, now)
    refresh_age = hb.get("continuous_loop_refresh_age_minutes")

    cycle_reference_age = last_ok_age
    if cycle_reference_age is None and refresh_age is not None:
        cycle_reference_age = refresh_age

    missed_cycles = 0
    if cycle_reference_age is not None and interval_minutes > 0:
        missed_cycles = max(0, int(cycle_reference_age // interval_minutes) - 1)

    warning_threshold = max(WARNING_GRACE_MINUTES, interval_minutes + 10)
    failure_threshold = max(FAILURE_GRACE_MINUTES, interval_minutes + 15)

    health = "HEALTHY"
    reasons: List[str] = []

    if not process_running and lock.get("exists"):
        health = "FAILED"
        reasons.append("orphaned_lock_process_dead")
    elif not process_running:
        if cycle_reference_age is None or cycle_reference_age > failure_threshold:
            health = "FAILED"
            reasons.append("process_not_running")
        elif cycle_reference_age > warning_threshold:
            health = "WARNING"
            reasons.append("process_not_running_delayed")
    elif process_running:
        if cycle_reference_age is not None and cycle_reference_age > failure_threshold:
            health = "FAILED"
            reasons.append("cycle_stale_while_process_alive")
        elif cycle_reference_age is not None and cycle_reference_age > warning_threshold:
            health = "WARNING"
            reasons.append("cycle_delayed")
        elif hb.get("stale_heartbeat_alerts_active", 0) > 0:
            health = "WARNING"
            reasons.append("stale_heartbeat_alert_active")

    if health == "HEALTHY" and not process_running and not lock.get("exists"):
        health = "FAILED"
        reasons.append("no_process_no_lock")

    failure_class = _classify_failure(process_running, lock, log_info, hb, health)

    next_cycle_dt = None
    if last_ok_dt:
        next_cycle_dt = last_ok_dt + timedelta(minutes=interval_minutes)

    metrics = {
        "assessed_at": _now_iso(),
        "health_classification": health,
        "process_running": process_running,
        "process_pids": pids,
        "lock": lock,
        "log": log_info,
        "heartbeat": hb,
        "interval_minutes": interval_minutes,
        "last_cycle_age_minutes": round(cycle_reference_age, 1) if cycle_reference_age else None,
        "consecutive_missed_cycles": missed_cycles,
        "failure_classification": failure_class,
        "root_cause_hypothesis": _investigate_root_cause(
            failure_class, process_running, lock, log_info
        ),
        "health_reasons": reasons,
        "next_expected_cycle": (
            next_cycle_dt.isoformat().replace("+00:00", "Z") if next_cycle_dt else None
        ),
        "operational_risk_level": (
            "CRITICAL" if health == "FAILED" else "ELEVATED" if health == "WARNING" else "NORMAL"
        ),
    }
    return health, metrics


def _duplicate_protection(metrics: Dict[str, Any]) -> Dict[str, Any]:
    running = metrics.get("process_running") is True
    lock = metrics.get("lock") or {}
    hb = metrics.get("heartbeat") or {}
    refresh_age = hb.get("continuous_loop_refresh_age_minutes")
    recent_hb = refresh_age is not None and refresh_age < WARNING_GRACE_MINUTES

    if running and lock.get("exists"):
        verdict = "RECOVERY_BLOCKED"
        reason = "Active continuous loop process and lock present"
    elif running:
        verdict = "RECOVERY_BLOCKED"
        reason = "Active continuous loop process detected"
    elif lock.get("exists") and metrics.get("process_running"):
        verdict = "RECOVERY_BLOCKED"
        reason = "Lock held by live process"
    elif running and recent_hb:
        verdict = "RECOVERY_BLOCKED"
        reason = "Recent heartbeat with active process"
    else:
        verdict = "SAFE_TO_RECOVER"
        reason = "No active loop process; recovery may proceed"

    return {
        "verdict": verdict,
        "reason": reason,
        "safe_to_recover": verdict == "SAFE_TO_RECOVER",
    }


def _clear_stale_lock(metrics: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    lock = metrics.get("lock") or {}
    if not lock.get("exists"):
        return {"action": "none", "cleared": False, "detail": "No lock file"}
    if metrics.get("process_running"):
        return {"action": "blocked", "cleared": False, "detail": "Process still running"}
    if dry_run:
        return {"action": "would_clear", "cleared": False, "detail": str(LOCK_PATH)}
    try:
        LOCK_PATH.unlink(missing_ok=True)
        _log(f"cleared stale lock {LOCK_PATH}")
        return {"action": "cleared", "cleared": True, "detail": str(LOCK_PATH)}
    except OSError as e:
        return {"action": "error", "cleared": False, "detail": str(e)}


def _restart_loop(interval_minutes: int, dry_run: bool) -> Dict[str, Any]:
    if not LOOP_SCRIPT.is_file():
        return {"action": "error", "started": False, "detail": "Loop script missing"}
    if dry_run:
        return {
            "action": "would_restart",
            "started": False,
            "detail": str(LOOP_SCRIPT),
            "command": f"powershell -File {LOOP_SCRIPT} -IntervalMinutes {interval_minutes}",
        }
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(LOOP_SCRIPT),
        "-IntervalMinutes",
        str(interval_minutes),
    ]
    try:
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        time.sleep(3)
        running, pids = _is_loop_process_running()
        return {
            "action": "restarted",
            "started": running,
            "spawn_pid": proc.pid,
            "detected_pids": pids,
            "detail": "Background restart initiated",
        }
    except OSError as e:
        return {"action": "error", "started": False, "detail": str(e)}


def _verify_recovery(
    before_refresh: Optional[str],
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
) -> Dict[str, Any]:
    from services.recovery_stabilization_monitor import adaptive_verify_recovery

    return adaptive_verify_recovery(
        before_refresh,
        interval_minutes=interval_minutes,
    )


def _attempt_recovery(
    metrics: Dict[str, Any],
    interval_minutes: int,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    dup = _duplicate_protection(metrics)
    if not dup["safe_to_recover"]:
        return {
            "attempted": False,
            "result": "RECOVERY_BLOCKED",
            "duplicate_protection": dup,
            "steps": [],
        }

    state_before = _read_json(WATCHDOG_STATE)
    refresh_before = state_before.get("heartbeat_refresh_ts")
    steps: List[Dict[str, Any]] = []

    lock_step = _clear_stale_lock(metrics, dry_run=dry_run)
    steps.append({"step": "clear_stale_lock", **lock_step})

    restart_step = _restart_loop(interval_minutes, dry_run=dry_run)
    steps.append({"step": "restart_loop", **restart_step})

    verify = {}
    if not dry_run and restart_step.get("started"):
        verify = _verify_recovery(refresh_before, interval_minutes=interval_minutes)
        steps.append({"step": "verify_stabilization", **verify})

    success = verify.get("recovery_successful") if verify else dry_run
    return {
        "attempted": True,
        "result": (
            "RECOVERY_SUCCESS" if success else "RECOVERY_PENDING" if dry_run else "RECOVERY_FAILED"
        ),
        "duplicate_protection": dup,
        "steps": steps,
        "verification": verify,
    }


def _load_history_stats() -> Dict[str, Any]:
    stats = {
        "total_failures_detected": 0,
        "total_recoveries_attempted": 0,
        "total_recoveries_successful": 0,
        "recovery_success_pct": None,
        "last_restart": None,
    }
    if OUT_FAILURES.is_file():
        df = pd.read_csv(OUT_FAILURES)
        stats["total_failures_detected"] = len(df[df["event_type"] == "FAILURE_DETECTED"])
        rec = df[df["event_type"].isin(["RECOVERY_ATTEMPT", "RECOVERY_SUCCESS", "RECOVERY_FAILED"])]
        stats["total_recoveries_attempted"] = len(rec)
        stats["total_recoveries_successful"] = len(df[df["event_type"] == "RECOVERY_SUCCESS"])
        if stats["total_recoveries_attempted"]:
            stats["recovery_success_pct"] = round(
                stats["total_recoveries_successful"] / stats["total_recoveries_attempted"] * 100, 1
            )
        restarts = df[df["event_type"] == "RECOVERY_SUCCESS"]
        if len(restarts):
            stats["last_restart"] = str(restarts.iloc[-1]["timestamp"])
    return stats


def _append_history_row(
    ts: str,
    health: str,
    metrics: Dict[str, Any],
    recovery: Optional[Dict[str, Any]],
) -> None:
    gate_status = health
    blocking = metrics.get("failure_classification") or metrics.get("health_reasons")
    row = {
        "timestamp": ts,
        "readiness_score": 100 if health == "HEALTHY" else 50 if health == "WARNING" else 0,
        "health_classification": health,
        "authorization_status": health,
        "process_running": metrics.get("process_running"),
        "blocking_reason": str(blocking),
        "recommended_action": (
            "No action required"
            if health == "HEALTHY"
            else (recovery or {}).get("result", "Investigate or run with --recover")
        ),
    }
    if OUT_HISTORY.is_file():
        hist = pd.read_csv(OUT_HISTORY)
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    hist.to_csv(OUT_HISTORY, index=False, encoding="utf-8")


def _append_failure_event(row: Dict[str, Any]) -> None:
    if OUT_FAILURES.is_file():
        df = pd.read_csv(OUT_FAILURES)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(OUT_FAILURES, index=False, encoding="utf-8")


def _executive_report(
    health: str,
    metrics: Dict[str, Any],
    recovery: Optional[Dict[str, Any]],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "report_type": "CONTINUOUS_LOOP_RESILIENCE_REPORT",
        "current_operational_status": health,
        "last_loop_restart": stats.get("last_restart"),
        "consecutive_successful_cycles": metrics.get("log", {}).get("consecutive_ok", 0),
        "total_failures_detected": stats.get("total_failures_detected"),
        "total_recoveries_completed": stats.get("total_recoveries_successful"),
        "recovery_success_percentage": stats.get("recovery_success_pct"),
        "operational_risk_level": metrics.get("operational_risk_level"),
        "remaining_operational_risks": [
            r
            for r in [
                "Stale lock without auto-recovery" if health == "FAILED" else None,
                "Manual restart required" if health == "FAILED" and not recovery else None,
                "Cycle delay" if health == "WARNING" else None,
            ]
            if r
        ],
        "recommended_operator_action": (
            "No action — loop healthy"
            if health == "HEALTHY"
            else (
                "Run with --recover if failure confirmed and market/testing window appropriate"
                if health == "FAILED"
                else "Monitor — cycle delayed but process may still be active"
            )
        ),
    }


def run_analysis(
    *,
    recover: bool = False,
    dry_run: bool = False,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
) -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LIVE.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    health, metrics = _assess_health(interval_minutes)
    dup = _duplicate_protection(metrics)
    recovery: Optional[Dict[str, Any]] = None

    if health == "FAILED" and recover:
        recovery = _attempt_recovery(metrics, interval_minutes, dry_run=dry_run)
        if recovery.get("result") == "RECOVERY_SUCCESS":
            health, metrics = _assess_health(interval_minutes)
    elif health == "FAILED":
        recovery = {
            "attempted": False,
            "result": "NOT_ATTEMPTED",
            "detail": "Use --recover to attempt safe self-healing",
            "duplicate_protection": dup,
        }

    stats = _load_history_stats()

    report = {
        "generated_at": ts,
        "phase": "194",
        "mode": "continuous_loop_recovery_manager",
        "disclaimer": (
            "Operational monitoring and optional loop recovery only — does NOT modify "
            "governance, lifecycle, allocation, execution strategy, or risk controls."
        ),
        "current_status": {
            "health_classification": health,
            "process_running": metrics.get("process_running"),
            "lock_exists": (metrics.get("lock") or {}).get("exists"),
            "interval_minutes": interval_minutes,
            "last_heartbeat": metrics.get("heartbeat"),
            "last_completed_cycle": metrics.get("log", {}).get("last_cycle_ok"),
            "next_expected_cycle": metrics.get("next_expected_cycle"),
            "uptime_cycles": metrics.get("log", {}).get("cycle_count"),
        },
        "health_metrics": metrics,
        "duplicate_loop_protection": dup,
        "recovery": recovery,
        "history_summary": stats,
        "executive_recommendation": _executive_report(health, metrics, recovery, stats),
        "summary": {
            "health": health,
            "failure_classification": metrics.get("failure_classification"),
            "safe_to_recover": dup.get("safe_to_recover"),
            "recovery_result": (recovery or {}).get("result"),
            "recover_mode": recover,
            "dry_run": dry_run,
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _append_history_row(ts, health, metrics, recovery)

    if health in ("FAILED", "WARNING"):
        _append_failure_event(
            {
                "timestamp": ts,
                "event_type": "FAILURE_DETECTED",
                "health_classification": health,
                "failure_classification": metrics.get("failure_classification"),
                "root_cause": metrics.get("root_cause_hypothesis"),
                "process_running": metrics.get("process_running"),
                "estimated_downtime_minutes": metrics.get("last_cycle_age_minutes"),
                "recovery_result": (recovery or {}).get("result"),
            }
        )

    if recovery and recovery.get("attempted"):
        event_type = (
            "RECOVERY_SUCCESS"
            if recovery.get("result") == "RECOVERY_SUCCESS"
            else (
                "RECOVERY_FAILED"
                if recovery.get("result") == "RECOVERY_FAILED"
                else "RECOVERY_ATTEMPT"
            )
        )
        _append_failure_event(
            {
                "timestamp": ts,
                "event_type": event_type,
                "health_classification": health,
                "failure_classification": metrics.get("failure_classification"),
                "root_cause": metrics.get("root_cause_hypothesis"),
                "process_running": metrics.get("process_running"),
                "estimated_downtime_minutes": metrics.get("last_cycle_age_minutes"),
                "recovery_result": recovery.get("result"),
            }
        )

    _log(f"health={health} failure={metrics.get('failure_classification')} recover={recover}")
    if recovery:
        _log(f"recovery_result={(recovery or {}).get('result')} safe={dup.get('verdict')}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="TRITON continuous loop health & recovery manager")
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Attempt safe recovery when health=FAILED and SAFE_TO_RECOVER",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate recovery steps without clearing lock or restarting",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=DEFAULT_INTERVAL_MINUTES,
        help=f"Expected cycle interval (default {DEFAULT_INTERVAL_MINUTES})",
    )
    args = parser.parse_args()
    run_analysis(
        recover=args.recover,
        dry_run=args.dry_run,
        interval_minutes=args.interval_minutes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
