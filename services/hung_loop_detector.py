"""
TRITON Phase 195 — Hung Loop Detection Intelligence Engine.

Detects continuous loop processes that appear alive but are no longer performing
work (frozen PID paradox). Does NOT modify trading, governance, or execution logic.

Outputs (data/results/):
    hung_loop_detection.json
    hung_loop_forensics.json
    hung_loop_history.csv  (appended)

Run:
    python -m services.hung_loop_detector
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from services.continuous_loop_recovery_manager import (
    DEFAULT_INTERVAL_MINUTES,
    FAILURE_GRACE_MINUTES,
    LOCK_PATH,
    LOOP_LOG_DIR,
    WARNING_GRACE_MINUTES,
    _assess_health,
    _heartbeat_snapshot,
    _is_loop_process_running,
    _latest_loop_log,
    _minutes_since,
    _now_iso,
    _parse_lock,
    _parse_loop_log,
    _parse_ts,
    _read_json,
)

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "hung_loop_detection.json"
OUT_FORENSICS = RESULTS / "hung_loop_forensics.json"
OUT_HISTORY = RESULTS / "hung_loop_history.csv"


def _log(msg: str) -> None:
    print(f"[hung_loop_detector] {msg}", flush=True)


def _process_details(pids: List[int]) -> List[Dict[str, Any]]:
    if not pids or sys.platform != "win32":
        return [{"pid": p, "details_available": False} for p in pids]
    import subprocess

    rows = []
    for pid in pids:
        ps = (
            f'$p = Get-CimInstance Win32_Process -Filter "ProcessId = {pid}" -ErrorAction SilentlyContinue; '
            "if ($p) { @{ ProcessId = $p.ProcessId; CreationDate = $p.CreationDate; "
            "CommandLine = $p.CommandLine; WorkingSetSize = $p.WorkingSetSize } | ConvertTo-Json -Compress }"
        )
        detail: Dict[str, Any] = {"pid": pid, "valid": False}
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if r.stdout.strip():
                data = json.loads(r.stdout.strip())
                detail["valid"] = True
                detail["creation_date"] = data.get("CreationDate")
                detail["command_line"] = (data.get("CommandLine") or "")[:200]
                detail["working_set_bytes"] = data.get("WorkingSetSize")
                created = _parse_ts(str(data.get("CreationDate", "")).split(".")[0] + "Z")
                runtime = _minutes_since(created)
                if runtime is not None:
                    detail["runtime_minutes"] = round(runtime, 1)
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
            detail["valid"] = False
        rows.append(detail)
    return rows


def _log_activity(log_path: Optional[Path]) -> Dict[str, Any]:
    if not log_path or not log_path.is_file():
        return {"log_exists": False}
    stat = log_path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    age_min = _minutes_since(mtime)
    return {
        "log_exists": True,
        "log_path": str(log_path),
        "log_size_bytes": stat.st_size,
        "log_mtime": mtime.isoformat().replace("+00:00", "Z"),
        "log_mtime_age_minutes": round(age_min, 1) if age_min is not None else None,
        "log_stale": age_min is not None and age_min > WARNING_GRACE_MINUTES,
    }


def _cpu_snapshot(pids: List[int]) -> Dict[str, Any]:
    if not pids or sys.platform != "win32":
        return {"available": False}
    import subprocess

    try:
        pid_list = ",".join(str(p) for p in pids)
        ps = (
            f"Get-Process -Id {pid_list} -ErrorAction SilentlyContinue | "
            "Select-Object Id, CPU, WorkingSet | ConvertTo-Json -Compress"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if not r.stdout.strip():
            return {"available": False, "detail": "no process metrics"}
        data = json.loads(r.stdout.strip())
        if isinstance(data, dict):
            data = [data]
        return {"available": True, "processes": data}
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as e:
        return {"available": False, "detail": str(e)}


def _classify_process_health(
    process_running: bool,
    pids: List[int],
    cycle_age_min: Optional[float],
    log_activity: Dict[str, Any],
    hb: Dict[str, Any],
    interval_minutes: float,
) -> Tuple[str, List[str], Optional[str]]:
    """Return (HEALTHY|DEGRADED|HUNG_PROCESS, reasons, suspected_root_cause)."""
    reasons: List[str] = []
    warning_threshold = max(WARNING_GRACE_MINUTES, interval_minutes + 10)
    failure_threshold = max(FAILURE_GRACE_MINUTES, interval_minutes + 15)

    if not process_running:
        if cycle_age_min is not None and cycle_age_min > failure_threshold:
            return (
                "HUNG_PROCESS",
                ["process_absent_cycle_stale"],
                "Process terminated; cycles stale",
            )
        return "HEALTHY", ["no_process_expected_or_idle"], None

    refresh_age = hb.get("continuous_loop_refresh_age_minutes")
    log_stale = log_activity.get("log_stale") is True
    cycle_stale = cycle_age_min is not None and cycle_age_min > failure_threshold
    cycle_delayed = cycle_age_min is not None and cycle_age_min > warning_threshold
    refresh_stale = refresh_age is not None and refresh_age > failure_threshold

    if cycle_stale and (log_stale or refresh_stale):
        reasons.extend(["pid_alive", "cycle_not_progressing", "log_or_refresh_stale"])
        return (
            "HUNG_PROCESS",
            reasons,
            "Process frozen in memory — PID survives but cycles and logs no longer advance",
        )

    if cycle_stale and process_running:
        reasons.extend(["pid_alive", "cycle_stale_max_grace_exceeded"])
        return (
            "HUNG_PROCESS",
            reasons,
            "Process likely hung — no cycle completion within maximum grace window",
        )

    if cycle_delayed or (refresh_age and refresh_age > warning_threshold):
        reasons.append("cycle_or_heartbeat_delayed")
        return "DEGRADED", reasons, "Cycle running longer than expected — monitor before recovery"

    if process_running and not cycle_stale:
        return "HEALTHY", ["pid_alive", "cycle_progressing"], None

    return "DEGRADED", ["indeterminate"], "Insufficient signals — continue monitoring"


def _recovery_decision(
    classification: str,
    process_running: bool,
    cycle_age_min: Optional[float],
    broker_connected: Optional[bool],
) -> Dict[str, Any]:
    warning_threshold = WARNING_GRACE_MINUTES
    recent_cycle = cycle_age_min is not None and cycle_age_min < warning_threshold

    if classification == "HEALTHY" or recent_cycle:
        return {
            "verdict": "RECOVERY_BLOCKED",
            "reason": "Healthy or recent cycle activity — duplicate protection preserved",
            "safe_to_terminate": False,
            "safe_to_recover": False,
            "operator_action": "No recovery — loop functioning or cycle recent",
        }
    if classification == "DEGRADED":
        return {
            "verdict": "MONITOR",
            "reason": "Process degraded but not confirmed hung",
            "safe_to_terminate": False,
            "safe_to_recover": False,
            "operator_action": "Monitor; re-run detector after one expected cycle interval",
        }
    if classification == "HUNG_PROCESS":
        if broker_connected is False:
            return {
                "verdict": "RECOVERY_DEFERRED",
                "reason": "Hung process suspected but broker state uncertain",
                "safe_to_terminate": False,
                "safe_to_recover": False,
                "operator_action": "Resolve broker connectivity before hung recovery",
            }
        return {
            "verdict": "SAFE_TERMINATION_CANDIDATE",
            "reason": "PID exists without cycle/log progression — safe termination candidate",
            "safe_to_terminate": True,
            "safe_to_recover": True,
            "operator_action": "Run hung_loop_recovery_manager --recover after operator review",
        }
    return {
        "verdict": "MONITOR",
        "reason": "Unclassified",
        "safe_to_terminate": False,
        "safe_to_recover": False,
        "operator_action": "Investigate manually",
    }


def _build_forensics(
    ts: str,
    classification: str,
    pids: List[int],
    process_details: List[Dict[str, Any]],
    lock: Dict[str, Any],
    log_info: Dict[str, Any],
    log_activity: Dict[str, Any],
    hb: Dict[str, Any],
    cycle_age_min: Optional[float],
    root_cause: Optional[str],
    recovery_decision: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "captured_at": ts,
        "classification": classification,
        "process_ids": pids,
        "process_details": process_details,
        "lock": lock,
        "last_heartbeat": hb.get("continuous_loop_refresh_ts") or hb.get("watchdog_heartbeat_ts"),
        "last_heartbeat_age_minutes": hb.get("continuous_loop_refresh_age_minutes"),
        "last_successful_cycle": log_info.get("last_cycle_ok"),
        "last_cycle_age_minutes": cycle_age_min,
        "lock_age_minutes": lock.get("lock_age_minutes"),
        "log_location": log_activity.get("log_path"),
        "log_mtime_age_minutes": log_activity.get("log_mtime_age_minutes"),
        "failure_classification": classification,
        "suspected_root_cause": root_cause,
        "recovery_decision": recovery_decision,
    }


def _executive_report(
    classification: str,
    process_running: bool,
    cycle_age_min: Optional[float],
    recovery_decision: Dict[str, Any],
    forensics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "report_type": "HUNG_LOOP_RECOVERY_REPORT",
        "loop_truly_alive": classification == "HEALTHY",
        "process_progressing": classification == "HEALTHY",
        "loop_hung": classification == "HUNG_PROCESS",
        "recovery_safe": recovery_decision.get("safe_to_recover") is True,
        "evidence_summary": {
            "pids": forensics.get("process_ids"),
            "last_cycle_age_minutes": cycle_age_min,
            "log_stale": forensics.get("log_mtime_age_minutes"),
            "classification": classification,
            "root_cause": forensics.get("suspected_root_cause"),
        },
        "recommended_operator_action": recovery_decision.get("operator_action"),
        "why_self_healing_refused": (
            None
            if recovery_decision.get("verdict") == "SAFE_TERMINATION_CANDIDATE"
            else recovery_decision.get("reason")
        ),
    }


def run_detection(*, interval_minutes: int = DEFAULT_INTERVAL_MINUTES) -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    now = datetime.now(timezone.utc)

    _, phase194_metrics = _assess_health(interval_minutes)
    process_running, pids = _is_loop_process_running()
    lock = _parse_lock(LOCK_PATH)
    log_path = _latest_loop_log()
    log_info = _parse_loop_log(log_path)
    log_activity = _log_activity(log_path)
    hb = _heartbeat_snapshot()
    process_details = _process_details(pids)
    cpu = _cpu_snapshot(pids)

    cycle_age_min = phase194_metrics.get("last_cycle_age_minutes")
    if cycle_age_min is None and log_info.get("last_cycle_ok_dt"):
        cycle_age_min = _minutes_since(log_info["last_cycle_ok_dt"], now)

    classification, reasons, root_cause = _classify_process_health(
        process_running,
        pids,
        cycle_age_min,
        log_activity,
        hb,
        interval_minutes,
    )

    wd = _read_json(RESULTS / "watchdog_status.json")
    broker_connected = wd.get("broker_connected")

    recovery_decision = _recovery_decision(
        classification, process_running, cycle_age_min, broker_connected
    )

    forensics = _build_forensics(
        ts,
        classification,
        pids,
        process_details,
        lock,
        log_info,
        log_activity,
        hb,
        cycle_age_min,
        root_cause,
        recovery_decision,
    )

    executive = _executive_report(
        classification, process_running, cycle_age_min, recovery_decision, forensics
    )

    intelligence = {
        "process_existence": process_running,
        "process_pids": pids,
        "process_details": process_details,
        "process_runtime_minutes": max(
            (d.get("runtime_minutes") or 0 for d in process_details), default=0
        ),
        "heartbeat": hb,
        "last_completed_cycle": log_info.get("last_cycle_ok"),
        "last_cycle_age_minutes": cycle_age_min,
        "expected_interval_minutes": interval_minutes,
        "consecutive_missed_cycles": phase194_metrics.get("consecutive_missed_cycles", 0),
        "log_activity": log_activity,
        "cpu_snapshot": cpu,
        "phase194_health": phase194_metrics.get("health_classification"),
    }

    report = {
        "generated_at": ts,
        "phase": "195",
        "mode": "hung_loop_detector",
        "disclaimer": (
            "Hung loop detection only — does NOT modify governance, lifecycle, allocation, "
            "execution strategy, broker trading logic, or risk controls."
        ),
        "process_health_classification": classification,
        "classification_reasons": reasons,
        "intelligence": intelligence,
        "recovery_decision": recovery_decision,
        "forensics": forensics,
        "executive_recommendation": executive,
        "summary": {
            "classification": classification,
            "process_running": process_running,
            "recovery_verdict": recovery_decision.get("verdict"),
            "safe_termination_candidate": recovery_decision.get("safe_to_terminate"),
            "safe_to_recover": recovery_decision.get("safe_to_recover"),
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    OUT_FORENSICS.write_text(json.dumps(forensics, indent=2, default=str), encoding="utf-8")

    hist_row = {
        "timestamp": ts,
        "classification": classification,
        "process_running": process_running,
        "pids": ",".join(str(p) for p in pids),
        "cycle_age_minutes": cycle_age_min,
        "recovery_verdict": recovery_decision.get("verdict"),
        "blocking_reason": recovery_decision.get("reason"),
    }
    if OUT_HISTORY.is_file():
        hist = pd.read_csv(OUT_HISTORY)
        hist = pd.concat([hist, pd.DataFrame([hist_row])], ignore_index=True)
    else:
        hist = pd.DataFrame([hist_row])
    hist.to_csv(OUT_HISTORY, index=False, encoding="utf-8")

    _log(
        f"classification={classification} verdict={recovery_decision.get('verdict')} "
        f"pids={pids} cycle_age={cycle_age_min}"
    )
    return report


def main() -> int:
    run_detection()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
