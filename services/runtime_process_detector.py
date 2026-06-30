"""
runtime_process_detector.py
Phase 197Q-FIX — Strict runtime loop process detection (watchdog vs continuous).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"

WATCHDOG_LOCK = LIVE / "watchdog_loop.lock"
CONTINUOUS_LOCK = LIVE / "continuous_loop.lock"

WATCHDOG_SCRIPT = "run_watchdog_loop.ps1"
CONTINUOUS_SCRIPT = "run_continuous_loop.ps1"

RUNTIME_PROCESS_CONFIRMED = "RUNTIME_PROCESS_CONFIRMED"
RUNTIME_PROCESS_STOPPED_LOCK_ABSENT = "RUNTIME_PROCESS_STOPPED_LOCK_ABSENT"
RUNTIME_PROCESS_STOPPED_LOCK_STALE = "RUNTIME_PROCESS_STOPPED_LOCK_STALE"
RUNTIME_PROCESS_RUNNING_LOCK_PRESENT = "RUNTIME_PROCESS_RUNNING_LOCK_PRESENT"
RUNTIME_PROCESS_RUNNING_LOCK_MISSING = "RUNTIME_PROCESS_RUNNING_LOCK_MISSING"
RUNTIME_PROCESS_INCONSISTENT = "RUNTIME_PROCESS_INCONSISTENT"

_STALE_LOCK_CLASSIFICATIONS = frozenset(
    {
        "STALE_ORPHAN_LOCK",
        "STALE_ACTIVE_LOCK",
        "CORRUPT_LOCK",
    }
)

_EXCLUDE_CMD_SUBSTRINGS = (
    "phase_197",
    "runtime_reentry_observation_validation",
    "runtime_process_and_env",
    "show_phase_197",
    "Get-CimInstance Win32_Process",
    "Where-Object",
)


def _script_invocation_pattern(script_name: str) -> re.Pattern[str]:
    escaped = re.escape(script_name)
    return re.compile(
        rf"(?:-File\s+[\"']?[^\s\"']*[/\\])?{escaped}(?:\s|$|\"|'|-)",
        re.IGNORECASE,
    )


def _command_invokes_loop_script(command_line: str, script_name: str) -> bool:
    cmd = command_line or ""
    lower = cmd.lower()
    if script_name.lower() not in lower:
        return False
    for fragment in _EXCLUDE_CMD_SUBSTRINGS:
        if fragment.lower() in lower:
            return False
    if _script_invocation_pattern(script_name).search(cmd):
        return True
    return False


def _find_loop_processes(script_name: str) -> Tuple[List[int], str, List[str]]:
    """
    Return (pids, method, warnings) for active loop script processes.
    Excludes the detection subprocess itself (self-match on query text).
    """
    warnings: List[str] = []
    if sys.platform != "win32":
        warnings.append("non_windows_platform_process_detection_unavailable")
        return [], "unavailable", warnings

    escaped = script_name.replace("'", "''")
    ps = (
        f"$pat = [regex]::Escape('{escaped}'); "
        f"$rx = '(?:-File\\s+[\"'']?[^\\s\"'']*[/\\\\])?' + $pat + '(?:\\s|$|\"|'')'; "
        "Get-CimInstance Win32_Process -Filter \"Name = 'powershell.exe' OR Name = 'pwsh.exe'\" "
        "| Where-Object { "
        "$_.ProcessId -ne $PID "
        "-and $_.CommandLine "
        "-and ($_.CommandLine -match $rx) "
        "-and ($_.CommandLine -notmatch 'Get-CimInstance Win32_Process') "
        "-and ($_.CommandLine -notmatch 'Where-Object') "
        "} "
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
        pids = [int(x.strip()) for x in (r.stdout or "").splitlines() if x.strip().isdigit()]
        method = "win32_process_commandline_file_invocation"
        if r.stderr and r.stderr.strip():
            warnings.append(f"powershell_stderr={r.stderr.strip()[:200]}")
        return pids, method, warnings
    except (subprocess.SubprocessError, OSError) as exc:
        warnings.append(f"process_scan_failed={exc}")
        return [], "failed", warnings


def classify_runtime_process_lock_state(
    *,
    process_running: bool,
    lock_exists: bool,
    lock_classification: Optional[str] = None,
) -> str:
    lock_class = str(lock_classification or "")
    if process_running and lock_exists:
        return RUNTIME_PROCESS_RUNNING_LOCK_PRESENT
    if process_running and not lock_exists:
        return RUNTIME_PROCESS_RUNNING_LOCK_MISSING
    if not process_running and not lock_exists:
        return RUNTIME_PROCESS_STOPPED_LOCK_ABSENT
    if not process_running and lock_exists:
        if lock_class in _STALE_LOCK_CLASSIFICATIONS:
            return RUNTIME_PROCESS_STOPPED_LOCK_STALE
        return RUNTIME_PROCESS_INCONSISTENT
    return RUNTIME_PROCESS_INCONSISTENT


def detect_runtime_loop_processes() -> Dict[str, Any]:
    """Detect watchdog and continuous loop processes with diagnostics."""
    from services.phase_195h_operational_readiness_freeze import _parse_lock
    from services.runtime_lock_classifier import audit_runtime_locks

    wd_pids, wd_method, wd_warnings = _find_loop_processes(WATCHDOG_SCRIPT)
    cl_pids, cl_method, cl_warnings = _find_loop_processes(CONTINUOUS_SCRIPT)

    watchdog_running = len(wd_pids) > 0
    continuous_running = len(cl_pids) > 0

    wd_lock = _parse_lock(WATCHDOG_LOCK)
    cl_lock = _parse_lock(CONTINUOUS_LOCK)
    lock_audit = audit_runtime_locks(
        watchdog_lock=wd_lock,
        continuous_lock=cl_lock,
        watchdog_running=watchdog_running,
        continuous_running=continuous_running,
        watchdog_pids=wd_pids,
        continuous_pids=cl_pids,
    )

    wd_lock_class = str(lock_audit.get("watchdog_lock_classification") or "")
    cl_lock_class = str(lock_audit.get("continuous_lock_classification") or "")

    warnings = list(wd_warnings) + list(cl_warnings)
    confidence = "high"
    if wd_method in ("failed", "unavailable") or cl_method in ("failed", "unavailable"):
        confidence = "low"
    elif not watchdog_running and wd_lock.get("exists"):
        warnings.append("watchdog_lock_without_matching_process")
    elif not continuous_running and cl_lock.get("exists"):
        warnings.append("continuous_lock_without_matching_process")
    elif continuous_running and not cl_lock.get("exists"):
        warnings.append("continuous_process_without_lock")

    return {
        "watchdog_loop_running": watchdog_running,
        "continuous_loop_running": continuous_running,
        "detected_watchdog_process_ids": wd_pids,
        "detected_continuous_process_ids": cl_pids,
        "watchdog_detection_method": wd_method,
        "continuous_detection_method": cl_method,
        "process_detection_confidence": confidence,
        "process_detection_warnings": warnings,
        "watchdog_lock_exists": WATCHDOG_LOCK.is_file(),
        "continuous_lock_exists": CONTINUOUS_LOCK.is_file(),
        "watchdog_runtime_process_lock_state": classify_runtime_process_lock_state(
            process_running=watchdog_running,
            lock_exists=bool(wd_lock.get("exists")),
            lock_classification=wd_lock_class,
        ),
        "continuous_runtime_process_lock_state": classify_runtime_process_lock_state(
            process_running=continuous_running,
            lock_exists=bool(cl_lock.get("exists")),
            lock_classification=cl_lock_class,
        ),
        "watchdog_lock_classification": wd_lock_class,
        "continuous_lock_classification": cl_lock_class,
        "watchdog_lock_healthy": lock_audit.get("watchdog_lock_healthy"),
        "continuous_lock_healthy": lock_audit.get("continuous_lock_healthy"),
    }


def find_ps_loop_processes(script_fragment: str) -> Tuple[bool, List[int]]:
    """Backward-compatible helper used by phase_195h and dashboard."""
    script = script_fragment
    if script.endswith(".ps1"):
        pass
    elif "watchdog" in script:
        script = WATCHDOG_SCRIPT
    elif "continuous" in script:
        script = CONTINUOUS_SCRIPT
    pids, _, _ = _find_loop_processes(script)
    return len(pids) > 0, pids
