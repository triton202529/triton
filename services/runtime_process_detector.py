"""
runtime_process_detector.py
Phase 197Q-FIX — Strict runtime loop process detection (watchdog vs continuous).
"""

from __future__ import annotations

import contextvars
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"

WATCHDOG_LOCK = LIVE / "watchdog_loop.lock"
CONTINUOUS_LOCK = LIVE / "continuous_loop.lock"

WATCHDOG_SCRIPT = "run_watchdog_loop.ps1"
CONTINUOUS_SCRIPT = "run_continuous_loop.ps1"
_KNOWN_LOOP_SCRIPTS = (WATCHDOG_SCRIPT, CONTINUOUS_SCRIPT)
_SCAN_SCOPE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "triton_process_scan_scope",
    default=None,
)
_SCAN_MEMO: Dict[Tuple[str, str], Any] = {}

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


def _scan_loop_processes(script_name: str) -> Tuple[List[int], str, List[str]]:
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


class _ProcessScanScope:
    """One dashboard rerun. Runtime callers never open this."""

    def __init__(self, token: str) -> None:
        self.token = token
        self._reset: Optional[contextvars.Token[Optional[str]]] = None

    def close(self) -> None:
        if self._reset is not None:
            _SCAN_SCOPE.reset(self._reset)
            self._reset = None
        _SCAN_MEMO.clear()


def open_process_scan_scope(token: Optional[str] = None) -> _ProcessScanScope:
    """Share loop-process scans for one dashboard render. The next render scans again."""
    _SCAN_MEMO.clear()
    scope = _ProcessScanScope(token or str(time.time_ns()))
    scope._reset = _SCAN_SCOPE.set(scope.token)
    return scope


def _scan_known_loop_processes() -> Dict[str, Tuple[List[int], str, List[str]]]:
    """One command-line query for both loop scripts. Same match rule as a single scan."""
    method = "win32_process_commandline_file_invocation"
    found: Dict[str, List[int]] = {name: [] for name in _KNOWN_LOOP_SCRIPTS}
    warnings: List[str] = []
    if sys.platform != "win32":
        warnings.append("non_windows_platform_process_detection_unavailable")
        return {name: ([], "unavailable", list(warnings)) for name in _KNOWN_LOOP_SCRIPTS}
    clauses = []
    for script_name in _KNOWN_LOOP_SCRIPTS:
        escaped = script_name.replace("'", "''")
        clauses.append(
            f"$pat = [regex]::Escape('{escaped}'); "
            "$rx = '(?:-File\\s+[\"'']?[^\\s\"'']*[/\\\\])?' + $pat + '(?:\\s|$|\"|'')'; "
            f"if ($_.CommandLine -match $rx) {{ Write-Output ('{escaped}' + \"`t\" + $_.ProcessId) }}"
        )
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name = 'powershell.exe' OR Name = 'pwsh.exe'\" "
        "| Where-Object { "
        "$_.ProcessId -ne $PID "
        "-and $_.CommandLine "
        "-and ($_.CommandLine -notmatch 'Get-CimInstance Win32_Process') "
        "-and ($_.CommandLine -notmatch 'Where-Object') "
        "} "
        "| ForEach-Object { " + " ".join(clauses) + " }"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        warnings.append(f"process_scan_failed={exc}")
        return {name: ([], "failed", list(warnings)) for name in _KNOWN_LOOP_SCRIPTS}
    if result.stderr and result.stderr.strip():
        warnings.append(f"powershell_stderr={result.stderr.strip()[:200]}")
    for line in (result.stdout or "").splitlines():
        script_name, separator, pid_text = line.partition("\t")
        if separator != "\t" or script_name not in found or not pid_text.strip().isdigit():
            continue
        found[script_name].append(int(pid_text.strip()))
    return {name: (pids, method, list(warnings)) for name, pids in found.items()}


def scan_loop_processes_for_control() -> Dict[str, Tuple[List[int], str, List[str]]]:
    """Uncached scan for decisions that launch, confirm, or stop a loop.

    Dashboard display lookups keep the per-render memo. This scan does not read
    or update that memo, so a process created after the display scan is visible
    to the next controller poll in the same render.
    """
    found = _scan_known_loop_processes()
    if any(item[1] == "failed" for item in found.values()):
        return {name: _scan_loop_processes(name) for name in _KNOWN_LOOP_SCRIPTS}
    return found


def _find_loop_processes(script_name: str) -> Tuple[List[int], str, List[str]]:
    """Scan once per dashboard render. Outside that render, every call scans again."""
    token = _SCAN_SCOPE.get()
    if token is None:
        return _scan_loop_processes(script_name)
    key = (token, script_name)
    cached = _SCAN_MEMO.get(key)
    if cached is not None:
        return cached
    filled_key = (token, "__filled__")
    if script_name in _KNOWN_LOOP_SCRIPTS and filled_key not in _SCAN_MEMO:
        try:
            found = _scan_known_loop_processes()
        except (subprocess.SubprocessError, OSError):
            found = None
        if found is not None and all(item[1] != "failed" for item in found.values()):
            for name, value in found.items():
                _SCAN_MEMO[(token, name)] = value
            _SCAN_MEMO[filled_key] = True
        else:
            _SCAN_MEMO[filled_key] = False
    cached = _SCAN_MEMO.get(key)
    if cached is not None:
        return cached
    scanned = _scan_loop_processes(script_name)
    _SCAN_MEMO[key] = scanned
    return scanned


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


def detect_runtime_loop_processes(*, fresh: bool = False) -> Dict[str, Any]:
    """Detect watchdog and continuous loop processes with diagnostics.

    fresh=False keeps the dashboard render memo. fresh=True is the controller
    boundary: Start, Stop, and Restart polls must not inherit that memo.
    """
    from services.phase_195h_operational_readiness_freeze import _parse_lock
    from services.runtime_lock_classifier import audit_runtime_locks

    if fresh:
        found = scan_loop_processes_for_control()
        wd_pids, wd_method, wd_warnings = found[WATCHDOG_SCRIPT]
        cl_pids, cl_method, cl_warnings = found[CONTINUOUS_SCRIPT]
    else:
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
