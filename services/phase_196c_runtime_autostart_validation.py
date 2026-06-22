"""
phase_196c_runtime_autostart_validation.py
Phase 196C — Validate market-open TRITON runtime autostart scheduler wiring.
Read-only static checks plus optional scheduled-task presence on Windows.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "data" / "results"

START_SCRIPT = SCRIPTS / "start_triton_runtime_931.ps1"
REGISTER_SCRIPT = SCRIPTS / "register_triton_runtime_931_task.ps1"
AUDIT_JSON = RESULTS / "runtime_autostart_931.json"
OUTPUT_JSON = RESULTS / "phase_196c_runtime_autostart_validation.json"

TASK_NAME = "TRITON_Runtime_931AM"

LIVE_EXEC_PATTERNS = (
    r"TRITON_ENABLE_LIVE",
    r"TRITON_ENABLE_LIVE_EXECUTION",
    r"live_execution_enabled\s*=\s*\$true",
    r"LIVE_EXECUTION\s*=\s*['\"]1['\"]",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _run_powershell(script: str) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(ROOT),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except Exception as exc:
        return 1, str(exc)


def _check_live_execution_blocked() -> Tuple[bool, str]:
    gov = _load_json(RESULTS / "governance_authorization.json")
    readiness = _load_json(RESULTS / "execution_readiness.json")
    live = bool(
        gov.get("live_execution_permitted", readiness.get("live_execution_permitted", True))
    )
    if live:
        return False, "live_execution_permitted=true in governance artifacts"
    return True, "live_execution_permitted=false"


def _get_scheduled_task_info() -> Dict[str, Any]:
    rc, out = _run_powershell(
        f"$t = Get-ScheduledTask -TaskName '{TASK_NAME}' -ErrorAction SilentlyContinue; "
        f"if ($null -eq $t) {{ '{{\"exists\":false}}' }} else {{ "
        f"$i = Get-ScheduledTaskInfo -TaskName '{TASK_NAME}'; "
        f"@{{exists=$true; state=$t.State; last_result=$i.LastTaskResult; "
        f"next_run=$i.NextRunTime; last_run=$i.LastRunTime}} | ConvertTo-Json -Compress }}"
    )
    if rc != 0 or not out:
        return {"exists": False, "query_error": out or f"exit={rc}"}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"exists": False, "raw": out}


def run_phase_196c_validation(*, verbose: bool = True) -> Dict[str, Any]:
    issues: List[str] = []
    checks: List[Dict[str, Any]] = []

    start_text = _read_text(START_SCRIPT)
    register_text = _read_text(REGISTER_SCRIPT)

    def add_check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": passed, "detail": detail})
        if not passed:
            issues.append(f"{name}: {detail}" if detail else name)

    add_check("start_script_exists", START_SCRIPT.is_file(), str(START_SCRIPT))
    add_check(
        "register_script_exists",
        REGISTER_SCRIPT.is_file(),
        str(REGISTER_SCRIPT),
    )

    add_check(
        "startup_no_execute_trades",
        not re.search(
            r"execute_trades\.py|python\s+-m\s+services\.execute_trades", start_text, re.I
        ),
        "must not invoke execute_trades module",
    )

    live_hits = [p for p in LIVE_EXEC_PATTERNS if re.search(p, start_text, re.IGNORECASE)]
    add_check(
        "startup_no_live_execution_enablement",
        not live_hits,
        f"patterns={live_hits}" if live_hits else "no live enable env vars",
    )
    add_check(
        "startup_live_flag_false_in_audit",
        bool(re.search(r"live_execution_enabled\s*=\s*\$false", start_text, re.IGNORECASE)),
        "audit records live_execution_enabled=false",
    )

    add_check(
        "watchdog_interval_1",
        '"-IntervalMinutes", "1"' in start_text or "-IntervalMinutes', '1'" in start_text,
        "IntervalMinutes 1 for watchdog",
    )
    add_check(
        "continuous_interval_45",
        '"-IntervalMinutes", "45"' in start_text or "-IntervalMinutes', '45'" in start_text,
        "IntervalMinutes 45 for continuous",
    )
    add_check(
        "process_check_before_start",
        "Test-LoopProcessRunning" in start_text and "run_watchdog_loop.ps1" in start_text,
        "uses loop_lock_utils process detection",
    )
    add_check(
        "writes_runtime_autostart_audit",
        "runtime_autostart_931.json" in start_text,
        str(AUDIT_JSON.name),
    )
    add_check(
        "no_lock_deletion_in_startup",
        "Remove-Item" not in start_text and "Clear-OrphanLoopLock" not in start_text,
        "must not delete lock files",
    )

    add_check(
        "register_task_name",
        TASK_NAME in register_text,
        TASK_NAME,
    )
    add_check(
        "register_daily_0931",
        bool(re.search(r"09:31|TimeOfDay\s*=\s*['\"]09:31['\"]", register_text)),
        "daily 09:31 schedule",
    )
    add_check(
        "register_no_bulk_enable",
        "Enable-ScheduledTask" not in register_text,
        "must not enable old disabled tasks",
    )
    add_check(
        "register_only_target_task_unregister",
        register_text.count("Unregister-ScheduledTask") <= 1,
        "only replaces TRITON_Runtime_931AM",
    )

    live_ok, live_detail = _check_live_execution_blocked()
    add_check("live_execution_remains_blocked", live_ok, live_detail)

    task_info = _get_scheduled_task_info()
    add_check(
        "scheduled_task_registered",
        bool(task_info.get("exists")),
        json.dumps(task_info, default=str),
    )

    audit_exists = AUDIT_JSON.is_file()
    add_check(
        "audit_file_path_defined",
        "runtime_autostart_931.json" in start_text,
        "written on startup run",
    )

    phase_pass = not issues

    payload: Dict[str, Any] = {
        "phase": "196C",
        "timestamp": _utc_now_iso(),
        "phase_pass": phase_pass,
        "issues": issues,
        "checks": checks,
        "scheduled_task": task_info,
        "live_execution_status": live_detail,
        "audit_file_exists": audit_exists,
        "artifacts": {
            "start_script": str(START_SCRIPT),
            "register_script": str(REGISTER_SCRIPT),
            "audit_json": str(AUDIT_JSON),
        },
    }

    _atomic_write_json(OUTPUT_JSON, payload)

    if verbose:
        print(
            json.dumps(
                {"phase_pass": phase_pass, "issues": issues, "scheduled_task": task_info}, indent=2
            )
        )

    return payload


def main() -> int:
    result = run_phase_196c_validation(verbose=True)
    return 0 if result.get("phase_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
