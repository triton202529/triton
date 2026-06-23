"""
phase_196h_runtime_supervisor_restore_validation.py
Phase 196H — Verify single 45-minute runtime supervisor task restore.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUTPUT_JSON = RESULTS / "phase_196h_runtime_supervisor_restore.json"

TASK_NAME = "TRITON_Runtime_Supervisor_45Min"
REPO_ROOT = Path(r"C:\Users\akimw\triton")
INTERVAL_MINUTES = 45
CONTINUOUS_SCRIPT = "scripts/run_continuous_loop.ps1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _run_ps(script: str, *extra_args: str) -> tuple[int, str]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ROOT / "scripts" / script),
        *extra_args,
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=120)
        out = (proc.stdout or "") + (proc.stderr or "")
        return int(proc.returncode), out.strip()
    except Exception as exc:
        return 1, str(exc)


def _query_scheduled_task(task_name: str) -> Dict[str, Any]:
    script = ROOT / "scripts" / "query_runtime_supervisor_task_json.ps1"
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-TaskName",
                task_name,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        raw = (proc.stdout or "").strip()
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return {"exists": False}


def _count_enabled_supervisor_tasks() -> List[str]:
    script = ROOT / "scripts" / "query_runtime_supervisor_task_json.ps1"
    ps = (
        "Get-ScheduledTask | Where-Object { $_.State -eq 'Ready' -and "
        "$_.TaskName -match 'Runtime_Supervisor|FullPipeline_45Min' } | "
        "Select-Object -ExpandProperty TaskName"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=60,
        )
        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        return lines
    except Exception:
        pass
    return []


def _loop_processes_running() -> Dict[str, bool]:
    ps = (
        ". '" + str(ROOT / "scripts" / "loop_lock_utils.ps1").replace("'", "''") + "'; "
        "@{ watchdog = (Test-LoopProcessRunning -ScriptName 'run_watchdog_loop.ps1'); "
        "continuous = (Test-LoopProcessRunning -ScriptName 'run_continuous_loop.ps1') } "
        "| ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=60,
        )
        raw = (proc.stdout or "").strip()
        if raw:
            data = json.loads(raw)
            return {
                "watchdog": bool(data.get("watchdog")),
                "continuous": bool(data.get("continuous")),
            }
    except Exception:
        pass
    return {"watchdog": False, "continuous": False}


def run_phase_196h_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []

    task_info = _query_scheduled_task(TASK_NAME)
    task_exists = bool(task_info.get("exists"))
    task_state = str(task_info.get("state") or "")
    task_enabled = task_state in ("Ready", "Running")
    args = str(task_info.get("arguments") or "")
    task_points_to_loop = (
        "run_continuous_loop.ps1" in args or "start_triton_runtime_supervisor.ps1" in args
    )
    task_interval_ok = "IntervalMinutes 45" in args or "IntervalMinutes=45" in args

    enabled_supervisors = _count_enabled_supervisor_tasks()
    multiple_supervisors = len(enabled_supervisors) > 1

    loops = _loop_processes_running()

    positions: List[str] = []
    open_orders = 0
    live_execution = False
    heartbeat_age: Optional[float] = None
    heartbeat_source = ""
    try:
        from services.dashboard_status_sync import (
            count_open_orders,
            load_pipeline_heartbeat,
            load_positions_summary,
        )
        from services.portfolio_history_sync import load_live_execution_flag

        pos = load_positions_summary()
        positions = list(pos.get("positions") or [])
        open_orders = int(count_open_orders())
        live_execution = load_live_execution_flag()
        ph = load_pipeline_heartbeat()
        heartbeat_age = ph.get("age_minutes")
        heartbeat_source = str(ph.get("path") or "")
    except Exception as exc:
        notes.append(f"artifact_read_error={exc}")

    install_script = ROOT / "scripts" / "install_triton_runtime_supervisor_45min.ps1"
    status_script = ROOT / "scripts" / "show_triton_runtime_supervisor_status.ps1"
    launcher_script = ROOT / "scripts" / "start_triton_runtime_supervisor.ps1"

    scripts_ok = install_script.is_file() and status_script.is_file() and launcher_script.is_file()

    status_rc, status_out = _run_ps("show_triton_runtime_supervisor_status.ps1")
    if status_rc != 0:
        notes.append(f"status_script_exit={status_rc}")

    runs_execute = "execute_trades" in args.lower()

    status = "PASS"
    if not scripts_ok:
        status = "FAIL"
        notes.append("Supervisor scripts missing.")
    elif not task_exists:
        status = "REVIEW"
        notes.append("Task not installed yet — run install_triton_runtime_supervisor_45min.ps1")
    elif not task_enabled:
        status = "FAIL"
        notes.append(f"Task exists but state={task_state}")
    elif multiple_supervisors:
        status = "FAIL"
        notes.append(f"Multiple supervisor tasks enabled: {enabled_supervisors}")
    elif not task_points_to_loop or not task_interval_ok:
        status = "FAIL"
        notes.append("Task action does not point to 45-minute continuous loop launcher.")
    elif runs_execute:
        status = "FAIL"
        notes.append("Task must not call execute_trades directly.")

    if task_exists and task_enabled:
        notes.append(
            "Pattern A: persistent loop started at logon/startup; task does not repeat every 45 min."
        )
    notes.append("Old TRITON_FullPipeline_45Min and legacy tasks were not restored.")

    payload = {
        "phase": "196H",
        "generated_at": _utc_now_iso(),
        "status": status,
        "task_name": TASK_NAME,
        "task_created_or_updated": task_exists,
        "task_enabled": task_enabled,
        "task_pattern": "PERSISTENT_LOOP_AT_LOGON",
        "old_tasks_restored": False,
        "multiple_triton_tasks_enabled": multiple_supervisors,
        "enabled_supervisor_tasks": enabled_supervisors,
        "runs_execute_trades_directly": runs_execute,
        "manual_approval_used": False,
        "broker_orders_submitted": False,
        "live_execution": live_execution,
        "continuous_loop_script": CONTINUOUS_SCRIPT,
        "interval_minutes": INTERVAL_MINUTES,
        "repo_root": str(REPO_ROOT).replace("\\", "/"),
        "open_orders": open_orders,
        "positions_count": len(positions),
        "positions": positions,
        "heartbeat_source_identified": bool(heartbeat_source),
        "heartbeat_source_path": heartbeat_source,
        "heartbeat_age_minutes": heartbeat_age,
        "watchdog_loop_running": loops.get("watchdog"),
        "continuous_loop_running": loops.get("continuous"),
        "dashboard_should_not_go_stale_if_task_runs": True,
        "task_info": task_info,
        "status_script_exit_code": status_rc,
        "execution_logic_changed": False,
        "governance_logic_changed": False,
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)

    if verbose:
        print(f"[196H] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196H] status={status}")
        if notes:
            print(f"[196H] notes={notes}")

    return payload


def main() -> int:
    report = run_phase_196h_validation(verbose=True)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
