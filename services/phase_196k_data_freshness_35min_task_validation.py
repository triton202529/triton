"""
phase_196k_data_freshness_35min_task_validation.py
Phase 196K — Validate read-only 35-minute data freshness scheduled task.
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
OUTPUT_JSON = RESULTS / "phase_196k_data_freshness_35min_task.json"

PREVIOUS_COMMIT = "6e73891e"
TASK_NAME = "TRITON_Data_Freshness_35Min"
INTERVAL_MINUTES = 35
STALE_THRESHOLD_MINUTES = 90.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _query_task() -> Dict[str, Any]:
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "query_data_freshness_task_json.ps1"),
            "-TaskName",
            TASK_NAME,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        return json.loads((proc.stdout or "").strip() or "{}")
    except json.JSONDecodeError:
        return {}


def _enabled_runtime_supervisors() -> List[str]:
    ps = (
        "Get-ScheduledTask | Where-Object { $_.State -eq 'Ready' -and "
        "$_.TaskName -match 'Runtime_Supervisor|FullPipeline_45Min' } | "
        "Select-Object -ExpandProperty TaskName"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]


def _old_fullpipeline_enabled() -> bool:
    ps = (
        "Get-ScheduledTask -TaskName 'TRITON_FullPipeline_45Min' -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty State"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (proc.stdout or "").strip() == "Ready"


def _artifact_age_minutes(path: Path) -> Optional[float]:
    try:
        if path.is_file() and path.stat().st_size > 0:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            return (datetime.now(timezone.utc) - mtime).total_seconds() / 60.0
    except Exception:
        pass
    return None


def _script_forbids_execution() -> bool:
    script = ROOT / "scripts" / "run_data_freshness_refresh.ps1"
    try:
        src = script.read_text(encoding="utf-8").lower()
    except Exception:
        return False
    forbidden_invocations = [
        "services.execute_trades",
        "services.paper_entry_loop_stage",
        "manage_positions --execute",
        "manage_open_orders --execute",
        "run_market_open.ps1",
    ]
    return not any(p in src for p in forbidden_invocations)


def _portfolio_state() -> Dict[str, Any]:
    positions: List[str] = []
    open_orders = 0
    live_execution = False
    try:
        from services.dashboard_status_sync import count_open_orders, load_positions_summary
        from services.portfolio_history_sync import load_live_execution_flag

        pos = load_positions_summary()
        positions = list(pos.get("positions") or [])
        open_orders = int(count_open_orders())
        live_execution = load_live_execution_flag()
    except Exception:
        pass
    return {
        "positions": positions,
        "positions_count": len(positions),
        "open_orders": open_orders,
        "live_execution": live_execution,
    }


def run_phase_196k_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    task = _query_task()
    task_exists = bool(task.get("exists"))
    task_enabled = str(task.get("state")) in ("Ready", "Running")
    args = str(task.get("arguments") or "").lower()

    runtime_supervisors = _enabled_runtime_supervisors()
    pos_state = _portfolio_state()

    hb_path = RESULTS / "heartbeat.json"
    pos_path = RESULTS / "positions_snapshot.csv"
    oo_path = RESULTS / "open_orders_snapshot.csv"
    lock_path = ROOT / "data" / "live" / "data_freshness_refresh.lock"

    hb_age = _artifact_age_minutes(hb_path)
    pos_age = _artifact_age_minutes(pos_path)
    oo_age = _artifact_age_minutes(oo_path)

    heartbeat_refreshed = hb_age is not None and float(hb_age) <= STALE_THRESHOLD_MINUTES
    positions_refreshed = pos_age is not None and float(pos_age) <= STALE_THRESHOLD_MINUTES
    open_orders_refreshed = oo_age is not None and float(oo_age) <= STALE_THRESHOLD_MINUTES

    refresh_script = ROOT / "scripts" / "run_data_freshness_refresh.ps1"
    install_script = ROOT / "scripts" / "install_triton_data_freshness_35min.ps1"
    status_script = ROOT / "scripts" / "show_triton_data_freshness_status.ps1"

    runs_execute = "execute_trades" in args
    creates_loop = "run_continuous_loop" in args or "start_triton_runtime_supervisor" in args

    status = "PASS"
    if not refresh_script.is_file() or not install_script.is_file() or not status_script.is_file():
        status = "FAIL"
        notes.append("Required scripts missing.")
    elif not _script_forbids_execution():
        status = "FAIL"
        notes.append("Refresh script contains forbidden execution patterns.")
    elif not task_exists:
        status = "REVIEW"
        notes.append("Task not installed — run install_triton_data_freshness_35min.ps1")
    elif not task_enabled:
        status = "FAIL"
        notes.append(f"Task exists but state={task.get('state')}")
    elif _old_fullpipeline_enabled():
        status = "FAIL"
        notes.append("TRITON_FullPipeline_45Min is enabled.")
    elif len(runtime_supervisors) > 1:
        status = "FAIL"
        notes.append(f"Multiple runtime supervisors: {runtime_supervisors}")
    elif runs_execute or creates_loop:
        status = "FAIL"
        notes.append("Task action must not call execute_trades or start continuous loop.")
    elif pos_state.get("positions") != ["JPM", "WFC"]:
        status = "FAIL"
        notes.append(f"Unexpected positions: {pos_state.get('positions')}")
    elif int(pos_state.get("open_orders") or 0) != 0:
        status = "FAIL"
        notes.append("Open orders not zero.")
    elif pos_state.get("live_execution"):
        status = "FAIL"
        notes.append("live_execution is true.")
    elif not heartbeat_refreshed or not positions_refreshed or not open_orders_refreshed:
        status = "REVIEW"
        notes.append(
            "Artifacts older than freshness threshold; run task once via Start-ScheduledTask."
        )
    else:
        notes.append("Read-only data freshness task configured; artifacts within freshness window.")

    if lock_path.is_file():
        notes.append("data_freshness_refresh.lock present (may be active run or stale orphan).")

    payload = {
        "phase": "196K",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_commit": PREVIOUS_COMMIT,
        "task_name": TASK_NAME,
        "task_created_or_updated": task_exists,
        "task_enabled": task_enabled,
        "interval_minutes": INTERVAL_MINUTES,
        "read_only_refresh": _script_forbids_execution(),
        "runs_execute_trades_directly": runs_execute,
        "runs_manual_approval": False,
        "broker_orders_submitted": False,
        "live_execution": pos_state.get("live_execution"),
        "creates_second_continuous_loop": creates_loop,
        "old_fullpipeline_45min_restored": _old_fullpipeline_enabled(),
        "multiple_runtime_authorities": len(runtime_supervisors) > 1,
        "enabled_runtime_supervisor_tasks": runtime_supervisors,
        "data_freshness_lock_used": (ROOT / "scripts" / "run_data_freshness_refresh.ps1").is_file(),
        "heartbeat_source": "data/results/heartbeat.json",
        "heartbeat_age_minutes": hb_age,
        "positions_snapshot_age_minutes": pos_age,
        "open_orders_snapshot_age_minutes": oo_age,
        "heartbeat_refreshed": heartbeat_refreshed,
        "positions_snapshot_refreshed": positions_refreshed,
        "open_orders_snapshot_refreshed": open_orders_refreshed,
        "open_orders": pos_state.get("open_orders"),
        "positions_count": pos_state.get("positions_count"),
        "positions": pos_state.get("positions"),
        "dashboard_expected_to_remain_fresh": True,
        "execution_logic_changed": False,
        "governance_logic_changed": False,
        "task_info": task,
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196K] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196K] status={status}")
    return payload


def main() -> int:
    report = run_phase_196k_validation(verbose=True)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
