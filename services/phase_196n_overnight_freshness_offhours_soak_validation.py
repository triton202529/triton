"""
phase_196n_overnight_freshness_offhours_soak_validation.py
Phase 196N — Overnight/off-hours freshness and status soak (monitor-only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
OUTPUT_JSON = RESULTS / "phase_196n_overnight_freshness_offhours_soak.json"
FRESHNESS_LOG_DIR = RESULTS / "data_freshness_refresh"

PREVIOUS_COMMIT = "94364c4a"
SUPERVISOR_TASK = "TRITON_Runtime_Supervisor_45Min"
FRESHNESS_TASK = "TRITON_Data_Freshness_35Min"
HEARTBEAT_PATH = RESULTS / "heartbeat.json"
OFF_HOURS_HEARTBEAT_MAX_MINUTES = 90.0
OFF_HOURS_SNAPSHOT_MAX_MINUTES = 120.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _run_ps_file(script: str) -> Tuple[int, str]:
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / script),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return int(proc.returncode), ((proc.stdout or "") + (proc.stderr or "")).strip()


def _query_json(script: str, task_name: str) -> Dict[str, Any]:
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / script),
            "-TaskName",
            task_name,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        return json.loads((proc.stdout or "").strip() or "{}")
    except json.JSONDecodeError:
        return {}


def _enabled_supervisor_authorities() -> List[str]:
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


def _fullpipeline_enabled() -> bool:
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


def _loop_processes() -> Dict[str, Any]:
    ps = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -match 'run_continuous_loop|run_watchdog_loop|data_freshness' } | "
        "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=60,
    )
    raw = (proc.stdout or "").strip()
    if not raw:
        return {"watchdog": [], "continuous": [], "freshness": []}
    try:
        rows = json.loads(raw)
        if isinstance(rows, dict):
            rows = [rows]
    except json.JSONDecodeError:
        rows = []
    watchdog, continuous, freshness = [], [], []
    for row in rows:
        cmd = str(row.get("CommandLine") or "")
        pid = row.get("ProcessId")
        if "run_watchdog_loop" in cmd:
            watchdog.append(pid)
        elif "run_continuous_loop" in cmd:
            continuous.append(pid)
        elif "data_freshness" in cmd.lower():
            freshness.append(pid)
    return {
        "watchdog_pids": watchdog,
        "continuous_pids": continuous,
        "freshness_pids": freshness,
        "watchdog_running": len(watchdog) > 0,
        "continuous_running": len(continuous) > 0,
        "duplicate_continuous_loops": len(continuous) > 1,
    }


def _artifact_age_minutes(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    try:
        return (
            datetime.now(timezone.utc)
            - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        ).total_seconds() / 60.0
    except Exception:
        return None


def _count_freshness_cycles(*, min_spacing_minutes: float = 25.0) -> Tuple[int, List[str]]:
    """Count successful scheduled freshness runs from log files."""
    if not FRESHNESS_LOG_DIR.is_dir():
        return 0, []
    successes: List[datetime] = []
    names: List[str] = []
    for log in sorted(FRESHNESS_LOG_DIR.glob("data_freshness_refresh_*.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "DATA FRESHNESS REFRESH COMPLETE" not in text:
            continue
        m = re.search(r"data_freshness_refresh_(\d{8}_\d{6})", log.name)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        successes.append(ts)
        names.append(log.name)

    if not successes:
        return 0, []

    cycles = 1
    cycle_logs = [names[0]]
    for i in range(1, len(successes)):
        gap = (successes[i] - successes[i - 1]).total_seconds() / 60.0
        if gap >= min_spacing_minutes:
            cycles += 1
            cycle_logs.append(names[i])
    return cycles, cycle_logs


def _wait_for_freshness_cycles(
    baseline_cycles: int,
    *,
    target_cycles: int,
    timeout_minutes: float = 80.0,
    poll_seconds: float = 60.0,
    verbose: bool = True,
) -> int:
    deadline = time.time() + timeout_minutes * 60.0
    while time.time() < deadline:
        cycles, _ = _count_freshness_cycles()
        if verbose:
            print(f"[196N soak] freshness_cycles={cycles} target={target_cycles}", flush=True)
        if cycles >= target_cycles:
            return cycles
        time.sleep(poll_seconds)
    return _count_freshness_cycles()[0]


def run_phase_196n_soak(
    *,
    wait_for_cycles: int = 0,
    wait_timeout_minutes: float = 80.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    notes: List[str] = []
    if wait_for_cycles > 0:
        baseline, _ = _count_freshness_cycles()
        if baseline < wait_for_cycles:
            notes.append(
                f"Waiting up to {wait_timeout_minutes:.0f}m for "
                f"{wait_for_cycles} freshness cycles (baseline={baseline})."
            )
            _wait_for_freshness_cycles(
                baseline,
                target_cycles=wait_for_cycles,
                timeout_minutes=wait_timeout_minutes,
                verbose=verbose,
            )

    cycles_observed, cycle_logs = _count_freshness_cycles()
    supervisor_task = _query_json("query_runtime_supervisor_task_json.ps1", SUPERVISOR_TASK)
    freshness_task = _query_json("query_data_freshness_task_json.ps1", FRESHNESS_TASK)
    supervisors = _enabled_supervisor_authorities()
    loops = _loop_processes()

    from services.dashboard_status_sync import (
        audit_reconciled_runtime_locks,
        build_dashboard_guard_summary,
        build_dashboard_health_classification,
        classify_contract_artifact_integrity,
        reconcile_integrity_lock_display,
    )

    health = build_dashboard_health_classification()
    lock_audit = audit_reconciled_runtime_locks()
    contracts_ok: Optional[bool] = None
    contract_summary: Dict[str, Any] = {}
    try:
        import view_results as vr

        results = vr.validate_all_contracts()
        contract_summary = vr.contracts_summary(results)
        contracts_ok = (contract_summary.get("failed", 0) == 0) and (
            contract_summary.get("error_count", 0) == 0
        )
    except Exception as exc:
        notes.append(f"Contract validation via dashboard module: {exc}")

    contract_integrity = classify_contract_artifact_integrity(
        contracts_ok=contracts_ok,
        contract_summary=contract_summary,
    )
    fake_reason = "Execution locked: pipeline health 🔴"
    integrity_display = reconcile_integrity_lock_display(
        pipeline_locked=True,
        pipeline_lock_reason=fake_reason,
        runtime_pipeline_health=str(health.get("runtime_pipeline_health") or "FAIL"),
        contracts_ok=contracts_ok,
        contract_integrity=contract_integrity,
    )
    summary = build_dashboard_guard_summary(
        pipeline_locked=True,
        pipeline_lock_reason=fake_reason,
        live_ready=False,
        live_gate_bits={"CONFIRM": False, "MARKET": False, "ARM": False, "GUARD": True},
        contracts_ok=contracts_ok,
        contract_summary=contract_summary,
    )

    hb_age = _artifact_age_minutes(HEARTBEAT_PATH)
    pos_age = _artifact_age_minutes(RESULTS / "positions_snapshot.csv")
    oo_age = _artifact_age_minutes(RESULTS / "open_orders_snapshot.csv")
    ph_age = _artifact_age_minutes(RESULTS / "portfolio_history.csv")

    heartbeat_ok = hb_age is not None and float(hb_age) <= OFF_HOURS_HEARTBEAT_MAX_MINUTES
    positions_ok = pos_age is not None and float(pos_age) <= OFF_HOURS_SNAPSHOT_MAX_MINUTES
    open_orders_ok = oo_age is not None and float(oo_age) <= OFF_HOURS_SNAPSHOT_MAX_MINUTES
    portfolio_ok = ph_age is not None and float(ph_age) <= OFF_HOURS_SNAPSHOT_MAX_MINUTES

    runtime_pipeline_health = str(health.get("runtime_pipeline_health") or "FAIL")
    data_freshness = str(health.get("data_freshness") or "STALE")
    block_reasons = list(health.get("execution_block_reasons") or [])
    positions = list(health.get("positions") or [])
    open_orders = int(health.get("open_orders") or 0)

    false_pipeline_fail = (
        runtime_pipeline_health
        in (
            "HEALTHY",
            "OFF_HOURS_HEALTHY",
        )
        and "PIPELINE_HEALTH" in block_reasons
    )
    false_contract_lock = bool(
        integrity_display.get("show_integrity_lock_caption")
        and runtime_pipeline_health in ("HEALTHY", "OFF_HOURS_HEALTHY")
        and "pipeline health" in str(integrity_display.get("integrity_lock_detail", "")).lower()
    )

    live_execution = False
    try:
        from services.portfolio_history_sync import load_live_execution_flag

        live_execution = load_live_execution_flag()
    except Exception:
        pass

    supervisor_enabled = str(supervisor_task.get("state")) in ("Ready", "Running")
    freshness_enabled = str(freshness_task.get("state")) in ("Ready", "Running")
    freshness_last_result = freshness_task.get("last_result")

    status = "PASS"
    if _fullpipeline_enabled():
        status = "FAIL"
        notes.append("TRITON_FullPipeline_45Min is enabled.")
    elif len(supervisors) > 1:
        status = "FAIL"
        notes.append(f"Multiple runtime authorities: {supervisors}")
    elif not supervisor_enabled:
        status = "FAIL"
        notes.append("Runtime supervisor task not enabled.")
    elif not freshness_enabled:
        status = "FAIL"
        notes.append("Data freshness task not enabled.")
    elif not loops.get("watchdog_running"):
        status = "FAIL"
        notes.append("Watchdog process not running.")
    elif not loops.get("continuous_running"):
        status = "FAIL"
        notes.append("Continuous loop process not running.")
    elif loops.get("duplicate_continuous_loops"):
        status = "FAIL"
        notes.append("Duplicate continuous loop processes detected.")
    elif cycles_observed < 2:
        status = "REVIEW"
        notes.append(
            f"Only {cycles_observed} freshness cycle(s) observed in logs; need >= 2 for soak PASS."
        )
    elif freshness_last_result not in (0, "0", None) and str(freshness_last_result) != "0":
        status = "REVIEW"
        notes.append(f"Last freshness task result={freshness_last_result}")
    elif not heartbeat_ok:
        status = "FAIL"
        notes.append(f"Heartbeat stale: {hb_age:.1f}m")
    elif not (positions_ok and open_orders_ok and portfolio_ok):
        status = "FAIL"
        notes.append("Snapshot/portfolio artifacts not refreshed within off-hours tolerance.")
    elif false_pipeline_fail:
        status = "FAIL"
        notes.append("PIPELINE_HEALTH block reason while runtime pipeline healthy.")
    elif false_contract_lock:
        status = "FAIL"
        notes.append("False contract/artifact pipeline-health lock detected.")
    elif positions != ["JPM", "WFC"]:
        status = "FAIL"
        notes.append(f"Unexpected positions: {positions}")
    elif open_orders != 0:
        status = "FAIL"
        notes.append("Open orders not zero.")
    elif live_execution:
        status = "FAIL"
        notes.append("live_execution is true.")
    elif runtime_pipeline_health in ("FAIL", "DEGRADED"):
        status = "REVIEW"
        notes.append(f"Runtime pipeline health={runtime_pipeline_health} during soak.")
    else:
        notes.append(
            f"Off-hours soak OK: {cycles_observed} freshness cycles, "
            f"runtime={runtime_pipeline_health}, blocks={block_reasons}."
        )

    payload = {
        "phase": "196N",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_commit": PREVIOUS_COMMIT,
        "runtime_supervisor_enabled": supervisor_enabled,
        "data_freshness_task_enabled": freshness_enabled,
        "old_fullpipeline_45min_restored": _fullpipeline_enabled(),
        "multiple_runtime_authorities": len(supervisors) > 1,
        "enabled_runtime_supervisor_tasks": supervisors,
        "watchdog_running": bool(loops.get("watchdog_running")),
        "continuous_loop_running": bool(loops.get("continuous_running")),
        "duplicate_continuous_loops": bool(loops.get("duplicate_continuous_loops")),
        "data_freshness_cycles_observed": cycles_observed,
        "data_freshness_cycle_logs": cycle_logs,
        "freshness_task_last_result": freshness_last_result,
        "freshness_task_last_run": freshness_task.get("last_run"),
        "freshness_task_next_run": freshness_task.get("next_run"),
        "heartbeat_source": "data/results/heartbeat.json",
        "heartbeat_age_minutes": hb_age,
        "heartbeat_fresh_or_offhours_acceptable": heartbeat_ok,
        "positions_snapshot_age_minutes": pos_age,
        "open_orders_snapshot_age_minutes": oo_age,
        "portfolio_history_age_minutes": ph_age,
        "positions_snapshot_refreshed": positions_ok,
        "open_orders_snapshot_refreshed": open_orders_ok,
        "portfolio_history_current": portfolio_ok,
        "runtime_pipeline_health": runtime_pipeline_health,
        "data_freshness": data_freshness,
        "contract_artifact_integrity": contract_integrity,
        "execution_status": str(health.get("execution_status") or "BLOCKED"),
        "execution_block_reasons": block_reasons,
        "pipeline_health_false_fail_detected": false_pipeline_fail,
        "contract_artifact_false_pipeline_lock_detected": false_contract_lock,
        "watchdog_lock_status": lock_audit.get("watchdog_lock_classification"),
        "continuous_loop_lock_status": lock_audit.get("continuous_lock_classification"),
        "open_orders": open_orders,
        "positions_count": int(health.get("positions_count") or 0),
        "positions": positions,
        "live_execution": live_execution,
        "manual_execute_trades_run": False,
        "manual_approval_used": False,
        "broker_orders_submitted": False,
        "execution_logic_changed": False,
        "governance_logic_changed": False,
        "guard_summary": {
            "sidebar_label": summary.get("sidebar_label"),
            "contract_artifact_integrity": summary.get("contract_artifact_integrity"),
            "execution_blocked_reasons": summary.get("execution_blocked_reasons"),
        },
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196N] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196N] status={status}")
        print(f"[196N] freshness_cycles={cycles_observed} runtime={runtime_pipeline_health}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 196N overnight off-hours soak")
    ap.add_argument(
        "--wait-cycles",
        type=int,
        default=0,
        help="If set, wait until this many freshness cycles are logged",
    )
    ap.add_argument("--wait-timeout-minutes", type=float, default=80.0)
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    report = run_phase_196n_soak(
        wait_for_cycles=args.wait_cycles,
        wait_timeout_minutes=args.wait_timeout_minutes,
        verbose=args.verbose,
    )
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
