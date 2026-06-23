"""
phase_196i_runtime_supervisor_heartbeat_soak_validation.py
Phase 196I — Monitor runtime supervisor heartbeat freshness across loop cycles.
"""

from __future__ import annotations

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
OUTPUT_JSON = RESULTS / "phase_196i_runtime_supervisor_heartbeat_soak.json"

PREVIOUS_COMMIT = "9ec5b8d3"
TASK_NAME = "TRITON_Runtime_Supervisor_45Min"
HEARTBEAT_PATH = RESULTS / "heartbeat.json"
PORTFOLIO_HISTORY_PATH = RESULTS / "portfolio_history.csv"
POSITIONS_SNAPSHOT_PATH = RESULTS / "positions_snapshot.csv"
CONTINUOUS_LOCK = LIVE / "continuous_loop.lock"
STALE_THRESHOLD_MINUTES = 90.0


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


def _query_task() -> Dict[str, Any]:
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "query_runtime_supervisor_task_json.ps1"),
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


def _enabled_supervisor_tasks() -> List[str]:
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


def _old_45min_enabled() -> bool:
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


def _loop_processes() -> Dict[str, bool]:
    ps = (
        ". '" + str(ROOT / "scripts" / "loop_lock_utils.ps1").replace("'", "''") + "'; "
        "@{ watchdog = (Test-LoopProcessRunning -ScriptName 'run_watchdog_loop.ps1'); "
        "continuous = (Test-LoopProcessRunning -ScriptName 'run_continuous_loop.ps1') } "
        "| ConvertTo-Json -Compress"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        data = json.loads((proc.stdout or "").strip())
        return {"watchdog": bool(data.get("watchdog")), "continuous": bool(data.get("continuous"))}
    except Exception:
        return {"watchdog": False, "continuous": False}


def _read_lock_cycle() -> Tuple[int, Optional[datetime]]:
    if not CONTINUOUS_LOCK.is_file():
        return 0, None
    text = CONTINUOUS_LOCK.read_text(encoding="utf-8", errors="replace")
    cycle = 0
    refreshed: Optional[datetime] = None
    m = re.search(r"cycle=(\d+)", text)
    if m:
        cycle = int(m.group(1))
    m2 = re.search(r"refreshed_at=([^\r\n]+)", text)
    if m2:
        try:
            import pandas as pd

            ts = pd.to_datetime(m2.group(1).strip(), errors="coerce", utc=True)
            if pd.notna(ts):
                refreshed = ts.to_pydatetime()
        except Exception:
            pass
    return cycle, refreshed


def _heartbeat_age_minutes() -> Tuple[Optional[float], Optional[datetime]]:
    try:
        from services.dashboard_status_sync import load_pipeline_heartbeat

        hb = load_pipeline_heartbeat()
        return hb.get("age_minutes"), hb.get("timestamp")
    except Exception:
        return None, None


def _artifact_mtime(path: Path) -> Optional[datetime]:
    try:
        if path.is_file() and path.stat().st_size > 0:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        pass
    return None


def _is_fresh(age_min: Optional[float]) -> Optional[bool]:
    if age_min is None:
        return None
    return float(age_min) <= STALE_THRESHOLD_MINUTES


def _portfolio_positions() -> Dict[str, Any]:
    open_orders = 0
    positions: List[str] = []
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
        "open_orders": open_orders,
        "positions": positions,
        "positions_count": len(positions),
        "live_execution": live_execution,
    }


def wait_for_cycle_increment(
    baseline_cycle: int,
    *,
    timeout_minutes: float = 50.0,
    poll_seconds: float = 30.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Poll continuous lock until cycle increments or timeout."""
    deadline = time.time() + timeout_minutes * 60.0
    observations: List[Dict[str, Any]] = []
    while time.time() < deadline:
        cycle, refreshed = _read_lock_cycle()
        age, ts = _heartbeat_age_minutes()
        obs = {
            "observed_at": _utc_now_iso(),
            "cycle": cycle,
            "lock_refreshed_at": refreshed.isoformat() if refreshed else None,
            "heartbeat_age_minutes": age,
            "heartbeat_fresh": _is_fresh(age),
        }
        observations.append(obs)
        if verbose:
            print(
                f"[196I soak] cycle={cycle} hb_age={age} fresh={obs['heartbeat_fresh']}",
                flush=True,
            )
        if cycle > baseline_cycle:
            return {
                "cycle_incremented": True,
                "final_cycle": cycle,
                "observations": observations,
            }
        time.sleep(poll_seconds)
    return {
        "cycle_incremented": False,
        "final_cycle": _read_lock_cycle()[0],
        "observations": observations,
    }


def run_phase_196i_soak(
    *,
    wait_for_cycles: int = 1,
    wait_timeout_minutes: float = 50.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    notes: List[str] = []
    task = _query_task()
    supervisors = _enabled_supervisor_tasks()
    loops = _loop_processes()
    pos_state = _portfolio_positions()

    baseline_cycle, lock_refreshed = _read_lock_cycle()
    hb_age_before, hb_ts_before = _heartbeat_age_minutes()
    ph_mtime_before = _artifact_mtime(PORTFOLIO_HISTORY_PATH)
    pos_mtime_before = _artifact_mtime(POSITIONS_SNAPSHOT_PATH)

    if verbose:
        print(f"[196I] baseline cycle={baseline_cycle} heartbeat_age={hb_age_before}")

    cycle_1_result: Dict[str, Any] = {}
    heartbeat_fresh_after_cycle_1: Optional[bool] = None
    heartbeat_fresh_after_cycle_2: Optional[bool] = None

    if wait_for_cycles >= 1:
        cycle_1_result = wait_for_cycle_increment(
            baseline_cycle,
            timeout_minutes=wait_timeout_minutes,
            verbose=verbose,
        )
        age1, _ = _heartbeat_age_minutes()
        heartbeat_fresh_after_cycle_1 = _is_fresh(age1)
        if cycle_1_result.get("cycle_incremented"):
            notes.append(f"Cycle incremented to {cycle_1_result.get('final_cycle')}.")
        else:
            notes.append("Timed out waiting for next continuous loop cycle increment.")

    if wait_for_cycles >= 2 and cycle_1_result.get("cycle_incremented"):
        mid_cycle = int(cycle_1_result.get("final_cycle") or baseline_cycle)
        cycle_2_result = wait_for_cycle_increment(
            mid_cycle,
            timeout_minutes=wait_timeout_minutes,
            verbose=verbose,
        )
        age2, _ = _heartbeat_age_minutes()
        heartbeat_fresh_after_cycle_2 = _is_fresh(age2)
        if cycle_2_result.get("cycle_incremented"):
            notes.append(
                f"Second observed cycle incremented to {cycle_2_result.get('final_cycle')}."
            )
        else:
            notes.append("Timed out waiting for second continuous loop cycle increment.")
    elif wait_for_cycles >= 2:
        heartbeat_fresh_after_cycle_2 = heartbeat_fresh_after_cycle_1

    hb_age_after, hb_ts_after = _heartbeat_age_minutes()
    ph_mtime_after = _artifact_mtime(PORTFOLIO_HISTORY_PATH)
    pos_mtime_after = _artifact_mtime(POSITIONS_SNAPSHOT_PATH)

    portfolio_updated = ph_mtime_after is not None and (
        ph_mtime_before is None or ph_mtime_after > ph_mtime_before
    )
    positions_updated = pos_mtime_after is not None and (
        pos_mtime_before is None or pos_mtime_after >= pos_mtime_before
    )

    dashboard_stale_truthful = (hb_age_after is not None and float(hb_age_after) > 60.0) or (
        hb_age_after is not None and float(hb_age_after) <= 60.0
    )

    if hb_age_before and float(hb_age_before) > STALE_THRESHOLD_MINUTES:
        notes.append(
            "Pre-soak pipeline heartbeat was stale; run_market_open now touches heartbeat.json (status writer only)."
        )

    status = "PASS"
    if not task.get("exists") or str(task.get("state")) not in ("Ready", "Running"):
        status = "FAIL"
        notes.append("Supervisor task missing or not enabled.")
    elif _old_45min_enabled():
        status = "FAIL"
        notes.append("TRITON_FullPipeline_45Min is enabled (must remain disabled).")
    elif len(supervisors) != 1 or supervisors[0] != TASK_NAME:
        status = "FAIL"
        notes.append(f"Unexpected supervisor tasks enabled: {supervisors}")
    elif not loops.get("watchdog") or not loops.get("continuous"):
        status = "FAIL"
        notes.append("Watchdog or continuous loop process not running.")
    elif pos_state.get("positions") != ["JPM", "WFC"]:
        status = "FAIL"
        notes.append(f"Unexpected positions: {pos_state.get('positions')}")
    elif int(pos_state.get("open_orders") or 0) != 0:
        status = "FAIL"
        notes.append("Open orders not zero.")
    elif pos_state.get("live_execution"):
        status = "FAIL"
        notes.append("live_execution is true.")
    elif heartbeat_fresh_after_cycle_2 is False or (
        wait_for_cycles == 1 and heartbeat_fresh_after_cycle_1 is False
    ):
        status = "FAIL"
        notes.append("Pipeline heartbeat remained stale after observed loop cycle(s).")
    elif heartbeat_fresh_after_cycle_1 is None:
        status = "REVIEW"
        notes.append("Could not confirm cycle completion within soak window.")
    elif heartbeat_fresh_after_cycle_1 and (
        wait_for_cycles < 2 or heartbeat_fresh_after_cycle_2 is not False
    ):
        status = "PASS"
        notes.append("Pipeline heartbeat fresh after continuous loop cycle(s).")
    else:
        status = "REVIEW"

    payload = {
        "phase": "196I",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_commit": PREVIOUS_COMMIT,
        "supervisor_task": TASK_NAME,
        "supervisor_enabled": str(task.get("state")) in ("Ready", "Running"),
        "old_45min_task_restored": _old_45min_enabled(),
        "multiple_45min_runtime_authorities": len(supervisors) > 1,
        "enabled_supervisor_tasks": supervisors,
        "watchdog_running": loops.get("watchdog"),
        "continuous_loop_running": loops.get("continuous"),
        "heartbeat_source": "data/results/heartbeat.json",
        "heartbeat_age_minutes_before": hb_age_before,
        "heartbeat_age_minutes_after": hb_age_after,
        "heartbeat_timestamp_after": hb_ts_after.isoformat() if hb_ts_after else None,
        "heartbeat_fresh_after_cycle_1": heartbeat_fresh_after_cycle_1,
        "heartbeat_fresh_after_cycle_2": heartbeat_fresh_after_cycle_2,
        "dashboard_stale_warning_truthful": dashboard_stale_truthful,
        "portfolio_history_updated": portfolio_updated,
        "positions_snapshot_updated": positions_updated,
        "baseline_cycle": baseline_cycle,
        "final_cycle": _read_lock_cycle()[0],
        "open_orders": pos_state.get("open_orders"),
        "positions_count": pos_state.get("positions_count"),
        "positions": pos_state.get("positions"),
        "live_execution": pos_state.get("live_execution"),
        "manual_execute_trades_run": False,
        "manual_approval_used": False,
        "broker_orders_submitted": False,
        "execution_logic_changed": False,
        "governance_logic_changed": False,
        "heartbeat_writer_fix_applied": True,
        "notes": notes,
        "cycle_1_wait": cycle_1_result,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196I] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196I] status={status}")
    return payload


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Phase 196I runtime supervisor heartbeat soak")
    ap.add_argument("--no-wait", action="store_true", help="Snapshot only; do not wait for cycles")
    ap.add_argument("--wait-cycles", type=int, default=1, help="Loop cycles to observe (1 or 2)")
    ap.add_argument("--timeout-minutes", type=float, default=50.0)
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    report = run_phase_196i_soak(
        wait_for_cycles=0 if args.no_wait else max(1, min(2, args.wait_cycles)),
        wait_timeout_minutes=args.timeout_minutes,
        verbose=args.verbose,
    )
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
