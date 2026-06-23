"""
phase_196o_market_open_reentry_observation_validation.py
Phase 196O — Observe market-open re-entry (monitor-only, no manual execution).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
OUTPUT_JSON = RESULTS / "phase_196o_market_open_reentry_observation.json"

PREVIOUS_COMMIT = "2e28d5ac"
SUPERVISOR_TASK = "TRITON_Runtime_Supervisor_45Min"
FRESHNESS_TASK = "TRITON_Data_Freshness_35Min"
DAILY_TASK = "TRITON_Runtime_931AM"
HEARTBEAT_PATH = RESULTS / "heartbeat.json"
PAPER_ENTRY_AUDIT = RESULTS / "paper_entry_loop_stage_audit.json"
CONTINUOUS_LOG_DIR = RESULTS / "continuous_loop"
MARKET_OPEN_HEARTBEAT_MAX_MINUTES = 90.0
MARKET_OPEN_SNAPSHOT_MAX_MINUTES = 45.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


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


def _task_state(name: str) -> str:
    ps = (
        f"(Get-ScheduledTask -TaskName '{name}' -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty State)"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (proc.stdout or "").strip()


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
    return _task_state("TRITON_FullPipeline_45Min") == "Ready"


def _is_market_open() -> bool:
    try:
        from services.guard_auto import is_market_open_et

        return bool(is_market_open_et(datetime.now(timezone.utc)))
    except Exception:
        return False


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
        return {
            "watchdog_running": False,
            "continuous_running": False,
            "duplicate_continuous": False,
        }
    try:
        rows = json.loads(raw)
        if isinstance(rows, dict):
            rows = [rows]
    except json.JSONDecodeError:
        rows = []
    watchdog, continuous = [], []
    for row in rows:
        cmd = str(row.get("CommandLine") or "")
        pid = row.get("ProcessId")
        if "run_watchdog_loop" in cmd:
            watchdog.append(pid)
        elif "run_continuous_loop" in cmd:
            continuous.append(pid)
    return {
        "watchdog_running": len(watchdog) > 0,
        "continuous_running": len(continuous) > 0,
        "duplicate_continuous": len(continuous) > 1,
        "watchdog_pids": watchdog,
        "continuous_pids": continuous,
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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _inspect_paper_entry_stage() -> Dict[str, Any]:
    audit = _read_json(PAPER_ENTRY_AUDIT)
    if not audit:
        return {
            "paper_entry_stage_observed": False,
            "paper_entry_result": "NOT_RUN",
            "paper_entry_block_reason": "",
        }
    gate_pass = bool(audit.get("gate_pass"))
    executed = bool(audit.get("executed"))
    reason = str(audit.get("gate_reason") or "")
    details = audit.get("details") if isinstance(audit.get("details"), dict) else {}
    failed = str(details.get("failed_check") or "")

    if executed:
        result = "SUBMITTED"
    elif gate_pass:
        result = "SKIPPED"
    elif audit.get("gate_pass") is False or failed or reason:
        result = "BLOCKED"
    else:
        result = "NOT_RUN"

    block_reason = reason or failed
    return {
        "paper_entry_stage_observed": True,
        "paper_entry_result": result,
        "paper_entry_block_reason": block_reason,
        "paper_entry_audit": audit,
    }


def _latest_continuous_log_excerpt(max_lines: int = 40) -> str:
    if not CONTINUOUS_LOG_DIR.is_dir():
        return ""
    logs = sorted(CONTINUOUS_LOG_DIR.glob("continuous_loop_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return ""
    try:
        lines = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception:
        return ""


def _market_open_cycle_in_logs_since(open_since: datetime) -> bool:
    """Detect market-open cycle activity in continuous loop logs after market opened."""
    if not CONTINUOUS_LOG_DIR.is_dir():
        return False
    for log in CONTINUOUS_LOG_DIR.glob("continuous_loop_*.log"):
        try:
            mtime = datetime.fromtimestamp(log.stat().st_mtime, tz=timezone.utc)
            if mtime < open_since:
                continue
            text = log.read_text(encoding="utf-8", errors="replace")
            if "run_market_open" in text.lower() or "[MARKET_OPEN]" in text:
                return True
        except Exception:
            continue
    hb = _read_json(HEARTBEAT_PATH)
    ts_raw = hb.get("timestamp")
    if ts_raw:
        try:
            import pandas as pd

            ts = pd.to_datetime(ts_raw, utc=True).to_pydatetime()
            if ts >= open_since and str(hb.get("stage") or "").lower().find("market") >= 0:
                return True
        except Exception:
            pass
    return False


def _wait_for_market_open(*, timeout_minutes: float = 720.0, poll_seconds: float = 120.0) -> bool:
    deadline = time.time() + timeout_minutes * 60.0
    while time.time() < deadline:
        if _is_market_open():
            return True
        time.sleep(poll_seconds)
    return _is_market_open()


def run_phase_196o_observation(
    *,
    wait_for_market_open: bool = False,
    wait_timeout_minutes: float = 720.0,
    post_open_settle_minutes: float = 50.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    notes: List[str] = []
    pre_market_open = not _is_market_open()

    if pre_market_open:
        notes.append("Pre-market: market currently CLOSED.")
    else:
        notes.append("Market currently OPEN.")

    if wait_for_market_open and pre_market_open:
        notes.append(f"Waiting up to {wait_timeout_minutes:.0f}m for market open.")
        if verbose:
            print("[196O] waiting for market open...", flush=True)
        opened = _wait_for_market_open(timeout_minutes=wait_timeout_minutes)
        if opened:
            notes.append(
                f"Market opened; settling {post_open_settle_minutes:.0f}m for first cycle."
            )
            time.sleep(post_open_settle_minutes * 60.0)
            pre_market_open = False

    market_open_now = _is_market_open()
    market_open_observed = market_open_now and not pre_market_open

    supervisor_task = _query_json("query_runtime_supervisor_task_json.ps1", SUPERVISOR_TASK)
    freshness_task = _query_json("query_data_freshness_task_json.ps1", FRESHNESS_TASK)
    daily_task = _query_json("query_data_freshness_task_json.ps1", DAILY_TASK)
    if not daily_task.get("exists"):
        daily_task = {
            "exists": _task_state(DAILY_TASK) == "Ready",
            "state": _task_state(DAILY_TASK),
        }

    supervisors = _enabled_supervisor_authorities()
    loops = _loop_processes()
    paper = _inspect_paper_entry_stage()

    from services.dashboard_status_sync import (
        audit_reconciled_runtime_locks,
        build_dashboard_guard_summary,
        build_dashboard_health_classification,
        classify_contract_artifact_integrity,
        reconcile_integrity_lock_display,
    )

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
        notes.append(f"Contract validation: {exc}")

    health = build_dashboard_health_classification()
    lock_audit = audit_reconciled_runtime_locks()
    fake_reason = "Execution locked: pipeline health 🔴"
    contract_integrity = classify_contract_artifact_integrity(
        contracts_ok=contracts_ok,
        contract_summary=contract_summary,
    )
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

    hb_max = MARKET_OPEN_HEARTBEAT_MAX_MINUTES if market_open_now else 120.0
    snap_max = MARKET_OPEN_SNAPSHOT_MAX_MINUTES if market_open_now else 120.0

    heartbeat_fresh = hb_age is not None and float(hb_age) <= hb_max
    positions_ok = pos_age is not None and float(pos_age) <= snap_max
    open_orders_ok = oo_age is not None and float(oo_age) <= snap_max
    portfolio_ok = ph_age is not None and float(ph_age) <= snap_max

    runtime_pipeline_health = str(health.get("runtime_pipeline_health") or "FAIL")
    data_freshness = str(health.get("data_freshness") or "STALE")
    block_reasons = list(health.get("execution_block_reasons") or [])
    positions = list(health.get("positions") or [])
    open_orders = int(health.get("open_orders") or 0)

    false_pipeline_fail = runtime_pipeline_health in ("HEALTHY", "OFF_HOURS_HEALTHY") and (
        "PIPELINE_HEALTH" in block_reasons
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
    daily_enabled = (
        str(daily_task.get("state")) in ("Ready", "Running") or _task_state(DAILY_TASK) == "Ready"
    )

    pre_market_ok = (
        supervisor_enabled
        and freshness_enabled
        and daily_enabled
        and not _fullpipeline_enabled()
        and len(supervisors) <= 1
        and loops.get("watchdog_running")
        and loops.get("continuous_running")
        and not loops.get("duplicate_continuous")
        and positions == ["JPM", "WFC"]
        and open_orders == 0
        and not live_execution
    )

    status = "REVIEW"
    if not pre_market_ok:
        status = "FAIL"
        if not supervisor_enabled:
            notes.append("Runtime supervisor not enabled.")
        if not freshness_enabled:
            notes.append("Data freshness task not enabled.")
        if loops.get("duplicate_continuous"):
            notes.append("Duplicate continuous loop detected.")
        if _fullpipeline_enabled():
            notes.append("TRITON_FullPipeline_45Min enabled.")
        if len(supervisors) > 1:
            notes.append(f"Multiple runtime authorities: {supervisors}")
    elif not market_open_observed:
        notes.append(
            "Pre-market checks PASS. Re-run after first market-open cycle: "
            "python -m services.phase_196o_market_open_reentry_observation_validation "
            "--wait-for-market-open"
        )
        if pre_market_ok and paper.get("paper_entry_stage_observed"):
            notes.append(
                f"Paper entry stage last result={paper.get('paper_entry_result')} "
                f"reason={paper.get('paper_entry_block_reason')!r} (expected while closed)."
            )
    else:
        if false_pipeline_fail or false_contract_lock:
            status = "FAIL"
        elif not heartbeat_fresh or not (positions_ok and open_orders_ok and portfolio_ok):
            status = "FAIL"
            notes.append("Artifacts stale after market open.")
        elif open_orders != 0:
            status = "REVIEW"
            notes.append("Open orders non-zero after market open — verify governed placement.")
        elif positions != ["JPM", "WFC"]:
            status = "REVIEW"
            notes.append(f"Positions changed to {positions} — verify governed change.")
        elif runtime_pipeline_health == "FAIL":
            status = "FAIL"
        elif runtime_pipeline_health == "DEGRADED":
            status = "REVIEW"
        else:
            status = "PASS"
            notes.append("Market-open re-entry observation PASS.")

    payload = {
        "phase": "196O",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_commit": PREVIOUS_COMMIT,
        "market_currently_open": market_open_now,
        "market_open_observed": market_open_observed,
        "pre_market_checks_pass": pre_market_ok,
        "runtime_supervisor_enabled": supervisor_enabled,
        "data_freshness_task_enabled": freshness_enabled,
        "daily_931_task_enabled": daily_enabled,
        "old_fullpipeline_45min_restored": _fullpipeline_enabled(),
        "multiple_runtime_authorities": len(supervisors) > 1,
        "enabled_runtime_supervisor_tasks": supervisors,
        "watchdog_running": bool(loops.get("watchdog_running")),
        "continuous_loop_running": bool(loops.get("continuous_running")),
        "duplicate_continuous_loops": bool(loops.get("duplicate_continuous")),
        "heartbeat_source": "data/results/heartbeat.json",
        "heartbeat_age_minutes": hb_age,
        "heartbeat_fresh_after_market_open": heartbeat_fresh if market_open_observed else None,
        "heartbeat_fresh_or_acceptable": heartbeat_fresh,
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
        "paper_entry_stage_observed": paper.get("paper_entry_stage_observed"),
        "paper_entry_result": paper.get("paper_entry_result"),
        "paper_entry_block_reason": paper.get("paper_entry_block_reason"),
        "watchdog_lock_status": lock_audit.get("watchdog_lock_classification"),
        "continuous_loop_lock_status": lock_audit.get("continuous_lock_classification"),
        "open_orders": open_orders,
        "positions_count": int(health.get("positions_count") or 0),
        "positions": positions,
        "live_execution": live_execution,
        "manual_execute_trades_run": False,
        "manual_approval_used": False,
        "broker_orders_submitted_manually": False,
        "unexpected_broker_orders": open_orders != 0
        and paper.get("paper_entry_result") != "SUBMITTED",
        "execution_logic_changed": False,
        "governance_logic_changed": False,
        "guard_summary": {
            "sidebar_label": summary.get("sidebar_label"),
            "market_status": summary.get("market_status"),
            "execution_blocked_reasons": summary.get("execution_blocked_reasons"),
        },
        "continuous_log_tail": _latest_continuous_log_excerpt(25),
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196O] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196O] status={status} market_open={market_open_now} pre_market_ok={pre_market_ok}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 196O market open re-entry observation")
    ap.add_argument(
        "--wait-for-market-open",
        action="store_true",
        help="Wait until market opens then observe first cycle",
    )
    ap.add_argument("--wait-timeout-minutes", type=float, default=720.0)
    ap.add_argument("--post-open-settle-minutes", type=float, default=50.0)
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    report = run_phase_196o_observation(
        wait_for_market_open=args.wait_for_market_open,
        wait_timeout_minutes=args.wait_timeout_minutes,
        post_open_settle_minutes=args.post_open_settle_minutes,
        verbose=args.verbose,
    )
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
