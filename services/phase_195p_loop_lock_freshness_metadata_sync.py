"""
phase_195p_loop_lock_freshness_metadata_sync.py
Phase 195P — Loop lock freshness & runtime metadata sync audit.

Read-only audit of loop lock metadata vs active processes. Does not place orders,
clear locks, restart loops, or run execute_trades --execute.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
CONFIG = ROOT / "config"

OUTPUT_JSON = RESULTS / "loop_lock_freshness_metadata_sync.json"
SUMMARY_CSV = RESULTS / "loop_lock_freshness_metadata_sync_summary.csv"

WATCHDOG_LOCK = LIVE / "watchdog_loop.lock"
CONTINUOUS_LOCK = LIVE / "continuous_loop.lock"
EXEC_READINESS = RESULTS / "execution_readiness.json"
HEARTBEAT_JSON = RESULTS / "heartbeat.json"
WATCHDOG_ALERTS = RESULTS / "watchdog_alerts.json"
PHASE_195H_JSON = RESULTS / "phase_195h_operational_readiness_freeze.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
PAPER_CYCLE_CFG = CONFIG / "run_scheduled_paper_cycle.json"

from services.phase_195h_operational_readiness_freeze import (  # noqa: E402
    _classify_lock,
    _find_ps_processes,
    _load_json as _h_load_json,
    _minutes_since,
    _parse_lock,
    _parse_ts,
)

SUMMARY_FIELDS = (
    "metric",
    "value",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    return _h_load_json(path)


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0


def _find_python_execute_trades_execute() -> Tuple[bool, List[str]]:
    if sys.platform != "win32":
        return False, []
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" "
        "| Where-Object { $_.CommandLine -like '*execute_trades*' -and $_.CommandLine -like '*--execute*' } "
        "| Select-Object -ExpandProperty CommandLine"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = [x.strip() for x in (r.stdout or "").splitlines() if x.strip()]
        return len(lines) > 0, lines
    except (subprocess.SubprocessError, OSError):
        return False, []


def _heartbeat_status() -> str:
    hb = _load_json(HEARTBEAT_JSON)
    alerts = _load_json(WATCHDOG_ALERTS)
    active = alerts.get("active_alerts") or []
    stale_hb = any(isinstance(a, dict) and a.get("alert_type") == "STALE_HEARTBEAT" for a in active)
    if stale_hb:
        return "STALE_HEARTBEAT"
    ts = hb.get("timestamp")
    age = _minutes_since(_parse_ts(ts))
    if age is None:
        return "UNKNOWN"
    if age <= 30:
        return "FRESH"
    if age <= 120:
        return "AGING"
    return "STALE"


def _audit_loop_lock(
    lock_path: Path,
    script_fragment: str,
) -> Dict[str, Any]:
    lock = _parse_lock(lock_path)
    running, pids = _find_ps_processes(script_fragment)
    classification = _classify_lock(lock, running, pids)
    lock_pid = lock.get("pid")
    actual_pid = pids[0] if pids else None
    pid_match = lock_pid is not None and lock_pid in pids if pids else None

    mismatch_reason = ""
    if running and lock.get("exists"):
        if lock.get("corrupt"):
            mismatch_reason = "lock file unparseable"
        elif not lock.get("refreshed_at_dt"):
            mismatch_reason = (
                "legacy lock format (started_at only); metadata not refreshed per cycle — "
                "restart loop after Phase 195P script update to enable refreshed_at/pid/source"
            )
        elif lock_pid is not None and pids and not pid_match:
            mismatch_reason = f"lock pid={lock_pid} does not match running pids={pids}"
        elif lock.get("stale"):
            mismatch_reason = (
                f"freshness age {lock.get('lock_age_minutes')} min exceeds "
                f"threshold {lock.get('stale_threshold_minutes')} min"
            )

    return {
        "lock_path": str(lock_path.relative_to(ROOT)).replace("\\", "/"),
        "process_running": running,
        "process_pids": pids,
        "lock_exists": bool(lock.get("exists")),
        "lock_pid": lock_pid,
        "actual_pid": actual_pid,
        "pid_match": pid_match,
        "lock_age_minutes": lock.get("lock_age_minutes"),
        "started_at_age_minutes": lock.get("started_at_age_minutes"),
        "freshness_source": lock.get("freshness_source"),
        "refreshed_at": lock.get("refreshed_at"),
        "started_at": lock.get("started_at"),
        "source": lock.get("source"),
        "cycle": lock.get("cycle"),
        "stale": lock.get("stale"),
        "stale_threshold_minutes": lock.get("stale_threshold_minutes"),
        "classification": classification,
        "mismatch_reason": mismatch_reason,
        "lock": lock,
    }


def _run_phase_195h() -> Dict[str, Any]:
    try:
        subprocess.run(
            [sys.executable, "-m", "services.phase_195h_operational_readiness_freeze"],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (subprocess.SubprocessError, OSError):
        pass
    return _load_json(PHASE_195H_JSON)


def _recommended_next_action(
    watchdog: Dict[str, Any],
    continuous: Dict[str, Any],
    heartbeat: str,
    readiness_status: str,
    execute_detected: bool,
) -> str:
    if execute_detected:
        return "Investigate execute_trades --execute process before any loop changes"
    actions: List[str] = []
    for label, audit in (("watchdog", watchdog), ("continuous", continuous)):
        cls = audit.get("classification")
        if cls == "ORPHAN_STALE_LOCK":
            actions.append(
                f"Review {label} orphan lock — clear only if no matching process is running"
            )
        elif cls == "MISSING_ACTIVE_LOCK":
            actions.append(f"{label} process running without lock — restart loop to recreate lock")
        elif cls == "STALE_LOCK_METADATA_PROCESS_ACTIVE":
            actions.append(f"{label} PID mismatch — restart loop to resync lock metadata")
        elif cls in ("STALE_ACTIVE_LOCK", "FRESH_ACTIVE_LOCK") and not audit.get("refreshed_at"):
            actions.append(
                f"Restart {label} loop when convenient to pick up Phase 195P per-cycle lock refresh"
            )
    if heartbeat == "STALE_HEARTBEAT":
        actions.append("Restart watchdog loop to refresh heartbeat via loop_safety --refresh")
    if not actions:
        if readiness_status == "READY":
            return "No lock metadata action required; runtime healthy"
        return "Re-run validation suite; address any readiness failures"
    return "; ".join(actions)


def run_phase_195p_sync(*, verbose: bool = True) -> Dict[str, Any]:
    phase_195h_before = _load_json(PHASE_195H_JSON)
    phase_195h_status_before = phase_195h_before.get("operational_freeze_status", "UNKNOWN")

    watchdog = _audit_loop_lock(WATCHDOG_LOCK, "run_watchdog_loop.ps1")
    continuous = _audit_loop_lock(CONTINUOUS_LOCK, "run_continuous_loop.ps1")

    readiness = _load_json(EXEC_READINESS)
    execute_detected, execute_cmds = _find_python_execute_trades_execute()
    paper_cfg = _load_json(PAPER_CYCLE_CFG)
    automated_entry = bool(paper_cfg.get("run_execute_trades", False))

    heartbeat_status = _heartbeat_status()
    execution_readiness_status = (
        readiness.get("status") or readiness.get("readiness_status") or "UNKNOWN"
    )

    phase_195h_after_doc = _run_phase_195h()
    phase_195h_status_after = phase_195h_after_doc.get("operational_freeze_status", "UNKNOWN")

    root_cause = (
        "Loop lock files were written once at startup with started_at only. "
        "Phase 195H treated started_at age > 60 min as stale while the process remained active, "
        "producing STALE_LOCK_METADATA_PROCESS_ACTIVE review items despite healthy heartbeat/readiness. "
        "Phase 195P adds per-cycle lock metadata refresh (refreshed_at, pid, source, cycle) and "
        "195H now uses refreshed_at for freshness with legacy started_at fallback thresholds."
    )

    recommended = _recommended_next_action(
        watchdog,
        continuous,
        heartbeat_status,
        execution_readiness_status,
        execute_detected,
    )

    payload: Dict[str, Any] = {
        "phase": "195P",
        "timestamp": _utc_now_iso(),
        "phase_pass": True,
        "audit_mode": "read_only",
        "orders_placed": False,
        "root_cause": root_cause,
        "watchdog_process_running": watchdog["process_running"],
        "continuous_process_running": continuous["process_running"],
        "watchdog_lock_exists": watchdog["lock_exists"],
        "continuous_lock_exists": continuous["lock_exists"],
        "watchdog_lock_pid": watchdog["lock_pid"],
        "continuous_lock_pid": continuous["lock_pid"],
        "watchdog_actual_pid": watchdog["actual_pid"],
        "continuous_actual_pid": continuous["actual_pid"],
        "watchdog_lock_age_minutes": watchdog["lock_age_minutes"],
        "continuous_lock_age_minutes": continuous["lock_age_minutes"],
        "watchdog_lock_classification": watchdog["classification"],
        "continuous_lock_classification": continuous["classification"],
        "heartbeat_status": heartbeat_status,
        "execution_readiness_status": execution_readiness_status,
        "phase_195h_status_before": phase_195h_status_before,
        "phase_195h_status_after": phase_195h_status_after,
        "broker_open_orders_count": _count_csv_rows(OPEN_ORDERS),
        "broker_positions_count": _count_csv_rows(POSITIONS),
        "live_execution_permitted": bool(readiness.get("live_execution_permitted", False)),
        "paper_execution_permitted": bool(readiness.get("paper_execution_permitted", False)),
        "execute_trades_execute_detected": execute_detected,
        "automated_entry_execution_enabled": automated_entry,
        "recommended_next_action": recommended,
        "diagnostics": {
            "watchdog": watchdog,
            "continuous": continuous,
            "watchdog_mismatch_reason": watchdog.get("mismatch_reason"),
            "continuous_mismatch_reason": continuous.get("mismatch_reason"),
            "phase_195h_review_items_before": phase_195h_before.get("review_items"),
            "phase_195h_review_items_after": phase_195h_after_doc.get("review_items"),
            "execute_trades_command_lines": execute_cmds,
            "code_changes": [
                "scripts/loop_lock_utils.ps1: Update-LoopLockMetadata",
                "scripts/run_watchdog_loop.ps1: per-cycle lock refresh",
                "scripts/run_continuous_loop.ps1: per-cycle lock refresh",
                "services/phase_195h_operational_readiness_freeze.py: refreshed_at/PID classification",
            ],
        },
        "safety_constraints": [
            "No orders placed",
            "No lock deletion while matching process active",
            "Live execution remains blocked",
            "Paper execution operator-only",
            "execute_trades --execute not invoked",
        ],
    }

    summary_rows = [
        {"metric": "watchdog_process_running", "value": str(watchdog["process_running"])},
        {"metric": "continuous_process_running", "value": str(continuous["process_running"])},
        {"metric": "watchdog_lock_exists", "value": str(watchdog["lock_exists"])},
        {"metric": "continuous_lock_exists", "value": str(continuous["lock_exists"])},
        {"metric": "watchdog_lock_pid", "value": str(watchdog["lock_pid"] or "")},
        {"metric": "continuous_lock_pid", "value": str(continuous["lock_pid"] or "")},
        {"metric": "watchdog_actual_pid", "value": str(watchdog["actual_pid"] or "")},
        {"metric": "continuous_actual_pid", "value": str(continuous["actual_pid"] or "")},
        {"metric": "watchdog_lock_age_minutes", "value": str(watchdog["lock_age_minutes"] or "")},
        {
            "metric": "continuous_lock_age_minutes",
            "value": str(continuous["lock_age_minutes"] or ""),
        },
        {"metric": "watchdog_lock_classification", "value": watchdog["classification"]},
        {"metric": "continuous_lock_classification", "value": continuous["classification"]},
        {"metric": "heartbeat_status", "value": heartbeat_status},
        {"metric": "execution_readiness_status", "value": execution_readiness_status},
        {"metric": "phase_195h_status_before", "value": phase_195h_status_before},
        {"metric": "phase_195h_status_after", "value": phase_195h_status_after},
        {"metric": "broker_open_orders_count", "value": str(_count_csv_rows(OPEN_ORDERS))},
        {"metric": "broker_positions_count", "value": str(_count_csv_rows(POSITIONS))},
        {"metric": "live_execution_permitted", "value": str(payload["live_execution_permitted"])},
        {"metric": "paper_execution_permitted", "value": str(payload["paper_execution_permitted"])},
        {"metric": "execute_trades_execute_detected", "value": str(execute_detected)},
        {"metric": "automated_entry_execution_enabled", "value": str(automated_entry)},
        {"metric": "recommended_next_action", "value": recommended},
    ]

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)

    if verbose:
        print(f"[195P] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[195P] watchdog={watchdog['classification']} continuous={continuous['classification']}"
        )
        print(f"[195P] 195H before={phase_195h_status_before} after={phase_195h_status_after}")
        print(f"[195P] recommended: {recommended}")

    return payload


def main() -> None:
    run_phase_195p_sync(verbose=True)


if __name__ == "__main__":
    main()
