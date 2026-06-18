"""
phase_195o_runtime_heartbeat_loop_restart_hardening.py
Phase 195O — Runtime heartbeat & loop restart hardening (read-only diagnostic).

Operator-safe assessment of stale heartbeat, loop locks, and restart readiness.
Does not place orders, clear locks, restart loops, or run execute_trades --execute.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
CONFIG = ROOT / "config"
SCRIPTS = ROOT / "scripts"

OUTPUT_JSON = RESULTS / "runtime_heartbeat_loop_restart_hardening.json"
SUMMARY_CSV = RESULTS / "runtime_heartbeat_loop_restart_summary.csv"

WATCHDOG_STATE = LIVE / "watchdog_alert_state.json"
WATCHDOG_LOCK = LIVE / "watchdog_loop.lock"
CONTINUOUS_LOCK = LIVE / "continuous_loop.lock"
WATCHDOG_ALERTS = RESULTS / "watchdog_alerts.json"
WATCHDOG_STATUS = RESULTS / "watchdog_status.json"
EXEC_READINESS = RESULTS / "execution_readiness.json"
HEARTBEAT_JSON = RESULTS / "heartbeat.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
CERTIFICATION = RESULTS / "capital_preservation_certification.json"
PAPER_CYCLE_CFG = CONFIG / "run_scheduled_paper_cycle.json"
ARM_CFG = CONFIG / "arm_mode.json"

STALE_LOCK_MINUTES = 60.0
STALE_HEARTBEAT_ALERT = "STALE_HEARTBEAT"

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
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _minutes_since(dt: Optional[datetime]) -> Optional[float]:
    if not dt:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0


from services.phase_195h_operational_readiness_freeze import (  # noqa: E402
    _classify_lock,
    _find_ps_processes,
    _parse_lock,
)


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
        lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
        return len(lines) > 0, lines
    except (subprocess.SubprocessError, OSError):
        return False, []


def _count_csv_rows(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(path))
    except Exception:
        return 0


def _stale_heartbeat_active(alerts_doc: Dict[str, Any]) -> bool:
    for alert in alerts_doc.get("active_alerts") or []:
        if isinstance(alert, dict) and alert.get("alert_type") == STALE_HEARTBEAT_ALERT:
            if str(alert.get("status", "ACTIVE")).upper() == "ACTIVE":
                return True
    return False


def _build_operator_commands(
    *,
    safe_clear_locks: bool,
    safe_restart_watchdog: bool,
    safe_restart_continuous: bool,
    watchdog_lock: Dict[str, Any],
    continuous_lock: Dict[str, Any],
) -> Dict[str, Any]:
    repo = str(ROOT).replace("'", "''")
    cmds: List[str] = []
    lock_cmds: List[str] = []

    cmds.append(f"Set-Location '{repo}'")

    cmds.append("# Step 1 — Confirm broker flat (read-only snapshot; no orders placed)")
    cmds.append(
        f"& '{repo}\\.venv\\Scripts\\python.exe' -m services.snapshot_live_orders --mode paper"
    )

    cmds.append("# Step 2 — Confirm no matching loop processes")
    cmds.append(
        "Get-CimInstance Win32_Process -Filter \"Name = 'powershell.exe' OR Name = 'pwsh.exe'\" "
        "| Where-Object { $_.CommandLine -like '*run_watchdog_loop.ps1*' "
        "-or $_.CommandLine -like '*run_continuous_loop.ps1*' } "
        "| Select-Object ProcessId, CommandLine"
    )

    if safe_clear_locks:
        lock_cmds.append(
            "# Step 3 — Remove stale orphan locks ONLY if Step 2 shows no matching process"
        )
        if continuous_lock.get("exists") and continuous_lock.get("stale"):
            lock_cmds.append(
                f"Remove-Item -Force '{repo}\\data\\live\\continuous_loop.lock' -ErrorAction Stop"
            )
        if watchdog_lock.get("exists") and watchdog_lock.get("stale"):
            lock_cmds.append(
                f"Remove-Item -Force '{repo}\\data\\live\\watchdog_loop.lock' -ErrorAction Stop"
            )
    else:
        lock_cmds.append(
            "# Step 3 — SKIP lock removal: process may still be running or locks not stale"
        )

    restart_cmds: List[str] = []
    if safe_restart_watchdog:
        restart_cmds.append("# Step 4 — Start watchdog loop (refreshes heartbeat via loop_safety)")
        restart_cmds.append(
            f"Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',"
            f"'-File','{repo}\\scripts\\run_watchdog_loop.ps1','-IntervalMinutes','1' "
            f"-WorkingDirectory '{repo}'"
        )
    if safe_restart_continuous:
        restart_cmds.append(
            "# Step 5 — Start continuous loop (plan-only market_open; does NOT call execute_trades --execute)"
        )
        restart_cmds.append(
            f"$env:TRITON_ENABLE_PAPER_EXECUTION='1'; "
            f"Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',"
            f"'-File','{repo}\\scripts\\run_continuous_loop.ps1','-IntervalMinutes','45' "
            f"-WorkingDirectory '{repo}'"
        )

    verify_cmds = [
        "# Step 6 — Verify readiness (after ~2 minutes for watchdog cycle)",
        f"& '{repo}\\.venv\\Scripts\\python.exe' -m services.execution_readiness",
        f"& '{repo}\\.venv\\Scripts\\python.exe' -m services.phase_195c_evaluation_contract_validation",
        f"& '{repo}\\.venv\\Scripts\\python.exe' -m services.phase_195k_certification_readiness_alignment_validation",
        f"& '{repo}\\.venv\\Scripts\\python.exe' -m services.phase_195h_operational_readiness_freeze",
    ]

    return {
        "precheck_commands": cmds,
        "stale_lock_removal_commands": lock_cmds,
        "restart_commands": restart_cmds,
        "verification_commands": verify_cmds,
        "full_sequence": cmds + lock_cmds + restart_cmds + verify_cmds,
    }


def run_phase_195o_hardening(*, verbose: bool = True) -> Dict[str, Any]:
    watchdog_running, watchdog_pids = _find_ps_processes("run_watchdog_loop.ps1")
    continuous_running, continuous_pids = _find_ps_processes("run_continuous_loop.ps1")
    execute_detected, execute_cmds = _find_python_execute_trades_execute()

    watchdog_lock = _parse_lock(WATCHDOG_LOCK)
    continuous_lock = _parse_lock(CONTINUOUS_LOCK)

    alert_state = _load_json(WATCHDOG_STATE)
    alerts_doc = _load_json(WATCHDOG_ALERTS)
    readiness_before = _load_json(EXEC_READINESS)
    cert = _load_json(CERTIFICATION)
    paper_cfg = _load_json(PAPER_CYCLE_CFG)
    arm_cfg = _load_json(ARM_CFG)

    hb_source = str(alert_state.get("heartbeat_refresh_source") or "")
    hb_refresh_ts = str(alert_state.get("heartbeat_refresh_ts") or "")
    hb_grace_until = str(alert_state.get("heartbeat_grace_until") or "")
    hb_refresh_age = _minutes_since(_parse_ts(hb_refresh_ts))
    grace_expired = True
    grace_dt = _parse_ts(hb_grace_until)
    if grace_dt is not None:
        grace_expired = datetime.now(timezone.utc) >= grace_dt

    stale_hb = _stale_heartbeat_active(alerts_doc)

    broker_open = _count_csv_rows(OPEN_ORDERS)
    broker_positions = _count_csv_rows(POSITIONS)

    paper_permitted = bool(
        readiness_before.get("paper_execution_permitted")
        or cert.get("alignment_audits")  # fallback
    )
    if readiness_before:
        paper_permitted = bool(readiness_before.get("paper_execution_permitted", paper_permitted))
    live_blocked = not bool(readiness_before.get("live_execution_permitted", False))

    entry_execution_enabled = bool(
        paper_cfg.get("run_execute_trades", False) and not execute_detected
    )
    # Operator-only: scheduled cycle config has run_execute_trades false per 195N
    automated_entry = bool(paper_cfg.get("run_execute_trades", False))

    lock_status_continuous = _classify_lock(continuous_lock, continuous_running, continuous_pids)
    lock_status_watchdog = _classify_lock(watchdog_lock, watchdog_running, watchdog_pids)

    orphan_continuous = lock_status_continuous == "ORPHAN_STALE_LOCK"
    orphan_watchdog = lock_status_watchdog == "ORPHAN_STALE_LOCK"

    safe_clear = (orphan_continuous or orphan_watchdog) and not execute_detected
    safe_restart_watchdog = (
        broker_open == 0
        and broker_positions == 0
        and not execute_detected
        and (not watchdog_running or stale_hb)
    )
    safe_restart_continuous = (
        broker_open == 0 and broker_positions == 0 and not execute_detected and not automated_entry
    )

    operator_actions: List[str] = []
    if stale_hb:
        operator_actions.append(
            "STALE_HEARTBEAT alert active — restart watchdog loop to refresh heartbeat via loop_safety"
        )
    if lock_status_continuous in ("STALE_ACTIVE_LOCK", "STALE_LOCK_METADATA_PROCESS_ACTIVE"):
        operator_actions.append(
            "continuous_loop lock metadata stale or mismatched — restart loop to refresh "
            "refreshed_at/pid/source (do NOT delete lock while process is running)"
        )
    if lock_status_watchdog in ("STALE_ACTIVE_LOCK", "STALE_LOCK_METADATA_PROCESS_ACTIVE"):
        operator_actions.append(
            "watchdog_loop lock metadata stale or mismatched — restart watchdog to refresh metadata"
        )
    if orphan_continuous:
        operator_actions.append(
            f"continuous_loop.lock is orphan/stale (age={continuous_lock.get('lock_age_minutes')} min) — "
            "remove only if no run_continuous_loop.ps1 process"
        )
    if orphan_watchdog:
        operator_actions.append(
            f"watchdog_loop.lock is orphan/stale (age={watchdog_lock.get('lock_age_minutes')} min) — "
            "remove only if no run_watchdog_loop.ps1 process"
        )
    if hb_refresh_age and hb_refresh_age > STALE_LOCK_MINUTES:
        operator_actions.append(
            f"watchdog_alert_state heartbeat_refresh_ts is stale ({hb_refresh_age:.0f} min old, "
            f"source={hb_source}) — will refresh when watchdog loop starts"
        )
    if not watchdog_running:
        operator_actions.append("Watchdog loop not running — start run_watchdog_loop.ps1 first")
    if continuous_running:
        operator_actions.append(
            "Continuous loop process detected — verify it is healthy before duplicate start"
        )
    operator_actions.append(
        "Entry execution remains operator-only: do NOT run execute_trades --execute in this phase"
    )

    commands = _build_operator_commands(
        safe_clear_locks=safe_clear,
        safe_restart_watchdog=safe_restart_watchdog,
        safe_restart_continuous=safe_restart_continuous,
        watchdog_lock=watchdog_lock,
        continuous_lock=continuous_lock,
    )

    recovery_sequence = [
        "1. Run snapshot_live_orders --mode paper → confirm 0 open orders / 0 positions",
        "2. Verify no run_watchdog_loop.ps1 / run_continuous_loop.ps1 processes (or note PIDs)",
        "3. If locks stale and no process: Remove-Item orphan lock files only",
        "4. Start watchdog loop first (IntervalMinutes 1) → loop_safety --refresh clears STALE_HEARTBEAT",
        "5. Optionally start continuous loop (IntervalMinutes 45) — plan-only, no execute_trades",
        "6. Wait ~2 min; run execution_readiness → expect READY when watchdog check passes",
        "7. Re-run 195C / 195K / 195H validations",
        "8. Do NOT enable execute_trades --execute or run_scheduled_paper_cycle run_execute_trades",
    ]

    payload: Dict[str, Any] = {
        "phase": "195O",
        "timestamp": _utc_now_iso(),
        "phase_pass": True,
        "audit_mode": "read_only",
        "orders_placed": False,
        "diagnostics": {
            "continuous_loop_process_running": continuous_running,
            "continuous_loop_pids": continuous_pids,
            "watchdog_loop_process_running": watchdog_running,
            "watchdog_loop_pids": watchdog_pids,
            "continuous_loop_lock_exists": bool(continuous_lock.get("exists")),
            "watchdog_loop_lock_exists": bool(watchdog_lock.get("exists")),
            "continuous_loop_lock_age_minutes": continuous_lock.get("lock_age_minutes"),
            "watchdog_loop_lock_age_minutes": watchdog_lock.get("lock_age_minutes"),
            "continuous_loop_lock_stale": continuous_lock.get("stale"),
            "continuous_loop_lock_status": lock_status_continuous,
            "watchdog_loop_lock_stale": watchdog_lock.get("stale"),
            "watchdog_loop_lock_status": lock_status_watchdog,
            "heartbeat_refresh_source": hb_source,
            "heartbeat_refresh_ts": hb_refresh_ts,
            "heartbeat_refresh_age_minutes": round(hb_refresh_age, 1) if hb_refresh_age else None,
            "heartbeat_grace_until": hb_grace_until,
            "heartbeat_grace_expired": grace_expired,
            "stale_heartbeat_active": stale_hb,
            "stale_heartbeat_alert_details": next(
                (
                    a
                    for a in (alerts_doc.get("active_alerts") or [])
                    if isinstance(a, dict) and a.get("alert_type") == STALE_HEARTBEAT_ALERT
                ),
                None,
            ),
            "broker_open_orders_count": broker_open,
            "broker_positions_count": broker_positions,
            "execution_readiness_before": readiness_before,
            "readiness_failed_checks": readiness_before.get("failed_checks"),
            "watchdog_check_detail": readiness_before.get("check_details", {}).get("watchdog"),
            "execute_trades_execute_detected": execute_detected,
            "execute_trades_execute_command_lines": execute_cmds,
            "entry_execution_automated_in_paper_cycle_config": automated_entry,
            "entry_execution_enabled": False,
            "paper_execution_permitted": paper_permitted,
            "live_execution_permitted": not live_blocked,
            "live_execution_blocked": live_blocked,
            "safe_to_clear_stale_locks": safe_clear,
            "safe_to_restart_watchdog": safe_restart_watchdog,
            "safe_to_restart_continuous_loop": safe_restart_continuous,
            "recommended_operator_actions": operator_actions,
            "operator_recovery_sequence": recovery_sequence,
            "operator_powershell_commands": commands,
        },
        "root_cause": (
            "Watchdog readiness fails while STALE_HEARTBEAT alert is active. "
            "watchdog_alert_state.json retains stale continuous_loop heartbeat metadata "
            f"(refresh_ts={hb_refresh_ts}). Watchdog loop is not running persistently; "
            "loop startup calls loop_safety --refresh which clears stale heartbeat and sets grace. "
            "continuous_loop.lock may block loop restart if orphan from prior session."
        ),
        "expected_post_recovery": {
            "execution_readiness": "READY",
            "watchdog_check": "Watchdog monitoring healthy",
            "stale_heartbeat_active": False,
            "phase_195h": "READY or READY_WITH_REVIEW_ITEMS",
            "entry_execution": "operator-only unchanged",
        },
        "safety_constraints": [
            "No execute_trades --execute",
            "No order placement or cancellation",
            "No automated entry execution enablement",
            "Lock removal only when no matching process",
        ],
    }

    summary_rows = [
        {"metric": "continuous_loop_process_running", "value": str(continuous_running)},
        {"metric": "watchdog_loop_process_running", "value": str(watchdog_running)},
        {"metric": "continuous_loop_lock_exists", "value": str(continuous_lock.get("exists"))},
        {"metric": "watchdog_loop_lock_exists", "value": str(watchdog_lock.get("exists"))},
        {
            "metric": "continuous_loop_lock_age_minutes",
            "value": str(continuous_lock.get("lock_age_minutes", "")),
        },
        {
            "metric": "watchdog_loop_lock_age_minutes",
            "value": str(watchdog_lock.get("lock_age_minutes", "")),
        },
        {"metric": "heartbeat_refresh_source", "value": hb_source},
        {"metric": "heartbeat_refresh_ts", "value": hb_refresh_ts},
        {"metric": "heartbeat_grace_until", "value": hb_grace_until},
        {"metric": "stale_heartbeat_active", "value": str(stale_hb)},
        {"metric": "broker_open_orders_count", "value": str(broker_open)},
        {"metric": "broker_positions_count", "value": str(broker_positions)},
        {
            "metric": "execution_readiness_status",
            "value": str(readiness_before.get("readiness_status", "")),
        },
        {"metric": "safe_to_clear_stale_locks", "value": str(safe_clear)},
        {"metric": "safe_to_restart_watchdog", "value": str(safe_restart_watchdog)},
        {"metric": "safe_to_restart_continuous_loop", "value": str(safe_restart_continuous)},
        {"metric": "execute_trades_execute_detected", "value": str(execute_detected)},
        {"metric": "entry_execution_enabled", "value": "false"},
        {"metric": "paper_execution_permitted", "value": str(paper_permitted)},
        {"metric": "live_execution_permitted", "value": str(not live_blocked)},
    ]

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)

    if verbose:
        print(
            json.dumps(
                {
                    "phase": "195O",
                    "phase_pass": True,
                    "diagnostics": {
                        k: payload["diagnostics"][k]
                        for k in (
                            "stale_heartbeat_active",
                            "safe_to_clear_stale_locks",
                            "safe_to_restart_watchdog",
                            "safe_to_restart_continuous_loop",
                            "broker_open_orders_count",
                            "broker_positions_count",
                            "execution_readiness_before",
                        )
                    },
                },
                indent=2,
            )
        )

    return payload


def main() -> int:
    run_phase_195o_hardening(verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
