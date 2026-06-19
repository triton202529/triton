"""
phase_195u_scheduler_stability_observation_gate.py
Phase 195U — Scheduler stability observation gate.

Read-only observation of TRITON_DailyRefresh and authoritative runtime health.
Does not modify scheduled tasks, place orders, or run isolation commands.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
CONFIG = ROOT / "config"

OUTPUT_JSON = RESULTS / "scheduler_stability_observation_gate.json"
SUMMARY_CSV = RESULTS / "scheduler_stability_observation_gate_summary.csv"
OBSERVATION_LOG_CSV = RESULTS / "scheduler_stability_observation_log.csv"

EXEC_READINESS = RESULTS / "execution_readiness.json"
PHASE_195C = RESULTS / "phase_195c_evaluation_contract_validation.json"
PHASE_195K = RESULTS / "phase_195k_certification_readiness_alignment_validation.json"
PHASE_195H = RESULTS / "phase_195h_operational_readiness_freeze.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
WATCHDOG_STATE = LIVE / "watchdog_alert_state.json"
PAPER_CYCLE_CFG = CONFIG / "run_scheduled_paper_cycle.json"
PLAN_195T = RESULTS / "legacy_scheduler_isolation_dry_run_plan.json"

AUTHORITATIVE_TASK = "TRITON_DailyRefresh"
MONITORING_TASKS = (
    "TRITON_MarketOpen",
    "TRITON_GuardAuto",
    "TRITON_SnapshotLive",
    "TRITON_EquityDailySnapshot",
)

REQUIRED_SUCCESSFUL_DAYS = 5
TASK_SUCCESS_RESULT = 0

LOG_FIELDS = (
    "observation_date",
    "generated_at",
    "daily_refresh_success_today",
    "authoritative_path_verified",
    "authoritative_task_last_result",
    "authoritative_task_last_run_time",
    "execution_readiness_ready",
    "phase_195c_pass",
    "phase_195k_pass",
    "phase_195h_ready",
    "broker_ok",
    "live_blocked",
    "paper_authorized",
    "execute_trades_detected",
    "watchdog_ok",
    "continuous_loop_ok",
    "observation_day_success",
    "notes",
)

SUMMARY_FIELDS = ("metric", "value")

GATE_STATUSES = (
    "OBSERVATION_IN_PROGRESS",
    "READY_FOR_OPERATOR_ISOLATION_REVIEW",
    "NOT_READY_SCHEDULER_FAILURE",
    "NOT_READY_READINESS_FAILURE",
    "NOT_READY_BROKER_STATE",
    "NOT_READY_LIVE_BLOCK_FAILURE",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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


def _fetch_tasks(task_names: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    names_filter = ",".join(f"'{n}'" for n in task_names)
    ps = (
        f"$names = @({names_filter}); "
        "Get-ScheduledTask | Where-Object { $_.TaskName -in $names } | ForEach-Object { "
        "$info = Get-ScheduledTaskInfo $_; "
        "[PSCustomObject]@{ "
        "TaskName=$_.TaskName; State=$_.State; "
        "Execute=$_.Actions.Execute; Arguments=$_.Actions.Arguments; "
        "WorkingDirectory=$_.Actions.WorkingDirectory; "
        "LastRunTime=$info.LastRunTime; LastTaskResult=$info.LastTaskResult; "
        "NextRunTime=$info.NextRunTime "
        "} } | ConvertTo-Json -Compress"
    )
    out: Dict[str, Dict[str, Any]] = {}
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=90,
        )
        raw = (proc.stdout or "").strip()
        if not raw:
            return out
        data = json.loads(raw)
        rows = [data] if isinstance(data, dict) else data
        for row in rows:
            if isinstance(row, dict) and row.get("TaskName"):
                out[str(row["TaskName"])] = row
    except Exception:
        pass
    return out


def _parse_ps_date(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    s = str(value).strip()
    if s.startswith("/Date(") and s.endswith(")/"):
        try:
            ms = int(s[6:-2].split("+")[0].split("-")[0])
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        except (TypeError, ValueError):
            return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _task_result_ok(result: Any) -> bool:
    try:
        return int(result) == TASK_SUCCESS_RESULT
    except (TypeError, ValueError):
        return False


def _verify_authoritative_path(task: Dict[str, Any]) -> Tuple[bool, str]:
    args = str(task.get("Arguments") or "").lower()
    exe = str(task.get("Execute") or "").lower()
    blob = exe + " " + args
    if "run_daily_refresh.ps1" not in blob:
        return False, "task does not invoke run_daily_refresh.ps1"
    if "triton" not in blob.replace("\\", "/"):
        return False, "task path does not reference triton repo"
    refresh_ps1 = ROOT / "run_daily_refresh.ps1"
    pipeline_py = ROOT / "run_full_pipeline.py"
    if not refresh_ps1.is_file():
        return False, "run_daily_refresh.ps1 missing on disk"
    if not pipeline_py.is_file():
        return False, "run_full_pipeline.py missing on disk"
    text = refresh_ps1.read_text(encoding="utf-8", errors="replace").lower()
    if "run_full_pipeline" not in text:
        return False, "run_daily_refresh.ps1 does not call run_full_pipeline.py"
    if ".venv" not in text:
        return False, "run_daily_refresh.ps1 does not use .venv"
    return True, "TRITON_DailyRefresh -> run_daily_refresh.ps1 -> run_full_pipeline.py (.venv)"


def _loops_running() -> Tuple[bool, bool]:
    if sys.platform != "win32":
        return False, False
    try:
        from services.phase_195h_operational_readiness_freeze import _find_ps_processes

        wd, _ = _find_ps_processes("run_watchdog_loop.ps1")
        cl, _ = _find_ps_processes("run_continuous_loop.ps1")
        return wd, cl
    except Exception:
        return False, False


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


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0


def _load_observation_log() -> List[Dict[str, str]]:
    if not OBSERVATION_LOG_CSV.is_file():
        return []
    with OBSERVATION_LOG_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_observation_log(rows: List[Dict[str, str]]) -> None:
    by_date = {r["observation_date"]: r for r in rows if r.get("observation_date")}
    merged = sorted(by_date.values(), key=lambda r: r["observation_date"])
    _write_csv(OBSERVATION_LOG_CSV, merged, LOG_FIELDS)


def _count_consecutive_successful_days(log_rows: List[Dict[str, str]]) -> int:
    by_date = {r["observation_date"]: r for r in log_rows if r.get("observation_date")}
    if not by_date:
        return 0
    streak = 0
    date = datetime.now(timezone.utc).date()
    for _ in range(365):
        key = date.isoformat()
        row = by_date.get(key)
        if not row or row.get("observation_day_success", "").lower() != "true":
            break
        streak += 1
        date -= timedelta(days=1)
    return streak


def _readiness_status(doc: Dict[str, Any]) -> str:
    inner = doc.get("readiness") if isinstance(doc.get("readiness"), dict) else doc
    return str(inner.get("readiness_status") or doc.get("readiness_status") or "UNKNOWN")


def _classify_gate(
    *,
    live_permitted: bool,
    broker_ok: bool,
    scheduler_ok: bool,
    readiness_ok: bool,
    consecutive: int,
    required: int,
) -> str:
    if live_permitted:
        return "NOT_READY_LIVE_BLOCK_FAILURE"
    if not broker_ok:
        return "NOT_READY_BROKER_STATE"
    if not scheduler_ok:
        return "NOT_READY_SCHEDULER_FAILURE"
    if not readiness_ok:
        return "NOT_READY_READINESS_FAILURE"
    if consecutive >= required:
        return "READY_FOR_OPERATOR_ISOLATION_REVIEW"
    return "OBSERVATION_IN_PROGRESS"


def run_phase_195u_gate(*, verbose: bool = True) -> Dict[str, Any]:
    today = _utc_today()
    generated_at = _utc_now_iso()

    all_names = (AUTHORITATIVE_TASK,) + MONITORING_TASKS
    tasks = _fetch_tasks(all_names)
    auth_task = tasks.get(AUTHORITATIVE_TASK, {})

    path_ok, path_reason = (
        _verify_authoritative_path(auth_task)
        if auth_task
        else (
            False,
            "TRITON_DailyRefresh task not found",
        )
    )

    last_run_dt = _parse_ps_date(auth_task.get("LastRunTime"))
    last_result = auth_task.get("LastTaskResult")
    last_result_ok = _task_result_ok(last_result)
    last_run_today = bool(last_run_dt and last_run_dt.date() == datetime.now(timezone.utc).date())
    daily_refresh_success_today = last_run_today and last_result_ok and path_ok

    # Scheduler OK if path verified and last run succeeded within 48h (daily cadence tolerance)
    scheduler_recent_ok = False
    if path_ok and last_result_ok and last_run_dt:
        age_h = (datetime.now(timezone.utc) - last_run_dt).total_seconds() / 3600.0
        scheduler_recent_ok = age_h <= 48.0

    readiness_doc = _load_json(EXEC_READINESS)
    readiness_status = _readiness_status(readiness_doc)
    readiness_ready = readiness_status == "READY"

    doc_195c = _load_json(PHASE_195C)
    doc_195k = _load_json(PHASE_195K)
    doc_195h = _load_json(PHASE_195H)

    phase_195c_pass = bool(doc_195c.get("phase_pass"))
    phase_195k_pass = bool(doc_195k.get("phase_pass"))
    phase_195h_status = str(
        doc_195h.get("operational_freeze_status") or doc_195k.get("phase_195h_status") or "UNKNOWN"
    )
    phase_195h_ready = phase_195h_status in ("READY", "READY_WITH_REVIEW_ITEMS")

    open_orders = _count_csv_rows(OPEN_ORDERS)
    positions = _count_csv_rows(POSITIONS)
    broker_ok = open_orders == 0 and positions == 0

    live_permitted = bool(
        readiness_doc.get("live_execution_permitted")
        or (readiness_doc.get("readiness") or {}).get("live_execution_permitted")
        or doc_195k.get("live_execution_permitted")
    )
    paper_permitted = bool(
        readiness_doc.get("paper_execution_permitted")
        or (readiness_doc.get("readiness") or {}).get("paper_execution_permitted")
        or doc_195k.get("paper_execution_permitted")
    )

    execute_detected, execute_cmds = _find_python_execute_trades_execute()
    watchdog_running, continuous_running = _loops_running()

    watchdog_status = str(
        doc_195h.get("watchdog_status") or ("RUNNING" if watchdog_running else "STOPPED")
    )
    continuous_status = str(
        doc_195h.get("continuous_loop_status") or ("RUNNING" if continuous_running else "STOPPED")
    )
    watchdog_ok = watchdog_running or watchdog_status in ("OK", "READY", "RUNNING")
    continuous_ok = continuous_running or continuous_status == "RUNNING"

    readiness_ok = (
        readiness_ready
        and phase_195c_pass
        and phase_195k_pass
        and phase_195h_ready
        and not execute_detected
        and paper_permitted
        and not live_permitted
    )

    observation_day_success = (
        path_ok
        and scheduler_recent_ok
        and readiness_ok
        and broker_ok
        and watchdog_ok
        and continuous_ok
    )

    notes: List[str] = []
    if not path_ok:
        notes.append(path_reason)
    if not last_result_ok:
        notes.append(f"TRITON_DailyRefresh LastTaskResult={last_result}")
    if not last_run_today:
        notes.append("DailyRefresh has not completed successfully yet today (UTC)")
    if not scheduler_recent_ok and path_ok:
        notes.append("Last successful DailyRefresh run older than 48h")

    log_row = {
        "observation_date": today,
        "generated_at": generated_at,
        "daily_refresh_success_today": str(daily_refresh_success_today).lower(),
        "authoritative_path_verified": str(path_ok).lower(),
        "authoritative_task_last_result": str(last_result if last_result is not None else ""),
        "authoritative_task_last_run_time": str(auth_task.get("LastRunTime") or ""),
        "execution_readiness_ready": str(readiness_ready).lower(),
        "phase_195c_pass": str(phase_195c_pass).lower(),
        "phase_195k_pass": str(phase_195k_pass).lower(),
        "phase_195h_ready": str(phase_195h_ready).lower(),
        "broker_ok": str(broker_ok).lower(),
        "live_blocked": str(not live_permitted).lower(),
        "paper_authorized": str(paper_permitted).lower(),
        "execute_trades_detected": str(execute_detected).lower(),
        "watchdog_ok": str(watchdog_ok).lower(),
        "continuous_loop_ok": str(continuous_ok).lower(),
        "observation_day_success": str(observation_day_success).lower(),
        "notes": "; ".join(notes),
    }

    log_rows = _load_observation_log()
    log_rows = [r for r in log_rows if r.get("observation_date") != today]
    log_rows.append(log_row)
    _save_observation_log(log_rows)

    consecutive = _count_consecutive_successful_days(log_rows)

    gate_status = _classify_gate(
        live_permitted=live_permitted,
        broker_ok=broker_ok,
        scheduler_ok=path_ok and scheduler_recent_ok,
        readiness_ok=readiness_ok,
        consecutive=consecutive,
        required=REQUIRED_SUCCESSFUL_DAYS,
    )

    scheduler_isolation_allowed = gate_status == "READY_FOR_OPERATOR_ISOLATION_REVIEW"

    if scheduler_isolation_allowed:
        recommended = (
            "Observation gate satisfied — operator may review Phase 195V legacy scheduler "
            "isolation execution using legacy_scheduler_operator_commands.txt"
        )
    elif gate_status == "OBSERVATION_IN_PROGRESS":
        recommended = (
            f"Continue observation — {consecutive}/{REQUIRED_SUCCESSFUL_DAYS} consecutive "
            f"successful days; do not disable legacy schedulers yet"
        )
    elif gate_status == "NOT_READY_SCHEDULER_FAILURE":
        recommended = (
            "Investigate TRITON_DailyRefresh failure or stale last run before any isolation"
        )
    elif gate_status == "NOT_READY_READINESS_FAILURE":
        recommended = "Restore execution_readiness and phase validations before isolation"
    elif gate_status == "NOT_READY_BROKER_STATE":
        recommended = "Resolve broker open orders/positions before scheduler isolation"
    else:
        recommended = "Restore live execution block before scheduler isolation"

    monitoring_task_summary = {
        name: {
            "state": str((tasks.get(name) or {}).get("State") or "NotFound"),
            "last_run_time": str((tasks.get(name) or {}).get("LastRunTime") or ""),
            "last_task_result": str((tasks.get(name) or {}).get("LastTaskResult") or ""),
            "next_run_time": str((tasks.get(name) or {}).get("NextRunTime") or ""),
        }
        for name in MONITORING_TASKS
    }

    payload: Dict[str, Any] = {
        "phase": "195U",
        "generated_at": generated_at,
        "observation_date": today,
        "phase_pass": True,
        "audit_mode": "observation_only",
        "orders_placed": False,
        "scheduled_tasks_changed": False,
        "scheduler_commands_executed": False,
        "authoritative_task_name": AUTHORITATIVE_TASK,
        "authoritative_task_state": str(auth_task.get("State") or "NotFound"),
        "authoritative_task_last_run_time": str(auth_task.get("LastRunTime") or ""),
        "authoritative_task_last_result": last_result,
        "authoritative_task_last_result_ok": last_result_ok,
        "authoritative_task_next_run_time": str(auth_task.get("NextRunTime") or ""),
        "authoritative_path_verified": path_ok,
        "authoritative_path_reason": path_reason,
        "daily_refresh_success_today": daily_refresh_success_today,
        "scheduler_recent_success_within_48h": scheduler_recent_ok,
        "consecutive_successful_days": consecutive,
        "required_successful_days": REQUIRED_SUCCESSFUL_DAYS,
        "observation_gate_status": gate_status,
        "execution_readiness_status": readiness_status,
        "phase_195c_status": "PASS" if phase_195c_pass else "FAIL",
        "phase_195k_status": "PASS" if phase_195k_pass else "FAIL",
        "phase_195h_status": phase_195h_status,
        "broker_open_orders_count": open_orders,
        "broker_positions_count": positions,
        "live_execution_permitted": live_permitted,
        "paper_execution_permitted": paper_permitted,
        "execute_trades_execute_detected": execute_detected,
        "execute_trades_command_lines": execute_cmds,
        "watchdog_status": watchdog_status,
        "watchdog_process_running": watchdog_running,
        "continuous_loop_status": continuous_status,
        "continuous_process_running": continuous_running,
        "scheduler_isolation_allowed": scheduler_isolation_allowed,
        "recommended_next_action": recommended,
        "monitoring_tasks": monitoring_task_summary,
        "195t_plan_reference": str(PLAN_195T.relative_to(ROOT)).replace("\\", "/"),
        "next_isolation_phase": "Phase 195V — Legacy Scheduler Isolation Execution (operator-approved)",
        "safety_statement": (
            "Phase 195U is observation-only. No scheduled tasks were modified, no isolation "
            "commands executed, no orders placed, and no runtime live state restored."
        ),
        "observation_log_row_today": log_row,
    }

    summary_rows = [
        {"metric": "observation_date", "value": today},
        {"metric": "observation_gate_status", "value": gate_status},
        {"metric": "consecutive_successful_days", "value": str(consecutive)},
        {"metric": "required_successful_days", "value": str(REQUIRED_SUCCESSFUL_DAYS)},
        {
            "metric": "scheduler_isolation_allowed",
            "value": str(scheduler_isolation_allowed).lower(),
        },
        {"metric": "authoritative_path_verified", "value": str(path_ok).lower()},
        {
            "metric": "daily_refresh_success_today",
            "value": str(daily_refresh_success_today).lower(),
        },
        {"metric": "execution_readiness_status", "value": readiness_status},
        {"metric": "phase_195c_status", "value": payload["phase_195c_status"]},
        {"metric": "phase_195k_status", "value": payload["phase_195k_status"]},
        {"metric": "phase_195h_status", "value": phase_195h_status},
        {"metric": "broker_open_orders_count", "value": str(open_orders)},
        {"metric": "broker_positions_count", "value": str(positions)},
        {"metric": "live_execution_permitted", "value": str(live_permitted).lower()},
        {"metric": "paper_execution_permitted", "value": str(paper_permitted).lower()},
        {"metric": "execute_trades_execute_detected", "value": str(execute_detected).lower()},
        {"metric": "watchdog_status", "value": watchdog_status},
        {"metric": "continuous_loop_status", "value": continuous_status},
        {"metric": "recommended_next_action", "value": recommended},
    ]

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)

    if verbose:
        print(f"[195U] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[195U] gate={gate_status} consecutive={consecutive}/{REQUIRED_SUCCESSFUL_DAYS}")
        print(f"[195U] isolation_allowed={scheduler_isolation_allowed}")
        print(f"[195U] recommended: {recommended}")

    return payload


def main() -> None:
    run_phase_195u_gate(verbose=True)


if __name__ == "__main__":
    main()
