"""
phase_195x_post_scheduler_cleanup_runtime_authority_certification.py
Phase 195X — Post-scheduler cleanup runtime authority certification.

Read-only certification that TRITON operates under loop-only authority after
Windows Scheduled Tasks were disabled. Does not place orders, modify scheduled
tasks, delete backups, or restore data/live while loops are running.
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

OUTPUT_JSON = RESULTS / "post_scheduler_cleanup_runtime_authority_certification.json"
SUMMARY_CSV = RESULTS / "post_scheduler_cleanup_runtime_authority_summary.csv"
PROCESSES_CSV = RESULTS / "post_scheduler_cleanup_runtime_processes.csv"
TASK_STATUS_CSV = RESULTS / "post_scheduler_cleanup_task_status.csv"

EXEC_READINESS = RESULTS / "execution_readiness.json"
PHASE_195C = RESULTS / "phase_195c_evaluation_contract_validation.json"
PHASE_195K = RESULTS / "phase_195k_certification_readiness_alignment_validation.json"
PHASE_195H = RESULTS / "phase_195h_operational_readiness_freeze.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
PAPER_CYCLE_CFG = CONFIG / "run_scheduled_paper_cycle.json"

TASK_BACKUP_PATH = Path(
    r"C:\Users\akimw\Documents\TRITON_TaskScheduler_Backups\task_scheduler_backup_20260618_193712"
)

RUNTIME_AUTHORITY_MODEL = (
    "loop_only: scripts/run_watchdog_loop.ps1 (1 min) + "
    "scripts/run_continuous_loop.ps1 (45 min) + Streamlit view_results.py"
)

EXPECTED_WATCHDOG_INTERVAL = 1
EXPECTED_CONTINUOUS_INTERVAL = 45

STATUS_CERTIFIED = "RUNTIME_AUTHORITY_CERTIFIED"
STATUS_LOOP_MISSING = "NOT_READY_LOOP_MISSING"
STATUS_TASK_ACTIVE = "NOT_READY_SCHEDULED_TASK_ACTIVE"
STATUS_BACKUP_MISSING = "NOT_READY_BACKUP_MISSING"
STATUS_READINESS = "NOT_READY_EXECUTION_READINESS"
STATUS_CERT_FAILURE = "NOT_READY_CERTIFICATION_FAILURE"
STATUS_BROKER = "NOT_READY_BROKER_STATE"
STATUS_LIVE_BLOCK = "NOT_READY_LIVE_BLOCK_FAILURE"
STATUS_GIT_DIRTY = "NOT_READY_GIT_DIRTY"
STATUS_REVIEW = "NEEDS_OPERATOR_REVIEW"

SUMMARY_FIELDS = ("metric", "value")
PROCESS_FIELDS = (
    "process_type",
    "running",
    "process_id",
    "interval_minutes",
    "command_line",
)
TASK_FIELDS = (
    "task_name",
    "state",
    "disabled",
    "running",
    "execute",
    "arguments",
    "last_run_time",
    "last_task_result",
    "next_run_time",
)

SOURCE_PATH_PREFIXES = ("services/", "scripts/", ".gitignore", ".gitattributes")


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


def _git_output(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        return (proc.stdout or "").strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def _parse_porcelain() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in _git_output("status", "--porcelain", "-uall").splitlines():
        if len(line) < 4:
            continue
        x, y = line[0], line[1]
        path = line[3:].strip().strip('"')
        rel = path.replace("\\", "/")
        rows.append({"path": rel, "index": x, "worktree": y})
    return rows


def _is_generated_artifact(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/")
    if norm.startswith("data/live/"):
        return True
    if norm.startswith("data/processed/"):
        return True
    if norm.startswith("data/results/"):
        return True
    return False


def _is_source_path(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/")
    return (
        any(norm.startswith(p) for p in SOURCE_PATH_PREFIXES)
        or norm.endswith((".py", ".ps1", ".json", ".md"))
        and norm.startswith(("services/", "scripts/", "config/"))
    )


from services.phase_195h_operational_readiness_freeze import (  # noqa: E402
    _find_ps_processes,
)


def _fetch_process_command_lines(
    script_fragment: str,
) -> List[Dict[str, Any]]:
    if sys.platform != "win32":
        return []
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name = 'powershell.exe' OR Name = 'pwsh.exe'\" "
        f"| Where-Object {{ $_.CommandLine -like '*{script_fragment}*' }} "
        "| Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=45,
        )
        raw = (proc.stdout or "").strip()
        if not raw:
            return []
        data = json.loads(raw)
        rows = [data] if isinstance(data, dict) else data
        out: List[Dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                out.append(row)
        return out
    except Exception:
        return []


def _parse_interval_minutes(command_line: str, default: int) -> int:
    m = re.search(r"-IntervalMinutes\s+(\d+)", command_line, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return default


def _find_streamlit_view_results() -> Tuple[int, List[Dict[str, Any]]]:
    if sys.platform != "win32":
        return 0, []
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" "
        "| Where-Object { "
        "($_.CommandLine -like '*streamlit*' -and $_.CommandLine -like '*view_results*') "
        "-or ($_.CommandLine -like '*view_results.py*') "
        "} | Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=45,
        )
        raw = (proc.stdout or "").strip()
        if not raw:
            return 0, []
        data = json.loads(raw)
        rows = [data] if isinstance(data, dict) else data
        procs = [r for r in rows if isinstance(r, dict)]
        return len(procs), procs
    except Exception:
        return 0, []


def _fetch_triton_scheduled_tasks() -> List[Dict[str, Any]]:
    if sys.platform != "win32":
        return []
    ps = (
        "Get-ScheduledTask | Where-Object { "
        "$_.TaskName -match 'TRITON|Triton' -or "
        "$_.Actions.Execute -match 'triton' -or $_.Actions.Arguments -match 'triton' "
        "} | ForEach-Object { "
        "$info = Get-ScheduledTaskInfo $_; "
        "[PSCustomObject]@{ "
        "TaskName=$_.TaskName; State=$_.State; "
        "Execute=$_.Actions.Execute; Arguments=$_.Actions.Arguments; "
        "WorkingDirectory=$_.Actions.WorkingDirectory; "
        "LastRunTime=$info.LastRunTime; LastTaskResult=$info.LastTaskResult; "
        "NextRunTime=$info.NextRunTime "
        "} } | ConvertTo-Json -Compress"
    )
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
            return []
        data = json.loads(raw)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
    except Exception:
        pass
    return []


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


def _normalize_task_state(state: Any) -> Tuple[str, bool, bool]:
    """Return (label, is_disabled, is_running) handling enum int or string from JSON."""
    if state is None or state == "":
        return "unknown", False, False
    s = str(state).strip()
    sl = s.lower()
    if sl in ("disabled", "1") or state == 1:
        return "Disabled", True, False
    if sl in ("running", "4") or state == 4:
        return "Running", False, True
    if sl in ("ready", "3") or state == 3:
        return "Ready", False, False
    if sl in ("queued", "2") or state == 2:
        return "Queued", False, False
    return s, False, False


def _phase_status(path: Path, *, pass_key: str = "phase_pass") -> str:
    doc = _load_json(path)
    if not doc:
        return "UNKNOWN"
    if pass_key in doc:
        return "PASS" if doc.get(pass_key) else "FAIL"
    status = doc.get("operational_freeze_status") or doc.get("readiness_status")
    if status:
        return str(status)
    return "UNKNOWN"


def _audit_git_state() -> Tuple[bool, List[str], List[str], List[str]]:
    porcelain = _parse_porcelain()
    staged_generated: List[str] = []
    data_live_issues: List[str] = []
    dirty_non_source: List[str] = []

    tracked_live = {
        line.strip().replace("\\", "/")
        for line in _git_output("ls-files", "--", "data/live/").splitlines()
        if line.strip()
    }

    for row in porcelain:
        rel = row["path"]
        x, y = row["index"], row["worktree"]
        is_staged = x not in (" ", "?")
        is_modified = y == "M" or x == "M"

        if rel.startswith("data/live/"):
            if is_staged or (rel in tracked_live and is_modified):
                data_live_issues.append(rel)
            if is_staged:
                staged_generated.append(rel)

        if _is_generated_artifact(rel) and is_staged:
            if rel not in staged_generated:
                staged_generated.append(rel)

        if not _is_source_path(rel) and (is_staged or is_modified or (x == "?" and y == "?")):
            if rel.startswith(("data/", "logs/")):
                dirty_non_source.append(rel)

    git_clean = not staged_generated and not data_live_issues
    return git_clean, staged_generated, data_live_issues, dirty_non_source


def _classify_runtime_authority(
    *,
    watchdog_running: bool,
    continuous_running: bool,
    watchdog_interval: Optional[int],
    continuous_interval: Optional[int],
    streamlit_count: int,
    tasks: List[Dict[str, Any]],
    backup_exists: bool,
    readiness_ok: bool,
    cert_ok: bool,
    broker_ok: bool,
    live_block_ok: bool,
    paper_ok: bool,
    execute_detected: bool,
    git_clean: bool,
    data_live_issues: List[str],
    tasks_exist: bool,
) -> Tuple[str, str]:
    if not watchdog_running or not continuous_running:
        return (
            STATUS_LOOP_MISSING,
            "Start missing loop process(es): watchdog (1 min) and/or continuous (45 min)",
        )
    if watchdog_interval != EXPECTED_WATCHDOG_INTERVAL:
        return (
            STATUS_LOOP_MISSING,
            f"Watchdog interval is {watchdog_interval}; expected {EXPECTED_WATCHDOG_INTERVAL} minute",
        )
    if continuous_interval != EXPECTED_CONTINUOUS_INTERVAL:
        return (
            STATUS_LOOP_MISSING,
            f"Continuous loop interval is {continuous_interval}; expected {EXPECTED_CONTINUOUS_INTERVAL} minutes",
        )
    if streamlit_count < 1:
        return (
            STATUS_LOOP_MISSING,
            "Start Streamlit Command Center: streamlit run view_results.py",
        )
    if not backup_exists:
        return (
            STATUS_BACKUP_MISSING,
            f"Task backup folder missing: {TASK_BACKUP_PATH}",
        )
    if not tasks_exist:
        return (
            STATUS_REVIEW,
            "No TRITON/Triton scheduled tasks found — verify tasks were disabled, not deleted from registry",
        )

    not_disabled = [
        str(t.get("TaskName")) for t in tasks if not _normalize_task_state(t.get("State"))[1]
    ]
    running_tasks = [
        str(t.get("TaskName")) for t in tasks if _normalize_task_state(t.get("State"))[2]
    ]
    if running_tasks or not_disabled:
        names = running_tasks or not_disabled
        return (
            STATUS_TASK_ACTIVE,
            f"Disable or stop active scheduled task(s): {', '.join(names)}",
        )
    if not readiness_ok:
        return STATUS_READINESS, "execution_readiness is not READY"
    if not cert_ok:
        return STATUS_CERT_FAILURE, "195C and/or 195K certification checks failed"
    if not broker_ok:
        return STATUS_BROKER, "Broker must be flat: 0 open orders and 0 positions"
    if not live_block_ok or not paper_ok:
        return (
            STATUS_LIVE_BLOCK,
            "live_execution_permitted must be false and paper_execution_permitted must be true",
        )
    if execute_detected:
        return STATUS_LIVE_BLOCK, "execute_trades --execute process detected — stop immediately"
    if data_live_issues or not git_clean:
        return (
            STATUS_GIT_DIRTY,
            "Unstage generated/runtime artifacts; do not commit or restore data/live while loops run",
        )
    return (
        STATUS_CERTIFIED,
        "Runtime authority certified under loop-only model; monitor loops and dashboard",
    )


def run_phase_195x_certification(*, verbose: bool = True) -> Dict[str, Any]:
    watchdog_running, watchdog_pids = _find_ps_processes("run_watchdog_loop.ps1")
    continuous_running, continuous_pids = _find_ps_processes("run_continuous_loop.ps1")

    wd_procs = _fetch_process_command_lines("run_watchdog_loop.ps1")
    cl_procs = _fetch_process_command_lines("run_continuous_loop.ps1")
    streamlit_count, streamlit_procs = _find_streamlit_view_results()
    execute_detected, execute_cmds = _find_python_execute_trades_execute()

    watchdog_interval: Optional[int] = None
    continuous_interval: Optional[int] = None
    watchdog_cmd = ""
    continuous_cmd = ""

    if wd_procs:
        watchdog_cmd = str(wd_procs[0].get("CommandLine") or "")
        watchdog_interval = _parse_interval_minutes(watchdog_cmd, EXPECTED_WATCHDOG_INTERVAL)
    elif watchdog_running:
        watchdog_interval = EXPECTED_WATCHDOG_INTERVAL

    if cl_procs:
        continuous_cmd = str(cl_procs[0].get("CommandLine") or "")
        continuous_interval = _parse_interval_minutes(continuous_cmd, EXPECTED_CONTINUOUS_INTERVAL)
    elif continuous_running:
        continuous_interval = EXPECTED_CONTINUOUS_INTERVAL

    tasks = _fetch_triton_scheduled_tasks()
    task_rows: List[Dict[str, str]] = []
    disabled_count = 0
    running_count = 0
    not_disabled: List[str] = []

    for task in tasks:
        name = str(task.get("TaskName") or "")
        state_raw = task.get("State")
        state_label, is_disabled, is_running = _normalize_task_state(state_raw)
        if is_disabled:
            disabled_count += 1
        if is_running:
            running_count += 1
        if not is_disabled:
            not_disabled.append(name)
        task_rows.append(
            {
                "task_name": name,
                "state": state_label,
                "disabled": str(is_disabled).lower(),
                "running": str(is_running).lower(),
                "execute": str(task.get("Execute") or ""),
                "arguments": str(task.get("Arguments") or ""),
                "last_run_time": str(task.get("LastRunTime") or ""),
                "last_task_result": str(task.get("LastTaskResult") or ""),
                "next_run_time": str(task.get("NextRunTime") or ""),
            }
        )

    backup_exists = TASK_BACKUP_PATH.is_dir()
    backup_file_count = 0
    if backup_exists:
        backup_file_count = sum(1 for p in TASK_BACKUP_PATH.rglob("*") if p.is_file())

    readiness = _load_json(EXEC_READINESS)
    paper_cfg = _load_json(PAPER_CYCLE_CFG)
    phase_195c = _load_json(PHASE_195C)
    phase_195k = _load_json(PHASE_195K)
    phase_195h = _load_json(PHASE_195H)

    execution_readiness_status = str(readiness.get("readiness_status") or "UNKNOWN")
    phase_195c_status = _phase_status(PHASE_195C)
    phase_195k_status = (
        "PASS" if phase_195k.get("phase_pass") else ("FAIL" if phase_195k else "UNKNOWN")
    )
    phase_195h_status = str(
        phase_195h.get("operational_freeze_status")
        or _phase_status(PHASE_195H, pass_key="phase_pass")
    )

    broker_open = _count_csv_rows(OPEN_ORDERS)
    broker_positions = _count_csv_rows(POSITIONS)
    live_permitted = bool(readiness.get("live_execution_permitted", True))
    paper_permitted = bool(readiness.get("paper_execution_permitted", False))
    automated_entry = bool(paper_cfg.get("run_execute_trades", False))

    git_clean, staged_generated, data_live_issues, dirty_non_source = _audit_git_state()
    tracked_live = [
        line.strip().replace("\\", "/")
        for line in _git_output("ls-files", "--", "data/live/").splitlines()
        if line.strip()
    ]
    data_live_tracked_or_staged = sorted(set(data_live_issues + tracked_live))

    readiness_ok = execution_readiness_status == "READY"
    cert_ok = phase_195c_status == "PASS" and phase_195k_status == "PASS"
    broker_ok = broker_open == 0 and broker_positions == 0
    live_block_ok = live_permitted is False
    paper_ok = paper_permitted is True
    h_ok = phase_195h_status == "READY"

    runtime_status, recommended = _classify_runtime_authority(
        watchdog_running=watchdog_running,
        continuous_running=continuous_running,
        watchdog_interval=watchdog_interval,
        continuous_interval=continuous_interval,
        streamlit_count=streamlit_count,
        tasks=tasks,
        backup_exists=backup_exists,
        readiness_ok=readiness_ok and h_ok,
        cert_ok=cert_ok,
        broker_ok=broker_ok,
        live_block_ok=live_block_ok,
        paper_ok=paper_ok,
        execute_detected=execute_detected,
        git_clean=git_clean,
        data_live_issues=data_live_tracked_or_staged,
        tasks_exist=len(tasks) > 0,
    )

    phase_pass = runtime_status == STATUS_CERTIFIED

    process_rows: List[Dict[str, str]] = [
        {
            "process_type": "watchdog_loop",
            "running": str(watchdog_running).lower(),
            "process_id": str(watchdog_pids[0]) if watchdog_pids else "",
            "interval_minutes": str(watchdog_interval or ""),
            "command_line": watchdog_cmd[:500],
        },
        {
            "process_type": "continuous_loop",
            "running": str(continuous_running).lower(),
            "process_id": str(continuous_pids[0]) if continuous_pids else "",
            "interval_minutes": str(continuous_interval or ""),
            "command_line": continuous_cmd[:500],
        },
    ]
    for proc in streamlit_procs:
        process_rows.append(
            {
                "process_type": "streamlit_view_results",
                "running": "true",
                "process_id": str(proc.get("ProcessId") or ""),
                "interval_minutes": "",
                "command_line": str(proc.get("CommandLine") or "")[:500],
            }
        )
    if streamlit_count == 0:
        process_rows.append(
            {
                "process_type": "streamlit_view_results",
                "running": "false",
                "process_id": "",
                "interval_minutes": "",
                "command_line": "",
            }
        )

    payload: Dict[str, Any] = {
        "phase": "195X",
        "generated_at": _utc_now_iso(),
        "phase_pass": phase_pass,
        "audit_mode": "read_only",
        "orders_placed": False,
        "scheduled_tasks_changed": False,
        "runtime_authority_model": RUNTIME_AUTHORITY_MODEL,
        "watchdog_process_running": watchdog_running,
        "watchdog_process_id": watchdog_pids[0] if watchdog_pids else None,
        "watchdog_interval_minutes": watchdog_interval,
        "continuous_process_running": continuous_running,
        "continuous_process_id": continuous_pids[0] if continuous_pids else None,
        "continuous_interval_minutes": continuous_interval,
        "streamlit_process_count": streamlit_count,
        "scheduled_tasks_total": len(tasks),
        "scheduled_tasks_disabled_count": disabled_count,
        "scheduled_tasks_running_count": running_count,
        "scheduled_tasks_not_disabled": not_disabled,
        "task_backup_path": str(TASK_BACKUP_PATH),
        "task_backup_exists": backup_exists,
        "task_backup_file_count": backup_file_count,
        "execution_readiness_status": execution_readiness_status,
        "phase_195c_status": phase_195c_status,
        "phase_195k_status": phase_195k_status,
        "phase_195h_status": phase_195h_status,
        "broker_open_orders_count": broker_open,
        "broker_positions_count": broker_positions,
        "live_execution_permitted": live_permitted,
        "paper_execution_permitted": paper_permitted,
        "execute_trades_execute_detected": execute_detected,
        "execute_trades_command_lines": execute_cmds,
        "automated_entry_execution_enabled": automated_entry,
        "git_status_clean": git_clean,
        "staged_generated_artifacts": staged_generated,
        "data_live_paths_tracked_or_staged": data_live_tracked_or_staged,
        "dirty_runtime_artifacts_unstaged": dirty_non_source,
        "runtime_authority_status": runtime_status,
        "recommended_next_action": recommended,
        "certification_checks": {
            "watchdog_loop_running": watchdog_running,
            "continuous_loop_running": continuous_running,
            "watchdog_interval_1_min": watchdog_interval == EXPECTED_WATCHDOG_INTERVAL,
            "continuous_interval_45_min": continuous_interval == EXPECTED_CONTINUOUS_INTERVAL,
            "streamlit_view_results_running": streamlit_count >= 1,
            "all_scheduled_tasks_disabled": len(not_disabled) == 0 and len(tasks) > 0,
            "no_scheduled_tasks_running": running_count == 0,
            "disabled_tasks_not_deleted": len(tasks) > 0,
            "task_backup_exists": backup_exists,
            "execution_readiness_ready": readiness_ok,
            "phase_195c_pass": phase_195c_status == "PASS",
            "phase_195k_pass": phase_195k_status == "PASS",
            "phase_195h_ready": h_ok,
            "broker_flat": broker_ok,
            "live_blocked": live_block_ok,
            "paper_authorized": paper_ok,
            "no_execute_trades_execute": not execute_detected,
            "git_no_staged_generated_artifacts": not staged_generated,
            "data_live_not_restored_or_staged": not data_live_tracked_or_staged,
        },
        "safety_statement": (
            "Phase 195X is read-only. No scheduled tasks modified or deleted, "
            "no orders placed, no data/live restore while loops run."
        ),
    }

    summary_rows: List[Dict[str, str]] = [
        {"metric": "runtime_authority_status", "value": runtime_status},
        {"metric": "phase_pass", "value": str(phase_pass).lower()},
        {"metric": "watchdog_process_running", "value": str(watchdog_running).lower()},
        {"metric": "watchdog_interval_minutes", "value": str(watchdog_interval or "")},
        {"metric": "continuous_process_running", "value": str(continuous_running).lower()},
        {"metric": "continuous_interval_minutes", "value": str(continuous_interval or "")},
        {"metric": "streamlit_process_count", "value": str(streamlit_count)},
        {"metric": "scheduled_tasks_total", "value": str(len(tasks))},
        {"metric": "scheduled_tasks_disabled_count", "value": str(disabled_count)},
        {"metric": "scheduled_tasks_running_count", "value": str(running_count)},
        {"metric": "task_backup_exists", "value": str(backup_exists).lower()},
        {"metric": "execution_readiness_status", "value": execution_readiness_status},
        {"metric": "phase_195c_status", "value": phase_195c_status},
        {"metric": "phase_195k_status", "value": phase_195k_status},
        {"metric": "phase_195h_status", "value": phase_195h_status},
        {"metric": "broker_open_orders_count", "value": str(broker_open)},
        {"metric": "broker_positions_count", "value": str(broker_positions)},
        {"metric": "live_execution_permitted", "value": str(live_permitted).lower()},
        {"metric": "paper_execution_permitted", "value": str(paper_permitted).lower()},
        {"metric": "execute_trades_execute_detected", "value": str(execute_detected).lower()},
        {"metric": "git_status_clean", "value": str(git_clean).lower()},
        {"metric": "recommended_next_action", "value": recommended},
    ]

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)
    _write_csv(PROCESSES_CSV, process_rows, PROCESS_FIELDS)
    _write_csv(TASK_STATUS_CSV, task_rows, TASK_FIELDS)

    if verbose:
        print(f"[195X] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[195X] runtime_authority_status={runtime_status}")
        print(f"[195X] phase_pass={phase_pass}")
        print(
            f"[195X] watchdog={watchdog_running} pid={watchdog_pids[:1]} interval={watchdog_interval}m"
        )
        print(
            f"[195X] continuous={continuous_running} pid={continuous_pids[:1]} "
            f"interval={continuous_interval}m"
        )
        print(f"[195X] streamlit_view_results count={streamlit_count}")
        print(
            f"[195X] scheduled_tasks total={len(tasks)} disabled={disabled_count} "
            f"running={running_count} not_disabled={not_disabled}"
        )
        print(f"[195X] task_backup_exists={backup_exists}")
        if not phase_pass:
            print(f"[195X] recommended: {recommended}")

    return payload


def main() -> None:
    result = run_phase_195x_certification(verbose=True)
    sys.exit(0 if result.get("phase_pass") else 1)


if __name__ == "__main__":
    main()
