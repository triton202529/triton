"""
phase_195v_dailyrefresh_failure_root_cause_audit.py
Phase 195V — Authoritative DailyRefresh failure root cause audit.

Read-only diagnosis of TRITON_DailyRefresh LastTaskResult failures.
Does not modify scheduled tasks, place orders, or run isolation commands.
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
LOGS = ROOT / "logs"
CONFIG = ROOT / "config"

OUTPUT_JSON = RESULTS / "dailyrefresh_failure_root_cause_audit.json"
SUMMARY_CSV = RESULTS / "dailyrefresh_failure_root_cause_summary.csv"
LOG_MANIFEST_CSV = RESULTS / "dailyrefresh_failure_log_manifest.csv"

TASK_NAME = "TRITON_DailyRefresh"
OBSERVATION_GATE = RESULTS / "scheduler_stability_observation_gate.json"
EXEC_READINESS = RESULTS / "execution_readiness.json"
PHASE_195C = RESULTS / "phase_195c_evaluation_contract_validation.json"
PHASE_195K = RESULTS / "phase_195k_certification_readiness_alignment_validation.json"
PHASE_195H = RESULTS / "phase_195h_operational_readiness_freeze.json"
ASOF_CONTRACT = RESULTS / "asof_contract.json"
LIFECYCLE_DIAG = RESULTS / "lifecycle_readiness_diagnostics.json"
HEARTBEAT = RESULTS / "heartbeat.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
PAPER_CYCLE_CFG = CONFIG / "run_scheduled_paper_cycle.json"

ROOT_CAUSE_CLASSES = (
    "DAILYREFRESH_SUCCESS_BUT_TASK_RESULT_STALE",
    "TASK_ACTION_PATH_ERROR",
    "TASK_WORKING_DIRECTORY_ERROR",
    "POWERSHELL_SCRIPT_ERROR",
    "PYTHON_IMPORT_ERROR",
    "MISSING_FILE_ERROR",
    "STALE_LOCK_OR_HUNG_PIPELINE",
    "PIPELINE_STAGE_FAILURE",
    "READINESS_VALIDATION_FAILURE",
    "BROKER_SNAPSHOT_FAILURE",
    "MARKET_CLOSED_HANDLING_ERROR",
    "LOG_NOT_FOUND",
    "UNKNOWN_NEEDS_OPERATOR_REVIEW",
)

ERROR_MARKERS = (
    "FAILED",
    "❌",
    "Error",
    "Exception",
    "Traceback",
    "exit=1",
    "code=1",
    "ModuleNotFoundError",
    "ImportError",
    "FileNotFoundError",
    "STALE_LOCK",
    "hung",
    "No positions returned",
)

COMPLETION_MARKERS = (
    "TRITON Daily Refresh OK",
    "Daily Refresh OK",
    "Full pipeline completed",
    "EXIT: snapshot_positions code=0",
    "[END] ExitCode=0",
    'stage": "done"',
)

LOG_GLOBS = (
    "logs/**/*.log",
    "logs/*",
    "data/logs/**/*.log",
    "data/results/*daily*.log",
    "data/results/*pipeline*.log",
    "data/results/pipeline_scheduled.log",
    "data/results/phase195a_pipeline.log",
    "logs/daily_refresh_*.log",
    "logs/daily_*.log",
    "logs/run_daily.log",
)

MANIFEST_FIELDS = (
    "path",
    "modified_at",
    "size_bytes",
    "log_type",
    "matches_dailyrefresh_window",
)

SUMMARY_FIELDS = ("metric", "value")


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


def _fetch_task(name: str) -> Dict[str, Any]:
    ps = (
        f"Get-ScheduledTask -TaskName '{name}' -ErrorAction SilentlyContinue | ForEach-Object {{ "
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
            timeout=60,
        )
        raw = (proc.stdout or "").strip()
        if raw:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _parse_ps_date(value: Any) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip()
    if s.startswith("/Date("):
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


def _find_log_candidates() -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []

    def add(path: Path) -> None:
        if path.is_file():
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                out.append(path)

    for p in LOGS.glob("daily_refresh_*.log"):
        add(p)
    for p in LOGS.glob("daily_*.log"):
        add(p)
    if (LOGS / "run_daily.log").is_file():
        add(LOGS / "run_daily.log")
    for name in ("pipeline_scheduled.log", "phase195a_pipeline.log", "preprocess_run.log"):
        p = RESULTS / name
        if p.is_file():
            add(p)
    for p in RESULTS.glob("*daily*.log"):
        add(p)
    for p in RESULTS.glob("*pipeline*.log"):
        add(p)
    data_logs = ROOT / "data" / "logs"
    if data_logs.is_dir():
        for p in data_logs.rglob("*.log"):
            add(p)
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def _log_type(path: Path) -> str:
    name = path.name.lower()
    if "daily_refresh" in name:
        return "daily_refresh"
    if "pipeline_scheduled" in name:
        return "pipeline_scheduled"
    if "pipeline" in name:
        return "pipeline"
    if "daily" in name:
        return "daily_other"
    return "other"


def _read_tail(path: Path, lines: int = 80) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        parts = text.splitlines()
        return "\n".join(parts[-lines:])
    except OSError:
        return ""


def _markers_in_text(text: str, markers: Tuple[str, ...]) -> List[str]:
    return [m for m in markers if m in text]


def _match_dailyrefresh_log(logs: List[Path], last_run: Optional[datetime]) -> Optional[Path]:
    if not logs:
        return None
    daily = [p for p in logs if "daily_refresh" in p.name.lower()]
    if not daily:
        return None
    if last_run:
        for p in daily:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            if abs((mtime - last_run).total_seconds()) < 7200:
                return p
    return daily[0]


def _analyze_failure_from_log(text: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "failed_step": "",
        "failed_exit_code": "",
        "failure_message": "",
    }
    m = re.search(r"EXIT: (\S+) code=(\d+)", text)
    failures = re.findall(r"EXIT: (\S+) code=(\d+)", text)
    last_fail = next(((s, c) for s, c in reversed(failures) if c != "0"), None)
    if last_fail:
        info["failed_step"] = last_fail[0]
        info["failed_exit_code"] = last_fail[1]
    if "snapshot_positions" in text and "code=1" in text:
        info["failed_step"] = "snapshot_positions"
        info["failed_exit_code"] = "1"
        for line in text.splitlines():
            if "No positions returned" in line:
                info["failure_message"] = line.strip()
                break
    if "No positions returned (empty). Use --empty-ok" in text:
        info["failure_message"] = (
            "snapshot_positions exits 1 when broker returns zero positions without --empty-ok"
        )
    return info


def _classify_root_cause(
    *,
    task: Dict[str, Any],
    path_ok: bool,
    latest_log: Optional[Path],
    log_text: str,
    failure_info: Dict[str, Any],
    heartbeat: Dict[str, Any],
    last_run: Optional[datetime],
) -> Tuple[str, str, str]:
    last_result = task.get("LastTaskResult")
    try:
        result_int = int(last_result)
    except (TypeError, ValueError):
        result_int = -1

    if not latest_log or not latest_log.is_file():
        if result_int != 0:
            return (
                "LOG_NOT_FOUND",
                "Phase 195W — DailyRefresh log retention / task logging hardening",
                "Inspect Task Scheduler history for TRITON_DailyRefresh; confirm logs/ writable",
            )

    if failure_info.get("failed_step") == "snapshot_positions" and (
        "No positions returned" in log_text or "empty-ok" in log_text.lower()
    ):
        return (
            "BROKER_SNAPSHOT_FAILURE",
            "Phase 195W — run_daily_refresh.ps1 snapshot_positions --empty-ok hardening",
            "Add --empty-ok to snapshot_positions in run_daily_refresh.ps1 (0 positions is expected); "
            "re-run observation gate after next scheduled DailyRefresh",
        )

    if "run_full_pipeline code=0" in log_text.replace(" ", "") or (
        "EXIT: run_full_pipeline code=0" in log_text
        and failure_info.get("failed_step") not in ("run_full_pipeline", "")
    ):
        if result_int != 0:
            return (
                "PIPELINE_STAGE_FAILURE",
                "Phase 195W — post-pipeline DailyRefresh step hardening",
                f"Pipeline succeeded but step {failure_info.get('failed_step')} failed; see log tail",
            )

    hb_stage = heartbeat.get("stage")
    hb_status = heartbeat.get("status")
    hb_ts = _parse_ps_date(heartbeat.get("timestamp"))
    if (
        hb_stage == "done"
        and hb_status == "ok"
        and hb_ts
        and last_run
        and hb_ts > last_run
        and result_int != 0
    ):
        return (
            "DAILYREFRESH_SUCCESS_BUT_TASK_RESULT_STALE",
            "Phase 195W — reconcile task result vs post-run snapshot steps",
            "Artifacts fresh from alternate pipeline path; fix failing DailyRefresh tail step",
        )

    if not path_ok:
        return (
            "TASK_ACTION_PATH_ERROR",
            "Phase 195W — scheduled task action verification",
            "Verify TRITON_DailyRefresh action points to run_daily_refresh.ps1 under repo root",
        )

    if "ImportError" in log_text or "ModuleNotFoundError" in log_text:
        return (
            "PYTHON_IMPORT_ERROR",
            "Phase 195W — dependency / venv verification",
            "Verify .venv activation in run_daily_refresh.ps1",
        )

    if "FileNotFoundError" in log_text or "missing" in log_text.lower():
        return (
            "MISSING_FILE_ERROR",
            "Phase 195W — missing target remediation",
            "Restore missing file referenced in daily refresh log",
        )

    if "STALE_LOCK" in log_text or "pipeline_scheduled.lock" in log_text:
        return (
            "STALE_LOCK_OR_HUNG_PIPELINE",
            "Phase 195W — lock/orphan pipeline review",
            "Review pipeline_scheduled.lock and hung pipeline protection",
        )

    if "run_full_pipeline" in (failure_info.get("failed_step") or ""):
        return (
            "PIPELINE_STAGE_FAILURE",
            "Phase 195W — pipeline stage failure triage",
            "Inspect failed pipeline stage in daily_refresh log",
        )

    if result_int != 0:
        return (
            "UNKNOWN_NEEDS_OPERATOR_REVIEW",
            "Phase 195W — operator log review",
            f"LastTaskResult={last_result}; review {latest_log}",
        )

    return (
        "DAILYREFRESH_SUCCESS_BUT_TASK_RESULT_STALE",
        "Phase 195W — observation gate recalibration",
        "Task result shows success; re-run phase_195u observation gate",
    )


def _verify_authoritative_path(task: Dict[str, Any]) -> Tuple[bool, str]:
    args = str(task.get("Arguments") or "")
    exe = str(task.get("Execute") or "")
    blob = (exe + " " + args).lower()
    if "run_daily_refresh.ps1" not in blob:
        return False, "task does not reference run_daily_refresh.ps1"
    if not (ROOT / "run_daily_refresh.ps1").is_file():
        return False, "run_daily_refresh.ps1 missing"
    if not (ROOT / "run_full_pipeline.py").is_file():
        return False, "run_full_pipeline.py missing"
    return True, "TRITON_DailyRefresh -> run_daily_refresh.ps1 -> run_full_pipeline.py"


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


def run_phase_195v_audit(*, verbose: bool = True) -> Dict[str, Any]:
    generated_at = _utc_now_iso()
    task = _fetch_task(TASK_NAME)
    last_run_dt = _parse_ps_date(task.get("LastRunTime"))
    next_run_dt = _parse_ps_date(task.get("NextRunTime"))

    args = str(task.get("Arguments") or "")
    exe = str(task.get("Execute") or "")
    workdir = str(task.get("WorkingDirectory") or "")
    blob = (exe + " " + args).lower()

    path_ok, path_reason = _verify_authoritative_path(task) if task else (False, "task not found")
    uses_dot_venv = ".venv" in blob or ".venv" in _read_tail(ROOT / "run_daily_refresh.ps1", 20)
    uses_legacy_venv = bool(re.search(r"(?<![.\w])venv\\", blob))

    refresh_ps1 = _read_tail(ROOT / "run_daily_refresh.ps1", 999)
    snapshot_cmd = "snapshot_positions" in refresh_ps1
    has_empty_ok = "--empty-ok" in refresh_ps1

    log_candidates = _find_log_candidates()
    latest_log = _match_dailyrefresh_log(log_candidates, last_run_dt)
    log_text = _read_tail(latest_log, 120) if latest_log else ""
    failure_info = _analyze_failure_from_log(log_text)

    errors_found = _markers_in_text(log_text, ERROR_MARKERS)
    completions_found = _markers_in_text(log_text, COMPLETION_MARKERS)

    heartbeat = _load_json(HEARTBEAT)
    observation = _load_json(OBSERVATION_GATE)
    readiness = _load_json(EXEC_READINESS)
    inner = (
        readiness.get("readiness") if isinstance(readiness.get("readiness"), dict) else readiness
    )

    execute_detected, execute_cmds = _find_python_execute_trades_execute()
    paper_cfg = _load_json(PAPER_CYCLE_CFG)

    root_cause, fix_phase, operator_actions = _classify_root_cause(
        task=task,
        path_ok=path_ok,
        latest_log=latest_log,
        log_text=log_text,
        failure_info=failure_info,
        heartbeat=heartbeat,
        last_run=last_run_dt,
    )

    scheduler_isolation_allowed = bool(observation.get("scheduler_isolation_allowed", False))

    manifest_rows: List[Dict[str, str]] = []
    for p in log_candidates[:40]:
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
        matches = "false"
        if last_run_dt:
            age = abs(p.stat().st_mtime - last_run_dt.timestamp())
            matches = str(age < 7200).lower()
        manifest_rows.append(
            {
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "modified_at": mtime,
                "size_bytes": str(p.stat().st_size),
                "log_type": _log_type(p),
                "matches_dailyrefresh_window": matches,
            }
        )

    payload: Dict[str, Any] = {
        "phase": "195V",
        "generated_at": generated_at,
        "phase_pass": True,
        "audit_mode": "read_only",
        "orders_placed": False,
        "scheduled_tasks_changed": False,
        "task_name": TASK_NAME,
        "task_state": str(task.get("State") or "NotFound"),
        "last_run_time": str(task.get("LastRunTime") or ""),
        "last_run_time_utc": last_run_dt.isoformat() if last_run_dt else None,
        "last_task_result": task.get("LastTaskResult"),
        "next_run_time": str(task.get("NextRunTime") or ""),
        "next_run_time_utc": next_run_dt.isoformat() if next_run_dt else None,
        "action_path": exe,
        "arguments": args,
        "working_directory": workdir,
        "uses_dot_venv": uses_dot_venv,
        "uses_legacy_venv": uses_legacy_venv,
        "authoritative_path_verified": path_ok,
        "authoritative_path_reason": path_reason,
        "run_daily_refresh_ps1_exists": (ROOT / "run_daily_refresh.ps1").is_file(),
        "run_full_pipeline_py_exists": (ROOT / "run_full_pipeline.py").is_file(),
        "snapshot_positions_in_daily_refresh": snapshot_cmd,
        "snapshot_positions_has_empty_ok_flag": has_empty_ok,
        "latest_log_path": (
            str(latest_log.relative_to(ROOT)).replace("\\", "/") if latest_log else None
        ),
        "latest_log_modified_at": (
            datetime.fromtimestamp(latest_log.stat().st_mtime, tz=timezone.utc).isoformat()
            if latest_log
            else None
        ),
        "latest_log_tail": log_text[-4000:] if log_text else "",
        "error_markers_found": errors_found,
        "completion_markers_found": completions_found,
        "failed_step": failure_info.get("failed_step"),
        "failed_exit_code": failure_info.get("failed_exit_code"),
        "failure_message": failure_info.get("failure_message"),
        "heartbeat_stage": heartbeat.get("stage"),
        "heartbeat_status": heartbeat.get("status"),
        "heartbeat_timestamp": heartbeat.get("timestamp"),
        "asof_contract_ok": _load_json(ASOF_CONTRACT).get("ok"),
        "execution_readiness_status": inner.get("readiness_status", "UNKNOWN"),
        "phase_195c_status": "PASS" if _load_json(PHASE_195C).get("phase_pass") else "FAIL",
        "phase_195k_status": "PASS" if _load_json(PHASE_195K).get("phase_pass") else "FAIL",
        "phase_195h_status": _load_json(PHASE_195H).get("operational_freeze_status", "UNKNOWN"),
        "observation_gate_status": observation.get("observation_gate_status"),
        "broker_open_orders_count": _count_csv_rows(OPEN_ORDERS),
        "broker_positions_count": _count_csv_rows(POSITIONS),
        "live_execution_permitted": bool(inner.get("live_execution_permitted", False)),
        "paper_execution_permitted": bool(inner.get("paper_execution_permitted", False)),
        "execute_trades_execute_detected": execute_detected,
        "root_cause_classification": root_cause,
        "root_cause_summary": (
            f"TRITON_DailyRefresh LastTaskResult={task.get('LastTaskResult')} because "
            f"{failure_info.get('failed_step') or 'unknown step'} failed: "
            f"{failure_info.get('failure_message') or 'see latest_log_tail'}"
        ),
        "scheduler_isolation_allowed": scheduler_isolation_allowed,
        "recommended_fix_phase": fix_phase,
        "recommended_operator_actions": operator_actions,
        "diagnostics": {
            "pipeline_completed_in_log": "run_full_pipeline code=0" in log_text,
            "daily_refresh_completed_marker": "TRITON Daily Refresh OK" in log_text,
            "195u_consecutive_days": observation.get("consecutive_successful_days"),
            "execute_trades_command_lines": execute_cmds,
            "automated_entry_execution_enabled": bool(paper_cfg.get("run_execute_trades", False)),
        },
        "safety_statement": (
            "Phase 195V is read-only. No scheduled tasks modified, no orders placed, "
            "no isolation commands executed."
        ),
    }

    summary_rows = [
        {"metric": "root_cause_classification", "value": root_cause},
        {"metric": "last_task_result", "value": str(task.get("LastTaskResult", ""))},
        {"metric": "failed_step", "value": str(failure_info.get("failed_step") or "")},
        {"metric": "authoritative_path_verified", "value": str(path_ok).lower()},
        {"metric": "snapshot_positions_has_empty_ok_flag", "value": str(has_empty_ok).lower()},
        {"metric": "latest_log_path", "value": payload["latest_log_path"] or ""},
        {"metric": "heartbeat_stage", "value": str(heartbeat.get("stage") or "")},
        {"metric": "execution_readiness_status", "value": payload["execution_readiness_status"]},
        {
            "metric": "observation_gate_status",
            "value": str(observation.get("observation_gate_status") or ""),
        },
        {
            "metric": "scheduler_isolation_allowed",
            "value": str(scheduler_isolation_allowed).lower(),
        },
        {"metric": "recommended_fix_phase", "value": fix_phase},
        {"metric": "recommended_operator_actions", "value": operator_actions},
    ]

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)
    _write_csv(LOG_MANIFEST_CSV, manifest_rows, MANIFEST_FIELDS)

    if verbose:
        print(f"[195V] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[195V] root_cause={root_cause}")
        print(
            f"[195V] failed_step={failure_info.get('failed_step')} result={task.get('LastTaskResult')}"
        )
        print(f"[195V] log={payload['latest_log_path']}")
        print(f"[195V] recommended: {operator_actions}")

    return payload


def main() -> None:
    run_phase_195v_audit(verbose=True)


if __name__ == "__main__":
    main()
