"""
phase_195t_legacy_scheduler_isolation_dry_run_plan.py
Phase 195T — Legacy scheduler isolation dry-run plan.

Read-only inspection of Windows Scheduled Tasks and runtime entry points.
Does not modify tasks, place orders, or execute isolation commands.
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
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
CONFIG = ROOT / "config"

OUTPUT_JSON = RESULTS / "legacy_scheduler_isolation_dry_run_plan.json"
TASK_PLAN_CSV = RESULTS / "legacy_scheduler_task_plan.csv"
OPERATOR_COMMANDS_TXT = RESULTS / "legacy_scheduler_operator_commands.txt"

PLAN_195R = RESULTS / "runtime_untrack_scheduler_isolation_plan.json"
ENTRYPOINT_195N = RESULTS / "runtime_entrypoint_consolidation_audit.json"
EXEC_READINESS = RESULTS / "execution_readiness.json"
PHASE_195H_JSON = RESULTS / "phase_195h_operational_readiness_freeze.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
PAPER_CYCLE_CFG = CONFIG / "run_scheduled_paper_cycle.json"

from services.phase_195n_runtime_entrypoint_consolidation_audit import (  # noqa: E402
    _fetch_scheduled_tasks,
    _find_missing_references,
    _match_patterns,
    _read_text,
)

CLASSIFICATIONS = (
    "KEEP_ACTIVE_AUTHORITATIVE",
    "KEEP_ACTIVE_MONITORING",
    "KEEP_ACTIVE_SNAPSHOT",
    "KEEP_OPERATOR_ONLY",
    "DISABLE_LATER_LEGACY_DUPLICATE",
    "REDIRECT_LATER_TO_AUTHORITATIVE",
    "REVIEW_LATER_BYPASS_RISK",
    "DO_NOT_TOUCH_YET",
    "BROKEN_MISSING_TARGET",
    "UNKNOWN_NEEDS_REVIEW",
)

TASK_PLAN_FIELDS = (
    "name",
    "type",
    "current_state",
    "last_run_time",
    "last_task_result",
    "next_run_time",
    "action_path",
    "arguments",
    "working_directory",
    "uses_dot_venv",
    "uses_legacy_venv",
    "references_missing_target",
    "calls_run_full_pipeline",
    "calls_pipeline_real_data",
    "calls_execute_trades",
    "calls_execute_trades_execute",
    "calls_manage_positions",
    "can_submit_orders",
    "live_block_respected",
    "paper_only",
    "classification",
    "recommended_later_action",
    "proposed_operator_command",
    "rollback_command",
    "safety_reason",
)

STATIC_ENTRYPOINTS: Tuple[Tuple[str, str], ...] = (
    ("run_full_pipeline.py", "python_orchestrator"),
    ("run_daily_refresh.ps1", "powershell_scheduler"),
    ("run_pipeline_scheduled.ps1", "powershell_scheduler"),
    ("run_guard_auto.ps1", "powershell_scheduler"),
    ("run_snapshot_live.ps1", "powershell_scheduler"),
    ("run-fetch.ps1", "powershell_scheduler"),
    ("run_daily.cmd", "batch_scheduler"),
    ("daily_refresh.bat", "batch_scheduler"),
    ("Makefile", "make_target"),
    ("scripts/run_watchdog_loop.ps1", "powershell_loop"),
    ("scripts/run_continuous_loop.ps1", "powershell_loop"),
    ("scripts/run_market_open.ps1", "powershell_scheduler"),
    ("scripts/run_all.py", "python_orchestrator"),
    ("pipeline_real_data.py", "python_legacy_pipeline"),
    ("services/execution_readiness.py", "python_module"),
    ("services/governance_execution_readiness.py", "python_module"),
    ("services/execute_trades.py", "python_execution"),
    ("services/manage_positions.py", "python_execution"),
    ("services/trade_executor.py", "python_legacy"),
    ("services/portfolio_manager.py", "python_legacy"),
    ("alpaca_client.py", "python_legacy_execution"),
    ("scripts/auto_trade.py", "python_legacy_execution"),
    ("scripts/auto_exit.py", "python_legacy_execution"),
    ("scripts/auto_execute_signals.py", "python_legacy_execution"),
    ("scripts/execute_trade.py", "python_legacy_execution"),
)

KEEP_ACTIVE_NAMES = frozenset(
    {
        "TRITON_DailyRefresh",
        "run_daily_refresh.ps1",
        "TRITON_MarketOpen",
        "TRITON_GuardAuto",
        "TRITON_SnapshotLive",
        "TRITON_EquityDailySnapshot",
        "scripts/run_watchdog_loop.ps1",
        "scripts/run_continuous_loop.ps1",
        "run_full_pipeline.py",
        "services/execution_readiness.py",
        "services/governance_execution_readiness.py",
        "services/execute_trades.py",
        "scripts/run_market_open.ps1",
    }
)

DISABLE_OR_REDIRECT_NAMES = frozenset(
    {
        "daily_refresh.bat",
        "Triton Daily Refresh",
        "run-fetch.ps1",
        "TritonFetchDaily",
        "Makefile",
        "scripts/run_all.py",
        "pipeline_real_data.py",
        "Triton Daily",
        "Triton Daily (ALL personas)",
        "Triton Daily (simons)",
        "run_daily.cmd",
    }
)

REVIEW_BYPASS_NAMES = frozenset(
    {
        "Triton Attach Brackets",
        "scripts/auto_trade.py",
        "scripts/auto_exit.py",
        "scripts/auto_execute_signals.py",
        "scripts/execute_trade.py",
        "services/trade_executor.py",
        "services/portfolio_manager.py",
        "alpaca_client.py",
    }
)

DO_NOT_TOUCH_NAMES = frozenset({"run_pipeline_scheduled.ps1"})


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


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _fetch_scheduled_tasks_extended() -> List[Dict[str, Any]]:
    ps = (
        "Get-ScheduledTask | Where-Object { "
        "$_.Actions.Execute -match 'triton' -or $_.Actions.Arguments -match 'triton' "
        "} | ForEach-Object { "
        "$info = Get-ScheduledTaskInfo $_; "
        "$tr = $_.Triggers | Select-Object -First 1; "
        "[PSCustomObject]@{ "
        "TaskName=$_.TaskName; State=$_.State; "
        "Execute=$_.Actions.Execute; Arguments=$_.Actions.Arguments; "
        "WorkingDirectory=$_.Actions.WorkingDirectory; "
        "LastRunTime=$info.LastRunTime; LastTaskResult=$info.LastTaskResult; "
        "NextRunTime=$info.NextRunTime; "
        "StartBoundary=$tr.StartBoundary; DaysInterval=$tr.DaysInterval "
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
            return _fetch_scheduled_tasks()
        data = json.loads(raw)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return _fetch_scheduled_tasks()


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


def _normalize_name(name: str) -> str:
    return name.replace("\\", "/").strip()


def _calls_pipeline_real_data(text: str) -> bool:
    return "pipeline_real_data" in text.lower()


def _can_submit_orders(flags: Dict[str, bool], text: str, rel: str) -> bool:
    if rel == "alpaca_client.py":
        return "submit_order" in text.lower()
    return bool(
        flags.get("calls_broker_submit_function")
        or flags.get("calls_execute_trades_execute_flag")
        or rel in {"services/execute_trades.py", "services/manage_positions.py"}
        or rel.startswith("scripts/auto_")
        or rel == "scripts/execute_trade.py"
    )


def _classify_195t(
    name: str,
    etype: str,
    *,
    exists: bool,
    missing: str,
    flags: Dict[str, bool],
    text: str,
) -> Tuple[str, str, str, str, str]:
    norm = _normalize_name(name)
    base = Path(norm).name

    if not exists and etype != "windows_scheduled_task":
        return (
            "BROKEN_MISSING_TARGET",
            "Remove or restore missing entrypoint; do not schedule",
            "# DO NOT RUN IN PHASE 195T",
            "# DO NOT RUN IN PHASE 195T",
            "Entrypoint file missing on disk",
        )

    if missing:
        return (
            "BROKEN_MISSING_TARGET",
            f"Redirect to run_full_pipeline.py / run_daily_refresh.ps1; fix missing: {missing}",
            "# DO NOT RUN IN PHASE 195T",
            "# DO NOT RUN IN PHASE 195T",
            f"References missing targets: {missing}",
        )

    if norm in DO_NOT_TOUCH_NAMES or base in DO_NOT_TOUCH_NAMES:
        return (
            "DO_NOT_TOUCH_YET",
            "Verify hang-protection overlap with TRITON_DailyRefresh before any disable",
            "# DO NOT RUN IN PHASE 195T: review run_pipeline_scheduled.ps1 overlap first",
            "# DO NOT RUN IN PHASE 195T",
            "Scheduled hang wrapper may still be required",
        )

    if norm in KEEP_ACTIVE_NAMES or base in KEEP_ACTIVE_NAMES:
        if "snapshot" in norm.lower() or base in {
            "TRITON_SnapshotLive",
            "TRITON_EquityDailySnapshot",
            "run_snapshot_live.ps1",
        }:
            return (
                "KEEP_ACTIVE_SNAPSHOT",
                "Keep task/script active — read-only broker snapshots",
                "# DO NOT RUN IN PHASE 195T: no change",
                "# DO NOT RUN IN PHASE 195T: Enable-ScheduledTask if ever disabled by mistake",
                "Authoritative snapshot path per Phase 195N",
            )
        if (
            norm in {"scripts/run_watchdog_loop.ps1", "scripts/run_continuous_loop.ps1"}
            or "watchdog" in norm.lower()
            and "loop" in norm.lower()
        ):
            return (
                "KEEP_ACTIVE_MONITORING",
                "Keep monitoring loops running — do not disable",
                "# DO NOT RUN IN PHASE 195T: no change",
                "# DO NOT RUN IN PHASE 195T",
                "Persistent runtime monitoring and plan-only continuous loop",
            )
        if norm == "services/execute_trades.py":
            return (
                "KEEP_OPERATOR_ONLY",
                "Operator-only CLI; never enable in scheduler without ARM + env gates",
                "# DO NOT RUN IN PHASE 195T: do not add execute_trades to any scheduled task",
                "# DO NOT RUN IN PHASE 195T",
                "Entry execution remains operator-only per Phase 195M",
            )
        if norm in {"TRITON_GuardAuto", "run_guard_auto.ps1"}:
            return (
                "KEEP_ACTIVE_MONITORING",
                "Keep guard auto task active — advisory protective snapshots",
                "# DO NOT RUN IN PHASE 195T: no change",
                "# DO NOT RUN IN PHASE 195T: Enable-ScheduledTask -TaskName 'TRITON_GuardAuto'",
                "Protective monitoring task",
            )
        return (
            "KEEP_ACTIVE_AUTHORITATIVE",
            "Keep active — authoritative production/readiness path",
            "# DO NOT RUN IN PHASE 195T: no change",
            "# DO NOT RUN IN PHASE 195T: Enable-ScheduledTask -TaskName '{name}'".format(name=base),
            "Authoritative chain confirmed by Phase 195N/195R",
        )

    if norm in REVIEW_BYPASS_NAMES or base in REVIEW_BYPASS_NAMES:
        cmd = f"# DO NOT RUN IN PHASE 195T: Disable-ScheduledTask -TaskName '{base}' -ErrorAction SilentlyContinue"
        rollback = f"# DO NOT RUN IN PHASE 195T: Enable-ScheduledTask -TaskName '{base}'"
        return (
            "REVIEW_LATER_BYPASS_RISK",
            "Code review for gate bypass; disable only after confirming no production dependency",
            cmd,
            rollback,
            "May submit orders or bypass execute_trades operator gates",
        )

    if norm in DISABLE_OR_REDIRECT_NAMES or base in DISABLE_OR_REDIRECT_NAMES:
        if "daily" in norm.lower() and "refresh" in norm.lower():
            cls = "REDIRECT_LATER_TO_AUTHORITATIVE"
            action = "Disable legacy task after confirming TRITON_DailyRefresh succeeds for 5+ days"
            cmd = f"# DO NOT RUN IN PHASE 195T: Disable-ScheduledTask -TaskName '{base}'"
            rollback = f"# DO NOT RUN IN PHASE 195T: Enable-ScheduledTask -TaskName '{base}'"
            reason = "Legacy duplicate of TRITON_DailyRefresh / run_daily_refresh.ps1"
        else:
            cls = "DISABLE_LATER_LEGACY_DUPLICATE"
            action = "Disable after confirming authoritative pipeline covers workload"
            cmd = f"# DO NOT RUN IN PHASE 195T: Disable-ScheduledTask -TaskName '{base}'"
            rollback = f"# DO NOT RUN IN PHASE 195T: Enable-ScheduledTask -TaskName '{base}'"
            reason = "Legacy duplicate or broken entry path per Phase 195N"
        return cls, action, cmd, rollback, reason

    if norm == "services/manage_positions.py":
        return (
            "KEEP_ACTIVE_MONITORING",
            "Plan-only exits/trims via continuous loop — not a scheduler task",
            "# DO NOT RUN IN PHASE 195T: no change",
            "# DO NOT RUN IN PHASE 195T",
            "No new entries; manage_positions is plan-only per 195M",
        )

    if etype == "windows_scheduled_task":
        return (
            "UNKNOWN_NEEDS_REVIEW",
            "Unmapped Triton scheduled task — operator review before Phase 195U isolation",
            f"# DO NOT RUN IN PHASE 195T: review task '{norm}' before Disable-ScheduledTask",
            f"# DO NOT RUN IN PHASE 195T: Enable-ScheduledTask -TaskName '{norm}'",
            "Task not in authoritative or legacy seed lists",
        )

    return (
        "UNKNOWN_NEEDS_REVIEW",
        "Review purpose and map to authoritative path",
        "# DO NOT RUN IN PHASE 195T",
        "# DO NOT RUN IN PHASE 195T",
        "Unclassified entrypoint",
    )


def _analyze_static(rel: str, etype: str) -> Dict[str, str]:
    path = ROOT / rel
    text = _read_text(path)
    flags = _match_patterns(text)
    missing_list = _find_missing_references(text, path)
    missing = ";".join(missing_list)
    exists = path.is_file()

    cls, action, prop_cmd, rollback, safety = _classify_195t(
        rel, etype, exists=exists, missing=missing, flags=flags, text=text
    )

    can_submit = _can_submit_orders(flags, text, rel)
    live_block = not flags.get("calls_execute_trades_execute_flag", False)
    if rel == "services/execute_trades.py":
        live_block = "live_execution_permitted" in text or "TRITON_ENABLE" in text

    return {
        "name": rel,
        "type": etype,
        "current_state": "file_exists" if exists else "missing",
        "last_run_time": "",
        "last_task_result": "",
        "next_run_time": "",
        "action_path": str(path) if exists else rel,
        "arguments": "",
        "working_directory": str(ROOT),
        "uses_dot_venv": str(flags.get("uses_dot_venv", False)).lower(),
        "uses_legacy_venv": str(flags.get("uses_legacy_venv", False)).lower(),
        "references_missing_target": missing,
        "calls_run_full_pipeline": str(flags.get("calls_run_full_pipeline", False)).lower(),
        "calls_pipeline_real_data": str(_calls_pipeline_real_data(text)).lower(),
        "calls_execute_trades": str(flags.get("calls_execute_trades", False)).lower(),
        "calls_execute_trades_execute": str(
            flags.get("calls_execute_trades_execute_flag", False)
        ).lower(),
        "calls_manage_positions": str(flags.get("calls_manage_positions", False)).lower(),
        "can_submit_orders": str(can_submit).lower(),
        "live_block_respected": str(live_block).lower(),
        "paper_only": "true",
        "classification": cls,
        "recommended_later_action": action,
        "proposed_operator_command": prop_cmd,
        "rollback_command": rollback,
        "safety_reason": safety,
    }


def _task_matches(name: str, rel: str) -> bool:
    stem = Path(rel).name.lower()
    return stem in name.lower() or rel.lower() in name.lower()


def _analyze_task(
    task: Dict[str, Any], static_by_name: Dict[str, Dict[str, str]]
) -> Dict[str, str]:
    task_name = str(task.get("TaskName") or "")
    execute = str(task.get("Execute") or "")
    arguments = str(task.get("Arguments") or "")
    workdir = str(task.get("WorkingDirectory") or "")
    blob = execute + " " + arguments
    flags = _match_patterns(blob)
    missing = ";".join(_find_missing_references(blob, ROOT))

    matched_static = None
    for rel, row in static_by_name.items():
        if _task_matches(blob, rel) or _task_matches(task_name, rel):
            matched_static = row
            break

    cls, action, prop_cmd, rollback, safety = _classify_195t(
        task_name,
        "windows_scheduled_task",
        exists=True,
        missing=missing,
        flags=flags,
        text=blob,
    )

    if matched_static and cls == "UNKNOWN_NEEDS_REVIEW":
        cls = matched_static["classification"]
        safety = matched_static["safety_reason"] + " (matched static entrypoint)"

    can_submit = _can_submit_orders(flags, blob, task_name)

    return {
        "name": task_name,
        "type": "windows_scheduled_task",
        "current_state": str(task.get("State") or "Unknown"),
        "last_run_time": str(task.get("LastRunTime") or ""),
        "last_task_result": str(task.get("LastTaskResult") or ""),
        "next_run_time": str(task.get("NextRunTime") or ""),
        "action_path": execute,
        "arguments": arguments,
        "working_directory": workdir,
        "uses_dot_venv": str(flags.get("uses_dot_venv", False)).lower(),
        "uses_legacy_venv": str(flags.get("uses_legacy_venv", False)).lower(),
        "references_missing_target": missing,
        "calls_run_full_pipeline": str(flags.get("calls_run_full_pipeline", False)).lower(),
        "calls_pipeline_real_data": str(_calls_pipeline_real_data(blob)).lower(),
        "calls_execute_trades": str(flags.get("calls_execute_trades", False)).lower(),
        "calls_execute_trades_execute": str(
            flags.get("calls_execute_trades_execute_flag", False)
        ).lower(),
        "calls_manage_positions": str(flags.get("calls_manage_positions", False)).lower(),
        "can_submit_orders": str(can_submit).lower(),
        "live_block_respected": str(
            not flags.get("calls_execute_trades_execute_flag", False)
        ).lower(),
        "paper_only": "true",
        "classification": cls,
        "recommended_later_action": action,
        "proposed_operator_command": prop_cmd,
        "rollback_command": rollback,
        "safety_reason": safety,
    }


def _build_operator_commands(rows: List[Dict[str, str]]) -> str:
    lines = [
        "=" * 72,
        "TRITON Phase 195T — Legacy Scheduler Isolation Operator Commands",
        "DO NOT RUN THESE COMMANDS IN PHASE 195T.",
        "DO NOT RUN UNTIL OPERATOR APPROVES Phase 195U EXECUTION.",
        "=" * 72,
        "",
        "# Preconditions before Phase 195U:",
        "# - execution_readiness READY, 195H READY",
        "# - Broker 0 open orders / 0 positions",
        "# - TRITON_DailyRefresh succeeded for 5+ consecutive days",
        "# - Watchdog + continuous loops healthy",
        "# - No execute_trades --execute processes running",
        "",
        "# --- KEEP ACTIVE (no commands) ---",
    ]
    for r in rows:
        if (
            r["classification"].startswith("KEEP_ACTIVE")
            or r["classification"] == "KEEP_OPERATOR_ONLY"
        ):
            lines.append(f"# KEEP: {r['name']} ({r['classification']})")

    lines.extend(["", "# --- DISABLE / REDIRECT LATER (Phase 195U) ---"])
    for r in rows:
        if r["classification"] in (
            "DISABLE_LATER_LEGACY_DUPLICATE",
            "REDIRECT_LATER_TO_AUTHORITATIVE",
            "REVIEW_LATER_BYPASS_RISK",
        ):
            lines.append(r["proposed_operator_command"])
            lines.append(f"# rollback: {r['rollback_command']}")
            lines.append(f"# reason: {r['safety_reason']}")
            lines.append("")

    lines.extend(
        [
            "# --- POST-ISOLATION VALIDATION (read-only) ---",
            "python -m services.execution_readiness",
            "python -m services.phase_195h_operational_readiness_freeze",
            "python -m services.snapshot_live_orders --mode paper",
            "",
            "# END — DO NOT RUN IN PHASE 195T",
        ]
    )
    return "\n".join(lines)


def run_phase_195t_dry_run(*, verbose: bool = True) -> Dict[str, Any]:
    tasks = _fetch_scheduled_tasks_extended()
    static_rows = [_analyze_static(rel, etype) for rel, etype in STATIC_ENTRYPOINTS]
    static_by_name = {r["name"]: r for r in static_rows}

    task_rows: List[Dict[str, str]] = []
    seen_tasks: Set[str] = set()
    for task in tasks:
        row = _analyze_task(task, static_by_name)
        task_rows.append(row)
        seen_tasks.add(row["name"])

    for rel, etype in STATIC_ENTRYPOINTS:
        base = Path(rel).name
        if not any(_task_matches(r["name"], rel) for r in task_rows):
            static_row = static_by_name[rel]
            if static_row["type"] != "windows_scheduled_task":
                task_rows.append(static_row)

    task_rows.sort(key=lambda r: (r["classification"], r["name"]))

    counts: Dict[str, int] = {}
    for r in task_rows:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1

    watchdog_running, continuous_running = _loops_running()
    execute_detected, execute_cmds = _find_python_execute_trades_execute()
    readiness = _load_json(EXEC_READINESS)
    phase_195h = _load_json(PHASE_195H_JSON)
    plan_195r = _load_json(PLAN_195R)
    paper_cfg = _load_json(PAPER_CYCLE_CFG)

    keep_active = [r["name"] for r in task_rows if r["classification"].startswith("KEEP_ACTIVE")]
    keep_active.append("KEEP_OPERATOR_ONLY entries: services/execute_trades.py")
    isolate_later = [
        r["name"]
        for r in task_rows
        if r["classification"]
        in (
            "DISABLE_LATER_LEGACY_DUPLICATE",
            "REDIRECT_LATER_TO_AUTHORITATIVE",
            "REVIEW_LATER_BYPASS_RISK",
        )
    ]

    operator_txt = _build_operator_commands(task_rows)

    payload: Dict[str, Any] = {
        "phase": "195T",
        "timestamp": _utc_now_iso(),
        "phase_pass": True,
        "audit_mode": "dry_run_planning_only",
        "orders_placed": False,
        "scheduled_tasks_changed": False,
        "scheduler_commands_executed": False,
        "safety_statement": (
            "Phase 195T is dry-run planning only. No orders placed, no scheduled tasks "
            "modified, no Disable-ScheduledTask or Enable-ScheduledTask commands executed."
        ),
        "watchdog_process_running": watchdog_running,
        "continuous_process_running": continuous_running,
        "execution_readiness_status": readiness.get("readiness_status", "UNKNOWN"),
        "phase_195h_status": phase_195h.get("operational_freeze_status", "UNKNOWN"),
        "broker_open_orders_count": _count_csv_rows(OPEN_ORDERS),
        "broker_positions_count": _count_csv_rows(POSITIONS),
        "live_execution_permitted": bool(readiness.get("live_execution_permitted", False)),
        "paper_execution_permitted": bool(readiness.get("paper_execution_permitted", False)),
        "execute_trades_execute_detected": execute_detected,
        "automated_entry_execution_enabled": bool(paper_cfg.get("run_execute_trades", False)),
        "windows_scheduled_tasks_observed_count": len(tasks),
        "entrypoints_planned_count": len(task_rows),
        "classification_counts": counts,
        "keep_active_authoritative_and_monitoring": keep_active,
        "isolate_or_review_later": isolate_later,
        "later_execution_phase": "Phase 195U — Legacy Scheduler Isolation Execution (operator-approved)",
        "195r_plan_reference": str(PLAN_195R.relative_to(ROOT)).replace("\\", "/"),
        "195n_audit_reference": str(ENTRYPOINT_195N.relative_to(ROOT)).replace("\\", "/"),
        "task_plan": task_rows,
        "raw_scheduled_tasks": tasks,
        "operator_commands_do_not_run_in_195t": operator_txt,
        "validation_plan_after_195u": [
            "Confirm TRITON_DailyRefresh LastTaskResult 0 after legacy disable",
            "python -m services.execution_readiness → READY",
            "python -m services.phase_195h_operational_readiness_freeze → READY",
            "python -m services.snapshot_live_orders --mode paper → 0 orders/positions",
            "Verify watchdog + continuous loops still running",
        ],
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(TASK_PLAN_CSV, task_rows, TASK_PLAN_FIELDS)
    _write_text(OPERATOR_COMMANDS_TXT, operator_txt)

    if verbose:
        print(f"[195T] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[195T] tasks_observed={len(tasks)} entrypoints_planned={len(task_rows)}")
        print(f"[195T] classifications={counts}")
        print("[195T] DO NOT RUN operator commands in Phase 195T")

    return payload


def main() -> None:
    run_phase_195t_dry_run(verbose=True)


if __name__ == "__main__":
    main()
