"""
phase_195n_runtime_entrypoint_consolidation_audit.py
Phase 195N — Runtime entry point consolidation & legacy scheduler isolation audit.

Read-only inventory of scripts, loops, schedulers, and launchers. Does not modify
tasks, locks, broker state, or runtime configuration.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
CONFIG = ROOT / "config"
SCRIPTS = ROOT / "scripts"
DATA_LIVE = ROOT / "data" / "live"

OUTPUT_JSON = RESULTS / "runtime_entrypoint_consolidation_audit.json"
ENTRYPOINTS_CSV = RESULTS / "runtime_entrypoint_consolidation_entrypoints.csv"
SUMMARY_CSV = RESULTS / "runtime_entrypoint_consolidation_summary.csv"

EXEC_READINESS_JSON = RESULTS / "execution_readiness.json"
EFFECTIVE_LIFECYCLE = RESULTS / "signal_lifecycle_effective.csv"
RAW_LIFECYCLE = RESULTS / "signal_lifecycle.csv"

CLASSIFICATIONS = (
    "AUTHORITATIVE_PRODUCTION_CHAIN",
    "AUTHORITATIVE_READINESS_WRITER",
    "AUTHORITATIVE_MONITORING_LOOP",
    "AUTHORITATIVE_SNAPSHOT_TASK",
    "SAFE_ADVISORY_ONLY",
    "LEGACY_DUPLICATE_ENTRYPOINT",
    "BROKEN_MISSING_TARGET",
    "DUAL_VENV_RISK",
    "EXECUTION_BYPASS_RISK",
    "SCHEDULER_HANG_RISK",
    "STALE_LOCK_RISK",
    "PLAN_ONLY_ENTRY_PATH",
    "OPERATOR_EXECUTE_ONLY_PATH",
    "NEEDS_OPERATOR_ISOLATION",
    "NEEDS_REDIRECT_TO_AUTHORITATIVE_PATH",
    "NEEDS_DEPRECATION_REVIEW",
)

ENTRYPOINT_COLUMNS = (
    "entrypoint_name",
    "path",
    "exists",
    "entrypoint_type",
    "scheduled_task_name",
    "schedule_cadence",
    "last_run_time",
    "last_task_result",
    "uses_dot_venv",
    "uses_legacy_venv",
    "calls_run_full_pipeline",
    "calls_execution_readiness",
    "calls_governance_execution_readiness",
    "calls_build_effective_lifecycle",
    "calls_build_trade_opportunities",
    "calls_execute_trades",
    "calls_execute_trades_execute_flag",
    "calls_manage_positions",
    "calls_broker_submit_function",
    "calls_place_live_orders",
    "calls_legacy_fetch_or_train_path",
    "references_missing_file",
    "can_mutate_orders",
    "live_execution_blocked",
    "paper_execution_only",
    "dry_run_default",
    "advisory_only",
    "authoritative_path",
    "issue_severity",
    "classification",
    "recommended_action",
    "responsible_file_or_function",
)

# Files whose primary role is launching other work (static seed list).
STATIC_ENTRYPOINTS: Tuple[Tuple[str, str], ...] = (
    ("run_full_pipeline.py", "python_orchestrator"),
    ("run_daily.cmd", "batch_scheduler"),
    ("run_daily_refresh.ps1", "powershell_scheduler"),
    ("run_guard_auto.ps1", "powershell_scheduler"),
    ("run_snapshot_live.ps1", "powershell_scheduler"),
    ("run-fetch.ps1", "powershell_scheduler"),
    ("run_pipeline_scheduled.ps1", "powershell_scheduler"),
    ("run_daily.py", "python_orchestrator"),
    ("run_adaptive_pipeline.py", "python_orchestrator"),
    ("daily_refresh.bat", "batch_scheduler"),
    ("Makefile", "make_target"),
    ("scripts/run_continuous_loop.ps1", "powershell_loop"),
    ("scripts/run_watchdog_loop.ps1", "powershell_loop"),
    ("scripts/run_market_open.ps1", "powershell_scheduler"),
    ("scripts/run_all.py", "python_orchestrator"),
    ("services/run_scheduled_paper_cycle.py", "python_orchestrator"),
    ("services/execution_readiness.py", "python_module"),
    ("services/governance_execution_readiness.py", "python_module"),
    ("services/build_effective_lifecycle.py", "python_module"),
    ("services/build_trade_opportunities.py", "python_module"),
    ("services/execute_trades.py", "python_execution"),
    ("services/manage_positions.py", "python_execution"),
    ("services/place_live_orders.py", "python_execution"),
    ("services/risk_watchdog.py", "python_monitoring"),
    ("services/trade_executor.py", "python_legacy"),
    ("services/portfolio_manager.py", "python_legacy"),
    ("scripts/auto_trade.py", "python_legacy_execution"),
    ("scripts/auto_exit.py", "python_legacy_execution"),
    ("scripts/auto_execute_signals.py", "python_legacy_execution"),
    ("scripts/execute_trade.py", "python_legacy_execution"),
    ("alpaca_client.py", "python_legacy_execution"),
    ("place_orders_from_csv.py", "python_manual_execution"),
    ("pipeline_real_data.py", "python_legacy_pipeline"),
    ("control_center.py", "streamlit_ui"),
    ("view_results.py", "streamlit_ui"),
)

LOCK_FILES: Tuple[Tuple[str, str], ...] = (
    ("continuous_loop.lock", "scripts/run_continuous_loop.ps1"),
    ("watchdog_loop.lock", "scripts/run_watchdog_loop.ps1"),
    ("pipeline_scheduled.lock", "run_pipeline_scheduled.ps1"),
)

MISSING_TARGET_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"train_models\.py", "services/train_models.py"),
    (r"scripts/train_model\.py", "scripts/train_model.py"),
    (r"scripts/control_center\.py", "scripts/control_center.py"),
    (r"scripts/fetch_and_prepare\.py", "scripts/fetch_and_prepare.py"),
    (r"run_market_open_sequence\.ps1", "scripts/run_market_open_sequence.ps1"),
)

FOCUS_NAMES = {
    "run_full_pipeline.py",
    "services/execution_readiness.py",
    "services/governance_execution_readiness.py",
    "scripts/run_continuous_loop.ps1",
    "scripts/run_watchdog_loop.ps1",
    "scripts/run_market_open.ps1",
    "run_daily.cmd",
    "daily_refresh.bat",
    "run-fetch.ps1",
    "Makefile",
    "scripts/run_all.py",
    "services/execute_trades.py",
    "services/manage_positions.py",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Tuple[str, ...]) -> None:
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


def _file_age_hours(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _fetch_scheduled_tasks() -> List[Dict[str, Any]]:
    """Read-only Windows Task Scheduler query."""
    ps = (
        "Get-ScheduledTask | Where-Object { "
        "$_.Actions.Execute -match 'triton' -or $_.Actions.Arguments -match 'triton' "
        "} | ForEach-Object { "
        "$info = Get-ScheduledTaskInfo $_; "
        "$tr = $_.Triggers | Select-Object -First 1; "
        "[PSCustomObject]@{ "
        "TaskName=$_.TaskName; State=$_.State; "
        "Execute=$_.Actions.Execute; Arguments=$_.Actions.Arguments; "
        "LastRunTime=$info.LastRunTime; LastTaskResult=$info.LastTaskResult; "
        "NextRunTime=$info.NextRunTime; "
        "StartBoundary=$tr.StartBoundary; DaysInterval=$tr.DaysInterval "
        "} } | ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(ROOT),
        )
        raw = (proc.stdout or "").strip()
        if not raw:
            return []
        data = json.loads(raw)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _match_patterns(text: str) -> Dict[str, bool]:
    lower = text.lower()
    return {
        "uses_dot_venv": ".venv" in lower or ".venv\\" in lower or ".venv/" in lower,
        "uses_legacy_venv": bool(
            re.search(r"(?<![.\w])venv\\scripts|(?<![.\w])venv/scripts", lower)
        ),
        "calls_run_full_pipeline": "run_full_pipeline" in lower,
        "calls_execution_readiness": bool(
            re.search(r"execution_readiness|services\.execution_readiness", lower)
        ),
        "calls_governance_execution_readiness": bool(
            re.search(
                r"governance_execution_readiness|persist_governance_execution_readiness",
                lower,
            )
        ),
        "calls_build_effective_lifecycle": "build_effective_lifecycle" in lower,
        "calls_build_trade_opportunities": "build_trade_opportunities" in lower,
        "calls_execute_trades": "execute_trades" in lower,
        "calls_execute_trades_execute_flag": bool(
            re.search(
                r"execute_trades[^\n]{0,80}--execute|execute_trades\.py[^\n]{0,40}--execute", lower
            )
            or ("--execute" in lower and "execute_trades" in lower)
        ),
        "calls_manage_positions": "manage_positions" in lower,
        "calls_broker_submit_function": bool(
            re.search(r"submit_order|place_bracket_market|api\.submit_order", lower)
        ),
        "calls_place_live_orders": "place_live_orders" in lower,
        "calls_legacy_fetch_or_train_path": bool(
            re.search(
                r"fetch_data\.py|train_models\.py|pipeline_real_data\.py|combine_predictions\.py",
                lower,
            )
        ),
    }


def _find_missing_references(text: str, base: Path) -> List[str]:
    missing: List[str] = []
    for pattern, rel in MISSING_TARGET_PATTERNS:
        if re.search(pattern, text, re.I):
            if not (ROOT / rel).is_file() and rel not in missing:
                missing.append(rel)
    # Explicit path references in batch/make
    for m in re.finditer(r"services\\([a-z_0-9]+\.py)", text, re.I):
        rel = f"services/{m.group(1).replace(chr(92), '/')}"
        if "train_models" in rel and not (ROOT / rel).is_file():
            if rel not in missing:
                missing.append(rel)
    for m in re.finditer(r"scripts\\([a-z_0-9]+\.py)", text, re.I):
        rel = f"scripts/{m.group(1).replace(chr(92), '/')}"
        if not (ROOT / rel).is_file() and rel not in missing:
            missing.append(rel)
    return missing


def _classify_entrypoint(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    name = row["entrypoint_name"]
    path = row["path"]
    exists = row.get("exists")
    missing = row.get("references_missing_file") or ""
    flags = row  # dict with bool fields

    if not exists:
        return (
            "BROKEN_MISSING_TARGET",
            "HIGH",
            "NEEDS_DEPRECATION_REVIEW",
            "Remove or restore missing entrypoint file",
        )

    if missing:
        return (
            "BROKEN_MISSING_TARGET",
            "HIGH",
            "NEEDS_REDIRECT_TO_AUTHORITATIVE_PATH",
            f"Fix references to missing targets: {missing}",
        )

    if name == "run_full_pipeline.py":
        return (
            "AUTHORITATIVE_PRODUCTION_CHAIN",
            "INFO",
            "AUTHORITATIVE_PRODUCTION_CHAIN",
            "Keep active; sole intelligence→lifecycle→trade_opportunities chain",
        )

    if name in {"run_daily_refresh.ps1", "TRITON_DailyRefresh"}:
        return (
            "AUTHORITATIVE_PRODUCTION_CHAIN",
            "INFO",
            "AUTHORITATIVE_PRODUCTION_CHAIN",
            "Keep active; modern daily refresh via run_full_pipeline",
        )

    if name in {"services/governance_execution_readiness.py", "services/risk_watchdog.py"}:
        return (
            "AUTHORITATIVE_READINESS_WRITER",
            "INFO",
            "AUTHORITATIVE_READINESS_WRITER",
            "Keep active; writes execution_readiness.json",
        )

    if name == "services/execution_readiness.py":
        return (
            "AUTHORITATIVE_READINESS_WRITER",
            "INFO",
            "AUTHORITATIVE_READINESS_WRITER",
            "CLI wrapper; invoke after lifecycle rebuild or via watchdog",
        )

    if name in {"scripts/run_watchdog_loop.ps1", "services/risk_watchdog.py"}:
        return (
            "AUTHORITATIVE_MONITORING_LOOP",
            "INFO",
            "AUTHORITATIVE_MONITORING_LOOP",
            "Keep active; read-only monitoring + readiness refresh",
        )

    if name in {"run_snapshot_live.ps1", "TRITON_SnapshotLive", "TRITON_EquityDailySnapshot"}:
        return (
            "AUTHORITATIVE_SNAPSHOT_TASK",
            "INFO",
            "AUTHORITATIVE_SNAPSHOT_TASK",
            "Keep active; broker snapshot refresh",
        )

    if name == "run_guard_auto.ps1" or name == "TRITON_GuardAuto":
        return (
            "AUTHORITATIVE_MONITORING_LOOP",
            "LOW",
            "AUTHORITATIVE_MONITORING_LOOP",
            "Keep active; weekend guard auto-clear only",
        )

    if name == "scripts/run_continuous_loop.ps1":
        return (
            "PLAN_ONLY_ENTRY_PATH",
            "MEDIUM",
            "NEEDS_REDIRECT_TO_AUTHORITATIVE_PATH",
            "Loop active for market_open/manage_positions but does not call execute_trades; "
            "redirect entry execution to operator or run_scheduled_paper_cycle",
        )

    if name == "scripts/run_market_open.ps1" or name == "TRITON_MarketOpen":
        return (
            "PLAN_ONLY_ENTRY_PATH",
            "MEDIUM",
            "AUTHORITATIVE_PRODUCTION_CHAIN",
            "Keep for exit/trim/analytics; not an entry submission path",
        )

    if name == "services/execute_trades.py":
        return (
            "OPERATOR_EXECUTE_ONLY_PATH",
            "INFO",
            "OPERATOR_EXECUTE_ONLY_PATH",
            "Authoritative entry executor; dry-run default; --execute requires operator+ARM",
        )

    if name == "services/run_scheduled_paper_cycle.py":
        return (
            "OPERATOR_EXECUTE_ONLY_PATH",
            "MEDIUM",
            "NEEDS_OPERATOR_ISOLATION",
            "Can call execute_trades --execute when config+ARM permit; isolate from legacy tasks",
        )

    if flags.get("calls_broker_submit_function") and not flags.get("calls_execute_trades"):
        if name in {
            "scripts/auto_trade.py",
            "scripts/auto_exit.py",
            "scripts/auto_execute_signals.py",
            "scripts/execute_trade.py",
            "alpaca_client.py",
        }:
            return (
                "EXECUTION_BYPASS_RISK",
                "CRITICAL",
                "NEEDS_DEPRECATION_REVIEW",
                "Legacy direct broker submit outside unified gate chain",
            )

    if name in {
        "daily_refresh.bat",
        "Makefile",
        "scripts/run_all.py",
        "run-fetch.ps1",
        "pipeline_real_data.py",
    }:
        return (
            "LEGACY_DUPLICATE_ENTRYPOINT",
            "HIGH",
            "NEEDS_REDIRECT_TO_AUTHORITATIVE_PATH",
            "Redirect to run_daily_refresh.ps1 + run_full_pipeline.py",
        )

    if flags.get("uses_legacy_venv") and flags.get("uses_dot_venv"):
        return (
            "DUAL_VENV_RISK",
            "MEDIUM",
            "NEEDS_REDIRECT_TO_AUTHORITATIVE_PATH",
            "Standardize on .venv across entrypoints",
        )

    if flags.get("uses_legacy_venv") and not flags.get("uses_dot_venv"):
        return (
            "DUAL_VENV_RISK",
            "MEDIUM",
            "NEEDS_REDIRECT_TO_AUTHORITATIVE_PATH",
            "Migrate from venv\\ to .venv\\",
        )

    if name in {"view_results.py", "control_center.py"}:
        return ("SAFE_ADVISORY_ONLY", "INFO", "SAFE_ADVISORY_ONLY", "UI only; no order mutation")

    if name == "run_pipeline_scheduled.ps1":
        return (
            "SCHEDULER_HANG_RISK",
            "MEDIUM",
            "AUTHORITATIVE_PRODUCTION_CHAIN",
            "Keep but monitor lock/TTL; pipeline can hang at fundamentals",
        )

    if flags.get("advisory_only"):
        return ("SAFE_ADVISORY_ONLY", "INFO", "SAFE_ADVISORY_ONLY", "Advisory/read-only path")

    if "Triton Daily" in name and "Refresh" not in name:
        return (
            "LEGACY_DUPLICATE_ENTRYPOINT",
            "HIGH",
            "NEEDS_REDIRECT_TO_AUTHORITATIVE_PATH",
            "Persona/legacy daily chain; redirect to TRITON_DailyRefresh",
        )

    return (
        "NEEDS_DEPRECATION_REVIEW",
        "LOW",
        "NEEDS_OPERATOR_ISOLATION",
        "Review task purpose and redirect",
    )


def _analyze_static_entrypoint(rel: str, etype: str) -> Dict[str, Any]:
    path = ROOT / rel
    text = _read_text(path)
    flags = _match_patterns(text)
    missing = _find_missing_references(text, path)
    missing_str = ";".join(missing)

    can_mutate = bool(
        flags["calls_broker_submit_function"]
        or flags["calls_execute_trades_execute_flag"]
        or (flags["calls_place_live_orders"] and "dry_run" not in text.lower())
        or (rel == "services/manage_positions.py")
    )
    if rel in {
        "services/execute_trades.py",
        "services/place_live_orders.py",
        "services/manage_positions.py",
    }:
        can_mutate = True  # capability exists; gated by flags

    advisory = rel in {"view_results.py", "control_center.py", "services/risk_watchdog.py"} or (
        "audit" in rel.lower() and "execute" not in rel.lower()
    )
    if rel == "services/risk_watchdog.py":
        advisory = False  # writes readiness but read-only broker

    dry_run_default = False
    if rel == "services/execute_trades.py":
        dry_run_default = (
            "dry_run = not bool(args.execute)" in text or "dry_run=not" in text.replace(" ", "")
        )
    if rel == "services/place_live_orders.py":
        dry_run_default = "--dry-run" in text and "default" in text.lower()

    row: Dict[str, Any] = {
        "entrypoint_name": rel.replace("\\", "/"),
        "path": str(path),
        "exists": path.is_file(),
        "entrypoint_type": etype,
        "scheduled_task_name": "",
        "schedule_cadence": "",
        "last_run_time": "",
        "last_task_result": "",
        "references_missing_file": missing_str,
        "can_mutate_orders": can_mutate,
        "live_execution_blocked": True,
        "paper_execution_only": etype
        in {"python_execution", "python_orchestrator", "powershell_loop"},
        "dry_run_default": dry_run_default,
        "advisory_only": advisory,
        "authoritative_path": rel
        in {
            "run_full_pipeline.py",
            "run_daily_refresh.ps1",
            "services/execute_trades.py",
            "services/build_trade_opportunities.py",
            "services/build_effective_lifecycle.py",
            "services/governance_execution_readiness.py",
        },
        **flags,
    }

    cls, sev, _sub, action = _classify_entrypoint(row)
    row["classification"] = cls
    row["issue_severity"] = sev
    row["recommended_action"] = action
    row["responsible_file_or_function"] = rel
    return row


def _task_matches_path(task: Dict[str, Any], rel: str) -> bool:
    exe = str(task.get("Execute") or "")
    args = str(task.get("Arguments") or "")
    blob = (exe + " " + args).lower()
    stem = Path(rel).name.lower()
    rel_lower = rel.replace("\\", "/").lower()
    return stem in blob or rel_lower in blob


def _analyze_scheduled_task(
    task: Dict[str, Any], static_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    task_name = str(task.get("TaskName") or "")
    execute = str(task.get("Execute") or "")
    arguments = str(task.get("Arguments") or "")
    blob = execute + " " + arguments
    flags = _match_patterns(blob)
    missing = _find_missing_references(blob, ROOT)
    missing_str = ";".join(missing)

    cadence = ""
    if task.get("DaysInterval"):
        cadence = f"daily interval={task.get('DaysInterval')}"
    if task.get("StartBoundary"):
        cadence = (cadence + " " + str(task.get("StartBoundary"))).strip()

    matched_static = next(
        (r for r in static_rows if _task_matches_path(task, r["entrypoint_name"])), None
    )

    can_mutate = bool(
        flags["calls_broker_submit_function"]
        or flags["calls_execute_trades_execute_flag"]
        or "auto_trade" in blob.lower()
        or "auto_execute" in blob.lower()
    )

    row: Dict[str, Any] = {
        "entrypoint_name": task_name,
        "path": execute if execute else arguments,
        "exists": True,
        "entrypoint_type": "windows_scheduled_task",
        "scheduled_task_name": task_name,
        "schedule_cadence": cadence,
        "last_run_time": str(task.get("LastRunTime") or ""),
        "last_task_result": str(task.get("LastTaskResult") or ""),
        "references_missing_file": missing_str,
        "can_mutate_orders": can_mutate,
        "live_execution_blocked": True,
        "paper_execution_only": True,
        "dry_run_default": not flags["calls_execute_trades_execute_flag"],
        "advisory_only": "snapshot" in task_name.lower() or "guard" in task_name.lower(),
        "authoritative_path": task_name
        in {
            "TRITON_DailyRefresh",
            "TRITON_MarketOpen",
            "TRITON_GuardAuto",
            "TRITON_SnapshotLive",
            "Triton_Run_Daily",
        },
        **flags,
    }
    if matched_static:
        row["responsible_file_or_function"] = matched_static["entrypoint_name"]
    else:
        row["responsible_file_or_function"] = execute or arguments

    cls, sev, _sub, action = _classify_entrypoint(row)
    row["classification"] = cls
    row["issue_severity"] = sev
    row["recommended_action"] = action
    return row


def _analyze_lock_file(name: str, script_rel: str) -> Dict[str, Any]:
    path = DATA_LIVE / name if "pipeline" not in name else RESULTS / name
    exists = path.is_file()
    age_min = None
    content = ""
    if exists:
        age_min = (_file_age_hours(path) or 0) * 60
        content = _read_text(path)[:200]

    stale = exists and age_min is not None and age_min > 120
    classification = "STALE_LOCK_RISK" if stale else "AUTHORITATIVE_MONITORING_LOOP"
    sev = "MEDIUM" if stale else "INFO"

    return {
        "entrypoint_name": f"lock:{name}",
        "path": str(path),
        "exists": exists,
        "entrypoint_type": "runtime_lock",
        "scheduled_task_name": "",
        "schedule_cadence": "",
        "last_run_time": "",
        "last_task_result": "",
        "uses_dot_venv": False,
        "uses_legacy_venv": False,
        "calls_run_full_pipeline": False,
        "calls_execution_readiness": False,
        "calls_governance_execution_readiness": False,
        "calls_build_effective_lifecycle": False,
        "calls_build_trade_opportunities": False,
        "calls_execute_trades": False,
        "calls_execute_trades_execute_flag": False,
        "calls_manage_positions": False,
        "calls_broker_submit_function": False,
        "calls_place_live_orders": False,
        "calls_legacy_fetch_or_train_path": False,
        "references_missing_file": "",
        "can_mutate_orders": False,
        "live_execution_blocked": True,
        "paper_execution_only": True,
        "dry_run_default": False,
        "advisory_only": True,
        "authoritative_path": True,
        "issue_severity": sev,
        "classification": classification,
        "recommended_action": (
            "Review stale lock before restart; do not delete during audit"
            if stale
            else "Normal runtime lock metadata"
        ),
        "responsible_file_or_function": script_rel,
        "lock_age_minutes": round(age_min, 1) if age_min is not None else "",
        "lock_content_preview": content.strip(),
    }


def _readiness_stale_analysis() -> Dict[str, Any]:
    pipeline_text = _read_text(ROOT / "run_full_pipeline.py")
    refreshes_in_pipeline = bool(
        re.search(r"execution_readiness|governance_execution_readiness", pipeline_text, re.I)
    )
    readiness_age = _file_age_hours(EXEC_READINESS_JSON)
    eff_age = _file_age_hours(EFFECTIVE_LIFECYCLE)
    raw_age = _file_age_hours(RAW_LIFECYCLE)
    readiness = _load_json(EXEC_READINESS_JSON)

    stale_risk = False
    reason = ""
    if refreshes_in_pipeline:
        reason = "pipeline includes readiness refresh"
    else:
        reason = "run_full_pipeline.py does NOT call execution_readiness or governance_execution_readiness"
        if eff_age is not None and readiness_age is not None and eff_age < readiness_age - 0.01:
            stale_risk = True
            reason += "; effective lifecycle newer than execution_readiness.json"

    return {
        "run_full_pipeline_refreshes_execution_readiness": refreshes_in_pipeline,
        "readiness_can_go_stale_after_lifecycle_rebuild": not refreshes_in_pipeline,
        "stale_risk_observed": stale_risk,
        "execution_readiness_age_hours": readiness_age,
        "effective_lifecycle_age_hours": eff_age,
        "raw_lifecycle_age_hours": raw_age,
        "readiness_status": readiness.get("readiness_status"),
        "readiness_failed_checks": readiness.get("failed_checks"),
        "analysis": reason,
        "readiness_refresh_paths": [
            "services/risk_watchdog.py → persist_governance_execution_readiness",
            "services/governance_execution_readiness.py (CLI)",
            "services/execution_readiness.py (CLI wrapper)",
        ],
    }


def run_phase_195n_audit(*, verbose: bool = True) -> Dict[str, Any]:
    static_rows = [_analyze_static_entrypoint(rel, etype) for rel, etype in STATIC_ENTRYPOINTS]
    tasks = _fetch_scheduled_tasks()
    task_rows = [_analyze_scheduled_task(t, static_rows) for t in tasks]
    lock_rows = [_analyze_lock_file(name, script) for name, script in LOCK_FILES]

    all_rows: List[Dict[str, Any]] = static_rows + task_rows + lock_rows

    counts = Counter(r.get("classification") for r in all_rows)
    sev_counts = Counter(r.get("issue_severity") for r in all_rows)

    authoritative = [
        r["entrypoint_name"]
        for r in all_rows
        if r.get("classification")
        in {
            "AUTHORITATIVE_PRODUCTION_CHAIN",
            "AUTHORITATIVE_READINESS_WRITER",
            "AUTHORITATIVE_MONITORING_LOOP",
            "AUTHORITATIVE_SNAPSHOT_TASK",
        }
        or r.get("authoritative_path")
    ]
    isolate = [
        r["entrypoint_name"]
        for r in all_rows
        if r.get("classification")
        in {
            "LEGACY_DUPLICATE_ENTRYPOINT",
            "BROKEN_MISSING_TARGET",
            "EXECUTION_BYPASS_RISK",
            "DUAL_VENV_RISK",
        }
    ]
    keep_active = sorted(
        {
            r["entrypoint_name"]
            for r in all_rows
            if r.get("classification")
            in {
                "AUTHORITATIVE_PRODUCTION_CHAIN",
                "AUTHORITATIVE_READINESS_WRITER",
                "AUTHORITATIVE_MONITORING_LOOP",
                "AUTHORITATIVE_SNAPSHOT_TASK",
                "PLAN_ONLY_ENTRY_PATH",
                "OPERATOR_EXECUTE_ONLY_PATH",
            }
            and r.get("issue_severity") != "CRITICAL"
        }
    )

    readiness = _readiness_stale_analysis()
    paper_cycle_cfg = _load_json(CONFIG / "run_scheduled_paper_cycle.json")

    safe_restart = [
        "1. Confirm no hung python pipeline (Get-Process python; do NOT kill during audit)",
        "2. Verify broker 0 open orders / 0 positions via snapshot_live_orders --mode paper",
        "3. Run run_full_pipeline.py to completion",
        "4. Run python -m services.execution_readiness (or wait for watchdog cycle)",
        "5. Verify execution_readiness.json READY and lifecycle diagnostics pass",
        "6. Review lock files: continuous_loop.lock, watchdog_loop.lock, pipeline_scheduled.lock",
        "7. If locks stale and no matching process: operator may clear locks then restart loops",
        "8. Start watchdog: scripts/run_watchdog_loop.ps1",
        "9. Start continuous loop only if paper entry path intentionally excluded or after paper_cycle config review",
        "10. Entry submissions remain operator-only: execute_trades --mode paper --execute with ARM+env",
    ]

    next_hardening = (
        "Phase 1: Disable/isolate legacy schedulers (Triton Daily Refresh bat, TritonFetchDaily, persona daily tasks) "
        "→ redirect to TRITON_DailyRefresh. Phase 2: Add execution_readiness refresh tail to run_full_pipeline.py "
        "(future change; not in this audit). Phase 3: Wire run_scheduled_paper_cycle or document operator-only "
        "execute_trades --execute path. Phase 4: Remove venv\\ references; deprecate auto_trade/auto_execute_signals."
    )

    focus_analysis = {
        name: next(
            (
                r
                for r in all_rows
                if r.get("entrypoint_name") == name or name in r.get("entrypoint_name", "")
            ),
            {},
        )
        for name in sorted(FOCUS_NAMES)
    }

    payload: Dict[str, Any] = {
        "phase": "195N",
        "timestamp": _utc_now_iso(),
        "phase_pass": True,
        "audit_mode": "read_only",
        "orders_placed": False,
        "summary": {
            "total_entrypoints_inspected": len(all_rows),
            "total_authoritative_paths": len(set(authoritative)),
            "total_safe_advisory_only": counts.get("SAFE_ADVISORY_ONLY", 0),
            "total_broken_missing_target": counts.get("BROKEN_MISSING_TARGET", 0),
            "total_legacy_duplicate_paths": counts.get("LEGACY_DUPLICATE_ENTRYPOINT", 0),
            "total_dual_venv_risks": counts.get("DUAL_VENV_RISK", 0),
            "total_execution_bypass_risks": counts.get("EXECUTION_BYPASS_RISK", 0),
            "total_scheduler_hang_risks": counts.get("SCHEDULER_HANG_RISK", 0),
            "classification_counts": dict(counts),
            "severity_counts": dict(sev_counts),
            "run_full_pipeline_refreshes_execution_readiness": readiness[
                "run_full_pipeline_refreshes_execution_readiness"
            ],
            "readiness_can_go_stale_after_lifecycle_rebuild": readiness[
                "readiness_can_go_stale_after_lifecycle_rebuild"
            ],
            "continuous_loop_authoritative": False,
            "continuous_loop_calls_execute_trades": False,
            "scheduled_paper_cycle_can_call_execute_trades": True,
            "scheduled_paper_cycle_config_run_execute_trades": bool(
                paper_cycle_cfg.get("run_execute_trades", True)
            ),
            "execute_trades_requires_operator_approval": True,
            "authoritative_production_chain": [
                "run_full_pipeline.py",
                "services.build_effective_lifecycle",
                "services.build_trade_opportunities",
                "trade_opportunities.csv",
                "services.execute_trades (operator --execute)",
            ],
            "entry_points_keep_active": keep_active,
            "entry_points_isolate_or_disable_later": sorted(set(isolate)),
            "recommended_next_hardening_step": next_hardening,
            "safe_restart_sequence": safe_restart,
        },
        "key_questions": {
            "1_single_authoritative_production_chain": (
                "run_full_pipeline.py → lifecycle → build_effective_lifecycle → "
                "build_trade_opportunities → trade_opportunities.csv → execute_trades (--execute operator path)"
            ),
            "2_scripts_launch_chain_correctly": [
                "run_daily_refresh.ps1 / TRITON_DailyRefresh",
                "run_pipeline_scheduled.ps1",
                "TRITON_MarketOpen (partial: no execute_trades)",
            ],
            "3_legacy_broken_chains": [
                "daily_refresh.bat",
                "run-fetch.ps1 / TritonFetchDaily",
                "Makefile refresh",
                "scripts/run_all.py",
                "persona Triton Daily* tasks",
            ],
            "4_dot_venv_correct": [
                "run_daily_refresh.ps1",
                "run_guard_auto.ps1",
                "run_snapshot_live.ps1",
                "run_pipeline_scheduled.ps1",
            ],
            "5_legacy_venv_paths": [
                "daily_refresh.bat",
                "Makefile",
                "Triton Attach Brackets",
                "Triton Daily",
            ],
            "6_missing_module_references": [
                "services/train_models.py",
                "scripts/train_model.py",
                "scripts/control_center.py",
            ],
            "7_calls_run_full_pipeline": [
                r["entrypoint_name"] for r in all_rows if r.get("calls_run_full_pipeline")
            ],
            "8_calls_execution_readiness": [
                r["entrypoint_name"] for r in all_rows if r.get("calls_execution_readiness")
            ],
            "9_calls_build_effective_lifecycle": [
                r["entrypoint_name"] for r in all_rows if r.get("calls_build_effective_lifecycle")
            ],
            "10_calls_execute_trades": [
                r["entrypoint_name"] for r in all_rows if r.get("calls_execute_trades")
            ],
            "11_calls_manage_positions": [
                r["entrypoint_name"] for r in all_rows if r.get("calls_manage_positions")
            ],
            "12_execution_bypass_paths": [
                r["entrypoint_name"]
                for r in all_rows
                if r.get("classification") == "EXECUTION_BYPASS_RISK"
            ],
            "13_can_mutate_orders": [
                r["entrypoint_name"] for r in all_rows if r.get("can_mutate_orders")
            ],
            "14_safe_advisory_only": [
                r["entrypoint_name"]
                for r in all_rows
                if r.get("classification") == "SAFE_ADVISORY_ONLY"
            ],
            "15_tasks_remain_active": keep_active,
            "16_tasks_disable_or_redirect_later": sorted(set(isolate)),
            "17_duplicate_entrypoints": counts.get("LEGACY_DUPLICATE_ENTRYPOINT", 0),
            "18_dual_pipeline_risk": counts.get("LEGACY_DUPLICATE_ENTRYPOINT", 0)
            + counts.get("DUAL_VENV_RISK", 0),
            "19_runtime_lock_files": [r["entrypoint_name"] for r in lock_rows],
            "20_safe_restart_sequence": safe_restart,
        },
        "readiness_stale_analysis": readiness,
        "focus_analysis": focus_analysis,
        "focus_entrypoints_detail": [
            r for r in all_rows if any(f in str(r.get("entrypoint_name", "")) for f in FOCUS_NAMES)
        ],
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(ENTRYPOINTS_CSV, all_rows, ENTRYPOINT_COLUMNS)
    summary_rows = [
        {"metric": k, "value": v}
        for k, v in payload["summary"].items()
        if not isinstance(v, (list, dict))
    ] + [
        {
            "metric": "entry_points_keep_active",
            "value": ";".join(keep_active[:20]) + ("..." if len(keep_active) > 20 else ""),
        },
        {"metric": "entry_points_isolate_later", "value": ";".join(sorted(set(isolate)))},
    ]
    _write_csv(SUMMARY_CSV, summary_rows, ("metric", "value"))

    if verbose:
        print(
            json.dumps(
                {"phase": "195N", "phase_pass": True, "summary": payload["summary"]}, indent=2
            )
        )

    return payload


def main() -> int:
    run_phase_195n_audit(verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
