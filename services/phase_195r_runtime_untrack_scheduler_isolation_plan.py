"""
phase_195r_runtime_untrack_scheduler_isolation_plan.py
Phase 195R — Runtime artifact untrack plan & legacy scheduler isolation safety review.

Planning-only: produces exact later cleanup commands without executing git rm --cached,
scheduler edits, lock deletion, or broker mutations.
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
LIVE = ROOT / "data" / "live"
CONFIG = ROOT / "config"

OUTPUT_JSON = RESULTS / "runtime_untrack_scheduler_isolation_plan.json"
FILE_PLAN_CSV = RESULTS / "runtime_untrack_file_plan.csv"
SCHEDULER_PLAN_CSV = RESULTS / "scheduler_isolation_plan.csv"
OPERATOR_COMMANDS_TXT = RESULTS / "runtime_cleanup_operator_commands.txt"

AUDIT_195Q = RESULTS / "runtime_artifact_git_hygiene_audit.json"
MANIFEST_195Q = RESULTS / "runtime_artifact_git_hygiene_file_manifest.csv"
ENTRYPOINT_195N = RESULTS / "runtime_entrypoint_consolidation_audit.json"
EXEC_READINESS = RESULTS / "execution_readiness.json"
PHASE_195H_JSON = RESULTS / "phase_195h_operational_readiness_freeze.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
PAPER_CYCLE_CFG = CONFIG / "run_scheduled_paper_cycle.json"

from services.phase_195q_runtime_artifact_git_hygiene_audit import (  # noqa: E402
    ACTIVE_LOOP_WRITERS,
    BASELINE_TRACKED_ARTIFACTS,
    LIVE_LOCK_FILES,
    LIVE_STATE_FILES,
)

FILE_CLASSIFICATIONS = (
    "UNTRACK_LATER_RUNTIME_LIVE_STATE",
    "UNTRACK_LATER_RUNTIME_RESULT",
    "UNTRACK_LATER_PROCESSED_DATA",
    "KEEP_TRACKED_BASELINE_EVIDENCE",
    "KEEP_TRACKED_SOURCE_OR_CONFIG",
    "NEEDS_OPERATOR_REVIEW",
)

SCHEDULER_CLASSIFICATIONS = (
    "KEEP_ACTIVE_AUTHORITATIVE",
    "KEEP_ACTIVE_MONITORING",
    "KEEP_OPERATOR_ONLY",
    "DISABLE_LATER_LEGACY_DUPLICATE",
    "REDIRECT_LATER_TO_AUTHORITATIVE",
    "REVIEW_LATER_BYPASS_RISK",
    "DO_NOT_TOUCH_YET",
)

FILE_PLAN_FIELDS = (
    "path",
    "plan_classification",
    "tracked_by_git",
    "ignored_by_gitignore",
    "currently_modified",
    "active_writer",
    "safe_to_restore_now",
    "safe_to_untrack_later",
    "should_remain_tracked",
    "reason",
    "proposed_later_command",
)

SCHEDULER_PLAN_FIELDS = (
    "entrypoint_or_task",
    "entrypoint_type",
    "plan_classification",
    "scheduled_task_name",
    "task_state",
    "last_run_time",
    "authoritative_replacement",
    "recommended_later_action",
    "do_not_execute_yet",
    "reason",
)

UNTRACK_LATER_RUNTIME_RESULT = frozenset(
    {
        "data/results/executive_outlook_reconciliation.json",
        "data/results/executive_outlook_scorecard.csv",
        "data/results/lifecycle_readiness_diagnostics.json",
    }
)

UNTRACK_LATER_LIVE = LIVE_LOCK_FILES | LIVE_STATE_FILES

STATIC_SCHEDULER_ENTRIES: Tuple[Tuple[str, str, str, str, str], ...] = (
    (
        "TRITON_DailyRefresh",
        "windows_scheduled_task",
        "run_daily_refresh.ps1",
        "KEEP_ACTIVE_AUTHORITATIVE",
        "Authoritative daily pipeline via .venv; replaces legacy bat/Makefile chains",
    ),
    (
        "run_daily_refresh.ps1",
        "powershell_scheduler",
        "run_full_pipeline.py",
        "KEEP_ACTIVE_AUTHORITATIVE",
        "Launcher for TRITON_DailyRefresh task",
    ),
    (
        "TRITON_MarketOpen",
        "windows_scheduled_task",
        "scripts/run_market_open.ps1",
        "KEEP_ACTIVE_AUTHORITATIVE",
        "Market-open plan-only chain; no automated execute_trades --execute",
    ),
    (
        "TRITON_GuardAuto",
        "windows_scheduled_task",
        "run_guard_auto.ps1",
        "KEEP_ACTIVE_MONITORING",
        "Protective guard snapshot; advisory only",
    ),
    (
        "TRITON_SnapshotLive",
        "windows_scheduled_task",
        "run_snapshot_live.ps1",
        "KEEP_ACTIVE_MONITORING",
        "Broker snapshot read-only",
    ),
    (
        "TRITON_EquityDailySnapshot",
        "windows_scheduled_task",
        "run_snapshot_live.ps1",
        "KEEP_ACTIVE_MONITORING",
        "Equity snapshot read-only",
    ),
    (
        "scripts/run_watchdog_loop.ps1",
        "powershell_loop",
        "services/risk_watchdog.py",
        "KEEP_ACTIVE_MONITORING",
        "Persistent watchdog; refreshes heartbeat and readiness",
    ),
    (
        "scripts/run_continuous_loop.ps1",
        "powershell_loop",
        "scripts/run_market_open.ps1",
        "KEEP_ACTIVE_MONITORING",
        "Plan-only continuous loop; no execute_trades --execute",
    ),
    (
        "run_full_pipeline.py",
        "python_orchestrator",
        "run_full_pipeline.py",
        "KEEP_ACTIVE_AUTHORITATIVE",
        "Single authoritative production chain per Phase 195N",
    ),
    (
        "services/execution_readiness.py",
        "python_module",
        "services/governance_execution_readiness.py",
        "KEEP_ACTIVE_AUTHORITATIVE",
        "Readiness CLI writer with lifecycle diagnostics",
    ),
    (
        "services/governance_execution_readiness.py",
        "python_module",
        "services/governance_execution_readiness.py",
        "KEEP_ACTIVE_AUTHORITATIVE",
        "Governance readiness persistence from watchdog cycle",
    ),
    (
        "services/execute_trades.py",
        "python_execution",
        "",
        "KEEP_OPERATOR_ONLY",
        "Operator-only entry path; requires --execute + ARM + TRITON_ENABLE_PAPER_EXECUTION",
    ),
    (
        "run_pipeline_scheduled.ps1",
        "powershell_scheduler",
        "run_full_pipeline.py",
        "DO_NOT_TOUCH_YET",
        "Hang-protection wrapper; verify no duplicate overlap with TRITON_DailyRefresh before changes",
    ),
    (
        "daily_refresh.bat",
        "batch_scheduler",
        "run_daily_refresh.ps1",
        "REDIRECT_LATER_TO_AUTHORITATIVE",
        "Legacy venv\\ chain; duplicate of TRITON_DailyRefresh",
    ),
    (
        "Triton Daily Refresh",
        "windows_scheduled_task",
        "run_daily_refresh.ps1",
        "REDIRECT_LATER_TO_AUTHORITATIVE",
        "Legacy task name; redirect to TRITON_DailyRefresh",
    ),
    (
        "run-fetch.ps1",
        "powershell_scheduler",
        "run_full_pipeline.py",
        "DISABLE_LATER_LEGACY_DUPLICATE",
        "Legacy pipeline_real_data fetch; superseded by authoritative pipeline",
    ),
    (
        "TritonFetchDaily",
        "windows_scheduled_task",
        "run_full_pipeline.py",
        "DISABLE_LATER_LEGACY_DUPLICATE",
        "Legacy fetch task; disable after confirming TRITON_DailyRefresh covers data refresh",
    ),
    (
        "Makefile",
        "make_target",
        "run_daily_refresh.ps1",
        "DISABLE_LATER_LEGACY_DUPLICATE",
        "Legacy refresh target using venv\\ paths",
    ),
    (
        "scripts/run_all.py",
        "python_orchestrator",
        "run_full_pipeline.py",
        "DISABLE_LATER_LEGACY_DUPLICATE",
        "Broken/missing modules per Phase 195N",
    ),
    (
        "pipeline_real_data.py",
        "python_legacy_pipeline",
        "run_full_pipeline.py",
        "DISABLE_LATER_LEGACY_DUPLICATE",
        "Legacy real-data chain; not authoritative",
    ),
    (
        "Triton Daily",
        "windows_scheduled_task",
        "run_daily_refresh.ps1",
        "REDIRECT_LATER_TO_AUTHORITATIVE",
        "Persona legacy daily task",
    ),
    (
        "Triton Daily (ALL personas)",
        "windows_scheduled_task",
        "run_daily_refresh.ps1",
        "REDIRECT_LATER_TO_AUTHORITATIVE",
        "Persona legacy daily task",
    ),
    (
        "Triton Daily (simons)",
        "windows_scheduled_task",
        "run_daily_refresh.ps1",
        "REDIRECT_LATER_TO_AUTHORITATIVE",
        "Persona legacy daily task",
    ),
    (
        "scripts/auto_trade.py",
        "python_legacy_execution",
        "",
        "REVIEW_LATER_BYPASS_RISK",
        "Legacy auto entry; may bypass operator-only execute_trades gates",
    ),
    (
        "scripts/auto_exit.py",
        "python_legacy_execution",
        "",
        "REVIEW_LATER_BYPASS_RISK",
        "Legacy auto exit path",
    ),
    (
        "scripts/auto_execute_signals.py",
        "python_legacy_execution",
        "",
        "REVIEW_LATER_BYPASS_RISK",
        "Legacy signal execution",
    ),
    (
        "scripts/execute_trade.py",
        "python_legacy_execution",
        "",
        "REVIEW_LATER_BYPASS_RISK",
        "Legacy single-trade script",
    ),
    (
        "services/trade_executor.py",
        "python_legacy",
        "services/execute_trades.py",
        "REVIEW_LATER_BYPASS_RISK",
        "Review for gate bypass before isolation",
    ),
    (
        "services/portfolio_manager.py",
        "python_legacy",
        "services/manage_positions.py",
        "REVIEW_LATER_BYPASS_RISK",
        "Review for direct submission outside gates",
    ),
    (
        "alpaca_client.py",
        "python_legacy_execution",
        "services/execute_trades.py",
        "REVIEW_LATER_BYPASS_RISK",
        "Direct submit_order API; isolate if used outside gated paths",
    ),
    (
        "services/manage_positions.py",
        "python_execution",
        "",
        "KEEP_ACTIVE_MONITORING",
        "Plan-only exits/trims; no new entry submission per Phase 195M",
    ),
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


def _tracked_runtime_files() -> Set[str]:
    tracked: Set[str] = set()
    for prefix in ("data/live/", "data/results/", "data/processed/"):
        for line in _git_output("ls-files", "--", prefix).splitlines():
            if line.strip():
                tracked.add(line.strip().replace("\\", "/"))
    return tracked


def _modified_paths() -> Set[str]:
    modified: Set[str] = set()
    for line in _git_output("status", "--porcelain").splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip().strip('"').replace("\\", "/")
        if line[1] == "M" or line[0] == "M":
            modified.add(path)
    return modified


def _gitignore_matches(rel_path: str) -> bool:
    return bool(_git_output("check-ignore", "-v", rel_path))


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


def _fetch_scheduled_tasks() -> List[Dict[str, Any]]:
    ps = (
        "Get-ScheduledTask | Where-Object { "
        "$_.Actions.Execute -match 'triton' -or $_.Actions.Arguments -match 'triton' "
        "} | ForEach-Object { "
        "$info = Get-ScheduledTaskInfo $_; "
        "[PSCustomObject]@{ "
        "TaskName=$_.TaskName; State=$_.State; "
        "Execute=$_.Actions.Execute; Arguments=$_.Actions.Arguments; "
        "LastRunTime=$info.LastRunTime; LastTaskResult=$info.LastTaskResult "
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


def _classify_tracked_file(
    rel_path: str,
    *,
    loops_active: bool,
) -> Tuple[str, str, str, str, str, str]:
    norm = rel_path.replace("\\", "/")
    active_writer = norm in ACTIVE_LOOP_WRITERS

    if norm == "data/processed/.gitkeep":
        return (
            "KEEP_TRACKED_SOURCE_OR_CONFIG",
            "false",
            "true",
            "true",
            "Directory placeholder; not runtime noise",
            "# (no untrack — keep .gitkeep tracked)",
        )

    if norm in UNTRACK_LATER_LIVE:
        safe_restore = "false" if loops_active else "true"
        return (
            "UNTRACK_LATER_RUNTIME_LIVE_STATE",
            safe_restore,
            "true",
            "false",
            "Live coordination/lock state; never restore while loops active",
            f"# DO NOT RUN YET: git rm --cached -- '{norm}'",
        )

    if norm in UNTRACK_LATER_RUNTIME_RESULT or active_writer:
        safe_restore = "true" if not norm.startswith("data/live/") else "false"
        return (
            "UNTRACK_LATER_RUNTIME_RESULT",
            safe_restore,
            "true",
            "false",
            "Generated result rewritten by watchdog/readiness cycles",
            f"# DO NOT RUN YET: git rm --cached -- '{norm}'",
        )

    if norm.startswith("data/processed/") and norm.endswith(".parquet"):
        return (
            "UNTRACK_LATER_PROCESSED_DATA",
            "true",
            "true",
            "false",
            "Market data parquet; pipeline refresh output not source code",
            f"# DO NOT RUN YET: git rm --cached -- '{norm}'",
        )

    if norm in BASELINE_TRACKED_ARTIFACTS:
        return (
            "KEEP_TRACKED_BASELINE_EVIDENCE",
            "true",
            "false",
            "true",
            "Certification/validation baseline evidence committed for audit reference",
            "# (no untrack — keep tracked baseline evidence)",
        )

    if norm.startswith("data/results/") and any(
        k in norm for k in ("certification", "validation", "stress", "replay", "oversight")
    ):
        return (
            "KEEP_TRACKED_BASELINE_EVIDENCE",
            "true",
            "false",
            "true",
            "Historical validation/certification artifact",
            "# (no untrack — keep tracked baseline evidence)",
        )

    return (
        "NEEDS_OPERATOR_REVIEW",
        "false",
        "false",
        "false",
        "Did not match automated untrack/baseline rules",
        "# NEEDS_OPERATOR_REVIEW before any git rm --cached",
    )


def _build_file_plan(
    tracked: Set[str],
    modified: Set[str],
    loops_active: bool,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(tracked):
        classification, safe_restore, safe_untrack, remain, reason, cmd = _classify_tracked_file(
            path, loops_active=loops_active
        )
        rows.append(
            {
                "path": path,
                "plan_classification": classification,
                "tracked_by_git": "true",
                "ignored_by_gitignore": str(_gitignore_matches(path)).lower(),
                "currently_modified": str(path in modified).lower(),
                "active_writer": str(path in ACTIVE_LOOP_WRITERS).lower(),
                "safe_to_restore_now": safe_restore,
                "safe_to_untrack_later": safe_untrack,
                "should_remain_tracked": remain,
                "reason": reason,
                "proposed_later_command": cmd,
            }
        )
    return rows


def _match_static_entry(name: str, task: Optional[Dict[str, Any]]) -> Optional[Tuple[str, ...]]:
    for entry in STATIC_SCHEDULER_ENTRIES:
        if name == entry[0] or name.lower() == entry[0].lower():
            return entry
    if task:
        exe = str(task.get("Execute") or "").lower()
        args = str(task.get("Arguments") or "").lower()
        combined = exe + " " + args
        for entry in STATIC_SCHEDULER_ENTRIES:
            key = entry[0].replace("\\", "/").lower()
            if key in combined or Path(key).name.lower() in combined:
                return entry
    return None


def _build_scheduler_plan(tasks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen: Set[str] = set()
    task_by_name = {str(t.get("TaskName") or ""): t for t in tasks}

    def add_row(
        name: str,
        entry_type: str,
        classification: str,
        replacement: str,
        action: str,
        reason: str,
        task: Optional[Dict[str, Any]] = None,
    ) -> None:
        if name in seen:
            return
        seen.add(name)
        t = task or task_by_name.get(name)
        rows.append(
            {
                "entrypoint_or_task": name,
                "entrypoint_type": entry_type,
                "plan_classification": classification,
                "scheduled_task_name": name if t else "",
                "task_state": str((t or {}).get("State") or "N/A"),
                "last_run_time": str((t or {}).get("LastRunTime") or ""),
                "authoritative_replacement": replacement,
                "recommended_later_action": action,
                "do_not_execute_yet": "true",
                "reason": reason,
            }
        )

    for entry in STATIC_SCHEDULER_ENTRIES:
        name, etype, replacement, classification, reason = entry
        task = task_by_name.get(name)
        if classification == "KEEP_ACTIVE_AUTHORITATIVE":
            action = "DO NOT EXECUTE YET: No change — keep task/script active"
        elif classification == "KEEP_ACTIVE_MONITORING":
            action = "DO NOT EXECUTE YET: No change — keep monitoring loop/task active"
        elif classification == "KEEP_OPERATOR_ONLY":
            action = "DO NOT EXECUTE YET: No scheduler enablement — operator CLI only"
        elif classification == "DISABLE_LATER_LEGACY_DUPLICATE":
            action = (
                f"DO NOT EXECUTE YET: After validation, disable task/script; "
                f"confirm {replacement} covers workload"
            )
        elif classification == "REDIRECT_LATER_TO_AUTHORITATIVE":
            action = (
                f"DO NOT EXECUTE YET: Disable legacy task and confirm {replacement} "
                "is sole daily authority"
            )
        elif classification == "REVIEW_LATER_BYPASS_RISK":
            action = "DO NOT EXECUTE YET: Code review for gate bypass; then disable or archive"
        else:
            action = "DO NOT EXECUTE YET: Review overlap with TRITON_DailyRefresh before changes"
        add_row(name, etype, classification, replacement, action, reason, task)

    for task in tasks:
        name = str(task.get("TaskName") or "")
        if not name or name in seen:
            continue
        matched = _match_static_entry(name, task)
        if matched:
            continue
        add_row(
            name,
            "windows_scheduled_task",
            "DO_NOT_TOUCH_YET",
            "run_daily_refresh.ps1",
            "DO NOT EXECUTE YET: Unmapped Triton task — review before disable",
            f"Execute={task.get('Execute')} Arguments={task.get('Arguments')}",
            task,
        )

    return sorted(rows, key=lambda r: r["entrypoint_or_task"])


def _build_operator_commands(
    untrack_paths: List[str],
    excluded_paths: List[str],
    loops_active: bool,
) -> str:
    repo = str(ROOT).replace("'", "''")
    lines = [
        "=" * 72,
        "TRITON Phase 195S+ — Runtime Git Untrack & Scheduler Isolation",
        "DO NOT RUN THIS BLOCK UNTIL OPERATOR REVIEWS Phase 195R PLAN",
        "=" * 72,
        "",
        "# Preconditions (manual verification before any step):",
        "# - Broker: 0 open orders / 0 positions (snapshot_live_orders --mode paper)",
        "# - execution_readiness: READY",
        "# - Phase 195H: READY",
        "# - Document current commit SHA for rollback",
        f"# - Loops running: watchdog={loops_active} (stop NOT required for git rm --cached)",
        "# - NEVER git restore data/live/* while loops are active",
        "",
        "# --- SECTION A: Optional restore of safe result noise (loops may stay running) ---",
        "git restore --worktree \\",
        "  'data/results/executive_outlook_reconciliation.json' \\",
        "  'data/results/executive_outlook_scorecard.csv' \\",
        "  'data/results/lifecycle_readiness_diagnostics.json'",
        "",
        "# --- SECTION B: Untrack runtime artifacts (index only; files remain on disk) ---",
        "# DO NOT RUN YET — execute only in approved later phase (195S):",
        "",
    ]

    batch_size = 15
    for i in range(0, len(untrack_paths), batch_size):
        batch = untrack_paths[i : i + batch_size]
        quoted = " ".join(f"'{p}'" for p in batch)
        lines.append(f"git rm --cached -- {quoted}")

    lines.extend(
        [
            "",
            "# --- SECTION C: Verify index ---",
            "git status --short",
            "git diff --cached --name-only",
            "# Expected: only untracked-on-disk runtime paths removed from index",
            "",
            "# --- SECTION D: Commit hygiene commit (code + index cleanup only) ---",
            "# git add .gitignore services/",
            '# git commit -m "chore: untrack runtime artifacts from Git index (Phase 195S)"',
            "",
            "# --- SECTION E: Scheduler isolation (DO NOT EXECUTE YET) ---",
            "# Example — disable legacy duplicate AFTER confirming TRITON_DailyRefresh succeeds:",
            "# Disable-ScheduledTask -TaskName 'Triton Daily Refresh' -ErrorAction SilentlyContinue",
            "# Disable-ScheduledTask -TaskName 'TritonFetchDaily' -ErrorAction SilentlyContinue",
            "# Disable-ScheduledTask -TaskName 'Triton Daily' -ErrorAction SilentlyContinue",
            "# DO NOT disable: TRITON_DailyRefresh, TRITON_MarketOpen, TRITON_GuardAuto,",
            "#   TRITON_SnapshotLive, TRITON_EquityDailySnapshot",
            "",
            "# --- SECTION F: Post-cleanup validation (read-only) ---",
            f"& '{repo}\\.venv\\Scripts\\python.exe' -m services.execution_readiness",
            f"& '{repo}\\.venv\\Scripts\\python.exe' -m services.phase_195c_evaluation_contract_validation",
            f"& '{repo}\\.venv\\Scripts\\python.exe' -m services.phase_195k_certification_readiness_alignment_validation",
            f"& '{repo}\\.venv\\Scripts\\python.exe' -m services.phase_195h_operational_readiness_freeze",
            f"& '{repo}\\.venv\\Scripts\\python.exe' -m services.snapshot_live_orders --mode paper",
            "git status --short",
            "",
            "# --- EXCLUDED FROM git rm --cached (keep tracked baseline evidence) ---",
        ]
    )
    for p in excluded_paths:
        lines.append(f"# KEEP TRACKED: {p}")
    lines.append("")
    return "\n".join(lines)


def _rollback_plan(untrack_paths: List[str]) -> List[str]:
    return [
        "Record PRE_UNTRACK_COMMIT=$(git rev-parse HEAD) before Phase 195S index cleanup",
        "If untrack must be reversed: git checkout PRE_UNTRACK_COMMIT -- "
        + " ".join(untrack_paths[:5])
        + " ...",
        "Or: git add -f <path> for each runtime file to re-track individually",
        "Scheduler rollback: Re-Enable-ScheduledTask -TaskName '<name>' for any disabled legacy task",
        "Runtime files on disk are never deleted by git rm --cached — only index entries removed",
        "Do not git restore data/live/* while watchdog/continuous loops are running",
    ]


def _validation_plan() -> List[str]:
    return [
        "python -m services.execution_readiness → readiness_status READY, live_execution_permitted false",
        "python -m services.phase_195c_evaluation_contract_validation → phase_pass true",
        "python -m services.phase_195k_certification_readiness_alignment_validation → phase_pass true",
        "python -m services.phase_195h_operational_readiness_freeze → operational_freeze_status READY",
        "python -m services.snapshot_live_orders --mode paper → open_orders_written 0, positions_written 0",
        "git status --short → no modified data/live/* or data/results/* in commit staging",
        "Confirm execute_trades --execute not running (Get-Process python)",
        "Confirm watchdog and continuous loops still healthy (lock + heartbeat)",
        "Do NOT place orders or enable automated entry execution during validation",
    ]


def run_phase_195r_plan(*, verbose: bool = True) -> Dict[str, Any]:
    audit_195q = _load_json(AUDIT_195Q)
    tracked = _tracked_runtime_files()
    modified = _modified_paths()
    watchdog_running, continuous_running = _loops_running()
    loops_active = watchdog_running or continuous_running
    execute_detected, execute_cmds = _find_python_execute_trades_execute()
    readiness = _load_json(EXEC_READINESS)
    phase_195h = _load_json(PHASE_195H_JSON)
    paper_cfg = _load_json(PAPER_CYCLE_CFG)

    file_plan = _build_file_plan(tracked, modified, loops_active)
    tasks = _fetch_scheduled_tasks()
    scheduler_plan = _build_scheduler_plan(tasks)

    untrack_paths = sorted(
        r["path"] for r in file_plan if r["plan_classification"].startswith("UNTRACK_LATER")
    )
    excluded_paths = sorted(r["path"] for r in file_plan if r["should_remain_tracked"] == "true")
    review_paths = sorted(
        r["path"] for r in file_plan if r["plan_classification"] == "NEEDS_OPERATOR_REVIEW"
    )

    operator_commands = _build_operator_commands(untrack_paths, excluded_paths, loops_active)
    rollback = _rollback_plan(untrack_paths)
    validation = _validation_plan()

    never_restore = sorted(UNTRACK_LATER_LIVE)

    payload: Dict[str, Any] = {
        "phase": "195R",
        "timestamp": _utc_now_iso(),
        "phase_pass": True,
        "audit_mode": "planning_only",
        "orders_placed": False,
        "scheduled_tasks_changed": False,
        "git_rm_cached_executed": False,
        "safety_statement": (
            "Phase 195R is planning-only. No orders were placed, no locks were deleted, "
            "no data/live files were restored, git rm --cached was not executed, and no "
            "Windows Scheduled Tasks were edited, disabled, or deleted."
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
        "195q_reference": {
            "tracked_runtime_files_count": audit_195q.get("tracked_runtime_files_count"),
            "active_loop_writers_tracked": audit_195q.get("git_hygiene_answers", {}).get(
                "2_tracked_files_rewritten_by_active_loops"
            ),
        },
        "untrack_plan_summary": {
            "total_tracked_runtime_files": len(tracked),
            "proposed_git_rm_cached_count": len(untrack_paths),
            "excluded_from_git_rm_cached_count": len(excluded_paths),
            "needs_operator_review_count": len(review_paths),
            "untrack_later_runtime_live_state_count": sum(
                1
                for r in file_plan
                if r["plan_classification"] == "UNTRACK_LATER_RUNTIME_LIVE_STATE"
            ),
            "untrack_later_runtime_result_count": sum(
                1 for r in file_plan if r["plan_classification"] == "UNTRACK_LATER_RUNTIME_RESULT"
            ),
            "untrack_later_processed_data_count": sum(
                1 for r in file_plan if r["plan_classification"] == "UNTRACK_LATER_PROCESSED_DATA"
            ),
            "keep_tracked_baseline_evidence_count": sum(
                1 for r in file_plan if r["plan_classification"] == "KEEP_TRACKED_BASELINE_EVIDENCE"
            ),
        },
        "exact_files_proposed_for_later_git_rm_cached": untrack_paths,
        "exact_files_excluded_from_git_rm_cached": excluded_paths,
        "files_needs_operator_review": review_paths,
        "never_restore_while_loops_active": never_restore,
        "operator_command_block_do_not_run_yet": operator_commands,
        "scheduler_isolation_recommendations_do_not_execute_yet": [
            r for r in scheduler_plan if r["do_not_execute_yet"] == "true"
        ],
        "rollback_plan": rollback,
        "validation_plan_after_cleanup": validation,
        "later_execution_phase": "Phase 195S — Runtime Index Untrack & Legacy Scheduler Isolation (operator-approved)",
        "file_plan": file_plan,
        "scheduler_plan": scheduler_plan,
        "windows_scheduled_tasks_observed": tasks,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(FILE_PLAN_CSV, file_plan, FILE_PLAN_FIELDS)
    _write_csv(SCHEDULER_PLAN_CSV, scheduler_plan, SCHEDULER_PLAN_FIELDS)
    _write_text(OPERATOR_COMMANDS_TXT, operator_commands)

    if verbose:
        print(f"[195R] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[195R] proposed git rm --cached: {len(untrack_paths)} files")
        print(f"[195R] excluded (keep tracked): {len(excluded_paths)} files")
        print(f"[195R] scheduler entries planned: {len(scheduler_plan)}")
        print("[195R] DO NOT RUN operator commands until operator approves Phase 195S")

    return payload


def main() -> None:
    run_phase_195r_plan(verbose=True)


if __name__ == "__main__":
    main()
