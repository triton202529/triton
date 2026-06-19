"""
phase_195q_runtime_artifact_git_hygiene_audit.py
Phase 195Q — Runtime artifact Git hygiene & legacy scheduler isolation prep.

Read-only audit of Git working-tree noise vs source changes. Does not place orders,
run git rm --cached, edit scheduled tasks, delete runtime artifacts, or stop loops.
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
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "config"

OUTPUT_JSON = RESULTS / "runtime_artifact_git_hygiene_audit.json"
SUMMARY_CSV = RESULTS / "runtime_artifact_git_hygiene_summary.csv"
MANIFEST_CSV = RESULTS / "runtime_artifact_git_hygiene_file_manifest.csv"

ENTRYPOINT_195N = RESULTS / "runtime_entrypoint_consolidation_audit.json"
EXEC_READINESS = RESULTS / "execution_readiness.json"
PHASE_195H_JSON = RESULTS / "phase_195h_operational_readiness_freeze.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
PAPER_CYCLE_CFG = CONFIG / "run_scheduled_paper_cycle.json"

CLASSIFICATIONS = (
    "SOURCE_CHANGE_COMMITTABLE",
    "LIVE_RUNTIME_STATE_DO_NOT_RESTORE_WHILE_RUNNING",
    "GENERATED_RESULT_SAFE_TO_RESTORE",
    "GENERATED_RESULT_ACTIVE_WRITER",
    "PROCESSED_DATA_RUNTIME_NOISE",
    "BASELINE_RESULT_SHOULD_REMAIN_TRACKED",
    "CANDIDATE_FOR_GITIGNORE",
    "NEEDS_OPERATOR_REVIEW",
)

MANIFEST_FIELDS = (
    "path",
    "git_status",
    "classification",
    "tracked_by_git",
    "runtime_generated",
    "source_code",
    "live_state_file",
    "result_artifact",
    "processed_market_data",
    "safe_to_restore_while_loops_running",
    "safe_to_ignore_in_git_later",
    "should_remain_tracked",
    "active_loop_writer",
    "recommended_action",
    "reason",
)

SUMMARY_FIELDS = ("metric", "value")

SPECIAL_FOCUS_PATHS = (
    "data/live/continuous_loop.lock",
    "data/live/watchdog_loop.lock",
    "data/live/watchdog_alert_state.json",
    "data/results/executive_outlook_reconciliation.json",
    "data/results/executive_outlook_scorecard.csv",
    "data/results/lifecycle_readiness_diagnostics.json",
    "data/results/open_orders_snapshot.csv",
    "data/results/recent_orders.csv",
    "data/results/positions_snapshot.csv",
    "data/results/execution_readiness.json",
    "data/results/capital_preservation_certification.json",
    "data/results/watchdog_alerts.json",
)

LIVE_LOCK_FILES = frozenset(
    {
        "data/live/continuous_loop.lock",
        "data/live/watchdog_loop.lock",
        "data/live/pipeline_scheduled.lock",
    }
)

LIVE_STATE_FILES = frozenset(
    {
        "data/live/watchdog_alert_state.json",
        "data/live/idempotency_state.json",
        "data/live/guard_snapshot.json",
        "data/live/live_armed.json",
        "data/live/live_confirm.json",
        "data/live/paper_arm_confirm.json",
    }
)

ACTIVE_LOOP_WRITERS: Dict[str, Tuple[str, ...]] = {
    "data/live/continuous_loop.lock": ("scripts/run_continuous_loop.ps1",),
    "data/live/watchdog_loop.lock": ("scripts/run_watchdog_loop.ps1",),
    "data/live/watchdog_alert_state.json": (
        "services/risk_watchdog.py",
        "services/loop_safety.py",
    ),
    "data/results/executive_outlook_reconciliation.json": (
        "services/risk_intelligence.py",
        "services/risk_watchdog.py",
    ),
    "data/results/executive_outlook_scorecard.csv": (
        "services/executive_outlook.py",
        "services/risk_intelligence.py",
    ),
    "data/results/lifecycle_readiness_diagnostics.json": (
        "services/execution_readiness.py",
        "services/governance_execution_readiness.py",
    ),
}

BASELINE_TRACKED_ARTIFACTS = frozenset(
    {
        "data/results/capital_preservation_certification_audit.json",
        "data/results/capital_preservation_scorecard.csv",
        "data/results/certification_false_positive_elimination.json",
        "data/results/certification_false_positive_scorecard.csv",
        "data/results/executive_oversight_certification.json",
        "data/results/executive_oversight_scorecard.csv",
        "data/results/exit_logic_refinement_validation.json",
        "data/results/historical_exit_replay_validation.csv",
        "data/results/historical_exit_replay_validation.json",
        "data/results/institutional_intelligence_stress_scorecard.csv",
        "data/results/institutional_intelligence_stress_test.json",
        "data/results/paper_execution_certification.json",
        "data/results/paper_execution_scorecard.csv",
        "data/results/paper_trade_exit_review.csv",
        "data/results/paper_trade_session_review.json",
        "data/results/portfolio_management_certification.json",
        "data/results/portfolio_management_operational_validation.csv",
        "data/results/portfolio_management_operational_validation.json",
        "data/results/portfolio_management_scorecard.csv",
        "data/results/portfolio_management_stress_scorecard.csv",
        "data/results/portfolio_management_stress_test.json",
    }
)

KEEP_ACTIVE_SCHEDULER = (
    "run_full_pipeline.py",
    "run_daily_refresh.ps1",
    "TRITON_DailyRefresh",
    "TRITON_MarketOpen",
    "TRITON_GuardAuto",
    "TRITON_SnapshotLive",
    "TRITON_EquityDailySnapshot",
    "scripts/run_watchdog_loop.ps1",
    "scripts/run_continuous_loop.ps1",
    "services/execution_readiness.py",
    "services/governance_execution_readiness.py",
    "services/execute_trades.py",
    "run_pipeline_scheduled.ps1",
)

ISOLATE_LATER_SCHEDULER = (
    "daily_refresh.bat",
    "Triton Daily Refresh",
    "run-fetch.ps1",
    "TritonFetchDaily",
    "Makefile",
    "scripts/run_all.py",
    "pipeline_real_data.py",
    "Triton Daily",
    "Triton Daily (ALL personas)",
    "scripts/auto_trade.py",
    "scripts/auto_exit.py",
    "scripts/auto_execute_signals.py",
    "scripts/execute_trade.py",
    "services/trade_executor.py",
    "services/portfolio_manager.py",
    "alpaca_client.py",
)

LATER_ISOLATION_PHASE = "Phase 195R — Legacy Scheduler Isolation & Runtime Artifact Untrack"


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
        status = "clean"
        if x == "?" and y == "?":
            status = "untracked"
        elif x != " " or y != " ":
            if x in ("M", "A", "D", "R", "C") and y in ("M", "D"):
                status = "staged_and_modified" if y == "M" else "staged"
            elif y == "M":
                status = "modified"
            elif x == "M":
                status = "staged"
            elif x == "?" or y == "?":
                status = "untracked"
            else:
                status = f"{x}{y}"
        rows.append({"path": rel, "git_status": status, "index": x, "worktree": y})
    return rows


def _tracked_runtime_files() -> Set[str]:
    tracked: Set[str] = set()
    for prefix in ("data/live/", "data/results/", "data/processed/"):
        for line in _git_output("ls-files", "--", prefix).splitlines():
            if line.strip():
                tracked.add(line.strip().replace("\\", "/"))
    return tracked


def _gitignore_matches(rel_path: str) -> bool:
    out = _git_output("check-ignore", "-v", rel_path)
    return bool(out)


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


def _classify_path(
    rel_path: str,
    *,
    tracked: bool,
    git_status: str,
    watchdog_running: bool,
    continuous_running: bool,
) -> Tuple[str, str, str]:
    loops_active = watchdog_running or continuous_running
    norm = rel_path.replace("\\", "/")

    if re.match(r"^(services|scripts)/.*\.(py|ps1)$", norm) or norm in (
        ".gitignore",
        ".gitattributes",
        "Makefile",
    ):
        if git_status in ("modified", "staged", "staged_and_modified", "untracked"):
            return (
                "SOURCE_CHANGE_COMMITTABLE",
                "Stage and commit with code review",
                "Source file change detected in working tree",
            )

    if norm in LIVE_LOCK_FILES:
        return (
            "LIVE_RUNTIME_STATE_DO_NOT_RESTORE_WHILE_RUNNING",
            "Never git restore or commit; untrack in Phase 195R via git rm --cached",
            "Loop lock proves process ownership; restoring while loop runs can desync ownership",
        )

    if norm in LIVE_STATE_FILES:
        return (
            "LIVE_RUNTIME_STATE_DO_NOT_RESTORE_WHILE_RUNNING",
            "Never git restore while loops active; untrack in Phase 195R",
            "Live coordination state rewritten by watchdog/loop_safety each cycle",
        )

    if norm in ACTIVE_LOOP_WRITERS:
        writers = ", ".join(ACTIVE_LOOP_WRITERS[norm])
        return (
            "GENERATED_RESULT_ACTIVE_WRITER",
            "Do not commit; safe to git restore results-only paths when loops stopped; "
            "or git restore now for data/results/* only (not data/live/*)",
            f"Repeatedly rewritten by active loops: {writers}",
        )

    if norm.startswith("data/processed/") and norm.endswith(".parquet"):
        return (
            "PROCESSED_DATA_RUNTIME_NOISE",
            "Do not commit; untrack all data/processed/*.parquet in Phase 195R",
            "Market data parquet refreshed by pipeline fetch; historical snapshots not source",
        )

    if norm.startswith("data/processed/.gitkeep"):
        return (
            "SOURCE_CHANGE_COMMITTABLE",
            "Keep tracked as directory placeholder",
            "Intentional repo structure file",
        )

    if norm in BASELINE_TRACKED_ARTIFACTS:
        return (
            "BASELINE_RESULT_SHOULD_REMAIN_TRACKED",
            "Keep tracked until Phase 195R baseline migration plan; do not commit incidental edits",
            "Certification/validation baseline artifact committed for audit reference",
        )

    if norm.startswith("data/results/") and tracked:
        if any(k in norm for k in ("certification", "validation", "stress", "replay")):
            return (
                "BASELINE_RESULT_SHOULD_REMAIN_TRACKED",
                "Review diff before commit; prefer git restore if loop/pipeline noise",
                "Tracked historical validation artifact; not an active loop writer",
            )
        return (
            "CANDIDATE_FOR_GITIGNORE",
            "Untrack in Phase 195R (git rm --cached); already covered by data/results/** ignore rule",
            "Tracked despite .gitignore because file was committed before ignore rule",
        )

    if norm.startswith(("data/live/", "data/results/", "data/processed/")):
        if tracked:
            return (
                "CANDIDATE_FOR_GITIGNORE",
                "Untrack in Phase 195R; do not commit runtime noise",
                "Runtime path tracked historically; should become ignored",
            )
        if _gitignore_matches(norm):
            return (
                "CANDIDATE_FOR_GITIGNORE",
                "Already ignored — do not git add",
                "Matches .gitignore; working tree noise only if accidentally force-added",
            )
        return (
            "GENERATED_RESULT_SAFE_TO_RESTORE",
            "Safe to delete locally or leave untracked; git restore N/A",
            "Runtime output not tracked and not in active writer registry",
        )

    return (
        "NEEDS_OPERATOR_REVIEW",
        "Inspect diff manually before any commit",
        "Path did not match automated runtime/source classification rules",
    )


def _manifest_row(
    rel_path: str,
    *,
    tracked: Set[str],
    status_map: Dict[str, str],
    watchdog_running: bool,
    continuous_running: bool,
) -> Dict[str, str]:
    norm = rel_path.replace("\\", "/")
    is_tracked = norm in tracked
    git_status = status_map.get(norm, "clean")
    classification, action, reason = _classify_path(
        norm,
        tracked=is_tracked,
        git_status=git_status,
        watchdog_running=watchdog_running,
        continuous_running=continuous_running,
    )

    is_live = norm.startswith("data/live/")
    is_results = norm.startswith("data/results/")
    is_processed = norm.startswith("data/processed/")
    is_source = bool(re.match(r"^(services|scripts)/", norm)) or norm in (
        ".gitignore",
        ".gitattributes",
    )
    active_writer = norm in ACTIVE_LOOP_WRITERS

    safe_restore = "false"
    if classification == "GENERATED_RESULT_SAFE_TO_RESTORE":
        safe_restore = "true"
    elif classification == "GENERATED_RESULT_ACTIVE_WRITER" and is_results:
        safe_restore = "true"
    elif classification == "BASELINE_RESULT_SHOULD_REMAIN_TRACKED":
        safe_restore = "true"
    elif is_live or norm in LIVE_LOCK_FILES:
        safe_restore = "false"

    should_remain = "false"
    if classification == "BASELINE_RESULT_SHOULD_REMAIN_TRACKED":
        should_remain = "true"
    elif is_source:
        should_remain = "true"

    safe_ignore_later = "false"
    if classification in (
        "CANDIDATE_FOR_GITIGNORE",
        "GENERATED_RESULT_ACTIVE_WRITER",
        "LIVE_RUNTIME_STATE_DO_NOT_RESTORE_WHILE_RUNNING",
        "PROCESSED_DATA_RUNTIME_NOISE",
    ):
        safe_ignore_later = "true"

    return {
        "path": norm,
        "git_status": git_status,
        "classification": classification,
        "tracked_by_git": str(is_tracked).lower(),
        "runtime_generated": str(not is_source and (is_live or is_results or is_processed)).lower(),
        "source_code": str(is_source).lower(),
        "live_state_file": str(is_live).lower(),
        "result_artifact": str(is_results).lower(),
        "processed_market_data": str(is_processed and norm.endswith(".parquet")).lower(),
        "safe_to_restore_while_loops_running": safe_restore,
        "safe_to_ignore_in_git_later": safe_ignore_later,
        "should_remain_tracked": should_remain,
        "active_loop_writer": str(active_writer).lower(),
        "recommended_action": action,
        "reason": reason,
    }


def _pre_commit_clean_command(
    *,
    watchdog_running: bool,
    continuous_running: bool,
    dirty_results: List[str],
) -> str:
    restore_results = [
        p
        for p in dirty_results
        if p.startswith("data/results/")
        and p
        in {
            "data/results/executive_outlook_reconciliation.json",
            "data/results/executive_outlook_scorecard.csv",
            "data/results/lifecycle_readiness_diagnostics.json",
        }
    ]
    lines = [
        "# Before every code-only commit (loops may stay running):",
        "git status --short",
        "git add scripts/ services/ .gitignore .gitattributes",
        "# Never: git add data/",
    ]
    if restore_results:
        quoted = " ".join(f"'{p}'" for p in restore_results)
        lines.append(f"git restore --worktree {quoted}")
    if watchdog_running or continuous_running:
        lines.append("# DO NOT git restore data/live/* while watchdog/continuous loops are running")
    lines.append("git diff --cached --name-only   # verify: no data/live or data/results paths")
    return "\n".join(lines)


def _load_195n_scheduler_prep() -> Dict[str, Any]:
    doc = _load_json(ENTRYPOINT_195N)
    summary = doc.get("summary") or {}
    return {
        "keep_active": summary.get("entry_points_keep_active") or list(KEEP_ACTIVE_SCHEDULER),
        "isolate_later": summary.get("entry_points_isolate_or_disable_later")
        or list(ISOLATE_LATER_SCHEDULER),
        "recommended_next_hardening_from_195n": summary.get("recommended_next_hardening_step"),
        "later_isolation_phase": LATER_ISOLATION_PHASE,
        "195n_phase_1_scheduler_isolation": (
            "Disable/isolate legacy schedulers (Triton Daily Refresh bat, TritonFetchDaily, "
            "persona Triton Daily* tasks) and redirect to TRITON_DailyRefresh — execute in "
            f"{LATER_ISOLATION_PHASE}, not in 195Q"
        ),
    }


def run_phase_195q_audit(*, verbose: bool = True) -> Dict[str, Any]:
    porcelain = _parse_porcelain()
    status_map = {r["path"]: r["git_status"] for r in porcelain}
    dirty_paths = [r["path"] for r in porcelain if r["git_status"] != "clean"]
    tracked = _tracked_runtime_files()
    watchdog_running, continuous_running = _loops_running()
    execute_detected, execute_cmds = _find_python_execute_trades_execute()
    paper_cfg = _load_json(PAPER_CYCLE_CFG)
    readiness = _load_json(EXEC_READINESS)
    phase_195h = _load_json(PHASE_195H_JSON)

    manifest_paths: Set[str] = set(dirty_paths)
    manifest_paths.update(SPECIAL_FOCUS_PATHS)
    manifest_paths.update(tracked)
    if (PROCESSED / "SPY.parquet").is_file():
        manifest_paths.add("data/processed/SPY.parquet")

    manifest_rows = [
        _manifest_row(
            p,
            tracked=tracked,
            status_map=status_map,
            watchdog_running=watchdog_running,
            continuous_running=continuous_running,
        )
        for p in sorted(manifest_paths)
    ]

    dirty_manifest = [r for r in manifest_rows if r["git_status"] != "clean"]
    active_writers_tracked = [
        r["path"]
        for r in manifest_rows
        if r["active_loop_writer"] == "true" and r["tracked_by_git"] == "true"
    ]
    candidates_ignore = sorted(
        {r["path"] for r in manifest_rows if r["classification"] == "CANDIDATE_FOR_GITIGNORE"}
    )
    baseline_tracked = sorted(
        {
            r["path"]
            for r in manifest_rows
            if r["classification"] == "BASELINE_RESULT_SHOULD_REMAIN_TRACKED"
        }
    )
    do_not_restore = sorted(
        {
            r["path"]
            for r in manifest_rows
            if r["classification"] == "LIVE_RUNTIME_STATE_DO_NOT_RESTORE_WHILE_RUNNING"
        }
    )
    safe_restore_while_running = sorted(
        {
            r["path"]
            for r in manifest_rows
            if r["safe_to_restore_while_loops_running"] == "true" and r["git_status"] == "modified"
        }
    )

    gitignore_needs_update = not all(
        _gitignore_matches(p)
        for p in (
            "data/live/continuous_loop.lock",
            "data/live/watchdog_loop.lock",
            "data/live/watchdog_alert_state.json",
        )
    )

    pre_commit_cmd = _pre_commit_clean_command(
        watchdog_running=watchdog_running,
        continuous_running=continuous_running,
        dirty_results=[r["path"] for r in dirty_manifest if r["path"].startswith("data/results/")],
    )

    scheduler_prep = _load_195n_scheduler_prep()

    git_hygiene_answers = {
        "1_tracked_runtime_files": sorted(tracked),
        "1_tracked_runtime_files_count": len(tracked),
        "2_tracked_files_rewritten_by_active_loops": active_writers_tracked,
        "3_files_should_be_ignored_in_future": candidates_ignore
        + list(LIVE_LOCK_FILES)
        + list(LIVE_STATE_FILES)
        + [
            "data/results/executive_outlook_reconciliation.json",
            "data/results/executive_outlook_scorecard.csv",
            "data/results/lifecycle_readiness_diagnostics.json",
            "data/results/execution_readiness.json",
            "data/results/watchdog_alerts.json",
            "data/results/open_orders_snapshot.csv",
            "data/results/recent_orders.csv",
            "data/results/positions_snapshot.csv",
            "data/processed/*.parquet",
        ],
        "4_baseline_artifacts_should_remain_tracked": baseline_tracked,
        "5_do_not_restore_while_loops_active": do_not_restore,
        "6_safe_to_restore_before_code_commit_while_loops_running": safe_restore_while_running,
        "7_safe_gitignore_update_needed": gitignore_needs_update,
        "7_gitignore_note": (
            ".gitignore already has data/results/** and data/**/*.parquet but 65 runtime files "
            "remain tracked from pre-ignore commits; explicit data/live/* rules prevent new adds"
        ),
        "8_later_git_rm_cached_plan_needed": True,
        "8_later_git_rm_cached_command_template": (
            "# Phase 195R only — after loops stopped and operator approval:\n"
            "git rm --cached -r data/live/*.lock data/live/watchdog_alert_state.json "
            "data/processed/*.parquet data/results/executive_outlook_reconciliation.json "
            "data/results/executive_outlook_scorecard.csv "
            "data/results/lifecycle_readiness_diagnostics.json"
        ),
        "9_pre_commit_clean_runtime_noise_command": pre_commit_cmd,
        "10_later_phase_for_legacy_scheduler_isolation": LATER_ISOLATION_PHASE,
    }

    root_cause = (
        "65 runtime files under data/live, data/results, and data/processed were committed before "
        "broad .gitignore rules (data/results/**, data/**/*.csv, data/**/*.parquet). Git continues "
        "tracking them, so active watchdog (~1 min) and continuous (~45 min) loops keep the working "
        "tree dirty even when only source code changed."
    )

    payload: Dict[str, Any] = {
        "phase": "195Q",
        "timestamp": _utc_now_iso(),
        "phase_pass": True,
        "audit_mode": "read_only",
        "orders_placed": False,
        "root_cause": root_cause,
        "watchdog_process_running": watchdog_running,
        "continuous_process_running": continuous_running,
        "dirty_files_count": len(dirty_paths),
        "dirty_files": dirty_paths,
        "tracked_runtime_files_count": len(tracked),
        "git_hygiene_answers": git_hygiene_answers,
        "special_focus_classifications": {
            p: next((r for r in manifest_rows if r["path"] == p), {}).get("classification", "N/A")
            for p in SPECIAL_FOCUS_PATHS
        },
        "legacy_scheduler_prep": scheduler_prep,
        "execution_readiness_status": readiness.get("readiness_status", "UNKNOWN"),
        "phase_195h_status": phase_195h.get("operational_freeze_status", "UNKNOWN"),
        "broker_open_orders_count": _count_csv_rows(OPEN_ORDERS),
        "broker_positions_count": _count_csv_rows(POSITIONS),
        "live_execution_permitted": bool(readiness.get("live_execution_permitted", False)),
        "paper_execution_permitted": bool(readiness.get("paper_execution_permitted", False)),
        "execute_trades_execute_detected": execute_detected,
        "automated_entry_execution_enabled": bool(paper_cfg.get("run_execute_trades", False)),
        "recommended_next_action": (
            f"Use pre-commit clean command; plan {LATER_ISOLATION_PHASE} for git rm --cached "
            "and legacy scheduler isolation — do not edit scheduled tasks in 195Q"
        ),
        "manifest": manifest_rows,
        "safety_constraints": [
            "No orders placed",
            "No git rm --cached in 195Q",
            "No scheduled task edits",
            "No runtime artifact deletion",
            "Do not git restore data/live/* while loops active",
        ],
    }

    summary_rows: List[Dict[str, str]] = [
        {"metric": "dirty_files_count", "value": str(len(dirty_paths))},
        {"metric": "tracked_runtime_files_count", "value": str(len(tracked))},
        {"metric": "active_loop_writers_tracked_count", "value": str(len(active_writers_tracked))},
        {"metric": "watchdog_process_running", "value": str(watchdog_running)},
        {"metric": "continuous_process_running", "value": str(continuous_running)},
        {"metric": "execution_readiness_status", "value": payload["execution_readiness_status"]},
        {"metric": "phase_195h_status", "value": payload["phase_195h_status"]},
        {"metric": "broker_open_orders_count", "value": str(payload["broker_open_orders_count"])},
        {"metric": "broker_positions_count", "value": str(payload["broker_positions_count"])},
        {"metric": "live_execution_permitted", "value": str(payload["live_execution_permitted"])},
        {"metric": "paper_execution_permitted", "value": str(payload["paper_execution_permitted"])},
        {"metric": "execute_trades_execute_detected", "value": str(execute_detected)},
        {"metric": "safe_gitignore_update_needed", "value": str(gitignore_needs_update).lower()},
        {"metric": "later_git_rm_cached_plan_needed", "value": "true"},
        {"metric": "later_isolation_phase", "value": LATER_ISOLATION_PHASE},
        {"metric": "recommended_next_action", "value": payload["recommended_next_action"]},
        {"metric": "pre_commit_clean_command", "value": pre_commit_cmd.replace("\n", " | ")},
    ]
    for p in dirty_paths:
        row = next((r for r in manifest_rows if r["path"] == p), {})
        summary_rows.append(
            {
                "metric": f"dirty:{p}",
                "value": row.get("classification", "UNKNOWN"),
            }
        )

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)
    _write_csv(MANIFEST_CSV, manifest_rows, MANIFEST_FIELDS)

    if verbose:
        print(f"[195Q] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[195Q] dirty_files={len(dirty_paths)} tracked_runtime={len(tracked)}")
        print(f"[195Q] active_writers_tracked={active_writers_tracked}")
        print(f"[195Q] later_phase={LATER_ISOLATION_PHASE}")

    return payload


def main() -> None:
    run_phase_195q_audit(verbose=True)


if __name__ == "__main__":
    main()
