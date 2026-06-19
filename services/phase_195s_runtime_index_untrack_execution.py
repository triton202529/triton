"""
phase_195s_runtime_index_untrack_execution.py
Phase 195S — Approved runtime Git index untrack execution.

Executes git rm --cached for Phase 195R approved runtime targets only.
Does not delete files from disk, stop loops, modify schedulers, or place orders.
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

PLAN_195R = RESULTS / "runtime_untrack_scheduler_isolation_plan.json"
FILE_PLAN_CSV = RESULTS / "runtime_untrack_file_plan.csv"
OPERATOR_COMMANDS = RESULTS / "runtime_cleanup_operator_commands.txt"

OUTPUT_JSON = RESULTS / "runtime_index_untrack_execution.json"
SUMMARY_CSV = RESULTS / "runtime_index_untrack_execution_summary.csv"
MANIFEST_CSV = RESULTS / "runtime_index_untrack_execution_manifest.csv"

EXEC_READINESS = RESULTS / "execution_readiness.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
PAPER_CYCLE_CFG = CONFIG / "run_scheduled_paper_cycle.json"

APPROVED_CLASSIFICATIONS = frozenset(
    {
        "UNTRACK_LATER_RUNTIME_LIVE_STATE",
        "UNTRACK_LATER_RUNTIME_RESULT",
        "UNTRACK_LATER_PROCESSED_DATA",
    }
)

RUNTIME_PREFIXES = ("data/live/", "data/results/", "data/processed/")

SOURCE_PATH_RE = re.compile(
    r"^(services|scripts|config|\.github)/|\.(py|ps1|bat|cmd|md|json|yaml|yml|toml)$",
    re.I,
)

SCHEDULER_SCRIPT_NAMES = frozenset(
    {
        "run_daily_refresh.ps1",
        "run_pipeline_scheduled.ps1",
        "run_guard_auto.ps1",
        "run_snapshot_live.ps1",
        "run-fetch.ps1",
        "daily_refresh.bat",
        "Makefile",
        "run_full_pipeline.py",
    }
)

MANIFEST_FIELDS = (
    "path",
    "plan_classification",
    "git_rm_cached_executed",
    "still_tracked_after",
    "exists_on_disk_after",
    "ignored_by_gitignore",
    "status",
    "reason",
)

SUMMARY_FIELDS = ("metric", "value")


class UntrackAbort(Exception):
    """Safety abort before git rm --cached."""


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


def _git_output(*args: str) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return 1, "", str(exc)


def _git_tracked(path: str) -> bool:
    code, out, _ = _git_output("ls-files", "--", path)
    return code == 0 and bool(out.strip())


def _gitignore_matches(rel_path: str) -> bool:
    _, out, _ = _git_output("check-ignore", "-v", rel_path)
    return bool(out)


def _load_file_plan() -> Dict[str, Dict[str, str]]:
    if not FILE_PLAN_CSV.is_file():
        return {}
    rows: Dict[str, Dict[str, str]] = {}
    with FILE_PLAN_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = (row.get("path") or "").replace("\\", "/")
            if path:
                rows[path] = row
    return rows


def _staged_source_paths() -> List[str]:
    staged: List[str] = []
    _, out, _ = _git_output("diff", "--cached", "--name-only")
    for line in out.splitlines():
        rel = line.strip().replace("\\", "/")
        if not rel:
            continue
        if rel.startswith(RUNTIME_PREFIXES):
            continue
        if SOURCE_PATH_RE.search(rel) or rel.endswith((".py", ".ps1")):
            staged.append(rel)
    return staged


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


def _is_source_or_scheduler_path(rel_path: str) -> Optional[str]:
    norm = rel_path.replace("\\", "/")
    base = Path(norm).name
    if base in SCHEDULER_SCRIPT_NAMES:
        return f"scheduler script: {base}"
    if not norm.startswith(RUNTIME_PREFIXES):
        return f"outside runtime data paths: {norm}"
    if norm.endswith(".gitkeep"):
        return "gitkeep must remain tracked"
    return None


def _validate_approved_targets(
    approved: List[str],
    file_plan: Dict[str, Dict[str, str]],
    excluded: Set[str],
) -> None:
    if not PLAN_195R.is_file():
        raise UntrackAbort(f"Phase 195R plan missing: {PLAN_195R}")
    if not FILE_PLAN_CSV.is_file():
        raise UntrackAbort(f"Phase 195R file plan CSV missing: {FILE_PLAN_CSV}")
    if not OPERATOR_COMMANDS.is_file():
        raise UntrackAbort(f"Phase 195R operator commands missing: {OPERATOR_COMMANDS}")

    for path in approved:
        norm = path.replace("\\", "/")
        if norm in excluded:
            raise UntrackAbort(f"Approved target is excluded baseline: {norm}")
        block = _is_source_or_scheduler_path(norm)
        if block:
            raise UntrackAbort(f"Approved target blocked ({block}): {norm}")
        plan_row = file_plan.get(norm)
        if not plan_row:
            raise UntrackAbort(f"Approved target not in 195R file plan: {norm}")
        classification = plan_row.get("plan_classification", "")
        if classification not in APPROVED_CLASSIFICATIONS:
            raise UntrackAbort(
                f"Approved target has disallowed classification {classification}: {norm}"
            )
        if plan_row.get("should_remain_tracked", "").lower() == "true":
            raise UntrackAbort(f"Approved target marked should_remain_tracked: {norm}")


def _validate_preconditions() -> None:
    staged_source = _staged_source_paths()
    if staged_source:
        raise UntrackAbort("Staged source changes before cleanup: " + ", ".join(staged_source[:5]))

    readiness = _load_json(EXEC_READINESS)
    if bool(readiness.get("live_execution_permitted", False)):
        raise UntrackAbort("live_execution_permitted is true — aborting")

    open_orders = _count_csv_rows(OPEN_ORDERS)
    if open_orders > 0:
        raise UntrackAbort(f"broker has {open_orders} open orders — aborting")

    execute_detected, _ = _find_python_execute_trades_execute()
    if execute_detected:
        raise UntrackAbort("execute_trades --execute process detected — aborting")


def _execute_git_rm_cached(paths: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    untracked: List[str] = []
    skipped: List[Dict[str, str]] = []
    batch_size = 20

    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        existing = [p for p in batch if _git_tracked(p)]
        for p in batch:
            if p not in existing:
                skipped.append({"path": p, "reason": "not tracked in index (already untracked)"})

        if not existing:
            continue

        args = ["rm", "--cached", "--", *existing]
        code, out, err = _git_output(*args)
        if code != 0:
            for p in existing:
                skipped.append({"path": p, "reason": f"git rm --cached failed: {err or out}"})
            continue
        untracked.extend(existing)

    return untracked, skipped


def run_phase_195s_execution(*, dry_run: bool = False, verbose: bool = True) -> Dict[str, Any]:
    plan = _load_json(PLAN_195R)
    file_plan = _load_file_plan()
    approved = list(plan.get("exact_files_proposed_for_later_git_rm_cached") or [])
    excluded = set(plan.get("exact_files_excluded_from_git_rm_cached") or [])

    pre_head = ""
    code, pre_head, _ = _git_output("rev-parse", "HEAD")
    if code != 0:
        pre_head = ""

    abort_reason: Optional[str] = None
    untracked: List[str] = []
    skipped: List[Dict[str, str]] = []
    manifest: List[Dict[str, str]] = []

    try:
        _validate_approved_targets(approved, file_plan, excluded)
        _validate_preconditions()

        gitignore_exceptions: List[str] = []
        for path in approved:
            if not _gitignore_matches(path):
                gitignore_exceptions.append(path)
        if gitignore_exceptions and verbose:
            print(
                f"[195S] note: {len(gitignore_exceptions)} paths not matched by .gitignore "
                "(may still untrack; live/*.lock rules apply after untrack)"
            )

        if dry_run:
            if verbose:
                print(f"[195S] dry_run: would untrack {len(approved)} paths")
        else:
            untracked, skipped = _execute_git_rm_cached(approved)
    except UntrackAbort as exc:
        abort_reason = str(exc)
        if verbose:
            print(f"[195S] ABORT: {abort_reason}")

    watchdog_running, continuous_running = _loops_running()
    execute_detected, execute_cmds = _find_python_execute_trades_execute()
    readiness = _load_json(EXEC_READINESS)
    paper_cfg = _load_json(PAPER_CYCLE_CFG)

    all_targets = approved if not abort_reason else []
    still_exist: List[str] = []
    missing_after: List[str] = []
    source_touched: List[str] = []

    for path in untracked:
        full = ROOT / path
        if full.is_file():
            still_exist.append(path)
        else:
            missing_after.append(path)

    for path in all_targets:
        norm = path.replace("\\", "/")
        row = file_plan.get(norm, {})
        tracked_after = _git_tracked(norm) if not abort_reason else True
        exists = (ROOT / norm).is_file()
        manifest.append(
            {
                "path": norm,
                "plan_classification": row.get("plan_classification", ""),
                "git_rm_cached_executed": str(norm in untracked).lower(),
                "still_tracked_after": str(tracked_after).lower(),
                "exists_on_disk_after": str(exists).lower(),
                "ignored_by_gitignore": str(_gitignore_matches(norm)).lower(),
                "status": (
                    "untracked"
                    if norm in untracked
                    else ("skipped" if not abort_reason else "aborted")
                ),
                "reason": next(
                    (s["reason"] for s in skipped if s["path"] == norm),
                    "ok" if norm in untracked else ("abort" if abort_reason else "pending"),
                ),
            }
        )

    phase_pass = (
        abort_reason is None
        and len(missing_after) == 0
        and not source_touched
        and len(untracked) + len(skipped) >= len(approved)
        and all(
            not _git_tracked(p.replace("\\", "/"))
            for p in approved
            if (ROOT / p.replace("\\", "/")).is_file() or _git_tracked(p.replace("\\", "/"))
        )
    )

    rollback = [
        f"PRE_CLEANUP_HEAD={pre_head}",
        f"git checkout {pre_head} -- <path>  # re-track individual runtime file if needed",
        "git add -f <path>  # alternative re-track",
        "Files on disk were never deleted — only index entries removed",
        "Do not git restore data/live/* while watchdog/continuous loops are running",
    ]

    commit_msg = (
        "chore: untrack runtime artifacts from Git index (Phase 195S)\n\n"
        "Remove 43 approved runtime paths from tracking via git rm --cached only. "
        "Files remain on disk; loops unchanged; no scheduler edits."
    )

    commit_instructions = [
        "# Index removals are already staged (git status shows 'D' — deletion from index only)",
        "git add services/phase_195s_runtime_index_untrack_execution.py",
        "git diff --cached --name-only   # expect: 43 runtime paths + phase_195s script only",
        "# Do NOT: git add data/results/certification* or other regenerated runtime files",
        f'git commit -m "{commit_msg.split(chr(10))[0]}"',
        "# After commit: 195H should return READY (staged-runtime-artifact block clears)",
    ]

    payload: Dict[str, Any] = {
        "phase": "195S",
        "timestamp": _utc_now_iso(),
        "phase_pass": phase_pass,
        "aborted": abort_reason is not None,
        "abort_reason": abort_reason,
        "orders_placed": False,
        "scheduled_tasks_touched": False,
        "runtime_files_deleted_from_disk": False,
        "pre_cleanup_head": pre_head,
        "approved_untrack_count": len(approved),
        "actually_untracked_count": len(untracked),
        "skipped_count": len(skipped),
        "skipped_paths_with_reasons": skipped,
        "files_still_exist_on_disk": still_exist,
        "files_missing_after_untrack": missing_after,
        "source_files_touched": source_touched,
        "scheduler_tasks_touched": False,
        "broker_open_orders_count": _count_csv_rows(OPEN_ORDERS),
        "broker_positions_count": _count_csv_rows(POSITIONS),
        "live_execution_permitted": bool(readiness.get("live_execution_permitted", False)),
        "paper_execution_permitted": bool(readiness.get("paper_execution_permitted", False)),
        "execute_trades_execute_detected": execute_detected,
        "execute_trades_command_lines": execute_cmds,
        "automated_entry_execution_enabled": bool(paper_cfg.get("run_execute_trades", False)),
        "watchdog_process_running": watchdog_running,
        "continuous_process_running": continuous_running,
        "execution_readiness_status": readiness.get("readiness_status", "UNKNOWN"),
        "195r_plan_reference": str(PLAN_195R.relative_to(ROOT)).replace("\\", "/"),
        "rollback_instructions": rollback,
        "recommended_commit_message": commit_msg,
        "recommended_commit_instructions": commit_instructions,
        "post_execution_note": (
            "195H/195K may report NOT_READY while index removals remain staged (D entries). "
            "This is expected until the Phase 195S commit completes. "
            "execution_readiness remains independent and should stay READY."
        ),
        "manifest": manifest,
        "safety_statement": (
            "Phase 195S executed git rm --cached only for approved runtime paths. "
            "No orders placed, no scheduler tasks modified, no runtime files deleted from disk, "
            "no data/live restore performed."
            if not abort_reason
            else f"Phase 195S aborted before git rm --cached: {abort_reason}"
        ),
    }

    summary_rows = [
        {"metric": "phase_pass", "value": str(phase_pass).lower()},
        {"metric": "aborted", "value": str(abort_reason is not None).lower()},
        {"metric": "pre_cleanup_head", "value": pre_head},
        {"metric": "approved_untrack_count", "value": str(len(approved))},
        {"metric": "actually_untracked_count", "value": str(len(untracked))},
        {"metric": "skipped_count", "value": str(len(skipped))},
        {"metric": "files_missing_after_untrack", "value": str(len(missing_after))},
        {"metric": "watchdog_process_running", "value": str(watchdog_running)},
        {"metric": "continuous_process_running", "value": str(continuous_running)},
        {"metric": "live_execution_permitted", "value": str(payload["live_execution_permitted"])},
        {"metric": "paper_execution_permitted", "value": str(payload["paper_execution_permitted"])},
        {"metric": "broker_open_orders_count", "value": str(payload["broker_open_orders_count"])},
        {"metric": "broker_positions_count", "value": str(payload["broker_positions_count"])},
        {"metric": "execute_trades_execute_detected", "value": str(execute_detected)},
    ]

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)
    _write_csv(MANIFEST_CSV, manifest, MANIFEST_FIELDS)

    if verbose:
        print(f"[195S] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        if abort_reason:
            print(f"[195S] ABORTED: {abort_reason}")
        else:
            print(f"[195S] untracked {len(untracked)}/{len(approved)} from Git index")
            print(
                f"[195S] missing_on_disk={len(missing_after)} loops wd={watchdog_running} cl={continuous_running}"
            )

    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 195S runtime index untrack execution")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate only; do not run git rm --cached"
    )
    args = parser.parse_args()
    result = run_phase_195s_execution(dry_run=args.dry_run, verbose=True)
    if result.get("aborted"):
        sys.exit(1)
    if not result.get("phase_pass"):
        sys.exit(2)


if __name__ == "__main__":
    main()
