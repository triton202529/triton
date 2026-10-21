"""Full DailyRefresh completion contract.

A fetch receipt is one stage. It cannot classify a session as complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from services.daily_refresh_stage_receipts import (
    resolve_stage_receipts,
    run_id_from_stage_receipt_filename,
)

SCHEMA = "DAILY-REFRESH-COMPLETION-1"
CLASS_COMPLETE = "FULL_DAILYREFRESH_COMPLETE"
CLASS_FETCH_ONLY = "FETCH_ONLY_INCOMPLETE"
REQUIRED_STAGES: Sequence[str] = (
    "fetch_raw_data",
    "stale_data_gate",
    "preprocess",
    "train_model",
    "daily_prediction_refresh_validate",
)
SUCCESS_LOG_LINE = "=== TRITON Daily Prediction Refresh OK ==="


def _result(
    ok: bool,
    detail: str,
    classification: str,
    target_session: str = "",
    *,
    run_id: str = "",
) -> Dict[str, Any]:
    return {
        "ok": bool(ok),
        "detail": detail,
        "classification": classification,
        "target_session": target_session,
        "run_id": run_id,
    }


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _lock_present(root: Path) -> bool:
    return (root / "data" / "live" / "daily_refresh.lock").is_file() or (
        root / "data" / "daily_refresh.lock"
    ).is_file()


def _stage_ok(stage: Mapping[str, Any], *, run_id: str, session_date: str) -> bool:
    if str(stage.get("name") or "") == "":
        return False
    if str(stage.get("run_id") or "") != run_id:
        return False
    if str(stage.get("target_session") or "") != session_date:
        return False
    try:
        exit_code = int(stage.get("exit_code"))
    except (TypeError, ValueError):
        return False
    outcome = str(stage.get("outcome") or "").lower()
    status = str(stage.get("status") or "").upper()
    return exit_code == 0 and outcome == "ok" and status in {"", "COMPLETE"}


def contract_is_complete(doc: Mapping[str, Any], session_date: str) -> Dict[str, Any]:
    """Validate an in-memory completion contract. Does not read or write files."""
    if str(doc.get("schema") or "") != SCHEMA:
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, session_date)
    run_id = str(doc.get("run_id") or "")
    target = str(doc.get("target_session") or "")
    if not run_id or target != session_date:
        return _result(
            False,
            "daily_refresh_session_mismatch",
            CLASS_FETCH_ONLY,
            target,
            run_id=run_id,
        )
    try:
        wrapper_exit = int(doc.get("wrapper_exit_code"))
    except (TypeError, ValueError):
        wrapper_exit = 1
    if wrapper_exit != 0 or doc.get("terminal_success") is not True:
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, target, run_id=run_id)
    if str(doc.get("baseline_run_id") or "") != run_id:
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, target, run_id=run_id)
    if str(doc.get("summary_run_id") or "") != run_id:
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, target, run_id=run_id)
    if str(doc.get("summary_target_session") or "") != session_date:
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, target, run_id=run_id)
    stages = list(doc.get("stages") or [])
    names = [str(stage.get("name") or "") for stage in stages if isinstance(stage, Mapping)]
    if names != list(REQUIRED_STAGES):
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, target, run_id=run_id)
    if not all(
        isinstance(stage, Mapping) and _stage_ok(stage, run_id=run_id, session_date=session_date)
        for stage in stages
    ):
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, target, run_id=run_id)
    return _result(True, "daily_refresh_complete", CLASS_COMPLETE, target, run_id=run_id)


def _legacy_fetch(receipt_doc: Mapping[str, Any], session_date: str) -> Dict[str, Any]:
    if not receipt_doc:
        return _result(False, "daily_refresh_receipt_missing", CLASS_FETCH_ONLY, session_date)
    target = str(receipt_doc.get("target_session") or "")
    if target and target != session_date:
        return _result(False, "daily_refresh_session_mismatch", CLASS_FETCH_ONLY, target)
    passed = receipt_doc.get("passed") is True and not receipt_doc.get("failure")
    if not passed:
        return _result(False, "daily_refresh_not_passed", CLASS_FETCH_ONLY, target or session_date)
    return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, target or session_date)


def _log_is_successful(text: str, run_id: str) -> bool:
    if f"run_id={run_id}" not in text or SUCCESS_LOG_LINE not in text:
        return False
    exit_lines = [line for line in text.splitlines() if "exit=" in line and "FinishedAt=" in line]
    if not exit_lines:
        exit_lines = [
            line
            for line in text.splitlines()
            if line.strip().endswith("exit=0") or " exit=0" in line
        ]
    if not exit_lines:
        return False
    last = exit_lines[-1]
    return "exit=0" in last and "exit=1" not in last.split("exit=0")[-1]


def _run_id_from_log(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            marker = "run_id="
            if marker in line and "daily_prediction_refresh_" in line:
                return line.split(marker, 1)[1].strip().split()[0]
    except OSError:
        return ""
    return ""


def _evaluate_run(root: Path, run_id: str, session_date: str) -> Dict[str, Any]:
    results = root / "data" / "results"
    logs = root / "logs"
    baseline = _read_json(results / f"daily_prediction_refresh_baseline_{run_id}.json")
    resolved = resolve_stage_receipts(results, run_id)
    if resolved.status in {"ambiguous", "malformed"} or (
        resolved.status == "missing" and resolved.generation is not None
    ):
        return _result(False, resolved.detail, CLASS_FETCH_ONLY, session_date, run_id=run_id)
    stages_doc = resolved.document
    summary = _read_json(results / "daily_prediction_refresh_summary.json")
    stamp = run_id.replace("daily_prediction_refresh_", "")
    log_path = logs / f"daily_refresh_{stamp}.log"
    if not isinstance(baseline, dict) or str(baseline.get("run_id") or "") != run_id:
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, session_date, run_id=run_id)
    if not isinstance(stages_doc, list):
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, session_date, run_id=run_id)
    names = [str(stage.get("name") or "") for stage in stages_doc if isinstance(stage, dict)]
    if names != list(REQUIRED_STAGES):
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, session_date, run_id=run_id)
    if not all(_stage_ok(stage, run_id=run_id, session_date=session_date) for stage in stages_doc):
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, session_date, run_id=run_id)
    if not isinstance(summary, dict):
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, session_date, run_id=run_id)
    if str(summary.get("run_id") or "") != run_id:
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, session_date, run_id=run_id)
    if str(summary.get("target_session") or "") != session_date:
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, session_date, run_id=run_id)
    if not log_path.is_file():
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, session_date, run_id=run_id)
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, session_date, run_id=run_id)
    if not _log_is_successful(log_text, run_id):
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, session_date, run_id=run_id)
    return _result(True, "daily_refresh_complete", CLASS_COMPLETE, session_date, run_id=run_id)


def _candidate_run_ids(root: Path, session_date: str) -> List[str]:
    found: List[str] = []
    results = root / "data" / "results"
    compact = session_date.replace("-", "")
    if results.is_dir():
        for path in sorted(
            results.glob(f"daily_prediction_refresh_stage_receipts_*{compact}*.json")
        ):
            run_id = run_id_from_stage_receipt_filename(path.name)
            if run_id and run_id not in found:
                found.append(run_id)
    logs = root / "logs"
    if logs.is_dir():
        for path in sorted(logs.glob(f"daily_refresh_*{compact}*.log")):
            run_id = _run_id_from_log(path)
            if run_id and run_id not in found:
                found.append(run_id)
    return found


def evaluate_daily_refresh_completion(
    root: Path,
    session_date: str,
    *,
    receipt_doc: Optional[Mapping[str, Any]] = None,
    refresh_running: bool = False,
) -> Dict[str, Any]:
    """Return the session completion classification. Read-only."""
    root = Path(root)
    session_date = str(session_date or "")
    if refresh_running or _lock_present(root):
        return _result(False, "daily_refresh_still_running", CLASS_FETCH_ONLY, session_date)
    if isinstance(receipt_doc, Mapping) and str(receipt_doc.get("schema") or "") == SCHEMA:
        return contract_is_complete(receipt_doc, session_date)
    if isinstance(receipt_doc, Mapping):
        return _legacy_fetch(receipt_doc, session_date)

    fetch = _read_json(root / "data" / "results" / "daily_refresh_fetch_receipt.json")
    fetch_target = str((fetch or {}).get("target_session") or "") if isinstance(fetch, dict) else ""
    fetch_passed = (
        isinstance(fetch, dict) and fetch.get("passed") is True and not fetch.get("failure")
    )
    if not session_date:
        return _result(False, "daily_refresh_receipt_missing", CLASS_FETCH_ONLY, "")
    blocked = None
    for run_id in _candidate_run_ids(root, session_date):
        verdict = _evaluate_run(root, run_id, session_date)
        if verdict["ok"]:
            return verdict
        if str(verdict.get("detail") or "").startswith("stage_receipt_generation_"):
            blocked = verdict
    if blocked is not None:
        return blocked
    if fetch_passed and fetch_target == session_date:
        return _result(False, CLASS_FETCH_ONLY, CLASS_FETCH_ONLY, session_date)
    if isinstance(fetch, dict) and fetch_target and fetch_target != session_date:
        return _result(False, "daily_refresh_session_mismatch", CLASS_FETCH_ONLY, fetch_target)
    if isinstance(fetch, dict) and fetch_target == session_date and not fetch_passed:
        return _result(False, "daily_refresh_not_passed", CLASS_FETCH_ONLY, session_date)
    return _result(False, "daily_refresh_receipt_missing", CLASS_FETCH_ONLY, session_date)
