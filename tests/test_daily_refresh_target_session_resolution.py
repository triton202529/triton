"""Session-wide DailyRefresh discovery. Temporary fixtures only. No production writes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from services.daily_refresh_completion import (
    CLASS_COMPLETE,
    CLASS_FETCH_ONLY,
    REQUIRED_STAGES,
    _evaluate_run,
    _select_valid_run,
    evaluate_daily_refresh_completion,
)
from services.daily_refresh_stage_receipts import generation_receipt_name
from services.operator_closeout_control import _daily_refresh_status

SESSION = "2026-10-20"
FAILED = "daily_prediction_refresh_20261020_211520"
RECOVERY = "daily_prediction_refresh_20261021_102134"
LEGACY = "daily_prediction_refresh_20261020_190815"


def _stage(run_id: str, name: str, session: str = SESSION, *, ok: bool = True) -> dict:
    return {
        "name": name,
        "run_id": run_id,
        "target_session": session,
        "exit_code": 0 if ok else 1,
        "outcome": "ok" if ok else "failed",
        "status": "COMPLETE" if ok else "FAILED",
    }


def _stages(run_id: str, session: str = SESSION, *, ok: bool = True) -> list[dict]:
    return [_stage(run_id, name, session, ok=ok) for name in REQUIRED_STAGES]


def _install(
    root: Path,
    run_id: str,
    stages: list[dict],
    *,
    session: str = SESSION,
    success_log: bool = True,
    asof: str = "",
    summary: bool = True,
) -> None:
    results = root / "data" / "results"
    logs = root / "logs"
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    (results / f"daily_prediction_refresh_baseline_{run_id}.json").write_text(
        json.dumps({"run_id": run_id}),
        encoding="utf-8",
    )
    stamp = run_id.replace("daily_prediction_refresh_", "")
    lines = [f"run_id={run_id}"]
    if asof:
        lines.append(f"[fetch_raw] asof={asof} end(exclusive)=2026-10-21")
    if success_log:
        lines.append("=== TRITON Daily Prediction Refresh OK ===")
        lines.append("FinishedAt=2026-10-21T14:49:18 exit=0")
    else:
        lines.append("FinishedAt=2026-10-20T21:20:00 exit=1")
    (logs / f"daily_refresh_{stamp}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    generation = results / generation_receipt_name(run_id, 1)
    generation.write_text(json.dumps(stages), encoding="utf-8")
    if summary:
        (results / "daily_prediction_refresh_summary.json").write_text(
            json.dumps({"run_id": run_id, "target_session": session, "passed": True}),
            encoding="utf-8",
        )


def _hashes(root: Path) -> dict[str, str]:
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            found[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def test_next_day_run_targets_the_prior_completed_session(tmp_path: Path) -> None:
    _install(tmp_path, RECOVERY, _stages(RECOVERY), asof=SESSION)
    before = _hashes(tmp_path)
    verdict = evaluate_daily_refresh_completion(tmp_path, SESSION)
    assert verdict["ok"] is True
    assert verdict["classification"] == CLASS_COMPLETE
    assert verdict["run_id"] == RECOVERY
    assert verdict["target_session"] == SESSION
    assert verdict["selection_reason"] == "only_fully_valid_run"
    assert _daily_refresh_status(tmp_path, SESSION) == verdict
    assert _hashes(tmp_path) == before


def test_failed_same_session_run_does_not_outrank_the_recovery(tmp_path: Path) -> None:
    _install(
        tmp_path,
        FAILED,
        [_stage(FAILED, "fetch_raw_data")],
        success_log=False,
        summary=False,
    )
    _install(tmp_path, RECOVERY, _stages(RECOVERY), asof=SESSION)
    failed = _evaluate_run(tmp_path, FAILED, SESSION)
    assert failed["ok"] is False
    assert failed["classification"] == CLASS_FETCH_ONLY
    verdict = evaluate_daily_refresh_completion(tmp_path, SESSION)
    assert verdict["ok"] is True
    assert verdict["run_id"] == RECOVERY
    assert verdict["selection_reason"] == "only_fully_valid_run"
    assert FAILED not in verdict["run_id"]


def test_malformed_newest_generation_fails_closed(tmp_path: Path) -> None:
    _install(tmp_path, RECOVERY, _stages(RECOVERY), asof=SESSION)
    results = tmp_path / "data" / "results"
    (results / generation_receipt_name(RECOVERY, 2)).write_text("{", encoding="utf-8")
    verdict = evaluate_daily_refresh_completion(tmp_path, SESSION)
    assert verdict["ok"] is False
    assert verdict["classification"] == CLASS_FETCH_ONLY
    assert verdict["detail"] == "stage_receipt_generation_malformed"
    assert verdict["run_id"] == RECOVERY


def test_mismatched_target_session_is_rejected(tmp_path: Path) -> None:
    _install(
        tmp_path, RECOVERY, _stages(RECOVERY, "2026-10-21"), session="2026-10-21", asof="2026-10-21"
    )
    verdict = evaluate_daily_refresh_completion(tmp_path, SESSION)
    assert verdict["ok"] is False
    assert verdict["classification"] != CLASS_COMPLETE
    assert verdict["run_id"] == ""


def test_legacy_same_day_run_still_completes(tmp_path: Path) -> None:
    _install(tmp_path, LEGACY, _stages(LEGACY))
    verdict = evaluate_daily_refresh_completion(tmp_path, SESSION)
    assert verdict["classification"] == CLASS_COMPLETE
    assert verdict["run_id"] == LEGACY
    assert verdict["selection_reason"] == "only_fully_valid_run"


def test_latest_valid_run_id_wins_when_several_are_complete() -> None:
    chosen = _select_valid_run(
        [
            {"ok": True, "run_id": FAILED, "classification": CLASS_COMPLETE},
            {"ok": True, "run_id": RECOVERY, "classification": CLASS_COMPLETE},
        ]
    )
    assert chosen["run_id"] == RECOVERY
    assert chosen["selection_reason"] == "latest_run_id_among_fully_valid"


def test_evaluation_does_not_write(tmp_path: Path) -> None:
    _install(tmp_path, RECOVERY, _stages(RECOVERY), asof=SESSION)
    before = _hashes(tmp_path)
    evaluate_daily_refresh_completion(tmp_path, SESSION)
    _evaluate_run(tmp_path, RECOVERY, SESSION)
    _daily_refresh_status(tmp_path, SESSION)
    assert _hashes(tmp_path) == before
