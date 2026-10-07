"""Temporary-fixture proof for the isolated D1a as-of validator."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

PRODUCTION = Path(r"C:\Users\akimw\triton")
PYTHON = PRODUCTION / ".venv" / "Scripts" / "python.exe"
SCRIPT = PRODUCTION / "scripts" / "validate_d1a_asof_run.py"
RUN_ID = "daily_prediction_refresh_20260928_223334"
SESSION = "2026-09-28"
PROTECTED = (
    "data/results/signals_asof_diagnostics.json",
    "data/results/signals.csv",
    "data/results/signals_with_rationale.csv",
    "data/results/heartbeat.json",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _production_hashes() -> dict[str, str]:
    return {_rel: _sha(PRODUCTION / _rel) for _rel in PROTECTED}


def _write_symbol(root: Path, symbol: str, day: str) -> None:
    frame = pd.DataFrame({"date": [day], "close": [1.0]})
    raw = root / "data" / "raw" / f"{symbol}.csv"
    processed = root / "data" / "processed" / f"{symbol}.parquet"
    prediction = root / "data" / "predictions" / f"{symbol}_predictions.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    processed.parent.mkdir(parents=True, exist_ok=True)
    prediction.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw, index=False)
    frame.to_parquet(processed, index=False)
    frame.to_parquet(prediction, index=False)


def _build(
    root: Path, *, binding_run_id: str = RUN_ID, omit_fetch: bool = False, stale: str | None = None
) -> None:
    (root / "data" / "config").mkdir(parents=True, exist_ok=True)
    (root / "data" / "results").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "data" / "config" / "tickers.txt").write_text("AAA\nBBB\n^IDX\n", encoding="utf-8")
    for symbol in ("AAA", "BBB", "^IDX"):
        _write_symbol(root, symbol, "2026-09-25" if symbol == stale else SESSION)
    for relative, body in (
        ("data/results/signals_asof_diagnostics.json", b"SHARED-SENTINEL\n"),
        ("data/results/signals.csv", b"SIGNALS-SENTINEL\n"),
        ("data/results/signals_with_rationale.csv", b"RATIONALE-SENTINEL\n"),
        ("data/results/heartbeat.json", b"HEARTBEAT-SENTINEL\n"),
    ):
        (root / relative).write_bytes(body)
    stamp = RUN_ID.removeprefix("daily_prediction_refresh_")
    (root / "logs" / f"daily_refresh_{stamp}.log").write_text(
        f"{RUN_ID}\nRefresh OK\n", encoding="utf-8"
    )
    fetch_path = root / "data" / "results" / "daily_refresh_fetch_receipt.json"
    fetch_path.write_text(
        json.dumps({"passed": True, "target_session": SESSION, "tradable_coverage": {"ratio": 1.0}})
        + "\n",
        encoding="utf-8",
    )
    fetch_bytes = fetch_path.read_bytes()
    captured = "2026-09-29T02:33:36.825439+00:00"
    baseline = {
        "run_id": RUN_ID,
        "captured_at_utc": captured,
    }
    (root / "data" / "results" / f"daily_prediction_refresh_baseline_{RUN_ID}.json").write_text(
        json.dumps(baseline) + "\n", encoding="utf-8"
    )
    rows = []
    if not omit_fetch:
        rows.append(
            {
                "name": "fetch_raw_data",
                "run_id": binding_run_id,
                "status": "COMPLETE",
                "outcome": "ok",
                "target_session": SESSION,
                "artifact_path": "data/results/daily_refresh_fetch_receipt.json",
                "artifact_sha256": hashlib.sha256(fetch_bytes).hexdigest(),
                "artifact_size_bytes": len(fetch_bytes),
                "artifact_written_at_utc": "2026-09-29T02:35:09.4990917Z",
                "baseline_captured_at_utc": captured,
                "artifact_after_baseline": True,
            }
        )
    (
        root / "data" / "results" / f"daily_prediction_refresh_stage_receipts_{RUN_ID}.json"
    ).write_text(json.dumps(rows) + "\n", encoding="utf-8")
    summary = {
        "run_id": RUN_ID,
        "target_session": SESSION,
        "passed": True,
        "broker_called": False,
        "network_called": False,
    }
    (root / "data" / "results" / "daily_prediction_refresh_summary.json").write_text(
        json.dumps(summary) + "\n", encoding="utf-8"
    )


def _sentinel_hashes(root: Path) -> dict[str, str]:
    return {_rel: _sha(root / _rel) for _rel in PROTECTED}


def _run(
    root: Path, validation_id: str, *, session: str = SESSION
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [
            str(PYTHON),
            "-B",
            str(SCRIPT),
            "--repo-root",
            str(root),
            "--source-run-id",
            RUN_ID,
            "--session",
            session,
            "--validation-id",
            validation_id,
        ],
        cwd=str(root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_validator_source_does_not_call_operational_publishers() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "_persist_diag" not in text
    assert "generate_signals.main" not in text
    assert "compute_asof_policy" in text


def test_valid_september_28_fixture_writes_one_run_bound_file(tmp_path: Path) -> None:
    before = _production_hashes()
    _build(tmp_path)
    sentinels = _sentinel_hashes(tmp_path)
    validation_id = "fixture_ok"
    completed = _run(tmp_path, validation_id)
    assert completed.returncode == 0, completed.stderr
    destination = tmp_path / "data" / "results" / f"signals_asof_diagnostics_{validation_id}.json"
    assert destination.is_file()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["validation_id"] == validation_id
    assert payload["source_run_id"] == RUN_ID
    assert payload["session_asof_date"] == SESSION
    assert payload["bound_fetch_receipt"]["run_id"] == RUN_ID
    assert payload["bound_fetch_receipt"]["post_baseline"] is True
    assert payload["tradable"]["have_session"] == 2
    assert payload["tradable"]["denominator"] == 2
    assert payload["tradable"]["coverage"] == 1.0
    assert payload["tradable"]["fail"] is False
    assert payload["shared_diagnostics_used_as_evidence"] is False
    assert _sentinel_hashes(tmp_path) == sentinels
    assert _production_hashes() == before


def test_missing_or_mismatched_source_run_binding_fails(tmp_path: Path) -> None:
    _build(tmp_path, binding_run_id="daily_prediction_refresh_20260928_000000")
    completed = _run(tmp_path, "fixture_mismatch")
    assert completed.returncode != 0
    assert not (
        tmp_path / "data" / "results" / "signals_asof_diagnostics_fixture_mismatch.json"
    ).exists()

    other = tmp_path / "missing"
    _build(other, omit_fetch=True)
    missing = _run(other, "fixture_missing")
    assert missing.returncode != 0
    assert not (
        other / "data" / "results" / "signals_asof_diagnostics_fixture_missing.json"
    ).exists()


def test_stale_non_september_28_data_fails(tmp_path: Path) -> None:
    _build(tmp_path, stale="AAA")
    completed = _run(tmp_path, "fixture_stale")
    assert completed.returncode != 0
    assert not (
        tmp_path / "data" / "results" / "signals_asof_diagnostics_fixture_stale.json"
    ).exists()


def test_existing_output_is_refused(tmp_path: Path) -> None:
    _build(tmp_path)
    destination = tmp_path / "data" / "results" / "signals_asof_diagnostics_fixture_exists.json"
    destination.write_bytes(b"KEEP\n")
    completed = _run(tmp_path, "fixture_exists")
    assert completed.returncode != 0
    assert destination.read_bytes() == b"KEEP\n"


def test_shared_diagnostic_and_signal_artifacts_are_untouched(tmp_path: Path) -> None:
    before = _production_hashes()
    _build(tmp_path)
    sentinels = _sentinel_hashes(tmp_path)
    completed = _run(tmp_path, "fixture_untouched")
    assert completed.returncode == 0, completed.stderr
    assert _sentinel_hashes(tmp_path) == sentinels
    assert _production_hashes() == before
