"""Shared session OHLC contract. Temporary files only."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from services.coherent_signal_bundle import BUNDLE_FILES, commit_signal_bundle
from services.daily_prediction_refresh_validate import (
    _raw_exact_session_errors,
    assess_required_layer,
)
from services.generate_signals import SKIP_INVALID_PROCESSED_SESSION_BAR, _session_ohlcv_skip_reason
from services.session_ohlc import OHLC_ABSOLUTE_TOLERANCE, session_ohlc_violation

SESSION = date(2026, 10, 20)
MRK_OPEN = 146.2899932861328
MRK_LOW = 146.3000030517578
T_OPEN = 23.360000610351562
T_LOW = 23.3799991607666


def _bar(**overrides) -> dict:
    row = {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100.0}
    row.update(overrides)
    return row


def _signal_reason(row: dict) -> str | None:
    return _session_ohlcv_skip_reason(pd.Series(row))


def _raw_errors(tmp_path: Path, row: dict) -> list[str]:
    path = tmp_path / "MRK.csv"
    tmp_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**row, "date": SESSION.isoformat(), "ticker": "MRK"}]).to_csv(path, index=False)
    return _raw_exact_session_errors(path, "MRK", SESSION)


def test_equality_and_subtolerance_noise_are_valid() -> None:
    equal = _bar(open=10.0, high=10.0, low=10.0, close=10.0)
    noisy = _bar(open=10.0, low=10.0 + 1e-7, high=10.0, close=10.0)
    for row in (equal, noisy):
        assert (
            session_ohlc_violation(
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                symbol="EQ",
                session_date=SESSION.isoformat(),
                source_layer="raw",
            )
            is None
        )
        assert _signal_reason(row) is None


def test_mrk_and_t_inversions_stay_invalid() -> None:
    mrk = _bar(open=MRK_OPEN, high=149.36000061035156, low=MRK_LOW, close=148.3000030517578)
    tee = _bar(open=T_OPEN, high=23.68000030517578, low=T_LOW, close=23.420000076293945)
    mrk_gap = session_ohlc_violation(
        open=mrk["open"],
        high=mrk["high"],
        low=mrk["low"],
        close=mrk["close"],
        symbol="MRK",
        session_date=SESSION.isoformat(),
        source_layer="raw",
    )
    tee_gap = session_ohlc_violation(
        open=tee["open"],
        high=tee["high"],
        low=tee["low"],
        close=tee["close"],
        symbol="T",
        session_date=SESSION.isoformat(),
        source_layer="processed",
    )
    assert mrk_gap is not None and tee_gap is not None
    assert mrk_gap["relationship"] == "low_gt_open"
    assert tee_gap["relationship"] == "low_gt_open"
    assert mrk_gap["tolerance"] == OHLC_ABSOLUTE_TOLERANCE == 1e-6
    assert (MRK_LOW - MRK_OPEN) > OHLC_ABSOLUTE_TOLERANCE
    assert (T_LOW - T_OPEN) > OHLC_ABSOLUTE_TOLERANCE
    assert _signal_reason(mrk) == SKIP_INVALID_PROCESSED_SESSION_BAR
    assert _signal_reason(tee) == SKIP_INVALID_PROCESSED_SESSION_BAR


def test_refresh_and_signal_generation_return_the_same_decision(tmp_path: Path) -> None:
    valid = _bar()
    invalid = _bar(open=MRK_OPEN, high=149.36, low=MRK_LOW, close=148.30)
    assert _signal_reason(valid) is None
    assert _raw_errors(tmp_path / "valid", valid) == []
    assert _signal_reason(invalid) == SKIP_INVALID_PROCESSED_SESSION_BAR
    assert _raw_errors(tmp_path / "invalid", invalid) == ["invalid_session_ohlc"]


def test_refresh_records_violation_without_rewriting_raw(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "MRK.csv"
    pd.DataFrame(
        [
            {
                "date": SESSION.isoformat(),
                "open": MRK_OPEN,
                "high": 149.36000061035156,
                "low": MRK_LOW,
                "close": 148.3000030517578,
                "volume": 9899397,
                "ticker": "MRK",
            }
        ]
    ).to_csv(path, index=False)
    before = path.read_bytes()
    assessed = assess_required_layer(
        repo_root=tmp_path,
        layer="raw",
        required_symbols=["MRK"],
        excluded_symbols=[],
        target=SESSION,
        baseline_artifacts={},
        run_started_at=datetime(2026, 10, 21, tzinfo=timezone.utc),
        require_advancement=False,
    )
    assert assessed["passed"] is False
    assert assessed["schema_invalid_symbols"] == ["MRK"]
    assert "MRK" not in assessed["ok_symbols"]
    violation = assessed["ohlc_violations"][0]
    assert violation["symbol"] == "MRK"
    assert violation["session_date"] == "2026-10-20"
    assert violation["relationship"] == "low_gt_open"
    assert violation["values"]["open"] == MRK_OPEN
    assert violation["values"]["low"] == MRK_LOW
    assert violation["tolerance"] == 1e-6
    assert violation["source_layer"] == "raw"
    assert path.read_bytes() == before


def test_commit_preserves_prior_bundle_when_ohlc_is_invalid(tmp_path: Path, monkeypatch) -> None:
    import services.coherent_signal_bundle as bundle

    results = tmp_path / "results"
    staging = tmp_path / "stage"
    results.mkdir()
    staging.mkdir()
    monkeypatch.setattr(bundle, "RESULTS_DIR", results)
    monkeypatch.setattr(bundle, "STAGING_DIR", staging)
    for name in BUNDLE_FILES:
        (results / name).write_text(f"prior-{name}\n", encoding="utf-8")
    before = {name: (results / name).read_bytes() for name in BUNDLE_FILES}
    (staging / bundle.STAGE_MARKER).write_text(
        json.dumps(
            {
                "status": "STAGED",
                "session_as_of_date": "2026-10-20",
                "generated_at_utc": "2026-10-21T13:31:49Z",
            }
        ),
        encoding="utf-8",
    )
    (staging / "signals_asof_diagnostics.json").write_text(
        json.dumps(
            {
                "per_symbol_skips": [
                    {
                        "symbol": "T",
                        "reason": "INVALID_PROCESSED_SESSION_BAR",
                        "class": "tradable",
                        "ohlc_violation": {
                            "symbol": "T",
                            "session_date": "2026-10-20",
                            "relationship": "low_gt_open",
                            "values": {
                                "open": T_OPEN,
                                "high": 23.68000030517578,
                                "low": T_LOW,
                                "close": 23.420000076293945,
                            },
                            "tolerance": 1e-6,
                            "source_layer": "processed",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert commit_signal_bundle() == 1
    assert {name: (results / name).read_bytes() for name in BUNDLE_FILES} == before
    receipt = json.loads((results / bundle.FAILURE_RECEIPT).read_text(encoding="utf-8"))
    assert receipt["published_signal_artifacts"] is False
    assert receipt["reason"] == "invalid_session_ohlc"
    assert receipt["ohlc_violations"][0]["symbol"] == "T"
    assert receipt["ohlc_violations"][0]["source_layer"] == "processed"
