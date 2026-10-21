"""Staged diagnostic provenance for coherent bundle lifecycle validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from services.coherent_signal_bundle import (
    PUBLIC_DIAGNOSTIC,
    PUBLISHED_ARTIFACTS,
    _fail_stage,
    bind_staged_diagnostic,
    commit_signal_bundle,
    publish_coherent_bundle,
)
from services.lifecycle_artifact_guard import evaluate_lifecycle_source_gate

SESSION = "2026-10-20"
GENERATED = "2026-10-21T15:35:47Z"


def _symbols() -> list[str]:
    return [f"S{i:02d}" for i in range(61)] + ["MRK", "T"]


def _index_skip() -> dict:
    return {
        "symbol": "^VIX",
        "reason": "INDEX_ONLY_EXCLUDED",
        "class": "index",
        "expected_session_date": SESSION,
    }


def _invalid_skip(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "reason": "INVALID_PROCESSED_SESSION_BAR",
        "class": "tradable",
        "expected_session_date": SESSION,
        "source_sha256": "ab" * 32,
    }


def _write_signals(path: Path, names: list[str]) -> None:
    rows = [
        {
            "date": SESSION,
            "as_of_date": SESSION,
            "session_as_of_date": SESSION,
            "ticker": name,
            "signal": "HOLD",
            "generated_at_utc": GENERATED,
        }
        for name in names
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_diagnostic(
    path: Path, names: list[str], skips: list[dict], *, have_session: int
) -> None:
    document = {
        "generated_at_utc": GENERATED,
        "session_asof_date": SESSION,
        "effective_asof_date": SESSION,
        "tradable_tickers": len(names),
        "have_session": have_session,
        "coverage": round(have_session / len(names), 4),
        "asof_coverage_threshold": 0.9,
        "batch_actionable": True,
        "fail": False,
        "current_tradable_symbols": list(names),
        "per_symbol_skips": skips,
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _candidate_from_provenance(provenance: dict) -> dict:
    return {
        "bundle_id": provenance["bundle_id"],
        "session_as_of_date": provenance["session_as_of_date"],
        "generated_at_utc": provenance["generated_at_utc"],
        "signals_sha256": provenance["signals_sha256"],
        "expected_symbols": list(provenance["expected_symbols"]),
    }


def _stage_valid(root: Path) -> tuple[Path, Path, pd.DataFrame, dict]:
    staging = root / "stage"
    names = _symbols()
    signals_path = staging / "signals.csv"
    _write_signals(signals_path, names)
    _write_diagnostic(
        staging / PUBLIC_DIAGNOSTIC,
        names,
        [_index_skip()],
        have_session=len(names),
    )
    provenance = bind_staged_diagnostic(staging, signals_path)
    frame = pd.read_csv(signals_path)
    return staging, signals_path, frame, _candidate_from_provenance(provenance)


def _stale_diagnostic(path: Path) -> None:
    document = {
        "generated_at_utc": "2026-10-21T13:31:49Z",
        "session_asof_date": SESSION,
        "tradable_tickers": 63,
        "have_session": 61,
        "coverage": 0.9683,
        "asof_coverage_threshold": 0.9,
        "batch_actionable": True,
        "fail": False,
        "per_symbol_skips": [_invalid_skip("MRK"), _invalid_skip("T")],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _forbid_production_diagnostic(monkeypatch, stale_path: Path) -> None:
    import services.lifecycle_artifact_guard as lag

    monkeypatch.setattr(lag, "ASOF_DIAGNOSTICS_JSON", stale_path)

    def forbid_default(*_args, **_kwargs):
        raise AssertionError("production diagnostic consulted")

    monkeypatch.setattr(lag, "minimum_required_rows", forbid_default)
    monkeypatch.setattr(lag, "_operational_universe_count", forbid_default)
    real_load = lag.load_asof_diagnostics

    def guarded_load(*, path=None):
        if path is None or Path(path) == stale_path:
            raise AssertionError("stale production diagnostic read")
        return real_load(path=path)

    monkeypatch.setattr(lag, "load_asof_diagnostics", guarded_load)


def _seven(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in PUBLISHED_ARTIFACTS:
        (directory / name).write_bytes(f"PRIOR-{name}\n".encode("utf-8"))


def _bytes(directory: Path) -> dict[str, bytes]:
    return {name: (directory / name).read_bytes() for name in PUBLISHED_ARTIFACTS}


def test_valid_63_symbol_candidate_ignores_stale_production_diagnostic(tmp_path, monkeypatch):
    production = tmp_path / "production"
    stale_path = production / PUBLIC_DIAGNOSTIC
    _stale_diagnostic(stale_path)
    before = stale_path.read_bytes()
    _forbid_production_diagnostic(monkeypatch, stale_path)
    staging, signals_path, frame, candidate = _stage_valid(tmp_path)
    gate = evaluate_lifecycle_source_gate(
        frame,
        diagnostics_path=staging / PUBLIC_DIAGNOSTIC,
        candidate=candidate,
        signals_artifact=signals_path,
        positions_path=tmp_path / "empty_positions.csv",
    )
    assert gate["ok"] is True
    assert gate["min_rows"] == 63
    assert gate["accepted_rows"] == 63
    assert {"MRK", "T"} <= set(frame["ticker"].astype(str))
    assert stale_path.read_bytes() == before


def test_lifecycle_main_receives_staged_diagnostic(tmp_path, monkeypatch):
    import services.apply_signal_lifecycle as lifecycle
    import services.lifecycle_artifact_guard as lag

    production = tmp_path / "production"
    stale_path = production / PUBLIC_DIAGNOSTIC
    _stale_diagnostic(stale_path)
    before = stale_path.read_bytes()
    _forbid_production_diagnostic(monkeypatch, stale_path)
    staging, signals_path, _frame, candidate = _stage_valid(tmp_path)
    monkeypatch.setattr(lifecycle, "RESULTS_DIR", staging)
    monkeypatch.setattr(lifecycle, "SIGNALS_RATIONALE", signals_path)
    monkeypatch.setattr(lifecycle, "SIGNALS_FALLBACK", staging / "missing_signals.csv")
    monkeypatch.setattr(lifecycle, "OUT_LIFECYCLE", staging / "signal_lifecycle.csv")
    monkeypatch.setattr(lifecycle, "STATE_PATH", staging / "signal_state.json")
    monkeypatch.setattr(lifecycle, "POSITIONS_SNAPSHOT", tmp_path / "empty_positions.csv")

    def forbid_broker():
        raise AssertionError("broker evidence requested")

    monkeypatch.setattr(lifecycle, "_lifecycle_position_evidence", forbid_broker)
    seen = []
    real_gate = lag.evaluate_lifecycle_source_gate

    def spy(df, **kwargs):
        seen.append(real_gate(df, **kwargs))
        return {"ok": False, "reason": "test_stop", "min_rows": 63, "details": {}}

    monkeypatch.setattr(lag, "evaluate_lifecycle_source_gate", spy)
    rc = lifecycle.main(
        [
            "--diagnostics",
            str(staging / PUBLIC_DIAGNOSTIC),
            "--signals-artifact",
            str(signals_path),
            "--bundle-id",
            candidate["bundle_id"],
            "--candidate-session",
            candidate["session_as_of_date"],
            "--generated-at",
            candidate["generated_at_utc"],
            "--signals-sha256",
            candidate["signals_sha256"],
            "--expected-symbols",
            ",".join(candidate["expected_symbols"]),
        ]
    )
    assert rc == 3
    assert seen[0]["ok"] is True
    assert seen[0]["min_rows"] == 63
    assert stale_path.read_bytes() == before


def test_stale_production_diagnostic_never_satisfies_staged_gate(tmp_path):
    stale_path = tmp_path / PUBLIC_DIAGNOSTIC
    _stale_diagnostic(stale_path)
    before = stale_path.read_bytes()
    _staging, signals_path, frame, candidate = _stage_valid(tmp_path / "valid")
    gate = evaluate_lifecycle_source_gate(
        frame,
        diagnostics_path=stale_path,
        candidate=candidate,
        signals_artifact=signals_path,
        positions_path=tmp_path / "empty_positions.csv",
    )
    assert gate["ok"] is False
    assert gate["reason"] == "staged_diagnostic_unbound"
    assert gate["min_rows"] == 63
    assert stale_path.read_bytes() == before


def test_session_bundle_and_signal_hash_mismatches_fail_closed(tmp_path):
    staging, signals_path, frame, candidate = _stage_valid(tmp_path)
    diagnostic = staging / PUBLIC_DIAGNOSTIC
    session = dict(candidate, session_as_of_date="2026-10-16")
    bundle = dict(candidate, bundle_id="a" * 64)
    hashed = dict(candidate, signals_sha256="b" * 64)
    common = {
        "diagnostics_path": diagnostic,
        "signals_artifact": signals_path,
        "positions_path": tmp_path / "empty_positions.csv",
    }
    assert (
        evaluate_lifecycle_source_gate(frame, candidate=session, **common)["reason"]
        == "staged_diagnostic_session_mismatch"
    )
    assert (
        evaluate_lifecycle_source_gate(frame, candidate=bundle, **common)["reason"]
        == "staged_diagnostic_bundle_mismatch"
    )
    assert (
        evaluate_lifecycle_source_gate(frame, candidate=hashed, **common)["reason"]
        == "staged_diagnostic_signals_hash_mismatch"
    )
    assert evaluate_lifecycle_source_gate(frame, candidate=candidate, **common)["ok"] is True


def test_non_index_tradable_and_unexplained_skips_fail(tmp_path):
    names = _symbols()
    present = [name for name in names if name != "MRK"]
    staging = tmp_path / "stage"
    signals_path = staging / "signals.csv"
    _write_signals(signals_path, present)
    _write_diagnostic(
        staging / PUBLIC_DIAGNOSTIC,
        present,
        [_invalid_skip("MRK"), _index_skip()],
        have_session=len(present),
    )
    provenance = bind_staged_diagnostic(staging, signals_path)
    gate = evaluate_lifecycle_source_gate(
        pd.read_csv(signals_path),
        diagnostics_path=staging / PUBLIC_DIAGNOSTIC,
        candidate=_candidate_from_provenance(provenance),
        signals_artifact=signals_path,
        positions_path=tmp_path / "empty_positions.csv",
    )
    assert gate["ok"] is False
    assert gate["reason"] == "tradable_skip"

    unexplained = tmp_path / "unexplained"
    unexplained_signals = unexplained / "signals.csv"
    _write_signals(unexplained_signals, names)
    _write_diagnostic(
        unexplained / PUBLIC_DIAGNOSTIC,
        names,
        [
            {
                "symbol": "S00",
                "reason": "MISSING_EXPECTED_SESSION_BAR",
                "class": "tradable",
                "expected_session_date": SESSION,
            }
        ],
        have_session=len(names),
    )
    provenance = bind_staged_diagnostic(unexplained, unexplained_signals)
    gate = evaluate_lifecycle_source_gate(
        pd.read_csv(unexplained_signals),
        diagnostics_path=unexplained / PUBLIC_DIAGNOSTIC,
        candidate=_candidate_from_provenance(provenance),
        signals_artifact=unexplained_signals,
        positions_path=tmp_path / "empty_positions.csv",
    )
    assert gate["ok"] is False
    assert gate["reason"] == "unexplained_skip"


def test_failure_preserves_seven_production_artifacts(tmp_path, monkeypatch):
    import services.coherent_signal_bundle as bundle

    production = tmp_path / "production"
    staging = tmp_path / "stage"
    _seven(production)
    before = _bytes(production)
    names = ["AAA"]
    _write_signals(staging / "signals.csv", names)
    _write_diagnostic(
        staging / PUBLIC_DIAGNOSTIC,
        names,
        [_invalid_skip("MRK")],
        have_session=1,
    )
    _fail_stage(production, staging, reason="apply_signal_lifecycle_failed", exit_code=3)
    assert _bytes(production) == before
    receipt = json.loads((production / bundle.FAILURE_RECEIPT).read_text(encoding="utf-8"))
    assert receipt["published_signal_artifacts"] is False
    assert receipt["reason"] == "apply_signal_lifecycle_failed"
    assert receipt["candidate_diagnostic"]["per_symbol_skips"][0]["symbol"] == "MRK"
    assert receipt["candidate_diagnostic"]["candidate_diagnostic_path"].endswith(
        f"signal_bundle_stage/{PUBLIC_DIAGNOSTIC}"
    )

    monkeypatch.setattr(bundle, "RESULTS_DIR", production)
    monkeypatch.setattr(bundle, "STAGING_DIR", staging)
    (staging / bundle.STAGE_MARKER).write_text(
        json.dumps(
            {
                "status": "STAGED",
                "bundle_id": "c" * 64,
                "session_as_of_date": SESSION,
                "generated_at_utc": GENERATED,
                "signals_sha256": "d" * 64,
            }
        ),
        encoding="utf-8",
    )
    assert commit_signal_bundle() == 1
    assert _bytes(production) == before


def test_publication_updates_seven_artifacts_atomically(tmp_path, monkeypatch):
    import services.coherent_signal_bundle as bundle

    production = tmp_path / "production"
    staging = tmp_path / "stage"
    _seven(production)
    before = _bytes(production)
    names = ["AAA", "MRK", "T"]
    _write_signals(staging / "signals.csv", names)
    _write_signals(staging / "signals_with_rationale.csv", names)
    for name in (
        "signal_lifecycle.csv",
        "signal_lifecycle_effective.csv",
        "portfolio_allocation_recommendations.csv",
    ):
        pd.DataFrame(
            [
                {"ticker": name_row, "position_state": "FLAT", "lifecycle_action": "HOLD"}
                for name_row in names
            ]
        ).to_csv(staging / name, index=False)
    (staging / "execution_readiness.json").write_text(
        json.dumps({"paper_only": True, "live_execution_permitted": False}) + "\n",
        encoding="utf-8",
    )
    _write_diagnostic(staging / PUBLIC_DIAGNOSTIC, names, [], have_session=len(names))
    provenance = bind_staged_diagnostic(staging, staging / "signals.csv")
    (staging / bundle.STAGE_MARKER).write_text(
        json.dumps(
            {
                "schema": bundle.SCHEMA_STAGE,
                "status": "STAGED",
                "bundle_id": provenance["bundle_id"],
                "session_as_of_date": SESSION,
                "generated_at_utc": GENERATED,
                "signals_sha256": provenance["signals_sha256"],
                "published_signal_artifacts": False,
            }
        ),
        encoding="utf-8",
    )
    (production / "execution_readiness.json").write_text(
        json.dumps({"paper_only": True}) + "\n",
        encoding="utf-8",
    )
    before = _bytes(production)

    def fake_allocation(argv):
        out = Path(argv[argv.index("--out-csv") + 1])
        pd.DataFrame([{"ticker": "AAA", "recommendation": "HOLD"}]).to_csv(out, index=False)
        return 0

    monkeypatch.setattr(bundle, "RESULTS_DIR", production)
    monkeypatch.setattr(bundle, "STAGING_DIR", staging)
    monkeypatch.setattr(
        "services.portfolio_allocation_engine.main",
        fake_allocation,
    )
    assert commit_signal_bundle() == 0
    published = _bytes(production)
    assert published != before
    for name in PUBLISHED_ARTIFACTS:
        assert published[name] == (staging / name).read_bytes()
    diagnostic = json.loads(published[PUBLIC_DIAGNOSTIC].decode("utf-8"))
    assert diagnostic["candidate_provenance"]["bundle_id"] == provenance["bundle_id"]
    assert diagnostic["candidate_provenance"]["session_as_of_date"] == SESSION
    from services.lifecycle_artifact_guard import file_sha256

    assert diagnostic["candidate_provenance"]["signals_sha256"] == file_sha256(
        production / "signals.csv"
    )

    restored = tmp_path / "restore"
    restore_stage = tmp_path / "restore_stage"
    restore_stage.mkdir()
    _seven(restored)
    prior = _bytes(restored)
    for name in PUBLISHED_ARTIFACTS:
        (restore_stage / name).write_bytes((staging / name).read_bytes())
    real_write = bundle._atomic_bytes

    def fail_last(target: Path, payload: bytes) -> None:
        if target.name == PUBLIC_DIAGNOSTIC:
            raise RuntimeError("diagnostic replace failed")
        real_write(target, payload)

    monkeypatch.setattr(bundle, "_atomic_bytes", fail_last)
    try:
        publish_coherent_bundle(
            restored,
            restore_stage,
            generated_at=GENERATED,
            session=SESSION,
        )
    except RuntimeError as exc:
        assert "diagnostic replace failed" in str(exc)
    else:
        raise AssertionError("partial publication was not rejected")
    assert _bytes(restored) == prior
