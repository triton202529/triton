"""Staged effective-lifecycle invocation for one coherent bundle candidate."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from services.build_effective_lifecycle import (
    build_staged_effective_lifecycle,
    main,
    read_positions_snapshot_evidence,
)
from services.coherent_signal_bundle import (
    PUBLISHED_ARTIFACTS,
    bind_staged_diagnostic,
    commit_signal_bundle,
    run_staged_allocation,
    stage_effective_from_candidate,
    write_staged_readiness,
)
from services.lifecycle_artifact_guard import file_sha256

SESSION = "2026-10-20"
GENERATED = "2026-10-21T17:06:08Z"
SNAPSHOT_TS = "2026-10-21T16:45:05Z"
BUNDLE_ID = "ab" * 32


@pytest.fixture(autouse=True)
def _isolate_lifecycle_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import services.lifecycle_artifact_guard as guard

    monkeypatch.setattr(guard, "DIAGNOSTICS_JSON", tmp_path / "lifecycle_diag.json")
    monkeypatch.setattr(guard, "BACKUPS_DIR", tmp_path / "backups")


def _symbols() -> list[str]:
    return [f"S{index:02d}" for index in range(58)] + ["MRK", "T", "NVDA", "VOO", "XLK"]


def _lifecycle(path: Path, names: list[str], *, nvda_action: str = "HOLD") -> None:
    rows = []
    for name in names:
        held = name in {"NVDA", "VOO", "XLK"}
        action = nvda_action if name == "NVDA" else ("HOLD" if held else "HOLD")
        rows.append(
            {
                "ticker": name,
                "stance": action,
                "lifecycle_action": action,
                "position_state": "LONG" if held else "FLAT",
                "delta_pct": -0.01 if name == "NVDA" else 0.0,
                "confidence": 0.9,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _snapshot(path: Path, names: list[str]) -> None:
    rows = [
        {
            "snapshot_ts": SNAPSHOT_TS,
            "ticker": name,
            "symbol": name,
            "qty": 1 if name in {"NVDA", "VOO", "XLK"} else 0,
            "side": "long",
            "market_value": 10 if name in {"NVDA", "VOO", "XLK"} else 0,
        }
        for name in names
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _recon(path: Path, rows: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows or [], columns=["ticker", "mismatch_reason"]).to_csv(path, index=False)


def _orders(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("snapshot_ts\n", encoding="utf-8")


def _signals(path: Path, names: list[str]) -> None:
    frame = pd.DataFrame(
        [
            {
                "date": SESSION,
                "session_as_of_date": SESSION,
                "ticker": name,
                "signal": "HOLD",
                "generated_at_utc": GENERATED,
            }
            for name in names
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _diagnostic(path: Path, names: list[str]) -> None:
    document = {
        "generated_at_utc": GENERATED,
        "session_asof_date": SESSION,
        "effective_asof_date": SESSION,
        "tradable_tickers": len(names),
        "have_session": len(names),
        "coverage": 1.0,
        "current_tradable_symbols": list(names),
        "per_symbol_skips": [],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _build(root: Path, names: list[str], *, recon_rows: list[dict] | None = None) -> dict:
    staging = root / "stage"
    production = root / "production"
    _lifecycle(staging / "signal_lifecycle.csv", names)
    _snapshot(production / "positions_snapshot.csv", names)
    _recon(production / "lifecycle_reconciliation.csv", recon_rows)
    _orders(production / "open_orders_snapshot.csv")
    evidence = read_positions_snapshot_evidence(production / "positions_snapshot.csv")
    return build_staged_effective_lifecycle(
        lifecycle_path=staging / "signal_lifecycle.csv",
        reconciliation_path=production / "lifecycle_reconciliation.csv",
        open_orders_path=production / "open_orders_snapshot.csv",
        output_path=staging / "signal_lifecycle_effective.csv",
        positions_snapshot=production / "positions_snapshot.csv",
        snapshot_evidence=evidence,
        bundle_id=BUNDLE_ID,
        session_as_of_date=SESSION,
        lifecycle_sha256=file_sha256(staging / "signal_lifecycle.csv"),
        expected_symbols=names,
        min_rows=len(names),
    )


def _row(frame: pd.DataFrame, symbol: str) -> pd.Series:
    return frame[frame["ticker"].astype(str) == symbol].iloc[0]


def test_staged_bundle_calls_reusable_api_not_cli_main(tmp_path: Path, monkeypatch) -> None:
    import services.build_effective_lifecycle as effective

    names = ["NVDA"]
    staging = tmp_path / "stage"
    production = tmp_path / "production"
    _lifecycle(staging / "signal_lifecycle.csv", names)
    _snapshot(production / "positions_snapshot.csv", names)
    _recon(production / "lifecycle_reconciliation.csv")
    _orders(production / "open_orders_snapshot.csv")
    seen: list[dict] = []
    real = effective.build_staged_effective_lifecycle

    def spy(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    def cli_main() -> int:
        raise AssertionError("cli main called")

    monkeypatch.setattr(effective, "build_staged_effective_lifecycle", spy)
    monkeypatch.setattr(effective, "main", cli_main)
    rc = stage_effective_from_candidate(
        production,
        staging,
        {
            "bundle_id": BUNDLE_ID,
            "session_as_of_date": SESSION,
            "expected_symbols": names,
        },
    )
    assert rc == 0
    assert seen
    assert seen[0]["lifecycle_path"] == staging / "signal_lifecycle.csv"
    assert seen[0]["output_path"] == staging / "signal_lifecycle_effective.csv"
    assert "snapshot_evidence" in seen[0]
    frame = pd.read_csv(staging / "signal_lifecycle_effective.csv")
    assert frame["candidate_bundle_id"].iloc[0] == BUNDLE_ID
    assert frame["candidate_session_as_of_date"].iloc[0] == SESSION


def test_cli_main_accepts_no_arguments(tmp_path: Path, monkeypatch) -> None:
    import services.lifecycle_artifact_guard as guard

    assert list(inspect.signature(main).parameters) == []
    lifecycle = tmp_path / "signal_lifecycle.csv"
    _lifecycle(lifecycle, ["NVDA"])
    _recon(tmp_path / "recon.csv")
    _orders(tmp_path / "orders.csv")
    output = tmp_path / "effective.csv"
    monkeypatch.setattr(guard, "minimum_required_rows", lambda: 1)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_effective_lifecycle",
            "--lifecycle",
            str(lifecycle),
            "--recon",
            str(tmp_path / "recon.csv"),
            "--open-orders",
            str(tmp_path / "orders.csv"),
            "--out",
            str(output),
        ],
    )
    assert main() == 0
    assert output.is_file()
    assert len(pd.read_csv(output).index) == 1


def test_63_symbol_candidate_completes_effective_allocation_and_readiness(tmp_path: Path) -> None:
    names = _symbols()
    assert len(names) == 63
    result = _build(tmp_path, names)
    staging = tmp_path / "stage"
    production = tmp_path / "production"
    _signals(staging / "signals_with_rationale.csv", names)
    (production / "execution_readiness.json").write_text(
        json.dumps({"paper_only": True}) + "\n",
        encoding="utf-8",
    )
    assert run_staged_allocation(production, staging) == 0
    write_staged_readiness(production, staging, generated_at=GENERATED, session=SESSION)
    effective = pd.read_csv(staging / "signal_lifecycle_effective.csv")
    allocation = pd.read_csv(staging / "portfolio_allocation_recommendations.csv")
    readiness = json.loads((staging / "execution_readiness.json").read_text(encoding="utf-8"))
    assert result["rows"] == 63
    assert set(effective["ticker"].astype(str)) == set(names)
    assert set(allocation["ticker"].astype(str)) == set(names)
    assert effective["candidate_lifecycle_sha256"].iloc[0] == file_sha256(
        staging / "signal_lifecycle.csv"
    )
    assert effective["positions_snapshot_sha256"].iloc[0] == file_sha256(
        production / "positions_snapshot.csv"
    )
    assert set(effective["positions_snapshot_ts"].astype(str)) == {SNAPSHOT_TS}
    nvda = _row(effective, "NVDA")
    assert nvda["effective_position_state"] == "LONG"
    assert nvda["effective_stance"] == "HOLD"
    for symbol in ("VOO", "XLK"):
        row = _row(effective, symbol)
        assert row["effective_position_state"] == "LONG"
        assert row["effective_stance"] == "HOLD"
        assert row["basis_classification"] == "BASIS_UNPROVEN_HOLD"
    assert readiness["bundle_session_as_of_date"] == SESSION
    assert readiness["bundle_generated_at_utc"] == GENERATED
    assert readiness["live_execution_permitted"] is False


def test_snapshot_hash_or_timestamp_mismatch_fails_closed(tmp_path: Path) -> None:
    names = ["NVDA", "VOO", "XLK"]
    staging = tmp_path / "stage"
    production = tmp_path / "production"
    _lifecycle(staging / "signal_lifecycle.csv", names)
    snapshot = production / "positions_snapshot.csv"
    _snapshot(snapshot, names)
    _recon(production / "lifecycle_reconciliation.csv")
    _orders(production / "open_orders_snapshot.csv")
    evidence = read_positions_snapshot_evidence(snapshot)
    lifecycle_hash = file_sha256(staging / "signal_lifecycle.csv")
    common = {
        "lifecycle_path": staging / "signal_lifecycle.csv",
        "reconciliation_path": production / "lifecycle_reconciliation.csv",
        "open_orders_path": production / "open_orders_snapshot.csv",
        "output_path": staging / "signal_lifecycle_effective.csv",
        "positions_snapshot": snapshot,
        "bundle_id": BUNDLE_ID,
        "session_as_of_date": SESSION,
        "lifecycle_sha256": lifecycle_hash,
        "expected_symbols": names,
    }
    with pytest.raises(ValueError, match="positions_snapshot_hash_mismatch"):
        build_staged_effective_lifecycle(
            **common,
            snapshot_evidence={"sha256": "0" * 64, "snapshot_ts": evidence["snapshot_ts"]},
        )
    with pytest.raises(ValueError, match="positions_snapshot_timestamp_mismatch"):
        build_staged_effective_lifecycle(
            **common,
            snapshot_evidence={"sha256": evidence["sha256"], "snapshot_ts": "2026-10-21T16:10:14Z"},
        )
    snapshot.write_text(snapshot.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positions_snapshot_hash_mismatch"):
        build_staged_effective_lifecycle(**common, snapshot_evidence=evidence)
    missing = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match="positions_snapshot_missing"):
        read_positions_snapshot_evidence(missing)
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("ticker,qty\nNVDA,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positions_snapshot_malformed"):
        read_positions_snapshot_evidence(malformed)


def test_nvda_candidate_hold_stays_long(tmp_path: Path) -> None:
    names = ["NVDA", "VOO", "XLK"]
    result = _build(
        tmp_path,
        names,
        recon_rows=[{"ticker": "NVDA", "mismatch_reason": "lifecycle_long_broker_flat"}],
    )
    frame = pd.read_csv(result["output_path"])
    nvda = _row(frame, "NVDA")
    assert nvda["effective_position_state"] == "LONG"
    assert nvda["effective_stance"] == "HOLD"
    assert nvda["lifecycle_action"] == "HOLD"
    for symbol in ("VOO", "XLK"):
        row = _row(frame, symbol)
        assert row["effective_position_state"] == "LONG"
        assert row["effective_stance"] == "HOLD"
        assert row["basis_classification"] == "BASIS_UNPROVEN_HOLD"


def test_effective_failure_preserves_seven_production_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    import services.coherent_signal_bundle as bundle

    production = tmp_path / "production"
    staging = tmp_path / "stage"
    production.mkdir()
    names = ["NVDA"]
    _lifecycle(staging / "signal_lifecycle.csv", names)
    _recon(production / "lifecycle_reconciliation.csv")
    _orders(production / "open_orders_snapshot.csv")
    for name in PUBLISHED_ARTIFACTS:
        (production / name).write_bytes(f"PRIOR-{name}\n".encode("utf-8"))
    before = {name: (production / name).read_bytes() for name in PUBLISHED_ARTIFACTS}
    rc = stage_effective_from_candidate(
        production,
        staging,
        {"bundle_id": BUNDLE_ID, "session_as_of_date": SESSION, "expected_symbols": names},
    )
    assert rc == 1
    assert {name: (production / name).read_bytes() for name in PUBLISHED_ARTIFACTS} == before
    receipt = json.loads((production / bundle.FAILURE_RECEIPT).read_text(encoding="utf-8"))
    assert receipt["published_signal_artifacts"] is False
    assert receipt["reason"] == "build_effective_lifecycle_failed"
    assert "positions_snapshot_missing" in receipt["detail"]

    _signals(staging / "signals.csv", names)
    _signals(staging / "signals_with_rationale.csv", names)
    _diagnostic(staging / bundle.PUBLIC_DIAGNOSTIC, names)
    provenance = bind_staged_diagnostic(staging, staging / "signals.csv")
    _snapshot(production / "positions_snapshot.csv", names)
    evidence = read_positions_snapshot_evidence(production / "positions_snapshot.csv")
    _lifecycle(staging / "signal_lifecycle.csv", names)
    pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "candidate_bundle_id": provenance["bundle_id"],
                "candidate_session_as_of_date": SESSION,
                "candidate_lifecycle_sha256": file_sha256(staging / "signal_lifecycle.csv"),
                "positions_snapshot_sha256": evidence["sha256"],
                "positions_snapshot_ts": evidence["snapshot_ts"],
                "effective_position_state": "LONG",
                "effective_stance": "HOLD",
            }
        ]
    ).to_csv(staging / "signal_lifecycle_effective.csv", index=False)
    snapshot_text = (production / "positions_snapshot.csv").read_text(encoding="utf-8")
    (production / "positions_snapshot.csv").write_text(
        snapshot_text.replace(SNAPSHOT_TS, "2026-10-21T16:45:06Z"),
        encoding="utf-8",
    )
    (staging / bundle.STAGE_MARKER).write_text(
        json.dumps(
            {
                "schema": bundle.SCHEMA_STAGE,
                "status": "STAGED",
                "bundle_id": provenance["bundle_id"],
                "session_as_of_date": SESSION,
                "generated_at_utc": GENERATED,
                "signals_sha256": provenance["signals_sha256"],
                "lifecycle_sha256": file_sha256(staging / "signal_lifecycle.csv"),
                "positions_snapshot_sha256": evidence["sha256"],
                "positions_snapshot_ts": evidence["snapshot_ts"],
                "expected_symbols": names,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bundle, "RESULTS_DIR", production)
    monkeypatch.setattr(bundle, "STAGING_DIR", staging)
    assert commit_signal_bundle() == 1
    assert {name: (production / name).read_bytes() for name in PUBLISHED_ARTIFACTS} == before
    receipt = json.loads((production / bundle.FAILURE_RECEIPT).read_text(encoding="utf-8"))
    assert receipt["published_signal_artifacts"] is False
    assert receipt["reason"] == "effective_lifecycle_rejected"
    assert receipt["detail"] == "positions_snapshot_changed"
