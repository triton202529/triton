"""Held-position guard reads the snapshot-bound effective lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from services.build_effective_lifecycle import (
    build_staged_effective_lifecycle,
    read_positions_snapshot_evidence,
)
from services.portfolio_allocation_engine import (
    _load_positions_map,
    _load_signals_map,
    _norm_symbol,
)
from services.symbol_identity import SymbolIdentityCollision, identity_key
from services.coherent_signal_bundle import (
    PUBLISHED_ARTIFACTS,
    assert_effective_held_positions,
    bind_staged_diagnostic,
    commit_signal_bundle,
    guard_staged_held_positions,
    run_staged_allocation,
    write_staged_readiness,
)
from services.lifecycle_artifact_guard import file_sha256

SESSION = "2026-10-20"
GENERATED = "2026-10-21T19:09:59Z"
SNAPSHOT_TS = "2026-10-21T19:05:05Z"
BUNDLE_ID = "cd" * 32
HELD = {"ARKK": 32, "NVDA": 1, "VOO": 2, "XLK": 7}


@pytest.fixture(autouse=True)
def _isolate_lifecycle_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import services.lifecycle_artifact_guard as guard

    monkeypatch.setattr(guard, "DIAGNOSTICS_JSON", tmp_path / "lifecycle_diag.json")
    monkeypatch.setattr(guard, "BACKUPS_DIR", tmp_path / "backups")


def _names() -> list[str]:
    return ["ARKK", "NVDA", "VOO", "XLK"]


def _raw_rows() -> list[dict]:
    rows = []
    for name in _names():
        if name == "ARKK":
            position, action = "FLAT", "WAIT"
        else:
            position, action = "LONG", "HOLD"
        rows.append(
            {
                "ticker": name,
                "stance": action,
                "lifecycle_action": action,
                "position_state": position,
                "delta_pct": 0.0,
                "confidence": 0.9,
            }
        )
    return rows


def _write_raw(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _snapshot(path: Path) -> None:
    rows = [
        {
            "snapshot_ts": SNAPSHOT_TS,
            "ticker": name,
            "symbol": name,
            "qty": qty,
            "side": "long",
            "market_value": float(qty),
        }
        for name, qty in HELD.items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _support(production: Path) -> None:
    production.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["ticker", "mismatch_reason"]).to_csv(
        production / "lifecycle_reconciliation.csv", index=False
    )
    (production / "open_orders_snapshot.csv").write_text("snapshot_ts\n", encoding="utf-8")
    (production / "execution_readiness.json").write_text(
        json.dumps({"paper_only": True}) + "\n", encoding="utf-8"
    )


def _signals(path: Path, names: list[str]) -> None:
    pd.DataFrame(
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
    ).to_csv(path, index=False)


def _diagnostic(path: Path, names: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": GENERATED,
                "session_asof_date": SESSION,
                "effective_asof_date": SESSION,
                "tradable_tickers": len(names),
                "have_session": len(names),
                "coverage": 1.0,
                "current_tradable_symbols": list(names),
                "per_symbol_skips": [],
            }
        ),
        encoding="utf-8",
    )


def _stage(root: Path, rows: list[dict] | None = None) -> tuple[Path, Path]:
    staging = root / "stage"
    production = root / "production"
    names = _names()
    _support(production)
    _snapshot(production / "positions_snapshot.csv")
    _write_raw(staging / "signal_lifecycle.csv", rows or _raw_rows())
    evidence = read_positions_snapshot_evidence(production / "positions_snapshot.csv")
    build_staged_effective_lifecycle(
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
    return production, staging


def _row(frame: pd.DataFrame, symbol: str) -> pd.Series:
    return frame[frame["ticker"].astype(str) == symbol].iloc[0]


def _plant_seven(production: Path) -> dict[str, bytes]:
    production.mkdir(parents=True, exist_ok=True)
    for name in PUBLISHED_ARTIFACTS:
        (production / name).write_bytes(f"PRIOR-{name}\n".encode("utf-8"))
    return {name: (production / name).read_bytes() for name in PUBLISHED_ARTIFACTS}


def test_arkk_flat_wait_becomes_effective_long_hold_and_progresses(tmp_path: Path) -> None:
    production, staging = _stage(tmp_path)
    raw_path = staging / "signal_lifecycle.csv"
    raw_bytes = raw_path.read_bytes()
    guard_staged_held_positions(production, staging)
    assert raw_path.read_bytes() == raw_bytes
    raw = _row(pd.read_csv(raw_path), "ARKK")
    assert raw["position_state"] == "FLAT"
    assert raw["lifecycle_action"] == "WAIT"

    effective = pd.read_csv(staging / "signal_lifecycle_effective.csv")
    arkk = _row(effective, "ARKK")
    assert arkk["effective_position_state"] == "LONG"
    assert arkk["effective_stance"] == "HOLD"
    nvda = _row(effective, "NVDA")
    assert nvda["effective_position_state"] == "LONG"
    assert nvda["effective_stance"] == "HOLD"
    for symbol in ("VOO", "XLK"):
        row = _row(effective, symbol)
        assert row["effective_position_state"] == "LONG"
        assert row["effective_stance"] == "HOLD"
        assert row["basis_classification"] == "BASIS_UNPROVEN_HOLD"

    names = _names()
    _signals(staging / "signals_with_rationale.csv", names)
    assert run_staged_allocation(production, staging) == 0
    write_staged_readiness(production, staging, generated_at=GENERATED, session=SESSION)
    allocation = pd.read_csv(staging / "portfolio_allocation_recommendations.csv")
    readiness = json.loads((staging / "execution_readiness.json").read_text(encoding="utf-8"))
    assert set(allocation["ticker"].astype(str)) == set(names)
    assert readiness["live_execution_permitted"] is False
    assert raw_path.read_bytes() == raw_bytes


def test_valid_exit_remains_long_exit_pending_terminal_sell(tmp_path: Path) -> None:
    rows = _raw_rows()
    for row in rows:
        if row["ticker"] == "NVDA":
            row["stance"] = "EXIT"
            row["lifecycle_action"] = "EXIT"
            row["position_state"] = "LONG"
    production, staging = _stage(tmp_path, rows)
    raw_path = staging / "signal_lifecycle.csv"
    raw_bytes = raw_path.read_bytes()
    guard_staged_held_positions(production, staging)
    assert raw_path.read_bytes() == raw_bytes
    raw = _row(pd.read_csv(raw_path), "NVDA")
    assert raw["position_state"] == "LONG"
    assert raw["lifecycle_action"] == "EXIT"
    nvda = _row(pd.read_csv(staging / "signal_lifecycle_effective.csv"), "NVDA")
    assert nvda["effective_position_state"] == "LONG"
    assert nvda["effective_stance"] == "EXIT"


def test_missing_or_flat_effective_lifecycle_fails_closed(tmp_path: Path, monkeypatch) -> None:
    import services.coherent_signal_bundle as bundle

    production, staging = _stage(tmp_path)
    before = _plant_seven(production)
    evidence = read_positions_snapshot_evidence(production / "positions_snapshot.csv")
    frame = pd.read_csv(staging / "signal_lifecycle_effective.csv")
    frame = frame[frame["ticker"].astype(str) != "ARKK"]
    frame.to_csv(staging / "signal_lifecycle_effective.csv", index=False)
    with pytest.raises(RuntimeError, match="held symbol ARKK missing from effective lifecycle"):
        guard_staged_held_positions(production, staging)
    bundle._fail_stage(
        production,
        staging,
        reason="held_position_guard",
        detail="held symbol ARKK missing from effective lifecycle",
    )
    assert {name: (production / name).read_bytes() for name in PUBLISHED_ARTIFACTS} == before
    assert not (staging / "portfolio_allocation_recommendations.csv").is_file()

    flat = pd.read_csv(staging / "signal_lifecycle_effective.csv")
    flat.loc[flat["ticker"].astype(str) == "NVDA", "effective_position_state"] = "FLAT"
    with pytest.raises(RuntimeError, match="terminal_flat_unproven"):
        assert_effective_held_positions(flat, {"NVDA"}, evidence)

    names = ["NVDA"]
    _signals(staging / "signals.csv", names)
    _diagnostic(staging / bundle.PUBLIC_DIAGNOSTIC, names)
    provenance = bind_staged_diagnostic(staging, staging / "signals.csv")
    lifecycle_hash = file_sha256(staging / "signal_lifecycle.csv")
    pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "candidate_bundle_id": provenance["bundle_id"],
                "candidate_session_as_of_date": SESSION,
                "candidate_lifecycle_sha256": lifecycle_hash,
                "positions_snapshot_sha256": evidence["sha256"],
                "positions_snapshot_ts": evidence["snapshot_ts"],
                "effective_position_state": "LONG",
                "effective_stance": "HOLD",
            }
        ]
    ).to_csv(staging / "signal_lifecycle_effective.csv", index=False)
    (staging / bundle.STAGE_MARKER).write_text(
        json.dumps(
            {
                "schema": bundle.SCHEMA_STAGE,
                "status": "STAGED",
                "bundle_id": provenance["bundle_id"],
                "session_as_of_date": SESSION,
                "generated_at_utc": GENERATED,
                "signals_sha256": provenance["signals_sha256"],
                "lifecycle_sha256": lifecycle_hash,
                "positions_snapshot_sha256": evidence["sha256"],
                "positions_snapshot_ts": evidence["snapshot_ts"],
                "expected_symbols": names,
            }
        ),
        encoding="utf-8",
    )
    called = {"allocation": 0}

    def _refuse_allocation(*_args, **_kwargs) -> int:
        called["allocation"] += 1
        return 0

    monkeypatch.setattr(bundle, "run_staged_allocation", _refuse_allocation)
    monkeypatch.setattr(bundle, "RESULTS_DIR", production)
    monkeypatch.setattr(bundle, "STAGING_DIR", staging)
    assert commit_signal_bundle() == 1
    assert called["allocation"] == 0
    assert {name: (production / name).read_bytes() for name in PUBLISHED_ARTIFACTS} == before
    receipt = json.loads((production / bundle.FAILURE_RECEIPT).read_text(encoding="utf-8"))
    assert receipt["published_signal_artifacts"] is False
    assert receipt["reason"] == "held_position_guard"
    assert "ARKK" in receipt["detail"]


def test_stale_or_mismatched_snapshot_fails_closed(tmp_path: Path) -> None:
    import services.coherent_signal_bundle as bundle

    production, staging = _stage(tmp_path)
    before = _plant_seven(production)
    raw_bytes = (staging / "signal_lifecycle.csv").read_bytes()
    snapshot = production / "positions_snapshot.csv"
    snapshot.write_text(
        snapshot.read_text(encoding="utf-8").replace(SNAPSHOT_TS, "2026-10-21T19:40:05Z"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="positions_snapshot_hash_mismatch"):
        guard_staged_held_positions(production, staging)
    bundle._fail_stage(
        production,
        staging,
        reason="held_position_guard",
        detail="positions_snapshot_hash_mismatch",
    )
    assert (staging / "signal_lifecycle.csv").read_bytes() == raw_bytes
    assert {name: (production / name).read_bytes() for name in PUBLISHED_ARTIFACTS} == before
    assert not (staging / "portfolio_allocation_recommendations.csv").is_file()
    receipt = json.loads((production / bundle.FAILURE_RECEIPT).read_text(encoding="utf-8"))
    assert receipt["published_signal_artifacts"] is False


def _held_snapshot(path: Path, held: dict[str, float]) -> None:
    rows = [
        {
            "snapshot_ts": SNAPSHOT_TS,
            "ticker": name,
            "symbol": name,
            "qty": qty,
            "side": "long",
            "market_value": float(qty),
        }
        for name, qty in held.items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _lifecycle_rows(names: list[str], flat_wait: set[str] | None = None) -> list[dict]:
    waiting = flat_wait or set()
    rows = []
    for name in names:
        if name in waiting:
            position, action = "FLAT", "WAIT"
        else:
            position, action = "LONG", "HOLD"
        rows.append(
            {
                "ticker": name,
                "stance": action,
                "lifecycle_action": action,
                "position_state": position,
                "delta_pct": 0.0,
                "confidence": 0.9,
            }
        )
    return rows


def _stage_symbols(
    root: Path,
    names: list[str],
    held: dict[str, float],
    *,
    flat_wait: set[str] | None = None,
) -> tuple[Path, Path]:
    staging = root / "stage"
    production = root / "production"
    _support(production)
    _held_snapshot(production / "positions_snapshot.csv", held)
    _write_raw(staging / "signal_lifecycle.csv", _lifecycle_rows(names, flat_wait))
    evidence = read_positions_snapshot_evidence(production / "positions_snapshot.csv")
    build_staged_effective_lifecycle(
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
    return production, staging


def test_brk_snapshot_alias_keeps_one_candidate_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import services.coherent_signal_bundle as bundle

    names = ["ARKK", "BRK-B", "NVDA", "VOO", "XLK"]
    held = {"ARKK": 32.0, "BRK.B": 2.0, "NVDA": 1.0, "VOO": 2.0, "XLK": 7.0}
    production, staging = _stage_symbols(tmp_path, names, held, flat_wait={"BRK-B"})
    snapshot = production / "positions_snapshot.csv"
    snapshot_bytes = snapshot.read_bytes()
    raw_bytes = (staging / "signal_lifecycle.csv").read_bytes()
    effective = pd.read_csv(staging / "signal_lifecycle_effective.csv")
    assert len(effective.index) == len(names)
    assert set(effective["ticker"].astype(str)) == set(names)
    assert list(effective["ticker"].astype(str)).count("BRK-B") == 1
    assert "BRK.B" not in set(effective["ticker"].astype(str))
    brk = _row(effective, "BRK-B")
    assert brk["effective_position_state"] == "LONG"
    for symbol in ("VOO", "XLK"):
        row = _row(effective, symbol)
        assert row["effective_position_state"] == "LONG"
        assert row["effective_stance"] == "HOLD"
        assert row["basis_classification"] == "BASIS_UNPROVEN_HOLD"

    guard_staged_held_positions(production, staging)
    assert snapshot.read_bytes() == snapshot_bytes
    assert (staging / "signal_lifecycle.csv").read_bytes() == raw_bytes
    guarded = pd.read_csv(staging / "signal_lifecycle_effective.csv")
    assert list(guarded["ticker"].astype(str)).count("BRK-B") == 1
    assert "BRK.B" not in set(guarded["ticker"].astype(str))
    brk = _row(guarded, "BRK-B")
    assert brk["effective_position_state"] == "LONG"
    assert brk["effective_stance"] == "HOLD"
    assert identity_key("BRK.B") == identity_key("BRK-B")

    _signals(staging / "signals_with_rationale.csv", names)
    assert run_staged_allocation(production, staging) == 0
    allocation = pd.read_csv(staging / "portfolio_allocation_recommendations.csv")
    assert set(allocation["ticker"].astype(str)) == set(names)
    assert list(allocation["ticker"].astype(str)).count("BRK-B") == 1
    assert "BRK.B" not in set(allocation["ticker"].astype(str))
    marker = {
        "positions_snapshot_sha256": str(guarded["positions_snapshot_sha256"].iloc[0]),
        "positions_snapshot_ts": str(guarded["positions_snapshot_ts"].iloc[0]),
        "bundle_id": str(guarded["candidate_bundle_id"].iloc[0]),
        "session_as_of_date": str(guarded["candidate_session_as_of_date"].iloc[0]),
        "lifecycle_sha256": str(guarded["candidate_lifecycle_sha256"].iloc[0]),
        "expected_symbols": names,
    }
    monkeypatch.setattr(bundle, "RESULTS_DIR", production)
    bundle._require_effective_snapshot(staging, marker)
    bundle._require_allocation_symbols(staging, marker)
    assert snapshot.read_bytes() == snapshot_bytes


def test_conflicting_brk_candidate_spellings_fail_closed(tmp_path: Path) -> None:
    names = ["BRK-B", "BRK.B", "VOO"]
    staging = tmp_path / "stage"
    production = tmp_path / "production"
    _support(production)
    snapshot = production / "positions_snapshot.csv"
    _held_snapshot(snapshot, {"BRK.B": 2.0, "VOO": 2.0})
    snapshot_bytes = snapshot.read_bytes()
    _write_raw(staging / "signal_lifecycle.csv", _lifecycle_rows(names))
    evidence = read_positions_snapshot_evidence(snapshot)
    with pytest.raises(SymbolIdentityCollision):
        build_staged_effective_lifecycle(
            lifecycle_path=staging / "signal_lifecycle.csv",
            reconciliation_path=production / "lifecycle_reconciliation.csv",
            open_orders_path=production / "open_orders_snapshot.csv",
            output_path=staging / "signal_lifecycle_effective.csv",
            positions_snapshot=snapshot,
            snapshot_evidence=evidence,
            bundle_id=BUNDLE_ID,
            session_as_of_date=SESSION,
            lifecycle_sha256=file_sha256(staging / "signal_lifecycle.csv"),
            expected_symbols=names,
            min_rows=len(names),
        )
    assert snapshot.read_bytes() == snapshot_bytes
    assert not (staging / "signal_lifecycle_effective.csv").is_file()


def test_unsupported_alias_fails_closed_without_a_new_row(tmp_path: Path) -> None:
    names = ["VOO", "XLK", "XYZ-B"]
    held = {"VOO": 2.0, "XLK": 7.0, "XYZ.B": 1.0}
    production, staging = _stage_symbols(tmp_path, names, held, flat_wait={"XYZ-B"})
    snapshot = production / "positions_snapshot.csv"
    snapshot_bytes = snapshot.read_bytes()
    effective = pd.read_csv(staging / "signal_lifecycle_effective.csv")
    assert set(effective["ticker"].astype(str)) == set(names)
    assert "XYZ.B" not in set(effective["ticker"].astype(str))
    assert identity_key("XYZ.B") != identity_key("XYZ-B")
    for symbol in ("VOO", "XLK"):
        row = _row(effective, symbol)
        assert row["basis_classification"] == "BASIS_UNPROVEN_HOLD"
    with pytest.raises(RuntimeError, match="held symbol XYZ.B missing from effective lifecycle"):
        guard_staged_held_positions(production, staging)
    assert snapshot.read_bytes() == snapshot_bytes
    after = pd.read_csv(staging / "signal_lifecycle_effective.csv")
    assert set(after["ticker"].astype(str)) == set(names)


def test_duplicate_brk_effective_rows_fail_closed(tmp_path: Path) -> None:
    production, staging = _stage(tmp_path)
    snapshot_bytes = (production / "positions_snapshot.csv").read_bytes()
    raw_bytes = (staging / "signal_lifecycle.csv").read_bytes()
    frame = pd.read_csv(staging / "signal_lifecycle_effective.csv")
    frame.loc[frame["ticker"].astype(str) == "ARKK", "ticker"] = "BRK-B"
    extra = frame.iloc[0].copy()
    extra["ticker"] = "BRK.B"
    frame = pd.concat([frame, pd.DataFrame([extra])], ignore_index=True)
    frame.to_csv(staging / "signal_lifecycle_effective.csv", index=False)
    with pytest.raises(RuntimeError, match="symbol_identity_collision"):
        guard_staged_held_positions(production, staging)
    assert (production / "positions_snapshot.csv").read_bytes() == snapshot_bytes
    assert (staging / "signal_lifecycle.csv").read_bytes() == raw_bytes


def test_allocation_join_rejects_conflicting_alias_and_keeps_unknown_forms() -> None:
    conflict = pd.DataFrame(
        {
            "ticker": ["BRK-B", "BRK.B"],
            "qty": [2.0, 4.0],
            "market_value": [10.0, 20.0],
            "signal": ["HOLD", "HOLD"],
        }
    )
    with pytest.raises(SymbolIdentityCollision):
        _load_positions_map(conflict)
    with pytest.raises(SymbolIdentityCollision):
        _load_signals_map(conflict)
    separate = pd.DataFrame(
        {
            "ticker": ["XYZ-B", "XYZ.B"],
            "qty": [1.0, 3.0],
            "market_value": [5.0, 9.0],
        }
    )
    loaded = _load_positions_map(separate)
    assert set(loaded) == {"XYZ-B", "XYZ.B"}
    assert loaded["XYZ.B"]["qty"] == 3.0
    assert loaded["XYZ-B"]["qty"] == 1.0
    assert _norm_symbol("BRK-B") == identity_key("BRK-B") == "BRK.B"
    assert _norm_symbol("VOO") == "VOO"
    assert _norm_symbol("XLK") == "XLK"
