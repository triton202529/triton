"""SF50 already-current raw evidence. No production writes."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

SESSION = date(2026, 9, 24)
RUN = "daily_prediction_refresh_20260924_sf50"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw(path: Path, symbol: str, *, session_rows: int = 1, future: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in range(5):
        day = SESSION - timedelta(days=5 - offset)
        rows.append(
            {
                "date": day.isoformat(),
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.2,
                "volume": 10,
                "ticker": symbol,
            }
        )
    for _ in range(session_rows):
        rows.append(
            {
                "date": SESSION.isoformat(),
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.3,
                "volume": 11,
                "ticker": symbol,
            }
        )
    if future:
        nxt = SESSION + timedelta(days=1)
        rows.append(
            {
                "date": nxt.isoformat(),
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.4,
                "volume": 12,
                "ticker": symbol,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _symbols(tmp: Path) -> list[str]:
    return [
        line.strip()
        for line in (tmp / "data/config/tickers.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _layout(tmp: Path, symbols: list[str]) -> Path:
    for folder in ("data/config", "data/raw", "data/processed", "data/predictions", "data/results"):
        (tmp / folder).mkdir(parents=True)
    (tmp / "data/config/tickers.txt").write_text("\n".join(symbols) + "\n", encoding="utf-8")
    for symbol in symbols:
        _raw(tmp / "data/raw" / f"{symbol}.csv", symbol)
        pd.DataFrame({"date": [SESSION.isoformat()], "close": [1.0]}).to_parquet(
            tmp / "data/processed" / f"{symbol}.parquet"
        )
        pd.DataFrame(
            {
                "date": [SESSION.isoformat()],
                "close": [1.0],
                "predicted_close": [1.1],
                "signal": ["HOLD"],
                "ticker": [symbol],
                "run_id": [RUN],
                "completed_session": [SESSION.isoformat()],
                "processed_sha256": ["abc"],
                "model_identity": ["RandomForest+GradientBoosting+LinearRegression"],
            }
        ).to_parquet(tmp / "data/predictions" / f"{symbol}_predictions.parquet")
    return tmp


def _fetch(tmp: Path, symbols: list[str], **overrides) -> None:
    decisions = [
        {
            "symbol": symbol,
            "action": "keep",
            "reason": "local_already_current",
            "local_max": SESSION.isoformat(),
            "operationally_eligible": True,
        }
        for symbol in symbols
    ]
    tradables = [symbol for symbol in symbols if not symbol.startswith("^")]
    body = {
        "target_session": SESSION.isoformat(),
        "run_id": RUN,
        "passed": True,
        "promoted": [],
        "tradable_coverage": {
            "covered": len(tradables),
            "denominator": len(tradables),
            "ratio": 1.0,
            "minimum": 0.9,
        },
        "decisions": decisions,
    }
    body.update(overrides)
    (tmp / "data/results/daily_refresh_fetch_receipt.json").write_text(
        json.dumps(body), encoding="utf-8"
    )


def _bind_predictions(tmp: Path, symbols: list[str]) -> None:
    pred_hashes = {}
    proc_hashes = {}
    for symbol in symbols:
        proc = tmp / "data/processed" / f"{symbol}.parquet"
        pred = tmp / "data/predictions" / f"{symbol}_predictions.parquet"
        proc_hash = _sha(proc)
        proc_hashes[symbol] = proc_hash
        frame = pd.read_parquet(pred)
        frame["processed_sha256"] = proc_hash
        frame.to_parquet(pred, index=False)
        pred_hashes[symbol] = _sha(pred)
    feature = tmp / "data/results/feature_importance.csv"
    model = tmp / "data/results/model_comparison.csv"
    feature.write_text("feature,importance\nx,1\n", encoding="utf-8")
    model.write_text("ticker,model\nPLTR,rf\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "run_id": RUN,
        "generated_at": "2026-09-24T23:00:00+00:00",
        "completed_session": SESSION.isoformat(),
        "governed_universe_sha256": _sha(tmp / "data/config/tickers.txt"),
        "required_count": len(symbols),
        "produced_count": len(symbols),
        "skipped_symbols": [],
        "processed_input_hashes": proc_hashes,
        "prediction_output_hashes": pred_hashes,
        "feature_importance_sha256": _sha(feature),
        "model_comparison_sha256": _sha(model),
        "publication_status": "COMPLETE",
    }
    (tmp / "data/results/training_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _other_stages():
    return [
        {"name": "stale_data_gate", "exit_code": 0, "outcome": "ok"},
        {"name": "preprocess", "exit_code": 0, "outcome": "ok"},
        {"name": "train_model", "exit_code": 0, "outcome": "ok"},
    ]


def _historical_stages():
    return [
        {
            "name": "fetch_raw_data",
            "started_at": "2026-09-25T02:01:03Z",
            "ended_at": "2026-09-25T02:03:12Z",
            "exit_code": 0,
            "outcome": "ok",
        },
        *_other_stages(),
    ]


def _binding(tmp: Path, baseline: dict, **overrides) -> list[dict]:
    path = tmp / "data/results/daily_refresh_fetch_receipt.json"
    blob = path.read_bytes()
    captured = datetime.fromisoformat(baseline["captured_at_utc"])
    stage = {
        "name": "fetch_raw_data",
        "run_id": RUN,
        "exit_code": 0,
        "outcome": "ok",
        "status": "COMPLETE",
        "artifact_path": "data/results/daily_refresh_fetch_receipt.json",
        "artifact_sha256": hashlib.sha256(blob).hexdigest(),
        "artifact_size_bytes": len(blob),
        "target_session": SESSION.isoformat(),
        "artifact_written_at_utc": (captured + timedelta(seconds=2)).isoformat(),
        "baseline_captured_at_utc": baseline["captured_at_utc"],
        "started_at": (captured + timedelta(seconds=1)).isoformat(),
        "ended_at": (captured + timedelta(seconds=3)).isoformat(),
        "artifact_after_baseline": True,
    }
    stage.update(overrides)
    return [stage, *_other_stages()]


def _prepare(tmp: Path):
    from services.daily_prediction_refresh_validate import capture_baseline

    started = "2026-09-24T22:00:00Z"
    baseline = capture_baseline(repo_root=tmp, run_id=RUN, run_started_at=started)
    receipt = tmp / "data/results/daily_refresh_fetch_receipt.json"
    if receipt.is_file():
        body = json.loads(receipt.read_text(encoding="utf-8"))
        body["generated_after_baseline"] = started
        receipt.write_text(json.dumps(body), encoding="utf-8")
    symbols = _symbols(tmp)
    for symbol in symbols:
        frame = pd.read_parquet(tmp / "data/processed" / f"{symbol}.parquet")
        frame["close"] = 2.0
        frame.to_parquet(tmp / "data/processed" / f"{symbol}.parquet", index=False)
        pred = pd.read_parquet(tmp / "data/predictions" / f"{symbol}_predictions.parquet")
        pred["predicted_close"] = 2.2
        pred.to_parquet(tmp / "data/predictions" / f"{symbol}_predictions.parquet", index=False)
    _bind_predictions(tmp, symbols)
    return started, baseline


def _validate(tmp: Path, stages=None):
    from services.daily_prediction_refresh_validate import validate_refresh

    started, baseline = _prepare(tmp)
    if stages is None and (tmp / "data/results/daily_refresh_fetch_receipt.json").is_file():
        stages = _binding(tmp, baseline)
    return validate_refresh(
        repo_root=tmp,
        target_session=SESSION,
        run_id=RUN,
        run_started_at=started,
        baseline=baseline,
        stage_receipts=stages if stages is not None else _historical_stages(),
        require_advancement=True,
    )


def test_unchanged_already_current_passes(tmp_path):
    symbols = ["PLTR", "SPCX", "^VIX"]
    root = _layout(tmp_path, symbols)
    _fetch(root, symbols)
    result = _validate(root)
    assert result["passed"] is True
    assert result["first_failure"] is None
    assert result["tradable_coverage"]["covered"] == 2
    assert result["tradable_coverage"]["denominator"] == 2
    assert result["raw"]["ok_count"] == 3


def test_missing_receipt_fails(tmp_path):
    root = _layout(tmp_path, ["PLTR"])
    result = _validate(root)
    assert result["passed"] is False
    assert "PLTR" in result["raw"]["not_advanced_symbols"]


def test_wrong_run_session_and_hash(tmp_path):
    from services.daily_prediction_refresh_validate import validate_refresh

    root = _layout(tmp_path, ["PLTR"])
    _fetch(root, ["PLTR"])
    started, baseline = _prepare(root)
    common = dict(
        repo_root=root,
        target_session=SESSION,
        run_id=RUN,
        run_started_at=started,
        baseline=baseline,
    )
    wrong_run = validate_refresh(**common, stage_receipts=_binding(root, baseline, run_id="other"))
    assert wrong_run["raw"]["already_current"]["rejected"]["receipt"] == "wrong_run_id"
    wrong_session = validate_refresh(
        **common, stage_receipts=_binding(root, baseline, target_session="2026-09-23")
    )
    assert wrong_session["raw"]["already_current"]["rejected"]["receipt"] == "wrong_session"
    forged = validate_refresh(
        **common, stage_receipts=_binding(root, baseline, artifact_sha256="ab" * 32)
    )
    assert forged["raw"]["already_current"]["rejected"]["receipt"] == "forged_fetch_artifact"
    body = json.loads(
        (root / "data/results/daily_refresh_fetch_receipt.json").read_text(encoding="utf-8")
    )
    body["decisions"][0]["raw_sha256"] = "cd" * 32
    (root / "data/results/daily_refresh_fetch_receipt.json").write_text(
        json.dumps(body), encoding="utf-8"
    )
    symbol_forged = validate_refresh(**common, stage_receipts=_binding(root, baseline))
    assert symbol_forged["raw"]["already_current"]["rejected"]["PLTR"] == "forged_raw_hash"


def test_structural_rejections(tmp_path):
    root = _layout(tmp_path, ["PLTR"])
    _fetch(root, ["PLTR"])
    _raw(root / "data/raw/PLTR.csv", "PLTR", session_rows=0)
    assert "PLTR" in _validate(root)["raw"]["stale_symbols"]
    _raw(root / "data/raw/PLTR.csv", "PLTR", session_rows=2)
    assert (
        _validate(root)["raw"]["already_current"]["rejected"]["PLTR"]
        == "missing_or_duplicate_session_row"
    )
    _raw(root / "data/raw/PLTR.csv", "PLTR", future=True)
    assert "PLTR" in _validate(root)["raw"]["stale_symbols"]
    _raw(root / "data/raw/PLTR.csv", "NOPE")
    assert _validate(root)["raw"]["already_current"]["rejected"]["PLTR"] == "wrong_symbol"
    pd.DataFrame(
        {
            "date": [SESSION.isoformat()],
            "open": ["x"],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [1],
        }
    ).to_csv(root / "data/raw/PLTR.csv", index=False)
    assert _validate(root)["raw"]["already_current"]["rejected"]["PLTR"] == "malformed_ohlcv"


def test_lagging_low_coverage_and_unpromoted_change(tmp_path):
    root = _layout(tmp_path, ["PLTR", "SPCX"])
    _fetch(root, ["PLTR", "SPCX"])
    body = json.loads(
        (root / "data/results/daily_refresh_fetch_receipt.json").read_text(encoding="utf-8")
    )
    body["decisions"][0]["reason"] = "provider_older_than_local"
    (root / "data/results/daily_refresh_fetch_receipt.json").write_text(
        json.dumps(body), encoding="utf-8"
    )
    lagging = _validate(root)
    assert lagging["raw"]["already_current"]["rejected"]["PLTR"] == "missing_provider_evidence"
    assert "PLTR" not in lagging["raw"].get("accepted_existing_local_evidence", [])
    _fetch(
        root,
        ["PLTR", "SPCX"],
        tradable_coverage={"covered": 1, "denominator": 2, "ratio": 0.5, "minimum": 0.9},
        passed=False,
    )
    low = _validate(root)
    assert low["raw"]["already_current"]["rejected"]["receipt"] == "coverage_below_minimum"
    _fetch(root, ["PLTR", "SPCX"])
    from services.daily_prediction_refresh_validate import validate_refresh

    started, baseline = _prepare(root)
    (root / "data/raw/PLTR.csv").write_bytes((root / "data/raw/PLTR.csv").read_bytes() + b"\n")
    changed = validate_refresh(
        repo_root=root,
        target_session=SESSION,
        run_id=RUN,
        run_started_at=started,
        baseline=baseline,
        stage_receipts=_binding(root, baseline),
    )
    assert changed["raw"]["already_current"]["rejected"]["PLTR"] == "changed_without_promotion"


def test_mixed_promotion_and_already_current(tmp_path):
    root = _layout(tmp_path, ["PLTR", "SPCX"])
    _fetch(root, ["PLTR", "SPCX"], promoted=["SPCX"])
    from services.daily_prediction_refresh_validate import validate_refresh

    started, baseline = _prepare(root)
    _raw(root / "data/raw/SPCX.csv", "SPCX")
    for symbol in ("PLTR", "SPCX"):
        frame = pd.read_parquet(root / "data/processed" / f"{symbol}.parquet")
        frame["close"] = 3.0
        frame.to_parquet(root / "data/processed" / f"{symbol}.parquet", index=False)
        pred = pd.read_parquet(root / "data/predictions" / f"{symbol}_predictions.parquet")
        pred["predicted_close"] = 3.3
        pred.to_parquet(root / "data/predictions" / f"{symbol}_predictions.parquet", index=False)
    _bind_predictions(root, ["PLTR", "SPCX"])
    result = validate_refresh(
        repo_root=root,
        target_session=SESSION,
        run_id=RUN,
        run_started_at=started,
        baseline=baseline,
        stage_receipts=_binding(root, baseline),
    )
    assert result["passed"] is True
    assert "PLTR" in result["raw"]["already_current"]["accepted"]


def _reason(tmp: Path, **overrides) -> str:
    from services.daily_prediction_refresh_validate import validate_refresh

    started, baseline = _prepare(tmp)
    result = validate_refresh(
        repo_root=tmp,
        target_session=SESSION,
        run_id=RUN,
        run_started_at=started,
        baseline=baseline,
        stage_receipts=_binding(tmp, baseline, **overrides),
    )
    return result["raw"]["already_current"]["rejected"]["receipt"]


def test_unbound_same_session_receipt_fails(tmp_path):
    root = _layout(tmp_path, ["PLTR"])
    _fetch(root, ["PLTR"])
    result = _validate(root, stages=_historical_stages())
    assert result["passed"] is False
    assert result["raw"]["already_current"]["rejected"]["receipt"] == "missing_binding_field"
    assert "PLTR" in result["raw"]["not_advanced_symbols"]


def test_binding_field_failures(tmp_path):
    root = _layout(tmp_path, ["PLTR"])
    _fetch(root, ["PLTR"])
    assert _reason(root, artifact_path="data/results/other.json") == "wrong_artifact_path"
    assert _reason(root, artifact_size_bytes=1) == "forged_fetch_artifact"
    assert (
        _reason(root, artifact_written_at_utc="2026-09-24T21:00:00Z")
        == "artifact_predates_baseline"
    )
    started, baseline = _prepare(root)
    stages = _binding(root, baseline)
    (root / "data/results/daily_refresh_fetch_receipt.json").write_bytes(
        (root / "data/results/daily_refresh_fetch_receipt.json").read_bytes() + b" "
    )
    from services.daily_prediction_refresh_validate import validate_refresh

    overwritten = validate_refresh(
        repo_root=root,
        target_session=SESSION,
        run_id=RUN,
        run_started_at=started,
        baseline=baseline,
        stage_receipts=stages,
    )
    assert overwritten["raw"]["already_current"]["rejected"]["receipt"] == "forged_fetch_artifact"
    assert _reason(root, exit_code=1, outcome="failed", status="FAILED") == "fetch_stage_failed"
    missing = _validate(root, stages=[])
    assert missing["raw"]["already_current"]["rejected"]["receipt"] == "missing_fetch_stage_binding"


def test_import_writes_nothing(tmp_path, monkeypatch):
    before = {path for path in tmp_path.rglob("*")}
    import socket

    import services.daily_prediction_refresh_validate  # noqa: F401

    assert {path for path in tmp_path.rglob("*")} == before
    calls = []

    def _blocked(*_args, **_kwargs):
        calls.append("network")
        raise AssertionError("network")

    monkeypatch.setattr(socket, "socket", _blocked)
    root = _layout(tmp_path, ["PLTR"])
    _fetch(root, ["PLTR"])
    result = _validate(root)
    assert result["broker_called"] is False
    assert result["network_called"] is False
    assert calls == []
