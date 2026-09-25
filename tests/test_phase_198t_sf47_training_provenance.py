"""SF47 training-run provenance and zero-skip receipt. No production writes."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

RUN = "daily_prediction_refresh_20260924_sf47_replay"
SESSION = date(2026, 9, 24)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _days(n: int = 80, end: date = SESSION) -> list[date]:
    return [end - timedelta(days=n - 1 - offset) for offset in range(n)]


def _ohlcv(symbol: str) -> pd.DataFrame:
    rows = []
    for offset, day in enumerate(_days()):
        price = 10.0 + offset
        rows.append(
            {
                "date": day.isoformat(),
                "open": price,
                "high": price + 1.0,
                "low": price - 0.5,
                "close": price + 0.2,
                "volume": 100 + offset,
                "ticker": symbol,
            }
        )
    return pd.DataFrame(rows)


def _layout(tmp_path: Path, symbols: list[str]) -> Path:
    for folder in (
        "data/config",
        "data/processed",
        "data/predictions",
        "data/results",
        "data/logs",
    ):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / "data/config/tickers.txt").write_text("\n".join(symbols) + "\n", encoding="utf-8")
    (tmp_path / "data/results/fundamentals.csv").write_text(
        "ticker,pe_ratio\nAAA,1\n", encoding="utf-8"
    )
    (tmp_path / "data/results/stock_scores.csv").write_text(
        "ticker,total_score\nAAA,1\n", encoding="utf-8"
    )
    for symbol in symbols:
        _ohlcv(symbol).to_parquet(tmp_path / "data/processed" / f"{symbol}.parquet", index=False)
    return tmp_path


def _bind(monkeypatch, root: Path):
    import services.train_model as train

    monkeypatch.setattr(train, "PROJECT_ROOT", str(root))
    monkeypatch.setattr(train, "RESULTS_DIR", str(root / "data/results"))
    monkeypatch.setattr(train, "PROCESSED_DIR", str(root / "data/processed"))
    monkeypatch.setattr(train, "PREDICTIONS_DIR", str(root / "data/predictions"))
    monkeypatch.setattr(train, "LOGS_DIR", str(root / "data/logs"))
    monkeypatch.setattr(train, "FUNDAMENTALS_PATH", str(root / "data/results/fundamentals.csv"))
    monkeypatch.setattr(train, "SCORES_PATH", str(root / "data/results/stock_scores.csv"))
    monkeypatch.setattr(train, "NEWS_SENTIMENT_PATH", str(root / "data/results/news_sentiment.csv"))
    monkeypatch.setattr(
        train, "FEATURES_OUT_PATH", str(root / "data/results/feature_importance.csv")
    )
    monkeypatch.setattr(
        train, "SKIPPED_TICKERS_LOG", str(root / "data/results/skipped_tickers.csv")
    )
    monkeypatch.setattr(
        train, "MODEL_COMPARISON_PATH", str(root / "data/results/model_comparison.csv")
    )
    monkeypatch.setattr(train, "write_heartbeat", lambda **kwargs: None)
    return train


def _pipeline(train, run_id: str = RUN) -> int:
    return train.main(
        [
            "--min-rows",
            "20",
            "--pipeline",
            "--run-id",
            run_id,
            "--completed-session",
            SESSION.isoformat(),
        ]
    )


def _receipt(root: Path, **overrides) -> None:
    symbols = [
        line.strip()
        for line in (root / "data/config/tickers.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pred_hashes = {}
    proc_hashes = {}
    for symbol in symbols:
        pred = root / "data" / "predictions" / f"{symbol}_predictions.parquet"
        proc = root / "data" / "processed" / f"{symbol}.parquet"
        proc_hash = _sha(proc)
        proc_hashes[symbol] = proc_hash
        if pred.is_file():
            frame = pd.read_parquet(pred)
        else:
            frame = pd.DataFrame(
                {
                    "date": [SESSION.isoformat()],
                    "close": [1.0],
                    "predicted_close": [1.0],
                    "signal": ["HOLD"],
                }
            )
        frame["ticker"] = symbol
        frame["run_id"] = overrides.get("prediction_run_id", RUN)
        frame["completed_session"] = SESSION.isoformat()
        frame["processed_sha256"] = proc_hash
        frame["model_identity"] = "RandomForest+GradientBoosting+LinearRegression"
        pred.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(pred, index=False)
        pred_hashes[symbol] = _sha(pred)
    feature = root / "data/results/feature_importance.csv"
    model = root / "data/results/model_comparison.csv"
    feature.write_text("feature,importance\nx,1\n", encoding="utf-8")
    model.write_text("ticker,model\nPLTR,rf\n", encoding="utf-8")
    body = {
        "schema_version": 1,
        "run_id": RUN,
        "generated_at": "2026-09-24T23:00:00+00:00",
        "completed_session": SESSION.isoformat(),
        "governed_universe_sha256": _sha(root / "data/config/tickers.txt"),
        "required_count": len(symbols),
        "produced_count": len(symbols),
        "skipped_symbols": [],
        "processed_input_hashes": proc_hashes,
        "prediction_output_hashes": pred_hashes,
        "feature_importance_sha256": _sha(feature),
        "model_comparison_sha256": _sha(model),
        "publication_status": "COMPLETE",
    }
    body.update({k: v for k, v in overrides.items() if k != "prediction_run_id"})
    (root / "data/results/training_receipt.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fresh(root: Path, run_id: str = RUN):
    from services.daily_prediction_refresh_validate import capture_baseline

    started = "2026-09-24T22:00:00Z"
    baseline = capture_baseline(repo_root=root, run_id=run_id, run_started_at=started)
    (root / "data/raw").mkdir(parents=True, exist_ok=True)
    for symbol in [
        line.strip()
        for line in (root / "data/config/tickers.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]:
        frame = pd.read_parquet(root / "data/processed" / f"{symbol}.parquet")
        frame.to_parquet(root / "data/processed" / f"{symbol}.parquet", index=False)
        pd.DataFrame({"date": [SESSION.isoformat()], "close": [1.0]}).to_csv(
            root / "data/raw" / f"{symbol}.csv", index=False
        )
    return started, baseline


def _validate(root: Path, run_id: str = RUN, *, baseline=None, started: str | None = None):
    from services.daily_prediction_refresh_validate import validate_refresh

    if baseline is None or started is None:
        started, baseline = _fresh(root, run_id)
    return validate_refresh(
        repo_root=root,
        target_session=SESSION,
        run_id=run_id,
        run_started_at=started,
        baseline=baseline,
        stage_receipts=[
            {"name": "fetch_raw_data", "exit_code": 0, "outcome": "ok"},
            {"name": "stale_data_gate", "exit_code": 0, "outcome": "ok"},
            {"name": "preprocess", "exit_code": 0, "outcome": "ok"},
            {"name": "train_model", "exit_code": 0, "outcome": "ok"},
        ],
        require_advancement=True,
    )


def test_pipeline_rejects_generic_train(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR"])
    train = _bind(monkeypatch, root)
    assert _pipeline(train, "train") == 4
    assert list((root / "data/predictions").glob("*")) == []


def test_pipeline_rejects_missing_run_id(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR"])
    train = _bind(monkeypatch, root)
    assert (
        train.main(["--min-rows", "20", "--pipeline", "--completed-session", SESSION.isoformat()])
        == 4
    )


def test_pipeline_trains_with_exact_run_id_and_zero_skip_receipt(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR", "SPCX"])
    train = _bind(monkeypatch, root)
    assert _pipeline(train) == 0
    for symbol in ("PLTR", "SPCX"):
        frame = pd.read_parquet(root / "data/predictions" / f"{symbol}_predictions.parquet")
        assert set(frame["run_id"].astype(str)) == {RUN}
    receipt = json.loads((root / "data/results/training_receipt.json").read_text(encoding="utf-8"))
    assert receipt["publication_status"] == "COMPLETE"
    assert receipt["run_id"] == RUN
    assert receipt["skipped_symbols"] == []
    assert receipt["required_count"] == 2
    assert receipt["produced_count"] == 2
    skips = pd.read_csv(root / "data/results/skipped_tickers.csv")
    assert list(skips.columns) == ["ticker", "reason"]
    assert skips.empty


def test_two_pipeline_runs_match_except_generated_at(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR"])
    train = _bind(monkeypatch, root)
    assert _pipeline(train) == 0
    first_pred = (root / "data/predictions/PLTR_predictions.parquet").read_bytes()
    first = json.loads((root / "data/results/training_receipt.json").read_text(encoding="utf-8"))
    assert _pipeline(train) == 0
    second_pred = (root / "data/predictions/PLTR_predictions.parquet").read_bytes()
    second = json.loads((root / "data/results/training_receipt.json").read_text(encoding="utf-8"))
    assert first_pred == second_pred
    assert {k: v for k, v in first.items() if k != "generated_at"} == {
        k: v for k, v in second.items() if k != "generated_at"
    }


def test_validator_accepts_bound_receipt(tmp_path):
    root = _layout(tmp_path, ["PLTR", "SPCX"])
    started, baseline = _fresh(root)
    _receipt(root)
    result = _validate(root, baseline=baseline, started=started)
    assert result["passed"] is True
    assert result["first_failure"] is None


def test_validator_rejects_generic_train_receipt(tmp_path):
    root = _layout(tmp_path, ["PLTR"])
    started, baseline = _fresh(root, "train")
    _receipt(root, run_id="train", prediction_run_id="train")
    result = _validate(root, "train", baseline=baseline, started=started)
    assert result["passed"] is False
    assert result["first_failure"] == "train_diagnostics_not_advanced"


def test_validator_rejects_missing_receipt(tmp_path):
    root = _layout(tmp_path, ["PLTR"])
    started, baseline = _fresh(root)
    _receipt(root)
    (root / "data/results/training_receipt.json").unlink()
    result = _validate(root, baseline=baseline, started=started)
    assert "missing_receipt" in result["train_diagnostics_advancement"]["errors"]


def test_validator_rejects_wrong_run_id(tmp_path):
    root = _layout(tmp_path, ["PLTR"])
    started, baseline = _fresh(root)
    _receipt(root, run_id="other")
    result = _validate(root, baseline=baseline, started=started)
    assert "run_id" in result["train_diagnostics_advancement"]["errors"]


def test_validator_rejects_stale_receipt(tmp_path):
    root = _layout(tmp_path, ["PLTR"])
    started, baseline = _fresh(root)
    _receipt(root)
    result = _validate(root, baseline=baseline, started=started)
    assert result["passed"] is True
    stale_started, stale_baseline = _fresh(root)
    again = _validate(root, baseline=stale_baseline, started=stale_started)
    assert "stale_receipt" in again["train_diagnostics_advancement"]["errors"]


def test_validator_rejects_forged_hash(tmp_path):
    root = _layout(tmp_path, ["PLTR"])
    started, baseline = _fresh(root)
    _receipt(root, feature_importance_sha256="0" * 64)
    result = _validate(root, baseline=baseline, started=started)
    assert "feature_importance" in result["train_diagnostics_advancement"]["errors"]


def test_validator_rejects_prediction_run_mismatch(tmp_path):
    root = _layout(tmp_path, ["PLTR"])
    started, baseline = _fresh(root)
    _receipt(root, prediction_run_id="other-run")
    result = _validate(root, baseline=baseline, started=started)
    assert "prediction_run_id" in result["train_diagnostics_advancement"]["errors"]


def test_validator_rejects_processed_hash_mismatch(tmp_path):
    root = _layout(tmp_path, ["PLTR"])
    started, baseline = _fresh(root)
    _receipt(root)
    body = json.loads((root / "data/results/training_receipt.json").read_text(encoding="utf-8"))
    body["processed_input_hashes"]["PLTR"] = "ab" * 32
    (root / "data/results/training_receipt.json").write_text(json.dumps(body), encoding="utf-8")
    result = _validate(root, baseline=baseline, started=started)
    assert "processed_hash" in result["train_diagnostics_advancement"]["errors"]


def test_missing_pltr_and_spcx_fail_before_receipt(tmp_path):
    root = _layout(tmp_path, ["PLTR", "SPCX"])
    started, baseline = _fresh(root)
    _receipt(root)
    (root / "data/predictions/PLTR_predictions.parquet").unlink()
    missing = _validate(root, baseline=baseline, started=started)
    assert missing["first_failure"] == "predictions_missing_required"
    started2, baseline2 = _fresh(root)
    _receipt(root)
    (root / "data/predictions/SPCX_predictions.parquet").unlink()
    missing_spcx = _validate(root, baseline=baseline2, started=started2)
    assert "SPCX" in missing_spcx["predictions"]["missing_required_symbols"]


def test_feature_importance_failure_writes_no_receipt(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR"])
    train = _bind(monkeypatch, root)
    real = train.atomic_publish_csv

    def fail_feature(df, path):
        if str(path).endswith("feature_importance.csv"):
            raise OSError("feature importance blocked")
        return real(df, path)

    monkeypatch.setattr(train, "atomic_publish_csv", fail_feature)
    assert _pipeline(train) == 6
    assert not (root / "data/results/training_receipt.json").exists()


def test_model_comparison_failure_writes_no_receipt(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR"])
    train = _bind(monkeypatch, root)
    real = train.atomic_publish_csv

    def fail_model(df, path):
        if str(path).endswith("model_comparison.csv"):
            raise OSError("model comparison blocked")
        return real(df, path)

    monkeypatch.setattr(train, "atomic_publish_csv", fail_model)
    assert _pipeline(train) == 6
    assert not (root / "data/results/training_receipt.json").exists()


def test_one_symbol_failure_publishes_nothing(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR", "SPCX"])
    (root / "data/processed/SPCX.parquet").write_text("not parquet", encoding="utf-8")
    train = _bind(monkeypatch, root)
    assert _pipeline(train) == 4
    assert list((root / "data/predictions").glob("*.parquet")) == []
    assert not (root / "data/results/training_receipt.json").exists()


def test_partial_staging_failure_publishes_nothing(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR"])
    train = _bind(monkeypatch, root)
    real_open = open

    def fail_stage(path, *args, **kwargs):
        if str(path).endswith(".parquet.tmp") or ".tmp" in str(path):
            raise OSError("staging blocked")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(train.os, "replace", train.os.replace)
    monkeypatch.setitem(__import__("builtins").__dict__, "open", fail_stage)
    rc = _pipeline(train)
    assert rc == 5
    assert list((root / "data/predictions").glob("*.parquet")) == []


def test_atomic_replacement_failure_is_not_retried(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR"])
    train = _bind(monkeypatch, root)
    calls = {"n": 0}
    real = train.os.replace

    def once(src, dst):
        calls["n"] += 1
        raise OSError("replace blocked")

    monkeypatch.setattr(train.os, "replace", once)
    assert _pipeline(train) == 5
    assert calls["n"] == 1
    assert not (root / "data/results/training_receipt.json").exists()
    del real


def test_import_writes_nothing(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR"])
    before = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    import services.train_model  # noqa: F401

    after = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert before == after
    monkeypatch.undo()
