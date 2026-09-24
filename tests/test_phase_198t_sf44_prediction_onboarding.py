"""SF44 new-symbol prediction publication. No production writes."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

SESSION = date(2026, 9, 23)


def _days(n: int = 80, end: date = SESSION) -> list[date]:
    return [end - timedelta(days=n - 1 - offset) for offset in range(n)]


def _ohlcv(symbol: str, days: list[date] | None = None) -> pd.DataFrame:
    rows = []
    for offset, day in enumerate(days or _days()):
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


def _layout(tmp_path: Path, symbols: list[str], frames: dict[str, pd.DataFrame] | None = None):
    root = tmp_path
    for folder in (
        "data/config",
        "data/processed",
        "data/predictions",
        "data/results",
        "data/logs",
    ):
        (root / folder).mkdir(parents=True)
    (root / "data/config/tickers.txt").write_text("\n".join(symbols) + "\n", encoding="utf-8")
    (root / "data/results/fundamentals.csv").write_text(
        "ticker,pe_ratio\nAAA,1\n", encoding="utf-8"
    )
    (root / "data/results/stock_scores.csv").write_text(
        "ticker,total_score\nAAA,1\n", encoding="utf-8"
    )
    frames = frames or {symbol: _ohlcv(symbol) for symbol in symbols}
    for symbol, frame in frames.items():
        frame.to_parquet(root / "data/processed" / f"{symbol}.parquet", index=False)
    return root


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


def test_new_symbol_without_prior_prediction_or_fundamentals(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR", "AAA", "SPCX"])
    train = _bind(monkeypatch, root)
    assert train.main(["--min-rows", "20"]) == 0
    pltr = pd.read_parquet(root / "data/predictions/PLTR_predictions.parquet")
    assert pltr["ticker"].nunique() == 1
    assert int((pd.to_datetime(pltr["date"]).dt.date == SESSION).sum()) == 1
    assert (root / "data/predictions/SPCX_predictions.parquet").is_file()
    assert (root / "data/predictions/AAA_predictions.parquet").is_file()


def test_stale_processed_maximum_publishes_nothing(tmp_path, monkeypatch):
    frame = _ohlcv("PLTR", _days(end=date(2026, 9, 22)))
    root = _layout(tmp_path, ["PLTR"], {"PLTR": frame})
    prior = root / "data/predictions/PLTR_predictions.parquet"
    prior.write_bytes(b"stale")
    train = _bind(monkeypatch, root)
    assert train.main(["--min-rows", "20"]) == 4
    assert prior.read_bytes() == b"stale"


def test_duplicate_session_publishes_nothing(tmp_path, monkeypatch):
    frame = _ohlcv("PLTR")
    frame = pd.concat([frame, frame.iloc[[-1]]], ignore_index=True)
    root = _layout(tmp_path, ["PLTR"], {"PLTR": frame})
    train = _bind(monkeypatch, root)
    assert train.main(["--min-rows", "20"]) == 4
    assert not (root / "data/predictions/PLTR_predictions.parquet").exists()


def test_wrong_symbol_publishes_nothing(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR"], {"PLTR": _ohlcv("AAA")})
    train = _bind(monkeypatch, root)
    assert train.main(["--min-rows", "20"]) == 4
    assert not (root / "data/predictions/PLTR_predictions.parquet").exists()


def test_insufficient_history_publishes_nothing(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR"], {"PLTR": _ohlcv("PLTR", _days(8))})
    train = _bind(monkeypatch, root)
    assert train.main(["--min-rows", "20"]) == 4
    assert not (root / "data/predictions/PLTR_predictions.parquet").exists()


def test_malformed_processed_publishes_nothing(tmp_path, monkeypatch):
    frame = _ohlcv("PLTR").drop(columns=["close"])
    root = _layout(tmp_path, ["PLTR"], {"PLTR": frame})
    train = _bind(monkeypatch, root)
    assert train.main(["--min-rows", "20"]) == 4


def test_one_symbol_exception_publishes_nothing(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR", "AAA"])
    train = _bind(monkeypatch, root)
    real = train.pd.read_parquet

    def boom(path, *args, **kwargs):
        if str(path).endswith("AAA.parquet"):
            raise RuntimeError("boom")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(train.pd, "read_parquet", boom)
    assert train.main(["--min-rows", "20"]) == 4
    assert list((root / "data/predictions").glob("*.parquet")) == []


def test_atomic_replacement_does_not_retry(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR"])
    train = _bind(monkeypatch, root)
    calls = {"n": 0}
    real = os.replace

    def fail_once(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("replace failed")
        return real(src, dst)

    monkeypatch.setattr(train.os, "replace", fail_once)
    assert train.main(["--min-rows", "20"]) == 5
    assert calls["n"] == 1
    assert not (root / "data/predictions/PLTR_predictions.parquet").exists()
    assert not (root / "data/results/feature_importance.csv").exists()


def test_partial_staging_failure_publishes_nothing(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR", "AAA"])
    prior = root / "data/predictions/PLTR_predictions.parquet"
    prior.write_bytes(b"prior")
    train = _bind(monkeypatch, root)

    def boom(_target):
        raise OSError("stage failed")

    monkeypatch.setattr(train, "_unique_sibling_tmp", boom)
    assert train.main(["--min-rows", "20"]) == 5
    assert prior.read_bytes() == b"prior"
    assert not (root / "data/predictions/AAA_predictions.parquet").exists()


def test_two_runs_match(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR"])
    train = _bind(monkeypatch, root)
    assert train.main(["--min-rows", "20"]) == 0
    first = (root / "data/predictions/PLTR_predictions.parquet").read_bytes()
    assert train.main(["--min-rows", "20"]) == 0
    assert (root / "data/predictions/PLTR_predictions.parquet").read_bytes() == first


def test_index_is_required_and_rf_shadow_flag_unchanged(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR", "^DJI", "SPCX"])
    train = _bind(monkeypatch, root)
    assert train.main(["--min-rows", "20"]) == 0
    assert (root / "data/predictions/^DJI_predictions.parquet").is_file()
    assert (root / "data/predictions/SPCX_predictions.parquet").is_file()
    from services.generate_signals import RF_SHADOW_ONLY

    assert RF_SHADOW_ONLY is True


def test_import_creates_no_prediction(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR"])
    _bind(monkeypatch, root)
    assert list((root / "data/predictions").glob("*.parquet")) == []


def test_required_symbol_sets_include_spcx_and_pltr():
    from services.daily_prediction_refresh_validate import required_symbol_sets
    from services.lifecycle_artifact_guard import parse_universe_ticker_identities

    tickers = parse_universe_ticker_identities(Path("data/config/tickers.txt"))
    sets = required_symbol_sets(tickers)
    assert len(tickers) == 67
    assert sets["prediction_required_count"] == 67
    assert "SPCX" in sets["prediction_required"]
    assert "PLTR" in sets["prediction_required"]
    assert sets["expected_excluded_symbols"] == []
    for idx in ("^DJI", "^GSPC", "^IXIC", "^VIX"):
        assert idx in sets["prediction_required"]
    assert sets["index_reference_predictions_required"] is True


def test_missing_spcx_publishes_nothing(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR", "SPCX"], {"PLTR": _ohlcv("PLTR")})
    train = _bind(monkeypatch, root)
    assert train.main(["--min-rows", "20"]) == 4
    assert not (root / "data/predictions/PLTR_predictions.parquet").exists()
    assert not (root / "data/predictions/SPCX_predictions.parquet").exists()


def test_stale_spcx_publishes_nothing(tmp_path, monkeypatch):
    root = _layout(
        tmp_path,
        ["SPCX"],
        {"SPCX": _ohlcv("SPCX", _days(end=date(2026, 9, 22)))},
    )
    prior = root / "data/predictions/SPCX_predictions.parquet"
    prior.write_bytes(b"stale")
    train = _bind(monkeypatch, root)
    assert train.main(["--min-rows", "20"]) == 4
    assert prior.read_bytes() == b"stale"


def test_malformed_spcx_publishes_nothing(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["SPCX"], {"SPCX": _ohlcv("SPCX").drop(columns=["close"])})
    train = _bind(monkeypatch, root)
    assert train.main(["--min-rows", "20"]) == 4
    assert not (root / "data/predictions/SPCX_predictions.parquet").exists()


def test_wrong_symbol_spcx_publishes_nothing(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["SPCX"], {"SPCX": _ohlcv("AAA")})
    train = _bind(monkeypatch, root)
    assert train.main(["--min-rows", "20"]) == 4
    assert not (root / "data/predictions/SPCX_predictions.parquet").exists()


def test_missing_pltr_publishes_nothing(tmp_path, monkeypatch):
    root = _layout(tmp_path, ["PLTR", "AAA"], {"AAA": _ohlcv("AAA")})
    train = _bind(monkeypatch, root)
    assert train.main(["--min-rows", "20"]) == 4
    assert list((root / "data/predictions").glob("*.parquet")) == []
