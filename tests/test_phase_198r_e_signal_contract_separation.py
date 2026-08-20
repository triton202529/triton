"""Phase 198R-E — separate training and operational signal contracts.

Deterministic. Injected disposable paths only. No broker, network, or
production artifact mutation. Must not run inside the production repository.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import io
import os
import socket
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
import sklearn.ensemble  # noqa: F401  imported before autouse socket patch

_PROD_ROOT = Path(r"C:\Users\akimw\triton")

_PROD_ROOT = Path(r"C:\Users\akimw\triton")
_REPO_ROOT = Path(__file__).resolve().parents[1]
if _REPO_ROOT.resolve() == _PROD_ROOT.resolve():
    raise RuntimeError(
        "Phase 198R-E tests must not run in the production repository. "
        "Use a disposable git-archive tree."
    )

_PROD_PREFIXES = (
    str(_PROD_ROOT.resolve()),
    str(_PROD_ROOT.resolve()).replace("\\", "/"),
    r"C:\Users\akimw\triton",
    "C:/Users/akimw/triton",
)


class ProductionWriteBlocked(RuntimeError):
    """Raised when a test attempts to write into the production tree."""


def _is_production_path(path: Any) -> bool:
    try:
        text = os.path.abspath(os.fspath(path))
    except TypeError:
        return False
    norm = text.replace("/", "\\").lower()
    prod = str(_PROD_ROOT.resolve()).replace("/", "\\").lower()
    return norm == prod or norm.startswith(prod + "\\")


def _guard_write(path: Any) -> None:
    if _is_production_path(path):
        raise ProductionWriteBlocked(f"production write blocked: {path}")


@pytest.fixture(autouse=True)
def _phase_198r_e_safety(monkeypatch):
    real_open = builtins.open
    real_replace = os.replace
    real_remove = os.remove
    real_rename = os.rename
    real_popen = __import__("subprocess").Popen

    def guarded_open(file, mode="r", *args, **kwargs):
        mode_s = str(mode)
        if any(flag in mode_s for flag in ("w", "a", "x", "+")):
            _guard_write(file)
        return real_open(file, mode, *args, **kwargs)

    def guarded_replace(src, dst):
        _guard_write(dst)
        return real_replace(src, dst)

    def guarded_remove(path):
        if _is_production_path(path):
            raise ProductionWriteBlocked(f"production delete blocked: {path}")
        return real_remove(path)

    def guarded_rename(src, dst):
        _guard_write(dst)
        return real_rename(src, dst)

    class GuardedPopen(real_popen):  # type: ignore[valid-type]
        def __init__(self, args, *a, **k):
            blob = " ".join(str(x) for x in (args if isinstance(args, (list, tuple)) else [args]))
            lowered = blob.lower()
            if any(
                token in lowered
                for token in (
                    "run_continuous_loop",
                    "run_watchdog",
                    "streamlit",
                    "alpaca",
                    "start-process",
                )
            ):
                raise RuntimeError(f"runtime subprocess blocked: {blob}")
            super().__init__(args, *a, **k)

    class BlockedSocket:
        def __init__(self, *a, **k):
            raise RuntimeError("network/broker calls are blocked in Phase 198R-E tests")

        def __call__(self, *a, **k):
            raise RuntimeError("network/broker calls are blocked in Phase 198R-E tests")

    def blocked_connect(*a, **k):
        raise RuntimeError("network/broker calls are blocked in Phase 198R-E tests")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "replace", guarded_replace)
    monkeypatch.setattr(os, "remove", guarded_remove)
    monkeypatch.setattr(os, "rename", guarded_rename)
    monkeypatch.setattr("subprocess.Popen", GuardedPopen)
    monkeypatch.setattr(socket, "socket", BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", blocked_connect)
    for key in (
        "TRITON_ENABLE_LIVE_EXECUTION",
        "TRITON_LIVE_EXECUTION",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
    ):
        monkeypatch.setenv(key, "0") if "LIVE" in key else monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    yield


def _bind_train_paths(monkeypatch, root: Path):
    import services.train_model as tm

    results = root / "data" / "results"
    processed = root / "data" / "processed"
    preds = root / "data" / "predictions"
    logs = root / "data" / "logs"
    for path in (results, processed, preds, logs):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tm, "RESULTS_DIR", str(results))
    monkeypatch.setattr(tm, "PROCESSED_DIR", str(processed))
    monkeypatch.setattr(tm, "PREDICTIONS_DIR", str(preds))
    monkeypatch.setattr(tm, "LOGS_DIR", str(logs))
    monkeypatch.setattr(tm, "FUNDAMENTALS_PATH", str(results / "fundamentals.csv"))
    monkeypatch.setattr(tm, "SCORES_PATH", str(results / "stock_scores.csv"))
    monkeypatch.setattr(tm, "NEWS_SENTIMENT_PATH", str(results / "news_sentiment.csv"))
    monkeypatch.setattr(tm, "FEATURES_OUT_PATH", str(results / "feature_importance.csv"))
    monkeypatch.setattr(tm, "SKIPPED_TICKERS_LOG", str(results / "skipped_tickers.csv"))
    monkeypatch.setattr(tm, "MODEL_COMPARISON_PATH", str(results / "model_comparison.csv"))
    monkeypatch.setattr(tm, "write_heartbeat", lambda **kwargs: None)
    monkeypatch.chdir(root)
    return tm, results, processed, preds


def _bind_generate_paths(monkeypatch, root: Path):
    import services.generate_signals as gs

    results = root / "data" / "results"
    preds = root / "data" / "predictions"
    results.mkdir(parents=True, exist_ok=True)
    preds.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(gs, "RESULTS_DIR", str(results))
    monkeypatch.setattr(gs, "PREDICTIONS_DIR", str(preds))
    monkeypatch.setattr(gs, "OUT_WITH_RATIONALE", str(results / "signals_with_rationale.csv"))
    monkeypatch.setattr(gs, "OUT_SIGNALS", str(results / "signals.csv"))
    monkeypatch.setattr(gs, "ASOF_DIAG_PATH", str(results / "signals_asof_diagnostics.json"))
    monkeypatch.chdir(root)
    return gs, results, preds


def _ohlcv_frame(n: int = 60, last: date = date(2026, 8, 18)) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(last), periods=n)
    close = np.linspace(100.0, 110.0, n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.4,
            "high": close + 0.8,
            "low": close - 0.9,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def _seed_training_inputs(root: Path) -> None:
    results = root / "data" / "results"
    processed = root / "data" / "processed"
    results.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "pe_ratio": 18.0,
                "eps": 4.0,
                "market_cap": 1e9,
                "pb_ratio": 3.0,
                "dividend_yield": 0.01,
            },
            {
                "ticker": "SKIPME",
                "pe_ratio": 12.0,
                "eps": 2.0,
                "market_cap": 1e8,
                "pb_ratio": 2.0,
                "dividend_yield": 0.0,
            },
        ]
    ).to_csv(results / "fundamentals.csv", index=False)
    pd.DataFrame(
        [{"ticker": "AAA", "total_score": 72.0}, {"ticker": "SKIPME", "total_score": 40.0}]
    ).to_csv(results / "stock_scores.csv", index=False)
    _ohlcv_frame().to_parquet(processed / "AAA.parquet", index=False)


def _pred_frame(ticker: str, last: date, rows: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(last), periods=rows)
    close = np.linspace(50.0, 55.0, rows)
    pred = close * 1.01
    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "predicted_close": pred,
            "signal": ["BUY"] * rows,
            "ticker": [ticker] * rows,
        }
    )


def _run_generate(
    monkeypatch, root: Path, *, last: date = date(2026, 8, 18), tickers=("AAA", "BBB")
):
    gs, results, preds = _bind_generate_paths(monkeypatch, root)
    for ticker in tickers:
        _pred_frame(ticker, last).to_parquet(preds / f"{ticker}_predictions.parquet", index=False)
    monkeypatch.setattr(gs, "_nyse_last_completed_session_date", lambda now_utc=None: last)
    monkeypatch.setattr(gs, "utc_now_z", lambda: "2026-08-19T12:00:00Z")
    monkeypatch.setattr(
        "services.signal_pressure_diagnostics.refresh_signal_pressure_diagnostics",
        lambda: None,
    )
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    gs.main()
    return gs, results, buf.getvalue()


def test_01_importing_train_model_performs_no_write(tmp_path, monkeypatch):
    sandbox = tmp_path / "import_safe"
    sandbox.mkdir()
    monkeypatch.chdir(sandbox)
    monkeypatch.setattr(sys, "argv", ["pytest", "--unexpected", "1"])
    writes: List[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        mode_s = str(mode)
        if any(flag in mode_s for flag in ("w", "a", "x")):
            writes.append(os.fspath(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)
    import services.train_model as tm

    importlib.reload(tm)
    artifact_writes = [
        w
        for w in writes
        if "signals.csv" in w.replace("\\", "/")
        or "signals_with_rationale.csv" in w.replace("\\", "/")
        or "model_comparison.csv" in w.replace("\\", "/")
        or "feature_importance.csv" in w.replace("\\", "/")
        or "_predictions.parquet" in w.replace("\\", "/")
    ]
    assert artifact_writes == []
    assert not list(sandbox.rglob("*.csv"))
    assert not list(sandbox.rglob("*.parquet"))


def test_02_importing_train_model_does_not_parse_process_cli(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog", "--min-rows", "not-an-int"])
    import services.train_model as tm

    importlib.reload(tm)
    args = tm.parse_args([])
    assert args.min_rows == 30


def test_03_importing_train_model_does_not_start_training(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    started = {"fit": False}

    def boom(*a, **k):
        started["fit"] = True
        raise AssertionError("training started on import")

    import sklearn.ensemble as ens

    monkeypatch.setattr(ens.RandomForestRegressor, "fit", boom)
    import services.train_model as tm

    importlib.reload(tm)
    assert started["fit"] is False
    assert tm.main.__name__ == "main"


def test_04_train_model_cli_retains_arguments_and_defaults():
    import services.train_model as tm

    args = tm.parse_args([])
    assert args.min_rows == 30
    args2 = tm.parse_args(["--min-rows", "12"])
    assert args2.min_rows == 12
    src = (_REPO_ROOT / "services" / "train_model.py").read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in src
    assert "raise SystemExit(main())" in src


def test_05_train_model_writes_per_ticker_prediction_parquet(tmp_path, monkeypatch):
    tm, results, processed, preds = _bind_train_paths(monkeypatch, tmp_path)
    _seed_training_inputs(tmp_path)
    assert tm.main(["--min-rows", "8"]) == 0
    out = preds / "AAA_predictions.parquet"
    assert out.is_file()
    df = pd.read_parquet(out)
    assert set(["date", "close", "predicted_close", "signal", "ticker"]).issubset(df.columns)
    assert (df["ticker"] == "AAA").all()


def test_06_train_model_writes_model_comparison(tmp_path, monkeypatch):
    tm, results, processed, preds = _bind_train_paths(monkeypatch, tmp_path)
    _seed_training_inputs(tmp_path)
    tm.main(["--min-rows", "8"])
    path = results / "model_comparison.csv"
    assert path.is_file()
    df = pd.read_csv(path)
    assert "ticker" in df.columns and "model" in df.columns
    assert (df["ticker"] == "AAA").any()


def test_07_train_model_writes_feature_importance(tmp_path, monkeypatch):
    tm, results, processed, preds = _bind_train_paths(monkeypatch, tmp_path)
    _seed_training_inputs(tmp_path)
    tm.main(["--min-rows", "8"])
    path = results / "feature_importance.csv"
    assert path.is_file()
    df = pd.read_csv(path)
    assert {"ticker", "model", "feature", "importance"}.issubset(df.columns)


def test_08_train_model_writes_skipped_tickers(tmp_path, monkeypatch):
    tm, results, processed, preds = _bind_train_paths(monkeypatch, tmp_path)
    _seed_training_inputs(tmp_path)
    tm.main(["--min-rows", "8"])
    path = results / "skipped_tickers.csv"
    assert path.is_file()
    df = pd.read_csv(path)
    assert "SKIPME" in set(df["ticker"].astype(str).str.upper())


def test_09_train_model_never_writes_signals_with_rationale(tmp_path, monkeypatch):
    tm, results, processed, preds = _bind_train_paths(monkeypatch, tmp_path)
    _seed_training_inputs(tmp_path)
    tm.main(["--min-rows", "8"])
    assert not (results / "signals_with_rationale.csv").exists()
    src = (_REPO_ROOT / "services" / "train_model.py").read_text(encoding="utf-8")
    assert "signals_with_rationale.csv" not in src


def test_10_train_model_never_writes_results_signals_csv(tmp_path, monkeypatch):
    tm, results, processed, preds = _bind_train_paths(monkeypatch, tmp_path)
    _seed_training_inputs(tmp_path)
    tm.main(["--min-rows", "8"])
    assert not (results / "signals.csv").exists()
    src = (_REPO_ROOT / "services" / "train_model.py").read_text(encoding="utf-8")
    assert "LEGACY_SIGNALS_PATH" not in src
    assert "data/results/signals.csv" not in src


def test_11_daily_refresh_validation_without_operational_signal_csvs(tmp_path):
    from tests.test_daily_prediction_refresh_owner import (
        _advance_all,
        _receipts_ok,
        _seed_repo,
    )
    from services.daily_prediction_refresh_validate import capture_baseline, validate_refresh

    src = (_REPO_ROOT / "services" / "daily_prediction_refresh_validate.py").read_text(
        encoding="utf-8"
    )
    assert "signals_with_rationale.csv" not in src
    assert "signals.csv" not in src
    target = date(2026, 8, 7)
    root = _seed_repo(tmp_path, target=target, include_index_preds=True)
    assert not (root / "data" / "results" / "signals_with_rationale.csv").exists()
    assert not (root / "data" / "results" / "signals.csv").exists()
    started = "2026-08-07T12:00:00Z"
    baseline = capture_baseline(repo_root=root, run_id="r198e", run_started_at=started)
    _advance_all(root, target)
    result = validate_refresh(
        repo_root=root,
        target_session=target,
        run_id="r198e",
        run_started_at=started,
        baseline=baseline,
        stage_receipts=_receipts_ok(),
        require_advancement=True,
    )
    assert result["passed"] is True
    assert not (root / "data" / "results" / "signals_with_rationale.csv").exists()
    assert not (root / "data" / "results" / "signals.csv").exists()


def test_12_generate_signals_is_operational_writer_for_rationale(tmp_path, monkeypatch):
    gs, results, _out = _run_generate(monkeypatch, tmp_path)
    path = results / "signals_with_rationale.csv"
    assert path.is_file()
    src = (_REPO_ROOT / "services" / "generate_signals.py").read_text(encoding="utf-8")
    assert "signals_with_rationale.csv" in src
    assert "publish_operational_signal_csvs" in src
    daily = (_REPO_ROOT / "run_daily_refresh.ps1").read_text(encoding="utf-8")
    assert "services.generate_signals" not in daily
    assert "services.train_model" in daily


def test_13_generate_signals_is_operational_writer_for_signals_csv(tmp_path, monkeypatch):
    gs, results, _out = _run_generate(monkeypatch, tmp_path)
    path = results / "signals.csv"
    assert path.is_file()
    df = pd.read_csv(path)
    assert "ticker" in df.columns
    market = (_REPO_ROOT / "scripts" / "run_market_open.ps1").read_text(encoding="utf-8")
    assert '-m","services.generate_signals"' in market or "services.generate_signals" in market
    assert "services.train_model" not in market


def test_14_generate_signals_one_current_row_per_ticker(tmp_path, monkeypatch):
    gs, results, _out = _run_generate(monkeypatch, tmp_path, tickers=("AAA", "BBB"))
    df = pd.read_csv(results / "signals_with_rationale.csv")
    assert df["ticker"].nunique() == len(df)
    assert set(df["ticker"].astype(str)) == {"AAA", "BBB"}


def test_15_generate_signals_preserves_generated_at_utc(tmp_path, monkeypatch):
    gs, results, _out = _run_generate(monkeypatch, tmp_path)
    df = pd.read_csv(results / "signals_with_rationale.csv")
    assert "generated_at_utc" in df.columns
    assert (df["generated_at_utc"] == "2026-08-19T12:00:00Z").all()


def test_16_generate_signals_preserves_as_of_date(tmp_path, monkeypatch):
    last = date(2026, 8, 18)
    gs, results, _out = _run_generate(monkeypatch, tmp_path, last=last)
    df = pd.read_csv(results / "signals_with_rationale.csv")
    assert "as_of_date" in df.columns
    as_of = pd.to_datetime(df["as_of_date"]).dt.date
    assert set(as_of) == {last}


def test_17_generate_signals_preserves_session_as_of_date(tmp_path, monkeypatch):
    last = date(2026, 8, 18)
    gs, results, _out = _run_generate(monkeypatch, tmp_path, last=last)
    df = pd.read_csv(results / "signals_with_rationale.csv")
    assert "session_as_of_date" in df.columns
    sess = pd.to_datetime(df["session_as_of_date"]).dt.date
    assert set(sess) == {last}


def test_18_generate_signals_preserves_freshness_refusal(tmp_path, monkeypatch):
    gs, results, preds = _bind_generate_paths(monkeypatch, tmp_path)
    stale = date(2026, 1, 2)
    session = date(2026, 8, 18)
    _pred_frame("AAA", stale).to_parquet(preds / "AAA_predictions.parquet", index=False)
    monkeypatch.setattr(gs, "_nyse_last_completed_session_date", lambda now_utc=None: session)
    monkeypatch.setattr(gs, "STRICT_ASOF", True)
    monkeypatch.setattr(gs, "MAX_LAG_SESSIONS", 1)
    monkeypatch.setattr(gs, "ASOF_COVERAGE", 0.9)
    monkeypatch.setattr(gs, "_next_session_date", lambda d: d + timedelta(days=1))
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        gs.main()
    _effective, diag = gs.compute_asof_policy({"AAA": stale}, session)
    assert diag.get("fail") is True


def test_19_operational_canonical_write_is_atomic(tmp_path, monkeypatch):
    gs, results, preds = _bind_generate_paths(monkeypatch, tmp_path)
    seen: List[str] = []
    real_replace = os.replace

    def tracking_replace(src, dst):
        seen.append(os.fspath(dst))
        assert Path(src).parent == Path(dst).parent
        return real_replace(src, dst)

    monkeypatch.setattr(gs.os, "replace", tracking_replace)
    _run_generate(monkeypatch, tmp_path)
    targets = [Path(p).name for p in seen]
    assert "signals_with_rationale.csv" in targets
    leftover = list(results.glob(".*.tmp"))
    assert leftover == []


def test_20_operational_legacy_write_is_atomic(tmp_path, monkeypatch):
    gs, results, preds = _bind_generate_paths(monkeypatch, tmp_path)
    seen: List[str] = []
    real_replace = os.replace

    def tracking_replace(src, dst):
        seen.append(Path(dst).name)
        assert str(src).endswith(".tmp") or ".tmp." in Path(src).name
        return real_replace(src, dst)

    monkeypatch.setattr(gs.os, "replace", tracking_replace)
    _run_generate(monkeypatch, tmp_path)
    assert "signals.csv" in seen


def test_21_training_parquet_replacement_is_atomic(tmp_path, monkeypatch):
    tm, results, processed, preds = _bind_train_paths(monkeypatch, tmp_path)
    seen: List[str] = []
    real_replace = os.replace

    def tracking_replace(src, dst):
        seen.append(os.fspath(dst))
        assert Path(src).parent == Path(dst).parent
        return real_replace(src, dst)

    monkeypatch.setattr(tm.os, "replace", tracking_replace)
    _seed_training_inputs(tmp_path)
    tm.main(["--min-rows", "8"])
    assert any(p.endswith("AAA_predictions.parquet") for p in seen)


def test_22_training_csv_replacements_are_atomic(tmp_path, monkeypatch):
    tm, results, processed, preds = _bind_train_paths(monkeypatch, tmp_path)
    seen: List[str] = []
    real_replace = os.replace

    def tracking_replace(src, dst):
        seen.append(Path(dst).name)
        return real_replace(src, dst)

    monkeypatch.setattr(tm.os, "replace", tracking_replace)
    _seed_training_inputs(tmp_path)
    tm.main(["--min-rows", "8"])
    names = set(seen)
    assert "model_comparison.csv" in names
    assert "feature_importance.csv" in names
    assert "skipped_tickers.csv" in names


def test_23_atomic_failure_preserves_previous_complete_artifact(tmp_path, monkeypatch):
    import services.train_model as tm

    target = tmp_path / "model_comparison.csv"
    target.write_text("old,complete\n1,2\n", encoding="utf-8")
    previous = target.read_bytes()

    def boom(src, dst):
        _guard_write(dst)
        raise OSError("injected replace failure")

    monkeypatch.setattr(tm.os, "replace", boom)
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(OSError, match="injected replace failure"):
        tm.atomic_publish_csv(df, str(target))
    assert target.read_bytes() == previous
    assert list(tmp_path.glob(".*.tmp")) == []


def test_24_atomic_failure_does_not_report_success(tmp_path, monkeypatch):
    gs, results, preds = _bind_generate_paths(monkeypatch, tmp_path)
    last = date(2026, 8, 18)
    _pred_frame("AAA", last).to_parquet(preds / "AAA_predictions.parquet", index=False)
    monkeypatch.setattr(gs, "_nyse_last_completed_session_date", lambda now_utc=None: last)
    monkeypatch.setattr(gs, "utc_now_z", lambda: "2026-08-19T12:00:00Z")
    monkeypatch.setattr(
        "services.signal_pressure_diagnostics.refresh_signal_pressure_diagnostics",
        lambda: None,
    )
    results.joinpath("signals.csv").write_text("previous,complete\n", encoding="utf-8")
    previous = (results / "signals.csv").read_bytes()
    real_publish = gs.atomic_publish_csv

    def flaky(df, target):
        if Path(target).name == "signals.csv":
            raise OSError("legacy publish failed")
        return real_publish(df, target)

    monkeypatch.setattr(gs, "atomic_publish_csv", flaky)
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    with pytest.raises(RuntimeError, match="already published"):
        gs.main()
    out = buf.getvalue()
    assert "✅ signals.csv" not in out
    assert "Signals generated successfully" not in out
    assert (results / "signals.csv").read_bytes() == previous


def test_25_partial_temporary_file_is_not_canonical(tmp_path, monkeypatch):
    import services.generate_signals as gs

    target = tmp_path / "signals_with_rationale.csv"
    df = pd.DataFrame(
        {
            "date": ["2026-08-18"],
            "as_of_date": ["2026-08-18"],
            "session_as_of_date": ["2026-08-18"],
            "ticker": ["AAA"],
            "close": [1.0],
            "predicted_close": [1.1],
            "delta_pct": [0.1],
            "signal": ["BUY"],
            "confidence": [0.6],
            "score": [0.7],
            "rationale": ["ok"],
            "generated_at_utc": ["2026-08-19T12:00:00Z"],
        }
    )
    gs.atomic_publish_csv(df, str(target))
    assert target.is_file()
    assert not target.name.startswith(".")
    assert list(tmp_path.glob(".*.tmp")) == []
    assert target.name == "signals_with_rationale.csv"


def test_26_import_run_uses_only_injected_disposable_paths(tmp_path, monkeypatch):
    tm, results, processed, preds = _bind_train_paths(monkeypatch, tmp_path)
    _seed_training_inputs(tmp_path)
    tm.main(["--min-rows", "8"])
    _run_generate(monkeypatch, tmp_path)
    for path in list(results.glob("*")) + list(preds.glob("*")):
        resolved = path.resolve()
        assert not _is_production_path(resolved)
        assert str(tmp_path.resolve()) in str(resolved)


def test_27_production_root_write_attempts_are_blocked(tmp_path):
    prod_target = _PROD_ROOT / "data" / "results" / "signals_with_rationale.csv"
    with pytest.raises(ProductionWriteBlocked):
        with builtins.open(prod_target, "w", encoding="utf-8") as fh:
            fh.write("nope")
    with pytest.raises(ProductionWriteBlocked):
        os.replace(str(tmp_path / "x"), str(prod_target))


def test_28_no_network_or_broker_call_occurs(tmp_path, monkeypatch):
    tm, results, processed, preds = _bind_train_paths(monkeypatch, tmp_path)
    _seed_training_inputs(tmp_path)
    tm.main(["--min-rows", "8"])
    _run_generate(monkeypatch, tmp_path)
    train_src = (_REPO_ROOT / "services" / "train_model.py").read_text(encoding="utf-8")
    gen_src = (_REPO_ROOT / "services" / "generate_signals.py").read_text(encoding="utf-8")
    for src in (train_src, gen_src):
        assert "alpaca" not in src.lower()
        assert "place_order" not in src
    with pytest.raises(RuntimeError, match="network/broker"):
        socket.socket()


def test_29_no_buy_sell_runtime_state_is_created(tmp_path, monkeypatch):
    tm, results, processed, preds = _bind_train_paths(monkeypatch, tmp_path)
    _seed_training_inputs(tmp_path)
    tm.main(["--min-rows", "8"])
    _run_generate(monkeypatch, tmp_path)
    forbidden = [
        tmp_path / "data" / "live" / "paper_sell_arm.sqlite3",
        tmp_path / "data" / "results" / "paper_arm_confirm.json",
        tmp_path / "runtime.lock",
        _PROD_ROOT / "data" / "live" / "paper_sell_arm.sqlite3",
    ]
    for path in forbidden[:3]:
        assert not path.exists()
    train_src = (_REPO_ROOT / "services" / "train_model.py").read_text(encoding="utf-8")
    gen_src = (_REPO_ROOT / "services" / "generate_signals.py").read_text(encoding="utf-8")
    for src in (train_src, gen_src):
        assert "paper_arm" not in src
        assert "Start-Triton" not in src
        assert "confirm_and_create_paper_sell_arm" not in src


def test_30_existing_market_open_stage_order_remains_unchanged():
    src = (_REPO_ROOT / "scripts" / "run_market_open.ps1").read_text(encoding="utf-8")
    d1a = src.index("D1a. GENERATE SIGNALS")
    d1b = src.index("D1b. APPLY SIGNAL LIFECYCLE")
    d1c = src.index("D1c. BUILD EFFECTIVE LIFECYCLE")
    assert d1a < d1b < d1c
    assert "services.generate_signals" in src
    assert "services.train_model" not in src
    daily = (_REPO_ROOT / "run_daily_refresh.ps1").read_text(encoding="utf-8")
    assert '-m", "services.train_model"' in daily or "services.train_model" in daily


def test_production_root_guard_constant_is_explicit():
    assert _PROD_ROOT != _REPO_ROOT
    assert _REPO_ROOT.exists()
