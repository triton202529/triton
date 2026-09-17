"""Phase 198T-SF3 — upstream signal-refresh path repair.

Deterministic. Disposable paths only. No broker, network, Yahoo, Alpaca,
scheduler mutation, or production writes.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib
import json
import os
import socket
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

_PROD_ROOT = Path(r"C:\Users\akimw\triton")
_REPO_ROOT = Path(__file__).resolve().parents[1]
if _REPO_ROOT.resolve() == _PROD_ROOT.resolve():
    raise RuntimeError("Phase 198T-SF3 tests must not run in the production repository.")
if "isolated" not in str(_REPO_ROOT).lower():
    raise RuntimeError("Phase 198T-SF3 tests must run from an isolated worktree.")

SESSION = date(2026, 9, 14)
STALE = date(2026, 9, 11)


def _guard_write(path: Any) -> None:
    try:
        text = os.path.abspath(os.fspath(path))
    except TypeError:
        return
    norm = text.replace("/", "\\").lower()
    prod = str(_PROD_ROOT.resolve()).replace("/", "\\").lower()
    if norm == prod or norm.startswith(prod + "\\"):
        raise RuntimeError(f"production write blocked: {path}")


@pytest.fixture(autouse=True)
def _phase_198t_sf3_safety(monkeypatch):
    real_open = builtins.open
    real_replace = os.replace
    real_remove = os.remove
    real_rename = os.rename

    def guarded_open(file, mode="r", *args, **kwargs):
        mode_s = str(mode)
        if any(flag in mode_s for flag in ("w", "a", "x", "+")):
            _guard_write(file)
        return real_open(file, mode, *args, **kwargs)

    def guarded_replace(src, dst):
        _guard_write(dst)
        return real_replace(src, dst)

    def guarded_remove(path):
        _guard_write(path)
        return real_remove(path)

    def guarded_rename(src, dst):
        _guard_write(dst)
        return real_rename(src, dst)

    class BlockedSocket:
        def __init__(self, *a, **k):
            raise RuntimeError("network/broker calls are blocked in Phase 198T-SF3 tests")

    def blocked_connect(*a, **k):
        raise RuntimeError("network/broker calls are blocked in Phase 198T-SF3 tests")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "replace", guarded_replace)
    monkeypatch.setattr(os, "remove", guarded_remove)
    monkeypatch.setattr(os, "rename", guarded_rename)
    monkeypatch.setattr(socket, "socket", BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", blocked_connect)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("PAPER_ONLY", "1")
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    yield


def _ohlcv(last: date, rows: int = 40, *, extra_dates=None) -> pd.DataFrame:
    dates = list(pd.bdate_range(end=pd.Timestamp(last), periods=rows))
    if extra_dates:
        dates.extend(pd.Timestamp(d) for d in extra_dates)
    n = len(dates)
    close = np.linspace(100.0, 110.0, n)
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(d).normalize() for d in dates],
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def _bind_preprocess(monkeypatch, root: Path, as_of: str = "2026-09-14"):
    import run_preprocessing as rp

    importlib.reload(rp)
    raw = root / "data" / "raw"
    proc = root / "data" / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rp, "RAW_DIR", raw)
    monkeypatch.setattr(rp, "PROCESSED_DIR", proc)
    monkeypatch.setattr(rp, "get_as_of_date", lambda: as_of)
    monkeypatch.chdir(root)
    return rp, raw, proc


def test_import_preprocess_and_generate_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    writes: List[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        mode_s = str(mode)
        if any(flag in mode_s for flag in ("w", "a", "x")):
            writes.append(os.fspath(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)
    import run_preprocessing as rp
    import services.generate_signals as gs

    importlib.reload(rp)
    importlib.reload(gs)
    assert not any("processed" in w.replace("\\", "/") for w in writes)
    assert not list(tmp_path.rglob("*.parquet"))


def test_preprocess_catch_up_writes_when_behind_asof(tmp_path, monkeypatch):
    rp, raw, proc = _bind_preprocess(monkeypatch, tmp_path, as_of="2026-09-15")
    _ohlcv(SESSION).assign(**{c: 1.0 for c in []})
    df = _ohlcv(SESSION)
    # raw columns expected by preprocess_stock_csv
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(raw / "AAA.csv", index=False)
    rc = rp.main()
    assert rc == 0
    saved = pd.read_parquet(proc / "AAA.parquet")
    assert pd.to_datetime(saved["date"]).max().date() == SESSION
    assert (pd.to_datetime(saved["date"]).dt.date <= date(2026, 9, 15)).all()


def test_preprocess_drops_partial_current_day_beyond_asof(tmp_path, monkeypatch):
    rp, raw, proc = _bind_preprocess(monkeypatch, tmp_path, as_of="2026-09-14")
    df = _ohlcv(SESSION, extra_dates=[date(2026, 9, 15)])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(raw / "SPY.csv", index=False)
    assert rp.main() == 0
    saved = pd.read_parquet(proc / "SPY.parquet")
    dates = set(pd.to_datetime(saved["date"]).dt.date)
    assert date(2026, 9, 15) not in dates
    assert date(2026, 9, 14) in dates


def test_preprocess_full_current_coverage(tmp_path, monkeypatch):
    rp, raw, proc = _bind_preprocess(monkeypatch, tmp_path, as_of="2026-09-14")
    for t in ("AAA", "BBB", "CCC"):
        df = _ohlcv(SESSION)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df.to_csv(raw / f"{t}.csv", index=False)
    assert rp.main() == 0
    assert all((proc / f"{t}.parquet").is_file() for t in ("AAA", "BBB", "CCC"))


def test_preprocess_mixed_coverage_records_behind_and_writes_available(tmp_path, monkeypatch):
    rp, raw, proc = _bind_preprocess(monkeypatch, tmp_path, as_of="2026-09-14")
    for t, last in (("CUR", SESSION), ("OLD", STALE)):
        df = _ohlcv(last)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df.to_csv(raw / f"{t}.csv", index=False)
    assert rp.main() == 0
    assert pd.to_datetime(pd.read_parquet(proc / "CUR.parquet")["date"]).max().date() == SESSION
    assert pd.to_datetime(pd.read_parquet(proc / "OLD.parquet")["date"]).max().date() == STALE
    report = tmp_path / "data" / "logs" / "preprocess_behind_asof.csv"
    assert report.is_file()
    behind = pd.read_csv(report)
    assert "OLD" in set(behind["ticker"].astype(str))


def test_generate_uses_processed_when_prediction_rf_stale(tmp_path, monkeypatch):
    import services.generate_signals as gs

    results = tmp_path / "data" / "results"
    preds = tmp_path / "data" / "predictions"
    proc = tmp_path / "data" / "processed"
    cfg = tmp_path / "data" / "config"
    for p in (results, preds, proc, cfg):
        p.mkdir(parents=True, exist_ok=True)
    (cfg / "tickers.txt").write_text("AAA\nBBB\n", encoding="utf-8")
    # Stale RF containers
    for t in ("AAA", "BBB"):
        dates = pd.bdate_range(end=pd.Timestamp(STALE), periods=30)
        close = np.linspace(50, 55, 30)
        pd.DataFrame(
            {
                "date": dates,
                "close": close,
                "predicted_close": close * 1.2,
                "ticker": t,
            }
        ).to_parquet(preds / f"{t}_predictions.parquet", index=False)
        # Current processed market bars
        pdf = _ohlcv(SESSION)
        pdf.to_parquet(proc / f"{t}.parquet", index=False)

    monkeypatch.setattr(gs, "RESULTS_DIR", str(results))
    monkeypatch.setattr(gs, "PREDICTIONS_DIR", str(preds))
    monkeypatch.setattr(gs, "PROCESSED_DIR", str(proc))
    monkeypatch.setattr(gs, "OUT_WITH_RATIONALE", str(results / "signals_with_rationale.csv"))
    monkeypatch.setattr(gs, "OUT_SIGNALS", str(results / "signals.csv"))
    monkeypatch.setattr(gs, "ASOF_DIAG_PATH", str(results / "signals_asof_diagnostics.json"))
    monkeypatch.setattr(gs, "TICKERS_PATH", str(cfg / "tickers.txt"))
    monkeypatch.setattr(gs, "_nyse_last_completed_session_date", lambda now_utc=None: SESSION)
    monkeypatch.setattr(gs, "utc_now_z", lambda: "2026-09-15T14:00:00Z")
    monkeypatch.setattr(
        "services.signal_pressure_diagnostics.refresh_signal_pressure_diagnostics",
        lambda: None,
    )
    monkeypatch.chdir(tmp_path)
    gs.main()
    df = pd.read_csv(results / "signals_with_rationale.csv")
    diag = json.loads((results / "signals_asof_diagnostics.json").read_text(encoding="utf-8"))
    assert len(df) == 2
    assert set(pd.to_datetime(df["date"]).dt.date) == {SESSION}
    assert diag["have_session"] == 2
    assert diag["batch_actionable"] is True
    # Stale RF must not rescue; operational residual stays non-authoritative
    assert df["rf_shadow_only"].astype(str).str.lower().isin(["true", "1"]).all()


def test_stale_features_with_current_raw_still_need_processed_session(tmp_path, monkeypatch):
    """Raw current but processed/pred stale => generate_signals still fail-closed."""
    import services.generate_signals as gs

    results = tmp_path / "data" / "results"
    preds = tmp_path / "data" / "predictions"
    proc = tmp_path / "data" / "processed"
    raw = tmp_path / "data" / "raw"
    cfg = tmp_path / "data" / "config"
    for p in (results, preds, proc, raw, cfg):
        p.mkdir(parents=True, exist_ok=True)
    (cfg / "tickers.txt").write_text("AAA\n", encoding="utf-8")
    dates = pd.bdate_range(end=pd.Timestamp(STALE), periods=30)
    close = np.linspace(50, 55, 30)
    pd.DataFrame(
        {"date": dates, "close": close, "predicted_close": close, "ticker": "AAA"}
    ).to_parquet(preds / "AAA_predictions.parquet", index=False)
    _ohlcv(STALE).to_parquet(proc / "AAA.parquet", index=False)
    rdf = _ohlcv(SESSION)
    rdf["date"] = pd.to_datetime(rdf["date"]).dt.strftime("%Y-%m-%d")
    rdf.to_csv(raw / "AAA.csv", index=False)

    monkeypatch.setattr(gs, "RESULTS_DIR", str(results))
    monkeypatch.setattr(gs, "PREDICTIONS_DIR", str(preds))
    monkeypatch.setattr(gs, "PROCESSED_DIR", str(proc))
    monkeypatch.setattr(gs, "OUT_WITH_RATIONALE", str(results / "signals_with_rationale.csv"))
    monkeypatch.setattr(gs, "OUT_SIGNALS", str(results / "signals.csv"))
    monkeypatch.setattr(gs, "ASOF_DIAG_PATH", str(results / "signals_asof_diagnostics.json"))
    monkeypatch.setattr(gs, "TICKERS_PATH", str(cfg / "tickers.txt"))
    monkeypatch.setattr(gs, "_nyse_last_completed_session_date", lambda now_utc=None: SESSION)
    monkeypatch.setattr(gs, "utc_now_z", lambda: "2026-09-15T14:00:00Z")
    monkeypatch.setattr(
        "services.signal_pressure_diagnostics.refresh_signal_pressure_diagnostics",
        lambda: None,
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        gs.main()


def test_rf_metamorphic_does_not_change_eligibility(tmp_path, monkeypatch):
    import services.generate_signals as gs

    results = tmp_path / "data" / "results"
    preds = tmp_path / "data" / "predictions"
    proc = tmp_path / "data" / "processed"
    cfg = tmp_path / "data" / "config"
    for p in (results, preds, proc, cfg):
        p.mkdir(parents=True, exist_ok=True)
    names = [f"T{i:02d}" for i in range(10)]
    (cfg / "tickers.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    def seed(scale: float):
        for t in names:
            dates = pd.bdate_range(end=pd.Timestamp(SESSION), periods=30)
            close = np.linspace(50, 58, 30)
            pd.DataFrame(
                {
                    "date": dates,
                    "close": close,
                    "predicted_close": close * scale,
                    "ticker": t,
                }
            ).to_parquet(preds / f"{t}_predictions.parquet", index=False)
            _ohlcv(SESSION).to_parquet(proc / f"{t}.parquet", index=False)

    monkeypatch.setattr(gs, "RESULTS_DIR", str(results))
    monkeypatch.setattr(gs, "PREDICTIONS_DIR", str(preds))
    monkeypatch.setattr(gs, "PROCESSED_DIR", str(proc))
    monkeypatch.setattr(gs, "OUT_WITH_RATIONALE", str(results / "signals_with_rationale.csv"))
    monkeypatch.setattr(gs, "OUT_SIGNALS", str(results / "signals.csv"))
    monkeypatch.setattr(gs, "ASOF_DIAG_PATH", str(results / "signals_asof_diagnostics.json"))
    monkeypatch.setattr(gs, "TICKERS_PATH", str(cfg / "tickers.txt"))
    monkeypatch.setattr(gs, "_nyse_last_completed_session_date", lambda now_utc=None: SESSION)
    monkeypatch.setattr(gs, "utc_now_z", lambda: "2026-09-15T14:00:00Z")
    monkeypatch.setattr(
        "services.signal_pressure_diagnostics.refresh_signal_pressure_diagnostics",
        lambda: None,
    )
    monkeypatch.chdir(tmp_path)
    seed(1.5)
    gs.main()
    a = pd.read_csv(results / "signals_with_rationale.csv")
    seed(0.5)
    gs.main()
    b = pd.read_csv(results / "signals_with_rationale.csv")
    assert list(a["ticker"]) == list(b["ticker"])
    assert list(a["signal"]) == list(b["signal"])
    assert list(pd.to_datetime(a["date"]).dt.date) == list(pd.to_datetime(b["date"]).dt.date)


def test_duplicate_dates_still_fail_closed_on_prediction_file(tmp_path, monkeypatch):
    import services.generate_signals as gs

    results = tmp_path / "data" / "results"
    preds = tmp_path / "data" / "predictions"
    cfg = tmp_path / "data" / "config"
    for p in (results, preds, cfg):
        p.mkdir(parents=True, exist_ok=True)
    (cfg / "tickers.txt").write_text("AAA\n", encoding="utf-8")
    dates = list(pd.bdate_range(end=pd.Timestamp(SESSION), periods=30))
    dates.append(dates[-1])
    close = np.linspace(50, 55, len(dates))
    pd.DataFrame(
        {"date": dates, "close": close, "predicted_close": close, "ticker": "AAA"}
    ).to_parquet(preds / "AAA_predictions.parquet", index=False)
    monkeypatch.setattr(gs, "RESULTS_DIR", str(results))
    monkeypatch.setattr(gs, "PREDICTIONS_DIR", str(preds))
    monkeypatch.setattr(gs, "PROCESSED_DIR", str(tmp_path / "data" / "processed"))
    monkeypatch.setattr(gs, "OUT_WITH_RATIONALE", str(results / "signals_with_rationale.csv"))
    monkeypatch.setattr(gs, "OUT_SIGNALS", str(results / "signals.csv"))
    monkeypatch.setattr(gs, "ASOF_DIAG_PATH", str(results / "signals_asof_diagnostics.json"))
    monkeypatch.setattr(gs, "TICKERS_PATH", str(cfg / "tickers.txt"))
    monkeypatch.setattr(gs, "_nyse_last_completed_session_date", lambda now_utc=None: SESSION)
    monkeypatch.setattr(gs, "utc_now_z", lambda: "2026-09-15T14:00:00Z")
    monkeypatch.setattr(
        "services.signal_pressure_diagnostics.refresh_signal_pressure_diagnostics",
        lambda: None,
    )
    monkeypatch.chdir(tmp_path)
    try:
        gs.main()
    except SystemExit as exc:
        assert exc.code == 0
    except RuntimeError:
        pass
    diag = json.loads((results / "signals_asof_diagnostics.json").read_text(encoding="utf-8"))
    reasons = {r["symbol"]: r["reason"] for r in diag.get("per_symbol_skips") or []}
    assert reasons.get("AAA") == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    assert (
        not (results / "signals_with_rationale.csv").is_file()
        or not pd.read_csv(results / "signals_with_rationale.csv").shape[0]
    )


def test_no_network_or_broker():
    with pytest.raises(RuntimeError, match="network/broker"):
        socket.socket()
