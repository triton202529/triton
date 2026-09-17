"""Phase 198T-SF8 — processed session-bar validity contract.

Deterministic. Injected disposable paths only. No broker, network,
production artifact mutation, or scheduled-task edits.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import socket
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

_PROD_ROOT = Path(r"C:\Users\akimw\triton")
_REPO_ROOT = Path(__file__).resolve().parents[1]
if _REPO_ROOT.resolve() == _PROD_ROOT.resolve():
    raise RuntimeError("Phase 198T-SF8 tests must not run in the production repository.")
if "isolated" not in str(_REPO_ROOT).lower():
    raise RuntimeError("Phase 198T-SF8 tests must run from an isolated worktree.")

SESSION = date(2026, 9, 16)
NOW_Z = "2026-09-17T01:23:45Z"
FROZEN_TS = pd.Timestamp("2026-09-17T01:23:45Z")
FROZEN_BARS = {
    "XLE": {
        "open": 65.22000122070312,
        "high": 65.13999938964844,
        "low": 63.68000030517578,
        "close": 64.02999877929688,
        "volume": 48594685.0,
    },
    "BLK": {
        "open": 1063.3800048828125,
        "high": 1061.425048828125,
        "low": 1017.4600219726562,
        "close": 1037.239990234375,
        "volume": 633917.0,
    },
    "C": {
        "open": 136.85000610351562,
        "high": 136.77000427246094,
        "low": 131.0500946044922,
        "close": 132.9499969482422,
        "volume": 9451142.0,
    },
}


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
def _phase_198t_sf8_safety(monkeypatch):
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
            raise RuntimeError("network/broker calls are blocked in Phase 198T-SF8 tests")

    def blocked_connect(*a, **k):
        raise RuntimeError("network/broker calls are blocked in Phase 198T-SF8 tests")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "replace", guarded_replace)
    monkeypatch.setattr(os, "remove", guarded_remove)
    monkeypatch.setattr(os, "rename", guarded_rename)
    monkeypatch.setattr(socket, "socket", BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", blocked_connect)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    monkeypatch.setenv("PAPER_ONLY", "1")
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    yield


def _bind(monkeypatch, root: Path):
    import services.generate_signals as gs

    results = root / "data" / "results"
    preds = root / "data" / "predictions"
    proc = root / "data" / "processed"
    cfg = root / "data" / "config"
    for path in (results, preds, proc, cfg):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(gs, "RESULTS_DIR", str(results))
    monkeypatch.setattr(gs, "PREDICTIONS_DIR", str(preds))
    monkeypatch.setattr(gs, "PROCESSED_DIR", str(proc))
    monkeypatch.setattr(gs, "OUT_WITH_RATIONALE", str(results / "signals_with_rationale.csv"))
    monkeypatch.setattr(gs, "OUT_SIGNALS", str(results / "signals.csv"))
    monkeypatch.setattr(gs, "ASOF_DIAG_PATH", str(results / "signals_asof_diagnostics.json"))
    monkeypatch.setattr(gs, "TICKERS_PATH", str(cfg / "tickers.txt"))
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        "services.signal_pressure_diagnostics.refresh_signal_pressure_diagnostics",
        lambda: None,
    )
    return gs, results, preds, proc, cfg


def _write_tickers(cfg: Path, tradable: Sequence[str], index: Sequence[str] = ()) -> None:
    (cfg / "tickers.txt").write_text(
        "\n".join(list(tradable) + list(index)) + "\n", encoding="utf-8"
    )


def _ohlcv(last: date, rows: int = 40, *, trend: str = "up") -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(last), periods=rows)
    if trend == "up":
        close = np.linspace(50.0, 58.0, rows)
    elif trend == "down":
        close = np.linspace(58.0, 50.0, rows)
    else:
        close = np.full(rows, 54.0)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(rows, 1_000_000.0),
        }
    )


def _write_processed(proc: Path, ticker: str, df: pd.DataFrame) -> Path:
    path = proc / f"{ticker}.parquet"
    df.to_parquet(path, index=False)
    return path


def _valid_processed(proc: Path, ticker: str, last: date = SESSION, **kwargs) -> Path:
    return _write_processed(proc, ticker, _ohlcv(last, **kwargs))


def _overwrite_last(df: pd.DataFrame, **fields: Any) -> pd.DataFrame:
    out = df.copy()
    for key, value in fields.items():
        out.loc[out.index[-1], key] = value
    return out


def _write_pred(preds: Path, ticker: str, last: date = SESSION) -> Path:
    dates = pd.bdate_range(end=pd.Timestamp(last), periods=30)
    close = np.linspace(50.0, 55.0, 30)
    df = pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "predicted_close": close * 1.01,
            "ticker": [ticker] * 30,
        }
    )
    path = preds / f"{ticker}_predictions.parquet"
    df.to_parquet(path, index=False)
    return path


def _fill_valid(proc: Path, preds: Path, count: int, prefix: str = "V") -> List[str]:
    names = [f"{prefix}{i}" for i in range(count)]
    for name in names:
        _valid_processed(proc, name)
        _write_pred(preds, name)
    return names


def _run(
    gs,
    *,
    session: date = SESSION,
    generated_at: str = NOW_Z,
    expect_error: Optional[type] = None,
) -> None:
    import sys

    buf = io.StringIO()
    old = sys.stdout
    nyse = gs._nyse_last_completed_session_date
    try:
        sys.stdout = buf
        gs._nyse_last_completed_session_date = lambda now_utc=None: session
        gs.utc_now_z = lambda: generated_at
        gs._now_utc_ts = lambda: FROZEN_TS
        if expect_error is not None:
            with pytest.raises(expect_error):
                gs.main()
        else:
            gs.main()
    finally:
        sys.stdout = old
        gs._nyse_last_completed_session_date = nyse


def _diag(results: Path) -> Dict[str, Any]:
    return json.loads((results / "signals_asof_diagnostics.json").read_text(encoding="utf-8"))


def _signals(results: Path) -> pd.DataFrame:
    path = results / "signals_with_rationale.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df if not df.empty and "ticker" in df.columns else pd.DataFrame()


def _skip_map(diag: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(r["symbol"]): r for r in diag.get("per_symbol_skips") or []}


def test_import_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    writes: List[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        mode_s = str(mode)
        if any(flag in mode_s for flag in ("w", "a", "x", "+")):
            writes.append(os.fspath(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)
    import services.generate_signals as gs

    assert gs.SKIP_INVALID_PROCESSED_SESSION_BAR == "INVALID_PROCESSED_SESSION_BAR"
    assert writes == []


@pytest.mark.parametrize("ticker", ["XLE", "BLK", "C"])
def test_frozen_sep16_invalid_bars_skip_and_do_not_count(tmp_path, monkeypatch, ticker):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    fillers = _fill_valid(proc, preds, 9)
    _write_tickers(cfg, fillers + [ticker])
    frozen = _overwrite_last(_ohlcv(SESSION), **FROZEN_BARS[ticker])
    _write_processed(proc, ticker, frozen)
    _write_pred(preds, ticker)
    _run(gs)
    diag = _diag(results)
    skipped = _skip_map(diag)
    df = _signals(results)
    assert ticker not in set(df["ticker"].astype(str))
    assert skipped[ticker]["reason"] == gs.SKIP_INVALID_PROCESSED_SESSION_BAR
    assert skipped[ticker]["actual_latest_bar_date"] == "2026-09-16"
    assert skipped[ticker]["source_sha256"]
    assert diag["have_session"] == 9
    assert diag["coverage"] == pytest.approx(0.9)
    assert ticker not in (diag.get("current_tradable_symbols") or [])


def test_rf_cannot_restore_invalid_processed_bar(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    fillers = _fill_valid(proc, preds, 9)
    _write_tickers(cfg, fillers + ["XLE"])
    _write_processed(proc, "XLE", _overwrite_last(_ohlcv(SESSION), **FROZEN_BARS["XLE"]))
    _write_pred(preds, "XLE")
    _run(gs)
    skipped = _skip_map(_diag(results))
    df = _signals(results)
    assert "XLE" not in set(df["ticker"].astype(str))
    assert skipped["XLE"]["reason"] == gs.SKIP_INVALID_PROCESSED_SESSION_BAR
    assert _diag(results)["have_session"] == 9


@pytest.mark.parametrize(
    "fields",
    [
        {"high": 9.0, "open": 10.0, "close": 9.5, "low": 8.0},
        {"high": 9.0, "open": 8.5, "close": 10.0, "low": 8.0},
        {"low": 10.5, "open": 10.0, "close": 9.5, "high": 11.0},
        {"low": 10.5, "open": 9.0, "close": 10.0, "high": 11.0},
        {"high": 9.0, "low": 10.0, "open": 9.5, "close": 9.5},
    ],
)
def test_ohlc_boundary_failures_skip(tmp_path, monkeypatch, fields):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    fillers = _fill_valid(proc, preds, 9)
    _write_tickers(cfg, fillers + ["BAD"])
    _write_processed(proc, "BAD", _overwrite_last(_ohlcv(SESSION), volume=1000.0, **fields))
    _write_pred(preds, "BAD")
    _run(gs)
    skipped = _skip_map(_diag(results))
    assert skipped["BAD"]["reason"] == gs.SKIP_INVALID_PROCESSED_SESSION_BAR
    assert "BAD" not in set(_signals(results)["ticker"].astype(str))
    assert _diag(results)["have_session"] == 9


@pytest.mark.parametrize(
    "fields",
    [
        {"open": 10.0, "high": 10.0, "low": 9.0, "close": 9.5},
        {"open": 9.5, "high": 10.0, "low": 9.0, "close": 10.0},
        {"open": 10.0, "high": 11.0, "low": 10.0, "close": 10.5},
        {"open": 10.5, "high": 11.0, "low": 10.0, "close": 10.0},
        {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
    ],
)
def test_valid_equality_cases_emit(tmp_path, monkeypatch, fields):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["EQ"])
    _write_processed(proc, "EQ", _overwrite_last(_ohlcv(SESSION), volume=1000.0, **fields))
    _write_pred(preds, "EQ")
    _run(gs)
    df = _signals(results)
    assert list(df["ticker"].astype(str)) == ["EQ"]
    assert _diag(results)["have_session"] == 1
    assert _diag(results)["batch_actionable"] is True


@pytest.mark.parametrize(
    "fields,reason",
    [
        ({"close": float("nan")}, "MISSING_EXPECTED_SESSION_BAR"),
        ({"open": float("inf")}, "INVALID_PROCESSED_SESSION_BAR"),
        ({"high": float("-inf")}, "INVALID_PROCESSED_SESSION_BAR"),
        ({"close": 0.0}, "INVALID_PROCESSED_SESSION_BAR"),
        ({"open": -1.0}, "INVALID_PROCESSED_SESSION_BAR"),
        ({"volume": -5.0}, "INVALID_PROCESSED_SESSION_BAR"),
    ],
)
def test_bad_numeric_values_skip(tmp_path, monkeypatch, fields, reason):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    fillers = _fill_valid(proc, preds, 9)
    _write_tickers(cfg, fillers + ["BAD"])
    _write_processed(proc, "BAD", _overwrite_last(_ohlcv(SESSION), **fields))
    _run(gs)
    skipped = _skip_map(_diag(results))
    assert skipped["BAD"]["reason"] == reason
    assert _diag(results)["have_session"] == 9


def test_missing_feature_history_skips(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    fillers = _fill_valid(proc, preds, 9)
    _write_tickers(cfg, fillers + ["SHORT"])
    _write_processed(proc, "SHORT", _ohlcv(SESSION, rows=10))
    _run(gs)
    skipped = _skip_map(_diag(results))
    assert skipped["SHORT"]["reason"] == gs.SKIP_INVALID_PROCESSED_SESSION_BAR
    assert _diag(results)["have_session"] == 9


def test_missing_session_bar_keeps_dated_missing_skip(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    fillers = _fill_valid(proc, preds, 9)
    _write_tickers(cfg, fillers + ["STALE"])
    _valid_processed(proc, "STALE", last=date(2026, 9, 15))
    _run(gs)
    skipped = _skip_map(_diag(results))
    assert skipped["STALE"]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    assert skipped["STALE"]["actual_latest_bar_date"] == "2026-09-15"
    assert _diag(results)["have_session"] == 9


def test_duplicate_session_date_skips(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    fillers = _fill_valid(proc, preds, 9)
    _write_tickers(cfg, fillers + ["DUP"])
    base = _ohlcv(SESSION)
    dup = pd.concat([base, base.iloc[[-1]]], ignore_index=True)
    _write_processed(proc, "DUP", dup)
    _run(gs)
    assert _skip_map(_diag(results))["DUP"]["reason"] == gs.SKIP_DUPLICATE_DATE
    assert _diag(results)["have_session"] == 9


def test_coverage_boundary_090(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = [f"T{i}" for i in range(10)]
    _write_tickers(cfg, names)
    for name in names:
        _valid_processed(proc, name, trend="up" if name != "T9" else "down")
        _write_pred(preds, name)
    _write_processed(proc, "T9", _overwrite_last(_ohlcv(SESSION), **FROZEN_BARS["XLE"]))
    _run(gs)
    diag = _diag(results)
    assert diag["have_session"] == 9
    assert diag["coverage"] == pytest.approx(0.9)
    assert diag["batch_actionable"] is True
    assert len(_signals(results)) == 9
    _write_processed(proc, "T8", _overwrite_last(_ohlcv(SESSION), **FROZEN_BARS["BLK"]))
    _run(gs, expect_error=RuntimeError)
    diag2 = _diag(results)
    assert diag2["have_session"] == 8
    assert diag2["coverage"] == pytest.approx(0.8)
    assert diag2["batch_actionable"] is False
    assert _signals(results).empty


def test_other_symbols_retain_operational_decisions(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    keepers = [f"K{i}" for i in range(9)]
    _write_tickers(cfg, keepers)
    for i, name in enumerate(keepers):
        _valid_processed(proc, name, trend="up" if i % 2 == 0 else "down")
        _write_pred(preds, name)
    _run(gs)
    baseline = {
        str(r.ticker): (str(r.signal), float(r.confidence)) for r in _signals(results).itertuples()
    }
    assert len(baseline) == 9
    _write_tickers(cfg, keepers + ["XLE"])
    _write_processed(
        proc, "XLE", _overwrite_last(_ohlcv(SESSION, trend="flat"), **FROZEN_BARS["XLE"])
    )
    _write_pred(preds, "XLE")
    _run(gs)
    df = _signals(results)
    assert set(df["ticker"].astype(str)) == set(keepers)
    for name, expected in baseline.items():
        row = df[df["ticker"] == name].iloc[0]
        assert str(row["signal"]) == expected[0]
        assert float(row["confidence"]) == expected[1]


def test_first_and_second_pass_use_one_validity_decision(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    fillers = _fill_valid(proc, preds, 9)
    _write_tickers(cfg, fillers + ["XLE"])
    _write_processed(proc, "XLE", _overwrite_last(_ohlcv(SESSION), **FROZEN_BARS["XLE"]))
    view = gs._inspect_processed_session("XLE", SESSION)
    close_hit, close_v, _ = gs._processed_session_close("XLE", SESSION)
    hist, _path, hist_close = gs._processed_session_history("XLE", SESSION)
    assert view.ok is False
    assert close_hit is None and close_v is None
    assert hist is None and hist_close is None
    _run(gs)
    diag = _diag(results)
    assert diag["have_session"] == 9
    assert "XLE" not in set(_signals(results)["ticker"].astype(str))


def test_independent_risk_overlays_still_function():
    from services.performance_risk_overlay import _classify_row
    from services.portfolio_allocation_engine import _decide_action

    force = _classify_row(
        total_pl=0.0,
        unrealized_pl=-250.0,
        drag_flag=False,
        win_rate=None,
        has_win_rate=False,
    )
    assert "FORCE_EXIT" in str(force)
    action, reason = _decide_action(
        is_held=True,
        risk_components=["FORCE_EXIT"],
        edge_score=0.2,
        sizing_tier="NO_EDGE",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="HOLD",
        confidence=0.4,
        delta_pct=0.0,
    )
    assert action == "EXIT"
    assert "FORCE_EXIT" in reason


def test_asof_coverage_threshold_unchanged():
    import services.generate_signals as gs

    assert gs.ASOF_COVERAGE == 0.90
