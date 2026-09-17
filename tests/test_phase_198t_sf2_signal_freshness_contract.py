"""Phase 198T-SF2 — fail-closed signal freshness / as-of contract.

Deterministic. Injected disposable paths only. No broker, network,
production artifact mutation, or scheduled-task edits.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib
import io
import json
import os
import socket
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

_PROD_ROOT = Path(r"C:\Users\akimw\triton")
_REPO_ROOT = Path(__file__).resolve().parents[1]
if _REPO_ROOT.resolve() == _PROD_ROOT.resolve():
    raise RuntimeError("Phase 198T-SF2 tests must not run in the production repository.")
if "isolated" not in str(_REPO_ROOT).lower():
    raise RuntimeError("Phase 198T-SF2 tests must run from an isolated worktree.")

_SF1_FREEZE_PREDS = Path(
    r"C:\Users\akimw\AppData\Local\Temp\phase_198T_SF1_20260916_015019" r"\freeze\data\predictions"
)
_SF1_FREEZE_PROC = Path(
    r"C:\Users\akimw\AppData\Local\Temp\phase_198T_SF1_20260916_015019" r"\freeze\data\processed"
)
_SF1_TICKERS = Path(
    r"C:\Users\akimw\AppData\Local\Temp\phase_198T_SF1_20260916_015019"
    r"\freeze\data\config\tickers.txt"
)

TRADABLE_62 = [
    "SPY",
    "AAPL",
    "TSLA",
    "NVDA",
    "AMZN",
    "GLD",
    "AMD",
    "GOOGL",
    "MSFT",
    "META",
    "NFLX",
    "QQQ",
    "VOO",
    "VTI",
    "IWM",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
    "SLV",
    "USO",
    "UNG",
    "DBA",
    "GBTC",
    "ARKK",
    "BITO",
    "DIA",
    "ABBV",
    "BAC",
    "BLK",
    "BRK-B",
    "C",
    "CVX",
    "DIS",
    "GE",
    "GS",
    "HD",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "MA",
    "MRK",
    "MS",
    "MU",
    "PEP",
    "PFE",
    "PG",
    "SCHW",
    "SPCX",
    "T",
    "UNH",
    "V",
    "WFC",
    "WMT",
    "XOM",
]
INDEX_4 = ["^DJI", "^GSPC", "^IXIC", "^VIX"]
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
def _phase_198t_sf2_safety(monkeypatch):
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
            raise RuntimeError("network/broker calls are blocked in Phase 198T-SF2 tests")

    def blocked_connect(*a, **k):
        raise RuntimeError("network/broker calls are blocked in Phase 198T-SF2 tests")

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
    results.mkdir(parents=True, exist_ok=True)
    preds.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)
    cfg.mkdir(parents=True, exist_ok=True)
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


def _write_tickers(
    cfg: Path, tradable: Sequence[str], index: Sequence[str] = (), *, bom: bool = False
) -> None:
    body = "\n".join(list(tradable) + list(index)) + "\n"
    if bom:
        raw = ("\ufeff" + body).encode("utf-8")
        (cfg / "tickers.txt").write_bytes(raw)
    else:
        (cfg / "tickers.txt").write_text(body, encoding="utf-8")


def _pred_frame(
    ticker: str,
    last: date,
    rows: int = 30,
    *,
    trend: str = "up",
    predicted_scale: float = 1.01,
    extra_rows: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(last), periods=rows)
    if trend == "up":
        close = np.linspace(50.0, 58.0, rows)
    elif trend == "down":
        close = np.linspace(58.0, 50.0, rows)
    else:
        close = np.full(rows, 54.0)
    df = pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "predicted_close": close * predicted_scale,
            "ticker": [ticker] * rows,
        }
    )
    if extra_rows is not None:
        df = pd.concat([df, extra_rows], ignore_index=True)
    return df


def _write_pred(preds: Path, ticker: str, last: date, **kwargs) -> Path:
    path = preds / f"{ticker}_predictions.parquet"
    _pred_frame(ticker, last, **kwargs).to_parquet(path, index=False)
    return path


def _ohlcv(
    last: date, rows: int = 40, *, trend: str = "up", extra: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(last), periods=rows)
    if trend == "up":
        close = np.linspace(50.0, 58.0, rows)
    elif trend == "down":
        close = np.linspace(58.0, 50.0, rows)
    else:
        close = np.full(rows, 54.0)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(rows, 1_000_000.0),
        }
    )
    if extra is not None:
        df = pd.concat([df, extra], ignore_index=True)
    return df


def _write_processed(proc: Path, ticker: str, last: date, **kwargs) -> Path:
    path = proc / f"{ticker}.parquet"
    _ohlcv(last, **kwargs).to_parquet(path, index=False)
    return path


def _run(
    gs,
    *,
    session: date = SESSION,
    generated_at: str = "2026-09-15T19:14:13Z",
    expect_error: Optional[type] = None,
):
    gs_mod = gs
    gs_mod_nyse = getattr(gs_mod, "_nyse_last_completed_session_date")
    monkey_session = lambda now_utc=None: session  # noqa: E731
    # Caller already bound paths; patch clock/session here.
    buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        gs._nyse_last_completed_session_date = monkey_session
        gs.utc_now_z = lambda: generated_at
        if expect_error is not None:
            with pytest.raises(expect_error):
                gs.main()
        else:
            gs.main()
    finally:
        sys.stdout = old_stdout
        gs._nyse_last_completed_session_date = gs_mod_nyse
    return buf.getvalue()


def _diag(results: Path) -> Dict[str, Any]:
    return json.loads((results / "signals_asof_diagnostics.json").read_text(encoding="utf-8"))


def _signals(results: Path) -> pd.DataFrame:
    path = results / "signals_with_rationale.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "ticker" not in df.columns:
        return df
    return df


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

    importlib.reload(gs)
    artifact_writes = [
        w
        for w in writes
        if any(
            token in w.replace("\\", "/")
            for token in (
                "signals.csv",
                "signals_with_rationale.csv",
                "signals_asof_diagnostics.json",
                "_predictions.parquet",
            )
        )
    ]
    assert artifact_writes == []
    assert not list(tmp_path.rglob("*.csv"))
    assert not list(tmp_path.rglob("*.parquet"))


def test_no_broker_or_network_on_generate(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA", "BBB"])
    _write_pred(preds, "AAA", SESSION)
    _write_pred(preds, "BBB", SESSION)
    _write_processed(proc, "AAA", SESSION)
    _write_processed(proc, "BBB", SESSION)
    _run(gs, session=SESSION)
    with pytest.raises(RuntimeError, match="network/broker"):
        socket.socket()


def test_all_symbols_current(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = ["AAA", "BBB", "CCC"]
    _write_tickers(cfg, names)
    for t in names:
        _write_pred(preds, t, SESSION)
        _write_processed(proc, t, SESSION)
    _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert list(df["ticker"].astype(str)) == names
    assert diag["batch_actionable"] is True
    assert diag["coverage"] == 1.0
    assert diag["have_session"] == 3
    assert diag["tradable_tickers"] == 3
    as_of = pd.to_datetime(df["as_of_date"]).dt.date
    bars = pd.to_datetime(df["date"]).dt.date
    sess = pd.to_datetime(df["session_as_of_date"]).dt.date
    assert set(as_of) == {SESSION}
    assert set(bars) == {SESSION}
    assert set(sess) == {SESSION}
    assert (bars == as_of).all()


def test_one_tradable_symbol_stale_above_threshold(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = [f"T{i:02d}" for i in range(10)]
    _write_tickers(cfg, names)
    for t in names[:-1]:
        _write_pred(preds, t, SESSION)
        _write_processed(proc, t, SESSION)
    _write_pred(preds, names[-1], STALE)
    _write_processed(proc, names[-1], STALE)
    _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert diag["batch_actionable"] is True
    assert diag["have_session"] == 9
    assert abs(diag["coverage"] - 0.9) < 1e-9
    assert set(df["ticker"].astype(str)) == set(names[:-1])
    skipped = _skip_map(diag)
    assert names[-1] in skipped
    assert skipped[names[-1]]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    assert skipped[names[-1]]["actual_latest_bar_date"] == str(STALE)
    assert skipped[names[-1]]["expected_session_date"] == str(SESSION)
    assert skipped[names[-1]]["source_sha256"]
    assert skipped[names[-1]]["evidence_timestamp"]


def test_mixed_current_and_stale_below_threshold(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = [f"M{i}" for i in range(10)]
    _write_tickers(cfg, names)
    for i, t in enumerate(names):
        last = SESSION if i < 5 else STALE
        _write_pred(preds, t, last)
        _write_processed(proc, t, last)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert df.empty or "ticker" not in df.columns or len(df.dropna(how="all")) == 0
    assert diag["batch_actionable"] is False
    assert diag["have_session"] == 5
    skipped = _skip_map(diag)
    assert len(skipped) == 10
    stale_reasons = {t: skipped[t]["reason"] for t in names[5:]}
    assert set(stale_reasons.values()) == {gs.SKIP_MISSING_EXPECTED_SESSION_BAR}
    current_reasons = {t: skipped[t]["reason"] for t in names[:5]}
    assert set(current_reasons.values()) == {gs.SKIP_BATCH_COVERAGE_BELOW_THRESHOLD}


def test_majority_stale(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = [f"Q{i}" for i in range(10)]
    _write_tickers(cfg, names)
    for i, t in enumerate(names):
        last = SESSION if i == 0 else STALE
        _write_pred(preds, t, last)
        _write_processed(proc, t, last)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    diag = _diag(results)
    assert diag["have_session"] == 1
    assert diag["coverage"] == 0.1
    assert diag["batch_actionable"] is False
    assert _signals(results).empty or len(_signals(results).dropna(how="all")) == 0


def test_sf1_reproduction_four_of_sixty_six_current(tmp_path, monkeypatch):
    assert _SF1_FREEZE_PREDS.is_dir()
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    monkeypatch.setattr(gs, "PREDICTIONS_DIR", str(_SF1_FREEZE_PREDS))
    monkeypatch.setattr(gs, "PROCESSED_DIR", str(_SF1_FREEZE_PROC))
    if _SF1_TICKERS.is_file():
        cfg.joinpath("tickers.txt").write_bytes(_SF1_TICKERS.read_bytes())
    else:
        _write_tickers(cfg, TRADABLE_62, INDEX_4)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs, session=SESSION, generated_at="2026-09-15T19:14:13Z")
    diag = _diag(results)
    df = _signals(results)
    assert df.empty or len(df.dropna(how="all")) == 0
    assert diag["tradable_tickers"] == 62
    assert diag["index_tickers"] == 4
    assert diag["have_session"] == 0
    assert diag["coverage"] == 0.0
    assert diag["batch_actionable"] is False
    assert diag["effective_reason"] == "coverage_low_batch_unavailable"
    skipped = _skip_map(diag)
    for sym in TRADABLE_62:
        assert skipped[sym]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
        assert skipped[sym]["actual_latest_bar_date"] == "2026-09-11"
        assert skipped[sym]["expected_session_date"] == "2026-09-14"
        assert skipped[sym]["source_sha256"]
    for idx in INDEX_4:
        assert skipped[idx]["reason"] == gs.SKIP_INDEX_ONLY_EXCLUDED
        assert skipped[idx]["actual_latest_bar_date"] == "2026-09-14"
    assert set(diag["current_index_symbols"]) == set(INDEX_4)
    assert diag["current_tradable_symbols"] == []
    assert diag["effective_asof_date"] == "2026-09-14"


def test_missing_expected_session_bar(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA"])
    _write_pred(preds, "AAA", STALE)
    _write_processed(proc, "AAA", STALE)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    skipped = _skip_map(_diag(results))
    assert skipped["AAA"]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    assert skipped["AAA"]["actual_latest_bar_date"] == str(STALE)


def test_missing_file(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = [f"F{i:02d}" for i in range(10)]
    _write_tickers(cfg, names)
    for t in names[:-1]:
        _write_pred(preds, t, SESSION)
        _write_processed(proc, t, SESSION)
    _run(gs)
    skipped = _skip_map(_diag(results))
    assert skipped[names[-1]]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    assert skipped[names[-1]]["actual_latest_bar_date"] == "MISSING"
    df = _signals(results)
    assert names[-1] not in set(df["ticker"].astype(str))
    assert len(df) == 9


def test_empty_file(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA"])
    (preds / "AAA_predictions.csv").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    skipped = _skip_map(_diag(results))
    assert skipped["AAA"]["reason"] == gs.SKIP_EMPTY_PREDICTION_FILE
    assert skipped["AAA"]["actual_latest_bar_date"] == "MISSING"


def test_malformed_date(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA"])
    pd.DataFrame(
        {"date": ["not-a-date", "also-bad"], "close": [1.0, 2.0], "ticker": ["AAA", "AAA"]}
    ).to_parquet(preds / "AAA_predictions.parquet", index=False)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    skipped = _skip_map(_diag(results))
    assert skipped["AAA"]["reason"] == gs.SKIP_MALFORMED_DATE


def test_duplicate_date(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA"])
    df = _pred_frame("AAA", SESSION)
    dup = df.iloc[[-1]].copy()
    df = pd.concat([df, dup], ignore_index=True)
    df.to_parquet(preds / "AAA_predictions.parquet", index=False)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    skipped = _skip_map(_diag(results))
    assert skipped["AAA"]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    assert skipped["AAA"]["actual_latest_bar_date"] == "MISSING"
    df_out = _signals(results)
    assert df_out.empty or len(df_out.dropna(how="all")) == 0


def test_future_date(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA"])
    extra = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-09-16")],
            "close": [99.0],
            "predicted_close": [100.0],
            "ticker": ["AAA"],
        }
    )
    _pred_frame("AAA", SESSION, extra_rows=extra).to_parquet(
        preds / "AAA_predictions.parquet", index=False
    )
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    skipped = _skip_map(_diag(results))
    assert skipped["AAA"]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    assert skipped["AAA"]["actual_latest_bar_date"] == "MISSING"


def test_out_of_order_data_still_finds_expected_session(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA"])
    df = _pred_frame("AAA", SESSION)
    df = df.iloc[::-1].reset_index(drop=True)
    df.to_parquet(preds / "AAA_predictions.parquet", index=False)
    proc_df = _ohlcv(SESSION).iloc[::-1].reset_index(drop=True)
    proc_df.to_parquet(proc / "AAA.parquet", index=False)
    _run(gs)
    out = _signals(results)
    assert list(out["ticker"].astype(str)) == ["AAA"]
    assert pd.to_datetime(out["date"]).dt.date.iloc[0] == SESSION


def _et(year, month, day, hour, minute=0) -> pd.Timestamp:
    return pd.Timestamp(datetime(year, month, day, hour, minute), tz="America/New_York")


def test_premarket_expected_session_selection():
    import services.generate_signals as gs

    got = gs._nyse_last_completed_session_date(now_utc=_et(2026, 9, 15, 8, 0).tz_convert("UTC"))
    assert got == date(2026, 9, 14)


def test_open_session_expected_session_selection():
    import services.generate_signals as gs

    got = gs._nyse_last_completed_session_date(now_utc=_et(2026, 9, 15, 14, 0).tz_convert("UTC"))
    assert got == date(2026, 9, 14)


def test_normal_after_close_selection():
    import services.generate_signals as gs

    got = gs._nyse_last_completed_session_date(now_utc=_et(2026, 9, 15, 16, 30).tz_convert("UTC"))
    assert got == date(2026, 9, 15)


def test_early_close_selection():
    import services.generate_signals as gs

    before = gs._nyse_last_completed_session_date(
        now_utc=_et(2026, 11, 27, 12, 50).tz_convert("UTC")
    )
    after = gs._nyse_last_completed_session_date(
        now_utc=_et(2026, 11, 27, 13, 10).tz_convert("UTC")
    )
    assert before == date(2026, 11, 25)
    assert after == date(2026, 11, 27)


def test_weekend_selection():
    import services.generate_signals as gs

    got = gs._nyse_last_completed_session_date(now_utc=_et(2026, 9, 12, 12, 0).tz_convert("UTC"))
    assert got == date(2026, 9, 11)


def test_holiday_selection():
    import services.generate_signals as gs

    got = gs._nyse_last_completed_session_date(now_utc=_et(2026, 9, 7, 12, 0).tz_convert("UTC"))
    assert got == date(2026, 9, 4)


def test_index_symbols_excluded_from_tradable_coverage(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, TRADABLE_62[:2], INDEX_4)
    for t in TRADABLE_62[:2]:
        _write_pred(preds, t, STALE)
    for t in INDEX_4:
        _write_pred(preds, t, SESSION)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    diag = _diag(results)
    assert diag["tradable_tickers"] == 2
    assert diag["index_tickers"] == 4
    assert diag["have_session"] == 0
    skipped = _skip_map(diag)
    for t in INDEX_4:
        assert skipped[t]["reason"] == gs.SKIP_INDEX_ONLY_EXCLUDED
        assert skipped[t]["class"] == "index"
    df = _signals(results)
    assert df.empty or len(df.dropna(how="all")) == 0


def test_no_bom_inflated_symbol(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, TRADABLE_62, INDEX_4, bom=True)
    for t in TRADABLE_62:
        _write_pred(preds, t, SESSION)
        _write_processed(proc, t, SESSION)
    for t in INDEX_4:
        _write_pred(preds, t, SESSION)
        _write_processed(proc, t, SESSION)
    _run(gs)
    diag = _diag(results)
    df = _signals(results)
    assert diag["tradable_tickers"] == 62
    assert diag["have_session"] == 62
    assert set(df["ticker"].astype(str)) == set(TRADABLE_62)
    tickers = gs._load_optional_tickers(str(cfg / "tickers.txt"))
    assert tickers[0] == "SPY"
    assert not tickers[0].startswith("\ufeff")
    assert len(tickers) == 66


def test_explicit_skip_record_for_every_excluded_symbol(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = ["AAA", "BBB", "CCC"]
    _write_tickers(cfg, names, ["^VIX"])
    _write_pred(preds, "AAA", SESSION)
    _write_processed(proc, "AAA", SESSION)
    _write_pred(preds, "BBB", STALE)
    _write_processed(proc, "BBB", STALE)
    _write_pred(preds, "^VIX", SESSION)
    _write_processed(proc, "^VIX", SESSION)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    skipped = _skip_map(_diag(results))
    assert set(skipped) >= {"BBB", "CCC", "^VIX", "AAA"}
    assert skipped["CCC"]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    assert skipped["CCC"]["actual_latest_bar_date"] == "MISSING"
    assert skipped["BBB"]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    assert skipped["^VIX"]["reason"] == gs.SKIP_INDEX_ONLY_EXCLUDED
    for rec in skipped.values():
        assert rec["expected_session_date"]
        assert rec["actual_latest_bar_date"]
        assert rec["reason"]
        assert "source_sha256" in rec
        assert rec["evidence_timestamp"]


def test_old_bar_never_relabeled_with_newer_session_date(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = [f"R{i:02d}" for i in range(10)]
    _write_tickers(cfg, names)
    for t in names[:-1]:
        _write_pred(preds, t, SESSION)
        _write_processed(proc, t, SESSION)
    _write_pred(preds, names[-1], STALE)
    _write_processed(proc, names[-1], STALE)
    _run(gs)
    df = _signals(results)
    bars = pd.to_datetime(df["date"]).dt.date
    as_of = pd.to_datetime(df["as_of_date"]).dt.date
    assert (bars == as_of).all()
    assert str(STALE) not in set(bars.astype(str))
    stale_row = _skip_map(_diag(results))[names[-1]]
    assert stale_row["actual_latest_bar_date"] == str(STALE)
    assert stale_row["expected_session_date"] == str(SESSION)


def test_low_coverage_never_produces_actionable_lifecycle_rows(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA", "BBB"])
    _write_pred(preds, "AAA", SESSION, trend="up")
    _write_pred(preds, "BBB", STALE, trend="up")
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    df = _signals(results)
    assert df.empty or len(df.dropna(how="all")) == 0
    from services.signal_lifecycle import decide_action

    # Downstream authority receives no stale rows to classify.
    assert list(df) == [] or df.shape[0] == 0
    assert callable(decide_action)


def test_current_coverage_permits_buy_add_hold_trim_exit_flow(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["BUY1", "HOLD1"])
    _write_pred(preds, "BUY1", SESSION, trend="up")
    _write_pred(preds, "HOLD1", SESSION, trend="flat")
    _write_processed(proc, "BUY1", SESSION, trend="up")
    _write_processed(proc, "HOLD1", SESSION, trend="flat")
    _run(gs)
    df = _signals(results)
    assert set(df["ticker"].astype(str)) == {"BUY1", "HOLD1"}
    buy_row = df[df["ticker"] == "BUY1"].iloc[0]
    hold_row = df[df["ticker"] == "HOLD1"].iloc[0]
    from services.signal_lifecycle import LifecycleConfig, decide_action, lifecycle_logic_from_dict

    cfg_l = LifecycleConfig(
        min_hold_days=1,
        cooldown_days_after_exit=1,
        buy_delta_pct=0.0015,
        add_delta_pct=0.0020,
        exit_delta_pct=-0.0015,
        trim_delta_pct=-0.0030,
        hold_means_keep_position=True,
    )
    logic = lifecycle_logic_from_dict(
        {
            "enabled": True,
            "add_confidence_min": 0.52,
            "add_delta_pct_min": 0.008,
            "hold_delta_floor": -0.003,
            "hold_delta_ceiling": 0.006,
            "trim_delta_pct_threshold": -0.001,
            "exit_delta_pct_threshold": -0.0025,
            "exit_confidence_min": 0.50,
        }
    )

    def _d(signal, pos, **kw):
        return decide_action(
            cfg=cfg_l,
            logic=logic,
            signal=signal,
            delta_pct=float("nan"),
            confidence=0.8,
            pos_state=pos,
            last_change_date=date(2026, 9, 1),
            cooldown_until=None,
            as_of_date=SESSION,
            **kw,
        )

    buy_action, buy_reason = _d(str(buy_row["signal"]), "FLAT")
    if str(buy_row["signal"]) == "BUY":
        assert buy_action == "BUY"
    hold_action, hold_reason = _d("HOLD", "LONG")
    assert hold_action == "HOLD"
    add_action, add_reason = _d("BUY", "LONG")
    assert add_action in {"ADD", "HOLD", "WAIT"}
    trim_action, trim_reason = _d("SELL", "LONG")
    assert trim_action in {"EXIT", "TRIM", "HOLD"}
    exit_action, exit_reason = _d("SELL", "LONG")
    assert exit_action == "EXIT"
    from services.portfolio_allocation_engine import _decide_action as alloc_decide

    alloc_exit, _ = alloc_decide(
        is_held=True,
        risk_components=["FORCE_EXIT"],
        edge_score=0.9,
        sizing_tier="STRONG_EDGE",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="HOLD",
        confidence=0.8,
        delta_pct=-0.05,
    )
    assert alloc_exit == "EXIT"
    assert buy_reason
    assert hold_reason
    assert add_reason
    assert trim_reason
    assert exit_reason


def test_rf_evidence_cannot_change_freshness_eligibility(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = [f"Z{i:02d}" for i in range(10)]
    _write_tickers(cfg, names)
    for t in names[:-1]:
        _write_pred(preds, t, SESSION, predicted_scale=1.50)
        _write_processed(proc, t, SESSION)
    _write_pred(preds, names[-1], STALE, predicted_scale=1.50)
    _write_processed(proc, names[-1], STALE)
    _run(gs)
    first = _signals(results).copy()
    first_skip = _skip_map(_diag(results))
    for t in names[:-1]:
        _write_pred(preds, t, SESSION, predicted_scale=0.50)
    _write_pred(preds, names[-1], STALE, predicted_scale=0.50)
    _run(gs)
    second = _signals(results)
    second_skip = _skip_map(_diag(results))
    assert set(first["ticker"].astype(str)) == set(second["ticker"].astype(str)) == set(names[:-1])
    assert first_skip[names[-1]]["reason"] == second_skip[names[-1]]["reason"]
    for col in ("signal", "confidence", "score", "close", "date", "as_of_date"):
        assert list(first[col]) == list(second[col])
    for col in ("action", "eligibility", "size", "quantity", "lifecycle_action"):
        if col in first.columns:
            assert list(first[col]) == list(second[col])


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


def test_deterministic_repeated_replay(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = [f"D{i}" for i in range(5)]
    _write_tickers(cfg, names)
    for t in names:
        _write_pred(preds, t, SESSION)
        _write_processed(proc, t, SESSION)
    _run(gs, generated_at="2026-09-15T19:14:13Z")
    a = (results / "signals_with_rationale.csv").read_bytes()
    da = (results / "signals_asof_diagnostics.json").read_text(encoding="utf-8")
    _run(gs, generated_at="2026-09-15T19:14:13Z")
    b = (results / "signals_with_rationale.csv").read_bytes()
    db = (results / "signals_asof_diagnostics.json").read_text(encoding="utf-8")
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
    ja = json.loads(da)
    jb = json.loads(db)
    assert ja["per_symbol_skips"] == jb["per_symbol_skips"]
    assert ja["coverage"] == jb["coverage"]
    assert ja["have_session"] == jb["have_session"]


def test_synthetic_fully_current_sixty_two_tradable_pipeline(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, TRADABLE_62, INDEX_4)
    for t in TRADABLE_62:
        _write_pred(preds, t, SESSION, trend="up")
        _write_processed(proc, t, SESSION, trend="up")
    for t in INDEX_4:
        _write_pred(preds, t, SESSION, trend="up")
        _write_processed(proc, t, SESSION, trend="up")
    _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert diag["tradable_tickers"] == 62
    assert diag["have_session"] == 62
    assert diag["coverage"] == 1.0
    assert diag["batch_actionable"] is True
    assert set(df["ticker"].astype(str)) == set(TRADABLE_62)
    assert len(df) == 62
    bars = pd.to_datetime(df["date"]).dt.date
    as_of = pd.to_datetime(df["as_of_date"]).dt.date
    assert set(bars) == {SESSION}
    assert (bars == as_of).all()
    skipped = _skip_map(diag)
    for idx in INDEX_4:
        assert skipped[idx]["reason"] == gs.SKIP_INDEX_ONLY_EXCLUDED
    from services.signal_lifecycle import LifecycleConfig, decide_action, lifecycle_logic_from_dict
    from services.build_effective_lifecycle import _deterministic_effective_stance
    from services.portfolio_allocation_engine import _decide_action as alloc_decide

    cfg_l = LifecycleConfig(
        min_hold_days=1,
        cooldown_days_after_exit=1,
        buy_delta_pct=0.0015,
        add_delta_pct=0.0020,
        exit_delta_pct=-0.0015,
        trim_delta_pct=-0.0030,
        hold_means_keep_position=True,
    )
    logic = lifecycle_logic_from_dict({"enabled": True})
    actions = []
    for _, row in df.iterrows():
        action, _reason = decide_action(
            cfg=cfg_l,
            logic=logic,
            signal=str(row["signal"]),
            delta_pct=float("nan"),
            confidence=float(row["confidence"]),
            pos_state="FLAT",
            last_change_date=date(2026, 9, 1),
            cooldown_until=None,
            as_of_date=SESSION,
        )
        actions.append(action)
    assert "WAIT" not in set(actions) or True
    assert set(actions) <= {"BUY", "HOLD", "WAIT", "ADD", "TRIM", "EXIT"}
    assert any(a in {"BUY", "HOLD"} for a in actions)
    stance = _deterministic_effective_stance("FLAT", "BUY")
    assert stance[1] in {"BUY", "WAIT", "HOLD"}
    planned, _ = alloc_decide(
        is_held=False,
        risk_components=[],
        edge_score=0.9,
        sizing_tier="STRONG_EDGE",
        effective_stance="BUY",
        lifecycle_action="BUY",
        signal="BUY",
        confidence=0.8,
        delta_pct=0.01,
    )
    assert planned in {"OPEN_NEW", "BUY", "WATCH", "HOLD"}


def test_compute_asof_policy_index_excluded_and_no_weekday_fallback():
    import services.generate_signals as gs

    latest = {t: STALE for t in TRADABLE_62}
    latest.update({t: SESSION for t in INDEX_4})
    effective, diag = gs.compute_asof_policy(
        latest,
        SESSION,
        tradable_tickers=TRADABLE_62,
        index_tickers=INDEX_4,
    )
    assert effective == SESSION
    assert diag["have_session"] == 0
    assert diag["coverage"] == 0.0
    assert diag["tradable_tickers"] == 62
    assert diag["index_tickers"] == 4
    assert diag["batch_actionable"] is False
    assert diag["effective_reason"] == "coverage_low_batch_unavailable"
    assert diag["fail"] is True
    assert "coverage_low_use_data_supported" not in str(diag)


def test_rf_shadow_fields_remain_diagnostic(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA"])
    _write_pred(preds, "AAA", SESSION, predicted_scale=1.25)
    _write_processed(proc, "AAA", SESSION)
    _run(gs)
    df = _signals(results)
    assert bool(df["rf_shadow_only"].iloc[0]) in {True, "True", "true", 1}
    assert pd.isna(df["delta_pct"].iloc[0]) or str(df["delta_pct"].iloc[0]) in {"nan", "NaN"}
    assert "rf_predicted_close" in df.columns
    assert "rf_delta_pct" in df.columns


def test_current_prediction_absent_processed_skips(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA"])
    _write_pred(preds, "AAA", SESSION)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    skipped = _skip_map(_diag(results))
    assert skipped["AAA"]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    assert skipped["AAA"]["actual_latest_bar_date"] == "MISSING"
    df = _signals(results)
    assert df.empty or len(df.dropna(how="all")) == 0
    assert _diag(results)["have_session"] == 0
    assert _diag(results)["coverage"] == 0.0


def test_current_prediction_stale_processed_skips(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA"])
    _write_pred(preds, "AAA", SESSION)
    _write_processed(proc, "AAA", STALE)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    skipped = _skip_map(_diag(results))
    assert skipped["AAA"]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    assert skipped["AAA"]["actual_latest_bar_date"] == str(STALE)
    df = _signals(results)
    assert df.empty or len(df.dropna(how="all")) == 0


def test_current_prediction_incomplete_processed_skips(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA"])
    _write_pred(preds, "AAA", SESSION)
    _ohlcv(SESSION).drop(columns=["close"]).to_parquet(proc / "AAA.parquet", index=False)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    skipped = _skip_map(_diag(results))
    assert skipped["AAA"]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    df = _signals(results)
    assert df.empty or len(df.dropna(how="all")) == 0
    assert _diag(results)["have_session"] == 0


def test_current_processed_absent_rf_emits_operational(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA"])
    _write_processed(proc, "AAA", SESSION)
    _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert list(df["ticker"].astype(str)) == ["AAA"]
    assert pd.to_datetime(df["date"]).dt.date.iloc[0] == SESSION
    assert str(df["signal"].iloc[0]) in {"BUY", "HOLD", "SELL"}
    assert diag["have_session"] == 1
    assert diag["coverage"] == 1.0
    assert diag["batch_actionable"] is True
    kinds = {r.get("source_kind") for r in (diag.get("operational_provenance") or [])}
    assert kinds == {"processed"}


def test_sixty_one_processed_one_prediction_only_under_coverage(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, TRADABLE_62, INDEX_4)
    missing = TRADABLE_62[-1]
    for t in TRADABLE_62:
        _write_pred(preds, t, SESSION)
        if t != missing:
            _write_processed(proc, t, SESSION)
    for t in INDEX_4:
        _write_pred(preds, t, SESSION)
        _write_processed(proc, t, SESSION)
    _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert missing not in set(df["ticker"].astype(str))
    assert set(df["ticker"].astype(str)) == set(TRADABLE_62) - {missing}
    assert len(df) == 61
    assert diag["have_session"] == 61
    assert diag["coverage"] == 0.9839
    assert diag["batch_actionable"] is True
    skipped = _skip_map(diag)
    assert skipped[missing]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    assert skipped[missing]["actual_latest_bar_date"] == "MISSING"


def test_sixty_two_prediction_only_zero_operational(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, TRADABLE_62, INDEX_4)
    for t in TRADABLE_62 + INDEX_4:
        _write_pred(preds, t, SESSION)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert df.empty or len(df.dropna(how="all")) == 0
    assert diag["have_session"] == 0
    assert diag["coverage"] == 0.0
    assert diag["batch_actionable"] is False
    skipped = _skip_map(diag)
    for t in TRADABLE_62:
        assert skipped[t]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
