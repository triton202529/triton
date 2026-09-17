"""Phase 198T-SF5 — processed-only operational signal path.

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
import shutil
import socket
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

_PROD_ROOT = Path(r"C:\Users\akimw\triton")
_REPO_ROOT = Path(__file__).resolve().parents[1]
if _REPO_ROOT.resolve() == _PROD_ROOT.resolve():
    raise RuntimeError("Phase 198T-SF5 tests must not run in the production repository.")
if "isolated" not in str(_REPO_ROOT).lower():
    raise RuntimeError("Phase 198T-SF5 tests must run from an isolated worktree.")

_SF5_TEMP = Path(r"C:\Users\akimw\AppData\Local\Temp\phase_198T_SF5_20260916_173028")
_SEP14_PROCESSED = _SF5_TEMP / "sep14_processed"
_FREEZE_TICKERS = _SF5_TEMP / "freeze" / "data" / "config" / "tickers.txt"
_SF1_PROCESSED = Path(
    r"C:\Users\akimw\AppData\Local\Temp\phase_198T_SF1_20260916_015019" r"\freeze\data\processed"
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
NOW_Z = "2026-09-15T15:00:00Z"
OPERATIONAL_COLS = [
    "ticker",
    "date",
    "as_of_date",
    "session_as_of_date",
    "close",
    "delta_pct",
    "signal",
    "confidence",
    "score",
    "momentum_score",
    "trend_score",
    "breakout_score",
    "volatility_score",
    "rationale",
]


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
def _phase_198t_sf5_safety(monkeypatch):
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
            raise RuntimeError("network/broker calls are blocked in Phase 198T-SF5 tests")

    def blocked_connect(*a, **k):
        raise RuntimeError("network/broker calls are blocked in Phase 198T-SF5 tests")

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
    body = "\n".join(list(tradable) + list(index)) + "\n"
    (cfg / "tickers.txt").write_text(body, encoding="utf-8")


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


def _pred_frame(
    ticker: str,
    last: date,
    rows: int = 30,
    *,
    predicted_scale: float = 1.01,
    extra_col: Optional[float] = None,
) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(last), periods=rows)
    close = np.linspace(50.0, 55.0, rows)
    data: Dict[str, Any] = {
        "date": dates,
        "close": close,
        "predicted_close": close * predicted_scale,
        "ticker": [ticker] * rows,
        "rsi": np.linspace(40.0, 60.0, rows),
    }
    if extra_col is not None:
        data["extra_feature"] = np.full(rows, extra_col)
    return pd.DataFrame(data)


def _write_pred(preds: Path, ticker: str, last: date, **kwargs) -> Path:
    path = preds / f"{ticker}_predictions.parquet"
    _pred_frame(ticker, last, **kwargs).to_parquet(path, index=False)
    return path


def _run(
    gs,
    *,
    session: date = SESSION,
    generated_at: str = NOW_Z,
    expect_error: Optional[type] = None,
):
    gs_mod_nyse = getattr(gs, "_nyse_last_completed_session_date")
    buf = io.StringIO()
    old_stdout = sys_stdout()
    try:
        import sys

        sys.stdout = buf
        gs._nyse_last_completed_session_date = lambda now_utc=None: session
        gs.utc_now_z = lambda: generated_at
        if expect_error is not None:
            with pytest.raises(expect_error):
                gs.main()
        else:
            gs.main()
    finally:
        import sys

        sys.stdout = old_stdout
        gs._nyse_last_completed_session_date = gs_mod_nyse
    return buf.getvalue()


def sys_stdout():
    import sys

    return sys.stdout


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


def _ops_frame(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in OPERATIONAL_COLS if c in df.columns]
    out = df[keep].copy()
    if "ticker" in out.columns:
        out = out.sort_values("ticker").reset_index(drop=True)
    return out


def _ops_sha(df: pd.DataFrame) -> str:
    return hashlib.sha256(_ops_frame(df).to_csv(index=False).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_pred_files_presence_gate_removed_from_source():
    src = (_REPO_ROOT / "services" / "generate_signals.py").read_text(encoding="utf-8")
    assert "if not pred_files" not in src
    assert "Prediction files are optional RF shadow evidence" in src


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
    _write_processed(proc, "AAA", SESSION)
    _write_processed(proc, "BBB", SESSION)
    _run(gs, session=SESSION)
    with pytest.raises(RuntimeError, match="network/broker"):
        socket.socket()


def test_absent_rf_keeps_current_processed_symbols(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = ["AAA", "BBB", "CCC"]
    _write_tickers(cfg, names)
    for t in names:
        _write_processed(proc, t, SESSION, trend="up")
    _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert list(df["ticker"].astype(str)) == names
    assert diag["batch_actionable"] is True
    assert diag["have_session"] == 3
    assert diag["coverage"] == 1.0
    assert df["rf_shadow_only"].astype(str).str.lower().isin(["true", "1"]).all()
    assert set(df["rf_evidence_status"].astype(str)) == {"unavailable"}
    prov = diag.get("operational_provenance") or []
    assert {r["ticker"] for r in prov} == set(names)
    for rec in prov:
        assert rec["input_date"] == str(SESSION)
        assert rec["source_kind"] == "processed"
        assert rec["source_sha256"]
        assert Path(rec["source_path"]).name.endswith(".parquet")


def test_partial_stale_malformed_and_changed_rf_do_not_change_ops(tmp_path, monkeypatch):
    names = ["AAA", "BBB", "CCC"]
    frames: Dict[str, pd.DataFrame] = {}
    diags: Dict[str, Dict[str, Any]] = {}

    def _once(tag: str, mutate) -> None:
        gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path / tag)
        _write_tickers(cfg, names)
        for t in names:
            _write_processed(proc, t, SESSION, trend="up")
        mutate(preds)
        _run(gs)
        frames[tag] = _signals(results)
        diags[tag] = _diag(results)

    _once("stale", lambda preds: [_write_pred(preds, t, STALE) for t in names])
    _once("absent", lambda preds: None)
    _once("partial", lambda preds: _write_pred(preds, "AAA", SESSION))

    def _malformed(preds: Path) -> None:
        for t in names:
            p = preds / f"{t}_predictions.parquet"
            p.write_bytes(b"not-a-parquet-file")

    _once("malformed", _malformed)

    def _changed(preds: Path) -> None:
        for t in names:
            _write_pred(preds, t, SESSION, predicted_scale=0.42, extra_col=99.0)

    _once("changed", _changed)

    base = _ops_sha(frames["stale"])
    for tag in ("absent", "partial", "malformed", "changed"):
        assert list(frames[tag]["ticker"].astype(str)) == names
        assert _ops_sha(frames[tag]) == base
        assert diags[tag]["have_session"] == 3
        assert diags[tag]["batch_actionable"] is True
        assert diags[tag]["coverage"] == 1.0
    assert set(frames["stale"]["rf_evidence_status"].astype(str)) == {"stale"}
    assert set(frames["absent"]["rf_evidence_status"].astype(str)) == {"unavailable"}
    assert (
        str(
            frames["partial"]
            .loc[frames["partial"]["ticker"] == "AAA", "rf_evidence_status"]
            .iloc[0]
        )
        == "valid"
    )
    assert set(
        frames["partial"]
        .loc[frames["partial"]["ticker"] != "AAA", "rf_evidence_status"]
        .astype(str)
    ) == {"unavailable"}
    assert set(frames["malformed"]["rf_evidence_status"].astype(str)) == {"unavailable"}
    assert set(frames["changed"]["rf_evidence_status"].astype(str)) == {"valid"}
    stale_rf = frames["stale"]["rf_predicted_close"].astype(float)
    changed_rf = frames["changed"]["rf_predicted_close"].astype(float)
    assert not stale_rf.equals(changed_rf)


def test_no_processed_files_zero_actionable_complete_skips(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = ["AAA", "BBB", "CCC"]
    _write_tickers(cfg, names)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert df.empty or "ticker" not in df.columns or len(df.dropna(how="all")) == 0
    assert diag["batch_actionable"] is False
    assert diag["have_session"] == 0
    skipped = _skip_map(diag)
    assert set(skipped) == set(names)
    assert len(skipped) == 3


def test_stale_processed_refuses_older_bar(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = [f"S{i}" for i in range(10)]
    _write_tickers(cfg, names)
    for t in names:
        _write_processed(proc, t, STALE)
        _write_pred(preds, t, STALE)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert df.empty or "ticker" not in df.columns or len(df.dropna(how="all")) == 0
    skipped = _skip_map(diag)
    assert len(skipped) == 10
    assert {r["reason"] for r in skipped.values()} == {gs.SKIP_MISSING_EXPECTED_SESSION_BAR}
    assert {r["actual_latest_bar_date"] for r in skipped.values()} == {str(STALE)}


def test_incomplete_processed_skips_without_older_bar(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = [f"P{i}" for i in range(10)]
    _write_tickers(cfg, names)
    for t in names[:-1]:
        _write_processed(proc, t, SESSION)
    bad = _ohlcv(SESSION).drop(columns=["close"])
    bad.to_parquet(proc / f"{names[-1]}.parquet", index=False)
    _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert set(df["ticker"].astype(str)) == set(names[:-1])
    skipped = _skip_map(diag)
    assert skipped[names[-1]]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    assert names[-1] not in set(df["ticker"].astype(str))


def test_exact_session_excludes_open_session_partial_bar(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA"])
    extra = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-09-15")],
            "open": [999.0],
            "high": [999.0],
            "low": [999.0],
            "close": [999.0],
            "volume": [1.0],
        }
    )
    _write_processed(proc, "AAA", SESSION, extra=extra)
    _run(gs)
    df = _signals(results)
    assert len(df) == 1
    assert pd.to_datetime(df.iloc[0]["date"]).date() == SESSION
    assert float(df.iloc[0]["close"]) != 999.0
    prov = _diag(results)["operational_provenance"]
    assert prov[0]["input_date"] == str(SESSION)


def test_future_rf_dates_do_not_drop_current_processed(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA", "BBB"])
    for t in ["AAA", "BBB"]:
        _write_processed(proc, t, SESSION)
        dates = pd.bdate_range(end=pd.Timestamp("2026-09-18"), periods=5)
        close = np.linspace(10, 12, 5)
        pd.DataFrame(
            {
                "date": dates,
                "close": close,
                "predicted_close": close * 3.0,
                "ticker": t,
            }
        ).to_parquet(preds / f"{t}_predictions.parquet", index=False)
    _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert set(df["ticker"].astype(str)) == {"AAA", "BBB"}
    assert diag["have_session"] == 2
    assert diag["batch_actionable"] is True


def test_index_only_never_inflates_tradable_coverage(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    tradable = [f"T{i:02d}" for i in range(10)]
    _write_tickers(cfg, tradable, INDEX_4)
    for t in tradable:
        _write_processed(proc, t, SESSION)
    for t in INDEX_4:
        _write_processed(proc, t, SESSION)
    _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert diag["tradable_tickers"] == 10
    assert diag["have_session"] == 10
    assert diag["coverage"] == 1.0
    assert set(df["ticker"].astype(str)) == set(tradable)
    skipped = _skip_map(diag)
    for idx in INDEX_4:
        assert skipped[idx]["reason"] == gs.SKIP_INDEX_ONLY_EXCLUDED
        assert skipped[idx]["class"] == "index"


def test_coverage_threshold_still_zero_nine(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = [f"C{i}" for i in range(10)]
    _write_tickers(cfg, names)
    for t in names[:-1]:
        _write_processed(proc, t, SESSION)
    _write_processed(proc, names[-1], STALE)
    _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert abs(diag["coverage"] - 0.9) < 1e-9
    assert diag["batch_actionable"] is True
    assert set(df["ticker"].astype(str)) == set(names[:-1])
    skipped = _skip_map(diag)
    assert skipped[names[-1]]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR


def test_deterministic_replay(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = ["AAA", "BBB"]
    _write_tickers(cfg, names)
    for t in names:
        _write_processed(proc, t, SESSION)
    _run(gs, generated_at=NOW_Z)
    a = (results / "signals_with_rationale.csv").read_bytes()
    da = (results / "signals_asof_diagnostics.json").read_text(encoding="utf-8")
    _run(gs, generated_at=NOW_Z)
    b = (results / "signals_with_rationale.csv").read_bytes()
    db = (results / "signals_asof_diagnostics.json").read_text(encoding="utf-8")
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
    ja = json.loads(da)
    jb = json.loads(db)
    assert ja["per_symbol_skips"] == jb["per_symbol_skips"]
    assert ja["operational_provenance"] == jb["operational_provenance"]


def test_current_processed_sixty_two_without_rf(tmp_path, monkeypatch):
    if not _SEP14_PROCESSED.is_dir() or not _FREEZE_TICKERS.is_file():
        pytest.skip("frozen SF5 sep14 processed tree missing")
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    shutil.copyfile(_FREEZE_TICKERS, cfg / "tickers.txt")
    for src in _SEP14_PROCESSED.glob("*.parquet"):
        shutil.copy2(src, proc / src.name)
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
    assert set(bars) == {SESSION}
    skipped = _skip_map(diag)
    for idx in INDEX_4:
        assert skipped[idx]["reason"] == gs.SKIP_INDEX_ONLY_EXCLUDED
    prov = {r["ticker"]: r for r in diag.get("operational_provenance") or []}
    assert set(prov) == set(TRADABLE_62)
    for ticker, rec in prov.items():
        assert rec["input_date"] == str(SESSION)
        assert rec["source_kind"] == "processed"
        assert rec["source_sha256"] == _file_sha(proc / f"{ticker}.parquet")
        assert rec["source_sha256"] != ""


def test_sf1_stale_processed_sixty_two_skips(tmp_path, monkeypatch):
    src_dir = _SF5_TEMP / "freeze" / "data" / "processed"
    if not src_dir.is_dir():
        src_dir = _SF1_PROCESSED
    if not src_dir.is_dir() or not _FREEZE_TICKERS.is_file():
        pytest.skip("frozen SF1 processed tree missing")
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    shutil.copyfile(_FREEZE_TICKERS, cfg / "tickers.txt")
    for src in src_dir.glob("*.parquet"):
        shutil.copy2(src, proc / src.name)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert df.empty or "ticker" not in df.columns or len(df.dropna(how="all")) == 0
    skipped = _skip_map(diag)
    tradable_skips = [
        r for r in skipped.values() if r.get("class") != "index" and r["symbol"] in TRADABLE_62
    ]
    assert len(tradable_skips) == 62
    assert {r["reason"] for r in tradable_skips} == {gs.SKIP_MISSING_EXPECTED_SESSION_BAR}


def test_downstream_mocked_path_and_risk_controls(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["BUY1", "HOLD1"])
    _write_processed(proc, "BUY1", SESSION, trend="up")
    _write_processed(proc, "HOLD1", SESSION, trend="flat")
    _run(gs)
    df = _signals(results)
    buy_row = df[df["ticker"] == "BUY1"].iloc[0]
    from services.signal_lifecycle import LifecycleConfig, decide_action, lifecycle_logic_from_dict
    from services.build_effective_lifecycle import _deterministic_effective_stance
    from services.performance_risk_overlay import _classify_row
    from services.portfolio_allocation_engine import _decide_action as alloc_decide
    from services.manage_positions import _max_add_qty_respecting_weight, _compute_weak_signal_score
    from services.risk_watchdog import _score_concentration, _score_exposure
    from services.paper_sell_arm_authority import allowed_transition, SellArmState

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
    life_action, life_reason = decide_action(
        cfg=cfg_l,
        logic=logic,
        signal=str(buy_row["signal"]),
        delta_pct=float("nan"),
        confidence=float(buy_row["confidence"]),
        pos_state="FLAT",
        last_change_date=date(2026, 9, 1),
        cooldown_until=None,
        as_of_date=SESSION,
    )
    assert life_action in {"BUY", "HOLD", "WAIT"}
    assert life_reason
    _pos, stance, _tag = _deterministic_effective_stance("FLAT", life_action)
    assert stance in {"BUY", "WAIT", "HOLD"}
    force = _classify_row(
        total_pl=0.0,
        unrealized_pl=-250.0,
        drag_flag=False,
        win_rate=None,
        has_win_rate=False,
    )
    assert "FORCE_EXIT" in str(force)
    force_action, force_reason = alloc_decide(
        is_held=True,
        risk_components=["FORCE_EXIT"],
        edge_score=0.9,
        sizing_tier="STRONG_EDGE",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="HOLD",
        confidence=0.8,
        delta_pct=float("nan"),
    )
    assert force_action == "EXIT"
    assert "FORCE_EXIT" in force_reason
    trim_action, trim_reason = alloc_decide(
        is_held=True,
        risk_components=["TRIM_PRIORITY"],
        edge_score=0.9,
        sizing_tier="STRONG_EDGE",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="HOLD",
        confidence=0.8,
        delta_pct=float("nan"),
    )
    assert trim_action == "TRIM"
    assert "TRIM_PRIORITY" in trim_reason
    block_action, block_reason = alloc_decide(
        is_held=False,
        risk_components=["BLOCK_NEW_BUY"],
        edge_score=0.9,
        sizing_tier="STRONG_EDGE",
        effective_stance="BUY",
        lifecycle_action="BUY",
        signal="BUY",
        confidence=0.8,
        delta_pct=float("nan"),
    )
    assert block_action == "BLOCK"
    assert "BLOCK_NEW_BUY" in block_reason
    planned, plan_reason = alloc_decide(
        is_held=False,
        risk_components=[],
        edge_score=0.9,
        sizing_tier="STRONG_EDGE",
        effective_stance="BUY",
        lifecycle_action="BUY",
        signal="BUY",
        confidence=float(buy_row["confidence"]),
        delta_pct=float("nan"),
    )
    assert planned in {"OPEN_NEW", "WATCH", "HOLD", "BLOCK"}
    assert plan_reason
    clipped = _max_add_qty_respecting_weight(
        "BUY1",
        100,
        50.0,
        {"BUY1": 10.0},
        {"BUY1": {"current_price": 50.0, "qty": 10.0}},
        {"BUY1": 500.0},
        1000.0,
        0.10,
    )
    assert clipped == 0
    weak_a = _compute_weak_signal_score(-0.05, 0.8)
    weak_b = _compute_weak_signal_score(0.05, 0.8)
    assert weak_a == weak_b
    conc = _score_concentration(
        [{"market_value": 800.0}, {"market_value": 200.0}],
        {},
        concentration_threshold_pct=25.0,
    )
    conc_ok = _score_concentration(
        [{"market_value": 200.0}, {"market_value": 200.0}],
        {},
        concentration_threshold_pct=25.0,
    )
    assert conc < conc_ok
    exposed = _score_exposure(
        [{"market_value": 900.0}],
        {"equity": 1000.0, "cash": 100.0},
    )
    cashy = _score_exposure(
        [{"market_value": 100.0}],
        {"equity": 1000.0, "cash": 900.0},
    )
    assert isinstance(exposed, int) and isinstance(cashy, int)
    assert allowed_transition(SellArmState.NO_SELLS_WAITING, SellArmState.ARMED) is False
    assert (
        allowed_transition(SellArmState.CONFIRMATION_CHALLENGE_ISSUED, SellArmState.ARMED) is True
    )
    with pytest.raises(RuntimeError, match="network/broker"):
        socket.socket()


def test_no_forward_fill_or_relabeled_bar(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, ["AAA"])
    df = _ohlcv(SESSION)
    prior_close = float(df.iloc[-2]["close"])
    session_close = float(df.iloc[-1]["close"])
    extra = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-09-15")],
            "open": [prior_close],
            "high": [prior_close],
            "low": [prior_close],
            "close": [prior_close],
            "volume": [1.0],
        }
    )
    out_df = pd.concat([df, extra], ignore_index=True)
    out_df.to_parquet(proc / "AAA.parquet", index=False)
    _run(gs)
    out = _signals(results)
    assert len(out) == 1
    assert pd.to_datetime(out.iloc[0]["date"]).date() == SESSION
    assert pd.to_datetime(out.iloc[0]["as_of_date"]).date() == SESSION
    assert float(out.iloc[0]["close"]) == pytest.approx(session_close)
    assert float(out.iloc[0]["close"]) != pytest.approx(prior_close)


SF5_ABSENT_RF_OPS_SHA = "21e3cda733b2db4cc8cabc417acf45f494a7ffee14541fd4d79ff193fff867f6"


def test_prediction_ohlcv_fallback_removed_from_operational_path():
    src = (_REPO_ROOT / "services" / "generate_signals.py").read_text(encoding="utf-8")
    assert "prediction_ohlcv_fallback" not in src
    assert "SF2 prediction-OHLCV fallback" not in src
    assert "Prediction OHLCV is never operational" in src


def test_current_pred_without_processed_does_not_emit(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = ["AAA", "BBB", "CCC"]
    _write_tickers(cfg, names)
    for t in names:
        _write_pred(preds, t, SESSION)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert df.empty or "ticker" not in df.columns or len(df.dropna(how="all")) == 0
    assert diag["have_session"] == 0
    assert diag["coverage"] == 0.0
    assert diag["batch_actionable"] is False
    skipped = _skip_map(diag)
    assert set(skipped) == set(names)
    assert {r["reason"] for r in skipped.values()} == {gs.SKIP_MISSING_EXPECTED_SESSION_BAR}
    assert {r["actual_latest_bar_date"] for r in skipped.values()} == {"MISSING"}
    assert not (diag.get("operational_provenance") or [])


def test_current_pred_stale_processed_does_not_emit(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = ["AAA", "BBB", "CCC"]
    _write_tickers(cfg, names)
    for t in names:
        _write_processed(proc, t, STALE)
        _write_pred(preds, t, SESSION)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert df.empty or "ticker" not in df.columns or len(df.dropna(how="all")) == 0
    assert diag["have_session"] == 0
    skipped = _skip_map(diag)
    assert set(skipped) == set(names)
    assert {r["reason"] for r in skipped.values()} == {gs.SKIP_MISSING_EXPECTED_SESSION_BAR}
    assert {r["actual_latest_bar_date"] for r in skipped.values()} == {str(STALE)}


def test_current_pred_incomplete_processed_does_not_emit(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    names = [f"P{i}" for i in range(10)]
    _write_tickers(cfg, names)
    for t in names[:-1]:
        _write_processed(proc, t, SESSION)
        _write_pred(preds, t, SESSION)
    bad = _ohlcv(SESSION).drop(columns=["close"])
    bad.to_parquet(proc / f"{names[-1]}.parquet", index=False)
    _write_pred(preds, names[-1], SESSION)
    _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert names[-1] not in set(df["ticker"].astype(str))
    assert set(df["ticker"].astype(str)) == set(names[:-1])
    assert diag["have_session"] == 9
    assert abs(diag["coverage"] - 0.9) < 1e-9
    skipped = _skip_map(diag)
    assert skipped[names[-1]]["reason"] == gs.SKIP_MISSING_EXPECTED_SESSION_BAR
    kinds = {r.get("source_kind") for r in (diag.get("operational_provenance") or [])}
    assert kinds == {"processed"}


def test_sixty_two_current_pred_no_processed_zero_coverage(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, TRADABLE_62, INDEX_4)
    for t in TRADABLE_62 + INDEX_4:
        _write_pred(preds, t, SESSION)
    with pytest.raises(RuntimeError, match="AS_OF invariant violated"):
        _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert df.empty or "ticker" not in df.columns or len(df.dropna(how="all")) == 0
    assert diag["tradable_tickers"] == 62
    assert diag["have_session"] == 0
    assert diag["coverage"] == 0.0
    skipped = _skip_map(diag)
    tradable_skips = [skipped[t] for t in TRADABLE_62]
    assert len(tradable_skips) == 62
    assert {r["reason"] for r in tradable_skips} == {gs.SKIP_MISSING_EXPECTED_SESSION_BAR}


def test_sixty_one_processed_one_pred_only_skips_under_coverage(tmp_path, monkeypatch):
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    _write_tickers(cfg, TRADABLE_62, INDEX_4)
    missing = TRADABLE_62[-1]
    for t in TRADABLE_62:
        _write_pred(preds, t, SESSION)
        if t != missing:
            _write_processed(proc, t, SESSION, trend="up")
    for t in INDEX_4:
        _write_processed(proc, t, SESSION)
        _write_pred(preds, t, SESSION)
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
    kinds = {r.get("source_kind") for r in (diag.get("operational_provenance") or [])}
    assert kinds == {"processed"}


def test_sixty_two_processed_absent_rf_matches_sf5_operational_sha(tmp_path, monkeypatch):
    if not _SEP14_PROCESSED.is_dir() or not _FREEZE_TICKERS.is_file():
        pytest.skip("frozen SF5 sep14 processed tree missing")
    gs, results, preds, proc, cfg = _bind(monkeypatch, tmp_path)
    shutil.copyfile(_FREEZE_TICKERS, cfg / "tickers.txt")
    for src in _SEP14_PROCESSED.glob("*.parquet"):
        shutil.copy2(src, proc / src.name)
    _run(gs)
    df = _signals(results)
    diag = _diag(results)
    assert len(df) == 62
    assert diag["have_session"] == 62
    assert diag["coverage"] == 1.0
    assert _ops_sha(df) == SF5_ABSENT_RF_OPS_SHA
    kinds = {r.get("source_kind") for r in (diag.get("operational_provenance") or [])}
    assert kinds == {"processed"}
    assert set(df["rf_evidence_status"].astype(str)) == {"unavailable"}
