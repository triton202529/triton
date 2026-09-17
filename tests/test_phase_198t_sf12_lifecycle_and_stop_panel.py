"""Phase 198T-SF12 — lifecycle coverage gate and Stop-panel presentation.

Deterministic. Injected disposable paths only. No broker, network,
production artifact mutation, process start, or scheduled-task edits.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

_PROD_ROOT = Path(r"C:\Users\akimw\triton")
_REPO_ROOT = Path(__file__).resolve().parents[1]
if _REPO_ROOT.resolve() == _PROD_ROOT.resolve():
    raise RuntimeError("Phase 198T-SF12 tests must not run in the production repository.")
if "isolated" not in str(_REPO_ROOT).lower():
    raise RuntimeError("Phase 198T-SF12 tests must run from an isolated worktree.")

SESSION = "2026-09-16"
SKIPPED = ("XLE", "BLK", "C")
INDEXES = ("^DJI", "^GSPC", "^IXIC", "^VIX")
SF11_SIGNAL_SHA = "27ac492ee23dade7bf4221320cd123bb5cfb436153eef57cd21e9766ae941ae3"
SF11_DIAG_SHA = "44335d7b8991f253cbf8f50807fe706106e9849138684813c5985f33526b5201"
TEMP_ROOT = Path(r"C:\Users\akimw\AppData\Local\Temp\phase_198T_SF12_20260917_133100")
FROZEN_DIR = TEMP_ROOT / "frozen_sf11_inputs"


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
def _phase_198t_sf12_safety(monkeypatch):
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
            raise RuntimeError("network/broker calls are blocked in Phase 198T-SF12 tests")

    def blocked_connect(*a, **k):
        raise RuntimeError("network/broker calls are blocked in Phase 198T-SF12 tests")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "replace", guarded_replace)
    monkeypatch.setattr(os, "remove", guarded_remove)
    monkeypatch.setattr(os, "rename", guarded_rename)
    monkeypatch.setattr(socket, "socket", BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", blocked_connect)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    monkeypatch.setenv("PAPER_ONLY", "1")
    monkeypatch.delenv("TRITON_LIFECYCLE_MIN_ROWS", raising=False)
    monkeypatch.delenv("TRITON_LIFECYCLE_ALLOW_PARTIAL", raising=False)
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    yield


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _skip(
    symbol: str,
    *,
    session: str = SESSION,
    sha: Optional[str] = None,
    reason: str = "INVALID_PROCESSED_SESSION_BAR",
    cls: str = "tradable",
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "expected_session_date": session,
        "actual_latest_bar_date": session,
        "reason": reason,
        "source_sha256": sha or ("a" * 64),
        "evidence_timestamp": "2026-09-17T14:56:51Z",
        "class": cls,
    }


def _diag_doc(
    *,
    tradable: int = 62,
    have_session: int = 59,
    skips: Optional[List[Dict[str, Any]]] = None,
    session: str = SESSION,
    fail: bool = False,
    batch_actionable: bool = True,
    threshold: float = 0.90,
) -> Dict[str, Any]:
    verified = skips
    if verified is None:
        verified = [_skip(s) for s in SKIPPED]
        verified.extend(_skip(s, reason="INDEX_ONLY_EXCLUDED", cls="index") for s in INDEXES)
    coverage = round(have_session / float(tradable), 4) if tradable else 0.0
    return {
        "session_asof_date": session,
        "tickers": tradable,
        "tradable_tickers": tradable,
        "have_session": have_session,
        "coverage": coverage,
        "asof_coverage_threshold": threshold,
        "batch_actionable": batch_actionable,
        "fail": fail,
        "per_symbol_skips": verified,
        "index_tickers": 4,
    }


def _bind_guard(monkeypatch, root: Path):
    import services.lifecycle_artifact_guard as lag

    results = root / "data" / "results"
    cfg = root / "data" / "config"
    results.mkdir(parents=True, exist_ok=True)
    cfg.mkdir(parents=True, exist_ok=True)
    tickers = cfg / "tickers.txt"
    if not tickers.is_file():
        tickers.write_text(
            "\n".join([f"T{i:02d}" for i in range(62)] + list(INDEXES)) + "\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(lag, "RESULTS_DIR", results)
    monkeypatch.setattr(lag, "ASOF_DIAGNOSTICS_JSON", results / "signals_asof_diagnostics.json")
    monkeypatch.setattr(lag, "POSITIONS_SNAPSHOT_PATH", results / "positions_snapshot.csv")
    monkeypatch.setattr(lag, "TICKERS_FILE", tickers)
    monkeypatch.setattr(lag, "RAW_LIFECYCLE_PATH", results / "signal_lifecycle.csv")
    monkeypatch.setattr(lag, "EFFECTIVE_LIFECYCLE_PATH", results / "signal_lifecycle_effective.csv")
    monkeypatch.setattr(
        lag, "DIAGNOSTICS_JSON", results / "lifecycle_artifact_write_diagnostics.json"
    )
    monkeypatch.setattr(lag, "BACKUPS_DIR", results / "backups")
    return lag, results


def _write_json(path: Path, doc: Dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _signals_df(tickers: List[str], *, session: str = SESSION) -> pd.DataFrame:
    rows = []
    for t in tickers:
        rows.append(
            {
                "date": session,
                "as_of_date": session,
                "session_as_of_date": session,
                "ticker": t,
                "close": 10.0,
                "predicted_close": 10.1,
                "delta_pct": 0.01,
                "signal": "HOLD",
                "confidence": 0.6,
                "confidence_score": 0.6,
                "score": 0.6,
            }
        )
    return pd.DataFrame(rows)


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
    import services.lifecycle_artifact_guard as lag
    import dashboard.operator_console as oc

    assert lag.COMMITTED_ASOF_COVERAGE == 0.90
    assert callable(oc._render_stop_control_column)
    assert writes == []


def test_accepts_59_when_coverage_and_dated_skips_verify(tmp_path, monkeypatch):
    lag, results = _bind_guard(monkeypatch, tmp_path)
    _write_json(results / "signals_asof_diagnostics.json", _diag_doc())
    present = [f"T{i:02d}" for i in range(59)]
    df = _signals_df(present)
    gate = lag.evaluate_lifecycle_source_gate(df)
    assert gate["ok"] is True
    assert gate["min_rows"] == 59
    assert lag.minimum_required_rows() == 59
    names = {r["symbol"] for r in gate["details"]["coverage"]["verified_invalid_bar_skips"]}
    assert names == set(SKIPPED)
    assert gate["details"]["coverage"]["index_exclusions"] == 4
    assert gate["details"]["coverage"]["coverage"] == pytest.approx(0.9516)


def test_rejects_missing_skip_evidence(tmp_path, monkeypatch):
    lag, results = _bind_guard(monkeypatch, tmp_path)
    present = [f"T{i:02d}" for i in range(59)]
    gate = lag.evaluate_lifecycle_source_gate(_signals_df(present))
    assert gate["ok"] is False
    assert gate["reason"] == "missing_skip_evidence"


def test_rejects_forged_skip_without_sha(tmp_path, monkeypatch):
    lag, results = _bind_guard(monkeypatch, tmp_path)
    doc = _diag_doc(skips=[_skip("XLE", sha="not-a-hash")] + [_skip("BLK"), _skip("C")])
    _write_json(results / "signals_asof_diagnostics.json", doc)
    gate = lag.evaluate_lifecycle_source_gate(_signals_df([f"T{i:02d}" for i in range(59)]))
    assert gate["ok"] is False
    assert gate["reason"] == "forged_skip_evidence"


def test_rejects_skip_symbol_still_in_source(tmp_path, monkeypatch):
    lag, results = _bind_guard(monkeypatch, tmp_path)
    _write_json(results / "signals_asof_diagnostics.json", _diag_doc())
    present = [f"T{i:02d}" for i in range(58)] + ["XLE"]
    gate = lag.evaluate_lifecycle_source_gate(_signals_df(present))
    assert gate["ok"] is False
    assert gate["reason"] == "forged_skip_evidence"


def test_rejects_duplicate_symbols(tmp_path, monkeypatch):
    lag, results = _bind_guard(monkeypatch, tmp_path)
    _write_json(results / "signals_asof_diagnostics.json", _diag_doc())
    present = [f"T{i:02d}" for i in range(58)] + ["T00"]
    gate = lag.evaluate_lifecycle_source_gate(_signals_df(present))
    assert gate["ok"] is False
    assert gate["reason"] == "duplicate_symbols"


def test_rejects_stale_september_15_inputs(tmp_path, monkeypatch):
    lag, results = _bind_guard(monkeypatch, tmp_path)
    _write_json(results / "signals_asof_diagnostics.json", _diag_doc())
    df = _signals_df([f"T{i:02d}" for i in range(59)], session="2026-09-15")
    gate = lag.evaluate_lifecycle_source_gate(df)
    assert gate["ok"] is False
    assert gate["reason"] == "stale_lifecycle_inputs"


def test_rejects_unknown_session(tmp_path, monkeypatch):
    lag, results = _bind_guard(monkeypatch, tmp_path)
    doc = _diag_doc()
    doc["session_asof_date"] = "UNKNOWN"
    _write_json(results / "signals_asof_diagnostics.json", doc)
    gate = lag.evaluate_lifecycle_source_gate(_signals_df([f"T{i:02d}" for i in range(59)]))
    assert gate["ok"] is False
    assert gate["reason"] == "unknown_session"


def test_rejects_coverage_below_threshold(tmp_path, monkeypatch):
    lag, results = _bind_guard(monkeypatch, tmp_path)
    extra = [_skip(f"Z{i}") for i in range(8)]
    skips = [_skip(s) for s in SKIPPED] + extra
    doc = _diag_doc(have_session=51, skips=skips, fail=True, batch_actionable=False)
    _write_json(results / "signals_asof_diagnostics.json", doc)
    gate = lag.evaluate_lifecycle_source_gate(_signals_df([f"T{i:02d}" for i in range(51)]))
    assert gate["ok"] is False
    assert gate["reason"] == "coverage_below_threshold"


def test_skipped_held_is_deferred_not_hold_or_sell(tmp_path, monkeypatch):
    lag, results = _bind_guard(monkeypatch, tmp_path)
    _write_json(results / "signals_asof_diagnostics.json", _diag_doc())
    pd.DataFrame([{"ticker": "XLE", "symbol": "XLE", "qty": 12, "market_value": 100.0}]).to_csv(
        results / "positions_snapshot.csv", index=False
    )
    present = [f"T{i:02d}" for i in range(59)]
    gate = lag.evaluate_lifecycle_source_gate(_signals_df(present))
    assert gate["ok"] is True
    assert gate["details"]["deferred_skipped_held"] == ["XLE"]
    deferred = lag.deferred_skipped_held_tickers(
        source_tickers=present,
        held_tickers=["XLE"],
        verified_skip_symbols=list(SKIPPED),
    )
    assert deferred == ["XLE"]

    import services.apply_signal_lifecycle as asl

    monkeypatch.setattr(asl, "RESULTS_DIR", results)
    monkeypatch.setattr(asl, "OUT_LIFECYCLE", results / "signal_lifecycle.csv")
    monkeypatch.setattr(asl, "STATE_PATH", results / "signal_state.json")
    monkeypatch.setattr(asl, "POSITIONS_SNAPSHOT", results / "positions_snapshot.csv")
    src = results / "signals.csv"
    _signals_df(present).to_csv(src, index=False)
    (results / "signal_state.json").write_text("{}", encoding="utf-8")
    rc = asl._rebuild_lifecycle(src)
    assert rc == 0
    out = pd.read_csv(results / "signal_lifecycle.csv")
    ticks = set(out["ticker"].astype(str).str.upper())
    assert "XLE" not in ticks
    assert "BLK" not in ticks
    assert "C" not in ticks
    assert len(ticks) == 59


def test_force_exit_independent_risk_remains_active():
    from services.portfolio_allocation_engine import _decide_action

    action, reason = _decide_action(
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
    assert action == "EXIT"
    assert "FORCE_EXIT" in reason
    src = (_REPO_ROOT / "services" / "generate_signals.py").read_text(encoding="utf-8")
    assert 'ASOF_COVERAGE = float(os.getenv("TRITON_ASOF_COVERAGE", "0.90"))' in src
    env_src = (_REPO_ROOT / "services" / "triton_env_loader.py").read_text(encoding="utf-8")
    assert "TRITON_ENABLE_LIVE_EXECUTION" in env_src


def test_stop_panel_defines_oui_and_does_not_request_on_render():
    src = (_REPO_ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    stop = src[
        src.index("def _render_stop_control_column") : src.index(
            "def _render_restart_control_column"
        )
    ]
    assert "from dashboard import operator_ui as oui" in stop
    assert stop.count("execute_stop_triton(") == 1
    assert 'key="stop_triton_request"' in stop
    assert 'key="stop_triton_confirm"' in stop
    assert "CONFIRMATION_PHRASE" in stop
    assert "issue_stop_preflight" in stop
    assert "preflight_token" in stop
    tree = ast.parse(stop)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "ui" not in names or "oui" in names


def _patch_streamlit(monkeypatch, session: Dict[str, Any]):
    import dashboard.operator_console as oc

    class Session(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

    st_session = Session(session)
    monkeypatch.setattr(oc.st, "session_state", st_session)
    monkeypatch.setattr(oc.st, "button", lambda *a, **k: False)
    monkeypatch.setattr(oc.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(oc.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(oc.st, "warning", lambda *a, **k: None)
    monkeypatch.setattr(oc.st, "checkbox", lambda *a, **k: False)
    monkeypatch.setattr(oc.st, "text_input", lambda *a, **k: "")
    monkeypatch.setattr(oc.st, "rerun", lambda *a, **k: None)

    class _Col:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(oc.st, "columns", lambda n: (_Col(), _Col()))
    return oc


def test_stop_panel_running_status_render_does_not_request_stop(monkeypatch):
    oc = _patch_streamlit(
        monkeypatch,
        {
            "stop_confirm_pending": True,
            "stop_in_progress": False,
            "stop_preflight_expires_at": "2026-09-17T16:40:00Z",
            "stop_preflight_decision": "STOP_ELIGIBLE_BOTH",
            "stop_preflight_targets": ["continuous", "watchdog"],
            "stop_preflight_token": "token",
            "stop_request_nonce": "nonce",
        },
    )
    import services.operator_stop_action as osa

    monkeypatch.setattr(
        osa,
        "execute_stop_triton",
        lambda **k: (_ for _ in ()).throw(AssertionError("page rendering must not request Stop")),
    )
    monkeypatch.setattr(
        osa,
        "issue_stop_preflight",
        lambda **k: (_ for _ in ()).throw(
            AssertionError("page rendering must not issue preflight")
        ),
    )
    oc._render_stop_control_column(
        {"decision": "STOP_ELIGIBLE_BOTH", "stop_button_enabled": True},
        start_busy=False,
    )


def test_stop_panel_already_stopped_status_render_does_not_request_stop(monkeypatch):
    oc = _patch_streamlit(monkeypatch, {"stop_confirm_pending": False, "stop_in_progress": False})
    import services.operator_stop_action as osa

    monkeypatch.setattr(
        osa,
        "execute_stop_triton",
        lambda **k: (_ for _ in ()).throw(AssertionError("page rendering must not request Stop")),
    )
    oc._render_stop_control_column(
        {"decision": "STOP_ALREADY_STOPPED", "stop_button_enabled": False},
        start_busy=False,
    )


def _freeze_sf11_inputs() -> Path:
    import shutil

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    prod_results = _PROD_ROOT / "data" / "results"
    mapping = {
        "signals.csv": prod_results / "signals.csv",
        "signals_with_rationale.csv": prod_results / "signals_with_rationale.csv",
        "signals_asof_diagnostics.json": prod_results / "signals_asof_diagnostics.json",
        "positions_snapshot.csv": prod_results / "positions_snapshot.csv",
        "lifecycle_reconciliation.csv": prod_results / "lifecycle_reconciliation.csv",
        "performance_intelligence_by_symbol.csv": prod_results
        / "performance_intelligence_by_symbol.csv",
        "edge_sizing.csv": prod_results / "edge_sizing.csv",
    }
    manifest: Dict[str, Any] = {
        "sf11_signal_sha": SF11_SIGNAL_SHA,
        "sf11_diag_sha": SF11_DIAG_SHA,
        "files": {},
    }
    for name, src in mapping.items():
        dest = FROZEN_DIR / name
        if src.is_file():
            shutil.copy2(src, dest)
            manifest["files"][name] = _sha256(dest)
        else:
            manifest["files"][name] = None
    (FROZEN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return FROZEN_DIR


def test_frozen_sf11_replay_lifecycle_risk_allocation(tmp_path, monkeypatch):
    lag, results = _bind_guard(monkeypatch, tmp_path)
    frozen = _freeze_sf11_inputs()
    frozen_signals = frozen / "signals.csv"
    frozen_diag = frozen / "signals_asof_diagnostics.json"
    import shutil

    shutil.copy2(frozen_signals, results / "signals.csv")
    shutil.copy2(frozen_diag, results / "signals_asof_diagnostics.json")
    diag = json.loads(frozen_diag.read_text(encoding="utf-8"))
    assert diag["session_asof_date"] == SESSION
    assert int(diag["have_session"]) == 59
    assert int(diag["tradable_tickers"]) == 62
    assert float(diag["coverage"]) >= 0.90
    skip_map = {str(r["symbol"]): r for r in diag.get("per_symbol_skips") or []}
    for sym in SKIPPED:
        assert skip_map[sym]["reason"] == "INVALID_PROCESSED_SESSION_BAR"
        assert skip_map[sym]["expected_session_date"] == SESSION
    assert sum(1 for r in skip_map.values() if r.get("reason") == "INDEX_ONLY_EXCLUDED") == 4
    for name in (
        "signals_with_rationale.csv",
        "positions_snapshot.csv",
        "lifecycle_reconciliation.csv",
        "performance_intelligence_by_symbol.csv",
        "edge_sizing.csv",
    ):
        src = frozen / name
        if src.is_file():
            shutil.copy2(src, results / name)
    if not (results / "positions_snapshot.csv").is_file():
        pd.DataFrame(columns=["ticker", "qty"]).to_csv(
            results / "positions_snapshot.csv", index=False
        )
    if not (results / "lifecycle_reconciliation.csv").is_file():
        pd.DataFrame(columns=["ticker", "mismatch_reason"]).to_csv(
            results / "lifecycle_reconciliation.csv", index=False
        )
    (results / "signal_state.json").write_text("{}", encoding="utf-8")
    (results / "open_orders_snapshot.csv").write_text("symbol,status,side\n", encoding="utf-8")

    import services.apply_signal_lifecycle as asl

    monkeypatch.setattr(asl, "RESULTS_DIR", results)
    monkeypatch.setattr(asl, "OUT_LIFECYCLE", results / "signal_lifecycle.csv")
    monkeypatch.setattr(asl, "STATE_PATH", results / "signal_state.json")
    monkeypatch.setattr(asl, "POSITIONS_SNAPSHOT", results / "positions_snapshot.csv")
    src_df = pd.read_csv(results / "signals.csv")
    assert src_df["ticker"].nunique() == 59
    assert src_df["as_of_date"].astype(str).str[:10].nunique() == 1
    assert str(src_df["as_of_date"].iloc[0])[:10] == SESSION
    for sym in SKIPPED:
        assert sym not in set(src_df["ticker"].astype(str).str.upper())

    rc = asl._rebuild_lifecycle(results / "signals.csv")
    assert rc == 0
    life = pd.read_csv(results / "signal_lifecycle.csv")
    life_ticks = set(life["ticker"].astype(str).str.upper())
    assert len(life_ticks) == 59
    for sym in SKIPPED:
        assert sym not in life_ticks
    assert "2026-09-15" not in set(life.get("as_of_date", pd.Series(dtype=str)).astype(str))

    import services.build_effective_lifecycle as bel
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_effective_lifecycle",
            "--lifecycle",
            str(results / "signal_lifecycle.csv"),
            "--recon",
            str(results / "lifecycle_reconciliation.csv"),
            "--out",
            str(results / "signal_lifecycle_effective.csv"),
            "--open-orders",
            str(results / "open_orders_snapshot.csv"),
        ],
    )
    assert bel.main() == 0
    eff = pd.read_csv(results / "signal_lifecycle_effective.csv")
    assert int(eff["ticker"].astype(str).str.upper().nunique()) == 59
    for sym in SKIPPED:
        assert sym not in set(eff["ticker"].astype(str).str.upper())

    import services.performance_risk_overlay as overlay

    overlay_rc = overlay.main(
        [
            "--input",
            str(results / "performance_intelligence_by_symbol.csv"),
            "--output",
            str(results / "performance_risk_overlay.csv"),
        ]
    )
    assert overlay_rc == 0
    overlay_df = pd.read_csv(results / "performance_risk_overlay.csv")
    assert not overlay_df.empty

    import services.portfolio_allocation_engine as pae

    alloc_args = [
        "--positions",
        str(results / "positions_snapshot.csv"),
        "--risk-overlay",
        str(results / "performance_risk_overlay.csv"),
        "--signals",
        str(results / "signals.csv"),
        "--lifecycle-effective",
        str(results / "signal_lifecycle_effective.csv"),
        "--out-csv",
        str(results / "portfolio_allocation_recommendations.csv"),
        "--out-json",
        str(results / "portfolio_allocation_recommendations.json"),
    ]
    if (results / "edge_sizing.csv").is_file():
        alloc_args.extend(["--edge-sizing", str(results / "edge_sizing.csv")])
    if (results / "performance_intelligence_by_symbol.csv").is_file():
        alloc_args.extend(
            ["--perf-by-symbol", str(results / "performance_intelligence_by_symbol.csv")]
        )
    assert pae.main(alloc_args) == 0
    alloc = pd.read_csv(results / "portfolio_allocation_recommendations.csv")
    alloc_ticks = (
        set(alloc["ticker"].astype(str).str.upper()) if "ticker" in alloc.columns else set()
    )
    if not alloc_ticks and "symbol" in alloc.columns:
        alloc_ticks = set(alloc["symbol"].astype(str).str.upper())
    for sym in SKIPPED:
        if sym in alloc_ticks:
            row = alloc[alloc.get("ticker", alloc.get("symbol")).astype(str).str.upper() == sym]
            actions = set(
                row.get("action", row.get("recommended_action", pd.Series(dtype=str)))
                .astype(str)
                .str.upper()
            )
            assert "SELL" not in actions

    from services.portfolio_allocation_engine import _decide_action

    planned = _decide_action(
        is_held=True,
        risk_components=["FORCE_EXIT"],
        edge_score=0.1,
        sizing_tier="WEAK",
        effective_stance="",
        lifecycle_action="",
        signal="",
        confidence=0.1,
        delta_pct=0.0,
    )
    assert planned[0] == "EXIT"

    summary = {
        "signal_rows": int(src_df["ticker"].nunique()),
        "lifecycle_rows": int(life["ticker"].nunique()),
        "effective_rows": int(eff["ticker"].nunique()),
        "skipped": list(SKIPPED),
        "index_exclusions": 4,
        "session": SESSION,
        "force_exit_active": True,
        "paper_only": True,
        "overlay_rows": int(len(overlay_df)),
        "allocation_rows": int(len(alloc)),
    }
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    (TEMP_ROOT / "replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert summary["signal_rows"] == 59
    assert summary["lifecycle_rows"] == 59
    assert summary["effective_rows"] == 59
