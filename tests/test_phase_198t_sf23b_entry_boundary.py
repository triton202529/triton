"""Phase 198T-SF23B — paper BUY/ADD boundary on execute_trades.

Isolated worktree only. Fake broker. No production writes or real submits.
"""

from __future__ import annotations

import builtins
import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

_PROD = Path(r"C:\Users\akimw\triton")
_REPO = Path(__file__).resolve().parents[1]
if _REPO.resolve() == _PROD.resolve():
    raise RuntimeError("SF23B tests must not run in the production repository.")
if "isolated" not in str(_REPO).lower():
    raise RuntimeError("SF23B tests must run from an isolated worktree.")

from services.edge_sizing_engine import (
    EDGE_PROVENANCE_SCHEMA,
    OPPS_PROVENANCE_SCHEMA,
    STATUS_VALID,
    provenance_sidecar_path,
    sha256_file,
    write_json_atomic,
)
from services.execute_trades import (
    REASON_EDGE_EVIDENCE_UNPROVEN,
    REASON_LIFECYCLE_STANCE_MISSING,
    REASON_LIFECYCLE_STANCE_NOT_ENTRY,
    apply_paper_buy_add_entry_boundary,
    main as execute_main,
    preview_planned_buy_symbols,
)
from services.portfolio_allocation_engine import _decide_action

_FORBIDDEN: List[str] = []
_NET: List[str] = []
_BROKER: List[str] = []


def _forbidden(path: Path) -> bool:
    return str(path.resolve()).lower().startswith(str((_PROD / "data").resolve()).lower())


@pytest.fixture(autouse=True)
def _block_prod_net(monkeypatch):
    _FORBIDDEN.clear()
    _NET.clear()
    _BROKER.clear()
    real_open = builtins.open

    def guarded(file, mode="r", *args, **kwargs):
        try:
            path = Path(file)
        except TypeError:
            return real_open(file, mode, *args, **kwargs)
        writing = any(ch in str(mode) for ch in "wxa+")
        if writing and _forbidden(path):
            _FORBIDDEN.append(str(path))
            raise RuntimeError(f"blocked production write: {path}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)

    def _net(*_a, **_k):
        _NET.append("socket")
        raise RuntimeError("network blocked")

    monkeypatch.setattr(socket.socket, "connect", _net)
    monkeypatch.setattr(socket, "create_connection", _net)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _bind_results(monkeypatch, tmp_path: Path) -> Path:
    results = tmp_path / "results"
    live = tmp_path / "live"
    results.mkdir()
    live.mkdir()
    import services.execute_trades as et

    monkeypatch.setattr(et, "RESULTS", results)
    monkeypatch.setattr(et, "LIVE", live)
    monkeypatch.setattr(et, "DEFAULT_OPPS", results / "trade_opportunities.csv")
    monkeypatch.setattr(
        et, "DEFAULT_ALLOC_RECOMMENDATIONS", results / "portfolio_allocation_recommendations.csv"
    )
    monkeypatch.setattr(
        et, "EDGE_SIZING_RECOMMENDATIONS_CSV", results / "edge_sizing_recommendations.csv"
    )
    monkeypatch.setattr(et, "DISPATCH_AUDIT_PATH", results / "execute_trades_dispatch_audit.json")
    monkeypatch.setattr(et, "DEFAULT_ORDERS_TODAY", live / "orders_today.csv")
    monkeypatch.setattr(et, "EXEC_PLAN_JSON", results / "execution_plan.json")
    monkeypatch.setattr(et, "EXEC_PLAN_CSV", results / "execution_plan.csv")
    monkeypatch.setattr(et, "EXEC_LOG_CSV", results / "execution_plan_log.csv")
    return results


def _lifecycle_row(ticker: str, stance: str, session: str = "2026-09-17") -> Dict[str, Any]:
    pos = "LONG" if stance in ("HOLD", "ADD", "TRIM", "EXIT") else "FLAT"
    return {
        "ticker": ticker,
        "session_as_of_date": session,
        "session_date": session,
        "effective_stance": stance,
        "lifecycle_action": stance,
        "effective_position_state": pos,
        "position_state": pos,
    }


def _write_proven_edge(results: Path, tickers: List[str], session: str = "2026-09-17") -> None:
    lc = results / "signal_lifecycle_effective.csv"
    opps = results / "trade_opportunities.csv"
    if not opps.is_file():
        _write_csv(
            opps,
            [
                {
                    "ticker": t,
                    "opportunity_type": "ENTRY",
                    "session_date": session,
                    "provenance_status": STATUS_VALID,
                }
                for t in tickers
            ],
        )
    write_json_atomic(
        provenance_sidecar_path(opps),
        {
            "schema": OPPS_PROVENANCE_SCHEMA,
            "status": STATUS_VALID,
            "session_date": session,
            "lifecycle_sha256": sha256_file(lc),
            "row_count": len(tickers),
            "tickers": tickers,
        },
    )
    lc_sha = sha256_file(lc) or ""
    opps_sha = sha256_file(opps) or ""
    rows = []
    for t in tickers:
        rows.append(
            {
                "ticker": t,
                "edge_score": 0.81,
                "sizing_tier": "STRONG_EDGE",
                "session_date": session,
                "lifecycle_sha256": lc_sha,
                "opportunities_sha256": opps_sha,
                "provenance_status": STATUS_VALID,
            }
        )
    edge = results / "edge_sizing_recommendations.csv"
    _write_csv(edge, rows)
    write_json_atomic(
        provenance_sidecar_path(edge),
        {
            "schema": EDGE_PROVENANCE_SCHEMA,
            "status": STATUS_VALID,
            "session_date": session,
            "lifecycle_sha256": lc_sha,
            "opportunities_sha256": opps_sha,
            "row_count": len(tickers),
            "tickers": tickers,
        },
    )


def test_bito_hold_june_or_fresh_edge_fails_closed(monkeypatch, tmp_path: Path):
    results = _bind_results(monkeypatch, tmp_path)
    _write_csv(results / "signal_lifecycle_effective.csv", [_lifecycle_row("BITO", "HOLD")])
    _write_csv(
        results / "trade_opportunities.csv",
        [{"ticker": "BITO", "opportunity_type": "ADD", "edge_score": 0.91}],
    )
    filtered, meta = apply_paper_buy_add_entry_boundary(
        pd.read_csv(results / "trade_opportunities.csv")
    )
    assert filtered.empty
    assert REASON_LIFECYCLE_STANCE_NOT_ENTRY in (meta["rejected"].get("BITO") or [])
    _write_proven_edge(results, ["BITO"])
    filtered2, meta2 = apply_paper_buy_add_entry_boundary(
        pd.read_csv(results / "trade_opportunities.csv")
    )
    assert filtered2.empty
    assert REASON_LIFECYCLE_STANCE_NOT_ENTRY in (meta2["rejected"].get("BITO") or [])


def test_missing_lifecycle_and_unproven_edge_fail_closed(monkeypatch, tmp_path: Path):
    results = _bind_results(monkeypatch, tmp_path)
    _write_csv(
        results / "trade_opportunities.csv",
        [{"ticker": "AAPL", "opportunity_type": "ENTRY"}],
    )
    filtered, meta = apply_paper_buy_add_entry_boundary(
        pd.read_csv(results / "trade_opportunities.csv")
    )
    assert filtered.empty
    reasons = meta["rejected"].get("AAPL") or []
    assert any(
        "lifecycle" in r.lower() or r.endswith("UNPROVEN") or r.endswith("MISSING") for r in reasons
    )
    assert REASON_EDGE_EVIDENCE_UNPROVEN in reasons


def test_wrong_session_symbol_fail_closed(monkeypatch, tmp_path: Path):
    results = _bind_results(monkeypatch, tmp_path)
    _write_csv(
        results / "signal_lifecycle_effective.csv", [_lifecycle_row("AAPL", "BUY", "2026-06-05")]
    )
    _write_csv(
        results / "trade_opportunities.csv",
        [{"ticker": "MSFT", "opportunity_type": "ENTRY"}],
    )
    _write_proven_edge(results, ["MSFT"], session="2026-06-05")
    filtered, meta = apply_paper_buy_add_entry_boundary(
        pd.read_csv(results / "trade_opportunities.csv")
    )
    assert filtered.empty
    assert REASON_LIFECYCLE_STANCE_MISSING in (meta["rejected"].get("MSFT") or [])


def test_valid_current_buy_reaches_boundary(monkeypatch, tmp_path: Path):
    results = _bind_results(monkeypatch, tmp_path)
    _write_csv(results / "signal_lifecycle_effective.csv", [_lifecycle_row("AAPL", "BUY")])
    _write_csv(
        results / "trade_opportunities.csv",
        [{"ticker": "AAPL", "opportunity_type": "ENTRY", "confidence": 0.82, "delta_pct": 0.01}],
    )
    _write_proven_edge(results, ["AAPL"])
    filtered, meta = apply_paper_buy_add_entry_boundary(
        pd.read_csv(results / "trade_opportunities.csv")
    )
    assert list(filtered["ticker"]) == ["AAPL"]
    assert meta["allowed"] == ["AAPL"]
    assert not meta["rejected"]


def test_no_contract_direct_main_does_not_touch_broker(monkeypatch, tmp_path: Path):
    results = _bind_results(monkeypatch, tmp_path)
    _write_csv(results / "signal_lifecycle_effective.csv", [_lifecycle_row("BITO", "HOLD")])
    _write_csv(
        results / "trade_opportunities.csv",
        [{"ticker": "BITO", "opportunity_type": "ADD"}],
    )
    import services.execute_trades as et

    def _broker(*_a, **_k):
        _BROKER.append("load_broker_state")
        raise AssertionError("broker must not run after BUY/ADD boundary reject")

    monkeypatch.setattr(et, "load_broker_state", _broker)
    monkeypatch.setattr(
        et,
        "evaluate_lifecycle_gate",
        lambda **_k: type(
            "G",
            (),
            {"status": "OK", "reason": None, "details": "", "format_block": lambda self: ""},
        )(),
    )
    rc = execute_main(["--mode", "paper", "--execute"])
    assert rc == 2
    assert _BROKER == []
    assert _FORBIDDEN == []


def test_stale_forged_contract_insufficient(monkeypatch, tmp_path: Path):
    from services.execute_trades import validate_paper_entry_dispatch_contract

    expired = {
        "source": "paper_entry_loop_stage",
        "mode": "paper",
        "live_execution_permitted": False,
        "contract_expires_at_utc": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        "final_selected_symbols": ["AAPL"],
        "allowed_symbols": ["AAPL"],
    }
    ok, reason, _ = validate_paper_entry_dispatch_contract(expired, mode="paper")
    assert ok is False
    assert reason == "DISPATCH_CONTRACT_EXPIRED"
    forged = dict(expired)
    forged["contract_expires_at_utc"] = (
        datetime.now(timezone.utc) + timedelta(hours=1)
    ).isoformat()
    forged["source"] = "operator"
    ok2, reason2, _ = validate_paper_entry_dispatch_contract(forged, mode="paper")
    assert ok2 is False
    assert reason2 == "DISPATCH_SOURCE_INVALID"


def test_hd_slv_and_rf_shadow_intact():
    hd, _ = _decide_action(
        is_held=True,
        risk_components=["FORCE_EXIT", "BLOCK_NEW_BUY"],
        edge_score=None,
        sizing_tier="",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="HOLD",
        confidence=0.4,
        delta_pct=-0.02,
    )
    slv, _ = _decide_action(
        is_held=True,
        risk_components=["FORCE_EXIT"],
        edge_score=None,
        sizing_tier="",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="HOLD",
        confidence=0.4,
        delta_pct=-0.03,
    )
    assert hd == "EXIT"
    assert slv == "EXIT"
    a, _ = _decide_action(
        is_held=False,
        risk_components=[],
        edge_score=0.81,
        sizing_tier="STRONG_EDGE",
        effective_stance="BUY",
        lifecycle_action="BUY",
        signal="BUY",
        confidence=0.82,
        delta_pct=-0.05,
    )
    b, _ = _decide_action(
        is_held=False,
        risk_components=[],
        edge_score=0.81,
        sizing_tier="STRONG_EDGE",
        effective_stance="BUY",
        lifecycle_action="BUY",
        signal="BUY",
        confidence=0.82,
        delta_pct=0.05,
    )
    assert a == b == "OPEN_NEW"


def test_sell_rows_preserved_without_buy_add_evidence(monkeypatch, tmp_path: Path):
    results = _bind_results(monkeypatch, tmp_path)
    _write_csv(results / "signal_lifecycle_effective.csv", [_lifecycle_row("HD", "HOLD")])
    df = pd.DataFrame([{"ticker": "HD", "opportunity_type": "EXIT"}])
    filtered, meta = apply_paper_buy_add_entry_boundary(df)
    assert list(filtered["ticker"]) == ["HD"]
    assert meta["sell_rows_preserved"] == 1
