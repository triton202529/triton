"""Phase 198T-SF23 — stale-edge provenance and paper-entry stance guards.

Isolated worktree only. Disposable paths. No broker, network, production
write, Stop request, or trading-control invocation.
"""

from __future__ import annotations

import builtins
import json
import socket
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

_PROD = Path(r"C:\Users\akimw\triton")
_REPO = Path(__file__).resolve().parents[1]
if _REPO.resolve() == _PROD.resolve():
    raise RuntimeError("SF23 tests must not run in the production repository.")
if "isolated" not in str(_REPO).lower():
    raise RuntimeError("SF23 tests must run from an isolated worktree.")

from services.edge_sizing_engine import (
    EDGE_PROVENANCE_SCHEMA,
    OPPS_PROVENANCE_SCHEMA,
    STATUS_FAIL_CLOSED,
    STATUS_VALID,
    load_proven_edge_sizing_map,
    provenance_sidecar_path,
    run as edge_run,
    sha256_file,
    write_fail_closed_edge,
    write_json_atomic,
)
from services.paper_entry_loop_stage import (
    REASON_LIFECYCLE_STANCE_NOT_ENTRY,
    _evaluate_allocation_entry_row,
    _load_governed_allocation_candidates,
)
from services.portfolio_allocation_engine import (
    STRONG_EDGE_SCORE,
    _decide_action,
    build_recommendations,
)

_FORBIDDEN_HIT: List[str] = []
_NETWORK_HIT: List[str] = []


def _forbidden(path: Path) -> bool:
    text = str(path.resolve()).lower()
    return text.startswith(str((_PROD / "data").resolve()).lower())


@pytest.fixture(autouse=True)
def _block_production_and_network(monkeypatch):
    _FORBIDDEN_HIT.clear()
    _NETWORK_HIT.clear()
    real_open = builtins.open

    def guarded(file, mode="r", *args, **kwargs):
        try:
            path = Path(file)
        except TypeError:
            return real_open(file, mode, *args, **kwargs)
        writing = any(ch in str(mode) for ch in "wxa+")
        if writing and _forbidden(path):
            _FORBIDDEN_HIT.append(str(path))
            raise RuntimeError(f"blocked production write: {path}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)

    def _net(*_a, **_k):
        _NETWORK_HIT.append("socket")
        raise RuntimeError("network blocked")

    monkeypatch.setattr(socket.socket, "connect", _net)
    yield
    assert not _FORBIDDEN_HIT
    assert not _NETWORK_HIT


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _lifecycle_row(
    ticker: str,
    *,
    session: str,
    stance: str,
    position: str,
    confidence: float = 0.80,
) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "session_as_of_date": session,
        "lifecycle_action": stance,
        "effective_stance": stance,
        "effective_position_state": position,
        "position_state": position,
        "stance": stance,
        "signal": stance,
        "confidence": confidence,
        "delta_pct": 0.01,
    }


def _write_proven_opps(path: Path, lifecycle_path: Path, rows: List[Dict[str, Any]]) -> str:
    session = rows[0]["session_as_of_date"] if rows else ""
    lc_sha = sha256_file(lifecycle_path) or ""
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "ticker": row["ticker"],
                "opportunity_type": "ENTRY",
                "session_date": session,
                "lifecycle_sha256": lc_sha,
                "provenance_status": STATUS_VALID,
                "confidence": row.get("confidence", 0.80),
            }
        )
    _write_csv(path, out_rows)
    write_json_atomic(
        provenance_sidecar_path(path),
        {
            "schema": OPPS_PROVENANCE_SCHEMA,
            "status": STATUS_VALID,
            "session_date": session,
            "lifecycle_sha256": lc_sha,
            "row_count": len(out_rows),
            "tickers": [r["ticker"] for r in out_rows],
            "invalid_reason": None,
        },
    )
    return sha256_file(path) or ""


def _write_proven_edge(
    path: Path,
    *,
    session: str,
    lifecycle_sha: str,
    opps_sha: str,
    rows: List[Dict[str, Any]],
    status: str = STATUS_VALID,
) -> None:
    out_rows = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "session_date": session,
                "lifecycle_sha256": lifecycle_sha,
                "opportunities_sha256": opps_sha,
                "provenance_status": status,
            }
        )
        out_rows.append(item)
    _write_csv(path, out_rows)
    write_json_atomic(
        provenance_sidecar_path(path),
        {
            "schema": EDGE_PROVENANCE_SCHEMA,
            "status": status,
            "session_date": session,
            "lifecycle_sha256": lifecycle_sha,
            "opportunities_sha256": opps_sha,
            "row_count": len(out_rows),
            "tickers": [r["ticker"] for r in out_rows],
            "invalid_reason": None if status == STATUS_VALID else "test_fail_closed",
        },
    )


def test_strong_edge_threshold_unchanged():
    assert STRONG_EDGE_SCORE == 0.50


def test_market_open_wires_producers_before_allocation():
    src = (_REPO / "scripts" / "run_market_open.ps1").read_text(encoding="utf-8")
    i_opps = src.find("E4.5 BUILD TRADE OPPORTUNITIES")
    i_edge = src.find("E4.6 EDGE SIZING ENGINE")
    i_alloc = src.find("E5. PORTFOLIO ALLOCATION ENGINE")
    assert 0 <= i_opps < i_edge < i_alloc
    assert "services.build_trade_opportunities" in src
    assert "services.edge_sizing_engine" in src
    assert "--invalidate-current-evidence" in src
    assert src.find("invalidate-current-evidence") < i_alloc


def test_june_csv_without_sidecar_cannot_grant_add(tmp_path: Path):
    session = "2026-09-17"
    lc = tmp_path / "signal_lifecycle_effective.csv"
    _write_csv(
        lc,
        [_lifecycle_row("BITO", session=session, stance="HOLD", position="LONG")],
    )
    opps = tmp_path / "trade_opportunities.csv"
    _write_proven_opps(opps, lc, [])
    edge = tmp_path / "edge_sizing_recommendations.csv"
    _write_csv(
        edge,
        [
            {
                "ticker": "BITO",
                "opportunity_type": "ENTRY",
                "edge_score": 0.908222,
                "sizing_tier": "STRONG_EDGE",
                "risk_flag": "OK",
                "confidence": 0.46,
            }
        ],
    )
    proven, diag = load_proven_edge_sizing_map(edge, lifecycle_path=lc, opportunities_path=opps)
    assert proven == {}
    assert diag["reason"] == "edge_provenance_sidecar_missing"
    action, _reason = _decide_action(
        is_held=True,
        risk_components=[],
        edge_score=None,
        sizing_tier="",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="HOLD",
        confidence=0.57,
        delta_pct=0.0,
    )
    assert action == "HOLD"


def test_missing_stale_forged_and_duplicate_edge_fail_closed(tmp_path: Path):
    session = "2026-09-17"
    lc = tmp_path / "lc.csv"
    _write_csv(lc, [_lifecycle_row("AAPL", session=session, stance="BUY", position="FLAT")])
    lc_sha = sha256_file(lc) or ""
    opps = tmp_path / "opps.csv"
    opps_sha = _write_proven_opps(
        opps, lc, [_lifecycle_row("AAPL", session=session, stance="BUY", position="FLAT")]
    )
    missing = tmp_path / "missing.csv"
    proven, diag = load_proven_edge_sizing_map(missing, lifecycle_path=lc, opportunities_path=opps)
    assert proven == {}
    assert diag["reason"] == "edge_provenance_sidecar_missing"

    stale = tmp_path / "stale.csv"
    _write_proven_edge(
        stale,
        session="2026-06-05",
        lifecycle_sha=lc_sha,
        opps_sha=opps_sha,
        rows=[{"ticker": "AAPL", "edge_score": 0.91, "sizing_tier": "STRONG_EDGE"}],
    )
    proven, diag = load_proven_edge_sizing_map(stale, lifecycle_path=lc, opportunities_path=opps)
    assert proven == {}
    assert diag["reason"] == "edge_session_unproven"

    forged = tmp_path / "forged.csv"
    _write_proven_edge(
        forged,
        session=session,
        lifecycle_sha="0" * 64,
        opps_sha=opps_sha,
        rows=[{"ticker": "AAPL", "edge_score": 0.91, "sizing_tier": "STRONG_EDGE"}],
    )
    proven, diag = load_proven_edge_sizing_map(forged, lifecycle_path=lc, opportunities_path=opps)
    assert proven == {}
    assert diag["reason"] == "edge_lifecycle_hash_mismatch"

    dup = tmp_path / "dup.csv"
    _write_proven_edge(
        dup,
        session=session,
        lifecycle_sha=lc_sha,
        opps_sha=opps_sha,
        rows=[
            {"ticker": "AAPL", "edge_score": 0.91, "sizing_tier": "STRONG_EDGE"},
            {"ticker": "AAPL", "edge_score": 0.92, "sizing_tier": "STRONG_EDGE"},
        ],
    )
    proven, diag = load_proven_edge_sizing_map(dup, lifecycle_path=lc, opportunities_path=opps)
    assert proven == {}
    assert diag["reason"] == "edge_duplicate_tickers"


def test_producer_failure_replaces_prior_edge_file(tmp_path: Path):
    prior = tmp_path / "edge_sizing_recommendations.csv"
    _write_csv(
        prior,
        [{"ticker": "BITO", "edge_score": 0.908, "sizing_tier": "STRONG_EDGE"}],
    )
    assert "BITO" in prior.read_text(encoding="utf-8")
    rc = edge_run(
        opps_path=tmp_path / "missing_opps.csv",
        risk_path=tmp_path / "missing_risk.csv",
        perf_path=tmp_path / "missing_perf.csv",
        out_path=prior,
        lifecycle_path=tmp_path / "missing_lc.csv",
    )
    assert rc == 2
    df = pd.read_csv(prior)
    assert "BITO" not in set(df.get("ticker", pd.Series(dtype=str)).astype(str))
    sidecar = json.loads(provenance_sidecar_path(prior).read_text(encoding="utf-8"))
    assert sidecar["status"] == STATUS_FAIL_CLOSED
    write_fail_closed_edge(prior, "producer_failed_or_incomplete")
    sidecar2 = json.loads(provenance_sidecar_path(prior).read_text(encoding="utf-8"))
    assert sidecar2["status"] == STATUS_FAIL_CLOSED


def test_hold_with_fresh_strong_edge_is_not_paper_eligible():
    action, reason = _decide_action(
        is_held=True,
        risk_components=[],
        edge_score=0.91,
        sizing_tier="STRONG_EDGE",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="HOLD",
        confidence=0.80,
        delta_pct=0.0,
    )
    assert action == "ADD"
    assert "strong_edge" in reason
    eligible, blocked, _flag = _evaluate_allocation_entry_row(
        sym="BITO",
        action="ADD",
        row={"risk_flag": "OK"},
        risk_map={},
        open_buy=set(),
        held={"BITO"},
        at_cap=False,
        lifecycle_stance="HOLD",
        lifecycle_ok=True,
    )
    assert eligible is False
    assert REASON_LIFECYCLE_STANCE_NOT_ENTRY in blocked


def test_allocation_open_new_alone_is_not_paper_eligible():
    eligible, blocked, _flag = _evaluate_allocation_entry_row(
        sym="AAPL",
        action="OPEN_NEW",
        row={"risk_flag": "OK"},
        risk_map={},
        open_buy=set(),
        held=set(),
        at_cap=False,
        lifecycle_stance="WAIT",
        lifecycle_ok=True,
    )
    assert eligible is False
    assert REASON_LIFECYCLE_STANCE_NOT_ENTRY in blocked


def test_current_buy_with_proven_edge_and_risk_ok_may_be_eligible(monkeypatch, tmp_path: Path):
    session = "2026-09-17"
    results = tmp_path / "results"
    results.mkdir()
    _write_csv(
        results / "signal_lifecycle_effective.csv",
        [_lifecycle_row("AAPL", session=session, stance="BUY", position="FLAT")],
    )
    _write_csv(
        results / "portfolio_allocation_recommendations.csv",
        [
            {
                "ticker": "AAPL",
                "recommended_action": "OPEN_NEW",
                "risk_flag": "OK",
                "edge_score": 0.81,
                "confidence": 0.82,
            }
        ],
    )
    _write_csv(results / "performance_risk_overlay.csv", [{"ticker": "AAPL", "risk_flag": "OK"}])
    _write_csv(results / "positions_snapshot.csv", [{"symbol": "CASH", "qty": 0}])
    _write_csv(results / "open_orders_snapshot.csv", [{"symbol": "NONE", "side": "sell"}])
    monkeypatch.setattr("services.paper_entry_loop_stage.RESULTS", results)
    monkeypatch.setattr(
        "services.paper_entry_loop_stage.ALLOC_RECOMMENDATIONS",
        results / "portfolio_allocation_recommendations.csv",
    )
    monkeypatch.setattr(
        "services.paper_entry_loop_stage.RISK_OVERLAY",
        results / "performance_risk_overlay.csv",
    )
    monkeypatch.setattr(
        "services.paper_entry_loop_stage.POSITIONS",
        results / "positions_snapshot.csv",
    )
    monkeypatch.setattr(
        "services.paper_entry_loop_stage.OPEN_ORDERS",
        results / "open_orders_snapshot.csv",
    )
    monkeypatch.setattr(
        "services.paper_entry_loop_stage.TRADE_OPPORTUNITIES",
        results / "missing_opps.csv",
    )
    monkeypatch.setattr("services.paper_entry_loop_stage._EXECUTE_INVOKED", False)
    monkeypatch.setattr(
        "services.duplicate_order_guard.check_duplicate_order",
        lambda **_k: (False, ""),
    )
    candidates, handoff = _load_governed_allocation_candidates()
    assert candidates
    assert candidates[0]["ticker"] == "AAPL"
    assert candidates[0]["eligible_for_auto_entry"] is True
    assert "AAPL" in handoff["eligible_entry_symbols"]
    from services import paper_entry_loop_stage as stage

    assert stage._EXECUTE_INVOKED is False


def test_uso_rule_c_and_hd_slv_independent_of_edge():
    uso, _ = _decide_action(
        is_held=False,
        risk_components=["BLOCK_NEW_BUY"],
        edge_score=0.91,
        sizing_tier="STRONG_EDGE",
        effective_stance="BUY",
        lifecycle_action="BUY",
        signal="BUY",
        confidence=0.80,
        delta_pct=0.0,
    )
    assert uso == "BLOCK"
    hd, _ = _decide_action(
        is_held=True,
        risk_components=["FORCE_EXIT", "BLOCK_NEW_BUY"],
        edge_score=None,
        sizing_tier="",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="HOLD",
        confidence=0.60,
        delta_pct=0.0,
    )
    assert hd == "EXIT"
    slv, _ = _decide_action(
        is_held=True,
        risk_components=["FORCE_EXIT"],
        edge_score=0.91,
        sizing_tier="STRONG_EDGE",
        effective_stance="HOLD",
        lifecycle_action="HOLD",
        signal="HOLD",
        confidence=0.60,
        delta_pct=0.0,
    )
    assert slv == "EXIT"


def test_proven_edge_join_feeds_allocation_build(tmp_path: Path):
    session = "2026-09-17"
    lc = tmp_path / "lc.csv"
    rows = [
        _lifecycle_row("BITO", session=session, stance="HOLD", position="LONG"),
        _lifecycle_row("AAPL", session=session, stance="BUY", position="FLAT"),
        _lifecycle_row("USO", session=session, stance="BUY", position="FLAT"),
        _lifecycle_row("HD", session=session, stance="HOLD", position="LONG"),
    ]
    _write_csv(lc, rows)
    opps = tmp_path / "opps.csv"
    opps_sha = _write_proven_opps(opps, lc, rows)
    edge = tmp_path / "edge.csv"
    _write_proven_edge(
        edge,
        session=session,
        lifecycle_sha=sha256_file(lc) or "",
        opps_sha=opps_sha,
        rows=[
            {"ticker": "BITO", "edge_score": 0.91, "sizing_tier": "STRONG_EDGE", "risk_flag": "OK"},
            {"ticker": "AAPL", "edge_score": 0.81, "sizing_tier": "STRONG_EDGE", "risk_flag": "OK"},
        ],
    )
    proven, diag = load_proven_edge_sizing_map(edge, lifecycle_path=lc, opportunities_path=opps)
    assert diag["status"] == STATUS_VALID
    assert set(proven) == {"BITO", "AAPL"}
    df, summary = build_recommendations(
        positions={
            "BITO": {"market_value": 1000.0, "qty": 10.0, "unrealized_pl": 0.0},
            "HD": {"market_value": 1000.0, "qty": 1.0, "unrealized_pl": -10.0},
        },
        edge_sizing=proven,
        risk_overlay={
            "USO": {"risk_flag": "BLOCK_NEW_BUY", "total_pl": 0.0, "unrealized_pl": 0.0},
            "HD": {"risk_flag": "FORCE_EXIT", "total_pl": -10.0, "unrealized_pl": -10.0},
        },
        perf_by_symbol={},
        signals={},
        lifecycle_eff={
            "BITO": {
                "lifecycle_action": "HOLD",
                "effective_stance": "HOLD",
                "confidence": 0.57,
                "delta_pct": 0.0,
                "signal": "HOLD",
            },
            "AAPL": {
                "lifecycle_action": "BUY",
                "effective_stance": "BUY",
                "confidence": 0.82,
                "delta_pct": 0.0,
                "signal": "BUY",
            },
            "USO": {
                "lifecycle_action": "BUY",
                "effective_stance": "BUY",
                "confidence": 0.80,
                "delta_pct": 0.0,
                "signal": "BUY",
            },
            "HD": {
                "lifecycle_action": "HOLD",
                "effective_stance": "HOLD",
                "confidence": 0.60,
                "delta_pct": 0.0,
                "signal": "HOLD",
            },
        },
        max_positions=25,
        max_positions_source="test",
    )
    actions = {r["ticker"]: r["recommended_action"] for _, r in df.iterrows()}
    assert actions["BITO"] == "ADD"
    assert actions["AAPL"] == "OPEN_NEW"
    assert actions["USO"] == "BLOCK"
    assert actions["HD"] == "EXIT"
    assert summary["add_candidates"] >= 1
    assert summary["open_new_candidates"] >= 1
    assert summary["blocked_candidates"] >= 1
    assert summary["exit_candidates"] >= 1
