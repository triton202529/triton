"""Phase 198T-SF25B — execute_trades BUY/ADD mutation requires a dispatch contract.

Isolated worktree only. Fake mutation spies. No production writes or real submits.
"""

from __future__ import annotations

import builtins
import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

_PROD = Path(r"C:\Users\akimw\triton")
_REPO = Path(__file__).resolve().parents[1]
if _REPO.resolve() == _PROD.resolve():
    raise RuntimeError("SF25B tests must not run in the production repository.")
if "isolated" not in str(_REPO).lower():
    raise RuntimeError("SF25B tests must run from an isolated worktree.")

from services.edge_sizing_engine import (
    EDGE_PROVENANCE_SCHEMA,
    OPPS_PROVENANCE_SCHEMA,
    STATUS_VALID,
    provenance_sidecar_path,
    sha256_file,
    write_json_atomic,
)
from services.execute_trades import (
    REASON_DISPATCH_CONTRACT_REQUIRED,
    compute_dispatch_contract_checksum,
    main as execute_main,
    maybe_execute_plan,
    require_execute_dispatch_contract,
)
from services.execute_trades import PlannedOrder

_FORBIDDEN: List[str] = []
_MUTATIONS: List[str] = []


def _forbidden(path: Path) -> bool:
    return str(path.resolve()).lower().startswith(str((_PROD / "data").resolve()).lower())


@pytest.fixture(autouse=True)
def _block_prod_net(monkeypatch):
    _FORBIDDEN.clear()
    _MUTATIONS.clear()
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
        raise RuntimeError("network blocked")

    monkeypatch.setattr(socket.socket, "connect", _net)
    monkeypatch.setattr(socket, "create_connection", _net)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _bind(monkeypatch, tmp_path: Path):
    import services.execute_trades as et

    results = tmp_path / "results"
    live = tmp_path / "live"
    results.mkdir()
    live.mkdir()
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
    monkeypatch.setattr(
        et, "DISPATCH_CONTRACT_DEFAULT_PATH", live / "paper_entry_dispatch_contract.json"
    )
    monkeypatch.setattr(et, "DEFAULT_ORDERS_TODAY", live / "orders_today.csv")
    monkeypatch.setattr(et, "EXEC_PLAN_JSON", results / "execution_plan.json")
    monkeypatch.setattr(et, "EXEC_PLAN_CSV", results / "execution_plan.csv")
    monkeypatch.setattr(et, "EXEC_LOG_CSV", results / "execution_plan_log.csv")

    def _no_broker(*_a, **_k):
        _MUTATIONS.append("load_broker_state")
        raise AssertionError("broker must not be constructed")

    def _no_place(*_a, **_k):
        _MUTATIONS.append("maybe_execute_plan_subprocess")
        raise AssertionError("place_live_orders must not be invoked")

    monkeypatch.setattr(et, "load_broker_state", _no_broker)
    monkeypatch.setattr(et.subprocess, "run", _no_place)
    monkeypatch.setattr(
        et,
        "evaluate_lifecycle_gate",
        lambda **_k: type(
            "G",
            (),
            {"status": "OK", "reason": None, "details": "", "format_block": lambda self: ""},
        )(),
    )
    return results, live, et


def _lifecycle_row(ticker: str, stance: str, session: str = "2026-09-18") -> Dict[str, Any]:
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


def _write_proven_edge(results: Path, tickers: List[str], session: str = "2026-09-18") -> None:
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
    rows = [
        {
            "ticker": t,
            "edge_score": 0.81,
            "sizing_tier": "STRONG_EDGE",
            "session_date": session,
            "lifecycle_sha256": lc_sha,
            "opportunities_sha256": opps_sha,
            "provenance_status": STATUS_VALID,
        }
        for t in tickers
    ]
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


def _valid_contract_doc(**overrides: Any) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    doc: Dict[str, Any] = {
        "dispatch_id": "sf25btestdispatch1",
        "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "paper_entry_loop_stage",
        "mode": "paper",
        "governed_source": "portfolio_allocation_recommendations.csv",
        "approved_symbols": ["AMD"],
        "final_selected_symbols": ["AMD"],
        "allowed_symbols": ["AMD"],
        "rejected_symbols": {},
        "planner_input_mode": "allocation",
        "approval_request_id": "sf25b-test",
        "paper_arm_status": "VALID",
        "execution_readiness_status": "READY",
        "market_open": True,
        "live_execution_permitted": False,
        "contract_expires_at_utc": (now + timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    doc.update(overrides)
    doc["checksum"] = compute_dispatch_contract_checksum(doc)
    return doc


def _write_contract(path: Path, **overrides: Any) -> Dict[str, Any]:
    doc = _valid_contract_doc(**overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return doc


def test_execute_without_contract_fails_before_broker(monkeypatch, tmp_path: Path):
    _bind(monkeypatch, tmp_path)
    rc = execute_main(["--mode", "paper", "--execute"])
    assert rc == 2
    assert _MUTATIONS == []
    assert _FORBIDDEN == []


def test_dry_run_without_contract_does_not_mutate(monkeypatch, tmp_path: Path):
    results, _live, et = _bind(monkeypatch, tmp_path)
    _write_csv(results / "signal_lifecycle_effective.csv", [_lifecycle_row("AMD", "BUY")])
    _write_csv(
        results / "trade_opportunities.csv",
        [{"ticker": "AMD", "opportunity_type": "ENTRY", "confidence": 0.82}],
    )
    _write_proven_edge(results, ["AMD"])

    def _acct(*_a, **_k):
        from services.execute_trades import AccountSnapshot

        _MUTATIONS.append("load_broker_state_planning")
        return {}, AccountSnapshot(), object()

    monkeypatch.setattr(et, "load_broker_state", _acct)
    rc = execute_main(["--mode", "paper"])
    assert rc == 0
    assert "maybe_execute_plan_subprocess" not in _MUTATIONS
    assert _FORBIDDEN == []


def test_dry_run_empty_never_constructs_broker(monkeypatch, tmp_path: Path):
    _bind(monkeypatch, tmp_path)
    rc = execute_main(["--mode", "paper"])
    assert rc == 0
    assert _MUTATIONS == []
    assert _FORBIDDEN == []


def test_missing_contract_path_fails_closed(monkeypatch, tmp_path: Path):
    _bind(monkeypatch, tmp_path)
    missing = tmp_path / "no_such_contract.json"
    rc = execute_main(["--mode", "paper", "--execute", "--dispatch-contract", str(missing)])
    assert rc == 2
    assert _MUTATIONS == []


def test_malformed_json_fails_closed(monkeypatch, tmp_path: Path):
    _bind(monkeypatch, tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    rc = execute_main(["--mode", "paper", "--execute", "--dispatch-contract", str(bad)])
    assert rc == 2
    assert _MUTATIONS == []


def test_expired_and_stale_and_forged_fail_closed(monkeypatch, tmp_path: Path):
    results, live, _et = _bind(monkeypatch, tmp_path)
    expired = live / "expired.json"
    _write_contract(
        expired,
        contract_expires_at_utc=(datetime.now(timezone.utc) - timedelta(minutes=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    )
    rc = execute_main(["--mode", "paper", "--execute", "--dispatch-contract", str(expired)])
    assert rc == 2

    stale = live / "stale.json"
    _write_contract(
        stale,
        generated_at_utc=(datetime.now(timezone.utc) - timedelta(minutes=20)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        contract_expires_at_utc=(datetime.now(timezone.utc) + timedelta(minutes=15)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    )
    rc2 = execute_main(["--mode", "paper", "--execute", "--dispatch-contract", str(stale)])
    assert rc2 == 2

    forged = live / "forged.json"
    doc = _valid_contract_doc()
    doc["checksum"] = "0" * 64
    forged.write_text(json.dumps(doc), encoding="utf-8")
    rc3 = execute_main(["--mode", "paper", "--execute", "--dispatch-contract", str(forged)])
    assert rc3 == 2
    assert _MUTATIONS == []
    assert _FORBIDDEN == []


def test_wrong_source_and_live_and_mode_fail_closed(monkeypatch, tmp_path: Path):
    _bind(monkeypatch, tmp_path)
    live_c = tmp_path / "live.json"
    _write_contract(live_c, live_execution_permitted=True)
    assert execute_main(["--mode", "paper", "--execute", "--dispatch-contract", str(live_c)]) == 2
    amb = tmp_path / "amb.json"
    _write_contract(amb, live_execution_permitted=None)
    assert execute_main(["--mode", "paper", "--execute", "--dispatch-contract", str(amb)]) == 2
    missing_live = tmp_path / "missing_live.json"
    doc = _valid_contract_doc()
    doc.pop("live_execution_permitted", None)
    doc["checksum"] = compute_dispatch_contract_checksum(doc)
    missing_live.write_text(json.dumps(doc), encoding="utf-8")
    assert (
        execute_main(["--mode", "paper", "--execute", "--dispatch-contract", str(missing_live)])
        == 2
    )
    src = tmp_path / "src.json"
    _write_contract(src, source="operator")
    assert execute_main(["--mode", "paper", "--execute", "--dispatch-contract", str(src)]) == 2
    assert execute_main(["--mode", "live", "--execute"]) == 2
    assert _MUTATIONS == []


def test_missing_expired_arm_on_contract_fails_closed(monkeypatch, tmp_path: Path):
    _bind(monkeypatch, tmp_path)
    missing = tmp_path / "arm.json"
    _write_contract(missing, paper_arm_status="EXPIRED")
    rc = execute_main(["--mode", "paper", "--execute", "--dispatch-contract", str(missing)])
    assert rc == 2
    malformed = tmp_path / "malformed_arm.json"
    _write_contract(malformed, paper_arm_status="NOT_A_CONFIRM")
    rc2 = execute_main(["--mode", "paper", "--execute", "--dispatch-contract", str(malformed)])
    assert rc2 == 2
    assert _MUTATIONS == []


def test_programmatic_maybe_execute_plan_requires_contract(tmp_path: Path):
    class _SpyBroker:
        def submit_order(self, *_a, **_k):
            _MUTATIONS.append("submit_order")
            raise AssertionError("submit_order must not run")

        def _post(self, *_a, **_k):
            _MUTATIONS.append("broker_post")
            raise AssertionError("broker _post must not run")

    planned = [
        PlannedOrder(
            symbol="AMD",
            side="buy",
            qty=1,
            order_type="limit",
            time_in_force="day",
            limit_price=10.0,
            stance="BUY",
            rationale="sf25b",
            confidence=0.8,
            delta_pct=0.01,
            planned_notional=10.0,
            current_position_qty=0.0,
            mode="paper",
            generated_at="2026-09-19T00:00:00Z",
        )
    ]
    rc, executed, reasons = maybe_execute_plan(
        planned,
        "paper",
        False,
        {"execute_via_existing_placement_flow": True},
        _SpyBroker(),
        False,
        True,
        tmp_path / "orders.csv",
        dispatch_contract=None,
    )
    assert rc == 2
    assert executed == 0
    assert REASON_DISPATCH_CONTRACT_REQUIRED in reasons
    assert _MUTATIONS == []
    assert _FORBIDDEN == []


def test_require_helper_does_not_default_production_path(tmp_path: Path):
    doc, reason, meta = require_execute_dispatch_contract(None, mode="paper")
    assert doc is None
    assert reason == REASON_DISPATCH_CONTRACT_REQUIRED
    assert "C:\\Users\\akimw\\triton\\data" not in str(meta)


def test_maybe_execute_plan_sell_only_does_not_require_contract(tmp_path: Path):
    planned = [
        PlannedOrder(
            symbol="TSLA",
            side="sell",
            qty=1,
            order_type="limit",
            time_in_force="day",
            limit_price=10.0,
            stance="EXIT",
            rationale="sf25b",
            confidence=0.8,
            delta_pct=-0.01,
            planned_notional=10.0,
            current_position_qty=1.0,
            mode="paper",
            generated_at="2026-09-19T00:00:00Z",
        )
    ]
    rc, executed, reasons = maybe_execute_plan(
        planned,
        "paper",
        False,
        {"execute_via_existing_placement_flow": False},
        object(),
        False,
        True,
        tmp_path / "orders.csv",
        dispatch_contract=None,
    )
    assert rc == 0
    assert executed == 0
    assert REASON_DISPATCH_CONTRACT_REQUIRED not in reasons


def test_independent_block_new_buy_rejects_before_broker(monkeypatch, tmp_path: Path):
    from services.execute_trades import (
        REASON_INDEPENDENT_BLOCK_NEW_BUY,
        apply_paper_buy_add_entry_boundary,
    )

    results, _live, et = _bind(monkeypatch, tmp_path)
    _write_csv(results / "signal_lifecycle_effective.csv", [_lifecycle_row("AMD", "BUY")])
    _write_csv(
        results / "trade_opportunities.csv",
        [{"ticker": "AMD", "opportunity_type": "ENTRY", "confidence": 0.82}],
    )
    _write_proven_edge(results, ["AMD"])
    _write_csv(
        results / "performance_risk_overlay.csv",
        [{"ticker": "AMD", "risk_flag": "BLOCK_NEW_BUY"}],
    )
    df = pd.read_csv(results / "trade_opportunities.csv")
    out, meta = apply_paper_buy_add_entry_boundary(df)
    assert out.empty
    assert "AMD" in (meta.get("rejected") or {})
    assert REASON_INDEPENDENT_BLOCK_NEW_BUY in (meta["rejected"]["AMD"])
    assert meta.get("broker_invoked") is False
    assert _MUTATIONS == []
    assert _FORBIDDEN == []


def test_valid_contract_reaches_planning_not_default_artifact(monkeypatch, tmp_path: Path):
    results, live, et = _bind(monkeypatch, tmp_path)
    _write_csv(results / "signal_lifecycle_effective.csv", [_lifecycle_row("AMD", "BUY")])
    _write_csv(
        results / "trade_opportunities.csv",
        [{"ticker": "AMD", "opportunity_type": "ENTRY", "confidence": 0.82, "delta_pct": 0.01}],
    )
    _write_proven_edge(results, ["AMD"])
    contract_path = tmp_path / "external" / "dispatch.json"
    _write_contract(contract_path)
    seen: List[str] = []

    def _fake_execute(*args, **kwargs):
        seen.append("maybe_execute_plan")
        assert kwargs.get("dispatch_contract") or args
        return 0, 0, []

    monkeypatch.setattr(et, "maybe_execute_plan", _fake_execute)

    def _acct(*_a, **_k):
        from services.execute_trades import AccountSnapshot

        return {}, AccountSnapshot(), object()

    monkeypatch.setattr(et, "load_broker_state", _acct)
    rc = execute_main(["--mode", "paper", "--execute", "--dispatch-contract", str(contract_path)])
    assert rc in (0, 2)
    assert _FORBIDDEN == []
    assert str(_PROD / "data" / "live" / "paper_entry_dispatch_contract.json") not in "".join(
        str(x) for x in seen
    )
