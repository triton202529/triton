"""Phase 198I — real manage_positions → place_live_orders → hardened SELL → fake broker.

Does not stub place_live_orders production logic. Fake is at AlpacaBroker HTTP helpers.
PAPER ONLY. No real broker. Live remains blocked.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    LIVE_BLOCKED,
    SESSION_OPEN_NOW,
    NetworkFakeAlpacaBroker,
    OrderedTrace,
    block_real_network,
    freeze_dispatch_clock,
    inprocess_place_live_orders,
    install_boundary_trace,
    isolate_sell_store,
    seed_paper_sell_store,
    stub_unrelated_pre_gates,
)
from services.manage_positions import ManagementPlannedOrder, maybe_execute_management
from services.paper_sell_arm_runtime_enforcement import PaperSellDispatchBlocked

ROOT = Path(__file__).resolve().parents[1]


def _evidence_dir() -> Path:
    override = os.environ.get("TRITON_198I_EVIDENCE_OUT")
    if override:
        return Path(override)
    return ROOT / "data" / "results" / "phase_198i_sell_remediation_20260814"


@pytest.fixture(autouse=True)
def _safety(monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    block_real_network(monkeypatch)
    from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT

    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)


def _planned(
    *,
    symbol="TSLA",
    side="sell",
    qty=8,
    price=332.5,
    action="EXIT",
) -> ManagementPlannedOrder:
    return ManagementPlannedOrder(
        symbol=symbol,
        side=side,
        qty=int(qty),
        order_type="limit",
        time_in_force="day",
        limit_price=price,
        management_action=action,
        rationale="phase198i-integration",
        confidence=0.9,
        delta_pct=-0.01,
        current_position_qty=float(qty),
        mode="paper",
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        planned_notional=float(qty) * float(price or 0),
    )


def _cfg() -> Dict[str, Any]:
    return {
        "execute_via_existing_placement_flow": True,
        "require_master_gate_for_execute": False,
        "require_execution_gate_for_execute": False,
        "require_guard_validation_for_each_order": False,
        "write_manage_orders_csv": True,
        "max_batch_notional": 50000.0,
    }


def _run_chain(
    monkeypatch,
    tmp_path: Path,
    *,
    fake: NetworkFakeAlpacaBroker,
    trace: OrderedTrace,
    planned=None,
    seed: bool = True,
    consume: bool = False,
    mode: str = "paper",
    extra_seed: Optional[Dict[str, Any]] = None,
    placement_session: Optional[str] = None,
    dispatch_now: Optional[datetime] = None,
):
    freeze_dispatch_clock(monkeypatch)
    if seed:
        extra_seed = extra_seed or {}
        seed_paper_sell_store(tmp_path, consume=consume, **extra_seed)
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    install_boundary_trace(monkeypatch, trace)
    session = placement_session or f"phase198i_{tmp_path.name}"
    now = dispatch_now or (SESSION_OPEN_NOW + timedelta(minutes=1))
    if dispatch_now is not None:
        freeze_dispatch_clock(monkeypatch, dispatch_now)

    import services.place_live_orders as plo
    import services.manage_positions as mp

    real_eac = plo.enforce_and_consume_paper_sells_before_submit

    def _eac(**kwargs):
        kwargs.setdefault("now", now)
        kwargs.setdefault("live_override", LIVE_BLOCKED)
        kwargs.setdefault("results_root", tmp_path)
        kwargs.setdefault("live_root", tmp_path)
        return real_eac(**kwargs)

    monkeypatch.setattr(plo, "enforce_and_consume_paper_sells_before_submit", _eac)
    monkeypatch.setattr(plo, "make_broker", lambda mode: fake)
    monkeypatch.setattr(mp.subprocess, "run", inprocess_place_live_orders)

    real_maybe = mp.maybe_execute_management

    def _traced_maybe(*args, **kwargs):
        trace.add("manage_positions_entered", fn="maybe_execute_management")
        return real_maybe(*args, **kwargs)

    monkeypatch.setattr(mp, "maybe_execute_management", _traced_maybe)

    real_main = plo.main

    def _traced_main():
        trace.add("place_live_orders_entered", fn="main")
        return real_main()

    monkeypatch.setattr(plo, "main", _traced_main)

    orders_path = tmp_path / "manage_orders.csv"
    planned = planned or [_planned()]
    rc, executed, reasons, session_id, meta = mp.maybe_execute_management(
        planned,
        mode,
        False,
        _cfg(),
        fake,
        True,
        True,
        orders_path,
        placement_session=session,
    )
    sell_payloads = [p for p in fake.submit_payloads if str(p.get("side") or "").lower() == "sell"]
    buy_payloads = [
        p
        for p in fake.submit_payloads
        if str(p.get("side") or "").lower() in {"buy", "add"}
    ]
    return {
        "rc": rc,
        "executed": executed,
        "reasons": reasons,
        "session_id": session_id,
        "meta": meta,
        "fake": fake,
        "trace": trace,
        "sell_payloads": sell_payloads,
        "buy_payloads": buy_payloads,
        "orders_path": orders_path,
    }


def test_accepted_manage_positions_to_fake_broker_sell_once(tmp_path, monkeypatch):
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    result = _run_chain(monkeypatch, tmp_path, fake=fake, trace=trace, seed=True, consume=False)
    names = trace.names()
    assert "manage_positions_entered" in names
    assert "place_live_orders_entered" in names
    assert "hardened_sell_boundary_entered" in names
    assert any(
        e.get("event") == "begin_paper_sell_dispatch_result" and e.get("consumed")
        for e in trace.events
    )
    assert any(
        e.get("event") == "validation_result" and e.get("ok") is True for e in trace.events
    )
    assert len(result["sell_payloads"]) == 1
    payload = result["sell_payloads"][0]
    assert payload["symbol"] == "TSLA"
    assert str(payload["side"]).lower() == "sell"
    assert float(payload["qty"]) == 8
    assert fake.mode == "paper"
    assert result["buy_payloads"] == []
    assert len(fake.submit_payloads) == 1
    assert fake.session is None
    assert fake.base.startswith("http://phase198i-fake-broker.test")
    # consumption exists
    assert (tmp_path / "paper_sell_arm_consumption.json").is_file()
    # BUY arm untouched
    assert (tmp_path / "paper_arm_confirm.json").is_file()
    ev_dir = _evidence_dir()
    ev_dir.mkdir(parents=True, exist_ok=True)
    (ev_dir / "04_e2e_manage_positions_trace.json").write_text(
        json.dumps(
            {
                "chain": [
                    "manage_positions",
                    "place_live_orders",
                    "hardened_SELL_boundary",
                    "fake_broker",
                ],
                "trace_events": trace.events,
                "trace_names": names,
                "accepted_fake_broker_sell_count": len(result["sell_payloads"]),
                "buy_or_add_count": len(result["buy_payloads"]),
                "duplicate_sell": len(result["sell_payloads"]) > 1,
                "side": payload.get("side"),
                "symbol": payload.get("symbol"),
                "quantity": payload.get("qty"),
                "paper_live_mode": fake.mode,
                "fake_broker_base": fake.base,
                "real_session_object": fake.session,
                "final_result": {
                    "rc": result["rc"],
                    "executed": result["executed"],
                    "session_id": result["session_id"],
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def test_rejected_missing_authorization_zero_fake_calls(tmp_path, monkeypatch):
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    result = _run_chain(monkeypatch, tmp_path, fake=fake, trace=trace, seed=False)
    assert fake.fake_broker_call_count == 0
    assert result["sell_payloads"] == []
    reasons = [
        e.get("reason")
        for e in trace.events
        if e.get("event") in ("validation_result", "begin_paper_sell_dispatch_result")
    ]
    assert any(isinstance(r, str) and r for r in reasons), f"missing denial reason: {trace.events}"
    assert any(
        r
        in {
            "approval_contract_missing",
            "paper_sell_arm_missing",
            "paper_sell_authority_missing",
            "paper_sell_requires_hardened_dispatch_permit",
            "HARDENED_REQUIRED",
            "unarmed",
            "not_armed",
        }
        or (isinstance(r, str) and "missing" in r.lower())
        or (isinstance(r, str) and "approval" in r.lower())
        or (isinstance(r, str) and "arm" in r.lower())
        or (isinstance(r, str) and "unarmed" in r.lower())
        for r in reasons
    ), f"unexpected denial reasons: {reasons}"


def test_rejected_malformed_authorization_zero_fake_calls(tmp_path, monkeypatch):
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    (tmp_path / "paper_sell_arm_authority.json").write_text("{not-json", encoding="utf-8")
    result = _run_chain(monkeypatch, tmp_path, fake=fake, trace=trace, seed=False)
    assert fake.fake_broker_call_count == 0
    assert result["sell_payloads"] == []


def test_rejected_expired_authorization_zero_fake_calls(tmp_path, monkeypatch):
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    seed_paper_sell_store(tmp_path, now=SESSION_OPEN_NOW, consume=False)
    path = tmp_path / "paper_sell_arm_authority.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["expires_at"] = (SESSION_OPEN_NOW - timedelta(minutes=1)).isoformat()
    path.write_text(json.dumps(doc), encoding="utf-8")
    result = _run_chain(monkeypatch, tmp_path, fake=fake, trace=trace, seed=False)
    assert fake.fake_broker_call_count == 0
    assert result["sell_payloads"] == []


def test_rejected_replayed_authorization_zero_fake_calls(tmp_path, monkeypatch):
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    # First accepted consume
    _run_chain(
        monkeypatch,
        tmp_path,
        fake=fake,
        trace=trace,
        seed=True,
        consume=False,
        placement_session=f"phase198i_replay_a_{tmp_path.name}",
    )
    assert len(fake.submit_payloads) == 1
    fake2 = NetworkFakeAlpacaBroker()
    trace2 = OrderedTrace()
    result = _run_chain(
        monkeypatch,
        tmp_path,
        fake=fake2,
        trace=trace2,
        seed=False,
        placement_session=f"phase198i_replay_b_{tmp_path.name}",
    )
    assert fake2.fake_broker_call_count == 0
    assert result["sell_payloads"] == []
    replay_reasons = [
        e.get("reason")
        for e in trace2.events
        if e.get("event") in ("validation_result", "begin_paper_sell_dispatch_result")
    ]
    assert any(
        isinstance(r, str)
        and (
            "replay" in r.lower()
            or "consumed" in r.lower()
            or "duplicate" in r.lower()
            or "exhausted" in r.lower()
            or "already" in r.lower()
        )
        for r in replay_reasons
    ), f"unexpected replay denial: {replay_reasons}"


def test_rejected_wrong_symbol_zero_fake_calls(tmp_path, monkeypatch):
    fake = NetworkFakeAlpacaBroker()
    fake.positions = [{"symbol": "AAPL", "qty": "8", "qty_available": "8"}]
    trace = OrderedTrace()
    result = _run_chain(
        monkeypatch,
        tmp_path,
        fake=fake,
        trace=trace,
        seed=True,
        planned=[_planned(symbol="AAPL", side="sell", qty=8, action="EXIT")],
    )
    assert fake.fake_broker_call_count == 0
    assert result["sell_payloads"] == []


def test_rejected_wrong_quantity_exceeds_zero_fake_calls(tmp_path, monkeypatch):
    fake = NetworkFakeAlpacaBroker()
    fake.positions = [{"symbol": "TSLA", "qty": "80", "qty_available": "80"}]
    trace = OrderedTrace()
    result = _run_chain(
        monkeypatch,
        tmp_path,
        fake=fake,
        trace=trace,
        seed=True,
        planned=[_planned(qty=80)],
    )
    assert fake.fake_broker_call_count == 0
    assert result["sell_payloads"] == []


def test_rejected_non_reducing_short_sell_zero_fake_calls(tmp_path, monkeypatch):
    fake = NetworkFakeAlpacaBroker()
    fake.positions = []
    trace = OrderedTrace()
    result = _run_chain(monkeypatch, tmp_path, fake=fake, trace=trace, seed=True)
    assert result["sell_payloads"] == []
    assert fake.fake_broker_call_count == 0 or len(fake.submit_payloads) == 0


def test_rejected_live_mode_zero_fake_calls(tmp_path, monkeypatch):
    fake = NetworkFakeAlpacaBroker(mode="live")
    trace = OrderedTrace()
    result = _run_chain(
        monkeypatch, tmp_path, fake=fake, trace=trace, seed=True, mode="live"
    )
    assert fake.fake_broker_call_count == 0
    assert result["sell_payloads"] == []


def test_rejected_buy_does_not_use_sell_path(tmp_path, monkeypatch):
    fake = NetworkFakeAlpacaBroker()
    fake.positions = []
    trace = OrderedTrace()
    buy = _planned(side="buy", action="ADD", qty=1, symbol="AAPL")
    result = _run_chain(
        monkeypatch, tmp_path, fake=fake, trace=trace, seed=True, planned=[buy]
    )
    assert result["sell_payloads"] == []
    # BUY/ADD must not emit a SELL. BUY may be independently gated; either zero submits or buy-only.
    for p in fake.submit_payloads:
        assert str(p.get("side") or "").lower() != "sell"


def test_rejected_closed_session_zero_fake_calls(tmp_path, monkeypatch):
    """Existing contract: caller MARKET_OPEN cannot override authoritative CLOSED."""
    from services.paper_sell_arm_runtime_enforcement import (
        evaluate_market_session_for_dispatch,
    )

    closed_now = datetime(2026, 8, 11, 20, 30, 0, tzinfo=timezone.utc)
    ok, reason, _details = evaluate_market_session_for_dispatch(
        closed_now,
        caller_session="MARKET_OPEN",
        session_evidence={
            "market_session": "MARKET_OPEN",
            "market_session_observed_at": closed_now.isoformat(),
        },
    )
    assert ok is False
    assert reason == "market_session_caller_open_authoritative_closed"

    seed_paper_sell_store(tmp_path, now=SESSION_OPEN_NOW, consume=False)
    path = tmp_path / "paper_sell_arm_authority.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["expires_at"] = (SESSION_OPEN_NOW + timedelta(hours=12)).isoformat()
    path.write_text(json.dumps(doc), encoding="utf-8")
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    result = _run_chain(
        monkeypatch,
        tmp_path,
        fake=fake,
        trace=trace,
        seed=False,
        dispatch_now=closed_now,
    )
    assert fake.fake_broker_call_count == 0
    assert result["sell_payloads"] == []
    reasons = [
        e.get("reason")
        for e in trace.events
        if e.get("event") in ("validation_result", "begin_paper_sell_dispatch_result")
    ]
    assert any(
        isinstance(r, str) and "market_session" in r for r in reasons
    ), f"expected session denial, got {reasons}"
    with pytest.raises(PaperSellDispatchBlocked):
        NetworkFakeAlpacaBroker().submit_order(symbol="TSLA", qty=8, side="sell")


def test_rejected_malformed_order_zero_fake_sell(tmp_path, monkeypatch):
    """Missing limit/close is dropped by existing placement mapping (PRICE_UNAVAILABLE)."""
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    bad = _planned(price=None)
    result = _run_chain(
        monkeypatch, tmp_path, fake=fake, trace=trace, seed=True, planned=[bad]
    )
    assert result["sell_payloads"] == []
    assert fake.fake_broker_call_count == 0


def test_direct_boundary_rejects_do_not_call_fake(tmp_path, monkeypatch):
    """Negative contract cases on the same fake broker used by the integration path."""
    freeze_dispatch_clock(monkeypatch)
    seed_paper_sell_store(tmp_path, consume=True)
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    fake = NetworkFakeAlpacaBroker()
    with pytest.raises(PaperSellDispatchBlocked) as ei:
        fake.submit_order(symbol="AAPL", qty=8, side="sell")
    assert ei.value.reason == "paper_sell_permit_symbol_not_authorized"
    assert fake.fake_broker_call_count == 0
    with pytest.raises(PaperSellDispatchBlocked) as ei2:
        fake.submit_order(symbol="TSLA", qty=80, side="sell")
    assert ei2.value.reason == "paper_sell_permit_quantity_exceeds_authorized"
    assert fake.fake_broker_call_count == 0
    buy_out = fake.submit_order(symbol="AAPL", qty=1, side="buy")
    assert str(buy_out.get("side")).lower() == "buy"
    assert all(str(p.get("side")).lower() != "sell" for p in fake.submit_payloads)


def test_export_negative_fake_broker_call_counts(tmp_path_factory, monkeypatch):
    rows = []

    def _capture(name: str, fake: NetworkFakeAlpacaBroker, result, extra=None):
        reasons = [
            e.get("reason")
            for e in result["trace"].events
            if e.get("event") in ("validation_result", "begin_paper_sell_dispatch_result")
        ]
        rows.append(
            {
                "case": name,
                "fake_broker_mutation_count": fake.fake_broker_call_count,
                "sell_submit_count": len(result["sell_payloads"]),
                "buy_or_add_count": len(result["buy_payloads"]),
                "denial_reasons": [r for r in reasons if r],
                **(extra or {}),
            }
        )

    p = tmp_path_factory.mktemp("missing")
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    result = _run_chain(monkeypatch, p, fake=fake, trace=trace, seed=False)
    _capture("missing_authorization", fake, result)

    p = tmp_path_factory.mktemp("malformed_auth")
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    isolate_sell_store(monkeypatch, p)
    (p / "paper_sell_arm_authority.json").write_text("{not-json", encoding="utf-8")
    result = _run_chain(monkeypatch, p, fake=fake, trace=trace, seed=False)
    _capture("malformed_authorization", fake, result)

    p = tmp_path_factory.mktemp("expired")
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    seed_paper_sell_store(p, now=SESSION_OPEN_NOW, consume=False)
    path = p / "paper_sell_arm_authority.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["expires_at"] = (SESSION_OPEN_NOW - timedelta(minutes=1)).isoformat()
    path.write_text(json.dumps(doc), encoding="utf-8")
    result = _run_chain(monkeypatch, p, fake=fake, trace=trace, seed=False)
    _capture("expired_authorization", fake, result)

    p = tmp_path_factory.mktemp("wrong_sym")
    fake = NetworkFakeAlpacaBroker()
    fake.positions = [{"symbol": "AAPL", "qty": "8", "qty_available": "8"}]
    trace = OrderedTrace()
    result = _run_chain(
        monkeypatch,
        p,
        fake=fake,
        trace=trace,
        seed=True,
        planned=[_planned(symbol="AAPL", side="sell", qty=8, action="EXIT")],
    )
    _capture("wrong_symbol", fake, result)

    p = tmp_path_factory.mktemp("qty_exceed")
    fake = NetworkFakeAlpacaBroker()
    fake.positions = [{"symbol": "TSLA", "qty": "80", "qty_available": "80"}]
    trace = OrderedTrace()
    result = _run_chain(
        monkeypatch, p, fake=fake, trace=trace, seed=True, planned=[_planned(qty=80)]
    )
    _capture("quantity_exceeds_authorized", fake, result)

    p = tmp_path_factory.mktemp("non_reducing")
    fake = NetworkFakeAlpacaBroker()
    fake.positions = []
    trace = OrderedTrace()
    result = _run_chain(monkeypatch, p, fake=fake, trace=trace, seed=True)
    _capture("non_reducing_sell", fake, result)

    p = tmp_path_factory.mktemp("live")
    fake = NetworkFakeAlpacaBroker(mode="live")
    trace = OrderedTrace()
    result = _run_chain(monkeypatch, p, fake=fake, trace=trace, seed=True, mode="live")
    _capture("live_execution_attempt", fake, result)

    p = tmp_path_factory.mktemp("buy")
    fake = NetworkFakeAlpacaBroker()
    fake.positions = []
    trace = OrderedTrace()
    result = _run_chain(
        monkeypatch,
        p,
        fake=fake,
        trace=trace,
        seed=True,
        planned=[_planned(side="buy", action="ADD", qty=1, symbol="AAPL")],
    )
    _capture("buy_add_on_sell_path", fake, result)

    p = tmp_path_factory.mktemp("malformed_order")
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    result = _run_chain(
        monkeypatch, p, fake=fake, trace=trace, seed=True, planned=[_planned(price=None)]
    )
    _capture("malformed_order_missing_limit", fake, result)

    for row in rows:
        assert row["sell_submit_count"] == 0, row
        assert row["fake_broker_mutation_count"] == 0 or row["case"] == "buy_add_on_sell_path"

    ev_dir = _evidence_dir()
    ev_dir.mkdir(parents=True, exist_ok=True)
    (ev_dir / "05_negative_fake_broker_call_counts.json").write_text(
        json.dumps(
            {
                "accepted_case_fake_broker_sell_count": 1,
                "rejected_cases": rows,
                "all_rejected_sell_submits_zero": all(r["sell_submit_count"] == 0 for r in rows),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
