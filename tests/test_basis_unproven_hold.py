"""Temporary-fixture proof for the basis-unproven hold guard.

No production ledger writes, no broker network, no forward-opening append,
and no append to the production hold audit file.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

import services.basis_unproven_hold as hold_mod
from services.basis_unproven_hold import (
    APPROVAL_IDENTITY,
    AUDIT_RELATIVE,
    HOLD_SYMBOLS,
    POLICY_RELATIVE,
    REPORTING_LABEL,
    BasisUnprovenHoldBlocked,
    _audit_path,
    _events_path,
    _policy_path,
    canonical_override_sha256,
    policy_path,
)
from services.broker_alpaca import AlpacaBroker, broker_order_auth_context
from services.performance_accounting import (
    EVENT_FEE,
    EVENT_FILL_BUY,
    EVENT_FILL_SELL,
    FORMULAS,
    GROSS_PROVISIONAL,
    UNAVAILABLE,
    build_event,
    build_forward_opening_basis,
    reduce_events,
)

ROOT = Path(__file__).resolve().parents[1]
PROD_AUDIT = ROOT / AUDIT_RELATIVE
PROD_POLICY = ROOT / POLICY_RELATIVE
PROD_ACCOUNTING = ROOT / "data/results/performance_accounting_events.jsonl"
PROD_ORDERS = ROOT / "data/results/paper_order_events.jsonl"
PROD_ARM = ROOT / "data/governance/paper_sell_arm_authority.json"
OCT9_STATEMENT = "22a1a6f94d724ef57a06606e4ec80ca60137b0170695f6928d1456870a286cfe"
OCT8_STATEMENT = "1f9cd8ee8fb0f76a4c40ffd65aa3f5bf798993b9889c663cb4245ca84b9607ed"
BASELINED = ("AAPL", "AMD", "GOOGL", "INTC", "META", "QQQ")
ROUTE_NAMES = (
    "submit_order_buy",
    "submit_order_sell",
    "submit_order_sell_force_exit",
    "submit_market_order_buy",
    "submit_market_order_sell",
    "submit_limit_order_buy",
    "submit_limit_order_sell",
    "place_order_buy",
    "place_order_sell",
    "submit_order_notional_buy",
    "submit_order_notional_sell",
    "submit_order_notional_sell_force_exit",
    "place_order_notional",
    "submit_market_order_notional",
    "submit_bracket_order_notional_buy",
    "place_bracket_market_buy",
    "place_bracket_market_sell",
    "place_bracket_buy",
    "place_bracket_sell",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Response:
    status_code = 200
    text = '{"id":"fixture","status":"accepted"}'

    def json(self):
        return {"id": "fixture", "status": "accepted"}


class RecordingSession:
    def __init__(self) -> None:
        self.posts = []
        self.gets = []
        self.deletes = []

    def post(self, url, data=None, timeout=None):
        self.posts.append({"url": url, "data": data})
        return _Response()

    def get(self, url, params=None, timeout=None):
        self.gets.append(url)
        return _Response()

    def delete(self, url, timeout=None):
        self.deletes.append(url)
        return _Response()


def _broker():
    broker = AlpacaBroker.__new__(AlpacaBroker)
    session = RecordingSession()
    broker.mode = "paper"
    broker.base = "https://paper-api.alpaca.markets"
    broker.timeout = 1
    broker.session = session
    broker._auth_diagnostics = {}
    return broker, session


def _call_route(broker, symbol: str, route_name: str):
    if route_name == "submit_order_buy":
        return broker.submit_order(symbol=symbol, qty=1, side="buy")
    if route_name == "submit_order_sell":
        return broker.submit_order(symbol=symbol, qty=1, side="sell")
    if route_name == "submit_order_sell_force_exit":
        return broker.submit_order(symbol=symbol, qty=1, side="sell", force_exit_override=True)
    if route_name == "submit_market_order_buy":
        return broker.submit_market_order(symbol=symbol, qty=1, side="buy")
    if route_name == "submit_market_order_sell":
        return broker.submit_market_order(symbol=symbol, qty=1, side="sell")
    if route_name == "submit_limit_order_buy":
        return broker.submit_limit_order(symbol=symbol, qty=1, side="buy", limit_price=10)
    if route_name == "submit_limit_order_sell":
        return broker.submit_limit_order(symbol=symbol, qty=1, side="sell", limit_price=10)
    if route_name == "place_order_buy":
        return broker.place_order(symbol=symbol, qty=1, side="buy")
    if route_name == "place_order_sell":
        return broker.place_order(symbol=symbol, qty=1, side="sell")
    if route_name == "submit_order_notional_buy":
        return broker.submit_order_notional(symbol=symbol, notional=10, side="buy")
    if route_name == "submit_order_notional_sell":
        return broker.submit_order_notional(symbol=symbol, notional=10, side="sell")
    if route_name == "submit_order_notional_sell_force_exit":
        return broker.submit_order_notional(
            symbol=symbol, notional=10, side="sell", force_exit_override=True
        )
    if route_name == "place_order_notional":
        return broker.place_order_notional(symbol=symbol, notional=10, side="buy")
    if route_name == "submit_market_order_notional":
        return broker.submit_market_order_notional(symbol=symbol, notional=10, side="sell")
    if route_name == "submit_bracket_order_notional_buy":
        return broker.submit_bracket_order_notional(
            symbol=symbol,
            notional=10,
            side="buy",
            take_profit={"limit_price": 12},
            stop_loss={"stop_price": 8},
        )
    if route_name == "place_bracket_market_buy":
        return broker.place_bracket_market(
            symbol=symbol, qty=1, side="buy", take_profit_price=12, stop_loss_price=8
        )
    if route_name == "place_bracket_market_sell":
        return broker.place_bracket_market(
            symbol=symbol, qty=1, side="sell", take_profit_price=8, stop_loss_price=12
        )
    if route_name == "place_bracket_buy":
        return broker.place_bracket(
            symbol=symbol, side="buy", qty=1, take_profit_price=12, stop_loss_price=8
        )
    if route_name == "place_bracket_sell":
        return broker.place_bracket(
            symbol=symbol, side="sell", qty=1, take_profit_price=8, stop_loss_price=12
        )
    raise AssertionError(route_name)


@pytest.fixture(autouse=True)
def _isolate_hold(tmp_path, monkeypatch):
    from tests._phase_198i_sell_fakes import block_real_network

    block_real_network(monkeypatch)
    policy = tmp_path / "basis_unproven_hold.json"
    audit = tmp_path / "basis_unproven_hold_audit.jsonl"
    events = tmp_path / "events.jsonl"
    policy.write_bytes(PROD_POLICY.read_bytes())
    audit.write_bytes(b"")
    events.write_bytes(b"")
    tokens = (
        _policy_path.set(str(policy)),
        _audit_path.set(str(audit)),
        _events_path.set(str(events)),
    )
    before = {
        "audit": PROD_AUDIT.read_bytes(),
        "accounting": PROD_ACCOUNTING.read_bytes(),
        "orders": PROD_ORDERS.read_bytes(),
        "arm": PROD_ARM.read_bytes(),
        "policy": PROD_POLICY.read_bytes(),
    }
    from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT

    _ACTIVE_PERMIT.set(None)
    yield {"policy": policy, "audit": audit, "events": events}
    _ACTIVE_PERMIT.set(None)
    for token, var in zip(tokens, (_policy_path, _audit_path, _events_path)):
        var.reset(token)
    assert PROD_AUDIT.read_bytes() == before["audit"]
    assert PROD_ACCOUNTING.read_bytes() == before["accounting"]
    assert PROD_ORDERS.read_bytes() == before["orders"]
    assert PROD_ARM.read_bytes() == before["arm"]
    assert PROD_POLICY.read_bytes() == before["policy"]
    assert b"FORWARD_OPENING_BASIS" not in PROD_ACCOUNTING.read_bytes()


def _stub_pre_post_gates(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.paper_sell_arm_runtime_enforcement.enforce_paper_sell_broker_submission",
        lambda **_k: None,
    )
    monkeypatch.setattr(
        "services.triton_execution_gate.enforce_broker_submission_gate",
        lambda **_k: {"allowed": True, "decision": "ALLOW"},
    )


def test_hold_runs_inside_post_before_session_post():
    src = inspect.getsource(AlpacaBroker._post)
    assert src.index("enforce_basis_unproven_hold_on_order_post") < src.index("session.post")
    assert "enforce_basis_unproven_hold_on_order_post" not in inspect.getsource(AlpacaBroker._get)
    assert "enforce_basis_unproven_hold_on_order_post" not in inspect.getsource(
        AlpacaBroker._delete
    )


def test_production_policy_binds_the_pending_statement():
    document = json.loads(PROD_POLICY.read_text(encoding="utf-8"))
    assert document["schema"] == "BASIS-UNPROVEN-HOLD-1"
    assert document["status"] == "ACTIVE"
    assert document["approval_identity"] == APPROVAL_IDENTITY
    assert document["created_at_utc"] == "2026-10-09T01:45:00Z"
    assert document["bound_partial_statement_sha256"] == OCT9_STATEMENT
    assert document["supersedes_partial_statement_sha256"] == OCT8_STATEMENT
    assert not hasattr(hold_mod, "BOUND_STATEMENT_SHA256")
    assert document["bound_statement_status_at_binding"] == "PENDING_FRESH_OPERATOR_APPROVAL"
    assert {row["symbol"] for row in document["symbols"]} == set(HOLD_SYMBOLS)
    assert all(row["classification"] == "BASIS_UNPROVEN_HOLD" for row in document["symbols"])
    assert document["operator_overrides"] == []
    assert PROD_AUDIT.read_bytes() == b""
    assert policy_path() != PROD_POLICY


@pytest.mark.parametrize("symbol", ("VOO", "XLK"))
@pytest.mark.parametrize("route_name", ROUTE_NAMES)
def test_every_hold_order_route_stops_before_broker_post(
    symbol, route_name, monkeypatch, _isolate_hold
):
    _stub_pre_post_gates(monkeypatch)
    broker, session = _broker()
    with pytest.raises(BasisUnprovenHoldBlocked) as caught:
        _call_route(broker, symbol, route_name)
    assert caught.value.reason == "basis_unproven_hold"
    assert caught.value.record["reporting_label"] == REPORTING_LABEL
    assert caught.value.record["broker_called"] is False
    assert caught.value.record["symbol"] == symbol
    assert session.posts == []
    audit_lines = _isolate_hold["audit"].read_text(encoding="utf-8").splitlines()
    assert len(audit_lines) == 1
    record = json.loads(audit_lines[0])
    assert record["broker_called"] is False
    assert record["bound_statement_sha256"] == OCT9_STATEMENT
    assert record["policy_sha256"] == _sha(_isolate_hold["policy"])


@pytest.mark.parametrize("symbol", BASELINED)
@pytest.mark.parametrize("side", ("buy", "sell"))
def test_baselined_symbols_pass_the_hold_and_can_post(symbol, side, monkeypatch, _isolate_hold):
    _stub_pre_post_gates(monkeypatch)
    broker, session = _broker()
    out = broker.submit_order(symbol=symbol, qty=1, side=side)
    assert out["status"] == "accepted"
    assert len(session.posts) == 1
    assert json.loads(session.posts[0]["data"])["symbol"] == symbol
    assert _isolate_hold["audit"].read_bytes() == b""


def test_get_and_delete_are_unchanged(monkeypatch):
    _stub_pre_post_gates(monkeypatch)
    broker, session = _broker()
    broker._get("/v2/orders")
    broker._get("/v2/positions/VOO")
    broker._delete("/v2/orders/voo-open-order")
    broker._delete("/v2/orders")
    assert session.posts == []
    assert session.gets == [
        "https://paper-api.alpaca.markets/v2/orders",
        "https://paper-api.alpaca.markets/v2/positions/VOO",
    ]
    assert session.deletes == [
        "https://paper-api.alpaca.markets/v2/orders/voo-open-order",
        "https://paper-api.alpaca.markets/v2/orders",
    ]


def test_missing_symbol_fails_closed_before_post():
    broker, session = _broker()
    with broker_order_auth_context():
        with pytest.raises(BasisUnprovenHoldBlocked) as caught:
            broker._post("/v2/orders", {"side": "buy", "type": "market"})
    assert caught.value.reason == "basis_unproven_hold_symbol_unreadable"
    assert session.posts == []


def test_missing_policy_blocks_hold_symbols_only(_isolate_hold):
    _isolate_hold["policy"].unlink()
    broker, session = _broker()
    with broker_order_auth_context():
        with pytest.raises(BasisUnprovenHoldBlocked) as caught:
            broker._post("/v2/orders", {"symbol": "VOO", "side": "buy"})
        assert caught.value.reason == "basis_unproven_hold_policy_missing"
        broker._post("/v2/orders", {"symbol": "AAPL", "side": "buy"})
    assert len(session.posts) == 1


def test_malformed_policy_blocks_hold_symbols_only(_isolate_hold):
    _isolate_hold["policy"].write_text("{", encoding="utf-8")
    broker, session = _broker()
    with broker_order_auth_context():
        with pytest.raises(BasisUnprovenHoldBlocked) as caught:
            broker._post("/v2/orders", {"symbol": "XLK", "side": "sell"})
        assert caught.value.reason == "basis_unproven_hold_policy_malformed"
        broker._post("/v2/orders", {"symbol": "QQQ", "side": "sell"})
    assert len(session.posts) == 1


def test_invalid_statement_hash_blocks_hold_symbols_only(_isolate_hold):
    document = json.loads(_isolate_hold["policy"].read_text(encoding="utf-8"))
    document["bound_partial_statement_sha256"] = "not-a-statement-hash"
    _isolate_hold["policy"].write_text(json.dumps(document), encoding="utf-8")
    broker, session = _broker()
    with broker_order_auth_context():
        with pytest.raises(BasisUnprovenHoldBlocked) as caught:
            broker._post("/v2/orders", {"symbol": "VOO", "side": "sell"})
        assert caught.value.reason == "basis_unproven_hold_policy_invalid_hash"
        assert caught.value.record["bound_statement_sha256"] is None
        broker._post("/v2/orders", {"symbol": "META", "side": "buy"})
    assert len(session.posts) == 1


def test_production_head_mismatch_fails_closed(monkeypatch):
    token = _policy_path.set(None)
    try:
        monkeypatch.setattr(
            hold_mod,
            "_committed_policy_bytes",
            lambda: PROD_POLICY.read_bytes().replace(
                OCT9_STATEMENT.encode("ascii"),
                b"a" * 64,
            ),
        )
        broker, session = _broker()
        with broker_order_auth_context():
            with pytest.raises(BasisUnprovenHoldBlocked) as caught:
                broker._post("/v2/orders", {"symbol": "VOO", "side": "buy"})
            assert caught.value.reason == "basis_unproven_hold_policy_mismatched"
            assert caught.value.record["bound_statement_sha256"] == OCT9_STATEMENT
            broker._post("/v2/orders", {"symbol": "AAPL", "side": "buy"})
        assert len(session.posts) == 1
    finally:
        _policy_path.reset(token)


def test_production_head_git_failure_fails_closed(monkeypatch):
    token = _policy_path.set(None)
    try:
        monkeypatch.setattr(hold_mod, "_committed_policy_bytes", lambda: None)
        broker, session = _broker()
        with broker_order_auth_context():
            with pytest.raises(BasisUnprovenHoldBlocked) as caught:
                broker._post("/v2/orders", {"symbol": "XLK", "side": "sell"})
            assert caught.value.reason == "basis_unproven_hold_policy_mismatched"
        assert session.posts == []
    finally:
        _policy_path.reset(token)


def test_matching_head_bytes_do_not_release_hold_symbols(monkeypatch):
    token = _policy_path.set(None)
    try:
        monkeypatch.setattr(hold_mod, "_committed_policy_bytes", lambda: PROD_POLICY.read_bytes())
        broker, session = _broker()
        with broker_order_auth_context():
            with pytest.raises(BasisUnprovenHoldBlocked) as caught:
                broker._post("/v2/orders", {"symbol": "VOO", "side": "sell"})
            assert caught.value.reason == "basis_unproven_hold"
            assert caught.value.record["bound_statement_sha256"] == OCT9_STATEMENT
        assert session.posts == []
    finally:
        _policy_path.reset(token)


def test_removing_a_symbol_from_the_file_does_not_release_it(_isolate_hold):
    document = json.loads(_isolate_hold["policy"].read_text(encoding="utf-8"))
    document["symbols"] = [row for row in document["symbols"] if row["symbol"] != "VOO"]
    _isolate_hold["policy"].write_text(json.dumps(document), encoding="utf-8")
    broker, session = _broker()
    with broker_order_auth_context():
        for symbol in ("VOO", "XLK"):
            with pytest.raises(BasisUnprovenHoldBlocked) as caught:
                broker._post("/v2/orders", {"symbol": symbol, "side": "buy"})
            assert caught.value.reason == "basis_unproven_hold_policy_mismatched"
    assert session.posts == []


def test_unreadable_policy_blocks_hold_symbols(monkeypatch, _isolate_hold):
    real_read = Path.read_bytes

    def _unreadable(self):
        if self == _isolate_hold["policy"]:
            raise OSError("policy unreadable")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", _unreadable)
    broker, session = _broker()
    with broker_order_auth_context():
        with pytest.raises(BasisUnprovenHoldBlocked) as caught:
            broker._post("/v2/orders", {"symbol": "XLK", "side": "buy"})
        assert caught.value.reason == "basis_unproven_hold_policy_unreadable"
        broker._post("/v2/orders", {"symbol": "INTC", "side": "buy"})
    assert len(session.posts) == 1


def test_audit_unwritable_fails_closed_without_post(tmp_path, _isolate_hold):
    audit_dir = tmp_path / "audit-dir"
    audit_dir.mkdir()
    token = _audit_path.set(str(audit_dir))
    try:
        broker, session = _broker()
        with broker_order_auth_context():
            with pytest.raises(BasisUnprovenHoldBlocked) as caught:
                broker._post("/v2/orders", {"symbol": "VOO", "side": "sell"})
        assert caught.value.reason == "basis_unproven_hold_audit_unwritable"
        assert caught.value.record["broker_called"] is False
        assert session.posts == []
        assert _isolate_hold["audit"].read_bytes() == b""
    finally:
        _audit_path.reset(token)


def test_published_opening_releases_only_that_symbol(monkeypatch, _isolate_hold):
    def _refuse_append(*_a, **_k):
        raise AssertionError("forward opening must not be appended")

    monkeypatch.setattr(
        "services.performance_accounting.append_forward_opening_basis",
        _refuse_append,
    )
    event = build_forward_opening_basis(
        symbol="VOO",
        qty="2",
        opening_basis="1422.780000",
        as_of_utc="2026-10-09T02:00:00Z",
        evidence_sha256=OCT9_STATEMENT,
        evidence_ref="fixture-ledger-only",
    )
    _isolate_hold["events"].write_text(json.dumps(event) + "\n", encoding="utf-8")
    broker, session = _broker()
    with broker_order_auth_context():
        broker._post("/v2/orders", {"symbol": "VOO", "side": "sell", "qty": 1})
        with pytest.raises(BasisUnprovenHoldBlocked) as caught:
            broker._post("/v2/orders", {"symbol": "XLK", "side": "sell", "qty": 1})
    assert caught.value.reason == "basis_unproven_hold"
    assert len(session.posts) == 1
    assert json.loads(session.posts[0]["data"])["symbol"] == "VOO"


def test_symbol_specific_override_releases_only_that_symbol(_isolate_hold):
    document = json.loads(_isolate_hold["policy"].read_text(encoding="utf-8"))
    override = {
        "symbol": "XLK",
        "approval_identity": APPROVAL_IDENTITY,
        "authorized_at_utc": "2026-10-09T02:05:00Z",
        "reason": "fixture override for one hold symbol",
        "authorization_reference": "fixture-override-xlk",
    }
    override["override_sha256"] = canonical_override_sha256(override)
    document["operator_overrides"] = [override]
    _isolate_hold["policy"].write_text(json.dumps(document), encoding="utf-8")
    broker, session = _broker()
    with broker_order_auth_context():
        broker._post("/v2/orders", {"symbol": "XLK", "side": "buy", "qty": 1})
        with pytest.raises(BasisUnprovenHoldBlocked) as caught:
            broker._post("/v2/orders", {"symbol": "VOO", "side": "buy", "qty": 1})
    assert caught.value.reason == "basis_unproven_hold"
    assert len(session.posts) == 1
    assert json.loads(session.posts[0]["data"])["symbol"] == "XLK"


def test_override_hash_mismatch_does_not_release(_isolate_hold):
    document = json.loads(_isolate_hold["policy"].read_text(encoding="utf-8"))
    document["operator_overrides"] = [
        {
            "symbol": "VOO",
            "approval_identity": APPROVAL_IDENTITY,
            "authorized_at_utc": "2026-10-09T02:05:00Z",
            "reason": "bad hash",
            "authorization_reference": "fixture-bad",
            "override_sha256": "0" * 64,
        }
    ]
    _isolate_hold["policy"].write_text(json.dumps(document), encoding="utf-8")
    broker, session = _broker()
    with broker_order_auth_context():
        with pytest.raises(BasisUnprovenHoldBlocked) as caught:
            broker._post("/v2/orders", {"symbol": "VOO", "side": "sell"})
    assert caught.value.reason == "basis_unproven_hold"
    assert session.posts == []


def test_consumed_sell_permit_and_force_exit_do_not_bypass(tmp_path, monkeypatch):
    from tests._phase_198i_sell_fakes import (
        freeze_dispatch_clock,
        isolate_sell_store,
        seed_paper_sell_store,
        stub_unrelated_pre_gates,
    )

    freeze_dispatch_clock(monkeypatch)
    seeded = seed_paper_sell_store(tmp_path, symbol="VOO", qty=2.0, price=100.0, consume=True)
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    assert seeded["guard"].consumed is True
    broker, session = _broker()
    with pytest.raises(BasisUnprovenHoldBlocked) as caught:
        broker.submit_order(symbol="VOO", qty=2, side="sell", force_exit_override=True)
    assert caught.value.reason == "basis_unproven_hold"
    assert session.posts == []


def test_gross_provisional_is_not_labeled_net():
    buy = build_event(
        EVENT_FILL_BUY,
        event_timestamp="2026-10-09T14:00:00Z",
        source="fixture",
        body={
            "symbol": "AAPL",
            "qty": "1",
            "price": "100",
            "broker_order_id": "ord-aapl-1",
            "broker_fill_id": "fill-buy-1",
        },
    )
    sell = build_event(
        EVENT_FILL_SELL,
        event_timestamp="2026-10-09T15:00:00Z",
        source="fixture",
        body={
            "symbol": "AAPL",
            "qty": "1",
            "price": "110",
            "broker_order_id": "ord-aapl-1",
            "broker_fill_id": "fill-sell-1",
        },
    )
    gross = reduce_events([buy, sell])["sales"][0]
    assert gross["gross_realized_pl_label"] == GROSS_PROVISIONAL
    assert gross["realized_pl"] == UNAVAILABLE
    assert gross["realized_pl_kind"] == UNAVAILABLE
    assert gross["fees_status"] == "PENDING"
    assert "net-P&L-certified" not in json.dumps(gross)
    fee = build_event(
        EVENT_FEE,
        event_timestamp="2026-10-09T15:00:01Z",
        source="fixture",
        body={
            "amount": "1.00",
            "activity_id": "fee-aapl-1",
            "fee_component": "REG",
            "allocates_to_event_id": sell["event_id"],
            "symbol": "AAPL",
        },
    )
    net = reduce_events([buy, sell, fee])["sales"][0]
    assert net["realized_pl_kind"] == "NET"
    assert net["fees_status"] == "ATTRIBUTED"
    assert net["gross_realized_pl_label"] == "GROSS"
    assert "NET" in FORMULAS["realized_pl"]
    assert "UNAVAILABLE" in FORMULAS["realized_pl"]
    assert "GROSS_PROVISIONAL" in FORMULAS["gross_realized_pl"]
    assert "never treated as net" in FORMULAS["gross_realized_pl"]


def test_ledger_hashes_match_the_pre_implementation_bytes():
    assert _sha(PROD_ACCOUNTING) == (
        "2777abc9e838333a946263ee0a18347460c0f5f364565f9f9773c1b8a357488a"
    )
    assert _sha(PROD_ORDERS) == ("f471049c339a3fc4569487f6654aa98e353c7e1bb76c8fff82194538f32ec7cf")
    assert _sha(PROD_AUDIT) == hashlib.sha256(b"").hexdigest()
