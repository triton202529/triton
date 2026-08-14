"""Phase 198I — close the 15 unresolved broker routes with execution evidence.

PAPER ONLY. Fake broker at the HTTP boundary. No real broker, no live execution.
Does not modify the verified paper-runtime/session-gating correction.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import os
from pathlib import Path

import pytest

from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    SESSION_OPEN_NOW,
    NetworkFakeAlpacaBroker,
    OrderedTrace,
    block_real_network,
    freeze_dispatch_clock,
    install_boundary_trace,
    isolate_sell_store,
    seed_paper_sell_store,
    stub_unrelated_pre_gates,
)
from services.paper_sell_arm_runtime_enforcement import PaperSellDispatchBlocked

ROOT = Path(__file__).resolve().parents[1]


def _evidence_dir() -> Path:
    override = os.environ.get("TRITON_198I_EVIDENCE_OUT")
    if override:
        return Path(override)
    return ROOT / "data" / "results" / "phase_198i_sell_remediation_20260814"


FORMERLY_UNRESOLVED = [
    "R3",
    "R4",
    "R5",
    "R6",
    "R23",
    "R33",
    "R34",
    "R36",
    "R37",
    "R38",
    "R39",
    "R40",
    "R41",
    "R42",
    "R43",
]


def _load_inventory_mod():
    path = ROOT / "scripts" / "evidence" / "phase_198i_sell_remediation_inventory.py"
    spec = importlib.util.spec_from_file_location("phase_198i_sell_remediation_inventory", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module", autouse=True)
def _materialize_inventory():
    _load_inventory_mod().write_inventory(_evidence_dir())


@pytest.fixture(autouse=True)
def _safety(monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    block_real_network(monkeypatch)
    from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT

    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)


def test_r40_argparse_help_is_non_execution():
    src = (ROOT / "services/place_live_orders.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    help_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in getattr(node, "keywords", []) or []:
                if kw.arg == "help" and isinstance(kw.value, ast.Constant):
                    if "never call broker.submit_order" in str(kw.value.value):
                        help_nodes.append(node)
    assert help_nodes, "R40 help string not found as argparse help="
    for node in help_nodes:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        assert name == "add_argument"


def test_r3_r6_page_submit_uses_gated_alpaca_broker(tmp_path, monkeypatch):
    freeze_dispatch_clock(monkeypatch)
    seed_paper_sell_store(tmp_path, consume=True)
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    for rel in ("pages/manual_order_desk.py", "pages_order_desk.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "AlpacaBroker" in src
        assert "broker.submit_order(" in src
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    install_boundary_trace(monkeypatch, trace)
    out = fake.submit_order(symbol="TSLA", qty=8, side="sell", order_type="market")
    assert out.get("status") == "accepted"
    assert len(fake.submit_payloads) == 1
    assert fake.submit_payloads[0]["side"] == "sell"
    assert "hardened_sell_boundary_entered" in trace.names()


def test_r3_r6_page_sell_blocked_without_permit(monkeypatch):
    fake = NetworkFakeAlpacaBroker()
    with pytest.raises(PaperSellDispatchBlocked) as ei:
        fake.submit_order(symbol="TSLA", qty=8, side="sell")
    assert ei.value.reason == "paper_sell_requires_hardened_dispatch_permit"
    assert fake.fake_broker_call_count == 0


def test_r4_r5_r23_cancel_routes_are_non_sell_and_gated(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    sell_fake = NetworkFakeAlpacaBroker(order_side_by_id={"oid-sell-1": "sell"})
    sell_fake.open_orders = [
        {"id": "oid-sell-1", "side": "sell", "symbol": "TSLA", "qty": "8"}
    ]
    with pytest.raises(PaperSellDispatchBlocked):
        sell_fake.cancel_order("oid-sell-1")
    assert sell_fake.deletes == []
    with pytest.raises(PaperSellDispatchBlocked):
        sell_fake.cancel_all_orders()
    assert sell_fake.deletes == []
    buy_fake = NetworkFakeAlpacaBroker(order_side_by_id={"oid-buy-1": "buy"})
    buy_fake.open_orders = [{"id": "oid-buy-1", "side": "buy", "symbol": "AAPL", "qty": "1"}]
    out = buy_fake.cancel_order("oid-buy-1")
    assert out.get("status") == "canceled"
    assert buy_fake.deletes == ["/v2/orders/oid-buy-1"]
    src = (ROOT / "scripts/opening_sequence.py").read_text(encoding="utf-8")
    assert "AlpacaBroker" in src and "cancel_all_orders()" in src


def test_r33_cancel_stale_is_cancel_only_through_alpaca_broker():
    src = (ROOT / "services/cancel_stale.py").read_text(encoding="utf-8")
    assert "AlpacaBroker" in src
    assert "b.cancel_order(" in src
    assert "submit_order" not in src


def test_r34_execute_cycle_cancel_open_orders(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    from services.execute_cycle import cancel_open_orders

    sell_fake = NetworkFakeAlpacaBroker(order_side_by_id={"oid-1": "sell"})
    cancelled, failed = cancel_open_orders(
        sell_fake, [{"id": "oid-1", "side": "sell", "symbol": "TSLA"}]
    )
    assert cancelled == 0
    assert failed == 1
    assert sell_fake.deletes == []
    buy_fake = NetworkFakeAlpacaBroker(order_side_by_id={"oid-2": "buy"})
    cancelled, failed = cancel_open_orders(
        buy_fake, [{"id": "oid-2", "side": "buy", "symbol": "AAPL"}]
    )
    assert cancelled == 1
    assert failed == 0
    assert buy_fake.deletes == ["/v2/orders/oid-2"]


def test_r36_r37_manage_open_orders_sell_replace_hits_boundary(tmp_path, monkeypatch):
    """R36 cancel and R37 submit both hit the canonical boundary.

    One permit is one-use per symbol: cancel of a SELL consumes it, so the
    replace submit is proven separately with a fresh permit. Production
    replace is fail-closed on the second mutation rather than a bypass.
    """
    src = (ROOT / "services/manage_open_orders.py").read_text(encoding="utf-8")
    assert "broker.cancel_order(oid)" in src
    assert "broker.submit_order(" in src

    freeze_dispatch_clock(monkeypatch)
    seed_paper_sell_store(tmp_path, consume=True)
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    fake = NetworkFakeAlpacaBroker(order_side_by_id={"oid-sell-reprice": "sell"})
    trace = OrderedTrace()
    install_boundary_trace(monkeypatch, trace)
    cancel_out = fake.cancel_order("oid-sell-reprice")
    assert cancel_out.get("status") == "canceled"
    assert fake.deletes == ["/v2/orders/oid-sell-reprice"]
    assert "hardened_sell_boundary_entered" in trace.names()
    with pytest.raises(PaperSellDispatchBlocked) as ei:
        fake.submit_order(
            symbol="TSLA",
            qty=8,
            side="sell",
            order_type="limit",
            time_in_force="day",
            limit_price=331.0,
            client_order_id="triton-om-test",
        )
    assert ei.value.reason in {
        "paper_sell_permit_exhausted_or_finalized",
        "paper_sell_permit_symbol_already_submitted",
    }
    assert fake.submit_payloads == []


def test_r37_replace_submit_hits_boundary(tmp_path, monkeypatch):
    freeze_dispatch_clock(monkeypatch)
    seed_paper_sell_store(tmp_path, consume=True)
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    install_boundary_trace(monkeypatch, trace)
    resp = fake.submit_order(
        symbol="TSLA",
        qty=8,
        side="sell",
        order_type="limit",
        time_in_force="day",
        limit_price=331.0,
        client_order_id="triton-om-test",
    )
    assert resp.get("status") == "accepted"
    assert len(fake.submit_payloads) == 1
    assert fake.submit_payloads[0]["side"] == "sell"
    assert "hardened_sell_boundary_entered" in trace.names()


def test_r37_sell_replace_blocked_without_permit(monkeypatch):
    fake = NetworkFakeAlpacaBroker()
    with pytest.raises(PaperSellDispatchBlocked):
        fake.submit_order(symbol="TSLA", qty=8, side="sell", order_type="limit", limit_price=331.0)
    assert fake.submit_payloads == []


def test_r38_run_cancel_sell_fail_closed(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    src = (ROOT / "services/manage_open_orders.py").read_text(encoding="utf-8")
    assert "br.cancel_order(oid)" in src
    fake = NetworkFakeAlpacaBroker(order_side_by_id={"stale-1": "sell"})
    with pytest.raises(PaperSellDispatchBlocked):
        fake.cancel_order("stale-1")
    assert fake.deletes == []


def test_r39_market_open_cycle_cancel_uses_alpaca_broker(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    fake = NetworkFakeAlpacaBroker(order_side_by_id={"oid-1": "sell"})
    fake.open_orders = [{"id": "oid-1", "side": "sell", "symbol": "TSLA", "qty": "8"}]

    def _factory(*_a, **_k):
        return fake

    monkeypatch.setattr("services.broker_alpaca.AlpacaBroker", _factory)
    from services.market_open_cycle import _cancel_open_orders

    result = _cancel_open_orders("paper")
    assert result.ok is True
    assert fake.deletes == []
    assert "failed=1" in (result.detail or "")


def test_r41_duplicate_sell_cancel_fail_closed(monkeypatch):
    src = (ROOT / "services/place_live_orders.py").read_text(encoding="utf-8")
    assert "broker.cancel_order(oid)" in src
    fake = NetworkFakeAlpacaBroker(order_side_by_id={"dup-1": "sell"})
    with pytest.raises(PaperSellDispatchBlocked):
        fake.cancel_order("dup-1")
    assert fake.deletes == []


def test_r43_position_protector_place_bracket_sell(tmp_path, monkeypatch):
    freeze_dispatch_clock(monkeypatch)
    seed_paper_sell_store(tmp_path, consume=True)
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    fake = NetworkFakeAlpacaBroker()
    trace = OrderedTrace()
    install_boundary_trace(monkeypatch, trace)
    from services.position_protector import place_bracket

    result = place_bracket(
        broker=fake,
        symbol="TSLA",
        side="sell",
        qty=8,
        entry_type="market",
        stop_loss_price=340.0,
        take_profit_price=320.0,
        client_order_id="coid-bracket",
    )
    assert result.ok, result.message
    assert len(fake.submit_payloads) == 1
    assert fake.submit_payloads[0]["side"] == "sell"
    assert "hardened_sell_boundary_entered" in trace.names()


def test_r43_place_bracket_sell_blocked_without_permit(monkeypatch):
    fake = NetworkFakeAlpacaBroker()
    from services.position_protector import place_bracket

    result = place_bracket(
        broker=fake,
        symbol="TSLA",
        side="sell",
        qty=8,
        entry_type="market",
        stop_loss_price=340.0,
        take_profit_price=320.0,
    )
    assert result.ok is False
    assert fake.submit_payloads == []


def test_r42_submit_site_is_real_place_live_orders_loop():
    src = (ROOT / "services/place_live_orders.py").read_text(encoding="utf-8")
    assert "resp = broker.submit_order(" in src
    assert "enforce_and_consume_paper_sells_before_submit" in src
    idx = src.find("resp = broker.submit_order(")
    assert "never call" not in src[idx - 80 : idx]


def test_formerly_unresolved_are_exactly_fifteen():
    assert len(FORMERLY_UNRESOLVED) == 15
    assert FORMERLY_UNRESOLVED == [
        "R3",
        "R4",
        "R5",
        "R6",
        "R23",
        "R33",
        "R34",
        "R36",
        "R37",
        "R38",
        "R39",
        "R40",
        "R41",
        "R42",
        "R43",
    ]


def test_inventory_file_reconciles_to_61():
    path = _evidence_dir() / "02_broker_route_inventory.csv"
    assert path.is_file(), f"canonical inventory missing: {path}"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 61
    allowed = {
        "VERIFIED_HARDENED_SELL_ROUTE",
        "VERIFIED_NON_SELL_ROUTE",
        "VERIFIED_NON_EXECUTION_ROUTE",
        "VERIFIED_DEAD_OR_UNREACHABLE",
        "VERIFIED_TEST_OR_FAKE_ONLY",
        "BLOCKED_UNRESOLVED",
    }
    disp = [r["final_disposition"] for r in rows]
    assert all(d in allowed for d in disp)
    assert "BLOCKED_UNRESOLVED" not in disp
    unresolved = set(FORMERLY_UNRESOLVED)
    for row in rows:
        if row["route_id"] in unresolved:
            assert row["prior_disposition"] == "Unresolved"
            assert row["final_disposition"] != "BLOCKED_UNRESOLVED"
