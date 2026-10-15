"""Manual portfolio paper exits stay off the defensive packet and production ledgers."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from dashboard import operator_modern
from services.defensive_exit_approval_readiness import (
    CLASS_AUTHORITY_READY,
    CLASS_NOT_SELECTED,
    apply_concentration_exit_precedence,
)
from services.manual_portfolio_exit import (
    GROSS_LABEL,
    REASON,
    UNAVAILABLE_HOLD,
    build_manual_exit_review,
    confirm_manual_paper_exit,
    provisional_gain_loss_text,
    review_with_quantity,
)
from services.paper_sell_arm_runtime_enforcement import (
    enforce_paper_sell_broker_submission,
)
from services.basis_unproven_hold import enforce_basis_unproven_hold_on_order_post

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = (
    ROOT / "data" / "results" / "paper_order_events.jsonl",
    ROOT / "data" / "results" / "performance_accounting_events.jsonl",
    ROOT / "data" / "governance" / "basis_unproven_hold_audit.jsonl",
    ROOT / "data" / "governance" / "basis_unproven_hold.json",
)


def _fingerprint(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


@pytest.fixture(scope="module", autouse=True)
def production_files_unchanged():
    before = {path: _fingerprint(path) for path in PRODUCTION}
    yield
    after = {path: _fingerprint(path) for path in PRODUCTION}
    assert after == before


class Broker:
    def __init__(self, symbol="AAPL", qty="10", price="20", explode=False):
        self.mode = "paper"
        self.base = "https://paper-api.alpaca.markets"
        self.symbol = symbol
        self.positions = [{"symbol": symbol, "qty": qty, "current_price": price}]
        self.orders = []
        self.gets = 0
        self.explode = explode

    def get_positions(self):
        self.gets += 1
        if self.explode:
            raise AssertionError("broker read")
        return list(self.positions)

    def submit_order(self, **kwargs):
        if self.explode:
            raise AssertionError("broker post")
        enforce_paper_sell_broker_submission(
            mode=self.mode,
            side=kwargs["side"],
            source="broker_alpaca.submit_order",
            symbol=kwargs["symbol"],
            qty=float(kwargs["qty"]),
            client_order_id=kwargs.get("client_order_id"),
        )
        enforce_basis_unproven_hold_on_order_post(
            {
                "symbol": kwargs["symbol"],
                "side": kwargs["side"],
                "qty": kwargs["qty"],
                "client_order_id": kwargs.get("client_order_id"),
            },
            source="broker_alpaca._post",
        )
        self.orders.append(kwargs)
        return {"id": "ord-1", "status": "accepted"}


def _row(symbol="AAPL", qty="10", price="20", average="12"):
    return {
        "symbol": symbol,
        "qty": qty,
        "current_price": price,
        "avg_entry_price": average,
    }


def _confirm(review, broker, store, **kwargs):
    return confirm_manual_paper_exit(
        review,
        broker=broker,
        store_root=store,
        live_execution_permitted=False,
        blocked_symbols=kwargs.pop("blocked_symbols", []),
        **kwargs,
    )


def test_full_and_partial_and_fractional_exits(tmp_path):
    full = build_manual_exit_review(_row(), blocked_symbols=[])
    assert full["selected_qty"] == "10"
    assert full["reason"] == REASON
    broker = Broker()
    done = _confirm(full, broker, tmp_path / "full")
    assert done["status"] == "submitted"
    assert done["broker_status"] == "accepted"
    assert done["submitted"] is True
    assert broker.orders[0]["qty"] == "10"
    assert broker.orders[0]["side"] == "sell"
    assert broker.orders[0]["symbol"] == "AAPL"

    partial = review_with_quantity(full, "4")
    partial_broker = Broker()
    partial_done = _confirm(partial, partial_broker, tmp_path / "partial")
    assert partial_done["status"] == "submitted"
    assert partial_broker.orders[0]["qty"] == "4"

    fractional_review = build_manual_exit_review(
        _row(qty="2.5", price="10"),
        blocked_symbols=[],
        selected_qty="0.25",
    )
    fractional_broker = Broker(qty="2.5", price="10")
    fractional = _confirm(fractional_review, fractional_broker, tmp_path / "fraction")
    assert fractional["status"] == "submitted"
    assert fractional_broker.orders[0]["qty"] == "0.25"


def test_invalid_above_held_and_cap_do_not_post(tmp_path):
    base = build_manual_exit_review(_row(qty="100", price="100"), blocked_symbols=[])
    broker = Broker(qty="100", price="100", explode=True)
    above = _confirm(review_with_quantity(base, "100.1"), broker, tmp_path / "above")
    zero = _confirm(review_with_quantity(base, "0"), broker, tmp_path / "zero")
    invalid = _confirm(review_with_quantity(base, "nope"), broker, tmp_path / "invalid")
    negative = _confirm(review_with_quantity(base, "-1"), broker, tmp_path / "negative")
    cap = _confirm(review_with_quantity(base, "51"), broker, tmp_path / "cap")
    assert above["status"] == "rejected"
    assert above["reason"] == "quantity_above_holdings"
    assert zero["reason"] == "quantity_invalid"
    assert invalid["reason"] == "quantity_invalid"
    assert negative["reason"] == "quantity_invalid"
    assert cap["reason"] == "quantity_exceeds_per_order_sell_notional_cap"
    assert broker.gets == 0
    assert broker.orders == []


def test_voo_and_xlk_are_blocked_with_no_post(tmp_path):
    broker = Broker(symbol="VOO", explode=True)
    for symbol in ("VOO", "XLK"):
        review = build_manual_exit_review(_row(symbol=symbol), blocked_symbols=["VOO", "XLK"])
        assert review["eligible"] is False
        assert review["unavailable_reason"] == UNAVAILABLE_HOLD
        result = _confirm(
            review,
            broker,
            tmp_path / symbol,
            blocked_symbols=["VOO", "XLK"],
        )
        assert result["status"] == "blocked"
        assert result["reason"] == UNAVAILABLE_HOLD
        assert result["submitted"] is False
    assert broker.orders == []
    assert broker.gets == 0


def test_consumed_authority_cannot_be_reused_and_confirm_is_idempotent(tmp_path):
    review = build_manual_exit_review(_row(), blocked_symbols=[])
    broker = Broker(explode=True)
    blocked = _confirm(
        review,
        broker,
        tmp_path / "reuse",
        supplied_authority={"status": "CONSUMED", "consumption_status": "CONSUMED"},
    )
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "supplied_authority_cannot_authorize_manual_exit"
    assert broker.orders == []

    live = Broker()
    store = tmp_path / "once"
    first = _confirm(review, live, store)
    second = _confirm(review, live, store)
    assert first["submitted"] is True
    assert first["broker_order_id"] == "ord-1"
    assert second["idempotent_replay"] is True
    assert second["broker_order_id"] == "ord-1"
    assert len(live.orders) == 1
    assert live.gets == 1


def test_stale_position_fails_closed_and_route_is_not_bypassed(tmp_path):
    review = build_manual_exit_review(_row(qty="10", price="20"), blocked_symbols=[])
    broker = Broker(qty="9", price="20")
    stale = _confirm(review, broker, tmp_path / "stale")
    assert stale["status"] == "stale"
    assert stale["reason"] == "position_or_price_changed_reopen_exit"
    assert broker.orders == []
    assert broker.gets == 1
    source = inspect.getsource(confirm_manual_paper_exit)
    assert "submit_order" in source
    assert "._post(" not in source
    assert "session.post" not in source


def test_gross_provisional_wording_and_googl_does_not_use_unavailable_sale():
    review = build_manual_exit_review(_row(symbol="GOOGL", qty="1", price="100", average="80"))
    text = provisional_gain_loss_text(review)
    assert text == f"{GROSS_LABEL} 20.00"
    lowered = text.lower()
    assert "realized" not in lowered
    assert "certified" not in lowered
    assert "guaranteed" not in lowered
    assert " net " not in f" {lowered} "
    missing = build_manual_exit_review(_row(symbol="GOOGL", average=None), blocked_symbols=[])
    assert missing["gross_provisional_gain_loss"] is None
    source = inspect.getsource(confirm_manual_paper_exit)
    assert "0e99c1320742c1c1c9f750b10b53437e80cdb06013c77171f4ed218a56db09b1" not in source


def test_automatic_trim_and_loss_exit_selection_remain_unchanged():
    names = ("HEAVY", "AAPL", "MSFT", "QQQ", "META")
    trim = apply_concentration_exit_precedence(
        [
            {
                "symbol": symbol,
                "action": "NONE",
                "classification": CLASS_NOT_SELECTED,
                "reason_code": "NO_ACTION_HOLD",
            }
            for symbol in names
        ],
        [
            {"symbol": "HEAVY", "current_position_qty": 400, "current_price": 10},
            {"symbol": "AAPL", "current_position_qty": 15, "current_price": 100},
            {"symbol": "MSFT", "current_position_qty": 15, "current_price": 100},
            {"symbol": "QQQ", "current_position_qty": 15, "current_price": 100},
            {"symbol": "META", "current_position_qty": 15, "current_price": 100},
        ],
        blocked_symbols=[],
    )
    assert trim["status"] == "applied"
    promoted = next(row for row in trim["classified"] if row["symbol"] == "HEAVY")
    assert promoted["action"] == "TRIM"
    assert promoted["reason_code"] != REASON

    loss_rows = [
        {
            "symbol": "LOSS",
            "action": "EXIT",
            "qty": 10,
            "classification": CLASS_AUTHORITY_READY,
            "reason_code": "EXIT_DRAWDOWN",
        }
    ]
    original = dict(loss_rows[0])
    loss = apply_concentration_exit_precedence(
        loss_rows,
        [{"symbol": "LOSS", "current_position_qty": 10, "current_price": 50}],
        blocked_symbols=[],
    )
    assert loss["status"] == "not_applicable"
    assert loss_rows[0] == original
    assert original["action"] == "EXIT"


def test_portfolio_page_keeps_exit_review_outside_the_holdings_table():
    page = inspect.getsource(operator_modern.render_portfolio_page)
    panel = inspect.getsource(operator_modern._render_manual_portfolio_exits)
    holdings = inspect.getsource(operator_modern._render_canonical_holdings)
    assert "_render_manual_portfolio_exits" in page
    assert "Confirm paper exit" in panel
    assert "Sell all" in panel
    assert "session_state" not in holdings
    assert "submit_order" not in holdings
