"""Portfolio effective action stays separate from the lifecycle model action."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone

from dashboard import operator_modern
from services.portfolio_effective_action import (
    BASIS_UNPROVEN_HOLD_STATUS,
    NO_PERMITTED_TRIM,
    PENDING_SELL_REVIEW,
    REASON_BASIS_UNPROVEN_HOLD,
    REASON_EXCESS_CONCENTRATION,
    resolve_portfolio_effective_actions,
)

NOW = datetime(2026, 10, 14, 23, 11, tzinfo=timezone.utc)
CONSUMED = {
    "status": "CONSUMED",
    "consumption_status": "CONSUMED",
    "consumed_at": "2026-10-14T19:25:34.654280+00:00",
    "expires_at": "2026-10-14T19:51:37.896242+00:00",
    "approved_symbols": ["BITO"],
    "orders_submitted": False,
}


def _resolve(rows, lifecycle, **kwargs):
    return resolve_portfolio_effective_actions(
        rows,
        lifecycle,
        blocked_symbols=kwargs.pop("blocked_symbols", []),
        now=NOW,
        **kwargs,
    )


def _breach_rows():
    return [
        {"symbol": "BITO", "qty": 400, "current_price": 10.0},
        {"symbol": "AAPL", "qty": 15, "current_price": 100.0},
        {"symbol": "MSFT", "qty": 15, "current_price": 100.0},
        {"symbol": "QQQ", "qty": 15, "current_price": 100.0},
        {"symbol": "META", "qty": 15, "current_price": 100.0},
    ]


def test_lifecycle_hold_with_concentration_breach_displays_effective_trim():
    rows = _breach_rows()
    lifecycle = {row["symbol"]: "HOLD" for row in rows}
    resolved = _resolve(rows, lifecycle, authority={})
    bito = resolved["BITO"]
    assert bito["effective_action"] == "TRIM"
    assert bito["lifecycle_action"] == "HOLD"
    assert bito["operator_status"] == PENDING_SELL_REVIEW
    assert bito["reason"] == REASON_EXCESS_CONCENTRATION
    assert resolved["AAPL"]["effective_action"] == "HOLD"
    assert resolved["AAPL"]["operator_status"] == ""


def test_consumed_or_expired_authority_is_not_an_executable_order():
    rows = _breach_rows()
    lifecycle = {row["symbol"]: "HOLD" for row in rows}
    consumed = _resolve(rows, lifecycle, authority=CONSUMED)
    expired = _resolve(
        rows,
        lifecycle,
        authority={
            "status": "UNUSED",
            "consumption_status": "UNUSED",
            "expires_at": "2026-10-14T19:51:37.896242+00:00",
            "approved_symbols": ["BITO"],
        },
    )
    for resolved in (consumed, expired):
        status = resolved["BITO"]["operator_status"]
        assert status == PENDING_SELL_REVIEW
        assert "executable" not in status.lower()
        assert "submitted" not in status.lower()
        assert resolved["BITO"]["effective_action"] == "TRIM"
        assert resolved["BITO"]["lifecycle_action"] == "HOLD"


def test_cleared_concentration_restores_lifecycle_display():
    rows = [
        {"symbol": "BITO", "qty": 10, "current_price": 10.0},
        {"symbol": "BALL", "qty": 10, "current_price": 10.0},
        {"symbol": "QQQ", "qty": 10, "current_price": 10.0},
        {"symbol": "AAPL", "qty": 10, "current_price": 10.0},
        {"symbol": "META", "qty": 10, "current_price": 10.0},
    ]
    lifecycle = {row["symbol"]: "HOLD" for row in rows}
    resolved = _resolve(rows, lifecycle, authority=CONSUMED)
    assert all(item["effective_action"] == "HOLD" for item in resolved.values())
    assert all(item["operator_status"] == "" for item in resolved.values())
    assert all(item["reason"] == "" for item in resolved.values())
    assert all(item["lifecycle_action"] == "HOLD" for item in resolved.values())


def test_no_permitted_trim_does_not_show_a_pending_trim():
    rows = [
        {"symbol": "HEAVY", "qty": 2, "current_price": 6000.0},
        {"symbol": "SMALL", "qty": 1, "current_price": 100.0},
    ]
    resolved = _resolve(rows, {"HEAVY": "HOLD", "SMALL": "HOLD"}, authority={})
    assert resolved["HEAVY"]["effective_action"] == "HOLD"
    assert resolved["HEAVY"]["lifecycle_action"] == "HOLD"
    assert resolved["HEAVY"]["operator_status"] == NO_PERMITTED_TRIM
    assert resolved["HEAVY"]["reason"] == REASON_EXCESS_CONCENTRATION
    assert PENDING_SELL_REVIEW not in resolved["HEAVY"]["operator_status"]


def test_voo_xlk_stay_hold_protected_and_googl_sale_is_not_created():
    rows = [
        {"symbol": "VOO", "qty": 50, "current_price": 100.0},
        {"symbol": "XLK", "qty": 30, "current_price": 100.0},
        {"symbol": "GOOGL", "qty": 1, "current_price": 100.0},
        {"symbol": "BALL", "qty": 20, "current_price": 100.0},
    ]
    resolved = _resolve(
        rows,
        {"VOO": "HOLD", "XLK": "EXIT", "GOOGL": "HOLD", "BALL": "HOLD"},
        blocked_symbols=["VOO", "XLK"],
        authority=CONSUMED,
    )
    for symbol in ("VOO", "XLK"):
        assert resolved[symbol]["effective_action"] == "HOLD"
        assert resolved[symbol]["operator_status"] == BASIS_UNPROVEN_HOLD_STATUS
        assert resolved[symbol]["reason"] == REASON_BASIS_UNPROVEN_HOLD
        assert resolved[symbol]["effective_action"] != "TRIM"
    assert resolved["XLK"]["lifecycle_action"] == "EXIT"
    assert resolved["GOOGL"]["effective_action"] == "HOLD"
    assert resolved["GOOGL"]["lifecycle_action"] == "HOLD"
    assert resolved["GOOGL"]["operator_status"] == ""
    source = inspect.getsource(resolve_portfolio_effective_actions)
    assert "0e99c1320742c1c1c9f750b10b53437e80cdb06013c77171f4ed218a56db09b1" not in source


def test_portfolio_page_shows_effective_action_separately_from_lifecycle():
    source = inspect.getsource(operator_modern._render_canonical_holdings)
    assert "Effective action" in source
    assert "Lifecycle action" in source
    assert "resolve_portfolio_effective_actions" in source
    lowered = source.lower()
    assert "broker_alpaca" not in lowered
    assert "submit_order" not in lowered
    assert "paper_arm" not in lowered
    assert "start_triton" not in lowered
