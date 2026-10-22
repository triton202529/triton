"""Exposure-reducing sells must not be dropped by entry-only gates."""

from __future__ import annotations

import pandas as pd

from services.execution_guard import ExecutionGuard
from services.execute_trades import (
    AccountSnapshot,
    PositionSnapshot,
    build_execution_plan,
    load_execute_trades_config,
)


class _Broker:
    mode = "paper"

    def __init__(self, *, buying_power: float, positions: list, price: float = 100.0):
        self.buying_power = buying_power
        self.positions = positions
        self.price = price

    def get_positions(self):
        return list(self.positions)

    def list_orders(self, **_kwargs):
        return []

    def get_account(self):
        return {"buying_power": self.buying_power, "equity": 10000.0}

    def get_latest_price(self, _symbol):
        return self.price

    def get_clock(self):
        return {"is_open": True}


def _guard(broker: _Broker) -> ExecutionGuard:
    guard = ExecutionGuard(broker)
    guard.cfg["kill_switch_path"] = "/tmp/triton-no-kill-switch.json"
    guard.cfg["require_market_open"] = False
    guard.cfg["enforce_buying_power"] = True
    guard.cfg["max_positions"] = 2
    guard.cfg["max_notional_usd"] = 1500.0
    guard.cfg["min_notional_usd"] = 25.0
    guard.cfg["cooldown_minutes"] = 0
    return guard


def test_sell_is_not_blocked_when_buying_power_is_below_notional():
    broker = _Broker(buying_power=100.0, positions=[])
    guard = _guard(broker)
    decision = guard.validate(
        {
            "symbol": "XYZ",
            "side": "sell",
            "qty": 10,
            "type": "limit",
            "limit_price": 100.0,
        }
    )
    assert decision.ok, decision.code
    assert decision.code != "INSUFFICIENT_BP"


def test_buy_is_still_blocked_when_buying_power_is_below_notional():
    broker = _Broker(buying_power=100.0, positions=[])
    guard = _guard(broker)
    decision = guard.validate(
        {
            "symbol": "AAA",
            "side": "buy",
            "qty": 10,
            "type": "limit",
            "limit_price": 100.0,
        }
    )
    assert not decision.ok
    assert decision.code == "INSUFFICIENT_BP"


def test_rotate_exit_is_not_blocked_by_full_book_or_entry_notional_cap():
    broker = _Broker(buying_power=50.0, positions=[{}, {}, {}, {}], price=100.0)
    guard = _guard(broker)
    decision = guard.validate(
        {
            "symbol": "XYZ",
            "side": "sell",
            "qty": 20,
            "type": "market",
            "management_action": "ROTATE_EXIT",
            "order_source": "manage_positions",
        }
    )
    assert decision.ok, decision.code


def test_untagged_sell_still_hits_max_notional():
    broker = _Broker(buying_power=100000.0, positions=[])
    guard = _guard(broker)
    decision = guard.validate(
        {
            "symbol": "XYZ",
            "side": "sell",
            "qty": 20,
            "type": "market",
        }
    )
    assert not decision.ok
    assert decision.code == "MAX_NOTIONAL"


def test_deterioration_exit_is_planned_beside_a_stronger_buy():
    cfg = load_execute_trades_config()
    cfg["sector_caps_enabled"] = False
    cfg["sector_exposure_enabled"] = False
    cfg["diversification_ranking_enabled"] = False
    cfg["capital_engine_enabled"] = False
    cfg["exec_score_percentile_normalize"] = True
    cfg["enforce_min_final_score"] = True
    cfg["min_final_score_threshold"] = 0.65
    cfg["block_negative_final_score"] = True
    cfg["score_sizing_enabled"] = True

    df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "opportunity_type": "ENTRY",
                "effective_stance": "BUY",
                "effective_position_state": "FLAT",
                "confidence": 0.9,
                "delta_pct": 0.02,
                "rationale": "entry",
            },
            {
                "ticker": "XYZ",
                "opportunity_type": "EXIT",
                "effective_stance": "EXIT",
                "effective_position_state": "LONG",
                "confidence": 0.8,
                "delta_pct": -0.02,
                "rationale": "deterioration",
            },
        ]
    )
    positions = {"XYZ": PositionSnapshot(symbol="XYZ", qty=10.0, market_value=1000.0)}
    acct = AccountSnapshot(equity=100000.0, buying_power=50000.0, cash=50000.0)
    broker = _Broker(buying_power=50000.0, positions=[], price=100.0)

    planned, plan_lines, skip_reasons, *_rest = build_execution_plan(
        df,
        cfg,
        positions,
        acct,
        broker,
        "paper",
        verbose=False,
    )
    exits = [p for p in planned if p.symbol == "XYZ" and p.side == "sell"]
    assert exits, f"exit was not planned; skips={skip_reasons}"
    assert exits[0].qty == 10
    exit_lines = [pl for pl in plan_lines if pl.symbol == "XYZ"]
    assert exit_lines
    assert exit_lines[-1].status == "planned"
    assert "BELOW_MIN_SCORE" not in str(exit_lines[-1].skip_reason)
    assert "SCORE_SIZED_TOO_SMALL" not in str(exit_lines[-1].skip_reason)
