"""Paper SELL quote-cap repair.

A higher bid must not zero an already-authorized whole-share exit, and must
not abort the rest of a protective slice. BUY notional limits stay unchanged.
Fake brokers only. No production authority writes and no live orders.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from services.entry_allocation_policy import select_bounded_entries
from services.paper_sell_arm_authority import (
    APPROVAL_CLI_CAP,
    RETAINED_WHOLE_SHARE_REASON,
    evaluate_readiness_packet_freshness,
    recompute_allowed_runtime_qty,
)
from services.paper_sell_arm_runtime_enforcement import (
    _downward_runtime_qty,
    _price_unavailable,
    build_pre_plan_planned_orders,
    protective_sell_qty_vs_stale_notional,
    validate_pre_plan_authority_bound_artifacts,
)

NOW = datetime(2026, 10, 23, 18, 14, 48, tzinfo=timezone.utc)

NVDA_CAP = 211.35
META_CAP = 643.84
ARKK_CAP = 2612.48
BATCH_CAP = 3467.67


class _QuoteBroker:
    def __init__(self, quotes, positions):
        self.quotes = {str(k).upper(): v for k, v in quotes.items()}
        self.positions = positions

    def get_quote(self, symbol):
        price = self.quotes.get(str(symbol).upper())
        if price is None:
            return None
        return {"bid": price, "ask": price}

    def get_trade(self, symbol):
        price = self.quotes.get(str(symbol).upper())
        if price is None:
            return None
        return {"last": price}

    def list_positions(self):
        return self.positions


def _slice_authority():
    return {
        "authorized_actions": [
            {"symbol": "NVDA", "action": "EXIT", "max_qty": 1.0, "max_notional": NVDA_CAP},
            {"symbol": "META", "action": "EXIT", "max_qty": 1.0, "max_notional": META_CAP},
            {"symbol": "ARKK", "action": "EXIT", "max_qty": 32.0, "max_notional": ARKK_CAP},
        ],
        "per_symbol_max_quantity": {"NVDA": 1.0, "META": 1.0, "ARKK": 32.0},
        "per_symbol_max_notional": {"NVDA": NVDA_CAP, "META": META_CAP, "ARKK": ARKK_CAP},
        "total_batch_max_notional": BATCH_CAP,
    }


def _positions(arkk="40"):
    return [
        {"symbol": "NVDA", "qty": "1", "qty_available": "1"},
        {"symbol": "META", "qty": "1", "qty_available": "1"},
        {"symbol": "ARKK", "qty": arkk, "qty_available": arkk},
    ]


def _plan(quotes, positions=None):
    return build_pre_plan_planned_orders(
        _slice_authority(),
        _QuoteBroker(quotes, positions if positions is not None else _positions()),
        now=NOW,
    )


def test_nvda_one_share_stays_executable_above_stale_cap():
    qty, reason = protective_sell_qty_vs_stale_notional(1, NVDA_CAP, 211.75)
    assert qty == 1
    assert reason == RETAINED_WHOLE_SHARE_REASON
    assert _downward_runtime_qty(1, NVDA_CAP, 211.75) == (
        0,
        "clipped_to_zero_vs_authorized_max_notional",
    )
    ok, reason, planned = _plan({"NVDA": 211.75, "META": 643.84, "ARKK": 81.64})
    assert ok, reason
    nvda = next(row for row in planned if row.symbol == "NVDA")
    assert int(nvda.qty) == 1
    assert float(nvda.ref_price) == 211.75


def test_meta_one_share_stays_executable_above_stale_cap():
    qty, reason = protective_sell_qty_vs_stale_notional(1, META_CAP, 644.90)
    assert qty == 1
    assert reason == RETAINED_WHOLE_SHARE_REASON
    ok, reason, planned = _plan({"NVDA": 211.35, "META": 644.90, "ARKK": 81.64})
    assert ok, reason
    meta = next(row for row in planned if row.symbol == "META")
    assert int(meta.qty) == 1
    assert int(meta.qty) <= 1


def test_arkk_partial_clip_stays_within_authorized_qty_and_holdings():
    qty, reason = protective_sell_qty_vs_stale_notional(32, ARKK_CAP, 82.00)
    assert qty == 31
    assert "downward_clip" in reason
    ok, reason, planned = _plan(
        {"NVDA": 211.35, "META": 643.84, "ARKK": 82.00},
        _positions(arkk="32"),
    )
    assert ok, reason
    arkk = next(row for row in planned if row.symbol == "ARKK")
    assert int(arkk.qty) == 31
    assert int(arkk.qty) <= 32
    held = 32
    assert int(arkk.qty) <= held


def test_invalid_missing_and_stale_quote_still_block():
    assert _price_unavailable(None)
    assert _price_unavailable(0)
    assert _price_unavailable(float("nan"))
    assert _price_unavailable(float("inf"))
    ok, reason, planned = _plan({"META": 643.84, "ARKK": 81.64})
    assert not ok
    assert reason == "price_unavailable:NVDA"
    assert planned == ()
    stale_packet = {
        "generated_at": (NOW - timedelta(minutes=51)).isoformat(),
        "session_date": "2026-10-23",
    }
    fresh_ok, fresh_reason, _gen = evaluate_readiness_packet_freshness(
        stale_packet,
        now=NOW,
        session_payload={"session_state": "OPEN", "session_date": "2026-10-23"},
    )
    assert not fresh_ok
    assert fresh_reason == "stale_readiness_packet"
    assert math.isfinite(211.75)


def test_expired_and_mismatched_authority_still_block(tmp_path):
    expired = {
        "consumption_status": "UNUSED",
        "status": "ARMED",
        "mode": "PAPER",
        "live_execution_required_state": "BLOCKED",
        "live_authorized": False,
        "expires_at": "2026-10-23T18:00:00+00:00",
        "trading_date": "2026-10-23",
        "authorized_actions": [
            {"symbol": "NVDA", "action": "EXIT", "max_qty": 1.0, "max_notional": NVDA_CAP}
        ],
        "total_batch_max_notional": NVDA_CAP,
    }
    ok, reason, _details = validate_pre_plan_authority_bound_artifacts(
        expired, results_root=tmp_path, now=NOW
    )
    assert not ok
    assert reason == "authority_expired"
    allowed, why = recompute_allowed_runtime_qty(1, NVDA_CAP, 211.75)
    assert allowed == 1
    assert why == RETAINED_WHOLE_SHARE_REASON
    mismatched = dict(expired)
    mismatched["expires_at"] = "2026-10-23T19:00:00+00:00"
    mismatched["readiness_packet_hash"] = "0" * 64
    ok_mismatch, mismatch_reason, _details = validate_pre_plan_authority_bound_artifacts(
        mismatched, results_root=tmp_path, now=NOW
    )
    assert not ok_mismatch
    assert mismatch_reason == "source_readiness_packet_missing"


def test_buy_max_notional_behavior_unchanged():
    blocked = select_bounded_entries(
        [{"symbol": "NVDA", "disposition": "OPEN_NEW", "score": 1.0}],
        cash=50.0,
        max_new=1,
        max_adds=0,
        max_notional=100.0,
        open_slots=1,
    )
    assert blocked[0]["disposition"] == "hard_block"
    assert blocked[0]["reason"] == "hard_block:per_order_notional_exceeds_settled_cash"
    allowed = select_bounded_entries(
        [{"symbol": "NVDA", "disposition": "OPEN_NEW", "score": 1.0}],
        cash=150.0,
        max_new=1,
        max_adds=0,
        max_notional=100.0,
        open_slots=1,
    )
    assert allowed[0]["disposition"] == "OPEN_NEW"
    assert allowed[0]["notional"] == 100.0


def _live_notional(planned) -> float:
    return round(
        sum(int(row.qty) * float(row.ref_price or row.limit_price) for row in planned),
        2,
    )


def test_current_nvda_meta_arkk_slice_stays_within_live_bid_cap():
    ok, reason, planned = _plan({"NVDA": 211.75, "META": 644.90, "ARKK": 82.00})
    assert ok, reason
    by_symbol = {row.symbol: int(row.qty) for row in planned}
    assert by_symbol == {"NVDA": 1, "META": 1, "ARKK": 31}
    total = _live_notional(planned)
    assert total == round(211.75 + 644.90 + 31 * 82.00, 2)
    assert total <= APPROVAL_CLI_CAP
    for row in planned:
        assert int(row.qty) <= int({"NVDA": 1, "META": 1, "ARKK": 32}[row.symbol])


def test_retained_one_share_exits_cannot_exceed_aggregate_cap():
    symbols = ("AAA", "BBB", "CCC")
    authority = {
        "authorized_actions": [
            {"symbol": symbol, "action": "EXIT", "max_qty": 1.0, "max_notional": 100.0}
            for symbol in symbols
        ],
        "per_symbol_max_quantity": {symbol: 1.0 for symbol in symbols},
        "per_symbol_max_notional": {symbol: 100.0 for symbol in symbols},
        "total_batch_max_notional": 300.0,
    }
    positions = [{"symbol": symbol, "qty": "1", "qty_available": "1"} for symbol in symbols]
    ok, reason, planned = build_pre_plan_planned_orders(
        authority,
        _QuoteBroker({symbol: 2000.0 for symbol in symbols}, positions),
        now=NOW,
    )
    assert ok, reason
    quantities = {row.symbol: int(row.qty) for row in planned}
    assert quantities["AAA"] == 1
    assert quantities["BBB"] == 1
    assert quantities["CCC"] == 0
    ccc = next(row for row in planned if row.symbol == "CCC")
    assert ccc.clip_reason == "clipped_to_zero_vs_remaining_batch_cap"
    total = _live_notional(planned)
    assert total == 4000.0
    assert total <= APPROVAL_CLI_CAP
    assert total + 2000.0 > APPROVAL_CLI_CAP
    for row in planned:
        assert int(row.qty) <= 1


def test_single_share_priced_above_approval_cap_blocks():
    authority = {
        "authorized_actions": [
            {"symbol": "NVDA", "action": "EXIT", "max_qty": 1.0, "max_notional": NVDA_CAP}
        ],
        "per_symbol_max_quantity": {"NVDA": 1.0},
        "per_symbol_max_notional": {"NVDA": NVDA_CAP},
        "total_batch_max_notional": NVDA_CAP,
    }
    ok, reason, planned = build_pre_plan_planned_orders(
        authority,
        _QuoteBroker(
            {"NVDA": 5000.01},
            [{"symbol": "NVDA", "qty": "1", "qty_available": "1"}],
        ),
        now=NOW,
    )
    assert not ok
    assert reason == "batch_notional_exceeds_approval_cli_cap"
    assert planned == ()


def test_multiline_slice_survives_favorable_quote_movement():
    ok, reason, planned = _plan({"NVDA": 211.75, "META": 644.90, "ARKK": 82.00})
    assert ok, reason
    assert reason != "SELL_AUTHORITY_NOT_EXECUTABLE"
    by_symbol = {row.symbol: int(row.qty) for row in planned}
    assert by_symbol == {"NVDA": 1, "META": 1, "ARKK": 31}
    assert _live_notional(planned) <= APPROVAL_CLI_CAP
    for row in planned:
        assert int(row.qty) <= int({"NVDA": 1, "META": 1, "ARKK": 32}[row.symbol])
