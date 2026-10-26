"""Session entry budget must submit the names that still fit.

A later cycle can propose more new names than the remaining daily slots.
Reserving that batch all-or-nothing used to refuse the legal prefix too, so
those slots stayed empty for the rest of the session.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from services.session_entry_budget import (
    budget_path,
    reserve_entries,
    select_orders_within_session_budget,
)

SESSION = "2026-10-26"
NOW = datetime(2026, 10, 26, 15, 0, tzinfo=timezone.utc)


def _buy(symbol: str, *, stance: str = "BUY", qty: float = 0.0, tag: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        side="buy",
        symbol=symbol,
        stance=stance,
        decision_tag=tag,
        current_position_qty=qty,
    )


def _sell(symbol: str) -> SimpleNamespace:
    return SimpleNamespace(
        side="sell",
        symbol=symbol,
        stance="EXIT",
        decision_tag="",
        current_position_qty=10,
    )


def _stored_symbols(root: Path) -> list[str]:
    path = budget_path(root, SESSION)
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [row["symbol"] for row in doc["reservations"]]


def _seed_open_new(root: Path, count: int) -> None:
    entries = [
        {"symbol": f"N{i:02d}", "action": "OPEN_NEW", "held_at_open": False}
        for i in range(count)
    ]
    result = reserve_entries(root, SESSION, entries, invocation_tag="cycle-1", now=NOW)
    assert result["ok"] is True


def test_over_cap_batch_refuses_every_name_including_the_ones_that_fit(tmp_path: Path):
    """The all-or-nothing reserve API still refuses a mixed batch and writes nothing."""
    _seed_open_new(tmp_path, 8)
    mixed = [
        {"symbol": "AAA", "action": "OPEN_NEW", "held_at_open": False},
        {"symbol": "BBB", "action": "OPEN_NEW", "held_at_open": False},
        {"symbol": "CCC", "action": "OPEN_NEW", "held_at_open": False},
        {"symbol": "DDD", "action": "OPEN_NEW", "held_at_open": False},
    ]
    refused = reserve_entries(tmp_path, SESSION, mixed, invocation_tag="cycle-2", now=NOW)
    assert refused["ok"] is False
    assert refused["reason"] == "session_new_name_cap"
    assert len(_stored_symbols(tmp_path)) == 8


def test_selector_reserves_remaining_slots_and_drops_only_the_overflow(tmp_path: Path):
    _seed_open_new(tmp_path, 8)
    orders = [_buy("AAA"), _buy("BBB"), _sell("XYZ"), _buy("CCC"), _buy("DDD")]
    selected = select_orders_within_session_budget(
        orders,
        tmp_path,
        SESSION,
        "cycle-2",
        now=NOW,
    )
    assert selected["ok"] is True
    assert selected["reason"] == "session_new_name_cap"
    assert [order.symbol for order in selected["orders"]] == ["AAA", "BBB", "XYZ"]
    assert [order.symbol for order in selected["dropped"]] == ["CCC", "DDD"]
    assert [row["symbol"] for row in selected["reserved"]] == ["AAA", "BBB"]
    assert _stored_symbols(tmp_path)[-2:] == ["AAA", "BBB"]
    assert len(_stored_symbols(tmp_path)) == 10


def test_full_cap_still_submits_a_sell_that_shares_the_batch(tmp_path: Path):
    _seed_open_new(tmp_path, 10)
    before = budget_path(tmp_path, SESSION).read_text(encoding="utf-8")
    selected = select_orders_within_session_budget(
        [_buy("AAA"), _sell("XYZ")],
        tmp_path,
        SESSION,
        "cycle-2",
        now=NOW,
    )
    assert selected["ok"] is True
    assert [order.symbol for order in selected["orders"]] == ["XYZ"]
    assert [order.symbol for order in selected["dropped"]] == ["AAA"]
    assert selected["reserved"] == []
    assert budget_path(tmp_path, SESSION).read_text(encoding="utf-8") == before


def test_unreadable_budget_drops_buys_and_keeps_the_sell(tmp_path: Path):
    path = budget_path(tmp_path, SESSION)
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")
    selected = select_orders_within_session_budget(
        [_buy("AAA"), _sell("XYZ")],
        tmp_path,
        SESSION,
        "cycle-2",
        now=NOW,
    )
    assert selected["ok"] is True
    assert selected["reason"] == "budget_unreadable"
    assert [order.symbol for order in selected["orders"]] == ["XYZ"]
    assert path.read_text(encoding="utf-8") == "{"


def test_missing_session_blocks_buys_only(tmp_path: Path):
    buys_only = select_orders_within_session_budget(
        [_buy("AAA")],
        tmp_path,
        "",
        "cycle-2",
        now=NOW,
    )
    assert buys_only["ok"] is False
    assert buys_only["reason"] == "budget_session_missing"
    assert buys_only["orders"] == []

    with_sell = select_orders_within_session_budget(
        [_buy("AAA"), _sell("XYZ")],
        tmp_path,
        "",
        "cycle-2",
        now=NOW,
    )
    assert with_sell["ok"] is True
    assert [order.symbol for order in with_sell["orders"]] == ["XYZ"]
    assert not budget_path(tmp_path, SESSION).exists()


def test_add_cap_keeps_the_one_add_that_fits(tmp_path: Path):
    seeded = reserve_entries(
        tmp_path,
        SESSION,
        [
            {"symbol": "HELD1", "action": "ADD", "held_at_open": True},
            {"symbol": "HELD2", "action": "ADD", "held_at_open": True},
        ],
        invocation_tag="cycle-1",
        now=NOW,
    )
    assert seeded["ok"] is True
    orders = [
        _buy("HELD3", stance="ADD", qty=5, tag="add_to_position"),
        _buy("HELD4", stance="ADD", qty=5, tag="add_to_position"),
    ]
    selected = select_orders_within_session_budget(
        orders,
        tmp_path,
        SESSION,
        "cycle-2",
        now=NOW,
    )
    assert [order.symbol for order in selected["orders"]] == ["HELD3"]
    assert [order.symbol for order in selected["dropped"]] == ["HELD4"]
    assert _stored_symbols(tmp_path) == ["HELD1", "HELD2", "HELD3"]


def test_executor_places_through_the_partial_selector():
    source = (Path(__file__).resolve().parents[1] / "services" / "execute_trades.py").read_text(
        encoding="utf-8"
    )
    body = source.split("def maybe_execute_plan", 1)[1].split("def preview_planned_buy_symbols", 1)[0]
    assert "select_orders_within_session_budget" in body
    assert "reserve_entries(" not in body
