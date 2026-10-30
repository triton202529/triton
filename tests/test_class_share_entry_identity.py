"""Class-share aliases must stay one name on the paper entry path.

Broker positions and orders use BRK.B. Allocation keeps the candidate
spelling BRK-B. Those two forms are the same security.
"""

from __future__ import annotations

from pathlib import Path

import services.duplicate_order_guard as duplicate_guard
import services.market_calendar as market_calendar
import services.paper_entry_loop_stage as entry
import services.session_entry_budget as budget


def _patch_entry_io(monkeypatch, results: Path) -> None:
    monkeypatch.setattr(entry, "RESULTS", results)
    monkeypatch.setattr(entry, "POSITIONS", results / "positions_snapshot.csv")
    monkeypatch.setattr(
        entry,
        "ALLOC_RECOMMENDATIONS",
        results / "portfolio_allocation_recommendations.csv",
    )
    monkeypatch.setattr(entry, "OPEN_ORDERS", results / "open_orders_snapshot.csv")
    monkeypatch.setattr(entry, "RISK_OVERLAY", results / "performance_risk_overlay.csv")
    monkeypatch.setattr(entry, "TRADE_OPPORTUNITIES", results / "trade_opportunities.csv")
    monkeypatch.setattr(
        market_calendar,
        "regular_session_state",
        lambda **_kwargs: {"session_date": "2026-10-30"},
    )
    monkeypatch.setattr(
        budget,
        "preview_entries",
        lambda *_args, **_kwargs: {"ok": True, "reason": "preview_ok", "refusals": []},
    )
    monkeypatch.setattr(
        duplicate_guard,
        "check_duplicate_order",
        lambda **_kwargs: (False, ""),
    )


def test_held_broker_alias_still_accepts_candidate_add(monkeypatch, tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "positions_snapshot.csv").write_text(
        "symbol,qty\nBRK.B,4\n",
        encoding="utf-8",
    )
    (results / "portfolio_allocation_recommendations.csv").write_text(
        "ticker,recommended_action,risk_flag,confidence,signal\n"
        "BRK-B,ADD,OK,0.82,BUY\n"
        "AAPL,OPEN_NEW,OK,0.80,BUY\n",
        encoding="utf-8",
    )
    (results / "signal_lifecycle_effective.csv").write_text(
        "ticker,effective_stance,session_as_of_date\n"
        "BRK-B,ADD,2026-10-30\n"
        "AAPL,BUY,2026-10-30\n",
        encoding="utf-8",
    )
    _patch_entry_io(monkeypatch, results)

    candidates, handoff = entry._load_governed_allocation_candidates()
    by_ticker = {row["ticker"]: row for row in candidates}

    assert "BRK-B" not in by_ticker
    assert by_ticker["BRK.B"]["allocation_action"] == "ADD"
    assert by_ticker["BRK.B"]["eligible_for_auto_entry"] is True
    assert "ADD_REQUIRES_EXISTING_POSITION" not in by_ticker["BRK.B"]["block_reasons"]
    assert by_ticker["AAPL"]["eligible_for_auto_entry"] is True
    assert handoff["eligible_entry_symbols"] == ["AAPL", "BRK.B"]

    from services.paper_entry_candidate_contract import evaluate_allocation_planner_contract

    planned = ["AAPL", "BRK.B"]
    ok, reason, detail = evaluate_allocation_planner_contract(
        handoff["eligible_entry_symbols"],
        planned,
        {
            "planner_considered_symbols": planned,
            "planner_selected_symbols": planned,
            "planner_rejected_symbols": {},
        },
    )
    assert ok is True
    assert reason == "OK"
    assert detail["contract_status"] == "ALIGNED"
    assert detail["contract_mismatch_detected"] is False


def test_open_new_candidate_alias_is_already_held(monkeypatch, tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "positions_snapshot.csv").write_text(
        "symbol,qty\nBRK.B,4\n",
        encoding="utf-8",
    )
    (results / "portfolio_allocation_recommendations.csv").write_text(
        "ticker,recommended_action,risk_flag,confidence,signal\n"
        "BRK-B,OPEN_NEW,OK,0.82,BUY\n",
        encoding="utf-8",
    )
    (results / "signal_lifecycle_effective.csv").write_text(
        "ticker,effective_stance,session_as_of_date\nBRK-B,BUY,2026-10-30\n",
        encoding="utf-8",
    )
    _patch_entry_io(monkeypatch, results)

    candidates, handoff = entry._load_governed_allocation_candidates()
    row = candidates[0]

    assert row["ticker"] == "BRK.B"
    assert row["eligible_for_auto_entry"] is False
    assert "ALREADY_HELD" in row["block_reasons"]
    assert handoff["eligible_entry_symbols"] == []


def test_open_broker_order_alias_blocks_only_that_name(monkeypatch, tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "positions_snapshot.csv").write_text(
        "symbol,qty\n",
        encoding="utf-8",
    )
    (results / "open_orders_snapshot.csv").write_text(
        "symbol,side,status\nBRK.B,buy,new\n",
        encoding="utf-8",
    )
    (results / "portfolio_allocation_recommendations.csv").write_text(
        "ticker,recommended_action,risk_flag,confidence,signal\n"
        "BRK-B,OPEN_NEW,OK,0.90,BUY\n"
        "AAPL,OPEN_NEW,OK,0.70,BUY\n",
        encoding="utf-8",
    )
    (results / "signal_lifecycle_effective.csv").write_text(
        "ticker,effective_stance,session_as_of_date\n"
        "BRK-B,BUY,2026-10-30\n"
        "AAPL,BUY,2026-10-30\n",
        encoding="utf-8",
    )
    _patch_entry_io(monkeypatch, results)

    candidates, handoff = entry._load_governed_allocation_candidates()
    by_ticker = {row["ticker"]: row for row in candidates}

    assert by_ticker["BRK.B"]["eligible_for_auto_entry"] is False
    assert "OPEN_BUY_ORDER" in by_ticker["BRK.B"]["block_reasons"]
    assert by_ticker["AAPL"]["eligible_for_auto_entry"] is True
    assert handoff["eligible_entry_symbols"] == ["AAPL"]


def test_session_budget_collapses_class_share_aliases(tmp_path: Path):
    session = "2026-10-30"
    reserved = budget.reserve_entries(
        tmp_path,
        session,
        [{"symbol": "BRK.B", "action": "OPEN_NEW", "held_at_open": False}],
        invocation_tag="cycle-1",
    )
    assert reserved["ok"] is True
    assert reserved["reserved"][0]["symbol"] == "BRK.B"

    preview = budget.preview_entries(
        tmp_path,
        session,
        [{"symbol": "BRK-B", "action": "OPEN_NEW", "held_at_open": False}],
    )
    assert preview["ok"] is False
    assert preview["reason"] == "session_symbol_already_used"

    again = budget.reserve_entries(
        tmp_path,
        session,
        [{"symbol": "BRK-B", "action": "ADD", "held_at_open": True}],
        invocation_tag="cycle-2",
    )
    assert again["ok"] is False
    assert again["reason"] == "opened_today_cannot_add"

    fresh = tmp_path / "fresh"
    both = budget.reserve_entries(
        fresh,
        session,
        [
            {"symbol": "BRK-B", "action": "OPEN_NEW", "held_at_open": False},
            {"symbol": "BRK.B", "action": "OPEN_NEW", "held_at_open": False},
        ],
        invocation_tag="same-batch",
    )
    assert both["ok"] is False
    assert both["reason"] == "session_symbol_already_used"
    assert not budget.budget_path(fresh, session).exists()
