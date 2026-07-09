"""Tests for Phase 198A defensive exit preview handoff."""

from __future__ import annotations

import json
from pathlib import Path

from services.defensive_exit_preview_handoff import (
    build_preview_handoff_rows_from_plan,
    evaluate_defensive_preview_eligibility,
    load_held_symbols,
)
from services.governance_degraded_mode import classify_portfolio_action_risk


def _orange_readiness() -> dict:
    return {
        "readiness_status": "PARTIALLY_READY",
        "live_execution_permitted": False,
        "paper_execution_permitted": True,
        "governance_mode": "ORANGE_DEGRADED",
        "risk_reducing_paper_actions_allowed": True,
        "failed_checks": ["governance"],
    }


def test_preview_eligible_orange_degraded_governance_only():
    elig = evaluate_defensive_preview_eligibility(_orange_readiness())
    assert elig["preview_generation_allowed"] is True
    assert elig["execution_allowed"] is False


def test_preview_blocked_when_operational_checks_fail():
    readiness = dict(_orange_readiness())
    readiness["failed_checks"] = ["governance", "broker"]
    elig = evaluate_defensive_preview_eligibility(readiness)
    assert elig["preview_generation_allowed"] is False


def test_preview_rows_only_include_held_symbols(tmp_path: Path, monkeypatch):
    plan = tmp_path / "manage_positions_plan.csv"
    plan.write_text(
        "symbol,status,final_action,management_action,qty,limit_price,current_price,rationale,reason_code,rank,priority_bucket\n"
        "GLD,planned,EXIT,EXIT,10,380,380,test,EXIT_HIGH_DRAG,1,EXIT\n"
        "AAPL,planned,EXIT,EXIT,5,100,100,stale,EXIT,2,EXIT\n",
        encoding="utf-8",
    )
    positions = tmp_path / "positions_snapshot.csv"
    positions.write_text(
        "symbol,qty\nGLD,10\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "services.defensive_exit_preview_handoff.PLAN_CSV",
        plan,
    )
    monkeypatch.setattr(
        "services.defensive_exit_preview_handoff.POSITIONS_CSV",
        positions,
    )
    rows = build_preview_handoff_rows_from_plan(held_symbols={"GLD"})
    assert len(rows) == 1
    assert rows[0]["symbol"] == "GLD"


def test_risk_increasing_still_blocked_under_orange():
    assert classify_portfolio_action_risk(recommended_action="ADD") == "RISK_INCREASING"
    assert classify_portfolio_action_risk(recommended_action="OPEN_NEW") == "RISK_INCREASING"
