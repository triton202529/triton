"""Tests for Phase 196B ORANGE governance degraded-mode hardening."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.governance_degraded_mode import (
    ACTION_RISK_INCREASING,
    ACTION_RISK_REDUCING,
    ACTION_NEUTRAL,
    GOVERNANCE_MODE_ORANGE_DEGRADED,
    GOVERNANCE_MODE_RED_BLOCKED,
    classify_governance_mode,
    classify_paper_entry_readiness_outcome,
    classify_portfolio_action_risk,
    enrich_execution_readiness,
)


def _orange_cpe() -> dict:
    return {
        "escalation_state": "ORANGE",
        "escalation_reason": ["EXCESS_CONCENTRATION", "EXCESS_POSITION_DRAWDOWN"],
        "escalation_reason_labels": ["Concentration Risk", "Drawdown Risk"],
    }


def _orange_gov() -> dict:
    return {
        "escalation_state": "ORANGE",
        "governance_awareness_level": "GOVERNANCE_REVIEW_RECOMMENDED",
        "governance_drivers": ["Concentration Risk", "Drawdown Risk"],
    }


def test_orange_concentration_classifies_degraded_not_ready():
    classification = classify_governance_mode(
        gov_doc=_orange_gov(),
        cpe_doc=_orange_cpe(),
        governance_authorized=False,
    )
    assert classification["governance_mode"] == GOVERNANCE_MODE_ORANGE_DEGRADED
    assert classification["risk_increasing_actions_blocked"] is True
    assert classification["risk_reducing_paper_actions_allowed"] is True


def test_orange_degraded_does_not_become_ready():
    readiness = {
        "readiness_status": "PARTIALLY_READY",
        "paper_execution_permitted": True,
        "live_execution_permitted": False,
        "failed_checks": ["governance"],
    }
    auth = {"governance_authorized": False, "execution_authorized": True}
    enriched = enrich_execution_readiness(
        readiness,
        gov_doc=_orange_gov(),
        cpe_doc=_orange_cpe(),
        auth_doc=auth,
        write_audit=False,
    )
    assert enriched["readiness_status"] == "PARTIALLY_READY"
    assert enriched["governance_mode"] == GOVERNANCE_MODE_ORANGE_DEGRADED
    assert enriched["live_execution_permitted"] is False


def test_orange_degraded_blocks_risk_increasing_portfolio_actions():
    assert (
        classify_portfolio_action_risk(recommended_action="OPEN_NEW", risk_flag="OK")
        == ACTION_RISK_INCREASING
    )
    assert (
        classify_portfolio_action_risk(recommended_action="ADD", risk_flag="OK")
        == ACTION_RISK_INCREASING
    )
    assert (
        classify_portfolio_action_risk(
            recommended_action="EXIT",
            risk_flag="FORCE_EXIT",
            reason="risk_flag_contains=FORCE_EXIT",
        )
        == ACTION_RISK_REDUCING
    )


def test_orange_degraded_allows_risk_reducing_classification():
    assert (
        classify_portfolio_action_risk(
            recommended_action="TRIM",
            risk_flag="TRIM_PRIORITY",
            reason="risk_flag_contains=TRIM_PRIORITY",
        )
        == ACTION_RISK_REDUCING
    )


def test_stale_lifecycle_still_blocks_paper_entry():
    readiness = {
        "readiness_status": "NOT_READY",
        "governance_mode": GOVERNANCE_MODE_ORANGE_DEGRADED,
    }
    label, reason, safe = classify_paper_entry_readiness_outcome(
        readiness, ["lifecycle", "governance"]
    )
    assert label == "FAIL"
    assert "lifecycle" in reason or "NOT_READY" in reason
    assert safe is False


def test_broker_failure_still_blocks_paper_entry():
    readiness = {
        "readiness_status": "NOT_READY",
        "governance_mode": GOVERNANCE_MODE_ORANGE_DEGRADED,
    }
    label, _, safe = classify_paper_entry_readiness_outcome(readiness, ["broker"])
    assert label == "FAIL"
    assert safe is False


def test_red_governance_still_blocks():
    classification = classify_governance_mode(
        gov_doc={"escalation_state": "RED"},
        cpe_doc={"escalation_state": "RED", "escalation_reason": ["CRITICAL"]},
        governance_authorized=False,
    )
    assert classification["governance_mode"] == GOVERNANCE_MODE_RED_BLOCKED
    readiness = {
        "readiness_status": "PARTIALLY_READY",
        "governance_mode": GOVERNANCE_MODE_RED_BLOCKED,
    }
    label, _, safe = classify_paper_entry_readiness_outcome(readiness, ["governance"])
    assert label == "FAIL"
    assert safe is False


def test_orange_degraded_paper_entry_safe_degraded():
    readiness = {
        "readiness_status": "PARTIALLY_READY",
        "governance_mode": GOVERNANCE_MODE_ORANGE_DEGRADED,
        "risk_increasing_actions_blocked": True,
    }
    label, reason, safe = classify_paper_entry_readiness_outcome(readiness, ["governance"])
    assert label == "SAFE_DEGRADED"
    assert "risk-increasing actions blocked" in reason
    assert safe is True


def test_neutral_actions_classified():
    assert classify_portfolio_action_risk(recommended_action="HOLD") == ACTION_NEUTRAL
    assert classify_portfolio_action_risk(recommended_action="WATCH") == ACTION_NEUTRAL


def test_enrich_writes_audit_artifact(tmp_path: Path):
    alloc = {
        "largest_position_symbol": "GLD",
        "largest_position_weight_pct": 63.1,
        "force_exit_symbols": ["GLD", "SLV"],
        "trim_priority_symbols": ["GLD", "TSLA"],
        "blocked_new_buy_symbols": ["GLD"],
    }
    (tmp_path / "portfolio_allocation_summary.json").write_text(json.dumps(alloc), encoding="utf-8")
    readiness = {
        "readiness_status": "PARTIALLY_READY",
        "paper_execution_permitted": True,
        "live_execution_permitted": False,
    }
    enrich_execution_readiness(
        readiness,
        gov_doc=_orange_gov(),
        cpe_doc=_orange_cpe(),
        auth_doc={"governance_authorized": False, "execution_authorized": True},
        results_dir=tmp_path,
        write_audit=True,
    )
    audit_path = tmp_path / "governance_degraded_mode_audit.json"
    assert audit_path.is_file()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["governance_mode"] == GOVERNANCE_MODE_ORANGE_DEGRADED
    assert audit["live_execution_permitted"] is False
    assert "safety_statement" in audit
