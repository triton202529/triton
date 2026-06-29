"""
defensive_exit_market_hours_governance_recheck.py
Phase 197O — Market-hours governance observation recheck (read-only, no execution).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"

RECHECK_WAITING = "MARKET_HOURS_RECHECK_WAITING_FOR_MARKET_OPEN"
RECHECK_READY = "MARKET_HOURS_RECHECK_READY_FOR_OPERATOR_DECISION"
RECHECK_GOV = "MARKET_HOURS_RECHECK_BLOCKED_BY_GOVERNANCE"
RECHECK_READINESS = "MARKET_HOURS_RECHECK_BLOCKED_BY_READINESS"
RECHECK_GOV_AND_READINESS = "MARKET_HOURS_RECHECK_BLOCKED_BY_GOVERNANCE_AND_READINESS"
RECHECK_OPEN_ORDERS = "MARKET_HOURS_RECHECK_BLOCKED_BY_OPEN_ORDERS"
RECHECK_LIVE_REGRESSION = "MARKET_HOURS_RECHECK_BLOCKED_BY_LIVE_BLOCK_REGRESSION"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _order_guard_authority_fields(
    *,
    broker_snapshot_ok: bool,
    open_orders_count: Optional[int] = None,
) -> Dict[str, Any]:
    from services.order_guard_snapshot_authority import evaluate_order_guard_authority

    authority = evaluate_order_guard_authority(
        broker_snapshot_ok=broker_snapshot_ok,
        open_orders_count=open_orders_count,
        refresh_guard_if_eligible=False,
        mode="paper",
    )
    return {
        "order_guard_authority_state": authority.get("order_guard_authority_state"),
        "order_guard_authority_reason": authority.get("order_guard_authority_reason"),
        "guard_snapshot_path": authority.get("guard_snapshot_path"),
        "guard_snapshot_age_minutes": authority.get("guard_snapshot_age_minutes"),
        "guard_snapshot_fresh": authority.get("guard_snapshot_fresh"),
        "open_orders_snapshot_age_minutes": authority.get("open_orders_snapshot_age_minutes"),
        "open_orders_snapshot_fresh": authority.get("open_orders_snapshot_fresh"),
        "gate_freshness_note": authority.get("gate_freshness_note"),
    }


def classify_market_hours_recheck(payload: Dict[str, Any]) -> Tuple[str, str]:
    """Classify market-hours recheck state. Returns (state, reason)."""
    if not payload.get("live_blocked"):
        return (
            RECHECK_LIVE_REGRESSION,
            str(payload.get("live_block_reason") or "live_blocked=false"),
        )

    open_orders = int(payload.get("open_orders_count") or 0)
    if open_orders > 0:
        return RECHECK_OPEN_ORDERS, f"open_orders_count={open_orders}"

    if not payload.get("market_open"):
        return RECHECK_WAITING, "market_closed_observation_only"

    governance_authorized = bool(payload.get("governance_authorized"))
    readiness_status = str(payload.get("readiness_status") or "")
    readiness_ready = readiness_status == "READY"

    if not governance_authorized and not readiness_ready:
        return (
            RECHECK_GOV_AND_READINESS,
            f"governance_authorized=false readiness={readiness_status}",
        )
    if not governance_authorized and readiness_ready:
        return (
            RECHECK_GOV,
            str(payload.get("management_review_reason") or "MANAGEMENT_REVIEW_REQUIRED"),
        )
    if governance_authorized and not readiness_ready:
        return RECHECK_READINESS, f"execution_readiness={readiness_status}"

    preview_ok = bool(payload.get("capped_preview_exists"))
    packet_ok = bool(payload.get("operator_packet_current"))
    no_real_approval = not bool(payload.get("real_approval_artifact_exists"))

    if (
        preview_ok
        and packet_ok
        and no_real_approval
        and payload.get("live_blocked")
        and open_orders == 0
    ):
        return RECHECK_READY, "ready_for_human_operator_decision_not_execution"

    return RECHECK_READINESS, "operator_decision_prerequisites_incomplete"


def run_market_hours_governance_recheck(
    *,
    broker_snapshot_ok: bool = True,
    broker_orders_submitted: bool = False,
) -> Dict[str, Any]:
    """Read-only market-hours governance observation."""
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.live_block_classifier import (
        classify_paper_entry_gate_for_validation,
        defensive_exit_phase_validator_gates,
    )
    from services.master_execution_gate import MasterExecutionGate

    criteria = evaluate_governance_review_criteria(
        broker_snapshot_ok=broker_snapshot_ok,
        broker_orders_submitted=broker_orders_submitted,
    )
    gates = defensive_exit_phase_validator_gates(verbose=False)
    gate_classification = classify_paper_entry_gate_for_validation(
        gates["gate_ok"], gates["gate_reason"], gates["gate_details"]
    )
    master_gate = MasterExecutionGate(project_root=ROOT).evaluate(mode="paper", verbose=False)
    freshness = _order_guard_authority_fields(
        broker_snapshot_ok=broker_snapshot_ok,
        open_orders_count=int(criteria.get("open_orders_count") or 0),
    )

    base = {
        "timestamp": _utc_now_iso(),
        "phase": "197O",
        "market_open": bool(criteria.get("market_open")),
        "live_blocked": bool(criteria.get("live_blocked")),
        "live_block_reason": criteria.get("live_block_reason"),
        "governance_authorized": criteria.get("governance_authorized"),
        "governance_review_state": criteria.get("governance_review_state"),
        "governance_review_reason": criteria.get("governance_review_reason"),
        "management_review_required": criteria.get("management_review_required"),
        "readiness_status": criteria.get("readiness_status"),
        "broker_snapshot_ok": broker_snapshot_ok,
        "open_orders_count": criteria.get("open_orders_count"),
        "positions_count": criteria.get("positions_count"),
        "capped_preview_exists": criteria.get("capped_preview_exists"),
        "capped_symbols": criteria.get("capped_symbols"),
        "deferred_symbols": criteria.get("deferred_symbols"),
        "capped_notional": criteria.get("capped_notional"),
        "configured_cap": criteria.get("configured_cap"),
        "notional_cap_status": criteria.get("notional_cap_status"),
        "operator_packet_exists": criteria.get("operator_packet_exists"),
        "operator_packet_current": criteria.get("operator_packet_current"),
        "real_approval_artifact_exists": criteria.get("real_approval_artifact_exists"),
        "broker_orders_submitted": broker_orders_submitted,
        **freshness,
    }

    state, reason = classify_market_hours_recheck(base)
    eligible_for_operator_decision = state == RECHECK_READY
    eligible_for_execution = False

    return {
        **base,
        "market_hours_recheck_state": state,
        "market_hours_recheck_reason": reason,
        "paper_entry_gate_status": gate_classification,
        "paper_entry_gate_reason": gates.get("gate_reason"),
        "paper_entry_gate_ok": gates.get("gate_ok"),
        "master_execution_gate_status": master_gate.status,
        "master_execution_gate_ok": bool(master_gate.ok),
        "master_execution_gate_summary": master_gate.summary,
        "eligible_for_operator_decision": eligible_for_operator_decision,
        "eligible_for_execution": eligible_for_execution,
        "observation_note": (
            "197O observation only. Does not approve, create approval artifacts, or execute."
        ),
        "criteria_snapshot": {
            "evidence_complete": criteria.get("evidence_complete"),
            "missing_evidence": criteria.get("missing_evidence"),
        },
    }
