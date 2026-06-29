"""
defensive_exit_market_hours_governance_recheck.py
Phase 197O — Market-hours governance observation recheck (read-only, no execution).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
OPEN_ORDERS_JSON = RESULTS / "open_orders_snapshot.csv"
GUARD_PATHS = [RESULTS / "guard_snapshot.json", LIVE / "guard_snapshot.json"]

RECHECK_WAITING = "MARKET_HOURS_RECHECK_WAITING_FOR_MARKET_OPEN"
RECHECK_READY = "MARKET_HOURS_RECHECK_READY_FOR_OPERATOR_DECISION"
RECHECK_GOV = "MARKET_HOURS_RECHECK_BLOCKED_BY_GOVERNANCE"
RECHECK_READINESS = "MARKET_HOURS_RECHECK_BLOCKED_BY_READINESS"
RECHECK_GOV_AND_READINESS = "MARKET_HOURS_RECHECK_BLOCKED_BY_GOVERNANCE_AND_READINESS"
RECHECK_OPEN_ORDERS = "MARKET_HOURS_RECHECK_BLOCKED_BY_OPEN_ORDERS"
RECHECK_LIVE_REGRESSION = "MARKET_HOURS_RECHECK_BLOCKED_BY_LIVE_BLOCK_REGRESSION"

DEFAULT_GUARD_MAX_AGE_MINUTES = 30.0
DEFAULT_OPEN_ORDERS_MAX_AGE_MINUTES = 120.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _file_age_minutes(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (_utc_now() - mtime).total_seconds() / 60.0


def _resolve_guard_path() -> Optional[Path]:
    existing = [p for p in GUARD_PATHS if p.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _guard_freshness() -> Dict[str, Any]:
    path = _resolve_guard_path()
    age = _file_age_minutes(path) if path else None
    fresh = age is not None and age <= DEFAULT_GUARD_MAX_AGE_MINUTES
    oo_age = _file_age_minutes(OPEN_ORDERS_JSON)
    oo_fresh = oo_age is not None and oo_age <= DEFAULT_OPEN_ORDERS_MAX_AGE_MINUTES
    note = ""
    if path and age is not None and not fresh:
        note = (
            "guard_snapshot stale; open_orders snapshot authoritative when open_orders=0 "
            "(guard not refreshed by snapshot_live_orders)"
        )
    elif not path:
        note = "guard_snapshot missing; diagnostic only unless live mode requires guard"
    return {
        "guard_snapshot_path": str(path.relative_to(ROOT)).replace("\\", "/") if path else None,
        "guard_snapshot_age_minutes": round(age, 2) if age is not None else None,
        "guard_snapshot_fresh": fresh if path else None,
        "open_orders_snapshot_age_minutes": round(oo_age, 2) if oo_age is not None else None,
        "open_orders_snapshot_fresh": oo_fresh if OPEN_ORDERS_JSON.is_file() else None,
        "gate_freshness_note": note,
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
    freshness = _guard_freshness()

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
