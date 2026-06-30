"""
defensive_exit_governance_review_criteria.py
Phase 197N — Formal governance review criteria for capped defensive paper exits.
Read-only; does not approve, execute, or alter governance/readiness artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
READINESS_JSON = RESULTS / "execution_readiness.json"
GOVERNANCE_AUTH_JSON = RESULTS / "governance_authorization.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
OPEN_ORDERS_JSON = RESULTS / "open_orders_snapshot.csv"
POSITIONS_JSON = RESULTS / "positions_snapshot.csv"

GOV_REVIEW_REQUIRED = "GOV_REVIEW_REQUIRED"
GOV_REVIEW_READY_FOR_OPERATOR_DECISION = "GOV_REVIEW_READY_FOR_OPERATOR_DECISION"
GOV_REVIEW_READY_FOR_OPERATOR_REVIEW = "GOV_REVIEW_READY_FOR_OPERATOR_REVIEW"
GOV_REVIEW_CURRENT_OPERATOR_APPROVAL_REQUIRED = "GOV_REVIEW_CURRENT_OPERATOR_APPROVAL_REQUIRED"
GOV_REVIEW_CURRENT_PAPER_APPROVAL_PRESENT = "GOV_REVIEW_CURRENT_PAPER_APPROVAL_PRESENT"
GOV_REVIEW_CURRENT_BUT_READINESS_BLOCKED = "GOV_REVIEW_CURRENT_BUT_READINESS_BLOCKED"
GOV_REVIEW_BLOCKED_BY_LIVE_BLOCK_REGRESSION = "GOV_REVIEW_BLOCKED_BY_LIVE_BLOCK_REGRESSION"
GOV_REVIEW_BLOCKED_BY_OPEN_ORDERS = "GOV_REVIEW_BLOCKED_BY_OPEN_ORDERS"
GOV_REVIEW_BLOCKED_BY_STALE_PACKET = "GOV_REVIEW_BLOCKED_BY_STALE_PACKET"
GOV_REVIEW_BLOCKED_BY_MISSING_PREVIEW = "GOV_REVIEW_BLOCKED_BY_MISSING_PREVIEW"
GOV_REVIEW_BLOCKED_BY_READINESS = "GOV_REVIEW_BLOCKED_BY_READINESS"
GOV_REVIEW_BLOCKED_BY_EXPIRED_APPROVAL = "GOV_REVIEW_BLOCKED_BY_EXPIRED_APPROVAL"
GOV_REVIEW_BLOCKED_BY_INVALID_APPROVAL = "GOV_REVIEW_BLOCKED_BY_INVALID_APPROVAL"
GOV_REVIEW_BLOCKED_BY_MARKET_CLOSED = "GOV_REVIEW_BLOCKED_BY_MARKET_CLOSED"

REQUIRED_EVIDENCE_KEYS = (
    "live_blocked_confirmed",
    "no_valid_live_confirmation",
    "paper_mode_confirmed",
    "broker_snapshot_ok",
    "open_orders_zero",
    "real_approval_artifact_absent",
    "capped_preview_exists",
    "capped_symbols_listed",
    "deferred_symbols_listed",
    "configured_cap_respected",
    "notional_cap_status_explicit",
    "readiness_status_recorded",
    "market_hours_status_recorded",
    "operator_packet_current",
    "source_preview_hash_recorded",
    "management_review_reason_recorded",
    "risk_reduction_intent_recorded",
    "no_broker_orders_submitted",
    "no_cap_increase_requested",
    "no_live_execution_requested",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _runtime_capped_preview(doc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        doc.get("capped_preview_runtime_cap")
        or doc.get("capped_preview")
        or doc.get("capped_preview_cli_cap")
        or {}
    )


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(path))
    except Exception:
        return 0


def _is_market_open() -> bool:
    try:
        from services.guard_auto import is_market_open_et

        return bool(is_market_open_et(_utc_now()))
    except Exception:
        return False


def _packet_current(packet: Dict[str, Any], preview_hash: str) -> Tuple[bool, str]:
    if not packet:
        return False, "operator_packet_missing"
    expires = _parse_iso(str(packet.get("expires_at") or ""))
    if expires and expires <= _utc_now():
        return False, "operator_packet_expired"
    pkt_hash = str(packet.get("source_preview_hash") or "")
    if preview_hash and pkt_hash != preview_hash:
        return False, "source_preview_hash_mismatch"
    if not str(packet.get("packet_id") or ""):
        return False, "packet_id_missing"
    return True, ""


def collect_governance_review_evidence(
    *,
    broker_snapshot_ok: bool = True,
    broker_orders_submitted: bool = False,
) -> Dict[str, Any]:
    """Collect required governance review evidence (read-only)."""
    from services.defensive_exit_operator_paper_approval import compute_capped_preview_hash
    from services.live_block_classifier import (
        evaluate_live_block_status,
        scan_live_confirm_artifacts,
    )

    live_block = evaluate_live_block_status(paper_mode=True)
    confirm_scan = scan_live_confirm_artifacts()
    preview_doc = _read_json(PREVIEW_JSON)
    packet = _read_json(OPERATOR_PACKET_JSON)
    readiness = _read_json(READINESS_JSON)
    auth = _read_json(GOVERNANCE_AUTH_JSON)

    capped_block = _runtime_capped_preview(preview_doc)
    capped_symbols = list(capped_block.get("capped_preview_symbols") or [])
    deferred_symbols = list(capped_block.get("deferred_symbols") or [])
    capped_notional = float(capped_block.get("batch_notional_capped_preview") or 0.0)
    configured_cap = float(
        preview_doc.get("configured_batch_cap_manage_positions")
        or preview_doc.get("configured_batch_cap_effective")
        or 0.0
    )
    preview_hash = compute_capped_preview_hash(preview_doc) if preview_doc else ""
    packet_current, packet_current_reason = _packet_current(packet, preview_hash)
    market_open = _is_market_open()
    open_orders = _count_csv_rows(OPEN_ORDERS_JSON)
    governance_reason = str(
        auth.get("gate_reasons", {}).get("governance")
        or readiness.get("check_details", {}).get("governance")
        or ""
    )
    management_review_required = (
        not bool(auth.get("governance_authorized"))
        or "MANAGEMENT_REVIEW_REQUIRED" in governance_reason.upper()
    )
    notional_cap_status = str(preview_doc.get("handoff_status") or "UNKNOWN")
    cap_respected = capped_notional <= configured_cap if configured_cap else bool(capped_symbols)
    risk_intent = packet.get("risk_reduction_intent") or {}

    evidence = {
        "live_blocked_confirmed": bool(live_block.get("live_blocked")),
        "no_valid_live_confirmation": confirm_scan.get("valid_live_confirm_count", 0) == 0,
        "paper_mode_confirmed": bool(live_block.get("paper_env_enabled")),
        "broker_snapshot_ok": broker_snapshot_ok,
        "open_orders_zero": open_orders == 0,
        "real_approval_artifact_absent": not REAL_APPROVAL_PATH.is_file(),
        "capped_preview_exists": bool(preview_doc) and bool(capped_symbols or deferred_symbols),
        "capped_symbols_listed": bool(capped_symbols),
        "deferred_symbols_listed": bool(deferred_symbols),
        "configured_cap_respected": cap_respected,
        "notional_cap_status_explicit": bool(
            notional_cap_status and notional_cap_status != "UNKNOWN"
        ),
        "readiness_status_recorded": bool(readiness.get("readiness_status")),
        "market_hours_status_recorded": True,
        "operator_packet_current": packet_current,
        "source_preview_hash_recorded": bool(preview_hash),
        "management_review_reason_recorded": bool(governance_reason),
        "risk_reduction_intent_recorded": bool(risk_intent),
        "no_broker_orders_submitted": not broker_orders_submitted,
        "no_cap_increase_requested": True,
        "no_live_execution_requested": not bool(live_block.get("live_explicitly_authorized")),
    }

    missing = [k for k in REQUIRED_EVIDENCE_KEYS if not evidence.get(k)]
    complete = not missing

    return {
        "timestamp": _utc_now_iso(),
        "phase": "197N",
        "evidence": evidence,
        "evidence_complete": complete,
        "missing_evidence": missing,
        "live_block": live_block,
        "live_blocked": bool(live_block.get("live_blocked")),
        "live_block_reason": live_block.get("live_block_reason"),
        "paper_mode": True,
        "market_open": market_open,
        "broker_snapshot_ok": broker_snapshot_ok,
        "open_orders_count": open_orders,
        "positions_count": _count_csv_rows(POSITIONS_JSON),
        "capped_preview_exists": evidence["capped_preview_exists"],
        "capped_symbols": capped_symbols,
        "deferred_symbols": deferred_symbols,
        "capped_notional": round(capped_notional, 2),
        "configured_cap": configured_cap,
        "notional_cap_status": notional_cap_status,
        "operator_packet_exists": OPERATOR_PACKET_JSON.is_file(),
        "operator_packet_current": packet_current,
        "operator_packet_current_reason": packet_current_reason,
        "source_preview_hash": preview_hash,
        "management_review_required": management_review_required,
        "management_review_reason": governance_reason,
        "readiness_status": str(readiness.get("readiness_status") or "UNKNOWN"),
        "governance_authorized": bool(auth.get("governance_authorized")),
        "real_approval_artifact_exists": REAL_APPROVAL_PATH.is_file(),
        "broker_orders_submitted": broker_orders_submitted,
        "risk_reduction_intent": risk_intent,
        "operator_packet_id": packet.get("packet_id"),
        "packet_expires_at": packet.get("expires_at"),
    }


def classify_governance_review_state(evidence_payload: Dict[str, Any]) -> Tuple[str, str]:
    """
    Classify governance review state from collected evidence.
    Returns (state, reason).
    """
    if not evidence_payload.get("live_blocked"):
        return (
            GOV_REVIEW_BLOCKED_BY_LIVE_BLOCK_REGRESSION,
            str(evidence_payload.get("live_block_reason") or "live_blocked=false"),
        )

    open_orders = int(evidence_payload.get("open_orders_count") or 0)
    if open_orders > 0:
        return GOV_REVIEW_BLOCKED_BY_OPEN_ORDERS, f"open_orders_count={open_orders}"

    if not evidence_payload.get("capped_preview_exists"):
        return GOV_REVIEW_BLOCKED_BY_MISSING_PREVIEW, "capped_preview_missing"

    if not evidence_payload.get("operator_packet_current"):
        return (
            GOV_REVIEW_BLOCKED_BY_STALE_PACKET,
            str(evidence_payload.get("operator_packet_current_reason") or "packet_not_current"),
        )

    if not evidence_payload.get("market_open"):
        return GOV_REVIEW_BLOCKED_BY_MARKET_CLOSED, "market_closed_safe_block"

    real_approval_exists = bool(evidence_payload.get("real_approval_artifact_exists"))
    readiness_status = str(evidence_payload.get("readiness_status") or "")

    if real_approval_exists:
        from services.paper_operator_approval_artifact_state import (
            PAPER_APPROVAL_EXPIRED,
            PAPER_APPROVAL_VALID_CURRENT,
            classify_paper_operator_approval_artifact,
        )

        ap_state, ap_reason, _ = classify_paper_operator_approval_artifact(
            broker_snapshot_ok=bool(evidence_payload.get("broker_snapshot_ok")),
        )
        if ap_state == PAPER_APPROVAL_VALID_CURRENT:
            return (
                GOV_REVIEW_CURRENT_PAPER_APPROVAL_PRESENT,
                "paper_operator_approval_valid_broker_execution_separate",
            )
        if ap_state == PAPER_APPROVAL_EXPIRED:
            return GOV_REVIEW_BLOCKED_BY_EXPIRED_APPROVAL, ap_reason
        return GOV_REVIEW_BLOCKED_BY_INVALID_APPROVAL, ap_reason

    if not real_approval_exists:
        if evidence_payload.get("management_review_required") or not evidence_payload.get(
            "governance_authorized"
        ):
            return (
                GOV_REVIEW_CURRENT_OPERATOR_APPROVAL_REQUIRED,
                str(
                    evidence_payload.get("management_review_reason")
                    or "operator_paper_defensive_exit_approval_required"
                ),
            )
        if readiness_status != "READY":
            return (
                GOV_REVIEW_CURRENT_BUT_READINESS_BLOCKED,
                f"execution_readiness={readiness_status}",
            )
        if evidence_payload.get("evidence_complete"):
            return (
                GOV_REVIEW_READY_FOR_OPERATOR_REVIEW,
                "governance_evidence_complete_packet_current",
            )
        return (
            GOV_REVIEW_CURRENT_OPERATOR_APPROVAL_REQUIRED,
            "operator_paper_defensive_exit_approval_required",
        )


def evaluate_governance_review_criteria(
    *,
    broker_snapshot_ok: bool = True,
    broker_orders_submitted: bool = False,
) -> Dict[str, Any]:
    """Full governance review criteria evaluation (read-only)."""
    evidence_payload = collect_governance_review_evidence(
        broker_snapshot_ok=broker_snapshot_ok,
        broker_orders_submitted=broker_orders_submitted,
    )
    state, reason = classify_governance_review_state(evidence_payload)

    eligible_for_operator_decision = state == GOV_REVIEW_READY_FOR_OPERATOR_DECISION
    eligible_for_execution = False

    return {
        **evidence_payload,
        "governance_review_state": state,
        "governance_review_reason": reason,
        "eligible_for_operator_decision": eligible_for_operator_decision,
        "eligible_for_execution": eligible_for_execution,
        "resolution_criteria_note": (
            "197N defines criteria only. Does not resolve MANAGEMENT_REVIEW_REQUIRED, "
            "create approval artifacts, or authorize execution."
        ),
        "required_evidence_keys": list(REQUIRED_EVIDENCE_KEYS),
        "governance_decision_states": [
            GOV_REVIEW_REQUIRED,
            GOV_REVIEW_READY_FOR_OPERATOR_DECISION,
            GOV_REVIEW_READY_FOR_OPERATOR_REVIEW,
            GOV_REVIEW_CURRENT_OPERATOR_APPROVAL_REQUIRED,
            GOV_REVIEW_CURRENT_PAPER_APPROVAL_PRESENT,
            GOV_REVIEW_CURRENT_BUT_READINESS_BLOCKED,
            GOV_REVIEW_BLOCKED_BY_LIVE_BLOCK_REGRESSION,
            GOV_REVIEW_BLOCKED_BY_OPEN_ORDERS,
            GOV_REVIEW_BLOCKED_BY_STALE_PACKET,
            GOV_REVIEW_BLOCKED_BY_MISSING_PREVIEW,
            GOV_REVIEW_BLOCKED_BY_READINESS,
            GOV_REVIEW_BLOCKED_BY_EXPIRED_APPROVAL,
            GOV_REVIEW_BLOCKED_BY_INVALID_APPROVAL,
            GOV_REVIEW_BLOCKED_BY_MARKET_CLOSED,
        ],
    }
