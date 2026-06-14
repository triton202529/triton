"""
execution_approval_workflow.py
THP-008 — Human approval workflow for execution governance council.

Wraps evaluate_execution_gate with authority level enforcement and approval requests.
"""

from __future__ import annotations

import csv
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from services.execution_escalation_engine import detect_escalations, record_escalations
from services.execution_gate_degradation_manager import load_operational_status
from services.execution_gate_evidence_registry import load_evidence
from services.execution_governance_council import (
    authority_level_name,
    determine_authority_level,
    load_authority_levels,
)
from services.triton_execution_gate import EXECUTION_PATHS, evaluate_execution_gate

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"
RESULTS_DIR = ROOT / "data" / "results"

APPROVAL_REQUESTS_JSON = GOVERNANCE_DIR / "execution_approval_requests.json"
APPROVAL_HISTORY_CSV = GOVERNANCE_DIR / "execution_approval_history.csv"

APPROVAL_STATUSES = ("PENDING", "APPROVED", "REJECTED", "EXPIRED", "CANCELLED")

APPROVAL_HISTORY_FIELDS = (
    "timestamp",
    "approval_id",
    "execution_path",
    "execution_identifier",
    "authority_level_required",
    "approval_status",
    "approving_authority",
    "action",
)

EMERGENCY_PATHS = {
    EXECUTION_PATHS["RISK_EMERGENCY_ACTION"],
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _approval_id() -> str:
    return f"eg-apr-{uuid.uuid4().hex[:12]}"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_requests() -> List[Dict[str, Any]]:
    if not APPROVAL_REQUESTS_JSON.is_file():
        return []
    try:
        doc = json.loads(APPROVAL_REQUESTS_JSON.read_text(encoding="utf-8"))
        return list(doc.get("records") or [])
    except Exception:
        return []


def _save_requests(records: List[Dict[str, Any]]) -> None:
    pending = sum(1 for r in records if r.get("approval_status") == "PENDING")
    doc = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-008",
        "summary": {
            "total": len(records),
            "pending": pending,
            "approved": sum(1 for r in records if r.get("approval_status") == "APPROVED"),
            "rejected": sum(1 for r in records if r.get("approval_status") == "REJECTED"),
            "expired": sum(1 for r in records if r.get("approval_status") == "EXPIRED"),
        },
        "records": records[-500:],
    }
    _atomic_write_json(APPROVAL_REQUESTS_JSON, doc)


def _append_history(row: Dict[str, Any]) -> None:
    GOVERNANCE_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if APPROVAL_HISTORY_CSV.is_file():
        with APPROVAL_HISTORY_CSV.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows.append({k: str(row.get(k, "")) for k in APPROVAL_HISTORY_FIELDS})
    tmp = APPROVAL_HISTORY_CSV.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(APPROVAL_HISTORY_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, APPROVAL_HISTORY_CSV)


def _load_risk_state() -> Dict[str, Any]:
    path = RESULTS_DIR / "adaptive_risk_state.json"
    if not path.is_file():
        return {"status": "unknown"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "unknown"}


def _evidence_links_for(execution_identifier: Optional[str]) -> List[str]:
    links: List[str] = []
    for ev in load_evidence():
        if execution_identifier and ev.get("decision_id"):
            links.append(str(ev.get("evidence_id") or ev.get("decision_id")))
    return links[:5]


def create_approval_request(
    *,
    path: str,
    context: Optional[Dict[str, Any]] = None,
    evaluation: Optional[Dict[str, Any]] = None,
    authority_level: int,
    triggering_conditions: List[str],
    execution_identifier: Optional[str] = None,
    symbols: Optional[Sequence[str]] = None,
    action_requested: Optional[str] = None,
    capital_exposure: Optional[Any] = None,
) -> Dict[str, Any]:
    levels_doc = load_authority_levels()
    expiry_hours = int((levels_doc.get("thresholds") or {}).get("approval_expiry_hours") or 24)
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=expiry_hours)).isoformat()

    request = {
        "approval_id": _approval_id(),
        "timestamp": _utc_now_iso(),
        "execution_identifier": execution_identifier or "",
        "execution_path": path,
        "requested_action": action_requested or path,
        "symbol(s)": ",".join(symbols) if symbols else "",
        "estimated_capital_exposure": capital_exposure,
        "authority_level_required": authority_level,
        "authority_level_name": authority_level_name(authority_level),
        "triggering_condition": "; ".join(triggering_conditions),
        "execution_gate_decision": (evaluation or {}).get("decision"),
        "current_risk_state": _load_risk_state(),
        "governance_context": {
            "operational_state": load_operational_status().get("operational_state"),
            "context_summary": {
                k: context.get(k)
                for k in (
                    "paperMode",
                    "liveMode",
                    "governanceApproved",
                    "lifecycleApproved",
                    "riskConstraintsSatisfied",
                )
                if context and k in context
            },
        },
        "evidence_links": _evidence_links_for(execution_identifier),
        "approval_status": "PENDING",
        "expires_at": expires_at,
    }

    records = _load_requests()
    records.append(request)
    _save_requests(records)
    _append_history(
        {
            "timestamp": request["timestamp"],
            "approval_id": request["approval_id"],
            "execution_path": path,
            "execution_identifier": execution_identifier or "",
            "authority_level_required": authority_level,
            "approval_status": "PENDING",
            "approving_authority": "",
            "action": "REQUEST_CREATED",
        }
    )
    return request


def find_approved_request(
    execution_identifier: Optional[str],
    path: str,
) -> Optional[Dict[str, Any]]:
    if not execution_identifier:
        return None
    now = datetime.now(timezone.utc)
    for req in reversed(_load_requests()):
        if req.get("approval_status") != "APPROVED":
            continue
        if req.get("execution_identifier") != execution_identifier:
            continue
        if req.get("execution_path") != path:
            continue
        expires = req.get("expires_at")
        if expires:
            try:
                exp_text = str(expires).replace("Z", "+00:00")
                exp_dt = datetime.fromisoformat(exp_text)
                if exp_dt.tzinfo is None:
                    exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                if now > exp_dt.astimezone(timezone.utc):
                    continue
            except ValueError:
                pass
        return req
    return None


def approve_request(
    approval_id: str,
    *,
    approving_authority: str,
    approval_rationale: str,
) -> Optional[Dict[str, Any]]:
    records = _load_requests()
    updated: Optional[Dict[str, Any]] = None
    for req in records:
        if req.get("approval_id") != approval_id:
            continue
        if req.get("approval_status") != "PENDING":
            return None
        req["approval_status"] = "APPROVED"
        req["approving_authority"] = approving_authority
        req["approval_rationale"] = approval_rationale
        req["approval_timestamp"] = _utc_now_iso()
        updated = req
        _append_history(
            {
                "timestamp": req["approval_timestamp"],
                "approval_id": approval_id,
                "execution_path": req.get("execution_path", ""),
                "execution_identifier": req.get("execution_identifier", ""),
                "authority_level_required": req.get("authority_level_required", ""),
                "approval_status": "APPROVED",
                "approving_authority": approving_authority,
                "action": "APPROVED",
            }
        )
        break
    if updated:
        _save_requests(records)
    return updated


def reject_request(
    approval_id: str,
    *,
    rejecting_authority: str,
    rejection_rationale: str,
) -> Optional[Dict[str, Any]]:
    records = _load_requests()
    updated: Optional[Dict[str, Any]] = None
    for req in records:
        if req.get("approval_id") != approval_id:
            continue
        if req.get("approval_status") != "PENDING":
            return None
        req["approval_status"] = "REJECTED"
        req["rejecting_authority"] = rejecting_authority
        req["rejection_rationale"] = rejection_rationale
        req["rejection_timestamp"] = _utc_now_iso()
        updated = req
        _append_history(
            {
                "timestamp": req["rejection_timestamp"],
                "approval_id": approval_id,
                "execution_path": req.get("execution_path", ""),
                "execution_identifier": req.get("execution_identifier", ""),
                "authority_level_required": req.get("authority_level_required", ""),
                "approval_status": "REJECTED",
                "approving_authority": rejecting_authority,
                "action": "REJECTED",
            }
        )
        break
    if updated:
        _save_requests(records)
    return updated


def expire_stale_requests() -> int:
    records = _load_requests()
    now = datetime.now(timezone.utc)
    expired_count = 0
    for req in records:
        if req.get("approval_status") != "PENDING":
            continue
        expires = req.get("expires_at")
        if not expires:
            continue
        try:
            exp_text = str(expires).replace("Z", "+00:00")
            exp_dt = datetime.fromisoformat(exp_text)
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            if now > exp_dt.astimezone(timezone.utc):
                req["approval_status"] = "EXPIRED"
                expired_count += 1
        except ValueError:
            continue
    if expired_count:
        _save_requests(records)
    return expired_count


def _apply_governance_overlay(
    evaluation: Dict[str, Any],
    *,
    authority_level: int,
    triggers: List[str],
    path: str,
    approved: bool,
    operational_state: str,
) -> Dict[str, Any]:
    result = dict(evaluation)
    result["governance"] = {
        "framework": "THP-008",
        "authority_level": authority_level,
        "authority_level_name": authority_level_name(authority_level),
        "triggering_conditions": triggers,
        "human_approval_required": authority_level >= 1 and not approved,
        "operational_state": operational_state,
    }

    if path in EMERGENCY_PATHS and evaluation.get("allowed"):
        result["governance"]["emergency_authority"] = True
        return result

    if operational_state == "LOCKDOWN" and path not in EMERGENCY_PATHS:
        result["decision"] = "BLOCK"
        result["allowed"] = False
        result["reason"] = (
            "Governance LOCKDOWN active: non-emergency capital deployment blocked. "
            f"Original gate: {evaluation.get('reason', '')}"
        )
        result["governance"]["lockdown_applied"] = True
        return result

    if approved and evaluation.get("decision") in ("ALLOW", "REVIEW_REQUIRED"):
        result["decision"] = "ALLOW"
        result["allowed"] = True
        result["reason"] = (
            f"Governance Council approved execution (level {authority_level}). "
            f"{evaluation.get('reason', '')}"
        )
        result["governance"]["approval_applied"] = True
        return result

    if authority_level >= 1 and not approved:
        if evaluation.get("allowed") or evaluation.get("decision") == "ALLOW":
            result["decision"] = "REVIEW_REQUIRED"
            result["allowed"] = False
            result["reason"] = (
                f"Human {authority_level_name(authority_level)} required: "
                f"{'; '.join(triggers) or 'governance review'}. "
                f"Gate would allow: {evaluation.get('reason', '')}"
            )
        elif evaluation.get("decision") == "BLOCK":
            result["governance"]["gate_blocked"] = True

    return result


def evaluate_governed_execution(
    path: str,
    context: Optional[Dict[str, Any]] = None,
    *,
    record_evidence: bool = True,
    source: str = "governed_execution",
    broker_mode: str = "",
    symbols: Optional[Sequence[str]] = None,
    execution_identifier: Optional[str] = None,
    side: str = "",
    quantity: Optional[Any] = None,
    capital_exposure: Optional[Any] = None,
    operator_identity: Optional[str] = None,
    action_requested: Optional[str] = None,
    skip_governance: bool = False,
    record_requests: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate execution with THP-008 human governance overlay.

    Wraps evaluate_execution_gate; may downgrade ALLOW to REVIEW_REQUIRED or apply
    LOCKDOWN blocks. Emergency risk actions remain available when gate permits.
    """
    ctx = context or {}
    evaluation = evaluate_execution_gate(
        path,
        ctx,
        record_evidence=record_evidence,
        source=source,
        broker_mode=broker_mode,
        symbols=symbols,
        execution_identifier=execution_identifier,
        side=side,
        quantity=quantity,
        capital_exposure=capital_exposure,
        operator_identity=operator_identity,
        action_requested=action_requested,
    )

    if skip_governance or source.startswith("stress") or source.startswith("chaos"):
        return evaluation

    operational = load_operational_status()
    op_state = str(operational.get("operational_state") or "MONITOR")

    authority_level, triggers = determine_authority_level(
        path=path,
        context=ctx,
        evaluation=evaluation,
        capital_exposure=capital_exposure,
        action_requested=action_requested,
    )

    approved_req = find_approved_request(execution_identifier, path)
    approved = approved_req is not None

    result = _apply_governance_overlay(
        evaluation,
        authority_level=authority_level,
        triggers=triggers,
        path=path,
        approved=approved,
        operational_state=op_state,
    )

    if result.get("governance", {}).get("human_approval_required") and record_requests:
        create_approval_request(
            path=path,
            context=ctx,
            evaluation=evaluation,
            authority_level=authority_level,
            triggering_conditions=triggers,
            execution_identifier=execution_identifier,
            symbols=symbols,
            action_requested=action_requested,
            capital_exposure=capital_exposure,
        )
        escalations = detect_escalations(
            path=path,
            context=ctx,
            evaluation=evaluation,
            capital_exposure=capital_exposure,
            action_requested=action_requested,
            execution_identifier=execution_identifier,
        )
        if escalations:
            record_escalations(escalations, dedupe_key=execution_identifier)

    return result


def load_approval_requests() -> Dict[str, Any]:
    if not APPROVAL_REQUESTS_JSON.is_file():
        return {}
    try:
        return json.loads(APPROVAL_REQUESTS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    expire_stale_requests()
    sample = evaluate_governed_execution(
        EXECUTION_PATHS["PAPER_EXECUTION"],
        {"paperMode": True, "lifecycleApproved": True},
        record_evidence=False,
        source="thp008_validation",
        execution_identifier="thp008-paper-test",
    )
    print(
        json.dumps({"decision": sample.get("decision"), "allowed": sample.get("allowed")}, indent=2)
    )
