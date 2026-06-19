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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        s = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _is_expired(req: Dict[str, Any], *, now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(timezone.utc)
    exp = _parse_dt(req.get("expires_at"))
    return exp is not None and now > exp


def _parse_symbol_list(value: Any) -> Set[str]:
    if not value:
        return set()
    return {s.strip().upper() for s in str(value).split(",") if s.strip()}


def _context_summary(req: Dict[str, Any]) -> Dict[str, Any]:
    gov = req.get("governance_context") or {}
    if isinstance(gov, dict):
        summary = gov.get("context_summary") or {}
        if isinstance(summary, dict):
            return summary
    return {}


def validate_paper_capital_deployment_approvable(
    req: Dict[str, Any],
    *,
    require_pending: bool = True,
) -> Tuple[bool, str]:
    """
    Safety checks for approving a paper-only capital_deployment council request.
    """
    if require_pending and req.get("approval_status") != "PENDING":
        return False, f"approval_status={req.get('approval_status')} (must be PENDING)"

    if req.get("approval_status") in ("REJECTED", "CANCELLED", "APPROVED", "EXPIRED"):
        return False, f"approval_status={req.get('approval_status')} cannot be approved"

    if _is_expired(req):
        return False, "approval expired"

    if req.get("authority_level_name") != "GOVERNANCE_COUNCIL_APPROVAL":
        return False, f"authority_level_name={req.get('authority_level_name')}"

    if req.get("execution_path") != "capital_deployment":
        return False, f"execution_path={req.get('execution_path')} (paper capital_deployment only)"

    if req.get("requested_action") != "capital_deployment":
        return False, f"requested_action={req.get('requested_action')}"

    if (
        req.get("execution_path") == "live_execution"
        or req.get("requested_action") == "live_execution"
    ):
        return False, "live_execution requests are never approvable via this CLI"

    ctx = _context_summary(req)
    if ctx.get("liveMode") is True:
        return False, "liveMode=true is not approvable"
    if ctx.get("paperMode") is not True:
        return False, "paperMode must be true"

    if str(req.get("execution_gate_decision") or "").upper() != "ALLOW":
        return (
            False,
            f"execution_gate_decision={req.get('execution_gate_decision')} (must be ALLOW)",
        )

    symbols = _parse_symbol_list(req.get("symbol(s)"))
    if not symbols:
        return False, "symbol(s) is empty"

    return True, "OK"


def _symbols_compatible(req: Dict[str, Any], symbols: Optional[Sequence[str]]) -> bool:
    req_symbols = _parse_symbol_list(req.get("symbol(s)"))
    if not req_symbols:
        return False
    if not symbols:
        return True
    caller = {str(s).strip().upper() for s in symbols if s}
    if not caller:
        return True
    return caller.issubset(req_symbols) or bool(caller & req_symbols)


def _context_compatible(req: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
    if not context:
        return True
    ctx = _context_summary(req)
    if context.get("liveMode") is True:
        return False
    if "paperMode" in context and context.get("paperMode") is not True:
        return False
    if ctx.get("liveMode") is True:
        return False
    if ctx.get("paperMode") is not True:
        return False
    return True


def find_approved_request(
    execution_identifier: Optional[str],
    path: str,
    *,
    symbols: Optional[Sequence[str]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    call_id = str(execution_identifier or "").strip()

    for req in reversed(_load_requests()):
        if req.get("approval_status") != "APPROVED":
            continue
        if req.get("execution_path") != path:
            continue
        if _is_expired(req, now=now):
            continue

        req_id = str(req.get("execution_identifier") or "").strip()
        if call_id and req_id:
            if call_id == req_id:
                return req
            continue

        if call_id and not req_id:
            continue

        if not _context_compatible(req, context):
            continue
        if not _symbols_compatible(req, symbols):
            continue
        return req
    return None


def approve_request(
    approval_id: str,
    *,
    approving_authority: str,
    approval_rationale: str,
    validate_paper_capital: bool = True,
) -> Tuple[Optional[Dict[str, Any]], str]:
    records = _load_requests()
    updated: Optional[Dict[str, Any]] = None
    for req in records:
        if req.get("approval_id") != approval_id:
            continue
        if validate_paper_capital:
            ok, reason = validate_paper_capital_deployment_approvable(req)
            if not ok:
                return None, reason
        elif req.get("approval_status") != "PENDING":
            return None, f"approval_status={req.get('approval_status')}"
        elif _is_expired(req):
            return None, "approval expired"

        if req.get("approval_status") != "PENDING":
            return None, f"approval_status={req.get('approval_status')}"

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
        return updated, "OK"
    return None, "approval_id not found"


def reject_request(
    approval_id: str,
    *,
    rejecting_authority: str,
    rejection_rationale: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    records = _load_requests()
    updated: Optional[Dict[str, Any]] = None
    for req in records:
        if req.get("approval_id") != approval_id:
            continue
        if req.get("approval_status") != "PENDING":
            return None, f"approval_status={req.get('approval_status')}"
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
        return updated, "OK"
    return None, "approval_id not found"


def list_pending_requests(*, paper_capital_only: bool = False) -> List[Dict[str, Any]]:
    expire_stale_requests()
    out: List[Dict[str, Any]] = []
    for req in _load_requests():
        if req.get("approval_status") != "PENDING":
            continue
        if _is_expired(req):
            continue
        if paper_capital_only:
            ok, _ = validate_paper_capital_deployment_approvable(req)
            if not ok:
                continue
        out.append(req)
    return out


def find_latest_pending_paper_capital_deployment() -> Optional[Dict[str, Any]]:
    expire_stale_requests()
    for req in reversed(_load_requests()):
        if req.get("approval_status") != "PENDING":
            continue
        ok, _ = validate_paper_capital_deployment_approvable(req)
        if ok:
            return req
    return None


def approve_latest_pending_paper_capital_deployment(
    *,
    approving_authority: str,
    approval_rationale: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    req = find_latest_pending_paper_capital_deployment()
    if not req:
        return None, "no pending paper capital_deployment request found"
    return approve_request(
        str(req.get("approval_id")),
        approving_authority=approving_authority,
        approval_rationale=approval_rationale,
        validate_paper_capital=True,
    )


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

    if authority_level >= 3 and evaluation.get("allowed"):
        result["governance"]["emergency_authority"] = True
        result["governance"]["human_approval_required"] = False
        result["decision"] = "ALLOW"
        result["allowed"] = True
        result["reason"] = (
            f"Emergency authority ({authority_level_name(authority_level)}): "
            f"{evaluation.get('reason', '')}"
        )
        return result

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

    approved_req = find_approved_request(
        execution_identifier,
        path,
        symbols=symbols,
        context=ctx,
    )
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


def _cli_list_pending(args: Any) -> int:
    rows = list_pending_requests(paper_capital_only=bool(args.paper_only))
    if not rows:
        print("No pending approval requests.")
        return 0
    for req in rows:
        ctx = _context_summary(req)
        print(
            f"{req.get('approval_id')} | {req.get('execution_path')} | "
            f"symbols={req.get('symbol(s)', '')} | paper={ctx.get('paperMode')} | "
            f"live={ctx.get('liveMode')} | gate={req.get('execution_gate_decision')} | "
            f"expires={req.get('expires_at')}"
        )
    print(f"total_pending={len(rows)}")
    return 0


def _cli_approve(args: Any) -> int:
    updated, reason = approve_request(
        args.id,
        approving_authority=args.authority,
        approval_rationale=args.rationale,
        validate_paper_capital=not args.force,
    )
    if not updated:
        print(f"[APPROVAL_REFUSED] {reason}", flush=True)
        return 1
    print(json.dumps({"approved": updated.get("approval_id"), "status": "APPROVED"}, indent=2))
    return 0


def _cli_reject(args: Any) -> int:
    updated, reason = reject_request(
        args.id,
        rejecting_authority=args.authority,
        rejection_rationale=args.rationale,
    )
    if not updated:
        print(f"[REJECTION_REFUSED] {reason}", flush=True)
        return 1
    print(json.dumps({"rejected": updated.get("approval_id"), "status": "REJECTED"}, indent=2))
    return 0


def _cli_approve_latest_paper(args: Any) -> int:
    updated, reason = approve_latest_pending_paper_capital_deployment(
        approving_authority=args.authority,
        approval_rationale=args.rationale,
    )
    if not updated:
        print(f"[APPROVAL_REFUSED] {reason}", flush=True)
        return 1
    print(
        json.dumps(
            {
                "approved": updated.get("approval_id"),
                "symbols": updated.get("symbol(s)"),
                "status": "APPROVED",
            },
            indent=2,
        )
    )
    return 0


def _build_cli() -> "argparse.ArgumentParser":
    import argparse

    ap = argparse.ArgumentParser(
        description="TRITON execution approval workflow (THP-008 governance council CLI)"
    )
    sub = ap.add_subparsers(dest="command")

    p_list = sub.add_parser("list-pending", help="List pending approval requests")
    p_list.add_argument(
        "--paper-only",
        action="store_true",
        help="Only show approvable paper capital_deployment requests",
    )
    p_list.set_defaults(func=_cli_list_pending)

    p_ap = sub.add_parser(
        "approve", help="Approve a pending request (paper capital_deployment safety enforced)"
    )
    p_ap.add_argument("--id", required=True, help="approval_id")
    p_ap.add_argument("--authority", required=True, help="Approving authority identity")
    p_ap.add_argument("--rationale", required=True, help="Approval rationale")
    p_ap.add_argument(
        "--force",
        action="store_true",
        help="Skip paper capital_deployment safety checks (not recommended)",
    )
    p_ap.set_defaults(func=_cli_approve)

    p_rej = sub.add_parser("reject", help="Reject a pending request")
    p_rej.add_argument("--id", required=True, help="approval_id")
    p_rej.add_argument("--authority", required=True, help="Rejecting authority identity")
    p_rej.add_argument("--rationale", required=True, help="Rejection rationale")
    p_rej.set_defaults(func=_cli_reject)

    p_latest = sub.add_parser(
        "approve-latest-paper",
        help="Approve latest pending paper-only capital_deployment council request",
    )
    p_latest.add_argument("--authority", required=True, help="Approving authority identity")
    p_latest.add_argument("--rationale", required=True, help="Approval rationale")
    p_latest.set_defaults(func=_cli_approve_latest_paper)

    p_exp = sub.add_parser("expire-stale", help="Mark expired pending requests as EXPIRED")
    p_exp.set_defaults(func=lambda _a: (print(f"expired={expire_stale_requests()}"), 0)[1])

    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = _build_cli()
    args = ap.parse_args(argv)
    if not getattr(args, "command", None):
        ap.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    import sys

    sys.exit(main())
