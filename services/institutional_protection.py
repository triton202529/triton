"""
TRITON Institutional Protection — Phases 25–27.

Risk committee oversight, accountability registry, and preservation governance board.
Paper-mode / simulation only. NO live trading, orders, or portfolio changes.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

GOVERNANCE_BLOCK_LEVELS = frozenset(
    {
        "MANAGEMENT_REVIEW_REQUIRED",
        "BOARD_REVIEW_REQUIRED",
        "CRITICAL_INTERVENTION",
    }
)

ESCALATION_SEVERITY = {
    "GREEN": 0,
    "YELLOW": 1,
    "ORANGE": 2,
    "RED": 3,
    "CRITICAL": 4,
}

DOMAIN_NAMES = (
    "Portfolio Health",
    "Governance Health",
    "Preservation Health",
    "Readiness Health",
    "Certification Health",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: Optional[datetime] = None) -> str:
    t = dt or _utc_now()
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _domain_status(score: int) -> str:
    if score >= 75:
        return "CLEAR"
    if score >= 55:
        return "MONITOR"
    if score >= 35:
        return "CONCERN"
    return "CRITICAL"


def _portfolio_health_score(cpi_doc: Dict[str, Any]) -> Tuple[int, List[str]]:
    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    concerns: List[str] = []
    components = cpi_doc.get("component_scores") or {}
    component_labels = {
        "drawdown": "Drawdown Risk",
        "concentration": "Concentration Risk",
        "exposure": "Exposure Risk",
        "operational": "Operational Risk",
        "execution": "Execution Risk",
    }
    for name, score in sorted(
        ((k, v) for k, v in components.items() if isinstance(v, (int, float))),
        key=lambda x: x[1],
    ):
        if score < 50:
            label = component_labels.get(name, name.title())
            if label not in concerns:
                concerns.append(label)
    for label in cpi_doc.get("escalation_reason_labels") or []:
        if label and label not in concerns:
            concerns.append(str(label))
    return cps, concerns


def _governance_health_score(
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
) -> Tuple[int, List[str]]:
    concerns: List[str] = []
    gov_level = str(gov_doc.get("governance_awareness_level") or "").upper()
    score = 85 if gov_level == "GREEN" else (65 if gov_level == "YELLOW" else 30)
    if gov_level in GOVERNANCE_BLOCK_LEVELS:
        score = min(score, 35)
        concerns.extend(
            gov_doc.get("governance_drivers") or gov_doc.get("governance_summary") or []
        )
    if not auth_doc.get("overall_authorization"):
        score = min(score, 45)
        if "Authorization Gap" not in concerns:
            concerns.append("Authorization Gap")
    if not auth_doc.get("governance_authorized"):
        score = min(score, 40)
    return score, concerns[:5]


def _preservation_health_score(
    governor_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
) -> Tuple[int, List[str]]:
    concerns: List[str] = list(cpe_doc.get("escalation_reason_labels") or [])
    posture = str(governor_doc.get("preservation_posture") or "UNKNOWN").upper()
    posture_scores = {
        "GREEN": 85,
        "YELLOW": 65,
        "ORANGE": 45,
        "RED": 30,
        "CRITICAL": 15,
    }
    score = posture_scores.get(posture, 50)
    escalation = str(cpe_doc.get("escalation_state") or "GREEN").upper()
    esc_penalty = ESCALATION_SEVERITY.get(escalation, 0) * 8
    score = max(10, score - esc_penalty)
    for driver in governor_doc.get("top_drivers") or []:
        if driver and driver not in concerns:
            concerns.append(str(driver))
    return score, concerns[:5]


def _readiness_health_score(readiness_doc: Dict[str, Any]) -> Tuple[int, List[str]]:
    concerns: List[str] = list(readiness_doc.get("failed_checks") or [])
    passing = readiness_doc.get("checks_passing_count", 0)
    total = readiness_doc.get("checks_total", 8) or 8
    score = int(round((passing / total) * 100)) if total else 0
    status = str(readiness_doc.get("readiness_status") or "NOT_READY")
    if status == "NOT_READY":
        score = min(score, 40)
    elif status == "PARTIALLY_READY":
        score = min(score, 65)
    if readiness_doc.get("live_execution_permitted"):
        score = 0
        concerns.insert(0, "Live execution gate open")
    return score, concerns[:5]


def _certification_health_score(cert_doc: Dict[str, Any]) -> Tuple[int, List[str]]:
    score = int(round(_safe_float(cert_doc.get("certification_score"), 0.0) or 0.0))
    concerns = list(cert_doc.get("failed_requirements") or [])[:5]
    status = str(cert_doc.get("certification_status") or "NOT_CERTIFIED")
    if status == "NOT_CERTIFIED":
        score = min(score, 40)
    elif status == "PARTIALLY_CERTIFIED":
        score = min(score, 70)
    return score, concerns


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _derive_committee_status(
    domains: List[Dict[str, Any]],
    cpe_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
) -> str:
    statuses = [str(d.get("status") or "MONITOR") for d in domains]
    escalation = str(cpe_doc.get("escalation_state") or "GREEN").upper()
    gov_level = str(gov_doc.get("governance_awareness_level") or "").upper()

    if (
        escalation == "CRITICAL"
        or gov_level == "CRITICAL_INTERVENTION"
        or statuses.count("CRITICAL") >= 2
    ):
        return "ESCALATE"
    if (
        gov_level in GOVERNANCE_BLOCK_LEVELS
        or escalation in ("RED", "ORANGE")
        or "CRITICAL" in statuses
        or statuses.count("CONCERN") >= 2
    ):
        return "REVIEW_REQUIRED"
    if "CONCERN" in statuses or "MONITOR" in statuses:
        return "MONITOR"
    return "CLEAR"


def _derive_overall_assessment(
    cpi_doc: Dict[str, Any],
    domains: List[Dict[str, Any]],
) -> str:
    band = str(cpi_doc.get("health_band") or "ELEVATED_RISK").upper()
    critical_count = sum(1 for d in domains if d.get("status") == "CRITICAL")
    concern_count = sum(1 for d in domains if d.get("status") == "CONCERN")
    if critical_count >= 2:
        return "CRITICAL"
    if band in ("CRITICAL", "HIGH_RISK") or critical_count >= 1:
        return "HIGH_RISK" if band != "CRITICAL" else "CRITICAL"
    if band == "ELEVATED_RISK" or concern_count >= 2:
        return "ELEVATED_RISK"
    if band in ("CAUTION",) or concern_count >= 1:
        return "CAUTION"
    if band in ("HEALTHY", "EXCELLENT"):
        return band
    return band or "ELEVATED_RISK"


def compute_risk_committee_oversight(
    *,
    cpi_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 25: committee assessment across five preservation domains."""
    ts = _iso_utc()

    portfolio_score, portfolio_concerns = _portfolio_health_score(cpi_doc)
    governance_score, governance_concerns = _governance_health_score(gov_doc, auth_doc)
    preservation_score, preservation_concerns = _preservation_health_score(governor_doc, cpe_doc)
    readiness_score, readiness_concerns = _readiness_health_score(readiness_doc)
    certification_score, certification_concerns = _certification_health_score(cert_doc)

    domain_specs = [
        ("Portfolio Health", portfolio_score, portfolio_concerns, cpi_doc),
        ("Governance Health", governance_score, governance_concerns, gov_doc),
        ("Preservation Health", preservation_score, preservation_concerns, governor_doc),
        ("Readiness Health", readiness_score, readiness_concerns, readiness_doc),
        ("Certification Health", certification_score, certification_concerns, cert_doc),
    ]

    domains: List[Dict[str, Any]] = []
    for name, score, concerns, source in domain_specs:
        domains.append(
            {
                "domain": name,
                "score": score,
                "status": _domain_status(score),
                "concerns": concerns,
                "source_generated_at": source.get("generated_at"),
            }
        )

    all_concerns: List[str] = []
    for d in domains:
        for c in d.get("concerns") or []:
            if c and c not in all_concerns:
                all_concerns.append(str(c))

    committee_status = _derive_committee_status(domains, cpe_doc, gov_doc)
    overall_assessment = _derive_overall_assessment(cpi_doc, domains)
    avg_score = round(sum(d["score"] for d in domains) / len(domains), 1) if domains else 0.0

    return {
        "generated_at": ts,
        "committee_status": committee_status,
        "overall_assessment": overall_assessment,
        "average_domain_score": avg_score,
        "top_concerns": all_concerns[:5],
        "domains": domains,
        "capital_preservation_score": cpi_doc.get("capital_preservation_score"),
        "health_band": cpi_doc.get("health_band"),
        "escalation_state": cpe_doc.get("escalation_state"),
        "preservation_posture": governor_doc.get("preservation_posture"),
        "certification_status": cert_doc.get("certification_status"),
        "disclaimer": "Committee oversight is advisory only. No trades or portfolio changes.",
    }


def _entry_certification_status(
    *,
    overall_auth: bool,
    execution_permitted: bool,
    queue_status: Optional[str] = None,
    evaluation: Optional[str] = None,
    cert_status: str,
) -> str:
    if execution_permitted or overall_auth:
        return "NOT_CERTIFIED"
    if queue_status == "APPROVED":
        return "APPROVED_PENDING_EXECUTION_BLOCK"
    if queue_status == "PENDING_REVIEW":
        return "PENDING_HUMAN_REVIEW"
    if queue_status == "REJECTED":
        return "REJECTED"
    if evaluation == "BENEFICIAL" and cert_status == "CERTIFIED_FOR_PAPER_PROTECTION":
        return "CERTIFIED_FOR_PAPER"
    if evaluation == "BENEFICIAL":
        return "SIMULATION_VALIDATED"
    return cert_status if cert_status else "NOT_CERTIFIED"


def compute_accountability_registry(
    *,
    governor_doc: Dict[str, Any],
    queue_doc: Dict[str, Any],
    candidates_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    audit_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    policy_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 26: traceability registry for protective decision paths."""
    ts = _iso_utc()
    cert_status = str(cert_doc.get("certification_status") or "NOT_CERTIFIED")
    entries: List[Dict[str, Any]] = []

    gov_ts = str(governor_doc.get("generated_at") or ts)
    entries.append(
        {
            "decision_id": f"governor-{gov_ts}",
            "origin": "Capital Preservation Governor",
            "governance_source": "Governance Risk Summary",
            "approval_source": "Human Approval Center",
            "policy_source": "Protective Action Policy Center",
            "certification_status": _entry_certification_status(
                overall_auth=bool(auth_doc.get("overall_authorization")),
                execution_permitted=bool(governor_doc.get("live_execution_permitted")),
                cert_status=cert_status,
            ),
            "decision_type": "PRESERVATION_POSTURE",
            "decision_result": governor_doc.get("preservation_posture"),
            "details": ", ".join(governor_doc.get("top_drivers") or [])[:120] or None,
            "timestamp": gov_ts,
            "execution_permitted": False,
        }
    )

    for req in queue_doc.get("requests") or []:
        if not isinstance(req, dict):
            continue
        status = str(req.get("status") or "PENDING_REVIEW")
        entries.append(
            {
                "decision_id": str(req.get("request_id") or req.get("candidate_fingerprint")),
                "origin": "Human Approval Center",
                "governance_source": "Governance Authorization",
                "approval_source": "Human Approval Center",
                "policy_source": req.get("policy_type") or "Protective Action Policy",
                "certification_status": _entry_certification_status(
                    overall_auth=bool(req.get("execution_permitted")),
                    execution_permitted=bool(req.get("execution_permitted")),
                    queue_status=status,
                    cert_status=cert_status,
                ),
                "decision_type": "APPROVAL_REQUEST",
                "decision_result": status,
                "details": req.get("reason") or req.get("candidate_action"),
                "timestamp": req.get("last_seen_at") or req.get("created_at"),
                "execution_permitted": False,
            }
        )

    seen_candidates = {
        e.get("decision_id") for e in entries if e.get("decision_type") == "APPROVAL_REQUEST"
    }
    for cand in candidates_doc.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        cid = str(cand.get("candidate_id") or "")
        if cid in seen_candidates:
            continue
        entries.append(
            {
                "decision_id": cid or f"candidate-{cand.get('candidate_action')}",
                "origin": "Defensive Automation Sandbox",
                "governance_source": "Governance Risk Summary",
                "approval_source": "Human Approval Center",
                "policy_source": cand.get("policy_type") or "Protective Action Policy",
                "certification_status": _entry_certification_status(
                    overall_auth=False,
                    execution_permitted=bool(cand.get("execution_permitted")),
                    cert_status=cert_status,
                ),
                "decision_type": "ACTION_CANDIDATE",
                "decision_result": cand.get("status") or "SIMULATION_ONLY",
                "details": cand.get("reason") or cand.get("summary"),
                "timestamp": candidates_doc.get("generated_at"),
                "execution_permitted": False,
            }
        )

    for ev in evaluation_doc.get("evaluations") or []:
        if not isinstance(ev, dict):
            continue
        entries.append(
            {
                "decision_id": str(ev.get("trial_id") or ev.get("trial_name")),
                "origin": "Protective Action Evaluation",
                "governance_source": "Governance Risk Summary",
                "approval_source": "Human Approval Center",
                "policy_source": "Protective Action Policy Center",
                "certification_status": _entry_certification_status(
                    overall_auth=False,
                    execution_permitted=bool(ev.get("execution_permitted")),
                    evaluation=str(ev.get("evaluation") or ""),
                    cert_status=cert_status,
                ),
                "decision_type": "TRIAL_EVALUATION",
                "decision_result": ev.get("evaluation"),
                "details": f"{ev.get('trial_name')} score={ev.get('effectiveness_score')}",
                "timestamp": evaluation_doc.get("generated_at"),
                "execution_permitted": False,
            }
        )

    for idx, event in enumerate((audit_doc.get("events") or [])[:20]):
        if not isinstance(event, dict):
            continue
        etype = str(event.get("event_type") or "UNKNOWN")
        if etype in ("ALERT", "GOVERNOR", "ESCALATION", "AUTHORIZATION", "EVALUATION"):
            entries.append(
                {
                    "decision_id": f"audit-{etype.lower()}-{idx}",
                    "origin": str(event.get("event_source") or "Capital Preservation Audit"),
                    "governance_source": "Governance Risk Summary",
                    "approval_source": "Human Approval Center",
                    "policy_source": "Protective Action Policy Center",
                    "certification_status": cert_status,
                    "decision_type": f"AUDIT_{etype}",
                    "decision_result": event.get("event_result"),
                    "details": event.get("details"),
                    "timestamp": event.get("audit_timestamp"),
                    "execution_permitted": False,
                }
            )

    entries.sort(key=lambda e: e.get("timestamp") or "", reverse=True)

    by_origin: Dict[str, int] = {}
    for e in entries:
        origin = str(e.get("origin") or "Unknown")
        by_origin[origin] = by_origin.get(origin, 0) + 1

    not_certified = sum(
        1
        for e in entries
        if str(e.get("certification_status") or "").startswith("NOT")
        or e.get("certification_status") == "PENDING_HUMAN_REVIEW"
    )

    return {
        "generated_at": ts,
        "entry_count": len(entries),
        "entries": entries,
        "summary_by_origin": by_origin,
        "not_certified_count": not_certified,
        "governance_awareness_level": gov_doc.get("governance_awareness_level"),
        "overall_authorization": auth_doc.get("overall_authorization", False),
        "policy_execution_enabled": policy_doc.get("global_execution_enabled", False),
        "disclaimer": "Accountability registry is read-only traceability. No execution.",
    }


def _board_status(committee_doc: Dict[str, Any]) -> str:
    committee = str(committee_doc.get("committee_status") or "MONITOR")
    if committee == "ESCALATE":
        return "SUSPENDED"
    if committee == "REVIEW_REQUIRED":
        return "REVIEW"
    return "ACTIVE"


def _governance_confidence(
    *,
    committee_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    accountability_doc: Dict[str, Any],
) -> int:
    avg_domain = _safe_float(committee_doc.get("average_domain_score"), 50.0) or 50.0
    cert_score = _safe_float(cert_doc.get("certification_score"), 0.0) or 0.0
    gov_conf = int(governor_doc.get("governor_confidence") or 0)
    entry_count = accountability_doc.get("entry_count", 0)
    trace_bonus = min(10, entry_count // 3)
    raw = avg_domain * 0.35 + cert_score * 0.35 + gov_conf * 0.25 + trace_bonus
    return int(max(0, min(99, round(raw))))


def compute_preservation_governance_board(
    *,
    committee_doc: Dict[str, Any],
    accountability_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 27: unified authority layer (advisory only, human-controlled)."""
    ts = _iso_utc()

    confidence = _governance_confidence(
        committee_doc=committee_doc,
        cert_doc=cert_doc,
        governor_doc=governor_doc,
        accountability_doc=accountability_doc,
    )

    recommendations: List[str] = []
    for concern in committee_doc.get("top_concerns") or []:
        recommendations.append(f"Committee review: {concern}")
    for action in governor_doc.get("recommended_review_actions") or []:
        if action and action not in recommendations:
            recommendations.append(str(action))
    for req in cert_doc.get("failed_requirements") or []:
        rec = f"Certification gap: {req}"
        if rec not in recommendations:
            recommendations.append(rec)
    if str(committee_doc.get("committee_status")) == "ESCALATE":
        recommendations.insert(0, "Escalate to senior governance review (advisory only)")

    return {
        "generated_at": ts,
        "board_status": _board_status(committee_doc),
        "governance_confidence": confidence,
        "preservation_authority": "HUMAN_CONTROLLED",
        "automation_authorized": False,
        "committee_status": committee_doc.get("committee_status"),
        "overall_assessment": committee_doc.get("overall_assessment"),
        "preservation_posture": governor_doc.get("preservation_posture"),
        "certification_status": cert_doc.get("certification_status"),
        "governance_awareness_level": gov_doc.get("governance_awareness_level"),
        "overall_authorization": auth_doc.get("overall_authorization", False),
        "accountability_entries": accountability_doc.get("entry_count", 0),
        "board_recommendations": recommendations[:8],
        "live_execution_permitted": False,
        "disclaimer": (
            "Governance board is advisory only. "
            "Preservation authority remains HUMAN_CONTROLLED. "
            "No automated trading or portfolio changes."
        ),
    }


def _append_committee_history(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "committee_status",
        "overall_assessment",
        "average_domain_score",
        "top_concern_1",
        "top_concern_2",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def _append_board_history(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "board_status",
        "governance_confidence",
        "preservation_authority",
        "automation_authorized",
        "committee_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def persist_institutional_protection(
    *,
    results_dir: Path,
    cpi_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    queue_doc: Dict[str, Any],
    candidates_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    audit_doc: Dict[str, Any],
    policy_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Run phases 25–27 and write JSON artifacts."""
    results_dir = Path(results_dir)

    committee_doc = compute_risk_committee_oversight(
        cpi_doc=cpi_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        governor_doc=governor_doc,
        cpe_doc=cpe_doc,
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
    )
    _atomic_write_json(committee_doc, results_dir / "risk_committee_oversight.json")

    accountability_doc = compute_accountability_registry(
        governor_doc=governor_doc,
        queue_doc=queue_doc,
        candidates_doc=candidates_doc,
        evaluation_doc=evaluation_doc,
        audit_doc=audit_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        policy_doc=policy_doc,
        cert_doc=cert_doc,
    )
    _atomic_write_json(accountability_doc, results_dir / "accountability_registry.json")

    board_doc = compute_preservation_governance_board(
        committee_doc=committee_doc,
        accountability_doc=accountability_doc,
        cert_doc=cert_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        governor_doc=governor_doc,
    )
    _atomic_write_json(board_doc, results_dir / "preservation_governance_board.json")

    concerns = committee_doc.get("top_concerns") or []
    _append_committee_history(
        results_dir / "risk_committee_oversight_history.csv",
        {
            "timestamp": committee_doc.get("generated_at"),
            "committee_status": committee_doc.get("committee_status"),
            "overall_assessment": committee_doc.get("overall_assessment"),
            "average_domain_score": committee_doc.get("average_domain_score"),
            "top_concern_1": concerns[0] if len(concerns) > 0 else "",
            "top_concern_2": concerns[1] if len(concerns) > 1 else "",
        },
    )

    _append_board_history(
        results_dir / "preservation_governance_board_history.csv",
        {
            "timestamp": board_doc.get("generated_at"),
            "board_status": board_doc.get("board_status"),
            "governance_confidence": board_doc.get("governance_confidence"),
            "preservation_authority": board_doc.get("preservation_authority"),
            "automation_authorized": board_doc.get("automation_authorized"),
            "committee_status": board_doc.get("committee_status"),
        },
    )

    return {
        "risk_committee_oversight": committee_doc,
        "accountability_registry": accountability_doc,
        "preservation_governance_board": board_doc,
    }
