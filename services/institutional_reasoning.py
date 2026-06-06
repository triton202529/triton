"""
TRITON Institutional Reasoning — Phases 37–39.

Causal reasoning, explainability, and institutional insight engines.
Paper-mode / simulation only. NO live trading, orders, or portfolio changes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from services.institutional_protection import _atomic_write_json, _iso_utc

MAX_HISTORY_ROWS = 5000

INSIGHTS_HISTORY_FIELDNAMES = [
    "timestamp",
    "top_insight",
    "insight_confidence",
    "most_important_risk",
    "most_important_weakness",
    "most_important_opportunity",
    "most_important_governance_concern",
]

ALERT_CAUSE_MAP: Dict[str, str] = {
    "EXCESS_CONCENTRATION": "Concentration Risk",
    "EXCESS_POSITION_DRAWDOWN": "Drawdown Risk",
    "STALE_HEARTBEAT": "Monitoring Risk",
    "OPEN_ORDER_AGING": "Execution Risk",
    "BROKER_DISCONNECTED": "Operational Risk",
    "EXCESS_EXPOSURE": "Exposure Risk",
}

COMPONENT_CAUSE_MAP: Dict[str, str] = {
    "concentration": "Concentration Risk",
    "drawdown": "Drawdown Risk",
    "operational": "Monitoring Risk",
    "exposure": "Exposure Risk",
    "execution": "Execution Risk",
}

CHECK_CAUSE_MAP: Dict[str, str] = {
    "governance": "Governance Block",
    "policy": "Policy Disabled",
    "watchdog": "Monitoring Health",
    "approval": "Approval Gate",
    "broker": "Broker Connectivity",
    "data_freshness": "Data Freshness",
    "lifecycle": "Lifecycle Artifacts",
    "signals": "Signal Freshness",
}


def _active_alert_types(alerts_doc: Dict[str, Any]) -> List[str]:
    return [
        str(a.get("alert_type") or "")
        for a in (alerts_doc.get("active_alerts") or [])
        if a.get("alert_type")
    ]


def _alert_evidence(alerts_doc: Dict[str, Any], alert_type: str) -> Optional[str]:
    for alert in alerts_doc.get("active_alerts") or []:
        if str(alert.get("alert_type") or "") == alert_type:
            subject = alert.get("subject")
            if subject:
                return f"active alert {alert_type}:{subject}"
            return f"active alert {alert_type}"
    return None


def _weak_cpi_components(cpi_doc: Dict[str, Any], threshold: int = 50) -> List[str]:
    weak: List[str] = []
    for name, score in (cpi_doc.get("component_scores") or {}).items():
        if isinstance(score, (int, float)) and score < threshold:
            label = COMPONENT_CAUSE_MAP.get(str(name).lower(), str(name).title())
            if label not in weak:
                weak.append(label)
    return weak


def _confidence_from_evidence(evidence: List[str], base: int = 55) -> int:
    return int(min(99, max(40, base + len(evidence) * 8)))


def _unique_preserve(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def compute_causal_reasoning(
    *,
    alerts_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpi_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    learning_doc: Dict[str, Any],
    memory_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 37 — identify likely cause-effect chains for key institutional issues."""
    ts = _iso_utc()
    analyses: List[Dict[str, Any]] = []

    alert_types = _active_alert_types(alerts_doc)
    cpe_labels = list(cpe_doc.get("escalation_reason_labels") or [])
    cpe_reasons = list(cpe_doc.get("escalation_reason") or [])
    weak_components = _weak_cpi_components(cpi_doc)
    escalation = str(cpe_doc.get("escalation_state") or "GREEN").upper()

    # --- Escalation / RED escalation ---
    if escalation in {"RED", "ORANGE", "CRITICAL", "YELLOW"}:
        causes: List[str] = []
        evidence: List[str] = []

        for label in cpe_labels:
            if label not in causes:
                causes.append(label)
        for comp in weak_components:
            if comp not in causes:
                causes.append(comp)
        for at in alert_types:
            mapped = ALERT_CAUSE_MAP.get(at, at.replace("_", " ").title())
            if mapped not in causes:
                causes.append(mapped)
            ev = _alert_evidence(alerts_doc, at)
            if ev:
                evidence.append(ev)
        for reason in cpe_reasons:
            evidence.append(f"CPE escalation_reason: {reason}")

        patterns = learning_doc.get("patterns") or {}
        for concern in (patterns.get("most_common_governance_concerns") or [])[:3]:
            if concern not in causes:
                causes.append(concern)

        memory_entries = memory_doc.get("entries") or []
        for entry in memory_entries[:5]:
            if entry.get("event_type") == "ESCALATION_CHANGE":
                evidence.append(f"memory: {entry.get('summary')}")
                break

        issue_label = f"{escalation} escalation"
        if not causes:
            causes = ["Unspecified preservation stress"]
        analyses.append(
            {
                "issue": issue_label,
                "likely_causes": _unique_preserve(causes)[:6],
                "confidence": _confidence_from_evidence(evidence, 60),
                "evidence": _unique_preserve(evidence)[:8],
            }
        )

    # --- Certification blocked / failed ---
    cert_status = str(cert_doc.get("certification_status") or "").upper()
    failed_reqs = list(cert_doc.get("failed_requirements") or [])
    if failed_reqs or cert_status in {"BLOCKED", "NOT_CERTIFIED", "PARTIALLY_CERTIFIED"}:
        causes = list(failed_reqs)
        evidence: List[str] = []

        if cert_status:
            evidence.append(f"certification_status={cert_status}")
        for req in failed_reqs:
            evidence.append(f"failed_requirement: {req}")

        blockers = (learning_doc.get("patterns") or {}).get(
            "most_common_certification_blockers"
        ) or []
        for blocker in blockers:
            if blocker not in causes and blocker not in {"PARTIALLY_CERTIFIED", "NOT_CERTIFIED"}:
                causes.append(blocker)

        readiness_status = str(readiness_doc.get("readiness_status") or "")
        if readiness_status and readiness_status != "READY":
            causes.append("Execution Readiness")
            evidence.append(f"readiness_status={readiness_status}")

        auth_gap = any("Authorization" in str(c) for c in causes)
        if auth_gap:
            evidence.append("organizational_learning: Authorization gap recurring")

        analyses.append(
            {
                "issue": (
                    "Certification blocked" if cert_status == "BLOCKED" else "Certification failure"
                ),
                "likely_causes": _unique_preserve(causes)[:6] or ["Uncertified governance gates"],
                "confidence": _confidence_from_evidence(evidence, 58),
                "evidence": _unique_preserve(evidence)[:8],
            }
        )

    # --- Readiness NOT_READY ---
    readiness_status = str(readiness_doc.get("readiness_status") or "").upper()
    failed_checks = list(readiness_doc.get("failed_checks") or [])
    if readiness_status == "NOT_READY" or failed_checks:
        causes = [CHECK_CAUSE_MAP.get(c, c.replace("_", " ").title()) for c in failed_checks]
        evidence: List[str] = [f"readiness_status={readiness_status or 'NOT_READY'}"]

        check_details = readiness_doc.get("check_details") or {}
        for check in failed_checks:
            detail = check_details.get(check)
            if detail:
                evidence.append(f"failed_check {check}: {detail}")
            else:
                evidence.append(f"failed_check: {check}")

        for lesson in (learning_doc.get("patterns") or {}).get("highest_value_lessons") or []:
            if "readiness" in str(lesson).lower():
                evidence.append(f"learning pattern: {lesson[:100]}")
                break

        analyses.append(
            {
                "issue": "Readiness NOT_READY",
                "likely_causes": _unique_preserve(causes)[:6] or ["Unresolved readiness gates"],
                "confidence": _confidence_from_evidence(evidence, 62),
                "evidence": _unique_preserve(evidence)[:8],
            }
        )

    # --- Concentration risk persistence ---
    concentration_alerts = [a for a in alert_types if a == "EXCESS_CONCENTRATION"]
    conc_score = (cpi_doc.get("component_scores") or {}).get("concentration")
    if concentration_alerts or (isinstance(conc_score, (int, float)) and conc_score < 50):
        causes = ["Concentration Risk"]
        evidence: List[str] = []

        for alert in alerts_doc.get("active_alerts") or []:
            if str(alert.get("alert_type") or "") == "EXCESS_CONCENTRATION":
                details = alert.get("details") or {}
                pct = details.get("portfolio_pct")
                subject = alert.get("subject") or "portfolio"
                if pct is not None:
                    evidence.append(f"active alert EXCESS_CONCENTRATION ({subject} at {pct:.1f}%)")
                else:
                    evidence.append(f"active alert EXCESS_CONCENTRATION ({subject})")
                duration = alert.get("duration_minutes")
                if duration and float(duration) > 60:
                    causes.append("Persistent Over-Concentration")
                    evidence.append(f"alert duration {float(duration):.0f} minutes")

        if isinstance(conc_score, (int, float)):
            evidence.append(f"CPI concentration component={conc_score}")
            if conc_score == 0:
                causes.append("Critical Concentration Score")

        for entry in memory_doc.get("entries") or []:
            summary = str(entry.get("summary") or "")
            if "CONCENTRATION" in summary.upper() or "concentration" in summary.lower():
                evidence.append(f"memory: {summary[:90]}")
                break

        analyses.append(
            {
                "issue": "Concentration risk persistence",
                "likely_causes": _unique_preserve(causes),
                "confidence": _confidence_from_evidence(evidence, 65),
                "evidence": _unique_preserve(evidence)[:8],
            }
        )

    return {
        "generated_at": ts,
        "reasoning_count": len(analyses),
        "analyses": analyses,
        "disclaimer": (
            "Causal reasoning is advisory diagnostic analysis only. "
            "No trades, orders, or portfolio modifications."
        ),
    }


def compute_institutional_explanations(
    *,
    governor_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    strategic_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    board_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 38 — plain-language explanations for institutional conclusions."""
    ts = _iso_utc()
    explanations: List[Dict[str, Any]] = []

    posture = str(governor_doc.get("preservation_posture") or "UNKNOWN").upper()
    top_drivers = list(governor_doc.get("top_drivers") or [])
    gov_conf = governor_doc.get("governor_confidence")
    cps = governor_doc.get("capital_preservation_score")

    posture_text = (
        f"The capital preservation governor is at {posture} posture because "
        f"{', '.join(top_drivers[:3]) if top_drivers else 'multiple preservation stress signals'} "
        f"are actively driving preservation risk."
    )
    explanations.append(
        {
            "topic": "Governor",
            "subject": "preservation_posture",
            "explanation": posture_text,
            "supporting_facts": _unique_preserve(
                [
                    f"preservation_posture={posture}",
                    f"governor_confidence={gov_conf}%",
                    f"capital_preservation_score={cps}",
                    f"escalation_state={cpe_doc.get('escalation_state')}",
                    *[f"driver: {d}" for d in top_drivers[:5]],
                ]
            ),
        }
    )

    cert_status = str(cert_doc.get("certification_status") or "UNKNOWN")
    failed_reqs = list(cert_doc.get("failed_requirements") or [])
    cert_score = cert_doc.get("certification_score")
    readiness_status = str(readiness_doc.get("readiness_status") or "NOT_READY")
    overall_auth = bool(auth_doc.get("overall_authorization"))

    if cert_status.upper() in {"BLOCKED", "NOT_CERTIFIED", "PARTIALLY_CERTIFIED"}:
        cert_explanation = (
            f"Certification remains {cert_status.lower().replace('_', ' ')} because "
            f"{', '.join(failed_reqs[:4]) if failed_reqs else 'key requirements'} "
            "are not satisfied."
        )
        if readiness_status != "READY":
            cert_explanation += " Execution readiness and authorization requirements remain unmet."
        explanations.append(
            {
                "topic": "Certification",
                "subject": "certification_status",
                "explanation": cert_explanation,
                "supporting_facts": _unique_preserve(
                    [
                        f"certification_status={cert_status}",
                        f"certification_score={cert_score}",
                        f"readiness_status={readiness_status}",
                        f"overall_authorization={overall_auth}",
                        *[f"failed_requirement: {r}" for r in failed_reqs],
                    ]
                ),
            }
        )

    failed_checks = list(readiness_doc.get("failed_checks") or [])
    check_details = readiness_doc.get("check_details") or {}
    if readiness_status.upper() != "READY":
        failed_labels = ", ".join(failed_checks) if failed_checks else "multiple gates"
        readiness_explanation = (
            f"Execution readiness is {readiness_status} because checks failed for "
            f"{failed_labels}. Live execution remains blocked in paper mode."
        )
        facts = [
            f"readiness_status={readiness_status}",
            f"checks_passing={readiness_doc.get('checks_passing_count')}/{readiness_doc.get('checks_total')}",
            f"live_execution_permitted={readiness_doc.get('live_execution_permitted', False)}",
        ]
        for check in failed_checks:
            detail = check_details.get(check)
            facts.append(f"{check}: {detail}" if detail else f"failed_check: {check}")

        explanations.append(
            {
                "topic": "Readiness",
                "subject": "readiness_status",
                "explanation": readiness_explanation,
                "supporting_facts": facts,
            }
        )

    automation_status = str(strategic_doc.get("automation_status") or "NOT_AUTHORIZED")
    automation_authorized = bool(board_doc.get("automation_authorized"))
    if automation_status.upper() != "AUTHORIZED" or not automation_authorized:
        gate_reasons = auth_doc.get("gate_reasons") or {}
        auth_facts = [
            f"automation_status={automation_status}",
            f"board_automation_authorized={automation_authorized}",
            f"overall_authorization={auth_doc.get('overall_authorization')}",
            f"governance_authorized={auth_doc.get('governance_authorized')}",
            f"policy_authorized={auth_doc.get('policy_authorized')}",
            f"execution_authorized={auth_doc.get('execution_authorized')}",
        ]
        for gate, reason in gate_reasons.items():
            auth_facts.append(f"{gate}: {reason}")

        auth_explanation = (
            "Automation is not authorized because governance, policy, and execution "
            "authorization gates are not collectively satisfied. Paper-mode lock "
            "prevents live execution by design."
        )
        if not auth_doc.get("governance_authorized"):
            auth_explanation = (
                "Automation is not authorized because governance awareness blocks action "
                "and overall authorization gates remain closed."
            )

        explanations.append(
            {
                "topic": "Authorization",
                "subject": "automation_status",
                "explanation": auth_explanation,
                "supporting_facts": auth_facts,
            }
        )

    return {
        "generated_at": ts,
        "explanation_count": len(explanations),
        "explanations": explanations,
        "disclaimer": (
            "Institutional explanations are advisory text only. "
            "No automated actions are executed."
        ),
    }


def _append_insights_history(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INSIGHTS_HISTORY_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in INSIGHTS_HISTORY_FIELDNAMES})


def compute_institutional_insights(
    *,
    results_dir: Path,
    causal_doc: Dict[str, Any],
    explanations_doc: Dict[str, Any],
    learning_doc: Dict[str, Any],
    improvement_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    exec_doc: Dict[str, Any],
    strategic_doc: Dict[str, Any],
    intelligence_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 39 — synthesize higher-level strategic observations."""
    ts = _iso_utc()
    results_dir = Path(results_dir)

    exec_summary = exec_doc.get("executive_summary") or exec_doc
    top_risks = list(exec_summary.get("top_risks") or exec_doc.get("top_risks") or [])
    if not top_risks:
        top_risks = list(governor_doc.get("top_drivers") or [])

    most_important_risk = top_risks[0] if top_risks else "Preservation Stress"

    weakest = list(improvement_doc.get("weakest_systems") or [])
    if not weakest:
        weakest = [str(intelligence_doc.get("weakest_area") or "Governance")]
    most_important_weakness = weakest[0]

    opportunities = list(improvement_doc.get("improvement_opportunities") or [])
    enhancements = list(improvement_doc.get("highest_leverage_enhancements") or [])
    most_important_opportunity = (
        enhancements[0]
        if enhancements
        else (opportunities[0] if opportunities else "Strengthen governance alignment")
    )

    gov_concerns = list(strategic_doc.get("top_strategic_concerns") or [])
    learning_concerns = (learning_doc.get("patterns") or {}).get(
        "most_common_governance_concerns"
    ) or []
    if not gov_concerns:
        gov_concerns = learning_concerns
    most_important_governance_concern = (
        gov_concerns[0] if gov_concerns else "Authorization and readiness gaps"
    )

    readiness_block = any(
        "readiness" in str(e.get("explanation", "")).lower()
        for e in (explanations_doc.get("explanations") or [])
    )
    cert_block = any(
        e.get("topic") == "Certification" for e in (explanations_doc.get("explanations") or [])
    )

    if readiness_block and cert_block:
        top_insight = (
            "Execution Readiness remains the dominant constraint preventing certification."
        )
    elif most_important_risk == "Concentration Risk":
        top_insight = (
            "Concentration Risk is the primary preservation threat requiring governance review."
        )
    elif str(governor_doc.get("preservation_posture") or "").upper() == "RED":
        top_insight = (
            f"{most_important_risk} drives RED preservation posture; "
            "certification and readiness gates remain unresolved."
        )
    else:
        top_insight = (
            f"{most_important_weakness} is the primary institutional weakness; "
            f"focus on {most_important_opportunity[:80]}."
        )

    insights: List[Dict[str, Any]] = [
        {
            "category": "risk",
            "insight": f"Most important risk: {most_important_risk}",
            "priority": 1,
        },
        {
            "category": "weakness",
            "insight": f"Most important weakness: {most_important_weakness}",
            "priority": 2,
        },
        {
            "category": "opportunity",
            "insight": f"Most important opportunity: {most_important_opportunity}",
            "priority": 3,
        },
        {
            "category": "governance_concern",
            "insight": f"Most important governance concern: {most_important_governance_concern}",
            "priority": 4,
        },
    ]

    for analysis in (causal_doc.get("analyses") or [])[:2]:
        insights.append(
            {
                "category": "causal",
                "insight": (
                    f"{analysis.get('issue')}: likely caused by "
                    f"{', '.join((analysis.get('likely_causes') or [])[:3])}"
                ),
                "priority": 5,
            }
        )

    depth_signals = [
        int(learning_doc.get("confidence") or 0),
        int(improvement_doc.get("improvement_score") or 0),
        int(governor_doc.get("governor_confidence") or 0),
        len(causal_doc.get("analyses") or []),
        len(explanations_doc.get("explanations") or []),
    ]
    insight_confidence = int(min(99, max(45, sum(depth_signals) / max(1, len(depth_signals)))))

    doc = {
        "generated_at": ts,
        "top_insight": top_insight,
        "insights": insights,
        "insight_confidence": insight_confidence,
        "most_important_risk": most_important_risk,
        "most_important_weakness": most_important_weakness,
        "most_important_opportunity": most_important_opportunity,
        "most_important_governance_concern": most_important_governance_concern,
        "disclaimer": (
            "Institutional insights are strategic observations only. "
            "No trades, orders, or portfolio modifications."
        ),
    }

    history_path = results_dir / "institutional_insights_history.csv"
    _append_insights_history(
        history_path,
        {
            "timestamp": ts,
            "top_insight": top_insight,
            "insight_confidence": insight_confidence,
            "most_important_risk": most_important_risk,
            "most_important_weakness": most_important_weakness,
            "most_important_opportunity": most_important_opportunity,
            "most_important_governance_concern": most_important_governance_concern,
        },
    )

    return doc


def persist_institutional_reasoning(
    *,
    results_dir: Path,
    alerts_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpi_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    strategic_doc: Dict[str, Any],
    board_doc: Dict[str, Any],
    exec_doc: Dict[str, Any],
    intelligence_doc: Dict[str, Any],
    memory_doc: Dict[str, Any],
    learning_doc: Dict[str, Any],
    improvement_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Run phases 37–39 and write JSON artifacts."""
    results_dir = Path(results_dir)

    causal_doc = compute_causal_reasoning(
        alerts_doc=alerts_doc,
        cpe_doc=cpe_doc,
        cpi_doc=cpi_doc,
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
        learning_doc=learning_doc,
        memory_doc=memory_doc,
    )
    _atomic_write_json(causal_doc, results_dir / "causal_reasoning.json")

    explanations_doc = compute_institutional_explanations(
        governor_doc=governor_doc,
        cert_doc=cert_doc,
        readiness_doc=readiness_doc,
        auth_doc=auth_doc,
        strategic_doc=strategic_doc,
        cpe_doc=cpe_doc,
        board_doc=board_doc,
    )
    _atomic_write_json(explanations_doc, results_dir / "institutional_explanations.json")

    insights_doc = compute_institutional_insights(
        results_dir=results_dir,
        causal_doc=causal_doc,
        explanations_doc=explanations_doc,
        learning_doc=learning_doc,
        improvement_doc=improvement_doc,
        governor_doc=governor_doc,
        exec_doc=exec_doc,
        strategic_doc=strategic_doc,
        intelligence_doc=intelligence_doc,
    )
    _atomic_write_json(insights_doc, results_dir / "institutional_insights.json")

    return {
        "causal_reasoning": causal_doc,
        "institutional_explanations": explanations_doc,
        "institutional_insights": insights_doc,
    }
