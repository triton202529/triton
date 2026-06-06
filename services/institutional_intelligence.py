"""
TRITON Institutional Intelligence — Phases 31–33.

Decision quality, institutional intelligence, and strategic self-improvement.
Paper-mode / simulation only. NO live trading, orders, or portfolio changes.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.institutional_operations import (
    MATURITY_BANDS,
    _monitoring_maturity,
    _oversight_maturity,
    _preservation_maturity,
)
from services.institutional_protection import (
    ESCALATION_SEVERITY,
    _atomic_write_json,
    _certification_health_score,
    _governance_health_score,
    _iso_utc,
    _safe_float,
)

QUALITY_BANDS = (
    ("LOW", 0, 49),
    ("MODERATE", 50, 69),
    ("HIGH", 70, 84),
    ("EXCELLENT", 85, 100),
)

DECISION_METRIC_NAMES = (
    "Advisory Accuracy",
    "Escalation Consistency",
    "Preservation Logic Consistency",
    "Governance Consistency",
    "Recommendation Stability",
)

INTELLIGENCE_AREAS = (
    "Monitoring",
    "Governance",
    "Certification",
    "Oversight",
    "Accountability",
    "Preservation",
)

ESCALATION_CPS_RANGES: Dict[str, Tuple[int, int]] = {
    "GREEN": (70, 100),
    "YELLOW": (55, 75),
    "ORANGE": (40, 60),
    "RED": (0, 50),
    "CRITICAL": (0, 35),
}


def _quality_band(score: float) -> str:
    s = int(round(score))
    for name, lo, hi in QUALITY_BANDS:
        if lo <= s <= hi:
            return name
    return "LOW"


def _maturity_band(score: float) -> str:
    s = int(round(score))
    for name, lo, hi in MATURITY_BANDS:
        if lo <= s <= hi:
            return name
    return "FOUNDATIONAL"


def _jaccard_score(a: set, b: set, default: int = 85) -> int:
    if not a and not b:
        return default
    if not a or not b:
        return max(20, default - 35)
    overlap = len(a & b)
    union = len(a | b)
    return int(round((overlap / union) * 100)) if union else default


def _advisory_accuracy(
    cpa_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    alerts_doc: Dict[str, Any],
) -> int:
    advisory_titles = {
        str(a.get("title") or "").strip()
        for a in (cpa_doc.get("advisories") or [])
        if a.get("title")
    }
    escalation_labels = {
        str(x).strip() for x in (cpe_doc.get("escalation_reason_labels") or []) if x
    }
    alert_titles = {
        str(a.get("title") or a.get("alert_type") or "").strip()
        for a in (alerts_doc.get("active_alerts") or [])
        if a.get("title") or a.get("alert_type")
    }

    label_score = _jaccard_score(advisory_titles, escalation_labels, default=88)
    alert_score = _jaccard_score(advisory_titles, alert_titles, default=88)

    cpa_count = len(cpa_doc.get("advisories") or [])
    active_count = len(alerts_doc.get("active_alerts") or [])
    count_delta = abs(cpa_count - active_count)
    count_penalty = min(20, count_delta * 6)

    raw = label_score * 0.45 + alert_score * 0.35 + max(0, 100 - count_penalty) * 0.2
    return int(max(0, min(100, round(raw))))


def _escalation_consistency(
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
) -> int:
    score = 82
    cpi_alerts = int(cpi_doc.get("active_alerts") or 0)
    cpe_alerts = int(cpe_doc.get("active_alerts") or 0)
    if cpi_alerts != cpe_alerts:
        score -= min(25, abs(cpi_alerts - cpe_alerts) * 8)

    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    esc = str(cpe_doc.get("escalation_state") or "GREEN").upper()
    lo, hi = ESCALATION_CPS_RANGES.get(esc, (0, 100))
    if lo <= cps <= hi:
        score += 8
    else:
        dist = min(abs(cps - lo), abs(cps - hi))
        score -= min(30, dist // 2)

    cpe_reasons = set(cpe_doc.get("escalation_reason") or [])
    weak_components = {
        name
        for name, val in (cpi_doc.get("component_scores") or {}).items()
        if isinstance(val, (int, float)) and val < 50
    }
    component_map = {
        "drawdown": "EXCESS_POSITION_DRAWDOWN",
        "concentration": "EXCESS_CONCENTRATION",
        "operational": "BROKER_DISCONNECTED",
        "execution": "OPEN_ORDER_AGING",
    }
    expected_reasons = {component_map[k] for k in weak_components if k in component_map}
    reason_score = _jaccard_score(cpe_reasons, expected_reasons, default=80)
    raw = score * 0.6 + reason_score * 0.4
    return int(max(0, min(100, round(raw))))


def _preservation_logic_consistency(
    governor_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    adaptive_doc: Dict[str, Any],
) -> int:
    posture = str(governor_doc.get("preservation_posture") or "UNKNOWN").upper()
    escalation = str(cpe_doc.get("escalation_state") or "GREEN").upper()
    posture_sev = ESCALATION_SEVERITY.get(posture, 2)
    esc_sev = ESCALATION_SEVERITY.get(escalation, 0)
    alignment = max(0, 100 - abs(posture_sev - esc_sev) * 18)

    gov_best = str(governor_doc.get("best_protection") or "").upper()
    adaptive_best = str(adaptive_doc.get("best_protection") or "").upper()
    protection_match = 95 if gov_best and gov_best == adaptive_best else 55

    avg_eff = _safe_float(evaluation_doc.get("average_effectiveness"), 50.0) or 50.0
    gov_conf = int(governor_doc.get("governor_confidence") or 0)
    eval_alignment = max(0, min(100, round(100 - abs(gov_conf - avg_eff))))

    raw = alignment * 0.4 + protection_match * 0.35 + eval_alignment * 0.25
    return int(max(0, min(100, round(raw))))


def _governance_consistency(
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
) -> int:
    gov_level = str(gov_doc.get("governance_awareness_level") or "").upper()
    gov_esc = str(gov_doc.get("escalation_state") or cpe_doc.get("escalation_state") or "").upper()
    cpe_esc = str(cpe_doc.get("escalation_state") or "GREEN").upper()
    esc_match = max(
        0, 100 - abs(ESCALATION_SEVERITY.get(gov_esc, 0) - ESCALATION_SEVERITY.get(cpe_esc, 0)) * 20
    )

    auth_ok = bool(auth_doc.get("overall_authorization"))
    gov_blocked = gov_level in {
        "MANAGEMENT_REVIEW_REQUIRED",
        "BOARD_REVIEW_REQUIRED",
        "CRITICAL_INTERVENTION",
    }
    auth_alignment = 90 if (not auth_ok and gov_blocked) or (auth_ok and not gov_blocked) else 45

    gov_drivers = {str(x).lower() for x in (gov_doc.get("governance_drivers") or [])}
    failed = {str(x).lower() for x in (readiness_doc.get("failed_checks") or [])}
    driver_score = _jaccard_score(gov_drivers, failed, default=75)

    readiness_status = str(readiness_doc.get("readiness_status") or "NOT_READY").upper()
    readiness_penalty = {"NOT_READY": 15, "PARTIALLY_READY": 8}.get(readiness_status, 0)
    if gov_blocked and readiness_status == "READY":
        readiness_penalty += 12

    raw = esc_match * 0.35 + auth_alignment * 0.35 + driver_score * 0.3 - readiness_penalty
    return int(max(0, min(100, round(raw))))


def _recommendation_stability(
    results_dir: Path,
    cpa_doc: Dict[str, Any],
) -> Tuple[int, bool]:
    history_path = results_dir / "capital_preservation_advisory_history.csv"
    if not history_path.is_file() or history_path.stat().st_size == 0:
        count = len(cpa_doc.get("advisories") or [])
        heuristic = 78 if count <= 2 else (68 if count <= 4 else 58)
        return heuristic, False

    rows: List[Dict[str, str]] = []
    try:
        with open(history_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return 70, False

    if len(rows) < 2:
        return 75, True

    recent = rows[-8:]
    counts = [_safe_float(r.get("advisory_count"), 0) or 0 for r in recent]
    issues = [str(r.get("top_issue") or "") for r in recent]

    count_std = statistics.pstdev(counts) if len(counts) > 1 else 0.0
    issue_changes = sum(1 for i in range(1, len(issues)) if issues[i] != issues[i - 1])
    change_rate = issue_changes / max(1, len(issues) - 1)

    count_score = max(0, 100 - count_std * 12)
    issue_score = max(0, 100 - change_rate * 55)
    return int(max(0, min(100, round(count_score * 0.45 + issue_score * 0.55)))), True


def compute_decision_quality_assessment(
    *,
    results_dir: Path,
    cpa_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpi_doc: Dict[str, Any],
    alerts_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    adaptive_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 31: decision quality scoring across five consistency dimensions."""
    ts = _iso_utc()
    results_dir = Path(results_dir)

    stability_score, history_used = _recommendation_stability(results_dir, cpa_doc)

    metrics = {
        "Advisory Accuracy": _advisory_accuracy(cpa_doc, cpe_doc, alerts_doc),
        "Escalation Consistency": _escalation_consistency(cpi_doc, cpe_doc),
        "Preservation Logic Consistency": _preservation_logic_consistency(
            governor_doc, cpe_doc, evaluation_doc, adaptive_doc
        ),
        "Governance Consistency": _governance_consistency(
            gov_doc, auth_doc, readiness_doc, cpe_doc
        ),
        "Recommendation Stability": stability_score,
    }

    metric_rows = [{"metric": k, "score": v} for k, v in metrics.items()]
    composite = round(sum(metrics.values()) / len(metrics), 1) if metrics else 0.0
    strongest = max(metrics, key=metrics.get)
    weakest = min(metrics, key=metrics.get)

    return {
        "generated_at": ts,
        "decision_quality_score": int(round(composite)),
        "quality_band": _quality_band(composite),
        "strongest_area": strongest.replace(" Consistency", "")
        .replace(" Accuracy", "")
        .replace(" Stability", ""),
        "weakest_area": weakest.replace(" Logic Consistency", " Consistency").replace(
            " Accuracy", " Consistency"
        ),
        "metrics": metric_rows,
        "history_used": history_used,
        "disclaimer": "Decision quality assessment is advisory only. No automated actions.",
    }


def _accountability_score(accountability_doc: Dict[str, Any]) -> int:
    entries = int(accountability_doc.get("entry_count") or 0)
    not_certified = int(accountability_doc.get("not_certified_count") or 0)
    trace = min(40, entries * 2)
    penalty = min(35, not_certified * 3)
    return int(max(0, min(100, round(55 + trace - penalty))))


def _coordination_score(scores: List[int]) -> int:
    if not scores:
        return 0
    if len(scores) == 1:
        return 100
    avg = sum(scores) / len(scores)
    variance = statistics.pvariance(scores)
    std = variance**0.5
    penalty = min(45, std * 1.8)
    return int(max(0, min(100, round(100 - penalty + (avg - 50) * 0.05))))


def compute_institutional_intelligence(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    alerts_doc: Dict[str, Any],
    exec_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    adaptive_doc: Dict[str, Any],
    committee_doc: Dict[str, Any],
    board_doc: Dict[str, Any],
    accountability_doc: Dict[str, Any],
    maturity_doc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 32: institutional intelligence across six operational layers."""
    ts = _iso_utc()

    maturity_lookup: Dict[str, int] = {}
    if maturity_doc:
        for cat in maturity_doc.get("categories") or []:
            if isinstance(cat, dict) and cat.get("category"):
                maturity_lookup[str(cat["category"])] = int(cat.get("score") or 0)

    governance_score, _ = _governance_health_score(gov_doc, auth_doc)
    certification_score, _ = _certification_health_score(cert_doc)

    area_scores = {
        "Monitoring": maturity_lookup.get("Monitoring")
        or _monitoring_maturity(cpi_doc, alerts_doc, exec_doc),
        "Governance": maturity_lookup.get("Governance") or governance_score,
        "Certification": maturity_lookup.get("Certification") or certification_score,
        "Oversight": maturity_lookup.get("Oversight")
        or _oversight_maturity(committee_doc, board_doc, accountability_doc),
        "Accountability": _accountability_score(accountability_doc),
        "Preservation": maturity_lookup.get("Preservation")
        or _preservation_maturity(governor_doc, cpe_doc, adaptive_doc),
    }

    areas = [
        {
            "area": name,
            "score": area_scores[name],
            "band": _maturity_band(area_scores[name]),
        }
        for name in INTELLIGENCE_AREAS
    ]

    scores_list = [a["score"] for a in areas]
    overall = round(sum(scores_list) / len(scores_list), 1) if scores_list else 0.0
    coordination = _coordination_score(scores_list)

    return {
        "generated_at": ts,
        "institutional_intelligence_score": int(round(overall)),
        "institutional_band": _maturity_band(overall),
        "coordination_score": coordination,
        "areas": areas,
        "strongest_area": max(areas, key=lambda a: a["score"])["area"],
        "weakest_area": min(areas, key=lambda a: a["score"])["area"],
        "disclaimer": "Institutional intelligence is diagnostic only. No execution permitted.",
    }


def _data_depth_score(docs: List[Dict[str, Any]]) -> int:
    if not docs:
        return 0
    present = sum(1 for d in docs if d)
    richness = sum(min(1.0, len(d) / 8) for d in docs if d)
    raw = (present / len(docs)) * 55 + (richness / len(docs)) * 45
    return int(max(0, min(99, round(raw))))


def compute_strategic_self_improvement(
    *,
    decision_doc: Dict[str, Any],
    intelligence_doc: Dict[str, Any],
    maturity_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    audit_doc: Dict[str, Any],
    strategic_doc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 33: strategic self-improvement prioritization from weakest signals."""
    ts = _iso_utc()

    candidates: List[Tuple[str, int, str]] = []

    for metric in decision_doc.get("metrics") or []:
        name = str(metric.get("metric") or "")
        score = int(metric.get("score") or 0)
        if score < 70:
            candidates.append((name, 100 - score, f"Decision quality gap in {name} ({score}/100)"))

    for cat in maturity_doc.get("categories") or []:
        name = str(cat.get("category") or "")
        score = int(cat.get("score") or 0)
        if score < 65:
            candidates.append((name, 100 - score, f"Maturity lag in {name} ({score}/100)"))

    for area_name, area in (cert_doc.get("areas") or {}).items():
        if isinstance(area, dict) and not area.get("certified"):
            score = int(area.get("score") or 0)
            candidates.append((str(area_name), 100 - score + 10, f"Certification gap: {area_name}"))

    for check in readiness_doc.get("failed_checks") or []:
        candidates.append((str(check).title(), 75, f"Readiness failure: {check}"))

    weakest_intel = intelligence_doc.get("weakest_area")
    if weakest_intel:
        intel_score = next(
            (
                a["score"]
                for a in (intelligence_doc.get("areas") or [])
                if a.get("area") == weakest_intel
            ),
            50,
        )
        candidates.append(
            (str(weakest_intel), 100 - intel_score, f"Weakest intelligence layer: {weakest_intel}")
        )

    if not auth_doc.get("overall_authorization"):
        candidates.append(("Authorization", 80, "Overall authorization gate failing"))

    if str(readiness_doc.get("readiness_status") or "").upper() == "NOT_READY":
        candidates.append(("Execution Readiness", 85, "System readiness status is NOT_READY"))

    avg_eff = _safe_float(evaluation_doc.get("average_effectiveness"), 50.0) or 50.0
    if avg_eff < 45:
        candidates.append(
            ("Protective Evaluation", int(100 - avg_eff), "Low trial effectiveness scores")
        )

    audit_events = int(audit_doc.get("event_count") or 0)
    if audit_events < 5:
        candidates.append(("Audit Trail", 40, "Limited preservation audit history"))

    candidates.sort(key=lambda x: x[1], reverse=True)

    system_weights: Dict[str, int] = {}
    for name, weight, _ in candidates:
        system_weights[name] = system_weights.get(name, 0) + weight

    weakest_systems = [
        k for k, _ in sorted(system_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    improvement_opportunities = [msg for _, _, msg in candidates[:8]]

    leverage_map = {
        "Execution Readiness": "Close readiness gate failures to unlock downstream certification",
        "Readiness": "Resolve failed readiness checks and refresh watchdog heartbeat",
        "Governance": "Align governance awareness with authorization posture",
        "Certification": "Certify failed governance, authorization, and readiness areas",
        "Authorization": "Reconcile authorization gates with governance drivers",
        "Monitoring": "Reduce active alert load and stabilize monitoring health",
        "Preservation": "Align governor posture with adaptive protection recommendations",
        "Oversight": "Strengthen committee and board oversight confidence",
        "Accountability": "Expand accountability registry coverage and certification linkage",
        "Advisory Accuracy": "Tighten advisory-to-alert alignment in CPA engine",
        "Escalation Consistency": "Synchronize CPI and CPE escalation signals",
    }
    highest_leverage = []
    for system in weakest_systems[:4]:
        hint = (
            leverage_map.get(system) or f"Improve {system} layer scoring and cross-layer alignment"
        )
        if hint not in highest_leverage:
            highest_leverage.append(hint)

    technical_debt: List[str] = []
    if readiness_doc.get("failed_checks"):
        technical_debt.append("Readiness gate failures unresolved across cycles")
    if cert_doc.get("failed_requirements"):
        for req in (cert_doc.get("failed_requirements") or [])[:4]:
            technical_debt.append(f"Uncertified requirement: {req}")
    if not auth_doc.get("policy_authorized"):
        technical_debt.append("Protective action policies disabled")
    if str(gov_doc.get("governance_awareness_level") or "").upper() in {
        "MANAGEMENT_REVIEW_REQUIRED",
        "BOARD_REVIEW_REQUIRED",
    }:
        technical_debt.append("Governance review backlog blocking authorization")
    if int(decision_doc.get("decision_quality_score") or 0) < 70:
        technical_debt.append("Decision quality below HIGH threshold")
    if strategic_doc and str(strategic_doc.get("automation_status") or "") == "NOT_AUTHORIZED":
        technical_debt.append("Automation remains NOT_AUTHORIZED by design (paper-mode)")

    top_priority = weakest_systems[0] if weakest_systems else "Monitoring"
    focus_pool = (
        weakest_systems[:3] if weakest_systems else ["Monitoring", "Governance", "Readiness"]
    )
    recommended_focus = [f.replace("_", " ").title() for f in focus_pool]

    depth = _data_depth_score(
        [decision_doc, intelligence_doc, maturity_doc, cert_doc, readiness_doc, audit_doc]
    )

    return {
        "generated_at": ts,
        "top_priority": top_priority.replace("_", " ").title(),
        "improvement_score": depth,
        "recommended_focus": recommended_focus,
        "improvement_opportunities": improvement_opportunities,
        "weakest_systems": weakest_systems,
        "highest_leverage_enhancements": highest_leverage,
        "technical_debt_areas": technical_debt[:8],
        "decision_quality_score": decision_doc.get("decision_quality_score"),
        "institutional_intelligence_score": intelligence_doc.get(
            "institutional_intelligence_score"
        ),
        "overall_maturity": maturity_doc.get("overall_maturity"),
        "disclaimer": (
            "Strategic self-improvement is advisory prioritization only. "
            "No trades, orders, or portfolio modifications."
        ),
    }


def _append_decision_quality_history(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "decision_quality_score",
        "quality_band",
        "strongest_area",
        "weakest_area",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def _append_institutional_intelligence_history(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "institutional_intelligence_score",
        "institutional_band",
        "coordination_score",
        "strongest_area",
        "weakest_area",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def persist_institutional_intelligence(
    *,
    results_dir: Path,
    cpa_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpi_doc: Dict[str, Any],
    alerts_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    adaptive_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    exec_doc: Dict[str, Any],
    committee_doc: Dict[str, Any],
    board_doc: Dict[str, Any],
    accountability_doc: Dict[str, Any],
    audit_doc: Dict[str, Any],
    maturity_doc: Dict[str, Any],
    strategic_doc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run phases 31–33 and write JSON artifacts."""
    results_dir = Path(results_dir)

    decision_doc = compute_decision_quality_assessment(
        results_dir=results_dir,
        cpa_doc=cpa_doc,
        cpe_doc=cpe_doc,
        cpi_doc=cpi_doc,
        alerts_doc=alerts_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        readiness_doc=readiness_doc,
        governor_doc=governor_doc,
        evaluation_doc=evaluation_doc,
        adaptive_doc=adaptive_doc,
    )
    _atomic_write_json(decision_doc, results_dir / "decision_quality_assessment.json")

    intelligence_doc = compute_institutional_intelligence(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        alerts_doc=alerts_doc,
        exec_doc=exec_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        cert_doc=cert_doc,
        readiness_doc=readiness_doc,
        governor_doc=governor_doc,
        adaptive_doc=adaptive_doc,
        committee_doc=committee_doc,
        board_doc=board_doc,
        accountability_doc=accountability_doc,
        maturity_doc=maturity_doc,
    )
    _atomic_write_json(intelligence_doc, results_dir / "institutional_intelligence.json")

    improvement_doc = compute_strategic_self_improvement(
        decision_doc=decision_doc,
        intelligence_doc=intelligence_doc,
        maturity_doc=maturity_doc,
        cert_doc=cert_doc,
        readiness_doc=readiness_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        evaluation_doc=evaluation_doc,
        audit_doc=audit_doc,
        strategic_doc=strategic_doc,
    )
    _atomic_write_json(improvement_doc, results_dir / "strategic_self_improvement.json")

    _append_decision_quality_history(
        results_dir / "decision_quality_assessment_history.csv",
        {
            "timestamp": decision_doc.get("generated_at"),
            "decision_quality_score": decision_doc.get("decision_quality_score"),
            "quality_band": decision_doc.get("quality_band"),
            "strongest_area": decision_doc.get("strongest_area"),
            "weakest_area": decision_doc.get("weakest_area"),
        },
    )

    _append_institutional_intelligence_history(
        results_dir / "institutional_intelligence_history.csv",
        {
            "timestamp": intelligence_doc.get("generated_at"),
            "institutional_intelligence_score": intelligence_doc.get(
                "institutional_intelligence_score"
            ),
            "institutional_band": intelligence_doc.get("institutional_band"),
            "coordination_score": intelligence_doc.get("coordination_score"),
            "strongest_area": intelligence_doc.get("strongest_area"),
            "weakest_area": intelligence_doc.get("weakest_area"),
        },
    )

    return {
        "decision_quality_assessment": decision_doc,
        "institutional_intelligence": intelligence_doc,
        "strategic_self_improvement": improvement_doc,
    }
