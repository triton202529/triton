"""
TRITON Institutional Operations — Phases 28–30.

Investment committee review, maturity assessment, and strategic oversight center.
Paper-mode / simulation only. NO live trading, orders, or portfolio changes.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.institutional_protection import (
    ESCALATION_SEVERITY,
    GOVERNANCE_BLOCK_LEVELS,
    _atomic_write_json,
    _certification_health_score,
    _domain_status,
    _governance_health_score,
    _iso_utc,
    _portfolio_health_score,
    _readiness_health_score,
    _safe_float,
)


MATURITY_CATEGORIES = (
    "Monitoring",
    "Governance",
    "Preservation",
    "Certification",
    "Readiness",
    "Oversight",
)

MATURITY_BANDS = (
    ("FOUNDATIONAL", 0, 39),
    ("DEVELOPING", 40, 59),
    ("ADVANCED", 60, 79),
    ("INSTITUTIONAL", 80, 100),
)


def _risk_health_score(
    cpe_doc: Dict[str, Any],
    alerts_doc: Dict[str, Any],
    exec_doc: Dict[str, Any],
) -> Tuple[int, List[str]]:
    concerns: List[str] = list(cpe_doc.get("escalation_reason_labels") or [])
    escalation = str(cpe_doc.get("escalation_state") or "GREEN").upper()
    esc_penalty = ESCALATION_SEVERITY.get(escalation, 0) * 10
    score = max(10, 85 - esc_penalty)

    active_count = len(alerts_doc.get("active_alerts") or [])
    if active_count >= 5:
        score = max(10, score - 20)
        if "Elevated Alert Count" not in concerns:
            concerns.append("Elevated Alert Count")
    elif active_count >= 2:
        score = max(10, score - 10)

    exec_summary = exec_doc.get("executive_summary") or {}
    portfolio_health = str(exec_summary.get("portfolio_health") or "").upper()
    if portfolio_health in ("CRITICAL", "HIGH_RISK"):
        score = min(score, 40)
        label = f"Executive Risk: {portfolio_health.replace('_', ' ').title()}"
        if label not in concerns:
            concerns.append(label)

    risk_direction = str(
        exec_doc.get("predictive_outlook", {}).get("risk_direction")
        or exec_summary.get("risk_direction")
        or ""
    ).upper()
    if risk_direction == "DETERIORATING":
        score = max(10, score - 8)
        if "Deteriorating Risk Trend" not in concerns:
            concerns.append("Deteriorating Risk Trend")

    for risk in exec_doc.get("top_risks") or exec_summary.get("top_risks") or []:
        if risk and risk not in concerns:
            concerns.append(str(risk))

    return score, concerns[:5]


def _derive_committee_recommendation(
    areas: List[Dict[str, Any]],
    cpe_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
) -> str:
    statuses = [str(a.get("status") or "MONITOR") for a in areas]
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


def _review_confidence(areas: List[Dict[str, Any]]) -> int:
    if not areas:
        return 0
    avg = sum(int(a.get("score") or 0) for a in areas) / len(areas)
    critical = sum(1 for a in areas if a.get("status") == "CRITICAL")
    concern = sum(1 for a in areas if a.get("status") == "CONCERN")
    penalty = critical * 8 + concern * 4
    return int(max(0, min(99, round(avg - penalty))))


def compute_investment_committee_review(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    alerts_doc: Dict[str, Any],
    exec_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 28: investment committee review across five health areas."""
    ts = _iso_utc()

    portfolio_score, portfolio_concerns = _portfolio_health_score(cpi_doc)
    risk_score, risk_concerns = _risk_health_score(cpe_doc, alerts_doc, exec_doc)
    governance_score, governance_concerns = _governance_health_score(gov_doc, auth_doc)
    certification_score, certification_concerns = _certification_health_score(cert_doc)
    readiness_score, readiness_concerns = _readiness_health_score(readiness_doc)

    area_specs = [
        ("Portfolio Health", portfolio_score, portfolio_concerns, cpi_doc),
        ("Risk Health", risk_score, risk_concerns, exec_doc),
        ("Governance Health", governance_score, governance_concerns, gov_doc),
        ("Certification Health", certification_score, certification_concerns, cert_doc),
        ("Readiness Health", readiness_score, readiness_concerns, readiness_doc),
    ]

    areas: List[Dict[str, Any]] = []
    for name, score, concerns, source in area_specs:
        areas.append(
            {
                "area": name,
                "score": score,
                "status": _domain_status(score),
                "concerns": concerns,
                "source_generated_at": source.get("generated_at"),
            }
        )

    all_concerns: List[str] = []
    for a in areas:
        for c in a.get("concerns") or []:
            if c and c not in all_concerns:
                all_concerns.append(str(c))

    committee_recommendation = _derive_committee_recommendation(areas, cpe_doc, gov_doc)
    confidence = _review_confidence(areas)
    avg_score = round(sum(a["score"] for a in areas) / len(areas), 1) if areas else 0.0

    return {
        "generated_at": ts,
        "committee_recommendation": committee_recommendation,
        "confidence": confidence,
        "top_concerns": all_concerns[:5],
        "average_area_score": avg_score,
        "review_areas": areas,
        "capital_preservation_score": cpi_doc.get("capital_preservation_score"),
        "health_band": cpi_doc.get("health_band"),
        "escalation_state": cpe_doc.get("escalation_state"),
        "active_alert_count": len(alerts_doc.get("active_alerts") or []),
        "certification_status": cert_doc.get("certification_status"),
        "readiness_status": readiness_doc.get("readiness_status"),
        "disclaimer": (
            "Investment committee review is advisory only. "
            "No trades, orders, or portfolio changes."
        ),
    }


def _maturity_band(score: float) -> str:
    s = int(round(score))
    for name, lo, hi in MATURITY_BANDS:
        if lo <= s <= hi:
            return name
    return "FOUNDATIONAL"


def _monitoring_maturity(
    cpi_doc: Dict[str, Any],
    alerts_doc: Dict[str, Any],
    exec_doc: Dict[str, Any],
) -> int:
    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    active = len(alerts_doc.get("active_alerts") or [])
    alert_penalty = min(25, active * 4)
    forecast = (
        _safe_float((exec_doc.get("predictive_outlook") or {}).get("forecast_confidence"), 50.0)
        or 50.0
    )
    raw = cps * 0.55 + forecast * 0.25 + max(0, 20 - alert_penalty)
    return int(max(0, min(100, round(raw))))


def _preservation_maturity(
    governor_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    adaptive_doc: Dict[str, Any],
) -> int:
    posture = str(governor_doc.get("preservation_posture") or "UNKNOWN").upper()
    posture_scores = {
        "GREEN": 85,
        "YELLOW": 68,
        "ORANGE": 50,
        "RED": 35,
        "CRITICAL": 15,
    }
    base = posture_scores.get(posture, 50)
    gov_conf = int(governor_doc.get("governor_confidence") or 0)
    adaptive_conf = int(adaptive_doc.get("confidence") or 0)
    escalation = str(cpe_doc.get("escalation_state") or "GREEN").upper()
    esc_penalty = ESCALATION_SEVERITY.get(escalation, 0) * 6
    raw = base * 0.45 + gov_conf * 0.35 + adaptive_conf * 0.2 - esc_penalty
    return int(max(0, min(100, round(raw))))


def _oversight_maturity(
    committee_doc: Dict[str, Any],
    board_doc: Dict[str, Any],
    accountability_doc: Dict[str, Any],
) -> int:
    avg_domain = _safe_float(committee_doc.get("average_domain_score"), 50.0) or 50.0
    gov_conf = int(board_doc.get("governance_confidence") or 0)
    entries = accountability_doc.get("entry_count", 0)
    trace_bonus = min(12, entries // 2)
    board_status = str(board_doc.get("board_status") or "ACTIVE").upper()
    status_penalty = {"SUSPENDED": 25, "REVIEW": 12, "STANDBY": 6}.get(board_status, 0)
    raw = avg_domain * 0.4 + gov_conf * 0.4 + trace_bonus - status_penalty
    return int(max(0, min(100, round(raw))))


def compute_triton_maturity_assessment(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    alerts_doc: Dict[str, Any],
    exec_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    adaptive_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    committee_doc: Dict[str, Any],
    board_doc: Dict[str, Any],
    accountability_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 29: institutional maturity scoring across six categories."""
    ts = _iso_utc()

    governance_score, _ = _governance_health_score(gov_doc, auth_doc)
    certification_score, _ = _certification_health_score(cert_doc)
    readiness_score, _ = _readiness_health_score(readiness_doc)

    category_scores = {
        "Monitoring": _monitoring_maturity(cpi_doc, alerts_doc, exec_doc),
        "Governance": governance_score,
        "Preservation": _preservation_maturity(governor_doc, cpe_doc, adaptive_doc),
        "Certification": certification_score,
        "Readiness": readiness_score,
        "Oversight": _oversight_maturity(committee_doc, board_doc, accountability_doc),
    }

    categories: List[Dict[str, Any]] = []
    for name in MATURITY_CATEGORIES:
        score = category_scores[name]
        categories.append(
            {
                "category": name,
                "score": score,
                "maturity_band": _maturity_band(score),
            }
        )

    overall = round(sum(c["score"] for c in categories) / len(categories), 1) if categories else 0.0
    strongest = max(categories, key=lambda c: c["score"])["category"]
    weakest = min(categories, key=lambda c: c["score"])["category"]

    return {
        "generated_at": ts,
        "overall_maturity": int(round(overall)),
        "maturity_band": _maturity_band(overall),
        "strongest_area": strongest,
        "weakest_area": weakest,
        "categories": categories,
        "disclaimer": "Maturity assessment is diagnostic only. No automated actions.",
    }


def _oversight_status(
    committee_recommendation: str,
    board_status: str,
) -> str:
    if committee_recommendation == "ESCALATE" or board_status == "SUSPENDED":
        return "SUSPENDED"
    if committee_recommendation == "REVIEW_REQUIRED" or board_status == "REVIEW":
        return "REVIEW"
    return "ACTIVE"


def _strategic_confidence(
    *,
    investment_doc: Dict[str, Any],
    maturity_doc: Dict[str, Any],
    board_doc: Dict[str, Any],
    exec_doc: Dict[str, Any],
) -> int:
    inv_conf = int(investment_doc.get("confidence") or 0)
    maturity = int(maturity_doc.get("overall_maturity") or 0)
    gov_conf = int(board_doc.get("governance_confidence") or 0)
    forecast = int((exec_doc.get("predictive_outlook") or {}).get("forecast_confidence") or 50)
    raw = inv_conf * 0.3 + maturity * 0.3 + gov_conf * 0.25 + forecast * 0.15
    return int(max(0, min(99, round(raw))))


def compute_strategic_oversight(
    *,
    governor_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    investment_doc: Dict[str, Any],
    maturity_doc: Dict[str, Any],
    exec_doc: Dict[str, Any],
    board_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 30: strategic oversight center aggregating institutional signals."""
    ts = _iso_utc()

    committee_rec = str(investment_doc.get("committee_recommendation") or "MONITOR")
    board_status = str(board_doc.get("board_status") or "ACTIVE")
    status = _oversight_status(committee_rec, board_status)
    confidence = _strategic_confidence(
        investment_doc=investment_doc,
        maturity_doc=maturity_doc,
        board_doc=board_doc,
        exec_doc=exec_doc,
    )
    institutional_readiness = maturity_doc.get("maturity_band", "DEVELOPING")

    strategic_concerns: List[str] = []
    for c in investment_doc.get("top_concerns") or []:
        if c and c not in strategic_concerns:
            strategic_concerns.append(str(c))
    for risk in exec_doc.get("top_risks") or []:
        if risk and risk not in strategic_concerns:
            strategic_concerns.append(str(risk))
    for req in cert_doc.get("failed_requirements") or []:
        label = f"Certification gap: {req}"
        if label not in strategic_concerns:
            strategic_concerns.append(label)

    recommendations: List[str] = []
    for rec in board_doc.get("board_recommendations") or []:
        if rec and rec not in recommendations:
            recommendations.append(str(rec))
    if committee_rec == "ESCALATE":
        recommendations.insert(0, "Escalate to senior strategic review (advisory only)")
    elif committee_rec == "REVIEW_REQUIRED":
        recommendations.insert(0, "Schedule investment committee review session")
    weakest = maturity_doc.get("weakest_area")
    if weakest:
        rec = f"Improve maturity in {weakest}"
        if rec not in recommendations:
            recommendations.append(rec)

    return {
        "generated_at": ts,
        "oversight_status": status,
        "strategic_confidence": confidence,
        "institutional_readiness": institutional_readiness,
        "automation_status": "NOT_AUTHORIZED",
        "strategic_readiness": institutional_readiness,
        "committee_recommendation": committee_rec,
        "preservation_posture": governor_doc.get("preservation_posture"),
        "certification_status": cert_doc.get("certification_status"),
        "board_status": board_status,
        "overall_maturity": maturity_doc.get("overall_maturity"),
        "maturity_band": maturity_doc.get("maturity_band"),
        "top_strategic_concerns": strategic_concerns[:6],
        "strategic_recommendations": recommendations[:8],
        "live_execution_permitted": False,
        "disclaimer": (
            "Strategic oversight is advisory only. "
            "automation_status defaults to NOT_AUTHORIZED. "
            "No automated trading or portfolio changes."
        ),
    }


def _append_maturity_history(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "overall_maturity",
        "maturity_band",
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


def _append_strategic_history(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "oversight_status",
        "strategic_confidence",
        "institutional_readiness",
        "automation_status",
        "committee_recommendation",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def persist_institutional_operations(
    *,
    results_dir: Path,
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
) -> Dict[str, Any]:
    """Run phases 28–30 and write JSON artifacts."""
    results_dir = Path(results_dir)

    investment_doc = compute_investment_committee_review(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        alerts_doc=alerts_doc,
        exec_doc=exec_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        cert_doc=cert_doc,
        readiness_doc=readiness_doc,
    )
    _atomic_write_json(investment_doc, results_dir / "investment_committee_review.json")

    maturity_doc = compute_triton_maturity_assessment(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        alerts_doc=alerts_doc,
        exec_doc=exec_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        governor_doc=governor_doc,
        adaptive_doc=adaptive_doc,
        cert_doc=cert_doc,
        readiness_doc=readiness_doc,
        committee_doc=committee_doc,
        board_doc=board_doc,
        accountability_doc=accountability_doc,
    )
    _atomic_write_json(maturity_doc, results_dir / "triton_maturity_assessment.json")

    strategic_doc = compute_strategic_oversight(
        governor_doc=governor_doc,
        cert_doc=cert_doc,
        investment_doc=investment_doc,
        maturity_doc=maturity_doc,
        exec_doc=exec_doc,
        board_doc=board_doc,
    )
    _atomic_write_json(strategic_doc, results_dir / "strategic_oversight.json")

    _append_maturity_history(
        results_dir / "triton_maturity_assessment_history.csv",
        {
            "timestamp": maturity_doc.get("generated_at"),
            "overall_maturity": maturity_doc.get("overall_maturity"),
            "maturity_band": maturity_doc.get("maturity_band"),
            "strongest_area": maturity_doc.get("strongest_area"),
            "weakest_area": maturity_doc.get("weakest_area"),
        },
    )

    _append_strategic_history(
        results_dir / "strategic_oversight_history.csv",
        {
            "timestamp": strategic_doc.get("generated_at"),
            "oversight_status": strategic_doc.get("oversight_status"),
            "strategic_confidence": strategic_doc.get("strategic_confidence"),
            "institutional_readiness": strategic_doc.get("institutional_readiness"),
            "automation_status": strategic_doc.get("automation_status"),
            "committee_recommendation": strategic_doc.get("committee_recommendation"),
        },
    )

    return {
        "investment_committee_review": investment_doc,
        "triton_maturity_assessment": maturity_doc,
        "strategic_oversight": strategic_doc,
    }
