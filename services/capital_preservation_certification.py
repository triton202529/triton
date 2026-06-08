"""
TRITON Capital Preservation Certification — Phase 24 / 150C alignment.

Scores certification areas from existing artifacts. Paper-mode protection only.
NO live trading, orders, or portfolio changes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

CERTIFICATION_AREAS = (
    "Monitoring",
    "Alerting",
    "Governance",
    "Authorization",
    "Readiness",
    "Simulation",
    "Evaluation",
    "Governor",
)

GOVERNANCE_BLOCK_LEVELS = frozenset(
    {
        "MANAGEMENT_REVIEW_REQUIRED",
        "BOARD_REVIEW_REQUIRED",
        "CRITICAL_INTERVENTION",
    }
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: Optional[datetime] = None) -> str:
    t = dt or _utc_now()
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _area_result(certified: bool, score: int, notes: str) -> Dict[str, Any]:
    return {"certified": certified, "score": score, "notes": notes}


def _live_execution_permitted(
    auth_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
) -> bool:
    """True only when live execution is explicitly permitted (paper auth does not count)."""
    if auth_doc.get("live_execution_permitted") is True:
        return True
    if readiness_doc.get("live_execution_permitted") is True:
        return True
    return False


def _paper_execution_authorized(auth_doc: Dict[str, Any]) -> bool:
    if auth_doc.get("paper_execution_permitted") is True:
        return True
    gate_reasons = auth_doc.get("gate_reasons") or {}
    execution_reason = str(gate_reasons.get("execution") or "").lower()
    if "paper" in execution_reason and auth_doc.get("execution_authorized") is True:
        return True
    return bool(
        auth_doc.get("execution_authorized")
        and auth_doc.get("governance_authorized")
        and auth_doc.get("overall_authorization")
    )


def _log_alignment_audit(
    *,
    requirement: str,
    expected: str,
    actual: str,
    status: str,
    reason: str,
) -> None:
    logger.info(
        "[CERTIFICATION_ALIGNMENT_AUDIT] requirement=%s expected=%s actual=%s status=%s reason=%s",
        requirement,
        expected,
        actual,
        status,
        reason,
    )
    print(
        f"[CERTIFICATION_ALIGNMENT_AUDIT] requirement={requirement} "
        f"expected={expected} actual={actual} status={status} reason={reason}"
    )


def _audit_requirement(
    *,
    requirement: str,
    expected: str,
    passed: bool,
    actual: str,
    reason: str,
) -> Tuple[bool, Dict[str, Any]]:
    status = "PASS" if passed else "FAIL"
    _log_alignment_audit(
        requirement=requirement,
        expected=expected,
        actual=actual,
        status=status,
        reason=reason,
    )
    return passed, {
        "requirement": requirement,
        "expected": expected,
        "actual": actual,
        "status": status,
        "reason": reason,
    }


def _evaluation_certification(
    evaluation_doc: Dict[str, Any],
) -> Tuple[bool, int, str, Dict[str, Any]]:
    eval_count = int(evaluation_doc.get("evaluation_count") or 0)
    avg_eff = _safe_float(evaluation_doc.get("average_effectiveness"), 0.0) or 0.0
    eval_score = int(min(100, avg_eff)) if eval_count else 0

    baseline_context = str(evaluation_doc.get("baseline_context") or "")
    beneficial = int(evaluation_doc.get("beneficial_count") or 0)
    negative = int(evaluation_doc.get("negative_count") or 0)
    neutral = int(evaluation_doc.get("neutral_count") or 0)
    high_cps_adj = bool(evaluation_doc.get("high_cps_adjustment_applied"))
    scoring_model = str(evaluation_doc.get("scoring_model_version") or "")

    if eval_count < 2:
        notes = f"{eval_count} trials evaluated (minimum 2 required)"
        return (
            False,
            eval_score,
            notes,
            {
                "baseline_context": baseline_context,
                "threshold": "eval_count>=2",
            },
        )

    if baseline_context == "HIGH_CPS_PRESERVATION" and high_cps_adj:
        threshold = 40.0
        eval_cert = avg_eff >= threshold and beneficial >= 1 and negative <= 1
        threshold_label = "high_cps_baseline_aware>=40"
    elif baseline_context == "MODERATE_CPS_IMPROVEMENT":
        threshold = 45.0
        eval_cert = avg_eff >= threshold and beneficial >= 1
        threshold_label = "moderate_cps>=45"
    else:
        threshold = 50.0
        eval_cert = avg_eff >= threshold
        threshold_label = "standard>=50"

    top_reason = ""
    evaluations = evaluation_doc.get("evaluations") or []
    if evaluations and isinstance(evaluations[0], dict):
        top_reason = str(evaluations[0].get("evaluation_reason") or "")

    notes = (
        f"{eval_count} trials, avg={avg_eff}, baseline={baseline_context}, "
        f"beneficial={beneficial}, neutral={neutral}, negative={negative}, "
        f"model={scoring_model or 'unknown'}, threshold={threshold_label}"
    )
    if top_reason:
        notes += f", top_reason={top_reason[:80]}"

    return (
        eval_cert,
        eval_score,
        notes,
        {
            "baseline_context": baseline_context,
            "high_cps_adjustment_applied": high_cps_adj,
            "scoring_model_version": scoring_model,
            "beneficial_count": beneficial,
            "neutral_count": neutral,
            "negative_count": negative,
            "average_effectiveness": avg_eff,
            "threshold": threshold_label,
            "threshold_value": threshold,
        },
    )


def compute_preservation_certification(
    *,
    cpi_doc: Dict[str, Any],
    alerts_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    sim_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Score certification areas from existing artifacts with alignment audit logging."""
    ts = _iso_utc()
    areas: Dict[str, Dict[str, Any]] = {}
    failed: List[str] = []
    alignment_audits: List[Dict[str, Any]] = []

    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    monitoring_score = int(min(100, max(0, cps)))
    monitoring_cert = bool(cpi_doc.get("generated_at")) and cps >= 30
    areas["Monitoring"] = _area_result(
        monitoring_cert,
        monitoring_score,
        f"CPS={cps}, band={cpi_doc.get('health_band')}, trend={cpi_doc.get('risk_trend')}",
    )

    active_count = len(alerts_doc.get("active_alerts") or [])
    alert_score = 70 if alerts_doc.get("generated_at") else 20
    if active_count > 0:
        alert_score = min(100, alert_score + 10)
    alert_cert = bool(alerts_doc.get("generated_at"))
    areas["Alerting"] = _area_result(
        alert_cert,
        alert_score,
        f"{active_count} active alerts tracked",
    )

    gov_level = str(gov_doc.get("governance_awareness_level") or "").upper()
    gov_score = 80 if gov_level == "GREEN" else (60 if gov_level == "YELLOW" else 35)
    gov_cert = gov_level not in GOVERNANCE_BLOCK_LEVELS and bool(gov_doc.get("generated_at"))
    areas["Governance"] = _area_result(
        gov_cert,
        gov_score,
        f"awareness={gov_level}, status={gov_doc.get('governance_status')}",
    )

    paper_ok = _paper_execution_authorized(auth_doc)
    live_open = _live_execution_permitted(auth_doc, readiness_doc)
    auth_score = 90 if paper_ok and not live_open else (25 if live_open else 55)
    auth_cert = bool(paper_ok and auth_doc.get("governance_authorized") and not live_open)
    areas["Authorization"] = _area_result(
        auth_cert,
        auth_score,
        (
            f"paper={auth_doc.get('paper_execution_permitted')}, "
            f"live={auth_doc.get('live_execution_permitted')}, "
            f"execution={auth_doc.get('execution_authorized')}"
        ),
    )

    passing = readiness_doc.get("checks_passing_count", 0)
    total = readiness_doc.get("checks_total", 8) or 8
    readiness_score = int(round((passing / total) * 100)) if total else 0
    readiness_cert = (
        readiness_doc.get("readiness_status") == "READY"
        and not readiness_doc.get("live_execution_permitted")
        and readiness_score >= 75
    )
    areas["Readiness"] = _area_result(
        readiness_cert,
        readiness_score,
        f"status={readiness_doc.get('readiness_status')}, {passing}/{total} checks passing",
    )

    sim_count = len(sim_doc.get("simulations") or [])
    sim_score = min(100, 40 + sim_count * 8) if sim_doc.get("generated_at") else 0
    sim_cert = sim_count >= 3 and bool(sim_doc.get("generated_at"))
    areas["Simulation"] = _area_result(
        sim_cert,
        sim_score,
        f"{sim_count} defensive simulations on file",
    )

    eval_cert, eval_score, eval_notes, eval_context = _evaluation_certification(evaluation_doc)
    areas["Evaluation"] = _area_result(eval_cert, eval_score, eval_notes)

    gov_conf = int(governor_doc.get("governor_confidence") or 0)
    posture = str(governor_doc.get("preservation_posture") or "UNKNOWN")
    governor_score = gov_conf
    governor_cert = bool(governor_doc.get("generated_at")) and posture not in ("CRITICAL",)
    areas["Governor"] = _area_result(
        governor_cert,
        governor_score,
        f"posture={posture}, confidence={gov_conf}%",
    )

    live_actual = (
        f"auth.live={auth_doc.get('live_execution_permitted')}, "
        f"readiness.live={readiness_doc.get('live_execution_permitted')}, "
        f"paper={auth_doc.get('paper_execution_permitted')}"
    )
    live_passed, live_audit = _audit_requirement(
        requirement="Live execution gates must remain blocked",
        expected="live_execution_permitted=false (paper authorization allowed)",
        passed=not live_open,
        actual=live_actual,
        reason=(
            "Live gates blocked; paper execution authorization is distinct"
            if not live_open
            else "Live execution gate open — certification blocked"
        ),
    )
    alignment_audits.append(live_audit)
    if not live_passed:
        failed.append("Live execution gates must remain blocked")

    for name, area in areas.items():
        if not area["certified"]:
            failed.append(name)

    certification_score = (
        round(sum(a["score"] for a in areas.values()) / len(areas), 1) if areas else 0.0
    )
    certified_count = sum(1 for a in areas.values() if a["certified"])

    if live_open:
        certification_status = "NOT_CERTIFIED"
    elif certified_count == len(CERTIFICATION_AREAS) and certification_score >= 80:
        certification_status = "CERTIFIED_FOR_PAPER_PROTECTION"
    elif certified_count >= 4 and certification_score >= 55:
        certification_status = "PARTIALLY_CERTIFIED"
    else:
        certification_status = "NOT_CERTIFIED"

    return {
        "generated_at": ts,
        "certification_status": certification_status,
        "certification_score": certification_score,
        "certified_area_count": certified_count,
        "total_areas": len(CERTIFICATION_AREAS),
        "areas": areas,
        "failed_requirements": failed,
        "alignment_audits": alignment_audits,
        "evaluation_context": eval_context,
        "escalation_state": cpe_doc.get("escalation_state"),
        "live_execution_permitted": live_open,
        "paper_execution_permitted": auth_doc.get("paper_execution_permitted"),
        "disclaimer": "Certification assesses paper-mode protection only. No live execution.",
    }
