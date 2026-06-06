"""
TRITON Strategic Intelligence — Phases 40–42.

Strategic reasoning, consequence forecasting, and institutional wisdom engines.
Paper-mode / simulation only. NO live trading, orders, or portfolio changes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from services.institutional_protection import _atomic_write_json, _iso_utc

MAX_HISTORY_ROWS = 5000
DEFAULT_FORECAST_HORIZON_DAYS = 90

WISDOM_HISTORY_FIELDNAMES = [
    "timestamp",
    "wisdom_statement",
    "confidence",
    "supporting_systems",
    "top_theme",
    "guidance_count",
]

SCOPE_RANK = {
    "LOCAL": 1,
    "DOMAIN": 2,
    "CROSS_DOMAIN": 3,
    "SYSTEM_WIDE": 4,
}

SEVERITY_RANK = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def _clamp_score(value: float, lo: int = 40, hi: int = 99) -> int:
    return int(min(hi, max(lo, round(value))))


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


def _issue_key(issue: str) -> str:
    return issue.strip().lower().replace("_", " ")


def _normalize_issue_label(label: str) -> str:
    text = label.strip()
    if not text:
        return "Unspecified Issue"
    replacements = {
        "readiness failure": "Execution Readiness",
        "not_ready": "Execution Readiness",
        "authorization gap": "Authorization Gap",
        "governance block": "Governance Block",
        "concentration risk": "Concentration Risk",
        "drawdown risk": "Drawdown Risk",
        "monitoring risk": "Monitoring Risk",
    }
    lower = text.lower()
    for needle, canonical in replacements.items():
        if needle in lower:
            return canonical
    if text.lower() == "readiness":
        return "Execution Readiness"
    if text.lower() == "governance":
        return "Governance Block"
    if text.lower() == "authorization":
        return "Authorization Gap"
    if text.lower() == "evaluation":
        return "Evaluation Gap"
    return text.title() if text.isupper() or "_" in text else text


def _scope_for_issue(issue: str, category: str) -> str:
    label = _issue_key(issue)
    if category == "bottleneck" or "readiness" in label or "certification" in label:
        return "SYSTEM_WIDE"
    if category == "governance" or "authorization" in label or "governance" in label:
        return "CROSS_DOMAIN"
    if "concentration" in label or "drawdown" in label:
        return "DOMAIN"
    if category == "weakness":
        return "CROSS_DOMAIN"
    return "DOMAIN"


def _score_issue(
    *,
    mentions: int,
    base: int,
    escalation_boost: int = 0,
    readiness_blocked: bool = False,
    cert_failed: bool = False,
) -> int:
    score = base + mentions * 6 + escalation_boost
    if readiness_blocked:
        score += 12
    if cert_failed:
        score += 8
    return _clamp_score(score)


def _collect_strategic_candidates(
    *,
    alerts_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpi_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    strategic_doc: Dict[str, Any],
    exec_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    learning_doc: Dict[str, Any],
    improvement_doc: Dict[str, Any],
    intelligence_doc: Dict[str, Any],
    maturity_doc: Dict[str, Any],
    investment_doc: Dict[str, Any],
    causal_doc: Dict[str, Any],
    insights_doc: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Aggregate strategic issue signals across institutional layers."""
    candidates: Dict[str, Dict[str, Any]] = {}

    def _bump(issue: str, category: str, base: int = 55) -> None:
        label = _normalize_issue_label(issue)
        key = _issue_key(label)
        entry = candidates.setdefault(
            key,
            {
                "issue": label,
                "category": category,
                "mentions": 0,
                "base": base,
            },
        )
        entry["mentions"] += 1
        if SCOPE_RANK.get(_scope_for_issue(label, category), 0) > SCOPE_RANK.get(
            _scope_for_issue(entry["issue"], entry["category"]), 0
        ):
            entry["category"] = category

    exec_summary = exec_doc.get("executive_summary") or exec_doc
    for risk in exec_summary.get("top_risks") or exec_doc.get("top_risks") or []:
        _bump(str(risk), "risk", 70)
    for driver in governor_doc.get("top_drivers") or []:
        _bump(str(driver), "risk", 68)
    for alert in alerts_doc.get("active_alerts") or []:
        alert_type = str(alert.get("alert_type") or "")
        if alert_type == "EXCESS_CONCENTRATION":
            _bump("Concentration Risk", "risk", 72)
        elif alert_type == "EXCESS_POSITION_DRAWDOWN":
            _bump("Drawdown Risk", "risk", 70)
        elif alert_type == "STALE_HEARTBEAT":
            _bump("Monitoring Risk", "risk", 65)
        elif alert_type:
            _bump(alert_type.replace("_", " ").title(), "risk", 60)

    escalation = str(cpe_doc.get("escalation_state") or "GREEN").upper()
    escalation_boost = {"RED": 15, "CRITICAL": 18, "ORANGE": 10, "YELLOW": 5}.get(escalation, 0)

    for concern in strategic_doc.get("top_strategic_concerns") or []:
        _bump(str(concern), "governance", 62)
    patterns = learning_doc.get("patterns") or {}
    for concern in patterns.get("most_common_governance_concerns") or []:
        _bump(str(concern), "governance", 58)
    for blocker in patterns.get("most_common_certification_blockers") or []:
        _bump(str(blocker), "governance", 60)

    if not auth_doc.get("overall_authorization"):
        _bump("Authorization Gap", "governance", 75)
    if str(governor_doc.get("governance_awareness_level") or "") in {
        "MANAGEMENT_REVIEW_REQUIRED",
        "BOARD_REVIEW_REQUIRED",
        "CRITICAL_INTERVENTION",
    }:
        _bump("Governance Block", "governance", 74)

    for failed in cert_doc.get("failed_requirements") or []:
        _bump(str(failed), "governance", 66)

    readiness_status = str(readiness_doc.get("readiness_status") or "")
    if readiness_status and readiness_status != "READY":
        _bump("Execution Readiness", "bottleneck", 78)
    for check in readiness_doc.get("failed_checks") or []:
        _bump(f"Readiness: {check}", "bottleneck", 64)

    for weak in improvement_doc.get("weakest_systems") or []:
        _bump(str(weak), "weakness", 63)
    if intelligence_doc.get("weakest_area"):
        _bump(str(intelligence_doc.get("weakest_area")), "weakness", 61)
    if maturity_doc.get("weakest_area"):
        _bump(str(maturity_doc.get("weakest_area")), "weakness", 59)
    if decision_weakest := (insights_doc.get("most_important_weakness")):
        _bump(str(decision_weakest), "weakness", 60)

    for concern in investment_doc.get("top_concerns") or []:
        _bump(str(concern), "risk", 64)

    for analysis in causal_doc.get("analyses") or []:
        issue = str(analysis.get("issue") or "")
        if issue:
            cat = (
                "bottleneck" if "readiness" in issue.lower() or "cert" in issue.lower() else "risk"
            )
            _bump(issue, cat, 67)

    if insights_doc.get("most_important_risk"):
        _bump(str(insights_doc.get("most_important_risk")), "risk", 71)
    if insights_doc.get("most_important_governance_concern"):
        _bump(str(insights_doc.get("most_important_governance_concern")), "governance", 65)

    weak_components = [
        name
        for name, score in (cpi_doc.get("component_scores") or {}).items()
        if isinstance(score, (int, float)) and score < 50
    ]
    for comp in weak_components:
        _bump(comp.replace("_", " ").title(), "weakness", 57)

    readiness_blocked = readiness_status != "READY"
    cert_failed = bool(cert_doc.get("failed_requirements")) or str(
        cert_doc.get("certification_status") or ""
    ).upper() in {"BLOCKED", "NOT_CERTIFIED", "PARTIALLY_CERTIFIED"}

    pred_boost = 0
    if str(pred_doc.get("risk_direction") or "").upper() == "DETERIORATING":
        pred_boost = 8

    issues: List[Dict[str, Any]] = []
    for entry in candidates.values():
        importance = _score_issue(
            mentions=entry["mentions"],
            base=entry["base"] + pred_boost,
            escalation_boost=escalation_boost,
            readiness_blocked=readiness_blocked and "readiness" in _issue_key(entry["issue"]),
            cert_failed=cert_failed
            and any(
                tok in _issue_key(entry["issue"])
                for tok in ("cert", "governance", "authorization", "evaluation")
            ),
        )
        category = entry["category"]
        scope = _scope_for_issue(entry["issue"], category)
        issues.append(
            {
                "issue": entry["issue"],
                "importance": importance,
                "scope": scope,
                "category": category,
            }
        )

    issues.sort(key=lambda x: (-x["importance"], -SCOPE_RANK.get(x["scope"], 0)))
    return issues


def compute_strategic_reasoning(
    *,
    alerts_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpi_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    strategic_doc: Dict[str, Any],
    exec_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    learning_doc: Dict[str, Any],
    improvement_doc: Dict[str, Any],
    intelligence_doc: Dict[str, Any],
    maturity_doc: Dict[str, Any],
    investment_doc: Dict[str, Any],
    causal_doc: Dict[str, Any],
    insights_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 40 — evaluate strategic implications across institutional layers."""
    ts = _iso_utc()
    strategic_issues = _collect_strategic_candidates(
        alerts_doc=alerts_doc,
        cpe_doc=cpe_doc,
        cpi_doc=cpi_doc,
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
        governor_doc=governor_doc,
        auth_doc=auth_doc,
        strategic_doc=strategic_doc,
        exec_doc=exec_doc,
        pred_doc=pred_doc,
        learning_doc=learning_doc,
        improvement_doc=improvement_doc,
        intelligence_doc=intelligence_doc,
        maturity_doc=maturity_doc,
        investment_doc=investment_doc,
        causal_doc=causal_doc,
        insights_doc=insights_doc,
    )

    if not strategic_issues:
        strategic_issues = [
            {
                "issue": "Preservation Monitoring",
                "importance": 55,
                "scope": "DOMAIN",
                "category": "risk",
            }
        ]

    top = strategic_issues[0]
    return {
        "generated_at": ts,
        "top_strategic_issue": top["issue"],
        "strategic_importance": top["importance"],
        "impact_scope": top["scope"],
        "strategic_issues": strategic_issues[:12],
        "issue_count": len(strategic_issues),
        "disclaimer": (
            "Strategic reasoning is advisory analysis only. "
            "No trades, orders, or portfolio modifications."
        ),
    }


def _forecast_for_issue(
    issue: str,
    *,
    horizon_days: int,
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    learning_doc: Dict[str, Any],
    causal_doc: Dict[str, Any],
) -> Dict[str, Any]:
    label = _issue_key(issue)
    cert_status = str(cert_doc.get("certification_status") or "").upper()
    readiness_status = str(readiness_doc.get("readiness_status") or "")
    posture = str(governor_doc.get("preservation_posture") or "").upper()
    risk_dir = str(pred_doc.get("risk_direction") or "").upper()
    forecast_conf = int(pred_doc.get("forecast_confidence") or 70)

    consequence = "Conditions likely persist with limited institutional progress"
    severity = "MEDIUM"
    confidence = _clamp_score(forecast_conf * 0.85, 45, 95)

    if "readiness" in label or issue == "Execution Readiness":
        consequence = "Certification remains blocked; automation gates stay closed"
        severity = "HIGH"
        confidence = _clamp_score(confidence + 10)
        if readiness_status == "NOT_READY":
            confidence = _clamp_score(confidence + 5)
    elif "concentration" in label:
        consequence = "Portfolio concentration risk remains elevated"
        severity = "HIGH" if posture == "RED" else "MEDIUM"
        if risk_dir == "DETERIORATING":
            consequence = "Concentration stress may intensify as risk momentum deteriorates"
            severity = "HIGH"
    elif "authorization" in label:
        consequence = "Governance authorization gap persists; execution remains prohibited"
        severity = "HIGH"
        confidence = _clamp_score(confidence + 8)
    elif "governance" in label:
        consequence = "Governance review requirements remain unresolved"
        severity = "HIGH" if posture in {"RED", "CRITICAL"} else "MEDIUM"
    elif "cert" in label or issue in {"Evaluation Gap", "Governance Block"}:
        consequence = "Certification gaps persist across governance and readiness domains"
        severity = "HIGH" if cert_status in {"BLOCKED", "PARTIALLY_CERTIFIED"} else "MEDIUM"
    elif "drawdown" in label:
        consequence = "Drawdown exposure may deepen without defensive review"
        severity = "HIGH" if posture == "RED" else "MEDIUM"
    elif "monitoring" in label:
        consequence = "Monitoring blind spots may delay escalation detection"
        severity = "MEDIUM"
    elif "evaluation" in label:
        consequence = (
            "Protective action evaluation confidence remains insufficient for certification"
        )
        severity = "MEDIUM"

    patterns = learning_doc.get("patterns") or {}
    repeated = patterns.get("repeated_failures") or []
    if any(label.split()[0] in str(r).lower() for r in repeated):
        confidence = _clamp_score(confidence + 6)
        if severity == "MEDIUM":
            severity = "HIGH"

    for analysis in causal_doc.get("analyses") or []:
        if _issue_key(str(analysis.get("issue") or "")) in label or label in _issue_key(
            str(analysis.get("issue") or "")
        ):
            confidence = _clamp_score(
                max(confidence, int(analysis.get("confidence") or confidence))
            )
            break

    projected_esc = (
        (pred_doc.get("escalation_forecast") or {}).get("projected_escalation") or ""
    ).upper()
    if projected_esc in {"RED", "CRITICAL"} and severity != "CRITICAL":
        severity = "HIGH"

    days_to_threshold = pred_doc.get("estimated_days_to_threshold")
    if isinstance(days_to_threshold, (int, float)) and days_to_threshold <= horizon_days:
        if severity == "MEDIUM":
            severity = "HIGH"

    return {
        "issue": issue,
        "forecast_horizon_days": horizon_days,
        "likely_consequence": consequence,
        "confidence": confidence,
        "severity": severity,
    }


def compute_consequence_forecasts(
    *,
    strategic_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    learning_doc: Dict[str, Any],
    causal_doc: Dict[str, Any],
    strategic_reasoning_doc: Dict[str, Any],
    forecast_horizon_days: int = DEFAULT_FORECAST_HORIZON_DAYS,
) -> Dict[str, Any]:
    """Phase 41 — forecast institutional consequences if current conditions persist."""
    ts = _iso_utc()

    seed_issues = [
        i.get("issue")
        for i in (strategic_reasoning_doc.get("strategic_issues") or [])[:6]
        if i.get("issue")
    ]
    if not seed_issues:
        seed_issues = list(strategic_doc.get("top_strategic_concerns") or [])[:4]
    if not seed_issues:
        seed_issues = ["Execution Readiness", "Authorization Gap", "Concentration Risk"]

    forecasts: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for issue in seed_issues:
        key = _issue_key(str(issue))
        if key in seen:
            continue
        seen.add(key)
        forecasts.append(
            _forecast_for_issue(
                str(issue),
                horizon_days=forecast_horizon_days,
                readiness_doc=readiness_doc,
                cert_doc=cert_doc,
                governor_doc=governor_doc,
                pred_doc=pred_doc,
                learning_doc=learning_doc,
                causal_doc=causal_doc,
            )
        )

    forecasts.sort(
        key=lambda f: (
            -SEVERITY_RANK.get(str(f.get("severity")), 0),
            -int(f.get("confidence") or 0),
        )
    )

    return {
        "generated_at": ts,
        "forecast_horizon_days": forecast_horizon_days,
        "forecast_count": len(forecasts),
        "forecasts": forecasts,
        "disclaimer": (
            "Consequence forecasts are simulation-only projections. "
            "No automated intervention or trading actions."
        ),
    }


def compute_institutional_wisdom(
    *,
    results_dir: Path,
    learning_doc: Dict[str, Any],
    memory_doc: Dict[str, Any],
    graph_doc: Dict[str, Any],
    decision_doc: Dict[str, Any],
    insights_doc: Dict[str, Any],
    strategic_reasoning_doc: Dict[str, Any],
    consequence_doc: Dict[str, Any],
    improvement_doc: Dict[str, Any],
    strategic_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 42 — synthesize long-term institutional guidance."""
    ts = _iso_utc()
    results_dir = Path(results_dir)

    top_issue = strategic_reasoning_doc.get("top_strategic_issue") or "Execution Readiness"
    top_importance = int(strategic_reasoning_doc.get("strategic_importance") or 70)
    top_insight = insights_doc.get("top_insight") or ""
    top_lesson = learning_doc.get("top_lesson") or ""
    high_severity = [
        f for f in (consequence_doc.get("forecasts") or []) if f.get("severity") == "HIGH"
    ]

    if "readiness" in _issue_key(str(top_issue)).lower():
        wisdom_statement = (
            "Execution readiness repeatedly emerges as the dominant constraint across "
            "certification, authorization, and governance layers. Institutional progress "
            "requires resolving readiness gates before automation or certification can advance."
        )
    elif high_severity:
        lead = high_severity[0]
        wisdom_statement = (
            f"{lead.get('issue')} is forecast to produce sustained institutional friction: "
            f"{lead.get('likely_consequence')}. Long-term stability depends on addressing "
            "root governance and preservation bottlenecks before risk momentum accelerates."
        )
    elif top_insight:
        wisdom_statement = top_insight
    elif top_lesson:
        wisdom_statement = (
            f"Organizational memory confirms a recurring pattern: {top_lesson}. "
            "Institutional wisdom favors resolving this gap before expanding automation scope."
        )
    else:
        wisdom_statement = (
            "Institutional signals remain mixed; prioritize governance alignment and "
            "preservation consistency before strategic expansion."
        )

    themes: List[Dict[str, Any]] = []
    theme_sources = [
        ("strategic_reasoning", top_issue, top_importance),
        (
            "organizational_learning",
            learning_doc.get("top_priority") or learning_doc.get("top_lesson") or "Governance",
            int(learning_doc.get("confidence") or 50),
        ),
        (
            "decision_quality",
            decision_doc.get("weakest_area") or "Governance Consistency",
            int(decision_doc.get("decision_quality_score") or 60),
        ),
        (
            "institutional_insights",
            insights_doc.get("most_important_governance_concern") or "Governance alignment",
            int(insights_doc.get("insight_confidence") or 55),
        ),
        (
            "knowledge_graph",
            graph_doc.get("most_connected_area") or "Preservation",
            min(99, int(graph_doc.get("relationships") or 0) * 3 + 40),
        ),
    ]
    for source, theme, weight in theme_sources:
        if theme:
            themes.append(
                {
                    "theme": str(theme),
                    "source": source,
                    "weight": _clamp_score(weight, 40, 99),
                }
            )

    guidance_items: List[Dict[str, Any]] = []
    priority = 1

    for forecast in (consequence_doc.get("forecasts") or [])[:4]:
        guidance_items.append(
            {
                "priority": priority,
                "guidance": (
                    f"Address {forecast.get('issue')}: {forecast.get('likely_consequence')} "
                    f"(severity={forecast.get('severity')}, confidence={forecast.get('confidence')}%)"
                ),
                "horizon_days": forecast.get("forecast_horizon_days"),
                "category": "consequence_mitigation",
            }
        )
        priority += 1

    lessons = (learning_doc.get("patterns") or {}).get("highest_value_lessons") or []
    for lesson in lessons[:3]:
        guidance_items.append(
            {
                "priority": priority,
                "guidance": str(lesson),
                "horizon_days": None,
                "category": "organizational_learning",
            }
        )
        priority += 1

    for enhancement in (improvement_doc.get("recommended_focus") or [])[:2]:
        guidance_items.append(
            {
                "priority": priority,
                "guidance": f"Strategic focus: {enhancement}",
                "horizon_days": None,
                "category": "self_improvement",
            }
        )
        priority += 1

    if strategic_doc.get("top_strategic_concerns"):
        guidance_items.append(
            {
                "priority": priority,
                "guidance": (
                    "Reconcile strategic oversight concerns: "
                    + "; ".join(strategic_doc.get("top_strategic_concerns")[:3])
                ),
                "horizon_days": 90,
                "category": "strategic_oversight",
            }
        )

    supporting_systems = sum(
        1
        for doc in (
            learning_doc,
            memory_doc,
            graph_doc,
            decision_doc,
            insights_doc,
            strategic_reasoning_doc,
            consequence_doc,
            improvement_doc,
        )
        if doc
    )

    depth_signals = [
        int(learning_doc.get("confidence") or 0),
        int(decision_doc.get("decision_quality_score") or 0),
        int(insights_doc.get("insight_confidence") or 0),
        top_importance,
        len(high_severity) * 10,
        min(99, int(memory_doc.get("memory_entries") or 0) * 2),
    ]
    confidence = _clamp_score(sum(depth_signals) / max(1, len(depth_signals)), 50, 99)

    doc = {
        "generated_at": ts,
        "wisdom_statement": wisdom_statement,
        "confidence": confidence,
        "supporting_systems": supporting_systems,
        "wisdom_themes": themes[:8],
        "guidance_items": guidance_items[:10],
        "disclaimer": (
            "Institutional wisdom is long-term advisory guidance only. "
            "No trades, orders, or portfolio modifications."
        ),
    }

    history_path = results_dir / "institutional_wisdom_history.csv"
    _append_wisdom_history(
        history_path,
        {
            "timestamp": ts,
            "wisdom_statement": wisdom_statement[:500],
            "confidence": confidence,
            "supporting_systems": supporting_systems,
            "top_theme": themes[0]["theme"] if themes else top_issue,
            "guidance_count": len(guidance_items),
        },
    )

    return doc


def _append_wisdom_history(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WISDOM_HISTORY_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in WISDOM_HISTORY_FIELDNAMES})


def persist_strategic_intelligence(
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
    exec_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    learning_doc: Dict[str, Any],
    memory_doc: Dict[str, Any],
    graph_doc: Dict[str, Any],
    improvement_doc: Dict[str, Any],
    intelligence_doc: Dict[str, Any],
    maturity_doc: Dict[str, Any],
    investment_doc: Dict[str, Any],
    decision_doc: Dict[str, Any],
    causal_doc: Dict[str, Any],
    insights_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Run phases 40–42 and write JSON artifacts."""
    results_dir = Path(results_dir)

    strategic_reasoning_doc = compute_strategic_reasoning(
        alerts_doc=alerts_doc,
        cpe_doc=cpe_doc,
        cpi_doc=cpi_doc,
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
        governor_doc=governor_doc,
        auth_doc=auth_doc,
        strategic_doc=strategic_doc,
        exec_doc=exec_doc,
        pred_doc=pred_doc,
        learning_doc=learning_doc,
        improvement_doc=improvement_doc,
        intelligence_doc=intelligence_doc,
        maturity_doc=maturity_doc,
        investment_doc=investment_doc,
        causal_doc=causal_doc,
        insights_doc=insights_doc,
    )
    _atomic_write_json(strategic_reasoning_doc, results_dir / "strategic_reasoning.json")

    consequence_doc = compute_consequence_forecasts(
        strategic_doc=strategic_doc,
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
        governor_doc=governor_doc,
        pred_doc=pred_doc,
        learning_doc=learning_doc,
        causal_doc=causal_doc,
        strategic_reasoning_doc=strategic_reasoning_doc,
    )
    _atomic_write_json(consequence_doc, results_dir / "consequence_forecasts.json")

    wisdom_doc = compute_institutional_wisdom(
        results_dir=results_dir,
        learning_doc=learning_doc,
        memory_doc=memory_doc,
        graph_doc=graph_doc,
        decision_doc=decision_doc,
        insights_doc=insights_doc,
        strategic_reasoning_doc=strategic_reasoning_doc,
        consequence_doc=consequence_doc,
        improvement_doc=improvement_doc,
        strategic_doc=strategic_doc,
    )
    _atomic_write_json(wisdom_doc, results_dir / "institutional_wisdom.json")

    return {
        "strategic_reasoning": strategic_reasoning_doc,
        "consequence_forecasts": consequence_doc,
        "institutional_wisdom": wisdom_doc,
    }
