"""
TRITON Institutional Planning — Phases 43–45.

Scenario planning, future path analysis, and strategic priorities engines.
Paper-mode / simulation only. NO live trading, orders, or portfolio changes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from services.institutional_protection import _atomic_write_json, _iso_utc

DEFAULT_HORIZON_DAYS = 90

PRIORITIES_HISTORY_FIELDNAMES = [
    "timestamp",
    "top_priority",
    "priority_count",
    "top_impact",
    "top_category",
]

SCENARIO_TYPES = (
    "BEST_CASE",
    "BASE_CASE",
    "WORST_CASE",
    "GOVERNANCE_FAILURE",
    "CERTIFICATION_SUCCESS",
    "READINESS_IMPROVEMENT",
)

PATH_TYPES = (
    "CURRENT_PATH",
    "ACCELERATED_IMPROVEMENT",
    "STALLED_READINESS",
    "GOVERNANCE_IMPROVEMENT",
)


def _clamp_score(value: float, lo: int = 0, hi: int = 100) -> int:
    return int(min(hi, max(lo, round(value))))


def _readiness_score(readiness_doc: Dict[str, Any]) -> int:
    status = str(readiness_doc.get("readiness_status") or "NOT_READY").upper()
    passing = int(readiness_doc.get("checks_passing_count") or 0)
    total = max(1, int(readiness_doc.get("checks_total") or 8))
    ratio = passing / total * 100
    if status == "READY":
        return _clamp_score(max(ratio, 85))
    if status == "PARTIALLY_READY":
        return _clamp_score(ratio * 0.85 + 10)
    return _clamp_score(ratio * 0.6)


def _cert_score(cert_doc: Dict[str, Any]) -> int:
    return _clamp_score(float(cert_doc.get("certification_score") or 0))


def _cpi_score(cpi_doc: Dict[str, Any]) -> int:
    return _clamp_score(float(cpi_doc.get("capital_preservation_score") or 50))


def _escalation_stress(cpe_doc: Dict[str, Any]) -> int:
    state = str(cpe_doc.get("escalation_state") or "GREEN").upper()
    ranks = {"GREEN": 10, "YELLOW": 35, "ORANGE": 60, "RED": 85, "CRITICAL": 95}
    return ranks.get(state, 40)


def _cert_gap(cert: int) -> int:
    return max(0, 100 - cert)


def _escalation_stress_from_posture(posture: str) -> int:
    return {"GREEN": 15, "YELLOW": 35, "ORANGE": 55, "RED": 80, "CRITICAL": 92}.get(posture, 45)


def _escalation_confidence(posture: str, pred_doc: Dict[str, Any]) -> float:
    base = int(pred_doc.get("forecast_confidence") or 70)
    boost = 12 if posture in {"RED", "CRITICAL"} else 6 if posture == "ORANGE" else 0
    return base + boost


def _normalize_probabilities(weights: Dict[str, float]) -> Dict[str, int]:
    total = sum(max(0.0, w) for w in weights.values()) or 1.0
    raw = {k: max(0.0, w) / total * 100 for k, w in weights.items()}
    rounded = {k: int(round(v)) for k, v in raw.items()}
    drift = 100 - sum(rounded.values())
    if drift and rounded:
        top_key = max(rounded, key=rounded.get)
        rounded[top_key] = max(0, rounded[top_key] + drift)
    return rounded


def compute_scenario_planning(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    strategic_reasoning_doc: Dict[str, Any],
    consequence_doc: Dict[str, Any],
    insights_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 43 — six institutional scenarios with weighted probabilities."""
    ts = _iso_utc()
    cpi = _cpi_score(cpi_doc)
    readiness = _readiness_score(readiness_doc)
    cert = _cert_score(cert_doc)
    escalation = _escalation_stress(cpe_doc)
    posture = str(governor_doc.get("preservation_posture") or "").upper()
    risk_dir = str(pred_doc.get("risk_direction") or "STABLE").upper()
    forecast_conf = int(pred_doc.get("forecast_confidence") or 70)
    top_issue = str(strategic_reasoning_doc.get("top_strategic_issue") or "Execution Readiness")
    cert_status = str(cert_doc.get("certification_status") or "").upper()
    readiness_status = str(readiness_doc.get("readiness_status") or "NOT_READY").upper()

    high_severity = sum(
        1 for f in (consequence_doc.get("forecasts") or []) if f.get("severity") == "HIGH"
    )
    gov_concern = bool(insights_doc.get("most_important_governance_concern"))

    improvement_headroom = max(0, 100 - readiness)
    cert_gap = max(0, 100 - cert)
    stress_factor = escalation / 100.0

    weights = {
        "BEST_CASE": max(
            5,
            (cpi * 0.25 + readiness * 0.35 + cert * 0.25 + (100 - escalation) * 0.15)
            * (1.1 if risk_dir == "IMPROVING" else 0.9 if risk_dir == "DETERIORATING" else 1.0),
        ),
        "BASE_CASE": 28 + forecast_conf * 0.08,
        "WORST_CASE": max(
            8,
            escalation * 0.45 + high_severity * 6 + (30 if risk_dir == "DETERIORATING" else 10),
        ),
        "GOVERNANCE_FAILURE": max(
            6,
            (45 if gov_concern else 20)
            + (25 if posture in {"RED", "CRITICAL"} else 10)
            + (15 if "governance" in top_issue.lower() else 5),
        ),
        "CERTIFICATION_SUCCESS": max(
            5,
            cert * 0.55 + (20 if cert_status == "PARTIALLY_CERTIFIED" else 8),
        ),
        "READINESS_IMPROVEMENT": max(
            5,
            improvement_headroom * 0.5 + (18 if readiness_status == "PARTIALLY_READY" else 6),
        ),
    }
    probs = _normalize_probabilities(weights)

    scenarios: List[Dict[str, Any]] = [
        {
            "scenario": "BEST_CASE",
            "probability": probs["BEST_CASE"],
            "primary_outcome": "Certification advances; readiness gates converge toward READY",
            "assumptions": [
                f"CPI stabilizes above {max(55, cpi)} with {cpi_doc.get('health_band', 'stable')} band",
                "Escalation eases from current posture without governance rupture",
                "Strategic focus on remediation yields measurable readiness gains",
            ],
            "time_horizon_days": DEFAULT_HORIZON_DAYS,
        },
        {
            "scenario": "BASE_CASE",
            "probability": probs["BASE_CASE"],
            "primary_outcome": "Incremental institutional progress; blockers persist in pockets",
            "assumptions": [
                f"Top strategic issue ({top_issue}) remains primary but managed",
                f"Consequence forecasts ({consequence_doc.get('forecast_count', 0)}) stay advisory",
                f"Predictive confidence ({forecast_conf}%) supports steady monitoring",
            ],
            "time_horizon_days": DEFAULT_HORIZON_DAYS,
        },
        {
            "scenario": "WORST_CASE",
            "probability": probs["WORST_CASE"],
            "primary_outcome": "Escalation deepens; certification and readiness stall",
            "assumptions": [
                f"Escalation state remains {cpe_doc.get('escalation_state', 'elevated')}",
                f"{high_severity} high-severity consequence forecasts materialize",
                "Risk direction deteriorates without compensating governance relief",
            ],
            "time_horizon_days": DEFAULT_HORIZON_DAYS,
        },
        {
            "scenario": "GOVERNANCE_FAILURE",
            "probability": probs["GOVERNANCE_FAILURE"],
            "primary_outcome": "Governance review backlog blocks authorization and oversight",
            "assumptions": [
                insights_doc.get("most_important_governance_concern")
                or "Governance alignment remains unresolved",
                f"Governor posture {posture or 'elevated'} sustains review requirements",
                "Authorization and policy layers remain misaligned",
            ],
            "time_horizon_days": DEFAULT_HORIZON_DAYS,
        },
        {
            "scenario": "CERTIFICATION_SUCCESS",
            "probability": probs["CERTIFICATION_SUCCESS"],
            "primary_outcome": "Certification score crosses threshold; uncertified areas close",
            "assumptions": [
                f"Certification score improves from {cert} toward full certification",
                f"Status progresses beyond {cert_status or 'PARTIALLY_CERTIFIED'}",
                "Simulation and monitoring areas remain certified anchors",
            ],
            "time_horizon_days": DEFAULT_HORIZON_DAYS,
        },
        {
            "scenario": "READINESS_IMPROVEMENT",
            "probability": probs["READINESS_IMPROVEMENT"],
            "primary_outcome": "Failed readiness checks clear; live execution remains prohibited",
            "assumptions": [
                f"Readiness moves from {readiness_status} with {readiness_doc.get('checks_passing_count', 0)}/"
                f"{readiness_doc.get('checks_total', 8)} checks passing",
                "Failed checks (governance, policy) addressed in paper mode only",
                "Paper-mode disclaimer preserved — no automated execution",
            ],
            "time_horizon_days": DEFAULT_HORIZON_DAYS,
        },
    ]

    return {
        "generated_at": ts,
        "scenario_count": len(scenarios),
        "probability_sum": sum(s["probability"] for s in scenarios),
        "probability_model": (
            "Six scenario lenses normalized to sum 100. "
            "Not mutually exclusive event probabilities."
        ),
        "inputs": {
            "cpi_score": cpi,
            "readiness_score": readiness,
            "certification_score": cert,
            "escalation_stress": escalation,
            "risk_direction": risk_dir,
            "top_strategic_issue": top_issue,
        },
        "scenarios": scenarios,
        "disclaimer": (
            "Scenario planning is simulation-only advisory analysis. "
            "No trades, orders, or portfolio modifications."
        ),
    }


def compute_future_path_analysis(
    *,
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    maturity_doc: Dict[str, Any],
    improvement_doc: Dict[str, Any],
    strategic_reasoning_doc: Dict[str, Any],
    scenario_doc: Dict[str, Any],
    learning_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 44 — evaluate institutional trajectory paths."""
    ts = _iso_utc()
    readiness = _readiness_score(readiness_doc)
    cert = _cert_score(cert_doc)
    maturity = _clamp_score(float(maturity_doc.get("overall_maturity") or 50))
    improvement = _clamp_score(float(improvement_doc.get("improvement_score") or 60))
    forecast_conf = int(pred_doc.get("forecast_confidence") or 70)
    top_issue = str(strategic_reasoning_doc.get("top_strategic_issue") or "Execution Readiness")
    posture = str(governor_doc.get("preservation_posture") or "YELLOW").upper()
    learning_conf = int(learning_doc.get("confidence") or 55)

    best_prob = 0
    for s in scenario_doc.get("scenarios") or []:
        if s.get("scenario") == "BEST_CASE":
            best_prob = int(s.get("probability") or 0)
            break

    paths: List[Dict[str, Any]] = [
        {
            "path": "CURRENT_PATH",
            "expected_benefit": _clamp_score(
                readiness * 0.35 + cert * 0.25 + maturity * 0.2 + improvement * 0.2
            ),
            "confidence": _clamp_score(forecast_conf * 0.9, 50, 95),
            "description": (
                "Maintain current institutional cadence: monitoring, advisory governance, "
                "and paper-mode readiness without acceleration."
            ),
            "milestones": [
                "Preserve CPI and escalation visibility each watchdog cycle",
                f"Continue addressing {top_issue} as top strategic issue",
                "Sustain consequence forecast review rhythm (90-day horizon)",
            ],
            "risks": [
                "Readiness gaps may persist across governance and policy checks",
                "Certification partially complete — authorization remains gated",
            ],
        },
        {
            "path": "ACCELERATED_IMPROVEMENT",
            "expected_benefit": _clamp_score(
                improvement * 0.4 + best_prob * 0.35 + learning_conf * 0.25
            ),
            "confidence": _clamp_score((learning_conf + improvement + best_prob) / 3, 45, 92),
            "description": (
                "Prioritize highest-leverage enhancements from self-improvement and "
                "organizational learning to compress certification and readiness timelines."
            ),
            "milestones": [
                "Close top 3 improvement opportunities from strategic self-improvement",
                "Lift readiness checks from failed to passing (paper validation only)",
                "Raise certification score toward full certification threshold",
            ],
            "risks": [
                "Acceleration without governance alignment may increase rework",
                "Resource contention across weakest systems",
            ],
        },
        {
            "path": "STALLED_READINESS",
            "expected_benefit": _clamp_score(100 - readiness - _cert_gap(cert), 15, 55),
            "confidence": _clamp_score(_escalation_confidence(posture, pred_doc), 55, 90),
            "description": (
                "Readiness and certification improvement stall; institutional blockers "
                "remain static while monitoring continues."
            ),
            "milestones": [
                "Readiness status unchanged — execution gates remain closed",
                "Failed checks persist without remediation sprint",
                "Consequence forecasts accumulate high-severity entries",
            ],
            "risks": [
                "Escalation posture may harden if risk direction deteriorates",
                "Strategic importance of readiness remains at system-wide scope",
                "Organizational learning lessons fail to convert to readiness passes",
            ],
        },
        {
            "path": "GOVERNANCE_IMPROVEMENT",
            "expected_benefit": _clamp_score(
                maturity * 0.3
                + improvement * 0.35
                + (100 - _escalation_stress_from_posture(posture)),
            ),
            "confidence": _clamp_score(maturity * 0.5 + learning_conf * 0.5, 50, 88),
            "description": (
                "Governance-first trajectory: align awareness, authorization, and committee "
                "oversight before expanding automation scope."
            ),
            "milestones": [
                "Resolve governance awareness blocks cited in readiness failures",
                "Align authorization posture with board preservation authority",
                "Reduce governance-related strategic issues in reasoning rankings",
            ],
            "risks": [
                "Governance improvement may lag concentration and drawdown risks",
                "Policy layer re-enablement required after governance alignment",
            ],
        },
    ]

    paths.sort(key=lambda p: (-p["expected_benefit"], -p["confidence"]))

    return {
        "generated_at": ts,
        "path_count": len(paths),
        "recommended_path": paths[0]["path"] if paths else "CURRENT_PATH",
        "paths": paths,
        "disclaimer": (
            "Future path analysis is advisory trajectory modeling only. "
            "No execution, orders, or portfolio changes."
        ),
    }


def compute_strategic_priorities(
    *,
    strategic_reasoning_doc: Dict[str, Any],
    improvement_doc: Dict[str, Any],
    wisdom_doc: Dict[str, Any],
    scenario_doc: Dict[str, Any],
    future_paths_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    insights_doc: Dict[str, Any],
    learning_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 45 — rank institutional objectives and synthesis."""
    ts = _iso_utc()
    top_issue = str(strategic_reasoning_doc.get("top_strategic_issue") or "Execution Readiness")
    leverage = (improvement_doc.get("highest_leverage_enhancements") or [None])[0] or (
        improvement_doc.get("recommended_focus") or ["Governance"]
    )[0]
    opportunities = improvement_doc.get("improvement_opportunities") or []
    top_opportunity = str(opportunities[0]) if opportunities else "Certification gap closure"

    bottlenecks = [
        i
        for i in (strategic_reasoning_doc.get("strategic_issues") or [])
        if i.get("category") == "bottleneck"
    ]
    bottleneck_issue = bottlenecks[0]["issue"] if bottlenecks else top_issue

    recommended_path = future_paths_doc.get("recommended_path") or "CURRENT_PATH"
    best_scenario = max(
        scenario_doc.get("scenarios") or [],
        key=lambda s: s.get("probability") or 0,
        default={},
    )

    candidates: List[Dict[str, Any]] = [
        {
            "focus_area": top_issue,
            "expected_impact": _clamp_score(
                strategic_reasoning_doc.get("strategic_importance") or 80
            ),
            "category": "highest_priority_issue",
            "rationale": (
                f"Strategic reasoning ranks {top_issue} as top issue with "
                f"{strategic_reasoning_doc.get('impact_scope', 'SYSTEM_WIDE')} scope."
            ),
        },
        {
            "focus_area": str(leverage)[:120],
            "expected_impact": _clamp_score(improvement_doc.get("improvement_score") or 75),
            "category": "highest_leverage_improvement",
            "rationale": (
                "Highest-leverage enhancement from strategic self-improvement synthesis."
            ),
        },
        {
            "focus_area": str(bottleneck_issue),
            "expected_impact": _clamp_score((bottlenecks[0]["importance"] if bottlenecks else 85)),
            "category": "most_important_bottleneck",
            "rationale": (
                "Primary bottleneck constraining certification and readiness convergence."
            ),
        },
        {
            "focus_area": top_opportunity[:120],
            "expected_impact": _clamp_score(
                int(learning_doc.get("confidence") or 55) * 0.6
                + int(insights_doc.get("insight_confidence") or 55) * 0.4
            ),
            "category": "most_important_opportunity",
            "rationale": (
                "Top improvement opportunity with organizational learning and insights support."
            ),
        },
    ]

    wisdom_preview = (wisdom_doc.get("wisdom_statement") or "")[:80]
    for idx, item in enumerate(candidates, start=1):
        item["priority_rank"] = idx

    extra: List[Dict[str, Any]] = []
    for focus in (improvement_doc.get("recommended_focus") or [])[:2]:
        if any(c["focus_area"] == focus for c in candidates):
            continue
        extra.append(
            {
                "priority_rank": len(candidates) + len(extra) + 1,
                "focus_area": str(focus),
                "expected_impact": _clamp_score(
                    (improvement_doc.get("improvement_score") or 70) - len(extra) * 5
                ),
                "category": "recommended_focus",
                "rationale": f"Recommended focus area; path bias={recommended_path}.",
            }
        )

    priorities = candidates + extra[:2]
    priorities.sort(key=lambda p: (-p["expected_impact"], p["priority_rank"]))
    for idx, item in enumerate(priorities, start=1):
        item["priority_rank"] = idx

    return {
        "generated_at": ts,
        "highest_priority_issue": top_issue,
        "highest_leverage_improvement": str(leverage)[:120],
        "most_important_bottleneck": str(bottleneck_issue),
        "most_important_opportunity": top_opportunity[:120],
        "top_priority": priorities[0]["focus_area"] if priorities else top_issue,
        "recommended_path": recommended_path,
        "dominant_scenario": best_scenario.get("scenario"),
        "wisdom_alignment": wisdom_preview,
        "priorities": priorities,
        "disclaimer": (
            "Strategic priorities are advisory rankings only. "
            "No trades, orders, or automated intervention."
        ),
    }


def _append_priorities_history(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PRIORITIES_HISTORY_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PRIORITIES_HISTORY_FIELDNAMES})


def persist_institutional_planning(
    *,
    results_dir: Path,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    strategic_reasoning_doc: Dict[str, Any],
    consequence_doc: Dict[str, Any],
    insights_doc: Dict[str, Any],
    maturity_doc: Dict[str, Any],
    improvement_doc: Dict[str, Any],
    wisdom_doc: Dict[str, Any],
    learning_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Run phases 43–45 and write JSON artifacts."""
    results_dir = Path(results_dir)

    scenario_doc = compute_scenario_planning(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
        governor_doc=governor_doc,
        pred_doc=pred_doc,
        strategic_reasoning_doc=strategic_reasoning_doc,
        consequence_doc=consequence_doc,
        insights_doc=insights_doc,
    )
    _atomic_write_json(scenario_doc, results_dir / "scenario_planning.json")

    future_paths_doc = compute_future_path_analysis(
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
        governor_doc=governor_doc,
        pred_doc=pred_doc,
        maturity_doc=maturity_doc,
        improvement_doc=improvement_doc,
        strategic_reasoning_doc=strategic_reasoning_doc,
        scenario_doc=scenario_doc,
        learning_doc=learning_doc,
    )
    _atomic_write_json(future_paths_doc, results_dir / "future_path_analysis.json")

    priorities_doc = compute_strategic_priorities(
        strategic_reasoning_doc=strategic_reasoning_doc,
        improvement_doc=improvement_doc,
        wisdom_doc=wisdom_doc,
        scenario_doc=scenario_doc,
        future_paths_doc=future_paths_doc,
        readiness_doc=readiness_doc,
        insights_doc=insights_doc,
        learning_doc=learning_doc,
    )
    _atomic_write_json(priorities_doc, results_dir / "strategic_priorities.json")

    top = priorities_doc["priorities"][0] if priorities_doc.get("priorities") else {}
    _append_priorities_history(
        results_dir / "strategic_priorities_history.csv",
        {
            "timestamp": priorities_doc["generated_at"],
            "top_priority": priorities_doc.get("top_priority"),
            "priority_count": len(priorities_doc.get("priorities") or []),
            "top_impact": top.get("expected_impact"),
            "top_category": top.get("category"),
        },
    )

    return {
        "scenario_planning": scenario_doc,
        "future_path_analysis": future_paths_doc,
        "strategic_priorities": priorities_doc,
    }
