"""
TRITON Capital Preservation Evaluation — Phases 19–21.

Protective action evaluation, adaptive learning, and preservation governor.
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

TRIAL_TYPE_TO_PROTECTION = {
    "concentration": "CONCENTRATION_REDUCTION",
    "drawdown": "DRAWDOWN_REDUCTION",
    "exposure": "EXPOSURE_REDUCTION",
    "risk_off": "RISK_OFF_TRANSITION",
}

ESCALATION_SEVERITY = {
    "GREEN": 0,
    "YELLOW": 1,
    "ORANGE": 2,
    "RED": 3,
    "CRITICAL": 4,
}

EVALUATION_WEIGHTS = {
    "cps_improvement": 0.30,
    "risk_reduction": 0.30,
    "concentration_reduction": 0.15,
    "drawdown_reduction": 0.15,
    "stability_improvement": 0.10,
}

NORMALIZATION_CAPS = {
    "cps_improvement": 22.0,
    "risk_reduction": 22.0,
    "concentration_reduction": 90.0,
    "drawdown_reduction": 15.0,
    "stability_improvement": 20.0,
}


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


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _normalize(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return max(0.0, min(100.0, (value / cap) * 100.0))


def _evaluation_state(score: float) -> str:
    if score >= 65:
        return "BENEFICIAL"
    if score >= 40:
        return "NEUTRAL"
    return "NEGATIVE"


def _stability_improvement(
    trial: Dict[str, Any],
    cpi_doc: Dict[str, Any],
) -> float:
    """Derive stability uplift from trial metrics and CPI risk trend."""
    cps_delta = _safe_float(trial.get("estimated_cps_improvement"), 0.0) or 0.0
    risk_delta = _safe_float(trial.get("estimated_risk_reduction"), 0.0) or 0.0
    trend = str(cpi_doc.get("risk_trend") or "STABLE").upper()
    trend_bonus = {"IMPROVING": 2.0, "STABLE": 0.0, "DETERIORATING": 4.0}.get(trend, 0.0)
    return round(min(20.0, cps_delta * 0.35 + risk_delta * 0.25 + trend_bonus), 2)


def _evaluate_single_trial(
    trial: Dict[str, Any],
    cpi_doc: Dict[str, Any],
) -> Dict[str, Any]:
    cps_improvement = round(_safe_float(trial.get("estimated_cps_improvement"), 0.0) or 0.0, 1)
    risk_reduction = round(_safe_float(trial.get("estimated_risk_reduction"), 0.0) or 0.0, 1)
    concentration_reduction = round(
        _safe_float(trial.get("estimated_concentration_reduction"), 0.0) or 0.0, 2
    )
    drawdown_reduction = round(
        _safe_float(trial.get("estimated_drawdown_improvement"), 0.0) or 0.0, 2
    )
    stability_improvement = _stability_improvement(trial, cpi_doc)

    components = {
        "cps_improvement": _normalize(cps_improvement, NORMALIZATION_CAPS["cps_improvement"]),
        "risk_reduction": _normalize(risk_reduction, NORMALIZATION_CAPS["risk_reduction"]),
        "concentration_reduction": _normalize(
            concentration_reduction, NORMALIZATION_CAPS["concentration_reduction"]
        ),
        "drawdown_reduction": _normalize(
            drawdown_reduction, NORMALIZATION_CAPS["drawdown_reduction"]
        ),
        "stability_improvement": _normalize(
            stability_improvement, NORMALIZATION_CAPS["stability_improvement"]
        ),
    }

    effectiveness_score = round(
        sum(components[k] * EVALUATION_WEIGHTS[k] for k in EVALUATION_WEIGHTS), 0
    )
    effectiveness_score = int(max(0, min(100, effectiveness_score)))

    return {
        "trial_id": trial.get("trial_id"),
        "trial_name": trial.get("trial_name"),
        "trial_type": trial.get("trial_type"),
        "effectiveness_score": effectiveness_score,
        "cps_improvement": cps_improvement,
        "risk_reduction": risk_reduction,
        "concentration_reduction": concentration_reduction,
        "drawdown_reduction": drawdown_reduction,
        "stability_improvement": stability_improvement,
        "evaluation": _evaluation_state(effectiveness_score),
        "status": trial.get("status", "SIMULATION_ONLY"),
        "execution_permitted": False,
        "mode": trial.get("mode", "PAPER"),
    }


def compute_protective_action_evaluation(
    *,
    trials_doc: Dict[str, Any],
    cpi_doc: Dict[str, Any],
    cpe_doc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 19: evaluate paper-mode protective action trials."""
    ts = _iso_utc()
    cpe_doc = cpe_doc or {}
    trials_in = trials_doc.get("trials") or []

    evaluations: List[Dict[str, Any]] = []
    for trial in trials_in:
        if not isinstance(trial, dict):
            continue
        evaluations.append(_evaluate_single_trial(trial, cpi_doc))

    evaluations.sort(key=lambda e: e.get("effectiveness_score", 0), reverse=True)

    beneficial = sum(1 for e in evaluations if e.get("evaluation") == "BENEFICIAL")
    neutral = sum(1 for e in evaluations if e.get("evaluation") == "NEUTRAL")
    negative = sum(1 for e in evaluations if e.get("evaluation") == "NEGATIVE")
    avg_score = (
        round(sum(e.get("effectiveness_score", 0) for e in evaluations) / len(evaluations), 1)
        if evaluations
        else 0.0
    )

    return {
        "generated_at": ts,
        "baseline_cps": trials_doc.get("baseline_cps"),
        "escalation_state": cpe_doc.get("escalation_state"),
        "evaluation_count": len(evaluations),
        "average_effectiveness": avg_score,
        "beneficial_count": beneficial,
        "neutral_count": neutral,
        "negative_count": negative,
        "evaluations": evaluations,
        "top_trial": evaluations[0] if evaluations else None,
        "disclaimer": "Evaluation only. No trades, orders, or portfolio modifications.",
    }


def _adaptive_confidence(evaluations: List[Dict[str, Any]]) -> int:
    if not evaluations:
        return 0
    n = len(evaluations)
    scores = [e.get("effectiveness_score", 0) for e in evaluations]
    spread = max(scores) - min(scores) if len(scores) > 1 else 0
    base = min(95, 55 + n * 8)
    spread_penalty = max(0, 15 - spread // 3)
    return int(max(40, min(99, base - spread_penalty + (10 if n >= 3 else 0))))


def compute_adaptive_capital_preservation(
    *,
    evaluation_doc: Dict[str, Any],
    trials_doc: Optional[Dict[str, Any]] = None,
    cpi_doc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 20: learn from simulation outcomes (no actions taken)."""
    ts = _iso_utc()
    trials_doc = trials_doc or {}
    cpi_doc = cpi_doc or {}
    evaluations: List[Dict[str, Any]] = evaluation_doc.get("evaluations") or []

    ranked = sorted(evaluations, key=lambda e: e.get("effectiveness_score", 0), reverse=True)
    best = ranked[0] if ranked else {}

    trial_type = str(best.get("trial_type") or "")
    best_protection = TRIAL_TYPE_TO_PROTECTION.get(trial_type, "UNKNOWN")

    most_effective = [
        {
            "protection": TRIAL_TYPE_TO_PROTECTION.get(str(e.get("trial_type")), "UNKNOWN"),
            "trial_name": e.get("trial_name"),
            "effectiveness_score": e.get("effectiveness_score"),
            "evaluation": e.get("evaluation"),
        }
        for e in ranked[:3]
    ]

    by_risk = sorted(evaluations, key=lambda e: e.get("risk_reduction", 0), reverse=True)
    lowest_risk = [
        {
            "protection": TRIAL_TYPE_TO_PROTECTION.get(str(e.get("trial_type")), "UNKNOWN"),
            "trial_name": e.get("trial_name"),
            "risk_reduction": e.get("risk_reduction"),
            "effectiveness_score": e.get("effectiveness_score"),
        }
        for e in by_risk[:3]
    ]

    by_cps = sorted(evaluations, key=lambda e: e.get("cps_improvement", 0), reverse=True)
    highest_cps = [
        {
            "protection": TRIAL_TYPE_TO_PROTECTION.get(str(e.get("trial_type")), "UNKNOWN"),
            "trial_name": e.get("trial_name"),
            "cps_improvement": e.get("cps_improvement"),
            "effectiveness_score": e.get("effectiveness_score"),
        }
        for e in by_cps[:3]
    ]

    avg_effectiveness = evaluation_doc.get("average_effectiveness", 0.0)
    confidence = _adaptive_confidence(evaluations)

    return {
        "generated_at": ts,
        "best_protection": best_protection,
        "best_trial_name": best.get("trial_name"),
        "best_effectiveness_score": best.get("effectiveness_score"),
        "average_effectiveness": avg_effectiveness,
        "confidence": confidence,
        "trial_count": trials_doc.get("trial_count", len(evaluations)),
        "baseline_cps": trials_doc.get("baseline_cps") or cpi_doc.get("capital_preservation_score"),
        "most_effective_protections": most_effective,
        "lowest_risk_protections": lowest_risk,
        "highest_cps_improvements": highest_cps,
        "learning_summary": (
            f"Top simulation: {best.get('trial_name', 'none')} "
            f"(score {best.get('effectiveness_score', 0)}). "
            f"Average effectiveness {avg_effectiveness} across {len(evaluations)} trials."
        ),
        "disclaimer": "Adaptive learning from paper simulations only. No automated actions.",
    }


def _governor_top_drivers(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
) -> List[str]:
    drivers: List[str] = []

    for label in cpe_doc.get("escalation_reason_labels") or []:
        if label and label not in drivers:
            drivers.append(str(label))

    component_scores = cpi_doc.get("component_scores") or {}
    weak = sorted(
        ((k, v) for k, v in component_scores.items() if isinstance(v, (int, float))),
        key=lambda x: x[1],
    )
    component_labels = {
        "drawdown": "Drawdown Risk",
        "concentration": "Concentration Risk",
        "exposure": "Exposure Risk",
        "operational": "Operational Risk",
        "execution": "Execution Risk",
    }
    for name, score in weak[:2]:
        if score < 50:
            label = component_labels.get(name, name.title())
            if label not in drivers:
                drivers.append(label)

    if readiness_doc.get("readiness_status") == "NOT_READY":
        if "Readiness Failure" not in drivers:
            drivers.append("Readiness Failure")
    if not auth_doc.get("overall_authorization"):
        if "Authorization Gap" not in drivers:
            drivers.append("Authorization Gap")

    negative_evals = [
        e for e in (evaluation_doc.get("evaluations") or []) if e.get("evaluation") == "NEGATIVE"
    ]
    if len(negative_evals) >= 2 and "Weak Trial Outcomes" not in drivers:
        drivers.append("Weak Trial Outcomes")

    return drivers[:5]


def _derive_preservation_posture(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    adaptive_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
) -> str:
    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    escalation = str(
        cpe_doc.get("escalation_state") or gov_doc.get("escalation_state") or "GREEN"
    ).upper()
    esc_sev = ESCALATION_SEVERITY.get(escalation, 0)
    readiness = str(readiness_doc.get("readiness_status") or "NOT_READY")
    negative_count = evaluation_doc.get("negative_count", 0)
    beneficial_count = evaluation_doc.get("beneficial_count", 0)
    avg_effectiveness = _safe_float(adaptive_doc.get("average_effectiveness"), 0.0) or 0.0

    if escalation == "CRITICAL" or cps < 25:
        return "CRITICAL"
    if escalation == "RED" or cps < 40 or (readiness == "NOT_READY" and cps < 50):
        return "RED"
    if esc_sev >= 2 or cps < 55 or negative_count >= 2:
        return "ORANGE"
    if (
        escalation == "YELLOW"
        or cps < 65
        or readiness == "PARTIALLY_READY"
        or not auth_doc.get("overall_authorization")
        or (beneficial_count == 0 and evaluation_doc.get("evaluation_count", 0) > 0)
    ):
        return "YELLOW"
    if cps >= 75 and escalation == "GREEN" and avg_effectiveness >= 60:
        return "GREEN"
    return "YELLOW" if cps < 75 else "GREEN"


def _governor_confidence(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    adaptive_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
) -> int:
    score = 50
    if cpi_doc.get("generated_at"):
        score += 10
    if cpe_doc.get("generated_at"):
        score += 10
    if evaluation_doc.get("evaluation_count", 0) >= 3:
        score += 10
    if adaptive_doc.get("confidence"):
        score += min(15, int(adaptive_doc.get("confidence", 0)) // 10)
    if readiness_doc.get("checks_passing_count", 0) >= 5:
        score += 5
    if auth_doc.get("governance_authorized"):
        score += 5
    if not readiness_doc.get("failed_checks"):
        score += 5
    return int(max(40, min(99, score)))


def compute_capital_preservation_governor(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    adaptive_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 21: unified preservation posture (advisory only)."""
    ts = _iso_utc()

    posture = _derive_preservation_posture(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        evaluation_doc=evaluation_doc,
        adaptive_doc=adaptive_doc,
        auth_doc=auth_doc,
        readiness_doc=readiness_doc,
        gov_doc=gov_doc,
    )
    confidence = _governor_confidence(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        evaluation_doc=evaluation_doc,
        adaptive_doc=adaptive_doc,
        readiness_doc=readiness_doc,
        auth_doc=auth_doc,
    )
    top_drivers = _governor_top_drivers(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        evaluation_doc=evaluation_doc,
        readiness_doc=readiness_doc,
        auth_doc=auth_doc,
    )

    recommended_actions: List[str] = []
    if posture in ("RED", "CRITICAL"):
        recommended_actions.extend(cpe_doc.get("recommended_review_actions") or [])
    elif posture == "ORANGE":
        recommended_actions.append("Review protective trial evaluations")
        recommended_actions.extend((cpe_doc.get("recommended_review_actions") or [])[:2])
    elif posture == "YELLOW":
        recommended_actions.append("Monitor capital preservation trends")
        if adaptive_doc.get("best_trial_name"):
            recommended_actions.append(
                f"Consider paper review of: {adaptive_doc.get('best_trial_name')}"
            )
    else:
        recommended_actions.append("Maintain current preservation monitoring")

    return {
        "generated_at": ts,
        "preservation_posture": posture,
        "governor_confidence": confidence,
        "capital_preservation_score": cpi_doc.get("capital_preservation_score"),
        "escalation_state": cpe_doc.get("escalation_state"),
        "readiness_status": readiness_doc.get("readiness_status"),
        "overall_authorization": auth_doc.get("overall_authorization", False),
        "best_protection": adaptive_doc.get("best_protection"),
        "average_trial_effectiveness": adaptive_doc.get("average_effectiveness"),
        "top_drivers": top_drivers,
        "recommended_review_actions": recommended_actions[:5],
        "governance_awareness_level": gov_doc.get("governance_awareness_level"),
        "live_execution_permitted": False,
        "disclaimer": "Governor advisory only. No automated trading or portfolio changes.",
    }


def _append_evaluation_history(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "evaluation_count",
        "average_effectiveness",
        "beneficial_count",
        "neutral_count",
        "negative_count",
        "top_trial",
        "top_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def _append_governor_history(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "preservation_posture",
        "governor_confidence",
        "capital_preservation_score",
        "escalation_state",
        "readiness_status",
        "top_driver_1",
        "top_driver_2",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def persist_capital_preservation_evaluation(
    *,
    results_dir: Path,
    trials_doc: Dict[str, Any],
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Run phases 19–21 and write JSON artifacts."""
    results_dir = Path(results_dir)

    evaluation_doc = compute_protective_action_evaluation(
        trials_doc=trials_doc,
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
    )
    _atomic_write_json(evaluation_doc, results_dir / "protective_action_evaluation.json")

    adaptive_doc = compute_adaptive_capital_preservation(
        evaluation_doc=evaluation_doc,
        trials_doc=trials_doc,
        cpi_doc=cpi_doc,
    )
    _atomic_write_json(adaptive_doc, results_dir / "adaptive_capital_preservation.json")

    governor_doc = compute_capital_preservation_governor(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        auth_doc=auth_doc,
        readiness_doc=readiness_doc,
        evaluation_doc=evaluation_doc,
        adaptive_doc=adaptive_doc,
        gov_doc=gov_doc,
    )
    _atomic_write_json(governor_doc, results_dir / "capital_preservation_governor.json")

    top = evaluation_doc.get("top_trial") or {}
    _append_evaluation_history(
        results_dir / "protective_action_evaluation_history.csv",
        {
            "timestamp": evaluation_doc.get("generated_at"),
            "evaluation_count": evaluation_doc.get("evaluation_count"),
            "average_effectiveness": evaluation_doc.get("average_effectiveness"),
            "beneficial_count": evaluation_doc.get("beneficial_count"),
            "neutral_count": evaluation_doc.get("neutral_count"),
            "negative_count": evaluation_doc.get("negative_count"),
            "top_trial": top.get("trial_name"),
            "top_score": top.get("effectiveness_score"),
        },
    )

    drivers = governor_doc.get("top_drivers") or []
    _append_governor_history(
        results_dir / "capital_preservation_governor_history.csv",
        {
            "timestamp": governor_doc.get("generated_at"),
            "preservation_posture": governor_doc.get("preservation_posture"),
            "governor_confidence": governor_doc.get("governor_confidence"),
            "capital_preservation_score": governor_doc.get("capital_preservation_score"),
            "escalation_state": governor_doc.get("escalation_state"),
            "readiness_status": governor_doc.get("readiness_status"),
            "top_driver_1": drivers[0] if len(drivers) > 0 else "",
            "top_driver_2": drivers[1] if len(drivers) > 1 else "",
        },
    )

    return {
        "protective_action_evaluation": evaluation_doc,
        "adaptive_capital_preservation": adaptive_doc,
        "capital_preservation_governor": governor_doc,
    }
