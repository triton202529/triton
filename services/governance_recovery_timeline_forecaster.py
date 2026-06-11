"""
TRITON Phase 182 — Governance Recovery Timeline Forecaster (forecasting only).

Forecasts when governance recovery is expected under approved concentration
recovery strategies. Introduces time-based forecasting into the governance
intelligence layer.

SAFETY: forecasting only — no governance, lifecycle, execution, broker, allocation,
or risk control changes. Does not place trades or submit orders.

Outputs (data/results/):
    governance_recovery_timeline.json
    governance_recovery_timeline.csv
    governance_recovery_scenarios.csv

Run:
    python -m services.governance_recovery_timeline_forecaster
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from services.concentration_governance_analysis import _f, _read_csv, _read_json
from services.concentration_recovery_simulator import CPS_GREEN, CPS_YELLOW
from services.pnl_ledger_builder import _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "governance_recovery_timeline.json"
OUT_TIMELINE = RESULTS / "governance_recovery_timeline.csv"
OUT_SCENARIOS = RESULTS / "governance_recovery_scenarios.csv"

FORECAST_DAYS = [1, 7, 14, 21, 30, 60, 90]
STRATEGY_MAP = {
    "CONSERVATIVE_PLAN": "conservative_plan",
    "BALANCED_PLAN": "balanced_plan",
    "AGGRESSIVE_PLAN": "aggressive_plan",
}


def _log(msg: str) -> None:
    print(f"[governance_recovery_timeline_forecaster] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


def _date_offset(base: datetime, days: int) -> str:
    return (base + timedelta(days=days)).strftime("%Y-%m-%d")


def _conf_label(score: float) -> str:
    if score >= 0.80:
        return "VERY_HIGH"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.35:
        return "MODERATE"
    return "LOW"


def _risk_label(score: float) -> str:
    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.50:
        return "HIGH"
    if score >= 0.25:
        return "MODERATE"
    return "LOW"


def _state_at_day(
    plan: Dict[str, Any],
    day: int,
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    steps = plan.get("steps") or []
    days_per = int(plan.get("days_per_step") or 5)
    if day <= 0 or not steps:
        return {
            "projected_cps": baseline.get("current_cps"),
            "projected_governance_status": baseline.get("current_governance_status"),
            "projected_escalation": baseline.get("current_escalation"),
            "projected_concentration": baseline.get("qqq_concentration_pct"),
            "projected_recovery_probability": baseline.get("recovery_probability", "MODERATE"),
            "recovery_probability_score": baseline.get("recovery_probability_score", 50),
            "step_number": 0,
        }
    step_idx = min(len(steps) - 1, max(0, (day - 1) // days_per))
    s = steps[step_idx]
    return {
        "projected_cps": s.get("projected_cps"),
        "projected_governance_status": s.get("projected_governance_status"),
        "projected_escalation": s.get("projected_escalation"),
        "projected_concentration": s.get("qqq_weight_pct"),
        "projected_recovery_probability": s.get("recovery_probability"),
        "recovery_probability_score": s.get("recovery_probability_score"),
        "step_number": s.get("step_number"),
    }


def _forecast_confidence(day: int, step: int, prob_score: float, days_to_target: int) -> float:
    horizon_penalty = min(day / 90.0, 1.0) * 0.25
    step_bonus = min(step / 4.0, 1.0) * 0.15 if step else 0
    proximity = max(0, 1.0 - abs(day - days_to_target) / 30.0) * 0.2 if days_to_target else 0
    prob = min(_to_float(prob_score, 50) / 100.0, 1.0)
    return float(np.clip(0.45 + prob * 0.35 + step_bonus + proximity - horizon_penalty, 0.25, 0.95))


def _build_strategy_forecast(
    strategy_key: str,
    plan: Dict[str, Any],
    baseline: Dict[str, Any],
    base_dt: datetime,
) -> Dict[str, Any]:
    days_yellow = int(plan.get("estimated_days_to_yellow") or 30)
    days_green = plan.get("estimated_days_to_green")
    rows: List[Dict[str, Any]] = []

    yellow_day: Optional[int] = None
    green_day: Optional[int] = None

    for day in FORECAST_DAYS:
        st = _state_at_day(plan, day, baseline)
        cps = _to_float(st.get("projected_cps"), baseline.get("current_cps", 66))
        prob_score = _to_float(st.get("recovery_probability_score"), 50)
        fc = _forecast_confidence(day, int(st.get("step_number") or 0), prob_score, days_yellow)

        if yellow_day is None and cps >= CPS_YELLOW:
            yellow_day = day
        if green_day is None and cps >= CPS_GREEN:
            green_day = day

        rows.append(
            {
                "strategy": strategy_key,
                "forecast_day": day,
                "forecast_date": _date_offset(base_dt, day),
                "projected_cps": int(cps) if np.isfinite(cps) else None,
                "projected_governance_status": st.get("projected_governance_status"),
                "projected_escalation": st.get("projected_escalation"),
                "projected_concentration": st.get("projected_concentration"),
                "projected_recovery_probability": st.get("projected_recovery_probability"),
                "recovery_probability_score": prob_score,
                "forecast_confidence": _f(fc * 100, 1),
                "forecast_confidence_label": _conf_label(fc),
                "step_number": st.get("step_number"),
            }
        )

    if yellow_day is None:
        yellow_day = days_yellow
    if green_day is None and days_green:
        green_day = int(days_green)

    last_prob = _to_float(rows[-1].get("recovery_probability_score"), 50) if rows else 50
    trans_conf = float(np.clip(0.5 + last_prob / 200.0, 0.3, 0.95))

    return {
        "strategy": strategy_key,
        "plan_name": plan.get("plan_name"),
        "days_per_step": plan.get("days_per_step"),
        "days_to_yellow": days_yellow,
        "days_to_green": days_green,
        "expected_yellow_date": _date_offset(base_dt, days_yellow),
        "expected_green_date": _date_offset(base_dt, int(days_green)) if days_green else None,
        "transition_confidence": _f(trans_conf * 100, 1),
        "transition_confidence_label": _conf_label(trans_conf),
        "forecast_points": rows,
    }


def _build_scenario_case(
    name: str,
    plan: Dict[str, Any],
    baseline: Dict[str, Any],
    base_dt: datetime,
    day_multiplier: float,
    bito_delay_days: int,
    confidence_adj: float,
) -> Dict[str, Any]:
    adj_plan = dict(plan)
    adj_plan["days_per_step"] = max(1, int((plan.get("days_per_step") or 5) * day_multiplier))
    days_yellow = int((plan.get("estimated_days_to_yellow") or 20) * day_multiplier)

    cps_path: List[Dict[str, Any]] = []
    gov_path: List[str] = []
    esc_path: List[str] = []
    prob_path: List[str] = []

    for day in FORECAST_DAYS:
        st = _state_at_day(adj_plan, day, baseline)
        cps_path.append({"day": day, "cps": st.get("projected_cps")})
        gov_path.append(str(st.get("projected_governance_status")))
        esc_path.append(str(st.get("projected_escalation")))
        prob_path.append(str(st.get("projected_recovery_probability")))

    green_day = days_yellow + bito_delay_days + int((adj_plan.get("days_per_step") or 5) * 2)
    fc = float(np.clip(0.55 + confidence_adj, 0.2, 0.95))

    return {
        "scenario": name,
        "strategy_basis": plan.get("plan_name"),
        "days_to_yellow": days_yellow,
        "days_to_green": green_day,
        "expected_yellow_date": _date_offset(base_dt, days_yellow),
        "expected_green_date": _date_offset(base_dt, green_day),
        "cps_path": cps_path,
        "governance_path": gov_path,
        "escalation_path": esc_path,
        "recovery_probability_path": prob_path,
        "timeline_confidence": _f(fc * 100, 1),
        "timeline_confidence_label": _conf_label(fc),
        "bito_assumption_days": bito_delay_days,
    }


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    base_dt = _parse_ts(ts)

    exit_strategy = _read_json(RESULTS / "concentration_exit_strategy.json")
    readiness = _read_json(RESULTS / "diversification_execution_readiness.json")
    monitor = _read_json(RESULTS / "recovery_basket_monitor.json")
    thresholds = _read_json(RESULTS / "governance_recovery_thresholds.json")
    simulation = _read_json(RESULTS / "concentration_recovery_simulation.json")

    cs = exit_strategy.get("current_state") or {}
    baseline = {
        "current_cps": cs.get("current_cps") or 66,
        "current_escalation": cs.get("current_escalation") or "ORANGE",
        "current_governance_status": cs.get("current_governance_status") or "BLOCKED",
        "qqq_concentration_pct": cs.get("qqq_concentration_pct") or 100.0,
        "recovery_probability": (readiness.get("executive_report") or {}).get(
            "recovery_probability", "HIGH"
        ),
        "recovery_probability_score": _to_float(
            (readiness.get("execution_probability") or {}).get("combined_recovery_probability"),
            75,
        ),
    }

    strategy_forecasts: Dict[str, Any] = {}
    timeline_rows: List[Dict[str, Any]] = []

    for key, plan_field in STRATEGY_MAP.items():
        plan = exit_strategy.get(plan_field) or {}
        if not plan:
            continue
        fc = _build_strategy_forecast(key, plan, baseline, base_dt)
        strategy_forecasts[key] = fc
        timeline_rows.extend(fc.get("forecast_points") or [])

    conservative = exit_strategy.get("conservative_plan") or {}
    balanced = exit_strategy.get("balanced_plan") or {}
    aggressive = exit_strategy.get("aggressive_plan") or {}
    recommended = (exit_strategy.get("recommended_plan") or {}).get(
        "recommended_plan", "BALANCED_EXIT_PLAN"
    )

    rec_plan = (
        balanced
        if "BALANCED" in recommended
        else (conservative if "CONSERVATIVE" in recommended else aggressive)
    )
    expected_fc = strategy_forecasts.get("BALANCED_PLAN") or strategy_forecasts.get(
        list(strategy_forecasts.keys())[0] if strategy_forecasts else ""
    )

    bito_gap = _to_float(
        (readiness.get("promotion_dependencies") or {})
        .get("bito_eligibility", {})
        .get("confidence_gap"),
        0.025,
    )

    best_case = _build_scenario_case(
        "BEST_CASE",
        aggressive,
        baseline,
        base_dt,
        day_multiplier=0.85,
        bito_delay_days=14,
        confidence_adj=0.25,
    )
    expected_case = _build_scenario_case(
        "EXPECTED_CASE",
        balanced,
        baseline,
        base_dt,
        day_multiplier=1.0,
        bito_delay_days=30,
        confidence_adj=0.10,
    )
    worst_case = _build_scenario_case(
        "WORST_CASE",
        conservative,
        baseline,
        base_dt,
        day_multiplier=1.5,
        bito_delay_days=90,
        confidence_adj=-0.20,
    )

    promo_risk = _risk_label(0.6 if bito_gap > 0 else 0.2)
    lifecycle_risk = _risk_label(0.25)
    alloc_risk = _risk_label(0.2)
    gov_risk = _risk_label(0.55 if baseline["current_governance_status"] == "BLOCKED" else 0.2)
    conc_risk = _risk_label(0.7 if baseline["qqq_concentration_pct"] >= 90 else 0.3)

    timeline_risks = {
        "promotion_risk": promo_risk,
        "lifecycle_risk": lifecycle_risk,
        "allocation_risk": alloc_risk,
        "governance_risk": gov_risk,
        "concentration_risk": conc_risk,
        "overall_risk": _risk_label(
            np.mean(
                [
                    0.6 if promo_risk in ("HIGH", "CRITICAL") else 0.3,
                    0.55 if gov_risk in ("HIGH", "CRITICAL") else 0.25,
                    0.7 if conc_risk in ("HIGH", "CRITICAL") else 0.3,
                ]
            )
        ),
    }

    accelerators = [
        {
            "factor": "BITO promotion to OPEN_NEW",
            "impact": f"Enables GREEN path (~{best_case['days_to_green']} days best case)",
            "confidence_gap": _f(bito_gap, 4),
        },
        {
            "factor": "HD + XLK OPEN_NEW already available",
            "impact": "YELLOW recovery executable at allocation layer now",
            "status": "ACTIVE",
        },
        {
            "factor": "Aggressive 15% step cadence",
            "impact": f"YELLOW in ~{aggressive.get('estimated_days_to_yellow')} days vs {balanced.get('estimated_days_to_yellow')} balanced",
            "status": "AVAILABLE",
        },
        {
            "factor": "Improved BITO confidence",
            "impact": "Closes 0.025 gap to OPEN_NEW floor — accelerates GREEN timeline",
            "status": "PENDING",
        },
    ]

    delay_factors = [
        {
            "factor": "Governance ORANGE authorization block",
            "impact": "Pre-diversification governance blocked; paradox until concentration reduced",
            "severity": gov_risk,
        },
        {
            "factor": "BITO WATCH status",
            "impact": "Blocks full GREEN recovery until promotion",
            "severity": promo_risk,
        },
        {
            "factor": "Conservative step cadence",
            "impact": f"Extends YELLOW timeline to ~{conservative.get('estimated_days_to_yellow')} days",
            "severity": "MODERATE",
        },
        {
            "factor": "Candidate demotion risk",
            "impact": "HD/XLK WATCH demotion would delay diversification",
            "severity": "LOW",
        },
        {
            "factor": "Governance tightening",
            "impact": "Escalation to RED would halt recovery path",
            "severity": "LOW",
        },
    ]

    timeline_conf = _to_float(expected_case.get("timeline_confidence"), 70) / 100.0
    forecast_conf = _to_float((expected_fc or {}).get("transition_confidence"), 70) / 100.0
    recovery_conf = (
        _to_float((monitor.get("executive_report") or {}).get("recovery_probability_score"), 75)
        / 100.0
    )
    gov_conf = 1.0 - (0.55 if baseline["current_governance_status"] == "BLOCKED" else 0.1)

    confidence_model = {
        "timeline_confidence": _conf_label(timeline_conf),
        "timeline_confidence_score": _f(timeline_conf * 100, 1),
        "forecast_confidence": _conf_label(forecast_conf),
        "forecast_confidence_score": _f(forecast_conf * 100, 1),
        "recovery_confidence": _conf_label(recovery_conf),
        "recovery_confidence_score": _f(recovery_conf * 100, 1),
        "governance_confidence": _conf_label(gov_conf),
        "governance_confidence_score": _f(gov_conf * 100, 1),
    }

    exp_days_yellow = int(rec_plan.get("estimated_days_to_yellow") or 20)
    exp_days_green = expected_case.get("days_to_green")

    executive = {
        "report_type": "GOVERNANCE_RECOVERY_TIMELINE_REPORT",
        "current_state": baseline,
        "recommended_recovery_path": recommended,
        "expected_yellow_date": _date_offset(base_dt, exp_days_yellow),
        "expected_green_date": expected_case.get("expected_green_date"),
        "days_to_yellow": exp_days_yellow,
        "days_to_green": exp_days_green,
        "forecast_confidence": confidence_model["forecast_confidence"],
        "recovery_confidence": confidence_model["recovery_confidence"],
        "major_risks": [d["factor"] for d in delay_factors[:3]],
        "major_accelerators": [a["factor"] for a in accelerators[:3]],
        "executive_recommendation": (
            f"Execute {recommended.replace('_EXIT_PLAN', '').replace('_', ' ').title()} "
            f"cadence — YELLOW expected ~{exp_days_yellow} days ({_date_offset(base_dt, exp_days_yellow)}); "
            f"GREEN ~{exp_days_green} days pending BITO promotion."
        ),
        "projected_governance_recovery": (
            f"ORANGE → YELLOW by day {exp_days_yellow}; "
            f"YELLOW → GREEN by day {exp_days_green} (expected case)"
        ),
    }

    scenario_summary = [
        {
            "scenario": best_case["scenario"],
            "days_to_yellow": best_case["days_to_yellow"],
            "days_to_green": best_case["days_to_green"],
            "timeline_confidence": best_case["timeline_confidence"],
            "strategy_basis": best_case["strategy_basis"],
        },
        {
            "scenario": expected_case["scenario"],
            "days_to_yellow": expected_case["days_to_yellow"],
            "days_to_green": expected_case["days_to_green"],
            "timeline_confidence": expected_case["timeline_confidence"],
            "strategy_basis": expected_case["strategy_basis"],
        },
        {
            "scenario": worst_case["scenario"],
            "days_to_yellow": worst_case["days_to_yellow"],
            "days_to_green": worst_case["days_to_green"],
            "timeline_confidence": worst_case["timeline_confidence"],
            "strategy_basis": worst_case["strategy_basis"],
        },
    ]

    pd.DataFrame(timeline_rows).to_csv(OUT_TIMELINE, index=False, encoding="utf-8")
    pd.DataFrame(scenario_summary).to_csv(OUT_SCENARIOS, index=False, encoding="utf-8")

    summary = {
        "forecast_horizon_days": max(FORECAST_DAYS),
        "strategies_forecast": len(strategy_forecasts),
        "expected_yellow_days": exp_days_yellow,
        "expected_green_days": exp_days_green,
        "most_realistic_strategy": "BALANCED_PLAN",
        "fastest_yellow": aggressive.get("estimated_days_to_yellow"),
        "safest_yellow": conservative.get("estimated_days_to_yellow"),
        "bito_blocks_green": cs.get("bito_status") == "WATCH",
    }

    report = {
        "generated_at": ts,
        "forecast_base_date": base_dt.strftime("%Y-%m-%d"),
        "executive_report": executive,
        "current_state": {**baseline, **cs},
        "strategy_forecasts": strategy_forecasts,
        "governance_transitions": {
            "orange_to_yellow": {
                "expected_day": exp_days_yellow,
                "expected_date": executive["expected_yellow_date"],
                "transition_confidence": confidence_model["forecast_confidence"],
            },
            "yellow_to_green": {
                "expected_day": exp_days_green,
                "expected_date": executive["expected_green_date"],
                "transition_confidence": confidence_model["recovery_confidence"],
                "requires_bito_promotion": True,
            },
        },
        "best_case": best_case,
        "expected_case": expected_case,
        "worst_case": worst_case,
        "timeline_risks": timeline_risks,
        "accelerators": accelerators,
        "delay_factors": delay_factors,
        "confidence_model": confidence_model,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "forecasting-only — no trades, orders, governance, lifecycle, or authorization modified",
        },
        "disclaimer": (
            "Timeline forecasts are planning intelligence only — does not place trades, "
            "submit orders, or modify governance, lifecycle, execution, broker, authorization, "
            "or allocation rules."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_TIMELINE.name}, {OUT_SCENARIOS.name}")
    _log(
        f"yellow~{exp_days_yellow}d green~{exp_days_green}d conf={confidence_model['forecast_confidence']}"
    )
    return report


if __name__ == "__main__":
    run()
