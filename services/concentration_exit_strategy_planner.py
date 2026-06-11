"""
TRITON Phase 181 — Concentration Exit Strategy Planner (analysis only).

Designs the governance-safe path for reducing concentrated QQQ exposure while
preserving lifecycle compliance, authorization integrity, and recovery objectives.

SAFETY: planning only — no governance, lifecycle, execution, broker, allocation,
or risk control changes. Does not place trades or submit orders.

Outputs (data/results/):
    concentration_exit_strategy.json
    concentration_exit_steps.csv
    concentration_exit_scenarios.csv

Run:
    python -m services.concentration_exit_strategy_planner
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from services.concentration_governance_analysis import (
    CPI_WEIGHTS,
    _concentration_score,
    _estimate_cps,
    _escalation_from_cps,
    _f,
    _load_positions,
    _read_csv,
    _read_json,
)
from services.concentration_recovery_simulator import (
    CPS_GREEN,
    CPS_YELLOW,
    _bito_eligible,
    _simulate_scenario,
)
from services.pnl_ledger_builder import _norm_sym, _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "concentration_exit_strategy.json"
OUT_STEPS = RESULTS / "concentration_exit_steps.csv"
OUT_SCENARIOS = RESULTS / "concentration_exit_scenarios.csv"

TARGET_YELLOW = {"QQQ": 60.0, "HD": 15.0, "XLK": 25.0}
TARGET_GREEN = {"QQQ": 25.0, "HD": 25.0, "XLK": 25.0, "BITO": 25.0}
HD_XLK_RATIO = (0.375, 0.625)  # 15:25 from threshold optimizer


def _log(msg: str) -> None:
    print(f"[concentration_exit_strategy_planner] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prob_label(score: float) -> str:
    if score >= 0.80:
        return "VERY_HIGH"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.35:
        return "MODERATE"
    return "LOW"


def _weights_at_qqq(
    qqq_pct: float,
    target: Dict[str, float],
    bito_ok: bool,
    green_mode: bool = False,
) -> Dict[str, float]:
    div = 100.0 - qqq_pct
    if div <= 0:
        return {"QQQ": 100.0}
    if green_mode and bito_ok:
        each = div / 3.0
        return {"QQQ": qqq_pct, "HD": each, "XLK": each, "BITO": each}
    hd_share, xlk_share = HD_XLK_RATIO
    hd = div * hd_share
    xlk = div * xlk_share
    return {"QQQ": qqq_pct, "HD": _f(hd, 2) or 0.0, "XLK": _f(xlk, 2) or 0.0}


def _step_recovery_probability(
    weights: Dict[str, float],
    readiness: Dict[str, Any],
    bito_ok: bool,
) -> Tuple[str, float]:
    adds = [s for s in weights if s != "QQQ" and weights.get(s, 0) > 0]
    cand_map = {c.get("ticker"): c for c in (readiness.get("candidate_readiness") or [])}
    scores = []
    for sym in adds:
        c = cand_map.get(sym, {})
        rs = c.get("readiness_status", "")
        if rs == "READY":
            scores.append(1.0)
        elif rs == "READY_WITH_CONDITIONS":
            scores.append(0.75)
        elif sym == "BITO" and not bito_ok:
            scores.append(0.35)
        else:
            scores.append(0.5)
    score = float(np.mean(scores)) if scores else 0.5
    return _prob_label(score), score


def _build_plan_steps(
    plan_name: str,
    step_size: float,
    risk_class: str,
    total_mv: float,
    base_components: Dict[str, int],
    baseline_largest: float,
    current_cps: int,
    target_qqq: float,
    readiness: Dict[str, Any],
    bito_ok: bool,
    days_per_step: int,
    include_green_extension: bool,
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    prev_qqq = 100.0
    prev_div = 0.0
    prev_cps = current_cps
    step_num = 0
    yellow_step: Optional[int] = None
    green_step: Optional[int] = None

    qqq_levels = []
    q = 100.0
    while q > target_qqq:
        q = max(target_qqq, q - step_size)
        qqq_levels.append(q)
    if not qqq_levels or qqq_levels[-1] != target_qqq:
        qqq_levels.append(target_qqq)

    for qqq_pct in qqq_levels:
        step_num += 1
        weights = _weights_at_qqq(qqq_pct, TARGET_YELLOW, bito_ok)
        sim = _simulate_scenario(weights, total_mv, base_components, baseline_largest, current_cps)
        div_pct = 100.0 - qqq_pct
        qqq_red = prev_qqq - qqq_pct
        div_add = div_pct - prev_div
        cps_imp = (sim.get("projected_cps") or 0) - prev_cps
        gov_imp = prev_cps < CPS_YELLOW <= (sim.get("projected_cps") or 0)

        alloc_eff = _f(cps_imp / div_add, 3) if div_add > 0 else 0.0
        rec_eff = _f((sim.get("recovery_score") or 0) / max(div_add, 1), 3)

        prob_label, prob_score = _step_recovery_probability(weights, readiness, bito_ok)

        step = {
            "plan_name": plan_name,
            "step_number": step_num,
            "qqq_weight_pct": _f(qqq_pct, 2),
            "qqq_reduction_pct": _f(qqq_red, 2),
            "diversification_addition_pct": _f(div_add, 2),
            "diversification_total_pct": _f(div_pct, 2),
            "weights": sim.get("weights"),
            "projected_cps": sim.get("projected_cps"),
            "cps_improvement": sim.get("cps_improvement"),
            "step_cps_delta": cps_imp,
            "concentration_reduction_pct": sim.get("concentration_reduction_pct"),
            "governance_improvement": gov_imp,
            "projected_governance_status": sim.get("governance_status"),
            "projected_escalation": sim.get("escalation_level"),
            "recovery_probability": prob_label,
            "recovery_probability_score": _f(prob_score * 100, 1),
            "allocation_efficiency": alloc_eff,
            "recovery_efficiency": rec_eff,
            "risk_classification": risk_class,
            "distance_to_yellow": sim.get("distance_to_yellow"),
            "distance_to_green": sim.get("distance_to_green"),
        }
        steps.append(step)

        if yellow_step is None and (sim.get("projected_cps") or 0) >= CPS_YELLOW:
            yellow_step = step_num
        if green_step is None and (sim.get("projected_cps") or 0) >= CPS_GREEN:
            green_step = step_num

        prev_qqq = qqq_pct
        prev_div = div_pct
        prev_cps = sim.get("projected_cps") or prev_cps

    if include_green_extension and bito_ok and (not green_step or green_step > step_num):
        step_num += 1
        weights = _weights_at_qqq(25.0, TARGET_GREEN, bito_ok, green_mode=True)
        sim = _simulate_scenario(weights, total_mv, base_components, baseline_largest, current_cps)
        prob_label, prob_score = _step_recovery_probability(weights, readiness, bito_ok)
        steps.append(
            {
                "plan_name": plan_name,
                "step_number": step_num,
                "qqq_weight_pct": 25.0,
                "qqq_reduction_pct": _f(prev_qqq - 25.0, 2),
                "diversification_addition_pct": _f(75.0 - prev_div, 2),
                "diversification_total_pct": 75.0,
                "weights": sim.get("weights"),
                "projected_cps": sim.get("projected_cps"),
                "cps_improvement": sim.get("cps_improvement"),
                "step_cps_delta": (sim.get("projected_cps") or 0) - prev_cps,
                "concentration_reduction_pct": sim.get("concentration_reduction_pct"),
                "governance_improvement": True,
                "projected_governance_status": sim.get("governance_status"),
                "projected_escalation": sim.get("escalation_level"),
                "recovery_probability": prob_label,
                "recovery_probability_score": _f(prob_score * 100, 1),
                "allocation_efficiency": None,
                "recovery_efficiency": sim.get("recovery_score"),
                "risk_classification": risk_class,
                "distance_to_yellow": sim.get("distance_to_yellow"),
                "distance_to_green": sim.get("distance_to_green"),
                "note": "GREEN extension with BITO",
            }
        )
        if (sim.get("projected_cps") or 0) >= CPS_GREEN:
            green_step = step_num

    final = steps[-1] if steps else {}
    return {
        "plan_name": plan_name,
        "risk_classification": risk_class,
        "step_size_pct": step_size,
        "days_per_step": days_per_step,
        "total_steps": len(steps),
        "steps_to_yellow": yellow_step,
        "steps_to_green": green_step,
        "estimated_days_to_yellow": (yellow_step or len(steps)) * days_per_step,
        "estimated_days_to_green": (green_step * days_per_step if green_step else None),
        "final_weights": final.get("weights"),
        "final_cps": final.get("projected_cps"),
        "final_escalation": final.get("projected_escalation"),
        "final_governance_status": final.get("projected_governance_status"),
        "steps": steps,
    }


def _recommend_plan(
    conservative: Dict[str, Any],
    balanced: Dict[str, Any],
    aggressive: Dict[str, Any],
) -> Dict[str, Any]:
    """Balanced preferred for governance-safe recovery; conservative if similar outcome."""
    for plan in (balanced, conservative, aggressive):
        if plan.get("steps_to_yellow") and plan.get("final_governance_status") == "AUTHORIZED":
            return {
                "recommended_plan": plan.get("plan_name"),
                "reason": (
                    "Balanced 10% step cadence aligns with Phase 179/180 threshold plan "
                    "while minimizing disruption per step."
                    if plan == balanced
                    else f"Selected {plan.get('plan_name')} for optimal risk/recovery tradeoff"
                ),
                **{
                    k: plan.get(k)
                    for k in ("total_steps", "steps_to_yellow", "final_cps", "final_escalation")
                },
            }
    return {"recommended_plan": "BALANCED_EXIT_PLAN", "reason": "Default governance-safe cadence"}


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    readiness = _read_json(RESULTS / "diversification_execution_readiness.json")
    thresholds = _read_json(RESULTS / "governance_recovery_thresholds.json")
    simulation = _read_json(RESULTS / "concentration_recovery_simulation.json")
    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    lifecycle = _read_csv(RESULTS / "signal_lifecycle_effective.csv")
    cpi = _read_json(RESULTS / "capital_preservation_intelligence.json")
    cpe = _read_json(RESULTS / "capital_preservation_escalation.json")
    auth = _read_json(RESULTS / "governance_authorization.json")

    positions = _load_positions()
    total_mv = sum(p["market_value"] for p in positions)
    baseline_largest = _concentration_score(positions)[1]

    base_components = dict(cpi.get("component_scores") or {})
    for k in CPI_WEIGHTS:
        base_components.setdefault(k, 70)
    current_cps = int(
        cpi.get("capital_preservation_score") or _estimate_cps(positions, base_components)[0]
    )
    current_esc = str(cpe.get("escalation_state") or _escalation_from_cps(current_cps))
    current_gov = "AUTHORIZED" if auth.get("governance_authorized") else "BLOCKED"
    bito_ok = _bito_eligible(recommendations, lifecycle)

    target_weights = (
        (readiness.get("recovery_plan") or {}).get("weights")
        or (thresholds.get("yellow_threshold") or {})
        .get("minimum_yellow_recovery_plan", {})
        .get("weights")
        or TARGET_YELLOW
    )
    target_qqq = _to_float(target_weights.get("QQQ"), 60.0)

    current_state = {
        "generated_at": ts,
        "qqq_concentration_pct": _f(baseline_largest, 2),
        "current_cps": current_cps,
        "current_escalation": current_esc,
        "current_governance_status": current_gov,
        "hd_status": "OPEN_NEW",
        "xlk_status": "OPEN_NEW",
        "bito_status": "OPEN_NEW" if bito_ok else "WATCH",
        "target_yellow_weights": target_weights,
        "execution_readiness": (readiness.get("executive_report") or {}).get("execution_readiness"),
    }

    conservative = _build_plan_steps(
        "CONSERVATIVE_EXIT_PLAN",
        step_size=5.0,
        risk_class="LOW_RISK",
        total_mv=total_mv,
        base_components=base_components,
        baseline_largest=baseline_largest,
        current_cps=current_cps,
        target_qqq=target_qqq,
        readiness=readiness,
        bito_ok=bito_ok,
        days_per_step=7,
        include_green_extension=True,
    )
    balanced = _build_plan_steps(
        "BALANCED_EXIT_PLAN",
        step_size=10.0,
        risk_class="MODERATE_RISK",
        total_mv=total_mv,
        base_components=base_components,
        baseline_largest=baseline_largest,
        current_cps=current_cps,
        target_qqq=target_qqq,
        readiness=readiness,
        bito_ok=bito_ok,
        days_per_step=5,
        include_green_extension=True,
    )
    aggressive = _build_plan_steps(
        "AGGRESSIVE_EXIT_PLAN",
        step_size=15.0,
        risk_class="HIGH_RISK",
        total_mv=total_mv,
        base_components=base_components,
        baseline_largest=baseline_largest,
        current_cps=current_cps,
        target_qqq=target_qqq,
        readiness=readiness,
        bito_ok=bito_ok,
        days_per_step=3,
        include_green_extension=True,
    )

    recommended = _recommend_plan(conservative, balanced, aggressive)
    rec_plan = (
        balanced
        if recommended.get("recommended_plan") == "BALANCED_EXIT_PLAN"
        else (
            conservative
            if recommended.get("recommended_plan") == "CONSERVATIVE_EXIT_PLAN"
            else aggressive
        )
    )

    timeline = {
        "conservative": {
            "steps_to_yellow": conservative.get("steps_to_yellow"),
            "steps_to_green": conservative.get("steps_to_green"),
            "estimated_days_to_yellow": conservative.get("estimated_days_to_yellow"),
            "estimated_days_to_green": conservative.get("estimated_days_to_green"),
        },
        "balanced": {
            "steps_to_yellow": balanced.get("steps_to_yellow"),
            "steps_to_green": balanced.get("steps_to_green"),
            "estimated_days_to_yellow": balanced.get("estimated_days_to_yellow"),
            "estimated_days_to_green": balanced.get("estimated_days_to_green"),
        },
        "aggressive": {
            "steps_to_yellow": aggressive.get("steps_to_yellow"),
            "steps_to_green": aggressive.get("steps_to_green"),
            "estimated_days_to_yellow": aggressive.get("estimated_days_to_yellow"),
            "estimated_days_to_green": aggressive.get("estimated_days_to_green"),
        },
        "recommended_plan": recommended.get("recommended_plan"),
        "yellow_achievable_at_step": rec_plan.get("steps_to_yellow"),
        "green_achievable": bito_ok and bool(rec_plan.get("steps_to_green")),
        "green_requires_bito_promotion": not bito_ok,
        "green_possible_after_bito_promotion": (
            "Additional step with equal-weight QQQ/HD/XLK/BITO at 25% each" if not bito_ok else None
        ),
    }

    cps_path = [
        {
            "plan": p.get("plan_name"),
            "step": s.get("step_number"),
            "qqq_pct": s.get("qqq_weight_pct"),
            "projected_cps": s.get("projected_cps"),
            "escalation": s.get("projected_escalation"),
        }
        for p in (conservative, balanced, aggressive)
        for s in p.get("steps") or []
    ]

    executive = {
        "report_type": "CONCENTRATION_EXIT_STRATEGY_REPORT",
        "current_concentration_pct": current_state["qqq_concentration_pct"],
        "recommended_exit_path": recommended.get("recommended_plan"),
        "projected_cps_path": [
            {"step": s.get("step_number"), "cps": s.get("projected_cps")}
            for s in rec_plan.get("steps") or []
        ],
        "expected_governance_outcome": (
            f"{current_esc} → {rec_plan.get('final_escalation')} "
            f"({rec_plan.get('final_governance_status')})"
        ),
        "expected_recovery_timeline_days": rec_plan.get("estimated_days_to_yellow"),
        "risk_classification": rec_plan.get("risk_classification"),
        "recommended_strategy": recommended.get("recommended_plan"),
        "safest_path": "CONSERVATIVE_EXIT_PLAN",
        "fastest_path": "AGGRESSIVE_EXIT_PLAN",
        "steps_required_for_yellow": rec_plan.get("steps_to_yellow"),
        "yellow_expected_at_step": rec_plan.get("steps_to_yellow"),
        "green_possible_when": (
            "After BITO OPEN_NEW promotion + extension step"
            if not bito_ok
            else f"Step {rec_plan.get('steps_to_green')}"
        ),
    }

    all_steps: List[Dict[str, Any]] = []
    for plan in (conservative, balanced, aggressive):
        all_steps.extend(plan.get("steps") or [])
    pd.DataFrame(all_steps).to_csv(OUT_STEPS, index=False, encoding="utf-8")

    scenario_rows = [
        {
            "plan_name": p.get("plan_name"),
            "risk_classification": p.get("risk_classification"),
            "step_size_pct": p.get("step_size_pct"),
            "total_steps": p.get("total_steps"),
            "steps_to_yellow": p.get("steps_to_yellow"),
            "steps_to_green": p.get("steps_to_green"),
            "estimated_days_to_yellow": p.get("estimated_days_to_yellow"),
            "estimated_days_to_green": p.get("estimated_days_to_green"),
            "final_cps": p.get("final_cps"),
            "final_escalation": p.get("final_escalation"),
            "final_governance_status": p.get("final_governance_status"),
            "recommended": p.get("plan_name") == recommended.get("recommended_plan"),
        }
        for p in (conservative, balanced, aggressive)
    ]
    pd.DataFrame(scenario_rows).to_csv(OUT_SCENARIOS, index=False, encoding="utf-8")

    summary = {
        "plans_generated": 3,
        "recommended_plan": recommended.get("recommended_plan"),
        "yellow_steps_balanced": balanced.get("steps_to_yellow"),
        "green_achievable_today": bito_ok and bool(balanced.get("steps_to_green")),
        "bito_blocks_green": not bito_ok,
        "target_qqq_pct": target_qqq,
    }

    report = {
        "generated_at": ts,
        "executive_report": executive,
        "current_state": current_state,
        "conservative_plan": conservative,
        "balanced_plan": balanced,
        "aggressive_plan": aggressive,
        "recommended_plan": {**recommended, "plan_detail": rec_plan},
        "timeline": timeline,
        "cps_path": cps_path,
        "governance_impact": {
            "current": {"cps": current_cps, "escalation": current_esc, "governance": current_gov},
            "post_recommended": {
                "cps": rec_plan.get("final_cps"),
                "escalation": rec_plan.get("final_escalation"),
                "governance": rec_plan.get("final_governance_status"),
            },
            "cps_improvement": (rec_plan.get("final_cps") or 0) - current_cps,
        },
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "planning-only — no trades, orders, positions, or authorization modified",
        },
        "disclaimer": (
            "Exit strategy planning only — does not place trades, submit orders, or modify "
            "governance, lifecycle, execution, broker, authorization, or allocation rules."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_STEPS.name}, {OUT_SCENARIOS.name}")
    _log(
        f"recommended={recommended.get('recommended_plan')} "
        f"yellow_step={rec_plan.get('steps_to_yellow')} final_cps={rec_plan.get('final_cps')}"
    )
    return report


if __name__ == "__main__":
    run()
