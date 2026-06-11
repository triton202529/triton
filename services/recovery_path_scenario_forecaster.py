"""
TRITON Phase 184 — Recovery Path Scenario Forecasting (forecasting only).

Forecasts multiple future recovery paths and quantifies the probability of each
governance outcome.

SAFETY: forecasting only — no governance, lifecycle, execution, broker, allocation,
or risk control changes.

Outputs (data/results/):
    recovery_path_scenarios.json
    recovery_path_scenarios.csv
    recovery_path_probability_matrix.csv

Run:
    python -m services.recovery_path_scenario_forecaster
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

from services.concentration_governance_analysis import _f, _read_csv, _read_json
from services.concentration_recovery_simulator import CPS_GREEN, CPS_YELLOW
from services.pnl_ledger_builder import _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "recovery_path_scenarios.json"
OUT_CSV = RESULTS / "recovery_path_scenarios.csv"
OUT_MATRIX = RESULTS / "recovery_path_probability_matrix.csv"


def _log(msg: str) -> None:
    print(f"[recovery_path_scenario_forecaster] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _outcome_class(success: float, green: float) -> str:
    if success < 0.15:
        return "RECOVERY_FAILED"
    if success < 0.35:
        return "RECOVERY_UNLIKELY"
    if green >= 0.55:
        return "RECOVERY_CERTAIN"
    if success >= 0.65:
        return "RECOVERY_LIKELY"
    return "RECOVERY_POSSIBLE"


def _conf_label(score: float) -> str:
    if score >= 0.80:
        return "VERY_HIGH"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.35:
        return "MODERATE"
    return "LOW"


def _build_scenario(
    name: str,
    description: str,
    days_yellow: int,
    days_green: Optional[int],
    final_cps: int,
    final_esc: str,
    final_gov: str,
    scenario_prob: float,
    bito_promotes: bool,
    plan_basis: str,
    readiness: Dict[str, Any],
    promo: Dict[str, Any],
) -> Dict[str, Any]:
    yellow_prob = 0.92 if days_yellow <= 30 else 0.75 if days_yellow <= 60 else 0.45
    if name == "FAILED_RECOVERY_CASE":
        yellow_prob = 0.08
    green_prob = 0.0
    if days_green and bito_promotes:
        green_prob = 0.85 if days_green <= 45 else 0.60 if days_green <= 90 else 0.35
    elif days_green and not bito_promotes:
        green_prob = 0.15
    if name == "FAILED_RECOVERY_CASE":
        green_prob = 0.02

    success = yellow_prob * 0.6 + green_prob * 0.4 if green_prob else yellow_prob * 0.85
    failure = max(0.0, 1.0 - success)
    forecast_conf = float(np.clip(0.4 + scenario_prob * 0.4 + yellow_prob * 0.2, 0.25, 0.95))

    recovery_prob = _conf_label(success)
    exec_ready = (readiness.get("executive_report") or {}).get("execution_readiness") == "READY"

    return {
        "scenario": name,
        "description": description,
        "plan_basis": plan_basis,
        "projected_cps": final_cps,
        "projected_governance_status": final_gov,
        "projected_escalation": final_esc,
        "projected_recovery_probability": recovery_prob,
        "projected_green_probability": _conf_label(green_prob),
        "timeline_to_yellow_days": days_yellow,
        "timeline_to_green_days": days_green,
        "scenario_probability": _f(scenario_prob * 100, 1),
        "success_probability": _f(success * 100, 1),
        "failure_probability": _f(failure * 100, 1),
        "yellow_probability": _f(yellow_prob * 100, 1),
        "green_probability": _f(green_prob * 100, 1),
        "forecast_confidence": _f(forecast_conf * 100, 1),
        "forecast_confidence_label": _conf_label(forecast_conf),
        "outcome_classification": _outcome_class(success, green_prob),
        "bito_promotion_required": not bito_promotes and days_green is not None,
        "execution_readiness": exec_ready,
        "key_dependencies": _scenario_deps(name, bito_promotes, promo),
    }


def _scenario_deps(name: str, bito_ok: bool, promo: Dict[str, Any]) -> List[str]:
    deps = [
        "HD OPEN_NEW stability",
        "XLK OPEN_NEW stability",
        "Concentration reduction via exit plan",
    ]
    if not bito_ok:
        deps.append("BITO WATCH → OPEN_NEW promotion")
    if name == "FAILED_RECOVERY_CASE":
        deps.extend(
            ["HD/XLK demotion risk", "Governance escalation to RED", "Execution gate closure"]
        )
    elif name == "ACCELERATED_CASE":
        deps.append("Immediate paper execution authorization")
    return deps


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    timeline = _read_json(RESULTS / "governance_recovery_timeline.json")
    promo = _read_json(RESULTS / "promotion_probability_forecast.json")
    exit_strat = _read_json(RESULTS / "concentration_exit_strategy.json")
    readiness = _read_json(RESULTS / "diversification_execution_readiness.json")
    thresholds = _read_json(RESULTS / "governance_recovery_thresholds.json")

    tl_ex = timeline.get("executive_report") or {}
    best_tl = timeline.get("best_case") or {}
    exp_tl = timeline.get("expected_case") or {}
    worst_tl = timeline.get("worst_case") or {}

    cs = exit_strat.get("current_state") or timeline.get("current_state") or {}
    current_cps = int(cs.get("current_cps") or 66)
    current_esc = str(cs.get("current_escalation") or "ORANGE")
    current_gov = str(cs.get("current_governance_status") or "BLOCKED")

    bito_prob = (
        _to_float(
            (promo.get("promotion_dependencies") or {}).get("bito_open_new_probability_30d"), 38.5
        )
        / 100.0
    )

    aggressive = exit_strat.get("aggressive_plan") or {}
    balanced = exit_strat.get("balanced_plan") or {}
    conservative = exit_strat.get("conservative_plan") or {}

    yellow_cps = int(
        balanced.get("final_cps")
        or thresholds.get("yellow_threshold", {}).get("cps_required")
        or 76
    )
    green_cps = CPS_GREEN

    scenarios = [
        _build_scenario(
            "BEST_CASE",
            "Aggressive exit cadence with early BITO promotion and full GREEN path",
            int(best_tl.get("days_to_yellow") or aggressive.get("estimated_days_to_yellow") or 9),
            int(best_tl.get("days_to_green") or 35),
            green_cps,
            "GREEN",
            "AUTHORIZED",
            0.12,
            True,
            "AGGRESSIVE_EXIT_PLAN",
            readiness,
            promo,
        ),
        _build_scenario(
            "EXPECTED_CASE",
            "Balanced exit cadence with BITO promotion within 30 days — baseline forecast",
            int(exp_tl.get("days_to_yellow") or tl_ex.get("days_to_yellow") or 20),
            int(exp_tl.get("days_to_green") or tl_ex.get("days_to_green") or 60),
            yellow_cps,
            "YELLOW",
            "AUTHORIZED",
            0.40,
            True,
            "BALANCED_EXIT_PLAN",
            readiness,
            promo,
        ),
        _build_scenario(
            "WORST_CASE",
            "Conservative cadence with delayed BITO and extended recovery timeline",
            int(
                worst_tl.get("days_to_yellow") or conservative.get("estimated_days_to_yellow") or 84
            ),
            int(worst_tl.get("days_to_green") or 194),
            yellow_cps,
            "YELLOW",
            "AUTHORIZED",
            0.15,
            False,
            "CONSERVATIVE_EXIT_PLAN",
            readiness,
            promo,
        ),
        _build_scenario(
            "ACCELERATED_CASE",
            "Fast-track execution with HD+XLK immediate diversification",
            int(aggressive.get("estimated_days_to_yellow") or 9),
            int(aggressive.get("estimated_days_to_yellow", 9) + 25),
            yellow_cps + 4,
            "YELLOW",
            "AUTHORIZED",
            0.18,
            bito_prob > 0.5,
            "AGGRESSIVE_EXIT_PLAN + immediate execution",
            readiness,
            promo,
        ),
        _build_scenario(
            "DELAYED_CASE",
            "Governance paradox delays plus BITO promotion beyond 60 days",
            int(conservative.get("estimated_days_to_yellow") or 56) + 14,
            int(exp_tl.get("days_to_green") or 60) + 45,
            max(current_cps + 5, 70),
            "ORANGE",
            "BLOCKED",
            0.10,
            False,
            "CONSERVATIVE_EXIT_PLAN + authorization delay",
            readiness,
            promo,
        ),
        _build_scenario(
            "FAILED_RECOVERY_CASE",
            "HD/XLK demotion or governance escalation prevents recovery",
            999,
            None,
            current_cps,
            "ORANGE",
            "BLOCKED",
            0.05,
            False,
            "Demotion / escalation failure mode",
            readiness,
            promo,
        ),
    ]

    prob_sum = sum(_to_float(s.get("scenario_probability"), 0) / 100.0 for s in scenarios)
    if prob_sum > 0:
        for s in scenarios:
            raw = _to_float(s.get("scenario_probability"), 0) / 100.0
            s["scenario_probability"] = _f(raw / prob_sum * 100, 1)

    yellow_agg = sum(
        _to_float(s.get("scenario_probability"), 0)
        / 100.0
        * _to_float(s.get("yellow_probability"), 0)
        / 100.0
        for s in scenarios
    )
    green_agg = sum(
        _to_float(s.get("scenario_probability"), 0)
        / 100.0
        * _to_float(s.get("green_probability"), 0)
        / 100.0
        for s in scenarios
    )

    ranked = sorted(
        scenarios, key=lambda s: _to_float(s.get("scenario_probability"), 0), reverse=True
    )
    fastest = min(
        [s for s in scenarios if s["scenario"] != "FAILED_RECOVERY_CASE"],
        key=lambda s: s.get("timeline_to_yellow_days") or 999,
    )
    slowest = max(
        [s for s in scenarios if s["scenario"] != "FAILED_RECOVERY_CASE"],
        key=lambda s: s.get("timeline_to_yellow_days") or 0,
    )

    hd = next((c for c in (promo.get("candidate_forecasts") or []) if c.get("ticker") == "HD"), {})
    xlk = next(
        (c for c in (promo.get("candidate_forecasts") or []) if c.get("ticker") == "XLK"), {}
    )
    bito = next(
        (c for c in (promo.get("candidate_forecasts") or []) if c.get("ticker") == "BITO"), {}
    )

    dependencies = {
        "hd_stability": {
            "status": hd.get("current_status", "OPEN_NEW"),
            "stability_30d": (hd.get("horizon_forecasts") or {})
            .get("30", {})
            .get("stability_probability"),
            "demotion_risk": (hd.get("horizon_forecasts") or {})
            .get("30", {})
            .get("demotion_probability"),
        },
        "xlk_stability": {
            "status": xlk.get("current_status", "OPEN_NEW"),
            "stability_30d": (xlk.get("horizon_forecasts") or {})
            .get("30", {})
            .get("stability_probability"),
            "demotion_risk": (xlk.get("horizon_forecasts") or {})
            .get("30", {})
            .get("demotion_probability"),
        },
        "bito_promotion": {
            "status": bito.get("current_status", "WATCH"),
            "promotion_prob_30d": bito.get("promotion_probability_30d"),
            "expected_date": bito.get("expected_promotion_date"),
            "blocks_green": True,
        },
        "governance_constraint": (readiness.get("bottleneck_analysis") or {}).get(
            "primary_bottleneck"
        ),
        "concentration_reduction": "Exit plan QQQ 100% → 60% required for YELLOW",
        "execution_readiness": (readiness.get("executive_report") or {}).get("execution_readiness"),
    }

    matrix_rows: List[Dict[str, Any]] = []
    outcomes = ["YELLOW", "GREEN", "ORANGE_BLOCKED", "FAILED"]
    for s in scenarios:
        yp = _to_float(s.get("yellow_probability"), 0) / 100.0
        gp = _to_float(s.get("green_probability"), 0) / 100.0
        sp = _to_float(s.get("scenario_probability"), 0) / 100.0
        fp = _to_float(s.get("failure_probability"), 0) / 100.0
        matrix_rows.extend(
            [
                {
                    "scenario": s["scenario"],
                    "outcome": "YELLOW",
                    "probability": _f(yp * sp * 100, 2),
                    "weighted_probability": _f(yp * sp * 100, 2),
                },
                {
                    "scenario": s["scenario"],
                    "outcome": "GREEN",
                    "probability": _f(gp * sp * 100, 2),
                    "weighted_probability": _f(gp * sp * 100, 2),
                },
                {
                    "scenario": s["scenario"],
                    "outcome": "ORANGE_BLOCKED",
                    "probability": _f(max(0, (1 - yp - gp)) * sp * 100, 2),
                    "weighted_probability": _f(max(0, (1 - yp - gp)) * sp * 100, 2),
                },
                {
                    "scenario": s["scenario"],
                    "outcome": "FAILED",
                    "probability": _f(fp * sp * 100, 2),
                    "weighted_probability": _f(fp * sp * 100, 2),
                },
            ]
        )

    forecast_summary = {
        "aggregate_yellow_probability": _f(yellow_agg * 100, 1),
        "aggregate_green_probability": _f(green_agg * 100, 1),
        "most_likely_scenario": ranked[0]["scenario"],
        "highest_probability_pct": ranked[0].get("scenario_probability"),
        "fastest_scenario": fastest["scenario"],
        "fastest_yellow_days": fastest.get("timeline_to_yellow_days"),
        "slowest_scenario": slowest["scenario"],
        "slowest_yellow_days": slowest.get("timeline_to_yellow_days"),
        "biggest_risk": "BITO promotion failure or HD/XLK demotion",
        "most_likely_failure_mode": "FAILED_RECOVERY_CASE — demotion or RED escalation",
        "forecast_confidence": _conf_label(0.65 + bito_prob * 0.2),
    }

    executive = {
        "report_type": "RECOVERY_PATH_SCENARIO_REPORT",
        "most_likely_scenario": ranked[0]["scenario"],
        "highest_probability_scenario": ranked[0]["scenario"],
        "fastest_recovery_scenario": fastest["scenario"],
        "slowest_recovery_scenario": slowest["scenario"],
        "probability_of_yellow": forecast_summary["aggregate_yellow_probability"],
        "probability_of_green": forecast_summary["aggregate_green_probability"],
        "forecast_confidence": forecast_summary["forecast_confidence"],
        "executive_recommendation": (
            f"Execute {ranked[0].get('plan_basis', 'BALANCED_EXIT_PLAN')} path — "
            f"{forecast_summary['aggregate_yellow_probability']}% aggregate YELLOW probability, "
            f"{forecast_summary['aggregate_green_probability']}% GREEN. "
            f"Monitor BITO promotion (30d prob {bito.get('promotion_probability_30d')}%). "
            f"Fastest YELLOW in ~{fastest.get('timeline_to_yellow_days')} days."
        ),
    }

    current_state = {
        "generated_at": ts,
        "current_cps": current_cps,
        "current_escalation": current_esc,
        "current_governance_status": current_gov,
        "hd_status": "OPEN_NEW",
        "xlk_status": "OPEN_NEW",
        "bito_status": "WATCH",
        "qqq_concentration_pct": cs.get("qqq_concentration_pct") or 100.0,
    }

    pd.DataFrame(scenarios).to_csv(OUT_CSV, index=False, encoding="utf-8")
    pd.DataFrame(matrix_rows).to_csv(OUT_MATRIX, index=False, encoding="utf-8")

    report = {
        "generated_at": ts,
        "executive_report": executive,
        "current_state": current_state,
        "scenarios": scenarios,
        "probability_matrix": {
            "by_scenario": matrix_rows,
            "aggregate": {
                "yellow": forecast_summary["aggregate_yellow_probability"],
                "green": forecast_summary["aggregate_green_probability"],
            },
        },
        "dependencies": dependencies,
        "forecast_summary": forecast_summary,
        "executive_recommendation": executive,
        "certification": {
            "status": "CERTIFIED",
            "safety": "forecasting-only — no trades, orders, governance, lifecycle, or authorization modified",
        },
        "disclaimer": (
            "Recovery path scenario forecasts are planning intelligence only — does not place trades "
            "or modify governance, lifecycle, execution, broker, authorization, or allocation rules."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_CSV.name}, {OUT_MATRIX.name}")
    _log(
        f"likely={ranked[0]['scenario']} yellow={forecast_summary['aggregate_yellow_probability']}% "
        f"green={forecast_summary['aggregate_green_probability']}%"
    )
    return report


if __name__ == "__main__":
    run()
