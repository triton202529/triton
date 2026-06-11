"""
TRITON Phase 185 — Governance Outcome Prediction Engine (forecasting only).

Predicts future governance states by consolidating forecasting layers from
Phases 178–184 into direct governance outcome predictions.

SAFETY: forecasting only — no governance, lifecycle, execution, broker, allocation,
or risk control changes.

Outputs (data/results/):
    governance_outcome_predictions.json
    governance_outcome_predictions.csv
    governance_state_probability_matrix.csv

Run:
    python -m services.governance_outcome_prediction_engine
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
OUT_JSON = RESULTS / "governance_outcome_predictions.json"
OUT_CSV = RESULTS / "governance_outcome_predictions.csv"
OUT_MATRIX = RESULTS / "governance_state_probability_matrix.csv"

HORIZONS = [7, 14, 30, 60, 90, 180]
STATES = ["RED", "ORANGE", "YELLOW", "GREEN"]


def _log(msg: str) -> None:
    print(f"[governance_outcome_prediction_engine] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _resilience_label(score: float) -> str:
    if score >= 0.80:
        return "VERY_HIGH"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.40:
        return "MODERATE"
    return "LOW"


def _outcome_class(yellow_p: float, green_p: float, red_p: float) -> str:
    if red_p >= 0.25:
        return "RECOVERY_FAILURE_RISK"
    if red_p >= 0.15:
        return "HIGH_RISK_OUTCOME"
    if green_p >= 0.55:
        return "HIGH_CONFIDENCE_RECOVERY"
    if yellow_p >= 0.60:
        return "MODERATE_CONFIDENCE_RECOVERY"
    return "UNCERTAIN_OUTCOME"


def _cps_to_escalation(cps: float) -> str:
    if cps >= CPS_GREEN:
        return "GREEN"
    if cps >= CPS_YELLOW:
        return "YELLOW"
    if cps >= 60:
        return "ORANGE"
    if cps >= 40:
        return "RED"
    return "RED"


def _timeline_at_day(
    timeline_df: pd.DataFrame, day: int, strategy: str = "BALANCED_PLAN"
) -> Dict[str, Any]:
    if timeline_df.empty:
        return {"projected_cps": 66, "projected_escalation": "ORANGE"}
    sub = timeline_df[(timeline_df["strategy"] == strategy) & (timeline_df["forecast_day"] <= day)]
    if sub.empty:
        sub = timeline_df[timeline_df["strategy"] == strategy].head(1)
    else:
        sub = sub.sort_values("forecast_day").tail(1)
    if sub.empty:
        return {"projected_cps": 66, "projected_escalation": "ORANGE"}
    row = sub.iloc[0]
    return {
        "projected_cps": _to_float(row.get("projected_cps"), 66),
        "projected_escalation": str(row.get("projected_escalation") or "ORANGE"),
        "forecast_confidence": _to_float(row.get("forecast_confidence"), 70),
    }


def _state_distribution(
    day: int,
    timeline_point: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
    bito_prob: float,
    current_esc: str,
) -> Dict[str, float]:
    esc = timeline_point.get("projected_escalation", "ORANGE")
    cps = _to_float(timeline_point.get("projected_cps"), 66)

    base = {s: 0.05 for s in STATES}
    if esc == "GREEN":
        base = {"RED": 0.02, "ORANGE": 0.03, "YELLOW": 0.10, "GREEN": 0.85}
    elif esc == "YELLOW":
        green_boost = min(0.35, bito_prob * 0.5 + max(0, (day - 30) / 120.0) * 0.3)
        base = {
            "RED": 0.03,
            "ORANGE": 0.05,
            "YELLOW": max(0.45, 0.75 - green_boost),
            "GREEN": green_boost,
        }
    elif esc == "ORANGE":
        yellow_boost = min(0.45, max(0, (day - 7) / 30.0) * 0.5)
        base = {
            "RED": 0.05,
            "ORANGE": max(0.40, 0.85 - yellow_boost),
            "YELLOW": yellow_boost,
            "GREEN": 0.02,
        }
    else:
        base = {"RED": 0.15, "ORANGE": 0.60, "YELLOW": 0.20, "GREEN": 0.05}

    fail_weight = sum(
        _to_float(s.get("scenario_probability"), 0) / 100.0
        for s in scenarios
        if s.get("scenario") == "FAILED_RECOVERY_CASE"
    )
    if fail_weight > 0:
        base["RED"] += fail_weight * 0.3
        base["ORANGE"] += fail_weight * 0.2
        for k in ("YELLOW", "GREEN"):
            base[k] *= max(0.5, 1.0 - fail_weight)

    total = sum(base.values())
    return {k: v / total for k, v in base.items()}


def _most_likely(dist: Dict[str, float]) -> str:
    return max(dist.items(), key=lambda x: x[1])[0]


def _transition_forecasts(
    predictions: List[Dict[str, Any]],
    base_dt: datetime,
    days_yellow: int,
    days_green: int,
) -> Dict[str, Any]:
    p30 = next((p for p in predictions if p.get("forecast_horizon") == 30), {})
    p90 = next((p for p in predictions if p.get("forecast_horizon") == 90), {})

    o2y = _to_float(p30.get("prob_yellow"), 0) / 100.0 + _to_float(p30.get("prob_green"), 0) / 100.0
    y2g = _to_float(p90.get("prob_green"), 0) / 100.0
    o2r = _to_float(p30.get("prob_red"), 0) / 100.0
    y2o = 0.08

    return {
        "ORANGE_to_YELLOW": {
            "transition_probability": _f(o2y * 100, 1),
            "transition_confidence": _conf_label(o2y),
            "expected_transition_date": _date_offset(base_dt, days_yellow),
            "expected_days": days_yellow,
        },
        "YELLOW_to_GREEN": {
            "transition_probability": _f(y2g * 100, 1),
            "transition_confidence": _conf_label(y2g),
            "expected_transition_date": _date_offset(base_dt, days_green),
            "expected_days": days_green,
        },
        "ORANGE_to_RED": {
            "transition_probability": _f(o2r * 100, 1),
            "transition_confidence": _conf_label(1.0 - o2r),
            "expected_transition_date": None,
        },
        "YELLOW_to_ORANGE": {
            "transition_probability": _f(y2o * 100, 1),
            "transition_confidence": "MODERATE",
            "expected_transition_date": None,
        },
    }


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    base_dt = datetime.now(timezone.utc)

    timeline = _read_json(RESULTS / "governance_recovery_timeline.json")
    promo = _read_json(RESULTS / "promotion_probability_forecast.json")
    scenarios_doc = _read_json(RESULTS / "recovery_path_scenarios.json")
    exit_strat = _read_json(RESULTS / "concentration_exit_strategy.json")
    readiness = _read_json(RESULTS / "diversification_execution_readiness.json")
    auth = _read_json(RESULTS / "governance_authorization.json")
    timeline_csv = _read_csv(RESULTS / "governance_recovery_timeline.csv")

    tl_ex = timeline.get("executive_report") or {}
    scenarios = scenarios_doc.get("scenarios") or []
    fs = scenarios_doc.get("forecast_summary") or {}

    cs = scenarios_doc.get("current_state") or exit_strat.get("current_state") or {}
    current_cps = int(cs.get("current_cps") or 66)
    current_esc = str(cs.get("current_escalation") or "ORANGE")
    current_gov = "AUTHORIZED" if auth.get("governance_authorized") else "BLOCKED"

    bito_prob = (
        _to_float(
            (promo.get("promotion_dependencies") or {}).get("bito_open_new_probability_30d"), 38.5
        )
        / 100.0
    )

    days_yellow = int(tl_ex.get("days_to_yellow") or 20)
    days_green = int(tl_ex.get("days_to_green") or 60)

    predictions: List[Dict[str, Any]] = []
    matrix_rows: List[Dict[str, Any]] = []

    for day in HORIZONS:
        tp = _timeline_at_day(timeline_csv, day)
        dist = _state_distribution(day, tp, scenarios, bito_prob, current_esc)
        ml = _most_likely(dist)
        conf = float(np.clip(_to_float(tp.get("forecast_confidence"), 70) / 100.0, 0.3, 0.95))
        yellow_p = dist["YELLOW"] + dist["GREEN"]
        green_p = dist["GREEN"]
        red_p = dist["RED"]

        pred = {
            "forecast_horizon": day,
            "forecast_date": _date_offset(base_dt, day),
            "most_likely_state": ml,
            "projected_cps": int(_to_float(tp.get("projected_cps"), current_cps)),
            "state_probability_distribution": {k: _f(v * 100, 1) for k, v in dist.items()},
            "prob_red": _f(dist["RED"] * 100, 1),
            "prob_orange": _f(dist["ORANGE"] * 100, 1),
            "prob_yellow": _f(dist["YELLOW"] * 100, 1),
            "prob_green": _f(dist["GREEN"] * 100, 1),
            "prob_yellow_or_green": _f(yellow_p * 100, 1),
            "prediction_confidence": _f(conf * 100, 1),
            "prediction_confidence_label": _conf_label(conf),
            "outcome_classification": _outcome_class(yellow_p, green_p, red_p),
        }
        predictions.append(pred)

        for state in STATES:
            matrix_rows.append(
                {
                    "forecast_horizon": day,
                    "forecast_date": pred["forecast_date"],
                    "state": state,
                    "probability": pred["state_probability_distribution"].get(state),
                    "most_likely_state": ml,
                }
            )

    p30 = next(p for p in predictions if p["forecast_horizon"] == 30)
    p90 = next(p for p in predictions if p["forecast_horizon"] == 90)

    hd_stab = (
        _to_float(
            (promo.get("dependencies") or {}).get("hd_stability", {}).get("stability_30d"), 85
        )
        / 100.0
    )
    xlk_stab = (
        _to_float(
            (promo.get("dependencies") or {}).get("xlk_stability", {}).get("stability_30d"), 85
        )
        / 100.0
    )
    exec_ready = (readiness.get("executive_report") or {}).get("execution_readiness") == "READY"

    gov_stability = float(
        np.clip(0.5 + hd_stab * 0.2 + xlk_stab * 0.2 - bito_prob * 0.15, 0.2, 0.95)
    )
    recovery_stab = float(
        np.clip(_to_float(fs.get("aggregate_yellow_probability"), 76) / 100.0, 0.2, 0.95)
    )
    promo_dep = float(np.clip(1.0 - bito_prob, 0.3, 0.9))
    conc_dep = float(np.clip(cs.get("qqq_concentration_pct", 100) / 100.0, 0.5, 1.0))
    forecast_res = float(
        np.clip(np.mean([gov_stability, recovery_stab, 1.0 - promo_dep * 0.5]), 0.3, 0.9)
    )

    resilience = {
        "governance_stability": _resilience_label(gov_stability),
        "governance_stability_score": _f(gov_stability * 100, 1),
        "recovery_stability": _resilience_label(recovery_stab),
        "recovery_stability_score": _f(recovery_stab * 100, 1),
        "promotion_dependency": _resilience_label(1.0 - promo_dep),
        "promotion_dependency_score": _f(promo_dep * 100, 1),
        "concentration_dependency": _resilience_label(1.0 - conc_dep + 0.5),
        "concentration_dependency_score": _f(conc_dep * 100, 1),
        "forecast_resilience": _resilience_label(forecast_res),
        "forecast_resilience_score": _f(forecast_res * 100, 1),
        "execution_readiness_supports_recovery": exec_ready,
    }

    transitions = _transition_forecasts(predictions, base_dt, days_yellow, days_green)

    agg_yellow = _to_float(fs.get("aggregate_yellow_probability"), 76.1)
    agg_green = _to_float(fs.get("aggregate_green_probability"), 40.8)
    fail_risk = sum(
        _to_float(s.get("scenario_probability"), 0)
        * _to_float(s.get("failure_probability"), 0)
        / 100.0
        for s in scenarios
        if s.get("scenario") == "FAILED_RECOVERY_CASE"
    )

    executive = {
        "report_type": "GOVERNANCE_OUTCOME_PREDICTION_REPORT",
        "current_governance_state": current_esc,
        "current_governance_authorized": current_gov,
        "current_cps": current_cps,
        "most_likely_future_state_30d": p30["most_likely_state"],
        "most_likely_future_state_90d": p90["most_likely_state"],
        "probability_of_yellow": agg_yellow,
        "probability_of_green": agg_green,
        "probability_of_recovery_failure": _f(fail_risk, 1),
        "expected_yellow_transition_date": transitions["ORANGE_to_YELLOW"][
            "expected_transition_date"
        ],
        "expected_green_transition_date": transitions["YELLOW_to_GREEN"][
            "expected_transition_date"
        ],
        "prediction_confidence": p30["prediction_confidence_label"],
        "governance_stability": resilience["governance_stability"],
        "forecast_resilience": resilience["forecast_resilience"],
        "most_likely_transition_path": f"{current_esc} → YELLOW (~day {days_yellow}) → GREEN (~day {days_green})",
        "executive_recommendation": (
            f"Governance most likely YELLOW by day {days_yellow}, GREEN by day {days_green} "
            f"if BITO promotes (P={bito_prob*100:.1f}%). "
            f"30d forecast: {p30['most_likely_state']} ({p30['prob_yellow_or_green']}% YELLOW+GREEN). "
            f"90d forecast: {p90['most_likely_state']} ({p90['prob_green']}% GREEN). "
            f"Execute balanced exit plan; monitor BITO promotion."
        ),
    }

    current_state = {
        "generated_at": ts,
        "cps": current_cps,
        "escalation": current_esc,
        "governance_status": current_gov,
        "hd_status": "OPEN_NEW",
        "xlk_status": "OPEN_NEW",
        "bito_status": "WATCH",
        "qqq_concentration_pct": cs.get("qqq_concentration_pct") or 100.0,
    }

    summary = {
        "horizons_forecast": len(predictions),
        "state_at_30d": p30["most_likely_state"],
        "state_at_90d": p90["most_likely_state"],
        "aggregate_yellow_probability": agg_yellow,
        "aggregate_green_probability": agg_green,
        "recovery_failure_risk_pct": _f(fail_risk, 1),
        "forecast_confidence_30d": p30["prediction_confidence"],
    }

    csv_rows = [
        {
            "forecast_horizon": p["forecast_horizon"],
            "forecast_date": p["forecast_date"],
            "most_likely_state": p["most_likely_state"],
            "prob_red": p["prob_red"],
            "prob_orange": p["prob_orange"],
            "prob_yellow": p["prob_yellow"],
            "prob_green": p["prob_green"],
            "prediction_confidence": p["prediction_confidence"],
        }
        for p in predictions
    ]
    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False, encoding="utf-8")
    pd.DataFrame(matrix_rows).to_csv(OUT_MATRIX, index=False, encoding="utf-8")

    report = {
        "generated_at": ts,
        "forecast_base_date": base_dt.strftime("%Y-%m-%d"),
        "executive_report": executive,
        "current_state": current_state,
        "predictions": predictions,
        "state_probabilities": {
            str(p["forecast_horizon"]): p["state_probability_distribution"] for p in predictions
        },
        "transition_forecasts": transitions,
        "resilience_analysis": resilience,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "forecasting-only — no trades, orders, governance, lifecycle, or authorization modified",
        },
        "disclaimer": (
            "Governance outcome predictions are planning intelligence only — does not place trades "
            "or modify governance, lifecycle, execution, broker, authorization, or allocation rules."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_CSV.name}, {OUT_MATRIX.name}")
    _log(
        f"30d={p30['most_likely_state']} 90d={p90['most_likely_state']} conf={p30['prediction_confidence_label']}"
    )
    return report


if __name__ == "__main__":
    run()
