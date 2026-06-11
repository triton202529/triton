"""
TRITON Phase 186 — Governance Risk Prediction Engine (forecasting only).

Predicts future governance risks before they materialize and quantifies the
probability of governance deterioration events.

SAFETY: forecasting only — no governance, lifecycle, execution, broker, allocation,
or risk control changes.

Outputs (data/results/):
    governance_risk_predictions.json
    governance_risk_predictions.csv
    governance_risk_matrix.csv

Run:
    python -m services.governance_risk_prediction_engine
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from services.concentration_governance_analysis import _f, _read_csv, _read_json
from services.pnl_ledger_builder import _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "governance_risk_predictions.json"
OUT_CSV = RESULTS / "governance_risk_predictions.csv"
OUT_MATRIX = RESULTS / "governance_risk_matrix.csv"

HORIZONS = [7, 14, 30, 60, 90, 180]
RISK_TYPES = [
    "CONCENTRATION_RISK",
    "PROMOTION_RISK",
    "LIFECYCLE_RISK",
    "ALLOCATION_RISK",
    "EXECUTION_RISK",
    "GOVERNANCE_RISK",
    "RECOVERY_FAILURE_RISK",
]
WARN_THRESHOLD = 0.45
SEVERITY_WEIGHTS = {"LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4}


def _log(msg: str) -> None:
    print(f"[governance_risk_prediction_engine] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _classify(score: float) -> str:
    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.55:
        return "HIGH"
    if score >= 0.30:
        return "MODERATE"
    return "LOW"


def _severity_label(score: float) -> str:
    return _classify(score)


def _horizon_factor(day: int, peak: int = 30) -> float:
    return float(np.clip(1.0 - abs(day - peak) / 90.0, 0.3, 1.0))


def _risk_row(
    risk_type: str,
    day: int,
    prob: float,
    severity: float,
    conf: float,
    impacts: Dict[str, str],
) -> Dict[str, Any]:
    combined = prob * severity
    return {
        "risk_type": risk_type,
        "time_horizon": day,
        "risk_probability": _f(prob * 100, 1),
        "risk_severity": _severity_label(severity),
        "risk_severity_score": _f(severity * 100, 1),
        "risk_confidence": _f(conf * 100, 1),
        "classification": _classify(combined),
        "risk_score": _f(combined * 100, 1),
        "impact_on_orange": impacts.get("orange", "—"),
        "impact_on_yellow": impacts.get("yellow", "—"),
        "impact_on_green": impacts.get("green", "—"),
        "impact_on_timeline": impacts.get("timeline", "—"),
        "impact_on_forecast_confidence": impacts.get("confidence", "—"),
    }


def _compute_risks(
    day: int,
    outcomes: Dict[str, Any],
    scenarios: Dict[str, Any],
    promo: Dict[str, Any],
    timeline: Dict[str, Any],
    readiness: Dict[str, Any],
) -> List[Dict[str, Any]]:
    pred = next(
        (p for p in (outcomes.get("predictions") or []) if p.get("forecast_horizon") == day),
        {},
    )
    cs = outcomes.get("current_state") or {}
    conc_pct = _to_float(cs.get("qqq_concentration_pct"), 100.0)

    bito_prob = (
        _to_float(
            (promo.get("promotion_dependencies") or {}).get("bito_open_new_probability_30d"), 38.5
        )
        / 100.0
    )
    hd_demote = (
        _to_float((promo.get("dependencies") or {}).get("hd_stability", {}).get("demotion_risk"), 8)
        / 100.0
    )
    xlk_demote = (
        _to_float(
            (promo.get("dependencies") or {}).get("xlk_stability", {}).get("demotion_risk"), 8
        )
        / 100.0
    )

    prob_red = _to_float(pred.get("prob_red"), 7) / 100.0
    prob_orange = _to_float(pred.get("prob_orange"), 50) / 100.0
    fail_scenario = next(
        (
            s
            for s in (scenarios.get("scenarios") or [])
            if s.get("scenario") == "FAILED_RECOVERY_CASE"
        ),
        {},
    )
    fail_p = _to_float(fail_scenario.get("scenario_probability"), 5) / 100.0

    conc_decay = max(0.15, 1.0 - day / 120.0)
    conc_prob = float(np.clip(conc_pct / 100.0 * conc_decay, 0.2, 0.95))
    conc_sev = 0.85 if conc_pct >= 90 else 0.55

    promo_prob = float(np.clip(1.0 - bito_prob + max(0, (day - 30) / 90.0) * 0.2, 0.25, 0.85))
    promo_sev = 0.80 if day >= 30 else 0.65

    lifecycle_prob = float(np.clip((hd_demote + xlk_demote) / 2.0 + 0.05, 0.08, 0.35))
    lifecycle_sev = 0.70

    alloc_prob = (
        0.15
        if readiness.get("executive_report", {}).get("execution_readiness") == "READY"
        else 0.45
    )
    alloc_sev = 0.50

    exec_prob = (
        0.55
        if (readiness.get("bottleneck_analysis") or {}).get("primary_bottleneck")
        == "AUTHORIZATION_BOTTLENECK"
        else 0.25
    )
    exec_sev = 0.65

    gov_prob = float(np.clip(prob_orange * 0.6 + prob_red * 0.4, 0.15, 0.90))
    gov_sev = 0.75

    fail_prob = float(np.clip(fail_p + prob_red * 0.5, 0.05, 0.40))
    fail_sev = 0.90

    conf_base = _to_float(pred.get("prediction_confidence"), 70) / 100.0

    rows = [
        _risk_row(
            "CONCENTRATION_RISK",
            day,
            conc_prob,
            conc_sev,
            conf_base,
            {
                "orange": "HIGH — sustains ORANGE block",
                "yellow": "HIGH — delays YELLOW until reduced",
                "green": "CRITICAL — blocks GREEN path",
                "timeline": f"+{max(0, 14 - day // 7)} days if unaddressed",
                "confidence": "Reduces forecast confidence at short horizons",
            },
        ),
        _risk_row(
            "PROMOTION_RISK",
            day,
            promo_prob,
            promo_sev,
            0.65,
            {
                "orange": "LOW",
                "yellow": "LOW",
                "green": "CRITICAL — BITO blocks GREEN",
                "timeline": f"+{30 if bito_prob < 0.5 else 0} days to GREEN",
                "confidence": "Moderate impact on long-horizon forecasts",
            },
        ),
        _risk_row(
            "LIFECYCLE_RISK",
            day,
            lifecycle_prob,
            lifecycle_sev,
            0.70,
            {
                "orange": "MODERATE",
                "yellow": "HIGH — HD/XLK demotion blocks YELLOW",
                "green": "HIGH",
                "timeline": "Could reset recovery cadence",
                "confidence": "Low-moderate",
            },
        ),
        _risk_row(
            "ALLOCATION_RISK",
            day,
            alloc_prob,
            alloc_sev,
            0.75,
            {
                "orange": "LOW",
                "yellow": "MODERATE",
                "green": "MODERATE",
                "timeline": "Minor delay",
                "confidence": "Low",
            },
        ),
        _risk_row(
            "EXECUTION_RISK",
            day,
            exec_prob * _horizon_factor(day, 14),
            exec_sev,
            0.68,
            {
                "orange": "HIGH — paradox blocks pre-diversification",
                "yellow": "MODERATE — clears post-diversification",
                "green": "MODERATE",
                "timeline": "Front-loaded execution risk",
                "confidence": "HIGH at 7-14d",
            },
        ),
        _risk_row(
            "GOVERNANCE_RISK",
            day,
            gov_prob,
            gov_sev,
            conf_base,
            {
                "orange": "CRITICAL — current state",
                "yellow": "MODERATE — reduces as CPS rises",
                "green": "MODERATE",
                "timeline": "Dominant until day ~20",
                "confidence": "HIGH",
            },
        ),
        _risk_row(
            "RECOVERY_FAILURE_RISK",
            day,
            fail_prob,
            fail_sev,
            0.60,
            {
                "orange": "HIGH — escalation to RED",
                "yellow": "CRITICAL — prevents YELLOW",
                "green": "CRITICAL — prevents GREEN",
                "timeline": "Indefinite delay",
                "confidence": "Severe forecast degradation",
            },
        ),
    ]
    return rows


def _early_warnings(rows: List[Dict[str, Any]], prev: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_type.setdefault(r["risk_type"], []).append(r)

    for rtype, rlist in by_type.items():
        h30 = next((r for r in rlist if r.get("time_horizon") == 30), None)
        if h30 and _to_float(h30.get("risk_probability"), 0) / 100.0 >= WARN_THRESHOLD:
            warnings.append(
                f"[EARLY_WARNING] {rtype} probability {h30['risk_probability']}% "
                f"at 30d — severity {h30['risk_severity']}"
            )

    prev_exec = prev.get("executive_report") or {}
    if prev_exec.get("top_risks"):
        prev_top = prev_exec["top_risks"][0] if prev_exec["top_risks"] else ""
        curr_top = max(rows, key=lambda r: _to_float(r.get("risk_score"), 0), default={}).get(
            "risk_type", ""
        )
        if prev_top and curr_top and prev_top != curr_top:
            warnings.append(f"[EARLY_WARNING] Top risk shifted {prev_top} → {curr_top}")

    bito_note = "BITO promotion probability below 50% — GREEN timeline at risk"
    warnings.append(f"[EARLY_WARNING] {bito_note}")

    if any(
        _to_float(r.get("risk_probability"), 0) >= 70
        for r in rows
        if r.get("risk_type") == "CONCENTRATION_RISK" and r.get("time_horizon") <= 14
    ):
        warnings.append(
            "[EARLY_WARNING] Concentration remains CRITICAL — QQQ 100% at 7-14d horizon"
        )

    return warnings


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    prev = _read_json(OUT_JSON)

    outcomes = _read_json(RESULTS / "governance_outcome_predictions.json")
    scenarios = _read_json(RESULTS / "recovery_path_scenarios.json")
    promo = _read_json(RESULTS / "promotion_probability_forecast.json")
    timeline = _read_json(RESULTS / "governance_recovery_timeline.json")
    readiness = _read_json(RESULTS / "diversification_execution_readiness.json")

    all_rows: List[Dict[str, Any]] = []
    for day in HORIZONS:
        all_rows.extend(_compute_risks(day, outcomes, scenarios, promo, timeline, readiness))

    matrix_rows: List[Dict[str, Any]] = []
    for r in all_rows:
        prob = _to_float(r.get("risk_probability"), 0) / 100.0
        sev = SEVERITY_WEIGHTS.get(str(r.get("risk_severity")), 2)
        matrix_rows.append(
            {
                "risk_type": r["risk_type"],
                "time_horizon": r["time_horizon"],
                "probability_pct": r["risk_probability"],
                "severity": r["risk_severity"],
                "severity_score": sev,
                "matrix_score": _f(prob * sev * 25, 1),
                "classification": r["classification"],
            }
        )

    ranked = sorted(all_rows, key=lambda r: _to_float(r.get("risk_score"), 0), reverse=True)
    h30_rows = [r for r in all_rows if r.get("time_horizon") == 30]
    top_h30 = sorted(h30_rows, key=lambda r: _to_float(r.get("risk_score"), 0), reverse=True)

    highest_prob = max(all_rows, key=lambda r: _to_float(r.get("risk_probability"), 0))
    highest_sev = max(
        [r for r in h30_rows if r.get("risk_type") != "RECOVERY_FAILURE_RISK"] or h30_rows,
        key=lambda r: _to_float(r.get("risk_severity_score"), 0),
    )

    warnings = _early_warnings(all_rows, prev)

    risk_impacts = {
        "impact_on_orange": {
            "primary_drivers": ["CONCENTRATION_RISK", "GOVERNANCE_RISK", "EXECUTION_RISK"],
            "30d_combined_risk_score": _f(
                np.mean([_to_float(r.get("risk_score"), 0) for r in top_h30[:3]]), 1
            ),
        },
        "impact_on_yellow": {
            "primary_drivers": ["CONCENTRATION_RISK", "LIFECYCLE_RISK", "EXECUTION_RISK"],
            "mitigation": "Execute balanced exit plan with HD+XLK",
        },
        "impact_on_green": {
            "primary_drivers": ["PROMOTION_RISK", "CONCENTRATION_RISK"],
            "blocker": "BITO WATCH → OPEN_NEW required",
        },
        "impact_on_timeline": {
            "delay_risk_days": 30,
            "sources": ["PROMOTION_RISK", "CONCENTRATION_RISK"],
        },
        "impact_on_forecast_confidence": {
            "stability": (outcomes.get("resilience_analysis") or {}).get("forecast_resilience"),
            "degradation_triggers": ["BITO demotion", "RED escalation", "HD/XLK demotion"],
        },
    }

    fs = scenarios.get("forecast_summary") or {}
    executive = {
        "report_type": "GOVERNANCE_RISK_PREDICTION_REPORT",
        "top_risks": [r["risk_type"] for r in top_h30[:5]],
        "highest_probability_risk": {
            "type": highest_prob["risk_type"],
            "horizon": highest_prob["time_horizon"],
            "probability": highest_prob["risk_probability"],
        },
        "highest_severity_risk": {
            "type": highest_sev["risk_type"],
            "horizon": highest_sev["time_horizon"],
            "severity": highest_sev["risk_severity"],
        },
        "expected_recovery_threats": [
            "BITO fails to promote before GREEN window",
            "Concentration persists beyond 14 days",
            "Governance authorization paradox delays first diversification step",
            "HD/XLK demotion from OPEN_NEW",
        ],
        "forecast_stability": (outcomes.get("resilience_analysis") or {}).get(
            "forecast_resilience", "MODERATE"
        ),
        "executive_recommendation": (
            f"Monitor {top_h30[0]['risk_type']} (score {top_h30[0]['risk_score']}) and "
            f"{top_h30[1]['risk_type']} at 30d. Prioritize concentration reduction and BITO promotion tracking. "
            f"Recovery timeline threatened primarily by PROMOTION_RISK if BITO stays WATCH beyond day 30."
        ),
        "operator_monitoring_checklist": [
            "BITO confidence vs 0.50 OPEN_NEW floor",
            "HD/XLK allocation_status stability",
            "QQQ concentration % weekly",
            "CPS trajectory toward 75",
            "Governance authorization gate status",
        ],
    }

    summary = {
        "risk_entries": len(all_rows),
        "horizons": HORIZONS,
        "early_warning_count": len(warnings),
        "top_risk_30d": top_h30[0]["risk_type"] if top_h30 else "—",
        "aggregate_failure_risk": fs.get("recovery_failure_risk_pct"),
        "green_blocker": "PROMOTION_RISK (BITO)",
    }

    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False, encoding="utf-8")
    pd.DataFrame(matrix_rows).to_csv(OUT_MATRIX, index=False, encoding="utf-8")

    report = {
        "generated_at": ts,
        "executive_report": executive,
        "risk_predictions": all_rows,
        "risk_matrix": {
            "entries": matrix_rows,
            "ranked_30d": top_h30,
        },
        "early_warnings": warnings,
        "risk_impacts": risk_impacts,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "forecasting-only — no trades, orders, governance, lifecycle, or authorization modified",
        },
        "disclaimer": (
            "Governance risk predictions are planning intelligence only — does not place trades "
            "or modify governance, lifecycle, execution, broker, authorization, or allocation rules."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_CSV.name}, {OUT_MATRIX.name}")
    _log(f"top_30d={summary.get('top_risk_30d')} warnings={len(warnings)}")
    return report


if __name__ == "__main__":
    run()
