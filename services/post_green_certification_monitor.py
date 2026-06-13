"""
TRITON Phase 206 — Post-GREEN Institutional Stability & Continuous Certification.

Permanent monitoring framework for GREEN certification durability and regression
early warning. Read-only — no trades, orders, or rule changes.

Outputs (data/results/):
    post_green_certification_status.json
    post_green_stability_history.csv
    post_green_regression_risks.csv
    post_green_certification_events.csv

Run:
    python -m services.post_green_certification_monitor
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from services.concentration_governance_analysis import (
    CPI_WEIGHTS,
    _concentration_score,
    _escalation_from_cps,
    _f,
    _load_positions,
    _read_csv,
    _read_json,
    _sector_weights,
)
from services.concentration_recovery_simulator import (
    CPS_GREEN,
    CPS_YELLOW,
    _simulate_scenario,
)
from services.diversification_feasibility_analysis import MIN_OPEN_NEW_CONFIDENCE
from services.pnl_ledger_builder import _norm_sym, _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "post_green_certification_status.json"
OUT_HISTORY = RESULTS / "post_green_stability_history.csv"
OUT_REGRESSION = RESULTS / "post_green_regression_risks.csv"
OUT_EVENTS = RESULTS / "post_green_certification_events.csv"

MAX_CONCENTRATION_PCT = 35.0
MIN_CPS_BUFFER = 5
TARGET_GREEN_WEIGHTS = {"QQQ": 25.0, "HD": 25.0, "XLK": 25.0, "BITO": 25.0}


def _log(msg: str) -> None:
    print(f"[post_green_certification_monitor] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _stability_band(score: float) -> str:
    if score >= 95:
        return "INSTITUTIONAL_EXCELLENCE"
    if score >= 85:
        return "GREEN_STABLE"
    if score >= 70:
        return "GREEN_WARNING"
    if score >= 50:
        return "RECOVERY_REQUIRED"
    return "GOVERNANCE_RISK"


def _warning_class(cps: int, qqq_pct: float, escalation: str) -> str:
    if escalation in ("RED", "CRITICAL") or cps < 40:
        return "CRITICAL_REGRESSION"
    if escalation == "ORANGE" or qqq_pct > 80:
        return "ORANGE_RISK"
    if cps < CPS_YELLOW or qqq_pct > 65:
        return "YELLOW_RISK"
    if cps < CPS_GREEN + MIN_CPS_BUFFER or qqq_pct > MAX_CONCENTRATION_PCT:
        return "GREEN_AT_RISK"
    if cps >= CPS_GREEN:
        return "GREEN_STABLE"
    return "GREEN_AT_RISK"


def _load_live_state() -> Dict[str, Any]:
    cpi = _read_json(RESULTS / "capital_preservation_intelligence.json")
    cpe = _read_json(RESULTS / "capital_preservation_escalation.json")
    auth = _read_json(RESULTS / "governance_authorization.json")
    daily = _read_json(RESULTS / "governance_daily_evidence.json")
    stability = _read_json(RESULTS / "execution_readiness_stability.json")
    green205 = _read_json(RESULTS / "green_authorization_strategy.json")
    sim204 = _read_json(RESULTS / "governance_recovery_simulation.json")
    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    lifecycle = _read_csv(RESULTS / "signal_lifecycle_effective.csv")

    positions = _load_positions()
    _, largest_pct, _ = _concentration_score(positions)
    total_mv = sum(p["market_value"] for p in positions) or 1438.0

    cp = (daily.get("daily_evidence") or {}).get("capital_preservation") or {}
    cps = int(cp.get("cps") or cpi.get("capital_preservation_score") or 66)
    qqq = float(
        (daily.get("daily_evidence") or {})
        .get("portfolio_recovery", {})
        .get("qqq_concentration_pct")
        or largest_pct
        or 100
    )
    escalation = str(cpe.get("escalation_state") or _escalation_from_cps(cps))
    gov_auth = bool(auth.get("governance_authorized"))

    expected_green = (green205.get("executive_recommendation") or {}).get(
        "expected_green_timeline"
    ) or (green205.get("summary") or {}).get("expected_green_date")
    yellow_date = (sim204.get("primary_scenario") or {}).get("summary", {}).get("yellow_date") or (
        green205.get("baseline") or {}
    ).get("yellow_date")

    open_new = recommendations[
        recommendations["recommended_action"].astype(str).str.upper() == "OPEN_NEW"
    ]
    buy_effective = 0
    if not lifecycle.empty:
        buy_effective = int(
            lifecycle[
                lifecycle["lifecycle_action"].astype(str).str.upper().isin(["BUY", "ADD"])
            ].shape[0]
        )

    return {
        "cps": cps,
        "qqq_pct": qqq,
        "escalation": escalation,
        "governance_authorized": gov_auth,
        "positions": positions,
        "total_mv": total_mv,
        "open_new_count": len(open_new),
        "buy_effective_count": buy_effective,
        "stability_doc": stability,
        "green205": green205,
        "expected_green_date": expected_green,
        "yellow_date": yellow_date,
        "recommendations": recommendations,
        "lifecycle": lifecycle,
        "base_components": dict(cpi.get("component_scores") or {}),
    }


def _certification_status(state: Dict[str, Any]) -> Dict[str, Any]:
    cps = state["cps"]
    qqq = state["qqq_pct"]
    esc = state["escalation"]
    authorized = state["governance_authorized"]

    is_green = cps >= CPS_GREEN and esc == "GREEN" and authorized
    is_yellow = cps >= CPS_YELLOW and esc in ("YELLOW", "GREEN")

    if is_green:
        status = "GREEN_CERTIFIED"
        label = "AUTHORIZED"
    elif is_yellow:
        status = "YELLOW_CERTIFIED"
        label = "CONDITIONALLY_AUTHORIZED"
    elif cps >= CPS_YELLOW - 3:
        status = "GREEN_CERTIFICATION_IN_PROGRESS"
        label = "APPROACHING_GREEN"
    else:
        status = "PRE_GREEN_MONITORING"
        label = "RECOVERY_IN_PROGRESS"

    buffer = cps - CPS_GREEN
    return {
        "certification_status": status,
        "authorization_label": label,
        "cps": cps,
        "cps_buffer_above_green": buffer,
        "minimum_buffer_required": MIN_CPS_BUFFER,
        "buffer_adequate": buffer >= MIN_CPS_BUFFER if is_green else False,
        "qqq_concentration_pct": _f(qqq, 2),
        "concentration_within_limit": qqq <= MAX_CONCENTRATION_PCT,
        "escalation_state": esc,
        "projected_green_date": state.get("expected_green_date"),
        "projected_yellow_date": state.get("yellow_date"),
    }


def _integrity_monitoring(state: Dict[str, Any]) -> Dict[str, Any]:
    cps = state["cps"]
    qqq = state["qqq_pct"]
    stab = state["stability_doc"]
    recs = state["recommendations"]
    lifecycle = state["lifecycle"]

    hist = _read_csv(RESULTS / "governance_daily_evidence_history.csv")
    cps_trend = "STABLE"
    cps_volatility = 0.0
    if not hist.empty and "cps" in hist.columns:
        vals = pd.to_numeric(hist["cps"], errors="coerce").dropna()
        if len(vals) >= 2:
            cps_trend = "IMPROVING" if vals.iloc[-1] > vals.iloc[0] else "DECLINING"
            cps_volatility = float(vals.std())

    sectors = _sector_weights(state["positions"])
    tech_pct = sectors.get("Tech", 0.0)
    div_quality = min(len(state["positions"]), 4) / 4.0 * 100.0

    open_new_syms = (
        recs[recs["recommended_action"].astype(str).str.upper() == "OPEN_NEW"]["ticker"]
        .map(_norm_sym)
        .tolist()
        if not recs.empty
        else []
    )
    bito_row = recs[recs["ticker"].astype(str).str.upper() == "BITO"]
    bito_watch = (
        not bito_row.empty
        and str(bito_row.iloc[0].get("recommended_action", "")).upper() != "OPEN_NEW"
    )

    wd = stab.get("watchdog_reliability") or {}
    loop_health = (stab.get("certification_impact") or {}).get(
        "operational_readiness_status", "UNKNOWN"
    )

    return {
        "portfolio_integrity": {
            "cps": cps,
            "cps_trend": cps_trend,
            "cps_volatility": _f(cps_volatility, 2),
            "concentration_pct": _f(qqq, 2),
            "diversification_quality_score": _f(div_quality, 1),
            "sector_balance_tech_pct": _f(tech_pct, 1),
            "position_count": len(state["positions"]),
        },
        "lifecycle_integrity": {
            "buy_effective_count": state["buy_effective_count"],
            "open_new_count": state["open_new_count"],
            "open_new_symbols": open_new_syms[:8],
            "bito_watch_status": bito_watch,
            "replacement_candidates_available": len(open_new_syms) >= 2,
            "deterioration_risk": (
                "ELEVATED" if bito_watch and state["open_new_count"] < 2 else "LOW"
            ),
        },
        "operational_integrity": {
            "continuous_loop_health": loop_health,
            "watchdog_stability_score": wd.get("watchdog_stability_score"),
            "false_positive_rate_pct": _f(
                _to_float(wd.get("false_positive_probability"), 0) * 100, 1
            ),
            "execution_readiness_stability": (stab.get("stability_score") or {}).get(
                "stability_score"
            ),
            "broker_reliability": (stab.get("stability_score") or {})
            .get("factors", {})
            .get("broker_connectivity"),
        },
        "governance_integrity": {
            "governance_authorized": state["governance_authorized"],
            "escalation_state": state["escalation"],
            "risk_trend": cps_trend,
            "evidence_quality_pct": (
                (state["green205"].get("executive_recommendation") or {})
                .get("executive_questions", {})
                .get("unmet_requirements_count")
            ),
            "institutional_confidence": "IMPROVING" if cps_trend == "IMPROVING" else "STABLE",
        },
    }


def _regression_warnings(state: Dict[str, Any], integrity: Dict[str, Any]) -> List[Dict[str, Any]]:
    cps = state["cps"]
    qqq = state["qqq_pct"]
    esc = state["escalation"]
    warnings: List[Dict[str, Any]] = []

    if cps < CPS_GREEN + MIN_CPS_BUFFER:
        gap = CPS_GREEN - cps
        warnings.append(
            {
                "warning_type": "YELLOW_RISK" if gap <= 15 else "ORANGE_RISK",
                "signal": "CPS below GREEN safety buffer",
                "probability_pct": _f(min(85, 40 + gap * 3), 1),
                "time_to_regression_days": max(7, int(gap * 2)),
                "severity": "HIGH" if gap > 20 else "MODERATE",
                "confidence": "HIGH",
                "preventative_evidence": "Complete YELLOW recovery; sustain CPS ≥75 for 14 days",
                "current_classification": _warning_class(cps, qqq, esc),
            }
        )

    if qqq > MAX_CONCENTRATION_PCT:
        warnings.append(
            {
                "warning_type": "ORANGE_RISK" if qqq > 65 else "YELLOW_RISK",
                "signal": f"Concentration {qqq:.1f}% exceeds GREEN limit {MAX_CONCENTRATION_PCT}%",
                "probability_pct": _f(min(90, 30 + (qqq - MAX_CONCENTRATION_PCT) * 2), 1),
                "time_to_regression_days": max(5, int((qqq - 60) / 5)),
                "severity": "CRITICAL" if qqq > 80 else "HIGH",
                "confidence": "VERY_HIGH",
                "preventative_evidence": "Execute concentration trim cadence per exit strategy",
                "current_classification": _warning_class(cps, qqq, esc),
            }
        )

    li = integrity.get("lifecycle_integrity") or {}
    if li.get("deterioration_risk") == "ELEVATED":
        warnings.append(
            {
                "warning_type": "YELLOW_RISK",
                "signal": "Lifecycle candidate quality declining",
                "probability_pct": 35.0,
                "time_to_regression_days": 21,
                "severity": "MODERATE",
                "confidence": "MODERATE",
                "preventative_evidence": "Monitor HD/XLK/BITO lifecycle daily; activate PATH_B alternatives",
                "current_classification": "GREEN_AT_RISK",
            }
        )

    oi = integrity.get("operational_integrity") or {}
    wd_score = _to_float(oi.get("watchdog_stability_score"), 50)
    if wd_score < 50:
        warnings.append(
            {
                "warning_type": "YELLOW_RISK",
                "signal": "Watchdog reliability degraded",
                "probability_pct": 28.0,
                "time_to_regression_days": 14,
                "severity": "MODERATE",
                "confidence": "HIGH",
                "preventative_evidence": "Resolve false-positive watchdog pattern; verify loop health",
                "current_classification": "GREEN_AT_RISK",
            }
        )

    if not warnings:
        warnings.append(
            {
                "warning_type": "GREEN_STABLE",
                "signal": "No active regression signals — monitoring baseline established",
                "probability_pct": 8.0,
                "time_to_regression_days": None,
                "severity": "LOW",
                "confidence": "MODERATE",
                "preventative_evidence": "Continue daily certification evidence collection",
                "current_classification": _warning_class(cps, qqq, esc),
            }
        )

    return warnings


def _institutional_stability_score(
    state: Dict[str, Any],
    integrity: Dict[str, Any],
    warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pi = integrity.get("portfolio_integrity") or {}
    li = integrity.get("lifecycle_integrity") or {}
    oi = integrity.get("operational_integrity") or {}
    gi = integrity.get("governance_integrity") or {}

    cps = state["cps"]
    cps_res = (
        min(cps / CPS_GREEN, 1.0) * 100 if cps < CPS_GREEN else min(100, 80 + (cps - CPS_GREEN) * 4)
    )
    conc_disc = max(0, 100 - max(0, state["qqq_pct"] - MAX_CONCENTRATION_PCT) * 2)
    adapt = 70.0 if li.get("replacement_candidates_available") else 40.0
    replace = min(li.get("open_new_count", 0) / 4.0, 1.0) * 100
    ops = _to_float(oi.get("execution_readiness_stability"), 64)
    gov = 90.0 if gi.get("governance_authorized") else max(30, cps / CPS_GREEN * 60)

    score = float(
        np.clip(
            cps_res * 0.25
            + conc_disc * 0.20
            + adapt * 0.15
            + replace * 0.10
            + ops * 0.15
            + gov * 0.15,
            0,
            100,
        )
    )
    max_warn_prob = max(_to_float(w.get("probability_pct"), 0) for w in warnings)

    return {
        "institutional_stability_score": _f(score, 1),
        "classification": _stability_band(score),
        "components": {
            "cps_resilience": _f(cps_res, 1),
            "concentration_discipline": _f(conc_disc, 1),
            "portfolio_adaptability": _f(adapt, 1),
            "replacement_readiness": _f(replace, 1),
            "operational_stability": _f(ops, 1),
            "governance_confidence": _f(gov, 1),
        },
        "regression_probability_pct": _f(max_warn_prob, 1),
    }


def _stress_scenarios(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    base_components = state["base_components"]
    for k in CPI_WEIGHTS:
        base_components.setdefault(k, 70)
    total_mv = state["total_mv"]
    cps = state["cps"]
    _, largest, _ = _concentration_score(state["positions"])

    scenarios = []

    # A — Market shock
    shock_cps = max(CPS_YELLOW - 5, cps - 8)
    scenarios.append(
        {
            "scenario": "A — Market Shock",
            "description": "Major drawdown + volatility spike",
            "cps_impact": -8,
            "projected_cps": shock_cps,
            "governance_impact": _escalation_from_cps(shock_cps),
            "recovery_difficulty": "HIGH",
            "days_to_stabilization": 14,
        }
    )

    # B — Concentration drift
    drift = dict(TARGET_GREEN_WEIGHTS)
    drift["QQQ"] = min(55, drift.get("QQQ", 25) + 20)
    cut = 20 / 3
    for k in list(drift.keys()):
        if k != "QQQ":
            drift[k] = max(5, drift[k] - cut)
    sim = _simulate_scenario(drift, total_mv, base_components, largest, cps)
    drift_cps = int(sim.get("projected_cps") or cps - 6)
    scenarios.append(
        {
            "scenario": "B — Concentration Drift",
            "description": "QQQ grows +20% without trim",
            "cps_impact": drift_cps - cps,
            "projected_cps": drift_cps,
            "governance_impact": sim.get("escalation_level"),
            "recovery_difficulty": "MODERATE",
            "days_to_stabilization": 10,
        }
    )

    # C — Candidate collapse
    scenarios.append(
        {
            "scenario": "C — Candidate Collapse",
            "description": "HD/XLK lose BUY-effective; OPEN_NEW pool empty",
            "cps_impact": -6,
            "projected_cps": max(CPS_YELLOW - 3, cps - 6),
            "governance_impact": "YELLOW",
            "recovery_difficulty": "HIGH",
            "days_to_stabilization": 21,
        }
    )

    # D — Operational failure
    scenarios.append(
        {
            "scenario": "D — Operational Failure",
            "description": "Loop interruption + watchdog anomaly + broker stress",
            "cps_impact": -4,
            "projected_cps": max(CPS_YELLOW, cps - 4),
            "governance_impact": "YELLOW" if cps - 4 >= CPS_YELLOW else "ORANGE",
            "recovery_difficulty": "MODERATE",
            "days_to_stabilization": 7,
        }
    )

    return scenarios


def _governance_memory(
    cert: Dict[str, Any],
    state: Dict[str, Any],
    warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    memory = _read_json(RESULTS / "institutional_memory.json")
    events_file = OUT_EVENTS
    prior_events = _read_csv(events_file)

    cert_status = cert.get("certification_status")
    is_green = cert_status == "GREEN_CERTIFIED"

    green_cert_date = None
    days_green = 0
    if is_green:
        green_cert_date = _today()
        days_green = 1
    elif not prior_events.empty and "event_type" in prior_events.columns:
        green_rows = prior_events[prior_events["event_type"] == "GREEN_CERTIFIED"]
        if not green_rows.empty:
            green_cert_date = str(green_rows.iloc[-1].get("date", ""))
            try:
                d0 = datetime.strptime(green_cert_date[:10], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                days_green = (datetime.now(timezone.utc) - d0).days
            except ValueError:
                days_green = 0

    longevity = {
        "30_day_durability_pct": _f(min(100, days_green / 30 * 100) if is_green else 0, 1),
        "90_day_durability_pct": _f(min(100, days_green / 90 * 100) if is_green else 0, 1),
        "365_day_durability_pct": _f(min(100, days_green / 365 * 100) if is_green else 0, 1),
    }
    longevity_score = float(
        np.mean(
            [
                longevity["30_day_durability_pct"] or 0,
                longevity["90_day_durability_pct"] or 0,
                longevity["365_day_durability_pct"] or 0,
            ]
        )
    )

    lessons = [
        "Concentration discipline is the primary GREEN maintenance lever",
        "BITO/alternative pipeline must stay validated even after certification",
        "14-day YELLOW stabilization evidence required before GREEN pursuit",
        "Watchdog false positives do not reduce ops score but require monitoring",
    ]
    if state["qqq_pct"] > 80:
        lessons.insert(
            0, "CURRENT: 100% QQQ — regression risk elevated until diversification executes"
        )

    hist_events = memory.get("events") or []
    regression_events = [e for e in hist_events if "ORANGE" in str(e.get("detail", "")).upper()][:5]

    return {
        "green_certification_date": green_cert_date,
        "days_continuously_green": days_green,
        "certification_status": cert_status,
        "historical_regression_events": len(regression_events),
        "near_miss_incidents": len(
            [w for w in warnings if w.get("severity") in ("HIGH", "CRITICAL")]
        ),
        "institutional_lessons": lessons,
        "green_longevity_score": _f(longevity_score, 1),
        "longevity_metrics": longevity,
        "projected_certification_date": state.get("expected_green_date"),
    }


def _certification_events(
    cert: Dict[str, Any],
    stability: Dict[str, Any],
    warnings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = [
        {
            "date": _today(),
            "timestamp": _now_iso(),
            "event_type": cert.get("certification_status"),
            "cps": cert.get("cps"),
            "escalation": cert.get("escalation_state"),
            "stability_score": stability.get("institutional_stability_score"),
            "summary": f"Daily certification scan — {stability.get('classification')}",
        }
    ]
    for w in warnings:
        if w.get("warning_type") != "GREEN_STABLE":
            events.append(
                {
                    "date": _today(),
                    "timestamp": _now_iso(),
                    "event_type": "REGRESSION_WARNING",
                    "cps": cert.get("cps"),
                    "escalation": cert.get("escalation_state"),
                    "stability_score": stability.get("institutional_stability_score"),
                    "summary": f"{w.get('warning_type')}: {w.get('signal')}",
                }
            )
    return events


def _update_history(stability: Dict[str, Any], cert: Dict[str, Any]) -> None:
    row = {
        "timestamp": _now_iso(),
        "date": _today(),
        "cps": cert.get("cps"),
        "qqq_concentration_pct": cert.get("qqq_concentration_pct"),
        "escalation_state": cert.get("escalation_state"),
        "certification_status": cert.get("certification_status"),
        "institutional_stability_score": stability.get("institutional_stability_score"),
        "stability_classification": stability.get("classification"),
        "regression_probability_pct": stability.get("regression_probability_pct"),
        "cps_buffer": cert.get("cps_buffer_above_green"),
    }
    existing = _read_csv(OUT_HISTORY)
    new_df = pd.DataFrame([row])
    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined.to_csv(OUT_HISTORY, index=False, encoding="utf-8")


def _executive_report(
    cert: Dict[str, Any],
    stability: Dict[str, Any],
    warnings: List[Dict[str, Any]],
    integrity: Dict[str, Any],
    memory: Dict[str, Any],
    stress: List[Dict[str, Any]],
) -> Dict[str, Any]:
    top_risks = [w.get("signal") for w in warnings if w.get("warning_type") != "GREEN_STABLE"][:4]
    if not top_risks:
        top_risks = ["Concentration at 100% — primary pre-GREEN risk"]

    return {
        "report_type": "POST_GREEN_INSTITUTIONAL_STABILITY_REPORT",
        "current_certification_status": cert.get("certification_status"),
        "authorization_label": cert.get("authorization_label"),
        "institutional_stability_score": stability.get("institutional_stability_score"),
        "stability_classification": stability.get("classification"),
        "days_since_green_certification": memory.get("days_continuously_green"),
        "projected_green_certification_date": memory.get("projected_certification_date"),
        "regression_probability_pct": stability.get("regression_probability_pct"),
        "top_emerging_risks": top_risks,
        "portfolio_health_trend": (integrity.get("portfolio_integrity") or {}).get("cps_trend"),
        "operational_health_trend": (integrity.get("operational_integrity") or {}).get(
            "continuous_loop_health"
        ),
        "recommended_preventative_actions": [
            "Execute QQQ concentration reduction per Phase 204 simulation",
            "Sustain daily CPS + concentration evidence logging",
            "Validate replacement candidate pipeline (XLE, PEP, USO, SLV)",
            "Maintain CPS buffer ≥5 above GREEN threshold post-certification",
            "Weekly concentration audit — alert if QQQ >35%",
            "Monitor watchdog false-positive rate and loop health",
        ],
        "executive_questions": {
            "is_green_stable": cert.get("certification_status") == "GREEN_CERTIFIED",
            "regression_likelihood": stability.get("regression_probability_pct"),
            "developing_risks": top_risks[0] if top_risks else "None",
            "days_maintaining_certification": memory.get("days_continuously_green"),
            "evidence_to_preserve_trust": "Daily CPS, concentration, lifecycle, and ops evidence",
            "shock_resilience": stress[0].get("recovery_difficulty") if stress else "UNKNOWN",
            "monitoring_mode": cert.get("certification_status"),
        },
    }


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    state = _load_live_state()
    cert = _certification_status(state)
    integrity = _integrity_monitoring(state)
    warnings = _regression_warnings(state, integrity)
    stability = _institutional_stability_score(state, integrity, warnings)
    stress = _stress_scenarios(state)
    memory = _governance_memory(cert, state, warnings)
    events = _certification_events(cert, stability, warnings)
    executive = _executive_report(cert, stability, warnings, integrity, memory, stress)

    report = {
        "generated_at": ts,
        "phase": "206",
        "mode": "post_green_certification_monitor",
        "disclaimer": (
            "Continuous certification intelligence only — does NOT execute trades, "
            "modify holdings, or change governance rules."
        ),
        "certification_status": cert,
        "continuous_monitoring": integrity,
        "regression_warnings": warnings,
        "institutional_stability": stability,
        "stress_survival_testing": stress,
        "governance_memory": memory,
        "certification_events": events,
        "executive_recommendation": executive,
        "summary": {
            "certification_status": cert.get("certification_status"),
            "stability_score": stability.get("institutional_stability_score"),
            "classification": stability.get("classification"),
            "regression_probability_pct": stability.get("regression_probability_pct"),
            "warning_count": len([w for w in warnings if w.get("warning_type") != "GREEN_STABLE"]),
            "projected_green_date": memory.get("projected_certification_date"),
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(warnings).to_csv(OUT_REGRESSION, index=False, encoding="utf-8")
    pd.DataFrame(events).to_csv(OUT_EVENTS, index=False, encoding="utf-8")
    _update_history(stability, cert)

    _log(f"wrote {OUT_JSON.name}, {OUT_HISTORY.name}, {OUT_REGRESSION.name}, {OUT_EVENTS.name}")
    _log(
        f"status={cert.get('certification_status')} stability={stability.get('classification')} "
        f"regression_prob={stability.get('regression_probability_pct')}%"
    )
    return report


if __name__ == "__main__":
    run()
