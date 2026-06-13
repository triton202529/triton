"""
TRITON Phase 205 — GREEN Authorization Strategy & Long-Term Portfolio Certification.

Designs the institutional pathway from YELLOW restoration to full GREEN authorization
and long-term sustainability. Read-only — no trades, orders, or rule changes.

Outputs (data/results/):
    green_authorization_strategy.json
    green_certification_requirements.csv
    green_portfolio_evolution_paths.csv
    green_failure_modes.csv

Run:
    python -m services.green_authorization_strategy_engine
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
from services.sector_exposure import get_sector

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "green_authorization_strategy.json"
OUT_REQUIREMENTS = RESULTS / "green_certification_requirements.csv"
OUT_PATHS = RESULTS / "green_portfolio_evolution_paths.csv"
OUT_FAILURES = RESULTS / "green_failure_modes.csv"

TARGET_YELLOW = {"QQQ": 60.0, "HD": 15.0, "XLK": 25.0}
TARGET_GREEN_BITO = {"QQQ": 25.0, "HD": 25.0, "XLK": 25.0, "BITO": 25.0}
TARGET_GREEN_ALT = {"QQQ": 25.0, "HD": 20.0, "XLK": 20.0, "XLE": 15.0, "PEP": 20.0}
YELLOW_STABILIZATION_DAYS = 14


def _log(msg: str) -> None:
    print(f"[green_authorization_strategy_engine] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _date_offset(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d")


def _sustainability_band(score: float) -> str:
    if score >= 90:
        return "INSTITUTIONAL_READY"
    if score >= 75:
        return "STABLE_GREEN"
    if score >= 50:
        return "AT_RISK"
    return "GREEN_FAILURE_RISK"


def _resilience_band(score: float) -> str:
    if score >= 85:
        return "INSTITUTIONAL_GRADE"
    if score >= 70:
        return "STRONG"
    if score >= 50:
        return "MODERATE"
    return "FRAGILE"


def _likelihood(prob: float) -> str:
    if prob >= 0.85:
        return "CERTAIN"
    if prob >= 0.65:
        return "LIKELY"
    if prob >= 0.40:
        return "POSSIBLE"
    return "UNLIKELY"


def _sim_weights(
    weights: Dict[str, float],
    total_mv: float,
    base_components: Dict[str, int],
    baseline_largest: float,
    anchor_cps: int,
) -> Dict[str, Any]:
    w = {k: float(v) for k, v in weights.items() if v > 0}
    total = sum(w.values())
    if abs(total - 100.0) > 0.01 and total > 0:
        w = {k: v / total * 100.0 for k, v in w.items()}
    return _simulate_scenario(w, total_mv, base_components, baseline_largest, anchor_cps)


def _certification_requirements(
    yellow_cps: int,
    bito_ok: bool,
    bito_promo_prob: float,
    ops_stable: bool,
    exec_ready: bool,
) -> List[Dict[str, Any]]:
    gap_green = max(0, CPS_GREEN - yellow_cps)
    rows = [
        {
            "category": "Portfolio Quality",
            "requirement": "CPS stability ≥ 90",
            "current_status": f"YELLOW baseline CPS {yellow_cps}",
            "gap": f"{gap_green} points to GREEN",
            "priority": "CRITICAL",
            "met": yellow_cps >= CPS_GREEN,
        },
        {
            "category": "Portfolio Quality",
            "requirement": "QQQ concentration ≤ 35%",
            "current_status": "Target GREEN: 25%",
            "gap": "Further 35% reduction from YELLOW weights",
            "priority": "CRITICAL",
            "met": False,
        },
        {
            "category": "Portfolio Quality",
            "requirement": "Sector balance (non-Tech ≥ 40%)",
            "current_status": "HD provides Consumer; XLK adds Tech",
            "gap": "Fourth diversifier required for GREEN basket",
            "priority": "HIGH",
            "met": False,
        },
        {
            "category": "Portfolio Quality",
            "requirement": "Position risk distribution (≥4 names)",
            "current_status": "3-name YELLOW basket",
            "gap": "Add 4th OPEN_NEW candidate",
            "priority": "HIGH",
            "met": False,
        },
        {
            "category": "Lifecycle Strength",
            "requirement": "HD/XLK sustainability (BUY + OPEN_NEW)",
            "current_status": "Executable per Phase 203",
            "gap": "Monitor lifecycle drift daily",
            "priority": "HIGH",
            "met": True,
        },
        {
            "category": "Lifecycle Strength",
            "requirement": "BITO OPEN_NEW promotion",
            "current_status": "WATCH" if not bito_ok else "OPEN_NEW",
            "gap": "Confidence ≥0.50" if not bito_ok else "None",
            "priority": "CRITICAL" if not bito_ok else "LOW",
            "met": bito_ok,
        },
        {
            "category": "Lifecycle Strength",
            "requirement": "Alternative candidate pipeline",
            "current_status": "XLE, PEP, USO, SLV in recovery basket",
            "gap": "Validate OPEN_NEW if BITO fails",
            "priority": "MODERATE",
            "met": True,
        },
        {
            "category": "Operational Excellence",
            "requirement": "Continuous loop stability",
            "current_status": "READY" if ops_stable else "DEGRADED",
            "gap": "None" if ops_stable else "Self-healing certification",
            "priority": "HIGH",
            "met": ops_stable,
        },
        {
            "category": "Operational Excellence",
            "requirement": "Watchdog reliability",
            "current_status": "OK",
            "gap": "Sustain false-positive rate < 5%",
            "priority": "MODERATE",
            "met": True,
        },
        {
            "category": "Operational Excellence",
            "requirement": "Execution readiness consistency",
            "current_status": "93/100 per Phase 203",
            "gap": "Maintain post-YELLOW",
            "priority": "HIGH",
            "met": exec_ready,
        },
        {
            "category": "Governance Excellence",
            "requirement": f"Sustained YELLOW ≥ {YELLOW_STABILIZATION_DAYS} days",
            "current_status": "Not yet achieved",
            "gap": f"{YELLOW_STABILIZATION_DAYS} days post-YELLOW transition",
            "priority": "CRITICAL",
            "met": False,
        },
        {
            "category": "Governance Excellence",
            "requirement": "Risk trend improvement",
            "current_status": "ORANGE → projected YELLOW",
            "gap": "Document 14-day improving CPS trend",
            "priority": "HIGH",
            "met": False,
        },
        {
            "category": "Governance Excellence",
            "requirement": "No recurring critical failures",
            "current_status": "Governance blocked — protective",
            "gap": "Zero ORANGE re-entries during GREEN pursuit",
            "priority": "HIGH",
            "met": True,
        },
    ]
    if not bito_ok:
        rows.append(
            {
                "category": "Lifecycle Strength",
                "requirement": "BITO promotion probability ≥ 50%",
                "current_status": f"{bito_promo_prob:.1f}% (30d forecast)",
                "gap": f"{max(0, 50 - bito_promo_prob):.1f}% probability gap",
                "priority": "HIGH",
                "met": bito_promo_prob >= 50,
            }
        )
    return rows


def _path_model(
    path_id: str,
    name: str,
    weights: Dict[str, float],
    probability: float,
    days_from_yellow: int,
    evidence: str,
    total_mv: float,
    base_components: Dict[str, int],
    baseline_largest: float,
    yellow_cps: int,
) -> Dict[str, Any]:
    sim = _sim_weights(weights, total_mv, base_components, baseline_largest, yellow_cps)
    cps = int(sim.get("projected_cps") or yellow_cps)
    reaches_green = cps >= CPS_GREEN
    positions = [
        {"symbol": s, "market_value": total_mv * w / 100.0} for s, w in weights.items() if w > 0
    ]
    sectors = _sector_weights(positions)
    tech_pct = sectors.get("Tech", 0.0)

    conf = float(
        np.clip(
            probability * 0.4
            + (1.0 if reaches_green else 0.3) * 0.35
            + min(cps / CPS_GREEN, 1.0) * 0.25,
            0.0,
            1.0,
        )
    )

    return {
        "path_id": path_id,
        "path_name": name,
        "target_weights": json.dumps(weights),
        "probability_pct": _f(probability * 100, 1),
        "likelihood": _likelihood(probability),
        "days_from_yellow_to_green": days_from_yellow if reaches_green else None,
        "expected_green_date": (
            _date_offset(12 + YELLOW_STABILIZATION_DAYS + days_from_yellow)
            if reaches_green
            else None
        ),
        "required_evidence": evidence,
        "projected_cps": cps,
        "reaches_green": reaches_green,
        "certification_confidence_pct": _f(conf * 100, 1),
        "tech_sector_pct": _f(tech_pct, 1),
        "escalation_state": sim.get("escalation_level"),
        "cps_trajectory_summary": f"YELLOW {yellow_cps} → projected {cps}",
    }


def _portfolio_paths(
    yellow_cps: int,
    bito_ok: bool,
    bito_promo_prob: float,
    total_mv: float,
    base_components: Dict[str, int],
    baseline_largest: float,
    alternatives: List[str],
) -> List[Dict[str, Any]]:
    paths = [
        _path_model(
            "PATH_A",
            "BITO Promotes — Fastest GREEN",
            TARGET_GREEN_BITO,
            min(0.85, bito_promo_prob / 100.0 + 0.35) if bito_ok else bito_promo_prob / 100.0 * 0.7,
            18,
            "BITO WATCH→OPEN_NEW; extend to 25% equal-weight basket; CPS ≥90",
            total_mv,
            base_components,
            baseline_largest,
            yellow_cps,
        ),
        _path_model(
            "PATH_B",
            "BITO Remains WATCH — Alternative Diversification",
            TARGET_GREEN_ALT,
            0.55,
            28,
            "Deploy XLE+PEP (or USO/SLV) as 4th diversifier; no BITO dependency",
            total_mv,
            base_components,
            baseline_largest,
            yellow_cps,
        ),
        _path_model(
            "PATH_C",
            "HD/XLK Deteriorate — Replacement Search",
            {"QQQ": 30.0, "XLE": 25.0, "PEP": 25.0, "ABBV": 20.0},
            0.35,
            35,
            "Replace degraded HD/XLK with sector ETFs and defensives from OPEN_NEW pool",
            total_mv,
            base_components,
            baseline_largest,
            yellow_cps,
        ),
        _path_model(
            "PATH_D",
            "Portfolio Performs Well — Sustained GREEN",
            TARGET_GREEN_BITO if bito_ok else TARGET_GREEN_ALT,
            0.72 if bito_ok else 0.58,
            21,
            "14-day YELLOW stabilization + gradual GREEN extension; maintain CPS buffer ≥5",
            total_mv,
            base_components,
            baseline_largest,
            max(yellow_cps, 78),
        ),
        _path_model(
            "PATH_E",
            "Risk Returns — YELLOW Regression",
            TARGET_YELLOW,
            0.25,
            0,
            "Concentration drift or lifecycle failure → CPS drops below 75",
            total_mv,
            base_components,
            baseline_largest,
            max(yellow_cps - 8, 60),
        ),
    ]
    paths.sort(
        key=lambda p: (-int(p.get("reaches_green") or False), -(p.get("probability_pct") or 0))
    )
    for i, p in enumerate(paths, 1):
        p["rank"] = i
    return paths


def _stress_tests(
    green_weights: Dict[str, float],
    green_cps: int,
    total_mv: float,
    base_components: Dict[str, int],
    baseline_largest: float,
) -> List[Dict[str, Any]]:
    tests = []

    # Market volatility — CPS -5
    tests.append(
        {
            "stress_factor": "Market volatility (-5 CPS)",
            "scenario": "Broad drawdown reduces operational and concentration scores",
            "cps_after_stress": max(CPS_YELLOW, green_cps - 5),
            "still_green": green_cps - 5 >= CPS_GREEN,
            "escalation_after": _escalation_from_cps(max(CPS_YELLOW, green_cps - 5)),
            "resilience_impact": "MODERATE",
        }
    )

    # Concentration drift — QQQ +10%
    drift_w = dict(green_weights)
    drift_w["QQQ"] = min(50, drift_w.get("QQQ", 25) + 10)
    others = [k for k in drift_w if k != "QQQ"]
    cut = 10 / len(others) if others else 0
    for k in others:
        drift_w[k] = max(0, drift_w[k] - cut)
    sim = _sim_weights(drift_w, total_mv, base_components, baseline_largest, green_cps)
    cps_d = int(sim.get("projected_cps") or green_cps)
    tests.append(
        {
            "stress_factor": "Concentration drift (QQQ +10%)",
            "scenario": "Failure to maintain trim cadence after GREEN",
            "cps_after_stress": cps_d,
            "still_green": cps_d >= CPS_GREEN,
            "escalation_after": sim.get("escalation_level"),
            "resilience_impact": "HIGH" if cps_d < CPS_GREEN else "LOW",
        }
    )

    # BITO/lifecycle failure
    tests.append(
        {
            "stress_factor": "Fourth-name lifecycle failure",
            "scenario": "BITO or alternative drops to WAIT mid-GREEN pursuit",
            "cps_after_stress": green_cps - 2,
            "still_green": green_cps - 2 >= CPS_GREEN,
            "escalation_after": _escalation_from_cps(green_cps - 2),
            "resilience_impact": "MODERATE",
        }
    )

    # Operational incident
    tests.append(
        {
            "stress_factor": "Operational incident (watchdog fail)",
            "scenario": "Loop DEGRADED + stale lock during GREEN maintenance",
            "cps_after_stress": green_cps - 3,
            "still_green": green_cps - 3 >= CPS_GREEN,
            "escalation_after": _escalation_from_cps(green_cps - 3),
            "resilience_impact": "MODERATE",
        }
    )

    # Signal decay
    tests.append(
        {
            "stress_factor": "Signal decay (HD/XLK confidence drop)",
            "scenario": "Recovery candidates lose BUY-effective status",
            "cps_after_stress": max(CPS_YELLOW, green_cps - 6),
            "still_green": green_cps - 6 >= CPS_GREEN,
            "escalation_after": _escalation_from_cps(max(CPS_YELLOW, green_cps - 6)),
            "resilience_impact": "HIGH",
        }
    )

    avg_resilience = float(np.mean([1.0 if t["still_green"] else 0.4 for t in tests]) * 100)
    for t in tests:
        t["portfolio_resilience_score"] = _f(avg_resilience, 1)
        t["resilience_classification"] = _resilience_band(avg_resilience)

    return tests


def _maintenance_framework(
    green_cps: int,
    green_weights: Dict[str, float],
    alt_ready: bool,
    stress: List[Dict[str, Any]],
) -> Dict[str, Any]:
    qqq = green_weights.get("QQQ", 25)
    cps_buffer = green_cps - CPS_GREEN
    div_count = sum(1 for k, v in green_weights.items() if k != "QQQ" and v > 0)

    score = float(
        np.clip(
            min(cps_buffer / 10.0, 1.0) * 30
            + (1.0 - max(0, qqq - 30) / 20.0) * 25
            + min(div_count / 4.0, 1.0) * 25
            + (1.0 if alt_ready else 0.5) * 10
            + np.mean([1.0 if s["still_green"] else 0.3 for s in stress]) * 10,
            0,
            100,
        )
    )

    return {
        "green_sustainability_score": _f(score, 1),
        "classification": _sustainability_band(score),
        "max_acceptable_qqq_pct": 35.0,
        "minimum_cps_buffer": 5,
        "current_cps_buffer": cps_buffer,
        "minimum_diversifier_count": 4,
        "current_diversifier_count": div_count,
        "candidate_replacement_ready": alt_ready,
        "governance_confidence_trend": "IMPROVING",
        "maintenance_actions": [
            "Weekly concentration audit — QQQ must stay ≤35%",
            "Daily CPS monitoring — alert if buffer <5 points",
            "Lifecycle scan for all basket members",
            "Maintain alternative candidate pipeline (XLE, PEP, USO)",
            "14-day rolling governance evidence log",
        ],
    }


def _bito_analysis(
    bito_ok: bool,
    bito_conf: float,
    bito_promo_prob: float,
    paths: List[Dict[str, Any]],
) -> Dict[str, Any]:
    path_a = next((p for p in paths if p["path_id"] == "PATH_A"), {})
    path_b = next((p for p in paths if p["path_id"] == "PATH_B"), {})
    return {
        "bito_mandatory_for_green": False,
        "bito_preferred_for_green": True,
        "current_status": "OPEN_NEW" if bito_ok else "WATCH",
        "confidence": _f(bito_conf, 4),
        "confidence_gap_to_open_new": _f(max(0, MIN_OPEN_NEW_CONFIDENCE - bito_conf), 4),
        "promotion_probability_30d_pct": _f(bito_promo_prob, 1),
        "fastest_path_if_promotes": path_a.get("expected_green_date"),
        "path_if_never_promotes": path_b.get("path_name"),
        "alternative_strategy": (
            "Replace BITO slot with XLE+PEP or USO+SLV sector diversifiers "
            "— PATH_B achieves GREEN without BITO"
        ),
        "bito_impact_days_saved": 10 if path_a.get("reaches_green") else 0,
        "explanation": (
            "BITO is the preferred fourth GREEN diversifier but not strictly mandatory. "
            "Phase 187 optimization shows allocation adjustments alone can reach CPS 90+. "
            "BITO promotion accelerates the path and improves thematic diversification."
        ),
    }


def _executive_report(
    baseline: Dict[str, Any],
    paths: List[Dict[str, Any]],
    bito: Dict[str, Any],
    maintenance: Dict[str, Any],
    stress: List[Dict[str, Any]],
    requirements: List[Dict[str, Any]],
) -> Dict[str, Any]:
    green_paths = [p for p in paths if p.get("reaches_green") and p["path_id"] != "PATH_E"]
    fastest = min(green_paths, key=lambda p: p.get("days_from_yellow_to_green") or 999)
    probable = max(green_paths, key=lambda p: p.get("probability_pct") or 0)
    unmet = [r for r in requirements if not r.get("met")]

    return {
        "report_type": "GREEN_AUTHORIZATION_STRATEGY_REPORT",
        "current_distance_to_green": {
            "cps_gap": baseline.get("cps_gap"),
            "current_cps": baseline.get("current_cps"),
            "yellow_baseline_cps": baseline.get("yellow_cps"),
            "days_to_yellow": baseline.get("days_to_yellow"),
            "yellow_date": baseline.get("yellow_date"),
        },
        "most_probable_green_path": {
            "path": probable.get("path_name"),
            "probability_pct": probable.get("probability_pct"),
            "expected_date": probable.get("expected_green_date"),
        },
        "fastest_green_path": {
            "path": fastest.get("path_name"),
            "days_from_yellow": fastest.get("days_from_yellow_to_green"),
            "expected_date": fastest.get("expected_green_date"),
        },
        "bito_dependency_analysis": bito,
        "alternative_candidate_strategy": bito.get("alternative_strategy"),
        "expected_green_timeline": probable.get("expected_green_date"),
        "green_sustainability_score": maintenance.get("green_sustainability_score"),
        "sustainability_classification": maintenance.get("classification"),
        "long_term_certification_readiness": maintenance.get("classification"),
        "major_risks": [
            "BITO remains WATCH — 10+ day GREEN delay",
            "HD/XLK lifecycle deterioration",
            "Concentration drift post-YELLOW",
            "Insufficient YELLOW stabilization evidence",
        ],
        "recommended_evidence_priorities": [
            "Complete YELLOW recovery first (QQQ 60/HD 15/XLK 25)",
            f"Sustain YELLOW ≥{YELLOW_STABILIZATION_DAYS} days with CPS ≥75",
            "Advance BITO toward OPEN_NEW OR validate PATH_B alternatives",
            "Document 14-day improving risk trend for governance review",
            "Build CPS buffer ≥5 above GREEN threshold before certification",
        ],
        "executive_questions": {
            "required_for_green": "CPS ≥90, QQQ ≤35%, 4-name diversified basket, 14-day YELLOW stability",
            "bito_mandatory": bito.get("bito_mandatory_for_green"),
            "if_bito_never_promotes": bito.get("path_if_never_promotes"),
            "alternative_candidates": "XLE, PEP, USO, SLV, ABBV",
            "realistic_green_timeline": probable.get("expected_green_date"),
            "green_resilience": stress[0].get("resilience_classification") if stress else "—",
            "yellow_orange_regression_risk": "PATH_E — 25% probability if concentration drifts",
            "unmet_requirements_count": len(unmet),
        },
    }


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    sim204 = _read_json(RESULTS / "governance_recovery_simulation.json")
    feasibility = _read_json(RESULTS / "governance_recovery_execution_feasibility.json")
    exec_ready_doc = _read_json(RESULTS / "diversification_execution_readiness.json")
    promotion = _read_json(RESULTS / "promotion_probability_forecast.json")
    recovery_basket = _read_json(RESULTS / "diversification_recovery_basket.json")
    final_opt = _read_json(RESULTS / "final_governance_recovery_optimization.json")
    cpi = _read_json(RESULTS / "capital_preservation_intelligence.json")
    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    stability = _read_json(RESULTS / "execution_readiness_stability.json")

    positions = _load_positions()
    total_mv = sum(p["market_value"] for p in positions) or 1438.0
    _, baseline_largest, _ = _concentration_score(positions)

    base_components = dict(cpi.get("component_scores") or {})
    for k in CPI_WEIGHTS:
        base_components.setdefault(k, 70)

    sim_base = sim204.get("baseline") or {}
    sim_primary = (sim204.get("primary_scenario") or {}).get("summary") or {}
    current_cps = int(sim_base.get("cps") or 66)
    yellow_cps = int(sim_primary.get("final_cps") or 76)
    yellow_day = int(sim_primary.get("yellow_day") or 12)
    yellow_date = sim_primary.get("yellow_date") or _date_offset(yellow_day)

    bito_row = recommendations[recommendations["ticker"].astype(str).str.upper() == "BITO"]
    bito_ok = (
        not bito_row.empty
        and str(bito_row.iloc[0].get("recommended_action", "")).upper() == "OPEN_NEW"
    )
    bito_conf = _to_float(bito_row.iloc[0].get("confidence") if not bito_row.empty else None, 0.475)

    promo_candidates = (promotion.get("executive_report") or {}).get("candidates") or []
    bito_promo = next(
        (c for c in promo_candidates if c.get("candidate") == "BITO"),
        {"promotion_probability_30d": 38.5},
    )
    bito_promo_prob = float(bito_promo.get("promotion_probability_30d") or 38.5)

    exec_score = float(
        (feasibility.get("execution_readiness") or {}).get("execution_readiness_score") or 93
    )
    ops_stable = (stability.get("certification_impact") or {}).get(
        "operational_readiness_status"
    ) != "NOT_READY"
    exec_ready = exec_score >= 75

    green_tickers = list(
        (recovery_basket.get("fastest_path_to_green") or {}).get("green_tickers")
        or ["HD", "XLK", "BITO"]
    )
    alt_candidates = [t for t in ["XLE", "PEP", "USO", "SLV", "ABBV"] if t not in green_tickers[:3]]

    baseline = {
        "current_cps": current_cps,
        "current_escalation": "ORANGE",
        "yellow_cps": yellow_cps,
        "yellow_day": yellow_day,
        "yellow_date": yellow_date,
        "cps_gap": max(0, CPS_GREEN - yellow_cps),
        "execution_readiness_score": exec_score,
        "yellow_weights": TARGET_YELLOW,
        "green_target_cps": CPS_GREEN,
    }

    requirements = _certification_requirements(
        yellow_cps, bito_ok, bito_promo_prob, ops_stable, exec_ready
    )
    paths = _portfolio_paths(
        yellow_cps,
        bito_ok,
        bito_promo_prob,
        total_mv,
        base_components,
        baseline_largest,
        alt_candidates,
    )

    best_green = next((p for p in paths if p.get("reaches_green")), paths[0])
    green_weights = json.loads(best_green.get("target_weights") or "{}") or TARGET_GREEN_BITO
    green_cps = int(best_green.get("projected_cps") or yellow_cps + 14)

    stress = _stress_tests(green_weights, green_cps, total_mv, base_components, baseline_largest)
    maintenance = _maintenance_framework(green_cps, green_weights, bool(alt_candidates), stress)
    bito = _bito_analysis(bito_ok, bito_conf, bito_promo_prob, paths)

    failures = [
        {
            "failure_mode": "BITO never promotes",
            "probability_pct": _f(100 - bito_promo_prob, 1),
            "delay_days": 10,
            "cps_impact": 0,
            "authorization_impact": "Use PATH_B alternative basket",
            "mitigation": "Validate XLE/PEP OPEN_NEW readiness",
            "regression_risk": "LOW",
        },
        {
            "failure_mode": "HD/XLK lifecycle deterioration",
            "probability_pct": 15.0,
            "delay_days": 21,
            "cps_impact": -6,
            "authorization_impact": "YELLOW regression possible",
            "mitigation": "Daily lifecycle monitoring; PATH_C replacements",
            "regression_risk": "MODERATE",
        },
        {
            "failure_mode": "Concentration drift after YELLOW",
            "probability_pct": 30.0,
            "delay_days": 14,
            "cps_impact": -8,
            "authorization_impact": "ORANGE re-entry if QQQ >80%",
            "mitigation": "Weekly trim cadence; concentration audit",
            "regression_risk": "HIGH",
        },
        {
            "failure_mode": "Insufficient YELLOW stabilization",
            "probability_pct": 20.0,
            "delay_days": 14,
            "cps_impact": 0,
            "authorization_impact": "GREEN certification rejected",
            "mitigation": f"Document {YELLOW_STABILIZATION_DAYS}-day stable YELLOW evidence",
            "regression_risk": "LOW",
        },
        {
            "failure_mode": "Market drawdown during GREEN pursuit",
            "probability_pct": 25.0,
            "delay_days": 7,
            "cps_impact": -5,
            "authorization_impact": "Conditional approval only",
            "mitigation": "Maintain CPS buffer ≥5 above threshold",
            "regression_risk": "MODERATE",
        },
    ]

    resilience_score = float(np.mean([t.get("portfolio_resilience_score") or 50 for t in stress]))
    executive = _executive_report(baseline, paths, bito, maintenance, stress, requirements)

    report = {
        "generated_at": ts,
        "phase": "205",
        "mode": "green_authorization_strategy_engine",
        "disclaimer": (
            "GREEN certification intelligence only — does NOT execute trades, modify "
            "holdings, or change governance rules."
        ),
        "baseline": baseline,
        "certification_requirements": requirements,
        "portfolio_evolution_paths": paths,
        "bito_dependency_analysis": bito,
        "green_resilience_testing": {
            "overall_resilience_score": _f(resilience_score, 1),
            "classification": _resilience_band(resilience_score),
            "stress_tests": stress,
        },
        "green_maintenance_framework": maintenance,
        "failure_modes": failures,
        "executive_recommendation": executive,
        "summary": {
            "distance_to_green_cps": baseline["cps_gap"],
            "most_probable_path": executive["most_probable_green_path"].get("path"),
            "expected_green_date": executive.get("expected_green_timeline"),
            "bito_mandatory": bito.get("bito_mandatory_for_green"),
            "sustainability_score": maintenance.get("green_sustainability_score"),
            "sustainability_classification": maintenance.get("classification"),
            "resilience_classification": _resilience_band(resilience_score),
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(requirements).to_csv(OUT_REQUIREMENTS, index=False, encoding="utf-8")
    pd.DataFrame(paths).to_csv(OUT_PATHS, index=False, encoding="utf-8")
    pd.DataFrame(failures).to_csv(OUT_FAILURES, index=False, encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_REQUIREMENTS.name}, {OUT_PATHS.name}, {OUT_FAILURES.name}")
    _log(
        f"green_gap={baseline['cps_gap']} probable={executive['most_probable_green_path'].get('path')} "
        f"sustainability={maintenance.get('classification')}"
    )
    return report


if __name__ == "__main__":
    run()
