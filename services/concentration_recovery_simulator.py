"""
TRITON Phase 178 — Concentration Recovery Simulation Engine (analysis only).

Forecasts governance recovery outcomes under realistic diversification scenarios
before any execution occurs.

SAFETY: analysis only — no governance, lifecycle, execution, allocation, broker,
or risk control changes.

Outputs (data/results/):
    concentration_recovery_simulation.json
    concentration_recovery_scenarios.csv

Run:
    python -m services.concentration_recovery_simulator
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
    _health_band,
    _load_positions,
    _read_csv,
    _read_json,
)
from services.diversification_feasibility_analysis import MIN_OPEN_NEW_CONFIDENCE
from services.pnl_ledger_builder import _norm_sym, _to_float
from services.sector_exposure import get_sector

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "concentration_recovery_simulation.json"
OUT_CSV = RESULTS / "concentration_recovery_scenarios.csv"

CPS_YELLOW = 75
CPS_GREEN = 90
PERMIT_ESCALATION = {"GREEN", "YELLOW"}
RECOVERY_FOCUS = ["HD", "XLK", "BITO"]


def _log(msg: str) -> None:
    print(f"[concentration_recovery_simulator] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _weights_to_positions(weights: Dict[str, float], total_mv: float) -> List[Dict[str, Any]]:
    positions: List[Dict[str, Any]] = []
    for sym, pct in weights.items():
        if pct <= 0:
            continue
        positions.append(
            {"symbol": _norm_sym(sym), "market_value": total_mv * pct / 100.0, "qty": np.nan}
        )
    return positions


def _simulate_scenario(
    weights: Dict[str, float],
    total_mv: float,
    base_components: Dict[str, int],
    baseline_largest_pct: float,
    current_cps: int,
) -> Dict[str, Any]:
    positions = _weights_to_positions(weights, total_mv)
    _, largest, _ = _concentration_score(positions)
    cps, comps = _estimate_cps(positions, base_components)
    esc = _escalation_from_cps(cps)
    conc_reduction = max(0.0, baseline_largest_pct - largest)
    dist_yellow = max(0, CPS_YELLOW - cps)
    dist_green = max(0, CPS_GREEN - cps)
    gov = "AUTHORIZED" if esc in PERMIT_ESCALATION else "BLOCKED"
    cps_improvement = cps - current_cps

    if cps_improvement <= 0 and esc == _escalation_from_cps(current_cps):
        classification = "NO_IMPROVEMENT"
    elif esc == "GREEN":
        classification = "GREEN_RECOVERY"
    elif esc == "YELLOW":
        classification = "YELLOW_RECOVERY"
    elif cps_improvement > 0:
        classification = "PARTIAL_RECOVERY"
    else:
        classification = "NO_IMPROVEMENT"

    recovery_score = float(
        np.clip(
            (
                0.35 * min(cps_improvement / 30.0, 1.0)
                + 0.30 * min(conc_reduction / 40.0, 1.0)
                + 0.20 * (1.0 if gov == "AUTHORIZED" else 0.0)
                + 0.15 * (1.0 - dist_yellow / 30.0 if dist_yellow < 30 else 0.0)
            )
            * 100.0,
            0.0,
            100.0,
        )
    )

    return {
        "weights": {k: _f(v, 2) for k, v in weights.items()},
        "projected_cps": cps,
        "cps_improvement": cps_improvement,
        "concentration_reduction_pct": _f(conc_reduction, 2),
        "largest_position_weight_pct": _f(largest, 2),
        "governance_status": gov,
        "escalation_level": esc,
        "distance_to_yellow": dist_yellow,
        "distance_to_green": dist_green,
        "recovery_score": _f(recovery_score, 1),
        "classification": classification,
        "position_count": len(positions),
        "portfolio_health": _health_band(cps),
        "component_scores": comps,
    }


def _bito_eligible(recommendations: pd.DataFrame, lifecycle: pd.DataFrame) -> bool:
    sym = "BITO"
    rec = recommendations[recommendations["ticker"].map(_norm_sym) == sym]
    if rec.empty:
        return False
    row = rec.iloc[0]
    action = str(row.get("recommended_action") or "").upper()
    conf = _to_float(row.get("confidence"), 0.0)
    if action == "OPEN_NEW":
        return True
    if conf >= MIN_OPEN_NEW_CONFIDENCE:
        return True
    lc = lifecycle[lifecycle["ticker"].map(_norm_sym) == sym]
    if not lc.empty:
        lc_action = str(lc.iloc[0].get("lifecycle_action") or "").upper()
        if lc_action == "OPEN_NEW":
            return True
    return False


def _build_scenario_f_weights(bito_ok: bool) -> Dict[str, float]:
    if bito_ok:
        return {"QQQ": 25.0, "HD": 25.0, "XLK": 25.0, "BITO": 25.0}
    return {"QQQ": 33.33, "HD": 33.33, "XLK": 33.34}


def _open_new_tickers(recommendations: pd.DataFrame) -> List[str]:
    if recommendations.empty or "recommended_action" not in recommendations.columns:
        return []
    df = recommendations[
        recommendations["recommended_action"].astype(str).str.upper() == "OPEN_NEW"
    ].copy()
    return [_norm_sym(t) for t in df["ticker"].tolist() if _norm_sym(t)]


def _candidate_confidence(recommendations: pd.DataFrame, sym: str) -> float:
    sub = recommendations[recommendations["ticker"].map(_norm_sym) == sym]
    if sub.empty:
        return 0.0
    return _to_float(sub.iloc[0].get("confidence"), 0.0)


def _risk_score(weights: Dict[str, float], recommendations: pd.DataFrame) -> float:
    """Lower is better — penalize low confidence additions and sector crowding."""
    tech_w = sum(w for s, w in weights.items() if get_sector(s) == "Tech")
    conf_penalty = 0.0
    for sym, w in weights.items():
        if sym == "QQQ" or w <= 0:
            continue
        conf = _candidate_confidence(recommendations, sym)
        if conf < MIN_OPEN_NEW_CONFIDENCE:
            conf_penalty += w * 0.5
    return tech_w + conf_penalty


def _find_min_weight_for_target(
    ticker: str,
    other_weights: Dict[str, float],
    total_mv: float,
    base_components: Dict[str, int],
    target_cps: int,
    baseline_largest: float,
    current_cps: int,
) -> Optional[float]:
    best: Optional[float] = None
    for pct in range(5, 51):
        weights = dict(other_weights)
        weights[ticker] = float(pct)
        remaining = 100.0 - sum(weights.values())
        if remaining < 0:
            continue
        weights["QQQ"] = weights.get("QQQ", 0.0) + remaining
        sim = _simulate_scenario(weights, total_mv, base_components, baseline_largest, current_cps)
        if sim["projected_cps"] >= target_cps:
            best = float(pct)
            break
    return best


def _find_min_combined_for_target(
    tickers: List[str],
    total_mv: float,
    base_components: Dict[str, int],
    target_cps: int,
    baseline_largest: float,
    current_cps: int,
) -> Optional[Dict[str, float]]:
    """Equal split among tickers; sweep combined allocation from 10% to 80%."""
    for combined in range(10, 81, 5):
        each = combined / len(tickers)
        weights = {"QQQ": 100.0 - combined}
        for t in tickers:
            weights[t] = each
        sim = _simulate_scenario(weights, total_mv, base_components, baseline_largest, current_cps)
        if sim["projected_cps"] >= target_cps:
            return {k: _f(v, 2) for k, v in weights.items()}
    return None


def _recovery_probability(
    scenario: Dict[str, Any],
    recommendations: pd.DataFrame,
    feasibility: Dict[str, Any],
) -> Tuple[str, float, float]:
    weights = scenario.get("weights") or {}
    adds = [s for s in weights if s != "QQQ" and (weights.get(s) or 0) > 0]
    if not adds:
        return "LOW", 0.15, 0.5

    feas_map = {
        _norm_sym(r.get("ticker")): r.get("feasibility_status")
        for r in (feasibility.get("feasibility_candidates") or [])
    }
    immediately = sum(1 for s in adds if feas_map.get(s) == "IMMEDIATELY_FEASIBLE")
    avail = immediately / len(adds)

    confs = [_candidate_confidence(recommendations, s) for s in adds]
    conf_factor = (
        float(np.mean([min(c / MIN_OPEN_NEW_CONFIDENCE, 1.0) for c in confs if c > 0]))
        if confs
        else 0.0
    )

    dist_y = scenario.get("distance_to_yellow") or 30
    dist_factor = max(0.0, 1.0 - dist_y / 30.0)
    gov_factor = 1.0 if scenario.get("governance_status") == "AUTHORIZED" else 0.3

    score = 0.35 * avail + 0.25 * conf_factor + 0.20 * dist_factor + 0.20 * gov_factor
    score = float(np.clip(score, 0.0, 1.0))
    forecast_conf = float(np.clip(0.5 + 0.3 * conf_factor + 0.2 * avail, 0.3, 0.95))

    if score >= 0.80:
        label = "VERY_HIGH"
    elif score >= 0.60:
        label = "HIGH"
    elif score >= 0.35:
        label = "MODERATE"
    else:
        label = "LOW"
    return label, score, forecast_conf


def _pick_path(
    scenarios: List[Dict[str, Any]],
    key: str,
    recommendations: pd.DataFrame,
) -> Dict[str, Any]:
    if not scenarios:
        return {}
    if key == "BEST_RECOVERY_PATH":
        best = max(scenarios, key=lambda s: _to_float(s.get("recovery_score"), 0))
    elif key == "FASTEST_PATH_TO_YELLOW":
        yellow = [
            s for s in scenarios if s.get("classification") in ("YELLOW_RECOVERY", "GREEN_RECOVERY")
        ]
        if not yellow:
            yellow = sorted(scenarios, key=lambda s: s.get("distance_to_yellow", 99))
        best = min(
            yellow,
            key=lambda s: sum(v for k, v in (s.get("weights") or {}).items() if k != "QQQ"),
        )
    elif key == "FASTEST_PATH_TO_GREEN":
        green = [s for s in scenarios if s.get("classification") == "GREEN_RECOVERY"]
        if not green:
            green = sorted(scenarios, key=lambda s: s.get("distance_to_green", 99))
        best = min(
            green,
            key=lambda s: sum(v for k, v in (s.get("weights") or {}).items() if k != "QQQ"),
        )
    elif key == "LOWEST_RISK_RECOVERY_PATH":
        best = min(scenarios, key=lambda s: _risk_score(s.get("weights") or {}, recommendations))
    elif key == "HIGHEST_PROBABILITY_RECOVERY_PATH":
        best = max(scenarios, key=lambda s: _to_float(s.get("recovery_probability_score"), 0))
    else:
        best = scenarios[0]

    return {
        "path_type": key,
        "scenario": best.get("scenario"),
        "weights": best.get("weights"),
        "projected_cps": best.get("projected_cps"),
        "escalation_level": best.get("escalation_level"),
        "governance_status": best.get("governance_status"),
        "classification": best.get("classification"),
        "recovery_score": best.get("recovery_score"),
    }


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    positions = _load_positions()
    held = [p["symbol"] for p in positions]
    total_mv = sum(p["market_value"] for p in positions)
    baseline_largest = _concentration_score(positions)[1]

    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    lifecycle = _read_csv(RESULTS / "signal_lifecycle_effective.csv")
    cpi = _read_json(RESULTS / "capital_preservation_intelligence.json")
    cpe = _read_json(RESULTS / "capital_preservation_escalation.json")
    auth = _read_json(RESULTS / "governance_authorization.json")
    recovery = _read_json(RESULTS / "diversification_recovery_basket.json")
    feasibility = _read_json(RESULTS / "diversification_feasibility_analysis.json")
    monitor = _read_json(RESULTS / "recovery_basket_monitor.json")

    base_components = dict(cpi.get("component_scores") or {})
    for k in CPI_WEIGHTS:
        base_components.setdefault(k, 70)

    current_cps = int(
        cpi.get("capital_preservation_score") or _estimate_cps(positions, base_components)[0]
    )
    current_esc = str(cpe.get("escalation_state") or _escalation_from_cps(current_cps))
    current_gov = "AUTHORIZED" if auth.get("governance_authorized") else "BLOCKED"

    current_state = {
        "generated_at": ts,
        "holdings": {
            p["symbol"]: _f(p["market_value"] / total_mv * 100 if total_mv else 0, 2)
            for p in positions
        },
        "largest_concentration_pct": _f(baseline_largest, 2),
        "current_cps": current_cps,
        "current_escalation": current_esc,
        "current_governance_status": current_gov,
        "position_count": len(positions),
        "open_new_opportunities": _open_new_tickers(recommendations),
        "recovery_basket": list(
            recovery.get("executive_report", {}).get("best_recovery_basket")
            or ["QQQ", "HD", "XLK", "BITO"]
        ),
    }

    bito_ok = _bito_eligible(recommendations, lifecycle)
    scenario_defs = [
        ("Scenario A — Baseline", {"QQQ": 100.0}),
        ("Scenario B — QQQ 90% / HD 10%", {"QQQ": 90.0, "HD": 10.0}),
        ("Scenario C — QQQ 80% / HD 10% / XLK 10%", {"QQQ": 80.0, "HD": 10.0, "XLK": 10.0}),
        ("Scenario D — QQQ 70% / HD 15% / XLK 15%", {"QQQ": 70.0, "HD": 15.0, "XLK": 15.0}),
        ("Scenario E — QQQ 60% / HD 20% / XLK 20%", {"QQQ": 60.0, "HD": 20.0, "XLK": 20.0}),
        (
            f"Scenario F — Dynamic Recovery Basket{' (+BITO)' if bito_ok else ''}",
            _build_scenario_f_weights(bito_ok),
        ),
    ]

    scenario_results: List[Dict[str, Any]] = []
    for name, weights in scenario_defs:
        sim = _simulate_scenario(weights, total_mv, base_components, baseline_largest, current_cps)
        prob_label, prob_score, forecast_conf = _recovery_probability(
            sim, recommendations, feasibility
        )
        scenario_results.append(
            {
                "scenario": name,
                **sim,
                "recovery_probability": prob_label,
                "recovery_probability_score": _f(prob_score * 100, 1),
                "forecast_confidence": _f(forecast_conf * 100, 1),
            }
        )

    scenario_results.sort(key=lambda s: _to_float(s.get("recovery_score"), 0), reverse=True)

    min_hd_yellow = _find_min_weight_for_target(
        "HD", {"QQQ": 100.0}, total_mv, base_components, CPS_YELLOW, baseline_largest, current_cps
    )
    min_xlk_yellow = _find_min_weight_for_target(
        "XLK", {"QQQ": 100.0}, total_mv, base_components, CPS_YELLOW, baseline_largest, current_cps
    )
    min_hd_xlk_yellow = _find_min_combined_for_target(
        ["HD", "XLK"], total_mv, base_components, CPS_YELLOW, baseline_largest, current_cps
    )
    min_hd_xlk_green = _find_min_combined_for_target(
        ["HD", "XLK"], total_mv, base_components, CPS_GREEN, baseline_largest, current_cps
    )
    min_full_green = None
    if bito_ok:
        min_full_green = _find_min_combined_for_target(
            ["HD", "XLK", "BITO"],
            total_mv,
            base_components,
            CPS_GREEN,
            baseline_largest,
            current_cps,
        )
    elif min_hd_xlk_green:
        min_full_green = min_hd_xlk_green

    threshold_discovery = {
        "minimum_hd_allocation_for_yellow_pct": _f(min_hd_yellow, 1),
        "minimum_xlk_allocation_for_yellow_pct": _f(min_xlk_yellow, 1),
        "minimum_combined_hd_xlk_for_yellow": min_hd_xlk_yellow,
        "minimum_combined_for_green": min_full_green or min_hd_xlk_green,
        "estimated_path_to_clear_governance": (
            "Add HD+XLK at equal weight (~"
            + str(
                _f(
                    sum(v for k, v in (min_hd_xlk_yellow or {}).items() if k != "QQQ"),
                    1,
                )
            )
            + "% combined) for YELLOW; "
            + (
                f"include BITO for GREEN (~{ _f(sum(v for k, v in (min_full_green or {}).items() if k != 'QQQ'), 1) }% additions)"
                if bito_ok
                else "BITO promotion required for full GREEN clearance"
            )
        ),
        "bito_included_in_green_path": bito_ok,
    }

    paths = {
        "BEST_RECOVERY_PATH": _pick_path(scenario_results, "BEST_RECOVERY_PATH", recommendations),
        "FASTEST_PATH_TO_YELLOW": _pick_path(
            scenario_results, "FASTEST_PATH_TO_YELLOW", recommendations
        ),
        "FASTEST_PATH_TO_GREEN": _pick_path(
            scenario_results, "FASTEST_PATH_TO_GREEN", recommendations
        ),
        "LOWEST_RISK_RECOVERY_PATH": _pick_path(
            scenario_results, "LOWEST_RISK_RECOVERY_PATH", recommendations
        ),
        "HIGHEST_PROBABILITY_RECOVERY_PATH": _pick_path(
            scenario_results, "HIGHEST_PROBABILITY_RECOVERY_PATH", recommendations
        ),
    }

    best = paths["BEST_RECOVERY_PATH"]
    projected_state = {
        "scenario": best.get("scenario"),
        "weights": best.get("weights"),
        "projected_cps": best.get("projected_cps"),
        "escalation_level": best.get("escalation_level"),
        "governance_status": best.get("governance_status"),
    }
    recovery_delta = {
        "cps_delta": (best.get("projected_cps") or 0) - current_cps,
        "escalation_change": f"{current_esc} → {best.get('escalation_level')}",
        "governance_change": f"{current_gov} → {best.get('governance_status')}",
    }
    prob_label, prob_score, forecast_conf = _recovery_probability(
        next((s for s in scenario_results if s.get("scenario") == best.get("scenario")), {}),
        recommendations,
        feasibility,
    )

    forecasting_foundation = {
        "current_state": {
            "cps": current_cps,
            "escalation": current_esc,
            "governance_status": current_gov,
            "concentration_pct": current_state["largest_concentration_pct"],
        },
        "projected_state": projected_state,
        "recovery_delta": recovery_delta,
        "recovery_probability": prob_label,
        "recovery_probability_score": _f(prob_score * 100, 1),
        "forecast_confidence": _f(forecast_conf * 100, 1),
        "scenario_projections": [
            {
                "scenario": s["scenario"],
                "projected_cps": s["projected_cps"],
                "escalation_level": s["escalation_level"],
                "governance_status": s["governance_status"],
                "classification": s["classification"],
            }
            for s in scenario_results
        ],
        "monitor_cross_reference": {
            "recovery_probability": (monitor.get("executive_report") or {}).get(
                "recovery_probability"
            ),
            "distance_to_yellow": (monitor.get("distance_to_recovery") or {}).get(
                "distance_to_yellow"
            ),
        },
    }

    csv_rows = [
        {
            "scenario": s["scenario"],
            "projected_cps": s["projected_cps"],
            "cps_improvement": s["cps_improvement"],
            "concentration_reduction_pct": s["concentration_reduction_pct"],
            "governance_status": s["governance_status"],
            "escalation_level": s["escalation_level"],
            "distance_to_yellow": s["distance_to_yellow"],
            "distance_to_green": s["distance_to_green"],
            "recovery_score": s["recovery_score"],
            "classification": s["classification"],
        }
        for s in scenario_results
    ]
    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False, encoding="utf-8")

    summary = {
        "scenarios_run": len(scenario_results),
        "best_scenario": best.get("scenario"),
        "best_projected_cps": best.get("projected_cps"),
        "best_classification": best.get("classification"),
        "current_cps": current_cps,
        "distance_to_yellow_baseline": max(0, CPS_YELLOW - current_cps),
        "distance_to_green_baseline": max(0, CPS_GREEN - current_cps),
        "yellow_achievable": any(
            s["classification"] in ("YELLOW_RECOVERY", "GREEN_RECOVERY") for s in scenario_results
        ),
        "green_achievable": any(s["classification"] == "GREEN_RECOVERY" for s in scenario_results),
        "bito_in_scenario_f": bito_ok,
    }

    report = {
        "generated_at": ts,
        "current_state": current_state,
        "scenario_results": scenario_results,
        "threshold_discovery": threshold_discovery,
        "best_recovery_path": paths["BEST_RECOVERY_PATH"],
        "yellow_recovery_path": paths["FASTEST_PATH_TO_YELLOW"],
        "green_recovery_path": paths["FASTEST_PATH_TO_GREEN"],
        "executive_paths": paths,
        "forecasting_foundation": forecasting_foundation,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED" if scenario_results else "NOT_CERTIFIED",
            "safety": "analysis-only — no governance, execution, lifecycle, broker, or allocation modified",
        },
        "disclaimer": (
            "Simulation forecasts only — does not place trades or modify governance, execution, "
            "lifecycle, broker, authorization, or allocation rules."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_CSV.name}")
    _log(
        f"CPS={current_cps} best={best.get('projected_cps')} "
        f"class={best.get('classification')} scenarios={len(scenario_results)}"
    )
    return report


if __name__ == "__main__":
    run()
