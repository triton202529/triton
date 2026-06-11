"""
TRITON Phase 179 — Governance Recovery Threshold Optimizer (analysis only).

Determines the minimum diversification changes required to move governance from
ORANGE → YELLOW and YELLOW → GREEN while minimizing portfolio disruption.

SAFETY: analysis only — no governance, lifecycle, execution, allocation, broker,
or risk control changes.

Outputs (data/results/):
    governance_recovery_thresholds.json
    governance_recovery_thresholds.csv
    recovery_efficiency_curve.csv

Run:
    python -m services.governance_recovery_threshold_optimizer
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
    PERMIT_ESCALATION,
    _bito_eligible,
    _candidate_confidence,
    _find_min_combined_for_target,
    _find_min_weight_for_target,
    _open_new_tickers,
    _recovery_probability,
    _risk_score,
    _simulate_scenario,
)
from services.diversification_feasibility_analysis import MIN_OPEN_NEW_CONFIDENCE
from services.pnl_ledger_builder import _norm_sym, _to_float
from services.sector_exposure import get_sector

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "governance_recovery_thresholds.json"
OUT_CSV = RESULTS / "governance_recovery_thresholds.csv"
OUT_CURVE = RESULTS / "recovery_efficiency_curve.csv"

RECOVERY_FOCUS = ["HD", "XLK", "BITO"]


def _log(msg: str) -> None:
    print(f"[governance_recovery_threshold_optimizer] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _diversification_pct(weights: Dict[str, float]) -> float:
    return sum(v for k, v in weights.items() if k != "QQQ" and v > 0)


def _allocation_change_pct(weights: Dict[str, float]) -> float:
    """Total non-QQQ weight added vs QQQ-only baseline."""
    return _diversification_pct(weights)


def _efficiency_class(efficiency: float, achieves_yellow: bool, rank_pct: float) -> str:
    if achieves_yellow and rank_pct >= 0.90:
        return "OPTIMAL"
    if rank_pct >= 0.75:
        return "EFFICIENT"
    if rank_pct >= 0.40:
        return "MODERATE"
    return "INEFFICIENT"


def _build_weight_grid(
    candidates: List[str],
    bito_ok: bool,
    step: int = 5,
) -> List[Dict[str, float]]:
    """HD × XLK grid; optional BITO slice when eligible."""
    grid: List[Dict[str, float]] = []
    focus = [c for c in RECOVERY_FOCUS if c in candidates or c in ("HD", "XLK")]
    hd_xlk = [c for c in ("HD", "XLK") if c in candidates or c in ("HD", "XLK")]

    for hd in range(0, 41, step):
        for xlk in range(0, 41, step):
            if hd + xlk > 80:
                continue
            if bito_ok:
                for bito in range(0, 41, step):
                    if hd + xlk + bito > 85:
                        continue
                    qqq = 100.0 - hd - xlk - bito
                    if qqq < 15:
                        continue
                    grid.append(
                        {
                            "QQQ": float(qqq),
                            "HD": float(hd),
                            "XLK": float(xlk),
                            "BITO": float(bito),
                        }
                    )
            else:
                qqq = 100.0 - hd - xlk
                if qqq < 20:
                    continue
                w = {"QQQ": float(qqq), "HD": float(hd), "XLK": float(xlk)}
                grid.append(w)

    # Add equal-weight OPEN_NEW-only combinations (non focus)
    extra = [c for c in candidates if c not in hd_xlk and c != "BITO"]
    for sym in extra[:3]:
        for pct in range(step, 41, step):
            w = {"QQQ": 100.0 - pct, sym: float(pct)}
            grid.append(w)

    # Deduplicate
    seen: set = set()
    unique: List[Dict[str, float]] = []
    for w in grid:
        key = tuple(sorted((k, round(v, 2)) for k, v in w.items()))
        if key not in seen:
            seen.add(key)
            unique.append(w)
    return unique


def _evaluate_combination(
    weights: Dict[str, float],
    total_mv: float,
    base_components: Dict[str, int],
    baseline_largest: float,
    current_cps: int,
    recommendations: pd.DataFrame,
    feasibility: Dict[str, Any],
) -> Dict[str, Any]:
    sim = _simulate_scenario(weights, total_mv, base_components, baseline_largest, current_cps)
    alloc_change = _allocation_change_pct(weights)
    div_pct = alloc_change
    cps_imp = sim["cps_improvement"] or 0
    efficiency = float(cps_imp / alloc_change) if alloc_change > 0 else 0.0
    cost = float(alloc_change / max(cps_imp, 0.5)) if cps_imp > 0 else 999.0

    prob_label, prob_score, forecast_conf = _recovery_probability(sim, recommendations, feasibility)
    achieves_yellow = (sim.get("projected_cps") or 0) >= CPS_YELLOW
    achieves_green = (sim.get("projected_cps") or 0) >= CPS_GREEN

    return {
        "weights": sim["weights"],
        "allocation_change_pct": _f(alloc_change, 2),
        "diversification_pct": _f(div_pct, 2),
        "projected_cps": sim["projected_cps"],
        "cps_improvement": sim["cps_improvement"],
        "governance_status": sim["governance_status"],
        "escalation_level": sim["escalation_level"],
        "recovery_efficiency_score": _f(efficiency, 3),
        "recovery_cost_score": _f(cost, 3),
        "recovery_probability": prob_label,
        "recovery_probability_score": _f(prob_score * 100, 1),
        "forecast_confidence": _f(forecast_conf * 100, 1),
        "achieves_yellow": achieves_yellow,
        "achieves_green": achieves_green,
        "risk_score": _f(_risk_score(weights, recommendations), 2),
        "distance_to_yellow": sim["distance_to_yellow"],
        "distance_to_green": sim["distance_to_green"],
    }


def _find_minimum_plan(
    evaluations: List[Dict[str, Any]],
    target: str,
) -> Optional[Dict[str, Any]]:
    if target == "YELLOW":
        pool = [e for e in evaluations if e.get("achieves_yellow")]
    else:
        pool = [e for e in evaluations if e.get("achieves_green")]
    if not pool:
        return None
    best = min(pool, key=lambda e: _to_float(e.get("allocation_change_pct"), 999))
    return {
        "target": target,
        "weights": best.get("weights"),
        "allocation_change_pct": best.get("allocation_change_pct"),
        "diversification_pct": best.get("diversification_pct"),
        "projected_cps": best.get("projected_cps"),
        "escalation_level": best.get("escalation_level"),
        "governance_status": best.get("governance_status"),
        "recovery_efficiency_score": best.get("recovery_efficiency_score"),
        "recovery_probability": best.get("recovery_probability"),
        "minimum_hd_pct": best.get("weights", {}).get("HD"),
        "minimum_xlk_pct": best.get("weights", {}).get("XLK"),
        "minimum_bito_pct": best.get("weights", {}).get("BITO"),
    }


def _pick_optimized_path(
    evaluations: List[Dict[str, Any]],
    key: str,
) -> Dict[str, Any]:
    yellow_pool = [e for e in evaluations if e.get("achieves_yellow")]
    green_pool = [e for e in evaluations if e.get("achieves_green")]
    if not evaluations:
        return {}

    if key == "OPTIMAL_RECOVERY_PATH":
        pool = yellow_pool or evaluations
        best = max(pool, key=lambda e: _to_float(e.get("recovery_efficiency_score"), 0))
    elif key == "FASTEST_RECOVERY_PATH":
        pool = yellow_pool or evaluations
        best = min(pool, key=lambda e: _to_float(e.get("allocation_change_pct"), 999))
    elif key == "LOWEST_RISK_RECOVERY_PATH":
        pool = yellow_pool or evaluations
        best = min(pool, key=lambda e: _to_float(e.get("risk_score"), 999))
    elif key == "LOWEST_DISRUPTION_PATH":
        pool = yellow_pool or evaluations
        best = min(pool, key=lambda e: _to_float(e.get("allocation_change_pct"), 999))
    elif key == "HIGHEST_PROBABILITY_PATH":
        pool = yellow_pool or evaluations
        best = max(pool, key=lambda e: _to_float(e.get("recovery_probability_score"), 0))
    else:
        best = evaluations[0]

    return {
        "path_type": key,
        "weights": best.get("weights"),
        "allocation_change_pct": best.get("allocation_change_pct"),
        "projected_cps": best.get("projected_cps"),
        "escalation_level": best.get("escalation_level"),
        "governance_status": best.get("governance_status"),
        "recovery_efficiency_score": best.get("recovery_efficiency_score"),
        "recovery_probability": best.get("recovery_probability"),
    }


def _threshold_sensitivity(curve_df: pd.DataFrame) -> Dict[str, Any]:
    if curve_df.empty or len(curve_df) < 3:
        return {"cps_per_diversification_pct": None, "yellow_sensitivity": None}
    sub = curve_df.sort_values("diversification_pct").drop_duplicates("diversification_pct")
    if len(sub) < 2:
        return {"cps_per_diversification_pct": None, "yellow_sensitivity": None}
    dx = sub["diversification_pct"].diff().dropna()
    dy = sub["projected_cps"].diff().dropna()
    if dx.empty or (dx == 0).all():
        return {"cps_per_diversification_pct": None, "yellow_sensitivity": None}
    slope = float(dy.sum() / dx.sum()) if dx.sum() != 0 else None
    yellow_row = sub[sub["projected_cps"] >= CPS_YELLOW].head(1)
    yellow_div = float(yellow_row["diversification_pct"].iloc[0]) if not yellow_row.empty else None
    return {
        "cps_per_diversification_pct": _f(slope, 3),
        "yellow_sensitivity": _f(yellow_div, 2),
        "recovery_resilience": "HIGH" if slope and slope > 0.3 else "MODERATE" if slope else "LOW",
    }


def _candidate_contributions(
    evaluations: List[Dict[str, Any]],
    candidates: List[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    yellow = [e for e in evaluations if e.get("achieves_yellow")]
    for sym in candidates:
        if sym == "QQQ":
            continue
        weights_in_yellow = [
            _to_float((e.get("weights") or {}).get(sym), 0) for e in yellow if yellow
        ]
        out[sym] = {
            "avg_weight_in_yellow_plans_pct": (
                _f(np.mean(weights_in_yellow), 2) if weights_in_yellow else None
            ),
            "min_weight_for_yellow_pct": (
                _f(min(weights_in_yellow), 2) if weights_in_yellow else None
            ),
        }
    return out


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    simulation = _read_json(RESULTS / "concentration_recovery_simulation.json")
    monitor = _read_json(RESULTS / "recovery_basket_monitor.json")
    feasibility = _read_json(RESULTS / "diversification_feasibility_analysis.json")
    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    lifecycle = _read_csv(RESULTS / "signal_lifecycle_effective.csv")
    cpi = _read_json(RESULTS / "capital_preservation_intelligence.json")
    cpe = _read_json(RESULTS / "capital_preservation_escalation.json")
    auth = _read_json(RESULTS / "governance_authorization.json")
    paradox = _read_json(RESULTS / "concentration_governance_analysis.json")

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

    open_new = _open_new_tickers(recommendations)
    bito_ok = _bito_eligible(recommendations, lifecycle)
    candidates = list(dict.fromkeys(["QQQ"] + RECOVERY_FOCUS + open_new))

    current_state = {
        "generated_at": ts,
        "current_cps": current_cps,
        "current_escalation": current_esc,
        "current_governance_status": current_gov,
        "largest_concentration_pct": _f(baseline_largest, 2),
        "distance_to_yellow": max(0, CPS_YELLOW - current_cps),
        "distance_to_green": max(0, CPS_GREEN - current_cps),
        "open_new_candidates": open_new,
        "recovery_basket": (simulation.get("current_state") or {}).get("recovery_basket")
        or ["QQQ", "HD", "XLK", "BITO"],
        "concentration_trigger": (paradox.get("executive_report") or {}).get(
            "concentration_trigger"
        ),
    }

    weight_grid = _build_weight_grid(candidates, bito_ok)
    evaluations: List[Dict[str, Any]] = []
    for weights in weight_grid:
        ev = _evaluate_combination(
            weights,
            total_mv,
            base_components,
            baseline_largest,
            current_cps,
            recommendations,
            feasibility,
        )
        evaluations.append(ev)

    eff_values = sorted(
        [_to_float(e.get("recovery_efficiency_score"), 0) for e in evaluations], reverse=True
    )

    ranked: List[Dict[str, Any]] = []
    for ev in evaluations:
        eff = _to_float(ev.get("recovery_efficiency_score"), 0)
        rank_pct = eff_values.index(eff) / max(len(eff_values) - 1, 1) if eff in eff_values else 0.0
        classification = _efficiency_class(eff, bool(ev.get("achieves_yellow")), 1.0 - rank_pct)
        ranked.append({**ev, "classification": classification})

    ranked.sort(
        key=lambda e: (
            not e.get("achieves_yellow"),
            -_to_float(e.get("recovery_efficiency_score"), 0),
        )
    )

    min_hd_yellow = _find_min_weight_for_target(
        "HD", {"QQQ": 100.0}, total_mv, base_components, CPS_YELLOW, baseline_largest, current_cps
    )
    min_xlk_yellow = _find_min_weight_for_target(
        "XLK", {"QQQ": 100.0}, total_mv, base_components, CPS_YELLOW, baseline_largest, current_cps
    )
    min_combined_yellow = _find_min_combined_for_target(
        ["HD", "XLK"], total_mv, base_components, CPS_YELLOW, baseline_largest, current_cps
    )
    green_tickers = ["HD", "XLK", "BITO"] if bito_ok else ["HD", "XLK"]
    min_combined_green = _find_min_combined_for_target(
        green_tickers, total_mv, base_components, CPS_GREEN, baseline_largest, current_cps
    )

    yellow_plan = _find_minimum_plan(ranked, "YELLOW") or {
        "target": "YELLOW",
        "weights": min_combined_yellow,
        "minimum_hd_pct": min_hd_yellow,
        "minimum_xlk_pct": min_xlk_yellow,
        "note": "from threshold sweep",
    }
    green_plan = _find_minimum_plan(ranked, "GREEN") or {
        "target": "GREEN",
        "weights": min_combined_green,
        "note": "BITO promotion may be required" if not bito_ok else None,
    }

    yellow_threshold = {
        "cps_required": CPS_YELLOW,
        "minimum_yellow_recovery_plan": yellow_plan,
        "minimum_hd_allocation_pct": _f(min_hd_yellow, 1),
        "minimum_xlk_allocation_pct": _f(min_xlk_yellow, 1),
        "minimum_combined_diversification": min_combined_yellow,
        "minimum_allocation_change_pct": (
            yellow_plan.get("allocation_change_pct") if isinstance(yellow_plan, dict) else None
        ),
    }

    green_threshold = {
        "cps_required": CPS_GREEN,
        "minimum_green_recovery_plan": green_plan,
        "minimum_combined_diversification": min_combined_green,
        "bito_required": not bito_ok,
        "minimum_allocation_change_pct": (
            green_plan.get("allocation_change_pct") if isinstance(green_plan, dict) else None
        ),
    }

    paths = {
        "OPTIMAL_RECOVERY_PATH": _pick_optimized_path(ranked, "OPTIMAL_RECOVERY_PATH"),
        "FASTEST_RECOVERY_PATH": _pick_optimized_path(ranked, "FASTEST_RECOVERY_PATH"),
        "LOWEST_RISK_RECOVERY_PATH": _pick_optimized_path(ranked, "LOWEST_RISK_RECOVERY_PATH"),
        "LOWEST_DISRUPTION_PATH": _pick_optimized_path(ranked, "LOWEST_DISRUPTION_PATH"),
        "HIGHEST_PROBABILITY_PATH": _pick_optimized_path(ranked, "HIGHEST_PROBABILITY_PATH"),
    }
    optimal = paths["OPTIMAL_RECOVERY_PATH"]

    curve_rows: List[Dict[str, Any]] = []
    curve_seen: set = set()
    for ev in ranked:
        div = _to_float(ev.get("diversification_pct"), 0)
        key = (div, ev.get("projected_cps"))
        if key in curve_seen:
            continue
        curve_seen.add(key)
        curve_rows.append(
            {
                "diversification_pct": ev.get("diversification_pct"),
                "projected_cps": ev.get("projected_cps"),
                "governance_status": ev.get("governance_status"),
                "escalation_level": ev.get("escalation_level"),
                "recovery_probability": ev.get("recovery_probability"),
                "recovery_probability_score": ev.get("recovery_probability_score"),
                "allocation_change_pct": ev.get("allocation_change_pct"),
            }
        )
    curve_df = pd.DataFrame(curve_rows).sort_values("diversification_pct")
    curve_df.to_csv(OUT_CURVE, index=False, encoding="utf-8")

    sensitivity = _threshold_sensitivity(curve_df)
    contrib = _candidate_contributions(ranked, candidates)

    dist_opt = {
        "current_distance_to_yellow": current_state["distance_to_yellow"],
        "current_distance_to_green": current_state["distance_to_green"],
        "optimized_distance_to_yellow": optimal.get("projected_cps")
        and max(0, CPS_YELLOW - int(optimal.get("projected_cps") or 0)),
        "optimized_distance_to_green": optimal.get("projected_cps")
        and max(0, CPS_GREEN - int(optimal.get("projected_cps") or 0)),
        "required_allocation_change_pct": optimal.get("allocation_change_pct"),
        "expected_cps_improvement": (
            (optimal.get("projected_cps") or 0) - current_cps
            if optimal.get("projected_cps")
            else None
        ),
    }

    prob_label = str(optimal.get("recovery_probability") or "MODERATE")
    threshold_confidence = float(
        np.clip(
            (
                0.55
                + 0.20 * (1.0 if yellow_plan.get("weights") else 0.0)
                + 0.15 * (1.0 if min_combined_yellow else 0.0)
                + 0.10 * (monitor.get("recovery_probability") or {}).get("score_pct", 0) / 100.0
                if isinstance(monitor.get("recovery_probability"), dict)
                else 0.0
            ),
            0.4,
            0.95,
        )
    )

    recovery_probability = {
        "classification": prob_label,
        "optimal_path_probability": prob_label,
        "monitor_probability": (monitor.get("executive_report") or {}).get("recovery_probability"),
        "threshold_confidence_pct": _f(threshold_confidence * 100, 1),
    }

    executive = {
        "report_type": "GOVERNANCE_RECOVERY_THRESHOLD_REPORT",
        "current_governance_status": current_gov,
        "current_escalation": current_esc,
        "current_cps": current_cps,
        "minimum_yellow_recovery_plan": yellow_plan,
        "minimum_green_recovery_plan": green_plan,
        "optimal_recovery_path": optimal,
        "expected_cps_improvement": dist_opt.get("expected_cps_improvement"),
        "recovery_efficiency": optimal.get("recovery_efficiency_score"),
        "recovery_probability": prob_label,
        "threshold_confidence_pct": recovery_probability["threshold_confidence_pct"],
        "recommended_recovery_plan": {
            "path": "OPTIMAL_RECOVERY_PATH",
            "weights": optimal.get("weights"),
            "allocation_change_pct": optimal.get("allocation_change_pct"),
            "expected_escalation": optimal.get("escalation_level"),
            "expected_governance": optimal.get("governance_status"),
        },
        "expected_outcome": (
            f"CPS {current_cps} → {optimal.get('projected_cps')} "
            f"({current_esc} → {optimal.get('escalation_level')})"
            if optimal.get("projected_cps")
            else "—"
        ),
        "probability_of_success": prob_label,
    }

    csv_rows = [
        {
            "allocation_change_pct": r.get("allocation_change_pct"),
            "projected_cps": r.get("projected_cps"),
            "cps_improvement": r.get("cps_improvement"),
            "governance_status": r.get("governance_status"),
            "escalation_level": r.get("escalation_level"),
            "recovery_efficiency_score": r.get("recovery_efficiency_score"),
            "recovery_cost_score": r.get("recovery_cost_score"),
            "recovery_probability": r.get("recovery_probability"),
            "classification": r.get("classification"),
            "diversification_pct": r.get("diversification_pct"),
            "weights": json.dumps(r.get("weights") or {}),
        }
        for r in ranked[:200]
    ]
    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False, encoding="utf-8")

    summary = {
        "combinations_evaluated": len(ranked),
        "yellow_achievable": any(r.get("achieves_yellow") for r in ranked),
        "green_achievable": any(r.get("achieves_green") for r in ranked),
        "min_yellow_diversification_pct": yellow_plan.get("diversification_pct")
        or yellow_plan.get("allocation_change_pct"),
        "min_green_diversification_pct": (
            green_plan.get("diversification_pct") if isinstance(green_plan, dict) else None
        ),
        "optimal_allocation_change_pct": optimal.get("allocation_change_pct"),
        "optimal_projected_cps": optimal.get("projected_cps"),
        "bito_required_for_green": not bito_ok,
    }

    report = {
        "generated_at": ts,
        "executive_report": executive,
        "current_state": current_state,
        "yellow_threshold": yellow_threshold,
        "green_threshold": green_threshold,
        "optimal_recovery_path": optimal,
        "recovery_paths": paths,
        "efficiency_rankings": ranked[:50],
        "recovery_curve": curve_rows,
        "distance_optimization": dist_opt,
        "governance_threshold_analysis": {
            "cps_yellow_required": CPS_YELLOW,
            "cps_green_required": CPS_GREEN,
            "candidate_contributions": contrib,
            "threshold_sensitivity": sensitivity,
            "recovery_resilience": sensitivity.get("recovery_resilience"),
        },
        "recovery_probability": recovery_probability,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED" if ranked else "NOT_CERTIFIED",
            "safety": "analysis-only — no governance, execution, lifecycle, broker, or allocation modified",
        },
        "disclaimer": (
            "Threshold optimization only — does not place trades or modify governance, "
            "execution, lifecycle, broker, authorization, or allocation rules."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_CSV.name}, {OUT_CURVE.name}")
    _log(
        f"evaluated={len(ranked)} yellow_min={summary.get('min_yellow_diversification_pct')}% "
        f"optimal_cps={summary.get('optimal_projected_cps')}"
    )
    return report


if __name__ == "__main__":
    run()
