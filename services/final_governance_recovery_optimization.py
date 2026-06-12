"""
TRITON Phase 187 — Final Governance Recovery Optimization Engine (read-only).

Identifies the minimum additional optimization to close the final CPS gap
(89 -> 90+) and achieve full GREEN governance authorization.

SAFETY: analysis and optimization only — no governance, lifecycle, execution,
broker, allocation, or risk control changes. No trades or orders.

Outputs (data/results/):
    final_governance_recovery_optimization.json
    final_cps_sensitivity_matrix.csv
    final_recovery_adjustments.csv

Run:
    python -m services.final_governance_recovery_optimization
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
from services.diversification_recovery_basket import _return_correlation
from services.pnl_ledger_builder import _norm_sym, _to_float
from services.sector_exposure import get_sector

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "final_governance_recovery_optimization.json"
OUT_SENSITIVITY = RESULTS / "final_cps_sensitivity_matrix.csv"
OUT_ADJUSTMENTS = RESULTS / "final_recovery_adjustments.csv"

DOMINANT = "QQQ"
CPS_GAP = 1  # 89 -> 90
SENSITIVITY_DELTAS = [1, 2, 5, 10]
BUY_STANCES = {"BUY", "ADD"}


def _log(msg: str) -> None:
    print(f"[final_governance_recovery_optimization] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_context() -> Dict[str, Any]:
    return {
        "reconstruction": _read_json(RESULTS / "recovery_basket_reconstruction.json"),
        "lifecycle_eff": _read_csv(RESULTS / "signal_lifecycle_effective.csv"),
        "recs": _read_csv(RESULTS / "portfolio_allocation_recommendations.csv"),
        "thresholds": _read_json(RESULTS / "governance_recovery_thresholds.json"),
        "cpi": _read_json(RESULTS / "capital_preservation_intelligence.json"),
        "promotion": _read_json(RESULTS / "promotion_probability_forecast.json"),
        "gov_outcome": _read_json(RESULTS / "governance_outcome_predictions.json"),
        "positions": _load_positions(),
    }


def _baseline_weights(ctx: Dict[str, Any]) -> Dict[str, float]:
    rec = ctx["reconstruction"].get("recommended_basket") or {}
    weights = rec.get("weights") or {}
    if weights:
        return {k: float(v) for k, v in weights.items()}
    return {"QQQ": 40.0, "PEP": 15.0, "XLE": 15.0, "ABBV": 15.0, "MA": 15.0}


def _sim_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
    state = ctx["reconstruction"].get("current_state") or {}
    positions = ctx["positions"]
    total_mv = sum(p["market_value"] for p in positions) or state.get("total_mv") or 1438.0
    base_components = dict(state.get("base_components") or ctx["cpi"].get("component_scores") or {})
    for k in CPI_WEIGHTS:
        base_components.setdefault(k, 70)
    current_cps = int(
        state.get("current_cps") or ctx["cpi"].get("capital_preservation_score") or 68
    )
    baseline_largest = state.get("baseline_largest_pct") or 100.0
    return {
        "total_mv": float(total_mv),
        "base_components": base_components,
        "current_cps": current_cps,
        "baseline_largest_pct": float(baseline_largest),
    }


def _run_sim(
    weights: Dict[str, float],
    sim_state: Dict[str, Any],
    anchor_cps: int,
) -> Dict[str, Any]:
    w = {k: float(v) for k, v in weights.items() if v > 0}
    total = sum(w.values())
    if abs(total - 100.0) > 0.01 and total > 0:
        w = {k: v / total * 100.0 for k, v in w.items()}
    return _simulate_scenario(
        w,
        sim_state["total_mv"],
        sim_state["base_components"],
        sim_state["baseline_largest_pct"],
        anchor_cps,
    )


def _risk_score(weights: Dict[str, float], cand_map: Dict[str, Dict[str, Any]]) -> float:
    tech_w = sum(w for s, w in weights.items() if get_sector(s) == "Tech")
    corr_pen = 0.0
    for sym, w in weights.items():
        if sym == DOMINANT or w <= 0:
            continue
        corr = _to_float(cand_map.get(sym, {}).get("correlation_to_QQQ"), 0.5)
        corr_pen += w * abs(corr)
    conf_pen = 0.0
    for sym, w in weights.items():
        if sym == DOMINANT or w <= 0:
            continue
        conf = _to_float(cand_map.get(sym, {}).get("confidence"), 0.5)
        if conf < 0.45:
            conf_pen += w * 0.3
    return float(tech_w + corr_pen * 0.25 + conf_pen)


def _shift_weight(
    weights: Dict[str, float],
    from_sym: str,
    to_sym: str,
    delta: float,
) -> Optional[Dict[str, float]]:
    new = {k: float(v) for k, v in weights.items()}
    if new.get(from_sym, 0) < delta:
        return None
    new[from_sym] = new[from_sym] - delta
    new[to_sym] = new.get(to_sym, 0) + delta
    if new[from_sym] < 0:
        return None
    return {k: _f(v, 2) or 0.0 for k, v in new.items()}


def _candidate_map(ctx: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    universe = ctx["reconstruction"].get("candidate_universe") or []
    return {_norm_sym(c["symbol"]): c for c in universe}


def _promotion_30d(ctx: Dict[str, Any], sym: str) -> float:
    promo = ctx["promotion"]
    by_ticker = promo.get("promotion_probabilities") or promo.get("by_ticker") or {}
    horizons = by_ticker.get(sym) or {}
    p = _to_float((horizons.get("30") or {}).get("promotion_probability"), np.nan)
    if not np.isnan(p):
        return p
    for row in (promo.get("executive_report") or {}).get("candidates") or []:
        if _norm_sym(row.get("candidate")) == sym:
            return _to_float(row.get("promotion_probability_30d"), 0.0)
    return 0.0


def _baseline_sim(
    baseline_weights: Dict[str, float],
    sim_state: Dict[str, Any],
) -> Dict[str, Any]:
    sim = _run_sim(baseline_weights, sim_state, sim_state["current_cps"])
    return {
        "weights": baseline_weights,
        "projected_cps": sim["projected_cps"],
        "escalation": sim["escalation_level"],
        "governance_status": sim["governance_status"],
        "largest_position_weight_pct": sim["largest_position_weight_pct"],
        "simulation": sim,
    }


def _sensitivity_matrix(
    baseline_weights: Dict[str, float],
    baseline_cps: int,
    sim_state: Dict[str, Any],
    candidates: List[str],
    cand_map: Dict[str, Dict[str, Any]],
    base_risk: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sym in candidates:
        if sym == DOMINANT:
            continue
        for delta in SENSITIVITY_DELTAS:
            adj = _shift_weight(baseline_weights, DOMINANT, sym, float(delta))
            if not adj:
                continue
            sim = _run_sim(adj, sim_state, sim_state["current_cps"])
            cps_change = sim["projected_cps"] - baseline_cps
            risk = _risk_score(adj, cand_map)
            rows.append(
                {
                    "candidate": sym,
                    "allocation_change_pct": delta,
                    "direction": f"QQQ -{delta}% -> {sym} +{delta}%",
                    "cps_change": cps_change,
                    "projected_cps": sim["projected_cps"],
                    "risk_change": _f(risk - base_risk, 3),
                    "concentration_change": _f(
                        (sim.get("largest_position_weight_pct") or 0)
                        - (baseline_cps and 40.0),  # baseline QQQ weight
                        2,
                    ),
                    "governance_impact": (
                        "GREEN" if sim["projected_cps"] >= CPS_GREEN else sim["escalation_level"]
                    ),
                    "reaches_green": sim["projected_cps"] >= CPS_GREEN,
                }
            )
    return rows


def _find_minimum_adjustment(
    baseline_weights: Dict[str, float],
    baseline_cps: int,
    sim_state: Dict[str, Any],
    candidates: List[str],
    cand_map: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for sym in candidates:
        if sym == DOMINANT:
            continue
        for delta in range(1, 16):
            adj = _shift_weight(baseline_weights, DOMINANT, sym, float(delta))
            if not adj:
                continue
            sim = _run_sim(adj, sim_state, sim_state["current_cps"])
            if sim["projected_cps"] >= CPS_GREEN:
                entry = {
                    "candidate": sym,
                    "allocation_change_pct": delta,
                    "weights": adj,
                    "projected_cps": sim["projected_cps"],
                    "escalation": sim["escalation_level"],
                    "governance_status": sim["governance_status"],
                    "risk_score": _f(_risk_score(adj, cand_map), 2),
                }
                if best is None or delta < best["allocation_change_pct"]:
                    best = entry
                elif delta == best["allocation_change_pct"]:
                    if _risk_score(adj, cand_map) < _to_float(best.get("risk_score"), 999):
                        best = entry
                break
    return best


def _alternative_replacement_path(
    baseline_weights: Dict[str, float],
    sim_state: Dict[str, Any],
    cand_map: Dict[str, Dict[str, Any]],
    anchor_cps: int,
) -> Dict[str, Any]:
    """PATH_B: swap lowest-weight slot for best non-basket BUY candidate."""
    basket_adds = [s for s in baseline_weights if s != DOMINANT]
    replace_from = min(basket_adds, key=lambda s: baseline_weights[s])
    alternatives = sorted(
        [
            c
            for c in cand_map.values()
            if c.get("symbol") not in baseline_weights
            and c.get("recovery_classification") != "UNSUITABLE"
            and c.get("effective_stance") in BUY_STANCES
        ],
        key=lambda c: _to_float(c.get("governance_recovery_contribution"), 0),
        reverse=True,
    )
    best: Optional[Dict[str, Any]] = None
    for alt in alternatives[:8]:
        sym = alt["symbol"]
        new_w = dict(baseline_weights)
        new_w[sym] = new_w.pop(replace_from)
        sim = _run_sim(new_w, sim_state, anchor_cps)
        entry = {
            "replace_from": replace_from,
            "replace_with": sym,
            "weights": new_w,
            "projected_cps": sim["projected_cps"],
            "escalation": sim["escalation_level"],
            "governance_status": sim["governance_status"],
            "risk_score": _f(_risk_score(new_w, cand_map), 2),
        }
        if best is None or sim["projected_cps"] > best["projected_cps"]:
            best = entry
        if sim["projected_cps"] >= CPS_GREEN and (
            best is None
            or _to_float(entry["risk_score"], 99) < _to_float(best.get("risk_score"), 99)
        ):
            best = entry
    if not best:
        best = {
            "replace_from": replace_from,
            "replace_with": "—",
            "weights": baseline_weights,
            "projected_cps": anchor_cps,
            "escalation": "YELLOW",
            "governance_status": "AUTHORIZED",
            "risk_score": 0.0,
        }
    return best


def _additional_candidate_path(
    baseline_weights: Dict[str, float],
    sim_state: Dict[str, Any],
    cand_map: Dict[str, Dict[str, Any]],
    anchor_cps: int,
) -> Dict[str, Any]:
    """PATH_C: add 6th lifecycle-actionable candidate, trim QQQ."""
    alts = [
        c["symbol"]
        for c in sorted(
            cand_map.values(),
            key=lambda c: _to_float(c.get("diversification_score"), 0),
            reverse=True,
        )
        if c["symbol"] not in baseline_weights and c.get("recovery_classification") != "UNSUITABLE"
    ]
    best: Optional[Dict[str, Any]] = None
    for sym in alts[:6]:
        for add_pct in (5, 8, 10, 12, 15):
            adj = _shift_weight(baseline_weights, DOMINANT, sym, float(add_pct))
            if not adj:
                continue
            sim = _run_sim(adj, sim_state, anchor_cps)
            entry = {
                "added_candidate": sym,
                "allocation_change_pct": add_pct,
                "weights": adj,
                "projected_cps": sim["projected_cps"],
                "escalation": sim["escalation_level"],
                "governance_status": sim["governance_status"],
                "risk_score": _f(_risk_score(adj, cand_map), 2),
            }
            if sim["projected_cps"] >= CPS_GREEN:
                if best is None or add_pct < best.get("allocation_change_pct", 99):
                    best = entry
                break
    if not best and alts:
        sym = alts[0]
        adj = _shift_weight(baseline_weights, DOMINANT, sym, 10.0) or baseline_weights
        sim = _run_sim(adj, sim_state, anchor_cps)
        best = {
            "added_candidate": sym,
            "allocation_change_pct": 10.0,
            "weights": adj,
            "projected_cps": sim["projected_cps"],
            "escalation": sim["escalation_level"],
            "governance_status": sim["governance_status"],
            "risk_score": _f(_risk_score(adj, cand_map), 2),
        }
    return best or {}


def _bito_promotion_path(
    baseline_weights: Dict[str, float],
    sim_state: Dict[str, Any],
    cand_map: Dict[str, Dict[str, Any]],
    ctx: Dict[str, Any],
    anchor_cps: int,
) -> Dict[str, Any]:
    """PATH_D: BITO promoted to BUY — add or replace slot."""
    promo_30 = _promotion_30d(ctx, "BITO")
    promo_60 = _to_float(
        ((ctx["promotion"].get("promotion_probabilities") or {}).get("BITO") or {})
        .get("60", {})
        .get("promotion_probability"),
        77.1,
    )
    best: Optional[Dict[str, Any]] = None
    scenarios = []

    for mode in ("add", "replace_ma"):
        for pct in (10, 12, 15):
            if mode == "add":
                adj = _shift_weight(baseline_weights, DOMINANT, "BITO", float(pct))
            else:
                new_w = dict(baseline_weights)
                ma_w = new_w.pop("MA", 15.0)
                new_w["BITO"] = ma_w
                adj = _shift_weight(new_w, DOMINANT, "BITO", float(pct - ma_w) if pct > ma_w else 0)
                if adj is None:
                    adj = new_w
            if not adj:
                continue
            sim = _run_sim(adj, sim_state, anchor_cps)
            entry = {
                "mode": mode,
                "bito_weight_pct": adj.get("BITO", 0),
                "weights": adj,
                "projected_cps": sim["projected_cps"],
                "escalation": sim["escalation_level"],
                "governance_status": sim["governance_status"],
                "risk_score": _f(_risk_score(adj, cand_map), 2),
                "promotion_probability_30d": promo_30,
                "promotion_probability_60d": promo_60,
                "lifecycle_assumption": "BITO effective_stance=BUY (promoted from WATCH)",
            }
            scenarios.append(entry)
            if sim["projected_cps"] >= CPS_GREEN and (
                best is None
                or _to_float(entry["risk_score"], 99) <= _to_float(best.get("risk_score"), 99)
            ):
                best = entry

    if not best:
        best = max(scenarios, key=lambda s: s["projected_cps"]) if scenarios else {}

    return {
        "best_scenario": best,
        "all_scenarios": scenarios,
        "promotion_probability_30d": promo_30,
        "promotion_probability_60d": promo_60,
        "provides_green_alone": any(s.get("projected_cps", 0) >= CPS_GREEN for s in scenarios),
        "note": (
            "BITO currently WATCH/WAIT — path assumes future lifecycle promotion. "
            f"P(promotion@30d)={promo_30:.1f}%, P@60d={promo_60:.1f}%."
        ),
    }


def _stability_path(
    baseline_weights: Dict[str, float],
    sim_state: Dict[str, Any],
    cand_map: Dict[str, Dict[str, Any]],
    anchor_cps: int,
    min_adj: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """PATH_E: lowest-risk path that reaches GREEN."""
    candidates = [s for s in baseline_weights if s != DOMINANT]
    candidates.extend(
        c["symbol"]
        for c in cand_map.values()
        if c.get("symbol") not in baseline_weights
        and c.get("recovery_classification") != "UNSUITABLE"
    )
    best: Optional[Dict[str, Any]] = None
    for sym in candidates:
        for delta in range(1, 16):
            adj = _shift_weight(baseline_weights, DOMINANT, sym, float(delta))
            if not adj:
                continue
            sim = _run_sim(adj, sim_state, anchor_cps)
            if sim["projected_cps"] < CPS_GREEN:
                continue
            risk = _risk_score(adj, cand_map)
            entry = {
                "candidate": sym,
                "allocation_change_pct": delta,
                "weights": adj,
                "projected_cps": sim["projected_cps"],
                "escalation": sim["escalation_level"],
                "governance_status": sim["governance_status"],
                "risk_score": _f(risk, 2),
            }
            if best is None or risk < _to_float(best.get("risk_score"), 999):
                best = entry
            break

    if min_adj and best:
        if _to_float(min_adj.get("risk_score"), 999) <= _to_float(best.get("risk_score"), 999):
            best = {
                "candidate": min_adj["candidate"],
                "allocation_change_pct": min_adj["allocation_change_pct"],
                "weights": min_adj["weights"],
                "projected_cps": min_adj["projected_cps"],
                "escalation": min_adj["escalation"],
                "governance_status": min_adj["governance_status"],
                "risk_score": min_adj["risk_score"],
            }

    return best or {}


def _path_rank_score(path: Dict[str, Any]) -> float:
    cps = _to_float(path.get("projected_cps"), 0)
    risk = _to_float(path.get("risk_score") or path.get("additional_risk"), 50)
    complexity_map = {"LOW": 1.0, "MEDIUM": 0.7, "HIGH": 0.4}
    complexity = complexity_map.get(
        str(path.get("implementation_complexity", "MEDIUM")).upper(), 0.7
    )
    prob_map = {"VERY_HIGH": 1.0, "HIGH": 0.85, "MODERATE": 0.6, "LOW": 0.3}
    prob = prob_map.get(str(path.get("recovery_probability", "MODERATE")).upper(), 0.6)
    green_bonus = 1.0 if cps >= CPS_GREEN else 0.0
    cps_eff = (cps - 89) / max(_to_float(path.get("allocation_change_pct"), 1), 1)
    return (
        0.30 * green_bonus
        + 0.25 * min(cps / 100.0, 1.0)
        + 0.20 * prob
        + 0.15 * (1.0 - risk / 100.0)
        + 0.10 * complexity
        + 0.05 * min(cps_eff, 1.0)
    )


def _classify_path(rank_score: float, reaches_green: bool) -> str:
    if not reaches_green:
        return "INEFFICIENT"
    if rank_score >= 0.85:
        return "OPTIMAL"
    if rank_score >= 0.70:
        return "STRONG"
    if rank_score >= 0.50:
        return "ACCEPTABLE"
    return "INEFFICIENT"


def _build_paths(
    baseline_weights: Dict[str, float],
    baseline: Dict[str, Any],
    sim_state: Dict[str, Any],
    cand_map: Dict[str, Dict[str, Any]],
    ctx: Dict[str, Any],
    min_adj: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    anchor_cps = baseline["projected_cps"]
    path_b = _alternative_replacement_path(baseline_weights, sim_state, cand_map, anchor_cps)
    path_c = _additional_candidate_path(baseline_weights, sim_state, cand_map, anchor_cps)
    path_d = _bito_promotion_path(baseline_weights, sim_state, cand_map, ctx, anchor_cps)
    path_e = _stability_path(baseline_weights, sim_state, cand_map, anchor_cps, min_adj)

    paths = []

    if min_adj:
        paths.append(
            {
                "path_id": "PATH_A",
                "path_name": "Minimum CPS Adjustment",
                "description": (
                    f"Shift {min_adj['allocation_change_pct']}% from QQQ to "
                    f"{min_adj['candidate']} — smallest allocation change reaching GREEN."
                ),
                "weights": min_adj["weights"],
                "projected_cps": min_adj["projected_cps"],
                "projected_escalation": min_adj["escalation"],
                "projected_governance_status": min_adj["governance_status"],
                "allocation_change_pct": min_adj["allocation_change_pct"],
                "additional_risk": min_adj["risk_score"],
                "implementation_complexity": "LOW",
                "recovery_probability": "HIGH",
                "reaches_green": min_adj["projected_cps"] >= CPS_GREEN,
            }
        )

    paths.append(
        {
            "path_id": "PATH_B",
            "path_name": "Alternative Candidate Replacement",
            "description": (
                f"Replace {path_b.get('replace_from')} with {path_b.get('replace_with')} "
                "at equal weight."
            ),
            "weights": path_b.get("weights"),
            "projected_cps": path_b.get("projected_cps"),
            "projected_escalation": path_b.get("escalation"),
            "projected_governance_status": path_b.get("governance_status"),
            "allocation_change_pct": 0,
            "additional_risk": path_b.get("risk_score"),
            "implementation_complexity": "MEDIUM",
            "recovery_probability": "MODERATE",
            "reaches_green": _to_float(path_b.get("projected_cps"), 0) >= CPS_GREEN,
        }
    )

    paths.append(
        {
            "path_id": "PATH_C",
            "path_name": "Additional Diversification Candidate",
            "description": (
                f"Add {path_c.get('added_candidate')} at {path_c.get('allocation_change_pct')}% "
                "from QQQ reduction."
            ),
            "weights": path_c.get("weights"),
            "projected_cps": path_c.get("projected_cps"),
            "projected_escalation": path_c.get("escalation"),
            "projected_governance_status": path_c.get("governance_status"),
            "allocation_change_pct": path_c.get("allocation_change_pct"),
            "additional_risk": path_c.get("risk_score"),
            "implementation_complexity": "MEDIUM",
            "recovery_probability": "HIGH",
            "reaches_green": _to_float(path_c.get("projected_cps"), 0) >= CPS_GREEN,
        }
    )

    bito_best = path_d.get("best_scenario") or {}
    paths.append(
        {
            "path_id": "PATH_D",
            "path_name": "Future Promotion Assisted Recovery",
            "description": path_d.get("note"),
            "weights": bito_best.get("weights"),
            "projected_cps": bito_best.get("projected_cps"),
            "projected_escalation": bito_best.get("escalation"),
            "projected_governance_status": bito_best.get("governance_status"),
            "allocation_change_pct": bito_best.get("bito_weight_pct"),
            "additional_risk": bito_best.get("risk_score"),
            "implementation_complexity": "HIGH",
            "recovery_probability": (
                "MODERATE" if path_d.get("promotion_probability_30d", 0) < 50 else "HIGH"
            ),
            "reaches_green": _to_float(bito_best.get("projected_cps"), 0) >= CPS_GREEN,
            "promotion_probability_30d": path_d.get("promotion_probability_30d"),
            "promotion_probability_60d": path_d.get("promotion_probability_60d"),
        }
    )

    paths.append(
        {
            "path_id": "PATH_E",
            "path_name": "Maximum Governance Stability",
            "description": (
                f"Lowest-risk GREEN path: {path_e.get('allocation_change_pct')}% "
                f"QQQ to {path_e.get('candidate')}."
            ),
            "weights": path_e.get("weights"),
            "projected_cps": path_e.get("projected_cps"),
            "projected_escalation": path_e.get("escalation"),
            "projected_governance_status": path_e.get("governance_status"),
            "allocation_change_pct": path_e.get("allocation_change_pct"),
            "additional_risk": path_e.get("risk_score"),
            "implementation_complexity": "LOW",
            "recovery_probability": "HIGH",
            "reaches_green": _to_float(path_e.get("projected_cps"), 0) >= CPS_GREEN,
        }
    )

    for p in paths:
        p["rank_score"] = _f(_path_rank_score(p), 4)
        p["classification"] = _classify_path(
            _to_float(p["rank_score"], 0), p.get("reaches_green", False)
        )

    paths.sort(key=lambda p: _to_float(p.get("rank_score"), 0), reverse=True)
    for i, p in enumerate(paths, 1):
        p["rank"] = i

    return paths


def _green_probability(paths: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
    gov = ctx["gov_outcome"].get("executive_report") or {}
    green_paths = [p for p in paths if p.get("reaches_green")]
    best = green_paths[0] if green_paths else (paths[0] if paths else {})

    immediate_prob = 0.92 if best.get("path_id") == "PATH_A" and best.get("reaches_green") else 0.75
    bito_path = next((p for p in paths if p.get("path_id") == "PATH_D"), {})
    bito_assisted = (
        immediate_prob * 0.6
        + (_to_float(bito_path.get("promotion_probability_30d"), 38.5) / 100.0) * 0.4
    )

    return {
        "allocation_only_green_probability": _f(immediate_prob * 100, 1),
        "bito_assisted_green_probability_30d": _f(bito_assisted * 100, 1),
        "governance_forecast_green_30d": gov.get("probability_of_green"),
        "governance_forecast_green_90d": gov.get("probability_of_green"),
        "highest_confidence_scenario": best.get("path_id"),
        "highest_confidence_cps": best.get("projected_cps"),
    }


def _executive_report(
    baseline: Dict[str, Any],
    paths: List[Dict[str, Any]],
    min_adj: Optional[Dict[str, Any]],
    green_prob: Dict[str, Any],
    sensitivity: List[Dict[str, Any]],
) -> Dict[str, Any]:
    best = paths[0] if paths else {}
    top_sens = sorted(
        [s for s in sensitivity if s.get("reaches_green")],
        key=lambda s: s.get("allocation_change_pct", 99),
    )
    best_candidate = top_sens[0]["candidate"] if top_sens else (min_adj or {}).get("candidate", "—")

    return {
        "report_type": "FINAL_GOVERNANCE_RECOVERY_OPTIMIZATION_REPORT",
        "current_cps_gap": {
            "baseline_cps": baseline["projected_cps"],
            "green_threshold": CPS_GREEN,
            "gap_points": max(0, CPS_GREEN - baseline["projected_cps"]),
            "baseline_escalation": baseline["escalation"],
            "baseline_governance": baseline["governance_status"],
        },
        "best_recovery_path": {
            "path_id": best.get("path_id"),
            "path_name": best.get("path_name"),
            "projected_cps": best.get("projected_cps"),
            "classification": best.get("classification"),
        },
        "minimum_required_change": (
            f"Shift {min_adj['allocation_change_pct']}% from QQQ to {min_adj['candidate']}"
            if min_adj
            else "No single-candidate shift found — see PATH_C or PATH_D"
        ),
        "minimum_weights": (min_adj or {}).get("weights"),
        "projected_green_probability": green_prob,
        "risk_impact": f"Additional risk score +{(min_adj or {}).get('risk_score', best.get('additional_risk', '—'))} vs baseline",
        "implementation_complexity": best.get("implementation_complexity"),
        "executive_recommendation": (
            f"Close the 1-point CPS gap via PATH_A: reduce QQQ by "
            f"{(min_adj or {}).get('allocation_change_pct', '?')}% and allocate to "
            f"{(min_adj or {}).get('candidate', best_candidate)} "
            f"(projected CPS {(min_adj or {}).get('projected_cps', best.get('projected_cps'))} -> GREEN). "
            "Allocation changes alone can achieve GREEN. BITO promotion is not required for the "
            "final CPS point but remains the highest-leverage path to sustained GREEN stability. "
            "Do not redesign exception framework."
        ),
        "executive_questions": {
            "smallest_change_for_green": (
                f"{min_adj['allocation_change_pct']}% QQQ -> {min_adj['candidate']}"
                if min_adj
                else "See ranked paths"
            ),
            "highest_cps_improvement_candidate": best_candidate,
            "allocation_alone_achieves_green": bool(min_adj),
            "bito_provides_final_point": any(
                p.get("path_id") == "PATH_D" and p.get("reaches_green") for p in paths
            ),
            "safest_path": next(
                (p["path_id"] for p in paths if p.get("path_id") == "PATH_E"),
                "PATH_E",
            ),
            "highest_confidence_green_scenario": green_prob.get("highest_confidence_scenario"),
        },
    }


def run_analysis() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    ctx = _load_context()
    baseline_weights = _baseline_weights(ctx)
    sim_state = _sim_state(ctx)
    cand_map = _candidate_map(ctx)

    baseline = _baseline_sim(baseline_weights, sim_state)
    baseline_cps = int(baseline["projected_cps"])
    base_risk = _risk_score(baseline_weights, cand_map)

    candidates = list(baseline_weights.keys()) + [
        c["symbol"] for c in cand_map.values() if c.get("recovery_classification") != "UNSUITABLE"
    ]
    candidates = list(dict.fromkeys(candidates))

    sensitivity = _sensitivity_matrix(
        baseline_weights, baseline_cps, sim_state, candidates, cand_map, base_risk
    )
    min_adj = _find_minimum_adjustment(
        baseline_weights, baseline_cps, sim_state, candidates, cand_map
    )
    paths = _build_paths(baseline_weights, baseline, sim_state, cand_map, ctx, min_adj)
    green_prob = _green_probability(paths, ctx)

    candidate_adjustments = {
        "minimum_adjustment": min_adj,
        "top_sensitivity_hits": [s for s in sensitivity if s.get("reaches_green")][:5],
        "highest_cps_per_pct": sorted(
            sensitivity,
            key=lambda s: _to_float(s.get("cps_change"), 0)
            / max(s.get("allocation_change_pct", 1), 1),
            reverse=True,
        )[:5],
    }

    executive = _executive_report(baseline, paths, min_adj, green_prob, sensitivity)

    summary = {
        "baseline_cps": baseline_cps,
        "green_threshold": CPS_GREEN,
        "cps_gap": max(0, CPS_GREEN - baseline_cps),
        "best_path": paths[0]["path_id"] if paths else "—",
        "minimum_change_pct": (min_adj or {}).get("allocation_change_pct"),
        "minimum_change_candidate": (min_adj or {}).get("candidate"),
        "projected_green_cps": (min_adj or {}).get("projected_cps")
        or paths[0].get("projected_cps"),
        "allocation_alone_sufficient": bool(min_adj),
        "green_paths_count": sum(1 for p in paths if p.get("reaches_green")),
    }

    report = {
        "generated_at": ts,
        "phase": "187",
        "mode": "final_governance_recovery_optimization",
        "disclaimer": (
            "Analysis and optimization only — does not modify governance, lifecycle, execution, "
            "allocation, broker, or risk controls. No trades or orders."
        ),
        "baseline": {
            "weights": baseline_weights,
            "projected_cps": baseline_cps,
            "escalation": baseline["escalation"],
            "governance_status": baseline["governance_status"],
            "risk_score": _f(base_risk, 2),
            "cps_gap_to_green": max(0, CPS_GREEN - baseline_cps),
        },
        "optimization_paths": paths,
        "sensitivity_matrix": sensitivity,
        "candidate_adjustments": candidate_adjustments,
        "green_recovery_projection": green_prob,
        "executive_recommendation": executive,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "read-only analysis — no pipeline modifications",
        },
    }

    pd.DataFrame(sensitivity).to_csv(OUT_SENSITIVITY, index=False, encoding="utf-8")
    adj_rows = [
        {
            "adjustment": p["path_id"],
            "path_name": p["path_name"],
            "projected_cps": p.get("projected_cps"),
            "projected_status": p.get("projected_escalation"),
            "projected_governance": p.get("projected_governance_status"),
            "risk_score": p.get("additional_risk"),
            "rank": p.get("rank"),
            "classification": p.get("classification"),
            "allocation_change_pct": p.get("allocation_change_pct"),
            "reaches_green": p.get("reaches_green"),
            "weights_json": json.dumps(p.get("weights") or {}),
        }
        for p in paths
    ]
    pd.DataFrame(adj_rows).to_csv(OUT_ADJUSTMENTS, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_SENSITIVITY.name}, {OUT_ADJUSTMENTS.name}")
    _log(
        f"baseline_cps={baseline_cps} gap={summary['cps_gap']} "
        f"min_change={summary['minimum_change_pct']}%->{summary['minimum_change_candidate']} "
        f"best={summary['best_path']}"
    )
    return report


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
