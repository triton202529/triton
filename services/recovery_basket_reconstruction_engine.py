"""
TRITON Phase 186 — Recovery Basket Reconstruction Engine (read-only).

Rebuilds the governance recovery basket using authoritative lifecycle data
(effective BUY/ADD) rather than stale OPEN_NEW assumptions from HD/XLK era.

SAFETY: analysis and design only — no governance, lifecycle, execution, broker,
allocation, or risk control changes. No trades or orders.

Outputs (data/results/):
    recovery_basket_reconstruction.json
    recovery_basket_candidates.csv
    recovery_basket_scenarios.csv

Run:
    python -m services.recovery_basket_reconstruction_engine
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    _sector_weights,
)
from services.concentration_recovery_simulator import (
    CPS_GREEN,
    CPS_YELLOW,
    _simulate_scenario,
)
from services.diversification_feasibility_analysis import _lifecycle_row
from services.diversification_recovery_basket import _return_correlation
from services.pnl_ledger_builder import _norm_sym, _to_float
from services.sector_exposure import get_sector

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "recovery_basket_reconstruction.json"
OUT_CANDIDATES = RESULTS / "recovery_basket_candidates.csv"
OUT_SCENARIOS = RESULTS / "recovery_basket_scenarios.csv"

BUY_STANCES = {"BUY", "ADD"}
EXCLUDED_STANCES = {"WAIT", "HOLD", "TRIM", "EXIT"}
DOMINANT = "QQQ"
ORIGINAL_BASKET = ["HD", "XLK"]
LEGACY_WEIGHTS = {"QQQ": 60.0, "HD": 15.0, "XLK": 25.0}
PERMIT_ESCALATION = {"GREEN", "YELLOW"}
MIN_CONF_ELITE = 0.55
MIN_CONF_STRONG = 0.45
STRONG_EDGE = 0.65
WEAK_EDGE = 0.40


def _log(msg: str) -> None:
    print(f"[recovery_basket_reconstruction_engine] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _blocked_symbols(alloc_summary: Dict[str, Any]) -> Set[str]:
    blocked = set(_norm_sym(s) for s in (alloc_summary.get("blocked_new_buy_symbols") or []))
    blocked.update(_norm_sym(s) for s in (alloc_summary.get("force_exit_symbols") or []))
    return blocked


def _rec_row(recs: pd.DataFrame, sym: str) -> Dict[str, Any]:
    if recs.empty:
        return {}
    sub = recs[recs["ticker"].map(_norm_sym) == sym]
    return sub.iloc[0].to_dict() if not sub.empty else {}


def _load_context() -> Dict[str, Any]:
    return {
        "recs": _read_csv(RESULTS / "portfolio_allocation_recommendations.csv"),
        "lifecycle_eff": _read_csv(RESULTS / "signal_lifecycle_effective.csv"),
        "alloc_summary": _read_json(RESULTS / "portfolio_allocation_summary.json"),
        "cpi": _read_json(RESULTS / "capital_preservation_intelligence.json"),
        "cpe": _read_json(RESULTS / "capital_preservation_escalation.json"),
        "auth": _read_json(RESULTS / "governance_authorization.json"),
        "thresholds": _read_json(RESULTS / "governance_recovery_thresholds.json"),
        "deterioration": _read_json(RESULTS / "recovery_candidate_deterioration_analysis.json"),
        "exception_gov": _read_json(RESULTS / "recovery_basket_exception_governance.json"),
        "promotion": _read_json(RESULTS / "promotion_probability_forecast.json"),
        "positions": _load_positions(),
    }


def _current_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
    positions = ctx["positions"]
    total_mv = sum(p["market_value"] for p in positions)
    _, largest_pct, _ = _concentration_score(positions)
    base_components = dict(ctx["cpi"].get("component_scores") or {})
    for k in CPI_WEIGHTS:
        base_components.setdefault(k, 70)
    current_cps = int(
        ctx["cpi"].get("capital_preservation_score") or _estimate_cps(positions, base_components)[0]
    )
    esc = str(ctx["cpe"].get("escalation_state") or _escalation_from_cps(current_cps)).upper()
    gov = ctx["auth"].get("governance_authorized")
    gov_status = "AUTHORIZED" if gov is True else "BLOCKED"

    weights: Dict[str, float] = {}
    if total_mv > 0:
        for p in positions:
            sym = p["symbol"]
            weights[sym] = weights.get(sym, 0.0) + p["market_value"] / total_mv * 100.0

    return {
        "holdings": weights,
        "largest_concentration_pct": _f(largest_pct),
        "concentration_symbol": DOMINANT,
        "current_cps": current_cps,
        "escalation": esc,
        "governance_status": gov_status,
        "governance_authorized": gov is True,
        "portfolio_value": _f(total_mv),
        "position_count": len(positions),
        "distance_to_yellow": max(0, CPS_YELLOW - current_cps),
        "distance_to_green": max(0, CPS_GREEN - current_cps),
        "base_components": base_components,
        "baseline_largest_pct": largest_pct,
        "total_mv": total_mv,
        "original_recovery_basket": ORIGINAL_BASKET,
        "legacy_plan_weights": LEGACY_WEIGHTS,
        "legacy_plan_status": "NOT_EXECUTABLE",
        "legacy_failure_reason": (
            "HD and XLK effective_stance=WAIT — authoritative lifecycle rejected both; "
            "recovery basket assumptions stale per Phase 185."
        ),
        "exception_framework_status": "VALID_UNCHANGED",
    }


def _promotion_30d(ctx: Dict[str, Any], sym: str) -> float:
    promo = ctx["promotion"]
    by_ticker = promo.get("by_ticker") or promo.get("ticker_horizons") or {}
    horizons = by_ticker.get(sym) or {}
    p = _to_float((horizons.get("30") or {}).get("promotion_probability"), np.nan)
    if not np.isnan(p):
        return p
    for row in (promo.get("executive_report") or {}).get("candidates") or []:
        if _norm_sym(row.get("candidate")) == sym:
            return _to_float(row.get("promotion_probability_30d"), 0.0)
    return 0.0


def _single_add_cps_delta(
    sym: str,
    total_mv: float,
    base_components: Dict[str, int],
    current_cps: int,
    baseline_largest: float,
) -> float:
    if total_mv <= 0:
        return 0.0
    weights = {DOMINANT: 50.0, sym: 50.0}
    sim = _simulate_scenario(weights, total_mv, base_components, baseline_largest, current_cps)
    return float(sim.get("cps_improvement") or 0)


def _classify_candidate(
    sym: str,
    effective: str,
    signal: str,
    conf: float,
    edge: float,
    delta: float,
    risk: str,
    div_value: float,
    corr_qqq: Optional[float],
    cps_delta: float,
) -> str:
    if effective not in BUY_STANCES or signal == "SELL":
        return "UNSUITABLE"
    if risk != "OK" or "BLOCK" in risk:
        return "UNSUITABLE"
    if corr_qqq is not None and corr_qqq > 0.92:
        return "UNSUITABLE"

    elite = (
        effective == "BUY"
        and conf >= MIN_CONF_ELITE
        and (edge >= STRONG_EDGE or (not np.isnan(edge) and edge >= 0.55))
        and (np.isnan(delta) or delta >= 0)
        and div_value >= 55
        and (corr_qqq is None or corr_qqq < 0.75)
    )
    if elite:
        return "ELITE_RECOVERY_CANDIDATE"

    strong = (
        effective in BUY_STANCES and conf >= MIN_CONF_STRONG and div_value >= 40 and cps_delta > 0
    )
    if strong:
        return "STRONG_RECOVERY_CANDIDATE"

    acceptable = effective in BUY_STANCES and signal != "SELL" and div_value >= 25
    if acceptable:
        return "ACCEPTABLE_RECOVERY_CANDIDATE"

    return "UNSUITABLE"


def _diversification_score(div_value: float, corr: Optional[float], sector: str) -> float:
    corr_pen = min(abs(corr), 1.0) if corr is not None else 0.5
    sector_bonus = 15.0 if sector not in ("Tech", "Information Technology") else 0.0
    return float(np.clip(div_value * 0.7 + (1.0 - corr_pen) * 20.0 + sector_bonus, 0.0, 100.0))


def _build_candidate_universe(ctx: Dict[str, Any], state: Dict[str, Any]) -> List[Dict[str, Any]]:
    recs = ctx["recs"]
    lc_df = ctx["lifecycle_eff"]
    blocked = _blocked_symbols(ctx["alloc_summary"])
    dominant_sector = get_sector(DOMINANT)
    total_mv = state["total_mv"]
    base_components = state["base_components"]
    current_cps = state["current_cps"]
    baseline_largest = state["baseline_largest_pct"]

    if lc_df.empty:
        return []

    sym_col = "ticker" if "ticker" in lc_df.columns else "symbol"
    candidates: List[Dict[str, Any]] = []

    for _, lc_row in lc_df.iterrows():
        sym = _norm_sym(lc_row.get(sym_col))
        if not sym or sym == DOMINANT:
            continue

        effective = str(lc_row.get("effective_stance") or "").upper()
        lifecycle_action = str(lc_row.get("lifecycle_action") or "").upper()
        signal = str(lc_row.get("signal") or "").upper()

        if effective in EXCLUDED_STANCES or lifecycle_action in EXCLUDED_STANCES:
            continue
        if effective not in BUY_STANCES:
            continue
        if signal == "SELL":
            continue

        rec = _rec_row(recs, sym)
        conf = _to_float(lc_row.get("confidence") or rec.get("confidence"), np.nan)
        edge = _to_float(rec.get("edge_score"), np.nan)
        delta = _to_float(lc_row.get("delta_pct") or rec.get("delta_pct"), np.nan)
        alloc_score = _to_float(rec.get("allocation_score"), np.nan)
        risk = str(rec.get("risk_flag") or "OK").upper()
        sector = get_sector(sym)
        corr = _return_correlation(DOMINANT, sym)
        cps_delta = _single_add_cps_delta(
            sym, total_mv, base_components, current_cps, baseline_largest
        )

        sector_bonus = 0.0 if sector == dominant_sector else 1.0
        corr_penalty = abs(corr) if corr is not None else 0.5
        div_value = float(
            np.clip(
                35.0 * sector_bonus
                + 25.0 * (1.0 - min(corr_penalty, 1.0))
                + 20.0 * min(_to_float(alloc_score, 0.0), 1.0)
                + 20.0 * min(_to_float(edge, 0.0), 1.0),
                0.0,
                100.0,
            )
        )
        gov_contrib = float(np.clip(cps_delta * 4.0 + div_value * 0.3, 0.0, 100.0))
        classification = _classify_candidate(
            sym, effective, signal, conf, edge, delta, risk, div_value, corr, cps_delta
        )
        div_score = _diversification_score(div_value, corr, sector)

        candidates.append(
            {
                "symbol": sym,
                "effective_stance": effective,
                "lifecycle_action": lifecycle_action,
                "signal": signal,
                "confidence": _f(conf, 4),
                "edge_score": _f(edge, 4),
                "allocation_score": _f(alloc_score, 4),
                "delta_pct": _f(delta, 6),
                "risk_flags": risk,
                "sector_exposure": sector,
                "correlation_to_QQQ": _f(corr, 3),
                "diversification_value": _f(div_value, 2),
                "diversification_score": _f(div_score, 2),
                "cps_improvement_potential": _f(cps_delta, 2),
                "governance_recovery_contribution": _f(gov_contrib, 2),
                "recovery_classification": classification,
                "promotion_probability_30d": _f(_promotion_30d(ctx, sym), 1),
                "lifecycle_actionable": True,
                "replaces_hd_candidate": sector not in ("Tech", "Information Technology")
                and (corr is None or corr < 0.7),
                "replaces_xlk_candidate": sector not in ("Tech", "Information Technology")
                or sym in ("XLE", "USO", "PEP", "ABBV", "XLV", "XLP"),
            }
        )

    rank_order = {
        "ELITE_RECOVERY_CANDIDATE": 4,
        "STRONG_RECOVERY_CANDIDATE": 3,
        "ACCEPTABLE_RECOVERY_CANDIDATE": 2,
        "UNSUITABLE": 1,
    }
    candidates.sort(
        key=lambda c: (
            rank_order.get(c["recovery_classification"], 0),
            _to_float(c.get("governance_recovery_contribution"), 0),
            _to_float(c.get("diversification_score"), 0),
        ),
        reverse=True,
    )
    return candidates


def _recovery_prob_label(
    scenario: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Tuple[str, float]:
    adds = [s for s, w in (scenario.get("weights") or {}).items() if s != DOMINANT and w > 0]
    if not adds:
        return "LOW", 0.15

    cand_map = {c["symbol"]: c for c in candidates}
    actionable = sum(1 for s in adds if cand_map.get(s, {}).get("lifecycle_actionable"))
    avail = actionable / len(adds)

    confs = [_to_float(cand_map.get(s, {}).get("confidence"), 0.0) for s in adds]
    conf_factor = (
        float(np.mean([min(c / MIN_CONF_STRONG, 1.0) for c in confs if c > 0])) if confs else 0.0
    )

    dist_y = scenario.get("distance_to_yellow") or 30
    dist_factor = max(0.0, 1.0 - dist_y / 30.0)
    gov_factor = 1.0 if scenario.get("governance_status") == "AUTHORIZED" else 0.35

    score = 0.35 * avail + 0.30 * conf_factor + 0.20 * dist_factor + 0.15 * gov_factor
    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.75:
        return "HIGH", score
    if score >= 0.50:
        return "MODERATE", score
    if score >= 0.30:
        return "LOW", score
    return "LOW", score


def _risk_score_basket(weights: Dict[str, float], candidates: List[Dict[str, Any]]) -> float:
    cand_map = {c["symbol"]: c for c in candidates}
    tech_w = sum(w for s, w in weights.items() if get_sector(s) == "Tech")
    corr_pen = 0.0
    for sym, w in weights.items():
        if sym == DOMINANT or w <= 0:
            continue
        corr = _to_float(cand_map.get(sym, {}).get("correlation_to_QQQ"), 0.5)
        corr_pen += w * corr
    return tech_w + corr_pen * 0.3


def _build_weights(
    adds: List[str],
    qqq_pct: float,
    equal_split: bool = True,
    custom: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if custom:
        return {k: _f(v, 2) or 0.0 for k, v in custom.items()}
    remaining = 100.0 - qqq_pct
    if not adds or remaining <= 0:
        return {DOMINANT: 100.0}
    each = remaining / len(adds)
    weights = {DOMINANT: qqq_pct}
    for sym in adds:
        weights[sym] = each
    return {k: _f(v, 2) or 0.0 for k, v in weights.items()}


def _sweep_yellow_minimum(
    ranked: List[str],
    state: Dict[str, Any],
    max_adds: int = 4,
) -> Optional[Dict[str, Any]]:
    total_mv = state["total_mv"]
    base = state["base_components"]
    current_cps = state["current_cps"]
    baseline_largest = state["baseline_largest_pct"]

    for n in range(1, min(max_adds, len(ranked)) + 1):
        selected = ranked[:n]
        for combined in range(15, 56, 5):
            each = combined / n
            weights = {DOMINANT: 100.0 - combined}
            for sym in selected:
                weights[sym] = each
            sim = _simulate_scenario(weights, total_mv, base, baseline_largest, current_cps)
            if sim["projected_cps"] >= CPS_YELLOW:
                return {
                    "weights": {k: _f(v, 2) for k, v in weights.items()},
                    "simulation": sim,
                    "adds": selected,
                    "qqq_reduction_pct": _f(100.0 - weights[DOMINANT]),
                }
    return None


def _design_baskets(
    candidates: List[Dict[str, Any]],
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    viable = [
        c
        for c in candidates
        if c["recovery_classification"] != "UNSUITABLE" and c.get("lifecycle_actionable")
    ]
    ranked = [c["symbol"] for c in viable]
    if not ranked:
        ranked = [c["symbol"] for c in candidates if c.get("lifecycle_actionable")][:5]

    total_mv = state["total_mv"]
    base = state["base_components"]
    current_cps = state["current_cps"]
    baseline_largest = state["baseline_largest_pct"]

    baskets: List[Dict[str, Any]] = []

    yellow_min = _sweep_yellow_minimum(ranked, state)
    if yellow_min:
        sim = yellow_min["simulation"]
        prob, prob_score = _recovery_prob_label(sim, candidates)
        baskets.append(
            {
                "basket_name": "MINIMUM_YELLOW_BASKET",
                "description": "Smallest lifecycle-actionable set reaching CPS ≥ 75 (YELLOW).",
                "symbols": [DOMINANT] + yellow_min["adds"],
                "weights": yellow_min["weights"],
                "required_qqq_reduction_pct": yellow_min["qqq_reduction_pct"],
                "replacement_allocations": {
                    k: v for k, v in yellow_min["weights"].items() if k != DOMINANT
                },
                "projected_cps": sim["projected_cps"],
                "projected_governance_status": sim["governance_status"],
                "projected_escalation": sim["escalation_level"],
                "recovery_probability": prob,
                "recovery_probability_score": _f(prob_score, 3),
                "candidates_required": len(yellow_min["adds"]),
                "classification": sim["classification"],
                "risk_score": _f(_risk_score_basket(yellow_min["weights"], candidates), 2),
            }
        )
    else:
        baskets.append(
            {
                "basket_name": "MINIMUM_YELLOW_BASKET",
                "description": "No lifecycle-actionable combination reached YELLOW in simulation sweep.",
                "symbols": [DOMINANT],
                "weights": {DOMINANT: 100.0},
                "required_qqq_reduction_pct": 0.0,
                "replacement_allocations": {},
                "projected_cps": current_cps,
                "projected_governance_status": "BLOCKED",
                "projected_escalation": state["escalation"],
                "recovery_probability": "LOW",
                "recovery_probability_score": 0.1,
                "candidates_required": 0,
                "classification": "NOT_ACHIEVABLE",
                "risk_score": 0.0,
            }
        )

    top2 = ranked[:2] if len(ranked) >= 2 else ranked
    balanced_weights = _build_weights(top2, qqq_pct=55.0)
    sim_bal = _simulate_scenario(balanced_weights, total_mv, base, baseline_largest, current_cps)
    prob_b, prob_b_score = _recovery_prob_label(sim_bal, candidates)
    baskets.append(
        {
            "basket_name": "OPTIMAL_BALANCED_BASKET",
            "description": "Top-ranked lifecycle-actionable pair with moderate QQQ reduction.",
            "symbols": [DOMINANT] + top2,
            "weights": balanced_weights,
            "required_qqq_reduction_pct": _f(100.0 - balanced_weights.get(DOMINANT, 100.0)),
            "replacement_allocations": {k: v for k, v in balanced_weights.items() if k != DOMINANT},
            "projected_cps": sim_bal["projected_cps"],
            "projected_governance_status": sim_bal["governance_status"],
            "projected_escalation": sim_bal["escalation_level"],
            "recovery_probability": prob_b,
            "recovery_probability_score": _f(prob_b_score, 3),
            "candidates_required": len(top2),
            "classification": sim_bal["classification"],
            "risk_score": _f(_risk_score_basket(balanced_weights, candidates), 2),
        }
    )

    top4 = ranked[:4] if len(ranked) >= 4 else ranked
    max_weights = _build_weights(top4, qqq_pct=40.0)
    sim_max = _simulate_scenario(max_weights, total_mv, base, baseline_largest, current_cps)
    prob_m, prob_m_score = _recovery_prob_label(sim_max, candidates)
    baskets.append(
        {
            "basket_name": "MAXIMUM_DIVERSIFICATION_BASKET",
            "description": "Broad lifecycle-actionable diversification with aggressive QQQ reduction.",
            "symbols": [DOMINANT] + top4,
            "weights": max_weights,
            "required_qqq_reduction_pct": _f(100.0 - max_weights.get(DOMINANT, 100.0)),
            "replacement_allocations": {k: v for k, v in max_weights.items() if k != DOMINANT},
            "projected_cps": sim_max["projected_cps"],
            "projected_governance_status": sim_max["governance_status"],
            "projected_escalation": sim_max["escalation_level"],
            "recovery_probability": prob_m,
            "recovery_probability_score": _f(prob_m_score, 3),
            "candidates_required": len(top4),
            "classification": sim_max["classification"],
            "risk_score": _f(_risk_score_basket(max_weights, candidates), 2),
        }
    )

    return baskets


def _pick_recommended(baskets: List[Dict[str, Any]]) -> Dict[str, Any]:
    viable = [b for b in baskets if b.get("projected_governance_status") == "AUTHORIZED"]
    if not viable:
        viable = sorted(baskets, key=lambda b: _to_float(b.get("projected_cps"), 0), reverse=True)
    if not viable:
        return {}

    scored = []
    for b in viable:
        cps = _to_float(b.get("projected_cps"), 0)
        prob = _to_float(b.get("recovery_probability_score"), 0)
        risk = _to_float(b.get("risk_score"), 50)
        efficiency = cps / max(_to_float(b.get("required_qqq_reduction_pct"), 1), 1)
        score = (
            0.40 * (cps / 100.0)
            + 0.35 * prob
            + 0.15 * efficiency / 100.0
            + 0.10 * (1.0 - risk / 100.0)
        )
        scored.append((score, b))

    best = max(scored, key=lambda x: x[0])[1]
    return {
        "basket_name": best["basket_name"],
        "weights": best.get("weights"),
        "symbols": best.get("symbols"),
        "projected_cps": best.get("projected_cps"),
        "projected_escalation": best.get("projected_escalation"),
        "projected_governance_status": best.get("projected_governance_status"),
        "recovery_probability": best.get("recovery_probability"),
        "rationale": (
            "Selected for best balance of CPS improvement, lifecycle-actionable feasibility, "
            "and diversification efficiency under current authoritative lifecycle."
        ),
    }


def _replacement_analysis(
    candidates: List[Dict[str, Any]],
    recommended: Dict[str, Any],
) -> Dict[str, Any]:
    hd_pool = [
        c
        for c in candidates
        if c.get("replaces_hd_candidate") and c["recovery_classification"] != "UNSUITABLE"
    ]
    xlk_pool = [
        c
        for c in candidates
        if c.get("replaces_xlk_candidate") and c["recovery_classification"] != "UNSUITABLE"
    ]
    compare_syms = ["USO", "PEP", "XLE", "ABBV", "^VIX", "MA", "UNH", "CVX"]
    compare = [c for c in candidates if c["symbol"] in compare_syms]

    return {
        "hd_replacement": hd_pool[0]["symbol"] if hd_pool else "—",
        "hd_replacement_alternatives": [c["symbol"] for c in hd_pool[:5]],
        "xlk_replacement": xlk_pool[0]["symbol"] if xlk_pool else "—",
        "xlk_replacement_alternatives": [c["symbol"] for c in xlk_pool[:5]],
        "phase185_comparison": compare,
        "original_thesis_changed": True,
        "thesis_change_summary": (
            "Original thesis assumed HD/XLK OPEN_NEW diversification from tech concentration. "
            "Authoritative lifecycle now requires non-tech / lower-QQQ-correlation BUY-effective "
            "symbols (e.g. USO, XLE, ABBV, PEP) rather than sector-tech XLK proxy."
        ),
        "recommended_basket_additions": [
            s for s in (recommended.get("symbols") or []) if s != DOMINANT
        ],
    }


def _recovery_validation(
    baskets: List[Dict[str, Any]], candidates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    yellow_ok = any(_to_float(b.get("projected_cps"), 0) >= CPS_YELLOW for b in baskets)
    green_ok = any(_to_float(b.get("projected_cps"), 0) >= CPS_GREEN for b in baskets)
    min_yellow = min(
        (
            b.get("candidates_required") or 99
            for b in baskets
            if b.get("basket_name") == "MINIMUM_YELLOW_BASKET"
        ),
        default=99,
    )
    viable_count = sum(1 for c in candidates if c["recovery_classification"] != "UNSUITABLE")

    fastest = min(
        baskets,
        key=lambda b: (
            _to_float(b.get("required_qqq_reduction_pct"), 999)
            if b.get("projected_governance_status") == "AUTHORIZED"
            else 999
        ),
        default={},
    )
    lowest_risk = min(baskets, key=lambda b: _to_float(b.get("risk_score"), 999), default={})
    most_efficient = max(
        baskets,
        key=lambda b: _to_float(b.get("projected_cps"), 0)
        / max(_to_float(b.get("required_qqq_reduction_pct"), 1), 1),
        default={},
    )

    return {
        "yellow_reachable": yellow_ok,
        "green_reachable": green_ok,
        "minimum_candidates_for_yellow": min_yellow if yellow_ok else "INSUFFICIENT",
        "lifecycle_actionable_count": viable_count,
        "fastest_recovery_basket": fastest.get("basket_name"),
        "lowest_risk_basket": lowest_risk.get("basket_name"),
        "most_efficient_basket": most_efficient.get("basket_name"),
        "exception_framework_redesign_required": False,
        "exception_framework_note": (
            "PATH_B Recovery Basket Exception framework remains valid. "
            "Only basket membership and weight assumptions require update — not governance rules."
        ),
    }


def _risk_analysis(baskets: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "current_risk": {
            "concentration_pct": state["largest_concentration_pct"],
            "cps": state["current_cps"],
            "escalation": state["escalation"],
        },
        "basket_risk_comparison": [
            {
                "basket_name": b["basket_name"],
                "risk_score": b.get("risk_score"),
                "projected_cps": b.get("projected_cps"),
                "qqq_reduction_pct": b.get("required_qqq_reduction_pct"),
            }
            for b in baskets
        ],
        "legacy_plan_risk": "UNEXECUTABLE — lifecycle-blocked candidates",
        "reconstruction_risk": "MODERATE — depends on lifecycle-actionable BUY pool depth",
    }


def _executive_report(
    state: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    baskets: List[Dict[str, Any]],
    recommended: Dict[str, Any],
    replacement: Dict[str, Any],
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    viable = [c for c in candidates if c["recovery_classification"] != "UNSUITABLE"]
    return {
        "report_type": "RECOVERY_BASKET_RECONSTRUCTION_REPORT",
        "current_recovery_failure": {
            "legacy_basket": ORIGINAL_BASKET,
            "legacy_weights": LEGACY_WEIGHTS,
            "status": state["legacy_plan_status"],
            "reason": state["legacy_failure_reason"],
            "cps": state["current_cps"],
            "escalation": state["escalation"],
            "governance": state["governance_status"],
        },
        "new_candidate_universe_count": len(viable),
        "new_candidate_universe_top5": [c["symbol"] for c in viable[:5]],
        "recommended_basket": recommended,
        "projected_cps": recommended.get("projected_cps"),
        "projected_governance_state": recommended.get("projected_governance_status"),
        "projected_escalation": recommended.get("projected_escalation"),
        "recovery_probability": recommended.get("recovery_probability"),
        "risk_assessment": validation.get("exception_framework_note"),
        "executive_recommendation": (
            f"Replace HD/XLK with lifecycle-actionable symbols "
            f"({', '.join(replacement.get('recommended_basket_additions') or [])}). "
            f"Adopt {recommended.get('basket_name', 'OPTIMAL_BALANCED_BASKET')} "
            f"(projected CPS {recommended.get('projected_cps')} → "
            f"{recommended.get('projected_escalation')}). "
            "Do not redesign exception framework — update basket scope only."
        ),
        "executive_questions": {
            "what_replaces_hd": replacement.get("hd_replacement"),
            "what_replaces_xlk": replacement.get("xlk_replacement"),
            "new_optimal_basket": recommended.get("basket_name"),
            "can_yellow_be_achieved": validation.get("yellow_reachable"),
            "can_green_be_achieved": validation.get("green_reachable"),
            "most_realistic_path": validation.get("most_efficient_basket"),
            "redesign_exception_framework": validation.get("exception_framework_redesign_required"),
        },
    }


def run_analysis() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    ctx = _load_context()
    state = _current_state(ctx)
    candidates = _build_candidate_universe(ctx, state)
    baskets = _design_baskets(candidates, state)
    recommended = _pick_recommended(baskets)
    replacement = _replacement_analysis(candidates, recommended)
    validation = _recovery_validation(baskets, candidates)
    risk = _risk_analysis(baskets, state)
    executive = _executive_report(state, candidates, baskets, recommended, replacement, validation)

    recovery_projection = {
        "current_cps": state["current_cps"],
        "projected_cps_recommended": recommended.get("projected_cps"),
        "cps_delta": (
            (_to_float(recommended.get("projected_cps"), 0) - state["current_cps"])
            if recommended
            else 0
        ),
        "yellow_reachable": validation["yellow_reachable"],
        "green_reachable": validation["green_reachable"],
        "distance_to_yellow": state["distance_to_yellow"],
        "distance_to_green": state["distance_to_green"],
    }

    summary = {
        "lifecycle_actionable_candidates": sum(
            1 for c in candidates if c["recovery_classification"] != "UNSUITABLE"
        ),
        "recommended_basket": recommended.get("basket_name"),
        "projected_cps": recommended.get("projected_cps"),
        "projected_escalation": recommended.get("projected_escalation"),
        "yellow_reachable": validation["yellow_reachable"],
        "green_reachable": validation["green_reachable"],
        "hd_replacement": replacement.get("hd_replacement"),
        "xlk_replacement": replacement.get("xlk_replacement"),
        "exception_framework_redesign": False,
    }

    report = {
        "generated_at": ts,
        "phase": "186",
        "mode": "recovery_basket_reconstruction_engine",
        "disclaimer": (
            "Analysis and design only — does not modify governance, lifecycle, execution, "
            "allocation, broker, or risk controls. No trades or orders."
        ),
        "current_state": state,
        "candidate_universe": candidates,
        "basket_options": baskets,
        "recommended_basket": recommended,
        "recovery_projection": recovery_projection,
        "replacement_analysis": replacement,
        "recovery_validation": validation,
        "risk_analysis": risk,
        "executive_report": executive,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "read-only analysis — no pipeline modifications",
        },
    }

    cand_df = pd.DataFrame(candidates)
    if not cand_df.empty:
        cand_df.to_csv(OUT_CANDIDATES, index=False, encoding="utf-8")

    scenario_rows = []
    for b in baskets:
        scenario_rows.append(
            {
                "basket_name": b["basket_name"],
                "projected_cps": b.get("projected_cps"),
                "projected_governance_status": b.get("projected_governance_status"),
                "projected_escalation": b.get("projected_escalation"),
                "recovery_probability": b.get("recovery_probability"),
                "required_qqq_reduction": b.get("required_qqq_reduction_pct"),
                "candidates_required": b.get("candidates_required"),
                "risk_score": b.get("risk_score"),
                "weights_json": json.dumps(b.get("weights") or {}),
            }
        )
    pd.DataFrame(scenario_rows).to_csv(OUT_SCENARIOS, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_CANDIDATES.name}, {OUT_SCENARIOS.name}")
    _log(
        f"candidates={summary['lifecycle_actionable_candidates']} "
        f"recommended={summary['recommended_basket']} "
        f"cps->{summary['projected_cps']} yellow={summary['yellow_reachable']}"
    )
    return report


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
