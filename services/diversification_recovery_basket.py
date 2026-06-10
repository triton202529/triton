"""
TRITON Phase 175 — Diversification Recovery Basket Design (analysis only).

Designs governance-approved diversification recovery baskets capable of moving
TRITON from ORANGE → YELLOW → GREEN without weakening governance controls.

SAFETY: portfolio design and analysis only — no governance rules, execution,
lifecycle, broker, authorization, or allocation rule changes.

Outputs (data/results/):
    diversification_recovery_basket.json
    diversification_recovery_candidates.csv
    diversification_scenarios.csv

Run:
    python -m services.diversification_recovery_basket
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
    _sector_weights,
)
from services.pnl_ledger_builder import _load_price_series, _norm_sym, _to_float
from services.sector_exposure import get_sector

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "diversification_recovery_basket.json"
OUT_CANDIDATES = RESULTS / "diversification_recovery_candidates.csv"
OUT_SCENARIOS = RESULTS / "diversification_scenarios.csv"

SECTOR_ETFS = frozenset(
    {
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLY",
        "XLI",
        "XLP",
        "XLU",
        "XLB",
        "XLRE",
        "QQQ",
        "SPY",
        "DIA",
        "IWM",
        "VTI",
        "VOO",
        "GLD",
        "SLV",
        "BITO",
        "USO",
        "UNG",
        "ARKK",
    }
)
PERMIT_ESCALATION = {"GREEN", "YELLOW"}


def _log(msg: str) -> None:
    print(f"[diversification_recovery_basket] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _asset_class(sym: str) -> str:
    return "ETF" if sym in SECTOR_ETFS or sym.startswith("X") else "EQUITY"


def _return_correlation(a: str, b: str, days: int = 60) -> Optional[float]:
    pa = _load_price_series(a)
    pb = _load_price_series(b)
    if pa.empty or pb.empty:
        return None
    pa = pa.sort_values("date").tail(days + 1)
    pb = pb.sort_values("date").tail(days + 1)
    merged = pa.merge(pb, on="date", suffixes=("_a", "_b"))
    if len(merged) < 10:
        return None
    ra = merged["close_a"].pct_change().dropna()
    rb = merged["close_b"].pct_change().dropna()
    if ra.empty or rb.empty:
        return None
    corr = float(ra.corr(rb))
    return corr if np.isfinite(corr) else None


def _equal_weight_positions(
    held: List[str],
    additions: List[str],
    total_mv: float,
) -> List[Dict[str, Any]]:
    names = list(dict.fromkeys(held + additions))
    if not names or total_mv <= 0:
        return []
    w = total_mv / len(names)
    return [{"symbol": s, "market_value": w, "qty": np.nan} for s in names]


def _simulate(positions: List[Dict[str, Any]], base_components: Dict[str, int]) -> Dict[str, Any]:
    _, largest, alerts = _concentration_score(positions)
    cps, comps = _estimate_cps(positions, base_components)
    sectors = _sector_weights(positions)
    top_sec = max(sectors.items(), key=lambda x: x[1]) if sectors else ("—", 0.0)
    esc = _escalation_from_cps(cps)
    return {
        "position_count": len(positions),
        "largest_position_weight_pct": _f(largest),
        "sector_concentration_top_pct": _f(top_sec[1]),
        "top_sector": top_sec[0],
        "concentration_score": comps.get("concentration"),
        "concentration_alerts": alerts,
        "portfolio_health": _health_band(cps),
        "capital_preservation_score": cps,
        "escalation_state": esc,
        "governance_authorized_estimate": esc in PERMIT_ESCALATION,
        "sector_exposure": {k: _f(v) for k, v in sectors.items()},
    }


def _rank_candidates(
    positions: List[Dict[str, Any]],
    recommendations: pd.DataFrame,
    base_components: Dict[str, int],
    current_cps: int,
) -> pd.DataFrame:
    held = [p["symbol"] for p in positions]
    dominant = held[0] if held else "QQQ"
    dominant_sector = get_sector(dominant)
    total_mv = sum(p["market_value"] for p in positions)
    sector_before = _sector_weights(positions)

    open_new = recommendations[
        recommendations["recommended_action"].astype(str).str.upper() == "OPEN_NEW"
    ].copy()

    rows: List[Dict[str, Any]] = []
    for _, r in open_new.iterrows():
        sym = _norm_sym(r.get("ticker"))
        if not sym or sym in held:
            continue
        sector = get_sector(sym)
        corr = _return_correlation(dominant, sym)
        corr_penalty = abs(corr) if corr is not None else 0.5
        sector_bonus = 0.0 if sector == dominant_sector else 1.0

        proj = _equal_weight_positions(held, [sym], total_mv)
        sim = _simulate(proj, base_components)
        cps_delta = (sim["capital_preservation_score"] or 0) - current_cps

        alloc = _to_float(r.get("allocation_score"), 0.0)
        edge = _to_float(r.get("edge_score"), 0.0)
        conf = _to_float(r.get("confidence"), 0.0)

        div_value = float(
            np.clip(
                40.0 * sector_bonus
                + 20.0 * (1.0 - min(corr_penalty, 1.0))
                + 25.0 * min(alloc, 1.0)
                + 15.0 * min(edge, 1.0),
                0.0,
                100.0,
            )
        )
        gov_score = float(
            np.clip(
                cps_delta * 3.0 + (10.0 if sim["governance_authorized_estimate"] else 0.0),
                0.0,
                100.0,
            )
        )

        rows.append(
            {
                "ticker": sym,
                "sector": sector,
                "asset_class": _asset_class(sym),
                "allocation_score": _f(alloc, 4),
                "edge_score": _f(edge, 4),
                "confidence": _f(conf, 4),
                "correlation_to_existing_portfolio": _f(corr, 3),
                "diversification_value": _f(div_value, 1),
                "governance_improvement_score": _f(gov_score, 1),
                "capital_preservation_score_projected": sim["capital_preservation_score"],
                "projected_escalation": sim["escalation_state"],
                "projected_largest_weight_pct": sim["largest_position_weight_pct"],
                "governance_authorized_estimate": sim["governance_authorized_estimate"],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["governance_improvement_score", "diversification_value"],
            ascending=False,
        ).reset_index(drop=True)
        df["rank"] = df.index + 1
    return df


def _sector_fill_pool(
    recommendations: pd.DataFrame, held: List[str], open_new_syms: List[str]
) -> List[str]:
    """Best-scoring ticker per sector from recommendations (sector diversification fill)."""
    df = recommendations.copy()
    df["ticker"] = df["ticker"].map(_norm_sym)
    df["allocation_score"] = pd.to_numeric(df["allocation_score"], errors="coerce").fillna(0)
    df = df[~df["ticker"].isin(held)]
    df["sector"] = df["ticker"].map(get_sector)
    df = df[df["sector"] != "Unknown"]
    # Prefer sector ETFs and OPEN_NEW, then WATCH with strong scores
    df["priority"] = 0
    df.loc[df["recommended_action"].astype(str).str.upper() == "OPEN_NEW", "priority"] = 3
    df.loc[df["ticker"].isin(SECTOR_ETFS), "priority"] = df["priority"] + 1
    df = df.sort_values(["priority", "allocation_score"], ascending=False)
    picked: List[str] = list(open_new_syms)
    seen_sectors = {get_sector(s) for s in held + picked}
    for _, r in df.iterrows():
        sym = str(r["ticker"])
        sec = str(r["sector"])
        if sym in picked:
            continue
        if sec in seen_sectors and sym not in open_new_syms:
            continue
        picked.append(sym)
        seen_sectors.add(sec)
    return picked


def _build_basket(
    name: str,
    held: List[str],
    picks: List[str],
    total_mv: float,
    base_components: Dict[str, int],
    current_cps: int,
) -> Dict[str, Any]:
    sim = _simulate(_equal_weight_positions(held, picks, total_mv), base_components)
    weights = _equal_weight_positions(held, picks, total_mv)
    w_pct = 100.0 / len(weights) if weights else 0.0
    members = []
    for p in weights:
        sec = get_sector(p["symbol"])
        members.append(
            {
                "ticker": p["symbol"],
                "target_weight_pct": _f(w_pct, 2),
                "sector": sec,
                "asset_class": _asset_class(p["symbol"]),
                "diversification_benefit": "HIGH" if sec != get_sector(held[0]) else "LOW",
                "governance_impact": sim["escalation_state"],
                "expected_cps_improvement": _f(
                    (sim["capital_preservation_score"] or 0) - current_cps, 1
                ),
            }
        )
    return {
        "basket_name": name,
        "members": members,
        "member_count": len(members),
        "metrics": sim,
    }


def _min_count_for_escalation(
    held: List[str],
    pool: List[str],
    total_mv: float,
    base_components: Dict[str, int],
    target: str,
) -> Tuple[int, Optional[List[str]], Optional[Dict[str, Any]]]:
    target_rank = {"YELLOW": 75, "GREEN": 90}[target]
    best_n, best_picks, best_sim = None, None, None
    for n in range(1, min(len(pool), 15) + 1):
        picks = pool[:n]
        sim = _simulate(_equal_weight_positions(held, picks, total_mv), base_components)
        if (sim["capital_preservation_score"] or 0) >= target_rank:
            return n, picks, sim
        if best_sim is None or (sim["capital_preservation_score"] or 0) > (
            best_sim["capital_preservation_score"] or 0
        ):
            best_n, best_picks, best_sim = n, picks, sim
    return best_n or 0, best_picks, best_sim


def _capital_preservation_assessment(basket: Dict[str, Any], current_cps: int) -> Dict[str, Any]:
    m = basket.get("metrics") or {}
    cps = m.get("capital_preservation_score") or 0
    delta = cps - current_cps
    factors = {
        "reduce_concentration": (m.get("largest_position_weight_pct") or 100) < 50,
        "reduce_portfolio_risk": delta > 0,
        "improve_governance": bool(m.get("governance_authorized_estimate")),
        "improve_capital_preservation": delta >= 5,
        "increase_complexity": (m.get("position_count") or 1) > 5,
    }
    pos = sum(1 for k, v in factors.items() if v and k != "increase_complexity")
    neg = 1 if factors["increase_complexity"] and not factors["improve_governance"] else 0
    if pos >= 3 and delta >= 10:
        verdict = "POSITIVE"
    elif delta >= 0:
        verdict = "NEUTRAL"
    else:
        verdict = "NEGATIVE"
    return {
        "verdict": verdict,
        "cps_delta": _f(delta, 1),
        "factors": factors,
        "summary": (
            "Diversification reduces concentration and improves CPS/governance posture "
            "with modest complexity increase."
            if verdict == "POSITIVE"
            else (
                "Partial benefit — may not alone clear governance gates."
                if verdict == "NEUTRAL"
                else "Net negative under current simulation assumptions."
            )
        ),
    }


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)

    positions = _load_positions()
    held = [p["symbol"] for p in positions]
    total_mv = sum(p["market_value"] for p in positions)

    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    alloc_summary = _read_json(RESULTS / "portfolio_allocation_summary.json")
    cpi = _read_json(RESULTS / "capital_preservation_intelligence.json")
    cpe = _read_json(RESULTS / "capital_preservation_escalation.json")
    auth = _read_json(RESULTS / "governance_authorization.json")
    trace = _read_json(RESULTS / "governance_authorization_trace.json")
    paradox = _read_json(RESULTS / "concentration_governance_analysis.json")

    base_components = dict(cpi.get("component_scores") or {})
    for k in CPI_WEIGHTS:
        base_components.setdefault(k, 70)
    current_cps = int(
        cpi.get("capital_preservation_score") or _estimate_cps(positions, base_components)[0]
    )
    current_esc = cpe.get("escalation_state") or _escalation_from_cps(current_cps)

    candidates = _rank_candidates(positions, recommendations, base_components, current_cps)
    candidates.to_csv(OUT_CANDIDATES, index=False, encoding="utf-8")

    open_new_syms = candidates["ticker"].tolist() if not candidates.empty else []
    pool = _sector_fill_pool(recommendations, held, open_new_syms)

    # Recovery analysis: add 1, 2, 3, 5, 10 candidates (equal-weight simulation)
    recovery_rows: List[Dict[str, Any]] = []
    for n in [1, 2, 3, 5, 10]:
        picks = pool[:n]
        if not picks:
            continue
        sim = _simulate(_equal_weight_positions(held, picks, total_mv), base_components)
        recovery_rows.append(
            {
                "candidates_added": n,
                "tickers": ",".join(picks),
                **sim,
            }
        )
    recovery_df = pd.DataFrame(recovery_rows)

    # Minimum diversification thresholds
    n_yellow, picks_yellow, sim_yellow = _min_count_for_escalation(
        held, pool, total_mv, base_components, "YELLOW"
    )
    n_green, picks_green, sim_green = _min_count_for_escalation(
        held, pool, total_mv, base_components, "GREEN"
    )

    # Baskets
    minimal_picks = pool[: max(1, n_yellow or 1)]
    balanced_n = max(n_yellow or 3, 5)
    balanced_picks = pool[: min(balanced_n, len(pool))]
    optimal_picks = picks_green or picks_yellow or pool[:5]

    basket_v1 = _build_basket(
        "RECOVERY_BASKET_V1", held, optimal_picks, total_mv, base_components, current_cps
    )
    minimal_basket = _build_basket(
        "Minimal Basket", held, minimal_picks, total_mv, base_components, current_cps
    )
    balanced_basket = _build_basket(
        "Balanced Basket", held, balanced_picks, total_mv, base_components, current_cps
    )
    optimal_basket = _build_basket(
        "Governance-Optimal Basket", held, optimal_picks, total_mv, base_components, current_cps
    )

    scenarios = [
        {
            "scenario": "Scenario A — Current Portfolio",
            "description": "No changes — baseline concentration block",
            **_simulate(positions, base_components),
        },
        {
            "scenario": "Scenario B — Minimal Basket",
            "description": f"Minimum names toward YELLOW ({len(minimal_picks)} additions)",
            **minimal_basket["metrics"],
            "basket_tickers": ",".join(minimal_picks),
        },
        {
            "scenario": "Scenario C — Balanced Basket",
            "description": f"Sector-diversified equal-weight ({len(balanced_picks)} additions)",
            **balanced_basket["metrics"],
            "basket_tickers": ",".join(balanced_picks),
        },
        {
            "scenario": "Scenario D — Governance-Optimal Basket",
            "description": f"Optimized for GREEN/YELLOW clearance ({len(optimal_picks)} additions)",
            **optimal_basket["metrics"],
            "basket_tickers": ",".join(optimal_picks),
        },
    ]
    scenarios_df = pd.DataFrame(scenarios)
    scenarios_df.to_csv(OUT_SCENARIOS, index=False, encoding="utf-8")

    best_candidate = candidates.iloc[0]["ticker"] if not candidates.empty else "—"
    cp_assessment = _capital_preservation_assessment(basket_v1, current_cps)

    fastest_green = {
        "basket_name": "RECOVERY_BASKET_V1",
        "tickers": optimal_picks,
        "target_weights_equal_pct": (
            _f(100.0 / (len(held) + len(optimal_picks)), 2) if optimal_picks else None
        ),
        "projected_cps": (sim_green or sim_yellow or basket_v1["metrics"]).get(
            "capital_preservation_score"
        ),
        "projected_escalation": (sim_green or sim_yellow or basket_v1["metrics"]).get(
            "escalation_state"
        ),
        "governance_authorized": (sim_green or sim_yellow or basket_v1["metrics"]).get(
            "governance_authorized_estimate"
        ),
        "minimum_additions_for_yellow": n_yellow,
        "minimum_additions_for_green": n_green,
        "yellow_tickers": picks_yellow,
        "green_tickers": picks_green,
        "strategy": (
            "Equal-weight rebalance simulation across existing holdings plus the "
            "lowest-count sector-diverse basket that clears CPS thresholds. "
            "Minimizes turnover vs full redeployment by adding sector ETFs from "
            "allocation recommendations."
        ),
    }

    executive = {
        "current_governance_state": {
            "governance_authorized": auth.get("governance_authorized"),
            "escalation_state": current_esc,
            "block_reason": (auth.get("gate_reasons") or {}).get("governance"),
            "trace_cause": (trace.get("executive_report") or {}).get("primary_cause"),
        },
        "current_cps": current_cps,
        "current_concentration_pct": _f(_concentration_score(positions)[1]),
        "concentration_cause": (paradox.get("executive_report") or {}).get("concentration_trigger")
        or cpe.get("escalation_reason_labels"),
        "best_recovery_candidate": best_candidate,
        "best_recovery_basket": [m["ticker"] for m in basket_v1["members"]],
        "projected_cps": basket_v1["metrics"].get("capital_preservation_score"),
        "projected_escalation_state": basket_v1["metrics"].get("escalation_state"),
        "projected_governance_status": (
            "AUTHORIZED"
            if basket_v1["metrics"].get("governance_authorized_estimate")
            else "BLOCKED"
        ),
        "fastest_path_to_green": fastest_green,
        "orange_to_yellow_minimum_names": n_yellow,
        "yellow_to_green_minimum_names": n_green,
    }

    cert_reasons: List[str] = []
    if candidates.empty:
        cert_reasons.append("no OPEN_NEW candidates in allocation pipeline")
    if not n_yellow:
        cert_reasons.append("could not simulate ORANGE→YELLOW with available pool")
    status = (
        "CERTIFIED"
        if candidates.shape[0] >= 1 and n_yellow and n_green
        else "PARTIALLY_CERTIFIED" if n_yellow else "NOT_CERTIFIED"
    )

    report = {
        "generated_at": _now_iso(),
        "executive_report": executive,
        "best_diversification_candidates": candidates.to_dict(orient="records"),
        "recovery_basket_v1": basket_v1,
        "recovery_analysis_by_count": recovery_df.to_dict(orient="records"),
        "governance_recovery_thresholds": {
            "orange_to_yellow": {
                "minimum_new_names": n_yellow,
                "cps_required": 75,
                "tickers": picks_yellow,
                "simulation": sim_yellow,
            },
            "yellow_to_green": {
                "minimum_new_names": n_green,
                "cps_required": 90,
                "tickers": picks_green,
                "simulation": sim_green,
            },
        },
        "scenarios": scenarios,
        "fastest_path_to_green": fastest_green,
        "capital_preservation_assessment": cp_assessment,
        "certification": {
            "status": status,
            "reasons": cert_reasons,
            "safety": "analysis-only — no rules, execution, or authorization modified",
        },
        "methodology": {
            "simulation": "equal-weight rebalance across held + selected names at constant total MV",
            "ranking": "diversification_value + governance_improvement_score from projected CPS",
            "correlation": "60-day return correlation vs dominant holding",
            "sector_mapping": "services/sector_exposure.get_sector",
            "permit_escalation": sorted(PERMIT_ESCALATION),
        },
        "disclaimer": (
            "Portfolio design only — does not place trades or modify governance, execution, "
            "lifecycle, broker, authorization, or allocation rules."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_CANDIDATES.name}, {OUT_SCENARIOS.name}")
    _log(f"RECOVERY_BASKET_V1: {[m['ticker'] for m in basket_v1['members']]}")
    _log(f"projected: CPS {executive['projected_cps']} {executive['projected_escalation_state']}")
    _log(f"min for YELLOW={n_yellow} GREEN={n_green}")
    return report


if __name__ == "__main__":
    run()
