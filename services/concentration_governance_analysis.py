"""
TRITON Phase 174 — Concentration Governance Paradox Analysis (read-only).

Investigates whether governance is unintentionally blocking diversification
trades that would reduce concentration risk. Evaluates OPEN_NEW candidates
and simulates portfolio scenarios without modifying any rules or execution.

Outputs (data/results/):
    concentration_governance_analysis.json
    diversification_candidates.csv
    concentration_scenarios.csv

Run:
    python -m services.concentration_governance_analysis
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

from services.pnl_ledger_builder import _norm_sym, _to_float
from services.sector_exposure import get_sector

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "concentration_governance_analysis.json"
OUT_CANDIDATES = RESULTS / "diversification_candidates.csv"
OUT_SCENARIOS = RESULTS / "concentration_scenarios.csv"

CONCENTRATION_THRESHOLD_PCT = 25.0  # risk_watchdog DEFAULT_CONCENTRATION_PCT
GOVERNANCE_TARGET_PCT = 35.0  # Phase 173 diversification target
CPI_WEIGHTS = {
    "drawdown": 0.3,
    "concentration": 0.25,
    "exposure": 0.2,
    "operational": 0.15,
    "execution": 0.1,
}


def _log(msg: str) -> None:
    print(f"[concentration_governance_analysis] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(v: Any, nd: int = 2) -> Optional[float]:
    x = _to_float(v, np.nan)
    return round(float(x), nd) if np.isfinite(x) else None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _clamp_score(score: float) -> int:
    return int(max(0, min(100, round(score))))


def _escalation_from_cps(cps: int) -> str:
    if cps >= 90:
        return "GREEN"
    if cps >= 75:
        return "YELLOW"
    if cps >= 60:
        return "ORANGE"
    if cps >= 40:
        return "RED"
    return "CRITICAL"


def _health_band(cps: int) -> str:
    if cps >= 80:
        return "HEALTHY"
    if cps >= 65:
        return "CAUTION"
    if cps >= 50:
        return "ELEVATED"
    return "CRITICAL"


def _load_positions() -> List[Dict[str, Any]]:
    snap = _read_csv(RESULTS / "positions_snapshot.csv")
    if snap.empty:
        return []
    snap.columns = [str(c).strip().lower() for c in snap.columns]
    sym_col = "ticker" if "ticker" in snap.columns else "symbol"
    out: List[Dict[str, Any]] = []
    for _, r in snap.iterrows():
        sym = _norm_sym(r.get(sym_col))
        qty = _to_float(r.get("qty") or r.get("qty_available"))
        mv = _to_float(r.get("market_value") or r.get("value"))
        if not sym or not np.isfinite(qty) or abs(qty) < 1e-9:
            continue
        if not np.isfinite(mv) or mv <= 0:
            px = _to_float(r.get("current_price") or r.get("lastday_price"))
            mv = px * abs(qty) if np.isfinite(px) else 0.0
        out.append({"symbol": sym, "market_value": abs(mv), "qty": abs(qty)})
    return out


def _concentration_score(positions: List[Dict[str, Any]]) -> Tuple[int, float, int]:
    """Mirror risk_watchdog _score_concentration; return (score, largest_pct, alert_count)."""
    mvs = [p["market_value"] for p in positions if p.get("market_value", 0) > 0]
    total = sum(mvs)
    if total <= 0:
        return 88, 0.0, 0
    largest_pct = (max(mvs) / total) * 100.0
    alert_count = sum(1 for mv in mvs if (mv / total) * 100.0 > CONCENTRATION_THRESHOLD_PCT)
    over = max(0.0, largest_pct - CONCENTRATION_THRESHOLD_PCT)
    score = 100.0 - over * 1.5 - 10.0 * alert_count
    return _clamp_score(score), largest_pct, alert_count


def _exposure_score(positions: List[Dict[str, Any]]) -> int:
    n = len(positions)
    if n == 0:
        return 70
    if n == 1:
        return 45
    if n <= 3:
        return 65
    if n <= 8:
        return 85
    return 100


def _estimate_cps(
    positions: List[Dict[str, Any]],
    base_components: Dict[str, int],
) -> Tuple[int, Dict[str, int]]:
    conc, _, _ = _concentration_score(positions)
    exp = _exposure_score(positions)
    components = dict(base_components)
    components["concentration"] = conc
    components["exposure"] = exp
    cps = _clamp_score(sum(components[k] * CPI_WEIGHTS[k] for k in CPI_WEIGHTS))
    return cps, components


def _sector_weights(positions: List[Dict[str, Any]]) -> Dict[str, float]:
    total = sum(p["market_value"] for p in positions)
    if total <= 0:
        return {}
    by_sector: Dict[str, float] = {}
    for p in positions:
        sec = get_sector(p["symbol"])
        w = p["market_value"] / total * 100.0
        by_sector[sec] = by_sector.get(sec, 0.0) + w
    return by_sector


def _classify_diversification(
    current_largest: float,
    projected_largest: float,
    sector_before: Dict[str, float],
    sector_after: Dict[str, float],
    candidate_sector: str,
) -> str:
    name_delta = current_largest - projected_largest
    if name_delta > 1.0:
        name_effect = "REDUCES"
    elif name_delta < -1.0:
        name_effect = "INCREASES"
    else:
        name_effect = "NEUTRAL"

    cand_sec_before = sector_before.get(candidate_sector, 0.0)
    cand_sec_after = sector_after.get(candidate_sector, 0.0)
    sec_delta = cand_sec_after - cand_sec_before

    if name_effect == "REDUCES" and sec_delta <= 1.0:
        return "REDUCES_CONCENTRATION"
    if name_effect == "INCREASES" or (name_effect == "NEUTRAL" and sec_delta > 5.0):
        return "INCREASES_CONCENTRATION"
    if name_effect == "REDUCES" and sec_delta > 5.0:
        return "REDUCES_CONCENTRATION"  # name-level benefit dominates for governance gate
    return "NEUTRAL"


def _new_position_notional(total_mv: float, max_positions: int, allocation_score: float) -> float:
    """Equal-slot baseline scaled by candidate allocation score."""
    slots = max(max_positions, 1)
    base = total_mv / slots if total_mv > 0 else 500.0
    scale = 0.75 + min(max(allocation_score, 0.0), 1.0) * 0.5
    return max(base * scale, 100.0)


def analyze_candidates(
    positions: List[Dict[str, Any]],
    recommendations: pd.DataFrame,
    max_positions: int,
    base_components: Dict[str, int],
    current_cps: int,
) -> pd.DataFrame:
    total_mv = sum(p["market_value"] for p in positions)
    _, current_largest, _ = _concentration_score(positions)
    sector_before = _sector_weights(positions)

    open_new = recommendations[
        recommendations["recommended_action"].astype(str).str.upper() == "OPEN_NEW"
    ].copy()

    rows: List[Dict[str, Any]] = []
    for _, r in open_new.iterrows():
        sym = _norm_sym(r.get("ticker"))
        if not sym:
            continue
        score = _to_float(r.get("allocation_score"), 0.0)
        notional = _new_position_notional(total_mv, max_positions, score)
        projected = [dict(p) for p in positions]
        projected.append({"symbol": sym, "market_value": notional, "qty": np.nan})

        _, proj_largest, _ = _concentration_score(projected)
        proj_cps, _ = _estimate_cps(projected, base_components)
        sector_after = _sector_weights(projected)
        cand_sector = get_sector(sym)
        classification = _classify_diversification(
            current_largest, proj_largest, sector_before, sector_after, cand_sector
        )
        div_benefit = current_largest - proj_largest
        weight_after = (
            (notional / (total_mv + notional)) * 100.0 if total_mv + notional > 0 else 0.0
        )

        rows.append(
            {
                "ticker": sym,
                "classification": classification,
                "current_concentration_pct": _f(current_largest),
                "projected_concentration_after_trade_pct": _f(proj_largest),
                "sector_concentration_after_trade_pct": _f(sector_after.get(cand_sector, 0.0)),
                "portfolio_weight_after_trade_pct": _f(weight_after),
                "diversification_benefit_pct_points": _f(div_benefit),
                "estimated_new_notional": _f(notional),
                "allocation_score": _f(score, 4),
                "confidence": _f(r.get("confidence"), 4),
                "projected_cps": proj_cps,
                "projected_escalation": _escalation_from_cps(proj_cps),
                "governance_would_improve": proj_cps > current_cps
                and proj_largest < current_largest,
                "governance_would_worsen": proj_cps < current_cps or proj_largest > current_largest,
                "sector": cand_sector,
                "reason": str(r.get("reason") or ""),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("diversification_benefit_pct_points", ascending=False).reset_index(
            drop=True
        )
    return df


def build_scenarios(
    positions: List[Dict[str, Any]],
    candidates: pd.DataFrame,
    max_positions: int,
    base_components: Dict[str, int],
) -> pd.DataFrame:
    total_mv = sum(p["market_value"] for p in positions)
    current_cps, _ = _estimate_cps(positions, base_components)
    _, cur_largest, _ = _concentration_score(positions)

    scenarios: List[Dict[str, Any]] = []

    def add_scenario(name: str, extra: List[Tuple[str, float]], *, rebalance: bool = False) -> None:
        if rebalance and positions and extra:
            # Equal-weight simulation: shrink existing book proportionally, add new names.
            half = total_mv / (1 + len(extra)) if total_mv > 0 else 0.0
            proj = [{"symbol": p["symbol"], "market_value": half, "qty": np.nan} for p in positions]
            add_mv = 0.0
            for sym, _ in extra:
                proj.append({"symbol": sym, "market_value": half, "qty": np.nan})
                add_mv += half
        else:
            proj = [dict(p) for p in positions]
            add_mv = 0.0
            for sym, notional in extra:
                proj.append({"symbol": sym, "market_value": notional, "qty": np.nan})
                add_mv += notional
        conc_score, largest, alerts = _concentration_score(proj)
        cps, comps = _estimate_cps(proj, base_components)
        sectors = _sector_weights(proj)
        top_sector = max(sectors.items(), key=lambda x: x[1])[0] if sectors else "—"
        top_sector_pct = max(sectors.values()) if sectors else 0.0
        scenarios.append(
            {
                "scenario": name,
                "position_count": len(proj),
                "largest_position_weight_pct": _f(largest),
                "concentration_score": conc_score,
                "concentration_alerts": alerts,
                "portfolio_health": _health_band(cps),
                "capital_preservation_score": cps,
                "escalation_state_estimate": _escalation_from_cps(cps),
                "governance_authorized_estimate": _escalation_from_cps(cps) in ("GREEN", "YELLOW"),
                "governance_impact": (
                    "IMPROVES"
                    if cps > current_cps
                    else "WORSENS" if cps < current_cps else "NEUTRAL"
                ),
                "cps_delta": _f(cps - current_cps, 1),
                "largest_weight_delta_pct": _f(cur_largest - largest, 1),
                "top_sector": top_sector,
                "top_sector_weight_pct": _f(top_sector_pct),
                "added_notional": _f(add_mv),
                "total_portfolio_value": _f(total_mv + add_mv),
            }
        )

    add_scenario("Current State — QQQ only", [])

    slot = _new_position_notional(total_mv, max_positions, 0.38)
    if not candidates.empty:
        top = str(candidates.iloc[0]["ticker"])
        add_scenario(f"Scenario A — QQQ + {top} (slot-sized add)", [(top, slot)])
        half = total_mv / 2.0 if total_mv > 0 else slot
        add_scenario(
            f"Scenario A2 — QQQ + {top} (equal-weight split)",
            [(top, half)],
            rebalance=True,
        )

        top3 = candidates.head(3)["ticker"].tolist()
        add_scenario(
            f"Scenario B — QQQ + Top 3 ({', '.join(top3)})",
            [(sym, slot) for sym in top3],
        )

    # Scenario C: equal-weight target book across diversified sectors
    diversified = [
        ("XLK", "Tech"),
        ("XLF", "Financials"),
        ("XLE", "Energy"),
        ("XLV", "Healthcare"),
        ("XLY", "Consumer Discretionary"),
    ]
    avail = [s for s, _ in diversified if s != "QQQ"]
    n_add = min(len(avail), max(1, max_positions - len(positions)))
    target_names = avail[:n_add]
    if total_mv > 0 and target_names:
        per = total_mv / len(target_names)  # equal split of existing book across new names
        add_scenario(
            f"Scenario C — QQQ + {len(target_names)} Sector ETFs (equal weight)",
            [(sym, per) for sym in target_names],
        )

    return pd.DataFrame(scenarios)


def _paradox_verdict(candidates: pd.DataFrame, auth: Dict[str, Any]) -> Dict[str, Any]:
    gov_blocked = auth.get("governance_authorized") is False
    reduces = (
        candidates[candidates["classification"] == "REDUCES_CONCENTRATION"]
        if not candidates.empty
        else pd.DataFrame()
    )
    best = (
        reduces.iloc[0]
        if not reduces.empty
        else (candidates.iloc[0] if not candidates.empty else None)
    )

    xlk = candidates[candidates["ticker"] == "XLK"] if not candidates.empty else pd.DataFrame()
    xlk_row = xlk.iloc[0].to_dict() if not xlk.empty else {}

    would_improve = bool(not reduces.empty and reduces["governance_would_improve"].any())
    name_level_benefit = bool(not reduces.empty)
    paradox = bool(gov_blocked and name_level_benefit)

    return {
        "governance_blocking_submissions": gov_blocked,
        "paradox_detected": paradox,
        "paradox_summary": (
            "YES — governance blocks paper submission while OPEN_NEW candidates would reduce "
            "name-level concentration, but trades cannot execute until escalation clears."
            if paradox
            else "NO — blocking aligns with concentration posture or candidates do not improve it."
        ),
        "xlk_analysis": {
            "would_reduce_name_concentration": xlk_row.get("classification")
            == "REDUCES_CONCENTRATION",
            "projected_largest_weight_pct": xlk_row.get("projected_concentration_after_trade_pct"),
            "sector_concentration_after_pct": xlk_row.get("sector_concentration_after_trade_pct"),
            "governance_would_improve": xlk_row.get("governance_would_improve"),
            "note": (
                "XLK reduces single-name weight but both QQQ and XLK map to Tech — "
                "sector concentration remains ~100% Tech until non-Tech names are added."
                if xlk_row
                else "XLK not in current OPEN_NEW set."
            ),
        },
        "best_diversification_candidate": (
            f"{best['ticker']} ({best['classification']}, benefit {best['diversification_benefit_pct_points']} pp)"
            if best is not None
            else "—"
        ),
        "is_governance_blocking_beneficial_trade": paradox,
    }


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)

    positions = _load_positions()
    alloc_summary = _read_json(RESULTS / "portfolio_allocation_summary.json")
    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    cpi = _read_json(RESULTS / "capital_preservation_intelligence.json")
    cpe = _read_json(RESULTS / "capital_preservation_escalation.json")
    auth = _read_json(RESULTS / "governance_authorization.json")
    trace = _read_json(RESULTS / "governance_authorization_trace.json")

    base_components = dict(cpi.get("component_scores") or {})
    for k in CPI_WEIGHTS:
        base_components.setdefault(k, 70)
    current_cps = int(
        cpi.get("capital_preservation_score") or _estimate_cps(positions, base_components)[0]
    )
    max_positions = int(alloc_summary.get("max_positions") or 35)

    candidates = analyze_candidates(
        positions, recommendations, max_positions, base_components, current_cps
    )
    candidates.to_csv(OUT_CANDIDATES, index=False, encoding="utf-8")

    scenarios = build_scenarios(positions, candidates, max_positions, base_components)
    scenarios.to_csv(OUT_SCENARIOS, index=False, encoding="utf-8")

    _, cur_largest, cur_alerts = _concentration_score(positions)
    paradox = _paradox_verdict(candidates, auth)

    # Paradox: governance blocks while OPEN_NEW trades reduce name-level concentration
    # even if full CPS/YELLOW not yet reached.
    if not candidates.empty:
        name_reducers = candidates[candidates["classification"] == "REDUCES_CONCENTRATION"]
        if auth.get("governance_authorized") is False and not name_reducers.empty:
            paradox["paradox_detected"] = True
            paradox["is_governance_blocking_beneficial_trade"] = True
            paradox["paradox_summary"] = (
                "YES — governance blocks paper submission while OPEN_NEW candidates reduce "
                "name-level concentration; however escalation may remain ORANGE until a "
                "multi-sector book (Scenario C) clears YELLOW/GREEN thresholds."
            )

    # Path to GREEN/YELLOW from scenarios
    path_rows = scenarios.sort_values("capital_preservation_score", ascending=False)
    best_scenario = path_rows.iloc[0] if not path_rows.empty else None
    yellow_scenarios = scenarios[scenarios["escalation_state_estimate"].isin(["GREEN", "YELLOW"])]

    executive = {
        "current_concentration_pct": _f(cur_largest),
        "current_escalation_state": cpe.get("escalation_state", "—"),
        "current_governance_authorized": auth.get("governance_authorized"),
        "concentration_trigger": cpe.get("escalation_reason_labels") or ["EXCESS_CONCENTRATION"],
        "best_diversification_candidate": paradox.get("best_diversification_candidate"),
        "concentration_reduction_potential_pp": (
            _f(candidates["diversification_benefit_pct_points"].max())
            if not candidates.empty
            else None
        ),
        "projected_governance_impact": (
            f"Best OPEN_NEW candidate moves CPS to {candidates.iloc[0]['projected_cps']} "
            f"({candidates.iloc[0]['projected_escalation']})"
            if not candidates.empty
            else "—"
        ),
        "would_trade_improve_governance": paradox.get("paradox_detected"),
        "would_trade_worsen_governance": (
            bool(candidates["governance_would_worsen"].any()) if not candidates.empty else False
        ),
        "xlk_reduces_concentration": (paradox.get("xlk_analysis") or {}).get(
            "would_reduce_name_concentration"
        ),
        "governance_blocking_beneficial_trade": paradox.get(
            "is_governance_blocking_beneficial_trade"
        ),
        "fastest_path_to_yellow": (
            f"{best_scenario['scenario']}: largest weight {best_scenario['largest_position_weight_pct']}% "
            f"→ CPS {best_scenario['capital_preservation_score']} ({best_scenario['escalation_state_estimate']})"
            if best_scenario is not None
            else "—"
        ),
        "fastest_path_to_green": (
            yellow_scenarios.iloc[0]["scenario"]
            if not yellow_scenarios.empty
            else "Requires largest position weight ≤25% with diversified sector book (Scenario C pattern)"
        ),
    }

    report = {
        "generated_at": _now_iso(),
        "executive_report": executive,
        "paradox_analysis": paradox,
        "current_portfolio": {
            "positions": positions,
            "largest_weight_pct": _f(cur_largest),
            "concentration_alerts": cur_alerts,
            "capital_preservation_score": current_cps,
            "escalation_state": cpe.get("escalation_state"),
            "governance_block_reason": (auth.get("gate_reasons") or {}).get("governance"),
            "trace_primary_cause": (trace.get("executive_report") or {}).get("primary_cause"),
        },
        "open_new_count": int(len(candidates)),
        "candidates": candidates.to_dict(orient="records"),
        "scenarios": scenarios.to_dict(orient="records"),
        "methodology": {
            "concentration_threshold_pct": CONCENTRATION_THRESHOLD_PCT,
            "governance_target_largest_weight_pct": GOVERNANCE_TARGET_PCT,
            "permit_escalation_states": ["GREEN", "YELLOW"],
            "new_position_notional": "equal-slot (total_mv / max_positions) scaled by allocation_score",
            "cps_estimate": "Recalculates concentration + exposure components; holds other CPI components constant",
            "sector_mapping": "services/sector_exposure.py get_sector()",
        },
        "disclaimer": (
            "Analysis only — no trades, orders, governance rules, or authorization decisions modified."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_CANDIDATES.name}, {OUT_SCENARIOS.name}")
    _log(f"paradox: {paradox.get('paradox_detected')} — {paradox.get('paradox_summary')}")
    return report


if __name__ == "__main__":
    run()
