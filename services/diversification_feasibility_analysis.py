"""
TRITON Phase 176 — Diversification Basket Feasibility Analysis (read-only).

Validates whether the Phase 175 governance recovery basket can actually be
assembled using current lifecycle-approved opportunities and allocation gates.

SAFETY: analysis only — no governance, execution, lifecycle, broker, or
allocation logic changes.

Outputs (data/results/):
    diversification_feasibility_analysis.json
    diversification_feasibility_candidates.csv
    diversification_feasibility_scenarios.csv

Run:
    python -m services.diversification_feasibility_analysis
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from services.concentration_governance_analysis import (
    _estimate_cps,
    _escalation_from_cps,
    _f,
    _health_band,
    _load_positions,
    _read_csv,
    _read_json,
)
from services.pnl_ledger_builder import _norm_sym, _to_float
from services.sector_exposure import get_sector

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "diversification_feasibility_analysis.json"
OUT_CANDIDATES = RESULTS / "diversification_feasibility_candidates.csv"
OUT_SCENARIOS = RESULTS / "diversification_feasibility_scenarios.csv"

MIN_OPEN_NEW_CONFIDENCE = 0.50
STRONG_EDGE_SCORE = 0.65
PERMIT_ESCALATION = {"GREEN", "YELLOW"}
BASKET_FOCUS = ["HD", "XLK", "BITO"]


def _log(msg: str) -> None:
    print(f"[diversification_feasibility_analysis] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _equal_weight(held: List[str], adds: List[str], total_mv: float) -> List[Dict[str, Any]]:
    names = list(dict.fromkeys(held + adds))
    if not names or total_mv <= 0:
        return []
    w = total_mv / len(names)
    return [{"symbol": s, "market_value": w, "qty": np.nan} for s in names]


def _lifecycle_row(lc: pd.DataFrame, sym: str) -> Dict[str, Any]:
    sub = lc[lc["ticker"].map(_norm_sym) == sym]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def _classify_feasibility(
    rec: Dict[str, Any],
    lc: Dict[str, Any],
    blocked_syms: Set[str],
    in_recovery_basket: bool,
) -> tuple[str, List[str]]:
    blockers: List[str] = []
    sym = _norm_sym(rec.get("ticker") or lc.get("ticker"))
    action = str(rec.get("recommended_action") or "").upper()
    lc_action = str(lc.get("lifecycle_action") or rec.get("lifecycle_action") or "").upper()
    stance = str(lc.get("effective_stance") or rec.get("effective_stance") or "").upper()
    signal = str(rec.get("signal") or lc.get("signal") or "").upper()
    conf = _to_float(rec.get("confidence") or lc.get("confidence"), np.nan)
    edge = _to_float(rec.get("edge_score"), np.nan)
    risk = str(rec.get("risk_flag") or "OK").upper()
    exec_blocked = lc.get("execution_blocked") is True
    reason = str(rec.get("reason") or "")

    if sym in blocked_syms or "BLOCK_NEW_BUY" in risk or "FORCE_EXIT" in risk:
        blockers.append("risk_flag or allocation block list")
    if exec_blocked:
        blockers.append(f"execution_blocked: {lc.get('execution_block_reason') or 'unknown'}")
    if lc_action not in ("BUY", "ADD") and stance not in ("BUY", "ADD"):
        blockers.append(f"lifecycle not buy-ready ({lc_action}/{stance})")
    if signal not in ("BUY",) and stance not in ("BUY", "ADD"):
        blockers.append(f"signal not positive ({signal})")

    open_new_ok = action == "OPEN_NEW"
    conf_ok = np.isfinite(conf) and conf >= MIN_OPEN_NEW_CONFIDENCE
    edge_ok = np.isfinite(edge) and edge >= STRONG_EDGE_SCORE
    lifecycle_ok = lc_action == "BUY" and stance == "BUY" and not blockers

    if lifecycle_ok and open_new_ok and conf_ok and not blockers:
        return "IMMEDIATELY_FEASIBLE", blockers

    if lifecycle_ok and not blockers and (action == "WATCH" or not conf_ok):
        if np.isfinite(conf) and conf >= 0.45 and edge_ok:
            blockers.append(
                f"allocation WATCH — confidence {conf:.3f} below OPEN_NEW floor {MIN_OPEN_NEW_CONFIDENCE}"
                if not conf_ok
                else "allocation WATCH — awaiting OPEN_NEW promotion"
            )
            return "FEASIBLE_WITH_TIME", blockers
        if action == "WATCH" and edge_ok:
            blockers.append("allocation WATCH — strong edge but not yet OPEN_NEW")
            return "FEASIBLE_WITH_TIME", blockers

    if not blockers and in_recovery_basket:
        blockers.append("recovery basket member but lifecycle/allocation gates not fully met")

    if blockers:
        return "CURRENTLY_UNAVAILABLE", blockers
    return "FEASIBLE_WITH_TIME", blockers or ["pending allocation promotion"]


def _evaluate_candidates(
    recovery_basket: List[str],
    recovery_additions: List[str],
    recommendations: pd.DataFrame,
    lifecycle: pd.DataFrame,
    alloc_summary: Dict[str, Any],
    recovery_rankings: Dict[str, float],
) -> pd.DataFrame:
    blocked = set(alloc_summary.get("blocked_new_buy_symbols") or [])
    focus = set(BASKET_FOCUS) | set(recovery_additions)

    open_new = recommendations[
        recommendations["recommended_action"].astype(str).str.upper() == "OPEN_NEW"
    ]
    eval_syms = list(dict.fromkeys(list(open_new["ticker"].map(_norm_sym)) + list(focus)))

    rows: List[Dict[str, Any]] = []
    for sym in eval_syms:
        rec_row = recommendations[recommendations["ticker"].map(_norm_sym) == sym]
        rec = rec_row.iloc[0].to_dict() if not rec_row.empty else {"ticker": sym}
        lc = _lifecycle_row(lifecycle, sym)
        in_basket = sym in recovery_basket or sym in recovery_additions
        feasibility, blockers = _classify_feasibility(rec, lc, blocked, in_basket)

        rows.append(
            {
                "ticker": sym,
                "in_recovery_basket": in_basket,
                "lifecycle_status": str(
                    lc.get("lifecycle_action") or rec.get("lifecycle_action") or "—"
                ),
                "effective_stance": str(
                    lc.get("effective_stance") or rec.get("effective_stance") or "—"
                ),
                "recommended_action": str(rec.get("recommended_action") or "—"),
                "confidence": _f(rec.get("confidence") or lc.get("confidence"), 4),
                "edge_score": _f(rec.get("edge_score"), 4),
                "allocation_score": _f(rec.get("allocation_score"), 4),
                "risk_flags": str(rec.get("risk_flag") or "OK"),
                "execution_blocked": bool(lc.get("execution_blocked")),
                "feasibility": feasibility,
                "blockers": "; ".join(blockers) if blockers else "",
                "governance_recovery_contribution": _f(recovery_rankings.get(sym), 1),
                "sector": get_sector(sym),
            }
        )

    df = pd.DataFrame(rows)
    order = {"IMMEDIATELY_FEASIBLE": 0, "FEASIBLE_WITH_TIME": 1, "CURRENTLY_UNAVAILABLE": 2}
    if not df.empty:
        df["_ord"] = df["feasibility"].map(lambda x: order.get(str(x), 9))
        df = df.sort_values(["_ord", "governance_recovery_contribution"], ascending=[True, False])
        df = df.drop(columns=["_ord"]).reset_index(drop=True)
    return df


def _scenario_metrics(
    name: str,
    held: List[str],
    additions: List[str],
    total_mv: float,
    base_components: Dict[str, int],
    feasible_count: int,
    required_count: int,
) -> Dict[str, Any]:
    positions = _equal_weight(held, additions, total_mv)
    cps, _ = _estimate_cps(positions, base_components)
    esc = _escalation_from_cps(cps)
    compat = feasible_count / required_count if required_count else 1.0
    prob = float(np.clip(compat * (1.0 if esc in PERMIT_ESCALATION else 0.7), 0.0, 1.0))
    return {
        "scenario": name,
        "basket_tickers": ",".join(additions) if additions else "—",
        "position_count": len(positions),
        "capital_preservation_score": cps,
        "escalation_state": esc,
        "governance_status": "AUTHORIZED" if esc in PERMIT_ESCALATION else "BLOCKED",
        "lifecycle_compatibility_pct": _f(compat * 100, 1),
        "recovery_probability_pct": _f(prob * 100, 1),
        "immediately_feasible_count": feasible_count,
        "required_additions": required_count,
    }


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)

    recovery = _read_json(RESULTS / "diversification_recovery_basket.json")
    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    lifecycle = _read_csv(RESULTS / "signal_lifecycle_effective.csv")
    alloc_summary = _read_json(RESULTS / "portfolio_allocation_summary.json")
    cpi = _read_json(RESULTS / "capital_preservation_intelligence.json")
    auth = _read_json(RESULTS / "governance_authorization.json")

    positions = _load_positions()
    held = [p["symbol"] for p in positions]
    total_mv = sum(p["market_value"] for p in positions)

    base_components = dict(cpi.get("component_scores") or {})
    for k in ("drawdown", "concentration", "exposure", "operational", "execution"):
        base_components.setdefault(k, 70)

    basket_members = list(recovery.get("executive_report", {}).get("best_recovery_basket") or [])
    green_adds = list((recovery.get("fastest_path_to_green") or {}).get("green_tickers") or [])
    recovery_adds = [s for s in green_adds if s not in held]

    rankings: Dict[str, float] = {}
    for row in recovery.get("best_diversification_candidates") or []:
        sym = _norm_sym(row.get("ticker"))
        if sym:
            rankings[sym] = _to_float(row.get("governance_improvement_score"), 0.0)

    candidates = _evaluate_candidates(
        basket_members, recovery_adds, recommendations, lifecycle, alloc_summary, rankings
    )
    candidates.to_csv(OUT_CANDIDATES, index=False, encoding="utf-8")

    immediately = candidates[candidates["feasibility"] == "IMMEDIATELY_FEASIBLE"]["ticker"].tolist()
    with_time = candidates[candidates["feasibility"] == "FEASIBLE_WITH_TIME"]["ticker"].tolist()
    unavailable = candidates[candidates["feasibility"] == "CURRENTLY_UNAVAILABLE"][
        "ticker"
    ].tolist()

    feasible_basket = [s for s in recovery_adds if s in immediately]
    missing = [s for s in recovery_adds if s not in immediately]

    scenarios = [
        _scenario_metrics(
            "Scenario A — Current Portfolio",
            held,
            [],
            total_mv,
            base_components,
            0,
            len(recovery_adds),
        ),
        _scenario_metrics(
            "Scenario B — Feasible Basket Only",
            held,
            feasible_basket,
            total_mv,
            base_components,
            len(feasible_basket),
            len(recovery_adds),
        ),
        _scenario_metrics(
            "Scenario C — Full Governance Recovery Basket",
            held,
            recovery_adds,
            total_mv,
            base_components,
            len(feasible_basket),
            len(recovery_adds),
        ),
    ]
    pd.DataFrame(scenarios).to_csv(OUT_SCENARIOS, index=False, encoding="utf-8")

    # Per-ticker key questions
    key_answers = {}
    for sym in BASKET_FOCUS:
        row = candidates[candidates["ticker"] == sym]
        if row.empty:
            key_answers[sym] = {"can_acquire": False, "reason": "not in evaluation set"}
        else:
            r = row.iloc[0]
            key_answers[sym] = {
                "can_acquire_now": r["feasibility"] == "IMMEDIATELY_FEASIBLE",
                "feasibility": r["feasibility"],
                "lifecycle_status": r["lifecycle_status"],
                "recommended_action": r["recommended_action"],
                "blockers": r["blockers"],
            }

    basket_assemblable_now = len(feasible_basket) == len(recovery_adds)
    partial_yellow = _scenario_metrics(
        "partial_feasible",
        held,
        feasible_basket,
        total_mv,
        base_components,
        len(feasible_basket),
        len(recovery_adds),
    )

    fastest_realistic = (
        f"Assemble {', '.join(feasible_basket)} now → {partial_yellow['escalation_state']} "
        f"(CPS {partial_yellow['capital_preservation_score']})"
        if feasible_basket
        else "No immediately feasible additions — wait for lifecycle promotion"
    )
    if missing:
        fastest_realistic += (
            f"; then add {', '.join(missing)} when FEASIBLE_WITH_TIME clears "
            f"(confidence ≥ {MIN_OPEN_NEW_CONFIDENCE} or OPEN_NEW promotion)"
        )

    executive = {
        "recovery_basket": basket_members,
        "recovery_additions_required": recovery_adds,
        "lifecycle_status_summary": {
            "immediately_feasible": immediately,
            "feasible_with_time": with_time,
            "currently_unavailable": unavailable,
        },
        "missing_components": missing,
        "current_feasibility": (
            "FULLY_FEASIBLE"
            if basket_assemblable_now
            else "PARTIALLY_FEASIBLE" if feasible_basket else "NOT_FEASIBLE"
        ),
        "basket_assemblable_now": basket_assemblable_now,
        "fastest_achievable_recovery_path": fastest_realistic,
        "projected_cps_feasible_only": partial_yellow["capital_preservation_score"],
        "projected_escalation_feasible_only": partial_yellow["escalation_state"],
        "projected_cps_full_basket": scenarios[2]["capital_preservation_score"],
        "projected_escalation_full_basket": scenarios[2]["escalation_state"],
        "governance_currently_authorized": auth.get("governance_authorized"),
    }

    report = {
        "generated_at": _now_iso(),
        "executive_report": executive,
        "key_questions": {
            "can_hd_be_acquired": key_answers.get("HD"),
            "can_xlk_be_acquired": key_answers.get("XLK"),
            "can_bito_be_acquired": key_answers.get("BITO"),
            "can_recovery_basket_be_assembled": {
                "now": basket_assemblable_now,
                "immediately_available": feasible_basket,
                "missing": missing,
                "missing_reason": (
                    "; ".join(
                        f"{s}: {candidates[candidates['ticker']==s]['blockers'].iloc[0]}"
                        for s in missing
                        if not candidates[candidates["ticker"] == s].empty
                    )
                    if missing
                    else "none"
                ),
            },
        },
        "feasibility_candidates": candidates.to_dict(orient="records"),
        "scenarios": scenarios,
        "certification": {
            "status": "CERTIFIED" if not candidates.empty else "NOT_CERTIFIED",
            "candidates_evaluated": int(len(candidates)),
            "safety": "analysis-only — no governance, execution, lifecycle, or allocation changes",
        },
        "disclaimer": (
            "Feasibility analysis only — validates lifecycle/allocation compatibility "
            "with the Phase 175 recovery basket design."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_CANDIDATES.name}, {OUT_SCENARIOS.name}")
    _log(f"feasibility: {executive['current_feasibility']} — feasible now: {feasible_basket}")
    _log(f"missing: {missing}")
    return report


if __name__ == "__main__":
    run()
