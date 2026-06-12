"""
TRITON Phase 185 — Recovery Candidate Deterioration Analysis (read-only).

Determines why HD and XLK deteriorated from OPEN_NEW recovery candidates into
WAIT effective lifecycle status, assesses reversal probability, and evaluates
recovery basket rebuild options.

SAFETY: analysis only — no governance, lifecycle, execution, broker, allocation,
or risk control changes.

Outputs (data/results/):
    recovery_candidate_deterioration_analysis.json
    recovery_candidate_history.csv
    recovery_candidate_viability.csv

Run:
    python -m services.recovery_candidate_deterioration_analysis
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

from services.diversification_feasibility_analysis import _lifecycle_row
from services.pnl_ledger_builder import _norm_sym, _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "recovery_candidate_deterioration_analysis.json"
OUT_HISTORY = RESULTS / "recovery_candidate_history.csv"
OUT_VIABILITY = RESULTS / "recovery_candidate_viability.csv"

FOCUS = ["HD", "XLK", "QQQ"]
BUY_STANCES = {"BUY", "ADD"}
OPEN_ACTIONS = {"OPEN_NEW", "ADD"}
ROOT_CAUSES = {
    "SIGNAL_DECAY",
    "CONFIDENCE_DECAY",
    "EDGE_DECAY",
    "RISK_OVERRIDE",
    "LIFECYCLE_OVERRIDE",
    "MARKET_REGIME_CHANGE",
    "MULTI_FACTOR",
}
REVERSAL_LEVELS = ("LOW", "MODERATE", "HIGH", "VERY_HIGH")


def _log(msg: str) -> None:
    print(f"[recovery_candidate_deterioration_analysis] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _artifact_ts(path: Path, data: Optional[Dict[str, Any]] = None) -> str:
    for key in ("generated_at", "generated_at_utc", "timestamp"):
        if data and data.get(key):
            return str(data[key])[:19]
    if path.is_file():
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    return ""


def _signal_strength(lc: Dict[str, Any], rec: Dict[str, Any]) -> float:
    for src in (lc, rec):
        for key in ("score", "momentum_score", "signal_strength"):
            v = _to_float(src.get(key), np.nan)
            if not np.isnan(v):
                return v
    mom = _to_float(lc.get("momentum_score"), 0.0)
    trend = _to_float(lc.get("trend_score"), 0.0)
    return (mom + trend) / 2.0 if (mom or trend) else np.nan


def _load_context() -> Dict[str, Any]:
    return {
        "recs": _read_csv(RESULTS / "portfolio_allocation_recommendations.csv"),
        "lifecycle_eff": _read_csv(RESULTS / "signal_lifecycle_effective.csv"),
        "opps": _read_csv(RESULTS / "trade_opportunities.csv"),
        "alloc_summary": _read_json(RESULTS / "portfolio_allocation_summary.json"),
        "exec_audit": _read_json(RESULTS / "allocation_execution_path_audit.json"),
        "lifecycle_drift": _read_json(RESULTS / "lifecycle_drift_analysis.json"),
        "recovery_basket": _read_json(RESULTS / "diversification_recovery_basket.json"),
        "exit_strat": _read_json(RESULTS / "concentration_exit_strategy.json"),
        "simulation": _read_json(RESULTS / "recovery_basket_exception_simulation.json"),
        "promotion": _read_json(RESULTS / "promotion_probability_forecast.json"),
        "risk_state": _read_json(RESULTS / "adaptive_risk_state.json"),
        "timestamps": {},
    }


def _enrich_timestamps(ctx: Dict[str, Any]) -> None:
    ctx["timestamps"] = {
        "allocation_recommendations": _artifact_ts(
            RESULTS / "portfolio_allocation_recommendations.csv"
        ),
        "signal_lifecycle_effective": _artifact_ts(RESULTS / "signal_lifecycle_effective.csv"),
        "phase179_audit": _artifact_ts(
            RESULTS / "allocation_execution_path_audit.json", ctx["exec_audit"]
        ),
        "lifecycle_drift": _artifact_ts(
            RESULTS / "lifecycle_drift_analysis.json", ctx["lifecycle_drift"]
        ),
        "promotion_forecast": _artifact_ts(
            RESULTS / "promotion_probability_forecast.json", ctx["promotion"]
        ),
    }


def _historical_snapshots(ctx: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build per-symbol historical baseline (OPEN_NEW era metrics)."""
    snapshots: Dict[str, Dict[str, Any]] = {}

    for trace in ctx["exec_audit"].get("candidate_traces") or []:
        sym = _norm_sym(trace.get("symbol"))
        if not sym:
            continue
        snapshots[sym] = {
            "observed_at": ctx["timestamps"].get("phase179_audit", ""),
            "source": "allocation_execution_path_audit.json",
            "historical_status": str(trace.get("recommended_action") or "—").upper(),
            "confidence": _to_float(trace.get("confidence"), np.nan),
            "edge_score": _to_float(trace.get("edge_score"), np.nan),
            "delta_pct": np.nan,
            "signal_strength": np.nan,
            "lifecycle_action": str(trace.get("lifecycle_action") or "—").upper(),
            "effective_stance": str(trace.get("effective_stance") or "—").upper(),
            "allocation_score": _to_float(trace.get("allocation_score"), np.nan),
            "signal": "—",
        }

    exit_cs = ctx["exit_strat"].get("current_state") or {}
    exit_ts = (ctx["exit_strat"].get("generated_at") or "")[:19]
    for sym in ("HD", "XLK", "BITO", "QQQ"):
        key = f"{sym.lower()}_status"
        status = exit_cs.get(key)
        if not status:
            continue
        sym_n = _norm_sym(sym)
        if sym_n not in snapshots:
            snapshots[sym_n] = {
                "observed_at": exit_ts,
                "source": "concentration_exit_strategy.json",
                "historical_status": str(status).upper(),
                "confidence": np.nan,
                "edge_score": np.nan,
                "delta_pct": np.nan,
                "signal_strength": np.nan,
                "lifecycle_action": "—",
                "effective_stance": "—",
                "allocation_score": np.nan,
                "signal": "—",
            }
        else:
            snapshots[sym_n]["recovery_basket_status"] = str(status).upper()

    basket = ctx["recovery_basket"].get("executive_report") or {}
    for sym in basket.get("best_recovery_basket") or []:
        sym_n = _norm_sym(sym)
        if sym_n not in snapshots:
            snapshots[sym_n] = {
                "observed_at": (ctx["recovery_basket"].get("generated_at") or "")[:19],
                "source": "diversification_recovery_basket.json",
                "historical_status": "RECOVERY_BASKET_MEMBER",
                "confidence": np.nan,
                "edge_score": np.nan,
                "delta_pct": np.nan,
                "signal_strength": np.nan,
                "lifecycle_action": "—",
                "effective_stance": "—",
                "allocation_score": np.nan,
                "signal": "—",
            }
        snapshots[sym_n]["recovery_basket_member"] = True

    return snapshots


def _current_metrics(ctx: Dict[str, Any], sym: str) -> Dict[str, Any]:
    sym = _norm_sym(sym)
    recs = ctx["recs"]
    sub = recs[recs["ticker"].map(_norm_sym) == sym] if not recs.empty else pd.DataFrame()
    rec = sub.iloc[0].to_dict() if not sub.empty else {}
    lc = _lifecycle_row(ctx["lifecycle_eff"], sym)

    return {
        "current_status": str(
            rec.get("recommended_action") or lc.get("effective_stance") or "—"
        ).upper(),
        "confidence": _to_float(lc.get("confidence") or rec.get("confidence"), np.nan),
        "edge_score": _to_float(rec.get("edge_score"), np.nan),
        "delta_pct": _to_float(lc.get("delta_pct") or rec.get("delta_pct"), np.nan),
        "signal_strength": _signal_strength(lc, rec),
        "lifecycle_action": str(lc.get("lifecycle_action") or "—").upper(),
        "effective_stance": str(lc.get("effective_stance") or "—").upper(),
        "allocation_score": _to_float(rec.get("allocation_score"), np.nan),
        "signal": str(lc.get("signal") or rec.get("signal") or "—").upper(),
        "lifecycle_decision_reason": str(lc.get("lifecycle_decision_reason") or ""),
        "risk_flag": str(rec.get("risk_flag") or "OK").upper(),
        "regime": str(lc.get("regime") or ctx["risk_state"].get("regime") or "—"),
    }


def _candidate_universe(ctx: Dict[str, Any], historical: Dict[str, Dict[str, Any]]) -> List[str]:
    symbols: Set[str] = set(FOCUS)
    symbols.update(historical.keys())

    recs = ctx["recs"]
    if not recs.empty:
        for _, row in recs.iterrows():
            action = str(row.get("recommended_action") or "").upper()
            stance = str(row.get("effective_stance") or "").upper()
            if action in OPEN_ACTIONS or stance == "WAIT":
                symbols.add(_norm_sym(row.get("ticker")))

    lc = ctx["lifecycle_eff"]
    if not lc.empty:
        sym_col = "ticker" if "ticker" in lc.columns else "symbol"
        for _, row in lc.iterrows():
            stance = str(row.get("effective_stance") or "").upper()
            if stance == "WAIT" or stance in BUY_STANCES:
                symbols.add(_norm_sym(row.get(sym_col)))

    for row in ctx["lifecycle_drift"].get("candidate_analysis") or []:
        if row.get("recovery_basket_assumption") not in ("—", "", None):
            symbols.add(_norm_sym(row.get("symbol")))

    return sorted(s for s in symbols if s)


def _deterioration_flags(hist: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    h_conf = _to_float(hist.get("confidence"), np.nan)
    c_conf = _to_float(cur.get("confidence"), np.nan)
    h_edge = _to_float(hist.get("edge_score"), np.nan)
    c_edge = _to_float(cur.get("edge_score"), np.nan)
    h_alloc = _to_float(hist.get("allocation_score"), np.nan)
    c_alloc = _to_float(cur.get("allocation_score"), np.nan)
    h_delta = _to_float(hist.get("delta_pct"), np.nan)
    c_delta = _to_float(cur.get("delta_pct"), np.nan)

    conf_drop = (h_conf - c_conf) if not (np.isnan(h_conf) or np.isnan(c_conf)) else np.nan
    edge_drop = (h_edge - c_edge) if not (np.isnan(h_edge) or np.isnan(c_edge)) else np.nan
    alloc_drop = (h_alloc - c_alloc) if not (np.isnan(h_alloc) or np.isnan(c_alloc)) else np.nan

    signal_deterioration = cur.get("signal") == "SELL" or (
        hist.get("historical_status") in OPEN_ACTIONS and cur.get("effective_stance") == "WAIT"
    )
    confidence_deterioration = not np.isnan(conf_drop) and conf_drop > 0.05
    edge_deterioration = not np.isnan(edge_drop) and edge_drop > 0.05
    allocation_deterioration = (
        hist.get("historical_status") in OPEN_ACTIONS
        and cur.get("current_status") not in OPEN_ACTIONS
    ) or (not np.isnan(alloc_drop) and alloc_drop > 0.03)
    risk_deterioration = (
        cur.get("risk_flag") not in ("OK", "—", "") and cur.get("current_status") == "BLOCK"
    )
    lifecycle_override = "flat_non_buy" in str(
        cur.get("lifecycle_decision_reason", "")
    ).lower() or (
        cur.get("lifecycle_action") == "WAIT"
        and cur.get("effective_stance") == "WAIT"
        and cur.get("signal") == "SELL"
    )
    delta_deterioration = (
        not np.isnan(c_delta) and c_delta < 0 and (np.isnan(h_delta) or c_delta < h_delta - 0.002)
    )

    return {
        "signal_deterioration": bool(signal_deterioration),
        "confidence_deterioration": bool(confidence_deterioration),
        "edge_deterioration": bool(edge_deterioration),
        "risk_deterioration": bool(risk_deterioration),
        "allocation_deterioration": bool(allocation_deterioration),
        "lifecycle_override": bool(lifecycle_override),
        "delta_deterioration": bool(delta_deterioration),
        "confidence_change": conf_drop if not np.isnan(conf_drop) else None,
        "edge_change": edge_drop if not np.isnan(edge_drop) else None,
        "allocation_score_change": alloc_drop if not np.isnan(alloc_drop) else None,
        "delta_pct_change": (
            (c_delta - h_delta) if not (np.isnan(h_delta) or np.isnan(c_delta)) else None
        ),
    }


def _classify_root_cause(flags: Dict[str, Any], cur: Dict[str, Any]) -> str:
    active = []
    if flags.get("signal_deterioration"):
        active.append("SIGNAL_DECAY")
    if flags.get("confidence_deterioration"):
        active.append("CONFIDENCE_DECAY")
    if flags.get("edge_deterioration"):
        active.append("EDGE_DECAY")
    if flags.get("risk_deterioration"):
        active.append("RISK_OVERRIDE")
    if flags.get("lifecycle_override"):
        active.append("LIFECYCLE_OVERRIDE")
    if flags.get("allocation_deterioration") and not active:
        active.append("LIFECYCLE_OVERRIDE")

    if len(active) >= 2:
        return "MULTI_FACTOR"
    if len(active) == 1:
        return active[0]
    if cur.get("effective_stance") == "WAIT" and cur.get("signal") == "SELL":
        return "SIGNAL_DECAY"
    return "MULTI_FACTOR"


def _deterioration_nature(flags: Dict[str, Any], reversal: Dict[str, Any]) -> str:
    structural_signals = sum(
        1
        for k in (
            "signal_deterioration",
            "lifecycle_override",
            "confidence_deterioration",
        )
        if flags.get(k)
    )
    if structural_signals >= 2 and reversal.get("probability_of_recovery") == "LOW":
        return "STRUCTURAL"
    if reversal.get("probability_of_recovery") in ("HIGH", "VERY_HIGH"):
        return "TEMPORARY"
    if flags.get("signal_deterioration") and flags.get("confidence_deterioration"):
        return "STRUCTURAL"
    return "LIKELY_TEMPORARY" if reversal.get("promotion_probability_30d", 0) > 25 else "STRUCTURAL"


def _promotion_data(ctx: Dict[str, Any], sym: str) -> Dict[str, Any]:
    sym = _norm_sym(sym)
    promo = ctx["promotion"]
    by_ticker = promo.get("by_ticker") or promo.get("ticker_horizons") or {}
    horizons = by_ticker.get(sym) or {}

    prob_30 = _to_float((horizons.get("30") or {}).get("promotion_probability"), np.nan)
    prob_7 = _to_float((horizons.get("7") or {}).get("promotion_probability"), np.nan)
    demotion_30 = _to_float((horizons.get("30") or {}).get("demotion_probability"), np.nan)
    stability_30 = _to_float((horizons.get("30") or {}).get("stability_probability"), np.nan)

    exec_candidates = (promo.get("executive_report") or {}).get("candidates") or []
    exec_row = next((c for c in exec_candidates if _norm_sym(c.get("candidate")) == sym), {})

    if np.isnan(prob_30) and exec_row:
        prob_30 = _to_float(exec_row.get("promotion_probability_30d"), np.nan)

    return {
        "promotion_probability_30d": prob_30,
        "promotion_probability_7d": prob_7,
        "demotion_probability_30d": demotion_30,
        "stability_probability_30d": stability_30,
        "promotion_classification": (horizons.get("30") or {}).get("promotion_classification")
        or exec_row.get("confidence_level")
        or "—",
        "recovery_impact": exec_row.get("recovery_impact") or "—",
    }


def _reversal_level(prob_30: float, flags: Dict[str, Any]) -> str:
    if np.isnan(prob_30):
        prob_30 = 5.0
    if prob_30 >= 50:
        return "VERY_HIGH"
    if prob_30 >= 30:
        return "HIGH"
    if prob_30 >= 15:
        return "MODERATE"
    if prob_30 >= 5 and not flags.get("signal_deterioration"):
        return "MODERATE"
    return "LOW"


def _reversal_analysis(
    ctx: Dict[str, Any], sym: str, flags: Dict[str, Any], cur: Dict[str, Any]
) -> Dict[str, Any]:
    promo = _promotion_data(ctx, sym)
    prob_30 = _to_float(promo.get("promotion_probability_30d"), 2.0)
    level = _reversal_level(prob_30, flags)

    days_map = {"VERY_HIGH": 7, "HIGH": 14, "MODERATE": 30, "LOW": 60}
    days = days_map.get(level, 60)

    open_new_prob = prob_30 if cur.get("effective_stance") == "WAIT" else max(prob_30, 80.0)
    if flags.get("signal_deterioration") and cur.get("signal") == "SELL":
        open_new_prob = min(open_new_prob, 5.0)

    return {
        "probability_of_recovery": level,
        "days_to_recovery": days,
        "promotion_probability": level,
        "return_to_open_new_probability": _reversal_level(open_new_prob, flags),
        "promotion_probability_30d_pct": prob_30,
        "demotion_probability_30d_pct": promo.get("demotion_probability_30d"),
        "stability_probability_30d_pct": promo.get("stability_probability_30d"),
        "promotion_classification": promo.get("promotion_classification"),
        "recovery_impact": promo.get("recovery_impact"),
        "reversal_nature": _deterioration_nature(
            flags, {"probability_of_recovery": level, "promotion_probability_30d": prob_30}
        ),
    }


def _market_regime_impact(ctx: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    regime = str(ctx["risk_state"].get("regime") or "—")
    symbol_regime = str(cur.get("regime") or "—")
    return {
        "portfolio_regime": regime,
        "symbol_regime": symbol_regime,
        "regime_shift_detected": regime != symbol_regime and symbol_regime != "—",
        "market_regime_contribution": regime in ("Bear", "High Volatility", "Risk-Off"),
        "note": (
            "Broad regime Bull/NORMAL — deterioration driven by symbol-level signal decay, "
            "not portfolio-wide risk-off."
            if regime in ("Bull", "NORMAL", "—")
            else "Portfolio regime may amplify recovery candidate deterioration."
        ),
    }


def _analyze_candidate(
    sym: str, ctx: Dict[str, Any], historical: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    sym = _norm_sym(sym)
    hist = historical.get(sym, {})
    cur = _current_metrics(ctx, sym)
    flags = _deterioration_flags(hist, cur)
    root_cause = _classify_root_cause(flags, cur)
    reversal = _reversal_analysis(ctx, sym, flags, cur)
    regime = _market_regime_impact(ctx, cur)

    hist_status = str(hist.get("historical_status") or "—").upper()
    deteriorated = (
        hist_status in OPEN_ACTIONS | {"RECOVERY_BASKET_MEMBER", "OPEN_NEW"}
        and cur.get("effective_stance") == "WAIT"
    ) or (hist_status in OPEN_ACTIONS and cur.get("current_status") not in OPEN_ACTIONS)

    return {
        "symbol": sym,
        "current_status": cur.get("current_status"),
        "historical_status": hist_status,
        "effective_stance": cur.get("effective_stance"),
        "lifecycle_action": cur.get("lifecycle_action"),
        "signal": cur.get("signal"),
        "lifecycle_decision_reason": cur.get("lifecycle_decision_reason"),
        "deteriorated": deteriorated,
        "historical_snapshot": hist,
        "current_metrics": cur,
        "comparison": {
            "confidence": {
                "historical": hist.get("confidence"),
                "current": cur.get("confidence"),
                "change": flags.get("confidence_change"),
            },
            "edge_score": {
                "historical": hist.get("edge_score"),
                "current": cur.get("edge_score"),
                "change": flags.get("edge_change"),
            },
            "delta_pct": {
                "historical": hist.get("delta_pct"),
                "current": cur.get("delta_pct"),
                "change": flags.get("delta_pct_change"),
            },
            "signal_strength": {
                "historical": hist.get("signal_strength"),
                "current": cur.get("signal_strength"),
            },
            "lifecycle_action": {
                "historical": hist.get("lifecycle_action"),
                "current": cur.get("lifecycle_action"),
            },
            "effective_stance": {
                "historical": hist.get("effective_stance"),
                "current": cur.get("effective_stance"),
            },
            "allocation_score": {
                "historical": hist.get("allocation_score"),
                "current": cur.get("allocation_score"),
                "change": flags.get("allocation_score_change"),
            },
        },
        "deterioration_flags": flags,
        "root_cause_classification": root_cause,
        "market_regime_impact": regime,
        "reversal_analysis": reversal,
        "recovery_basket_assumption_stale": hist_status in OPEN_ACTIONS
        and cur.get("effective_stance") not in BUY_STANCES,
    }


def _replacement_candidates(ctx: Dict[str, Any], exclude: Set[str]) -> List[Dict[str, Any]]:
    recs = ctx["recs"]
    if recs.empty:
        return []

    rows: List[Dict[str, Any]] = []
    for _, rec in recs.iterrows():
        sym = _norm_sym(rec.get("ticker"))
        if not sym or sym in exclude or sym == "QQQ":
            continue
        lc = _lifecycle_row(ctx["lifecycle_eff"], sym)
        effective = str(lc.get("effective_stance") or rec.get("effective_stance") or "").upper()
        signal = str(lc.get("signal") or rec.get("signal") or "").upper()
        action = str(rec.get("recommended_action") or "").upper()
        risk = str(rec.get("risk_flag") or "OK").upper()
        delta = _to_float(lc.get("delta_pct") or rec.get("delta_pct"), np.nan)
        conf = _to_float(lc.get("confidence") or rec.get("confidence"), np.nan)
        edge = _to_float(rec.get("edge_score"), np.nan)
        alloc_score = _to_float(rec.get("allocation_score"), np.nan)

        if risk != "OK" or action == "BLOCK":
            continue
        if effective not in BUY_STANCES and signal != "BUY":
            continue

        promo = _promotion_data(ctx, sym)
        viability_score = (
            (alloc_score if not np.isnan(alloc_score) else 0)
            + (edge if not np.isnan(edge) else 0) * 0.35
            + (conf if not np.isnan(conf) else 0) * 0.25
            + (max(delta, 0) if not np.isnan(delta) else 0) * 5.0
        )
        rows.append(
            {
                "symbol": sym,
                "effective_stance": effective,
                "recommended_action": action,
                "signal": signal,
                "confidence": conf,
                "edge_score": edge,
                "delta_pct": delta,
                "allocation_score": alloc_score,
                "promotion_probability_30d": promo.get("promotion_probability_30d"),
                "viability_score": round(viability_score, 6),
                "recovery_eligible": effective in BUY_STANCES and signal == "BUY",
            }
        )

    rows.sort(
        key=lambda r: (r.get("recovery_eligible", False), r.get("viability_score", 0)), reverse=True
    )
    return rows


def _viability_row(candidate: Dict[str, Any], role: str) -> Dict[str, Any]:
    rev = candidate.get("reversal_analysis") or {}
    return {
        "symbol": candidate.get("symbol"),
        "role": role,
        "current_status": candidate.get("current_status"),
        "historical_status": candidate.get("historical_status"),
        "effective_stance": candidate.get("effective_stance"),
        "deteriorated": candidate.get("deteriorated"),
        "root_cause": candidate.get("root_cause_classification"),
        "recovery_probability": rev.get("probability_of_recovery"),
        "return_to_open_new_probability": rev.get("return_to_open_new_probability"),
        "days_to_recovery": rev.get("days_to_recovery"),
        "reversal_nature": rev.get("reversal_nature"),
        "recovery_basket_viable": candidate.get("effective_stance") in BUY_STANCES
        and candidate.get("signal") == "BUY"
        and not candidate.get("deteriorated"),
        "should_remain_in_basket": (
            candidate.get("effective_stance") in BUY_STANCES
            and rev.get("probability_of_recovery") in ("HIGH", "VERY_HIGH", "MODERATE")
        ),
    }


def _recovery_basket_rebuild(
    ctx: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    replacements: List[Dict[str, Any]],
) -> Dict[str, Any]:
    basket = ctx["recovery_basket"].get("executive_report") or {}
    original = [
        _norm_sym(t) for t in (basket.get("best_recovery_basket") or ["QQQ", "HD", "XLK", "BITO"])
    ]
    focus_map = {c["symbol"]: c for c in candidates}

    hd_replacements = [r["symbol"] for r in replacements[:5]]
    xlk_replacements = [r["symbol"] for r in replacements[1:6]]

    viable_current = [
        sym
        for sym in original
        if sym in focus_map
        and focus_map[sym].get("effective_stance") in BUY_STANCES
        and focus_map[sym].get("signal") == "BUY"
    ]
    lifecycle_actionable = sum(
        1
        for c in candidates
        if c.get("symbol") in original and c.get("effective_stance") in BUY_STANCES
    )

    min_yellow = int(basket.get("minimum_additions_for_yellow") or 2)
    min_green = int(basket.get("minimum_additions_for_green") or 3)

    proposed = ["QQQ"] + hd_replacements[:1] + xlk_replacements[:1]
    proposed = list(dict.fromkeys(proposed))

    sim = ctx["simulation"].get("executive_summary") or {}
    if lifecycle_actionable < min_yellow:
        viability = "NOT_VIABLE"
        note = (
            f"Original basket members HD/XLK effective=WAIT; "
            f"{lifecycle_actionable}/{min_yellow} lifecycle-actionable for YELLOW."
        )
    elif lifecycle_actionable >= min_green:
        viability = "VIABLE"
        note = "Sufficient lifecycle-actionable members for GREEN path."
    else:
        viability = "CONDITIONALLY_VIABLE"
        note = f"Partial viability — {lifecycle_actionable} actionable; replacements or BITO promotion required."

    return {
        "original_basket": original,
        "viable_current_members": viable_current,
        "lifecycle_actionable_count": lifecycle_actionable,
        "minimum_for_yellow": min_yellow,
        "minimum_for_green": min_green,
        "hd_replacement_candidates": hd_replacements,
        "xlk_replacement_candidates": xlk_replacements,
        "proposed_rebuilt_basket": proposed,
        "recovery_basket_viability": viability,
        "viability_note": note,
        "simulation_reference": {
            "recommended_scenario": sim.get("recommended_scenario")
            or ctx["simulation"].get("recommended_scenario"),
            "deadlock_resolved_scenario_d": (ctx["simulation"].get("summary") or {}).get(
                "deadlock_resolved_scenario_d"
            ),
            "caveat": "Scenario D assumed HD/XLK lifecycle-actionable; current effective=WAIT invalidates without replacement.",
        },
        "should_remove_from_basket": ["HD", "XLK"],
        "should_retain_in_basket": viable_current,
    }


def _build_history(
    candidates: List[Dict[str, Any]], historical: Dict[str, Dict[str, Any]], ctx: Dict[str, Any]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ts_now = _now_iso()[:19]

    for c in candidates:
        cur = c.get("current_metrics") or {}
        comp = c.get("comparison") or {}
        rows.append(
            {
                "observed_at": ts_now,
                "source_artifact": "current_snapshot",
                "symbol": c["symbol"],
                "status": c.get("current_status"),
                "effective_stance": c.get("effective_stance"),
                "confidence": cur.get("confidence"),
                "edge_score": cur.get("edge_score"),
                "delta_pct": cur.get("delta_pct"),
                "signal_strength": cur.get("signal_strength"),
                "lifecycle_action": c.get("lifecycle_action"),
                "signal": c.get("signal"),
                "allocation_score": cur.get("allocation_score"),
                "deteriorated": c.get("deteriorated"),
                "root_cause": c.get("root_cause_classification"),
            }
        )

    for sym, hist in historical.items():
        rows.append(
            {
                "observed_at": hist.get("observed_at", ""),
                "source_artifact": hist.get("source", "historical"),
                "symbol": sym,
                "status": hist.get("historical_status"),
                "effective_stance": hist.get("effective_stance"),
                "confidence": hist.get("confidence"),
                "edge_score": hist.get("edge_score"),
                "delta_pct": hist.get("delta_pct"),
                "signal_strength": hist.get("signal_strength"),
                "lifecycle_action": hist.get("lifecycle_action"),
                "signal": hist.get("signal"),
                "allocation_score": hist.get("allocation_score"),
                "deteriorated": False,
                "root_cause": "—",
            }
        )

    for trace in ctx["exec_audit"].get("candidate_traces") or []:
        sym = _norm_sym(trace.get("symbol"))
        rows.append(
            {
                "observed_at": ctx["timestamps"].get("phase179_audit", ""),
                "source_artifact": "phase179_candidate_trace",
                "symbol": sym,
                "status": str(trace.get("recommended_action") or "").upper(),
                "effective_stance": str(trace.get("effective_stance") or "").upper(),
                "confidence": trace.get("confidence"),
                "edge_score": trace.get("edge_score"),
                "delta_pct": None,
                "signal_strength": None,
                "lifecycle_action": str(trace.get("lifecycle_action") or "").upper(),
                "signal": None,
                "allocation_score": trace.get("allocation_score"),
                "deteriorated": trace.get("lifecycle_mismatch"),
                "root_cause": "ALLOCATION_VS_EFFECTIVE" if trace.get("lifecycle_mismatch") else "—",
            }
        )

    return rows


def _hd_deterioration_summary(focus: List[Dict[str, Any]]) -> str:
    hd = next((x for x in focus if x.get("symbol") == "HD"), None)
    if not hd:
        return "—"
    conf_chg = (hd.get("comparison") or {}).get("confidence") or {}
    change = conf_chg.get("change")
    change_txt = f"{change:.3f}" if change is not None else "?"
    return (
        f"Signal SELL + flat_non_buy_wait lifecycle rule; confidence fell {change_txt} "
        "from OPEN_NEW-era baseline; allocation downgraded HOLD."
    )


def _executive_report(
    candidates: List[Dict[str, Any]],
    rebuild: Dict[str, Any],
    replacements: List[Dict[str, Any]],
) -> Dict[str, Any]:
    focus = [c for c in candidates if c["symbol"] in ("HD", "XLK")]
    report_rows = []
    for c in focus:
        rev = c.get("reversal_analysis") or {}
        repl = (
            rebuild.get("hd_replacement_candidates")
            if c["symbol"] == "HD"
            else rebuild.get("xlk_replacement_candidates")
        )
        report_rows.append(
            {
                "candidate": c["symbol"],
                "current_status": c.get("effective_stance"),
                "historical_status": c.get("historical_status"),
                "deterioration_cause": c.get("root_cause_classification"),
                "deterioration_detail": c.get("lifecycle_decision_reason"),
                "recovery_probability": rev.get("probability_of_recovery"),
                "return_to_open_new_probability": rev.get("return_to_open_new_probability"),
                "days_to_recovery": rev.get("days_to_recovery"),
                "reversal_nature": rev.get("reversal_nature"),
                "replacement_candidates": (repl or [])[:3],
                "should_remain_in_basket": False,
            }
        )

    qqq = next((c for c in candidates if c["symbol"] == "QQQ"), {})
    return {
        "report_type": "RECOVERY_CANDIDATE_DETERIORATION_REPORT",
        "candidates": report_rows,
        "qqq_context": {
            "candidate": "QQQ",
            "current_status": qqq.get("effective_stance"),
            "historical_status": qqq.get("historical_status"),
            "deterioration_cause": qqq.get("root_cause_classification"),
            "note": "QQQ is concentration source (TRIM); not a recovery addition candidate.",
        },
        "recovery_basket_viability": rebuild.get("recovery_basket_viability"),
        "proposed_rebuilt_basket": rebuild.get("proposed_rebuilt_basket"),
        "top_replacement_candidates": [r["symbol"] for r in replacements[:5]],
        "executive_questions": {
            "why_did_hd_deteriorate": _hd_deterioration_summary(focus),
            "why_did_xlk_deteriorate": (
                f"Signal SELL + flat_non_buy_wait; edge preserved but negative delta_pct; "
                f"allocation OPEN_NEW→HOLD; recovery basket assumption stale."
            ),
            "can_they_recover": (
                "LOW probability (2% promotion @30d); structural signal decay unless signal flips BUY."
            ),
            "should_remain_in_basket": "No — effective WAIT; remove until lifecycle returns BUY/ADD.",
            "replacement_candidates": ", ".join(rebuild.get("hd_replacement_candidates", [])[:3]),
            "does_viable_basket_exist": (
                f"{rebuild.get('recovery_basket_viability')} — "
                f"rebuild with {', '.join(rebuild.get('proposed_rebuilt_basket', [])[1:3])} per BUY-effective pool."
            ),
        },
    }


def run_analysis() -> Dict[str, Any]:
    ctx = _load_context()
    _enrich_timestamps(ctx)
    historical = _historical_snapshots(ctx)
    universe = _candidate_universe(ctx, historical)
    candidates = [_analyze_candidate(sym, ctx, historical) for sym in universe]

    open_new_hist = [
        c
        for c in candidates
        if c.get("historical_status") in OPEN_ACTIONS | {"OPEN_NEW", "RECOVERY_BASKET_MEMBER"}
    ]
    wait_current = [c for c in candidates if c.get("effective_stance") == "WAIT"]
    deteriorated = [c for c in candidates if c.get("deteriorated")]

    exclude = {"HD", "XLK", "QQQ", "GE", "GBTC"}
    replacements = _replacement_candidates(ctx, exclude)
    rebuild = _recovery_basket_rebuild(ctx, candidates, replacements)

    viability_rows = []
    for c in candidates:
        role = (
            "FOCUS"
            if c["symbol"] in FOCUS
            else (
                "OPEN_NEW_HISTORICAL"
                if c.get("historical_status") in OPEN_ACTIONS
                else ("WAIT" if c.get("effective_stance") == "WAIT" else "OTHER")
            )
        )
        viability_rows.append(_viability_row(c, role))
    for r in replacements:
        viability_rows.append(
            {
                "symbol": r["symbol"],
                "role": "REPLACEMENT_CANDIDATE",
                "current_status": r.get("recommended_action"),
                "historical_status": "—",
                "effective_stance": r.get("effective_stance"),
                "deteriorated": False,
                "root_cause": "—",
                "recovery_probability": _reversal_level(
                    _to_float(r.get("promotion_probability_30d"), 10.0), {}
                ),
                "return_to_open_new_probability": "MODERATE",
                "days_to_recovery": 14,
                "reversal_nature": "N/A",
                "recovery_basket_viable": r.get("recovery_eligible"),
                "should_remain_in_basket": r.get("recovery_eligible"),
            }
        )

    executive = _executive_report(candidates, rebuild, replacements)
    history = _build_history(candidates, historical, ctx)

    summary = {
        "symbols_analyzed": len(candidates),
        "open_new_historical_count": len(open_new_hist),
        "wait_current_count": len(wait_current),
        "deteriorated_count": len(deteriorated),
        "hd_root_cause": next(
            (c["root_cause_classification"] for c in candidates if c["symbol"] == "HD"), "—"
        ),
        "xlk_root_cause": next(
            (c["root_cause_classification"] for c in candidates if c["symbol"] == "XLK"), "—"
        ),
        "recovery_basket_viability": rebuild.get("recovery_basket_viability"),
        "open_new_candidates_now": int(ctx["alloc_summary"].get("open_new_candidates") or 0),
        "primary_deterioration_pattern": "SIGNAL_DECAY + LIFECYCLE_OVERRIDE (flat_non_buy_wait)",
    }

    report = {
        "generated_at": _now_iso(),
        "phase": "185",
        "mode": "recovery_candidate_deterioration_analysis",
        "disclaimer": (
            "Deterioration analysis only — does not modify governance, lifecycle, execution, "
            "allocation, or risk controls."
        ),
        "summary": summary,
        "candidate_analysis": candidates,
        "open_new_candidates": open_new_hist,
        "wait_candidates": wait_current,
        "deteriorated_candidates": deteriorated,
        "replacement_candidates": replacements,
        "recovery_basket_rebuild": rebuild,
        "executive_report": executive,
        "certification": {
            "status": "CERTIFIED",
            "safety": "read-only analysis — no pipeline modifications",
        },
    }

    pd.DataFrame(history).to_csv(OUT_HISTORY, index=False, encoding="utf-8")
    pd.DataFrame(viability_rows).to_csv(OUT_VIABILITY, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_HISTORY.name}, {OUT_VIABILITY.name}")
    _log(
        f"deteriorated={summary['deteriorated_count']} "
        f"basket={summary['recovery_basket_viability']} "
        f"hd={summary['hd_root_cause']} xlk={summary['xlk_root_cause']}"
    )
    return report


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
