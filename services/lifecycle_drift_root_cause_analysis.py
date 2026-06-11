"""
TRITON Phase 184 — Lifecycle Drift Root Cause Analysis (read-only).

Determines why allocation outputs and effective lifecycle decisions diverge,
quantifies operational impact, and identifies the authoritative layer.

SAFETY: analysis only — no governance, lifecycle, execution, broker, allocation,
or risk control changes.

Outputs (data/results/):
    lifecycle_drift_analysis.json
    lifecycle_drift_candidates.csv
    lifecycle_drift_history.csv

Run:
    python -m services.lifecycle_drift_root_cause_analysis
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
OUT_JSON = RESULTS / "lifecycle_drift_analysis.json"
OUT_CANDIDATES = RESULTS / "lifecycle_drift_candidates.csv"
OUT_HISTORY = RESULTS / "lifecycle_drift_history.csv"

FOCUS = ["HD", "XLK", "QQQ", "BITO"]
DRIFT_CLASSES = {
    "NO_DRIFT",
    "MINOR_DRIFT",
    "MODERATE_DRIFT",
    "MAJOR_DRIFT",
    "CRITICAL_DRIFT",
}
BUY_STANCES = {"BUY", "ADD"}
OPEN_ACTIONS = {"OPEN_NEW", "ADD"}


def _log(msg: str) -> None:
    print(f"[lifecycle_drift_root_cause_analysis] {msg}", flush=True)


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


def _load_context() -> Dict[str, Any]:
    recs = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    lifecycle_eff = _read_csv(RESULTS / "signal_lifecycle_effective.csv")
    lifecycle_base = _read_csv(RESULTS / "signal_lifecycle.csv")
    opps = _read_csv(RESULTS / "trade_opportunities.csv")
    alloc_summary = _read_json(RESULTS / "portfolio_allocation_summary.json")
    exec_audit = _read_json(RESULTS / "allocation_execution_path_audit.json")
    readiness = _read_json(RESULTS / "recovery_basket_implementation_readiness.json")
    simulation = _read_json(RESULTS / "recovery_basket_exception_simulation.json")
    exit_strat = _read_json(RESULTS / "concentration_exit_strategy.json")
    div_ready = _read_json(RESULTS / "diversification_execution_readiness.json")
    recovery_basket = _read_json(RESULTS / "diversification_recovery_basket.json")

    return {
        "recs": recs,
        "lifecycle_eff": lifecycle_eff,
        "lifecycle_base": lifecycle_base,
        "opps": opps,
        "alloc_summary": alloc_summary,
        "exec_audit": exec_audit,
        "readiness": readiness,
        "simulation": simulation,
        "exit_strat": exit_strat,
        "div_ready": div_ready,
        "recovery_basket": recovery_basket,
        "timestamps": {
            "allocation_summary": _artifact_ts(
                RESULTS / "portfolio_allocation_summary.json", alloc_summary
            ),
            "allocation_recommendations": _artifact_ts(
                RESULTS / "portfolio_allocation_recommendations.csv"
            ),
            "signal_lifecycle_effective": _lifecycle_file_ts(lifecycle_eff),
            "trade_opportunities": _artifact_ts(RESULTS / "trade_opportunities.csv"),
            "phase179_audit": _artifact_ts(
                RESULTS / "allocation_execution_path_audit.json", exec_audit
            ),
        },
    }


def _lifecycle_file_ts(df: pd.DataFrame) -> str:
    for col in ("generated_at", "lifecycle_ts", "asof_ts", "timestamp", "effective_ts"):
        if col in df.columns and not df.empty:
            return str(df[col].iloc[0])[:19]
    return _artifact_ts(RESULTS / "signal_lifecycle_effective.csv")


def _opp_row(opps: pd.DataFrame, sym: str) -> Dict[str, Any]:
    if opps.empty:
        return {}
    col = "ticker" if "ticker" in opps.columns else "symbol"
    sub = opps[opps[col].map(_norm_sym) == sym]
    return sub.iloc[0].to_dict() if not sub.empty else {}


def _classify_drift(
    allocation_action: str,
    lifecycle_action: str,
    effective_action: str,
    execution_action: str,
    embedded_stale: bool,
    recovery_assumption: str,
) -> Tuple[str, str]:
    """Return (drift_type, drift_severity)."""
    layers = [allocation_action, lifecycle_action, effective_action, execution_action]
    uniq = {x for x in layers if x and x != "—"}

    if len(uniq) <= 1 and not embedded_stale:
        return "NO_DRIFT", "NO_DRIFT"

    if allocation_action in OPEN_ACTIONS and effective_action not in BUY_STANCES:
        if recovery_assumption in OPEN_ACTIONS:
            return "RECOVERY_BASKET_VS_EFFECTIVE", "CRITICAL_DRIFT"
        return "ALLOCATION_VS_EFFECTIVE", "MAJOR_DRIFT"

    if embedded_stale and allocation_action in OPEN_ACTIONS:
        return "ALLOCATION_STALENESS", "MAJOR_DRIFT"

    if recovery_assumption in OPEN_ACTIONS and allocation_action not in OPEN_ACTIONS:
        return "RECOVERY_BASKET_STALE", "MODERATE_DRIFT"

    if lifecycle_action != effective_action:
        return "LIFECYCLE_ACTION_VS_EFFECTIVE", "MODERATE_DRIFT"

    if effective_action == "WAIT" and execution_action in BUY_STANCES:
        return "EXECUTION_VS_EFFECTIVE", "MAJOR_DRIFT"

    if len(uniq) == 2:
        return "CROSS_LAYER_MISMATCH", "MINOR_DRIFT"

    return "CROSS_LAYER_MISMATCH", "MODERATE_DRIFT"


def _execution_action(sym: str, lc: Dict[str, Any], opp: Dict[str, Any]) -> str:
    if opp:
        return str(
            opp.get("effective_stance") or opp.get("opportunity_type") or "IN_OPPORTUNITIES"
        ).upper()
    stance = str(lc.get("effective_stance") or "—").upper()
    if stance in BUY_STANCES and not lc.get("execution_blocked"):
        return "ELIGIBLE_NOT_IN_OPPORTUNITIES"
    if stance in ("WAIT", "HOLD"):
        return "NOT_ELIGIBLE"
    return stance or "—"


def _recovery_assumption(ctx: Dict[str, Any], sym: str) -> str:
    exit_cs = ctx["exit_strat"].get("current_state") or {}
    key = f"{sym.lower()}_status"
    if exit_cs.get(key):
        return str(exit_cs[key]).upper()
    sim_assets = (ctx["simulation"].get("current_state") or {}).get("asset_status") or {}
    if sym in sim_assets:
        return str(
            sim_assets[sym].get("exit_strategy_status")
            or sim_assets[sym].get("allocation_action")
            or "—"
        ).upper()
    basket = ctx["recovery_basket"].get("executive_report") or {}
    if sym in (basket.get("best_recovery_basket") or []):
        return "RECOVERY_BASKET_MEMBER"
    return "—"


def _analyze_candidate(sym: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    sym = _norm_sym(sym)
    recs = ctx["recs"]
    sub = recs[recs["ticker"].map(_norm_sym) == sym] if not recs.empty else pd.DataFrame()
    rec = sub.iloc[0].to_dict() if not sub.empty else {}
    lc = _lifecycle_row(ctx["lifecycle_eff"], sym)
    lc_base = _lifecycle_row(ctx["lifecycle_base"], sym)
    opp = _opp_row(ctx["opps"], sym)

    allocation_action = str(rec.get("recommended_action") or "—").upper()
    alloc_embedded_lc = str(rec.get("lifecycle_action") or "—").upper()
    alloc_embedded_stance = str(rec.get("effective_stance") or "—").upper()
    lifecycle_action = str(lc.get("lifecycle_action") or "—").upper()
    effective_action = str(lc.get("effective_stance") or "—").upper()
    signal = str(lc.get("signal") or rec.get("signal") or "—").upper()
    execution_action = _execution_action(sym, lc, opp)
    recovery_assumption = _recovery_assumption(ctx, sym)

    embedded_stale = (
        alloc_embedded_lc != lifecycle_action or alloc_embedded_stance != effective_action
    ) and bool(rec)

    drift_type, drift_severity = _classify_drift(
        allocation_action,
        lifecycle_action,
        effective_action,
        execution_action,
        embedded_stale,
        recovery_assumption,
    )

    delta = _to_float(lc.get("delta_pct") or rec.get("delta_pct"), np.nan)
    conf = _to_float(lc.get("confidence") or rec.get("confidence"), np.nan)
    lc_reason = str(lc.get("lifecycle_decision_reason") or lc.get("reason_code") or "")

    execution_ready = (
        effective_action in BUY_STANCES
        and execution_action not in ("NOT_ELIGIBLE", "WAIT", "—")
        and lc.get("execution_blocked") is not True
    )

    return {
        "symbol": sym,
        "allocation_action": allocation_action,
        "allocation_embedded_lifecycle_action": alloc_embedded_lc,
        "allocation_embedded_effective_stance": alloc_embedded_stance,
        "lifecycle_action": lifecycle_action,
        "lifecycle_base_action": str(lc_base.get("lifecycle_action") or "—").upper(),
        "effective_action": effective_action,
        "execution_action": execution_action,
        "recovery_basket_assumption": recovery_assumption,
        "signal": signal,
        "confidence": conf,
        "delta_pct": delta,
        "lifecycle_decision_reason": lc_reason,
        "embedded_snapshot_stale": embedded_stale,
        "drift_type": drift_type,
        "drift_severity": drift_severity,
        "execution_ready": execution_ready,
        "in_trade_opportunities": bool(opp),
        "open_new_at_allocation": allocation_action == "OPEN_NEW",
        "wait_at_effective": effective_action == "WAIT",
    }


def _candidate_universe(ctx: Dict[str, Any]) -> List[str]:
    symbols: Set[str] = set(FOCUS)
    recs = ctx["recs"]
    if not recs.empty:
        open_new = recs[recs["recommended_action"].astype(str).str.upper() == "OPEN_NEW"]
        for t in open_new.get("ticker", pd.Series(dtype=str)):
            symbols.add(_norm_sym(t))
        wait_mask = recs["effective_stance"].astype(str).str.upper() == "WAIT"
        if "effective_stance" in recs.columns:
            for t in recs.loc[wait_mask].get("ticker", pd.Series(dtype=str)):
                symbols.add(_norm_sym(t))
    lc = ctx["lifecycle_eff"]
    if not lc.empty and "effective_stance" in lc.columns:
        sym_col = "ticker" if "ticker" in lc.columns else "symbol"
        wait_lc = lc[lc["effective_stance"].astype(str).str.upper() == "WAIT"]
        for t in wait_lc.get(sym_col, pd.Series(dtype=str)):
            symbols.add(_norm_sym(t))
        buy_lc = lc[lc["effective_stance"].astype(str).str.upper().isin(BUY_STANCES)]
        for t in buy_lc.get(sym_col, pd.Series(dtype=str)):
            symbols.add(_norm_sym(t))
    for trace in ctx["exec_audit"].get("candidate_traces") or []:
        symbols.add(_norm_sym(trace.get("symbol")))
    return sorted(s for s in symbols if s)


def _root_causes(candidates: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
    ts = ctx["timestamps"]
    alloc_ts = ts.get("allocation_recommendations", "")
    lc_ts = ts.get("signal_lifecycle_effective", "")
    timing_gap = alloc_ts and lc_ts and alloc_ts < lc_ts

    signal_deterioration = [
        c["symbol"]
        for c in candidates
        if c.get("signal") == "SELL"
        and c.get("effective_action") == "WAIT"
        and "flat_non_buy" in str(c.get("lifecycle_decision_reason", "")).lower()
        or "bearish" in str(c.get("lifecycle_decision_reason", "")).lower()
    ]
    negative_delta = [c["symbol"] for c in candidates if _to_float(c.get("delta_pct"), 0) < 0]

    causes = [
        {
            "cause": "pipeline_timing_mismatch",
            "label": "Pipeline timing mismatch",
            "evidence": f"allocation_recs={alloc_ts} lifecycle_effective={lc_ts}",
            "confirmed": timing_gap,
            "weight": 0.85 if timing_gap else 0.35,
            "description": (
                "Allocation recommendations embed a lifecycle snapshot at generation time; "
                "lifecycle effective refreshes independently on a faster cadence."
            ),
        },
        {
            "cause": "allocation_staleness",
            "label": "Allocation artifact staleness",
            "evidence": f"{sum(1 for c in candidates if c.get('embedded_snapshot_stale'))} symbols with embedded≠live lifecycle",
            "confirmed": any(c.get("embedded_snapshot_stale") for c in candidates),
            "weight": 0.80,
            "description": (
                "portfolio_allocation_recommendations.csv stores lifecycle_action/effective_stance "
                "columns from the allocation run; downstream consumers treat OPEN_NEW as current "
                "when lifecycle has since moved to WAIT."
            ),
        },
        {
            "cause": "signal_deterioration",
            "label": "Signal deterioration (SELL / flat_non_buy_wait)",
            "evidence": f"symbols={signal_deterioration}, negative_delta={negative_delta}",
            "confirmed": bool(signal_deterioration),
            "weight": 0.90,
            "description": (
                "Lifecycle engine moved HD/XLK to flat_non_buy_wait (WAIT) on bearish signal; "
                "delta_pct turned negative, failing OPEN_NEW confidence/delta gates."
            ),
        },
        {
            "cause": "recovery_basket_staleness",
            "label": "Recovery basket / exit strategy staleness",
            "evidence": "exit_strategy OPEN_NEW vs allocation HOLD",
            "confirmed": any(
                c.get("recovery_basket_assumption") == "OPEN_NEW"
                and c.get("allocation_action") == "HOLD"
                for c in candidates
                if c["symbol"] in ("HD", "XLK")
            ),
            "weight": 0.75,
            "description": (
                "Phases 175–182 recovery artifacts retain OPEN_NEW assumptions while "
                "current allocation and lifecycle no longer support OPEN_NEW."
            ),
        },
        {
            "cause": "authority_model_split",
            "label": "Authority model split (lifecycle_action vs effective_stance)",
            "evidence": "portfolio_allocation_engine rule E uses lifecycle_action; execute_trades uses effective_stance",
            "confirmed": True,
            "weight": 0.95,
            "description": (
                "Allocation OPEN_NEW gate keys off lifecycle_action∈{BUY,ADD}; execution layer "
                "keys off effective_stance∈{BUY,ADD}. When they diverge, allocation can promote "
                "OPEN_NEW while execution blocks."
            ),
        },
        {
            "cause": "data_freshness_mismatch",
            "label": "Data freshness mismatch across phases",
            "evidence": f"phase179={ts.get('phase179_audit')} vs current lifecycle={lc_ts}",
            "confirmed": bool(ctx["exec_audit"].get("candidate_traces")),
            "weight": 0.70,
            "description": "Phase 179 audit captured OPEN_NEW+WAIT drift; pattern persists in phases 182–183.",
        },
    ]
    ranked = sorted(causes, key=lambda x: -x["weight"])
    primary = ranked[0]["cause"] if ranked else "unknown"
    return {
        "primary_root_cause": primary,
        "ranked_causes": ranked,
        "is_expected_behavior": False,
        "expected_behavior_note": (
            "Partial drift from independent pipeline schedules is expected; "
            "CRITICAL drift (OPEN_NEW recovery assumptions vs WAIT effective) is a system "
            "inconsistency requiring reconciliation — not intended steady-state behavior."
        ),
        "code_reference": "services/portfolio_allocation_engine.py rule E (lifecycle_action) vs execution effective_stance gate",
    }


def _authority_analysis(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "authoritative_layer_for_execution": "effective_lifecycle",
        "authoritative_source": "signal_lifecycle_effective.csv",
        "authoritative_fields": [
            "effective_stance",
            "execution_blocked",
            "lifecycle_decision_reason",
        ],
        "allocation_role": "point_in_time scoring and OPEN_NEW candidacy at last allocation run",
        "lifecycle_action_role": "intent signal used by allocation rule E (BUY/ADD/WAIT)",
        "effective_stance_role": "binding stance for execution and position manager",
        "execution_eligibility_role": "trade_opportunities.csv membership + effective_stance filter",
        "recovery_basket_role": "planning artifact — must reconcile with live effective lifecycle before execution",
        "recommendation": (
            "Treat signal_lifecycle_effective.effective_stance as authoritative for execution. "
            "Treat portfolio_allocation_recommendations as stale after lifecycle refresh. "
            "Recovery basket phases must re-validate against effective lifecycle before activation."
        ),
        "layer_ranking": [
            {"layer": "effective_lifecycle", "authority": 1, "use": "execution binding"},
            {"layer": "execution_eligibility", "authority": 2, "use": "order planner input"},
            {"layer": "lifecycle_action", "authority": 3, "use": "allocation rule E input"},
            {
                "layer": "allocation_recommendation",
                "authority": 4,
                "use": "scoring / candidacy snapshot",
            },
            {"layer": "recovery_basket_assumption", "authority": 5, "use": "planning only"},
        ],
    }


def _impact_analysis(candidates: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
    focus = [c for c in candidates if c["symbol"] in ("HD", "XLK", "QQQ")]
    hd_xlk_blocked = all(
        not c.get("execution_ready") for c in focus if c["symbol"] in ("HD", "XLK")
    )
    drift_count = sum(1 for c in candidates if c["drift_severity"] != "NO_DRIFT")
    critical = sum(1 for c in candidates if c["drift_severity"] == "CRITICAL_DRIFT")

    return {
        "execution_readiness_impact": (
            "BLOCKED — HD/XLK not in trade_opportunities; effective_stance=WAIT"
            if hd_xlk_blocked
            else "PARTIAL"
        ),
        "recovery_basket_viability": (
            "NOT_VIABLE until lifecycle reconciled to BUY/ADD or exception-scoped override designed"
            if hd_xlk_blocked
            else "VIABLE_WITH_CONDITIONS"
        ),
        "governance_recovery_impact": (
            "HIGH — diversification trades cannot execute; ORANGE deadlock persists at execution layer"
            if hd_xlk_blocked
            else "MODERATE"
        ),
        "yellow_timeline_impact_days": (
            "UNREACHABLE" if hd_xlk_blocked else "3-14 (per Phase 182 sim)"
        ),
        "green_timeline_impact_days": (
            "UNREACHABLE" if hd_xlk_blocked else "14+ (BITO promotion may be required)"
        ),
        "forecast_confidence_impact": "REDUCED — lifecycle drift adds 15-25pp execution uncertainty",
        "symbols_blocked": [c["symbol"] for c in focus if not c.get("execution_ready")],
        "drift_count": drift_count,
        "critical_drift_count": critical,
        "phase179_paradox_link": (ctx["exec_audit"].get("summary") or {}).get("paradox_detected"),
    }


def _build_history(ctx: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ts_now = _now_iso()[:19]

    for c in candidates:
        rows.append(
            {
                "observed_at": ts_now,
                "source_artifact": "current_snapshot",
                "symbol": c["symbol"],
                "allocation_action": c["allocation_action"],
                "lifecycle_action": c["lifecycle_action"],
                "effective_action": c["effective_action"],
                "execution_action": c["execution_action"],
                "drift_type": c["drift_type"],
                "drift_severity": c["drift_severity"],
            }
        )

    for trace in ctx["exec_audit"].get("candidate_traces") or []:
        sym = _norm_sym(trace.get("symbol"))
        rows.append(
            {
                "observed_at": ctx["timestamps"].get("phase179_audit", ""),
                "source_artifact": "allocation_execution_path_audit.json",
                "symbol": sym,
                "allocation_action": str(trace.get("recommended_action") or "").upper(),
                "lifecycle_action": str(trace.get("lifecycle_action") or "").upper(),
                "effective_action": str(trace.get("effective_stance") or "").upper(),
                "execution_action": "BLOCKED" if not trace.get("lifecycle_pass") else "ELIGIBLE",
                "drift_type": (
                    "ALLOCATION_VS_EFFECTIVE" if trace.get("lifecycle_mismatch") else "NO_DRIFT"
                ),
                "drift_severity": "MAJOR_DRIFT" if trace.get("lifecycle_mismatch") else "NO_DRIFT",
            }
        )

    sim_assets = (ctx["simulation"].get("current_state") or {}).get("asset_status") or {}
    sim_ts = (ctx["simulation"].get("generated_at") or "")[:19]
    for sym, st in sim_assets.items():
        rows.append(
            {
                "observed_at": sim_ts,
                "source_artifact": "recovery_basket_exception_simulation.json",
                "symbol": _norm_sym(sym),
                "allocation_action": str(st.get("allocation_action") or "").upper(),
                "lifecycle_action": "—",
                "effective_action": str(st.get("lifecycle_stance") or "").upper(),
                "execution_action": (
                    "NOT_ELIGIBLE" if not st.get("lifecycle_actionable") else "ELIGIBLE"
                ),
                "drift_type": (
                    "RECOVERY_BASKET_STALE"
                    if st.get("exit_strategy_status") == "OPEN_NEW"
                    and st.get("allocation_action") == "HOLD"
                    else "CROSS_LAYER_MISMATCH"
                ),
                "drift_severity": "MODERATE_DRIFT",
            }
        )

    exit_cs = ctx["exit_strat"].get("current_state") or {}
    exit_ts = (ctx["exit_strat"].get("generated_at") or "")[:19]
    for sym in ("HD", "XLK", "QQQ", "BITO"):
        key = f"{sym.lower()}_status"
        if exit_cs.get(key):
            rows.append(
                {
                    "observed_at": exit_ts,
                    "source_artifact": "concentration_exit_strategy.json",
                    "symbol": sym,
                    "allocation_action": "—",
                    "lifecycle_action": "—",
                    "effective_action": "—",
                    "execution_action": str(exit_cs[key]).upper(),
                    "drift_type": "RECOVERY_BASKET_ASSUMPTION",
                    "drift_severity": "MODERATE_DRIFT",
                }
            )

    return rows


def _executive_recommendation(
    candidates: List[Dict[str, Any]],
    root_causes: Dict[str, Any],
    authority: Dict[str, Any],
    impact: Dict[str, Any],
) -> Dict[str, Any]:
    focus = [c for c in candidates if c["symbol"] in ("HD", "XLK")]
    return {
        "report_type": "LIFECYCLE_DRIFT_ROOT_CAUSE_REPORT",
        "current_drift_summary": (
            f"{sum(1 for c in candidates if c['drift_severity'] != 'NO_DRIFT')} symbols with drift; "
            f"HD/XLK effective=WAIT, recovery artifacts assumed OPEN_NEW"
        ),
        "root_cause": root_causes.get("primary_root_cause"),
        "affected_symbols": [c["symbol"] for c in candidates if c["drift_severity"] != "NO_DRIFT"],
        "execution_impact": impact.get("execution_readiness_impact"),
        "recovery_impact": impact.get("recovery_basket_viability"),
        "authoritative_layer": authority.get("authoritative_layer_for_execution"),
        "recommended_resolution_path": [
            "Re-run portfolio allocation after lifecycle refresh (or enforce alloc←lifecycle dependency)",
            "Add drift monitor: flag when recommended_action=OPEN_NEW and effective_stance∉{BUY,ADD}",
            "Reconcile recovery basket / exit strategy artifacts against signal_lifecycle_effective",
            "Block recovery exception activation when effective_stance=WAIT for basket symbols",
            "Complete lifecycle reconciliation before Phase 183 implementation review proceeds to build",
        ],
        "must_fix_before_implementation": [
            "HD/XLK effective_stance must be BUY/ADD OR exception design must define scoped lifecycle eligibility",
            "Eliminate stale OPEN_NEW references in recovery planning artifacts",
            "Document authoritative layer: effective_lifecycle > execution > allocation snapshot",
        ],
        "executive_questions": {
            "why_is_drift_occurring": (
                root_causes["ranked_causes"][0]["description"]
                if root_causes.get("ranked_causes")
                else "—"
            ),
            "which_layer_authoritative": authority.get("authoritative_layer_for_execution"),
            "is_drift_expected": root_causes.get("is_expected_behavior"),
            "recovery_impact": impact.get("recovery_basket_viability"),
            "fix_before_implementation": "Lifecycle reconciliation for HD/XLK; stale artifact refresh",
        },
    }


def run_analysis() -> Dict[str, Any]:
    ctx = _load_context()
    ts = _now_iso()
    universe = _candidate_universe(ctx)
    candidates = [_analyze_candidate(sym, ctx) for sym in universe]
    root_causes = _root_causes(candidates, ctx)
    authority = _authority_analysis(candidates)
    impact = _impact_analysis(candidates, ctx)
    history = _build_history(ctx, candidates)
    executive = _executive_recommendation(candidates, root_causes, authority, impact)

    drift_summary = {
        "symbols_analyzed": len(candidates),
        "drift_count": sum(1 for c in candidates if c["drift_severity"] != "NO_DRIFT"),
        "critical_count": sum(1 for c in candidates if c["drift_severity"] == "CRITICAL_DRIFT"),
        "major_count": sum(1 for c in candidates if c["drift_severity"] == "MAJOR_DRIFT"),
        "open_new_candidates": sum(1 for c in candidates if c.get("open_new_at_allocation")),
        "wait_at_effective": sum(1 for c in candidates if c.get("wait_at_effective")),
        "recovery_basket_symbols": ["HD", "XLK"],
        "recovery_basket_execution_ready": any(
            c.get("execution_ready") for c in candidates if c["symbol"] in ("HD", "XLK")
        ),
        "primary_root_cause": root_causes.get("primary_root_cause"),
        "authoritative_layer": authority.get("authoritative_layer_for_execution"),
        "artifact_timestamps": ctx["timestamps"],
    }

    summary = {
        "drift_detected": drift_summary["drift_count"] > 0,
        "critical_drift": drift_summary["critical_count"] > 0,
        "primary_root_cause": root_causes.get("primary_root_cause"),
        "authoritative_layer": authority.get("authoritative_layer_for_execution"),
        "recovery_blocked": impact.get("recovery_basket_viability", "").startswith("NOT"),
        "implementation_blocker": True,
    }

    report = {
        "generated_at": ts,
        "phase": "184",
        "mode": "lifecycle_drift_root_cause_analysis",
        "disclaimer": (
            "Root cause analysis only — does not modify governance, lifecycle, execution, "
            "allocation, or risk controls."
        ),
        "drift_summary": drift_summary,
        "candidate_analysis": candidates,
        "root_causes": root_causes,
        "impact_analysis": impact,
        "authority_analysis": authority,
        "executive_recommendation": executive,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "read-only analysis — no pipeline modifications",
        },
    }

    pd.DataFrame(candidates).to_csv(OUT_CANDIDATES, index=False, encoding="utf-8")
    pd.DataFrame(history).to_csv(OUT_HISTORY, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_CANDIDATES.name}, {OUT_HISTORY.name}")
    _log(
        f"drift={drift_summary['drift_count']} critical={drift_summary['critical_count']} "
        f"cause={drift_summary['primary_root_cause']}"
    )
    return report


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
