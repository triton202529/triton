# services/execute_trades.py
"""TRITON execution engine — plan from trade_opportunities.csv, optional handoff to place_live_orders."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from services.execution_drop_diagnostics import (
    finalize_artifacts,
    make_row,
    build_summary_payload,
)
from services.session_fill_pressure import write_last_execution_session_snapshot
from services.adaptive_position_sizing import (
    compute_size_factor_breakdown,
    max_order_notional_usd,
)
from services.execution_quality import entry_priority_score
from services.sector_exposure import (
    UNKNOWN_SECTOR_LABEL,
    current_sector_pct,
    evaluate_sector_cap,
    get_sector,
    load_exposure_for_planning,
    projected_sector_pct_after_buy,
    sector_exposure_pcts,
    should_block_buy_for_sector,
)
from services.lifecycle_truth import evaluate_lifecycle_gate
from services.position_sizer import join_key
from services.execution_intelligence import (
    ExecutionIntelligenceConfig,
    annotate_order as _ei_annotate_order,
    ANNOTATE_ORDER_KEYS as _EI_KEYS,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
DEFAULT_OPPS = RESULTS / "trade_opportunities.csv"
DEFAULT_ORDERS_TODAY = LIVE / "orders_today.csv"
EXEC_PLAN_JSON = RESULTS / "execution_plan.json"
EXEC_PLAN_CSV = RESULTS / "execution_plan.csv"
EXEC_LOG_CSV = RESULTS / "execution_plan_log.csv"
ORDERS_SIZED_CSV = RESULTS / "orders_sized.csv"
EDGE_SIZING_RECOMMENDATIONS_CSV = RESULTS / "edge_sizing_recommendations.csv"
CONFIG_PATH = ROOT / "config" / "execute_trades.json"


def _stance_to_opportunity_type(stance: str) -> str:
    return {"BUY": "ENTRY", "ADD": "ADD", "TRIM": "TRIM", "EXIT": "EXIT"}.get(
        str(stance or "").strip().upper(), ""
    )


def map_to_diagnostic_reason(internal: str) -> str:
    """Map internal skip/placement codes to stable diagnostic reason_code (never changes trading behavior)."""
    c = str(internal or "").strip().upper()
    m = {
        "LOW_CONFIDENCE": "BELOW_MIN_CONFIDENCE",
        "BELOW_MIN_ORDER_NOTIONAL": "BELOW_MIN_NOTIONAL",
        "NO_PRICE_AVAILABLE": "INVALID_PRICE",
        "MAX_NEW_POSITIONS_REACHED": "EXCEEDS_POSITION_LIMIT",
        "MAX_ADDS_REACHED": "EXCEEDS_POSITION_LIMIT",
        "MAX_ORDERS_REACHED": "EXCEEDS_POSITION_LIMIT",
        "MAX_GROSS_EXPOSURE": "MAX_NOTIONAL_EXCEEDED",
        "INSUFFICIENT_BUYING_POWER": "MAX_NOTIONAL_EXCEEDED",
        "ZERO_QTY_AFTER_ROUNDING": "MAX_QTY_EXCEEDED",
        "POSITION_NOT_FLAT_FOR_BUY": "POSITION_ALREADY_EXISTS",
        "DUPLICATE_SYMBOL_IN_RUN": "DUPLICATE_ORDER",
        "BUY_BLOCKED_BY_RISK": "RISK_GATE_BLOCK",
        "ADD_BLOCKED_BY_RISK": "RISK_GATE_BLOCK",
        "BUY_BLOCKED_BY_CPM": "RISK_GATE_BLOCK",
        "ADD_BLOCKED_BY_CPM": "RISK_GATE_BLOCK",
        "LIFECYCLE_GATE_BLOCK": "LIFECYCLE_GATE_BLOCK",
        "BUY_BLOCKED_BY_SECTOR": "RISK_GATE_BLOCK",
        "ADD_BLOCKED_BY_SECTOR": "RISK_GATE_BLOCK",
        "BUY_BLOCKED_BY_SECTOR_SOFT": "RISK_GATE_BLOCK",
        "BUY_BLOCKED_BY_SECTOR_HARD": "RISK_GATE_BLOCK",
        "ADD_BLOCKED_BY_SECTOR_SOFT": "RISK_GATE_BLOCK",
        "ADD_BLOCKED_BY_SECTOR_HARD": "RISK_GATE_BLOCK",
        "UNKNOWN_SECTOR_BLOCK": "RISK_GATE_BLOCK",
        "PLACEMENT_BLOCKED": "RISK_GATE_BLOCK",
        "EXECUTION_GUARD_BLOCK": "RISK_GATE_BLOCK",
        "NO_OPPORTUNITIES": "UNKNOWN",
        "UNKNOWN_DROP_REASON": "UNKNOWN",
        "NON_ACTIONABLE_STANCE": "UNKNOWN",
        "POSITION_NOT_FOUND_FOR_ADD": "UNKNOWN",
        "POSITION_NOT_FOUND_FOR_TRIM": "UNKNOWN",
        "POSITION_NOT_FOUND_FOR_EXIT": "UNKNOWN",
        "ILLEGAL_SELL_PREVENTED": "MAX_QTY_EXCEEDED",
        "EXECUTION_QUALITY_NO_CONFIRMATION": "FILTERED_BY_EXECUTION_QUALITY",
        "EXECUTION_QUALITY_NO_PRICE": "INVALID_PRICE",
        "EXECUTION_QUALITY_TOP_N_CAP": "FILTERED_BY_EXECUTION_QUALITY",
        "DELAYED_ENTRY": "FILTERED_BY_EXECUTION_QUALITY",
        "IN_FLIGHT_ALREADY_SATISFIED": "IN_FLIGHT_SATISFIED",
        "BELOW_MIN_SCORE": "FILTERED_BY_QUALITY",
        "NEGATIVE_SCORE_BLOCK": "FILTERED_BY_QUALITY",
        "SCORE_SIZED_TOO_SMALL": "FILTERED_BY_QUALITY",
        "ADD_BELOW_SCORE_THRESHOLD": "FILTERED_BY_QUALITY",
        "ADD_TOO_SMALL": "FILTERED_BY_QUALITY",
        "MAX_POSITIONS": "EXCEEDS_POSITION_LIMIT",
        "ALREADY_IN_PORTFOLIO": "POSITION_ALREADY_EXISTS",
        "ALREADY_LONG_SKIP": "POSITION_ALREADY_EXISTS",
        "ALREADY_LONG_LOW_CONFIDENCE_SKIP": "POSITION_ALREADY_EXISTS",
    }
    if c in m:
        return m[c]
    allowed = {
        "MAX_NOTIONAL_EXCEEDED",
        "MAX_QTY_EXCEEDED",
        "RISK_GATE_BLOCK",
        "BELOW_MIN_NOTIONAL",
        "BELOW_MIN_CONFIDENCE",
        "INVALID_PRICE",
        "POSITION_ALREADY_EXISTS",
        "DUPLICATE_ORDER",
        "IN_FLIGHT_ORDER_EXISTS",
        "IN_FLIGHT_SATISFIED",
        "EXCEEDS_POSITION_LIMIT",
        "UNKNOWN",
        "FILTERED_BY_QUALITY",
    }
    if c in allowed:
        return c
    return "UNKNOWN"


def _emit_execution_drop_payload(
    rows: List[Dict[str, Any]],
    mode: str,
    *,
    blocked: bool = False,
    run_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        p = build_summary_payload(
            mode=mode, rows=rows, blocked=blocked, source_hint="execute_trades"
        )
        p["rows"] = rows
        p["source"] = "execute_trades"
        if run_id:
            p["run_id"] = run_id
        if extra:
            p.update(extra)
        finalize_artifacts(p, write_log=True)
    except Exception:
        pass


def build_drop_rows_from_plan(
    plan_lines: List[PlanLine],
    planned: List[PlannedOrder],
    mode: str,
    session: str,
    orders_file: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pl in plan_lines:
        sym = pl.symbol or ""
        if pl.status == "skipped":
            internal = pl.skip_reason or "UNKNOWN_DROP_REASON"
            diag = map_to_diagnostic_reason(internal)
            if internal == "IN_FLIGHT_ALREADY_SATISFIED":
                out.append(
                    make_row(
                        run_mode=mode,
                        symbol=sym,
                        ticker=sym,
                        stance=pl.stance,
                        opportunity_type=pl.opportunity_type,
                        confidence=pl.confidence,
                        delta_pct=pl.delta_pct,
                        phase="preflight_in_flight",
                        status="satisfied_in_flight",
                        reason_code=diag,
                        reason_detail=(
                            "internal=IN_FLIGHT_ALREADY_SATISFIED; "
                            "open same-side order already covers intent (execute_trades prefilter)"
                        ),
                        source="execute_trades",
                        session=session,
                    )
                )
            else:
                out.append(
                    make_row(
                        run_mode=mode,
                        symbol=sym,
                        ticker=sym,
                        stance=pl.stance,
                        opportunity_type=pl.opportunity_type,
                        confidence=pl.confidence,
                        delta_pct=pl.delta_pct,
                        phase="planning",
                        status="dropped",
                        reason_code=diag,
                        reason_detail=f"internal={internal}",
                        source="execute_trades",
                        session=session,
                    )
                )
        elif pl.planned is not None:
            p = pl.planned
            ot = pl.opportunity_type or _stance_to_opportunity_type(pl.stance)
            out.append(
                make_row(
                    run_mode=mode,
                    symbol=p.symbol,
                    ticker=p.symbol,
                    stance=pl.stance,
                    opportunity_type=ot,
                    confidence=p.confidence,
                    delta_pct=p.delta_pct,
                    planned_qty=p.qty,
                    planned_notional=round(float(p.planned_notional), 4),
                    phase="planning",
                    status="planned",
                    reason_code="KEPT",
                    reason_detail="Planned order",
                    source="execute_trades",
                    session=session,
                )
            )
    for p in planned:
        out.append(
            make_row(
                run_mode=mode,
                symbol=p.symbol,
                ticker=p.symbol,
                stance=p.stance,
                opportunity_type=_stance_to_opportunity_type(p.stance),
                confidence=p.confidence,
                delta_pct=p.delta_pct,
                planned_qty=p.qty,
                planned_notional=round(float(p.planned_notional), 4),
                phase="prewrite",
                status="kept",
                reason_code="WRITTEN_TO_ORDERS_TODAY",
                reason_detail=f"orders_today path {orders_file}",
                source="execute_trades",
                session=session,
            )
        )
    return out


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        o = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return o if isinstance(o, dict) else None
    except Exception:
        return None


def _effective_max_positions(cfg: Dict[str, Any]) -> int:
    """Same cap source as ExecutionGuard (execute_trades.json max_positions merged in guard)."""
    try:
        mp = cfg.get("max_positions")
        if mp is None:
            mp = cfg.get("max_portfolio_positions")
        if mp is not None:
            return max(1, int(mp))
    except Exception:
        pass
    return 25


# Portfolio-state guard codes that are knowable from broker snapshot; not "system failures".
_SOFT_GUARD_PORTFOLIO_CODES = frozenset({"MAX_POSITIONS", "MAX_OPEN_ORDERS"})


def load_execute_trades_config() -> Dict[str, Any]:
    cfg = {
        "enabled": True,
        "default_mode": "paper",
        "dry_run_default": True,
        "min_confidence": 0.0,
        "max_new_positions_per_run": 10,
        "max_adds_per_run": 3,
        "trim_fraction": 0.25,
        "default_position_weight": 0.05,
        "add_position_weight": 0.02,
        "max_position_weight_hard_cap": 0.10,
        "reserve_cash_pct": 0.20,
        "min_order_notional": 50.0,
        "round_share_qty_down": True,
        "prefer_limit_orders": True,
        "limit_price_buffer_buy_bps": 35,
        "limit_price_buffer_sell_bps": 20,
        "max_orders_per_run": 15,
        "require_master_gate_for_execute": True,
        "require_guard_validation_for_each_order": True,
        "write_orders_today": True,
        "execute_via_existing_placement_flow": True,
        "max_batch_notional": 50000.0,
        "adaptive_sizing_enabled": True,
        "size_factor_min": 0.5,
        "size_factor_max": 1.5,
        "delta_pct_boost_scale": 50.0,
        "volatility_shrink_k": 0.25,
        "max_order_notional_usd": None,
        "size_factor_final_min": 0.3,
        "size_factor_final_max": 1.5,
        "volatility_impact_strength": 3.0,
        "use_quadratic_vol": True,
        "quadratic_vol_scale": 10.0,
        "vol_adjustment_floor": 0.3,
        "sector_exposure_enabled": True,
        "sector_warn_pct": 0.40,
        "sector_critical_pct": 0.60,
        "sector_block_pct": 0.40,
        "sector_caps_enabled": True,
        "sector_soft_cap_pct": 0.30,
        "sector_hard_cap_pct": 0.40,
        "sector_add_soft_cap_pct": 0.30,
        "sector_add_hard_cap_pct": 0.35,
        "allow_new_position_under_hard_cap": True,
        "allow_adds_under_soft_cap_only": True,
        "allow_unknown_sector": False,
        "positions_snapshot_path": None,
        "snapshot_hygiene_enabled": True,
        "snapshot_hygiene_max_age_minutes": 25.0,
        "snapshot_hygiene_report_guard": True,
        "diversification_ranking_enabled": True,
        "diversification_bonus_underweight_sector": 0.15,
        "diversification_bonus_new_sector": 0.10,
        "diversification_penalty_dominant_sector": 0.15,
        "diversification_penalty_near_cap": 0.20,
        "diversification_scale": 0.02,
        "override_low_score_threshold": 0.45,
        "block_negative_final_score": True,
        "enforce_min_final_score": True,
        "min_final_score_threshold": 0.65,
        "exec_score_percentile_normalize": True,
        "delayed_entry_soft_penalty_enabled": True,
        "delayed_entry_penalty_factor": 0.85,
        "add_min_final_score_threshold": 0.67,
        "add_min_qty_increase": 2,
        "capital_deploy_under_deployed_ratio": 0.05,
        "capital_engine_enabled": True,
        "capital_engine_under_deploy_trigger": 0.05,
        "capital_engine_relaxed_add_score_threshold": 0.60,
        "capital_engine_relaxed_min_final_score_threshold": 0.55,
        "capital_engine_max_borderline_trades": 1,
        "score_sizing_enabled": True,
        "score_sizing_tier_a_threshold": 0.85,
        "score_sizing_tier_b_threshold": 0.75,
        "score_sizing_tier_c_threshold": 0.65,
        "score_sizing_tier_a_multiplier": 1.00,
        "score_sizing_tier_b_multiplier": 0.75,
        "score_sizing_tier_c_multiplier": 0.50,
        "score_sizing_mode": "nonlinear",
        "score_sizing_reference_score": 0.65,
        "score_sizing_gamma": 1.5,
        "score_sizing_min_multiplier": 0.20,
        "score_sizing_max_multiplier": 1.00,
        "planner_rank_by_final_score": True,
        "entry_guard_precap_buys": True,
    }
    try:
        if CONFIG_PATH.is_file():
            u = json.loads(CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
            if isinstance(u, dict):
                cfg.update(u)
    except Exception:
        pass
    return cfg


def _planned_score_breakdown(p: PlannedOrder) -> Tuple[float, float, float]:
    """(base_score, diversification_adjustment, final_score) for diagnostics only."""
    div_adj = float(getattr(p, "diversification_adjustment", 0.0) or 0.0)
    er = getattr(p, "execution_rank_score", None)
    if er is not None:
        try:
            final = float(er)
            if not math.isnan(final):
                return final - div_adj, div_adj, final
        except Exception:
            pass
    base = entry_priority_score(float(p.confidence), float(p.delta_pct))
    return base, div_adj, base + div_adj


def _decision_reason_short(p: PlannedOrder, max_len: int = 64) -> str:
    r = str(p.rationale or "").replace("\n", " ").strip()
    if len(r) > max_len:
        return r[: max_len - 3] + "..."
    return r or "-"


def _print_execution_diversification_table(planned: List[PlannedOrder]) -> None:
    """Pre-placement [EXECUTION] snapshot from execute_trades planned orders (diagnostic)."""
    if not planned:
        return
    print("[EXECUTION] planned_orders (execute_trades diversification snapshot)")
    print("ticker | stance | action | qty | base | adj | final | decision_reason")
    for p in planned:
        b, a, f = _planned_score_breakdown(p)
        dr = _decision_reason_short(p)
        print(f"{p.symbol} | {p.stance} | {p.side} | {p.qty} | {b:.4f} | {a:+.4f} | {f:.4f} | {dr}")


def _print_placement_override_diagnostics(
    planned: List[PlannedOrder],
    cfg: Dict[str, Any],
    placement_rc: int,
) -> None:
    """After placement subprocess: negative / low-score execution diagnostics (no side effects)."""
    if placement_rc != 0 or not planned:
        return
    thr = float(cfg.get("override_low_score_threshold", 0.45) or 0.45)
    scored: List[Tuple[PlannedOrder, float, float, float]] = []
    for p in planned:
        b, a, f = _planned_score_breakdown(p)
        scored.append((p, b, a, f))

    neg = [(p, f) for p, _b, _a, f in scored if f < 0]
    for p, f in neg:
        print(
            f"[OVERRIDE_WARNING] symbol={p.symbol} final_score={f:.4f} "
            f"reason=executed_despite_negative_score source=planner/lifecycle"
        )

    low = [(p, f) for p, _b, _a, f in scored if f < thr]
    neg_n = len(neg)
    low_n = len(low)
    worst = sorted(scored, key=lambda x: x[3])[:8]
    top_overrides = [f"{p.symbol}(final={f:.4f})" for p, _b, _a, f in worst]

    print("[OVERRIDE_SUMMARY]")
    print(f"executed_negative_scores={neg_n}")
    print(f"executed_low_scores(final_score<{thr:.4f})={low_n}")
    print(f"top_overrides={top_overrides}")


def _planned_div_fields(row: pd.Series) -> Tuple[float, Optional[float]]:
    """Read diversification columns set by _maybe_reorder_df_for_diversification."""
    if "_div_final" not in row.index:
        return 0.0, None
    fin = row.get("_div_final")
    try:
        if fin is None or pd.isna(fin):
            return 0.0, None
        fr = float(fin)
        if math.isnan(fr):
            return 0.0, None
    except Exception:
        return 0.0, None
    return _safe_float(row.get("_div_adj"), 0.0), fr


def _score_sizing_mode_normalized(cfg: Dict[str, Any]) -> str:
    """Default nonlinear when missing; only 'tier' selects tiered curve."""
    m = str(cfg.get("score_sizing_mode") or "nonlinear").strip().lower()
    if m == "tier":
        return "tier"
    return "nonlinear"


def _nonlinear_score_sizing_multipliers(
    fs: float, cfg: Dict[str, Any]
) -> Tuple[float, float, float, float]:
    """Gamma-scaled: raw = max(base_ratio,0)^gamma, then clamp to [min, max]. Returns
    (base_ratio, gamma, raw_multiplier_pre_clamp, multiplier).
    """
    ref = float(cfg.get("score_sizing_reference_score", 0.65) or 0.65)
    if ref <= 0:
        ref = 0.65
    gamma = float(cfg.get("score_sizing_gamma", 1.5) or 1.5)
    min_m = float(cfg.get("score_sizing_min_multiplier", 0.20) or 0.20)
    max_m = float(cfg.get("score_sizing_max_multiplier", 1.0) or 1.0)
    if min_m > max_m:
        min_m, max_m = max_m, min_m
    base_ratio = fs / ref
    raw_unc = max(base_ratio, 0.0) ** gamma
    clamped = max(min_m, min(max_m, raw_unc))
    return base_ratio, gamma, raw_unc, clamped


def _score_sizing_multiplier(fs: float, cfg: Dict[str, Any]) -> Tuple[float, str]:
    """Tier label A/B/C for logging; fs is final_score (same as quality gating)."""
    ta = float(cfg.get("score_sizing_tier_a_threshold", 0.85) or 0.85)
    tb = float(cfg.get("score_sizing_tier_b_threshold", 0.75) or 0.75)
    tc = float(cfg.get("score_sizing_tier_c_threshold", 0.65) or 0.65)
    ma = float(cfg.get("score_sizing_tier_a_multiplier", 1.0) or 1.0)
    mb = float(cfg.get("score_sizing_tier_b_multiplier", 0.75) or 0.75)
    mc = float(cfg.get("score_sizing_tier_c_multiplier", 0.5) or 0.5)
    if fs >= ta:
        return ma, "A"
    if fs >= tb:
        return mb, "B"
    if fs >= tc:
        return mc, "C"
    return mc, "C"


def _row_final_quality_score(row: pd.Series) -> float:
    """
    Authoritative execution score for this opportunity row — single pipeline for:
    quality gates, score sizing, conviction ranking, and capital-engine borderline logic.

    Uses diversification-adjusted `_div_final` when present (set by sector diversification pass);
    then post-attach / post-adaptation `_exec_final_score` when set (so runtime adaptation
    actually affects gating and sizing); otherwise confidence × delta_pct via
    entry_priority_score (same base as pre-div BUY/ADD scoring).
    """
    if "_div_final" in row.index:
        fin = row.get("_div_final")
        try:
            if fin is not None and not pd.isna(fin):
                v = float(fin)
                if not math.isnan(v):
                    return v
        except Exception:
            pass
    if "_exec_final_score" in row.index:
        fin = row.get("_exec_final_score")
        try:
            if fin is not None and not pd.isna(fin):
                v = float(fin)
                if not math.isnan(v):
                    return v
        except Exception:
            pass
    if "final_score" in row.index:
        fin = row.get("final_score")
        try:
            if fin is not None and not pd.isna(fin):
                v = float(fin)
                if not math.isnan(v):
                    return v
        except Exception:
            pass
    conf = _safe_float(row.get("confidence"), 0.0)
    d_pct = _safe_float(row.get("delta_pct"), 0.0)
    return entry_priority_score(conf, d_pct)


def _row_execution_final_score(row: pd.Series) -> float:
    """Explicit alias for the same authoritative score (readability at call sites)."""
    return _row_final_quality_score(row)


def _attach_exec_final_score_column(df: pd.DataFrame) -> pd.DataFrame:
    """STAGE A (part): one authoritative final score per row — used for rank, quality, sizing, ADD, rescue."""
    if df is None or df.empty:
        return df
    out = df.copy()
    xs = [_row_execution_final_score(out.iloc[i]) for i in range(len(out))]
    out["_exec_final_score"] = xs
    out["final_score"] = xs
    return out


def _normalize_exec_final_score_percentile_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map _exec_final_score to [0, 1] using min-rank / (n-1). Preserves total preorder
    (higher raw score => higher or equal normalized score); tied raw scores share the
    same normalized value. This aligns absolute thresholds (e.g. min_final_score 0.55–0.65)
    with typical opportunity score ranges (~0.3–0.7 raw) without changing sort order.
    """
    if df is None or df.empty or "_exec_final_score" not in df.columns:
        return df
    out = df.copy()
    raw = pd.to_numeric(out["_exec_final_score"], errors="coerce")
    if bool(raw.isna().any()):
        fill = float(raw.min()) if raw.notna().any() else 0.0
        raw = raw.fillna(fill)
    n = int(len(out))
    print(
        f"[SCORE_DISTRIBUTION] n={n} min={float(raw.min()):.6f} max={float(raw.max()):.6f} "
        f"mean={float(raw.mean()):.6f} median={float(raw.median()):.6f}"
    )
    if n < 2:
        x = 1.0
        out["_exec_final_score"] = x
        out["final_score"] = x
        if "_div_final" in out.columns:
            out["_div_final"] = x
        print(
            "[SCORE_DISTRIBUTION] normalized=percentile_rank n=1 min=1.0 max=1.0 mean=1.0 median=1.0"
        )
        return out
    rmin = raw.rank(method="min", ascending=True)
    norm = (rmin - 1.0) / float(n - 1)
    out["_exec_final_score"] = norm
    out["final_score"] = norm
    if "_div_final" in out.columns:
        out["_div_final"] = norm
    npv = pd.to_numeric(out["_exec_final_score"], errors="coerce")
    print(
        f"[SCORE_DISTRIBUTION] normalized=percentile_rank min={float(npv.min()):.6f} max={float(npv.max()):.6f} "
        f"mean={float(npv.mean()):.6f} median={float(npv.median()):.6f}"
    )
    return out


def _rank_working_df_by_final_score(df: pd.DataFrame) -> pd.DataFrame:
    """STAGE B: final_score desc; then adaptation_delta desc (penalties rank after ties); then ticker."""
    if df is None or df.empty or "_exec_final_score" not in df.columns:
        return df
    dfr = df.copy()
    if "adaptation_delta_applied" in dfr.columns:
        dfr["_adapt_rank_tie"] = pd.to_numeric(
            dfr["adaptation_delta_applied"], errors="coerce"
        ).fillna(0.0)
        out = dfr.sort_values(
            by=["_exec_final_score", "_adapt_rank_tie", "ticker"],
            ascending=[False, False, True],
            na_position="last",
        )
        out = out.drop(columns=["_adapt_rank_tie"], errors="ignore")
    else:
        out = dfr.sort_values(
            by=["_exec_final_score", "ticker"],
            ascending=[False, True],
            na_position="last",
        )
    return out.reset_index(drop=True)


def _inflight_summary_fields(skip_reasons: Dict[str, int]) -> Dict[str, int]:
    """
    Visibility for in-flight suppression (current prefilter uses one code for open+intent overlap).
    blocked_open_orders / blocked_already_satisfied both count IN_FLIGHT_ALREADY_SATISFIED today.
    """
    sr = skip_reasons or {}
    satisfied = int(sr.get("IN_FLIGHT_ALREADY_SATISFIED", 0) or 0)
    other = 0
    for k, v in sr.items():
        ks = str(k)
        if ks.startswith("IN_FLIGHT") and ks != "IN_FLIGHT_ALREADY_SATISFIED":
            other += int(v)
    return {
        "blocked_open_orders": satisfied,
        "blocked_already_satisfied": satisfied,
        "other_inflight_blocks": other,
    }


def _planner_transparency_counts(skip_reasons: Dict[str, int]) -> Dict[str, int]:
    """
    Desk counts aligned to planner stages (each skip reason counted once).
    filtered_* = rows skipped for that reason category in this run.
    """
    sr = skip_reasons or {}

    def n(k: str) -> int:
        return int(sr.get(k, 0) or 0)

    fq = n("NEGATIVE_SCORE_BLOCK") + n("BELOW_MIN_SCORE")
    fa = n("ADD_BELOW_SCORE_THRESHOLD") + n("ADD_TOO_SMALL")
    fsector = 0
    for k, v in sr.items():
        kk = str(k)
        if "SECTOR" in kk or kk == "UNKNOWN_SECTOR_BLOCK":
            fsector += int(v)
    finflight = n("IN_FLIGHT_ALREADY_SATISFIED")
    fbp = n("INSUFFICIENT_BUYING_POWER") + n("MAX_GROSS_EXPOSURE")
    fmin = n("BELOW_MIN_ORDER_NOTIONAL")
    return {
        "filtered_quality": fq,
        "filtered_add": fa,
        "filtered_sector": fsector,
        "filtered_inflight": finflight,
        "filtered_bp": fbp,
        "filtered_min_notional": fmin,
    }


def _categorize_planner_skips(skip_reasons: Dict[str, int]) -> Dict[str, int]:
    """Roll up skip_reason codes into desk-friendly buckets (counts may overlap none — each code once)."""
    cats = {
        "quality": 0,
        "add_rules": 0,
        "sector": 0,
        "buying_power_notional": 0,
        "score_sizing": 0,
        "position_mismatch": 0,
        "limits": 0,
        "risk_cpm": 0,
        "in_flight": 0,
        "execution_quality": 0,
        "discipline": 0,
        "other": 0,
    }
    quality_codes = frozenset({"NEGATIVE_SCORE_BLOCK", "BELOW_MIN_SCORE"})
    add_codes = frozenset({"ADD_BELOW_SCORE_THRESHOLD", "ADD_TOO_SMALL"})
    bp_codes = frozenset({"INSUFFICIENT_BUYING_POWER", "BELOW_MIN_ORDER_NOTIONAL"})
    sz_codes = frozenset({"SCORE_SIZED_TOO_SMALL"})
    pos_codes = frozenset(
        {
            "POSITION_NOT_FLAT_FOR_BUY",
            "ALREADY_IN_PORTFOLIO",
            "ALREADY_LONG_SKIP",
            "ALREADY_LONG_LOW_CONFIDENCE_SKIP",
            "POSITION_NOT_FOUND_FOR_ADD",
            "POSITION_NOT_FOUND_FOR_TRIM",
            "POSITION_NOT_FOUND_FOR_EXIT",
        }
    )
    lim_codes = frozenset(
        {
            "MAX_ORDERS_REACHED",
            "MAX_ADDS_REACHED",
            "MAX_NEW_POSITIONS_REACHED",
            "MAX_GROSS_EXPOSURE",
            "DUPLICATE_SYMBOL_IN_RUN",
            "MAX_POSITIONS",
        }
    )
    risk_codes = frozenset(
        {
            "BUY_BLOCKED_BY_RISK",
            "ADD_BLOCKED_BY_RISK",
            "BUY_BLOCKED_BY_CPM",
            "ADD_BLOCKED_BY_CPM",
        }
    )
    for code, n in (skip_reasons or {}).items():
        v = int(n)
        c = str(code or "")
        if c in quality_codes:
            cats["quality"] += v
        elif c in add_codes:
            cats["add_rules"] += v
        elif "SECTOR" in c or c == "UNKNOWN_SECTOR_BLOCK":
            cats["sector"] += v
        elif c in bp_codes:
            cats["buying_power_notional"] += v
        elif c in sz_codes:
            cats["score_sizing"] += v
        elif c in pos_codes:
            cats["position_mismatch"] += v
        elif c in lim_codes:
            cats["limits"] += v
        elif c in risk_codes:
            cats["risk_cpm"] += v
        elif c == "IN_FLIGHT_ALREADY_SATISFIED":
            cats["in_flight"] += v
        elif c.startswith("EXECUTION_QUALITY"):
            cats["execution_quality"] += v
        elif c == "ORDER_DISCIPLINE":
            cats["discipline"] += v
        else:
            cats["other"] += v
    return cats


def _top_skip_reasons(skip_reasons: Dict[str, int], n: int = 8) -> List[str]:
    items = sorted((skip_reasons or {}).items(), key=lambda x: -int(x[1]))
    return [f"{k}={v}" for k, v in items[: max(0, n)]]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper()


@dataclass
class PositionSnapshot:
    symbol: str
    qty: float
    market_value: Optional[float] = None


@dataclass
class AccountSnapshot:
    equity: float = 0.0
    buying_power: float = 0.0
    cash: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpportunityRow:
    ticker: str
    effective_stance: str
    effective_position_state: str
    opportunity_type: str
    confidence: float
    delta_pct: float
    rationale: str
    healed: bool
    exploration_flag: bool
    row_index: int


@dataclass
class PlannedOrder:
    symbol: str
    side: str
    qty: int
    order_type: str
    time_in_force: str
    limit_price: Optional[float]
    stance: str
    rationale: str
    confidence: float
    delta_pct: float
    planned_notional: float
    current_position_qty: float
    mode: str
    generated_at: str
    source: str = "execute_trades"
    exploration_flag: bool = False
    discipline_allowed: bool = True
    discipline_reason: str = ""
    diversification_adjustment: float = 0.0
    execution_rank_score: Optional[float] = None
    decision_tag: str = ""  # e.g. add_to_position; handoff to placement (planner = source of truth)


@dataclass
class PlanLine:
    symbol: str
    action: str
    status: str
    skip_reason: str
    stance: str
    planned: Optional[PlannedOrder] = None
    opportunity_type: str = ""
    confidence: float = 0.0
    delta_pct: float = 0.0
    decision_tag: str = ""  # e.g. add_to_position (planned); empty for other rows


@dataclass
class ExecutionPlanSummary:
    timestamp: str
    mode: str
    dry_run: bool
    opportunities_seen: int
    orders_planned: int
    orders_executed: int
    orders_skipped: int
    skip_reasons: Dict[str, int]
    blocked: bool
    block_reasons: List[str]
    source_file: str
    orders_file: str
    warnings: List[str] = field(default_factory=list)


def load_trade_opportunities(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "ticker" not in df.columns:
        return pd.DataFrame()
    return df


def _opp_to_stance(row: pd.Series) -> Optional[str]:
    ot = str(row.get("opportunity_type") or "").strip().upper()
    mp = {
        "ENTRY": "BUY",
        "ADD": "ADD",
        "TRIM": "TRIM",
        "EXIT": "EXIT",
    }
    if ot in mp:
        return mp[ot]
    st = str(row.get("effective_stance") or "").strip().upper()
    if st in ("BUY", "ADD", "TRIM", "EXIT"):
        return st
    return None


def _batch_buy_add_confidence_median(dfw: pd.DataFrame) -> Optional[float]:
    """Median confidence across rows with ENTRY/ADD (or effective BUY/ADD) in this run — for long-BUY add gate."""
    if dfw is None or dfw.empty:
        return None
    xs: List[float] = []
    for _, r in dfw.iterrows():
        st = _opp_to_stance(r)
        if st in ("BUY", "ADD"):
            xs.append(_safe_float(r.get("confidence"), 0.0))
    if not xs:
        return None
    return float(statistics.median(xs))


def load_broker_state(
    mode: str,
) -> Tuple[Dict[str, PositionSnapshot], AccountSnapshot, Optional[Any]]:
    positions: Dict[str, PositionSnapshot] = {}
    acct = AccountSnapshot()
    broker = None
    try:
        from services.broker_alpaca import AlpacaBroker

        broker = AlpacaBroker(mode=mode)
        raw_pos = broker.get_positions() or []
        for p in raw_pos:
            sym = _norm_sym(p.get("symbol"))
            if not sym:
                continue
            q = _safe_float(p.get("qty") or p.get("quantity"), 0.0)
            mv = (
                _safe_float(p.get("market_value"), None)
                if p.get("market_value") is not None
                else None
            )
            positions[sym] = PositionSnapshot(symbol=sym, qty=q, market_value=mv)
        a = broker.get_account() or {}
        acct.equity = _safe_float(a.get("equity") or a.get("portfolio_value"), 0.0)
        acct.buying_power = _safe_float(a.get("buying_power"), 0.0)
        acct.cash = _safe_float(a.get("cash"), 0.0)
        acct.raw = dict(a)
    except Exception:
        pass
    return positions, acct, broker


def _risk_cpm_flags() -> Tuple[bool, bool, float, float, float]:
    """allow_new_orders, allow_new_trades, exposure_mult, max_pos_w, max_gross."""
    allow_new = True
    allow_trades = True
    exp_mult = 1.0
    max_pos_w = 0.10
    max_gross = 1.0

    rj = _load_json(RESULTS / "adaptive_risk_state.json")
    if isinstance(rj, dict):
        ctrl = rj.get("controls") if isinstance(rj.get("controls"), dict) else {}
        if "risk_on" in ctrl:
            allow_new = allow_new and bool(ctrl.get("risk_on", True))
        if "allow_new_orders" in ctrl:
            allow_new = allow_new and bool(ctrl.get("allow_new_orders", True))
        mg = _safe_float(ctrl.get("max_gross_exposure"), 1.0)
        if mg > 0:
            max_gross = min(max_gross, mg)
        mw = _safe_float(ctrl.get("max_position_weight"), 0.10)
        if mw > 0:
            max_pos_w = min(max_pos_w, mw)

    for cpm_path in (
        RESULTS / "capital_preservation_mode.json",
        RESULTS / "capital_preservation_state.json",
    ):
        cj = _load_json(cpm_path)
        if isinstance(cj, dict):
            if "allow_new_trades" in cj:
                allow_trades = allow_trades and bool(cj.get("allow_new_trades", True))
            em = _safe_float(cj.get("exposure_multiplier"), 1.0)
            if em >= 0:
                exp_mult = min(exp_mult, em) if exp_mult > 0 else em
            break

    return allow_new, allow_trades, exp_mult, max_pos_w, max_gross


def _ref_price(broker: Any, symbol: str, side: str) -> Optional[float]:
    if broker is None:
        return None
    try:
        px = broker.get_latest_price(symbol)
        if px is not None and px > 0:
            return float(px)
    except Exception:
        pass
    try:
        from services.place_live_orders import get_ref_price

        return get_ref_price(broker, symbol, side)
    except Exception:
        return None


def _limit_from_ref(ref: float, side: str, buy_bps: float, sell_bps: float) -> float:
    if side == "buy":
        return ref * (1.0 + buy_bps / 10000.0)
    return ref * (1.0 - sell_bps / 10000.0)


def _round_price(p: float) -> float:
    if p >= 1.0:
        return round(p, 2)
    return round(p, 4)


def _estimate_buy_add_notional_for_ranking(
    sym: str,
    row: pd.Series,
    stance: str,
    *,
    positions: Dict[str, PositionSnapshot],
    broker: Any,
    cfg: Dict[str, Any],
    equity: float,
    spendable_bp: float,
    cap_pos_w: float,
    exp_mult: float,
    cap_gross: float,
    dw: float,
    aw: float,
    min_notional: float,
    buy_bps: float,
    sell_bps: float,
    max_n_order: float,
) -> Optional[float]:
    """Best-effort notional for diversification projection; mirrors plan sizing for BUY/ADD."""
    st = str(stance or "").strip().upper()
    if st not in ("BUY", "ADD"):
        return None
    pos_qty = positions[sym].qty if sym in positions else 0.0
    is_long_broker = pos_qty > 1e-9
    pos_state = str(row.get("effective_position_state") or "").strip().upper()
    is_long_csv = pos_state == "LONG"
    conf = _safe_float(row.get("confidence"), 0.0)
    d_pct = _safe_float(row.get("delta_pct"), 0.0)
    if st == "ADD" and not is_long_broker:
        return None
    # CSV-long without broker: cannot project an add; broker-long BUY rows use add-sized notional (no early None).
    if st == "BUY" and (is_long_broker or is_long_csv) and not is_long_broker:
        return None
    ref = _ref_price(broker, sym, "buy")
    if ref is None or ref <= 0:
        return None
    bd = compute_size_factor_breakdown(conf, d_pct, row, cfg=cfg)
    sf = float(bd["size_factor_final"])
    # New entries use dw; ADD and BUY when already long use aw (same as plan-time ADD) — do not return None.
    _long = is_long_broker or is_long_csv
    use_add_sizing = st == "ADD" or (st == "BUY" and _long)
    if use_add_sizing:
        w = min(aw * sf, cap_pos_w) * exp_mult
    else:
        w = min(dw * sf, cap_pos_w) * exp_mult
    target_notional = equity * w
    target_notional = min(target_notional, max_n_order)
    qty = int(math.floor(target_notional / ref))
    if bool(cfg.get("round_share_qty_down", True)) and st == "BUY" and not use_add_sizing:
        qty = max(0, qty)
    if qty < 1:
        return None
    lim = _limit_from_ref(ref, "buy", buy_bps, sell_bps)
    lim = _round_price(lim)
    notional = qty * lim
    if notional < min_notional or notional > spendable_bp + 1e-6:
        return None
    return float(notional)


def _compute_diversification_adjustment(
    symbol: str,
    stance: str,
    sector_exp: Dict[str, Any],
    cfg: Dict[str, Any],
    estimated_notional: Optional[float],
) -> Tuple[float, List[str]]:
    """
    Ranking-only adjustment (not a gate). Unknown sector -> neutral (0).
    Missing notional -> skip near_cap; still use new/existing exposure for other terms.
    """
    reasons: List[str] = []
    try:
        b_under = float(cfg.get("diversification_bonus_underweight_sector", 0.15))
        b_new = float(cfg.get("diversification_bonus_new_sector", 0.10))
        p_dom = float(cfg.get("diversification_penalty_dominant_sector", 0.15))
        p_near = float(cfg.get("diversification_penalty_near_cap", 0.20))
    except Exception:
        b_under, b_new, p_dom, p_near = 0.15, 0.10, 0.15, 0.20

    sector = get_sector(symbol)
    if sector == UNKNOWN_SECTOR_LABEL:
        return 0.0, []

    total_value = float(sector_exp.get("total_value") or 0.0)
    if total_value <= 1e-9:
        return 0.0, []

    sv0 = sector_exp.get("sector_values") or {}
    if not isinstance(sv0, dict):
        sv0 = {}
    sv: Dict[str, float] = {str(k): float(v) for k, v in sv0.items()}
    base_sec_val = float(sv.get(sector, 0.0))

    cur = current_sector_pct(sector, sector_exp, {}, 0.0)
    adj = 0.0

    pcts: List[float] = []
    for _k, v in sv.items():
        if float(v) > 1e-9:
            pcts.append(float(v) / total_value)
    median_pct = sorted(pcts)[len(pcts) // 2] if pcts else 0.0

    if base_sec_val <= 1e-9:
        adj += b_new
        reasons.append("new_sector")
    elif pcts and cur < median_pct * 0.75:
        adj += b_under
        reasons.append("underweight_sector")

    if sv and sector not in ("Diversified",):
        dominant_sec = max(sv.keys(), key=lambda k: float(sv.get(k, 0.0)))
        if str(sector) == str(dominant_sec) and cur >= 0.15:
            adj -= p_dom
            reasons.append("dominant_sector")

    notional = float(estimated_notional or 0.0)
    if notional > 1e-9:
        proj = projected_sector_pct_after_buy(
            sector,
            notional,
            total_value=total_value,
            sector_values=sv,
            pending_sector_add={},
            pending_total_add=0.0,
        )
        st = str(stance or "").strip().upper()
        soft_buy = float(cfg.get("sector_soft_cap_pct", 0.30))
        soft_add = float(cfg.get("sector_add_soft_cap_pct", 0.30))
        soft = soft_buy if st == "BUY" else soft_add
        if proj >= soft * 0.90:
            adj -= p_near
            reasons.append("near_cap")
    return adj, reasons


def _diversification_diagnostics_stats(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Min/median/max and dominance note; dominance uses median(|effective_adj|) vs median(base)."""
    if not scored:
        return {}
    bases = [float(x["base"]) for x in scored]
    raw_adjs = [float(x["raw_adj"]) for x in scored]
    eff_adjs = [float(x["eff_adj"]) for x in scored]
    finals = [float(x["final"]) for x in scored]
    abs_eff = [abs(a) for a in eff_adjs]
    med_base = statistics.median(bases)
    med_abs_eff = statistics.median(abs_eff)
    note = (
        "adjustment_dominates_base_signal"
        if med_abs_eff > med_base
        else "base_signal_dominates_or_balanced"
    )
    return {
        "base_score_min": min(bases),
        "base_score_median": statistics.median(bases),
        "base_score_max": max(bases),
        "raw_adj_min": min(raw_adjs),
        "raw_adj_max": max(raw_adjs),
        "eff_adj_min": min(eff_adjs),
        "eff_adj_max": max(eff_adjs),
        "final_score_min": min(finals),
        "final_score_median": statistics.median(finals),
        "final_score_max": max(finals),
        "note": note,
    }


def _maybe_reorder_df_for_diversification(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    positions: Dict[str, PositionSnapshot],
    _acct: AccountSnapshot,
    broker: Any,
    sector_exp: Dict[str, Any],
    sector_enforce: bool,
    *,
    equity: float,
    spendable_bp: float,
    cap_pos_w: float,
    exp_mult: float,
    cap_gross: float,
    dw: float,
    aw: float,
    min_notional: float,
    buy_bps: float,
    sell_bps: float,
    max_n_order: float,
    verbose: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    empty: Dict[str, Any] = {
        "enabled": False,
        "adjusted": 0,
        "top_positive": [],
        "top_negative": [],
    }
    if not bool(cfg.get("diversification_ranking_enabled", True)):
        return df, {**empty, "enabled": False}

    tv = float(sector_exp.get("total_value") or 0.0)
    if not sector_enforce or tv <= 1e-9:
        return df, {**empty, "enabled": True, "skipped": "no_sector_baseline"}

    work = df.reset_index(drop=True)
    rows_list = list(work.iterrows())
    if not rows_list:
        return df, {**empty, "enabled": True, "adjusted": 0}

    ba_positions: List[int] = []
    for pos, (_idx, row) in enumerate(rows_list):
        st = _opp_to_stance(row)
        if st in ("BUY", "ADD"):
            ba_positions.append(pos)

    if not ba_positions:
        return df, {**empty, "enabled": True, "adjusted": 0}

    scored: List[Dict[str, Any]] = []
    for pos in ba_positions:
        _idx, row = rows_list[pos]
        sym = _norm_sym(row.get("ticker"))
        stance = _opp_to_stance(row)
        if not sym or stance not in ("BUY", "ADD"):
            continue
        conf = _safe_float(row.get("confidence"), 0.0)
        d_pct = _safe_float(row.get("delta_pct"), 0.0)
        base = entry_priority_score(conf, d_pct)
        est = _estimate_buy_add_notional_for_ranking(
            sym,
            row,
            stance,
            positions=positions,
            broker=broker,
            cfg=cfg,
            equity=equity,
            spendable_bp=spendable_bp,
            cap_pos_w=cap_pos_w,
            exp_mult=exp_mult,
            cap_gross=cap_gross,
            dw=dw,
            aw=aw,
            min_notional=min_notional,
            buy_bps=buy_bps,
            sell_bps=sell_bps,
            max_n_order=max_n_order,
        )
        raw_adj, reasons = _compute_diversification_adjustment(sym, stance, sector_exp, cfg, est)
        raw_adj = float(raw_adj)
        scale = float(cfg.get("diversification_scale", 0.02) or 0.02)
        eff_adj = raw_adj * scale
        final = base + eff_adj
        sec_lab = get_sector(sym)
        scored.append(
            {
                "pos": pos,
                "symbol": sym,
                "sector": sec_lab,
                "confidence": conf,
                "delta_pct": d_pct,
                "base": base,
                "raw_adj": raw_adj,
                "eff_adj": eff_adj,
                "final": final,
                "reasons": reasons,
            }
        )
        if verbose:
            rs = ",".join(reasons) if reasons else "neutral"
            print(
                f"[DIVERSIFICATION_SCORE] symbol={sym} sector={sec_lab} "
                f"conf={conf:.4f} delta_pct={d_pct:.4f} "
                f"base_score={base:.4f} raw_adj={raw_adj:+.4f} eff_adj={eff_adj:+.4f} final_score={final:.4f} "
                f"reason={rs}"
            )

    if not scored:
        return df, {**empty, "enabled": True, "adjusted": 0}

    order = sorted(range(len(scored)), key=lambda i: (-scored[i]["final"], scored[i]["pos"]))
    out = work.copy()
    for c in ("_div_base", "_div_adj", "_div_final", "_div_raw_adj"):
        if c not in out.columns:
            out[c] = float("nan")

    for slot, dest_pos in enumerate(ba_positions):
        src = scored[order[slot]]
        src_pos = int(src["pos"])
        row = rows_list[src_pos][1].copy()
        row["_div_base"] = src["base"]
        row["_div_adj"] = src["eff_adj"]
        row["_div_raw_adj"] = src["raw_adj"]
        row["_div_final"] = src["final"]
        aligned = row.reindex(out.columns)
        out.iloc[dest_pos] = aligned

    by_raw = sorted(scored, key=lambda x: -x["raw_adj"])
    top_pos = [
        f"{x['symbol']}(raw_adj={x['raw_adj']:+.2f})" for x in by_raw[:5] if x["raw_adj"] > 1e-9
    ]
    top_neg = [
        f"{x['symbol']}(raw_adj={x['raw_adj']:+.2f})"
        for x in sorted(scored, key=lambda x: x["raw_adj"])[:5]
        if x["raw_adj"] < -1e-9
    ]

    summary = {
        "enabled": True,
        "adjusted": len(ba_positions),
        "top_positive": top_pos,
        "top_negative": top_neg,
        "diagnostics": _diversification_diagnostics_stats(scored),
    }
    return out, summary


def build_execution_plan(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    positions: Dict[str, PositionSnapshot],
    acct: AccountSnapshot,
    broker: Any,
    mode: str,
    verbose: bool = False,
) -> Tuple[
    List[PlannedOrder],
    List[PlanLine],
    Dict[str, int],
    List[str],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, int],
    Dict[str, Any],
    List[str],
    Dict[str, Any],
    Dict[str, Any],
]:
    # --- Stage 1: candidate intake (opportunities dataframe from caller) ---
    warnings: List[str] = []
    allow_new_risk, allow_trades_cpm, exp_mult, cap_pos_w, cap_gross = _risk_cpm_flags()
    cap_pos_w = min(cap_pos_w, float(cfg.get("max_position_weight_hard_cap", 0.10)))

    pending_sector_add: Dict[str, float] = defaultdict(float)
    pending_total_add = 0.0
    sector_exp: Dict[str, Any] = {}
    sector_cap_reason_counts: Dict[str, int] = {}
    sector_summary: Dict[str, Any] = {
        "enabled": False,
        "caps_mode": "off",
        "blocked_total": 0,
        "reason_counts": sector_cap_reason_counts,
        "sector_exposure_before": {},
        "sector_exposure_after_planned": {},
        "zero_portfolio_warned": False,
    }
    caps_on = bool(cfg.get("sector_caps_enabled", False))
    legacy_on = bool(cfg.get("sector_exposure_enabled", True)) and not caps_on
    sector_enforce = False
    if caps_on or legacy_on:
        sector_summary["enabled"] = True
        sector_summary["caps_mode"] = "new" if caps_on else "legacy"
        try:
            psp = cfg.get("positions_snapshot_path")
            snap_path = RESULTS / "positions_snapshot.csv"
            if psp:
                snap_path = Path(str(psp))
            sector_exp = load_exposure_for_planning(
                snap_path,
                warn_pct=float(cfg.get("sector_warn_pct", 0.40)),
                critical_pct=float(cfg.get("sector_critical_pct", 0.60)),
            )
            tv = float(sector_exp.get("total_value") or 0.0)
            if tv <= 1e-9:
                wmsg = "[SECTOR_CAP] portfolio total value is zero/missing — sector concentration caps skipped (safe mode)"
                warnings.append(wmsg)
                sector_summary["zero_portfolio_warned"] = True
                print(wmsg)
            else:
                sector_enforce = True
            for rline in sector_exp.get("risk_lines") or []:
                print(rline)
        except Exception:
            sector_exp = {}

    equity = acct.equity if acct.equity > 0 else max(acct.buying_power, 1.0)
    reserve = float(cfg.get("reserve_cash_pct", 0.2))
    spendable_bp = (
        acct.buying_power * (1.0 - reserve) if acct.buying_power > 0 else equity * (1.0 - reserve)
    )

    min_conf = float(cfg.get("min_confidence", 0.0))
    max_orders = int(cfg.get("max_orders_per_run", 5))
    max_new = int(cfg.get("max_new_positions_per_run", 3))
    max_adds = int(cfg.get("max_adds_per_run", 3))
    trim_frac = float(cfg.get("trim_fraction", 0.25))
    dw = float(cfg.get("default_position_weight", 0.05))
    aw = float(cfg.get("add_position_weight", 0.02))
    min_notional = float(cfg.get("min_order_notional", 50.0))
    buy_bps = float(cfg.get("limit_price_buffer_buy_bps", 35))
    sell_bps = float(cfg.get("limit_price_buffer_sell_bps", 20))
    max_n_order = max_order_notional_usd(cfg)

    # -------------------------------------------------------------------------
    # Planner pipeline (stages — see _run_once for C–I per-row processing)
    #  A  candidate intake: opportunities dataframe from caller
    #  A' sector/diversification pre-order on BUY/ADD (when sector baseline exists)
    #  B  conviction ranking: final_score desc before caps / max_orders bite (see DEFAULT max_* — low by design for competition)
    #  C  quality filter: NEGATIVE_SCORE_BLOCK / BELOW_MIN_SCORE (+ rescue pass)
    #  D  capital-engine borderline rescue: second _run_once with relaxed thresholds
    #  E  base quantity: existing sizing formulas (unchanged)
    #  F  nonlinear score sizing: _apply_score_sizing_qty
    #  G  ADD overlay: ADD_BELOW_SCORE_THRESHOLD, ADD_TOO_SMALL
    #  H  safeguards: sector caps, BP, min-notional, position checks, ADD_POSITION_MISS
    #  I  planned order append: PlannedOrder + PlanLine planned
    # -------------------------------------------------------------------------
    div_summary: Dict[str, Any] = {
        "enabled": False,
        "adjusted": 0,
        "top_positive": [],
        "top_negative": [],
    }
    working_df = df
    if not working_df.empty:
        working_df, div_summary = _maybe_reorder_df_for_diversification(
            working_df,
            cfg,
            positions,
            acct,
            broker,
            sector_exp,
            sector_enforce,
            equity=equity,
            spendable_bp=spendable_bp,
            cap_pos_w=cap_pos_w,
            exp_mult=exp_mult,
            cap_gross=cap_gross,
            dw=dw,
            aw=aw,
            min_notional=min_notional,
            buy_bps=buy_bps,
            sell_bps=sell_bps,
            max_n_order=max_n_order,
            verbose=verbose,
        )

    # STAGE A (score) + STAGE B (rank): authoritative final_score column; process strongest first
    planner_intake_rows = int(len(df))
    planner_rank_applied = False
    adapt_summary: Dict[str, Any] = {}
    if not working_df.empty:
        working_df = _attach_exec_final_score_column(working_df)
        if bool(cfg.get("exec_score_percentile_normalize", True)):
            working_df = _normalize_exec_final_score_percentile_rank(working_df)
        try:
            from services.adaptation_simulation import apply_runtime_score_influence

            _enf = bool(cfg.get("enforce_min_final_score", True))
            _min_thr: Optional[float] = (
                float(cfg.get("min_final_score_threshold", 0.65) or 0.0) if _enf else None
            )
            _blk = bool(cfg.get("block_negative_final_score", True))
            working_df, adapt_summary = apply_runtime_score_influence(
                working_df,
                min_final_score_threshold=_min_thr,
                block_negative_final_score=_blk,
            )
        except Exception:
            adapt_summary = {}
        if adapt_summary:
            ac = int(adapt_summary.get("active_adjustment_count", 0) or 0)
            ra = int(adapt_summary.get("rows_affected", 0) or 0)
            if ac > 0 or ra > 0:
                print("[ADAPTATION_IMPACT_SUMMARY]")
                print(f"active_adjustments={ac}")
                sup = int(adapt_summary.get("supported_adjustments", 0) or 0)
                uns = int(adapt_summary.get("unsupported_adjustments", 0) or 0)
                print(f"supported_adjustments={sup}")
                print(f"unsupported_adjustments={uns}")
                print(f"rows_affected={ra}")
                print(
                    f"rows_score_delta_nonzero="
                    f"{int(adapt_summary.get('rows_score_delta_nonzero', 0) or 0)}"
                )
                print(f"avg_score_delta={adapt_summary.get('avg_score_delta', 0.0)}")
                print(f"matched_targets={adapt_summary.get('matched_targets', '')}")
                print(f"unmatched_targets={adapt_summary.get('unmatched_targets', '')}")
                print(f"top_adjustments={adapt_summary.get('top_adjustments', '')}")
                prm = adapt_summary.get("per_target_rows_matched") or {}
                prc = adapt_summary.get("per_target_rows_changed") or {}
                pru = adapt_summary.get("per_target_reason_if_unused") or {}
                for tgt in sorted(set(list(prm.keys()) + list(prc.keys()) + list(pru.keys()))):
                    print(
                        f"[ADAPT_PER_TARGET] target={tgt} rows_matched={int(prm.get(tgt, 0) or 0)} "
                        f"rows_changed={int(prc.get(tgt, 0) or 0)} "
                        f"reason_if_unused={pru.get(tgt, '')}"
                    )
                avail = adapt_summary.get("available_runtime_fields") or ""
                if avail:
                    print(f"available_runtime_fields={avail[:400]}")
                ri = adapt_summary.get("runtime_input") or []
                if ri:
                    print("[ADAPTATION_RUNTIME_INPUT]")
                    for entry in ri:
                        if not isinstance(entry, dict):
                            continue
                        tgt_e = entry.get("adaptation_target", "")
                        sup_e = entry.get("supported_rule", "")
                        eff = entry.get("effect", "")
                        ed = entry.get("effective_delta", "")
                        rb = entry.get("related_bucket", "")
                        rf = entry.get("related_flag", "")
                        req = str(entry.get("required_fields_hint") or "")[:120]
                        rm = entry.get("rows_matched", "")
                        rc = entry.get("rows_changed", "")
                        unr = str(entry.get("likely_no_match_reason") or "")[:160]
                        print(
                            f"  target={tgt_e} supported={sup_e} effect={eff} "
                            f"delta={ed} related_bucket={rb} related_flag={rf} "
                            f"rows_matched={rm} rows_changed={rc} hint={req} "
                            f"unused={unr}"
                        )
            _dc2 = adapt_summary.get("decision_changes")
            if isinstance(_dc2, dict):
                print(
                    f"[ADAPTATION_DECISIONS] newly_rejected={int(_dc2.get('newly_rejected', 0) or 0)} "
                    f"newly_accepted={int(_dc2.get('newly_accepted', 0) or 0)} "
                    f"rank_changes={int(_dc2.get('rank_changes', 0) or 0)}"
                )
            print(
                f"[ADAPTATION_BORDERLINE] band_rows={int(adapt_summary.get('borderline_band_rows', 0) or 0)} "
                f"amplified_rows={int(adapt_summary.get('borderline_amplified_rows', 0) or 0)}"
            )
        if bool(cfg.get("planner_rank_by_final_score", True)):
            working_df = _rank_working_df_by_final_score(working_df)
            planner_rank_applied = True
            _nr = min(3, len(working_df))
            _top = ", ".join(
                f"{_norm_sym(working_df.iloc[i].get('ticker'))}"
                f"({float(working_df['_exec_final_score'].iloc[i]):.4f})"
                for i in range(_nr)
            )
            print(
                f"[PLANNER_RANK] ordered_by=final_score_desc rows={len(working_df)} "
                f"intake_rows={planner_intake_rows} top={_top}"
            )
        else:
            print(
                f"[PLANNER_RANK] ordered_by=file_order rows={len(working_df)} "
                f"intake_rows={planner_intake_rows} (planner_rank_by_final_score=False)"
            )

    # STAGE C–I: per-row loop inside _run_once; STAGE D capital rescue = optional second _run_once

    def _run_once(
        rescue_global: Set[str],
        rescue_add: Set[str],
        rescue_sizing: Set[str],
    ) -> Tuple[
        List[PlannedOrder],
        List[PlanLine],
        Dict[str, int],
        List[str],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, int],
        Dict[str, Any],
        List[str],
    ]:
        nonlocal pending_total_add

        skip_reasons: Dict[str, int] = {}
        planned: List[PlannedOrder] = []
        lines: List[PlanLine] = []
        add_position_miss_symbols: List[str] = []

        def bump(code: str) -> None:
            skip_reasons[code] = skip_reasons.get(code, 0) + 1

        quality_summary: Dict[str, int] = {
            "skipped_low_score": 0,
            "blocked_negative": 0,
            "passed_entry_filter": 0,
        }

        def _quality_skip_reason(sym_q: str, row_q: pd.Series) -> Optional[str]:
            fs = _row_execution_final_score(row_q)
            if bool(cfg.get("block_negative_final_score", True)) and fs < 0:
                print(f"[QUALITY_BLOCK] symbol={sym_q} final_score={fs:.4f}")
                quality_summary["blocked_negative"] = quality_summary.get("blocked_negative", 0) + 1
                return "NEGATIVE_SCORE_BLOCK"
            if bool(cfg.get("enforce_min_final_score", True)):
                min_s = float(cfg.get("min_final_score_threshold", 0.65) or 0.0)
                relaxed_min = float(
                    cfg.get("capital_engine_relaxed_min_final_score_threshold", 0.55) or 0.0
                )
                if fs < min_s:
                    if sym_q in rescue_global and fs >= relaxed_min:
                        return None
                    print(
                        f"[QUALITY_SKIP] symbol={sym_q} final_score={fs:.4f} threshold={min_s:.4f}"
                    )
                    quality_summary["skipped_low_score"] = (
                        quality_summary.get("skipped_low_score", 0) + 1
                    )
                    return "BELOW_MIN_SCORE"
            return None

        score_sizing_summary: Dict[str, Any] = {
            "full_size": 0,
            "reduced_size": 0,
            "skipped_too_small": 0,
            "mode": "off",
            "multiplier_sum": 0.0,
            "multiplier_n": 0,
            "min_multiplier_used": None,
            "max_multiplier_used": None,
        }
        score_sizing_enabled = bool(cfg.get("score_sizing_enabled", True))
        if score_sizing_enabled:
            print("[SCORE_SIZING_ACTIVE] enabled=True")
            score_sizing_summary["mode"] = _score_sizing_mode_normalized(cfg)
        else:
            print("[SCORE_SIZING_ACTIVE] enabled=False OR NOT_TRIGGERED")
            score_sizing_summary["mode"] = "off"

        def _record_multiplier_used(m: float) -> None:
            score_sizing_summary["multiplier_sum"] = (
                float(score_sizing_summary.get("multiplier_sum", 0.0) or 0.0) + m
            )
            score_sizing_summary["multiplier_n"] = (
                int(score_sizing_summary.get("multiplier_n", 0) or 0) + 1
            )
            mn = score_sizing_summary.get("min_multiplier_used")
            mx = score_sizing_summary.get("max_multiplier_used")
            if mn is None or m < mn:
                score_sizing_summary["min_multiplier_used"] = m
            if mx is None or m > mx:
                score_sizing_summary["max_multiplier_used"] = m

        def _apply_score_sizing_qty(
            sym_q: str,
            row_q: pd.Series,
            base_qty: int,
            lim_q: float,
            *,
            is_buy: bool,
        ) -> Tuple[int, Optional[str]]:
            fs = _row_execution_final_score(row_q)
            if not score_sizing_enabled:
                print(
                    f"[SCORE_SIZING] symbol={sym_q} final_score={fs:.4f} base_qty={base_qty} "
                    f"multiplier=1.0000 sized_qty={base_qty} mode=off"
                )
                return base_qty, None

            curve = _score_sizing_mode_normalized(cfg)
            base_ratio: Optional[float] = None
            gamma_v: Optional[float] = None
            raw_pre: Optional[float] = None

            if curve == "tier":
                mult, tier = _score_sizing_multiplier(fs, cfg)
            else:
                base_ratio, gamma_v, raw_pre, mult = _nonlinear_score_sizing_multipliers(fs, cfg)

            sized = int(math.floor(float(base_qty) * mult))
            if bool(cfg.get("round_share_qty_down", True)):
                sized = max(0, sized)

            def _skip_print(sized_q: int, skip_mode: str) -> None:
                print(
                    f"[SCORE_SIZING_SKIP] symbol={sym_q} final_score={fs:.4f} base_qty={base_qty} "
                    f"sized_qty={sized_q} reason=SCORE_SIZED_TOO_SMALL mode={skip_mode}"
                )

            rescue_mode = sym_q in rescue_sizing

            if sized < 1:
                if rescue_mode:
                    print(
                        f"[CAPITAL_ENGINE_RESCUE] symbol={sym_q} forced_qty=1 "
                        f"original_reason=SCORE_SIZED_TOO_SMALL"
                    )
                    sized = 1
                else:
                    _skip_print(sized, curve)
                    score_sizing_summary["skipped_too_small"] = (
                        int(score_sizing_summary.get("skipped_too_small", 0) or 0) + 1
                    )
                    return base_qty, "SCORE_SIZED_TOO_SMALL"
            n = sized * lim_q
            if n < min_notional:
                if rescue_mode:
                    print(
                        f"[CAPITAL_ENGINE_RESCUE] symbol={sym_q} forced_qty={sized} "
                        f"original_reason=SCORE_SIZED_TOO_SMALL note=below_min_notional"
                    )
                else:
                    _skip_print(sized, curve)
                    score_sizing_summary["skipped_too_small"] = (
                        int(score_sizing_summary.get("skipped_too_small", 0) or 0) + 1
                    )
                    return base_qty, "SCORE_SIZED_TOO_SMALL"
            if is_buy and n > spendable_bp + 1e-6:
                _skip_print(sized, curve)
                score_sizing_summary["skipped_too_small"] = (
                    int(score_sizing_summary.get("skipped_too_small", 0) or 0) + 1
                )
                return base_qty, "SCORE_SIZED_TOO_SMALL"

            if curve == "tier":
                if verbose:
                    print(
                        f"[SCORE_SIZING] symbol={sym_q} final_score={fs:.4f} base_qty={base_qty} "
                        f"multiplier={mult:.4f} sized_qty={sized} tier={tier} mode=tier"
                    )
                else:
                    print(
                        f"[SCORE_SIZING] symbol={sym_q} final_score={fs:.4f} base_qty={base_qty} "
                        f"multiplier={mult:.2f} sized_qty={sized} tier={tier} mode=tier"
                    )
                _record_multiplier_used(mult)
                if tier == "A":
                    score_sizing_summary["full_size"] = (
                        int(score_sizing_summary.get("full_size", 0) or 0) + 1
                    )
                else:
                    score_sizing_summary["reduced_size"] = (
                        int(score_sizing_summary.get("reduced_size", 0) or 0) + 1
                    )
            else:
                if verbose:
                    print(
                        f"[SCORE_SIZING] symbol={sym_q} final_score={fs:.4f} base_qty={base_qty} "
                        f"base_ratio={base_ratio:.4f} gamma={gamma_v:.4f} raw_multiplier={raw_pre:.4f} "
                        f"multiplier={mult:.4f} sized_qty={sized} mode=nonlinear"
                    )
                else:
                    print(
                        f"[SCORE_SIZING] symbol={sym_q} final_score={fs:.4f} base_qty={base_qty} "
                        f"multiplier={mult:.4f} sized_qty={sized} mode=nonlinear"
                    )
                _record_multiplier_used(mult)
                max_cap = float(cfg.get("score_sizing_max_multiplier", 1.0) or 1.0)
                if abs(mult - max_cap) < 1e-9:
                    score_sizing_summary["full_size"] = (
                        int(score_sizing_summary.get("full_size", 0) or 0) + 1
                    )
                else:
                    score_sizing_summary["reduced_size"] = (
                        int(score_sizing_summary.get("reduced_size", 0) or 0) + 1
                    )

            return sized, None

        # Planner vs ExecutionGuard handshake: ExecutionGuard._check_open_counts blocks ANY buy when
        # len(broker_positions) >= max_positions (see execution_guard.py). Until this gate ran only
        # at validate time, planned_success could include BUY/ADD rows that were guaranteed to fail
        # with MAX_POSITIONS. We mirror that cap here using the same cfg max_positions + live
        # positions snapshot. POSITION_NOT_FLAT / already-long checks also belong before planned rows.
        new_pos_count = 0
        add_count = 0
        gross_notional = 0.0
        pending_new_opens = (
            0  # new symbols not yet in `positions` but planned this run (guard-accurate)
        )
        seen_syms: set = set()
        batch_entry_conf_median: Optional[float] = (
            _batch_buy_add_confidence_median(working_df) if not working_df.empty else None
        )

        for idx, row in working_df.iterrows():
            sym = _norm_sym(row.get("ticker"))
            ot = str(row.get("opportunity_type") or "").strip()
            stance = _opp_to_stance(row)
            conf = _safe_float(row.get("confidence"), 0.0)
            d_pct = _safe_float(row.get("delta_pct"), 0.0)
            rationale = str(row.get("rationale") or "")
            pos_state = str(row.get("effective_position_state") or "").strip().upper()
            expl = (
                bool(row.get("exploration_flag"))
                if pd.notna(row.get("exploration_flag"))
                else False
            )

            if not sym:
                lines.append(
                    PlanLine(
                        "", "skip", "skipped", "NON_ACTIONABLE_STANCE", "", None, ot, conf, d_pct
                    )
                )
                bump("NON_ACTIONABLE_STANCE")
                continue

            if sym in seen_syms:
                lines.append(
                    PlanLine(
                        sym,
                        "skip",
                        "skipped",
                        "DUPLICATE_SYMBOL_IN_RUN",
                        stance or "",
                        None,
                        ot,
                        conf,
                        d_pct,
                    )
                )
                bump("DUPLICATE_SYMBOL_IN_RUN")
                continue
            seen_syms.add(sym)

            if stance is None or stance in ("HOLD", "WAIT", ""):
                lines.append(
                    PlanLine(
                        sym,
                        "skip",
                        "skipped",
                        "NON_ACTIONABLE_STANCE",
                        str(stance or ""),
                        None,
                        ot,
                        conf,
                        d_pct,
                    )
                )
                bump("NON_ACTIONABLE_STANCE")
                continue

            if conf < min_conf:
                lines.append(
                    PlanLine(
                        sym, "skip", "skipped", "LOW_CONFIDENCE", stance, None, ot, conf, d_pct
                    )
                )
                bump("LOW_CONFIDENCE")
                continue

            pos_qty = positions[sym].qty if sym in positions else 0.0
            is_long_broker = pos_qty > 1e-9
            is_long_csv = pos_state == "LONG"

            if len(planned) >= max_orders:
                lines.append(
                    PlanLine(
                        sym, "skip", "skipped", "MAX_ORDERS_REACHED", stance, None, ot, conf, d_pct
                    )
                )
                bump("MAX_ORDERS_REACHED")
                continue

            # Single place: already-long is evaluated for BUY (ADD routing vs low-confidence skip).
            already_long = is_long_broker or is_long_csv
            effective_stance: Optional[str] = stance
            if stance == "BUY" and already_long:
                print("[DEBUG_ALREADY_LONG_PATH] ticker=", sym, "confidence=", conf, flush=True)
                med = batch_entry_conf_median
                add_threshold = max(0.5, float(med)) if med is not None else 0.5
                if conf < add_threshold:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "ALREADY_LONG_LOW_CONFIDENCE_SKIP",
                            str(stance or "BUY"),
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("ALREADY_LONG_LOW_CONFIDENCE_SKIP")
                    continue
                if not is_long_broker:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "ALREADY_LONG_SKIP",
                            str(stance or "BUY"),
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("ALREADY_LONG_SKIP")
                    continue
                print("[ADD_DECISION]", flush=True)
                print(f"ticker={sym}", flush=True)
                print(f"confidence={conf}", flush=True)
                print(f"threshold={add_threshold}", flush=True)
                print("action=ADD", flush=True)
                effective_stance = "ADD"

            if effective_stance == "BUY":
                if not allow_new_risk:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "BUY_BLOCKED_BY_RISK",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("BUY_BLOCKED_BY_RISK")
                    continue
                if not allow_trades_cpm:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "BUY_BLOCKED_BY_CPM",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("BUY_BLOCKED_BY_CPM")
                    continue
                if new_pos_count >= max_new:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "MAX_NEW_POSITIONS_REACHED",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("MAX_NEW_POSITIONS_REACHED")
                    continue

                mp_cap = _effective_max_positions(cfg)
                if len(positions) + pending_new_opens >= mp_cap:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "MAX_POSITIONS",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("MAX_POSITIONS")
                    continue

                ref = _ref_price(broker, sym, "buy")
                if ref is None or ref <= 0:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "NO_PRICE_AVAILABLE",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("NO_PRICE_AVAILABLE")
                    continue

                bd = compute_size_factor_breakdown(conf, d_pct, row, cfg=cfg)
                sf = float(bd["size_factor_final"])
                w = min(dw * sf, cap_pos_w) * exp_mult
                target_notional = equity * w
                target_notional = min(target_notional, max_n_order)
                if gross_notional + target_notional > equity * cap_gross:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "MAX_GROSS_EXPOSURE",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("MAX_GROSS_EXPOSURE")
                    continue

                qty = int(math.floor(target_notional / ref))
                if bool(cfg.get("round_share_qty_down", True)):
                    qty = max(0, qty)
                if qty < 1:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "ZERO_QTY_AFTER_ROUNDING",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("ZERO_QTY_AFTER_ROUNDING")
                    continue

                lim = _limit_from_ref(ref, "buy", buy_bps, sell_bps)
                lim = _round_price(lim)
                notional = qty * lim
                if notional < min_notional:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "BELOW_MIN_ORDER_NOTIONAL",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("BELOW_MIN_ORDER_NOTIONAL")
                    continue
                if notional > spendable_bp + 1e-6:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "INSUFFICIENT_BUYING_POWER",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("INSUFFICIENT_BUYING_POWER")
                    continue

                if sector_enforce and sector_exp:
                    try:
                        allow_unk = bool(cfg.get("allow_unknown_sector", False))
                        if caps_on:
                            blocked, rcode, cur_pct, proj_pct, sec_lab = evaluate_sector_cap(
                                sym,
                                stance,
                                notional,
                                sector_exp,
                                pending_sector_add=pending_sector_add,
                                pending_total_add=pending_total_add,
                                cfg=cfg,
                            )
                            if blocked and rcode:
                                if verbose:
                                    print(
                                        f"[SECTOR_CAP] block symbol={sym} sector={sec_lab} action={stance} "
                                        f"current={cur_pct:.4f} projected={proj_pct:.4f} reason={rcode}"
                                    )
                                lines.append(
                                    PlanLine(
                                        sym,
                                        "skip",
                                        "skipped",
                                        rcode,
                                        stance,
                                        None,
                                        ot,
                                        conf,
                                        d_pct,
                                    )
                                )
                                bump(rcode)
                                sector_cap_reason_counts[rcode] = (
                                    sector_cap_reason_counts.get(rcode, 0) + 1
                                )
                                sector_summary["blocked_total"] = (
                                    int(sector_summary.get("blocked_total") or 0) + 1
                                )
                                continue
                        elif legacy_on:
                            blocked, det = should_block_buy_for_sector(
                                sym,
                                notional,
                                sector_exp,
                                pending_sector_add=pending_sector_add,
                                pending_total_add=pending_total_add,
                                block_pct=float(cfg.get("sector_block_pct", 0.40)),
                                allow_unknown_sector=allow_unk,
                            )
                            if blocked:
                                internal = (
                                    "UNKNOWN_SECTOR_BLOCK"
                                    if str(det or "") == "UNKNOWN_SECTOR_BLOCK"
                                    else "BUY_BLOCKED_BY_SECTOR"
                                )
                                if verbose:
                                    print(
                                        f"[SECTOR_CAP] block symbol={sym} sector={get_sector(sym)} action={stance} "
                                        f"reason={internal} legacy_detail={det}"
                                    )
                                lines.append(
                                    PlanLine(
                                        sym,
                                        "skip",
                                        "skipped",
                                        internal,
                                        stance,
                                        None,
                                        ot,
                                        conf,
                                        d_pct,
                                    )
                                )
                                bump(internal)
                                sector_cap_reason_counts[internal] = (
                                    sector_cap_reason_counts.get(internal, 0) + 1
                                )
                                sector_summary["blocked_total"] = (
                                    int(sector_summary.get("blocked_total") or 0) + 1
                                )
                                continue
                    except Exception:
                        pass

                skip_q = _quality_skip_reason(sym, row)
                if skip_q:
                    lines.append(
                        PlanLine(sym, "skip", "skipped", skip_q, stance, None, ot, conf, d_pct)
                    )
                    bump(skip_q)
                    continue
                quality_summary["passed_entry_filter"] = (
                    int(quality_summary.get("passed_entry_filter", 0) or 0) + 1
                )

                nq, sz_err = _apply_score_sizing_qty(sym, row, qty, lim, is_buy=True)
                if sz_err:
                    lines.append(
                        PlanLine(sym, "skip", "skipped", sz_err, stance, None, ot, conf, d_pct)
                    )
                    bump(sz_err)
                    continue
                qty = nq
                notional = qty * lim

                div_adj, exec_rank = _planned_div_fields(row)
                po = PlannedOrder(
                    symbol=sym,
                    side="buy",
                    qty=qty,
                    order_type="limit" if cfg.get("prefer_limit_orders", True) else "market",
                    time_in_force="day",
                    limit_price=lim if cfg.get("prefer_limit_orders", True) else None,
                    stance=stance,
                    rationale=rationale,
                    confidence=conf,
                    delta_pct=d_pct,
                    planned_notional=notional,
                    current_position_qty=pos_qty,
                    mode=mode,
                    generated_at=_utc_iso(),
                    exploration_flag=expl,
                    diversification_adjustment=div_adj,
                    execution_rank_score=exec_rank,
                )
                planned.append(po)
                gross_notional += notional
                new_pos_count += 1
                pending_new_opens += 1
                pending_sector_add[get_sector(sym)] += float(notional)
                pending_total_add += float(notional)
                lines.append(PlanLine(sym, "plan", "planned", "", stance, po, ot, conf, d_pct))

            elif effective_stance == "ADD":
                from_long_buy = str(stance or "").upper() == "BUY"
                ot_effective = _stance_to_opportunity_type("ADD") if from_long_buy else ot
                if not is_long_broker:
                    _raw_t = str(row.get("ticker") or "").strip()
                    print(
                        f"[ADD_POSITION_MISS] symbol={_raw_t} stance=ADD reason=POSITION_NOT_FOUND_FOR_ADD "
                        f"available_positions={sorted(positions.keys())} normalized_lookup_symbol={sym}"
                    )
                    add_position_miss_symbols.append(sym)
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "POSITION_NOT_FOUND_FOR_ADD",
                            effective_stance,
                            None,
                            ot_effective,
                            conf,
                            d_pct,
                        )
                    )
                    bump("POSITION_NOT_FOUND_FOR_ADD")
                    continue
                if not allow_new_risk:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "ADD_BLOCKED_BY_RISK",
                            effective_stance,
                            None,
                            ot_effective,
                            conf,
                            d_pct,
                        )
                    )
                    bump("ADD_BLOCKED_BY_RISK")
                    continue
                if not allow_trades_cpm:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "ADD_BLOCKED_BY_CPM",
                            effective_stance,
                            None,
                            ot_effective,
                            conf,
                            d_pct,
                        )
                    )
                    bump("ADD_BLOCKED_BY_CPM")
                    continue
                if add_count >= max_adds:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "MAX_ADDS_REACHED",
                            effective_stance,
                            None,
                            ot_effective,
                            conf,
                            d_pct,
                        )
                    )
                    bump("MAX_ADDS_REACHED")
                    continue

                mp_cap_add = _effective_max_positions(cfg)
                if len(positions) >= mp_cap_add:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "MAX_POSITIONS",
                            effective_stance,
                            None,
                            ot_effective,
                            conf,
                            d_pct,
                        )
                    )
                    bump("MAX_POSITIONS")
                    continue

                ref = _ref_price(broker, sym, "buy")
                if ref is None or ref <= 0:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "NO_PRICE_AVAILABLE",
                            effective_stance,
                            None,
                            ot_effective,
                            conf,
                            d_pct,
                        )
                    )
                    bump("NO_PRICE_AVAILABLE")
                    continue

                bd = compute_size_factor_breakdown(conf, d_pct, row, cfg=cfg)
                sf = float(bd["size_factor_final"])
                w = min(aw * sf, cap_pos_w) * exp_mult
                target_notional = equity * w
                target_notional = min(target_notional, max_n_order)
                qty = int(math.floor(target_notional / ref))
                if qty < 1:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "ZERO_QTY_AFTER_ROUNDING",
                            effective_stance,
                            None,
                            ot_effective,
                            conf,
                            d_pct,
                        )
                    )
                    bump("ZERO_QTY_AFTER_ROUNDING")
                    continue

                lim = _limit_from_ref(ref, "buy", buy_bps, sell_bps)
                lim = _round_price(lim)
                notional = qty * lim
                if notional < min_notional:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "BELOW_MIN_ORDER_NOTIONAL",
                            effective_stance,
                            None,
                            ot_effective,
                            conf,
                            d_pct,
                        )
                    )
                    bump("BELOW_MIN_ORDER_NOTIONAL")
                    continue
                if notional > spendable_bp + 1e-6:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "INSUFFICIENT_BUYING_POWER",
                            effective_stance,
                            None,
                            ot_effective,
                            conf,
                            d_pct,
                        )
                    )
                    bump("INSUFFICIENT_BUYING_POWER")
                    continue

                if sector_enforce and sector_exp:
                    try:
                        allow_unk = bool(cfg.get("allow_unknown_sector", False))
                        if caps_on:
                            blocked, rcode, cur_pct, proj_pct, sec_lab = evaluate_sector_cap(
                                sym,
                                effective_stance,
                                notional,
                                sector_exp,
                                pending_sector_add=pending_sector_add,
                                pending_total_add=pending_total_add,
                                cfg=cfg,
                            )
                            if blocked and rcode:
                                if verbose:
                                    print(
                                        f"[SECTOR_CAP] block symbol={sym} sector={sec_lab} action={effective_stance} "
                                        f"current={cur_pct:.4f} projected={proj_pct:.4f} reason={rcode}"
                                    )
                                lines.append(
                                    PlanLine(
                                        sym,
                                        "skip",
                                        "skipped",
                                        rcode,
                                        effective_stance,
                                        None,
                                        ot_effective,
                                        conf,
                                        d_pct,
                                    )
                                )
                                bump(rcode)
                                sector_cap_reason_counts[rcode] = (
                                    sector_cap_reason_counts.get(rcode, 0) + 1
                                )
                                sector_summary["blocked_total"] = (
                                    int(sector_summary.get("blocked_total") or 0) + 1
                                )
                                continue
                        elif legacy_on:
                            blocked, det = should_block_buy_for_sector(
                                sym,
                                notional,
                                sector_exp,
                                pending_sector_add=pending_sector_add,
                                pending_total_add=pending_total_add,
                                block_pct=float(cfg.get("sector_block_pct", 0.40)),
                                allow_unknown_sector=allow_unk,
                            )
                            if blocked:
                                internal = (
                                    "UNKNOWN_SECTOR_BLOCK"
                                    if str(det or "") == "UNKNOWN_SECTOR_BLOCK"
                                    else "ADD_BLOCKED_BY_SECTOR"
                                )
                                if verbose:
                                    print(
                                        f"[SECTOR_CAP] block symbol={sym} sector={get_sector(sym)} action={effective_stance} "
                                        f"reason={internal} legacy_detail={det}"
                                    )
                                lines.append(
                                    PlanLine(
                                        sym,
                                        "skip",
                                        "skipped",
                                        internal,
                                        effective_stance,
                                        None,
                                        ot_effective,
                                        conf,
                                        d_pct,
                                    )
                                )
                                bump(internal)
                                sector_cap_reason_counts[internal] = (
                                    sector_cap_reason_counts.get(internal, 0) + 1
                                )
                                sector_summary["blocked_total"] = (
                                    int(sector_summary.get("blocked_total") or 0) + 1
                                )
                                continue
                    except Exception:
                        pass

                fs_add = _row_execution_final_score(row)
                add_thr = float(cfg.get("add_min_final_score_threshold", 0.67) or 0.0)
                relaxed_add = float(
                    cfg.get("capital_engine_relaxed_add_score_threshold", 0.60) or 0.0
                )
                eff_add_thr = relaxed_add if sym in rescue_add else add_thr
                if fs_add < eff_add_thr:
                    print(
                        f"[ADD_FILTER] symbol={sym} final_score={fs_add:.4f} threshold={eff_add_thr:.4f} "
                        f"reason=ADD_BELOW_SCORE_THRESHOLD"
                    )
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "ADD_BELOW_SCORE_THRESHOLD",
                            effective_stance,
                            None,
                            ot_effective,
                            conf,
                            d_pct,
                        )
                    )
                    bump("ADD_BELOW_SCORE_THRESHOLD")
                    continue

                nq, sz_err = _apply_score_sizing_qty(sym, row, qty, lim, is_buy=True)
                if sz_err:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            sz_err,
                            effective_stance,
                            None,
                            ot_effective,
                            conf,
                            d_pct,
                        )
                    )
                    bump(sz_err)
                    continue
                qty = nq
                min_add_q = int(cfg.get("add_min_qty_increase", 2) or 0)
                if min_add_q > 0 and qty < min_add_q:
                    print(
                        f"[ADD_FILTER] symbol={sym} final_score={fs_add:.4f} sized_qty={qty} "
                        f"min_add_qty={min_add_q} reason=ADD_TOO_SMALL"
                    )
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "ADD_TOO_SMALL",
                            effective_stance,
                            None,
                            ot_effective,
                            conf,
                            d_pct,
                        )
                    )
                    bump("ADD_TOO_SMALL")
                    continue
                notional = qty * lim

                div_adj, exec_rank = _planned_div_fields(row)
                _dt = "add_to_position" if from_long_buy else ""
                po = PlannedOrder(
                    symbol=sym,
                    side="buy",
                    qty=qty,
                    order_type="limit" if cfg.get("prefer_limit_orders", True) else "market",
                    time_in_force="day",
                    limit_price=lim if cfg.get("prefer_limit_orders", True) else None,
                    stance=effective_stance,
                    rationale=rationale,
                    confidence=conf,
                    delta_pct=d_pct,
                    planned_notional=notional,
                    current_position_qty=pos_qty,
                    mode=mode,
                    generated_at=_utc_iso(),
                    exploration_flag=expl,
                    diversification_adjustment=div_adj,
                    execution_rank_score=exec_rank,
                    decision_tag=_dt,
                )
                planned.append(po)
                gross_notional += notional
                add_count += 1
                pending_sector_add[get_sector(sym)] += float(notional)
                pending_total_add += float(notional)
                lines.append(
                    PlanLine(
                        sym,
                        "plan",
                        "planned",
                        "",
                        effective_stance,
                        po,
                        ot_effective,
                        conf,
                        d_pct,
                        decision_tag=_dt,
                    )
                )

            elif effective_stance == "TRIM":
                if not is_long_broker:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "POSITION_NOT_FOUND_FOR_TRIM",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("POSITION_NOT_FOUND_FOR_TRIM")
                    continue
                ref = _ref_price(broker, sym, "sell")
                if ref is None or ref <= 0:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "NO_PRICE_AVAILABLE",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("NO_PRICE_AVAILABLE")
                    continue
                q = max(1, int(math.floor(pos_qty * trim_frac)))
                q = min(q, int(math.floor(pos_qty)))
                if q < 1:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "ZERO_QTY_AFTER_ROUNDING",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("ZERO_QTY_AFTER_ROUNDING")
                    continue
                lim = _limit_from_ref(ref, "sell", buy_bps, sell_bps)
                lim = _round_price(lim)
                notional = q * lim
                skip_q = _quality_skip_reason(sym, row)
                if skip_q:
                    lines.append(
                        PlanLine(sym, "skip", "skipped", skip_q, stance, None, ot, conf, d_pct)
                    )
                    bump(skip_q)
                    continue
                quality_summary["passed_entry_filter"] = (
                    int(quality_summary.get("passed_entry_filter", 0) or 0) + 1
                )

                nq, sz_err = _apply_score_sizing_qty(sym, row, q, lim, is_buy=False)
                if sz_err:
                    lines.append(
                        PlanLine(sym, "skip", "skipped", sz_err, stance, None, ot, conf, d_pct)
                    )
                    bump(sz_err)
                    continue
                q = nq
                notional = q * lim

                po = PlannedOrder(
                    symbol=sym,
                    side="sell",
                    qty=q,
                    order_type="limit" if cfg.get("prefer_limit_orders", True) else "market",
                    time_in_force="day",
                    limit_price=lim if cfg.get("prefer_limit_orders", True) else None,
                    stance=stance,
                    rationale=rationale,
                    confidence=conf,
                    delta_pct=d_pct,
                    planned_notional=notional,
                    current_position_qty=pos_qty,
                    mode=mode,
                    generated_at=_utc_iso(),
                    exploration_flag=expl,
                )
                planned.append(po)
                lines.append(PlanLine(sym, "plan", "planned", "", stance, po, ot, conf, d_pct))

            elif effective_stance == "EXIT":
                if not is_long_broker:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "POSITION_NOT_FOUND_FOR_EXIT",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("POSITION_NOT_FOUND_FOR_EXIT")
                    continue
                q = int(math.floor(pos_qty))
                if q < 1:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "ILLEGAL_SELL_PREVENTED",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("ILLEGAL_SELL_PREVENTED")
                    continue
                ref = _ref_price(broker, sym, "sell")
                if ref is None or ref <= 0:
                    lines.append(
                        PlanLine(
                            sym,
                            "skip",
                            "skipped",
                            "NO_PRICE_AVAILABLE",
                            stance,
                            None,
                            ot,
                            conf,
                            d_pct,
                        )
                    )
                    bump("NO_PRICE_AVAILABLE")
                    continue
                lim = _limit_from_ref(ref, "sell", buy_bps, sell_bps)
                lim = _round_price(lim)
                notional = q * lim
                skip_q = _quality_skip_reason(sym, row)
                if skip_q:
                    lines.append(
                        PlanLine(sym, "skip", "skipped", skip_q, stance, None, ot, conf, d_pct)
                    )
                    bump(skip_q)
                    continue
                quality_summary["passed_entry_filter"] = (
                    int(quality_summary.get("passed_entry_filter", 0) or 0) + 1
                )

                nq, sz_err = _apply_score_sizing_qty(sym, row, q, lim, is_buy=False)
                if sz_err:
                    lines.append(
                        PlanLine(sym, "skip", "skipped", sz_err, stance, None, ot, conf, d_pct)
                    )
                    bump(sz_err)
                    continue
                q = nq
                notional = q * lim

                po = PlannedOrder(
                    symbol=sym,
                    side="sell",
                    qty=q,
                    order_type="limit" if cfg.get("prefer_limit_orders", True) else "market",
                    time_in_force="day",
                    limit_price=lim if cfg.get("prefer_limit_orders", True) else None,
                    stance=stance,
                    rationale=rationale,
                    confidence=conf,
                    delta_pct=d_pct,
                    planned_notional=notional,
                    current_position_qty=pos_qty,
                    mode=mode,
                    generated_at=_utc_iso(),
                    exploration_flag=expl,
                )
                planned.append(po)
                lines.append(PlanLine(sym, "plan", "planned", "", stance, po, ot, conf, d_pct))

        sector_summary["sector_exposure_before"] = (
            sector_exposure_pcts(sector_exp) if sector_exp else {}
        )
        sector_summary["sector_exposure_after_planned"] = (
            sector_exposure_pcts(sector_exp, dict(pending_sector_add), float(pending_total_add))
            if sector_exp
            else {}
        )
        sector_summary["reason_counts"] = dict(sector_cap_reason_counts)

        return (
            planned,
            lines,
            skip_reasons,
            warnings,
            sector_summary,
            div_summary,
            quality_summary,
            score_sizing_summary,
            add_position_miss_symbols,
        )

    first = _run_once(set(), set(), set())
    (
        planned,
        lines,
        skip_reasons,
        warnings,
        sector_summary,
        div_summary,
        quality_summary,
        score_sizing_summary,
        add_position_miss_symbols,
    ) = first

    ces: Dict[str, Any] = {
        "enabled": bool(cfg.get("capital_engine_enabled", True)),
        "under_deployed": False,
        "borderline_candidates": 0,
        "rescued": 0,
        "rescued_symbols": [],
    }
    from services.capital_engine import (
        deploy_ratio_metrics,
        find_row_for_symbol,
        pick_borderline_rescues,
    )

    trigger = float(cfg.get("capital_engine_under_deploy_trigger", 0.05))
    eng = bool(cfg.get("capital_engine_enabled", True))
    _pn, _avail, dr = deploy_ratio_metrics(planned, acct, cfg)
    under = dr < trigger
    ces["under_deployed"] = under
    if eng:
        print("[CAPITAL_ENGINE] enabled=True")
        print(
            f"[CAPITAL_ENGINE] deploy_ratio={dr:.4f} trigger={trigger:.4f} under_deployed={under}"
        )
    else:
        print("[CAPITAL_ENGINE] enabled=False")
    print("[CAPITAL_DEPLOYMENT]")
    print(f"planned_notional={_pn:.2f} available_after_reserve={_avail:.2f} deploy_ratio={dr:.4f}")
    try:
        udr = float(cfg.get("capital_deploy_under_deployed_ratio", 0.05) or 0.05)
        if bool(ces.get("under_deployed")) and dr < udr:
            print(
                f"[CAPITAL_DEPLOYMENT] note=materially_under_deployed deploy_ratio={dr:.4f} "
                f"below_ratio={udr:.4f}"
            )
    except Exception:
        pass
    max_bd = int(cfg.get("capital_engine_max_borderline_trades", 1) or 0)
    if eng and under and max_bd > 0:
        rg, ra, rs, bc, rsym, rcnt = pick_borderline_rescues(
            lines, df, cfg, _row_execution_final_score
        )
        ces["borderline_candidates"] = bc
        if rg or ra or rs:
            for sym_r in rsym:
                row_r = find_row_for_symbol(df, sym_r)
                fs_r = _row_execution_final_score(row_r) if row_r is not None else 0.0
                if sym_r in rg:
                    src_r = "BELOW_MIN_SCORE"
                elif sym_r in ra:
                    src_r = "ADD_BELOW_SCORE_THRESHOLD"
                else:
                    src_r = "SCORE_SIZED_TOO_SMALL"
                print(
                    f"[CAPITAL_ENGINE_RESCUE] symbol={sym_r} final_score={fs_r:.4f} source={src_r}"
                )
            second = _run_once(rg, ra, rs)
            (
                planned,
                lines,
                skip_reasons,
                warnings,
                sector_summary,
                div_summary,
                quality_summary,
                score_sizing_summary,
                add_position_miss_symbols,
            ) = second
            ces["rescued"] = rcnt
            ces["rescued_symbols"] = list(rsym)
    print("[CAPITAL_ENGINE_SUMMARY]")
    print(f"enabled={ces['enabled']}")
    print(f"under_deployed={ces['under_deployed']}")
    print(f"borderline_candidates={ces['borderline_candidates']}")
    print(f"rescued={ces['rescued']}")
    print(f"rescued_symbols={ces['rescued_symbols']}")
    _pn_tot = sum(float(p.planned_notional or 0) for p in planned)
    planner_core_meta: Dict[str, Any] = {
        "intake_rows": planner_intake_rows,
        "ranked_rows": int(len(working_df)) if not df.empty else 0,
        "rank_applied": planner_rank_applied,
        "planned_count": len(planned),
        "total_planned_notional": _pn_tot,
        "planner_caps": {
            "max_orders_per_run": int(max_orders),
            "max_new_positions_per_run": int(max_new),
            "max_adds_per_run": int(max_adds),
        },
        "adaptation_decision_changes": (adapt_summary or {}).get("decision_changes", {}),
        "adaptation_borderline": {
            "band_rows": int((adapt_summary or {}).get("borderline_band_rows", 0) or 0),
            "amplified_rows": int((adapt_summary or {}).get("borderline_amplified_rows", 0) or 0),
        },
    }
    return (
        planned,
        lines,
        skip_reasons,
        warnings,
        sector_summary,
        div_summary,
        quality_summary,
        score_sizing_summary,
        add_position_miss_symbols,
        ces,
        planner_core_meta,
    )


def _broker_open_same_side_keys(broker: Any) -> Optional[Set[Tuple[str, str]]]:
    if broker is None:
        return None
    try:
        from services.place_live_orders import build_open_index, list_open_orders

        oo = list_open_orders(broker)
        idx = build_open_index(oo or [])
        return set(idx.keys())
    except Exception:
        return None


def _sum_broker_open_qty_for_side(broker: Any, sym: str, side: str) -> Optional[float]:
    """Sum remaining open order qty for (symbol, side); None if unavailable."""
    if broker is None:
        return None
    try:
        from services.place_live_orders import build_open_index, list_open_orders, safe_int

        idx = build_open_index(list_open_orders(broker) or [])
        total = 0
        for o in idx.get((sym, side), []):
            q = safe_int(o.get("qty") or o.get("quantity"))
            if q and q > 0:
                total += int(q)
        return float(total)
    except Exception:
        return None


def _sum_snapshot_open_qty_for_side(snap_path: Path, sym: str, side: str) -> float:
    """Best-effort sum of open qty from snapshot CSV (non-terminal statuses)."""
    terminal = {
        "filled",
        "canceled",
        "cancelled",
        "done_for_day",
        "expired",
        "replaced",
        "failed",
    }
    if not snap_path.is_file():
        return 0.0
    try:
        sdf = pd.read_csv(snap_path, on_bad_lines="skip", keep_default_na=False)
    except Exception:
        return 0.0
    if sdf is None or sdf.empty:
        return 0.0
    sdf.columns = [str(c).strip() for c in sdf.columns]
    if "symbol" not in sdf.columns or "side" not in sdf.columns:
        return 0.0
    qcol = "qty" if "qty" in sdf.columns else ("quantity" if "quantity" in sdf.columns else None)
    if not qcol:
        return 0.0
    stcol = "status" if "status" in sdf.columns else None
    total = 0.0
    for _, r in sdf.iterrows():
        rsym = _norm_sym(r.get("symbol"))
        if rsym != sym:
            continue
        rsd = str(r.get("side") or "").strip().lower()
        if rsd != side:
            continue
        if stcol:
            st = str(r.get(stcol) or "").strip().lower()
            if st in terminal:
                continue
        try:
            q = float(r.get(qcol) or 0.0)
        except (TypeError, ValueError):
            q = 0.0
        if q > 0:
            total += q
    return total


def _position_qty_for_symbol(positions: Optional[Dict[str, PositionSnapshot]], sym: str) -> float:
    if not positions:
        return 0.0
    ps = positions.get(sym)
    if ps is None:
        return 0.0
    return float(ps.qty)


def apply_in_flight_prefilter(
    planned: List[PlannedOrder],
    plan_lines: List[PlanLine],
    skip_reasons: Dict[str, int],
    broker: Any,
    *,
    positions: Optional[Dict[str, PositionSnapshot]] = None,
    verbose: bool = False,
    session: str = "",
) -> Tuple[List[PlannedOrder], List[PlanLine], Dict[str, int], Dict[str, Any]]:
    """
    Drop planned orders when an open broker/snapshot order already matches (symbol, side).
    Marks plan_lines as skipped with IN_FLIGHT_ALREADY_SATISFIED (order_discipline remains final net).
    """
    meta: Dict[str, Any] = {
        "count": 0,
        "symbols": [],
        "suppressed_in_flight_satisfied": 0,
        "reason_breakdown": {"IN_FLIGHT_ALREADY_SATISFIED": 0},
    }
    try:
        from services.order_discipline import load_open_same_side_keys
    except Exception:
        return planned, plan_lines, skip_reasons, meta

    snap = RESULTS / "open_orders_snapshot.csv"
    broker_keys = _broker_open_same_side_keys(broker)
    open_keys = load_open_same_side_keys(snap, broker_open_keys=broker_keys)
    if not open_keys:
        return planned, plan_lines, skip_reasons, meta

    confs: List[float] = []
    for _pl0 in plan_lines:
        if _pl0.status != "planned" or _pl0.planned is None:
            continue
        _p0 = _pl0.planned
        _sd0 = str(_p0.side or "").strip().lower()
        if _sd0 not in ("buy", "sell"):
            continue
        try:
            _c0 = float(_p0.confidence)
        except (TypeError, ValueError):
            _c0 = float("nan")
        if math.isfinite(_c0):
            confs.append(_c0)
    batch_median: Optional[float] = None
    if confs:
        try:
            batch_median = float(statistics.median(confs))
        except Exception:
            batch_median = None

    def bump(code: str) -> None:
        skip_reasons[code] = skip_reasons.get(code, 0) + 1

    for pl in plan_lines:
        if pl.status != "planned" or pl.planned is None:
            continue
        p = pl.planned
        sym = str(p.symbol or "").strip().upper()
        sd = str(p.side or "").strip().lower()
        if not sym or sd not in ("buy", "sell"):
            continue
        if (sym, sd) in open_keys:
            bq = _sum_broker_open_qty_for_side(broker, sym, sd)
            sq = _sum_snapshot_open_qty_for_side(snap, sym, sd)
            if bq is None:
                oq = sq
            elif bq <= 0 and sq > 0:
                oq = sq
            else:
                oq = bq
            pq = _position_qty_for_symbol(positions, sym)
            tgt = int(p.qty)
            try:
                c_conf = float(p.confidence)
            except (TypeError, ValueError):
                c_conf = float("nan")
            if batch_median is not None and math.isfinite(c_conf) and c_conf >= float(batch_median):
                print(
                    f"[IN_FLIGHT_OVERRIDE] symbol={sym} "
                    f"confidence={c_conf:.6f} reason=high_confidence_override",
                    flush=True,
                )
                continue
            if verbose:
                print(
                    f"[IN_FLIGHT_SKIP] symbol={sym} stance={p.stance} reason=IN_FLIGHT_ALREADY_SATISFIED "
                    f"existing_open_qty={oq:.4f} existing_position_qty={pq:.4f} target_qty={tgt} session={session}"
                )
            meta["count"] = int(meta.get("count") or 0) + 1
            syms: List[str] = list(meta.get("symbols") or [])
            syms.append(sym)
            meta["symbols"] = syms
            pl.action = "skip"
            pl.status = "skipped"
            pl.skip_reason = "IN_FLIGHT_ALREADY_SATISFIED"
            pl.planned = None
            bump("IN_FLIGHT_ALREADY_SATISFIED")

    meta["suppressed_in_flight_satisfied"] = int(meta.get("count") or 0)
    meta["reason_breakdown"]["IN_FLIGHT_ALREADY_SATISFIED"] = int(meta.get("count") or 0)

    rebuilt: List[PlannedOrder] = []
    for pl in plan_lines:
        if pl.status == "planned" and pl.planned is not None:
            rebuilt.append(pl.planned)
    return rebuilt, plan_lines, skip_reasons, meta


def write_execution_plan(
    planned: List[PlannedOrder],
    plan_lines: List[PlanLine],
    summary: ExecutionPlanSummary,
    cfg: Dict[str, Any],
    orders_path: Path,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LIVE.mkdir(parents=True, exist_ok=True)

    with EXEC_PLAN_JSON.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    rows_out: List[Dict[str, Any]] = []
    _ei_cfg = ExecutionIntelligenceConfig()
    for pl in plan_lines:
        d: Dict[str, Any] = {
            "symbol": pl.symbol,
            "action": pl.action,
            "status": pl.status,
            "skip_reason": pl.skip_reason,
            "stance": pl.stance,
            "decision_tag": pl.decision_tag,
        }
        if pl.planned:
            p = pl.planned
            d.update(
                {
                    "qty": p.qty,
                    "side": p.side,
                    "limit_price": p.limit_price,
                    "planned_notional": p.planned_notional,
                    "rationale": p.rationale,
                }
            )
            for k, v in asdict(p).items():
                if k not in d:
                    d[k] = v
        # Execution-intelligence annotations (planning stage — quote unavailable here,
        # so style is driven by liquidity proxy / action class with neutral fallbacks).
        try:
            _ei = _ei_annotate_order(
                action=pl.stance or pl.action,
                side=getattr(pl.planned, "side", None) if pl.planned else None,
                close=getattr(pl.planned, "limit_price", None) if pl.planned else None,
                order_qty=getattr(pl.planned, "qty", None) if pl.planned else None,
                order_notional=(
                    getattr(pl.planned, "planned_notional", None) if pl.planned else None
                ),
                intended_price=getattr(pl.planned, "limit_price", None) if pl.planned else None,
                submitted_limit_price=(
                    getattr(pl.planned, "limit_price", None) if pl.planned else None
                ),
                cfg=_ei_cfg,
            )
            for _k in _EI_KEYS:
                if _k not in d:
                    d[_k] = _ei.get(_k)
        except Exception:
            pass
        rows_out.append(d)

    if rows_out:
        plan_keys = sorted(set().union(*(r.keys() for r in rows_out)))
        with EXEC_PLAN_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=plan_keys, extrasaction="ignore")
            w.writeheader()
            for r in rows_out:
                w.writerow(r)

    emit_orders_today_csv(planned, orders_path, cfg)


def append_exec_log(summary: ExecutionPlanSummary) -> None:
    try:
        row = {
            "ts_utc": summary.timestamp,
            "mode": summary.mode,
            "dry_run": summary.dry_run,
            "planned": summary.orders_planned,
            "skipped": summary.orders_skipped,
            "blocked": summary.blocked,
        }
        write_h = not EXEC_LOG_CSV.is_file() or EXEC_LOG_CSV.stat().st_size == 0
        with EXEC_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_h:
                w.writeheader()
            w.writerow(row)
    except Exception:
        pass


def _artifact_age_minutes(path: Path) -> Optional[float]:
    try:
        if not path.is_file():
            return None
        return (time.time() - path.stat().st_mtime) / 60.0
    except OSError:
        return None


def maybe_refresh_snapshots_for_execution(mode: str, cfg: Dict[str, Any], verbose: bool) -> None:
    """
    If CSV snapshot artifacts are stale vs snapshot_hygiene_max_age_minutes, run snapshot_live_orders once.
    Does not modify MasterExecutionGate rules; runs before gate.evaluate. Guard JSON is reported but not written here.
    """
    if not bool(cfg.get("snapshot_hygiene_enabled", True)):
        return
    max_age = float(cfg.get("snapshot_hygiene_max_age_minutes", 25.0))
    report_guard = bool(cfg.get("snapshot_hygiene_report_guard", True))

    positions_path = RESULTS / "positions_snapshot.csv"
    recent_path = RESULTS / "recent_orders.csv"
    open_path = RESULTS / "open_orders_snapshot.csv"
    guard_paths = [LIVE / "guard_snapshot.json", RESULTS / "guard_snapshot.json"]

    stale_parts: List[str] = []
    need_csv_refresh = False

    for label, p in (
        ("positions_snapshot.csv", positions_path),
        ("recent_orders.csv", recent_path),
        ("open_orders_snapshot.csv", open_path),
    ):
        age = _artifact_age_minutes(p)
        if not p.is_file():
            stale_parts.append(f"{label}(missing)")
            need_csv_refresh = True
        elif age is not None and age > max_age:
            stale_parts.append(f"{label}(age={age:.1f}m)")
            need_csv_refresh = True

    if report_guard:
        for gp in guard_paths:
            ga = _artifact_age_minutes(gp)
            if not gp.is_file():
                stale_parts.append(f"guard({gp.parent.name}/guard_snapshot.json missing)")
            elif ga is not None and ga > max_age:
                stale_parts.append(f"guard({gp.name} age={ga:.1f}m)")

    if not stale_parts:
        return

    print(f"[SNAPSHOT_HYGIENE] stale artifacts detected: {', '.join(stale_parts)}")

    if not need_csv_refresh:
        print(
            "[SNAPSHOT_HYGIENE] CSV snapshots within tolerance; refresh skipped "
            "(guard/report-only staleness — not updated by snapshot_live_orders)"
        )
        return

    print("[SNAPSHOT_HYGIENE] refresh triggered")
    try:
        cmd = [sys.executable, "-m", "services.snapshot_live_orders", "--mode", mode]
        if verbose:
            rc = subprocess.run(cmd, cwd=str(ROOT), timeout=120)
        else:
            rc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                timeout=120,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        if int(rc.returncode) == 0:
            print("[SNAPSHOT_HYGIENE] refresh complete")
        else:
            err = (getattr(rc, "stderr", None) or "").strip()[:800]
            print(f"[SNAPSHOT_HYGIENE] refresh failed: rc={rc.returncode} {err}")
    except Exception as e:
        print(f"[SNAPSHOT_HYGIENE] refresh failed: {e}")


def emit_orders_today_csv(
    planned: List[PlannedOrder], orders_path: Path, cfg: Dict[str, Any]
) -> None:
    """Write orders_today.csv from planned orders (shared with write_execution_plan / guard filtering)."""
    if not cfg.get("write_orders_today", True):
        return
    if planned:
        out_rows: List[Dict[str, Any]] = []
        for p in planned:
            eff_pos = "LONG" if p.stance in ("ADD", "TRIM", "EXIT") else "FLAT"
            lp = p.limit_price if p.limit_price is not None else ""
            eff_st = str(p.stance or "").strip().upper()
            # Handoff for place_live: _stance_from_row() uses `stance` and `lifecycle_action` first, not
            # `effective_stance` alone. Planner stance must win (ADD vs BUY) to avoid "already_long_no_new_buy".
            row: Dict[str, Any] = {
                "ticker": p.symbol,
                "stance": eff_st,
                "effective_stance": eff_st,
                "lifecycle_action": eff_st,
                "effective_position_state": eff_pos,
                "opportunity_type": _stance_to_opportunity_type(eff_st),
                "confidence": p.confidence,
                "delta_pct": p.delta_pct,
                "rationale": p.rationale,
                "close": lp,
                "limit_price": lp,
                "qty": p.qty,
                "exploration_flag": p.exploration_flag,
                "discipline_allowed": getattr(p, "discipline_allowed", True),
                "discipline_reason": getattr(p, "discipline_reason", ""),
            }
            for k, v in asdict(p).items():
                if k not in row:
                    row[k] = v
            row["stance"] = eff_st
            row["effective_stance"] = eff_st
            row["lifecycle_action"] = eff_st
            row["opportunity_type"] = _stance_to_opportunity_type(eff_st)
            dreason = (getattr(p, "decision_tag", None) or "").strip() or eff_st
            print("[EXECUTION_PATH_DEBUG]", flush=True)
            print(f"ticker={p.symbol}", flush=True)
            print(f"effective_stance={eff_st}", flush=True)
            print(f"decision_reason={dreason}", flush=True)
            out_rows.append(row)
        fieldnames = sorted(set().union(*(r.keys() for r in out_rows))) if out_rows else []
        orders_path.parent.mkdir(parents=True, exist_ok=True)
        with orders_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in out_rows:
                w.writerow(r)
    else:
        empty_cols = [
            "ticker",
            "effective_stance",
            "effective_position_state",
            "opportunity_type",
            "confidence",
            "delta_pct",
            "rationale",
            "close",
            "qty",
            "exploration_flag",
        ]
        orders_path.parent.mkdir(parents=True, exist_ok=True)
        with orders_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=empty_cols, extrasaction="ignore")
            w.writeheader()


def _validate_guard_orders(
    planned: List[PlannedOrder],
    broker: Any,
    cfg: Dict[str, Any],
    *,
    verbose: bool = False,
) -> Tuple[List[PlannedOrder], List[str], bool]:
    """
    Per-order ExecutionGuard.validate. Returns:
      eligible_planned — orders that pass the guard
      failures — symbol:code for each failure (same order as planned scan)
      has_hard_failure — True if any failure is not a known portfolio-state soft code
    """
    if not planned:
        return [], [], False
    if not cfg.get("require_guard_validation_for_each_order", True):
        return list(planned), [], False
    try:
        from services.execution_guard import ExecutionGuard

        guard = ExecutionGuard(broker)
        if verbose:
            try:
                positions = broker.get_positions()
                n = len(positions) if isinstance(positions, list) else 0
                mp = int(guard.cfg.get("max_positions", 25))
                cap_src = str(guard.cfg.get("_position_cap_source") or "")
                src = "config" if cap_src == "execute_trades" else "execution_guard"
                print(
                    f"[POSITION_CAP] current_positions={n} max_positions={mp} source={src}",
                    flush=True,
                )
            except Exception:
                pass
    except Exception as e:
        return [], [f"GUARD_INIT:{e}"], True

    eligible: List[PlannedOrder] = []
    failures: List[str] = []
    has_hard = False
    for p in planned:
        payload = {
            "symbol": p.symbol,
            "side": p.side,
            "qty": p.qty,
            "type": p.order_type,
            "limit_price": p.limit_price,
            "time_in_force": p.time_in_force,
        }
        d = guard.validate(payload)
        if d.ok:
            eligible.append(p)
        else:
            code = str(d.code or "")
            failures.append(f"{p.symbol}:{code}")
            if code not in _SOFT_GUARD_PORTFOLIO_CODES:
                has_hard = True
    return eligible, failures, has_hard


def _load_orders_sized_by_key(
    path: Path,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """
    Load orders_sized.csv: join_key -> {shares, position_value, weight}.
    position_sizer output is source of truth; last row per symbol wins (bottom-up scan).
    Requires ticker, shares. position_value and weight default to 0 if absent.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return "false", {}
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        if "ticker" not in df.columns or "shares" not in df.columns:
            return f"{path} (need ticker+shares)", {}
        m: Dict[str, Dict[str, float]] = {}
        for _, row in df.iloc[::-1].iterrows():
            jk = join_key(row.get("ticker"))
            if not jk or jk in m:
                continue
            v = pd.to_numeric(row.get("shares"), errors="coerce")
            if pd.isna(v):
                continue
            fv = float(v)
            if not math.isfinite(fv) or fv <= 0.0:
                continue
            pv = row.get("position_value")
            pvf = 0.0
            if pv is not None and str(pv).strip() != "":
                pv2 = pd.to_numeric(pv, errors="coerce")
                if not pd.isna(pv2) and math.isfinite(float(pv2)):
                    pvf = float(pv2)
            wgt = row.get("weight")
            wf = 0.0
            if wgt is not None and str(wgt).strip() != "":
                w2 = pd.to_numeric(wgt, errors="coerce")
                if not pd.isna(w2) and math.isfinite(float(w2)):
                    wf = float(w2)
            m[jk] = {"shares": fv, "position_value": pvf, "weight": wf}
        return str(path), m
    except Exception as e:
        return f"error: {e}", {}


def _int_qty_from_sized_shares(shares: float) -> int:
    if not math.isfinite(float(shares)) or float(shares) <= 0.0:
        return 0
    return max(1, int(math.floor(float(shares))))


def _recompute_planned_notional_after_qty_change(p: PlannedOrder, old_qty: int) -> None:
    new_q = int(p.qty)
    lp = p.limit_price
    if lp is not None:
        try:
            lpv = float(lp)
            if math.isfinite(lpv) and lpv > 0.0:
                p.planned_notional = float(new_q) * lpv
                return
        except (TypeError, ValueError):
            pass
    oq = int(old_qty)
    if oq > 0:
        p.planned_notional = float(p.planned_notional) * (float(new_q) / float(oq))


def apply_orders_sized_qty_overlay(
    planned: List[PlannedOrder], *, path: Path = ORDERS_SIZED_CSV
) -> Dict[str, Any]:
    """
    After the planner + filters: primary qty from data/results/orders_sized.csv when the symbol
    matches (join_key, same as position_sizer). Only mutates buy-side planned orders; planner qty
    is kept when there is no sized row, invalid shares, or shares <= 0. Entry-guard precap still runs after.
    """
    loaded, by_key = _load_orders_sized_by_key(path)
    n_buy = sum(1 for p in planned if str(p.side or "").lower() == "buy")
    matched = 0
    used = 0
    applied: List[Tuple[str, int]] = []
    applied_syms: List[str] = []
    for p in planned:
        if str(p.side or "").lower() != "buy":
            continue
        jk = join_key(p.symbol)
        if not jk or jk not in by_key:
            continue
        matched += 1
        rec = by_key[jk]
        nq = _int_qty_from_sized_shares(float(rec.get("shares", 0.0)))
        if nq <= 0:
            continue
        old = int(p.qty)
        p.qty = nq
        _recompute_planned_notional_after_qty_change(p, old)
        used += 1
        sym = str(p.symbol or "").strip().upper() or jk
        applied.append((sym, nq))
        applied_syms.append(sym)
        print(
            f"[SIZING_EXECUTION_DETAIL] ticker={sym} "
            f"shares={float(rec.get('shares', 0.0)):.4f} "
            f"position_value={float(rec.get('position_value', 0.0)):.4f} "
            f"weight={float(rec.get('weight', 0.0)):.6f}",
            flush=True,
        )
    applied.sort(key=lambda t: -t[1])
    top = [f"{a}:{b}" for a, b in applied[:5]]
    fallback = max(0, n_buy - used)
    sym_csv = ",".join(sorted(set(applied_syms)))
    print(
        f"[SIZING_EXECUTION_HANDOFF] matched={matched} used_sized_qty={used} "
        f"fallback_qty={fallback} symbols={sym_csv}",
        flush=True,
    )
    return {
        "sizing_file_loaded": loaded,
        "matched_rows": matched,
        "used_sized_qty": used,
        "fallback_default_qty": fallback,
        "top_sized_symbols": top,
    }


# ─────────────────────────────────────────────────────────────────────
# Edge-sizing overlay (read-only join from edge_sizing_engine.py output)
# ─────────────────────────────────────────────────────────────────────
_EDGE_TIER_MULTIPLIER_FALLBACK: Dict[str, float] = {
    "STRONG_EDGE": 1.25,
    "NORMAL_EDGE": 1.00,
    "WEAK_EDGE": 0.50,
    "BLOCKED": 0.00,
}


def _load_edge_sizing_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Return {join_key: {"tier": <TIER>, "multiplier": <float>}}.

    Defensive on every layer: missing / empty / malformed file produces an
    empty map with no warnings. The CSV's ``size_multiplier`` column is
    used when present and finite; otherwise we fall back to the canonical
    per-tier multiplier so the overlay still works on minimally-populated
    files.
    """
    out: Dict[str, Dict[str, Any]] = {}
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return out
    except OSError:
        return out
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return out
    if df is None or df.empty:
        return out
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    sym_col: Optional[str] = None
    for cand in ("ticker", "symbol"):
        if cand in df.columns:
            sym_col = cand
            break
    if sym_col is None or "sizing_tier" not in df.columns:
        return out
    has_mult_col = "size_multiplier" in df.columns
    for _, r in df.iterrows():
        jk = join_key(r.get(sym_col))
        if not jk or jk in out:
            continue
        tier = str(r.get("sizing_tier") or "").strip().upper()
        if not tier:
            continue
        mult: Optional[float] = None
        if has_mult_col:
            try:
                mult = float(r.get("size_multiplier"))
            except (TypeError, ValueError):
                mult = None
            if mult is not None and not math.isfinite(mult):
                mult = None
        if mult is None:
            mult = _EDGE_TIER_MULTIPLIER_FALLBACK.get(tier, 1.0)
        out[jk] = {"tier": tier, "multiplier": float(mult)}
    return out


def apply_edge_sizing_overlay(
    planned: List[PlannedOrder],
    plan_lines: List[PlanLine],
    skip_reasons: Dict[str, int],
    *,
    path: Path = EDGE_SIZING_RECOMMENDATIONS_CSV,
) -> Dict[str, Any]:
    """
    Apply per-symbol edge multipliers to BUY/ADD orders, AFTER the
    concentration sizer (orders_sized.csv) has produced base_qty and
    BEFORE the entry-guard precap clamps against max_notional / max_qty.

    Pipeline ordering (intentional):
        1. apply_orders_sized_qty_overlay       -> base_qty (concentration)
        2. apply_edge_sizing_overlay            -> base_qty × edge_multiplier
        3. apply_entry_guard_max_notional_precap -> clamps against caps

    This means STRONG_EDGE × 1.25 increases that exceed
    max_position_weight / max_notional are naturally clamped by step 3 —
    we never bypass any existing risk cap. WEAK_EDGE × 0.50 reductions
    leave a minimum of 1 share so we still ship the trade (lighter, not
    blocked). BLOCKED orders are dropped using the same pattern as the
    entry-guard precap (skip_reasons bookkeeping + plan-line detach).

    EXIT and TRIM are sell-side and are naturally excluded by the
    ``side == "buy"`` filter, matching the convention already established
    by ``apply_orders_sized_qty_overlay``. ADD is buy-side broker action
    and IS sized by this overlay (per spec: "Only apply to BUY / ADD").

    Read-only on the CSV. Missing / empty file is a clean no-op.
    """
    by_key = _load_edge_sizing_map(path)
    n_buy = sum(1 for p in planned if str(p.side or "").lower() == "buy")
    summary: Dict[str, Any] = {
        "sizing_file_loaded": bool(by_key),
        "matched": 0,
        "strong": 0,
        "normal": 0,
        "weak": 0,
        "blocked": 0,
        "unmatched": 0,
    }
    if not by_key:
        summary["unmatched"] = n_buy
        return summary

    to_remove: List[PlannedOrder] = []
    tier_counter_keys = {
        "STRONG_EDGE": "strong",
        "NORMAL_EDGE": "normal",
        "WEAK_EDGE": "weak",
    }
    for p in planned:
        if str(p.side or "").lower() != "buy":
            continue
        jk = join_key(p.symbol)
        if not jk or jk not in by_key:
            summary["unmatched"] = int(summary["unmatched"]) + 1
            continue
        rec = by_key[jk]
        tier = str(rec.get("tier") or "").upper()
        # Bare `or 1.0` would coerce a legitimate 0.0 (BLOCKED) to 1.0 and
        # mis-report it in [EDGE_SIZING_APPLIED]. Test for None explicitly.
        m_val = rec.get("multiplier")
        if m_val is None:
            mult = 1.0
        else:
            try:
                mult = float(m_val)
            except (TypeError, ValueError):
                mult = 1.0
        if not math.isfinite(mult):
            mult = 1.0
        summary["matched"] = int(summary["matched"]) + 1
        sym = str(p.symbol or "").strip().upper() or jk
        base_qty = int(p.qty)

        if tier == "BLOCKED" or mult <= 0.0:
            print(
                f"[EDGE_SIZING_APPLIED] symbol={sym} base_qty={base_qty} "
                f"multiplier={mult:.2f} final_qty=0 tier=BLOCKED",
                flush=True,
            )
            summary["blocked"] = int(summary["blocked"]) + 1
            skip_reasons["EDGE_SIZING_BLOCKED"] = (
                int(skip_reasons.get("EDGE_SIZING_BLOCKED", 0) or 0) + 1
            )
            try:
                _detach_planned_for_precap_drop(p, plan_lines, "EDGE_SIZING_BLOCKED")
            except Exception:
                pass
            to_remove.append(p)
            continue

        new_q = max(1, int(round(float(base_qty) * mult)))
        if new_q != base_qty:
            old = int(p.qty)
            p.qty = new_q
            _recompute_planned_notional_after_qty_change(p, old)
        ck = tier_counter_keys.get(tier)
        if ck:
            summary[ck] = int(summary[ck]) + 1
        print(
            f"[EDGE_SIZING_APPLIED] symbol={sym} base_qty={base_qty} "
            f"multiplier={mult:.2f} final_qty={p.qty} tier={tier}",
            flush=True,
        )

    for p in to_remove:
        if p in planned:
            planned.remove(p)

    print(
        f"[EDGE_SIZING_HANDOFF] matched={summary['matched']} "
        f"strong={summary['strong']} normal={summary['normal']} "
        f"weak={summary['weak']} blocked={summary['blocked']} "
        f"unmatched={summary['unmatched']}",
        flush=True,
    )
    return summary


def _read_guard_sizing_config(broker: Any) -> Optional[Tuple[float, float, int]]:
    """(min_notional, max_notional, max_qty) from ExecutionGuard config (no rule changes)."""
    try:
        from services.execution_guard import CONFIG_PATH, ExecutionGuard

        c: Dict[str, Any] = {}
        if broker is not None:
            c = dict(ExecutionGuard(broker).cfg)
        elif CONFIG_PATH.is_file() and CONFIG_PATH.stat().st_size > 0:
            o = json.loads(CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
            if isinstance(o, dict):
                c = o
        if not c:
            return None
        ExecutionGuard._merge_execute_trades_position_cap(c)  # type: ignore[attr-defined]
        mn = float(c.get("min_notional_usd", 25.0) or 25.0)
        mx = float(c.get("max_notional_usd", 1500.0) or 1500.0)
        mq = int(c.get("max_qty", 200) or 200)
        if not math.isfinite(mn) or not math.isfinite(mx) or mx <= 0 or mq < 1:
            return None
        return (max(0.0, mn), float(mx), max(1, mq))
    except Exception:
        return None


def _px_for_entry_guard(p: PlannedOrder, broker: Any) -> Optional[float]:
    if str(p.side or "").lower() != "buy":
        return None
    lp = p.limit_price
    if lp is not None:
        try:
            v = float(lp)
            if math.isfinite(v) and v > 0.0:
                return v
        except (TypeError, ValueError):
            pass
    if broker is None:
        return None
    try:
        px = broker.get_latest_price(str(p.symbol or "").strip().upper())
        if px is not None and float(px) > 0.0:
            return float(px)
    except Exception:
        pass
    return None


def _detach_planned_for_precap_drop(
    p: PlannedOrder, plan_lines: List[PlanLine], reason: str
) -> None:
    for pl in plan_lines:
        if pl.planned is p:
            pl.planned = None
            pl.status = "skipped"
            pl.skip_reason = reason
            pl.action = "skip"
            return


def apply_entry_guard_max_notional_precap(
    planned: List[PlannedOrder],
    plan_lines: List[PlanLine],
    broker: Any,
    skip_reasons: Dict[str, int],
    *,
    enabled: bool = True,
) -> Dict[str, Any]:
    """
    Shrink buy qty so estimated notional fits ExecutionGuard max_notional / max_qty before placement.
    Drops orders that cannot meet min notional after shrinking.
    """
    out: Dict[str, Any] = {
        "adjusted_symbols": 0,
        "unchanged_symbols": 0,
        "dropped_symbols": 0,
        "adjusted_list": [],
    }
    if not enabled or not planned:
        n_buy = sum(1 for p in planned if str(p.side or "").lower() == "buy")
        out["unchanged_symbols"] = n_buy
        return out

    cfg_t = _read_guard_sizing_config(broker)
    if cfg_t is None:
        n_buy = sum(1 for p in planned if str(p.side or "").lower() == "buy")
        out["unchanged_symbols"] = n_buy
        return out
    mn, mx, max_q = cfg_t
    to_remove: List[PlannedOrder] = []

    for p in list(planned):
        if str(p.side or "").lower() != "buy":
            continue
        sym = str(p.symbol or "").strip().upper()
        orig_q = int(p.qty)
        px = _px_for_entry_guard(p, broker)
        if px is None or px <= 0.0 or orig_q < 1:
            out["unchanged_symbols"] = int(out["unchanged_symbols"]) + 1
            continue

        n_est = float(orig_q) * float(px)
        over_n = n_est > mx + 1e-6
        over_q = orig_q > max_q
        if not over_n and not over_q:
            out["unchanged_symbols"] = int(out["unchanged_symbols"]) + 1
            continue

        q = orig_q
        if over_q:
            q = min(q, max_q)
        if q * float(px) > mx + 1e-6:
            q_n = int(math.floor((mx - 1e-6) / float(px)))
            q = min(q, max(0, q_n))
        if q < 0:
            q = 0

        new_est = float(q) * float(px) if q > 0 else 0.0
        if (q < 1) or (q > 0 and new_est < mn - 1e-6):
            print(
                f"[ENTRY_SIZE_ADJUSTMENT] symbol={sym} original_qty={orig_q} "
                f"adjusted_qty=0 reason=dropped_due_to_precap",
                flush=True,
            )
            to_remove.append(p)
            skip_reasons["DROPPED_DUE_TO_PRECAP"] = (
                int(skip_reasons.get("DROPPED_DUE_TO_PRECAP", 0) or 0) + 1
            )
            out["dropped_symbols"] = int(out["dropped_symbols"]) + 1
            _detach_planned_for_precap_drop(p, plan_lines, "DROPPED_DUE_TO_PRECAP")
            continue
        if q < orig_q:
            old = int(p.qty)
            p.qty = int(q)
            _recompute_planned_notional_after_qty_change(p, old)
            print(
                f"[ENTRY_SIZE_ADJUSTMENT] symbol={sym} original_qty={orig_q} "
                f"adjusted_qty={p.qty} reason=MAX_NOTIONAL_PRECAP",
                flush=True,
            )
            out["adjusted_symbols"] = int(out["adjusted_symbols"]) + 1
            out["adjusted_list"].append(sym)
        else:
            out["unchanged_symbols"] = int(out["unchanged_symbols"]) + 1

    for p in to_remove:
        if p in planned:
            planned.remove(p)

    return out


def maybe_execute_plan(
    planned: List[PlannedOrder],
    mode: str,
    dry_run: bool,
    cfg: Dict[str, Any],
    broker: Any,
    verbose: bool,
    ignore_market_closed: bool,
    orders_path: Path,
    drop_rows: Optional[List[Dict[str, Any]]] = None,
    session_tag: str = "",
    plan_extra: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, List[str]]:
    """Returns (exit_code, orders_executed_as_intent, block_reasons)."""
    if dry_run or not planned:
        return 0, 0, []

    if broker is None:
        return 1, 0, ["BROKER_UNAVAILABLE"]

    if not cfg.get("execute_via_existing_placement_flow", True):
        return 0, 0, []

    block_reasons: List[str] = []
    sess = session_tag or f"exec_trades_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    maybe_refresh_snapshots_for_execution(mode, cfg, verbose)

    if cfg.get("require_master_gate_for_execute", True):
        try:
            from services.master_execution_gate import (
                MasterExecutionGate,
                append_gate_log_csv,
                write_snapshot,
            )

            gate = MasterExecutionGate(project_root=ROOT)
            dec = gate.evaluate(
                mode=mode,
                broker=broker,
                verbose=verbose,
                require_market_open=(False if (mode == "live" and ignore_market_closed) else None),
            )
            write_snapshot(dec)
            append_gate_log_csv(dec)
            if not dec.ok:
                if drop_rows is not None:
                    for rsn in list(dec.reasons):
                        drop_rows.append(
                            make_row(
                                run_mode=mode,
                                symbol="",
                                stance="",
                                phase="placement_validation",
                                status="blocked",
                                reason_code=map_to_diagnostic_reason("PLACEMENT_BLOCKED"),
                                reason_detail=f"internal=PLACEMENT_BLOCKED; {rsn}",
                                source="execute_trades",
                                session=sess,
                            )
                        )
                    _emit_execution_drop_payload(drop_rows, mode, blocked=True, extra=plan_extra)
                return 2, 0, list(dec.reasons)
        except Exception as e:
            return 1, 0, [str(e)]

    eligible, guard_failures, has_hard = _validate_guard_orders(
        planned, broker, cfg, verbose=verbose
    )
    if guard_failures and has_hard:
        if drop_rows is not None:
            for bad in guard_failures:
                sym = bad.split(":", 1)[0].strip() if ":" in bad else ""
                drop_rows.append(
                    make_row(
                        run_mode=mode,
                        symbol=sym,
                        stance="",
                        phase="placement_validation",
                        status="blocked",
                        reason_code=map_to_diagnostic_reason("EXECUTION_GUARD_BLOCK"),
                        reason_detail=f"internal=EXECUTION_GUARD_BLOCK; {bad}",
                        source="execute_trades",
                        session=sess,
                    )
                )
            _emit_execution_drop_payload(drop_rows, mode, blocked=True, extra=plan_extra)
        return 2, 0, guard_failures

    if not eligible:
        if guard_failures and not has_hard:
            print(
                "[execute_trades] guard: no eligible orders after validation; "
                "failures are portfolio-state (soft) — not treating as fatal."
            )
            return 0, 0, guard_failures
        return 0, 0, []

    if len(eligible) < len(planned):
        print(
            f"[GUARD_FILTER] planned={len(planned)} eligible={len(eligible)} "
            f"(rewrote orders CSV to eligible subset)"
        )
        emit_orders_today_csv(eligible, orders_path, cfg)

    run_id = uuid.uuid4().hex[:12]
    if drop_rows is not None:
        _emit_execution_drop_payload(
            drop_rows, mode, blocked=False, run_id=run_id, extra=plan_extra
        )

    _print_execution_diversification_table(eligible)

    cmd = [
        sys.executable,
        "-m",
        "services.place_live_orders",
        "--mode",
        mode,
        "--orders",
        str(orders_path),
        "--no-dry-run",
        "--session",
        sess,
    ]
    if ignore_market_closed:
        cmd.append("--ignore-market-closed")
    if verbose:
        cmd.append("--verbose")

    mbn = cfg.get("max_batch_notional")
    if mbn is not None:
        try:
            cmd.extend(["--max-batch-notional", str(float(mbn))])
        except Exception:
            pass

    try:
        env = os.environ.copy()
        env["TRITON_EXEC_TRADES_RUN_ID"] = run_id
        p = subprocess.run(cmd, cwd=str(ROOT), env=env)
        rc = int(p.returncode)
        ex = len(eligible) if rc == 0 else 0
        _print_placement_override_diagnostics(eligible, cfg, rc)
        # Non-zero: place_live_orders writes diagnostics (merge + drops); do not overwrite here.
        return rc, ex, []
    except Exception as e:
        if drop_rows is not None:
            drop_rows.append(
                make_row(
                    run_mode=mode,
                    symbol="",
                    stance="",
                    phase="placement_validation",
                    status="dropped",
                    reason_code=map_to_diagnostic_reason("UNKNOWN_DROP_REASON"),
                    reason_detail=f"internal=UNKNOWN_DROP_REASON; {e}",
                    source="execute_trades",
                    session=sess,
                )
            )
            _emit_execution_drop_payload(drop_rows, mode, blocked=False, extra=plan_extra)
        return 1, 0, [str(e)]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TRITON execute_trades — plan from trade_opportunities.csv"
    )
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument(
        "--execute",
        action="store_true",
        help="After plan: master gate + place_live_orders (default is plan-only).",
    )
    ap.add_argument("--max-orders", type=int, default=None)
    ap.add_argument("--min-confidence", type=float, default=None)
    ap.add_argument(
        "--orders-path",
        type=str,
        default=None,
        help="Override output orders_today.csv path (still writes plan).",
    )
    ap.add_argument("--opportunities-path", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ignore-market-closed", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_execute_trades_config()
    if args.max_orders is not None:
        cfg["max_orders_per_run"] = args.max_orders
    if args.min_confidence is not None:
        cfg["min_confidence"] = args.min_confidence

    dry_run = not bool(args.execute)

    op_path = Path(args.opportunities_path) if args.opportunities_path else DEFAULT_OPPS
    orders_path = Path(args.orders_path) if args.orders_path else DEFAULT_ORDERS_TODAY

    _lg = evaluate_lifecycle_gate(path=RESULTS / "signal_lifecycle_effective.csv")
    print(_lg.format_block())
    lifecycle_gate_blocked = _lg.status == "BLOCKED"
    if lifecycle_gate_blocked:
        print(
            "[execute_trades] lifecycle gate BLOCKED — skipping trade_opportunities; "
            "no orders will be planned from lifecycle artifacts."
        )
        df = pd.DataFrame()
    else:
        df = load_trade_opportunities(op_path)
    positions, acct, broker = load_broker_state(args.mode)

    planned: List[PlannedOrder] = []
    plan_lines: List[PlanLine] = []
    skip_reasons: Dict[str, int] = {}
    warnings: List[str] = []
    session_tag = f"exec_trades_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    drop_rows: List[Dict[str, Any]] = []
    in_flight_meta: Dict[str, Any] = {
        "count": 0,
        "symbols": [],
        "suppressed_in_flight_satisfied": 0,
        "reason_breakdown": {"IN_FLIGHT_ALREADY_SATISFIED": 0},
    }

    if df.empty:
        if lifecycle_gate_blocked:
            _detail = f"internal=LIFECYCLE_GATE_BLOCK; reason={_lg.reason}; details={_lg.details}"
            _rc = map_to_diagnostic_reason("LIFECYCLE_GATE_BLOCK")
        else:
            _detail = "internal=NO_OPPORTUNITIES; trade_opportunities missing or empty"
            _rc = map_to_diagnostic_reason("NO_OPPORTUNITIES")
        drop_rows.append(
            make_row(
                run_mode=args.mode,
                symbol="",
                stance="",
                phase="planning",
                status="dropped",
                reason_code=_rc,
                reason_detail=_detail,
                source="execute_trades",
                session=session_tag,
            )
        )
        _emit_execution_drop_payload(drop_rows, args.mode, blocked=False)
        summary = ExecutionPlanSummary(
            timestamp=_utc_iso(),
            mode=args.mode,
            dry_run=dry_run,
            opportunities_seen=0,
            orders_planned=0,
            orders_executed=0,
            orders_skipped=0,
            skip_reasons={},
            blocked=False,
            block_reasons=[],
            source_file=str(op_path),
            orders_file=str(orders_path),
            warnings=[],
        )
        write_execution_plan([], [], summary, cfg, orders_path)
        append_exec_log(summary)
        write_last_execution_session_snapshot(
            session=session_tag,
            mode=args.mode,
            orders_planned=0,
            dry_run=dry_run,
            opportunities_seen=0,
        )
        print("[execute_trades] No opportunities (missing or empty). Plan written.")
        print("[PLANNER_SUMMARY]")
        print("candidates_total=0")
        print("candidates_ranked=0")
        print("rank_applied=False")
        print("filtered_quality=0")
        print("filtered_add=0")
        print("filtered_sector=0")
        print("filtered_inflight=0")
        print("filtered_bp=0")
        print("filtered_min_notional=0")
        print("planned_success=0")
        print("total_planned_notional=0.00 plan_lines_skipped=0")
        print("skip_buckets={}")
        print("top_skip_reasons=[]")
        print("[SCORE_SIZING_ACTIVE] enabled=False OR NOT_TRIGGERED")
        print("[IN_FLIGHT_SUMMARY]")
        print("blocked_open_orders=0")
        print("blocked_already_satisfied=0")
        print("other_inflight_blocks=0")
        print(f"count={in_flight_meta.get('count', 0)}")
        print(f"symbols={in_flight_meta.get('symbols') or []}")
        print("[QUALITY_SUMMARY]")
        print("skipped_low_score=0")
        print("blocked_negative=0")
        print("[SCORE_SIZING_SUMMARY]")
        _sse0 = bool(cfg.get("score_sizing_enabled", True))
        print(f"enabled={_sse0}")
        print(f"mode={'off' if not _sse0 else _score_sizing_mode_normalized(cfg)}")
        print("full_size=0")
        print("reduced_size=0")
        print("skipped_too_small=0")
        print("avg_multiplier=0.00")
        print("min_multiplier_used=n/a")
        print("max_multiplier_used=n/a")
        print("[ADD_FILTER_SUMMARY]")
        print("skip_add_below_score_threshold=0")
        print("skip_add_too_small=0")
        print("[ADD_POSITION_MISS_SUMMARY]")
        print("count=0")
        print("symbols=[]")
        return 0

    (
        planned,
        plan_lines,
        skip_reasons,
        warnings,
        sector_summary,
        div_summary,
        quality_summary,
        score_sizing_summary,
        add_position_miss_symbols,
        _capital_engine_summary,
        planner_core_meta,
    ) = build_execution_plan(
        df, cfg, positions, acct, broker, args.mode, verbose=bool(args.verbose)
    )

    planned, plan_lines, skip_reasons, in_flight_meta = apply_in_flight_prefilter(
        planned,
        plan_lines,
        skip_reasons,
        broker,
        positions=positions,
        verbose=bool(args.verbose),
        session=session_tag,
    )

    entry_eq_stats: Dict[str, int] = {"delayed_penalized": 0, "delayed_entry_skipped": 0}
    try:
        from services.execution_quality import (
            apply_execution_quality_filters,
            load_execution_quality_config,
        )

        _eq = load_execution_quality_config()
        for _ek in (
            "min_final_score_threshold",
            "delayed_entry_soft_penalty_enabled",
            "delayed_entry_penalty_factor",
        ):
            if _ek in cfg:
                _eq[_ek] = cfg[_ek]
        if _eq.get("enabled", True):
            planned, plan_lines, skip_reasons, entry_eq_stats = apply_execution_quality_filters(
                planned, plan_lines, skip_reasons, broker=broker, eq_cfg=_eq
            )
    except Exception:
        pass

    _thr_entry = float(cfg.get("min_final_score_threshold", 0.65) or 0.0)
    _pass_n = int(quality_summary.get("passed_entry_filter", 0) or 0)
    _rej_n = int(
        (quality_summary.get("skipped_low_score", 0) or 0)
        + (quality_summary.get("blocked_negative", 0) or 0)
    )
    _del_pen = int(entry_eq_stats.get("delayed_penalized", 0) or 0)
    print("[ENTRY_FILTER_SUMMARY]")
    print(f"threshold={_thr_entry:.6f}")
    print(f"passed={_pass_n}")
    print(f"rejected={_rej_n}")
    print(f"delayed_penalized={_del_pen}")

    if planned and not dry_run:
        try:
            from services.order_discipline import apply_discipline_to_planned_generic

            planned, _disc_meta = apply_discipline_to_planned_generic(
                planned,
                session=session_tag,
                source_module="execute_trades",
                mode=args.mode,
                context=None,
            )
            nb = int(_disc_meta.get("orders_blocked") or 0)
            if nb:
                skip_reasons["ORDER_DISCIPLINE"] = skip_reasons.get("ORDER_DISCIPLINE", 0) + nb
        except Exception:
            pass

    apply_orders_sized_qty_overlay(planned, path=ORDERS_SIZED_CSV)
    apply_edge_sizing_overlay(
        planned,
        plan_lines,
        skip_reasons,
        path=EDGE_SIZING_RECOMMENDATIONS_CSV,
    )

    _precap = apply_entry_guard_max_notional_precap(
        planned,
        plan_lines,
        broker,
        skip_reasons,
        enabled=bool(cfg.get("entry_guard_precap_buys", True)),
    )
    print(
        "[ENTRY_SIZE_ADJUSTMENT_SUMMARY] "
        f"adjusted_symbols={int(_precap.get('adjusted_symbols', 0) or 0)} "
        f"unchanged_symbols={int(_precap.get('unchanged_symbols', 0) or 0)} "
        f"dropped_symbols={int(_precap.get('dropped_symbols', 0) or 0)}",
        flush=True,
    )

    _chk_pre = int(planner_core_meta.get("ranked_rows", 0) or 0)
    _bmp_pre = int(skip_reasons.get("MAX_POSITIONS", 0) or 0)
    _baip_pre = int(skip_reasons.get("ALREADY_IN_PORTFOLIO", 0) or 0)
    _bnf_pre = int(skip_reasons.get("POSITION_NOT_FLAT_FOR_BUY", 0) or 0)
    _als_pre = int(skip_reasons.get("ALREADY_LONG_LOW_CONFIDENCE_SKIP", 0) or 0)
    _alnb_pre = int(skip_reasons.get("ALREADY_LONG_SKIP", 0) or 0)
    _ppass_pre = int(len(planned))
    print("[PRE_EXEC_ELIGIBILITY_SUMMARY]")
    print(f"checked={_chk_pre}")
    print(f"blocked_max_positions={_bmp_pre}")
    print(f"blocked_already_in_portfolio={_baip_pre}")
    print(f"blocked_position_not_flat={_bnf_pre}")
    print(f"blocked_already_long_low_confidence={_als_pre}")
    print(f"blocked_already_long_no_broker_add={_alnb_pre}")
    print(f"passed={_ppass_pre}")

    skipped = sum(1 for pl in plan_lines if pl.status == "skipped")
    drop_rows.extend(
        build_drop_rows_from_plan(plan_lines, planned, args.mode, session_tag, str(orders_path))
    )

    _pn_run = sum(float(p.planned_notional or 0) for p in planned)
    _ptc = _planner_transparency_counts(skip_reasons)
    print("[PLANNER_SUMMARY]")
    print(f"candidates_total={planner_core_meta.get('intake_rows', 0)}")
    print(f"candidates_ranked={planner_core_meta.get('ranked_rows', 0)}")
    print(f"rank_applied={planner_core_meta.get('rank_applied', False)}")
    print(f"filtered_quality={_ptc['filtered_quality']}")
    print(f"filtered_add={_ptc['filtered_add']}")
    print(f"filtered_sector={_ptc['filtered_sector']}")
    print(f"filtered_inflight={_ptc['filtered_inflight']}")
    print(f"filtered_bp={_ptc['filtered_bp']}")
    print(f"filtered_min_notional={_ptc['filtered_min_notional']}")
    print(f"planned_success={len(planned)}")
    print(f"total_planned_notional={_pn_run:.2f} plan_lines_skipped={skipped}")
    print(f"skip_buckets={_categorize_planner_skips(skip_reasons)}")
    print(f"top_skip_reasons={_top_skip_reasons(skip_reasons, 8)}")
    _opp_n = int(planner_core_meta.get("intake_rows", 0) or 0) or int(len(df))
    _qpass = int(quality_summary.get("passed_entry_filter", 0) or 0)
    _mord = int(cfg.get("max_orders_per_run", 5) or 0)
    _mnew = int(cfg.get("max_new_positions_per_run", 3) or 0)
    _madd = int(cfg.get("max_adds_per_run", 3) or 0)
    print("[SELECTION_COMPETITION]")
    print(f"opportunities_total={_opp_n}")
    print(f"quality_passed={_qpass}")
    print(f"planned_selected={len(planned)}")
    print(
        f"planner_caps max_orders_per_run={_mord} max_new_positions_per_run={_mnew} max_adds_per_run={_madd}"
    )

    summary = ExecutionPlanSummary(
        timestamp=_utc_iso(),
        mode=args.mode,
        dry_run=dry_run,
        opportunities_seen=len(df),
        orders_planned=len(planned),
        orders_executed=0,
        orders_skipped=skipped,
        skip_reasons=skip_reasons,
        blocked=False,
        block_reasons=[],
        source_file=str(op_path),
        orders_file=str(orders_path),
        warnings=warnings,
    )

    write_execution_plan(planned, plan_lines, summary, cfg, orders_path)
    append_exec_log(summary)
    write_last_execution_session_snapshot(
        session=session_tag,
        mode=args.mode,
        orders_planned=len(planned),
        dry_run=dry_run,
        opportunities_seen=len(df),
    )

    print(
        f"[execute_trades] mode={args.mode} dry_run={dry_run} opportunities={len(df)} planned={len(planned)} skipped={skipped}"
    )
    if skip_reasons:
        print(f"[execute_trades] skip_reasons={skip_reasons}")

    sc_blocked = int(sector_summary.get("blocked_total") or 0)
    sc_en = bool(sector_summary.get("enabled"))
    print(
        f"[SECTOR_CAP_SUMMARY] enabled={sc_en} blocked={sc_blocked} "
        f"reasons={sector_summary.get('reason_counts') or {}} "
        f"sector_exposure_before={sector_summary.get('sector_exposure_before') or {}} "
        f"sector_exposure_after_planned={sector_summary.get('sector_exposure_after_planned') or {}}"
    )
    print("[DIVERSIFICATION_SUMMARY]")
    print(f"enabled={div_summary.get('enabled')} adjusted={div_summary.get('adjusted')}")
    print(f"top_positive={div_summary.get('top_positive') or []}")
    print(f"top_negative={div_summary.get('top_negative') or []}")
    _diag = div_summary.get("diagnostics") or {}
    if _diag:
        print("[DIVERSIFICATION_DIAGNOSTICS]")
        print(f"base_score_min={_diag['base_score_min']:.4f}")
        print(f"base_score_median={_diag['base_score_median']:.4f}")
        print(f"base_score_max={_diag['base_score_max']:.4f}")
        print(f"raw_adj_min={_diag['raw_adj_min']:.4f}")
        print(f"raw_adj_max={_diag['raw_adj_max']:.4f}")
        print(f"eff_adj_min={_diag['eff_adj_min']:.4f}")
        print(f"eff_adj_max={_diag['eff_adj_max']:.4f}")
        print(f"final_score_min={_diag['final_score_min']:.4f}")
        print(f"final_score_median={_diag['final_score_median']:.4f}")
        print(f"final_score_max={_diag['final_score_max']:.4f}")
        print(f"[DIVERSIFICATION_DIAGNOSTICS] note={_diag.get('note', '')}")

    _isf = _inflight_summary_fields(skip_reasons)
    print("[IN_FLIGHT_SUMMARY]")
    print(f"blocked_open_orders={_isf['blocked_open_orders']}")
    print(f"blocked_already_satisfied={_isf['blocked_already_satisfied']}")
    print(f"other_inflight_blocks={_isf['other_inflight_blocks']}")
    print(f"count={in_flight_meta.get('count', 0)}")
    print(f"symbols={in_flight_meta.get('symbols') or []}")

    print("[QUALITY_SUMMARY]")
    print(f"skipped_low_score={quality_summary.get('skipped_low_score', 0)}")
    print(f"blocked_negative={quality_summary.get('blocked_negative', 0)}")

    print("[ADD_FILTER_SUMMARY]")
    print(f"skip_add_below_score_threshold={skip_reasons.get('ADD_BELOW_SCORE_THRESHOLD', 0)}")
    print(f"skip_add_too_small={skip_reasons.get('ADD_TOO_SMALL', 0)}")

    print("[SCORE_SIZING_SUMMARY]")
    _sse = bool(cfg.get("score_sizing_enabled", True))
    print(f"enabled={_sse}")
    print(f"mode={score_sizing_summary.get('mode', 'off')}")
    print(f"full_size={score_sizing_summary.get('full_size', 0)}")
    print(f"reduced_size={score_sizing_summary.get('reduced_size', 0)}")
    print(f"skipped_too_small={score_sizing_summary.get('skipped_too_small', 0)}")
    _mn = int(score_sizing_summary.get("multiplier_n") or 0)
    _avg_m = (float(score_sizing_summary.get("multiplier_sum") or 0.0) / _mn) if _mn else 0.0
    print(f"avg_multiplier={_avg_m:.2f}")
    _minu = score_sizing_summary.get("min_multiplier_used")
    _maxu = score_sizing_summary.get("max_multiplier_used")
    print(f"min_multiplier_used={_minu:.2f}" if _minu is not None else "min_multiplier_used=n/a")
    print(f"max_multiplier_used={_maxu:.2f}" if _maxu is not None else "max_multiplier_used=n/a")

    print("[ADD_POSITION_MISS_SUMMARY]")
    print(f"count={len(add_position_miss_symbols)}")
    print(f"symbols={add_position_miss_symbols}")

    if dry_run:
        _emit_execution_drop_payload(
            drop_rows,
            args.mode,
            blocked=False,
            extra={"sector_cap_summary": sector_summary},
        )
        return 0

    if not planned:
        _emit_execution_drop_payload(
            drop_rows,
            args.mode,
            blocked=False,
            extra={"sector_cap_summary": sector_summary},
        )
        print("[execute_trades] No planned orders; skipping placement.")
        return 0

    rc, executed, brs = maybe_execute_plan(
        planned,
        args.mode,
        False,
        cfg,
        broker,
        args.verbose,
        args.ignore_market_closed,
        orders_path,
        drop_rows=drop_rows,
        session_tag=session_tag,
        plan_extra={"sector_cap_summary": sector_summary},
    )
    summary.orders_executed = executed
    summary.blocked = rc == 2
    summary.block_reasons = brs
    try:
        with EXEC_PLAN_JSON.open("w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)
    except Exception:
        pass

    if rc == 2:
        print(f"[execute_trades] BLOCKED: {brs}")
        return 2
    if rc != 0:
        print(f"[execute_trades] placement rc={rc}")
        return rc if rc in (1, 2, 3) else 1
    print(f"[execute_trades] placement subprocess completed OK (planned={len(planned)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
