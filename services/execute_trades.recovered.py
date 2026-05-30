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

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
DEFAULT_OPPS = RESULTS / "trade_opportunities.csv"
DEFAULT_ORDERS_TODAY = LIVE / "orders_today.csv"
EXEC_PLAN_JSON = RESULTS / "execution_plan.json"
EXEC_PLAN_CSV = RESULTS / "execution_plan.csv"
EXEC_LOG_CSV = RESULTS / "execution_plan_log.csv"
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


def load_execute_trades_config() -> Dict[str, Any]:
    cfg = {
        "enabled": True,
        "default_mode": "paper",
        "dry_run_default": True,
        "min_confidence": 0.0,
        "max_new_positions_per_run": 9,
        "max_adds_per_run": 9,
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
        "max_orders_per_run": 18,
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
        "override_low_score_threshold": 0.02,
        "block_negative_final_score": True,
        "enforce_min_final_score": True,
        "min_final_score_threshold": 0.0025,
        "score_sizing_enabled": True,
        "score_sizing_tier_a_threshold": 0.0100,
        "score_sizing_tier_b_threshold": 0.0050,
        "score_sizing_tier_c_threshold": 0.0025,
        "score_sizing_tier_a_multiplier": 1.00,
        "score_sizing_tier_b_multiplier": 0.75,
        "score_sizing_tier_c_multiplier": 0.50,
        "score_sizing_mode": "nonlinear",
        "score_sizing_reference_score": 0.0115,
        "score_sizing_gamma": 1.5,
        "score_sizing_min_multiplier": 0.20,
        "score_sizing_max_multiplier": 1.00,
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
    thr = float(cfg.get("override_low_score_threshold", 0.02) or 0.02)
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
    ref = float(cfg.get("score_sizing_reference_score", 0.0115) or 0.0115)
    if ref <= 0:
        ref = 0.0115
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
    ta = float(cfg.get("score_sizing_tier_a_threshold", 0.01) or 0.01)
    tb = float(cfg.get("score_sizing_tier_b_threshold", 0.005) or 0.005)
    tc = float(cfg.get("score_sizing_tier_c_threshold", 0.0025) or 0.0025)
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
    """Final score used for diversification ranking: _div_final when present, else confidence * delta_pct."""
    if "_div_final" in row.index:
        fin = row.get("_div_final")
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
    if st == "BUY" and (is_long_broker or is_long_csv):
        return None
    if st == "ADD" and not is_long_broker:
        return None
    ref = _ref_price(broker, sym, "buy")
    if ref is None or ref <= 0:
        return None
    bd = compute_size_factor_breakdown(conf, d_pct, row, cfg=cfg)
    sf = float(bd["size_factor_final"])
    if st == "BUY":
        w = min(dw * sf, cap_pos_w) * exp_mult
    else:
        w = min(aw * sf, cap_pos_w) * exp_mult
    target_notional = equity * w
    target_notional = min(target_notional, max_n_order)
    qty = int(math.floor(target_notional / ref))
    if bool(cfg.get("round_share_qty_down", True)) and st == "BUY":
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
]:
    skip_reasons: Dict[str, int] = {}
    warnings: List[str] = []
    planned: List[PlannedOrder] = []
    lines: List[PlanLine] = []
    add_position_miss_symbols: List[str] = []

    def bump(code: str) -> None:
        skip_reasons[code] = skip_reasons.get(code, 0) + 1

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
    max_orders = int(cfg.get("max_orders_per_run", 18))
    max_new = int(cfg.get("max_new_positions_per_run", 9))
    max_adds = int(cfg.get("max_adds_per_run", 9))
    trim_frac = float(cfg.get("trim_fraction", 0.25))
    dw = float(cfg.get("default_position_weight", 0.05))
    aw = float(cfg.get("add_position_weight", 0.02))
    min_notional = float(cfg.get("min_order_notional", 50.0))
    buy_bps = float(cfg.get("limit_price_buffer_buy_bps", 35))
    sell_bps = float(cfg.get("limit_price_buffer_sell_bps", 20))
    max_n_order = max_order_notional_usd(cfg)

    div_summary: Dict[str, Any] = {
        "enabled": False,
        "adjusted": 0,
        "top_positive": [],
        "top_negative": [],
    }
    if not df.empty:
        df, div_summary = _maybe_reorder_df_for_diversification(
            df,
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

    quality_summary: Dict[str, int] = {"skipped_low_score": 0, "blocked_negative": 0}

    def _quality_skip_reason(sym_q: str, row_q: pd.Series) -> Optional[str]:
        fs = _row_final_quality_score(row_q)
        if bool(cfg.get("block_negative_final_score", True)) and fs < 0:
            print(f"[QUALITY_BLOCK] symbol={sym_q} final_score={fs:.4f}")
            quality_summary["blocked_negative"] = quality_summary.get("blocked_negative", 0) + 1
            return "NEGATIVE_SCORE_BLOCK"
        if bool(cfg.get("enforce_min_final_score", True)):
            min_s = float(cfg.get("min_final_score_threshold", 0.0025) or 0.0)
            if fs < min_s:
                print(f"[QUALITY_SKIP] symbol={sym_q} final_score={fs:.4f} threshold={min_s:.4f}")
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
        fs = _row_final_quality_score(row_q)
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

        if sized < 1:
            _skip_print(sized, curve)
            score_sizing_summary["skipped_too_small"] = (
                int(score_sizing_summary.get("skipped_too_small", 0) or 0) + 1
            )
            return base_qty, "SCORE_SIZED_TOO_SMALL"
        n = sized * lim_q
        if n < min_notional:
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

    new_pos_count = 0
    add_count = 0
    gross_notional = 0.0
    seen_syms: set = set()

    for idx, row in df.iterrows():
        sym = _norm_sym(row.get("ticker"))
        ot = str(row.get("opportunity_type") or "").strip()
        stance = _opp_to_stance(row)
        conf = _safe_float(row.get("confidence"), 0.0)
        d_pct = _safe_float(row.get("delta_pct"), 0.0)
        rationale = str(row.get("rationale") or "")
        pos_state = str(row.get("effective_position_state") or "").strip().upper()
        expl = bool(row.get("exploration_flag")) if pd.notna(row.get("exploration_flag")) else False

        if not sym:
            lines.append(
                PlanLine("", "skip", "skipped", "NON_ACTIONABLE_STANCE", "", None, ot, conf, d_pct)
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
                PlanLine(sym, "skip", "skipped", "LOW_CONFIDENCE", stance, None, ot, conf, d_pct)
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

        if stance == "BUY":
            if is_long_broker or is_long_csv:
                lines.append(
                    PlanLine(
                        sym,
                        "skip",
                        "skipped",
                        "POSITION_NOT_FLAT_FOR_BUY",
                        stance,
                        None,
                        ot,
                        conf,
                        d_pct,
                    )
                )
                bump("POSITION_NOT_FLAT_FOR_BUY")
                continue
            if not allow_new_risk:
                lines.append(
                    PlanLine(
                        sym, "skip", "skipped", "BUY_BLOCKED_BY_RISK", stance, None, ot, conf, d_pct
                    )
                )
                bump("BUY_BLOCKED_BY_RISK")
                continue
            if not allow_trades_cpm:
                lines.append(
                    PlanLine(
                        sym, "skip", "skipped", "BUY_BLOCKED_BY_CPM", stance, None, ot, conf, d_pct
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

            ref = _ref_price(broker, sym, "buy")
            if ref is None or ref <= 0:
                lines.append(
                    PlanLine(
                        sym, "skip", "skipped", "NO_PRICE_AVAILABLE", stance, None, ot, conf, d_pct
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
                        sym, "skip", "skipped", "MAX_GROSS_EXPOSURE", stance, None, ot, conf, d_pct
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
            pending_sector_add[get_sector(sym)] += float(notional)
            pending_total_add += float(notional)
            lines.append(PlanLine(sym, "plan", "planned", "", stance, po, ot, conf, d_pct))

        elif stance == "ADD":
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
                        stance,
                        None,
                        ot,
                        conf,
                        d_pct,
                    )
                )
                bump("POSITION_NOT_FOUND_FOR_ADD")
                continue
            if not allow_new_risk:
                lines.append(
                    PlanLine(
                        sym, "skip", "skipped", "ADD_BLOCKED_BY_RISK", stance, None, ot, conf, d_pct
                    )
                )
                bump("ADD_BLOCKED_BY_RISK")
                continue
            if not allow_trades_cpm:
                lines.append(
                    PlanLine(
                        sym, "skip", "skipped", "ADD_BLOCKED_BY_CPM", stance, None, ot, conf, d_pct
                    )
                )
                bump("ADD_BLOCKED_BY_CPM")
                continue
            if add_count >= max_adds:
                lines.append(
                    PlanLine(
                        sym, "skip", "skipped", "MAX_ADDS_REACHED", stance, None, ot, conf, d_pct
                    )
                )
                bump("MAX_ADDS_REACHED")
                continue

            ref = _ref_price(broker, sym, "buy")
            if ref is None or ref <= 0:
                lines.append(
                    PlanLine(
                        sym, "skip", "skipped", "NO_PRICE_AVAILABLE", stance, None, ot, conf, d_pct
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
                                else "ADD_BLOCKED_BY_SECTOR"
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
            add_count += 1
            pending_sector_add[get_sector(sym)] += float(notional)
            pending_total_add += float(notional)
            lines.append(PlanLine(sym, "plan", "planned", "", stance, po, ot, conf, d_pct))

        elif stance == "TRIM":
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
                        sym, "skip", "skipped", "NO_PRICE_AVAILABLE", stance, None, ot, conf, d_pct
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

        elif stance == "EXIT":
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
                        sym, "skip", "skipped", "NO_PRICE_AVAILABLE", stance, None, ot, conf, d_pct
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
    meta: Dict[str, Any] = {"count": 0, "symbols": []}
    try:
        from services.order_discipline import load_open_same_side_keys
    except Exception:
        return planned, plan_lines, skip_reasons, meta

    snap = RESULTS / "open_orders_snapshot.csv"
    broker_keys = _broker_open_same_side_keys(broker)
    open_keys = load_open_same_side_keys(snap, broker_open_keys=broker_keys)
    if not open_keys:
        return planned, plan_lines, skip_reasons, meta

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
    for pl in plan_lines:
        d: Dict[str, Any] = {
            "symbol": pl.symbol,
            "action": pl.action,
            "status": pl.status,
            "skip_reason": pl.skip_reason,
            "stance": pl.stance,
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
        rows_out.append(d)

    if rows_out:
        plan_keys = sorted(set().union(*(r.keys() for r in rows_out)))
        with EXEC_PLAN_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=plan_keys, extrasaction="ignore")
            w.writeheader()
            for r in rows_out:
                w.writerow(r)

    if cfg.get("write_orders_today", True) and planned:
        out_rows: List[Dict[str, Any]] = []
        for p in planned:
            eff_pos = "LONG" if p.stance in ("ADD", "TRIM", "EXIT") else "FLAT"
            lp = p.limit_price if p.limit_price is not None else ""
            row = {
                "ticker": p.symbol,
                "effective_stance": p.stance,
                "effective_position_state": eff_pos,
                "opportunity_type": {
                    "BUY": "ENTRY",
                    "ADD": "ADD",
                    "TRIM": "TRIM",
                    "EXIT": "EXIT",
                }.get(p.stance, ""),
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
            out_rows.append(row)
        fieldnames = sorted(set().union(*(r.keys() for r in out_rows))) if out_rows else []
        orders_path.parent.mkdir(parents=True, exist_ok=True)
        with orders_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in out_rows:
                w.writerow(r)
    elif cfg.get("write_orders_today", True):
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


def _validate_guard_orders(
    planned: List[PlannedOrder], broker: Any, cfg: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    if not planned:
        return True, []
    if not cfg.get("require_guard_validation_for_each_order", True):
        return True, []
    try:
        from services.execution_guard import ExecutionGuard

        guard = ExecutionGuard(broker)
    except Exception as e:
        return False, [f"GUARD_INIT:{e}"]

    bad: List[str] = []
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
        if not d.ok:
            bad.append(f"{p.symbol}:{d.code}")
    return (len(bad) == 0), bad


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

    ok_g, guard_bad = _validate_guard_orders(planned, broker, cfg)
    if not ok_g:
        if drop_rows is not None:
            for bad in guard_bad:
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
        return 2, 0, guard_bad

    run_id = uuid.uuid4().hex[:12]
    if drop_rows is not None:
        _emit_execution_drop_payload(
            drop_rows, mode, blocked=False, run_id=run_id, extra=plan_extra
        )

    _print_execution_diversification_table(planned)

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
        ex = len(planned) if rc == 0 else 0
        _print_placement_override_diagnostics(planned, cfg, rc)
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

    df = load_trade_opportunities(op_path)
    positions, acct, broker = load_broker_state(args.mode)

    planned: List[PlannedOrder] = []
    plan_lines: List[PlanLine] = []
    skip_reasons: Dict[str, int] = {}
    warnings: List[str] = []
    session_tag = f"exec_trades_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    drop_rows: List[Dict[str, Any]] = []
    in_flight_meta: Dict[str, Any] = {"count": 0, "symbols": []}

    if df.empty:
        drop_rows.append(
            make_row(
                run_mode=args.mode,
                symbol="",
                stance="",
                phase="planning",
                status="dropped",
                reason_code=map_to_diagnostic_reason("NO_OPPORTUNITIES"),
                reason_detail="internal=NO_OPPORTUNITIES; trade_opportunities missing or empty",
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
        print("[SCORE_SIZING_ACTIVE] enabled=False OR NOT_TRIGGERED")
        print("[IN_FLIGHT_SUMMARY]")
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

    try:
        from services.execution_quality import (
            apply_execution_quality_filters,
            load_execution_quality_config,
        )

        _eq = load_execution_quality_config()
        if _eq.get("enabled", True):
            planned, plan_lines, skip_reasons, _eq_entry_stats = apply_execution_quality_filters(
                planned, plan_lines, skip_reasons, broker=broker, eq_cfg=_eq
            )
    except Exception:
        pass

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

    skipped = sum(1 for pl in plan_lines if pl.status == "skipped")
    drop_rows.extend(
        build_drop_rows_from_plan(plan_lines, planned, args.mode, session_tag, str(orders_path))
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

    print("[IN_FLIGHT_SUMMARY]")
    print(f"count={in_flight_meta.get('count', 0)}")
    print(f"symbols={in_flight_meta.get('symbols') or []}")

    print("[QUALITY_SUMMARY]")
    print(f"skipped_low_score={quality_summary.get('skipped_low_score', 0)}")
    print(f"blocked_negative={quality_summary.get('blocked_negative', 0)}")

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
