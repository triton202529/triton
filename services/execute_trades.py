# services/execute_trades.py
"""TRITON execution engine — plan from trade_opportunities.csv, optional handoff to place_live_orders."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
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
from services.sector_exposure import (
    get_sector,
    load_exposure_for_planning,
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
        "positions_snapshot_path": None,
    }
    try:
        if CONFIG_PATH.is_file():
            u = json.loads(CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
            if isinstance(u, dict):
                cfg.update(u)
    except Exception:
        pass
    return cfg


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


def build_execution_plan(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    positions: Dict[str, PositionSnapshot],
    acct: AccountSnapshot,
    broker: Any,
    mode: str,
) -> Tuple[List[PlannedOrder], List[PlanLine], Dict[str, int], List[str]]:
    skip_reasons: Dict[str, int] = {}
    warnings: List[str] = []
    planned: List[PlannedOrder] = []
    lines: List[PlanLine] = []

    def bump(code: str) -> None:
        skip_reasons[code] = skip_reasons.get(code, 0) + 1

    allow_new_risk, allow_trades_cpm, exp_mult, cap_pos_w, cap_gross = _risk_cpm_flags()
    cap_pos_w = min(cap_pos_w, float(cfg.get("max_position_weight_hard_cap", 0.10)))

    pending_sector_add: Dict[str, float] = defaultdict(float)
    pending_total_add = 0.0
    sector_exp: Dict[str, Any] = {}
    if bool(cfg.get("sector_exposure_enabled", True)):
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

    new_pos_count = 0
    add_count = 0
    gross_notional = 0.0
    seen_syms: set = set()
    max_n_order = max_order_notional_usd(cfg)

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

            if bool(cfg.get("sector_exposure_enabled", True)) and sector_exp:
                try:
                    blocked, _det = should_block_buy_for_sector(
                        sym,
                        notional,
                        sector_exp,
                        pending_sector_add=pending_sector_add,
                        pending_total_add=pending_total_add,
                        block_pct=float(cfg.get("sector_block_pct", 0.40)),
                    )
                    if blocked:
                        lines.append(
                            PlanLine(
                                sym,
                                "skip",
                                "skipped",
                                "BUY_BLOCKED_BY_SECTOR",
                                stance,
                                None,
                                ot,
                                conf,
                                d_pct,
                            )
                        )
                        bump("BUY_BLOCKED_BY_SECTOR")
                        continue
                except Exception:
                    pass

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
            )
            planned.append(po)
            gross_notional += notional
            new_pos_count += 1
            pending_sector_add[get_sector(sym)] += float(notional)
            pending_total_add += float(notional)
            lines.append(PlanLine(sym, "plan", "planned", "", stance, po, ot, conf, d_pct))

        elif stance == "ADD":
            if not is_long_broker:
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

            if bool(cfg.get("sector_exposure_enabled", True)) and sector_exp:
                try:
                    blocked, _det = should_block_buy_for_sector(
                        sym,
                        notional,
                        sector_exp,
                        pending_sector_add=pending_sector_add,
                        pending_total_add=pending_total_add,
                        block_pct=float(cfg.get("sector_block_pct", 0.40)),
                    )
                    if blocked:
                        lines.append(
                            PlanLine(
                                sym,
                                "skip",
                                "skipped",
                                "ADD_BLOCKED_BY_SECTOR",
                                stance,
                                None,
                                ot,
                                conf,
                                d_pct,
                            )
                        )
                        bump("ADD_BLOCKED_BY_SECTOR")
                        continue
                except Exception:
                    pass

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

    return planned, lines, skip_reasons, warnings


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


def apply_in_flight_prefilter(
    planned: List[PlannedOrder],
    plan_lines: List[PlanLine],
    skip_reasons: Dict[str, int],
    broker: Any,
) -> Tuple[List[PlannedOrder], List[PlanLine], Dict[str, int]]:
    """
    Drop planned orders when an open broker/snapshot order already matches (symbol, side).
    Marks plan_lines as skipped with IN_FLIGHT_ALREADY_SATISFIED (order_discipline remains final net).
    """
    try:
        from services.order_discipline import load_open_same_side_keys
    except Exception:
        return planned, plan_lines, skip_reasons

    snap = RESULTS / "open_orders_snapshot.csv"
    broker_keys = _broker_open_same_side_keys(broker)
    open_keys = load_open_same_side_keys(snap, broker_open_keys=broker_keys)
    if not open_keys:
        return planned, plan_lines, skip_reasons

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
            pl.action = "skip"
            pl.status = "skipped"
            pl.skip_reason = "IN_FLIGHT_ALREADY_SATISFIED"
            pl.planned = None
            bump("IN_FLIGHT_ALREADY_SATISFIED")

    rebuilt: List[PlannedOrder] = []
    for pl in plan_lines:
        if pl.status == "planned" and pl.planned is not None:
            rebuilt.append(pl.planned)
    return rebuilt, plan_lines, skip_reasons


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
                    _emit_execution_drop_payload(drop_rows, mode, blocked=True)
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
            _emit_execution_drop_payload(drop_rows, mode, blocked=True)
        return 2, 0, guard_bad

    run_id = uuid.uuid4().hex[:12]
    if drop_rows is not None:
        _emit_execution_drop_payload(drop_rows, mode, blocked=False, run_id=run_id)

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
            _emit_execution_drop_payload(drop_rows, mode, blocked=False)
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
        return 0

    planned, plan_lines, skip_reasons, warnings = build_execution_plan(
        df, cfg, positions, acct, broker, args.mode
    )

    planned, plan_lines, skip_reasons = apply_in_flight_prefilter(
        planned, plan_lines, skip_reasons, broker
    )

    try:
        from services.execution_quality import (
            apply_execution_quality_filters,
            load_execution_quality_config,
        )

        _eq = load_execution_quality_config()
        if _eq.get("enabled", True):
            planned, plan_lines, skip_reasons = apply_execution_quality_filters(
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

    if dry_run:
        _emit_execution_drop_payload(drop_rows, args.mode, blocked=False)
        return 0

    if not planned:
        _emit_execution_drop_payload(drop_rows, args.mode, blocked=False)
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
