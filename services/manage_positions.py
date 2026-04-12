# services/manage_positions.py
"""TRITON position management — EXIT/TRIM for existing longs; entries stay in execute_trades."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
DEFAULT_LIFECYCLE_EFFECTIVE = RESULTS / "signal_lifecycle_effective.csv"
DEFAULT_LIFECYCLE_RAW = RESULTS / "signal_lifecycle.csv"
DEFAULT_POSITIONS_SNAPSHOT = RESULTS / "positions_snapshot.csv"
DEFAULT_OPEN_ORDERS_SNAPSHOT = RESULTS / "open_orders_snapshot.csv"
DEFAULT_MANAGE_ORDERS = LIVE / "manage_orders.csv"
PLAN_JSON = RESULTS / "manage_positions_plan.json"
PLAN_CSV = RESULTS / "manage_positions_plan.csv"
LOG_CSV = RESULTS / "manage_positions_log.csv"
CONFIG_PATH = ROOT / "config" / "manage_positions.json"
EXEC_GUARD_CONFIG = ROOT / "config" / "execution_guard.json"

# No lifecycle EXIT/TRIM this cycle → synthetic ROTATE_EXIT on bottom delta_pct*confidence names
FORCED_ROTATION_CAND_REASON = "forced_rotation_weakest"
# Diagnostics: full-position EXIT/ROTATE_EXIT may proceed below min_order_notional (partial trims still gated).
FULL_EXIT_MIN_NOTIONAL_BYPASS_REASON = "full_exit_bypasses_min_notional"


def _exit_pool_sort_key(x: Dict[str, Any]) -> float:
    """Weakest-first: forced_rotation uses weak_score (= delta_pct * confidence); else profit/signal score."""
    if x.get("forced_rotation") and "weak_score" in x:
        return float(x["weak_score"])
    return float(x["score"])


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


def load_manage_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "enabled": True,
        "default_mode": "paper",
        "dry_run_default": True,
        "trim_fraction_default": 0.25,
        "trim_fraction_by_signal": {"TRIM": 0.25},
        "trim_profit_pct": 0.03,
        "weak_signal_trim_delta_pct": -0.001,
        "exit_delta_pct_confirm": -0.003,
        "stale_hold_exit_cycles": 2,
        "require_profit_for_trim": True,
        "min_order_notional": 50.0,
        "max_management_orders_per_run": 10,
        "prefer_limit_orders": True,
        "limit_price_buffer_sell_bps": 20,
        "require_master_gate_for_execute": True,
        "require_guard_validation_for_each_order": True,
        "skip_if_open_order_exists_same_side": True,
        "skip_if_pending_exit_exists": True,
        "write_manage_orders_csv": True,
        "execute_via_existing_placement_flow": True,
        "allow_market_orders_if_no_quote": False,
        "max_batch_notional": 50000.0,
        "profit_weight": 0.6,
        "signal_weight": 0.4,
        "rotation_enabled": True,
        "rotation_force_count": 2,
        "rotation_mild_exit_delta_pct": -0.001,
        "rotation_small_trim_profit_pct": 0.005,
        "rotation_weak_trim_delta_pct": -0.0005,
        "rotation_trim_fraction": 0.15,
        "rotation_max_order_notional_usd": 2500.0,
        "max_portfolio_positions": None,
        # Forced rotation when no lifecycle EXIT/TRIM (weakest by delta_pct * confidence)
        "forced_rotation_no_signal_enabled": True,
        "forced_rotation_weakest_pct": 0.25,
        "forced_rotation_min_positions": 2,
        "forced_rotation_max_orders": None,
        "forced_rotation_max_turnover_pct": 0.12,
        "forced_rotation_exit_fraction": 0.35,
        "forced_rotation_positions_snapshot": None,
        # Cap total notional of forced_rotation_weakest ROTATE_EXIT per cycle (normal EXIT unaffected)
        "max_rotation_turnover_pct": 0.20,
    }
    try:
        if CONFIG_PATH.is_file():
            u = json.loads(CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
            if isinstance(u, dict):
                cfg.update(u)
    except Exception:
        pass
    return cfg


def _load_max_positions_mgmt(cfg: Dict[str, Any]) -> int:
    o = cfg.get("max_portfolio_positions")
    if o is not None:
        try:
            return max(1, int(o))
        except Exception:
            pass
    try:
        if EXEC_GUARD_CONFIG.is_file():
            u = json.loads(EXEC_GUARD_CONFIG.read_text(encoding="utf-8", errors="replace"))
            if isinstance(u, dict) and u.get("max_positions") is not None:
                return max(1, int(u.get("max_positions", 25)))
    except Exception:
        pass
    return 25


def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def _round_price(p: float) -> float:
    if p >= 1.0:
        return round(p, 2)
    return round(p, 4)


def _limit_sell(ref: float, sell_bps: float) -> float:
    return _round_price(ref * (1.0 - sell_bps / 10000.0))


def _ref_price_sell(broker: Any, symbol: str) -> Optional[float]:
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

        return get_ref_price(broker, symbol, "sell")
    except Exception:
        return None


def resolve_lifecycle_path(override: Optional[str]) -> Path:
    if override:
        return Path(override)
    if DEFAULT_LIFECYCLE_EFFECTIVE.is_file() and DEFAULT_LIFECYCLE_EFFECTIVE.stat().st_size > 0:
        return DEFAULT_LIFECYCLE_EFFECTIVE
    return DEFAULT_LIFECYCLE_RAW


def load_lifecycle_df(path: Path) -> pd.DataFrame:
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
    df["ticker"] = df["ticker"].apply(_norm_sym)
    df = df.drop_duplicates(subset=["ticker"], keep="last")
    return df


def resolve_management_stance(row: pd.Series) -> str:
    for col in ("effective_stance", "lifecycle_action", "stance"):
        if col in row.index:
            v = str(row.get(col) or "").strip().upper()
            if v:
                return v
    sig = str(row.get("signal") or "").strip().upper()
    if sig in ("EXIT", "TRIM", "HOLD", "WAIT", "BUY", "ADD"):
        return sig
    return ""


def resolve_authoritative_stance(row: pd.Series) -> Tuple[str, str]:
    """
    Single stance for EXIT/TRIM/HOLD/ADD decisions — same precedence as resolve_management_stance.
    Returns (stance_upper, source_column_name or 'signal' or 'empty').
    """
    for col in ("effective_stance", "lifecycle_action", "stance"):
        if col in row.index:
            v = str(row.get(col) or "").strip().upper()
            if v:
                return v, col
    sig = str(row.get("signal") or "").strip().upper()
    if sig in ("EXIT", "TRIM", "HOLD", "WAIT", "BUY", "ADD"):
        return sig, "signal"
    return "", "empty"


def tri_stance(row: pd.Series) -> Tuple[str, str, str]:
    eff = str(row.get("effective_stance") or "").strip().upper()
    lc = str(row.get("lifecycle_action") or "").strip().upper()
    st = str(row.get("stance") or "").strip().upper()
    return eff, lc, st


def _buy_add_skip(stance: str) -> bool:
    """Skip management sells when the authoritative stance is an entry/add (handled by execute_trades)."""
    return str(stance or "").strip().upper() in ("BUY", "ADD")


def compute_held_stance_debug(
    lifecycle: pd.DataFrame,
    pos_map: Dict[str, float],
) -> Tuple[Dict[str, int], Dict[str, int], str, Dict[str, int]]:
    """Per held symbol: stance counts, stance column source counts, primary source, position_state counts."""
    stance_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    pos_state_counts: Dict[str, int] = {}
    if lifecycle.empty or not pos_map:
        return stance_counts, source_counts, "n/a", pos_state_counts
    try:
        lc_index = lifecycle.set_index("ticker", drop=False)
    except Exception:
        return stance_counts, source_counts, "n/a", pos_state_counts
    for sym in pos_map.keys():
        if float(pos_map.get(sym) or 0.0) <= 1e-9:
            continue
        if sym not in lc_index.index:
            continue
        row = lc_index.loc[sym]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        st, src = resolve_authoritative_stance(row)
        source_counts[src] = source_counts.get(src, 0) + 1
        if not st:
            key = "EMPTY"
        elif st in ("EXIT", "TRIM", "HOLD", "ADD", "BUY", "WAIT"):
            key = st
        else:
            key = "OTHER"
        stance_counts[key] = stance_counts.get(key, 0) + 1
        eps = ""
        for col in ("effective_position_state", "position_state"):
            if col in row.index:
                eps = str(row.get(col) or "").strip().upper()
                if eps:
                    break
        if eps:
            pos_state_counts[eps] = pos_state_counts.get(eps, 0) + 1
    if not source_counts:
        primary = "n/a"
    elif len(source_counts) == 1:
        primary = next(iter(source_counts.keys()))
    else:
        dom = max(source_counts.items(), key=lambda x: x[1])[0]
        primary = f"mixed (dominant={dom})"
    return stance_counts, source_counts, primary, pos_state_counts


def load_position_metrics(pos_path: Path, pos_map: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Per-symbol metrics from positions_snapshot (profit_pct, prices)."""
    out: Dict[str, Dict[str, Any]] = {}
    if not pos_path.is_file() or not pos_map:
        return {sym: _default_metrics(sym, pos_map[sym]) for sym in pos_map}
    try:
        df = pd.read_csv(pos_path)
        df.columns = [str(c).strip() for c in df.columns]
        tc = "ticker" if "ticker" in df.columns else "symbol"
        for _, r in df.iterrows():
            sym = _norm_sym(r.get(tc))
            if sym not in pos_map:
                continue
            pr: Optional[float] = None
            known = False
            ulpc = r.get("unrealized_plpc")
            if ulpc is not None and not (isinstance(ulpc, float) and pd.isna(ulpc)):
                try:
                    pr = float(ulpc)
                    known = True
                except Exception:
                    pr = None
            if pr is None:
                ap = _safe_float(r.get("avg_entry_price"), 0.0)
                cp = _safe_float(r.get("current_price"), 0.0)
                if ap > 0 and cp > 0:
                    pr = (cp - ap) / ap
                    known = True
            out[sym] = {
                "qty": _safe_float(r.get("qty") or r.get("qty_available"), pos_map[sym]),
                "avg_entry_price": _safe_float(r.get("avg_entry_price"), 0.0) or None,
                "current_price": _safe_float(r.get("current_price"), 0.0) or None,
                "profit_pct": pr,
                "profit_pct_known": known and pr is not None,
            }
    except Exception:
        pass
    for sym in pos_map:
        if sym not in out:
            out[sym] = _default_metrics(sym, pos_map[sym])
    return out


def _default_metrics(sym: str, qty: float) -> Dict[str, Any]:
    return {
        "qty": qty,
        "avg_entry_price": None,
        "current_price": None,
        "profit_pct": None,
        "profit_pct_known": False,
    }


def _compute_rotation_score(
    profit_pct: Optional[float],
    delta_pct: float,
    *,
    profit_weight: float,
    signal_weight: float,
) -> float:
    """Lower score = weaker (more negative profit / more negative delta)."""
    pp = float(profit_pct) if profit_pct is not None else 0.0
    return float(profit_weight) * pp + float(signal_weight) * float(delta_pct)


def _compute_weak_signal_score(delta_pct: float, confidence: float) -> float:
    """Weakest names sort ascending on this (lifecycle signal quality)."""
    return float(delta_pct) * float(confidence)


def _market_values_by_symbol(pos_path: Path, pos_map: Dict[str, float]) -> Dict[str, float]:
    """symbol -> market_value from positions_snapshot (empty if unreadable)."""
    out: Dict[str, float] = {}
    if not pos_path.is_file() or not pos_map:
        return out
    try:
        df = pd.read_csv(pos_path)
        df.columns = [str(c).strip() for c in df.columns]
        tc = "ticker" if "ticker" in df.columns else "symbol"
        for _, r in df.iterrows():
            sym = _norm_sym(r.get(tc))
            if sym not in pos_map:
                continue
            mv = _safe_float(r.get("market_value"), 0.0)
            if mv <= 0:
                mv = _safe_float(r.get("value"), 0.0)
            if mv > 0:
                out[sym] = mv
    except Exception:
        pass
    return out


def _compute_total_portfolio_value(
    ps_path: Path,
    pos_map: Dict[str, float],
    pos_metrics: Dict[str, Dict[str, Any]],
) -> float:
    """Sum of market values for held positions (same fallback as forced rotation block)."""
    mv_by = _market_values_by_symbol(ps_path, pos_map)
    pv = float(sum(mv_by.values())) if mv_by else 0.0
    if pv <= 0 and pos_map:
        for sym, q in pos_map.items():
            m = pos_metrics.get(sym) or {}
            cp = m.get("current_price")
            if cp is not None and float(cp) > 0:
                pv += float(q) * float(cp)
    return pv


def _estimate_forced_rotation_exit_notional(
    broker: Any, x: Dict[str, Any], cfg: Dict[str, Any]
) -> float:
    """Planned notional for synthetic ROTATE_EXIT (forced_rotation_weakest); matches plan qty * ref."""
    sym = str(x.get("sym") or "")
    qty_pos = float(x.get("qty_pos") or 0.0)
    ref = _ref_price_sell(broker, sym)
    if ref is None or ref <= 0 or qty_pos <= 1e-9:
        return 0.0
    rot_frac = float(cfg.get("forced_rotation_exit_fraction", 0.35))
    q = max(1, int(math.floor(qty_pos * rot_frac)))
    q = min(q, int(math.floor(qty_pos)))
    return float(q * ref)


def load_long_positions(
    mode: str,
    positions_snapshot_path: Path,
) -> Tuple[Dict[str, float], Any]:
    """symbol -> qty (>0 long only)."""
    m: Dict[str, float] = {}
    broker: Any = None
    try:
        from services.broker_alpaca import AlpacaBroker

        broker = AlpacaBroker(mode=mode)
        for p in broker.get_positions() or []:
            sym = _norm_sym(p.get("symbol"))
            qty = _safe_float(p.get("qty"), 0.0)
            side = str(p.get("side") or "long").lower()
            if sym and qty > 1e-9 and side in ("long", ""):
                m[sym] = float(qty)
    except Exception:
        broker = None

    if not m and positions_snapshot_path.is_file():
        try:
            sdf = pd.read_csv(positions_snapshot_path)
            sdf.columns = [str(c).strip() for c in sdf.columns]
            tc = "ticker" if "ticker" in sdf.columns else "symbol"
            for _, r in sdf.iterrows():
                sym = _norm_sym(r.get(tc))
                qty = _safe_float(r.get("qty") or r.get("qty_available"), 0.0)
                sd = str(r.get("side") or "long").lower()
                if sym and qty > 1e-9 and sd in ("long", ""):
                    m[sym] = float(qty)
        except Exception:
            pass

    return m, broker


def collect_symbols_with_open_sell(
    broker: Any,
    open_orders_snapshot_path: Path,
) -> Set[str]:
    out: Set[str] = set()
    if broker is not None:
        try:
            for o in broker.list_orders(status="open", nested=True, limit=500) or []:
                if str(o.get("side") or "").lower() != "sell":
                    continue
                sym = _norm_sym(o.get("symbol"))
                st = str(o.get("status") or "").lower()
                if sym and st in (
                    "open",
                    "accepted",
                    "new",
                    "pending_new",
                    "partially_filled",
                    "held",
                ):
                    out.add(sym)
        except Exception:
            pass

    if open_orders_snapshot_path.is_file():
        try:
            df = pd.read_csv(open_orders_snapshot_path)
            df.columns = [str(c).strip() for c in df.columns]
            for _, r in df.iterrows():
                if str(r.get("side") or "").lower() != "sell":
                    continue
                sym = _norm_sym(r.get("symbol"))
                st = str(r.get("status") or "").lower()
                if sym and st in (
                    "open",
                    "accepted",
                    "new",
                    "pending_new",
                    "partially_filled",
                    "held",
                    "pending_replace",
                ):
                    out.add(sym)
        except Exception:
            pass

    return out


@dataclass
class ManagementPlannedOrder:
    symbol: str
    side: str
    qty: int
    order_type: str
    time_in_force: str
    limit_price: Optional[float]
    management_action: str
    rationale: str
    confidence: float
    delta_pct: float
    current_position_qty: float
    mode: str
    generated_at: str
    source: str = "manage_positions"
    planned_notional: float = 0.0
    forced_rotation: bool = False
    discipline_allowed: bool = True
    discipline_reason: str = ""


@dataclass
class PlanRow:
    symbol: str
    action: str
    status: str
    skip_reason: str
    stance: str
    planned: Optional[ManagementPlannedOrder] = None
    lifecycle_stance: str = ""
    effective_stance: str = ""
    current_position_qty: float = 0.0
    avg_entry_price: Optional[float] = None
    current_price: Optional[float] = None
    profit_pct: Optional[float] = None
    management_action_candidate: str = ""
    final_action: str = ""
    reason_code: str = ""
    reason_detail: str = ""
    side: str = "LONG"
    score: Optional[float] = None
    rank: Optional[int] = None
    priority_bucket: str = ""
    selected_for_execution: bool = False
    forced_rotation: bool = False
    rotation_mode: str = ""


@dataclass
class ManagePlanSummary:
    timestamp: str
    mode: str
    dry_run: bool
    symbols_seen: int
    positions_seen: int
    orders_planned: int
    orders_executed: int
    orders_skipped: int
    trim_candidates: int
    exit_candidates: int
    approved_actions: int
    total_positions: int
    selected_actions: int
    profit_weight: float
    signal_weight: float
    skip_reasons: Dict[str, int]
    blocked: bool
    block_reasons: List[str]
    source_file: str
    orders_file: str


def build_management_plan(
    lifecycle: pd.DataFrame,
    pos_map: Dict[str, float],
    open_sell_syms: Set[str],
    cfg: Dict[str, Any],
    broker: Any,
    mode: str,
    pos_metrics: Dict[str, Dict[str, Any]],
    state: Dict[str, Any],
    positions_snapshot_path: Optional[Path] = None,
) -> Tuple[
    List[ManagementPlannedOrder], List[PlanRow], Dict[str, int], Dict[str, Any], int, int, bool
]:
    from services.position_management_state import get_symbol_state

    skip_reasons: Dict[str, int] = {}
    planned: List[ManagementPlannedOrder] = []
    lines: List[PlanRow] = []
    trim_candidates = 0
    exit_candidates = 0
    exit_pool: List[Dict[str, Any]] = []
    trim_pool: List[Dict[str, Any]] = []
    rotation_applied = False

    def bump(code: str) -> None:
        skip_reasons[code] = skip_reasons.get(code, 0) + 1

    max_n = int(cfg.get("max_management_orders_per_run", 10))
    min_notional = float(cfg.get("min_order_notional", 50.0))
    sell_bps = float(cfg.get("limit_price_buffer_sell_bps", 20))
    trim_default = float(cfg.get("trim_fraction_default", 0.25))
    trim_by = (
        cfg.get("trim_fraction_by_signal")
        if isinstance(cfg.get("trim_fraction_by_signal"), dict)
        else {}
    )
    prefer_limit = bool(cfg.get("prefer_limit_orders", True))
    allow_mkt = bool(cfg.get("allow_market_orders_if_no_quote", False))
    skip_open = bool(cfg.get("skip_if_open_order_exists_same_side", True)) or bool(
        cfg.get("skip_if_pending_exit_exists", True)
    )
    trim_profit_pct = float(cfg.get("trim_profit_pct", 0.03))
    weak_trim = float(cfg.get("weak_signal_trim_delta_pct", -0.001))
    exit_delta = float(cfg.get("exit_delta_pct_confirm", -0.003))
    stale_n = max(1, int(cfg.get("stale_hold_exit_cycles", 2)))
    require_profit_trim = bool(cfg.get("require_profit_for_trim", True))
    pw = float(cfg.get("profit_weight", 0.6))
    sw = float(cfg.get("signal_weight", 0.4))

    lc_index = lifecycle.set_index("ticker", drop=False) if not lifecycle.empty else None

    for sym in sorted(pos_map.keys()):
        qty_pos = float(pos_map[sym])
        met = pos_metrics.get(sym) or _default_metrics(sym, qty_pos)
        profit_pct = met.get("profit_pct")
        profit_known = bool(met.get("profit_pct_known"))
        avg_e = met.get("avg_entry_price")
        cur_p = met.get("current_price")

        if qty_pos <= 1e-9:
            lines.append(
                PlanRow(
                    sym,
                    "skip",
                    "skipped",
                    "ZERO_POSITION_QTY",
                    "",
                    reason_code="ZERO_POSITION_QTY",
                    current_position_qty=qty_pos,
                )
            )
            bump("ZERO_POSITION_QTY")
            continue

        if lc_index is None or sym not in lc_index.index:
            lines.append(
                PlanRow(
                    sym,
                    "skip",
                    "skipped",
                    "POSITION_NOT_FOUND",
                    "",
                    reason_code="POSITION_NOT_FOUND",
                    reason_detail="no lifecycle row for symbol",
                    current_position_qty=qty_pos,
                    profit_pct=profit_pct,
                    avg_entry_price=avg_e,
                    current_price=cur_p,
                )
            )
            bump("POSITION_NOT_FOUND")
            continue

        row = lc_index.loc[sym]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]

        eff, lc, st = tri_stance(row)
        mgmt_stance, _stance_src = resolve_authoritative_stance(row)
        stance_disp = mgmt_stance or resolve_management_stance(row)
        rationale = str(row.get("rationale") or "")
        conf = _safe_float(row.get("confidence"), 0.0)
        d_pct = _safe_float(row.get("delta_pct"), 0.0)
        score = _compute_rotation_score(profit_pct, d_pct, profit_weight=pw, signal_weight=sw)

        sym_state = get_symbol_state(state, sym)
        sym_state["last_seen_cycle_ts"] = _utc_iso()
        sym_state["last_effective_stance"] = mgmt_stance

        if _buy_add_skip(mgmt_stance):
            sym_state["last_management_action"] = "SKIP_BUY_ADD"
            lines.append(
                PlanRow(
                    sym,
                    "skip",
                    "skipped",
                    "BUY_ADD_HANDLED_ELSEWHERE",
                    stance_disp,
                    lifecycle_stance=st,
                    effective_stance=eff,
                    current_position_qty=qty_pos,
                    avg_entry_price=avg_e,
                    current_price=cur_p,
                    profit_pct=profit_pct,
                    reason_code="BUY_ADD_HANDLED_ELSEWHERE",
                    reason_detail="entries/adds handled by execute_trades",
                    final_action="NONE",
                    score=score,
                    priority_bucket="HOLD",
                    selected_for_execution=False,
                )
            )
            bump("BUY_ADD_HANDLED_ELSEWHERE")
            continue

        if skip_open and sym in open_sell_syms:
            sym_state["last_management_action"] = "SKIP_OPEN_SELL"
            lines.append(
                PlanRow(
                    sym,
                    "skip",
                    "skipped",
                    "OPEN_SELL_ALREADY_EXISTS",
                    stance_disp,
                    lifecycle_stance=st,
                    effective_stance=eff,
                    current_position_qty=qty_pos,
                    avg_entry_price=avg_e,
                    current_price=cur_p,
                    profit_pct=profit_pct,
                    reason_code="OPEN_SELL_ALREADY_EXISTS",
                    final_action="NONE",
                    score=score,
                    priority_bucket="HOLD",
                    selected_for_execution=False,
                )
            )
            bump("OPEN_SELL_ALREADY_EXISTS")
            continue

        exit_stance = mgmt_stance == "EXIT"
        trim_stance = mgmt_stance == "TRIM"
        exit_immediate = exit_stance or (d_pct <= exit_delta)

        c_weak = int(sym_state.get("consecutive_weak_cycles") or 0)
        stale_exit = False
        if not exit_immediate:
            if mgmt_stance in ("HOLD", "WAIT", "FLAT") and exit_delta < d_pct < 0:
                c_weak += 1
                if c_weak >= stale_n:
                    stale_exit = True
                    c_weak = 0
            elif d_pct >= 0:
                c_weak = 0
            elif d_pct <= exit_delta:
                c_weak = 0
        else:
            c_weak = 0
        sym_state["consecutive_weak_cycles"] = c_weak

        want_exit = exit_immediate or stale_exit
        want_trim = False
        trim_reason = ""

        if want_exit:
            exit_candidates += 1
            sym_state["last_management_action"] = "EXIT"
            cand_reason = "EXIT_APPROVED" if exit_immediate else "EXIT_APPROVED_STALE_WEAK_CYCLES"
            detail = (
                f"exit_stance={exit_stance} delta={d_pct:.6f} exit_delta={exit_delta} "
                f"immediate={exit_immediate} stale={stale_exit}"
            )
            exit_pool.append(
                {
                    "sym": sym,
                    "score": score,
                    "qty_pos": qty_pos,
                    "eff": eff,
                    "lc": lc,
                    "st": st,
                    "stance_disp": stance_disp,
                    "rationale": rationale,
                    "conf": conf,
                    "d_pct": d_pct,
                    "cand_reason": cand_reason,
                    "detail": detail,
                    "avg_e": avg_e,
                    "cur_p": cur_p,
                    "profit_pct": profit_pct,
                }
            )
            continue
        weak_trim_sig = mgmt_stance in ("HOLD", "WAIT") and d_pct <= weak_trim
        want_trim = trim_stance or weak_trim_sig
        if want_trim:
            trim_reason = "TRIM_STANCE" if trim_stance else "TRIM_WEAK_SIGNAL"
            if require_profit_trim:
                if not profit_known or profit_pct is None:
                    lines.append(
                        PlanRow(
                            sym,
                            "skip",
                            "skipped",
                            "MISSING_PROFIT_DATA",
                            stance_disp,
                            lifecycle_stance=st,
                            effective_stance=eff,
                            management_action_candidate="TRIM",
                            final_action="NONE",
                            reason_code="MISSING_PROFIT_DATA",
                            reason_detail="profit_pct required for trim",
                            current_position_qty=qty_pos,
                            avg_entry_price=avg_e,
                            current_price=cur_p,
                            profit_pct=profit_pct,
                            score=score,
                            priority_bucket="TRIM",
                            selected_for_execution=False,
                        )
                    )
                    bump("MISSING_PROFIT_DATA")
                    sym_state["last_management_action"] = "SKIP_TRIM"
                    continue
                if float(profit_pct) < trim_profit_pct:
                    lines.append(
                        PlanRow(
                            sym,
                            "skip",
                            "skipped",
                            "TRIM_PROFIT_THRESHOLD_NOT_MET",
                            stance_disp,
                            lifecycle_stance=st,
                            effective_stance=eff,
                            management_action_candidate="TRIM",
                            final_action="NONE",
                            reason_code="TRIM_PROFIT_THRESHOLD_NOT_MET",
                            reason_detail=f"profit_pct={profit_pct} < {trim_profit_pct}",
                            current_position_qty=qty_pos,
                            avg_entry_price=avg_e,
                            current_price=cur_p,
                            profit_pct=profit_pct,
                            score=score,
                            priority_bucket="TRIM",
                            selected_for_execution=False,
                        )
                    )
                    bump("TRIM_PROFIT_THRESHOLD_NOT_MET")
                    sym_state["last_management_action"] = "SKIP_TRIM"
                    continue
            trim_candidates += 1
            sym_state["last_management_action"] = "TRIM"
            cand_reason = "TRIM_APPROVED"
            detail = trim_reason
            trim_pool.append(
                {
                    "sym": sym,
                    "score": score,
                    "qty_pos": qty_pos,
                    "eff": eff,
                    "lc": lc,
                    "st": st,
                    "stance_disp": stance_disp,
                    "rationale": rationale,
                    "conf": conf,
                    "d_pct": d_pct,
                    "cand_reason": cand_reason,
                    "detail": detail,
                    "avg_e": avg_e,
                    "cur_p": cur_p,
                    "profit_pct": profit_pct,
                }
            )
            continue

        lines.append(
            PlanRow(
                sym,
                "skip",
                "skipped",
                "NO_ACTION_HOLD",
                stance_disp,
                lifecycle_stance=st,
                effective_stance=eff,
                management_action_candidate="HOLD",
                final_action="NONE",
                reason_code="NO_ACTION_HOLD",
                reason_detail="no exit/trim rule matched",
                current_position_qty=qty_pos,
                avg_entry_price=avg_e,
                current_price=cur_p,
                profit_pct=profit_pct,
                score=score,
                priority_bucket="HOLD",
                selected_for_execution=False,
            )
        )
        bump("NO_ACTION_HOLD")
        sym_state["last_management_action"] = "HOLD"

    pos_count = len(pos_map)
    max_pos = _load_max_positions_mgmt(cfg)
    ps_path = (
        Path(positions_snapshot_path) if positions_snapshot_path else DEFAULT_POSITIONS_SNAPSHOT
    )
    psp = cfg.get("forced_rotation_positions_snapshot")
    if psp:
        ps_path = Path(str(psp))

    # Forced rotation when lifecycle produced no EXIT and no TRIM:
    # rank LONGs by weak_score = delta_pct * confidence (ascending = weakest), take bottom ~20–30%,
    # plan ROTATE_EXIT (partial by forced_rotation_exit_fraction), cap turnover via max_turnover_pct.
    if (
        bool(cfg.get("forced_rotation_no_signal_enabled", True))
        and len(exit_pool) == 0
        and len(trim_pool) == 0
        and lc_index is not None
    ):
        mv_by = _market_values_by_symbol(ps_path, pos_map)
        portfolio_gv = float(sum(mv_by.values())) if mv_by else 0.0
        if portfolio_gv <= 0 and pos_map:
            for sym, q in pos_map.items():
                m = pos_metrics.get(sym) or {}
                cp = m.get("current_price")
                if cp is not None and float(cp) > 0:
                    portfolio_gv += float(q) * float(cp)
        fr_cands: List[Dict[str, Any]] = []
        for sym in sorted(pos_map.keys()):
            qty_pos = float(pos_map[sym])
            if qty_pos <= 1e-9:
                continue
            if sym not in lc_index.index:
                continue
            row = lc_index.loc[sym]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            rot_mgmt_stance, _ = resolve_authoritative_stance(row)
            if _buy_add_skip(rot_mgmt_stance):
                continue
            if skip_open and sym in open_sell_syms:
                continue
            conf = _safe_float(row.get("confidence"), 0.0)
            d_pct = _safe_float(row.get("delta_pct"), 0.0)
            weak_score = _compute_weak_signal_score(d_pct, conf)
            eff, lc, st = tri_stance(row)
            stance_disp = rot_mgmt_stance or resolve_management_stance(row)
            rationale = str(row.get("rationale") or "")
            met = pos_metrics.get(sym) or _default_metrics(sym, qty_pos)
            profit_pct = met.get("profit_pct")
            avg_e = met.get("avg_entry_price")
            cur_p = met.get("current_price")
            sc = _compute_rotation_score(profit_pct, d_pct, profit_weight=pw, signal_weight=sw)
            fr_cands.append(
                {
                    "sym": sym,
                    "score": float(sc),
                    "weak_score": float(weak_score),
                    "qty_pos": qty_pos,
                    "eff": eff,
                    "lc": lc,
                    "st": st,
                    "stance_disp": stance_disp,
                    "rationale": rationale,
                    "conf": conf,
                    "d_pct": d_pct,
                    "avg_e": avg_e,
                    "cur_p": cur_p,
                    "profit_pct": profit_pct,
                }
            )
        n = len(fr_cands)
        min_pos = max(1, int(cfg.get("forced_rotation_min_positions", 2)))
        if n >= min_pos:
            pct_w = float(cfg.get("forced_rotation_weakest_pct", 0.25))
            pct_w = max(0.20, min(0.30, pct_w))
            k = max(1, int(math.ceil(n * pct_w)))
            fm = cfg.get("forced_rotation_max_orders")
            if fm is not None:
                try:
                    k = min(k, max(1, int(fm)))
                except Exception:
                    k = min(k, max_n)
            else:
                k = min(k, max_n)
            fr_cands.sort(key=lambda x: float(x["weak_score"]))
            take = fr_cands[:k]
            max_turn_pct = float(cfg.get("forced_rotation_max_turnover_pct", 0.12))
            max_turn_usd = portfolio_gv * max_turn_pct if portfolio_gv > 0 else float("inf")
            frac = float(cfg.get("forced_rotation_exit_fraction", 0.35))
            filtered_take: List[Dict[str, Any]] = []
            cum_turn = 0.0
            for tc in take:
                sym = tc["sym"]
                mv = float(mv_by.get(sym, 0.0))
                if mv <= 0:
                    m = pos_metrics.get(sym) or {}
                    cp = _safe_float(m.get("current_price"), 0.0)
                    mv = float(tc["qty_pos"]) * cp if cp > 0 else 0.0
                est_sell = mv * frac if mv > 0 else 0.0
                if portfolio_gv > 0 and cum_turn + est_sell > max_turn_usd + 1e-6:
                    continue
                filtered_take.append(tc)
                cum_turn += est_sell
            take = filtered_take
            if take:
                syms = [x["sym"] for x in take]
                print(f"[ROTATION] Forcing exit on weakest positions: {', '.join(syms)}")
                rotation_applied = True
                exit_candidates += len(take)
                for tc in take:
                    sym = tc["sym"]
                    sym_state = get_symbol_state(state, sym)
                    sym_state["last_management_action"] = "ROTATE_EXIT"
                    detail = (
                        f"decision_reason={FORCED_ROTATION_CAND_REASON} "
                        f"weak_score={float(tc['weak_score']):.6f} "
                        f"delta_pct={float(tc['d_pct']):.6f} conf={float(tc['conf']):.4f}"
                    )
                    exit_pool.append(
                        {
                            "sym": sym,
                            "score": float(tc["score"]),
                            "qty_pos": tc["qty_pos"],
                            "eff": tc["eff"],
                            "lc": tc["lc"],
                            "st": tc["st"],
                            "stance_disp": tc["stance_disp"],
                            "rationale": tc["rationale"],
                            "conf": float(tc["conf"]),
                            "d_pct": float(tc["d_pct"]),
                            "cand_reason": FORCED_ROTATION_CAND_REASON,
                            "detail": detail,
                            "avg_e": tc["avg_e"],
                            "cur_p": tc["cur_p"],
                            "profit_pct": tc["profit_pct"],
                            "forced_rotation": True,
                            "weak_score": float(tc["weak_score"]),
                        }
                    )

    stall = (
        bool(cfg.get("rotation_enabled", True))
        and pos_count >= max_pos
        and len(exit_pool) == 0
        and len(trim_pool) == 0
    )
    if stall and lc_index is not None:
        mild = float(cfg.get("rotation_mild_exit_delta_pct", -0.001))
        small_p = float(cfg.get("rotation_small_trim_profit_pct", 0.005))
        weak_tr = float(cfg.get("rotation_weak_trim_delta_pct", -0.0005))
        n_force = min(3, max(1, int(cfg.get("rotation_force_count", 2))))

        pool_sym = {x["sym"] for x in exit_pool} | {x["sym"] for x in trim_pool}
        cands: List[Dict[str, Any]] = []
        for sym in sorted(pos_map.keys()):
            if sym in pool_sym:
                continue
            qty_pos = float(pos_map[sym])
            if qty_pos <= 1e-9:
                continue
            if sym not in lc_index.index:
                continue
            row = lc_index.loc[sym]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            eff, lc, st = tri_stance(row)
            rot_mgmt_stance, _ = resolve_authoritative_stance(row)
            if _buy_add_skip(rot_mgmt_stance):
                continue
            if skip_open and sym in open_sell_syms:
                continue
            met = pos_metrics.get(sym) or _default_metrics(sym, qty_pos)
            profit_pct = met.get("profit_pct")
            profit_known = bool(met.get("profit_pct_known"))
            avg_e = met.get("avg_entry_price")
            cur_p = met.get("current_price")
            stance_disp = rot_mgmt_stance or resolve_management_stance(row)
            rationale = str(row.get("rationale") or "")
            conf = _safe_float(row.get("confidence"), 0.0)
            d_pct = _safe_float(row.get("delta_pct"), 0.0)
            score = _compute_rotation_score(profit_pct, d_pct, profit_weight=pw, signal_weight=sw)
            cands.append(
                {
                    "sym": sym,
                    "score": score,
                    "qty_pos": qty_pos,
                    "eff": eff,
                    "lc": lc,
                    "st": st,
                    "stance_disp": stance_disp,
                    "rationale": rationale,
                    "conf": conf,
                    "d_pct": d_pct,
                    "avg_e": avg_e,
                    "cur_p": cur_p,
                    "profit_pct": profit_pct,
                    "profit_known": profit_known,
                }
            )
        cands.sort(key=lambda x: float(x["score"]))
        take = cands[:n_force]
        for tc in take:
            sym = tc["sym"]
            sym_state = get_symbol_state(state, sym)
            d_pct = float(tc["d_pct"])
            profit_pct = tc["profit_pct"]
            profit_known = tc["profit_known"]
            detail_rot = (
                f"rotation_pressure=1 mild={mild} small_profit={small_p} weak_trim={weak_tr} "
                f"delta={d_pct:.6f}"
            )
            base = {
                "sym": tc["sym"],
                "score": tc["score"],
                "qty_pos": tc["qty_pos"],
                "eff": tc["eff"],
                "lc": tc["lc"],
                "st": tc["st"],
                "stance_disp": tc["stance_disp"],
                "rationale": tc["rationale"],
                "conf": tc["conf"],
                "d_pct": tc["d_pct"],
                "avg_e": tc["avg_e"],
                "cur_p": tc["cur_p"],
                "profit_pct": tc["profit_pct"],
                "forced_rotation": True,
            }
            if d_pct <= mild:
                exit_candidates += 1
                sym_state["last_management_action"] = "EXIT"
                exit_pool.append(
                    {
                        **base,
                        "cand_reason": "EXIT_ROTATION_PRESSURE",
                        "detail": detail_rot + " path=mild_exit",
                    }
                )
            elif (profit_known and profit_pct is not None and float(profit_pct) >= small_p) or (
                d_pct <= weak_tr
            ):
                trim_candidates += 1
                sym_state["last_management_action"] = "TRIM"
                trim_pool.append(
                    {
                        **base,
                        "cand_reason": "TRIM_ROTATION_PRESSURE",
                        "detail": detail_rot + " path=profit_or_weak_signal",
                    }
                )
            else:
                trim_candidates += 1
                sym_state["last_management_action"] = "TRIM"
                trim_pool.append(
                    {
                        **base,
                        "cand_reason": "TRIM_ROTATION_PRESSURE",
                        "detail": detail_rot + " path=default_small_trim",
                    }
                )
        if take:
            rotation_applied = True
            print("[rotation] ACTIVE — forcing rotation on weakest positions")

    exit_sorted = sorted(exit_pool, key=_exit_pool_sort_key)
    trim_sorted = sorted(trim_pool, key=lambda x: float(x["score"]))
    rn = 1
    for x in exit_sorted:
        x["rank"] = rn
        rn += 1
    for x in trim_sorted:
        x["rank"] = rn
        rn += 1

    total_portfolio_value = _compute_total_portfolio_value(ps_path, pos_map, pos_metrics)
    max_rot_pct = float(cfg.get("max_rotation_turnover_pct", 0.20))
    max_rotation_notional_cap = (
        total_portfolio_value * max_rot_pct if total_portfolio_value > 0 else float("inf")
    )
    normal_exit_list = [
        x for x in exit_sorted if x.get("cand_reason") != FORCED_ROTATION_CAND_REASON
    ]
    forced_rotation_list = [
        x for x in exit_sorted if x.get("cand_reason") == FORCED_ROTATION_CAND_REASON
    ]

    selected_exit: List[Dict[str, Any]] = []
    for x in normal_exit_list:
        if len(selected_exit) >= max_n:
            break
        selected_exit.append(x)

    slots_left = max_n - len(selected_exit)
    rotation_turnover_capped = False
    cap_dropped_syms: Set[str] = set()
    cum_rot = 0.0
    for i, x in enumerate(forced_rotation_list):
        if slots_left <= 0:
            break
        est = _estimate_forced_rotation_exit_notional(broker, x, cfg)
        if cum_rot + est <= max_rotation_notional_cap + 1e-6:
            selected_exit.append(x)
            cum_rot += est
            slots_left -= 1
        else:
            rotation_turnover_capped = True
            cap_dropped_syms.update(y["sym"] for y in forced_rotation_list[i:])
            break

    planned_rotation_notional = cum_rot
    if rotation_turnover_capped:
        pct_disp = max_rot_pct * 100.0
        print(f"[ROTATION_CAP] limiting exits to {pct_disp:.1f}% of portfolio")

    n_ex = len(selected_exit)
    remaining = max(0, max_n - n_ex)
    selected_trim = trim_sorted[:remaining]
    exit_sym_sel = {c["sym"] for c in selected_exit}

    for x in exit_sorted:
        if x["sym"] in exit_sym_sel:
            continue
        sym = x["sym"]
        eff, lc, st = x["eff"], x["lc"], x["st"]
        mac = (
            "ROTATE_EXIT"
            if str(x.get("cand_reason") or "") == FORCED_ROTATION_CAND_REASON
            else "EXIT"
        )
        cap_drop = sym in cap_dropped_syms
        skip_r = "rotation_turnover_capped" if cap_drop else "NOT_SELECTED_THIS_RUN"
        if cap_drop:
            bump("rotation_turnover_capped")
        lines.append(
            PlanRow(
                sym,
                "skip",
                "skipped",
                skip_r,
                x["stance_disp"],
                lifecycle_stance=st,
                effective_stance=eff,
                current_position_qty=x["qty_pos"],
                avg_entry_price=x["avg_e"],
                current_price=x["cur_p"],
                profit_pct=x["profit_pct"],
                management_action_candidate=mac,
                final_action="NONE",
                reason_code="rotation_turnover_capped" if cap_drop else "NOT_SELECTED_THIS_RUN",
                reason_detail=(
                    "rotation_turnover_capped"
                    if cap_drop
                    else "lower priority vs other EXIT/TRIM candidates this run"
                ),
                score=float(x["score"]),
                rank=int(x["rank"]),
                priority_bucket="EXIT",
                selected_for_execution=False,
                forced_rotation=bool(x.get("forced_rotation", False)),
            )
        )
        bump("NOT_SELECTED_THIS_RUN")

    for x in trim_sorted[remaining:]:
        sym = x["sym"]
        eff, lc, st = x["eff"], x["lc"], x["st"]
        lines.append(
            PlanRow(
                sym,
                "skip",
                "skipped",
                "NOT_SELECTED_THIS_RUN",
                x["stance_disp"],
                lifecycle_stance=st,
                effective_stance=eff,
                current_position_qty=x["qty_pos"],
                avg_entry_price=x["avg_e"],
                current_price=x["cur_p"],
                profit_pct=x["profit_pct"],
                management_action_candidate="TRIM",
                final_action="NONE",
                reason_code="NOT_SELECTED_THIS_RUN",
                reason_detail="lower priority vs other EXIT/TRIM candidates this run",
                score=float(x["score"]),
                rank=int(x["rank"]),
                priority_bucket="TRIM",
                selected_for_execution=False,
                forced_rotation=bool(x.get("forced_rotation", False)),
            )
        )
        bump("NOT_SELECTED_THIS_RUN")

    for cand in selected_exit + selected_trim:
        sym = cand["sym"]
        qty_pos = float(cand["qty_pos"])
        eff, lc, st = cand["eff"], cand["lc"], cand["st"]
        stance_disp = cand["stance_disp"]
        rationale = cand["rationale"]
        conf = float(cand["conf"])
        d_pct = float(cand["d_pct"])
        cand_reason = cand["cand_reason"]
        detail = cand["detail"]
        avg_e = cand["avg_e"]
        cur_p = cand["cur_p"]
        profit_pct = cand["profit_pct"]
        if cand_reason == FORCED_ROTATION_CAND_REASON:
            mgmt_action = "ROTATE_EXIT"
            bucket = "EXIT"
        else:
            mgmt_action = "EXIT" if sym in exit_sym_sel else "TRIM"
            bucket = "EXIT" if mgmt_action == "EXIT" else "TRIM"
        sc = float(cand["score"])
        rk = int(cand["rank"])
        fr = bool(cand.get("forced_rotation", False))
        rot_tf = float(cfg.get("rotation_trim_fraction", 0.15))
        cap_usd = float(cfg.get("rotation_max_order_notional_usd", 2500.0))

        if mgmt_action == "TRIM":
            try:
                from services.execution_quality import (
                    load_execution_quality_config,
                    trim_allowed_after_cooldown,
                )

                _eqx = load_execution_quality_config()
                if _eqx.get("enabled", True):
                    st_sym = get_symbol_state(state, sym)
                    ok_t, tr_sk = trim_allowed_after_cooldown(
                        sym,
                        st_sym,
                        float(_eqx.get("trim_min_interval_minutes", 60) or 0),
                    )
                    if not ok_t:
                        lines.append(
                            PlanRow(
                                sym,
                                "skip",
                                "skipped",
                                tr_sk or "TRIM_COOLDOWN",
                                stance_disp,
                                lifecycle_stance=st,
                                effective_stance=eff,
                                current_position_qty=qty_pos,
                                avg_entry_price=avg_e,
                                current_price=cur_p,
                                profit_pct=profit_pct,
                                management_action_candidate="TRIM",
                                final_action="NONE",
                                reason_code=tr_sk or "TRIM_COOLDOWN",
                                reason_detail="trim spacing (execution_quality)",
                                score=sc,
                                rank=rk,
                                priority_bucket="TRIM",
                                selected_for_execution=False,
                                forced_rotation=fr,
                            )
                        )
                        bump(tr_sk or "TRIM_COOLDOWN")
                        continue
            except Exception:
                pass

        ref = _ref_price_sell(broker, sym)
        lim: Optional[float] = None
        ot = "limit"
        if ref is not None and ref > 0 and prefer_limit:
            lim = _limit_sell(ref, sell_bps)
        elif allow_mkt and ref is None:
            ot = "market"
        else:
            lines.append(
                PlanRow(
                    sym,
                    "skip",
                    "skipped",
                    "NO_PRICE_AVAILABLE",
                    stance_disp,
                    lifecycle_stance=st,
                    effective_stance=eff,
                    current_position_qty=qty_pos,
                    avg_entry_price=avg_e,
                    current_price=cur_p,
                    profit_pct=profit_pct,
                    management_action_candidate=mgmt_action,
                    reason_code="NO_PRICE_AVAILABLE",
                    score=sc,
                    rank=rk,
                    priority_bucket=bucket,
                    selected_for_execution=False,
                    forced_rotation=fr,
                )
            )
            bump("NO_PRICE_AVAILABLE")
            continue

        q = 0
        if mgmt_action == "ROTATE_EXIT":
            rot_frac = float(cfg.get("forced_rotation_exit_fraction", 0.35))
            q = max(1, int(math.floor(qty_pos * rot_frac)))
            q = min(q, int(math.floor(qty_pos)))
        elif mgmt_action == "EXIT":
            q = int(math.floor(qty_pos))
            if fr and ref is not None and ref > 0 and cand_reason != FORCED_ROTATION_CAND_REASON:
                cap_q = max(1, int(math.floor(cap_usd / ref)))
                q = min(q, cap_q)
        else:
            tf = rot_tf if fr else float(trim_by.get("TRIM", trim_default))
            q = max(1, int(math.floor(qty_pos * tf)))
            q = min(q, int(math.floor(qty_pos)))

        if q < 1:
            lines.append(
                PlanRow(
                    sym,
                    "skip",
                    "skipped",
                    "ZERO_QTY_AFTER_ROUNDING",
                    stance_disp,
                    lifecycle_stance=st,
                    effective_stance=eff,
                    current_position_qty=qty_pos,
                    avg_entry_price=avg_e,
                    current_price=cur_p,
                    profit_pct=profit_pct,
                    management_action_candidate=mgmt_action,
                    reason_code="ZERO_QTY_AFTER_ROUNDING",
                    score=sc,
                    rank=rk,
                    priority_bucket=bucket,
                    selected_for_execution=False,
                    forced_rotation=fr,
                )
            )
            bump("ZERO_QTY_AFTER_ROUNDING")
            continue

        notional = q * (lim if lim is not None else (ref or 0.0))
        qty_full = int(math.floor(qty_pos))
        full_position_exit = (
            mgmt_action in ("EXIT", "ROTATE_EXIT") and qty_full >= 1 and q == qty_full
        )
        min_notional_bypass = False
        if lim is not None and notional < min_notional:
            if full_position_exit:
                min_notional_bypass = True
            else:
                lines.append(
                    PlanRow(
                        sym,
                        "skip",
                        "skipped",
                        "BELOW_MIN_ORDER_NOTIONAL",
                        stance_disp,
                        lifecycle_stance=st,
                        effective_stance=eff,
                        current_position_qty=qty_pos,
                        avg_entry_price=avg_e,
                        current_price=cur_p,
                        profit_pct=profit_pct,
                        management_action_candidate=mgmt_action,
                        reason_code="BELOW_MIN_ORDER_NOTIONAL",
                        score=sc,
                        rank=rk,
                        priority_bucket=bucket,
                        selected_for_execution=False,
                        forced_rotation=fr,
                    )
                )
                bump("BELOW_MIN_ORDER_NOTIONAL")
                continue

        rationale_out = str(rationale or "")
        if cand_reason == FORCED_ROTATION_CAND_REASON:
            rationale_out = (rationale_out + " | " + detail).strip(" |")

        po = ManagementPlannedOrder(
            symbol=sym,
            side="sell",
            qty=q,
            order_type=ot,
            time_in_force="day",
            limit_price=lim,
            management_action=mgmt_action,
            rationale=rationale_out,
            confidence=conf,
            delta_pct=d_pct,
            current_position_qty=qty_pos,
            mode=mode,
            generated_at=_utc_iso(),
            planned_notional=float(notional),
            forced_rotation=fr,
            discipline_reason=(FULL_EXIT_MIN_NOTIONAL_BYPASS_REASON if min_notional_bypass else ""),
        )
        planned.append(po)
        if mgmt_action == "TRIM":
            get_symbol_state(state, sym)["last_trim_ts_utc"] = _utc_iso()
        lines.append(
            PlanRow(
                sym,
                "plan",
                "planned",
                "",
                stance_disp,
                po,
                lifecycle_stance=st,
                effective_stance=eff,
                current_position_qty=qty_pos,
                avg_entry_price=avg_e,
                current_price=cur_p,
                profit_pct=profit_pct,
                management_action_candidate=mgmt_action,
                final_action=mgmt_action,
                reason_code=cand_reason,
                reason_detail=(
                    f"{detail} | {FULL_EXIT_MIN_NOTIONAL_BYPASS_REASON}"
                    if min_notional_bypass
                    else detail
                ),
                score=sc,
                rank=rk,
                priority_bucket=bucket,
                selected_for_execution=True,
                forced_rotation=fr,
            )
        )

    rm = "ACTIVE" if rotation_applied else "NORMAL"
    for pr in lines:
        pr.rotation_mode = rm

    return planned, lines, skip_reasons, state, trim_candidates, exit_candidates, rotation_applied


def _fmt_top_skip_reasons(skip_reasons: Dict[str, int], limit: int = 8) -> str:
    if not skip_reasons:
        return "none"
    items = sorted(skip_reasons.items(), key=lambda x: (-x[1], x[0]))[:limit]
    return ", ".join(f"{k}={v}" for k, v in items)


def _count_planned_actions(planned: List[ManagementPlannedOrder]) -> Tuple[int, int, int]:
    """Returns (exit_count, trim_count, forced_rotation_rotate_exit_count)."""
    exits = 0
    trims = 0
    fr = 0
    for p in planned:
        ma = str(p.management_action or "").upper()
        if ma == "EXIT":
            exits += 1
        elif ma == "TRIM":
            trims += 1
        elif ma == "ROTATE_EXIT" and bool(getattr(p, "forced_rotation", False)):
            fr += 1
    return exits, trims, fr


def _guard_codes_for_display(guard_bad: List[str]) -> str:
    seen: Set[str] = set()
    codes: List[str] = []
    for b in guard_bad:
        c = b.split(":", 1)[1] if ":" in b else b
        if c not in seen:
            seen.add(c)
            codes.append(c)
    return ", ".join(codes) if codes else "unknown"


def write_manage_artifacts(
    planned: List[ManagementPlannedOrder],
    plan_rows: List[PlanRow],
    summary: ManagePlanSummary,
    cfg: Dict[str, Any],
    orders_path: Path,
    orders_for_csv: Optional[List[ManagementPlannedOrder]] = None,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LIVE.mkdir(parents=True, exist_ok=True)

    with PLAN_JSON.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    out: List[Dict[str, Any]] = []
    for pr in plan_rows:
        d: Dict[str, Any] = {
            "symbol": pr.symbol,
            "side": pr.side,
            "action": pr.action,
            "status": pr.status,
            "skip_reason": pr.skip_reason,
            "stance": pr.stance,
            "current_position_qty": pr.current_position_qty,
            "avg_entry_price": pr.avg_entry_price if pr.avg_entry_price is not None else "",
            "current_price": pr.current_price if pr.current_price is not None else "",
            "profit_pct": pr.profit_pct if pr.profit_pct is not None else "",
            "lifecycle_stance": pr.lifecycle_stance,
            "effective_stance": pr.effective_stance,
            "management_action_candidate": pr.management_action_candidate,
            "final_action": pr.final_action,
            "reason_code": pr.reason_code,
            "reason_detail": pr.reason_detail,
            "score": pr.score if pr.score is not None else "",
            "rank": pr.rank if pr.rank is not None else "",
            "priority_bucket": pr.priority_bucket,
            "selected_for_execution": pr.selected_for_execution,
            "forced_rotation": pr.forced_rotation,
            "rotation_mode": pr.rotation_mode,
        }
        if pr.planned:
            p = pr.planned
            d.update(
                {
                    "qty": p.qty,
                    "order_side": p.side,
                    "limit_price": p.limit_price,
                    "management_action": p.management_action,
                    "planned_notional": p.planned_notional,
                    "rationale": p.rationale,
                    "discipline_allowed": getattr(p, "discipline_allowed", True),
                    "discipline_reason": getattr(p, "discipline_reason", ""),
                }
            )
        out.append(d)

    if out:
        all_keys: List[str] = []
        seen_k: Set[str] = set()
        for d in out:
            for k in d.keys():
                if k not in seen_k:
                    seen_k.add(k)
                    all_keys.append(k)
        with PLAN_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            for r in out:
                w.writerow({k: r.get(k, "") for k in all_keys})

    csv_orders = orders_for_csv if orders_for_csv is not None else planned
    if cfg.get("write_manage_orders_csv", True):
        if csv_orders:
            rows = []
            for p in csv_orders:
                lp = p.limit_price if p.limit_price is not None else ""
                _ma = str(p.management_action or "").upper()
                _opp = "EXIT" if _ma in ("EXIT", "ROTATE_EXIT") else "TRIM"
                rows.append(
                    {
                        "ticker": p.symbol,
                        "effective_stance": p.management_action,
                        "effective_position_state": "LONG",
                        "opportunity_type": _opp,
                        "confidence": p.confidence,
                        "delta_pct": p.delta_pct,
                        "rationale": p.rationale,
                        "close": lp,
                        "limit_price": lp,
                        "qty": p.qty,
                        "exploration_flag": False,
                        "discipline_allowed": getattr(p, "discipline_allowed", True),
                        "discipline_reason": getattr(p, "discipline_reason", ""),
                    }
                )
            pd.DataFrame(rows).to_csv(orders_path, index=False)
        else:
            pd.DataFrame(
                columns=[
                    "ticker",
                    "effective_stance",
                    "effective_position_state",
                    "opportunity_type",
                    "confidence",
                    "delta_pct",
                    "rationale",
                    "close",
                    "limit_price",
                    "qty",
                    "exploration_flag",
                ]
            ).to_csv(orders_path, index=False)


def append_manage_log(summary: ManagePlanSummary) -> None:
    try:
        row = {
            "ts_utc": summary.timestamp,
            "mode": summary.mode,
            "dry_run": summary.dry_run,
            "planned": summary.orders_planned,
            "trim_candidates": summary.trim_candidates,
            "exit_candidates": summary.exit_candidates,
            "skipped": summary.orders_skipped,
            "blocked": summary.blocked,
        }
        h = not LOG_CSV.is_file() or LOG_CSV.stat().st_size == 0
        with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if h:
                w.writeheader()
            w.writerow(row)
    except Exception:
        pass


def _validate_guard_mgmt(
    planned: List[ManagementPlannedOrder], broker: Any, cfg: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    if not planned or not cfg.get("require_guard_validation_for_each_order", True):
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
            "order_source": "manage_positions",
            "management_action": p.management_action,
        }
        d = guard.validate(payload)
        if not d.ok:
            bad.append(f"{p.symbol}:{d.code}")
    return (len(bad) == 0), bad


def maybe_execute_management(
    planned: List[ManagementPlannedOrder],
    mode: str,
    dry_run: bool,
    cfg: Dict[str, Any],
    broker: Any,
    verbose: bool,
    ignore_market_closed: bool,
    orders_path: Path,
    placement_session: Optional[str] = None,
) -> Tuple[int, int, List[str], str]:
    if dry_run or not planned:
        return 0, 0, [], ""

    if broker is None:
        return 1, 0, ["BROKER_UNAVAILABLE"], ""

    if not cfg.get("execute_via_existing_placement_flow", True):
        return 0, 0, [], ""

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
                rs = list(dec.reasons)
                print(f"[MANAGE_BLOCK] execution blocked by master gate: {', '.join(rs)}")
                return 2, 0, rs, ""
        except Exception as e:
            return 1, 0, [str(e)], ""

    ok_g, guard_bad = _validate_guard_mgmt(planned, broker, cfg)
    if not ok_g:
        print(f"[MANAGE_BLOCK] execution blocked by guard: {_guard_codes_for_display(guard_bad)}")
        return 2, 0, guard_bad, ""

    session_id = (
        placement_session or f"manage_pos_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
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
        session_id,
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
        rc = int(subprocess.call(cmd, cwd=str(ROOT)))
        ex = len(planned) if rc == 0 else 0
        return rc, ex, [], session_id
    except Exception as e:
        return 1, 0, [str(e)], ""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TRITON manage_positions — EXIT/TRIM for existing longs"
    )
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--orders-path", type=str, default=None)
    ap.add_argument("--lifecycle-path", type=str, default=None)
    ap.add_argument("--positions-path", type=str, default=None)
    ap.add_argument("--max-orders", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ignore-market-closed", action="store_true")
    ap.add_argument(
        "--trim-profit-pct",
        type=float,
        default=None,
        help="Override trim profit threshold (fraction, e.g. 0.03).",
    )
    ap.add_argument(
        "--exit-delta-confirm",
        type=float,
        default=None,
        help="Override exit delta_pct threshold (e.g. -0.003).",
    )
    ap.add_argument(
        "--stale-hold-exit-cycles",
        type=int,
        default=None,
        help="Consecutive weak cycles before stale exit.",
    )
    ap.add_argument(
        "--reallocate-after-exit",
        action="store_true",
        help="After successful paper manage execution: if freed capital and master gate OK, run execute_trades (paper).",
    )
    args = ap.parse_args(argv)

    cfg = load_manage_config()
    if args.max_orders is not None:
        cfg["max_management_orders_per_run"] = args.max_orders
    if args.trim_profit_pct is not None:
        cfg["trim_profit_pct"] = float(args.trim_profit_pct)
    if args.exit_delta_confirm is not None:
        cfg["exit_delta_pct_confirm"] = float(args.exit_delta_confirm)
    if args.stale_hold_exit_cycles is not None:
        cfg["stale_hold_exit_cycles"] = int(args.stale_hold_exit_cycles)

    dry_run = not bool(args.execute)
    lc_path = resolve_lifecycle_path(args.lifecycle_path)
    pos_path = Path(args.positions_path) if args.positions_path else DEFAULT_POSITIONS_SNAPSHOT
    oo_path = DEFAULT_OPEN_ORDERS_SNAPSHOT
    orders_path = Path(args.orders_path) if args.orders_path else DEFAULT_MANAGE_ORDERS

    lifecycle = load_lifecycle_df(lc_path)
    pos_map, broker = load_long_positions(args.mode, pos_path)
    open_sells = collect_symbols_with_open_sell(broker, oo_path)
    pos_metrics = load_position_metrics(pos_path, pos_map)

    from services.position_management_state import load_state, save_state

    state = load_state()

    symbols_seen = int(len(lifecycle)) if not lifecycle.empty else 0
    positions_seen = len(pos_map)

    pw = float(cfg.get("profit_weight", 0.6))
    sw = float(cfg.get("signal_weight", 0.4))

    if lifecycle.empty:
        summary = ManagePlanSummary(
            timestamp=_utc_iso(),
            mode=args.mode,
            dry_run=dry_run,
            symbols_seen=0,
            positions_seen=positions_seen,
            orders_planned=0,
            orders_executed=0,
            orders_skipped=0,
            trim_candidates=0,
            exit_candidates=0,
            approved_actions=0,
            total_positions=positions_seen,
            selected_actions=0,
            profit_weight=pw,
            signal_weight=sw,
            skip_reasons={},
            blocked=False,
            block_reasons=[],
            source_file=str(lc_path),
            orders_file=str(orders_path),
        )
        write_manage_artifacts([], [], summary, cfg, orders_path)
        append_manage_log(summary)
        print("[manage_positions] Empty or missing lifecycle; plan written.")
        return 0

    planned, plan_lines, skip_reasons, state, trim_c, exit_c, rotation_applied = (
        build_management_plan(
            lifecycle,
            pos_map,
            open_sells,
            cfg,
            broker,
            args.mode,
            pos_metrics,
            state,
            positions_snapshot_path=pos_path,
        )
    )
    try:
        save_state(state)
    except Exception:
        pass
    skipped = sum(1 for x in plan_lines if x.status == "skipped")
    planned_after_build = list(planned)

    if not lifecycle.empty and pos_map:
        sc_stance, sc_src, primary, sc_pos = compute_held_stance_debug(lifecycle, pos_map)
        et_counts = {k: sc_stance.get(k, 0) for k in ("EXIT", "TRIM", "HOLD", "ADD")}
        print(
            f"[manage_positions] authoritative_stance_source={primary} "
            f"held_counts_EXIT_TRIM_HOLD_ADD={et_counts}"
        )
        if args.verbose:
            print(
                f"[manage_positions] stance_column_hits={sc_src} "
                f"effective_position_state_counts={sc_pos} full_stance_counts={sc_stance}"
            )

    mgmt_place_session = f"manage_pos_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if planned and not dry_run:
        try:
            from services.order_discipline import apply_discipline_to_planned_generic

            planned, _disc_meta = apply_discipline_to_planned_generic(
                planned,
                session=mgmt_place_session,
                source_module="manage_positions",
                mode=args.mode,
                context=None,
            )
            nb = int(_disc_meta.get("orders_blocked") or 0)
            if nb:
                skip_reasons["ORDER_DISCIPLINE"] = skip_reasons.get("ORDER_DISCIPLINE", 0) + nb
        except Exception:
            pass

    fr_selected = sum(
        1
        for p in planned_after_build
        if str(p.management_action or "").upper() == "ROTATE_EXIT"
        and bool(getattr(p, "forced_rotation", False))
    )
    fr_after = sum(
        1
        for p in planned
        if str(p.management_action or "").upper() == "ROTATE_EXIT"
        and bool(getattr(p, "forced_rotation", False))
    )
    if rotation_applied and fr_selected > 0 and fr_after < fr_selected:
        _rot_parts: List[str] = []
        od = int(skip_reasons.get("ORDER_DISCIPLINE") or 0)
        if od:
            _rot_parts.append(f"ORDER_DISCIPLINE={od}")
        _rstr = ",".join(_rot_parts) if _rot_parts else "none"
        print(
            f"[ROTATION_RESULT] selected={fr_selected} executable={fr_after} "
            f"blocked={fr_selected - fr_after} reasons={_rstr}"
        )

    summary = ManagePlanSummary(
        timestamp=_utc_iso(),
        mode=args.mode,
        dry_run=dry_run,
        symbols_seen=symbols_seen,
        positions_seen=positions_seen,
        orders_planned=len(planned),
        orders_executed=0,
        orders_skipped=skipped,
        trim_candidates=trim_c,
        exit_candidates=exit_c,
        approved_actions=len(planned),
        total_positions=positions_seen,
        selected_actions=len(planned),
        profit_weight=pw,
        signal_weight=sw,
        skip_reasons=skip_reasons,
        blocked=False,
        block_reasons=[],
        source_file=str(lc_path),
        orders_file=str(orders_path),
    )

    csv_planned = planned if planned else planned_after_build
    write_manage_artifacts(
        planned, plan_lines, summary, cfg, orders_path, orders_for_csv=csv_planned
    )
    append_manage_log(summary)

    print(
        f"[manage_positions] mode={args.mode} dry_run={dry_run} "
        f"lifecycle_rows={symbols_seen} positions={positions_seen} planned={len(planned)} skipped={skipped}"
    )
    if skip_reasons:
        print(f"[manage_positions] skip_reasons={skip_reasons}")

    if args.verbose:
        rot = [
            x
            for x in plan_lines
            if x.priority_bucket in ("EXIT", "TRIM") and x.score is not None and x.rank is not None
        ]
        rot.sort(key=lambda x: (int(x.rank or 0), x.symbol))
        if rot:
            print("[manage_positions] rotation_rank (weakest first; scored EXIT/TRIM pool):")
            for r in rot:
                print(
                    f"  rank={r.rank} score={r.score:.6f} bucket={r.priority_bucket} "
                    f"selected={r.selected_for_execution} symbol={r.symbol}"
                )

    placement_rc = 0
    mgmt_session = ""
    if dry_run:
        try:
            from services.capital_reallocation import run_reallocation_pipeline

            run_reallocation_pipeline(
                planned,
                mode=args.mode,
                manage_executed=False,
                manage_session="",
                placement_rc=0,
                broker=broker,
                verbose=args.verbose,
                reallocate_after_exit=False,
                ignore_market_closed=args.ignore_market_closed,
            )
        except Exception:
            pass
        ex_n, tr_n, fr_n = _count_planned_actions(planned)
        print(
            f"[MANAGE_SUMMARY] exits={ex_n} trims={tr_n} forced_rotation={fr_n} planned={len(planned)} "
            f"executed=0 skipped={skipped} blocked=0 csv_rows={len(csv_planned)}"
        )
        print(f"[MANAGE_SUMMARY] top_skip_reasons: {_fmt_top_skip_reasons(skip_reasons)}")
        return 0

    rc, executed, brs, mgmt_session = maybe_execute_management(
        planned,
        args.mode,
        False,
        cfg,
        broker,
        args.verbose,
        args.ignore_market_closed,
        orders_path,
        placement_session=mgmt_place_session if planned else None,
    )
    placement_rc = rc
    summary.orders_executed = executed
    summary.blocked = rc == 2
    summary.block_reasons = brs
    try:
        with PLAN_JSON.open("w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)
    except Exception:
        pass

    try:
        from services.capital_reallocation import run_reallocation_pipeline

        run_reallocation_pipeline(
            planned,
            mode=args.mode,
            manage_executed=True,
            manage_session=mgmt_session,
            placement_rc=placement_rc,
            broker=broker,
            verbose=args.verbose,
            reallocate_after_exit=bool(args.reallocate_after_exit),
            ignore_market_closed=args.ignore_market_closed,
        )
    except Exception as e:
        if args.verbose:
            print(f"[manage_positions] capital_reallocation skipped: {e}")

    ex_n, tr_n, fr_n = _count_planned_actions(planned)
    blocked_n = len(brs) if rc == 2 else 0
    print(
        f"[MANAGE_SUMMARY] exits={ex_n} trims={tr_n} forced_rotation={fr_n} planned={len(planned)} "
        f"executed={executed} skipped={skipped} blocked={blocked_n} csv_rows={len(csv_planned)}"
    )
    print(f"[MANAGE_SUMMARY] top_skip_reasons: {_fmt_top_skip_reasons(skip_reasons)}")

    if rc == 2:
        print(f"[manage_positions] BLOCKED: {brs}")
        return 2
    if rc != 0:
        print(f"[manage_positions] placement rc={rc}")
        return rc if rc in (1, 2, 3) else 1
    print("[manage_positions] placement subprocess completed OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
