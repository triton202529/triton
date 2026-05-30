# services/manage_positions.py
"""TRITON position management — EXIT/TRIM for risk reduction; optional ADDs to scale winning longs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
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
OPEN_POSITION_RISK_CSV = RESULTS / "open_position_risk_diagnostics.csv"
PERFORMANCE_RISK_OVERLAY_CSV = RESULTS / "performance_risk_overlay.csv"
PORTFOLIO_ALLOCATION_RECS_CSV = RESULTS / "portfolio_allocation_recommendations.csv"
EDGE_SIZING_RECOMMENDATIONS_CSV = RESULTS / "edge_sizing_recommendations.csv"
CONFIG_PATH = ROOT / "config" / "manage_positions.json"

# No lifecycle EXIT/TRIM this cycle → synthetic ROTATE_EXIT on bottom delta_pct*confidence names
FORCED_ROTATION_CAND_REASON = "forced_rotation_weakest"
# Diagnostics: full-position EXIT/ROTATE_EXIT may proceed below min_order_notional (partial trims still gated).
FULL_EXIT_MIN_NOTIONAL_BYPASS_REASON = "full_exit_bypasses_min_notional"


def _exit_pool_sort_key(x: Dict[str, Any]) -> Tuple[int, float, float]:
    """
    Tiered sort:
      Tier 0 — high-drag-forced exits: worst total_pl first (most negative first).
               This makes capital-protection EXITs eat the limited per-run slots
               before lower-priority exits and adds.
      Tier 1 — everything else: previous behavior (forced_rotation by weak_score,
               otherwise rotation/profit score).
    Returning a tuple keeps Python's stable sort happy across mixed shapes.
    """
    if bool(x.get("high_drag_forced")):
        return (0, float(x.get("high_drag_total_pl", 0.0)), 0.0)
    if x.get("forced_rotation") and "weak_score" in x:
        return (1, float(x["weak_score"]), 0.0)
    return (1, float(x["score"]), 0.0)


def _load_high_drag_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read pnl_diagnostics_by_symbol.csv into:
        {symbol: {"total_pl": float, "severity": str, "loss_source": str}}
    Last row per symbol wins. Missing or malformed file -> {} (no crash).
    Read-only; never modifies the source file.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file() or path.stat().st_size == 0:
        return out
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return out
    if df is None or df.empty:
        return out
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "symbol" not in df.columns:
        return out
    for _, r in df.iloc[::-1].iterrows():
        s = _norm_sym(r.get("symbol"))
        if not s or s in out:
            continue
        try:
            v = float(r.get("total_pl"))
            tpl = 0.0 if (math.isnan(v) or math.isinf(v)) else v
        except Exception:
            tpl = 0.0
        sev = str(r.get("severity_bucket") or "").strip().upper()
        ls = str(r.get("loss_source") or "").strip().upper()
        out[s] = {"total_pl": tpl, "severity": sev, "loss_source": ls}
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


def load_manage_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "enabled": True,
        "default_mode": "paper",
        "dry_run_default": True,  # ignored: paper always runs placement; live uses --execute
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
        "use_position_risk_diagnostics": True,
        "dynamic_add_to_long_enabled": True,
        "add_fraction_of_position": 0.25,
        "max_add_management_orders_per_run": 5,
        # High-drag forced exits (capital protection). Read-only inputs:
        # data/results/pnl_diagnostics_by_symbol.csv (total_pl, severity_bucket).
        "force_exit_high_drag_enabled": True,
        "force_exit_total_pl_threshold": -50.0,
        "force_exit_severity_buckets": ["HIGH"],
        "force_exit_high_drag_pnl_path": None,
    }
    try:
        if CONFIG_PATH.is_file():
            u = json.loads(CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
            if isinstance(u, dict):
                cfg.update(u)
    except Exception:
        pass
    return cfg


def _max_positions_for_manage() -> int:
    """
    Same cap as execute_trades / ExecutionGuard: only config/execute_trades.json
    (max_positions or legacy max_portfolio_positions). No duplicate rules or numeric fallback here.
    """
    from services.execution_guard import ExecutionGuard

    cfg: Dict[str, Any] = {}
    ExecutionGuard._merge_execute_trades_position_cap(cfg)
    mp = cfg.get("max_positions")
    if mp is None:
        raise RuntimeError(
            "config/execute_trades.json must define max_positions or max_portfolio_positions "
            "(required for manage_positions; must match execute_trades)."
        )
    return max(1, int(mp))


def _print_position_cap_sync(pos_count: int, max_pos: int) -> None:
    print("[POSITION_CAP_SYNC]", flush=True)
    print("module=manage_positions", flush=True)
    print(f"current_positions={pos_count}", flush=True)
    print(f"max_positions={max_pos}", flush=True)
    print("source=config", flush=True)


def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper()


def _load_position_risk_candidate_map(path: Path) -> Dict[str, str]:
    """
    symbol -> candidate_action (EXIT_CANDIDATE, TRIM_CANDIDATE, HOLD_OK), last row wins per symbol.
    REVIEW/unknown actions are not mapped (treated as no override).
    """
    out: Dict[str, str] = {}
    if not path.is_file() or path.stat().st_size == 0:
        return out
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return out
    if df is None or df.empty:
        return out
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "symbol" not in df.columns or "candidate_action" not in df.columns:
        return out
    for _, r in df.iloc[::-1].iterrows():
        s = _norm_sym(r.get("symbol"))
        if not s or s in out:
            continue
        a = str(r.get("candidate_action") or "").strip().upper()
        if a in ("EXIT_CANDIDATE", "TRIM_CANDIDATE", "HOLD_OK"):
            out[s] = a
    return out


def _load_performance_risk_overlay_map(path: Path) -> Dict[str, str]:
    """
    symbol -> primary_risk_flag (FORCE_EXIT, TRIM_PRIORITY, BLOCK_NEW_BUY, OK).

    Reads ``data/results/performance_risk_overlay.csv`` defensively. If the
    file is missing, empty, or malformed the map is empty and the caller
    silently skips overlay application (per spec).

    The overlay's ``risk_flag`` column may be a pipe-joined union (e.g.
    ``FORCE_EXIT|TRIM_PRIORITY``); we collapse to the highest-severity
    component because callers only key off the dominant action.

    Note: BLOCK_NEW_BUY only applies to NEW buys (not in scope for
    manage_positions, which never plans buys here), so it is loaded for
    diagnostics but treated as no-op for EXIT/TRIM decisions.
    """
    out: Dict[str, str] = {}
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
    if sym_col is None or "risk_flag" not in df.columns:
        return out

    severity_rank = {"FORCE_EXIT": 0, "TRIM_PRIORITY": 1, "BLOCK_NEW_BUY": 2, "OK": 3}
    for _, r in df.iloc[::-1].iterrows():
        s = _norm_sym(r.get(sym_col))
        if not s or s in out:
            continue
        raw = str(r.get("risk_flag") or "").strip().upper()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split("|") if p.strip()]
        if not parts:
            continue
        primary = min(parts, key=lambda p: severity_rank.get(p, 99))
        if primary in ("FORCE_EXIT", "TRIM_PRIORITY", "BLOCK_NEW_BUY", "OK"):
            out[s] = primary
    return out


def _load_portfolio_allocation_overlay_map(path: Path) -> Dict[str, str]:
    """
    symbol -> recommended_action (EXIT, TRIM, BLOCK_NEW_BUY, INCREASE, HOLD).

    Reads ``data/results/portfolio_allocation_recommendations.csv`` defensively.
    If the file is missing, empty, or malformed the map is empty and the
    caller silently skips overlay application (per spec safety rule
    "If file missing → skip silently").

    Only manage_positions-actionable recommendations (EXIT, TRIM) are used to
    influence planning; BLOCK_NEW_BUY belongs to the execution layer and
    INCREASE belongs to the sizing layer (per spec rules C and D), so they
    are loaded for diagnostic visibility only and never alter EXIT/TRIM
    flow here.
    """
    out: Dict[str, str] = {}
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
    if sym_col is None or "recommended_action" not in df.columns:
        return out

    valid_actions = {"EXIT", "TRIM", "BLOCK_NEW_BUY", "INCREASE", "HOLD"}
    # Last-write-wins semantics if duplicate rows exist for a ticker
    # (the engine emits one row per symbol, but be defensive against hand-
    # edited CSVs). The reverse iteration matches the perf-overlay loader.
    for _, r in df.iloc[::-1].iterrows():
        s = _norm_sym(r.get(sym_col))
        if not s or s in out:
            continue
        action = str(r.get("recommended_action") or "").strip().upper()
        if action in valid_actions:
            out[s] = action
    return out


def _load_edge_score_map(path: Path) -> Dict[str, float]:
    """
    symbol -> edge_score (float).

    Reads ``data/results/edge_sizing_recommendations.csv`` defensively. Used
    only for the rotation-pressure tie-break ranking (lower edge_score = weaker
    forward expectancy). Missing/empty/malformed file or missing required
    columns -> empty map; the caller treats absent symbols as edge_score=0.0
    so the tie-break degrades gracefully without crashing or biasing decisions.

    Read-only; never writes the source file. No effect on any execution path
    when the file is absent (the rotation-pressure ranking is the only
    consumer of this signal in manage_positions).
    """
    out: Dict[str, float] = {}
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
    if sym_col is None or "edge_score" not in df.columns:
        return out
    for _, r in df.iloc[::-1].iterrows():
        s = _norm_sym(r.get(sym_col))
        if not s or s in out:
            continue
        try:
            out[s] = float(r.get("edge_score"))
        except (TypeError, ValueError):
            continue
    return out


def _load_overweight_symbols(path: Path) -> Set[str]:
    """
    Set of symbols whose `allocation_band == "OVERWEIGHT"` per
    portfolio_allocation_recommendations.csv.

    Used only by the rotation-pressure ranking to break ties (overweight
    positions rank ahead of normal/underweight ones). Missing/empty file or
    missing `allocation_band` column -> empty set; rotation behaviour
    degrades to the next tie-breaker without crashing.

    Read-only. Distinct from `_load_portfolio_allocation_overlay_map` (which
    reads `recommended_action`); both can coexist without interference.
    """
    out: Set[str] = set()
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
    if sym_col is None or "allocation_band" not in df.columns:
        return out
    for _, r in df.iterrows():
        if str(r.get("allocation_band") or "").strip().upper() != "OVERWEIGHT":
            continue
        s = _norm_sym(r.get(sym_col))
        if s:
            out.add(s)
    return out


def collect_symbols_with_open_buy(
    broker: Any,
    open_orders_snapshot_path: Path,
) -> Set[str]:
    out: Set[str] = set()
    if broker is not None:
        try:
            for o in broker.list_orders(status="open", nested=True, limit=500) or []:
                if str(o.get("side") or "").lower() != "buy":
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
                if str(r.get("side") or "").lower() != "buy":
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


def _ref_price_buy(broker: Any, symbol: str) -> Optional[float]:
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

        return get_ref_price(broker, symbol, "buy")
    except Exception:
        return None


def _limit_buy(ref: float, buy_bps: float) -> float:
    return _round_price(ref * (1.0 + buy_bps / 10000.0))


def _profit_or_neutral(profit_known: bool, profit_pct: Optional[float]) -> bool:
    """
    Add allowed for winners or flat; block known losers. Unknown P/L treated as neutral.
    """
    if not profit_known or profit_pct is None:
        return True
    try:
        return float(profit_pct) >= 0.0
    except (TypeError, ValueError):
        return True


def _max_add_qty_respecting_weight(
    sym: str,
    q_raw: int,
    ref: float,
    pos_map: Dict[str, float],
    pos_metrics: Dict[str, Dict[str, Any]],
    mv_by: Dict[str, float],
    total_portfolio_value: float,
    weight_cap: float,
) -> int:
    """Scale down add qty if current MV + add would exceed max_position_weight * portfolio."""
    if q_raw < 1 or not math.isfinite(float(ref)) or float(ref) <= 0.0:
        return 0
    if not math.isfinite(float(total_portfolio_value)) or total_portfolio_value <= 0.0:
        return max(0, int(q_raw))
    if not math.isfinite(float(weight_cap)) or weight_cap <= 0.0:
        return max(0, int(q_raw))
    max_symbol_mv = float(total_portfolio_value) * float(weight_cap)
    cur_mv = float(mv_by.get(sym, 0.0) or 0.0)
    if cur_mv <= 0.0:
        m = pos_metrics.get(sym) or {}
        cp = _safe_float(m.get("current_price"), 0.0)
        q0 = _safe_float(m.get("qty", pos_map.get(sym, 0.0)))
        if cp > 0.0 and q0 > 0.0:
            cur_mv = float(q0) * float(cp)
    headroom = max_symbol_mv - cur_mv
    if headroom <= 0.0:
        return 0
    max_q_from_weight = int(math.floor(headroom / float(ref)))
    return max(0, min(int(q_raw), max_q_from_weight))


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
    # When True, this EXIT order originated from a FORCE_EXIT performance
    # risk overlay flag. main() will short-circuit ORDER_DISCIPLINE for these
    # orders so critical capital-protection exits are not silently dropped.
    # MAX_QTY/TRIM_COOLDOWN are also conceptually bypassed (FORCE_EXIT is
    # always EXIT-side and full-position, so the manage_positions ADD-side
    # MAX_QTY clamp and TRIM-side cooldown gate are structurally inapplicable).
    force_exit_override: bool = False


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
    open_buy_syms: Optional[Set[str]] = None,
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
    risk_ux = 0
    risk_tx = 0
    risk_syms: Set[str] = set()
    _use_risk = bool(cfg.get("use_position_risk_diagnostics", True))
    _risk_map = _load_position_risk_candidate_map(OPEN_POSITION_RISK_CSV) if _use_risk else {}

    # Optional performance-aware risk overlay (off by default).
    # Only loaded when the operator passed --use-performance-risk-overlay.
    # If the file is missing/malformed we fall back to an empty map and
    # the rest of the planner behaves identically to the unflagged path.
    _use_perf_overlay = bool(cfg.get("use_performance_risk_overlay", False))
    _perf_overlay_map: Dict[str, str] = (
        _load_performance_risk_overlay_map(PERFORMANCE_RISK_OVERLAY_CSV)
        if _use_perf_overlay
        else {}
    )
    perf_overlay_force_exit_count = 0
    perf_overlay_force_trim_count = 0
    perf_overlay_force_exit_syms: Set[str] = set()
    perf_overlay_force_trim_syms: Set[str] = set()
    if _use_perf_overlay:
        # One-shot diagnostic so the operator can verify the overlay loaded.
        print(
            f"[PERF_RISK_OVERLAY_LOAD] enabled=True path={PERFORMANCE_RISK_OVERLAY_CSV} "
            f"symbols={len(_perf_overlay_map)}",
            flush=True,
        )

    # Optional portfolio-allocation overlay (off by default).
    # Loads `portfolio_allocation_recommendations.csv` only when the operator
    # passes --use-portfolio-allocation. Behaves identically to the perf
    # overlay above: missing/malformed file → empty map → no behavior change.
    # Action precedence (EXIT > TRIM > BLOCK_NEW_BUY > INCREASE > HOLD) is
    # already collapsed into a single `recommended_action` per symbol by the
    # upstream `services.portfolio_allocation_engine`, so here we simply
    # honor that single field.
    _use_pae_overlay = bool(cfg.get("use_portfolio_allocation", False))
    _pae_overlay_map: Dict[str, str] = (
        _load_portfolio_allocation_overlay_map(PORTFOLIO_ALLOCATION_RECS_CSV)
        if _use_pae_overlay
        else {}
    )
    pae_overlay_force_exit_count = 0
    pae_overlay_force_trim_count = 0
    pae_overlay_force_exit_syms: Set[str] = set()
    pae_overlay_force_trim_syms: Set[str] = set()
    if _use_pae_overlay:
        print(
            f"[PORTFOLIO_ALLOCATION_OVERLAY_LOAD] enabled=True "
            f"path={PORTFOLIO_ALLOCATION_RECS_CSV} symbols={len(_pae_overlay_map)}",
            flush=True,
        )

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
    _open_buys: Set[str] = set(open_buy_syms or ())

    # ----------------------------------------------------------
    # High-drag protection (capital protection only).
    # Loads pnl_diagnostics_by_symbol.csv read-only and uses it to
    # promote high-drag positions into the EXIT pool with priority.
    # Disabled cleanly if file missing/malformed or feature flag off.
    # ----------------------------------------------------------
    hd_enabled = bool(cfg.get("force_exit_high_drag_enabled", True))
    hd_path_cfg = cfg.get("force_exit_high_drag_pnl_path")
    hd_path = Path(str(hd_path_cfg)) if hd_path_cfg else (RESULTS / "pnl_diagnostics_by_symbol.csv")
    high_drag_map = _load_high_drag_map(hd_path) if hd_enabled else {}
    try:
        hd_threshold = float(cfg.get("force_exit_total_pl_threshold", -50.0) or -50.0)
    except Exception:
        hd_threshold = -50.0
    _hd_sev_raw = cfg.get("force_exit_severity_buckets") or ["HIGH"]
    if not isinstance(_hd_sev_raw, (list, tuple, set)):
        _hd_sev_raw = ["HIGH"]
    hd_sev_set: Set[str] = {str(s).strip().upper() for s in _hd_sev_raw if str(s).strip()}
    # "STRONG BUY" interpretation: protect lifecycle stances that explicitly
    # want to keep buying/scaling (BUY or ADD). Everything else is exit-eligible
    # under high-drag protection.
    hd_strong_buy_stances = {"BUY", "ADD"}
    forced_high_drag_count = 0

    batch_median: Optional[float] = None
    _batch_confs: List[float] = []
    if lc_index is not None:
        for _sym0 in pos_map.keys():
            if _sym0 not in lc_index.index:
                continue
            _r0 = lc_index.loc[_sym0]
            if isinstance(_r0, pd.DataFrame):
                _r0 = _r0.iloc[-1]
            _c0 = _safe_float(_r0.get("confidence"), 0.0)
            if math.isfinite(_c0):
                _batch_confs.append(_c0)
    if _batch_confs:
        try:
            batch_median = float(statistics.median(_batch_confs))
        except Exception:
            batch_median = None

    add_pool: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────
    # Stale-lifecycle diagnostic pass (read-only).
    #
    # Lifecycle/effective rows whose symbol has no broker position AND no
    # active open order AND whose stance is non-actionable for management
    # (HOLD / WAIT / FLAT / BUY) are not iterated by the main loop below
    # (which walks pos_map). Such rows survive ghost-order reconciliation
    # and would historically have been mistaken for POSITION_NOT_FOUND
    # cases — they are not. They are harmless stale lifecycle leftovers.
    #
    # We log them so the operator can verify lifecycle/broker drift, but
    # we do NOT bump skip_reasons and we do NOT add a PlanRow (these are
    # not part of the planned execution set — manage_positions only acts
    # on symbols actually held at the broker).
    #
    # Real EXIT/TRIM-against-missing-position remains the responsibility
    # of execute_trades.POSITION_NOT_FOUND_FOR_{TRIM,EXIT}; this pass
    # never touches that classification.
    # ──────────────────────────────────────────────────────────────────
    _stale_lc_log_stances = {"HOLD", "WAIT", "FLAT", "BUY"}
    if lc_index is not None:
        for _sym_lc in lc_index.index:
            _sym_lc_s = str(_sym_lc).strip().upper()
            if not _sym_lc_s:
                continue
            if _sym_lc_s in pos_map:
                continue
            if _sym_lc_s in _open_buys or _sym_lc_s in open_sell_syms:
                continue
            _row_lc = lc_index.loc[_sym_lc]
            if isinstance(_row_lc, pd.DataFrame):
                _row_lc = _row_lc.iloc[-1]
            _stance_lc, _ = resolve_authoritative_stance(_row_lc)
            _stance_lc = (_stance_lc or "").strip().upper()
            if _stance_lc not in _stale_lc_log_stances:
                continue
            _action_lc = (
                str(_row_lc.get("lifecycle_action") or "").strip().upper() or _stance_lc or "NONE"
            )
            print(
                f"[STALE_LIFECYCLE_ROW_IGNORED] symbol={_sym_lc_s} "
                f"action={_action_lc} stance={_stance_lc} "
                f"reason=no_broker_position_no_open_order",
                flush=True,
            )

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
            # Broker holds this symbol but lifecycle has no directive for
            # it. Per the new contract, POSITION_NOT_FOUND is reserved for
            # explicit EXIT/TRIM-against-missing-position cases (handled in
            # execute_trades as POSITION_NOT_FOUND_FOR_{ADD,TRIM,EXIT}).
            # This branch is non-actionable — no EXIT/TRIM is being denied,
            # we just have nothing to do — so we classify it as
            # STALE_LIFECYCLE_NO_POSITION for traceability in the PlanRow
            # CSV but intentionally DO NOT bump skip_reasons (which counts
            # actionable skips only).
            lines.append(
                PlanRow(
                    sym,
                    "skip",
                    "skipped",
                    "STALE_LIFECYCLE_NO_POSITION",
                    "",
                    reason_code="STALE_LIFECYCLE_NO_POSITION",
                    reason_detail="no lifecycle directive for broker position",
                    current_position_qty=qty_pos,
                    profit_pct=profit_pct,
                    avg_entry_price=avg_e,
                    current_price=cur_p,
                )
            )
            print(
                f"[STALE_LIFECYCLE_ROW_IGNORED] symbol={sym} "
                f"action=NONE stance=NONE "
                f"reason=no_lifecycle_directive_for_broker_position",
                flush=True,
            )
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
            if not bool(cfg.get("dynamic_add_to_long_enabled", True)):
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
            if batch_median is None or not math.isfinite(conf) or conf < float(batch_median):
                sym_state["last_management_action"] = "SKIP_ADD_LOW_CONF"
                lines.append(
                    PlanRow(
                        sym,
                        "skip",
                        "skipped",
                        "ADD_BELOW_MEDIAN_CONF",
                        stance_disp,
                        lifecycle_stance=st,
                        effective_stance=eff,
                        current_position_qty=qty_pos,
                        avg_entry_price=avg_e,
                        current_price=cur_p,
                        profit_pct=profit_pct,
                        reason_code="ADD_BELOW_MEDIAN_CONF",
                        reason_detail=f"conf={conf} batch_median={batch_median}",
                        final_action="NONE",
                        score=score,
                        priority_bucket="HOLD",
                        selected_for_execution=False,
                    )
                )
                bump("ADD_BELOW_MEDIAN_CONF")
                continue
            if not _profit_or_neutral(profit_known, profit_pct):
                sym_state["last_management_action"] = "SKIP_ADD_LOSER"
                lines.append(
                    PlanRow(
                        sym,
                        "skip",
                        "skipped",
                        "ADD_LOSING_POSITION",
                        stance_disp,
                        lifecycle_stance=st,
                        effective_stance=eff,
                        current_position_qty=qty_pos,
                        avg_entry_price=avg_e,
                        current_price=cur_p,
                        profit_pct=profit_pct,
                        reason_code="ADD_LOSING_POSITION",
                        reason_detail="add only when P/L is neutral or positive",
                        final_action="NONE",
                        score=score,
                        priority_bucket="HOLD",
                        selected_for_execution=False,
                    )
                )
                bump("ADD_LOSING_POSITION")
                continue
            if bool(cfg.get("skip_if_open_order_exists_same_side", True)) and sym in _open_buys:
                sym_state["last_management_action"] = "SKIP_ADD_OPEN_BUY"
                lines.append(
                    PlanRow(
                        sym,
                        "skip",
                        "skipped",
                        "OPEN_BUY_ALREADY_EXISTS",
                        stance_disp,
                        lifecycle_stance=st,
                        effective_stance=eff,
                        current_position_qty=qty_pos,
                        avg_entry_price=avg_e,
                        current_price=cur_p,
                        profit_pct=profit_pct,
                        reason_code="OPEN_BUY_ALREADY_EXISTS",
                        final_action="NONE",
                        score=score,
                        priority_bucket="HOLD",
                        selected_for_execution=False,
                    )
                )
                bump("OPEN_BUY_ALREADY_EXISTS")
                continue
            sym_state["last_management_action"] = "ADD_QUEUED"
            add_pool.append(
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
                }
            )
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

        risk_action = str(_risk_map.get(sym) or "").strip().upper() if _risk_map else ""
        if risk_action not in ("EXIT_CANDIDATE", "TRIM_CANDIDATE", "HOLD_OK"):
            risk_action = ""
        force_exit = False
        force_trim = False
        if _risk_map and risk_action and risk_action != "HOLD_OK":
            if risk_action == "EXIT_CANDIDATE" and mgmt_stance != "EXIT":
                if mgmt_stance in ("HOLD", "WAIT", "TRIM", "FLAT"):
                    force_exit = True
            elif (
                risk_action == "TRIM_CANDIDATE" and not force_exit and mgmt_stance not in ("EXIT",)
            ):
                if mgmt_stance in ("HOLD", "WAIT", "FLAT", "TRIM"):
                    force_trim = True

        # Performance risk overlay (optional, additive).
        # Layers FORCE_EXIT / TRIM_PRIORITY on top of the existing
        # `force_exit`/`force_trim` plumbing so downstream pool placement,
        # discipline checks, and risk caps remain identical. Conviction
        # stances (BUY/ADD) are intentionally protected to avoid stomping
        # on lifecycle-strong longs, mirroring the existing risk map gating.
        # `perf_force_exit_active` is a per-symbol flag: True only when this
        # iteration's FORCE_EXIT was triggered by the perf overlay. It is
        # propagated into the exit_pool dict so the placement loop can stamp
        # `force_exit_override=True` on the resulting ManagementPlannedOrder
        # and main() can short-circuit ORDER_DISCIPLINE for it.
        perf_force_exit_active = False
        if _use_perf_overlay and _perf_overlay_map:
            perf_flag = str(_perf_overlay_map.get(sym) or "").strip().upper()
            if perf_flag in ("FORCE_EXIT", "TRIM_PRIORITY", "BLOCK_NEW_BUY", "OK"):
                perf_action_override = "NONE"
                if perf_flag == "FORCE_EXIT" and mgmt_stance != "EXIT":
                    if mgmt_stance in ("HOLD", "WAIT", "TRIM", "FLAT"):
                        if not force_exit:
                            perf_overlay_force_exit_count += 1
                            perf_overlay_force_exit_syms.add(sym)
                        force_exit = True
                        perf_force_exit_active = True
                        perf_action_override = "EXIT"
                elif perf_flag == "TRIM_PRIORITY" and not force_exit and mgmt_stance != "EXIT":
                    if mgmt_stance in ("HOLD", "WAIT", "FLAT", "TRIM"):
                        if not force_trim:
                            perf_overlay_force_trim_count += 1
                            perf_overlay_force_trim_syms.add(sym)
                        force_trim = True
                        perf_action_override = "TRIM"
                # FORCE_EXIT / TRIM_PRIORITY that we did not promote (e.g.
                # BUY/ADD stance) are still logged so the operator can see
                # the overlay considered them; OK / BLOCK_NEW_BUY simply
                # log as no-op.
                print(
                    f"[PERF_RISK_APPLIED] symbol={sym} risk_flag={perf_flag} "
                    f"action_override={perf_action_override}",
                    flush=True,
                )

        # Portfolio-allocation overlay (optional, additive).
        # Layers EXIT/TRIM hints from the portfolio allocation engine on top
        # of the same `force_exit` / `force_trim` plumbing the perf overlay
        # uses, so downstream pool placement, discipline checks, and risk
        # caps all behave identically. This intentionally does NOT touch
        # BUY/ADD logic (spec rule 6: "Do NOT affect BUY/ADD logic"); it
        # only influences EXIT/TRIM prioritization.
        #
        # Reuses the existing `perf_force_exit_active` flag for FORCE_EXIT
        # propagation: the downstream contract is "force_exit_override=True
        # on the planned ManagementPlannedOrder for any high-conviction
        # exit", and the source of conviction (perf overlay vs. portfolio
        # allocation) is irrelevant to the placement layer. Counters and
        # the `[PORTFOLIO_ALLOCATION_APPLIED]` audit log keep the two
        # overlays' contributions distinguishable.
        if _use_pae_overlay and _pae_overlay_map:
            pae_action = str(_pae_overlay_map.get(sym) or "").strip().upper()
            if pae_action in ("EXIT", "TRIM", "BLOCK_NEW_BUY", "INCREASE", "HOLD"):
                pae_override = "NONE"
                if pae_action == "EXIT" and mgmt_stance != "EXIT":
                    if mgmt_stance in ("HOLD", "WAIT", "TRIM", "FLAT"):
                        if not force_exit:
                            pae_overlay_force_exit_count += 1
                            pae_overlay_force_exit_syms.add(sym)
                        force_exit = True
                        perf_force_exit_active = True
                        pae_override = "EXIT"
                elif pae_action == "TRIM" and not force_exit and mgmt_stance != "EXIT":
                    if mgmt_stance in ("HOLD", "WAIT", "FLAT", "TRIM"):
                        if not force_trim:
                            pae_overlay_force_trim_count += 1
                            pae_overlay_force_trim_syms.add(sym)
                        force_trim = True
                        pae_override = "TRIM"
                # BLOCK_NEW_BUY → execution layer concern (place_live_orders
                # already enforces it); INCREASE → sizing layer concern
                # (execute_trades / edge_sizing); HOLD → no-op. We still
                # emit the audit line so the operator can see what the
                # overlay considered for every symbol.
                print(
                    f"[PORTFOLIO_ALLOCATION_APPLIED] symbol={sym} "
                    f"recommended_action={pae_action} override={pae_override}",
                    flush=True,
                )

        # High-drag protection: force EXIT_CANDIDATE for positions whose
        # cumulative damage is large or whose severity bucket says HIGH
        # (only when the position is actually losing — never on winners).
        # Skipped for stances that the lifecycle still treats as STRONG BUY
        # (mgmt_stance in BUY/ADD) so we do not stomp on conviction longs.
        high_drag_forced = False
        hd_total_pl_for_sort: float = 0.0
        hd_total_pl_for_log: Optional[float] = None
        hd_severity: str = ""
        hd_loss_source: str = ""
        if hd_enabled and high_drag_map:
            hd_entry = high_drag_map.get(sym) or {}
            if hd_entry:
                try:
                    _tpl = float(hd_entry.get("total_pl", 0.0) or 0.0)
                except Exception:
                    _tpl = 0.0
                hd_total_pl_for_log = _tpl
                hd_total_pl_for_sort = _tpl
                hd_severity = str(hd_entry.get("severity") or "").strip().upper()
                hd_loss_source = str(hd_entry.get("loss_source") or "").strip().upper()
                # Severity arm only fires when the position is actually losing,
                # so a winner (e.g., severity=HIGH due to large positive PnL) is
                # not exited by mistake.
                sev_trigger = bool(hd_sev_set) and (hd_severity in hd_sev_set) and (_tpl < 0.0)
                pl_trigger = _tpl < hd_threshold
                if (sev_trigger or pl_trigger) and (mgmt_stance not in hd_strong_buy_stances):
                    high_drag_forced = True

        exit_stance = mgmt_stance == "EXIT"
        trim_stance = mgmt_stance == "TRIM"
        exit_immediate = exit_stance or (d_pct <= exit_delta) or force_exit or high_drag_forced

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
            if high_drag_forced:
                cand_reason = "EXIT_HIGH_DRAG_PROTECTION"
                # Risk diagnostics may also have flagged this symbol; preserve
                # accounting if so, but the high-drag reason takes log priority.
                if force_exit:
                    risk_ux += 1
                risk_syms.add(sym)
                forced_high_drag_count += 1
                detail = (
                    f"high_drag_protection total_pl={hd_total_pl_for_log} "
                    f"severity={hd_severity or 'NONE'} loss_source={hd_loss_source or 'NONE'} "
                    f"mgmt_stance={mgmt_stance} threshold={hd_threshold} "
                    f"delta={d_pct:.6f} exit_delta={exit_delta} stale={stale_exit}"
                )
                print(
                    f"[FORCED_EXIT] symbol={sym} "
                    f"total_pl={hd_total_pl_for_log if hd_total_pl_for_log is not None else 'NA'} "
                    f"severity={hd_severity or 'NONE'} reason=high_drag_protection",
                    flush=True,
                )
            elif force_exit:
                cand_reason = "EXIT_RISK_OVERRIDE"
                risk_ux += 1
                risk_syms.add(sym)
                detail = (
                    f"open_position_risk=EXIT_CANDIDATE; mgmt_stance={mgmt_stance} "
                    f"delta={d_pct:.6f} exit_delta={exit_delta} stale={stale_exit}"
                )
            else:
                cand_reason = (
                    "EXIT_APPROVED" if exit_immediate else "EXIT_APPROVED_STALE_WEAK_CYCLES"
                )
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
                    "high_drag_forced": bool(high_drag_forced),
                    "high_drag_total_pl": float(hd_total_pl_for_sort),
                    "high_drag_severity": hd_severity,
                    "perf_force_exit": bool(perf_force_exit_active),
                }
            )
            continue
        weak_trim_sig = (mgmt_stance in ("HOLD", "WAIT") and d_pct <= weak_trim) or force_trim
        want_trim = trim_stance or weak_trim_sig
        if want_trim:
            if trim_stance:
                trim_reason = "TRIM_STANCE"
            elif force_trim:
                trim_reason = "TRIM_RISK_FORCED"
            else:
                trim_reason = "TRIM_WEAK_SIGNAL"
            if require_profit_trim and not force_trim:
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
            if force_trim:
                risk_tx += 1
                risk_syms.add(sym)
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
    max_pos = _max_positions_for_manage()
    _print_position_cap_sync(pos_count, max_pos)
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
            # ──────────────────────────────────────────────────────────────
            # Rotation throughput: --max-rotation-exits N (cfg.max_rotation_exits)
            #
            # When the operator passes --max-rotation-exits N AND N > 1, the
            # forced-rotation cap is widened from the default
            #   k = min(min(ceil(n * pct_w), forced_rotation_max_orders), max_n)
            # to N, AND candidates are re-ranked by a tiered key that prefers:
            #   1) FORCE_EXIT (perf-risk overlay or PAE overlay)
            #   2) worst total_pl  (most negative first)
            #   3) worst unrealized_pl (most negative first)
            #   4) lowest edge_score (weakest forward edge)
            #   5) overweight allocation_band first
            #   6) lowest rotation score (weakest combined signal)
            #
            # When the flag is absent OR N <= 1, the original path runs
            # untouched (weak_score sort + percentage/fm/max_n cap), so the
            # default behaviour is preserved bit-for-bit.
            #
            # Safety: this branch only re-orders and recaps the same
            # `fr_cands` list that already passed every per-symbol filter
            # upstream (no BUY/ADD stance, qty_pos > 0, in lifecycle index,
            # not in open_sell_syms when skip_open is on). It does NOT
            # bypass: broker connectivity, market closed, denylist, or any
            # downstream guard. The existing turnover cap below STILL
            # applies to the resulting take, and FORCE_EXIT
            # protections in main() / _partition_planned_by_guard /
            # place_live_orders are unchanged.
            # ──────────────────────────────────────────────────────────────
            mre_raw = cfg.get("max_rotation_exits")
            try:
                mre_n = int(mre_raw) if mre_raw is not None else None
            except (TypeError, ValueError):
                mre_n = None
            mre_active = mre_n is not None and mre_n > 1

            rank_skipped_syms: List[str] = []
            if mre_active:
                # Lazy load ranking enrichment maps. Each loader degrades
                # to an empty result if its source file is missing, keeping
                # the rotation-pressure path fully optional and safe.
                edge_score_map = _load_edge_score_map(EDGE_SIZING_RECOMMENDATIONS_CSV)
                overweight_set = _load_overweight_symbols(PORTFOLIO_ALLOCATION_RECS_CSV)
                for _c in fr_cands:
                    _sym = _c["sym"]
                    _perf_flag = str(_perf_overlay_map.get(_sym, "") or "").strip().upper()
                    _pae_action = str(_pae_overlay_map.get(_sym, "") or "").strip().upper()
                    _c["_rp_force_exit"] = "FORCE_EXIT" in _perf_flag or _pae_action == "EXIT"
                    _hd = high_drag_map.get(_sym) or {}
                    try:
                        _c["_rp_total_pl"] = float(_hd.get("total_pl") or 0.0)
                    except (TypeError, ValueError):
                        _c["_rp_total_pl"] = 0.0
                    _met = pos_metrics.get(_sym) or {}
                    _ap = _met.get("avg_entry_price")
                    _cp = _met.get("current_price")
                    try:
                        if _ap and _cp and float(_ap) > 0 and float(_cp) > 0:
                            _c["_rp_unrealized_pl"] = float(_c["qty_pos"]) * (
                                float(_cp) - float(_ap)
                            )
                        else:
                            _c["_rp_unrealized_pl"] = 0.0
                    except (TypeError, ValueError):
                        _c["_rp_unrealized_pl"] = 0.0
                    try:
                        _c["_rp_edge_score"] = float(edge_score_map.get(_sym, 0.0))
                    except (TypeError, ValueError):
                        _c["_rp_edge_score"] = 0.0
                    _c["_rp_overweight"] = _sym in overweight_set
                fr_cands.sort(
                    key=lambda c: (
                        0 if c.get("_rp_force_exit") else 1,
                        float(c.get("_rp_total_pl") or 0.0),
                        float(c.get("_rp_unrealized_pl") or 0.0),
                        float(c.get("_rp_edge_score") or 0.0),
                        0 if c.get("_rp_overweight") else 1,
                        float(c.get("score") or 0.0),
                    )
                )
                cap_n = max(1, mre_n)
                take = fr_cands[:cap_n]
                rank_skipped_syms = [c["sym"] for c in fr_cands[cap_n:]]
                print(
                    f"[ROTATION_PRESSURE] enabled=True max_rotation_exits={mre_n} "
                    f"candidates={len(fr_cands)} selected={len(take)}",
                    flush=True,
                )
                for _rank, _tc in enumerate(take, start=1):
                    _reason = "FORCE_EXIT" if _tc.get("_rp_force_exit") else "WEAK_ROTATION"
                    print(
                        f"[ROTATION_PRESSURE_SELECTED] rank={_rank} "
                        f"symbol={_tc['sym']} reason={_reason} "
                        f"score={float(_tc.get('score') or 0.0):.6f} "
                        f"total_pl={float(_tc.get('_rp_total_pl') or 0.0):.2f} "
                        f"unrealized_pl={float(_tc.get('_rp_unrealized_pl') or 0.0):.2f} "
                        f"edge_score={float(_tc.get('_rp_edge_score') or 0.0):.4f}",
                        flush=True,
                    )
            else:
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
            take_pre_turnover_syms = [tc["sym"] for tc in take]
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
            if mre_active:
                _selected_syms = [x["sym"] for x in take]
                _selected_syms_set = set(_selected_syms)
                _turnover_skipped_syms = [
                    s for s in take_pre_turnover_syms if s not in _selected_syms_set
                ]
                _skipped_syms = list(rank_skipped_syms) + _turnover_skipped_syms
                _skip_reasons: Dict[str, int] = {}
                if rank_skipped_syms:
                    _skip_reasons["below_rank_cutoff"] = len(rank_skipped_syms)
                if _turnover_skipped_syms:
                    _skip_reasons["turnover_cap_exceeded"] = len(_turnover_skipped_syms)
                print(
                    f"[ROTATION_PRESSURE_SUMMARY] "
                    f"selected_symbols={_selected_syms} "
                    f"skipped_symbols={_skipped_syms} "
                    f"skip_reasons={_skip_reasons}",
                    flush=True,
                )
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

    # ──────────────────────────────────────────────────────────────────
    # FORCE_EXIT pre-placement carve-out (batch/order discipline limit).
    #
    # Performance-risk-overlay and portfolio-allocation-overlay decisions
    # mark a candidate with cand["perf_force_exit"]=True. Without this
    # carve-out, a high-EXIT-volume run could silently drop such a cand
    # whose natural rank in normal_exit_list is past
    # max_management_orders_per_run BEFORE it is materialized into a
    # ManagementPlannedOrder — defeating the downstream bypasses keyed on
    # ManagementPlannedOrder.force_exit_override (main()'s discipline
    # partition + _partition_planned_by_guard's per-symbol guard bypass).
    #
    # Behaviour:
    #   • FORCE_EXIT cands are ALWAYS selected, regardless of max_n.
    #   • The cap continues to apply to non-FORCE_EXIT cands (max_n slots).
    #   • Sort order is preserved (cands are appended in their
    #     normal_exit_list order; FORCE_EXITs do not jump ahead of higher-
    #     priority high_drag_forced cands or vice versa).
    #   • Per-symbol log is emitted for every FORCE_EXIT kept in the batch
    #     so the audit trail is unambiguous regardless of cap pressure:
    #         [FORCE_EXIT_PREPLACEMENT_OVERRIDE] symbol=<S>
    #             reason=batch_cap_bypass action=kept_in_batch
    #     followed by a single
    #         [FORCE_EXIT_PREPLACEMENT_OVERRIDE_SUMMARY] count=<N>
    #             symbols=<S1,S2,…>
    #     summary line.
    #
    # Safety:
    #   • Only triggers for cand["perf_force_exit"]=True. BUY/ADD/TRIM
    #     and normal-EXIT cands without this flag are unaffected.
    #   • Critical guards (broker, market, symbol denylist, no broker
    #     position) are downstream of this and continue to enforce.
    #   • Forced-rotation EXITs (cand_reason=FORCED_ROTATION_CAND_REASON)
    #     are in forced_rotation_list, not normal_exit_list, so this
    #     carve-out does not interact with the rotation turnover cap.
    #   • De-duplication via fx_seen_syms guarantees a symbol is included
    #     at most once even if it appears more than once in the pool.
    # ──────────────────────────────────────────────────────────────────
    selected_exit: List[Dict[str, Any]] = []
    fx_kept_syms: List[str] = []
    fx_seen_syms: Set[str] = set()
    non_fx_kept = 0
    for x in normal_exit_list:
        sym_x = str(x.get("sym") or "")
        is_fx = bool(x.get("perf_force_exit", False))
        if is_fx:
            if sym_x in fx_seen_syms:
                continue
            fx_seen_syms.add(sym_x)
            selected_exit.append(x)
            fx_kept_syms.append(sym_x)
            print(
                f"[FORCE_EXIT_PREPLACEMENT_OVERRIDE] symbol={sym_x} "
                f"reason=batch_cap_bypass action=kept_in_batch",
                flush=True,
            )
            continue
        if non_fx_kept >= max_n:
            continue
        selected_exit.append(x)
        non_fx_kept += 1
    if fx_kept_syms:
        print(
            f"[FORCE_EXIT_PREPLACEMENT_OVERRIDE_SUMMARY] count={len(fx_kept_syms)} "
            f"symbols={','.join(fx_kept_syms)}",
            flush=True,
        )

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
            force_exit_override=bool(cand.get("perf_force_exit", False)) and mgmt_action == "EXIT",
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

    if add_pool and bool(cfg.get("dynamic_add_to_long_enabled", True)):
        add_frac = float(cfg.get("add_fraction_of_position", 0.25) or 0.25)
        add_frac = max(0.0, min(0.5, add_frac))
        try:
            from services.execute_trades import load_execute_trades_config

            _etx = load_execute_trades_config()
            w_cap = float(_etx.get("max_position_weight_hard_cap", 0.10) or 0.10)
            buy_bps = float(_etx.get("limit_price_buffer_buy_bps", 35) or 35)
        except Exception:
            w_cap, buy_bps = 0.10, 35.0
        _gcfg: Dict[str, Any] = {}
        try:
            from services.execution_guard import ExecutionGuard

            if broker is not None:
                _gcfg = ExecutionGuard(broker).cfg
        except Exception:
            pass
        g_min_n = float(_gcfg.get("min_notional_usd", min_notional) or min_notional)
        g_max_n = float(_gcfg.get("max_notional_usd", 1500) or 1500)
        g_max_q = int(_gcfg.get("max_qty", 200) or 200)
        eff_min = max(float(min_notional), g_min_n)
        if not math.isfinite(g_max_n) or g_max_n <= 0.0:
            g_max_n = 1500.0

        add_pool.sort(key=lambda a: -float(a.get("conf", 0.0)))
        _slots = max(0, max_n - len(planned))
        _sub = int(cfg.get("max_add_management_orders_per_run", 5) or 5)
        _sub = max(0, _sub)
        take_n = min(_slots, _sub, len(add_pool)) if _sub else min(_slots, len(add_pool))
        not_selected = add_pool[take_n:] if take_n < len(add_pool) else []
        to_place = add_pool[:take_n]

        mv_add = _market_values_by_symbol(ps_path, pos_map)
        t_pv = float(total_portfolio_value)
        for ax in not_selected:
            s = str(ax.get("sym") or "")
            lines.append(
                PlanRow(
                    s,
                    "skip",
                    "skipped",
                    "ADD_NOT_SELECTED_THIS_RUN",
                    str(ax.get("stance_disp") or ""),
                    planned=None,
                    lifecycle_stance=str(ax.get("st") or ""),
                    effective_stance=str(ax.get("eff") or ""),
                    current_position_qty=float(ax.get("qty_pos", 0.0)),
                    avg_entry_price=ax.get("avg_e"),
                    current_price=ax.get("cur_p"),
                    profit_pct=ax.get("profit_pct"),
                    reason_code="ADD_NOT_SELECTED_THIS_RUN",
                    final_action="NONE",
                    score=float(ax.get("score", 0.0)) if ax.get("score") is not None else None,
                    priority_bucket="HOLD",
                    selected_for_execution=False,
                )
            )
            bump("ADD_NOT_SELECTED_THIS_RUN")

        ad_rank = 10000
        for ax in to_place:
            sym = str(ax.get("sym") or "")
            ad_rank += 1
            eff, lc, st = ax.get("eff"), ax.get("lc"), ax.get("st")
            stance_disp = str(ax.get("stance_disp") or "")
            conf = float(ax.get("conf", 0.0))
            d_pct = float(ax.get("d_pct", 0.0))
            rationale = str(ax.get("rationale") or "")
            qty_pos = float(ax.get("qty_pos", 0.0))
            avg_e = ax.get("avg_e")
            cur_p = ax.get("cur_p")
            profit_pct = ax.get("profit_pct")
            sc = float(ax.get("score", 0.0))
            ref = _ref_price_buy(broker, sym)
            if ref is None or ref <= 0.0:
                lines.append(
                    PlanRow(
                        sym,
                        "skip",
                        "skipped",
                        "NO_PRICE_AVAILABLE",
                        stance_disp,
                        planned=None,
                        lifecycle_stance=str(st or ""),
                        effective_stance=str(eff or ""),
                        current_position_qty=qty_pos,
                        avg_entry_price=avg_e,
                        current_price=cur_p,
                        profit_pct=profit_pct,
                        reason_code="NO_PRICE_AVAILABLE",
                        final_action="NONE",
                        score=sc,
                        rank=ad_rank,
                        priority_bucket="ADD",
                        selected_for_execution=False,
                    )
                )
                bump("NO_PRICE_AVAILABLE")
                continue
            lim: Optional[float] = None
            ot = "limit"
            if prefer_limit:
                lim = _limit_buy(float(ref), buy_bps)
            elif allow_mkt:
                ot = "market"
            if not prefer_limit and not allow_mkt and lim is None:
                lim = _limit_buy(float(ref), buy_bps)
            if prefer_limit and lim is None:
                lines.append(
                    PlanRow(
                        sym,
                        "skip",
                        "skipped",
                        "NO_PRICE_AVAILABLE",
                        stance_disp,
                        planned=None,
                        lifecycle_stance=str(st or ""),
                        effective_stance=str(eff or ""),
                        current_position_qty=qty_pos,
                        avg_entry_price=avg_e,
                        current_price=cur_p,
                        profit_pct=profit_pct,
                        reason_code="NO_PRICE_AVAILABLE",
                        final_action="NONE",
                        score=sc,
                        rank=ad_rank,
                        priority_bucket="ADD",
                        selected_for_execution=False,
                    )
                )
                bump("NO_PRICE_AVAILABLE")
                continue
            q_raw = max(1, int(math.floor(qty_pos * add_frac)))
            exec_px = float(lim) if lim is not None else float(ref)
            q = _max_add_qty_respecting_weight(
                sym,
                q_raw,
                exec_px,
                pos_map,
                pos_metrics,
                mv_add,
                t_pv,
                w_cap,
            )
            if int(q) < 1:
                lines.append(
                    PlanRow(
                        sym,
                        "skip",
                        "skipped",
                        "ADD_EXCEEDS_MAX_POSITION_WEIGHT",
                        stance_disp,
                        planned=None,
                        lifecycle_stance=str(st or ""),
                        effective_stance=str(eff or ""),
                        current_position_qty=qty_pos,
                        avg_entry_price=avg_e,
                        current_price=cur_p,
                        profit_pct=profit_pct,
                        reason_code="ADD_EXCEEDS_MAX_POSITION_WEIGHT",
                        final_action="NONE",
                        score=sc,
                        rank=ad_rank,
                        priority_bucket="ADD",
                        selected_for_execution=False,
                    )
                )
                bump("ADD_EXCEEDS_MAX_POSITION_WEIGHT")
                continue
            q = int(q)
            if exec_px * float(q) > g_max_n + 1e-9 and exec_px > 0.0:
                cap_q = int(math.floor(g_max_n / exec_px))
                q = min(q, max(0, cap_q))
            if g_max_q and q > g_max_q:
                q = g_max_q
            if q < 1:
                lines.append(
                    PlanRow(
                        sym,
                        "skip",
                        "skipped",
                        "ADD_ZERO_QTY",
                        stance_disp,
                        planned=None,
                        lifecycle_stance=str(st or ""),
                        effective_stance=str(eff or ""),
                        current_position_qty=qty_pos,
                        avg_entry_price=avg_e,
                        current_price=cur_p,
                        profit_pct=profit_pct,
                        reason_code="ADD_ZERO_QTY",
                        final_action="NONE",
                        score=sc,
                        rank=ad_rank,
                        priority_bucket="ADD",
                        selected_for_execution=False,
                    )
                )
                bump("ADD_ZERO_QTY")
                continue
            notional = float(q) * exec_px
            if notional < eff_min + 1e-9:
                lines.append(
                    PlanRow(
                        sym,
                        "skip",
                        "skipped",
                        "ADD_BELOW_MIN_ORDER_NOTIONAL",
                        stance_disp,
                        planned=None,
                        lifecycle_stance=str(st or ""),
                        effective_stance=str(eff or ""),
                        current_position_qty=qty_pos,
                        avg_entry_price=avg_e,
                        current_price=cur_p,
                        profit_pct=profit_pct,
                        reason_code="ADD_BELOW_MIN_ORDER_NOTIONAL",
                        final_action="NONE",
                        score=sc,
                        rank=ad_rank,
                        priority_bucket="ADD",
                        selected_for_execution=False,
                    )
                )
                bump("ADD_BELOW_MIN_ORDER_NOTIONAL")
                continue
            ex_qty = int(math.floor(qty_pos))
            print(
                f"[ADD_EXECUTION] symbol={sym} existing_qty={ex_qty} add_qty={q} confidence={conf:.6f}",
                flush=True,
            )
            po = ManagementPlannedOrder(
                symbol=sym,
                side="buy",
                qty=q,
                order_type=ot,
                time_in_force="day",
                limit_price=lim,
                management_action="ADD",
                rationale=rationale,
                confidence=conf,
                delta_pct=d_pct,
                current_position_qty=qty_pos,
                mode=mode,
                generated_at=_utc_iso(),
                planned_notional=float(notional),
            )
            planned.append(po)
            get_symbol_state(state, sym)["last_management_action"] = "ADD"
            lines.append(
                PlanRow(
                    sym,
                    "plan",
                    "planned",
                    "",
                    stance_disp,
                    po,
                    lifecycle_stance=str(st or ""),
                    effective_stance=str(eff or ""),
                    current_position_qty=qty_pos,
                    avg_entry_price=avg_e,
                    current_price=cur_p,
                    profit_pct=profit_pct,
                    management_action_candidate="ADD",
                    final_action="ADD",
                    reason_code="ADD_APPROVED",
                    reason_detail="dynamic add to long (median+ confidence, neutral/profit)",
                    score=sc,
                    rank=ad_rank,
                    priority_bucket="ADD",
                    selected_for_execution=True,
                )
            )

    rm = "ACTIVE" if rotation_applied else "NORMAL"
    for pr in lines:
        pr.rotation_mode = rm

    state["position_risk_overrides"] = {
        "exit_forced": int(risk_ux),
        "trim_forced": int(risk_tx),
        "symbols_affected": ",".join(sorted(risk_syms)),
    }
    if risk_ux or risk_tx:
        print(
            f"[POSITION_OVERRIDE] exit_forced={risk_ux} trim_forced={risk_tx} "
            f"symbols_affected={','.join(sorted(risk_syms))}",
            flush=True,
        )

    if hd_enabled:
        print(
            f"[FORCED_EXIT_SUMMARY] count={int(forced_high_drag_count)} "
            f"threshold={hd_threshold} severity_buckets={sorted(hd_sev_set)}",
            flush=True,
        )

    if _use_perf_overlay:
        print(
            f"[PERF_RISK_OVERLAY_SUMMARY] enabled=True "
            f"force_exit={int(perf_overlay_force_exit_count)} "
            f"trim_priority={int(perf_overlay_force_trim_count)} "
            f"force_exit_syms={sorted(perf_overlay_force_exit_syms)} "
            f"trim_priority_syms={sorted(perf_overlay_force_trim_syms)}",
            flush=True,
        )

    if _use_pae_overlay:
        print(
            f"[PORTFOLIO_ALLOCATION_OVERLAY_SUMMARY] enabled=True "
            f"force_exit={int(pae_overlay_force_exit_count)} "
            f"trim_priority={int(pae_overlay_force_trim_count)} "
            f"force_exit_syms={sorted(pae_overlay_force_exit_syms)} "
            f"trim_priority_syms={sorted(pae_overlay_force_trim_syms)}",
            flush=True,
        )

    return planned, lines, skip_reasons, state, trim_candidates, exit_candidates, rotation_applied


def _fmt_top_skip_reasons(skip_reasons: Dict[str, int], limit: int = 8) -> str:
    if not skip_reasons:
        return "none"
    items = sorted(skip_reasons.items(), key=lambda x: (-x[1], x[0]))[:limit]
    return ", ".join(f"{k}={v}" for k, v in items)


def _count_planned_actions(planned: List[ManagementPlannedOrder]) -> Tuple[int, int, int, int]:
    """Returns (exit_count, trim_count, forced_rotation_rotate_exit_count, add_count)."""
    exits = 0
    trims = 0
    fr = 0
    adds = 0
    for p in planned:
        ma = str(p.management_action or "").upper()
        if ma == "EXIT":
            exits += 1
        elif ma == "TRIM":
            trims += 1
        elif ma == "ADD":
            adds += 1
        elif ma == "ROTATE_EXIT" and bool(getattr(p, "forced_rotation", False)):
            fr += 1
    return exits, trims, fr, adds


def _rotation_execution_stats(
    planned: List[ManagementPlannedOrder], execution_succeeded: bool
) -> Tuple[int, int, float]:
    """
    After a successful place_live_orders batch, approximate EXIT+ROTATE_EXIT vs TRIM
    and sum planned_notional as capital_freed estimate (sells reduce exposure).
    """
    if not execution_succeeded or not planned:
        return 0, 0, 0.0
    exits_n = 0
    trims_n = 0
    cap = 0.0
    for p in planned:
        ma = str(p.management_action or "").upper()
        n = float(getattr(p, "planned_notional", 0.0) or 0.0)
        if ma in ("EXIT", "ROTATE_EXIT"):
            exits_n += 1
            cap += n
        elif ma == "TRIM":
            trims_n += 1
            cap += n
    return exits_n, trims_n, cap


def _guard_codes_for_display(guard_bad: List[str]) -> str:
    seen: Set[str] = set()
    codes: List[str] = []
    for b in guard_bad:
        c = b.split(":", 1)[1] if ":" in b else b
        if c not in seen:
            seen.add(c)
            codes.append(c)
    return ", ".join(codes) if codes else "unknown"


# Per-symbol / policy blocks — batch may continue with other symbols; not system failure.
MANAGE_GUARD_CONTROLLED_CODES: Set[str] = {
    "MAX_QTY",
    "MAX_NOTIONAL",
    "MIN_NOTIONAL",
    "MAX_POSITIONS",
    "MAX_POSITIONS_REACHED",
    "MAX_OPEN_ORDERS",
    "ALREADY_IN_PORTFOLIO",
    "POSITION_ALREADY_EXISTS",
    "NO_PRICE_AVAILABLE",
    "NO_LATEST_TRADE",
    "STALE_QUOTE",
    "STALE_DATA",
    "INSUFFICIENT_BP",
    "COOLDOWN",
    "SYMBOL_DENYLIST",
    "SYMBOL_NOT_ALLOWED",
    "NO_SYMBOL",
    "BRACKET_NO_TP",
    "BRACKET_NO_SL",
    "BRACKET_TP_INVALID",
    "BRACKET_SL_INVALID",
}

# Abort entire management step — same outcome for all orders in one batch.
MANAGE_GUARD_GLOBAL_CODES: Set[str] = {
    "KILL_SWITCH",
    "LIVE_BLOCKED",
    "MARKET_CLOSED",
    "CLOCK_FAIL",
    "BP_CHECK_FAIL",
}

# Per-symbol guard codes that REMAIN blocking even for orders flagged with
# force_exit_override=True. Everything else in MANAGE_GUARD_CONTROLLED_CODES
# (MAX_QTY, MAX_NOTIONAL, COOLDOWN, MAX_POSITIONS, BRACKET_*, etc.) is
# bypassed for FORCE_EXIT — see _partition_planned_by_guard.
#
# Rationale: these three are *symbol-fundamental* (no symbol / denylisted /
# disallowed by allowlist). They cannot be safely overridden by a downstream
# risk-overlay decision; the order would not be place-able regardless.
# All other per-symbol codes are sizing/policy gates that the FORCE_EXIT
# spec explicitly authorizes bypassing. Global codes (broker init failure,
# market closed, kill switch, etc.) remain handled by MANAGE_GUARD_GLOBAL_CODES
# and continue to abort the entire batch — including any FORCE_EXIT in it.
MANAGE_GUARD_FORCE_EXIT_CRITICAL_CODES: Set[str] = {
    "NO_SYMBOL",
    "SYMBOL_DENYLIST",
    "SYMBOL_NOT_ALLOWED",
}


def _print_manage_block_summary(meta: Dict[str, Any]) -> None:
    rows = list(meta.get("blocked_rows") or [])
    if not rows and meta.get("executed", 0) in (0, None) and not meta.get("degraded_batch"):
        return
    syms = ",".join(str(x.get("symbol", "")) for x in rows)
    rsns = ",".join(f"{x.get('symbol')}:{x.get('code')}" for x in rows)
    print(
        f"[MANAGE_BLOCK_SUMMARY] blocked_symbols={syms} blocked_reasons={rsns} "
        f"executed={int(meta.get('executed', 0) or 0)} "
        f"skipped={int(meta.get('guard_skipped', 0) or 0)} "
        f"degraded_batch={bool(meta.get('degraded_batch', False))}"
    )


def _write_management_orders_subset(
    planned: List[ManagementPlannedOrder], orders_path: Path
) -> None:
    if not planned:
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
        return
    rows: List[Dict[str, Any]] = []
    for p in planned:
        lp = p.limit_price if p.limit_price is not None else ""
        _ma = str(p.management_action or "").upper()
        if _ma == "ADD":
            _opp = "ADD"
        else:
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


def _partition_planned_by_guard(
    planned: List[ManagementPlannedOrder], broker: Any, cfg: Dict[str, Any]
) -> Tuple[
    List[ManagementPlannedOrder],
    List[Dict[str, str]],
    Optional[str],
]:
    """
    Returns (allowed_orders, blocked_rows, global_abort_reason or None).
    global_abort_reason set => caller must not place any part of the batch.
    """
    if not planned or not cfg.get("require_guard_validation_for_each_order", True):
        return list(planned), [], None
    try:
        from services.execution_guard import ExecutionGuard

        guard = ExecutionGuard(broker)
    except Exception as e:
        return [], [], f"GUARD_INIT:{e}"

    allowed: List[ManagementPlannedOrder] = []
    blocked: List[Dict[str, str]] = []
    fx_overrides: List[Dict[str, str]] = []
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
        if d.ok:
            allowed.append(p)
            continue
        code = str(d.code or "UNKNOWN").strip().upper()
        sym = str(p.symbol or "").upper().strip()
        if code in MANAGE_GUARD_GLOBAL_CODES:
            return [], [], f"GLOBAL:{code}:{sym}"
        # FORCE_EXIT pre-placement bypass:
        # If this order originated from a performance-risk-overlay FORCE_EXIT
        # decision, every per-symbol guard EXCEPT the symbol-fundamental ones
        # (no symbol / denylisted / not allowed) is overridden so the order
        # survives to placement. Critical safety conditions remain in force:
        # broker unavailable -> GUARD_INIT (returned as global_abort above),
        # market closed -> MARKET_CLOSED (in GLOBAL_CODES, returned above),
        # invalid symbol -> NO_SYMBOL/SYMBOL_DENYLIST/SYMBOL_NOT_ALLOWED
        # (in this critical set), and no broker position is enforced
        # structurally upstream by build_management_plan (only LONG symbols
        # produce EXIT candidates).
        if (
            bool(getattr(p, "force_exit_override", False))
            and code not in MANAGE_GUARD_FORCE_EXIT_CRITICAL_CODES
        ):
            print(
                f"[FORCE_EXIT_PREPLACEMENT_OVERRIDE] symbol={sym} "
                f"bypassed_guard={code} action=kept_in_batch",
                flush=True,
            )
            fx_overrides.append({"symbol": sym, "code": code})
            allowed.append(p)
            continue
        blocked.append({"symbol": sym, "code": code})
    if fx_overrides:
        bypassed_summary = ",".join(f"{x['symbol']}:{x['code']}" for x in fx_overrides)
        print(
            f"[FORCE_EXIT_PREPLACEMENT_OVERRIDE_SUMMARY] count={len(fx_overrides)} "
            f"bypassed={bypassed_summary}",
            flush=True,
        )
    return allowed, blocked, None


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
                if _ma == "ADD":
                    _opp = "ADD"
                else:
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
                        "force_exit_override": bool(getattr(p, "force_exit_override", False)),
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


def _empty_manage_exec_meta() -> Dict[str, Any]:
    return {
        "blocked_rows": [],
        "guard_skipped": 0,
        "executed": 0,
        "degraded_batch": False,
        "executed_planned": [],
        "subprocess_ran": False,
        "batch_empty_noop": False,
    }


def _placement_subprocess_is_batch_empty(rc: int, combined_output: str) -> bool:
    """
    place_live_orders uses die(..., 2) for many paths; only BATCH_EMPTY is downgraded to no-op.
    """
    if int(rc) != 2:
        return False
    s = combined_output or ""
    if "[BATCH_EMPTY]" in s:
        return True
    if "[PLACE_MANAGE_ORDERS] BATCH_EMPTY" in s:
        return True
    return False


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
) -> Tuple[int, int, List[str], str, Dict[str, Any]]:
    meta = _empty_manage_exec_meta()
    if dry_run or not planned:
        return 0, 0, [], "", meta

    if broker is None:
        return 1, 0, ["BROKER_UNAVAILABLE"], "", meta

    if not cfg.get("execute_via_existing_placement_flow", True):
        return 0, 0, [], "", meta

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
                return 2, 0, rs, "", meta
        except Exception as e:
            return 1, 0, [str(e)], "", meta

    allowed, blocked_rows, global_abort = _partition_planned_by_guard(planned, broker, cfg)
    if global_abort is not None:
        meta["blocked_rows"] = list(blocked_rows)
        if str(global_abort).startswith("GUARD_INIT"):
            print(f"[MANAGE_BLOCK] {global_abort}")
            return 2, 0, [str(global_abort)], "", meta
        print(f"[MANAGE_BLOCK] execution aborted: {global_abort}")
        return 2, 0, [str(global_abort)], "", meta

    if not allowed:
        meta["blocked_rows"] = list(blocked_rows)
        meta["guard_skipped"] = len(blocked_rows)
        if blocked_rows:
            codes = {str(b.get("code", "")).strip().upper() for b in blocked_rows}
            all_controlled = codes.issubset(MANAGE_GUARD_CONTROLLED_CODES)
            meta["degraded_batch"] = bool(blocked_rows)
            _print_manage_block_summary(meta)
            if all_controlled:
                print(
                    f"[MANAGE_BLOCK] all {len(blocked_rows)} order(s) blocked by per-symbol "
                    f"guard; recognized codes: {','.join(sorted(codes))}"
                )
                return 0, 0, [], "", meta
            return (
                1,
                0,
                [f"{b.get('symbol')}:{b.get('code')}" for b in blocked_rows],
                "",
                meta,
            )
        return 0, 0, [], "", meta

    meta["blocked_rows"] = list(blocked_rows)
    meta["guard_skipped"] = len(blocked_rows)
    meta["degraded_batch"] = bool(blocked_rows)
    if blocked_rows:
        _bd = [f"{b.get('symbol')}:{b.get('code')}" for b in blocked_rows]
        print(
            f"[MANAGE_BLOCK] {len(blocked_rows)} symbol(s) skipped by guard; "
            f"placing {len(allowed)}/{len(planned)} — codes={_guard_codes_for_display(_bd)}"
        )
    _write_management_orders_subset(allowed, orders_path)

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

    meta["subprocess_ran"] = True
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if out.strip():
            print(out.rstrip(), flush=True)
        rc = int(proc.returncode)
        if _placement_subprocess_is_batch_empty(rc, out):
            meta["batch_empty_noop"] = True
            planned_n = int(len(allowed))
            print(
                "[MANAGE_NOOP] reason=BATCH_EMPTY_AFTER_FILTERS",
                flush=True,
            )
            print(f"planned={planned_n}", flush=True)
            print("executed=0", flush=True)
            print(
                "[MANAGE_NOOP] warning: placement returned no executable rows after "
                "in-flight/discipline/validation filters; treating as no-op (not a failure).",
                flush=True,
            )
            _print_manage_block_summary({**meta, "executed": 0})
            return 0, 0, [], session_id, meta
        ex = len(allowed) if rc == 0 else 0
        if rc == 0:
            meta["executed"] = ex
            meta["executed_planned"] = list(allowed)
        _print_manage_block_summary({**meta, "executed": ex})
        return rc, ex, [], session_id, meta
    except Exception as e:
        _print_manage_block_summary(meta)
        return 1, 0, [str(e)], "", meta


_AUTO_REFRESH_REASONS = frozenset({"STALE_LIFECYCLE", "STALE_EFFECTIVE"})


def _maybe_auto_refresh_lifecycle(gate_reason: str) -> bool:
    """
    Attempt to refresh stale lifecycle artifacts when the lifecycle gate
    blocked us with a refresh-able staleness reason.

    Behavior matches the spec:
      - STALE_LIFECYCLE  -> rebuild signal_lifecycle.csv (via the public
        :func:`apply_signal_lifecycle.ensure_lifecycle_fresh` API, which
        is itself a no-op when already fresh), THEN rebuild the effective
        layer (because regenerating the base lifecycle invalidates the
        derived effective layer by definition).
      - STALE_EFFECTIVE  -> rebuild signal_lifecycle_effective.csv only.
      - Anything else    -> return False without touching files.

    The effective rebuild runs as a subprocess (``python -m
    services.build_effective_lifecycle``) because that module's ``main()``
    parses ``sys.argv`` directly and would otherwise consume
    manage_positions's own argv.

    Returns True iff at least one rebuild action reported success. The
    caller MUST re-evaluate the lifecycle gate after this returns — this
    helper deliberately does not bypass the gate, only refreshes inputs.

    Each attempt emits a single [MANAGE_LIFECYCLE_AUTO_REFRESH] line so
    the audit trail is complete even when refresh fails.
    """
    reason = (gate_reason or "").strip().upper()
    if reason not in _AUTO_REFRESH_REASONS:
        return False

    refreshed_any = False

    if reason == "STALE_LIFECYCLE":
        try:
            from services.apply_signal_lifecycle import ensure_lifecycle_fresh

            did_rebuild = bool(ensure_lifecycle_fresh(verbose=False))
            print(
                f"[MANAGE_LIFECYCLE_AUTO_REFRESH] reason={reason} "
                f"action=rebuild_signal_lifecycle "
                f"refreshed={'true' if did_rebuild else 'false'}",
                flush=True,
            )
            refreshed_any = refreshed_any or did_rebuild
        except Exception as e:
            print(
                f"[MANAGE_LIFECYCLE_AUTO_REFRESH] reason={reason} "
                f"action=rebuild_signal_lifecycle refreshed=false "
                f"error={type(e).__name__}",
                flush=True,
            )

    cmd = [sys.executable, "-m", "services.build_effective_lifecycle"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if proc.stdout and proc.stdout.strip():
            print(proc.stdout.rstrip(), flush=True)
        if proc.stderr and proc.stderr.strip():
            print(proc.stderr.rstrip(), flush=True)
        rc = int(proc.returncode)
        ok = rc == 0
        print(
            f"[MANAGE_LIFECYCLE_AUTO_REFRESH] reason={reason} "
            f"action=rebuild_signal_lifecycle_effective "
            f"refreshed={'true' if ok else 'false'} rc={rc}",
            flush=True,
        )
        refreshed_any = refreshed_any or ok
    except Exception as e:
        print(
            f"[MANAGE_LIFECYCLE_AUTO_REFRESH] reason={reason} "
            f"action=rebuild_signal_lifecycle_effective refreshed=false "
            f"error={type(e).__name__}",
            flush=True,
        )

    return refreshed_any


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
    ap.add_argument(
        "--use-performance-risk-overlay",
        action="store_true",
        help=(
            "Optional, non-destructive: layer performance_risk_overlay.csv on top of "
            "EXIT/TRIM decisions. FORCE_EXIT promotes to exit; TRIM_PRIORITY raises "
            "trim likelihood. Existing logic still runs; OK is a no-op."
        ),
    )
    ap.add_argument(
        "--use-portfolio-allocation",
        action="store_true",
        help=(
            "Optional, non-destructive: layer "
            "portfolio_allocation_recommendations.csv on top of EXIT/TRIM decisions. "
            "EXIT promotes to force-exit (force_exit_override=True); TRIM raises "
            "trim likelihood. BLOCK_NEW_BUY / INCREASE / HOLD are no-ops here "
            "(handled in execution and sizing layers respectively). Existing "
            "logic still runs; BUY/ADD lifecycle stances are protected."
        ),
    )
    ap.add_argument(
        "--max-rotation-exits",
        type=int,
        default=None,
        help=(
            "Override the per-cycle cap on forced-rotation EXIT candidates "
            "(applies ONLY to the forced-rotation / weakest-position path). "
            "When provided AND > 1, candidates are re-ranked by FORCE_EXIT > "
            "worst total_pl > worst unrealized_pl > lowest edge_score > "
            "overweight > lowest rotation score, and the top N are selected "
            "(still subject to forced_rotation_max_turnover_pct). When omitted "
            "or <= 1, behaviour is unchanged. Critical safety guards (broker, "
            "market closed, denylist, no-broker-position, open-sell skip, "
            "BUY/ADD stance protection, FORCE_EXIT pre-placement bypasses) "
            "remain enforced."
        ),
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
    if bool(getattr(args, "use_performance_risk_overlay", False)):
        cfg["use_performance_risk_overlay"] = True
    if bool(getattr(args, "use_portfolio_allocation", False)):
        cfg["use_portfolio_allocation"] = True
    _mre_arg = getattr(args, "max_rotation_exits", None)
    if _mre_arg is not None:
        cfg["max_rotation_exits"] = int(_mre_arg)

    # Paper: EXIT/TRIM run through placement by default (rotation; no extra flag).
    # Live: real orders only with --execute (safety).
    if str(args.mode).lower() == "paper":
        dry_run = False
    else:
        dry_run = not bool(args.execute)
    lc_path = resolve_lifecycle_path(args.lifecycle_path)
    pos_path = Path(args.positions_path) if args.positions_path else DEFAULT_POSITIONS_SNAPSHOT
    oo_path = DEFAULT_OPEN_ORDERS_SNAPSHOT
    orders_path = Path(args.orders_path) if args.orders_path else DEFAULT_MANAGE_ORDERS

    _lg = None
    lifecycle_gate_blocked = False
    try:
        from services.lifecycle_truth import evaluate_lifecycle_gate

        _lg = evaluate_lifecycle_gate(path=lc_path)
        print(_lg.format_block())
        # Pre-placement auto-refresh: if the gate blocked us purely because
        # an upstream artifact is stale (signals newer than lifecycle, or
        # lifecycle newer than effective), regenerate the stale artifact in
        # place and re-evaluate the gate ONCE. We never bypass the gate —
        # if the re-check still says BLOCKED, we fall through to the
        # existing block path. Other block reasons (MISSING_LIFECYCLE_FILE,
        # INCONSISTENT, row-date issues, etc.) are intentionally NOT
        # auto-fixed here; those need upstream pipeline attention.
        if _lg.status == "BLOCKED" and _lg.reason in _AUTO_REFRESH_REASONS:
            refreshed = _maybe_auto_refresh_lifecycle(_lg.reason)
            if refreshed:
                print(
                    "[manage_positions] lifecycle gate re-check after auto-refresh:",
                    flush=True,
                )
                _lg = evaluate_lifecycle_gate(path=lc_path)
                print(_lg.format_block())
        if _lg.status == "BLOCKED":
            lifecycle_gate_blocked = True
            print(
                "[manage_positions] lifecycle gate BLOCKED — no EXIT/TRIM/ADD planning from lifecycle."
            )
            lifecycle = pd.DataFrame()
        else:
            lifecycle = load_lifecycle_df(lc_path)
    except Exception as e:
        print(f"[manage_positions] lifecycle gate error (loading lifecycle anyway): {e}")
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
        brs: List[str] = []
        if lifecycle_gate_blocked and _lg is not None:
            brs = [f"LIFECYCLE_GATE:{_lg.reason}; {_lg.details}"]
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
            blocked=bool(lifecycle_gate_blocked),
            block_reasons=brs,
            source_file=str(lc_path),
            orders_file=str(orders_path),
        )
        write_manage_artifacts([], [], summary, cfg, orders_path)
        append_manage_log(summary)
        if lifecycle_gate_blocked:
            print("[manage_positions] lifecycle gate BLOCKED; plan written (no lifecycle actions).")
        else:
            print("[manage_positions] Empty or missing lifecycle; plan written.")
        return 0

    open_buys = collect_symbols_with_open_buy(broker, oo_path)
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
            open_buy_syms=open_buys,
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
        # FORCE_EXIT performance-risk-overlay orders must not be silently
        # dropped by per-batch heuristics. Partition them out, run discipline
        # only on the remainder, then re-merge so the executor still sees them.
        # This is the single, audited bypass point for the three guards listed
        # in the spec (MAX_QTY/ORDER_DISCIPLINE/TRIM_COOLDOWN) — the other two
        # are structurally inapplicable to EXIT orders and so are bypassed
        # by virtue of branch separation in build_management_plan. Critical
        # broker-level guards (connectivity / market closed / invalid symbol)
        # are downstream of this and remain enforced.
        force_exit_bypass = [p for p in planned if bool(getattr(p, "force_exit_override", False))]
        for _p in force_exit_bypass:
            print(
                f"[FORCE_EXIT_OVERRIDE] symbol={_p.symbol} "
                f"bypassed_guards=MAX_QTY,ORDER_DISCIPLINE,TRIM_COOLDOWN",
                flush=True,
            )
        normal_planned = [p for p in planned if not bool(getattr(p, "force_exit_override", False))]
        try:
            from services.order_discipline import apply_discipline_to_planned_generic

            normal_planned, _disc_meta = apply_discipline_to_planned_generic(
                normal_planned,
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
        # Re-merge: FORCE_EXIT exits are placed first so they get priority
        # in any per-run order limit and cannot be displaced by lower-severity
        # actions. Order of normal_planned (post-discipline) is preserved.
        if force_exit_bypass:
            planned = force_exit_bypass + list(normal_planned)
            print(
                f"[FORCE_EXIT_OVERRIDE_SUMMARY] count={len(force_exit_bypass)} "
                f"symbols={sorted({_p.symbol for _p in force_exit_bypass})}",
                flush=True,
            )
        else:
            planned = list(normal_planned)

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
        ex_n, tr_n, fr_n, ad_n = _count_planned_actions(planned)
        print("[ROTATION_EXECUTION]")
        print("exits_executed=0")
        print("trims_executed=0")
        print("capital_freed_estimate=0.0")
        print(
            f"[MANAGE_SUMMARY] exits={ex_n} trims={tr_n} adds={ad_n} forced_rotation={fr_n} "
            f"planned={len(planned)} executed=0 skipped={skipped} blocked=0 csv_rows={len(csv_planned)}"
        )
        print(f"[MANAGE_SUMMARY] top_skip_reasons: {_fmt_top_skip_reasons(skip_reasons)}")
        return 0

    rc, executed, brs, mgmt_session, exec_meta = maybe_execute_management(
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
            manage_executed=bool(executed > 0 and exec_meta.get("subprocess_ran")),
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

    ex_n, tr_n, fr_n, ad_n = _count_planned_actions(planned)
    blocked_n = len(brs) if rc == 2 else 0
    ex_pl: List[ManagementPlannedOrder] = list(
        exec_meta.get("executed_planned") or (planned if (rc == 0 and executed > 0) else [])
    )
    ok_batch = bool(rc == 0 and executed > 0 and ex_pl)
    rex, rtr, rcap = _rotation_execution_stats(ex_pl, ok_batch)
    print("[ROTATION_EXECUTION]")
    print(f"exits_executed={rex}")
    print(f"trims_executed={rtr}")
    print(f"capital_freed_estimate={rcap:.2f}")
    print(
        f"[MANAGE_SUMMARY] exits={ex_n} trims={tr_n} adds={ad_n} forced_rotation={fr_n} "
        f"planned={len(planned)} executed={executed} skipped={skipped} blocked={blocked_n} csv_rows={len(csv_planned)}"
    )
    print(f"[MANAGE_SUMMARY] top_skip_reasons: {_fmt_top_skip_reasons(skip_reasons)}")

    if rc == 2:
        print(f"[manage_positions] BLOCKED: {brs}")
        return 2
    if rc != 0:
        if rc == 1:
            print(
                "[MANAGE] DEGRADED reason=control_block",
                flush=True,
            )
        # 1 = placement/unknown guard degradation; still a calm exit for the runner
        print(
            f"[manage_positions] placement rc={rc} (non-fatal for partial guard/degraded placement)"
        )
        return rc if rc in (1, 2, 3) else 1
    if exec_meta.get("batch_empty_noop"):
        print(
            "[manage_positions] done: placement BATCH_EMPTY after filters (MANAGE_NOOP, exit 0, not failed)"
        )
        return 0
    print("[manage_positions] placement subprocess completed OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
