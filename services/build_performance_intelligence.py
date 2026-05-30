# services/build_performance_intelligence.py
"""
Performance Intelligence (read-only analytics layer).

This module is the bridge between execution and learning. It reads existing
local artifacts produced by the trading pipeline and writes three derived
analytics outputs:

    data/results/performance_intelligence.csv             (one-row system summary)
    data/results/performance_intelligence_by_symbol.csv   (per-symbol diagnostics)
    data/results/performance_intelligence_summary.json    (rich summary blob)

It MUST NOT modify broker, execution, lifecycle, manage_positions, sizing, or
adaptation logic. It does not feed results into execution. It only measures.

Inputs (best-effort; missing or malformed files do not crash the run):

    data/results/trade_outcomes.csv
    data/results/trade_outcomes_by_symbol.csv
    data/results/pnl_diagnostics_by_symbol.csv
    data/results/positions_snapshot.csv
    data/results/live_orders_log.csv
    data/results/signal_lifecycle_effective.csv
    data/results/execution_drop_diagnostics.json

Run:

    python -m services.build_performance_intelligence
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path bootstrap (lets us run as a module from repo root)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

IN_TRADE_OUTCOMES = RESULTS_DIR / "trade_outcomes.csv"
IN_TRADE_OUTCOMES_BY_SYMBOL = RESULTS_DIR / "trade_outcomes_by_symbol.csv"
IN_PNL_DIAGNOSTICS_BY_SYMBOL = RESULTS_DIR / "pnl_diagnostics_by_symbol.csv"
IN_POSITIONS_SNAPSHOT = RESULTS_DIR / "positions_snapshot.csv"
IN_LIVE_ORDERS_LOG = RESULTS_DIR / "live_orders_log.csv"
IN_LIFECYCLE_EFFECTIVE = RESULTS_DIR / "signal_lifecycle_effective.csv"
IN_EXEC_DROP_DIAG = RESULTS_DIR / "execution_drop_diagnostics.json"

OUT_SYSTEM_CSV = RESULTS_DIR / "performance_intelligence.csv"
OUT_BY_SYMBOL_CSV = RESULTS_DIR / "performance_intelligence_by_symbol.csv"
OUT_SUMMARY_JSON = RESULTS_DIR / "performance_intelligence_summary.json"


# -----------------------------------------------------------
# Tunables (analytics-only; no behavioral effect)
# -----------------------------------------------------------
RECENT_ORDER_WINDOW_DAYS = 14
TOP_N_WINNERS = 10
TOP_N_LOSERS = 10

# Bucket thresholds in dollars of total_pl.
BUCKET_THRESHOLDS = {
    "STRONG_WINNER": 100.0,
    "WINNER": 10.0,
    "NEUTRAL_HI": 10.0,
    "NEUTRAL_LO": -10.0,
    "DRAG": -10.0,
    "HIGH_DRAG": -100.0,
}


# -----------------------------------------------------------
# Safe loaders
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[PERFORMANCE_INTELLIGENCE_WARN] {msg}", flush=True)


def _safe_read_csv(path: Path, *, label: str) -> pd.DataFrame:
    """Read a CSV defensively; return empty DataFrame on any failure."""
    try:
        if not path.is_file():
            _warn(f"missing input: {label} ({path}); using empty frame")
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, keep_default_na=False)
        except Exception:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip", keep_default_na=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        _warn(f"failed to read {label} ({path}): {type(e).__name__}: {e}")
        return pd.DataFrame()


def _safe_read_json(path: Path, *, label: str) -> Any:
    """Read a JSON file defensively; return {} on any failure."""
    try:
        if not path.is_file():
            _warn(f"missing input: {label} ({path}); using empty object")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _warn(f"failed to read {label} ({path}): {type(e).__name__}: {e}")
        return {}


# -----------------------------------------------------------
# Small utilities
# -----------------------------------------------------------
def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _norm_symbol(x: Any) -> str:
    s = str(x or "").strip().upper()
    # Normalize common share-class variants to a single canonical form.
    if s == "BRK-B":
        s = "BRK.B"
    return s


def _parse_ts_utc(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None
    try:
        return datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except Exception:
        try:
            return pd.to_datetime(txt, utc=True, errors="coerce").to_pydatetime()
        except Exception:
            return None


def _round(x: float, n: int = 4) -> float:
    try:
        return round(float(x), n)
    except Exception:
        return 0.0


def _none_or_round(x: Optional[float], n: int = 4) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, n)
    except Exception:
        return None


# -----------------------------------------------------------
# Per-source extraction
# -----------------------------------------------------------
def _build_pnl_table(pnl_df: pd.DataFrame, outcomes_by_sym_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine pnl_diagnostics_by_symbol + trade_outcomes_by_symbol into one
    canonical per-symbol PnL frame. pnl_diagnostics is preferred where
    columns overlap (it carries loss_source and severity_bucket).
    """
    cols = [
        "symbol",
        "realized_pl",
        "unrealized_pl",
        "total_pl",
        "open_qty",
        "trade_rows",
        "loss_source",
        "severity_bucket",
    ]
    if pnl_df is None or pnl_df.empty:
        base = pd.DataFrame(columns=cols)
    else:
        base = pnl_df.copy()
        base["symbol"] = base["symbol"].apply(_norm_symbol)
        for c in cols:
            if c not in base.columns:
                base[c] = "" if c in ("loss_source", "severity_bucket", "symbol") else 0.0
        base = base[cols].copy()

    if outcomes_by_sym_df is not None and not outcomes_by_sym_df.empty:
        out2 = outcomes_by_sym_df.copy()
        out2["symbol"] = out2["symbol"].apply(_norm_symbol)
        # add any symbols missing from pnl table
        existing = set(base["symbol"].astype(str))
        add = out2[~out2["symbol"].astype(str).isin(existing)].copy()
        if not add.empty:
            for c in cols:
                if c not in add.columns:
                    add[c] = "" if c in ("loss_source", "severity_bucket") else 0.0
            add = add[cols]
            base = pd.concat([base, add], ignore_index=True)

    if base.empty:
        return base

    base["realized_pl"] = base["realized_pl"].apply(_to_float)
    base["unrealized_pl"] = base["unrealized_pl"].apply(_to_float)
    base["total_pl"] = base.apply(
        lambda r: (
            _to_float(r.get("total_pl"))
            if _to_float(r.get("total_pl")) != 0
            else _to_float(r.get("realized_pl")) + _to_float(r.get("unrealized_pl"))
        ),
        axis=1,
    )
    base["open_qty"] = base["open_qty"].apply(_to_float)
    base["trade_rows"] = base["trade_rows"].apply(_to_int)
    base["loss_source"] = (
        base["loss_source"].astype(str).str.strip().str.upper().replace({"": "NONE", "NAN": "NONE"})
    )
    base["severity_bucket"] = (
        base["severity_bucket"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": "NONE", "NAN": "NONE"})
    )
    return base


def _positions_summary(positions_df: pd.DataFrame) -> pd.DataFrame:
    """Latest per-symbol open position view (qty, market_value, unrealized_pl)."""
    if positions_df is None or positions_df.empty:
        return pd.DataFrame(
            columns=["symbol", "open_qty_live", "market_value_live", "unrealized_pl_live"]
        )

    df = positions_df.copy()
    sym_col = "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
    if sym_col is None:
        return pd.DataFrame(
            columns=["symbol", "open_qty_live", "market_value_live", "unrealized_pl_live"]
        )

    df["symbol"] = df[sym_col].apply(_norm_symbol)

    if "snapshot_ts" in df.columns:
        df["__ts"] = df["snapshot_ts"].astype(str).apply(_parse_ts_utc)
        df = df.sort_values("__ts")
    df = df.groupby("symbol", as_index=False).tail(1).copy()

    df["open_qty_live"] = df.get("qty", 0).apply(_to_float)
    df["market_value_live"] = df.get("market_value", 0).apply(_to_float)
    df["unrealized_pl_live"] = df.get("unrealized_pl", 0).apply(_to_float)
    return df[["symbol", "open_qty_live", "market_value_live", "unrealized_pl_live"]].copy()


def _lifecycle_per_symbol(lifecycle_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce lifecycle_effective to one row per ticker (latest stance/state)."""
    if lifecycle_df is None or lifecycle_df.empty:
        return pd.DataFrame(
            columns=["symbol", "current_lifecycle_stance", "effective_position_state"]
        )
    df = lifecycle_df.copy()
    sym_col = "ticker" if "ticker" in df.columns else ("symbol" if "symbol" in df.columns else None)
    if sym_col is None:
        return pd.DataFrame(
            columns=["symbol", "current_lifecycle_stance", "effective_position_state"]
        )
    df["symbol"] = df[sym_col].apply(_norm_symbol)

    # latest row per symbol by generated_at_utc (fallback: row order)
    if "generated_at_utc" in df.columns:
        df["__ts"] = df["generated_at_utc"].astype(str).apply(_parse_ts_utc)
        df = df.sort_values("__ts")
    df = df.groupby("symbol", as_index=False).tail(1).copy()

    stance_col = (
        "effective_stance"
        if "effective_stance" in df.columns
        else ("stance" if "stance" in df.columns else None)
    )
    state_col = (
        "effective_position_state"
        if "effective_position_state" in df.columns
        else ("position_state" if "position_state" in df.columns else None)
    )
    df["current_lifecycle_stance"] = (
        df[stance_col].astype(str).str.strip().str.upper() if stance_col else ""
    )
    df["effective_position_state"] = (
        df[state_col].astype(str).str.strip().str.upper() if state_col else ""
    )
    return df[["symbol", "current_lifecycle_stance", "effective_position_state"]].copy()


def _orderlog_metrics(orders_df: pd.DataFrame, recent_window_days: int) -> pd.DataFrame:
    """
    Derive activity & decision-effectiveness counts per symbol from
    live_orders_log.csv. We deliberately stay defensive because this file is
    known to contain some legacy/corrupt rows (handled by repair elsewhere).

    Returns columns:
        symbol,
        recent_order_count, recent_buy_count, recent_sell_count,
        buy_count, sell_count,
        filled_count, open_order_count,
        add_count, exit_count, trim_count
    """
    cols_out = [
        "symbol",
        "recent_order_count",
        "recent_buy_count",
        "recent_sell_count",
        "buy_count",
        "sell_count",
        "filled_count",
        "open_order_count",
        "add_count",
        "exit_count",
        "trim_count",
    ]
    if orders_df is None or orders_df.empty:
        return pd.DataFrame(columns=cols_out)

    df = orders_df.copy()

    needed = ["timestamp", "action", "symbol", "side", "qty", "status", "filled_qty"]
    for c in needed:
        if c not in df.columns:
            df[c] = ""

    df["action_l"] = df["action"].astype(str).str.strip().str.lower()
    df["side_l"] = df["side"].astype(str).str.strip().str.lower()
    df["status_l"] = df["status"].astype(str).str.strip().str.lower()
    df["symbol_n"] = df["symbol"].apply(_norm_symbol)
    df["qty_n"] = df["qty"].apply(_to_int)
    df["filled_qty_n"] = df["filled_qty"].apply(_to_int)
    df["__ts"] = df["timestamp"].astype(str).apply(_parse_ts_utc)

    # Keep only rows that look like real order rows (drop the legacy submit-shift junk).
    valid_action = df["action_l"].isin(["submit", "poll", "cancel"])
    valid_side = df["side_l"].isin(["buy", "sell"])
    valid_symbol = df["symbol_n"].astype(bool) & df["symbol_n"].str.match(
        r"^[A-Z][A-Z0-9.\-^]{0,9}$"
    )
    df = df[valid_action & valid_side & valid_symbol].copy()
    if df.empty:
        return pd.DataFrame(columns=cols_out)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=int(max(1, recent_window_days)))

    # ---- Per-symbol aggregates over SUBMIT rows (the canonical order-creation event)
    submits = df[df["action_l"] == "submit"].copy()
    base = pd.DataFrame({"symbol": sorted(submits["symbol_n"].unique())})

    buys_per_sym = submits[submits["side_l"] == "buy"].groupby("symbol_n").size().to_dict()
    sells_per_sym = submits[submits["side_l"] == "sell"].groupby("symbol_n").size().to_dict()
    base["buy_count"] = base["symbol"].map(buys_per_sym).fillna(0).astype(int)
    base["sell_count"] = base["symbol"].map(sells_per_sym).fillna(0).astype(int)

    recent_mask = submits["__ts"].notna() & (submits["__ts"] >= cutoff)
    recent_submits = submits[recent_mask]
    recent_per_sym = recent_submits.groupby("symbol_n").size().to_dict()
    recent_buys_per_sym = (
        recent_submits[recent_submits["side_l"] == "buy"].groupby("symbol_n").size().to_dict()
    )
    recent_sells_per_sym = (
        recent_submits[recent_submits["side_l"] == "sell"].groupby("symbol_n").size().to_dict()
    )
    base["recent_order_count"] = base["symbol"].map(recent_per_sym).fillna(0).astype(int)
    base["recent_buy_count"] = base["symbol"].map(recent_buys_per_sym).fillna(0).astype(int)
    base["recent_sell_count"] = base["symbol"].map(recent_sells_per_sym).fillna(0).astype(int)

    # ---- Filled / open order counts come from poll-side last status per order_id
    # Use the LAST status observed per (symbol, order_id).
    if "order_id" not in df.columns:
        df["order_id"] = ""
    last_status_df = (
        df[df["__ts"].notna()]
        .sort_values("__ts")
        .groupby(["symbol_n", "order_id"], as_index=False)
        .tail(1)
    )
    closed_terminal = {
        "filled",
        "canceled",
        "expired",
        "rejected",
        "stopped",
        "suspended",
        "calculated",
        "replaced",
        "done_for_day",
    }
    last_status_df["is_filled"] = last_status_df["status_l"].eq("filled").astype(int)
    last_status_df["is_open"] = (~last_status_df["status_l"].isin(closed_terminal)).astype(
        int
    ) & last_status_df["order_id"].astype(str).str.strip().ne("").astype(int)

    fc = last_status_df.groupby("symbol_n")["is_filled"].sum().astype(int).to_dict()
    oc = last_status_df.groupby("symbol_n")["is_open"].sum().astype(int).to_dict()

    base["filled_count"] = base["symbol"].map(fc).fillna(0).astype(int)
    base["open_order_count"] = base["symbol"].map(oc).fillna(0).astype(int)

    # ---- ADD / EXIT / TRIM heuristic
    # Replay filled events per symbol in chronological order, tracking net qty.
    # - buy when net_qty > 0  -> ADD
    # - buy when net_qty == 0 -> initial BUY (not ADD)
    # - sell that drops net_qty to 0 -> EXIT
    # - sell that leaves net_qty > 0 -> TRIM
    add_counts: Dict[str, int] = {}
    exit_counts: Dict[str, int] = {}
    trim_counts: Dict[str, int] = {}

    fills_only = last_status_df[last_status_df["is_filled"] == 1].copy()
    fills_only["use_qty"] = fills_only.apply(
        lambda r: int(r["filled_qty_n"]) if int(r["filled_qty_n"]) > 0 else int(r["qty_n"]),
        axis=1,
    )
    fills_only = fills_only[fills_only["use_qty"] > 0]
    fills_only = fills_only.sort_values("__ts")

    for sym, g in fills_only.groupby("symbol_n"):
        net = 0
        for _, row in g.iterrows():
            q = int(row["use_qty"])
            side = str(row["side_l"])
            if side == "buy":
                if net > 0:
                    add_counts[sym] = add_counts.get(sym, 0) + 1
                # initial buy: no add
                net += q
            elif side == "sell":
                new_net = max(0, net - q)
                if new_net == 0 and net > 0:
                    exit_counts[sym] = exit_counts.get(sym, 0) + 1
                elif new_net > 0:
                    trim_counts[sym] = trim_counts.get(sym, 0) + 1
                net = new_net

    base["add_count"] = base["symbol"].map(add_counts).fillna(0).astype(int)
    base["exit_count"] = base["symbol"].map(exit_counts).fillna(0).astype(int)
    base["trim_count"] = base["symbol"].map(trim_counts).fillna(0).astype(int)

    return base[cols_out].copy()


# -----------------------------------------------------------
# Bucketing
# -----------------------------------------------------------
def _performance_bucket(
    total_pl: float, severity: str, drag_flag_like: bool, has_open_position: bool
) -> str:
    sev = (severity or "").upper()
    if total_pl >= BUCKET_THRESHOLDS["STRONG_WINNER"]:
        return "STRONG_WINNER"
    if total_pl >= BUCKET_THRESHOLDS["WINNER"]:
        return "WINNER"
    if total_pl <= BUCKET_THRESHOLDS["HIGH_DRAG"] or sev == "HIGH":
        return "HIGH_DRAG"
    if total_pl <= BUCKET_THRESHOLDS["DRAG"] or drag_flag_like:
        return "DRAG"
    if (sev == "MED" and total_pl < 0) or (has_open_position and total_pl < 0):
        return "WATCH"
    return "NEUTRAL"


# -----------------------------------------------------------
# Build per-symbol intelligence
# -----------------------------------------------------------
def build_by_symbol(
    pnl_df: pd.DataFrame,
    outcomes_by_sym_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    lifecycle_df: pd.DataFrame,
    orders_df: pd.DataFrame,
) -> pd.DataFrame:
    pnl_table = _build_pnl_table(pnl_df, outcomes_by_sym_df)
    pos_table = _positions_summary(positions_df)
    life_table = _lifecycle_per_symbol(lifecycle_df)
    ord_table = _orderlog_metrics(orders_df, RECENT_ORDER_WINDOW_DAYS)

    # Build full universe of symbols from all sources.
    all_syms = set()
    for d in (pnl_table, pos_table, life_table, ord_table):
        if d is not None and not d.empty and "symbol" in d.columns:
            all_syms.update([s for s in d["symbol"].astype(str) if s])
    if not all_syms:
        return pd.DataFrame(
            columns=[
                "symbol",
                "realized_pl",
                "unrealized_pl",
                "total_pl",
                "open_qty",
                "trade_rows",
                "loss_source",
                "severity_bucket",
                "current_lifecycle_stance",
                "effective_position_state",
                "recent_order_count",
                "recent_buy_count",
                "recent_sell_count",
                "buy_count",
                "sell_count",
                "filled_count",
                "open_order_count",
                "add_count",
                "exit_count",
                "trim_count",
                "performance_bucket",
            ]
        )

    base = pd.DataFrame({"symbol": sorted(all_syms)})

    if not pnl_table.empty:
        base = base.merge(pnl_table, on="symbol", how="left")
    if not pos_table.empty:
        base = base.merge(pos_table, on="symbol", how="left")
    if not life_table.empty:
        base = base.merge(life_table, on="symbol", how="left")
    if not ord_table.empty:
        base = base.merge(ord_table, on="symbol", how="left")

    # Fill defaults for every output column.
    defaults_num = [
        "realized_pl",
        "unrealized_pl",
        "total_pl",
        "open_qty",
        "open_qty_live",
        "market_value_live",
        "unrealized_pl_live",
        "trade_rows",
        "recent_order_count",
        "recent_buy_count",
        "recent_sell_count",
        "buy_count",
        "sell_count",
        "filled_count",
        "open_order_count",
        "add_count",
        "exit_count",
        "trim_count",
    ]
    for c in defaults_num:
        if c not in base.columns:
            base[c] = 0
        base[c] = base[c].apply(_to_float)

    defaults_str = [
        "loss_source",
        "severity_bucket",
        "current_lifecycle_stance",
        "effective_position_state",
    ]
    for c in defaults_str:
        if c not in base.columns:
            base[c] = ""
        base[c] = base[c].astype(str).str.strip().str.upper().replace({"NAN": ""})

    # Prefer live broker open_qty when available; fall back to PnL table open_qty.
    base["open_qty"] = base.apply(
        lambda r: (
            r["open_qty_live"]
            if _to_float(r.get("open_qty_live")) > 0
            else _to_float(r.get("open_qty"))
        ),
        axis=1,
    )
    # If realized/unrealized PnL are both zero but live unrealized exists, use it.
    base["unrealized_pl"] = base.apply(
        lambda r: (
            r["unrealized_pl"]
            if _to_float(r["unrealized_pl"]) != 0
            else _to_float(r.get("unrealized_pl_live"))
        ),
        axis=1,
    )
    base["total_pl"] = base["realized_pl"].apply(_to_float) + base["unrealized_pl"].apply(_to_float)

    # Performance bucket
    has_open = base["open_qty"].apply(_to_float) > 0
    drag_like = base["loss_source"].isin(["REALIZED", "UNREALIZED"])
    base["performance_bucket"] = [
        _performance_bucket(
            _to_float(t),
            str(s),
            bool(d),
            bool(o),
        )
        for t, s, d, o in zip(base["total_pl"], base["severity_bucket"], drag_like, has_open)
    ]

    # Final column order
    out_cols = [
        "symbol",
        "realized_pl",
        "unrealized_pl",
        "total_pl",
        "open_qty",
        "trade_rows",
        "loss_source",
        "severity_bucket",
        "current_lifecycle_stance",
        "effective_position_state",
        "recent_order_count",
        "recent_buy_count",
        "recent_sell_count",
        "buy_count",
        "sell_count",
        "filled_count",
        "open_order_count",
        "add_count",
        "exit_count",
        "trim_count",
        "performance_bucket",
    ]
    for c in out_cols:
        if c not in base.columns:
            base[c] = 0 if c in defaults_num else ""
    out = base[out_cols].copy()

    # Tidy numeric rounding
    for c in ("realized_pl", "unrealized_pl", "total_pl"):
        out[c] = out[c].apply(lambda v: _round(v, 4))
    for c in (
        "open_qty",
        "trade_rows",
        "recent_order_count",
        "recent_buy_count",
        "recent_sell_count",
        "buy_count",
        "sell_count",
        "filled_count",
        "open_order_count",
        "add_count",
        "exit_count",
        "trim_count",
    ):
        out[c] = out[c].apply(_to_int)

    out = out.sort_values(["total_pl", "symbol"], ascending=[False, True]).reset_index(drop=True)
    return out


# -----------------------------------------------------------
# Build summary objects
# -----------------------------------------------------------
def _top_n(by_sym: pd.DataFrame, n: int, ascending: bool) -> List[Dict[str, Any]]:
    if by_sym.empty:
        return []
    sub = by_sym.sort_values("total_pl", ascending=ascending).head(int(n))
    return [
        {
            "symbol": str(r["symbol"]),
            "total_pl": _round(_to_float(r["total_pl"]), 4),
            "realized_pl": _round(_to_float(r["realized_pl"]), 4),
            "unrealized_pl": _round(_to_float(r["unrealized_pl"]), 4),
            "performance_bucket": str(r["performance_bucket"]),
        }
        for _, r in sub.iterrows()
    ]


def build_summary(by_sym: pd.DataFrame, exec_drop: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (summary_dict_for_json, one_row_dict_for_csv).
    """
    notes: List[str] = []
    if by_sym is None or by_sym.empty:
        notes.append("no per-symbol data available; outputs are empty placeholders")
        zero_summary = {
            "generated_at_utc": _now_iso_utc(),
            "total_symbols": 0,
            "winners": 0,
            "losers": 0,
            "high_drag_symbols": 0,
            "total_realized_pl": 0.0,
            "total_unrealized_pl": 0.0,
            "total_combined_pl": 0.0,
            "best_symbol": "",
            "worst_symbol": "",
            "open_positions": 0,
            "symbols_to_monitor": [],
            "top_winners": [],
            "top_losers": [],
            "execution_drop_observed": bool(
                isinstance(exec_drop, dict) and int(exec_drop.get("dropped_orders", 0) or 0) > 0
            ),
            "notes": notes,
        }
        zero_row = {
            "generated_at_utc": zero_summary["generated_at_utc"],
            "total_symbols": 0,
            "winners": 0,
            "losers": 0,
            "high_drag_symbols": 0,
            "total_realized_pl": 0.0,
            "total_unrealized_pl": 0.0,
            "total_combined_pl": 0.0,
            "best_symbol": "",
            "worst_symbol": "",
            "open_positions": 0,
        }
        return zero_summary, zero_row

    df = by_sym.copy()
    df["total_pl"] = df["total_pl"].apply(_to_float)
    df["realized_pl"] = df["realized_pl"].apply(_to_float)
    df["unrealized_pl"] = df["unrealized_pl"].apply(_to_float)
    df["open_qty"] = df["open_qty"].apply(_to_float)

    total_symbols = int(len(df))
    winners = int((df["total_pl"] > 0).sum())
    losers = int((df["total_pl"] < 0).sum())
    high_drag = int((df["performance_bucket"] == "HIGH_DRAG").sum())
    open_positions = int((df["open_qty"] > 0).sum())

    best_row = df.sort_values("total_pl", ascending=False).head(1)
    worst_row = df.sort_values("total_pl", ascending=True).head(1)
    best_symbol = str(best_row["symbol"].iloc[0]) if not best_row.empty else ""
    worst_symbol = str(worst_row["symbol"].iloc[0]) if not worst_row.empty else ""

    # Symbols to monitor: WATCH / DRAG / HIGH_DRAG, plus any open position with negative unrealized.
    monitor_mask = df["performance_bucket"].isin(["WATCH", "DRAG", "HIGH_DRAG"]) | (
        (df["open_qty"] > 0) & (df["unrealized_pl"] < 0)
    )
    symbols_to_monitor = sorted(set(df.loc[monitor_mask, "symbol"].astype(str).tolist()))

    total_realized = _round(float(df["realized_pl"].sum()), 4)
    total_unrealized = _round(float(df["unrealized_pl"].sum()), 4)
    total_combined = _round(total_realized + total_unrealized, 4)

    # Loss-source breakdown for the notes section (analytics only).
    loss_breakdown = (
        df.assign(loss_source=df["loss_source"].astype(str).str.upper())
        .groupby("loss_source", as_index=False)["total_pl"]
        .sum()
    )
    realized_loss_share = float(
        loss_breakdown.loc[loss_breakdown["loss_source"] == "REALIZED", "total_pl"].sum()
    )
    unrealized_loss_share = float(
        loss_breakdown.loc[loss_breakdown["loss_source"] == "UNREALIZED", "total_pl"].sum()
    )

    # Decision-effectiveness rollups (analytics only)
    decision_rollup = {
        "total_buys": int(df["buy_count"].sum()),
        "total_sells": int(df["sell_count"].sum()),
        "total_adds": int(df["add_count"].sum()),
        "total_exits": int(df["exit_count"].sum()),
        "total_trims": int(df["trim_count"].sum()),
        "total_filled": int(df["filled_count"].sum()),
        "total_open_orders": int(df["open_order_count"].sum()),
    }

    if isinstance(exec_drop, dict):
        dropped = int(exec_drop.get("dropped_orders", 0) or 0)
        if dropped > 0:
            notes.append(
                f"execution_drop_diagnostics: {dropped} planned orders not submitted "
                f"in last execute_trades run (see drop_reasons)"
            )
        if exec_drop.get("blocked"):
            notes.append("execution_drop_diagnostics: most recent run reported BLOCKED state")

    if realized_loss_share < 0:
        notes.append(
            f"loss composition: realized={_round(realized_loss_share,2)} "
            f"unrealized={_round(unrealized_loss_share,2)} (negative=loss)"
        )
    if high_drag > 0:
        notes.append(f"{high_drag} HIGH_DRAG symbol(s) detected; review for tighter risk control")
    if not symbols_to_monitor:
        notes.append("no symbols currently flagged for monitoring")

    summary = {
        "generated_at_utc": _now_iso_utc(),
        "total_symbols": total_symbols,
        "winners": winners,
        "losers": losers,
        "high_drag_symbols": high_drag,
        "total_realized_pl": total_realized,
        "total_unrealized_pl": total_unrealized,
        "total_combined_pl": total_combined,
        "best_symbol": best_symbol,
        "worst_symbol": worst_symbol,
        "open_positions": open_positions,
        "symbols_to_monitor": symbols_to_monitor,
        "top_winners": _top_n(df, TOP_N_WINNERS, ascending=False),
        "top_losers": _top_n(df, TOP_N_LOSERS, ascending=True),
        "decision_rollup": decision_rollup,
        "execution_drop_observed": bool(
            isinstance(exec_drop, dict) and int(exec_drop.get("dropped_orders", 0) or 0) > 0
        ),
        "notes": notes,
    }

    one_row = {
        "generated_at_utc": summary["generated_at_utc"],
        "total_symbols": total_symbols,
        "winners": winners,
        "losers": losers,
        "high_drag_symbols": high_drag,
        "total_realized_pl": total_realized,
        "total_unrealized_pl": total_unrealized,
        "total_combined_pl": total_combined,
        "best_symbol": best_symbol,
        "worst_symbol": worst_symbol,
        "open_positions": open_positions,
    }
    return summary, one_row


# -----------------------------------------------------------
# IO writers
# -----------------------------------------------------------
def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main() -> int:
    print("[PERFORMANCE_INTELLIGENCE] starting (read-only analytics)", flush=True)

    pnl_df = _safe_read_csv(IN_PNL_DIAGNOSTICS_BY_SYMBOL, label="pnl_diagnostics_by_symbol.csv")
    outcomes_df = _safe_read_csv(
        IN_TRADE_OUTCOMES, label="trade_outcomes.csv"
    )  # not joined per-row, but loaded for trace
    outcomes_sym_df = _safe_read_csv(
        IN_TRADE_OUTCOMES_BY_SYMBOL, label="trade_outcomes_by_symbol.csv"
    )
    positions_df = _safe_read_csv(IN_POSITIONS_SNAPSHOT, label="positions_snapshot.csv")
    orders_df = _safe_read_csv(IN_LIVE_ORDERS_LOG, label="live_orders_log.csv")
    lifecycle_df = _safe_read_csv(IN_LIFECYCLE_EFFECTIVE, label="signal_lifecycle_effective.csv")
    exec_drop = _safe_read_json(IN_EXEC_DROP_DIAG, label="execution_drop_diagnostics.json")

    # Touch outcomes_df only to allow extending later (kept loaded for parity with spec).
    _ = len(outcomes_df) if isinstance(outcomes_df, pd.DataFrame) else 0

    by_sym = build_by_symbol(
        pnl_df=pnl_df,
        outcomes_by_sym_df=outcomes_sym_df,
        positions_df=positions_df,
        lifecycle_df=lifecycle_df,
        orders_df=orders_df,
    )

    summary, one_row = build_summary(by_sym, exec_drop)

    try:
        _atomic_write_csv(by_sym, OUT_BY_SYMBOL_CSV)
    except Exception as e:
        _warn(f"failed to write {OUT_BY_SYMBOL_CSV.name}: {type(e).__name__}: {e}")
        return 2

    try:
        _atomic_write_csv(pd.DataFrame([one_row]), OUT_SYSTEM_CSV)
    except Exception as e:
        _warn(f"failed to write {OUT_SYSTEM_CSV.name}: {type(e).__name__}: {e}")
        return 2

    try:
        _atomic_write_json(summary, OUT_SUMMARY_JSON)
    except Exception as e:
        _warn(f"failed to write {OUT_SUMMARY_JSON.name}: {type(e).__name__}: {e}")
        return 2

    print(
        "[PERFORMANCE_INTELLIGENCE] "
        f"total_symbols={summary['total_symbols']} "
        f"winners={summary['winners']} "
        f"losers={summary['losers']} "
        f"high_drag_symbols={summary['high_drag_symbols']} "
        f"best_symbol={summary['best_symbol'] or '-'} "
        f"worst_symbol={summary['worst_symbol'] or '-'} "
        f"total_combined_pl={summary['total_combined_pl']}",
        flush=True,
    )
    print(
        "[PERFORMANCE_INTELLIGENCE_OUT] "
        f"by_symbol={OUT_BY_SYMBOL_CSV.as_posix()} "
        f"system={OUT_SYSTEM_CSV.as_posix()} "
        f"summary={OUT_SUMMARY_JSON.as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
