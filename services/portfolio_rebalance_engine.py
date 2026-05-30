"""
Portfolio Rebalance Decision Engine — Step 6 of the WATCH → DEPLOY funnel.

Reads:
    data/results/portfolio_construction_recommendations.csv
    data/results/portfolio_construction_summary.json
    data/results/positions_snapshot.csv
    data/results/capital_deployment_summary.json
    data/results/performance_risk_overlay.csv

Writes:
    data/results/portfolio_rebalance_plan.csv
    data/results/portfolio_rebalance_summary.json

Purpose
-------
Step 5 produced a target portfolio (per-symbol target_weight_pct). This
engine produces the *exact set of trades* required to transform the
current portfolio into that target portfolio, sequenced and prioritized
so capital is freed (EXITs/TRIMs) before it is redeployed (ADDs/BUYs).

It answers:

    "What trades are required to transform the current portfolio
    into the target portfolio?"

Per row it emits a `rebalance_action` (FULL_EXIT / SELL / TRIM / ADD /
BUY_NEW / HOLD / NO_ACTION), a dollar `rebalance_amount_usd`, a
normalized `priority` (risk exits first), an `execution_order` integer
following the spec's sequence (EXIT/SELL → TRIM → ADD → BUY_NEW →
HOLD), and `execution_ready` for downstream consumers.

Safety
------
* Read-only. No orders, no broker calls, no mutation of execute_trades
  or manage_positions. The output is purely a trade *plan*.
* Missing inputs warn and continue (empty maps).
* Atomic writes via `*.tmp` + `os.replace`.
* main() returns 0 on success, 2 only on output-write failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_CONSTRUCTION_CSV = RESULTS_DIR / "portfolio_construction_recommendations.csv"
DEFAULT_CONSTRUCTION_JSON = RESULTS_DIR / "portfolio_construction_summary.json"
DEFAULT_POSITIONS_CSV = RESULTS_DIR / "positions_snapshot.csv"
DEFAULT_CAPITAL_SUMMARY_JSON = RESULTS_DIR / "capital_deployment_summary.json"
DEFAULT_RISK_OVERLAY_CSV = RESULTS_DIR / "performance_risk_overlay.csv"

DEFAULT_OUTPUT_CSV = RESULTS_DIR / "portfolio_rebalance_plan.csv"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "portfolio_rebalance_summary.json"

# -----------------------------------------------------------
# Tunables
# -----------------------------------------------------------
# Trades below this notional are ignored as noise (and execution_ready=False).
MIN_TRADE_USD_DEFAULT = 25.0

# Step 11 runtime policy override (optional, additive).
# When data/results/runtime_policy.json exists, ROTATION_PRESSURE is
# overridden for the current cycle. Default 0.50 is neutral -- the
# priority multiplier is 1.0 at neutral, so behaviour is identical to
# pre-Step-11 unless an override is supplied.
DEFAULT_RUNTIME_POLICY_JSON = RESULTS_DIR / "runtime_policy.json"
ROTATION_PRESSURE_DEFAULT = 0.50
ROTATION_PRESSURE = ROTATION_PRESSURE_DEFAULT
# Sell/trim priority multiplier as a function of rotation_pressure:
#   multiplier = 1.0 + ROTATION_PRESSURE_GAIN * (pressure - 0.5)
# Gain 0.40 means pressure=0.85 -> +14% sell urgency,
# pressure=0.40 -> -4% sell urgency, pressure=0.50 -> no change.
ROTATION_PRESSURE_GAIN = 0.40

# Construction-layer action labels (spec §4 of Step 5)
CONSTR_ACTION_OPEN = "OPEN_POSITION"
CONSTR_ACTION_ADD = "ADD_TO_POSITION"
CONSTR_ACTION_HOLD = "HOLD"
CONSTR_ACTION_TRIM = "TRIM"
CONSTR_ACTION_EXIT = "EXIT"
CONSTR_ACTION_BLOCK = "BLOCK"

# Rebalance-layer action labels (spec §2)
REBAL_BUY_NEW = "BUY_NEW"
REBAL_ADD = "ADD"
REBAL_TRIM = "TRIM"
REBAL_SELL = "SELL"
REBAL_FULL_EXIT = "FULL_EXIT"
REBAL_HOLD = "HOLD"
REBAL_NO_ACTION = "NO_ACTION"

# Execution sequence tiers (spec §4). Lower = earlier in plan.
EXECUTION_SEQUENCE: Dict[str, int] = {
    REBAL_FULL_EXIT: 1,
    REBAL_SELL: 1,
    REBAL_TRIM: 2,
    REBAL_ADD: 3,
    REBAL_BUY_NEW: 4,
    REBAL_HOLD: 5,
    REBAL_NO_ACTION: 6,
}

# Priority tier bands (spec §3). Within each band we add a small bonus
# proportional to deploy_priority / urgency so high-conviction or
# high-risk items rank first within their tier.
PRIORITY_BAND_FULL_EXIT = (0.92, 1.00)
PRIORITY_BAND_FORCED_TRIM = (0.75, 0.92)
PRIORITY_BAND_HIGH_CONVICTION_DEPLOY = (0.55, 0.75)
PRIORITY_BAND_MODERATE_DEPLOY = (0.35, 0.55)
PRIORITY_BAND_HOLD = (0.05, 0.20)
PRIORITY_NO_ACTION = 0.0

# Risk tokens (mirror upstream conventions)
RISK_FORCE_EXIT = "FORCE_EXIT"
RISK_TRIM_PRIORITY = "TRIM_PRIORITY"
RISK_BLOCK_NEW_BUY = "BLOCK_NEW_BUY"
RISK_OK = "OK"

# Deploy_priority gate that splits "high conviction" from "moderate".
HIGH_CONVICTION_DEPLOY_PRIORITY = 0.60

OUTPUT_COLUMNS = [
    "ticker",
    "current_weight_pct",
    "target_weight_pct",
    "delta_weight_pct",
    "current_value_usd",
    "target_value_usd",
    "rebalance_amount_usd",
    "portfolio_action",
    "rebalance_action",
    "priority",
    "execution_order",
    "execution_ready",
    "reason",
]


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[PORTFOLIO_REBALANCE_WARN] {msg}", flush=True)


def _safe_read_csv(path: Path, *, label: str) -> pd.DataFrame:
    try:
        if not path.is_file():
            _warn(f"missing input: {label} ({path}); continuing without it")
            return pd.DataFrame()
    except OSError as e:
        _warn(f"stat failed for {label} ({path}): {type(e).__name__}: {e}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except Exception:
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip", keep_default_na=False)
        except Exception as e:
            _warn(f"failed to read {label} ({path}): {type(e).__name__}: {e}")
            return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _safe_read_json(path: Path, *, label: str) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception as e:
        _warn(f"failed to read {label} ({path}): {type(e).__name__}: {e}")
        return None


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_safe)
    os.replace(tmp, path)


def _json_safe(o: Any) -> Any:
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            return str(o)
    try:
        return float(o)
    except Exception:
        return str(o)


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# -----------------------------------------------------------
# Coercion helpers
# -----------------------------------------------------------
def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _to_float_or_zero(x: Any) -> float:
    v = _to_float(x)
    return 0.0 if v is None else v


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x or "").strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_symbol(x: Any) -> str:
    s = str(x or "").strip().upper()
    if s == "BRK-B":
        s = "BRK.B"
    return s


def _norm_upper(x: Any) -> str:
    return str(x or "").strip().upper()


def _pick_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _risk_components(risk_flag: str) -> List[str]:
    if not risk_flag:
        return []
    parts = [p.strip().upper() for p in str(risk_flag).split("|")]
    return [p for p in parts if p and p != RISK_OK]


def _band_score(band: Tuple[float, float], fraction: float) -> float:
    """Map fraction in [0,1] linearly into the [lo, hi] band."""
    lo, hi = band
    return round(lo + (hi - lo) * _clamp(fraction, 0.0, 1.0), 6)


# -----------------------------------------------------------
# Loaders
# -----------------------------------------------------------
def _load_construction_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Pull every row from Step 5's CSV; this is the primary input."""
    if df is None or df.empty:
        return []
    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        _warn("construction CSV missing ticker/symbol column; no plan")
        return []
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        rows.append(
            {
                "ticker": sym,
                "current_weight_pct": _to_float_or_zero(r.get("current_weight_pct")),
                "target_weight_pct": _to_float_or_zero(r.get("target_weight_pct")),
                "delta_weight_pct": _to_float_or_zero(r.get("delta_weight_pct")),
                "target_position_size_usd": _to_float_or_zero(r.get("target_position_size_usd")),
                "deploy_priority": _to_float_or_zero(r.get("deploy_priority")),
                "portfolio_action": _norm_upper(r.get("portfolio_action")),
                "construction_reason": str(r.get("reason") or ""),
                "sector_bucket": str(r.get("sector_bucket") or ""),
            }
        )
    return rows


def _load_positions_value_map(df: pd.DataFrame) -> Tuple[Dict[str, float], float]:
    """{ticker: market_value} + sum-of-positions for the broker-truth side."""
    out: Dict[str, float] = {}
    total = 0.0
    if df is None or df.empty:
        return out, 0.0
    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    mv_col = _pick_first_present(df, ("market_value", "value"))
    qty_col = _pick_first_present(df, ("qty", "qty_available"))
    if not sym_col or not mv_col:
        return out, 0.0
    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        mv = _to_float_or_zero(r.get(mv_col))
        qty = _to_float_or_zero(r.get(qty_col)) if qty_col else 1.0
        if mv <= 0 or qty <= 0:
            continue
        out[sym] = mv
        total += mv
    return out, total


def _load_risk_overlay_map(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if df is None or df.empty:
        return out
    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        return out
    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        rf = _norm_upper(r.get("risk_flag"))
        if rf:
            out[sym] = rf
    return out


def _load_construction_summary(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not d:
        return {}
    return {
        "total_portfolio_value": _to_float(d.get("total_portfolio_value")),
        "cash_reserve_pct": _to_float(d.get("cash_reserve_pct")),
        "portfolio_construction_score": _to_float(d.get("portfolio_construction_score")),
    }


def _load_capital_summary(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not d:
        return {}
    return {
        "deployable_capital_estimate": _to_float(d.get("deployable_capital_estimate")),
        "cash_estimate": _to_float(d.get("cash_estimate")),
        "max_positions": int(_to_float_or_zero(d.get("max_positions")) or 0),
        "available_slots": int(_to_float_or_zero(d.get("available_slots")) or 0),
    }


# -----------------------------------------------------------
# Decision logic
# -----------------------------------------------------------
def _map_to_rebalance_action(
    *,
    portfolio_action: str,
    current_value_usd: float,
    target_value_usd: float,
) -> str:
    """
    Translate Step 5's portfolio_action into a concrete trade action.

    Precedence: EXIT and BLOCK are absolute (regardless of dollar deltas).
    OPEN/ADD/TRIM map directly. HOLD with non-zero delta surfaces as
    SELL (target<current) or ADD (target>current) so reporting reflects
    reality even if Step 5 didn't tag it explicitly.
    """
    if portfolio_action == CONSTR_ACTION_EXIT:
        return REBAL_FULL_EXIT
    if portfolio_action == CONSTR_ACTION_BLOCK:
        return REBAL_NO_ACTION
    if portfolio_action == CONSTR_ACTION_TRIM:
        return REBAL_TRIM
    if portfolio_action == CONSTR_ACTION_OPEN:
        return REBAL_BUY_NEW
    if portfolio_action == CONSTR_ACTION_ADD:
        return REBAL_ADD
    if portfolio_action == CONSTR_ACTION_HOLD:
        delta = target_value_usd - current_value_usd
        # A "HOLD" row should generally have zero delta — but if Step 5
        # produced a slightly different target weight (e.g. rounding), we
        # surface the action so the plan reflects what would actually
        # need to trade. The min_trade_usd gate below filters noise.
        if delta > 0 and current_value_usd > 0:
            return REBAL_ADD
        if delta < 0 and current_value_usd > 0:
            return REBAL_SELL
        return REBAL_HOLD
    return REBAL_NO_ACTION


def _priority_for(
    *,
    rebalance_action: str,
    portfolio_action: str,
    deploy_priority: float,
    risk_components: List[str],
) -> float:
    """
    Spec §3 priority bands. Risk-driven trades sit above any DEPLOY in
    the same cycle so freed capital is always available before new
    deployments execute.
    """
    if rebalance_action == REBAL_FULL_EXIT:
        # Urgency rises with the number of severe risk flags.
        sev = (1.0 if RISK_FORCE_EXIT in risk_components else 0.0) + (
            0.5 if RISK_TRIM_PRIORITY in risk_components else 0.0
        )
        return _band_score(PRIORITY_BAND_FULL_EXIT, min(1.0, sev / 1.5 if sev > 0 else 0.5))

    if rebalance_action in {REBAL_TRIM, REBAL_SELL}:
        sev = 1.0 if RISK_TRIM_PRIORITY in risk_components else 0.5
        return _band_score(PRIORITY_BAND_FORCED_TRIM, sev)

    if rebalance_action in {REBAL_BUY_NEW, REBAL_ADD}:
        dp = _clamp(deploy_priority, 0.0, 1.0)
        if dp >= HIGH_CONVICTION_DEPLOY_PRIORITY:
            # Map [HIGH_CONVICTION_DEPLOY_PRIORITY, 1.0] into the high band.
            span = max(1e-9, 1.0 - HIGH_CONVICTION_DEPLOY_PRIORITY)
            frac = (dp - HIGH_CONVICTION_DEPLOY_PRIORITY) / span
            return _band_score(PRIORITY_BAND_HIGH_CONVICTION_DEPLOY, frac)
        # Below the gate → diversification / optional adds tier.
        frac = dp / max(1e-9, HIGH_CONVICTION_DEPLOY_PRIORITY)
        return _band_score(PRIORITY_BAND_MODERATE_DEPLOY, frac)

    if rebalance_action == REBAL_HOLD:
        return _band_score(PRIORITY_BAND_HOLD, 0.5)

    return PRIORITY_NO_ACTION


def _is_execution_ready(
    *,
    rebalance_action: str,
    rebalance_amount_usd: float,
    portfolio_action: str,
    risk_components: List[str],
    min_trade_usd: float,
) -> Tuple[bool, Optional[str]]:
    """
    A row is execution_ready iff a real, sized, non-blocked trade can be
    placed against it. Returns (ready, blocker_reason).
    """
    if portfolio_action == CONSTR_ACTION_BLOCK or rebalance_action == REBAL_NO_ACTION:
        return False, "blocked"
    if rebalance_action == REBAL_HOLD:
        return False, "hold_no_trade"
    if abs(rebalance_amount_usd) < float(min_trade_usd):
        return False, f"below_min_trade(${min_trade_usd:.2f})"
    # BUY/ADD into a name with severe risk flag → not ready, even though
    # Step 5 normally would have already BLOCKed it. Defensive check.
    if rebalance_action in {REBAL_BUY_NEW, REBAL_ADD}:
        if RISK_FORCE_EXIT in risk_components or RISK_BLOCK_NEW_BUY in risk_components:
            return False, "severe_risk_lock_for_buy"
    return True, None


# -----------------------------------------------------------
# Pipeline
# -----------------------------------------------------------
def build_rebalance_plan(
    *,
    construction_rows: List[Dict[str, Any]],
    positions_value_map: Dict[str, float],
    positions_value_total: float,
    construction_summary: Dict[str, Any],
    capital_summary: Dict[str, Any],
    risk_overlay_map: Dict[str, str],
    min_trade_usd: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Pure planner — no IO. Returns (plan_df, summary)."""
    # Total book value preference order:
    #   1. construction summary (cash-adjusted, canonical for weights)
    #   2. capital_deployment.cash_estimate + positions_value (cash-aware)
    #   3. raw sum of positions (pre-cash fallback)
    total_book_value = _to_float(construction_summary.get("total_portfolio_value")) or 0.0
    if total_book_value <= 0:
        cash = _to_float(capital_summary.get("cash_estimate")) or 0.0
        total_book_value = positions_value_total + max(0.0, cash)
    if total_book_value <= 0:
        total_book_value = positions_value_total

    rows: List[Dict[str, Any]] = []
    for r in construction_rows:
        sym = r["ticker"]
        portfolio_action = r["portfolio_action"]
        deploy_priority = float(r.get("deploy_priority") or 0.0)
        # Use broker-truth current_value when present; otherwise derive
        # from construction's current_weight_pct × book value.
        current_value_usd = positions_value_map.get(
            sym,
            (float(r.get("current_weight_pct") or 0.0) / 100.0) * total_book_value,
        )
        # Step 5 already published target_position_size_usd; trust it but
        # backstop with target_weight_pct math in case the upstream value
        # was missing or stale.
        target_value_usd = float(r.get("target_position_size_usd") or 0.0)
        if target_value_usd <= 0 and portfolio_action not in {
            CONSTR_ACTION_EXIT,
            CONSTR_ACTION_BLOCK,
        }:
            target_value_usd = (float(r.get("target_weight_pct") or 0.0) / 100.0) * total_book_value
        if portfolio_action == CONSTR_ACTION_EXIT:
            target_value_usd = 0.0

        # FULL_EXIT amount = -current_value_usd (sell entire position).
        if portfolio_action == CONSTR_ACTION_EXIT:
            rebalance_amount_usd = -current_value_usd
        elif portfolio_action == CONSTR_ACTION_BLOCK:
            rebalance_amount_usd = 0.0
        else:
            rebalance_amount_usd = target_value_usd - current_value_usd

        rebalance_action = _map_to_rebalance_action(
            portfolio_action=portfolio_action,
            current_value_usd=current_value_usd,
            target_value_usd=target_value_usd,
        )

        risk_flag = risk_overlay_map.get(sym, RISK_OK)
        risk_comps = _risk_components(risk_flag)

        priority = _priority_for(
            rebalance_action=rebalance_action,
            portfolio_action=portfolio_action,
            deploy_priority=deploy_priority,
            risk_components=risk_comps,
        )

        # Step 11: rotation_pressure biases sell/trim urgency.
        # Multiplier centred on 1.0 when pressure==0.5 (neutral) so
        # behaviour is unchanged unless a non-neutral runtime policy is
        # loaded. Buy-side priorities are NOT touched (additive only).
        if rebalance_action in (REBAL_FULL_EXIT, REBAL_SELL, REBAL_TRIM):
            rp_mult = 1.0 + ROTATION_PRESSURE_GAIN * (ROTATION_PRESSURE - 0.5)
            priority = max(0.0, min(1.0, priority * rp_mult))

        ready, blocker = _is_execution_ready(
            rebalance_action=rebalance_action,
            rebalance_amount_usd=rebalance_amount_usd,
            portfolio_action=portfolio_action,
            risk_components=risk_comps,
            min_trade_usd=min_trade_usd,
        )

        # Compose reason: keep upstream construction reason and append a
        # rebalance-specific tag so downstream filters can grep cleanly.
        reason_parts = [rebalance_action.lower()]
        if not ready and blocker:
            reason_parts.append(f"not_ready:{blocker}")
        if r.get("construction_reason"):
            reason_parts.append(f"from:{r['construction_reason']}")

        rows.append(
            {
                "ticker": sym,
                "current_weight_pct": round(float(r.get("current_weight_pct") or 0.0), 4),
                "target_weight_pct": round(float(r.get("target_weight_pct") or 0.0), 4),
                "delta_weight_pct": round(float(r.get("delta_weight_pct") or 0.0), 4),
                "current_value_usd": round(float(current_value_usd), 2),
                "target_value_usd": round(float(target_value_usd), 2),
                "rebalance_amount_usd": round(float(rebalance_amount_usd), 2),
                "portfolio_action": portfolio_action,
                "rebalance_action": rebalance_action,
                "priority": priority,
                "execution_order": 0,  # resolved below
                "execution_ready": bool(ready),
                "reason": "|".join(reason_parts),
                "_sequence_tier": EXECUTION_SEQUENCE.get(rebalance_action, 9),
            }
        )

    # ── Sequence (sell-before-buy is non-negotiable) ───────────────
    rows.sort(
        key=lambda r: (
            r["_sequence_tier"],  # tier asc (FULL_EXIT first)
            -float(r["priority"]),  # priority desc within tier
            -abs(float(r["rebalance_amount_usd"])),  # bigger trades first
            r["ticker"],  # deterministic tiebreaker
        )
    )
    for i, r in enumerate(rows, start=1):
        r["execution_order"] = i
        r.pop("_sequence_tier", None)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # ── Summary stats ─────────────────────────────────────────────
    def _count(action: str) -> int:
        return int(sum(1 for r in rows if r["rebalance_action"] == action))

    exits = _count(REBAL_FULL_EXIT)
    sells = _count(REBAL_SELL)
    trims = _count(REBAL_TRIM)
    adds = _count(REBAL_ADD)
    buys = _count(REBAL_BUY_NEW)
    holds = _count(REBAL_HOLD)
    blocks = _count(REBAL_NO_ACTION)

    capital_freed = sum(
        -float(r["rebalance_amount_usd"])
        for r in rows
        if r["rebalance_action"] in {REBAL_FULL_EXIT, REBAL_TRIM, REBAL_SELL}
        and float(r["rebalance_amount_usd"]) < 0
    )
    capital_deployed = sum(
        float(r["rebalance_amount_usd"])
        for r in rows
        if r["rebalance_action"] in {REBAL_BUY_NEW, REBAL_ADD}
        and float(r["rebalance_amount_usd"]) > 0
    )

    # Turnover = sum(|amount|) for actionable trades / book × 100. HOLD
    # and NO_ACTION rows don't count toward turnover even if a tiny
    # rounding delta exists, because they aren't executable.
    actionable_notional = sum(
        abs(float(r["rebalance_amount_usd"]))
        for r in rows
        if r["rebalance_action"] not in {REBAL_HOLD, REBAL_NO_ACTION}
    )
    turnover_pct = (
        round(100.0 * actionable_notional / total_book_value, 4) if total_book_value > 0 else 0.0
    )

    ready_rows = [r for r in rows if r["execution_ready"]]

    # Top-actions: execution_ready, ranked by execution_order asc (the
    # plan's natural sequence), capped at 10. We surface the dollar
    # amount and direction so dashboards can render it cleanly.
    top_actions: List[Dict[str, Any]] = []
    for r in ready_rows[:10]:
        top_actions.append(
            {
                "execution_order": int(r["execution_order"]),
                "ticker": r["ticker"],
                "rebalance_action": r["rebalance_action"],
                "rebalance_amount_usd": float(r["rebalance_amount_usd"]),
                "delta_weight_pct": float(r["delta_weight_pct"]),
                "priority": float(r["priority"]),
                "reason": r["reason"],
            }
        )

    summary: Dict[str, Any] = {
        "generated_at_utc": _now_iso_utc(),
        "total_rebalance_actions": int(len(rows)),
        "executable_actions": int(len(ready_rows)),
        "exits": exits,
        "sells": sells,
        "trims": trims,
        "buys": buys,
        "adds": adds,
        "holds": holds,
        "blocked": blocks,
        "estimated_capital_freed": round(float(capital_freed), 2),
        "estimated_capital_deployed": round(float(capital_deployed), 2),
        "net_capital_flow": round(float(capital_deployed - capital_freed), 2),
        "portfolio_turnover_pct": turnover_pct,
        "total_portfolio_value": round(float(total_book_value), 4),
        "min_trade_usd": float(min_trade_usd),
        "construction_score": construction_summary.get("portfolio_construction_score"),
        "available_slots": capital_summary.get("available_slots"),
        "deployable_capital_estimate": capital_summary.get("deployable_capital_estimate"),
        "execution_sequence": EXECUTION_SEQUENCE,
        "top_actions": top_actions,
    }
    return df, summary


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only portfolio rebalance planner (step 6 of WATCH funnel). "
            "Converts Step 5 construction targets into a sequenced trade plan."
        ),
    )
    p.add_argument("--construction", default=str(DEFAULT_CONSTRUCTION_CSV))
    p.add_argument("--construction-summary", default=str(DEFAULT_CONSTRUCTION_JSON))
    p.add_argument("--positions", default=str(DEFAULT_POSITIONS_CSV))
    p.add_argument("--capital-summary", default=str(DEFAULT_CAPITAL_SUMMARY_JSON))
    p.add_argument("--risk-overlay", default=str(DEFAULT_RISK_OVERLAY_CSV))
    p.add_argument(
        "--min-trade-usd",
        type=float,
        default=MIN_TRADE_USD_DEFAULT,
        help="Notional threshold below which a trade is ignored as noise.",
    )
    p.add_argument("--out-csv", default=str(DEFAULT_OUTPUT_CSV))
    p.add_argument("--out-json", default=str(DEFAULT_OUTPUT_JSON))
    return p.parse_args(argv)


def _apply_runtime_policy(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Step 11 integration. Reads runtime_policy.json (if present) and
    overrides ROTATION_PRESSURE for the current cycle. Safe to call
    every cycle -- missing/malformed file leaves the neutral default
    untouched, which means zero behaviour change.
    Path resolves via module attribute at call time so tests can
    monkey-patch ``DEFAULT_RUNTIME_POLICY_JSON``.
    """
    global ROTATION_PRESSURE
    if path is None:
        path = DEFAULT_RUNTIME_POLICY_JSON
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            rp = json.load(f) or {}
    except Exception as e:
        print(
            f"[PORTFOLIO_REBALANCE_WARN] runtime_policy.json present but unreadable "
            f"({type(e).__name__}: {e}); keeping defaults",
            flush=True,
        )
        return None
    v = rp.get("rotation_pressure")
    if v is not None:
        try:
            ROTATION_PRESSURE = max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            pass
    mult = 1.0 + ROTATION_PRESSURE_GAIN * (ROTATION_PRESSURE - 0.5)
    print(
        "[PORTFOLIO_REBALANCE_POLICY] "
        f"regime={rp.get('regime', 'UNKNOWN')} "
        f"rotation_pressure={ROTATION_PRESSURE:.2f} "
        f"sell_priority_multiplier={mult:.3f}",
        flush=True,
    )
    return rp


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[PORTFOLIO_REBALANCE] starting (read-only planner)", flush=True)
    _apply_runtime_policy()

    construction_df = _safe_read_csv(
        Path(args.construction), label="portfolio_construction_recommendations.csv"
    )
    construction_summary = _load_construction_summary(
        _safe_read_json(
            Path(args.construction_summary),
            label="portfolio_construction_summary.json",
        )
    )
    positions_df = _safe_read_csv(Path(args.positions), label="positions_snapshot.csv")
    capital_summary = _load_capital_summary(
        _safe_read_json(Path(args.capital_summary), label="capital_deployment_summary.json")
    )
    risk_df = _safe_read_csv(Path(args.risk_overlay), label="performance_risk_overlay.csv")

    construction_rows = _load_construction_rows(construction_df)
    positions_value_map, positions_value_total = _load_positions_value_map(positions_df)
    risk_overlay_map = _load_risk_overlay_map(risk_df)

    df, summary = build_rebalance_plan(
        construction_rows=construction_rows,
        positions_value_map=positions_value_map,
        positions_value_total=positions_value_total,
        construction_summary=construction_summary,
        capital_summary=capital_summary,
        risk_overlay_map=risk_overlay_map,
        min_trade_usd=float(args.min_trade_usd),
    )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    try:
        _atomic_write_csv(df, out_csv)
    except Exception as e:
        _warn(f"failed to write {out_csv}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(summary, out_json)
    except Exception as e:
        _warn(f"failed to write {out_json}: {type(e).__name__}: {e}")
        return 2

    print(
        "[PORTFOLIO_REBALANCE] "
        f"actions={summary['total_rebalance_actions']} "
        f"executable={summary['executable_actions']} "
        f"exits={summary['exits']} "
        f"trims={summary['trims']} "
        f"buys={summary['buys']} "
        f"adds={summary['adds']} "
        f"holds={summary['holds']} "
        f"blocked={summary['blocked']} "
        f"turnover={summary['portfolio_turnover_pct']:.2f}% "
        f"capital_freed=${summary['estimated_capital_freed']:.2f} "
        f"capital_deployed=${summary['estimated_capital_deployed']:.2f}",
        flush=True,
    )
    print(
        "[REBALANCE_TOP_ACTIONS] symbols=" f"{[a['ticker'] for a in summary['top_actions']]}",
        flush=True,
    )
    print(
        f"[PORTFOLIO_REBALANCE_OUT] csv={out_csv.as_posix()} summary={out_json.as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
