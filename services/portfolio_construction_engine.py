"""
Portfolio Construction Engine — Step 5 of the WATCH → DEPLOY funnel.

Reads:
    data/results/capital_deployment_recommendations.csv
    data/results/capital_deployment_summary.json
    data/results/positions_snapshot.csv
    data/results/performance_risk_overlay.csv
    data/results/opportunity_persistence_recommendations.csv
    data/results/portfolio_allocation_recommendations.csv

Writes:
    data/results/portfolio_construction_recommendations.csv
    data/results/portfolio_construction_summary.json

Purpose
-------
Step 4 (capital deployment) sizes each opportunity in isolation. This
engine answers the next, portfolio-level question:

    "What should the entire portfolio look like?"

It merges currently-held positions with the new deploy candidates,
enforces *institutional* portfolio constraints (per-position cap,
sector cap, correlated-cluster cap, cash reserve, max-new-per-cycle),
and emits one row per ticker with a final `portfolio_action`
(OPEN_POSITION / ADD_TO_POSITION / HOLD / TRIM / EXIT / BLOCK) plus a
`portfolio_construction_score` describing overall portfolio health.

Safety
------
* Read-only. No orders, no broker calls, no mutation of execute_trades
  or manage_positions. All sizes are recommendations.
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

# Reuse the canonical ticker→sector mapping already maintained for the
# sector exposure guard — keeping one source of truth across services.
try:
    from services.sector_exposure import get_sector, UNKNOWN_SECTOR_LABEL  # type: ignore
except Exception:  # pragma: no cover — degrade gracefully if import fails
    UNKNOWN_SECTOR_LABEL = "Unknown"

    def get_sector(symbol: str) -> str:  # type: ignore[misc]
        return UNKNOWN_SECTOR_LABEL


# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"
CONFIG_DIR = ROOT / "config"

DEFAULT_CAPITAL_DEPLOY_CSV = RESULTS_DIR / "capital_deployment_recommendations.csv"
DEFAULT_CAPITAL_DEPLOY_JSON = RESULTS_DIR / "capital_deployment_summary.json"
DEFAULT_POSITIONS_CSV = RESULTS_DIR / "positions_snapshot.csv"
DEFAULT_RISK_OVERLAY_CSV = RESULTS_DIR / "performance_risk_overlay.csv"
DEFAULT_PERSISTENCE_CSV = RESULTS_DIR / "opportunity_persistence_recommendations.csv"
DEFAULT_ALLOCATION_CSV = RESULTS_DIR / "portfolio_allocation_recommendations.csv"

DEFAULT_OUTPUT_CSV = RESULTS_DIR / "portfolio_construction_recommendations.csv"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "portfolio_construction_summary.json"

# -----------------------------------------------------------
# Portfolio constraint constants (institutional defaults)
# -----------------------------------------------------------
MAX_SINGLE_POSITION_PCT = 8.0
MIN_POSITION_PCT = 0.5

MAX_SECTOR_PCT = 25.0
MAX_CLUSTER_PCT = 30.0

MIN_CASH_RESERVE_PCT = 10.0
MAX_CASH_RESERVE_PCT = 20.0
TARGET_CASH_RESERVE_PCT = 15.0

MAX_NEW_POSITIONS_PER_CYCLE_DEFAULT = 5

# Step 11 runtime policy override (optional, additive).
# When data/results/runtime_policy.json exists, MAX_SINGLE_POSITION_PCT
# is overridden for the current cycle. Missing/malformed file = defaults.
DEFAULT_RUNTIME_POLICY_JSON = RESULTS_DIR / "runtime_policy.json"

# Penalty thresholds used for the construction score components.
DIVERSIFICATION_TARGET_TOP_SECTOR_PCT = 20.0  # below this gets full credit
SECTOR_HEAVY_FLOOR_PCT = 5.0  # sectors > this floor count as represented
SECTOR_TARGET_REPRESENTED = 4  # >= this many distinct sectors = full credit

# Portfolio actions
ACTION_OPEN = "OPEN_POSITION"
ACTION_ADD = "ADD_TO_POSITION"
ACTION_HOLD = "HOLD"
ACTION_TRIM = "TRIM"
ACTION_EXIT = "EXIT"
ACTION_BLOCK = "BLOCK"

# Deployment engine labels (mirror Step 4 source of truth)
DEPLOY_DECISION_DEPLOY = "DEPLOY_NOW"
DEPLOY_DECISION_SHADOW = "SHADOW_WATCH"
DEPLOY_DECISION_BLOCK = "BLOCK"

# Allocation engine labels (mirror Step 1)
ALLOC_ACTION_EXIT = "EXIT"
ALLOC_ACTION_TRIM = "TRIM"
ALLOC_ACTION_BLOCK = "BLOCK"
ALLOC_ACTION_ADD = "ADD"
ALLOC_ACTION_OPEN_NEW = "OPEN_NEW"
ALLOC_ACTION_WATCH = "WATCH"
ALLOC_ACTION_HOLD = "HOLD"

# Persistence engine labels (deterioration signals for currently-held)
PERSISTENCE_DEMOTE = "DEMOTE_WATCH"
PERSISTENCE_REJECT = "REJECT"

# Risk flag tokens (pipe-joined upstream)
RISK_FORCE_EXIT = "FORCE_EXIT"
RISK_TRIM_PRIORITY = "TRIM_PRIORITY"
RISK_BLOCK_NEW_BUY = "BLOCK_NEW_BUY"
RISK_OK = "OK"

# -----------------------------------------------------------
# Correlation clusters
# -----------------------------------------------------------
# A ticker may legitimately belong to multiple clusters (e.g. NVDA is
# both MEGA_CAP_TECH and SEMICONDUCTORS). We cap each cluster
# independently, which naturally tightens exposure on multi-cluster
# names. Lower-cased on lookup via _ticker_clusters().
CORRELATION_CLUSTERS: Dict[str, frozenset] = {
    "BROAD_US_EQUITY": frozenset({"SPY", "VOO", "VTI", "QQQ", "DIA", "IWM", "ITOT", "SCHB"}),
    "MEGA_CAP_TECH": frozenset(
        {"AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN", "NFLX", "AVGO", "ORCL"}
    ),
    "HIGH_BETA_INNOVATION": frozenset({"ARKK", "TSLA", "RIVN", "COIN", "PLTR", "SNOW", "CRWD"}),
    "CRYPTO_PROXY": frozenset({"BITO", "GBTC", "ETHE", "MSTR", "COIN", "MARA", "RIOT"}),
    "PRECIOUS_METALS": frozenset({"GLD", "SLV", "IAU", "GDX", "GDXJ"}),
    "ENERGY_COMMODITY": frozenset({"USO", "UNG", "XLE", "XOP"}),
    "AGRICULTURE": frozenset({"DBA", "CORN", "WEAT", "SOYB"}),
    "SEMICONDUCTORS": frozenset({"NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "SMH", "SOXX"}),
    "FINANCIALS_CLUSTER": frozenset({"JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "XLF"}),
}


def _ticker_clusters(sym: str) -> List[str]:
    s = (sym or "").strip().upper()
    if not s:
        return []
    return [name for name, members in CORRELATION_CLUSTERS.items() if s in members]


OUTPUT_COLUMNS = [
    "ticker",
    "current_weight_pct",
    "target_weight_pct",
    "target_position_size_usd",
    "delta_weight_pct",
    "sector_bucket",
    "concentration_penalty",
    "diversification_bonus",
    "correlation_penalty",
    "deploy_priority",
    "portfolio_action",
    "reason",
]


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[PORTFOLIO_CONSTRUCTION_WARN] {msg}", flush=True)


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


# -----------------------------------------------------------
# Loaders
# -----------------------------------------------------------
def _load_positions(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, Any]], float]:
    """Return ({ticker: {value, qty}}, total_portfolio_value)."""
    out: Dict[str, Dict[str, Any]] = {}
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
        out[sym] = {"market_value": mv, "qty": qty}
        total += mv
    return out, total


def _load_capital_deploy_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """{ticker: capital_deployment_engine row} for ALL rows (DEPLOY/SHADOW/BLOCK)."""
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return out
    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        return out
    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        out[sym] = {
            "deployment_decision": _norm_upper(r.get("deployment_decision")),
            "target_weight_pct": _to_float(r.get("target_weight_pct")),
            "suggested_position_size_usd": _to_float(r.get("suggested_position_size_usd")),
            "deploy_priority": _to_float(r.get("deploy_priority")),
            "persistence_score": _to_float(r.get("persistence_score")),
            "confidence": _to_float(r.get("confidence")),
            "delta_pct": _to_float(r.get("delta_pct")),
            "risk_flag": _norm_upper(r.get("risk_flag")) or RISK_OK,
            "reason": str(r.get("reason") or ""),
        }
    return out


def _load_allocation_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """{ticker: portfolio_allocation_engine row} — supplies recommended_action."""
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return out
    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        return out
    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        out[sym] = {
            "recommended_action": _norm_upper(r.get("recommended_action")),
            "risk_flag": _norm_upper(r.get("risk_flag")) or RISK_OK,
            "current_weight_pct": _to_float(r.get("current_weight_pct")),
            "is_currently_held": _to_bool(r.get("is_currently_held")),
            "allocation_score": _to_float(r.get("allocation_score")),
            "confidence": _to_float(r.get("confidence")),
            "delta_pct": _to_float(r.get("delta_pct")),
            "edge_score": _to_float(r.get("edge_score")),
        }
    return out


def _load_risk_overlay_map(df: pd.DataFrame) -> Dict[str, str]:
    """{ticker: risk_flag} — canonical risk source (overrides allocation when both present)."""
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


def _load_persistence_map(df: pd.DataFrame) -> Dict[str, str]:
    """{ticker: promotion_decision} — used to detect deterioration on held names."""
    out: Dict[str, str] = {}
    if df is None or df.empty:
        return out
    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col or "promotion_decision" not in df.columns:
        return out
    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        out[sym] = _norm_upper(r.get("promotion_decision"))
    return out


def _load_capital_summary(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract capacity hints from capital_deployment_summary.json."""
    if not d:
        return {}
    return {
        "deployable_capital_estimate": _to_float(d.get("deployable_capital_estimate")),
        "max_positions": int(_to_float_or_zero(d.get("max_positions")) or 0),
        "available_slots": int(_to_float_or_zero(d.get("available_slots")) or 0),
        "cash_estimate": _to_float(d.get("cash_estimate")),
        "reserve_pct": _to_float(d.get("reserve_pct")),
        "total_portfolio_value": _to_float(d.get("total_portfolio_value")),
    }


# -----------------------------------------------------------
# Construction pipeline
# -----------------------------------------------------------
def _initial_portfolio_action(
    *,
    sym: str,
    is_held: bool,
    deploy_decision: str,
    alloc_action: str,
    risk_components: List[str],
    persistence_decision: str,
) -> Tuple[str, List[str]]:
    """
    Map upstream signals → a *pre-constraint* portfolio_action.

    Precedence (top wins):
      1. EXIT  — held & (FORCE_EXIT risk OR allocation said EXIT)
      2. TRIM  — held & (TRIM_PRIORITY risk OR allocation said TRIM)
      3. BLOCK — deployment BLOCKED, or BLOCK_NEW_BUY for non-held openers,
                 or SHADOW_WATCH (still informational only)
      4. OPEN_POSITION — DEPLOY_NOW + not held
      5. ADD_TO_POSITION — DEPLOY_NOW + held
      6. HOLD — fallback for held names
      7. BLOCK — fallback for non-held with no actionable directive
    Returns the (action, reason_tags).
    """
    tags: List[str] = []

    # ── 1. EXIT (held only) ─────────────────────────────────────
    if is_held and RISK_FORCE_EXIT in risk_components:
        return ACTION_EXIT, ["force_exit_risk"]
    if is_held and alloc_action == ALLOC_ACTION_EXIT:
        return ACTION_EXIT, ["allocation_exit"]

    # ── 2. TRIM (held only) ─────────────────────────────────────
    if is_held and RISK_TRIM_PRIORITY in risk_components:
        tags.append("trim_priority_risk")
        if persistence_decision in {PERSISTENCE_DEMOTE, PERSISTENCE_REJECT}:
            tags.append("persistence_deteriorating")
        return ACTION_TRIM, tags
    if is_held and alloc_action == ALLOC_ACTION_TRIM:
        return ACTION_TRIM, ["allocation_trim"]

    # ── 3. BLOCK (deployment-level) ─────────────────────────────
    if deploy_decision == DEPLOY_DECISION_BLOCK:
        return ACTION_BLOCK, ["deployment_blocked"]
    if not is_held and RISK_BLOCK_NEW_BUY in risk_components:
        return ACTION_BLOCK, ["block_new_buy_risk"]
    if deploy_decision == DEPLOY_DECISION_SHADOW:
        return ACTION_BLOCK, ["shadow_waiting_confirmation"]

    # ── 4 / 5. DEPLOY_NOW ───────────────────────────────────────
    if deploy_decision == DEPLOY_DECISION_DEPLOY:
        if is_held:
            return ACTION_ADD, ["deploy_now_held"]
        return ACTION_OPEN, ["deploy_now_new"]

    # ── 6. HOLD for held names with no actionable directive ────
    if is_held:
        if persistence_decision in {PERSISTENCE_DEMOTE, PERSISTENCE_REJECT}:
            tags.append("persistence_warning")
        return ACTION_HOLD, tags or ["held_no_action"]

    # ── 7. Fallback (non-held, no signal) ───────────────────────
    return ACTION_BLOCK, ["no_actionable_directive"]


def _apply_portfolio_constraints(
    rows: List[Dict[str, Any]],
    *,
    max_new_per_cycle: int,
    cash_reserve_pct: float,
) -> List[str]:
    """
    In-place: enforce institutional portfolio caps on OPEN/ADD candidates.

    Caps applied:
      * Per-position max  = MAX_SINGLE_POSITION_PCT (8%)
      * Per-sector max    = MAX_SECTOR_PCT (25%)
      * Per-cluster max   = MAX_CLUSTER_PCT (30%)
      * Aggregate budget  = 100 - cash_reserve_pct - (anticipated post-action exposure)
      * Max OPEN per run  = max_new_per_cycle

    Semantic distinctions:
      * OPEN consumes its *full* target_weight against all caps.
      * ADD consumes only the *increment* (target - current). If the
        engine target is below current, the ADD signal is downgraded to
        HOLD (we never want to TRIM a position that has DEPLOY_NOW).
      * EXIT frees its full current_weight from sector/cluster/budget.
      * TRIM frees ~half its current_weight (we recommend 50% trim).

    Demotions:
      * OPEN below MIN_POSITION_PCT after caps → BLOCK.
      * ADD increment below MIN_POSITION_PCT after caps → revert to HOLD
        (keeping the existing position untouched is safer than blocking
        a held name).
      * OPEN beyond max_new_per_cycle (lowest priorities) → BLOCK.

    Returns a list of [PORTFOLIO_CAP] log lines describing every action.
    """
    cap_logs: List[str] = []

    # ── Step A: seed running totals with the *post-action* exposure of
    # held rows (EXIT removes; TRIM halves; HOLD stays). This lets the
    # deploy sweep model a portfolio after EXIT/TRIM have freed capacity.
    sector_totals: Dict[str, float] = defaultdict(float)
    cluster_totals: Dict[str, float] = defaultdict(float)
    held_baseline_exposure = 0.0  # post-action exposure of held names

    for r in rows:
        cw = float(r.get("current_weight_pct") or 0.0)
        if cw <= 0:
            continue
        action = r["portfolio_action"]
        if action == ACTION_EXIT:
            contributing = 0.0
        elif action == ACTION_TRIM:
            contributing = cw * 0.5
        elif action in {ACTION_HOLD, ACTION_ADD}:
            contributing = cw
        else:
            # BLOCK on a held name keeps the position untouched (we are
            # *not* selling); count its current weight.
            contributing = cw
        sector_totals[r["sector_bucket"]] += contributing
        for cluster in r.get("_clusters", []):
            cluster_totals[cluster] += contributing
        held_baseline_exposure += contributing

    budget_remaining = max(0.0, 100.0 - cash_reserve_pct - held_baseline_exposure)

    # ── Step B: collect OPEN/ADD rows in priority order. ───────────
    deploy_rows = [r for r in rows if r["portfolio_action"] in {ACTION_OPEN, ACTION_ADD}]
    deploy_rows.sort(
        key=lambda r: (float(r.get("deploy_priority") or 0.0), r["ticker"]),
        reverse=True,
    )

    # ── Step C: enforce max_new_per_cycle on OPEN rows only. ──────
    if max_new_per_cycle > 0:
        opens = [r for r in deploy_rows if r["portfolio_action"] == ACTION_OPEN]
        if len(opens) > max_new_per_cycle:
            keep_ids = set(id(r) for r in opens[:max_new_per_cycle])
            for r in opens[max_new_per_cycle:]:
                r["portfolio_action"] = ACTION_BLOCK
                r["target_weight_pct"] = 0.0
                r["_reason_tags"].append(f"max_new_per_cycle({max_new_per_cycle})_exceeded")
                cap_logs.append(
                    f"[PORTFOLIO_CAP] {r['ticker']} new_position_cap_exhausted "
                    f"max_new_per_cycle={max_new_per_cycle}"
                )
            deploy_rows = [
                r for r in deploy_rows if r["portfolio_action"] != ACTION_BLOCK or id(r) in keep_ids
            ]

    # ── Step D: per-candidate cap sweep. ──────────────────────────
    for r in deploy_rows:
        if r["portfolio_action"] not in {ACTION_OPEN, ACTION_ADD}:
            continue

        sector = r["sector_bucket"]
        clusters = r.get("_clusters", [])
        current_w = float(r.get("current_weight_pct") or 0.0)
        proposed_target = float(r.get("target_weight_pct") or 0.0)

        # For ADD, only the *increment* competes for caps; the existing
        # current_w is already counted in sector/cluster totals above.
        if r["portfolio_action"] == ACTION_ADD:
            increment = proposed_target - current_w
            if increment <= 0:
                # Engine target is already met — downgrade to HOLD.
                r["portfolio_action"] = ACTION_HOLD
                r["target_weight_pct"] = round(current_w, 4)
                r["_reason_tags"].append("deploy_target_already_met")
                cap_logs.append(
                    f"[PORTFOLIO_CAP] {r['ticker']} add_to_hold "
                    f"current={current_w:.2f}% target={proposed_target:.2f}%"
                )
                continue
            proposed_increment = increment
        else:
            proposed_increment = proposed_target

        # Per-position absolute cap (always applies to FINAL weight).
        max_increment_from_position_cap = max(0.0, MAX_SINGLE_POSITION_PCT - current_w)
        proposed_increment = min(proposed_increment, max_increment_from_position_cap)

        original_increment = proposed_increment

        # Sector cap on the *increment*.
        sector_room = max(0.0, MAX_SECTOR_PCT - sector_totals[sector])
        proposed_sector_increment = min(proposed_increment, sector_room)
        if proposed_sector_increment < proposed_increment:
            cap_logs.append(
                f"[PORTFOLIO_CAP] {r['ticker']} sector={sector} "
                f"reduced {proposed_increment:.2f}%->{proposed_sector_increment:.2f}% "
                f"(sector_used={sector_totals[sector]:.2f}%)"
            )
        proposed_increment = proposed_sector_increment

        # Cluster caps on the *increment*.
        proposed_after_cluster = proposed_increment
        for cluster in clusters:
            room = max(0.0, MAX_CLUSTER_PCT - cluster_totals[cluster])
            if proposed_after_cluster > room:
                cap_logs.append(
                    f"[PORTFOLIO_CAP] {r['ticker']} cluster={cluster} "
                    f"reduced {proposed_after_cluster:.2f}%->{room:.2f}% "
                    f"(cluster_used={cluster_totals[cluster]:.2f}%)"
                )
                proposed_after_cluster = room

        # Aggregate budget cap on the *increment*.
        if proposed_after_cluster > budget_remaining:
            cap_logs.append(
                f"[PORTFOLIO_CAP] {r['ticker']} budget_cap "
                f"reduced {proposed_after_cluster:.2f}%->{budget_remaining:.2f}% "
                f"(budget_remaining={budget_remaining:.2f}%)"
            )
            proposed_after_cluster = budget_remaining

        final_increment = proposed_after_cluster

        # Track informational penalties (pct-points removed by caps).
        r["concentration_penalty"] = round(
            max(0.0, original_increment - proposed_sector_increment), 4
        )
        r["correlation_penalty"] = round(max(0.0, proposed_sector_increment - final_increment), 4)

        if final_increment < MIN_POSITION_PCT:
            if r["portfolio_action"] == ACTION_ADD:
                # Don't dump a held position just because we can't add
                # the increment; hold the existing weight.
                r["portfolio_action"] = ACTION_HOLD
                r["target_weight_pct"] = round(current_w, 4)
                r["_reason_tags"].append("add_increment_below_min")
                cap_logs.append(
                    f"[PORTFOLIO_CAP] {r['ticker']} add_to_hold "
                    f"increment<{MIN_POSITION_PCT:.2f}% "
                    f"(kept existing {current_w:.2f}%)"
                )
            else:
                r["portfolio_action"] = ACTION_BLOCK
                r["target_weight_pct"] = 0.0
                r["_reason_tags"].append("constraint_below_min_position")
                cap_logs.append(
                    f"[PORTFOLIO_CAP] {r['ticker']} demoted_to_block "
                    f"final<{MIN_POSITION_PCT:.2f}%"
                )
            continue

        final_total = current_w + final_increment
        final_total = round(_clamp(final_total, MIN_POSITION_PCT, MAX_SINGLE_POSITION_PCT), 4)
        r["target_weight_pct"] = final_total

        # Diversification bonus: small credit when this name lands in an
        # under-represented sector (informational).
        if sector != UNKNOWN_SECTOR_LABEL and sector_totals[sector] < SECTOR_HEAVY_FLOOR_PCT:
            r["diversification_bonus"] = round(min(2.0, MAX_SECTOR_PCT - sector_totals[sector]), 4)
        else:
            r["diversification_bonus"] = 0.0

        # Update running totals with the actually-deployed increment.
        sector_totals[sector] += final_increment
        for cluster in clusters:
            cluster_totals[cluster] += final_increment
        budget_remaining = max(0.0, budget_remaining - final_increment)

    return cap_logs


def _finalize_rows_and_sizes(
    rows: List[Dict[str, Any]],
    *,
    total_portfolio_value: float,
) -> List[Dict[str, Any]]:
    """Resolve target sizes/deltas and trim internal-only keys.

    For EXIT we drive target_weight_pct to 0; for TRIM we halve current.
    These are *recommendations* — actual exit/trim execution stays in
    `execute_trades` / `manage_positions`.
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        action = r["portfolio_action"]
        current = float(r.get("current_weight_pct") or 0.0)

        if action == ACTION_EXIT:
            r["target_weight_pct"] = 0.0
        elif action == ACTION_TRIM:
            # Recommend halving the position as the default TRIM amount.
            r["target_weight_pct"] = round(max(MIN_POSITION_PCT, current * 0.5), 4)
        elif action == ACTION_HOLD:
            r["target_weight_pct"] = round(current, 4)
        elif action == ACTION_BLOCK:
            r["target_weight_pct"] = 0.0
        # OPEN / ADD already set by the constraint sweep.

        target = float(r["target_weight_pct"])
        r["delta_weight_pct"] = round(target - current, 4)
        r["target_position_size_usd"] = round((target / 100.0) * total_portfolio_value, 2)

        reason_tags = r.pop("_reason_tags", [])
        if reason_tags:
            r["reason"] = "|".join(reason_tags)
        elif not r.get("reason"):
            r["reason"] = action.lower()

        # Drop internal-only keys before emit.
        for k in ("_clusters",):
            r.pop(k, None)
        out.append(r)
    return out


def _compute_construction_score(
    rows: List[Dict[str, Any]],
    *,
    cash_reserve_pct: float,
) -> Dict[str, float]:
    """Combine diversification, conviction, risk-balance, cash discipline → [0,1]."""
    sector_totals: Dict[str, float] = defaultdict(float)
    for r in rows:
        tw = float(r.get("target_weight_pct") or 0.0)
        if tw <= 0:
            continue
        sector_totals[r["sector_bucket"]] += tw

    # Diversification: penalize when any sector is over the soft target.
    if sector_totals:
        top_sector_pct = max(sector_totals.values())
        represented = sum(1 for v in sector_totals.values() if v >= SECTOR_HEAVY_FLOOR_PCT)
    else:
        top_sector_pct = 0.0
        represented = 0

    if top_sector_pct <= DIVERSIFICATION_TARGET_TOP_SECTOR_PCT:
        top_sector_score = 1.0
    elif top_sector_pct >= MAX_SECTOR_PCT:
        top_sector_score = 0.0
    else:
        span = MAX_SECTOR_PCT - DIVERSIFICATION_TARGET_TOP_SECTOR_PCT
        top_sector_score = _clamp(
            1.0 - (top_sector_pct - DIVERSIFICATION_TARGET_TOP_SECTOR_PCT) / max(span, 1e-9),
            0.0,
            1.0,
        )
    sector_breadth_score = _clamp(represented / float(SECTOR_TARGET_REPRESENTED), 0.0, 1.0)
    diversification_quality = round(0.6 * top_sector_score + 0.4 * sector_breadth_score, 6)

    # Conviction: weighted avg deploy_priority for OPEN+ADD rows.
    deploy_rows = [r for r in rows if r["portfolio_action"] in {ACTION_OPEN, ACTION_ADD}]
    if deploy_rows:
        weights = [max(0.0, float(r["target_weight_pct"])) for r in deploy_rows]
        priorities = [_clamp(float(r.get("deploy_priority") or 0.0), 0.0, 1.0) for r in deploy_rows]
        wsum = sum(weights) or 1.0
        conviction_quality = round(sum(w * p for w, p in zip(weights, priorities)) / wsum, 6)
    else:
        # No new deploys — neutral score so we don't punish a passive cycle.
        conviction_quality = 0.5

    # Risk balance: fraction of *non-blocked* rows out of (open/add/hold/trim/exit/block).
    total = len(rows)
    blocked = sum(1 for r in rows if r["portfolio_action"] == ACTION_BLOCK)
    risk_balance = round(1.0 - (blocked / float(total)), 6) if total else 1.0

    # Cash reserve discipline: peak when cash is in the target band.
    if cash_reserve_pct < MIN_CASH_RESERVE_PCT:
        # Under-reserved (over-deployed).
        gap = MIN_CASH_RESERVE_PCT - cash_reserve_pct
        cash_discipline = _clamp(1.0 - gap / MIN_CASH_RESERVE_PCT, 0.0, 1.0)
    elif cash_reserve_pct > MAX_CASH_RESERVE_PCT:
        # Over-reserved (under-deployed).
        gap = cash_reserve_pct - MAX_CASH_RESERVE_PCT
        cash_discipline = _clamp(1.0 - gap / (100.0 - MAX_CASH_RESERVE_PCT), 0.0, 1.0)
    else:
        cash_discipline = 1.0
    cash_discipline = round(cash_discipline, 6)

    construction_score = round(
        0.30 * diversification_quality
        + 0.30 * conviction_quality
        + 0.20 * risk_balance
        + 0.20 * cash_discipline,
        6,
    )

    return {
        "portfolio_construction_score": construction_score,
        "diversification_quality": diversification_quality,
        "conviction_quality": conviction_quality,
        "risk_balance": risk_balance,
        "cash_discipline": cash_discipline,
        "top_sector_pct": round(top_sector_pct, 4),
        "represented_sectors": int(represented),
    }


# -----------------------------------------------------------
# Build pipeline
# -----------------------------------------------------------
def build_construction(
    *,
    positions_map: Dict[str, Dict[str, Any]],
    total_portfolio_value: float,
    capital_deploy_map: Dict[str, Dict[str, Any]],
    allocation_map: Dict[str, Dict[str, Any]],
    risk_overlay_map: Dict[str, str],
    persistence_map: Dict[str, str],
    capital_summary: Dict[str, Any],
    max_new_per_cycle: int,
    target_cash_reserve_pct: float,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """Pure planner — no IO. Returns (df, summary, cap_log_lines)."""
    universe = set(positions_map.keys()) | set(capital_deploy_map.keys())
    rows: List[Dict[str, Any]] = []

    for sym in sorted(universe):
        pos = positions_map.get(sym, {})
        cap_row = capital_deploy_map.get(sym, {})
        alloc_row = allocation_map.get(sym, {})
        risk_flag = (
            risk_overlay_map.get(sym)
            or cap_row.get("risk_flag")
            or alloc_row.get("risk_flag")
            or RISK_OK
        )
        risk_comps = _risk_components(risk_flag)

        is_held = sym in positions_map
        current_value = float(pos.get("market_value") or 0.0)
        current_weight_pct = (
            round(100.0 * current_value / total_portfolio_value, 4)
            if (is_held and total_portfolio_value > 0)
            else 0.0
        )

        deploy_decision = cap_row.get("deployment_decision") or ""
        alloc_action = alloc_row.get("recommended_action") or ""
        persistence_decision = persistence_map.get(sym, "")

        action, reason_tags = _initial_portfolio_action(
            sym=sym,
            is_held=is_held,
            deploy_decision=deploy_decision,
            alloc_action=alloc_action,
            risk_components=risk_comps,
            persistence_decision=persistence_decision,
        )

        # Proposed target_weight_pct from Step 4 (only meaningful for
        # OPEN/ADD; for EXIT/TRIM/HOLD we resolve later from current).
        proposed_target = float(cap_row.get("target_weight_pct") or 0.0)
        sector = get_sector(sym) or UNKNOWN_SECTOR_LABEL
        clusters = _ticker_clusters(sym)

        rows.append(
            {
                "ticker": sym,
                "current_weight_pct": round(current_weight_pct, 4),
                "target_weight_pct": (
                    round(proposed_target, 4) if action in {ACTION_OPEN, ACTION_ADD} else 0.0
                ),
                "target_position_size_usd": 0.0,  # resolved post-constraints
                "delta_weight_pct": 0.0,  # resolved post-constraints
                "sector_bucket": sector,
                "concentration_penalty": 0.0,
                "diversification_bonus": 0.0,
                "correlation_penalty": 0.0,
                "deploy_priority": round(
                    _clamp(_to_float_or_zero(cap_row.get("deploy_priority")), 0.0, 1.0), 6
                ),
                "portfolio_action": action,
                "reason": cap_row.get("reason") or "",  # may be overwritten by tags
                "_reason_tags": list(reason_tags),
                "_clusters": clusters,
            }
        )

    # Apply portfolio-level constraints, scaling target weights as needed.
    cap_logs = _apply_portfolio_constraints(
        rows,
        max_new_per_cycle=int(max_new_per_cycle),
        cash_reserve_pct=float(target_cash_reserve_pct),
    )

    # Resolve EXIT/TRIM/HOLD/BLOCK target weights + sizes + deltas.
    rows = _finalize_rows_and_sizes(rows, total_portfolio_value=total_portfolio_value)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # ── Distributions ────────────────────────────────────────────
    sector_distribution: Dict[str, float] = defaultdict(float)
    cluster_distribution: Dict[str, float] = defaultdict(float)
    for r in rows:
        tw = float(r["target_weight_pct"])
        if tw <= 0:
            continue
        sector_distribution[r["sector_bucket"]] += tw
        for cluster in _ticker_clusters(r["ticker"]):
            cluster_distribution[cluster] += tw

    total_target_weight = sum(sector_distribution.values())
    cash_buffer_pct = round(max(0.0, 100.0 - total_target_weight), 4)
    concentration_risk = round(max(sector_distribution.values()) if sector_distribution else 0.0, 4)

    scores = _compute_construction_score(rows, cash_reserve_pct=cash_buffer_pct)

    def _count(action: str) -> int:
        return int(sum(1 for r in rows if r["portfolio_action"] == action))

    # Top allocations: OPEN + ADD ranked by target_weight_pct desc, then deploy_priority.
    candidates = [r for r in rows if r["portfolio_action"] in {ACTION_OPEN, ACTION_ADD}]
    candidates.sort(
        key=lambda r: (
            float(r["target_weight_pct"]),
            float(r["deploy_priority"]),
            r["ticker"],
        ),
        reverse=True,
    )
    top_allocations = [
        {
            "ticker": r["ticker"],
            "portfolio_action": r["portfolio_action"],
            "target_weight_pct": float(r["target_weight_pct"]),
            "target_position_size_usd": float(r["target_position_size_usd"]),
            "sector_bucket": r["sector_bucket"],
            "deploy_priority": float(r["deploy_priority"]),
            "reason": r["reason"],
        }
        for r in candidates[:5]
    ]

    blocked = [
        {
            "ticker": r["ticker"],
            "reason": r["reason"],
        }
        for r in rows
        if r["portfolio_action"] == ACTION_BLOCK
    ]

    deployable_capital = _to_float(capital_summary.get("deployable_capital_estimate")) or 0.0
    max_positions = int(capital_summary.get("max_positions") or 0)
    available_slots = int(capital_summary.get("available_slots") or 0)
    cash_estimate = capital_summary.get("cash_estimate")
    reserve_pct = capital_summary.get("reserve_pct")

    summary: Dict[str, Any] = {
        "generated_at_utc": _now_iso_utc(),
        "total_target_positions": int(
            sum(1 for r in rows if float(r["target_weight_pct"]) >= MIN_POSITION_PCT)
        ),
        "current_positions": len(positions_map),
        "deploy_candidates": int(
            sum(1 for r in rows if r["portfolio_action"] in {ACTION_OPEN, ACTION_ADD})
        ),
        "open_position_count": _count(ACTION_OPEN),
        "add_to_position_count": _count(ACTION_ADD),
        "hold_count": _count(ACTION_HOLD),
        "trim_count": _count(ACTION_TRIM),
        "exit_count": _count(ACTION_EXIT),
        "block_count": _count(ACTION_BLOCK),
        "avg_position_weight": (
            round(
                float(
                    sum(
                        float(r["target_weight_pct"])
                        for r in rows
                        if float(r["target_weight_pct"]) > 0
                    )
                    / max(
                        1,
                        sum(1 for r in rows if float(r["target_weight_pct"]) > 0),
                    )
                ),
                4,
            )
            if rows
            else 0.0
        ),
        "sector_distribution": {k: round(v, 4) for k, v in sorted(sector_distribution.items())},
        "correlation_cluster_distribution": {
            k: round(v, 4) for k, v in sorted(cluster_distribution.items())
        },
        "concentration_risk": concentration_risk,
        "cash_reserve_pct": cash_buffer_pct,
        "target_cash_pct": round(float(target_cash_reserve_pct), 4),
        "total_portfolio_value": round(float(total_portfolio_value), 4),
        "deployable_capital": round(float(deployable_capital), 4),
        "cash_estimate": cash_estimate,
        "reserve_pct": reserve_pct,
        "max_positions": max_positions,
        "available_slots": available_slots,
        "max_new_positions_per_cycle": int(max_new_per_cycle),
        "diversification_score": scores["diversification_quality"],
        "portfolio_construction_score": scores["portfolio_construction_score"],
        "score_components": {
            k: scores[k]
            for k in (
                "diversification_quality",
                "conviction_quality",
                "risk_balance",
                "cash_discipline",
                "top_sector_pct",
                "represented_sectors",
            )
        },
        "constraints": {
            "max_single_position_pct": MAX_SINGLE_POSITION_PCT,
            "min_position_pct": MIN_POSITION_PCT,
            "max_sector_pct": MAX_SECTOR_PCT,
            "max_cluster_pct": MAX_CLUSTER_PCT,
            "min_cash_reserve_pct": MIN_CASH_RESERVE_PCT,
            "max_cash_reserve_pct": MAX_CASH_RESERVE_PCT,
        },
        "top_allocations": top_allocations,
        "blocked_allocations": blocked,
    }
    return df, summary, cap_logs


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only portfolio construction engine (step 5 of WATCH funnel). "
            "Merges held positions with deploy candidates and enforces "
            "portfolio-level constraints to produce a target portfolio."
        ),
    )
    p.add_argument("--capital-deploy", default=str(DEFAULT_CAPITAL_DEPLOY_CSV))
    p.add_argument("--capital-summary", default=str(DEFAULT_CAPITAL_DEPLOY_JSON))
    p.add_argument("--positions", default=str(DEFAULT_POSITIONS_CSV))
    p.add_argument("--risk-overlay", default=str(DEFAULT_RISK_OVERLAY_CSV))
    p.add_argument("--persistence", default=str(DEFAULT_PERSISTENCE_CSV))
    p.add_argument("--allocation", default=str(DEFAULT_ALLOCATION_CSV))
    p.add_argument("--out-csv", default=str(DEFAULT_OUTPUT_CSV))
    p.add_argument("--out-json", default=str(DEFAULT_OUTPUT_JSON))
    p.add_argument(
        "--max-new-per-cycle",
        type=int,
        default=MAX_NEW_POSITIONS_PER_CYCLE_DEFAULT,
        help="Hard cap on OPEN_POSITION actions per cycle (default %(default)s).",
    )
    p.add_argument(
        "--target-cash-pct",
        type=float,
        default=TARGET_CASH_RESERVE_PCT,
        help="Target cash reserve %% used by the constraint sweep.",
    )
    return p.parse_args(argv)


def _apply_runtime_policy(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Step 11 integration. Reads runtime_policy.json (if present) and
    overrides the construction engine's per-position cap. Safe to call
    every cycle -- missing/malformed file leaves defaults untouched.
    Path resolves via the module attribute at call time so tests can
    monkey-patch ``DEFAULT_RUNTIME_POLICY_JSON``.
    """
    global MAX_SINGLE_POSITION_PCT
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
            f"[PORTFOLIO_CONSTRUCTION_WARN] runtime_policy.json present but "
            f"unreadable ({type(e).__name__}: {e}); keeping defaults",
            flush=True,
        )
        return None
    aliases = rp.get("aliases") or {}
    max_pos = rp.get("max_position_pct")
    if max_pos is None:
        max_pos = aliases.get("max_single_position_pct")
    if max_pos is not None:
        try:
            MAX_SINGLE_POSITION_PCT = float(max_pos)
        except (TypeError, ValueError):
            pass
    print(
        "[PORTFOLIO_CONSTRUCTION_POLICY] "
        f"regime={rp.get('regime', 'UNKNOWN')} "
        f"max_single_position_pct={MAX_SINGLE_POSITION_PCT:.1f}",
        flush=True,
    )
    return rp


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[PORTFOLIO_CONSTRUCTION] starting (read-only intelligence layer)", flush=True)
    _apply_runtime_policy()

    cap_df = _safe_read_csv(
        Path(args.capital_deploy), label="capital_deployment_recommendations.csv"
    )
    cap_summary = _load_capital_summary(
        _safe_read_json(Path(args.capital_summary), label="capital_deployment_summary.json")
    )
    positions_df = _safe_read_csv(Path(args.positions), label="positions_snapshot.csv")
    risk_df = _safe_read_csv(Path(args.risk_overlay), label="performance_risk_overlay.csv")
    persistence_df = _safe_read_csv(
        Path(args.persistence), label="opportunity_persistence_recommendations.csv"
    )
    allocation_df = _safe_read_csv(
        Path(args.allocation), label="portfolio_allocation_recommendations.csv"
    )

    positions_map, positions_value = _load_positions(positions_df)
    # Build the canonical book value as positions + cash so that
    # current_weight_pct, sector totals, and the cash buffer all
    # reflect the *real* portfolio (positions-only weights would
    # collapse to 100% even when the broker holds material cash).
    cash_estimate = _to_float(cap_summary.get("cash_estimate"))
    if cash_estimate is None or cash_estimate < 0:
        cash_estimate = 0.0
    total_portfolio_value = positions_value + float(cash_estimate)
    if total_portfolio_value <= 0 and cap_summary.get("total_portfolio_value"):
        total_portfolio_value = float(cap_summary["total_portfolio_value"])

    capital_deploy_map = _load_capital_deploy_map(cap_df)
    allocation_map = _load_allocation_map(allocation_df)
    risk_overlay_map = _load_risk_overlay_map(risk_df)
    persistence_map = _load_persistence_map(persistence_df)

    df, summary, cap_logs = build_construction(
        positions_map=positions_map,
        total_portfolio_value=total_portfolio_value,
        capital_deploy_map=capital_deploy_map,
        allocation_map=allocation_map,
        risk_overlay_map=risk_overlay_map,
        persistence_map=persistence_map,
        capital_summary=cap_summary,
        max_new_per_cycle=int(args.max_new_per_cycle),
        target_cash_reserve_pct=float(args.target_cash_pct),
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

    for line in cap_logs:
        print(line, flush=True)

    print(
        "[PORTFOLIO_CONSTRUCTION] "
        f"positions={summary['current_positions']} "
        f"deploy={summary['deploy_candidates']} "
        f"open={summary['open_position_count']} "
        f"add={summary['add_to_position_count']} "
        f"trim={summary['trim_count']} "
        f"exit={summary['exit_count']} "
        f"block={summary['block_count']} "
        f"cash_buffer={summary['cash_reserve_pct']:.2f}% "
        f"construction_score={summary['portfolio_construction_score']:.3f}",
        flush=True,
    )
    print(
        "[PORTFOLIO_TOP_ALLOCATIONS] symbols="
        f"{[a['ticker'] for a in summary['top_allocations']]}",
        flush=True,
    )
    print(
        f"[PORTFOLIO_CONSTRUCTION_OUT] csv={out_csv.as_posix()} summary={out_json.as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
