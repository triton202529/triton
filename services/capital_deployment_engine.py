"""
Capital Deployment Decision Engine — Step 4 of the WATCH → DEPLOY funnel.

Reads:
    data/results/opportunity_persistence_recommendations.csv
    data/results/opportunity_persistence_summary.json
    data/results/portfolio_allocation_recommendations.csv
    data/results/positions_snapshot.csv
    data/results/performance_risk_overlay.csv
    data/results/edge_sizing_recommendations.csv
    data/results/signals_with_rationale.csv
    data/results/signal_lifecycle_effective.csv
    data/results/guard_snapshot.json            (best-effort cash hint)
    config/execute_trades.json                  (max_positions)

Writes:
    data/results/capital_deployment_recommendations.csv
    data/results/capital_deployment_summary.json

Purpose
-------
Step 3 (persistence) confirms which WATCH opportunities are persistent.
This engine answers the next question:

    "How much capital should Triton allocate, to which symbol, and why?"

It sizes each PROMOTE_CONFIRMED_OPEN_NEW (and high-persistence
KEEP_WATCH shadows) into a conviction-driven `target_weight_pct`, a
dollar `suggested_position_size_usd`, and a final `deployment_decision`
of DEPLOY_NOW / SHADOW_WATCH / BLOCK.

Safety
------
* Read-only. No orders, no broker calls, no mutation of execute_trades
  or manage_positions. Sizes are recommendations only; execute_trades
  retains all real capital controls.
* Missing inputs warn and continue (each loader yields empty data).
* Atomic writes for all outputs via `*.tmp` + `os.replace`.
* main() returns 0 on success, 2 only on output-write failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
CONFIG_DIR = ROOT / "config"

DEFAULT_PERSISTENCE_CSV = RESULTS_DIR / "opportunity_persistence_recommendations.csv"
DEFAULT_PERSISTENCE_JSON = RESULTS_DIR / "opportunity_persistence_summary.json"
DEFAULT_ALLOCATION_CSV = RESULTS_DIR / "portfolio_allocation_recommendations.csv"
DEFAULT_POSITIONS_CSV = RESULTS_DIR / "positions_snapshot.csv"
DEFAULT_RISK_OVERLAY_CSV = RESULTS_DIR / "performance_risk_overlay.csv"
DEFAULT_EDGE_SIZING_CSV = RESULTS_DIR / "edge_sizing_recommendations.csv"
DEFAULT_SIGNALS_CSV = RESULTS_DIR / "signals_with_rationale.csv"
DEFAULT_LIFECYCLE_EFFECTIVE_CSV = RESULTS_DIR / "signal_lifecycle_effective.csv"
DEFAULT_GUARD_SNAPSHOT_JSON = RESULTS_DIR / "guard_snapshot.json"
DEFAULT_EXECUTE_TRADES_CONFIG = CONFIG_DIR / "execute_trades.json"

DEFAULT_OUTPUT_CSV = RESULTS_DIR / "capital_deployment_recommendations.csv"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "capital_deployment_summary.json"

# -----------------------------------------------------------
# Tunables (analytics-only thresholds)
# -----------------------------------------------------------
# Decision labels
DECISION_DEPLOY = "DEPLOY_NOW"
DECISION_SHADOW = "SHADOW_WATCH"
DECISION_BLOCK = "BLOCK"

# Candidate filter
SHADOW_PERSISTENCE_FLOOR = 0.75  # KEEP_WATCH with persistence >= this becomes SHADOW
DEPLOY_PERSISTENCE_FLOOR = 0.60  # rule A persistence gate
MIN_DEPLOY_CONFIDENCE = 0.50  # rule A confidence gate
MIN_KEEP_CONFIDENCE = 0.45  # rule C confidence rejection floor

# Step 11 runtime policy override (optional, additive).
# When data/results/runtime_policy.json exists, these defaults are
# overridden for the current cycle. If the file is missing or malformed,
# behaviour is identical to the baseline above.
DEFAULT_RUNTIME_POLICY_JSON = RESULTS_DIR / "runtime_policy.json"

# Conviction tiers for base target weight (%).
TIER_VERY_HIGH_PERSISTENCE = 0.85
TIER_VERY_HIGH_CONFIDENCE = 0.65
TIER_HIGH_PERSISTENCE = 0.75
TIER_MODERATE_PERSISTENCE = 0.60

BASE_WEIGHT_VERY_HIGH = 7.0  # midpoint of 6–8%
BASE_WEIGHT_HIGH = 5.0  # midpoint of 4–6%
BASE_WEIGHT_MODERATE = 3.0  # midpoint of 2–4%
BASE_WEIGHT_WEAK = 1.0  # <= 2%

TARGET_WEIGHT_MIN_PCT = 0.5
TARGET_WEIGHT_MAX_PCT = 8.0

# Adjustment magnitudes (percentage points).
ADJ_STRONG_EDGE = +1.0
ADJ_STRONG_DELTA = +0.5
ADJ_UNDERWEIGHT_BOOK = +0.5
ADJ_TRIM_PRIORITY = -2.0
ADJ_FORCE_EXIT = -4.0
ADJ_BLOCK_NEW_BUY = -3.0
ADJ_OVERWEIGHT_POSITION = -1.0
ADJ_WEAK_CONFIDENCE = -1.0

STRONG_EDGE_SCORE = 0.50
STRONG_DELTA_PCT = 0.005
WEAK_CONFIDENCE_THRESHOLD = 0.50

# Capacity fallbacks.
DEPLOYABLE_FALLBACK_FRACTION = 0.20  # used only when broker buying_power unknown
RESERVE_PCT_FALLBACK = 0.20
MAX_POSITIONS_FALLBACK = 25
UNDERWEIGHT_BOOK_FRACTION = 0.50  # available_slots/max_positions above this = bonus
OVERWEIGHT_POSITION_PCT = 15.0  # mirrors allocation_engine OVERWEIGHT band

# Risk flag string components (consistent with allocation engine).
RISK_FLAG_FORCE_EXIT = "FORCE_EXIT"
RISK_FLAG_TRIM_PRIORITY = "TRIM_PRIORITY"
RISK_FLAG_BLOCK_NEW_BUY = "BLOCK_NEW_BUY"
RISK_FLAG_OK = "OK"

# Labels treated as positive intent.
POSITIVE_SIGNAL_LABELS: frozenset = frozenset({"BUY", "ADD"})
LIFECYCLE_ACTION_POSITIVE: frozenset = frozenset({"BUY", "ADD"})

# Persistence-engine decisions
PERSISTENCE_PROMOTE_CONFIRMED = "PROMOTE_CONFIRMED_OPEN_NEW"
PERSISTENCE_KEEP_WATCH = "KEEP_WATCH"

OUTPUT_COLUMNS = [
    "ticker",
    "persistence_score",
    "promotion_score",
    "allocation_score",
    "confidence",
    "delta_pct",
    "edge_score",
    "risk_flag",
    "lifecycle_action",
    "effective_stance",
    "current_position_weight",
    "target_weight_pct",
    "suggested_position_size_usd",
    "deploy_priority",
    "deployment_decision",
    "reason",
]


# -----------------------------------------------------------
# Safe IO helpers
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[CAPITAL_DEPLOYMENT_WARN] {msg}", flush=True)


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
    if isinstance(o, (datetime,)):
        return o.replace(microsecond=0).isoformat()
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            return str(o)
    try:
        return float(o)
    except Exception:
        return str(o)


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


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# -----------------------------------------------------------
# Capacity probes
# -----------------------------------------------------------
def _load_max_positions(config_path: Path) -> Tuple[int, str]:
    """Read max_positions from execute_trades.json; fall back to constant."""
    try:
        if not config_path.is_file():
            return MAX_POSITIONS_FALLBACK, "fallback_default"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        v = cfg.get("max_positions")
        if v is None:
            return MAX_POSITIONS_FALLBACK, "fallback_default"
        return max(1, int(v)), "execute_trades.json"
    except Exception as e:
        _warn(
            f"could not read max_positions from {config_path}: "
            f"{type(e).__name__}: {e}; using fallback {MAX_POSITIONS_FALLBACK}"
        )
        return MAX_POSITIONS_FALLBACK, "fallback_default"


def _load_capacity(
    positions_df: pd.DataFrame,
    guard_snapshot: Optional[Dict[str, Any]],
    max_positions: int,
    deployable_fallback_fraction: float,
) -> Dict[str, Any]:
    """
    Compute portfolio capacity. Cash is best-effort:
      * guard_snapshot.json.buying_power when present and > 0 → real cash
      * otherwise → fraction-of-book heuristic so sizing degrades gracefully
    """
    total_value = 0.0
    held_syms: List[str] = []
    if positions_df is not None and not positions_df.empty:
        sym_col = _pick_first_present(positions_df, ("ticker", "symbol"))
        mv_col = _pick_first_present(positions_df, ("market_value", "value"))
        qty_col = _pick_first_present(positions_df, ("qty", "qty_available"))
        if sym_col and mv_col:
            for _, r in positions_df.iterrows():
                mv = _to_float_or_zero(r.get(mv_col))
                qty = _to_float_or_zero(r.get(qty_col)) if qty_col else 1.0
                if mv <= 0 or qty <= 0:
                    continue
                sym = _norm_symbol(r.get(sym_col))
                if not sym:
                    continue
                total_value += mv
                held_syms.append(sym)

    current_positions = len(set(held_syms))

    cash_estimate: Optional[float] = None
    reserve_pct = RESERVE_PCT_FALLBACK
    if guard_snapshot:
        bp = _to_float(guard_snapshot.get("buying_power"))
        if bp is not None and bp > 0:
            cash_estimate = float(bp)
        rp = _to_float(guard_snapshot.get("reserve_pct"))
        if rp is not None and 0.0 <= rp < 1.0:
            reserve_pct = float(rp)

    if cash_estimate is not None:
        deployable = max(0.0, cash_estimate * (1.0 - reserve_pct))
        deployable_source = "guard_snapshot.buying_power"
    else:
        # Fraction-of-book heuristic: assume only this fraction of the book
        # is realistically deployable in any one run. Reserve is NOT applied
        # again because the fraction itself already encodes conservatism.
        deployable = max(0.0, total_value * float(deployable_fallback_fraction))
        deployable_source = "fraction_of_book_fallback"

    current_exposure_pct = (
        100.0
        if (total_value > 0 and not cash_estimate)
        else (round(100.0 * total_value / max(total_value + (cash_estimate or 0.0), 1e-9), 4))
    )

    available_slots = max(0, int(max_positions) - current_positions)

    return {
        "total_portfolio_value": round(total_value, 4),
        "cash_estimate": (round(cash_estimate, 4) if cash_estimate is not None else None),
        "reserve_pct": round(reserve_pct, 4),
        "current_positions": current_positions,
        "max_positions": int(max_positions),
        "available_slots": available_slots,
        "current_exposure_pct": round(current_exposure_pct, 4),
        "deployable_capital_estimate": round(deployable, 4),
        "deployable_capital_source": deployable_source,
    }


# -----------------------------------------------------------
# Loaders for enrichment
# -----------------------------------------------------------
def _load_persistence_candidates(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """{ticker: persistence fields}. Filters to PROMOTE_CONFIRMED + KEEP_WATCH≥0.75."""
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return out

    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col or "promotion_decision" not in df.columns:
        _warn(
            "opportunity_persistence_recommendations.csv missing ticker/symbol "
            "or promotion_decision column; no deployment candidates"
        )
        return out

    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        decision = _norm_upper(r.get("promotion_decision"))
        persistence = _to_float(r.get("persistence_score"))
        if decision == PERSISTENCE_PROMOTE_CONFIRMED:
            tier = "PRIMARY"
        elif (
            decision == PERSISTENCE_KEEP_WATCH and (persistence or 0.0) >= SHADOW_PERSISTENCE_FLOOR
        ):
            tier = "SHADOW"
        else:
            continue

        out[sym] = {
            "candidate_tier": tier,
            "persistence_score": persistence,
            "promotion_score": _to_float(r.get("latest_promotion_score"))
            or _to_float(r.get("promotion_score")),
            "confidence": _to_float(r.get("latest_confidence")) or _to_float(r.get("confidence")),
            "delta_pct": _to_float(r.get("latest_delta_pct")) or _to_float(r.get("delta_pct")),
            "promotion_score_trend": _to_float(r.get("promotion_score_trend")) or 0.0,
            "confidence_trend": _to_float(r.get("confidence_trend")) or 0.0,
            "delta_trend": _to_float(r.get("delta_trend")) or 0.0,
            "consecutive_watch_cycles": int(_to_float_or_zero(r.get("consecutive_watch_cycles"))),
        }
    return out


def _load_allocation_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """{ticker: allocation engine row}. Canonical risk_flag / edge_score / weight."""
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
            "allocation_score": _to_float(r.get("allocation_score")),
            "edge_score": _to_float(r.get("edge_score")),
            "risk_flag": _norm_upper(r.get("risk_flag")),
            "lifecycle_action": _norm_upper(r.get("lifecycle_action")),
            "effective_stance": _norm_upper(r.get("effective_stance")),
            "signal": _norm_upper(r.get("signal")),
            "sizing_tier": _norm_upper(r.get("sizing_tier")),
            "current_weight_pct": _to_float(r.get("current_weight_pct")),
            "is_currently_held": _to_bool(r.get("is_currently_held")),
        }
    return out


def _load_risk_overlay_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Canonical risk_flag source (overrides allocation when both present)."""
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
            "risk_flag": _norm_upper(r.get("risk_flag")),
        }
    return out


def _load_edge_sizing_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
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
            "edge_score": _to_float(r.get("edge_score")),
            "sizing_tier": _norm_upper(r.get("sizing_tier")),
            "size_multiplier": _to_float(r.get("size_multiplier")),
            "risk_flag": _norm_upper(r.get("risk_flag")),
        }
    return out


def _load_signals_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return out
    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        return out
    delta_col = _pick_first_present(df, ("delta_pct", "delta_pct_snapshot"))
    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        out[sym] = {
            "signal": _norm_upper(r.get("signal")),
            "confidence": _to_float(r.get("confidence")),
            "delta_pct": _to_float(r.get(delta_col)) if delta_col else None,
        }
    return out


def _load_lifecycle_effective_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return out
    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        return out
    delta_col = _pick_first_present(df, ("delta_pct", "delta_pct_snapshot"))
    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        out[sym] = {
            "lifecycle_action": _norm_upper(r.get("lifecycle_action")),
            "effective_stance": _norm_upper(r.get("effective_stance"))
            or _norm_upper(r.get("stance")),
            "signal": _norm_upper(r.get("signal")),
            "confidence": _to_float(r.get("confidence")),
            "delta_pct": _to_float(r.get(delta_col)) if delta_col else None,
        }
    return out


# -----------------------------------------------------------
# Sizing & decision logic
# -----------------------------------------------------------
def _risk_components(risk_flag: str) -> List[str]:
    if not risk_flag:
        return []
    parts = [p.strip().upper() for p in str(risk_flag).split("|")]
    return [p for p in parts if p and p != RISK_FLAG_OK]


def _is_strong_edge(edge_score: Optional[float]) -> bool:
    return edge_score is not None and edge_score >= STRONG_EDGE_SCORE


def _base_target_weight(
    persistence: Optional[float], confidence: Optional[float]
) -> Tuple[float, str]:
    """Return (base_weight_pct, conviction_tier_label)."""
    p = persistence or 0.0
    c = confidence or 0.0
    if p >= TIER_VERY_HIGH_PERSISTENCE and c >= TIER_VERY_HIGH_CONFIDENCE:
        return BASE_WEIGHT_VERY_HIGH, "very_high"
    if p >= TIER_HIGH_PERSISTENCE:
        return BASE_WEIGHT_HIGH, "high"
    if p >= TIER_MODERATE_PERSISTENCE:
        return BASE_WEIGHT_MODERATE, "moderate"
    return BASE_WEIGHT_WEAK, "weak"


def _weight_adjustments(
    *,
    edge_score: Optional[float],
    delta_pct: Optional[float],
    confidence: Optional[float],
    current_weight_pct: Optional[float],
    risk_components: List[str],
    available_slots: int,
    max_positions: int,
) -> Tuple[float, List[str]]:
    """Sum adjustments and return (delta_pct_points, applied_reason_tags)."""
    adj = 0.0
    tags: List[str] = []

    if _is_strong_edge(edge_score):
        adj += ADJ_STRONG_EDGE
        tags.append(f"+strong_edge({edge_score:.2f})")
    if delta_pct is not None and delta_pct >= STRONG_DELTA_PCT:
        adj += ADJ_STRONG_DELTA
        tags.append(f"+strong_delta({delta_pct:.4f})")

    # Underweight book bonus: when we have plenty of available slots
    # (>UNDERWEIGHT_BOOK_FRACTION of max), favor adding new positions.
    if max_positions > 0:
        slot_frac = available_slots / float(max_positions)
        if slot_frac >= UNDERWEIGHT_BOOK_FRACTION:
            adj += ADJ_UNDERWEIGHT_BOOK
            tags.append(f"+underweight_book({slot_frac:.2f})")

    if RISK_FLAG_TRIM_PRIORITY in risk_components:
        adj += ADJ_TRIM_PRIORITY
        tags.append("-trim_priority")
    if RISK_FLAG_FORCE_EXIT in risk_components:
        adj += ADJ_FORCE_EXIT
        tags.append("-force_exit")
    if RISK_FLAG_BLOCK_NEW_BUY in risk_components:
        adj += ADJ_BLOCK_NEW_BUY
        tags.append("-block_new_buy")

    if current_weight_pct is not None and current_weight_pct > OVERWEIGHT_POSITION_PCT:
        adj += ADJ_OVERWEIGHT_POSITION
        tags.append(f"-overweight({current_weight_pct:.1f}%)")

    if confidence is not None and confidence < WEAK_CONFIDENCE_THRESHOLD:
        adj += ADJ_WEAK_CONFIDENCE
        tags.append(f"-weak_confidence({confidence:.2f})")

    return adj, tags


def _deploy_priority(
    *,
    persistence_score: Optional[float],
    promotion_score: Optional[float],
    allocation_score: Optional[float],
    confidence: Optional[float],
    edge_score: Optional[float],
    delta_pct: Optional[float],
    risk_components: List[str],
    available_slots: int,
    max_positions: int,
) -> float:
    """Convex combination of conviction signals minus a risk penalty, clamped to [0,1]."""
    persistence_n = _clamp(_to_float_or_zero(persistence_score), 0.0, 1.0)
    # promotion / allocation are roughly in [-1, 1]; map to [0, 1] via (x+1)/2.
    promotion_n = _clamp((_to_float_or_zero(promotion_score) + 1.0) * 0.5, 0.0, 1.0)
    allocation_n = _clamp((_to_float_or_zero(allocation_score) + 1.0) * 0.5, 0.0, 1.0)
    confidence_n = _clamp(_to_float_or_zero(confidence), 0.0, 1.0)
    edge_n = _clamp((_to_float_or_zero(edge_score) + 1.0) * 0.5, 0.0, 1.0)
    delta_n = _clamp((_to_float_or_zero(delta_pct) * 20.0 + 1.0) * 0.5, 0.0, 1.0)

    diversification_bonus = 0.0
    if max_positions > 0:
        diversification_bonus = _clamp(available_slots / float(max_positions), 0.0, 1.0)

    risk_penalty = 0.0
    if RISK_FLAG_FORCE_EXIT in risk_components:
        risk_penalty += 0.50
    if RISK_FLAG_BLOCK_NEW_BUY in risk_components:
        risk_penalty += 0.30
    if RISK_FLAG_TRIM_PRIORITY in risk_components:
        risk_penalty += 0.20

    score = (
        0.25 * persistence_n
        + 0.20 * promotion_n
        + 0.15 * allocation_n
        + 0.15 * confidence_n
        + 0.10 * edge_n
        + 0.10 * delta_n
        + 0.05 * diversification_bonus
        - risk_penalty
    )
    return round(_clamp(score, 0.0, 1.0), 6)


def _decide_deployment(
    *,
    persistence_score: Optional[float],
    confidence: Optional[float],
    delta_pct: Optional[float],
    signal: str,
    lifecycle_action: str,
    risk_components: List[str],
    candidate_tier: str,
) -> Tuple[str, str]:
    """
    Apply precedence: BLOCK → DEPLOY_NOW → SHADOW_WATCH.

    Reasons mirror the spec examples so dashboards can filter cleanly.
    """
    pos_sig = signal in POSITIVE_SIGNAL_LABELS
    pos_lc = lifecycle_action in LIFECYCLE_ACTION_POSITIVE
    pos_either = pos_sig or pos_lc

    # ── C: BLOCK ────────────────────────────────────────────────────
    if RISK_FLAG_FORCE_EXIT in risk_components or RISK_FLAG_BLOCK_NEW_BUY in risk_components:
        return DECISION_BLOCK, "blocked_by_risk_overlay"
    if confidence is None or confidence < MIN_KEEP_CONFIDENCE:
        return DECISION_BLOCK, "confidence_too_low"
    if delta_pct is None or delta_pct <= 0:
        return DECISION_BLOCK, "negative_delta"
    if not pos_either:
        return DECISION_BLOCK, "no_positive_signal_or_lifecycle"

    # ── A: DEPLOY_NOW ───────────────────────────────────────────────
    deploy_ok = (
        candidate_tier == "PRIMARY"
        and (persistence_score is not None and persistence_score >= DEPLOY_PERSISTENCE_FLOOR)
        and confidence >= MIN_DEPLOY_CONFIDENCE
        and delta_pct > 0
        and pos_either
    )
    if deploy_ok:
        # Tiered reason: high-conviction setups get a stronger label.
        if (
            persistence_score is not None
            and persistence_score >= TIER_VERY_HIGH_PERSISTENCE
            and confidence >= TIER_VERY_HIGH_CONFIDENCE
        ):
            return DECISION_DEPLOY, "high_conviction_setup"
        return DECISION_DEPLOY, "confirmed_persistent_signal"

    # ── B: SHADOW_WATCH ────────────────────────────────────────────
    return DECISION_SHADOW, "shadow_candidate_waiting_confirmation"


# -----------------------------------------------------------
# Build recommendations
# -----------------------------------------------------------
def build_deployments(
    *,
    persistence_map: Dict[str, Dict[str, Any]],
    allocation_map: Dict[str, Dict[str, Any]],
    risk_overlay_map: Dict[str, Dict[str, Any]],
    edge_sizing_map: Dict[str, Dict[str, Any]],
    signals_map: Dict[str, Dict[str, Any]],
    lifecycle_map: Dict[str, Dict[str, Any]],
    capacity: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """For each persistence-confirmed candidate produce one sized row."""
    rows: List[Dict[str, Any]] = []

    deployable_capital = float(capacity.get("deployable_capital_estimate") or 0.0)
    available_slots = int(capacity.get("available_slots") or 0)
    max_positions = int(capacity.get("max_positions") or MAX_POSITIONS_FALLBACK)

    for sym in sorted(persistence_map.keys()):
        p = persistence_map[sym]
        alloc = allocation_map.get(sym, {})
        risk_row = risk_overlay_map.get(sym, {})
        edge_row = edge_sizing_map.get(sym, {})
        sig_row = signals_map.get(sym, {})
        lc_row = lifecycle_map.get(sym, {})

        # ── Field resolution with documented precedence ──────────────
        confidence = p.get("confidence") or lc_row.get("confidence") or sig_row.get("confidence")
        delta_pct = p.get("delta_pct") or lc_row.get("delta_pct") or sig_row.get("delta_pct")
        edge_score = (
            alloc.get("edge_score")
            if alloc.get("edge_score") is not None
            else edge_row.get("edge_score")
        )
        # risk_flag precedence: overlay (canonical) > allocation (mirrored) > edge.
        risk_flag = (
            risk_row.get("risk_flag")
            or alloc.get("risk_flag")
            or edge_row.get("risk_flag")
            or RISK_FLAG_OK
        )
        if not risk_flag:
            risk_flag = RISK_FLAG_OK
        risk_comps = _risk_components(risk_flag)

        lifecycle_action = lc_row.get("lifecycle_action") or alloc.get("lifecycle_action") or ""
        effective_stance = lc_row.get("effective_stance") or alloc.get("effective_stance") or ""
        signal = lc_row.get("signal") or alloc.get("signal") or sig_row.get("signal") or ""

        allocation_score = alloc.get("allocation_score")
        current_weight_pct = alloc.get("current_weight_pct")

        # ── Base + adjusted target weight ───────────────────────────
        base_weight, tier_label = _base_target_weight(p.get("persistence_score"), confidence)
        adj_pts, adj_tags = _weight_adjustments(
            edge_score=edge_score,
            delta_pct=delta_pct,
            confidence=confidence,
            current_weight_pct=current_weight_pct,
            risk_components=risk_comps,
            available_slots=available_slots,
            max_positions=max_positions,
        )
        raw_weight = base_weight + adj_pts
        target_weight_pct = round(
            _clamp(raw_weight, TARGET_WEIGHT_MIN_PCT, TARGET_WEIGHT_MAX_PCT), 4
        )

        # Dollar size — capped at deployable capital so the sum across
        # all candidates can never exceed available capital even in
        # extreme cases (executor still enforces its own caps).
        size_usd = round((target_weight_pct / 100.0) * deployable_capital, 2)

        priority = _deploy_priority(
            persistence_score=p.get("persistence_score"),
            promotion_score=p.get("promotion_score"),
            allocation_score=allocation_score,
            confidence=confidence,
            edge_score=edge_score,
            delta_pct=delta_pct,
            risk_components=risk_comps,
            available_slots=available_slots,
            max_positions=max_positions,
        )

        decision, decision_reason = _decide_deployment(
            persistence_score=p.get("persistence_score"),
            confidence=confidence,
            delta_pct=delta_pct,
            signal=signal,
            lifecycle_action=lifecycle_action,
            risk_components=risk_comps,
            candidate_tier=p.get("candidate_tier", "PRIMARY"),
        )

        # Compose a single human-readable reason with tier + sizing notes.
        reason_parts: List[str] = [decision_reason]
        reason_parts.append(f"tier={tier_label}({p.get('candidate_tier','PRIMARY')})")
        if adj_tags:
            reason_parts.append("adj=" + ",".join(adj_tags))
        reason = "|".join(reason_parts)

        rows.append(
            {
                "ticker": sym,
                "persistence_score": (round(float(p.get("persistence_score") or 0.0), 6)),
                "promotion_score": (
                    round(float(p["promotion_score"]), 6)
                    if p.get("promotion_score") is not None
                    else None
                ),
                "allocation_score": (
                    round(float(allocation_score), 6) if allocation_score is not None else None
                ),
                "confidence": (round(float(confidence), 6) if confidence is not None else None),
                "delta_pct": (round(float(delta_pct), 6) if delta_pct is not None else None),
                "edge_score": (round(float(edge_score), 6) if edge_score is not None else None),
                "risk_flag": risk_flag,
                "lifecycle_action": lifecycle_action,
                "effective_stance": effective_stance,
                "current_position_weight": (
                    round(float(current_weight_pct), 4) if current_weight_pct is not None else 0.0
                ),
                "target_weight_pct": target_weight_pct,
                "suggested_position_size_usd": size_usd,
                "deploy_priority": priority,
                "deployment_decision": decision,
                "reason": reason,
            }
        )

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    def _count(dec: str) -> int:
        if df.empty:
            return 0
        return int((df["deployment_decision"] == dec).sum())

    deploy_n = _count(DECISION_DEPLOY)
    shadow_n = _count(DECISION_SHADOW)
    blocked_n = _count(DECISION_BLOCK)

    avg_target = (
        float(round(df["target_weight_pct"].astype(float).mean(), 4)) if not df.empty else 0.0
    )
    total_target_capital = (
        float(
            round(
                df.loc[df["deployment_decision"] == DECISION_DEPLOY, "suggested_position_size_usd"]
                .astype(float)
                .sum(),
                2,
            )
        )
        if not df.empty
        else 0.0
    )

    # top_deployments: PRIMARY DEPLOY_NOW first, then SHADOW, ranked by
    # deploy_priority desc, confidence desc, ticker asc; cap at 5.
    top_list: List[Dict[str, Any]] = []
    if not df.empty:
        d = df[df["deployment_decision"].isin([DECISION_DEPLOY, DECISION_SHADOW])].copy()
        if not d.empty:
            d["__class"] = (d["deployment_decision"] == DECISION_DEPLOY).astype(int)
            d["__pri"] = pd.to_numeric(d["deploy_priority"], errors="coerce").fillna(-1e9)
            d["__conf"] = pd.to_numeric(d["confidence"], errors="coerce").fillna(-1e9)
            d = d.sort_values(
                by=["__class", "__pri", "__conf", "ticker"],
                ascending=[False, False, False, True],
            )
            for _, r in d.head(5).iterrows():
                top_list.append(
                    {
                        "ticker": str(r["ticker"]),
                        "deployment_decision": str(r["deployment_decision"]),
                        "deploy_priority": float(r["__pri"]),
                        "target_weight_pct": float(r["target_weight_pct"]),
                        "suggested_position_size_usd": float(r["suggested_position_size_usd"]),
                        "persistence_score": float(r["persistence_score"]),
                        "confidence": (
                            float(r["confidence"]) if pd.notna(r["confidence"]) else None
                        ),
                        "reason": str(r["reason"]),
                    }
                )

    summary: Dict[str, Any] = {
        "generated_at_utc": _now_iso_utc(),
        "total_candidates": int(len(df)),
        "deploy_now_count": deploy_n,
        "shadow_watch_count": shadow_n,
        "blocked_count": blocked_n,
        "available_slots": int(available_slots),
        "current_positions": int(capacity.get("current_positions") or 0),
        "max_positions": int(capacity.get("max_positions") or MAX_POSITIONS_FALLBACK),
        "total_portfolio_value": capacity.get("total_portfolio_value"),
        "cash_estimate": capacity.get("cash_estimate"),
        "reserve_pct": capacity.get("reserve_pct"),
        "current_exposure_pct": capacity.get("current_exposure_pct"),
        "deployable_capital_estimate": capacity.get("deployable_capital_estimate"),
        "deployable_capital_source": capacity.get("deployable_capital_source"),
        "avg_target_weight": avg_target,
        "total_target_capital": total_target_capital,
        "thresholds": {
            "shadow_persistence_floor": SHADOW_PERSISTENCE_FLOOR,
            "deploy_persistence_floor": DEPLOY_PERSISTENCE_FLOOR,
            "min_deploy_confidence": MIN_DEPLOY_CONFIDENCE,
            "min_keep_confidence": MIN_KEEP_CONFIDENCE,
            "target_weight_min_pct": TARGET_WEIGHT_MIN_PCT,
            "target_weight_max_pct": TARGET_WEIGHT_MAX_PCT,
            "base_weight_very_high": BASE_WEIGHT_VERY_HIGH,
            "base_weight_high": BASE_WEIGHT_HIGH,
            "base_weight_moderate": BASE_WEIGHT_MODERATE,
            "base_weight_weak": BASE_WEIGHT_WEAK,
        },
        "top_deployments": top_list,
    }
    return df, summary


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only capital deployment decision engine (step 4 of WATCH funnel). "
            "Converts persistence-confirmed opportunities into sized recommendations."
        ),
    )
    p.add_argument("--persistence", default=str(DEFAULT_PERSISTENCE_CSV))
    p.add_argument("--persistence-summary", default=str(DEFAULT_PERSISTENCE_JSON))
    p.add_argument("--allocation", default=str(DEFAULT_ALLOCATION_CSV))
    p.add_argument("--positions", default=str(DEFAULT_POSITIONS_CSV))
    p.add_argument("--risk-overlay", default=str(DEFAULT_RISK_OVERLAY_CSV))
    p.add_argument("--edge-sizing", default=str(DEFAULT_EDGE_SIZING_CSV))
    p.add_argument("--signals", default=str(DEFAULT_SIGNALS_CSV))
    p.add_argument("--lifecycle-effective", default=str(DEFAULT_LIFECYCLE_EFFECTIVE_CSV))
    p.add_argument("--guard-snapshot", default=str(DEFAULT_GUARD_SNAPSHOT_JSON))
    p.add_argument("--config", default=str(DEFAULT_EXECUTE_TRADES_CONFIG))
    p.add_argument(
        "--deployable-fallback-fraction",
        type=float,
        default=DEPLOYABLE_FALLBACK_FRACTION,
        help=(
            "Used only when broker buying_power is unknown: estimate deployable "
            "capital as this fraction of total_portfolio_value."
        ),
    )
    p.add_argument("--out-csv", default=str(DEFAULT_OUTPUT_CSV))
    p.add_argument("--out-json", default=str(DEFAULT_OUTPUT_JSON))
    return p.parse_args(argv)


def _apply_runtime_policy(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Step 11 integration. Reads runtime_policy.json (if present) and
    overrides the conviction/persistence floors used by `_decide_action`.

    Returns the runtime policy dict (or None when absent). Safe to call
    every cycle — missing or malformed files leave defaults untouched
    and only emit a warning.

    The default path is looked up via the module attribute at call time
    so tests can monkey-patch ``DEFAULT_RUNTIME_POLICY_JSON`` to point
    at a synthetic location.
    """
    global MIN_DEPLOY_CONFIDENCE, DEPLOY_PERSISTENCE_FLOOR
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
            f"[CAPITAL_DEPLOYMENT_WARN] runtime_policy.json present but unreadable "
            f"({type(e).__name__}: {e}); keeping defaults",
            flush=True,
        )
        return None
    aliases = rp.get("aliases") or {}
    conf_v = rp.get("confidence_threshold")
    if conf_v is None:
        conf_v = aliases.get("min_deploy_confidence")
    pers_v = rp.get("persistence_threshold")
    if pers_v is None:
        pers_v = aliases.get("deploy_persistence_floor")
    if conf_v is not None:
        try:
            MIN_DEPLOY_CONFIDENCE = float(conf_v)
        except (TypeError, ValueError):
            pass
    if pers_v is not None:
        try:
            DEPLOY_PERSISTENCE_FLOOR = float(pers_v)
        except (TypeError, ValueError):
            pass
    print(
        "[CAPITAL_DEPLOYMENT_POLICY] "
        f"regime={rp.get('regime', 'UNKNOWN')} "
        f"confidence>={MIN_DEPLOY_CONFIDENCE:.2f} "
        f"persistence>={DEPLOY_PERSISTENCE_FLOOR:.2f}",
        flush=True,
    )
    return rp


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    print("[CAPITAL_DEPLOYMENT] starting (read-only intelligence layer)", flush=True)
    _apply_runtime_policy()

    persistence_df = _safe_read_csv(
        Path(args.persistence), label="opportunity_persistence_recommendations.csv"
    )
    _ = _safe_read_json(
        Path(args.persistence_summary), label="opportunity_persistence_summary.json"
    )
    allocation_df = _safe_read_csv(
        Path(args.allocation), label="portfolio_allocation_recommendations.csv"
    )
    positions_df = _safe_read_csv(Path(args.positions), label="positions_snapshot.csv")
    risk_df = _safe_read_csv(Path(args.risk_overlay), label="performance_risk_overlay.csv")
    edge_df = _safe_read_csv(Path(args.edge_sizing), label="edge_sizing_recommendations.csv")
    signals_df = _safe_read_csv(Path(args.signals), label="signals_with_rationale.csv")
    lifecycle_df = _safe_read_csv(
        Path(args.lifecycle_effective), label="signal_lifecycle_effective.csv"
    )
    guard_snapshot = _safe_read_json(Path(args.guard_snapshot), label="guard_snapshot.json")

    max_positions, max_positions_source = _load_max_positions(Path(args.config))

    capacity = _load_capacity(
        positions_df=positions_df,
        guard_snapshot=guard_snapshot,
        max_positions=max_positions,
        deployable_fallback_fraction=float(args.deployable_fallback_fraction),
    )
    capacity["max_positions_source"] = max_positions_source

    persistence_map = _load_persistence_candidates(persistence_df)
    allocation_map = _load_allocation_map(allocation_df)
    risk_overlay_map = _load_risk_overlay_map(risk_df)
    edge_sizing_map = _load_edge_sizing_map(edge_df)
    signals_map = _load_signals_map(signals_df)
    lifecycle_map = _load_lifecycle_effective_map(lifecycle_df)

    df, summary = build_deployments(
        persistence_map=persistence_map,
        allocation_map=allocation_map,
        risk_overlay_map=risk_overlay_map,
        edge_sizing_map=edge_sizing_map,
        signals_map=signals_map,
        lifecycle_map=lifecycle_map,
        capacity=capacity,
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
        "[CAPITAL_DEPLOYMENT] "
        f"positions={summary['current_positions']} "
        f"available_slots={summary['available_slots']} "
        f"deploy_now={summary['deploy_now_count']} "
        f"shadow={summary['shadow_watch_count']} "
        f"blocked={summary['blocked_count']} "
        f"deployable_capital={summary['deployable_capital_estimate']}",
        flush=True,
    )
    print(
        "[CAPITAL_TOP_DEPLOYMENTS] symbols="
        f"{[o['ticker'] for o in summary.get('top_deployments', [])]}",
        flush=True,
    )
    print(
        f"[CAPITAL_DEPLOYMENT_OUT] csv={out_csv.as_posix()} summary={out_json.as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
