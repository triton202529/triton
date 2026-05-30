# services/edge_ranking.py
"""
TRITON — Edge ranking and sizing helpers (allocation layer only).

This module is SHARED by:
  - services/build_trade_opportunities.py (writes data/results/trade_opportunities.csv)
  - services/regime_weighting.py          (writes data/results/target_weights.csv)

It is purely a ranking + sizing-multiplier computation. It does NOT:
  - touch broker logic
  - touch order placement
  - touch lifecycle decisions
  - touch hard portfolio safety constraints (caps, reserves, gross limits)
  - perform any I/O of its own

Stronger ideas get larger multipliers; weaker ideas get smaller multipliers or are
flagged as FILTERED_LOW_EDGE (multiplier 0). Existing weights / caps elsewhere remain
authoritative — these helpers only produce additive fields downstream code can use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Configurable constants (kept near the logic per spec)
# ─────────────────────────────────────────────────────────────

# Component weights for the composite edge score.
# Sum to 1.0 by design; reweighted proportionally when a component is missing.
EDGE_WEIGHTS: Dict[str, float] = {
    "confidence": 0.35,
    "score": 0.25,
    "momentum": 0.20,
    "trend": 0.15,
    "breakout": 0.05,
}

# Sizing bucket thresholds (descending).
# edge_score >= threshold -> bucket name.
SIZING_BUCKETS: List[Tuple[float, str]] = [
    (0.82, "HIGH_CONVICTION"),
    (0.72, "STRONG"),
    (0.62, "STANDARD"),
    (0.55, "SMALL"),
]
FILTERED_BUCKET = "FILTERED_LOW_EDGE"

# Bucket multipliers (relative sizing factor on top of the existing base weight).
BUCKET_MULTIPLIERS: Dict[str, float] = {
    "HIGH_CONVICTION": 1.50,
    "STRONG": 1.20,
    "STANDARD": 1.00,
    "SMALL": 0.60,
    FILTERED_BUCKET: 0.00,
}

# Neutral fallback for missing component values (rough mid-range).
NEUTRAL_FALLBACK = 0.5

# Action types for which positive edge sizing is meaningful (long-only).
SIZING_ELIGIBLE_OPPORTUNITIES = {"ENTRY", "ADD"}
# Action types where edge sizing must NOT override de-risking behavior.
DE_RISKING_OPPORTUNITIES = {"TRIM", "EXIT"}


# ─────────────────────────────────────────────────────────────
# Component helpers
# ─────────────────────────────────────────────────────────────


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _clamp01(v: float) -> float:
    if not math.isfinite(v):
        return NEUTRAL_FALLBACK
    return max(0.0, min(1.0, v))


def _component_value(
    row: pd.Series,
    primary_col: str,
    fallback_cols: Tuple[str, ...] = (),
) -> Optional[float]:
    """
    Pull a component from a row by trying the primary column then ordered fallbacks.
    Returns None when nothing usable is found (signals the caller to reweight).
    """
    if primary_col in row.index:
        v = _safe_float(row.get(primary_col))
        if math.isfinite(v):
            return _clamp01(v)
    for c in fallback_cols:
        if c in row.index:
            v = _safe_float(row.get(c))
            if math.isfinite(v):
                return _clamp01(v)
    return None


def compute_edge_score(row: pd.Series) -> Tuple[float, Dict[str, Optional[float]]]:
    """
    Compute the composite edge_score in [0, 1] for a single row.

    Component mapping (graceful fallback to neutral when a column is absent):
      - confidence_component = confidence
      - score_component      = score, else confidence
      - momentum_component   = momentum_score
      - trend_component      = trend_score
      - breakout_component   = breakout_score

    When some components are unavailable we proportionally reweight across
    the present components so the final score still lives in [0, 1].

    Returns (edge_score, component_values_used).
    """
    components: Dict[str, Optional[float]] = {
        "confidence": _component_value(row, "confidence"),
        "score": _component_value(row, "score", fallback_cols=("confidence",)),
        "momentum": _component_value(row, "momentum_score"),
        "trend": _component_value(row, "trend_score"),
        "breakout": _component_value(row, "breakout_score"),
    }

    # Reweight across only the components we actually have a value for.
    weighted_sum = 0.0
    weight_total = 0.0
    for key, weight in EDGE_WEIGHTS.items():
        v = components.get(key)
        if v is None:
            continue
        weighted_sum += weight * v
        weight_total += weight

    if weight_total <= 0.0:
        # Nothing at all — degrade gracefully to neutral so we never crash.
        return NEUTRAL_FALLBACK, components

    edge_score = weighted_sum / weight_total
    return _clamp01(edge_score), components


def bucket_for_edge(edge_score: float) -> str:
    """Map an edge_score into a sizing bucket name."""
    s = _safe_float(edge_score, default=0.0)
    if not math.isfinite(s):
        return FILTERED_BUCKET
    for threshold, name in SIZING_BUCKETS:
        if s >= threshold:
            return name
    return FILTERED_BUCKET


def multiplier_for_bucket(bucket: str) -> float:
    """Map a sizing bucket name to its allocation multiplier."""
    return float(BUCKET_MULTIPLIERS.get(str(bucket or "").strip().upper(), 0.0))


def allocation_reason_for(
    bucket: str,
    components: Dict[str, Optional[float]],
    edge_score: float,
) -> str:
    """
    Short, deterministic, human-readable explanation for the assigned bucket.
    Uses the strongest 1-2 contributing components when available.
    """
    b = str(bucket or "").strip().upper()
    # Pick the top contributing components (those with values and weight).
    contributors = sorted(
        [(k, v) for k, v in components.items() if v is not None and EDGE_WEIGHTS.get(k, 0.0) > 0.0],
        key=lambda kv: kv[1] * EDGE_WEIGHTS.get(kv[0], 0.0),
        reverse=True,
    )
    top_names = [k for k, _ in contributors[:2]]

    def _drivers_phrase() -> str:
        if not top_names:
            return ""
        if len(top_names) == 1:
            return f" (driver: {top_names[0]})"
        return f" (drivers: {top_names[0]} + {top_names[1]})"

    if b == "HIGH_CONVICTION":
        return f"High conviction setup with strong {' and '.join(top_names) if top_names else 'edge'} support"
    if b == "STRONG":
        return f"Strong edge ({edge_score:.2f}); above-standard size{_drivers_phrase()}"
    if b == "STANDARD":
        return f"Standard-sized position based on moderate edge ({edge_score:.2f})"
    if b == "SMALL":
        return f"Reduced size due to weaker relative edge ({edge_score:.2f})"
    if b == FILTERED_BUCKET:
        return f"Filtered out due to low edge score ({edge_score:.2f})"
    return f"Edge score {edge_score:.2f}"


# ─────────────────────────────────────────────────────────────
# DataFrame enrichment (additive, never destructive)
# ─────────────────────────────────────────────────────────────


@dataclass
class EnrichmentSpec:
    """How to interpret each row when enriching with edge fields."""

    # Column whose value (uppercased) decides whether positive sizing applies.
    # Common choices: "opportunity_type" (ENTRY/ADD/TRIM/EXIT) or "decision_action".
    opportunity_col: Optional[str] = "opportunity_type"
    # When the row is a TRIM/EXIT, sizing must not override de-risking.
    suppress_sizing_for_de_risking: bool = True
    # When the row is not eligible for sizing (e.g. HOLD/WAIT) the multiplier is forced to 0.
    force_zero_for_ineligible: bool = True


def _row_is_sizing_eligible(row: pd.Series, spec: EnrichmentSpec) -> Tuple[bool, str]:
    """Decide whether this row should receive a positive sizing multiplier."""
    if spec.opportunity_col is None or spec.opportunity_col not in row.index:
        # No opportunity classification available — assume eligible.
        return True, "no_opportunity_col"
    op = str(row.get(spec.opportunity_col) or "").strip().upper()
    if op in DE_RISKING_OPPORTUNITIES and spec.suppress_sizing_for_de_risking:
        return False, "de_risking"
    if not op:
        return True, "blank_opportunity"
    if op in SIZING_ELIGIBLE_OPPORTUNITIES:
        return True, "sizing_eligible"
    return (not spec.force_zero_for_ineligible), "non_sizing_action"


def enrich_with_edge(
    df: pd.DataFrame,
    spec: Optional[EnrichmentSpec] = None,
    *,
    rank_within_eligible_only: bool = True,
) -> pd.DataFrame:
    """
    Return a copy of `df` with new edge / ranking / sizing columns appended.

    New columns added (existing columns are NEVER renamed or removed):
      - edge_score              float in [0, 1]
      - edge_rank               int rank descending within ranking pool (1 = strongest)
      - edge_percentile         float in [0, 100]; 100 = strongest
      - sizing_bucket           HIGH_CONVICTION | STRONG | STANDARD | SMALL | FILTERED_LOW_EDGE
      - allocation_multiplier   float (0.0 to 1.5 with default buckets)
      - allocation_reason       short human-readable string
    """
    if df is None or df.empty:
        out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        for col, default in (
            ("edge_score", 0.0),
            ("edge_rank", 0),
            ("edge_percentile", 0.0),
            ("sizing_bucket", FILTERED_BUCKET),
            ("allocation_multiplier", 0.0),
            ("allocation_reason", ""),
        ):
            if col not in out.columns:
                out[col] = pd.Series(dtype=float if isinstance(default, float) else object)
        return out

    spec = spec or EnrichmentSpec()
    out = df.copy()

    edge_scores: List[float] = []
    buckets: List[str] = []
    multipliers: List[float] = []
    reasons: List[str] = []
    eligible_flags: List[bool] = []

    for _, row in out.iterrows():
        eligible, _why = _row_is_sizing_eligible(row, spec)
        edge, components = compute_edge_score(row)

        if not eligible:
            # De-risking row or non-sizing action: keep edge_score informational
            # but force allocation contribution to zero.
            bucket = FILTERED_BUCKET
            mult = 0.0
            reason = (
                "Sizing suppressed: de-risking action (lifecycle authority)"
                if str(row.get(spec.opportunity_col or "") or "").strip().upper()
                in DE_RISKING_OPPORTUNITIES
                else "Sizing suppressed: non-actionable row"
            )
        else:
            bucket = bucket_for_edge(edge)
            mult = multiplier_for_bucket(bucket)
            reason = allocation_reason_for(bucket, components, edge)

        edge_scores.append(float(edge))
        buckets.append(bucket)
        multipliers.append(float(mult))
        reasons.append(reason)
        eligible_flags.append(bool(eligible))

    out["edge_score"] = pd.Series(edge_scores, index=out.index, dtype=float).round(6)
    out["sizing_bucket"] = pd.Series(buckets, index=out.index, dtype=object)
    out["allocation_multiplier"] = pd.Series(multipliers, index=out.index, dtype=float).round(6)
    out["allocation_reason"] = pd.Series(reasons, index=out.index, dtype=object)

    # Ranking pool: by default rank only sizing-eligible rows so de-risking rows do
    # not consume rank slots ahead of strong BUY/ADD ideas.
    if rank_within_eligible_only:
        pool_mask = pd.Series(eligible_flags, index=out.index)
    else:
        pool_mask = pd.Series([True] * len(out), index=out.index)

    pool_scores = out["edge_score"].where(pool_mask)
    # Higher score => smaller rank number (1 = best).
    # Suppressed/ineligible rows (NaN here) get rank 0 (i.e. "no rank").
    rank = pool_scores.rank(method="min", ascending=False, na_option="keep")
    rank = rank.fillna(0).astype(int)
    out["edge_rank"] = rank

    # Percentile within the pool (100 = best). Suppressed rows are 0.
    pool_only = pool_scores.dropna()
    if len(pool_only) >= 2:
        pct = pool_scores.rank(pct=True, ascending=True, na_option="keep") * 100.0
        pct = pct.fillna(0.0)
    elif len(pool_only) == 1:
        pct = pd.Series(np.where(pool_mask, 100.0, 0.0), index=out.index, dtype=float)
    else:
        pct = pd.Series([0.0] * len(out), index=out.index, dtype=float)
    out["edge_percentile"] = pct.round(2)

    return out
