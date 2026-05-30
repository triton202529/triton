# services/portfolio_intelligence.py
"""
TRITON — Portfolio intelligence (allocation tilt layer).

Sits AFTER edge-based ranking/sizing and BEFORE the final allocation is written.
Adjusts proposed weights so the resulting book is a sensible PORTFOLIO and not
just a list of strong individual ideas.

This module:
  - applies a SINGLE-NAME soft concentration cap (gentle decay, never zeroing)
  - applies a SECTOR soft cap (proportional reduction within crowded sectors)
  - dampens ADDs into already-large existing positions
  - mildly penalizes lower-ranked names within a crowded sector

It DOES NOT:
  - touch broker / order placement code
  - touch lifecycle decisions (TRIM/EXIT remain authoritative)
  - touch hard portfolio safety constraints (caps / reserves / risk gates)
  - perform any execution I/O

It is purely additive: every column it produces is appended to the input
DataFrame; nothing is renamed or removed. Optional metadata sources
(positions_snapshot.csv, sector mapping) are loaded best-effort and
gracefully degrade when missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Internal sector resolver — read-only; tolerated if import fails.
# We canonicalize the missing-sector sentinel to "UNKNOWN" (uppercase) for the
# portfolio intelligence column so it is unambiguous, regardless of what the
# upstream resolver returns ("Unknown", "", None, etc.).
try:
    from services.sector_exposure import get_sector as _get_sector
except Exception:  # pragma: no cover — defensive degrade
    _get_sector = None  # type: ignore[assignment]

_UNKNOWN_SECTOR = "UNKNOWN"
# Values returned by upstream resolvers that should be normalized to _UNKNOWN_SECTOR.
_UNKNOWN_SECTOR_ALIASES = {"", "UNKNOWN", "UNK", "NONE", "NAN", "N/A"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_POSITIONS_SNAPSHOT = RESULTS_DIR / "positions_snapshot.csv"


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────


@dataclass
class PortfolioIntelligenceConfig:
    """All knobs for the v1 portfolio-aware tilt layer."""

    # Single-name soft cap on portfolio weight (e.g. 12%). Above this, a smooth
    # decay shrinks the proposed weight rather than hard-clipping it.
    soft_cap_single_name: float = 0.12
    # Floor on the concentration penalty so a strong single name is reduced,
    # not extinguished. (0.40 = at most -60% size from concentration alone.)
    concentration_penalty_floor: float = 0.40

    # Sector soft cap on aggregate exposure (e.g. 30% of book in one sector).
    soft_cap_sector: float = 0.30
    # Floor on per-row sector penalty.
    sector_penalty_floor: float = 0.50

    # ADD dampener thresholds on existing position weight (as fraction of book).
    add_small_threshold: float = 0.02  # below this -> no dampening
    add_medium_threshold: float = 0.05  # below this -> mild dampening
    add_large_threshold: float = 0.08  # below this -> stronger dampening
    add_dampener_mild: float = 0.85
    add_dampener_strong: float = 0.65
    add_dampener_extreme: float = 0.45

    # Crowding penalty applied to lower-ranked names INSIDE a sector that has
    # multiple ideas competing. Top-ranked name keeps full size; each
    # subsequent rank loses `crowding_step` (clamped at `crowding_floor`).
    crowding_step: float = 0.07
    crowding_floor: float = 0.70

    # The opportunity-action column to inspect (for ADD/TRIM/EXIT semantics).
    # If absent, every row is treated as a generic BUY-equivalent (long entry).
    opportunity_col: str = "opportunity_type"

    # Source of pre-adjustment weights — column the layer starts from.
    pre_adjustment_col: str = "allocation_weight_final"
    # Column carrying the strength signal used for crowding rank/order.
    edge_score_col: str = "edge_score"
    edge_rank_col: str = "edge_rank"

    # Action types whose sizing must NOT be lifted by portfolio re-tilt.
    de_risking_actions: Tuple[str, ...] = ("TRIM", "EXIT")
    # Action types where the ADD dampener applies.
    add_actions: Tuple[str, ...] = ("ADD",)

    # Estimated portfolio equity used to translate raw market values from
    # positions_snapshot.csv into existing weights when no explicit total is
    # provided (best-effort; we fall back to the snapshot's own total).
    fallback_equity: Optional[float] = None


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _normalize_weights(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    total = float(s.sum())
    if total <= 0:
        return s * 0.0
    return s / total


def load_existing_positions_optional(
    *,
    snapshot_path: Path = DEFAULT_POSITIONS_SNAPSHOT,
    fallback_equity: Optional[float] = None,
) -> Dict[str, float]:
    """
    Best-effort map of {ticker -> existing portfolio weight in [0, 1]}.

    Reads `positions_snapshot.csv` (read-only) and converts market_value into
    weights using either `fallback_equity` (if provided) or the sum of all
    positions' market_value as the denominator. Returns {} on any failure.
    """
    if not snapshot_path.exists() or snapshot_path.stat().st_size == 0:
        return {}
    try:
        df = pd.read_csv(snapshot_path)
    except Exception:
        return {}
    if df is None or df.empty:
        return {}

    sym_col = None
    for c in ("ticker", "symbol"):
        if c in df.columns:
            sym_col = c
            break
    if sym_col is None:
        return {}

    val_col = None
    for c in ("market_value", "value", "marketValue"):
        if c in df.columns:
            val_col = c
            break
    if val_col is None:
        return {}

    df = df.copy()
    df[sym_col] = df[sym_col].astype(str).str.strip().str.upper()
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)

    # Prefer the freshest snapshot rows when duplicates exist.
    if "snapshot_ts" in df.columns:
        try:
            df["_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)
            df = df.sort_values(["_ts"], kind="mergesort").drop(columns=["_ts"])
        except Exception:
            pass
    df = df.drop_duplicates(subset=[sym_col], keep="last")

    total = _safe_float(fallback_equity, 0.0)
    if total <= 0:
        total = float(df[val_col].sum())
    if total <= 0:
        return {}

    out: Dict[str, float] = {}
    for sym, val in zip(df[sym_col].tolist(), df[val_col].tolist()):
        v = _safe_float(val, 0.0)
        if v <= 0 or not sym:
            continue
        w = max(0.0, min(1.0, v / total))
        out[sym] = w
    return out


def _resolve_sector(ticker: str) -> str:
    if not ticker:
        return _UNKNOWN_SECTOR
    if _get_sector is None:
        return _UNKNOWN_SECTOR
    try:
        raw = _get_sector(ticker)
    except Exception:
        return _UNKNOWN_SECTOR
    if raw is None:
        return _UNKNOWN_SECTOR
    s = str(raw).strip()
    if not s or s.upper() in _UNKNOWN_SECTOR_ALIASES:
        return _UNKNOWN_SECTOR
    return s


# ─────────────────────────────────────────────────────────────
# Penalty primitives (deterministic, monotonic, NaN-safe)
# ─────────────────────────────────────────────────────────────


def concentration_penalty(weight: float, soft_cap: float, floor: float = 0.40) -> float:
    """
    Smooth single-name concentration penalty.

    weight <= soft_cap            -> 1.0  (no penalty)
    weight  > soft_cap            -> factor = max(floor, soft_cap / weight)

    The geometric (cap / w) form is gentle: at w = 1.5 * cap the factor is
    ~0.67; at w = 2 * cap it's 0.50. The floor prevents extinction.
    """
    w = _safe_float(weight, 0.0)
    cap = _safe_float(soft_cap, 0.12)
    flr = _safe_float(floor, 0.40)
    if w <= cap or cap <= 0:
        return 1.0
    return max(flr, cap / w)


def add_dampener(
    existing_weight: float,
    cfg: PortfolioIntelligenceConfig,
) -> float:
    """Step-shaped ADD dampener based on the existing position weight."""
    w = _safe_float(existing_weight, 0.0)
    if w <= cfg.add_small_threshold:
        return 1.0
    if w <= cfg.add_medium_threshold:
        return cfg.add_dampener_mild
    if w <= cfg.add_large_threshold:
        return cfg.add_dampener_strong
    return cfg.add_dampener_extreme


def crowding_factor(
    rank_within_sector: int,
    cfg: PortfolioIntelligenceConfig,
) -> float:
    """
    Mild penalty for lower-ranked names competing inside the same sector.
    Rank 1 (top) keeps full size; each subsequent rank loses `crowding_step`,
    clamped at `crowding_floor`. Rank <= 0 (no rank) returns 1.0.
    """
    r = int(rank_within_sector or 0)
    if r <= 1:
        return 1.0
    factor = 1.0 - cfg.crowding_step * (r - 1)
    return max(cfg.crowding_floor, factor)


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────


def apply_portfolio_intelligence(
    df: pd.DataFrame,
    cfg: Optional[PortfolioIntelligenceConfig] = None,
    *,
    existing_positions: Optional[Dict[str, float]] = None,
    sector_resolver=None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply the v1 portfolio-aware adjustment layer.

    Inputs (read from `df` when present; safely defaulted otherwise):
      - ticker
      - allocation_weight_final  (pre-adjustment proposal; cfg.pre_adjustment_col)
      - edge_score / edge_rank   (used for crowding ordering)
      - opportunity_type         (drives ADD dampener / TRIM-EXIT suppression)

    Outputs (appended; nothing removed or renamed):
      - sector_name
      - existing_position_weight
      - sector_weight_current
      - sector_weight_proposed
      - single_name_over_cap_flag
      - sector_over_cap_flag
      - add_overcrowded_flag
      - concentration_penalty
      - sector_penalty
      - add_dampener
      - crowding_penalty
      - crowded_group_rank
      - portfolio_adjustment_factor
      - portfolio_weight_pre_adjustment
      - portfolio_weight_post_adjustment
      - portfolio_adjustment_reason
      - portfolio_fallback_used
    """
    cfg = cfg or PortfolioIntelligenceConfig()
    diagnostics: Dict[str, Any] = {
        "rows_in": 0 if df is None else len(df),
        "soft_cap_single_name": cfg.soft_cap_single_name,
        "soft_cap_sector": cfg.soft_cap_sector,
        "fallback_used": False,
        "sectors_over_cap": [],
        "single_names_over_cap": 0,
        "add_dampened_rows": 0,
        "metadata": {
            "sector_resolver_available": _get_sector is not None or sector_resolver is not None,
            "existing_positions_count": 0,
        },
    }

    if df is None or df.empty:
        out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        for col, default in _empty_output_defaults().items():
            if col not in out.columns:
                out[col] = pd.Series(dtype=type(default))
        return out, diagnostics

    out = df.copy()

    # Lazy-load existing positions if caller didn't pass them.
    if existing_positions is None:
        existing_positions = load_existing_positions_optional(
            fallback_equity=cfg.fallback_equity,
        )
    diagnostics["metadata"]["existing_positions_count"] = len(existing_positions)

    resolver = sector_resolver or _resolve_sector

    # ── Step 0: collect derived per-row context ────────────────────────────
    tickers = (
        out["ticker"].astype(str).str.strip().str.upper()
        if "ticker" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )

    sector_names = [resolver(t) for t in tickers.tolist()]
    out["sector_name"] = sector_names

    out["existing_position_weight"] = [
        float(existing_positions.get(t, 0.0)) for t in tickers.tolist()
    ]

    pre_col = cfg.pre_adjustment_col
    if pre_col not in out.columns:
        # Nothing to tilt — emit pass-through outputs and bail safely.
        out["portfolio_weight_pre_adjustment"] = 0.0
        for col, default in _empty_output_defaults().items():
            if col not in out.columns:
                out[col] = default
        diagnostics["fallback_used"] = True
        diagnostics["fallback_reason"] = f"missing_pre_adjustment_column:{pre_col}"
        return out, diagnostics

    pre_w = pd.to_numeric(out[pre_col], errors="coerce").fillna(0.0).astype(float)
    out["portfolio_weight_pre_adjustment"] = pre_w

    op_col = cfg.opportunity_col
    if op_col in out.columns:
        op_series = out[op_col].fillna("").astype(str).str.strip().str.upper()
    else:
        op_series = pd.Series([""] * len(out), index=out.index, dtype=object)

    is_de_risking = op_series.isin([a.upper() for a in cfg.de_risking_actions])
    is_add = op_series.isin([a.upper() for a in cfg.add_actions])

    edge_score = (
        pd.to_numeric(out[cfg.edge_score_col], errors="coerce")
        if cfg.edge_score_col in out.columns
        else pd.Series([np.nan] * len(out), index=out.index, dtype=float)
    )

    # ── Step 1: single-name concentration penalty ──────────────────────────
    conc_pen: List[float] = []
    over_single: List[bool] = []
    for w in pre_w.tolist():
        if w <= 0:
            conc_pen.append(1.0)
            over_single.append(False)
            continue
        f = concentration_penalty(w, cfg.soft_cap_single_name, cfg.concentration_penalty_floor)
        conc_pen.append(f)
        over_single.append(w > cfg.soft_cap_single_name)
    out["concentration_penalty"] = conc_pen
    out["single_name_over_cap_flag"] = over_single
    diagnostics["single_names_over_cap"] = int(sum(over_single))

    # ── Step 2: sector exposure + sector penalty + crowding rank ───────────
    # Compute proposed sector exposure on PRE-adjustment, eligible (non-de-risking) rows.
    eligible_mask = ~is_de_risking
    sector_proposed: Dict[str, float] = {}
    for sec, w in zip(sector_names, pre_w.where(eligible_mask, 0.0).tolist()):
        sector_proposed[sec] = sector_proposed.get(sec, 0.0) + float(w)

    # Existing exposure per sector from current positions.
    sector_current: Dict[str, float] = {}
    for t, w in existing_positions.items():
        sec = resolver(t)
        sector_current[sec] = sector_current.get(sec, 0.0) + float(w)

    out["sector_weight_current"] = [float(sector_current.get(sec, 0.0)) for sec in sector_names]
    out["sector_weight_proposed"] = [float(sector_proposed.get(sec, 0.0)) for sec in sector_names]

    # Per-sector crowding rank (1 = strongest by edge_score within sector).
    crowding_rank_by_idx: List[int] = [0] * len(out)
    sector_to_idxs: Dict[str, List[int]] = {}
    for i, sec in enumerate(sector_names):
        sector_to_idxs.setdefault(sec, []).append(i)
    for sec, idxs in sector_to_idxs.items():
        # Only rank eligible (non-de-risking) rows so TRIM/EXIT don't take rank slots.
        eligible_idxs = [i for i in idxs if not bool(is_de_risking.iloc[i])]

        # Sort by edge_score desc; missing scores go last.
        def _key(i: int) -> float:
            v = edge_score.iloc[i]
            return -float(v) if pd.notna(v) else float("inf")

        eligible_idxs.sort(key=_key)
        for r, i in enumerate(eligible_idxs, start=1):
            crowding_rank_by_idx[i] = r
    out["crowded_group_rank"] = crowding_rank_by_idx

    # Sector penalty: if proposed sector exposure > soft_cap_sector, scale rows
    # in that sector down proportionally. Lower-ranked rows get a slightly
    # larger reduction (handled separately by crowding_penalty below).
    sector_pen: List[float] = []
    sector_over_flag: List[bool] = []
    sectors_over_cap: List[str] = []
    for sec, w_proposed in sector_proposed.items():
        if w_proposed > cfg.soft_cap_sector and sec != _UNKNOWN_SECTOR:
            sectors_over_cap.append(sec)
    diagnostics["sectors_over_cap"] = sorted(set(sectors_over_cap))

    for i, sec in enumerate(sector_names):
        prop = sector_proposed.get(sec, 0.0)
        if prop <= cfg.soft_cap_sector or sec == _UNKNOWN_SECTOR or bool(is_de_risking.iloc[i]):
            sector_pen.append(1.0)
            sector_over_flag.append(False)
            continue
        # Proportional shrink toward cap, with floor.
        f = max(cfg.sector_penalty_floor, cfg.soft_cap_sector / prop)
        sector_pen.append(f)
        sector_over_flag.append(True)
    out["sector_penalty"] = sector_pen
    out["sector_over_cap_flag"] = sector_over_flag

    # ── Step 3: ADD dampener ───────────────────────────────────────────────
    add_pen: List[float] = []
    add_over_flag: List[bool] = []
    add_dampened = 0
    for i, t in enumerate(tickers.tolist()):
        existing = float(existing_positions.get(t, 0.0))
        if not bool(is_add.iloc[i]):
            # Even non-ADD rows can be flagged "overcrowded" if they have a
            # large existing position (informational), but we don't dampen them.
            add_pen.append(1.0)
            add_over_flag.append(existing > cfg.add_large_threshold)
            continue
        f = add_dampener(existing, cfg)
        add_pen.append(f)
        add_over_flag.append(existing > cfg.add_medium_threshold)
        if f < 1.0:
            add_dampened += 1
    out["add_dampener"] = add_pen
    out["add_overcrowded_flag"] = add_over_flag
    diagnostics["add_dampened_rows"] = add_dampened

    # ── Step 4: crowding penalty (intra-sector rank decay) ─────────────────
    crowding_pen: List[float] = []
    for i, r in enumerate(crowding_rank_by_idx):
        # Only meaningful when sector has >1 eligible name.
        sec = sector_names[i]
        n_in_sec = sum(1 for j in sector_to_idxs.get(sec, []) if not bool(is_de_risking.iloc[j]))
        if r <= 0 or n_in_sec <= 1 or sec == _UNKNOWN_SECTOR:
            crowding_pen.append(1.0)
            continue
        crowding_pen.append(crowding_factor(r, cfg))
    out["crowding_penalty"] = crowding_pen

    # ── Step 5: combined factor + post-adjustment weight + reason ──────────
    factors = (
        np.array(conc_pen, dtype=float)
        * np.array(sector_pen, dtype=float)
        * np.array(add_pen, dtype=float)
        * np.array(crowding_pen, dtype=float)
    )

    # Suppress sizing for de-risking rows entirely (lifecycle authority).
    factors_arr = np.where(is_de_risking.values, 0.0, factors)
    out["portfolio_adjustment_factor"] = factors_arr.round(6)

    raw_post = pre_w.values * factors_arr
    raw_post_series = pd.Series(raw_post, index=out.index, dtype=float)

    # Renormalize across eligible names so book sums to 1.0 of the eligible mass.
    eligible = (~is_de_risking) & (raw_post_series > 0)
    eligible_sum = float(raw_post_series.where(eligible, 0.0).sum())

    if eligible_sum > 0:
        # Preserve the original eligible mass (i.e. how much of the book the
        # pre-adjustment proposal allocated to non-de-risking names) so we
        # don't accidentally inflate exposure when penalties shrink things.
        original_eligible_mass = float(pre_w.where(eligible, 0.0).sum())
        scale = original_eligible_mass / eligible_sum if eligible_sum > 0 else 0.0
        post = raw_post_series * scale
        # De-risking rows stay at zero allocation in the post-adjustment view.
        post = post.where(~is_de_risking, 0.0)
        out["portfolio_weight_post_adjustment"] = post.round(6).astype(float)
        out["portfolio_fallback_used"] = False
    else:
        # Fallback: portfolio adjustment collapsed the book (shouldn't happen
        # with floors in place, but be safe). Revert to pre-adjustment.
        out["portfolio_weight_post_adjustment"] = pre_w.where(~is_de_risking, 0.0).astype(float)
        out["portfolio_fallback_used"] = True
        diagnostics["fallback_used"] = True
        diagnostics["fallback_reason"] = "post_adjustment_eligible_mass_zero"

    # Reason text — short, deterministic.
    reasons: List[str] = []
    for i in range(len(out)):
        if bool(is_de_risking.iloc[i]):
            reasons.append("Sizing suppressed: de-risking action (lifecycle authority)")
            continue
        if bool(out["portfolio_fallback_used"].iloc[i]):
            reasons.append("Fallback to pre-adjustment weights")
            continue
        parts: List[str] = []
        if bool(over_single[i]):
            parts.append("Reduced due to single-name concentration")
        if bool(sector_over_flag[i]):
            parts.append("Reduced due to sector crowding")
        if bool(is_add.iloc[i]) and add_pen[i] < 1.0:
            parts.append("Reduced add size because existing position is already large")
        if crowding_pen[i] < 1.0 and not bool(sector_over_flag[i]):
            parts.append("Mild penalty: lower-ranked name in same sector group")
        if not parts:
            parts.append("No portfolio adjustment applied")
        reasons.append("; ".join(parts))
    out["portfolio_adjustment_reason"] = reasons

    diagnostics["rows_out"] = int(len(out))
    diagnostics["eligible_rows"] = int(eligible.sum())
    diagnostics["pre_adjustment_eligible_mass"] = float(
        pre_w.where(eligible | (~is_de_risking), 0.0).sum()
    )
    diagnostics["post_adjustment_eligible_mass"] = float(
        out["portfolio_weight_post_adjustment"].sum()
    )
    return out, diagnostics


def _empty_output_defaults() -> Dict[str, Any]:
    """Default values for output columns (used when input is empty)."""
    return {
        "sector_name": "",
        "existing_position_weight": 0.0,
        "sector_weight_current": 0.0,
        "sector_weight_proposed": 0.0,
        "single_name_over_cap_flag": False,
        "sector_over_cap_flag": False,
        "add_overcrowded_flag": False,
        "concentration_penalty": 1.0,
        "sector_penalty": 1.0,
        "add_dampener": 1.0,
        "crowding_penalty": 1.0,
        "crowded_group_rank": 0,
        "portfolio_adjustment_factor": 1.0,
        "portfolio_weight_pre_adjustment": 0.0,
        "portfolio_weight_post_adjustment": 0.0,
        "portfolio_adjustment_reason": "",
        "portfolio_fallback_used": False,
    }
