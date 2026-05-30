"""
services/adaptation_simulation.py — Triton Phase-2 Adaptive Behavior Simulation.

Answers: "If these applied adjustments were active, what would change?"

Read-only by construction. This module:

    * reads data/results/applied_adjustments.csv (the approved/applied registry)
    * reads existing baseline artifacts (trade_opportunities.csv,
      execution_intelligence.csv, execution_plan.csv) for context
    * applies each ACTIVE adjustment IN MEMORY against the opportunity rows
    * classifies each row as
          UNCHANGED_ACCEPT | UNCHANGED_REJECT | NEWLY_REJECTED | NEWLY_ACCEPTED
    * writes
          data/results/adaptation_simulation.csv           (per-row)
          data/results/adaptation_simulation_summary.json  (aggregate)

Hard non-goals
--------------
* NO changes to trade_opportunities.csv, applied_adjustments.csv, or any
  production config / CSV.
* NO broker interaction.
* NO execution, no lifecycle, no risk-sizing decisions.
* NO import of services.execute_trades, services.run_execution_loop,
  services.place_live_orders, services.apply_signal_lifecycle, etc.

Run
---
    python -m services.adaptation_simulation
optional flags:
    --score-floor 0.30     override baseline score floor for rejection
    --quiet                suppress stdout print output

Outputs are always written (even when zero rows or zero active adjustments),
so dashboard consumers see a stable schema.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# ──────────────────────────────────────────────────────────────
# Paths / constants
# ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

APPLIED_ADJUSTMENTS_CSV = RESULTS / "applied_adjustments.csv"
TRADE_OPPORTUNITIES_CSV = RESULTS / "trade_opportunities.csv"
EXECUTION_INTELLIGENCE_CSV = RESULTS / "execution_intelligence.csv"
EXECUTION_PLAN_CSV = RESULTS / "execution_plan.csv"

SIM_CSV = RESULTS / "adaptation_simulation.csv"
SIM_SUMMARY_JSON = RESULTS / "adaptation_simulation_summary.json"

SCHEMA_VERSION = 1
PHASE = "2-simulation"

# Decision-change categories (deterministic, CSV-safe)
DC_UNCHANGED_ACCEPT = "UNCHANGED_ACCEPT"
DC_UNCHANGED_REJECT = "UNCHANGED_REJECT"
DC_NEWLY_REJECTED = "NEWLY_REJECTED"
DC_NEWLY_ACCEPTED = "NEWLY_ACCEPTED"

DECISION_CHANGE_ORDER = [
    DC_UNCHANGED_ACCEPT,
    DC_NEWLY_REJECTED,
    DC_NEWLY_ACCEPTED,
    DC_UNCHANGED_REJECT,
]

# Effect types used by the rule engine
EFFECT_PENALTY = "penalty"  # subtract delta from simulated_score
EFFECT_BOOST = "boost"  # add delta to simulated_score
EFFECT_FLOOR_RAISE = "floor_raise"  # raise required score for matching rows
EFFECT_TRIM_THRESH = "trim_threshold"  # annotates TRIM rows only
EFFECT_ANNOTATE = "annotate_only"  # record match, no accept/reject impact

# Output schema — stable even on empty runs
SIM_COLUMNS: List[str] = [
    "row_id",
    "symbol",
    "opportunity_type",
    "effective_stance",
    "sizing_bucket",
    "confidence",
    "edge_score",
    "baseline_score",
    "baseline_threshold",
    "baseline_accepted",
    "baseline_reject_reason",
    "simulated_score",
    "simulated_threshold",
    "simulated_penalty_total",
    "simulated_boost_total",
    "simulated_accepted",
    "simulated_reject_reason",
    "decision_change",
    "score_delta",
    "adjustments_applied",
    "adjustment_details",
    "applied_delta_total",
    "match_reason",
    "trim_threshold_tightened",
    "cooldown_bias_applied",
    "spread_bucket",
    "spread_bps",
    "spread_pct",
    "liquidity_pressure_bucket",
    "quote_is_stale",
    "quote_reason",
    "quote_staleness_flag",
    "execution_risk_flag",
    "execution_style",
    "execution_quality_score",
    "execution_context_source",
    "execution_blocked",
    "execution_block_reason",
]


# ──────────────────────────────────────────────────────────────
# Safe loaders
# ──────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_csv_safe(path: Path) -> Tuple[pd.DataFrame, str]:
    """Return (df, status). Status ∈ {'ok','missing','empty','malformed'}."""
    if not path.exists():
        return pd.DataFrame(), "missing"
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "empty"
    except Exception:
        return pd.DataFrame(), "malformed"
    if df is None or df.empty:
        return (df if isinstance(df, pd.DataFrame) else pd.DataFrame()), "empty"
    return df, "ok"


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def _safe_bool(x: Any, default: Optional[bool] = None) -> Optional[bool]:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    try:
        if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
            return bool(int(x))
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    if s in ("", "nan", "none"):
        return default
    return default


def _safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    try:
        if isinstance(x, float) and math.isnan(x):
            return default
    except Exception:
        pass
    s = str(x).strip()
    return s if s.lower() not in ("nan", "none") else default


# ──────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────


@dataclass
class SimInputs:
    applied: pd.DataFrame = field(default_factory=pd.DataFrame)
    opps: pd.DataFrame = field(default_factory=pd.DataFrame)
    exec_intel: pd.DataFrame = field(default_factory=pd.DataFrame)
    exec_plan: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: Dict[str, str] = field(default_factory=dict)

    def missing(self) -> List[str]:
        return [k for k, v in self.status.items() if v != "ok"]


def load_inputs() -> SimInputs:
    applied, s_app = load_csv_safe(APPLIED_ADJUSTMENTS_CSV)
    opps, s_opps = load_csv_safe(TRADE_OPPORTUNITIES_CSV)
    ei, s_ei = load_csv_safe(EXECUTION_INTELLIGENCE_CSV)
    ep, s_ep = load_csv_safe(EXECUTION_PLAN_CSV)
    return SimInputs(
        applied=applied,
        opps=opps,
        exec_intel=ei,
        exec_plan=ep,
        status={
            "applied_adjustments_csv": s_app,
            "trade_opportunities_csv": s_opps,
            "execution_intelligence_csv": s_ei,
            "execution_plan_csv": s_ep,
        },
    )


# ──────────────────────────────────────────────────────────────
# Adjustment filter — only ACTIVE rows count
# ──────────────────────────────────────────────────────────────


# Statuses that count an applied-adjustment row as live. APPLIED is the
# canonical active label emitted by apply_layer; ACTIVE is accepted as an
# equivalent synonym for forwards compatibility.
_ACTIVE_STATUSES = {"APPLIED", "ACTIVE"}


def _normalize_applied_row(row: pd.Series) -> Dict[str, Any]:
    """
    Normalize one applied_adjustments.csv row before simulation use.

    Strips whitespace on strings, collapses blank / "nan" / "none" to None,
    uppercases categorical fields, and canonicalizes active_flag across
    True/False/1/0/"True"/"true"/"yes"/"y". Numeric deltas are parsed
    defensively so downstream code never sees a stray NaN.
    """

    def _str_or_none(key: str, upper: bool = False) -> Optional[str]:
        v = row.get(key) if isinstance(row, (pd.Series, dict)) else None
        s = _safe_str(v)
        if not s:
            return None
        return s.upper() if upper else s

    status = _str_or_none("status", upper=True)
    active = _safe_bool(
        row.get("active_flag") if isinstance(row, (pd.Series, dict)) else None, default=None
    )

    effective_delta = _safe_float(
        row.get("effective_delta") if isinstance(row, (pd.Series, dict)) else None
    )
    proposed_delta = _safe_float(
        row.get("proposed_delta") if isinstance(row, (pd.Series, dict)) else None
    )
    delta = effective_delta if effective_delta is not None else proposed_delta
    if delta is None:
        delta = 0.0

    return {
        "application_id": _str_or_none("application_id") or "",
        "proposal_id": _str_or_none("proposal_id") or "",
        "adaptation_target": (_str_or_none("adaptation_target") or "").lower(),
        "effective_delta": float(delta),
        "related_bucket": _str_or_none("related_bucket", upper=True) or "",
        "related_flag": _str_or_none("related_flag", upper=True) or "",
        "related_style": _str_or_none("related_style", upper=True) or "",
        "status": status or "",
        "active_flag": bool(active) if active is not None else None,
    }


def _is_active_by_flag(active_flag: Optional[bool]) -> bool:
    # Permissive default: if active_flag is absent/unparseable, we only accept
    # the row when its status indicates liveness (handled by caller).
    return bool(active_flag) if active_flag is True else False


def active_adjustments(applied: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows that are *genuinely* active.

    A row counts as active when EITHER:
      - active_flag resolves to True (True/1/"true"/"yes"/"y"), OR
      - status ∈ {APPLIED, ACTIVE} (case-insensitive), if active_flag
        is unavailable / unparseable.

    When both columns are missing entirely the function returns every row
    (permissive — same behavior as before this fix, but now honest about it).
    """
    if applied is None or applied.empty:
        return pd.DataFrame()

    df = applied.copy()
    df.columns = [str(c).strip() for c in df.columns]

    has_active = "active_flag" in df.columns
    has_status = "status" in df.columns
    if not has_active and not has_status:
        return df.reset_index(drop=True)

    masks: List[pd.Series] = []
    if has_active:
        active_parsed = df["active_flag"].apply(lambda v: _safe_bool(v, default=None))
        masks.append(active_parsed == True)  # noqa: E712 — want elementwise compare
    if has_status:
        status_up = df["status"].apply(lambda v: _safe_str(v).upper())
        masks.append(status_up.isin(_ACTIVE_STATUSES))

    # A row is active if ANY of the available liveness signals confirms it.
    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m
    out = df[mask.fillna(False).astype(bool)].copy()
    return out.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Rule engine — allow-listed targets only
# ──────────────────────────────────────────────────────────────


@dataclass
class AdjustmentRule:
    """One bounded simulation rule for a single adaptation_target."""

    target: str
    effect: str
    matches: Callable[[Dict[str, Any]], bool]
    applies_to_actions: Optional[List[str]] = None  # e.g. ["ADD", "ENTRY"]; None=all
    description: str = ""


def _lower(x: Any) -> str:
    return _safe_str(x).lower()


# Canonical action groups. Each rule reads these so the "side" of a row is
# detected via whichever of (opportunity_type, action, effective_stance,
# side, stance) is first populated.
_ACTION_KEYS: Tuple[str, ...] = (
    "opportunity_type",
    "action",
    "effective_stance",
    "side",
    "stance",
)

_WIDE_SPREAD_BUCKETS = {"WIDE", "TOO_WIDE", "VERY_WIDE", "EXTREMELY_WIDE"}
_STRESSED_LIQUIDITY_BUCKETS = {"HIGH", "ELEVATED", "STRESSED", "WIDE", "TOO_WIDE"}
_CONFIRM_FLAG_TOKENS = {"TRUE", "1", "YES", "Y", "T"}

# Sentinel values that mean "no information / fill placeholder". Writers
# upstream (build_trade_opportunities) now always populate spread_bucket /
# execution_risk_flag with UNKNOWN when no exec data exists, so the simulator
# must treat these as absent — otherwise an UNKNOWN row would spuriously
# win over a richer exec_intelligence join.
_EMPTY_CATEGORICAL_SENTINELS = {"", "UNKNOWN", "NONE", "NULL", "NAN", "NA", "N/A"}


def _is_empty_categorical(s: Any) -> bool:
    """True if the value is effectively empty (blank / NaN / UNKNOWN/NA placeholder)."""
    return _safe_str(s).upper() in _EMPTY_CATEGORICAL_SENTINELS


def _first_meaningful_str(*values: Any) -> str:
    """
    Return the first value that is both non-empty AND not an UNKNOWN-style
    sentinel. This is stricter than `_first_nonempty_str`: it falls through
    placeholder 'UNKNOWN' values so downstream (exec_ctx) gets a chance.
    """
    for v in values:
        s = _safe_str(v)
        if s and s.upper() not in _EMPTY_CATEGORICAL_SENTINELS:
            return s
    return ""


def _action_of(row: Dict[str, Any]) -> str:
    for k in _ACTION_KEYS:
        v = _safe_str(row.get(k))
        if v:
            return v.upper()
    return ""


def _is_entry_like(row: Dict[str, Any]) -> bool:
    # Prefer explicit opportunity_type (canonical) before other action keys.
    ot = _safe_str(row.get("opportunity_type")).upper()
    if ot in ("TRIM", "EXIT", "SELL", "HOLD", "WAIT"):
        return False
    if ot in ("ENTRY", "ADD", "BUY"):
        return True
    a = _action_of(row)
    if a in ("ENTRY", "ADD", "BUY"):
        return True
    # Defensive: rows without an action field are treated as entry-like
    # candidates so wide-spread / stale / low-confidence rules can still
    # inspect their execution context. Explicit TRIM/EXIT/SELL/HOLD rows
    # are excluded below.
    if a in ("TRIM", "EXIT", "SELL", "HOLD", "WAIT"):
        return False
    return a == ""  # unknown → permissive for entry-penalty checks


def _is_trim_like(row: Dict[str, Any]) -> bool:
    a = _action_of(row)
    return a in ("TRIM", "EXIT", "SELL")


def _is_add_like(row: Dict[str, Any]) -> bool:
    a = _action_of(row)
    return a == "ADD"


def _confidence_band_label(c: Optional[float]) -> str:
    """Triton's planner bands: LOW < 0.55, MEDIUM 0.55–0.70, HIGH > 0.70."""
    if c is None:
        return ""
    if c < 0.55:
        return "LOW"
    if c <= 0.70:
        return "MEDIUM"
    return "HIGH"


def _match_low_confidence(row: Dict[str, Any]) -> bool:
    """
    True when the row is an entry / buy candidate and confidence is in the
    low band, using numeric confidence, confidence_bucket, or derived band.
    """
    if not _is_entry_like(row):
        return False
    c = _safe_float(row.get("confidence"))
    if c is not None and c < 0.55:
        return True
    cb = _safe_str(row.get("confidence_bucket")).upper()
    if cb in (
        "LOW",
        "SIGNAL_CAUTION",
        "WEAK",
        "WEAK_SIGNAL",
        "CAUTION",
        "LOW_CONF",
        "LOW_CONFIDENCE",
        "FILTERED_LOW_EDGE",
    ):
        return True
    b = _safe_str(row.get("confidence_band")).upper()
    if b == "LOW":
        return True
    return False


def _match_trim_only(row: Dict[str, Any]) -> bool:
    """Original strict: TRIM/EXIT/SELL in action only."""
    a = _action_of(row)
    return a in ("TRIM", "EXIT", "SELL")


def _match_trim_scenario(row: Dict[str, Any]) -> bool:
    """
    TRIM / EXIT style opportunities: explicit types, stance + position,
    lifecycle text on longs, and a conservative 'profit-take' hint when
    a held winner is indicated by score percentiles.
    """
    if _match_trim_only(row):
        return True
    ot = _safe_str(row.get("opportunity_type")).upper()
    if ot in ("TRIM", "EXIT"):
        return True
    es = _safe_str(row.get("effective_stance")).upper()
    eps = _safe_str(row.get("effective_position_state")).upper()
    if es in ("TRIM", "REDUCE", "SELL", "EXIT") and eps == "LONG":
        return True
    if eps == "LONG":
        blob = " ".join(
            [
                _safe_str(row.get("lifecycle_decision_reason")),
                _safe_str(row.get("reason_code")),
                _safe_str(row.get("stance_adjustment")),
            ]
        ).upper()
        for kw in (
            "TRIM",
            "REDUCE",
            "DE-RISK",
            "DERISK",
            "TAKE PROFIT",
            "TAKE_PROFIT",
            "PARE",
            "PROFIT TAKE",
            "PARTIAL",
            "EXIT",
            "CLOSE",
            "SELL",
            "PAYOFF",
            "SCALING",
        ):
            if kw in blob and len(blob) > 2:
                return True
    if eps == "LONG":
        edge = _safe_float(row.get("edge_score"))
        epct = _safe_float(row.get("edge_percentile"))
        if epct is not None and epct >= 80.0 and edge is not None and edge >= 0.5:
            return True
    return False


def _row_indicates_wide_spread(row: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Return (is_wide, reason). Looks across the likely variants of fields
    that could indicate spread stress on a row:

        spread_bucket / liquidity_pressure_bucket    (categorical buckets)
        entry_penalty_flag / wide_spread_flag         (boolean-ish flags)
        execution_risk_flag                           (HIGH alone is treated
                                                       as secondary evidence)
        spread_bps / spread_pct                       (numeric > guardrails)
    """
    # Treat UNKNOWN/NA/None as absent: they're placeholders, not real signals.
    sb_raw = _safe_str(row.get("spread_bucket")).upper()
    sb = "" if sb_raw in _EMPTY_CATEGORICAL_SENTINELS else sb_raw
    if sb in _WIDE_SPREAD_BUCKETS:
        return True, f"spread_bucket={sb}"

    lpb_raw = _safe_str(row.get("liquidity_pressure_bucket")).upper()
    lpb = "" if lpb_raw in _EMPTY_CATEGORICAL_SENTINELS else lpb_raw
    if lpb in _STRESSED_LIQUIDITY_BUCKETS:
        return True, f"liquidity_pressure_bucket={lpb}"

    lb_raw = _safe_str(row.get("liquidity_bucket")).upper()
    lb = "" if lb_raw in _EMPTY_CATEGORICAL_SENTINELS else lb_raw
    if lb in _STRESSED_LIQUIDITY_BUCKETS:
        return True, f"liquidity_bucket={lb}"

    exr = _safe_str(row.get("execution_reason")).upper()
    if exr and ("WIDE_SPREAD" in exr or ("WIDE" in exr and "TIGHT" not in exr)):
        return True, f"execution_reason={exr}"
    if exr and "SPREAD" in exr and "TIGHT" not in exr:
        return True, f"execution_reason={exr}"

    for flag_key in ("entry_penalty_flag", "wide_spread_flag"):
        v = _safe_str(row.get(flag_key)).upper()
        if v in _CONFIRM_FLAG_TOKENS:
            return True, f"{flag_key}={v}"

    # Stale quotes with a spread-flavored reason imply an unusable quote,
    # which is honestly the same operational risk the penalty targets.
    stale = _safe_bool(row.get("quote_is_stale"), default=None)
    qreason = _safe_str(row.get("quote_reason")).upper()
    if stale and ("SPREAD" in qreason or "WIDE" in qreason):
        return True, f"quote_is_stale,quote_reason={qreason}"

    # execution_risk_flag=HIGH alone is a *secondary* signal — only count it
    # when no explicit spread bucket was present, so we don't hide "HIGH
    # because of slippage" inside a wide-spread match.
    if not sb:
        erf_raw = _safe_str(row.get("execution_risk_flag")).upper()
        erf = "" if erf_raw in _EMPTY_CATEGORICAL_SENTINELS else erf_raw
        style_raw = _safe_str(row.get("execution_style")).upper()
        style = "" if style_raw in _EMPTY_CATEGORICAL_SENTINELS else style_raw
        if erf == "HIGH" and ("WIDE" in style or "DEFER" in style or "SPREAD" in style):
            return True, f"execution_risk_flag=HIGH,style={style}"
        if "WIDE" in style or "DEFER" in style:
            if "TIGHT" not in style:
                return True, f"execution_style={style}"

    # Numeric fallbacks — only used if explicit buckets aren't present.
    if not sb:
        bps = _safe_float(row.get("spread_bps"))
        if bps is not None and bps >= 50.0:  # 50 bps ≈ "wide" across most tickers
            return True, f"spread_bps={bps:.1f}"
        if bps is not None and bps >= 35.0:
            erf2 = _safe_str(row.get("execution_risk_flag")).upper()
            erf2 = "" if erf2 in _EMPTY_CATEGORICAL_SENTINELS else erf2
            if erf2 in ("MEDIUM", "HIGH", "ELEVATED"):
                return True, f"spread_bps={bps:.1f}+elevated_risk={erf2}"
        pct = _safe_float(row.get("spread_pct"))
        if pct is not None and pct >= 0.005:  # 50 bps threshold on %
            return True, f"spread_pct={pct:.4f}"
    return False, ""


def _match_wide_spread(row: Dict[str, Any]) -> bool:
    if not _is_entry_like(row):
        return False
    is_wide, _ = _row_indicates_wide_spread(row)
    return is_wide


def _match_stale_quote(row: Dict[str, Any]) -> bool:
    if not _is_entry_like(row):
        return False
    # Accept boolean, "True"/"1"/"yes", and an explicit STALE flag on
    # quote_reason / execution_risk_flag as fallback evidence.
    stale = _safe_bool(row.get("quote_is_stale"), default=None)
    if stale:
        return True
    reason_raw = _safe_str(row.get("quote_reason")).upper()
    reason = "" if reason_raw in _EMPTY_CATEGORICAL_SENTINELS else reason_raw
    if "STALE" in reason:
        return True
    # Also respect a boolean quote_staleness_flag carry-through if present.
    sflag = _safe_str(row.get("quote_staleness_flag")).upper()
    if sflag in ("STALE",) or sflag in _CONFIRM_FLAG_TOKENS:
        return True
    return False


def _match_high_exec_risk(row: Dict[str, Any]) -> bool:
    if not _is_entry_like(row):
        return False
    flag_raw = _safe_str(row.get("execution_risk_flag")).upper()
    flag = "" if flag_raw in _EMPTY_CATEGORICAL_SENTINELS else flag_raw
    return flag in ("HIGH", "ELEVATED")


def _match_add_only(row: Dict[str, Any]) -> bool:
    return _is_add_like(row)


def _match_high_conviction(row: Dict[str, Any]) -> bool:
    if not _is_entry_like(row):
        return False
    b = _safe_str(row.get("sizing_bucket")).upper()
    return b in ("HIGH_CONVICTION", "HIGH-CONVICTION", "HIGHCONVICTION", "CONVICTION_HIGH")


def _match_post_stop_out(row: Dict[str, Any], related_flag: str = "") -> bool:
    if not _is_entry_like(row):
        return False
    flag = _safe_str(related_flag).upper()
    if flag != "POST_STOP_OUT":
        return False
    oflag = _safe_str(row.get("post_stop_out_flag")).upper()
    return oflag in ("TRUE", "1", "YES", "Y", "POST_STOP_OUT")


# Static rule registry — every allow-listed target maps to one rule.
RULE_REGISTRY: Dict[str, AdjustmentRule] = {
    "wide_spread_entry_penalty": AdjustmentRule(
        target="wide_spread_entry_penalty",
        effect=EFFECT_PENALTY,
        matches=_match_wide_spread,
        applies_to_actions=["ENTRY", "ADD", "BUY"],
        description="Penalty on entries when spread_bucket ∈ {WIDE, TOO_WIDE}.",
    ),
    "stale_quote_penalty": AdjustmentRule(
        target="stale_quote_penalty",
        effect=EFFECT_PENALTY,
        matches=_match_stale_quote,
        applies_to_actions=["ENTRY", "ADD", "BUY"],
        description="Penalty on entries when quote_is_stale=True.",
    ),
    "stale_quote_entry_caution": AdjustmentRule(
        target="stale_quote_entry_caution",
        effect=EFFECT_PENALTY,
        matches=_match_stale_quote,
        applies_to_actions=["ENTRY", "ADD", "BUY"],
        description="Caution (penalty) on entries when quote_is_stale=True.",
    ),
    "low_confidence_entry_penalty": AdjustmentRule(
        target="low_confidence_entry_penalty",
        effect=EFFECT_PENALTY,
        matches=_match_low_confidence,
        applies_to_actions=["ENTRY", "ADD", "BUY"],
        description=(
            "Penalty on entry/buy rows in the low confidence band "
            "(numeric <0.55, LOW bucket, or derived LOW band)."
        ),
    ),
    "high_execution_risk_entry_trust": AdjustmentRule(
        target="high_execution_risk_entry_trust",
        effect=EFFECT_PENALTY,
        matches=_match_high_exec_risk,
        applies_to_actions=["ENTRY", "ADD", "BUY"],
        description="Penalty on entries when execution_risk_flag == HIGH.",
    ),
    "add_score_threshold": AdjustmentRule(
        target="add_score_threshold",
        effect=EFFECT_FLOOR_RAISE,
        matches=_match_add_only,
        applies_to_actions=["ADD"],
        description="Raise the required score floor for ADD opportunities.",
    ),
    "trim_profit_threshold": AdjustmentRule(
        target="trim_profit_threshold",
        effect=EFFECT_TRIM_THRESH,
        matches=_match_trim_scenario,
        applies_to_actions=["TRIM", "EXIT", "SELL"],
        description="Annotate TRIM rows — threshold would trigger earlier.",
    ),
    "high_conviction_bucket_validation": AdjustmentRule(
        target="high_conviction_bucket_validation",
        effect=EFFECT_BOOST,
        matches=_match_high_conviction,
        applies_to_actions=["ENTRY", "ADD", "BUY"],
        description="Small positive boost for HIGH_CONVICTION-bucket entries.",
    ),
    "position_cooldown_bias": AdjustmentRule(
        target="position_cooldown_bias",
        effect=EFFECT_ANNOTATE,
        matches=lambda r, rf="": _match_post_stop_out(r, rf),
        applies_to_actions=["ENTRY", "ADD", "BUY"],
        description="Annotate POST_STOP_OUT-flagged re-entries (cooldown bias).",
    ),
    "execution_entry_aggressiveness": AdjustmentRule(
        target="execution_entry_aggressiveness",
        effect=EFFECT_ANNOTATE,
        matches=lambda r: _is_entry_like(r),
        applies_to_actions=["ENTRY", "ADD", "BUY"],
        description=(
            "Annotate: execution aggressiveness dial (no accept/reject " "change in simulation)."
        ),
    ),
    "sizing_bucket_multiplier_adjustment": AdjustmentRule(
        target="sizing_bucket_multiplier_adjustment",
        effect=EFFECT_ANNOTATE,
        matches=lambda r: _is_entry_like(r),
        applies_to_actions=["ENTRY", "ADD", "BUY"],
        description=(
            "Annotate: sizing-bucket multiplier shift " "(exposure-only, no accept/reject change)."
        ),
    ),
}


# ──────────────────────────────────────────────────────────────
# Context assembly — per-row inputs for the rule engine
# ──────────────────────────────────────────────────────────────


def _pick_symbol_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for want in ("symbol", "ticker"):
        if want in lower_map:
            return str(lower_map[want])
    return None


def _norm_symbol_key(x: Any) -> str:
    return _safe_str(x).strip().upper()


def _series_get_ci(series: pd.Series, *names: str) -> Any:
    """First matching column (case-insensitive)."""
    idx = {str(c).strip().lower(): c for c in series.index}
    for n in names:
        key = n.lower()
        if key in idx:
            return series.get(idx[key])
    return None


_SPREAD_BUCKET_RANK: Dict[str, int] = {
    "": 0,
    "UNKNOWN": 0,
    "TIGHT": 1,
    "NARROW": 1,
    "NORMAL": 2,
    "MODERATE": 3,
    "WIDE": 4,
    "TOO_WIDE": 5,
    "VERY_WIDE": 6,
    "EXTREMELY_WIDE": 7,
}


def _spread_bucket_rank(sb: str) -> int:
    s = _safe_str(sb).upper()
    return int(_SPREAD_BUCKET_RANK.get(s, 0))


def _wider_spread_bucket(a: Optional[str], b: Optional[str]) -> str:
    ra = _spread_bucket_rank(a or "")
    rb = _spread_bucket_rank(b or "")
    aa = _safe_str(a)
    bb = _safe_str(b)
    return bb if rb > ra else aa


def _risk_rank(flag: str) -> int:
    f = _safe_str(flag).upper()
    if f in ("HIGH",):
        return 3
    if f in ("ELEVATED", "MEDIUM"):
        return 2
    if f in ("LOW",):
        return 1
    return 0


def _merge_ctx_fragment(acc: Dict[str, Any], frag: Dict[str, Any]) -> None:
    """Merge one execution-intel fragment into accumulator (stress-style: wider spread, max bps)."""
    for fk, fv in frag.items():
        if fv is None:
            continue
        if isinstance(fv, float) and (math.isnan(fv) or math.isinf(fv)):
            continue
        if fk in ("spread_bps", "spread_pct", "quote_age_sec"):
            cur = _safe_float(acc.get(fk))
            add = _safe_float(fv)
            if add is None:
                continue
            if cur is None or add > cur:
                acc[fk] = add
        elif fk == "spread_bucket":
            sb = _safe_str(fv).upper()
            if sb in _EMPTY_CATEGORICAL_SENTINELS:
                continue
            prev = acc.get("spread_bucket")
            acc["spread_bucket"] = _wider_spread_bucket(
                _safe_str(prev) if prev is not None else "",
                sb,
            )
        elif fk == "quote_is_stale":
            if _safe_bool(fv, default=False):
                acc["quote_is_stale"] = True
        elif fk == "execution_risk_flag":
            nf = _safe_str(fv)
            pf = _safe_str(acc.get("execution_risk_flag"))
            if _risk_rank(nf) >= _risk_rank(pf):
                acc["execution_risk_flag"] = nf
        elif fk not in acc or not _safe_str(acc.get(fk)):
            acc[fk] = fv


def _ei_row_to_fragment(row: pd.Series) -> Dict[str, Any]:
    """Extract execution-context fields from one execution_intel / plan row."""
    frag: Dict[str, Any] = {}
    sb = _series_get_ci(row, "spread_bucket")
    if sb is not None and not pd.isna(sb):
        sbu = _safe_str(sb).upper()
        if sbu and sbu not in _EMPTY_CATEGORICAL_SENTINELS:
            frag["spread_bucket"] = sbu
    lpb = _series_get_ci(row, "liquidity_pressure_bucket", "liquidity_bucket")
    if lpb is not None and not pd.isna(lpb):
        lpbu = _safe_str(lpb).upper()
        if lpbu and lpbu not in _EMPTY_CATEGORICAL_SENTINELS:
            frag["liquidity_pressure_bucket"] = lpbu
    for key in (
        "quote_reason",
        "execution_style",
        "execution_reason",
        "execution_skip_reason",
        "quote_staleness_flag",
        "entry_penalty_flag",
        "wide_spread_flag",
        "execution_context_source",
    ):
        v = _series_get_ci(row, key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = _safe_str(v)
        if s and s.upper() not in _EMPTY_CATEGORICAL_SENTINELS:
            frag[key] = v
    qst = _series_get_ci(row, "quote_is_stale")
    if qst is not None and not pd.isna(qst):
        b = _safe_bool(qst, default=None)
        if b is not None:
            frag["quote_is_stale"] = b
    erf = _series_get_ci(row, "execution_risk_flag")
    if erf is not None and not pd.isna(erf):
        eu = _safe_str(erf).upper()
        if eu and eu not in _EMPTY_CATEGORICAL_SENTINELS:
            frag["execution_risk_flag"] = eu
    for key in ("spread_bps", "spread_pct", "quote_age_sec", "execution_quality_score"):
        v = _series_get_ci(row, key)
        f = _safe_float(v)
        if f is not None:
            frag[key] = f
    return frag


def _build_exec_context_map(ei: pd.DataFrame, ep: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Per-symbol merged execution context: aggregates ALL rows per symbol so we
    keep the widest spread / max spread_bps / any stale quote / worst risk flag.
    execution_intelligence rows are merged before execution_plan (ep augments ei).
    """
    sym_frags: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def _collect(df: pd.DataFrame, *, prepend: bool) -> None:
        if df is None or df.empty:
            return
        sym_col = _pick_symbol_col(df)
        if sym_col is None:
            return
        for _, row in df.iterrows():
            sym = _norm_symbol_key(row.get(sym_col))
            if not sym:
                continue
            frag = _ei_row_to_fragment(row)
            if frag:
                if prepend:
                    sym_frags[sym].insert(0, frag)
                else:
                    sym_frags[sym].append(frag)

    _collect(ei, prepend=False)
    _collect(ep, prepend=False)

    ctx: Dict[str, Dict[str, Any]] = {}
    for sym, frags in sym_frags.items():
        acc: Dict[str, Any] = {}
        for frag in frags:
            _merge_ctx_fragment(acc, frag)
        if acc:
            ctx[sym] = acc
    return ctx


def _first_nonempty_str(*values: Any) -> str:
    for v in values:
        s = _safe_str(v)
        if s:
            return s
    return ""


def _first_non_none_bool(*values: Any) -> Optional[bool]:
    for v in values:
        b = _safe_bool(v, default=None)
        if b is not None:
            return b
    return None


def _first_non_none_float(*values: Any) -> Optional[float]:
    for v in values:
        f = _safe_float(v)
        if f is not None:
            return f
    return None


def _lifecycle_implies_trim_or_exit(r: pd.Series) -> str:
    """
    When the row clearly references trim/exit in lifecycle text and the
    account is long, override the surface opportunity_type.
    """
    eps = _safe_str(r.get("effective_position_state")).upper()
    if eps != "LONG":
        return ""
    blob = " ".join(
        [
            _safe_str(r.get("lifecycle_decision_reason")),
            _safe_str(r.get("reason_code")),
            _safe_str(r.get("stance_adjustment")),
        ]
    ).upper()
    if not blob.strip():
        return ""
    exit_phrases = (
        "FULL EXIT",
        "CLOSE POSITION",
        "LIQUIDATE",
        "STOP OUT",
        "FLAT TO FLAT",
        "EXIT POSITION",
        "CLOSE LONG",
    )
    for ph in exit_phrases:
        if ph in blob:
            return "EXIT"
    trim_phrases = (
        "TRIM",
        "REDUCE",
        "TAKE PROFIT",
        "TAKE_PROFIT",
        "DE-RISK",
        "DERISK",
        "PARE",
        "SCALING OUT",
        "PARTIAL",
        "PROFIT TAKE",
        "REDUCE SIZE",
    )
    for ph in trim_phrases:
        if ph in blob:
            return "TRIM"
    return ""


def _canonical_opportunity_type_from_series(r: pd.Series) -> str:
    """Infer ENTRY/ADD/TRIM/EXIT when opportunity_type is blank (runtime + simulation)."""
    ot = _safe_str(r.get("opportunity_type")).upper()
    if ot in ("ENTRY", "ADD", "TRIM", "EXIT"):
        return ot
    es = _safe_str(r.get("effective_stance")).upper()
    eps = _safe_str(r.get("effective_position_state")).upper()
    if es == "BUY" and eps == "FLAT":
        return "ENTRY"
    if es == "BUY" and eps == "LONG":
        return "ADD"
    if es in ("TRIM", "REDUCE") and eps == "LONG":
        return "TRIM"
    if es in ("EXIT", "CLOSE", "SELL") and eps == "LONG":
        return "EXIT"
    if es == "BUY":
        return "ENTRY" if eps == "FLAT" else "ADD"
    life = _lifecycle_implies_trim_or_exit(r)
    if life:
        return life
    return ot


def build_row_contexts(
    opps: pd.DataFrame, exec_ctx: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Return list of per-row dicts with all fields the rules need.

    Field precedence: direct columns on the opportunity row win over
    execution-context joins (the row-level writer is more specific). This
    also means future writers can ship spread / stale info directly on
    trade_opportunities.csv without requiring an exec_context join.
    """
    if opps is None or opps.empty:
        return []
    sym_col = _pick_symbol_col(opps)
    rows: List[Dict[str, Any]] = []
    for i, (_, r) in enumerate(opps.iterrows()):
        sym = _norm_symbol_key(r.get(sym_col)) if sym_col else ""
        ctx = dict(exec_ctx.get(sym, {})) if sym else {}
        ot_canon = _canonical_opportunity_type_from_series(r)
        rows.append(
            {
                "row_id": i + 1,
                "symbol": sym,
                "opportunity_type": ot_canon or _safe_str(r.get("opportunity_type")),
                "effective_stance": _safe_str(r.get("effective_stance")),
                "sizing_bucket": _safe_str(r.get("sizing_bucket")),
                "confidence": _safe_float(r.get("confidence")),
                "confidence_bucket": _safe_str(r.get("confidence_bucket")).upper(),
                "edge_score": _safe_float(r.get("edge_score")),
                "final_score": _safe_float(r.get("final_score")),
                "side": _safe_str(r.get("side")).upper(),
                "execution_blocked": _safe_bool(r.get("execution_blocked"), default=False),
                "execution_block_reason": _safe_str(r.get("execution_block_reason")),
                "post_stop_out_flag": _safe_str(r.get("post_stop_out_flag")),
                "effective_position_state": _safe_str(r.get("effective_position_state")).upper(),
                "lifecycle_decision_reason": _safe_str(r.get("lifecycle_decision_reason")),
                "reason_code": _safe_str(r.get("reason_code")),
                "stance_adjustment": _safe_str(r.get("stance_adjustment")),
                "edge_percentile": _safe_float(r.get("edge_percentile")),
                "confidence_band": _confidence_band_label(_safe_float(r.get("confidence"))),
                # Pulled-in execution context. Categoricals use _first_meaningful_str
                # so UNKNOWN-style placeholders written by build_trade_opportunities
                # fall through to the richer exec_ctx join (and finally stay empty
                # when neither side has a real signal).
                "spread_bucket": _first_meaningful_str(
                    r.get("spread_bucket"), ctx.get("spread_bucket")
                ).upper(),
                "liquidity_pressure_bucket": _first_meaningful_str(
                    r.get("liquidity_pressure_bucket"),
                    ctx.get("liquidity_pressure_bucket"),
                ).upper(),
                "quote_is_stale": _first_non_none_bool(
                    r.get("quote_is_stale"), ctx.get("quote_is_stale")
                ),
                "quote_reason": _first_meaningful_str(
                    r.get("quote_reason"), ctx.get("quote_reason")
                ),
                "quote_staleness_flag": _first_meaningful_str(
                    r.get("quote_staleness_flag"), ctx.get("quote_staleness_flag")
                ).upper(),
                "execution_risk_flag": _first_meaningful_str(
                    r.get("execution_risk_flag"), ctx.get("execution_risk_flag")
                ).upper(),
                "execution_style": _first_meaningful_str(
                    r.get("execution_style"), ctx.get("execution_style")
                ).upper(),
                "execution_reason": _first_meaningful_str(
                    r.get("execution_reason"), ctx.get("execution_reason")
                ),
                "execution_quality_score": _first_non_none_float(
                    r.get("execution_quality_score"),
                    ctx.get("execution_quality_score"),
                ),
                "liquidity_bucket": _first_meaningful_str(
                    r.get("liquidity_bucket"), ctx.get("liquidity_bucket")
                ).upper(),
                "entry_penalty_flag": _first_meaningful_str(
                    r.get("entry_penalty_flag"), ctx.get("entry_penalty_flag")
                ),
                "wide_spread_flag": _first_meaningful_str(
                    r.get("wide_spread_flag"), ctx.get("wide_spread_flag")
                ),
                "spread_bps": _first_non_none_float(r.get("spread_bps"), ctx.get("spread_bps")),
                "spread_pct": _first_non_none_float(r.get("spread_pct"), ctx.get("spread_pct")),
                # Provenance tag — helps diagnostics distinguish "no context data
                # available" from "we have context but it doesn't flag wide spread".
                "execution_context_source": _first_meaningful_str(
                    r.get("execution_context_source"),
                    ctx.get("execution_context_source"),
                ).upper(),
            }
        )
    return rows


def _runtime_derive_ctx_fields(ctx: Dict[str, Any]) -> None:
    """
    Honest enrichment for planner-time adaptation only (mutates ctx in place).
    Fills missing spread_bucket from numerics; maps liquidity_bucket → pressure.
    """
    sb_raw = _safe_str(ctx.get("spread_bucket")).upper()
    sb = "" if sb_raw in _EMPTY_CATEGORICAL_SENTINELS else sb_raw
    if not sb:
        bps = _safe_float(ctx.get("spread_bps"))
        pct = _safe_float(ctx.get("spread_pct"))
        if bps is not None and bps >= 50.0:
            ctx["spread_bucket"] = "WIDE"
            ctx["_spread_derived_from"] = "spread_bps"
        elif pct is not None and pct >= 0.005:
            ctx["spread_bucket"] = "WIDE"
            ctx["_spread_derived_from"] = "spread_pct"
        elif bps is not None and bps >= 35.0:
            erf = _safe_str(ctx.get("execution_risk_flag")).upper()
            if erf in ("MEDIUM", "HIGH", "ELEVATED"):
                ctx["spread_bucket"] = "WIDE"
                ctx["_spread_derived_from"] = "spread_bps+risk"
    lpb = _safe_str(ctx.get("liquidity_pressure_bucket")).upper()
    if lpb in _EMPTY_CATEGORICAL_SENTINELS or not lpb:
        lb = _safe_str(ctx.get("liquidity_bucket")).upper()
        if lb and lb not in _EMPTY_CATEGORICAL_SENTINELS:
            ctx["liquidity_pressure_bucket"] = lb


def build_runtime_row_contexts(
    opps: pd.DataFrame,
    exec_ctx: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """build_row_contexts + runtime-only enrichment for adaptation matching."""
    rows = build_row_contexts(opps, exec_ctx)
    for ctx in rows:
        _runtime_derive_ctx_fields(ctx)
    return rows


# ──────────────────────────────────────────────────────────────
# Simulation engine
# ──────────────────────────────────────────────────────────────


@dataclass
class AdjustmentInstance:
    """One ACTIVE applied adjustment, already parsed for simulation use."""

    application_id: str
    proposal_id: str
    adaptation_target: str
    effective_delta: float
    related_bucket: str
    related_flag: str
    related_style: str
    rule: AdjustmentRule
    rows_matched: int = 0
    rows_changed: int = 0
    match_reasons: List[str] = field(default_factory=list)
    reason_if_unused: str = ""
    # Per-miss diagnostic counters (populated in simulate()). These are only
    # used to build a specific `reason_if_unused` when rows_matched == 0.
    miss_predicate: int = 0  # rule predicate didn't fire on this row
    miss_bucket_mismatch: int = 0  # related_bucket set, row bucket was different
    miss_bucket_missing: int = 0  # related_bucket set, row had no bucket field
    miss_flag_scope: int = 0  # related_flag scope rejected the row
    miss_action: int = 0  # row action type didn't match rule (trim/add/entry)

    @property
    def delta_abs(self) -> float:
        return abs(self.effective_delta)


@dataclass
class UnsupportedAdjustment:
    """An active adjustment whose target has no simulation mapping."""

    application_id: str
    proposal_id: str
    adaptation_target: str
    effective_delta: float
    reason: str = "target_not_in_rule_registry"


def _split_applied_rows(
    active_df: pd.DataFrame,
) -> Tuple[List[AdjustmentInstance], List[UnsupportedAdjustment]]:
    """
    Normalize every active row, split into supported (mapped to RULE_REGISTRY)
    and unsupported (no known mapping). Also handles blank targets as
    unsupported so we never silently drop a row.
    """
    supported: List[AdjustmentInstance] = []
    unsupported: List[UnsupportedAdjustment] = []
    if active_df is None or active_df.empty:
        return supported, unsupported

    for _, raw_row in active_df.iterrows():
        n = _normalize_applied_row(raw_row)
        target = n["adaptation_target"]
        if not target:
            unsupported.append(
                UnsupportedAdjustment(
                    application_id=n["application_id"],
                    proposal_id=n["proposal_id"],
                    adaptation_target="",
                    effective_delta=float(n["effective_delta"]),
                    reason="missing_adaptation_target",
                )
            )
            continue
        rule = RULE_REGISTRY.get(target)
        if rule is None:
            unsupported.append(
                UnsupportedAdjustment(
                    application_id=n["application_id"],
                    proposal_id=n["proposal_id"],
                    adaptation_target=target,
                    effective_delta=float(n["effective_delta"]),
                    reason="target_not_in_rule_registry",
                )
            )
            continue
        supported.append(
            AdjustmentInstance(
                application_id=n["application_id"],
                proposal_id=n["proposal_id"],
                adaptation_target=target,
                effective_delta=float(n["effective_delta"]),
                related_bucket=n["related_bucket"],
                related_flag=n["related_flag"],
                related_style=n["related_style"],
                rule=rule,
            )
        )
    return supported, unsupported


def _build_instances(active_df: pd.DataFrame) -> List[AdjustmentInstance]:
    """Back-compat entry point — returns only the supported list."""
    supported, _ = _split_applied_rows(active_df)
    return supported


def _bucket_scope_matches(
    inst_bucket: str, row: Dict[str, Any], target: str
) -> Tuple[bool, str, str]:
    """
    Honour `related_bucket` scoping on the adjustment when the row actually
    carries the relevant bucket. Returns (ok, reason_if_not_ok, category).

    `category` is one of:
        ""                  — ok (or no scope set)
        "bucket_missing"    — related_bucket set but row's relevant field empty
        "bucket_mismatch"   — related_bucket set and conflicts with row

    Key fix: the old logic compared `related_bucket` against whichever of
    `spread_bucket`/`sizing_bucket` happened to be populated — which meant
    a TRIM row with `sizing_bucket=HIGH_CONVICTION` would be rejected if its
    spread bucket was missing. Here we pick the FIELD relevant to the
    target category, and we treat a missing value as *not a mismatch* so
    the row can still be matched by its primary rule predicate.
    """
    if not inst_bucket:
        return True, "", ""
    expected = inst_bucket.upper()
    # Targets that talk about SPREAD / LIQUIDITY scope to spread_bucket.
    if target in (
        "wide_spread_entry_penalty",
        "stale_quote_penalty",
        "stale_quote_entry_caution",
    ):
        row_bucket_raw = _safe_str(row.get("spread_bucket")).upper()
        row_bucket = "" if row_bucket_raw in _EMPTY_CATEGORICAL_SENTINELS else row_bucket_raw
        if not row_bucket:
            # Missing spread bucket — already handled by _row_indicates_wide_spread
            # fallbacks; don't enforce scope here (but record the miss category).
            return True, "", "bucket_missing"
        if expected in (row_bucket, ""):
            return True, "", ""
        # "TOO_WIDE" scope still counts the stricter "VERY_WIDE" / "EXTREMELY_WIDE"
        # as a match; "WIDE" scope accepts anything in _WIDE_SPREAD_BUCKETS.
        if expected == "TOO_WIDE" and row_bucket in {"VERY_WIDE", "EXTREMELY_WIDE"}:
            return True, "", ""
        if expected == "WIDE" and row_bucket in _WIDE_SPREAD_BUCKETS:
            return True, "", ""
        return False, f"spread_bucket={row_bucket} != related_bucket={expected}", "bucket_mismatch"

    # Confidence / signal family — never compare to sizing_bucket or spread_bucket.
    if target in ("low_confidence_entry_penalty",):
        if not expected:
            return True, "", ""
        c = _safe_float(row.get("confidence"))
        band = (
            _confidence_band_label(c)
            if c is not None
            else _safe_str(row.get("confidence_band")).upper()
        )
        if expected in (
            "LOW_CONF",
            "LOW",
            "LOW_CONFIDENCE",
            "SIGNAL_CAUTION",
            "WEAK_SIGNAL",
        ):
            if c is not None and c < 0.55:
                return True, "", ""
            if band == "LOW":
                return True, "", ""
            cb2 = _safe_str(row.get("confidence_bucket")).upper()
            if cb2 in ("LOW", "SIGNAL_CAUTION", "WEAK", "WEAK_SIGNAL", "CAUTION"):
                return True, "", ""
            return False, "related_bucket expects_low_confidence_band", "bucket_mismatch"
        if expected in ("MEDIUM", "MID", "NEUTRAL", "NEUTRAL_CONFIDENCE"):
            if c is not None and 0.55 <= c <= 0.70:
                return True, "", ""
            if band == "MEDIUM":
                return True, "", ""
            return False, "related_bucket expects_medium_confidence_band", "bucket_mismatch"
        if expected in ("HIGH", "STRONG", "STRONG_CONFIDENCE", "HIGH_CONFIDENCE"):
            if c is not None and c > 0.70:
                return True, "", ""
            if band == "HIGH":
                return True, "", ""
            return False, "related_bucket expects_high_confidence_band", "bucket_mismatch"
        cb = _safe_str(row.get("confidence_bucket")).upper()
        if cb and expected in (cb,):
            return True, "", ""
        if not cb:
            return True, "", "bucket_missing"
        return False, f"confidence_bucket={cb} != related_bucket={expected}", "bucket_mismatch"

    # Targets scoped to sizing bucket (HIGH_CONVICTION validation, trim etc.)
    row_bucket_raw = _safe_str(row.get("sizing_bucket")).upper()
    row_bucket = "" if row_bucket_raw in _EMPTY_CATEGORICAL_SENTINELS else row_bucket_raw
    if not row_bucket:
        return True, "", "bucket_missing"
    if expected in (row_bucket, ""):
        return True, "", ""
    return False, f"sizing_bucket={row_bucket} != related_bucket={expected}", "bucket_mismatch"


def _instance_matches(inst: AdjustmentInstance, row: Dict[str, Any]) -> Tuple[bool, str, str]:
    """
    Apply both rule.matches and related_* scoping (bucket/flag/style).

    Returns (matched, reason, miss_category). `reason` is a short
    human-readable string describing the primary hit signal (for the
    matching branch) or the first scope violation (for the non-match branch).
    `miss_category` is one of:
        "", "predicate", "action", "bucket_missing", "bucket_mismatch", "flag_scope"
    and is used by simulate() to aggregate honest reasons-not-matched.
    """
    target = inst.adaptation_target

    # cooldown rule consumes related_flag from the instance itself
    if target == "position_cooldown_bias":
        try:
            ok = bool(inst.rule.matches(row, inst.related_flag))  # type: ignore[call-arg]
        except TypeError:
            ok = False
        if ok:
            return True, "cooldown_post_stop_out", ""
        return False, "no_cooldown_flag", "flag_scope"

    if not inst.rule.matches(row):
        # Distinguish action-type mismatch (e.g. trim rule on an ENTRY row)
        # from a "right action but predicate conditions not met".
        if target in ("trim_profit_threshold",) and not _match_trim_scenario(row):
            return False, f"not_a_trim_row:{target}", "action"
        if target in ("add_score_threshold",) and not _is_add_like(row):
            return False, f"not_an_add_row:{target}", "action"
        entry_targets = (
            "wide_spread_entry_penalty",
            "stale_quote_penalty",
            "stale_quote_entry_caution",
            "low_confidence_entry_penalty",
            "high_execution_risk_entry_trust",
            "high_conviction_bucket_validation",
            "execution_entry_aggressiveness",
            "sizing_bucket_multiplier_adjustment",
        )
        if target in entry_targets and not _is_entry_like(row):
            return False, f"not_an_entry_row:{target}", "action"
        return False, f"rule_predicate_missed:{target}", "predicate"

    # related_bucket scope (uses the right field for the target category)
    bucket_ok, bucket_reason, bucket_cat = _bucket_scope_matches(inst.related_bucket, row, target)
    if not bucket_ok:
        return False, bucket_reason, bucket_cat or "bucket_mismatch"

    # related_flag scope for non-cooldown rules
    if inst.related_flag and target != "position_cooldown_bias":
        rf = inst.related_flag.upper()
        row_exec_risk_raw = _safe_str(row.get("execution_risk_flag")).upper()
        row_exec_risk = (
            "" if row_exec_risk_raw in _EMPTY_CATEGORICAL_SENTINELS else row_exec_risk_raw
        )
        row_stale = _safe_bool(row.get("quote_is_stale"), default=False) or (
            "STALE" in _safe_str(row.get("quote_reason")).upper()
        )
        if rf == "STALE" and not row_stale:
            return False, "related_flag=STALE but row not stale", "flag_scope"
        if rf in ("HIGH", "ELEVATED") and row_exec_risk not in ("HIGH", "ELEVATED"):
            return (
                False,
                f"related_flag={rf} but execution_risk_flag={row_exec_risk or 'N/A'}",
                "flag_scope",
            )
        if rf == "LOW_CONF":
            c = _safe_float(row.get("confidence"))
            if c is None or c >= 0.55:
                return False, f"related_flag=LOW_CONF but confidence={c}", "flag_scope"
        if rf == "POST_STOP_OUT":
            oflag = _safe_str(row.get("post_stop_out_flag")).upper()
            if oflag not in _CONFIRM_FLAG_TOKENS:
                return False, "related_flag=POST_STOP_OUT but no post_stop_out_flag", "flag_scope"
        # Other flags (WEAK_ADD, etc.) are informational — not a hard scope gate.

    # Build a human-readable match reason tied to the rule category.
    if target == "wide_spread_entry_penalty":
        _, hit = _row_indicates_wide_spread(row)
        return True, f"wide_spread_hit[{hit or 'generic'}]", ""
    if target in ("stale_quote_penalty", "stale_quote_entry_caution"):
        return True, "stale_quote_hit", ""
    if target == "low_confidence_entry_penalty":
        c = _safe_float(row.get("confidence"))
        band = (
            _confidence_band_label(c) if c is not None else _safe_str(row.get("confidence_bucket"))
        )
        return True, f"low_confidence_hit[c={c},band={band}]", ""
    if target == "high_execution_risk_entry_trust":
        return True, "high_exec_risk_hit", ""
    if target == "add_score_threshold":
        return True, "add_floor_raised", ""
    if target == "trim_profit_threshold":
        return True, "trim_annotation", ""
    if target == "high_conviction_bucket_validation":
        return True, "high_conviction_boost", ""
    return True, f"{target}_matched", ""


@dataclass
class SimResult:
    rows: List[Dict[str, Any]] = field(default_factory=list)
    instances: List[AdjustmentInstance] = field(default_factory=list)


def simulate(
    row_contexts: List[Dict[str, Any]],
    instances: List[AdjustmentInstance],
    *,
    score_floor: float = 0.0,
) -> SimResult:
    """
    Apply each instance to each row in memory and classify decisions.

    Baseline acceptance (read-only mirror of what's already in opps):
        not execution_blocked AND edge_score >= score_floor
    Simulated acceptance:
        identical + any active adjustment penalties / boosts / floor raises.
    """
    result = SimResult(instances=list(instances))

    for ctx in row_contexts:
        edge = _safe_float(ctx.get("edge_score"))
        execution_blocked = bool(ctx.get("execution_blocked") or False)
        block_reason = _safe_str(ctx.get("execution_block_reason"))

        baseline_score = edge if edge is not None else 0.0
        baseline_threshold = float(score_floor)
        baseline_accepted = (not execution_blocked) and (
            baseline_score >= baseline_threshold - 1e-9
        )
        baseline_reject_reason = ""
        if not baseline_accepted:
            if execution_blocked:
                baseline_reject_reason = block_reason or "execution_blocked"
            else:
                baseline_reject_reason = f"edge_score<{baseline_threshold:g}"

        penalty_total = 0.0
        boost_total = 0.0
        floor_raise_total = 0.0
        adjustments_hit: List[str] = []
        detail_bits: List[str] = []
        match_reasons: List[str] = []
        trim_hit = False
        cooldown_hit = False

        for inst in instances:
            matched, reason, miss_cat = _instance_matches(inst, ctx)
            if not matched:
                if miss_cat == "predicate":
                    inst.miss_predicate += 1
                elif miss_cat == "action":
                    inst.miss_action += 1
                elif miss_cat == "bucket_missing":
                    inst.miss_bucket_missing += 1
                elif miss_cat == "bucket_mismatch":
                    inst.miss_bucket_mismatch += 1
                elif miss_cat == "flag_scope":
                    inst.miss_flag_scope += 1
                continue
            inst.rows_matched += 1
            if reason:
                # Track unique reason examples per adjustment for the summary
                if reason not in inst.match_reasons:
                    inst.match_reasons.append(reason)
            tag = inst.adaptation_target
            adjustments_hit.append(tag)
            match_reasons.append(f"{tag}:{reason}" if reason else tag)

            effect = inst.rule.effect
            changed_state = False
            if effect == EFFECT_PENALTY:
                penalty = inst.delta_abs
                penalty_total += penalty
                detail_bits.append(f"{tag}:-{penalty:.4f}")
                changed_state = True
            elif effect == EFFECT_BOOST:
                boost = inst.delta_abs
                boost_total += boost
                detail_bits.append(f"{tag}:+{boost:.4f}")
                changed_state = True
            elif effect == EFFECT_FLOOR_RAISE:
                fr = inst.delta_abs
                floor_raise_total += fr
                detail_bits.append(f"{tag}:floor+{fr:.4f}")
                changed_state = True
            elif effect == EFFECT_TRIM_THRESH:
                trim_hit = True
                detail_bits.append(f"{tag}:trim_tighten_{inst.effective_delta:+.4f}")
                changed_state = True
            elif effect == EFFECT_ANNOTATE:
                if tag == "position_cooldown_bias":
                    cooldown_hit = True
                detail_bits.append(f"{tag}:annotate")
            if changed_state:
                inst.rows_changed += 1

        simulated_score = baseline_score - penalty_total + boost_total
        simulated_threshold = baseline_threshold + floor_raise_total

        # Simulated acceptance: baseline block always kills it (we don't
        # simulate un-blocking real execution gates in Phase 2).
        if execution_blocked:
            simulated_accepted = False
            simulated_reject_reason = block_reason or "execution_blocked"
        else:
            simulated_accepted = simulated_score >= simulated_threshold - 1e-9
            if simulated_accepted:
                simulated_reject_reason = ""
            else:
                if penalty_total > 0 and floor_raise_total > 0:
                    simulated_reject_reason = "penalty_and_threshold"
                elif penalty_total > 0:
                    simulated_reject_reason = "penalty_below_floor"
                elif floor_raise_total > 0:
                    simulated_reject_reason = "threshold_raised"
                else:
                    simulated_reject_reason = f"edge_score<{simulated_threshold:g}"

        if baseline_accepted and simulated_accepted:
            decision = DC_UNCHANGED_ACCEPT
        elif baseline_accepted and not simulated_accepted:
            decision = DC_NEWLY_REJECTED
        elif (not baseline_accepted) and simulated_accepted:
            decision = DC_NEWLY_ACCEPTED
        else:
            decision = DC_UNCHANGED_REJECT

        result.rows.append(
            {
                "row_id": ctx.get("row_id"),
                "symbol": ctx.get("symbol"),
                "opportunity_type": ctx.get("opportunity_type"),
                "effective_stance": ctx.get("effective_stance"),
                "sizing_bucket": ctx.get("sizing_bucket"),
                "confidence": ctx.get("confidence"),
                "edge_score": edge,
                "baseline_score": round(baseline_score, 6),
                "baseline_threshold": round(baseline_threshold, 6),
                "baseline_accepted": bool(baseline_accepted),
                "baseline_reject_reason": baseline_reject_reason,
                "simulated_score": round(simulated_score, 6),
                "simulated_threshold": round(simulated_threshold, 6),
                "simulated_penalty_total": round(penalty_total, 6),
                "simulated_boost_total": round(boost_total, 6),
                "simulated_accepted": bool(simulated_accepted),
                "simulated_reject_reason": simulated_reject_reason,
                "decision_change": decision,
                "score_delta": round(simulated_score - baseline_score, 6),
                "adjustments_applied": ";".join(sorted(set(adjustments_hit))),
                "adjustment_details": ";".join(detail_bits),
                "applied_delta_total": round(boost_total - penalty_total, 6),
                "match_reason": ";".join(match_reasons),
                "trim_threshold_tightened": bool(trim_hit),
                "cooldown_bias_applied": bool(cooldown_hit),
                "spread_bucket": ctx.get("spread_bucket"),
                "spread_bps": ctx.get("spread_bps"),
                "spread_pct": ctx.get("spread_pct"),
                "liquidity_pressure_bucket": ctx.get("liquidity_pressure_bucket"),
                "quote_is_stale": ctx.get("quote_is_stale"),
                "quote_reason": ctx.get("quote_reason"),
                "quote_staleness_flag": ctx.get("quote_staleness_flag"),
                "execution_risk_flag": ctx.get("execution_risk_flag"),
                "execution_style": ctx.get("execution_style"),
                "execution_quality_score": ctx.get("execution_quality_score"),
                "execution_context_source": ctx.get("execution_context_source"),
                "execution_blocked": bool(execution_blocked),
                "execution_block_reason": block_reason,
            }
        )
    return result


# ──────────────────────────────────────────────────────────────
# Runtime score influence (execute_trades planner integration)
# ──────────────────────────────────────────────────────────────
#
# Uses the same rule matching + active_adjustments filter as simulation,
# but mutates planner-facing scores so ranking / quality gates see them.
# Does not write CSVs or touch broker / lifecycle / risk configs.

RUNTIME_MAX_DELTA_PER_RULE = 0.08
# Default per-row cap; borderline (near min threshold) can use max row cap below.
RUNTIME_MAX_ROW_SCORE_DELTA = 0.20
RUNTIME_MAX_ROW_SCORE_BORDERLINE = 0.22
# When score is within ±10% of min_final_score threshold, net delta is nudged and
# a slightly higher per-row cap applies so adaptation can tip accept/reject.
RUNTIME_BORDERLINE_THRESH_PCT = 0.10
RUNTIME_BORDERLINE_DELTA_MULT = 1.10
RUNTIME_MAX_EDGE_DELTA = 0.10


def _runtime_quality_would_pass(
    fs: Optional[float],
    *,
    min_final_score_threshold: Optional[float],
    block_negative_final_score: bool,
) -> bool:
    """Mirror planner quality: optional min threshold + optional negative block."""
    f = _safe_float(fs)
    if f is None:
        f = 0.0
    if block_negative_final_score and f < 0.0:
        return False
    if min_final_score_threshold is not None and f < float(min_final_score_threshold) - 1e-12:
        return False
    return True


def _score_in_borderline_band(
    before: float,
    threshold: Optional[float],
) -> bool:
    """
    True when pre-adaptation score is within ±RUNTIME_BORDERLINE_THRESH_PCT of
    the quality threshold (relative band around the threshold value).
    """
    t = _safe_float(threshold)
    if t is None or t <= 0.0:
        return False
    b = _safe_float(before)
    if b is None:
        return False
    lo = t * (1.0 - RUNTIME_BORDERLINE_THRESH_PCT)
    hi = t * (1.0 + RUNTIME_BORDERLINE_THRESH_PCT)
    return lo <= b <= hi


def _count_adaptation_rank_reorders(out: pd.DataFrame) -> int:
    """
    How many rows changed position in score-desc + ticker tiebreak ordering
    after adaptation (vs baseline score order).
    """
    if (
        out is None
        or out.empty
        or "_adapt_rank_baseline" not in out.columns
        or "_exec_final_score" not in out.columns
    ):
        return 0
    try:
        tmp = out.copy()
        tmp["_b"] = pd.to_numeric(tmp["_adapt_rank_baseline"], errors="coerce")
        tmp["_a"] = pd.to_numeric(tmp["_exec_final_score"], errors="coerce")
        tcol = "ticker" if "ticker" in tmp.columns else None
        if tcol is None:
            tmp["_t"] = tmp.index.astype(str)
            tkey = "_t"
        else:
            tmp["_t"] = tmp[tcol].astype(str)
            tkey = "_t"
        o_b = tmp.sort_values(["_b", tkey], ascending=[False, True]).index
        o_a = tmp.sort_values(["_a", tkey], ascending=[False, True]).index
        pos_b = {ix: p for p, ix in enumerate(o_b)}
        pos_a = {ix: p for p, ix in enumerate(o_a)}
        nchg = 0
        for ix in tmp.index:
            if pos_b.get(ix) != pos_a.get(ix):
                nchg += 1
        return int(nchg)
    except Exception:
        return 0


def _runtime_score_delta_for_row(
    ctx: Dict[str, Any],
    instances: List[AdjustmentInstance],
) -> Tuple[float, List[str], List[str]]:
    """
    Aggregate bounded score delta for one row. Positive = higher final_score.
    Sums all matching adjustments (capped per rule and per row).
    Returns (delta, adaptation_target tags, human-readable match detail lines).
    """
    penalty_total = 0.0
    boost_total = 0.0
    floor_sub = 0.0
    trim_adj = 0.0
    tags: List[str] = []
    details: List[str] = []
    cap = float(RUNTIME_MAX_DELTA_PER_RULE)

    for inst in instances:
        matched, reason, _miss = _instance_matches(inst, ctx)
        if not matched:
            continue
        tags.append(inst.adaptation_target)
        effect = inst.rule.effect
        tag = inst.adaptation_target
        if effect == EFFECT_PENALTY:
            p = min(inst.delta_abs, cap)
            penalty_total += p
            details.append(f"{tag}:penalty{-p:.4f}:{reason or 'hit'}")
        elif effect == EFFECT_BOOST:
            b = min(inst.delta_abs, cap)
            boost_total += b
            details.append(f"{tag}:boost+{b:.4f}:{reason or 'hit'}")
        elif effect == EFFECT_FLOOR_RAISE:
            if _is_add_like(ctx):
                f = min(inst.delta_abs, cap)
                floor_sub += f
                details.append(f"{tag}:add_floor-{f:.4f}:{reason or 'hit'}")
            else:
                details.append(f"{tag}:floor_skip_not_add:{reason or ''}")
        elif effect == EFFECT_TRIM_THRESH:
            if _match_trim_scenario(ctx):
                ed = float(inst.effective_delta)
                if ed < 0:
                    t = min(abs(ed), cap)
                    trim_adj += t
                    details.append(f"{tag}:trim_priority+{t:.4f}:{reason or 'hit'}")
                else:
                    t = min(abs(ed), cap)
                    trim_adj -= t
                    details.append(f"{tag}:trim_priority-{t:.4f}:{reason or 'hit'}")
            else:
                details.append(f"{tag}:trim_skip_not_trim:{reason or ''}")
        elif effect == EFFECT_ANNOTATE:
            details.append(f"{tag}:annotate:{reason or 'hit'}")

    raw = boost_total - penalty_total - floor_sub + trim_adj
    # Per-row cap applied in apply_runtime_score_influence (borderline may use
    # RUNTIME_MAX_ROW_SCORE_BORDERLINE and RUNTIME_BORDERLINE_DELTA_MULT).
    return raw, tags, details


_RUNTIME_TRACE_DEFAULTS: List[Tuple[str, Any]] = [
    ("adjustment_applied", False),
    ("adjustment_type", ""),
    ("score_before_adjustment", float("nan")),
    ("score_after_adjustment", float("nan")),
    ("adaptation_match_reason", ""),
    ("adaptation_delta_applied", float("nan")),
]


def _ensure_runtime_trace_columns(out: pd.DataFrame) -> None:
    for c, default in _RUNTIME_TRACE_DEFAULTS:
        if c not in out.columns:
            out[c] = default


def _runtime_ctx_field_coverage(row_contexts: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = (
        "spread_bucket",
        "spread_bps",
        "spread_pct",
        "liquidity_pressure_bucket",
        "liquidity_bucket",
        "execution_risk_flag",
        "execution_style",
        "execution_reason",
        "quote_is_stale",
        "quote_reason",
        "confidence",
        "edge_score",
        "opportunity_type",
        "sizing_bucket",
        "confidence_bucket",
        "confidence_band",
        "edge_percentile",
        "effective_position_state",
    )
    n = max(1, len(row_contexts))
    cov: Dict[str, float] = {}

    def _meaningful(ctx: Dict[str, Any], k: str) -> bool:
        v = ctx.get(k)
        if v is None:
            return False
        if isinstance(v, bool):
            return True
        f = _safe_float(v)
        if f is not None:
            return True
        s = _safe_str(v).upper()
        return bool(s) and s not in _EMPTY_CATEGORICAL_SENTINELS

    for k in keys:
        nz = sum(1 for c in row_contexts if _meaningful(c, k))
        cov[k] = round(nz / n, 4)
    return cov


def _required_fields_hint_for_target(target: str) -> str:
    hints = {
        "wide_spread_entry_penalty": "entry_like + (spread_bucket|bps|pct|liq_pressure|exec_reason|style)",
        "stale_quote_penalty": "entry_like + quote_is_stale / stale reason",
        "stale_quote_entry_caution": "entry_like + quote_is_stale / stale reason",
        "low_confidence_entry_penalty": (
            "entry_like + (conf<0.55|LOW bucket|band); scope=MEDIUM/HIGH via numeric"
        ),
        "high_execution_risk_entry_trust": "entry_like + execution_risk_flag",
        "add_score_threshold": "ADD opportunity rows",
        "trim_profit_threshold": "TRIM/EXIT/LONG+profit-take or lifecycle trim hints",
        "high_conviction_bucket_validation": "entry_like + sizing_bucket",
        "position_cooldown_bias": "entry_like + post_stop_out_flag",
        "execution_entry_aggressiveness": "entry_like",
        "sizing_bucket_multiplier_adjustment": "entry_like",
    }
    return hints.get(target, "see RULE_REGISTRY")


def _build_runtime_adjustment_input_block(
    instances: List[AdjustmentInstance],
    unsupported: List[UnsupportedAdjustment],
    df_columns: List[str],
    coverage: Dict[str, float],
) -> List[Dict[str, Any]]:
    cols_l = {str(c).strip().lower() for c in df_columns}
    rows: List[Dict[str, Any]] = []
    for inst in instances:
        tgt = inst.adaptation_target
        avail = [k for k, frac in coverage.items() if frac > 0]
        rows.append(
            {
                "adaptation_target": tgt,
                "effect": inst.rule.effect,
                "effective_delta": round(float(inst.effective_delta), 6),
                "related_bucket": inst.related_bucket or "",
                "related_flag": inst.related_flag or "",
                "supported_rule": True,
                "required_fields_hint": _required_fields_hint_for_target(tgt),
                "df_has_ticker": "ticker" in cols_l or "symbol" in cols_l,
                "field_coverage_nonzero": ",".join(
                    f"{k}:{coverage[k]:.2f}" for k in sorted(coverage) if coverage[k] > 0
                )[:500],
            }
        )
    for u in unsupported:
        rows.append(
            {
                "adaptation_target": u.adaptation_target,
                "effect": "",
                "effective_delta": round(float(u.effective_delta), 6),
                "related_bucket": "",
                "related_flag": "",
                "supported_rule": False,
                "required_fields_hint": u.reason,
                "df_has_ticker": "ticker" in cols_l or "symbol" in cols_l,
                "field_coverage_nonzero": "",
            }
        )
    return rows


def _build_coverage_by_target(
    instances: List[AdjustmentInstance],
    row_contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Per-target diagnostics: how many rows were eligible (predicate) vs
    actually matched, plus match/changed rates.
    """
    n = max(1, len(row_contexts))
    out: Dict[str, Any] = {}
    for inst in instances:
        tgt = inst.adaptation_target
        if tgt == "trim_profit_threshold":
            elig = sum(1 for c in row_contexts if _match_trim_scenario(c))
        elif tgt == "low_confidence_entry_penalty":
            elig = sum(1 for c in row_contexts if _match_low_confidence(c))
        elif tgt == "wide_spread_entry_penalty":
            elig = sum(1 for c in row_contexts if _is_entry_like(c) and _match_wide_spread(c))
        elif tgt in ("stale_quote_penalty", "stale_quote_entry_caution"):
            elig = sum(1 for c in row_contexts if _is_entry_like(c) and _match_stale_quote(c))
        elif tgt == "high_execution_risk_entry_trust":
            elig = sum(1 for c in row_contexts if _is_entry_like(c) and _match_high_exec_risk(c))
        elif tgt == "add_score_threshold":
            elig = sum(1 for c in row_contexts if _is_add_like(c))
        elif tgt == "high_conviction_bucket_validation":
            elig = sum(1 for c in row_contexts if _match_high_conviction(c))
        else:
            rule = RULE_REGISTRY.get(tgt)
            if rule and rule.applies_to_actions:
                acts = set(rule.applies_to_actions)
                if acts <= {"TRIM", "EXIT", "SELL"}:
                    elig = sum(1 for c in row_contexts if _match_trim_scenario(c))
                elif "ADD" in acts and len(acts) == 1:
                    elig = sum(1 for c in row_contexts if _is_add_like(c))
                else:
                    elig = sum(1 for c in row_contexts if _is_entry_like(c))
            else:
                elig = n
        rm = int(inst.rows_matched)
        rc = int(inst.rows_changed)
        out[tgt] = {
            "eligible_rows": int(elig),
            "eligible_rate": round(elig / n, 4),
            "matched_rows": rm,
            "match_rate": round(rm / n, 4),
            "changed_rows": rc,
            "change_rate": round(rc / n, 4),
        }
    return out


def _finalize_runtime_unused_reasons(
    instances: List[AdjustmentInstance],
    row_contexts: List[Dict[str, Any]],
) -> None:
    n_entry = sum(1 for c in row_contexts if _is_entry_like(c))
    n_trim = sum(1 for c in row_contexts if _match_trim_scenario(c))
    n_add = sum(1 for c in row_contexts if _is_add_like(c))
    any_wide = any(_row_indicates_wide_spread(c)[0] for c in row_contexts)
    any_stale = any(_match_stale_quote(c) for c in row_contexts)
    any_low_conf = any(_match_low_confidence(c) for c in row_contexts)

    for inst in instances:
        if inst.rows_matched > 0:
            inst.reason_if_unused = ""
            continue
        t = inst.adaptation_target
        if t == "trim_profit_threshold":
            inst.reason_if_unused = (
                "no_trim_rows_for_target" if n_trim == 0 else "no_trim_rows_matched_predicate"
            )
        elif t in (
            "wide_spread_entry_penalty",
            "stale_quote_penalty",
            "stale_quote_entry_caution",
            "low_confidence_entry_penalty",
            "high_execution_risk_entry_trust",
            "high_conviction_bucket_validation",
            "execution_entry_aggressiveness",
            "sizing_bucket_multiplier_adjustment",
        ):
            if n_entry == 0:
                inst.reason_if_unused = "no_entry_rows_for_target"
            elif t == "wide_spread_entry_penalty" and not any_wide:
                inst.reason_if_unused = "missing_spread_context_on_opportunities"
            elif t in ("stale_quote_penalty", "stale_quote_entry_caution") and not any_stale:
                inst.reason_if_unused = "no_stale_quote_signal_on_rows"
            elif t == "low_confidence_entry_penalty" and not any_low_conf:
                inst.reason_if_unused = "no_low_confidence_rows"
            else:
                inst.reason_if_unused = "predicate_or_bucket_family_mismatch"
        elif t == "add_score_threshold":
            inst.reason_if_unused = (
                "no_add_rows_for_target" if n_add == 0 else "no_add_rows_matched_predicate"
            )
        else:
            inst.reason_if_unused = "no_qualifying_rows"


def apply_runtime_score_influence(
    df: pd.DataFrame,
    *,
    applied_path: Optional[Path] = None,
    exec_intel_path: Optional[Path] = None,
    exec_plan_path: Optional[Path] = None,
    min_final_score_threshold: Optional[float] = None,
    block_negative_final_score: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply ACTIVE applied_adjustments to planner scores on a trade_opportunities dataframe.

    Expects columns: ticker, _exec_final_score (set by execute_trades after diversification).
    Adds trace columns including adaptation_match_reason and adaptation_delta_applied.

    min_final_score_threshold: when not None, summary decision_changes compares pass/fail
    against this threshold; align with execute_trades enforce_min_final_score.
    block_negative_final_score: align with block_negative_final_score in execute_trades.

    Returns (df_out, summary_dict). If no active adjustments or empty df, df is unchanged
    (aside from default trace columns) and summary counts are zero.
    """
    _dc = {"newly_rejected": 0, "newly_accepted": 0, "rank_changes": 0}
    empty_summary: Dict[str, Any] = {
        "rows_affected": 0,
        "rows_score_delta_nonzero": 0,
        "avg_score_delta": 0.0,
        "top_adjustments": "",
        "active_adjustment_count": 0,
        "supported_adjustments": 0,
        "unsupported_adjustments": 0,
        "matched_targets": "",
        "unmatched_targets": "",
        "per_target_rows_matched": {},
        "per_target_rows_changed": {},
        "per_target_reason_if_unused": {},
        "runtime_input": [],
        "field_coverage": {},
        "available_runtime_fields": "",
        "coverage_by_target": {},
        "decision_changes": _dc,
        "borderline_band_rows": 0,
        "borderline_amplified_rows": 0,
    }
    if df is None or df.empty:
        return df, empty_summary

    ap_path = applied_path or APPLIED_ADJUSTMENTS_CSV
    ei_path = exec_intel_path or EXECUTION_INTELLIGENCE_CSV
    ep_path = exec_plan_path or EXECUTION_PLAN_CSV

    applied, st_ap = load_csv_safe(ap_path)
    if st_ap != "ok" or applied.empty:
        out = df.copy()
        _ensure_runtime_trace_columns(out)
        return out, empty_summary

    active = active_adjustments(applied)
    if active.empty:
        out = df.copy()
        _ensure_runtime_trace_columns(out)
        return out, empty_summary

    instances, unsupported = _split_applied_rows(active)
    if not instances:
        out = df.copy()
        _ensure_runtime_trace_columns(out)
        return out, {
            **empty_summary,
            "unsupported_adjustments": len(unsupported),
        }

    for inst in instances:
        inst.rows_matched = 0
        inst.rows_changed = 0

    ei, _ = load_csv_safe(ei_path)
    ep, _ = load_csv_safe(ep_path)
    exec_ctx = _build_exec_context_map(ei, ep)

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if "_exec_final_score" not in out.columns:
        _ensure_runtime_trace_columns(out)
        return out, {
            **empty_summary,
            "active_adjustment_count": len(active),
            "supported_adjustments": 0,
            "unsupported_adjustments": len(unsupported),
            "error": "missing_exec_final_score",
        }

    row_contexts = build_runtime_row_contexts(out, exec_ctx)
    if len(row_contexts) != len(out):
        _ensure_runtime_trace_columns(out)
        return out, {**empty_summary, "error": "context_row_count_mismatch"}

    coverage = _runtime_ctx_field_coverage(row_contexts)
    input_block = _build_runtime_adjustment_input_block(
        instances, unsupported, list(out.columns), coverage
    )

    for ctx in row_contexts:
        for inst in instances:
            m, _, _ = _instance_matches(inst, ctx)
            if m:
                inst.rows_matched += 1

    _ensure_runtime_trace_columns(out)
    out["_adapt_rank_baseline"] = pd.to_numeric(out["_exec_final_score"], errors="coerce")
    out["adjustment_applied"] = False
    out["adjustment_type"] = ""
    out["score_before_adjustment"] = float("nan")
    out["score_after_adjustment"] = float("nan")
    out["adaptation_match_reason"] = ""
    out["adaptation_delta_applied"] = float("nan")

    deltas: List[float] = []
    tag_counts: Dict[str, int] = {}
    rows_touched = 0
    borderline_band_rows = 0
    borderline_amplified_rows = 0
    changed_by_target: Dict[str, int] = {inst.adaptation_target: 0 for inst in instances}

    for i in range(len(out)):
        ctx = row_contexts[i]
        before = _safe_float(out.iloc[i].get("_exec_final_score"))
        if before is None:
            before = 0.0
        raw, tags, details = _runtime_score_delta_for_row(ctx, instances)
        raw0 = float(raw)
        in_bz = min_final_score_threshold is not None and _score_in_borderline_band(
            before, min_final_score_threshold
        )
        if in_bz:
            borderline_band_rows += 1
        if in_bz and abs(raw0) > 1e-15:
            raw0 *= float(RUNTIME_BORDERLINE_DELTA_MULT)
            borderline_amplified_rows += 1
        row_cap = (
            float(RUNTIME_MAX_ROW_SCORE_BORDERLINE) if in_bz else float(RUNTIME_MAX_ROW_SCORE_DELTA)
        )
        delta = max(-row_cap, min(row_cap, raw0))
        if in_bz:
            details = list(details or []) + [
                f"borderline_band(±{RUNTIME_BORDERLINE_THRESH_PCT:.0%}threshold)"
            ]
        after = before + delta
        ix = out.index[i]
        out.loc[ix, "score_before_adjustment"] = before
        out.loc[ix, "score_after_adjustment"] = after
        out.loc[ix, "adaptation_match_reason"] = " | ".join(details) if details else ""
        out.loc[ix, "adaptation_delta_applied"] = delta
        if tags:
            out.loc[ix, "adjustment_applied"] = True
            out.loc[ix, "adjustment_type"] = ";".join(sorted(set(tags)))
            rows_touched += 1
            for t in set(tags):
                tag_counts[t] = tag_counts.get(t, 0) + 1
        if abs(delta) > 1e-12:
            out.loc[ix, "_exec_final_score"] = after
            if "final_score" in out.columns:
                out.loc[ix, "final_score"] = after
            if "_div_final" in out.columns:
                out.loc[ix, "_div_final"] = after
            deltas.append(delta)
            for t in set(tags):
                changed_by_target[t] = changed_by_target.get(t, 0) + 1
            if "edge_score" in out.columns:
                e0 = _safe_float(out.iloc[i].get("edge_score"))
                if e0 is not None:
                    e_delta = max(
                        -RUNTIME_MAX_EDGE_DELTA,
                        min(RUNTIME_MAX_EDGE_DELTA, delta * 0.85),
                    )
                    e1 = max(0.0, min(1.0, e0 + e_delta))
                    out.loc[ix, "edge_score"] = e1

    for inst in instances:
        if inst.rule.effect != EFFECT_ANNOTATE:
            inst.rows_changed = int(changed_by_target.get(inst.adaptation_target, 0))
        else:
            inst.rows_changed = 0

    _finalize_runtime_unused_reasons(instances, row_contexts)

    inst_by_target = {inst.adaptation_target: inst for inst in instances}
    for row in input_block:
        if not row.get("supported_rule"):
            row["rows_matched"] = 0
            row["rows_changed"] = 0
            row["likely_no_match_reason"] = str(row.get("required_fields_hint") or "")
            continue
        inst = inst_by_target.get(str(row.get("adaptation_target") or ""))
        if inst is None:
            continue
        row["rows_matched"] = int(inst.rows_matched)
        row["rows_changed"] = int(inst.rows_changed)
        row["likely_no_match_reason"] = (
            str(inst.reason_if_unused or "") if inst.rows_matched == 0 else ""
        )

    top_adj = sorted(tag_counts.items(), key=lambda x: -x[1])[:8]
    top_str = ",".join(f"{k}:{v}" for k, v in top_adj)
    avg_d = float(sum(deltas) / len(deltas)) if deltas else 0.0

    matched_t = [inst.adaptation_target for inst in instances if inst.rows_matched > 0]
    unmatched_t = [inst.adaptation_target for inst in instances if inst.rows_matched == 0]

    decision_changes: Dict[str, int] = {
        "newly_rejected": 0,
        "newly_accepted": 0,
        "rank_changes": 0,
    }
    if min_final_score_threshold is not None or block_negative_final_score:
        nr = na = 0
        for i in range(len(out)):
            b = _safe_float(out["score_before_adjustment"].iloc[i])
            a = _safe_float(out["score_after_adjustment"].iloc[i])
            if b is None:
                b = 0.0
            if a is None:
                a = 0.0
            ok_b = _runtime_quality_would_pass(
                b,
                min_final_score_threshold=min_final_score_threshold,
                block_negative_final_score=block_negative_final_score,
            )
            ok_a = _runtime_quality_would_pass(
                a,
                min_final_score_threshold=min_final_score_threshold,
                block_negative_final_score=block_negative_final_score,
            )
            if ok_b and (not ok_a):
                nr += 1
            if (not ok_b) and ok_a:
                na += 1
        decision_changes["newly_rejected"] = nr
        decision_changes["newly_accepted"] = na
    decision_changes["rank_changes"] = _count_adaptation_rank_reorders(out)

    summary = {
        "rows_affected": int(rows_touched),
        "rows_score_delta_nonzero": len(deltas),
        "avg_score_delta": round(avg_d, 6),
        "top_adjustments": top_str,
        "active_adjustment_count": int(len(active)),
        "supported_adjustments": len(instances),
        "unsupported_adjustments": len(unsupported),
        "matched_targets": ";".join(sorted(set(matched_t))),
        "unmatched_targets": ";".join(sorted(set(unmatched_t))),
        "per_target_rows_matched": {
            inst.adaptation_target: inst.rows_matched for inst in instances
        },
        "per_target_rows_changed": {
            inst.adaptation_target: inst.rows_changed for inst in instances
        },
        "per_target_reason_if_unused": {
            inst.adaptation_target: inst.reason_if_unused for inst in instances
        },
        "runtime_input": input_block,
        "field_coverage": coverage,
        "available_runtime_fields": ",".join(
            k for k, v in sorted(coverage.items()) if float(v or 0) > 0
        ),
        "coverage_by_target": _build_coverage_by_target(instances, row_contexts),
        "decision_changes": decision_changes,
        "borderline_band_rows": int(borderline_band_rows),
        "borderline_amplified_rows": int(borderline_amplified_rows),
    }
    return out, summary


# ──────────────────────────────────────────────────────────────
# Summary aggregation
# ──────────────────────────────────────────────────────────────


def _decision_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {k: 0 for k in DECISION_CHANGE_ORDER}
    for r in rows:
        dc = r.get("decision_change")
        if dc in counts:
            counts[dc] += 1
    return counts


def _accept_counts(rows: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    b_acc = sum(1 for r in rows if r.get("baseline_accepted"))
    b_rej = sum(1 for r in rows if not r.get("baseline_accepted"))
    s_acc = sum(1 for r in rows if r.get("simulated_accepted"))
    s_rej = sum(1 for r in rows if not r.get("simulated_accepted"))
    return b_acc, b_rej, s_acc, s_rej


def _exec_context_present(row_contexts: List[Dict[str, Any]]) -> bool:
    """True if ANY row carries at least one non-placeholder execution field."""
    for r in row_contexts:
        for k in (
            "spread_bucket",
            "execution_risk_flag",
            "liquidity_pressure_bucket",
            "execution_style",
        ):
            if _safe_str(r.get(k)).upper() not in _EMPTY_CATEGORICAL_SENTINELS:
                return True
        if _safe_float(r.get("spread_bps")) is not None:
            return True
        if _safe_float(r.get("spread_pct")) is not None:
            return True
        if _safe_bool(r.get("quote_is_stale"), default=False):
            return True
        src = _safe_str(r.get("execution_context_source")).upper()
        if src and src not in _EMPTY_CATEGORICAL_SENTINELS:
            return True
    return False


def _reason_if_unused_for(
    inst: AdjustmentInstance,
    row_contexts: List[Dict[str, Any]],
) -> str:
    """
    Build an honest, short, debug-friendly reason explaining why a
    supported adjustment didn't match anything.

    Prefers specific categories in this order:
        1. no_entry_candidates / no_trim_candidates / no_add_candidates
        2. bucket_scope_missing / bucket_scope_mismatch (if scope was the cause)
        3. no_execution_context_for_symbols (no exec data available anywhere)
        4. spread_context_present_but_not_wide / stale/low-conf equivalents
        5. supported_rule_but_no_rows_matched (catch-all)
    """
    target = inst.adaptation_target
    entry_like = sum(1 for r in row_contexts if _is_entry_like(r))
    trim_like = sum(1 for r in row_contexts if _match_trim_scenario(r))
    add_like = sum(1 for r in row_contexts if _is_add_like(r))
    total = len(row_contexts)

    # If the adjustment has a related_bucket and every miss was bucket-related,
    # surface that directly — it's actionable (operator can drop / retarget scope).
    if inst.related_bucket:
        total_misses = (
            inst.miss_predicate
            + inst.miss_action
            + inst.miss_bucket_missing
            + inst.miss_bucket_mismatch
            + inst.miss_flag_scope
        )
        if total_misses > 0 and inst.miss_bucket_mismatch > 0 and inst.miss_bucket_missing == 0:
            return f"bucket_scope_mismatch related_bucket={inst.related_bucket}"
        if total_misses > 0 and inst.miss_bucket_missing > 0 and inst.miss_bucket_mismatch == 0:
            return f"bucket_scope_missing related_bucket={inst.related_bucket}"

    if target == "trim_profit_threshold":
        if trim_like == 0:
            return "no_trim_candidates_in_opportunities"
        return "supported_rule_but_no_rows_matched (trim action)"
    if target == "add_score_threshold":
        if add_like == 0:
            return "no_add_candidates_in_opportunities"
        return "supported_rule_but_no_rows_matched (add action)"

    if target == "wide_spread_entry_penalty":
        if entry_like == 0:
            return "no_entry_candidates_in_opportunities"
        # Honest tiering: do we have ANY execution context at all?
        has_any_ctx = _exec_context_present(row_contexts)
        if not has_any_ctx:
            return (
                "no_execution_context_for_symbols — neither trade_opportunities "
                "nor execution_intelligence/execution_plan cover these symbols"
            )
        with_spread = sum(
            1
            for r in row_contexts
            if _safe_str(r.get("spread_bucket")).upper() not in _EMPTY_CATEGORICAL_SENTINELS
        )
        with_any_spread_signal = sum(
            1
            for r in row_contexts
            if (_safe_str(r.get("spread_bucket")).upper() not in _EMPTY_CATEGORICAL_SENTINELS)
            or (_safe_float(r.get("spread_bps")) is not None)
            or (_safe_float(r.get("spread_pct")) is not None)
            or (
                _safe_str(r.get("liquidity_pressure_bucket")).upper() in _STRESSED_LIQUIDITY_BUCKETS
            )
        )
        if with_any_spread_signal == 0:
            return "spread_context_present_but_not_spread_related"
        if with_spread > 0 or with_any_spread_signal > 0:
            return (
                f"spread_context_present_but_not_wide — "
                f"{with_any_spread_signal}/{total} rows carry a spread signal, "
                "none cross WIDE/TOO_WIDE or bps/pct thresholds"
            )
        return "supported_rule_but_no_rows_matched (wide spread)"

    if target in ("stale_quote_penalty", "stale_quote_entry_caution"):
        any_stale = any(
            _safe_bool(r.get("quote_is_stale"), default=False)
            or _safe_str(r.get("quote_staleness_flag")).upper() == "STALE"
            for r in row_contexts
        )
        has_any_ctx = _exec_context_present(row_contexts)
        if not has_any_ctx:
            return "no_execution_context_for_symbols"
        if not any_stale:
            return "no_stale_quotes_observed"
        return "stale_rows_failed_related_flag_scope"

    if target == "low_confidence_entry_penalty":
        low = sum(1 for r in row_contexts if (_safe_float(r.get("confidence")) or 1.0) < 0.55)
        if low == 0:
            return "no_rows_below_confidence_0_55"
        return "supported_rule_but_no_rows_matched (low confidence)"

    if target == "high_execution_risk_entry_trust":
        hi = sum(
            1
            for r in row_contexts
            if _safe_str(r.get("execution_risk_flag")).upper() in ("HIGH", "ELEVATED")
        )
        if hi == 0:
            has_any_ctx = _exec_context_present(row_contexts)
            if not has_any_ctx:
                return "no_execution_context_for_symbols"
            return "no_rows_with_high_execution_risk_flag"
        return "supported_rule_but_no_rows_matched (high exec risk)"

    if target == "high_conviction_bucket_validation":
        hc = sum(
            1
            for r in row_contexts
            if _safe_str(r.get("sizing_bucket")).upper() == "HIGH_CONVICTION"
        )
        if hc == 0:
            return "no_high_conviction_sizing_rows"
        return "supported_rule_but_no_rows_matched (high conviction)"

    if target == "position_cooldown_bias":
        cd = sum(
            1
            for r in row_contexts
            if _safe_str(r.get("post_stop_out_flag")).upper() in _CONFIRM_FLAG_TOKENS
        )
        if cd == 0:
            return "no_rows_with_post_stop_out_flag"
        return "supported_rule_but_no_rows_matched (cooldown)"

    return "supported_rule_but_no_rows_matched"


def build_summary(
    inp: SimInputs,
    result: SimResult,
    *,
    score_floor: float,
    notes: List[str],
    row_contexts: Optional[List[Dict[str, Any]]] = None,
    unsupported: Optional[List[UnsupportedAdjustment]] = None,
) -> Dict[str, Any]:
    rows = result.rows
    total = len(rows)
    b_acc, b_rej, s_acc, s_rej = _accept_counts(rows)
    dc_counts = _decision_counts(rows)
    row_contexts = row_contexts or []
    unsupported = unsupported or []

    # Threshold utilization counters
    add_floor_hits = sum(
        1
        for inst in result.instances
        if inst.rule.effect == EFFECT_FLOOR_RAISE and inst.rows_matched > 0
    )
    add_floor_total_matches = sum(
        inst.rows_matched for inst in result.instances if inst.rule.effect == EFFECT_FLOOR_RAISE
    )
    trim_rows = sum(1 for r in rows if r.get("trim_threshold_tightened"))
    cooldown_rows = sum(1 for r in rows if r.get("cooldown_bias_applied"))

    # Risk impact (counts-only — never exposes any real risk metric).
    # UNKNOWN/NA placeholders are treated as "no signal" here to avoid
    # false positives from the build_trade_opportunities carry-through.
    def _cat(v: Any) -> str:
        s = _safe_str(v).upper()
        return "" if s in _EMPTY_CATEGORICAL_SENTINELS else s

    wide_spread_rows = sum(1 for r in rows if _cat(r.get("spread_bucket")) in _WIDE_SPREAD_BUCKETS)
    stale_rows = sum(
        1
        for r in rows
        if _safe_bool(r.get("quote_is_stale"), default=False)
        or _cat(r.get("quote_staleness_flag")) == "STALE"
    )
    high_exec_risk_rows = sum(
        1 for r in rows if _cat(r.get("execution_risk_flag")) in ("HIGH", "ELEVATED")
    )

    # Exposure delta estimate — sum of edge_score*sizing_multiplier proxies
    # for newly-rejected rows (approximation, read-only, no real exposure calc)
    def _exposure_proxy(r: Dict[str, Any]) -> float:
        es = _safe_float(r.get("edge_score"), default=0.0) or 0.0
        # sizing bucket multiplier defaults: STANDARD=1.0, HIGH_CONVICTION=1.25,
        # LOW_CONVICTION=0.75 — heuristic proxy only.
        bucket = _safe_str(r.get("sizing_bucket")).upper()
        mult = {"STANDARD": 1.0, "HIGH_CONVICTION": 1.25, "LOW_CONVICTION": 0.75}.get(bucket, 1.0)
        return es * mult

    nr_exposure = sum(
        _exposure_proxy(r) for r in rows if r.get("decision_change") == DC_NEWLY_REJECTED
    )
    na_exposure = sum(
        _exposure_proxy(r) for r in rows if r.get("decision_change") == DC_NEWLY_ACCEPTED
    )

    # Per-adjustment usage block with rows_matched / rows_changed / reason_if_unused
    adj_rows: List[Dict[str, Any]] = []
    for inst in result.instances:
        reason_if_unused = ""
        if inst.rows_matched == 0:
            reason_if_unused = _reason_if_unused_for(inst, row_contexts)
            inst.reason_if_unused = reason_if_unused
        adj_rows.append(
            {
                "application_id": inst.application_id,
                "proposal_id": inst.proposal_id,
                "adaptation_target": inst.adaptation_target,
                "effect": inst.rule.effect,
                "effective_delta": round(inst.effective_delta, 6),
                "related_bucket": inst.related_bucket,
                "related_flag": inst.related_flag,
                "related_style": inst.related_style,
                "rows_matched": int(inst.rows_matched),
                "rows_changed": int(inst.rows_changed),
                "match_reasons_sample": list(inst.match_reasons)[:3],
                "reason_if_unused": reason_if_unused,
                "miss_breakdown": {
                    "predicate": int(inst.miss_predicate),
                    "action": int(inst.miss_action),
                    "bucket_missing": int(inst.miss_bucket_missing),
                    "bucket_mismatch": int(inst.miss_bucket_mismatch),
                    "flag_scope": int(inst.miss_flag_scope),
                },
                "supported": True,
                "description": inst.rule.description,
            }
        )
    adj_rows_used = [a for a in adj_rows if a["rows_matched"] > 0]

    unsupported_rows = [
        {
            "application_id": u.application_id,
            "proposal_id": u.proposal_id,
            "adaptation_target": u.adaptation_target,
            "effective_delta": round(u.effective_delta, 6),
            "reason": u.reason,
            "supported": False,
            "rows_matched": 0,
            "rows_changed": 0,
            "reason_if_unused": u.reason,
        }
        for u in unsupported
    ]

    matched_targets = sorted({a["adaptation_target"] for a in adj_rows_used})
    unmatched_targets = sorted({a["adaptation_target"] for a in adj_rows if a["rows_matched"] == 0})
    unsupported_targets = sorted({u.adaptation_target for u in unsupported})

    total_adjustment_row_matches = sum(a["rows_matched"] for a in adj_rows)

    summary = {
        "generated_at_utc": _utc_now_iso(),
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "simulation_only": True,
        "advisory_only": True,
        "auto_apply_allowed": False,
        "source_availability": {
            "applied_adjustments_csv": {
                "status": inp.status.get("applied_adjustments_csv"),
                "rows": int(inp.applied.shape[0]),
                "path": str(APPLIED_ADJUSTMENTS_CSV),
            },
            "trade_opportunities_csv": {
                "status": inp.status.get("trade_opportunities_csv"),
                "rows": int(inp.opps.shape[0]),
                "path": str(TRADE_OPPORTUNITIES_CSV),
            },
            "execution_intelligence_csv": {
                "status": inp.status.get("execution_intelligence_csv"),
                "rows": int(inp.exec_intel.shape[0]),
                "path": str(EXECUTION_INTELLIGENCE_CSV),
            },
            "execution_plan_csv": {
                "status": inp.status.get("execution_plan_csv"),
                "rows": int(inp.exec_plan.shape[0]),
                "path": str(EXECUTION_PLAN_CSV),
            },
        },
        "missing_inputs": inp.missing(),
        "score_floor": float(score_floor),
        "opportunities_seen": total,
        "active_adjustments_count": len(result.instances) + len(unsupported),
        "active_adjustments_loaded_count": len(result.instances) + len(unsupported),
        "active_adjustments_used_count": len(adj_rows_used),
        "total_adjustment_row_matches": int(total_adjustment_row_matches),
        "supported_adjustments_count": len(result.instances),
        "unsupported_adjustments_count": len(unsupported),
        "unsupported_adjustment_targets": unsupported_targets,
        "matched_targets": matched_targets,
        "unmatched_targets": unmatched_targets,
        "row_context_coverage": {
            "rows_with_any_execution_context": sum(
                1
                for r in row_contexts
                if (_safe_str(r.get("spread_bucket")).upper() not in _EMPTY_CATEGORICAL_SENTINELS)
                or (
                    _safe_str(r.get("execution_risk_flag")).upper()
                    not in _EMPTY_CATEGORICAL_SENTINELS
                )
                or (_safe_str(r.get("quote_staleness_flag")).upper() == "STALE")
                or _safe_bool(r.get("quote_is_stale"), default=False)
                or (_safe_float(r.get("spread_bps")) is not None)
                or (_safe_float(r.get("spread_pct")) is not None)
            ),
            "rows_total": total,
            "note": (
                "count of opportunity rows that carry at least one "
                "non-placeholder execution-context signal usable by "
                "simulation rules."
            ),
        },
        "active_adjustments": adj_rows,
        "unsupported_adjustments": unsupported_rows,
        "baseline_counts": {
            "accepted": b_acc,
            "rejected": b_rej,
            "total": total,
        },
        "simulated_counts": {
            "accepted": s_acc,
            "rejected": s_rej,
            "total": total,
        },
        "decision_change_counts": dc_counts,
        "exposure_delta_estimate": {
            "newly_rejected_exposure_proxy": round(nr_exposure, 6),
            "newly_accepted_exposure_proxy": round(na_exposure, 6),
            "net_exposure_change_proxy": round(na_exposure - nr_exposure, 6),
            "note": (
                "heuristic proxy: edge_score × bucket multiplier on rows "
                "whose decision would have changed; not a real risk figure."
            ),
        },
        "threshold_utilization": {
            "add_score_threshold_rules_used": add_floor_hits,
            "add_score_threshold_matches": add_floor_total_matches,
            "trim_profit_threshold_rows": trim_rows,
            "position_cooldown_bias_rows": cooldown_rows,
        },
        "risk_impact_counts": {
            "wide_or_too_wide_spread_rows": wide_spread_rows,
            "stale_quote_rows": stale_rows,
            "high_execution_risk_rows": high_exec_risk_rows,
            "note": "counts only — the simulation never alters real risk state.",
        },
        "notes": list(notes),
    }
    return summary


# ──────────────────────────────────────────────────────────────
# Writers — always emit stable schema, even when empty
# ──────────────────────────────────────────────────────────────


def _empty_sim_df() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in SIM_COLUMNS})


def _df_for_csv(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty_sim_df()
    df = pd.DataFrame(rows)
    # Preserve schema column order; append any unexpected columns at the end.
    extra = [c for c in df.columns if c not in SIM_COLUMNS]
    ordered = [c for c in SIM_COLUMNS if c in df.columns] + extra
    df = df.reindex(columns=ordered)
    # Bool columns → int for CSV stability, then back to str
    for bcol in (
        "baseline_accepted",
        "simulated_accepted",
        "trim_threshold_tightened",
        "cooldown_bias_applied",
        "quote_is_stale",
        "execution_blocked",
    ):
        if bcol in df.columns:
            df[bcol] = df[bcol].apply(
                lambda v: "" if v is None else ("True" if bool(v) else "False")
            )
    # Fill any missing schema columns
    for c in SIM_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[[c for c in SIM_COLUMNS] + extra]
    return df


def write_outputs(
    summary: Dict[str, Any],
    result: SimResult,
) -> Dict[str, str]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sim_df = _df_for_csv(result.rows)
    sim_df.to_csv(SIM_CSV, index=False)
    SIM_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {
        "simulation_csv": str(SIM_CSV),
        "summary_json": str(SIM_SUMMARY_JSON),
    }


# ──────────────────────────────────────────────────────────────
# Orchestrator + CLI
# ──────────────────────────────────────────────────────────────


def run_simulation(*, score_floor: float = 0.0, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    inp = load_inputs()
    if inp.missing():
        notes.append(
            "Missing or unreadable input(s): "
            + ", ".join(inp.missing())
            + ". Simulation ran on best-effort partial data."
        )

    if inp.opps is None or inp.opps.empty:
        notes.append("No trade_opportunities.csv rows — simulation is empty by construction.")

    active_df = active_adjustments(inp.applied)
    if active_df.empty:
        notes.append(
            "No active applied adjustments found — simulation reports baseline "
            "as identical to simulated."
        )

    exec_ctx = _build_exec_context_map(inp.exec_intel, inp.exec_plan)
    contexts = build_row_contexts(inp.opps, exec_ctx)
    instances, unsupported = _split_applied_rows(active_df)

    if unsupported:
        notes.append(
            "Active adjustment target(s) with no simulation rule "
            f"(recorded as unsupported): "
            f"{sorted({u.adaptation_target or '(blank)' for u in unsupported})}."
        )

    if verbose:
        # Per-adjustment load log BEFORE simulation so operators can see what
        # was actually loaded.
        for inst in instances:
            print(
                f"[adaptation_simulation] active adjustment loaded: "
                f"target={inst.adaptation_target} "
                f"effect={inst.rule.effect} "
                f"delta={inst.effective_delta:+.4f} "
                f"related_bucket={inst.related_bucket or '-'} "
                f"related_flag={inst.related_flag or '-'}"
            )
        for u in unsupported:
            print(
                f"[adaptation_simulation] unsupported adjustment: "
                f"target={u.adaptation_target or '(blank)'} "
                f"reason={u.reason}"
            )

    result = simulate(contexts, instances, score_floor=score_floor)
    summary = build_summary(
        inp,
        result,
        score_floor=score_floor,
        notes=notes,
        row_contexts=contexts,
        unsupported=unsupported,
    )
    written = write_outputs(summary, result)

    if verbose:
        for inst in result.instances:
            print(
                f"[adaptation_simulation] matched rows for "
                f"target={inst.adaptation_target}: {inst.rows_matched} "
                f"(changed={inst.rows_changed})"
                + (
                    f" — reason_if_unused={inst.reason_if_unused}"
                    if inst.rows_matched == 0 and inst.reason_if_unused
                    else ""
                )
            )
        print(
            f"[adaptation_simulation] opportunities={summary['opportunities_seen']}, "
            f"active_adjustments_loaded={summary['active_adjustments_count']} "
            f"(supported={summary['supported_adjustments_count']}, "
            f"unsupported={summary['unsupported_adjustments_count']}, "
            f"used={summary['active_adjustments_used_count']}, "
            f"total_row_matches={summary['total_adjustment_row_matches']})"
        )
        print(
            f"[adaptation_simulation] baseline accepted="
            f"{summary['baseline_counts']['accepted']}, "
            f"simulated accepted={summary['simulated_counts']['accepted']}, "
            f"newly_rejected={summary['decision_change_counts'][DC_NEWLY_REJECTED]}, "
            f"newly_accepted={summary['decision_change_counts'][DC_NEWLY_ACCEPTED]}"
        )
        for k, v in written.items():
            print(f"[adaptation_simulation] {k}: {v}")

    return {
        "summary": summary,
        "result": result,
        "written": written,
        "unsupported": unsupported,
    }


# ══════════════════════════════════════════════════════════════════
#  STRESS TEST HARNESS — TEST-ONLY.
#
#  Everything below this banner is a SELF-CONTAINED test harness.
#  It drives the SAME live rule engine (`simulate`, `build_summary`,
#  `_df_for_csv`) against synthetic in-memory rows so we can prove
#  the simulator behaves correctly under adversarial conditions.
#
#  Hard guarantees:
#    * Does NOT read or modify real `trade_opportunities.csv`,
#      `applied_adjustments.csv`, or any other production artifact.
#    * Writes ONLY to dedicated stress-test output files:
#        data/results/adaptation_simulation.stress_test.csv
#        data/results/adaptation_simulation_summary.stress_test.json
#    * No change to execution logic, broker code, signal generation,
#      lifecycle logic, risk sizing, or portfolio logic.
#
#  Invoke with:  python -m services.adaptation_simulation --stress-test
# ══════════════════════════════════════════════════════════════════

STRESS_SIM_CSV = RESULTS / "adaptation_simulation.stress_test.csv"
STRESS_SIM_SUMMARY_JSON = RESULTS / "adaptation_simulation_summary.stress_test.json"
STRESS_SCORE_FLOOR = 0.50  # Chosen so penalties and boosts actually flip decisions.


def _stress_synthetic_contexts() -> List[Dict[str, Any]]:
    """Nine synthetic row contexts spanning every category we need:

    1-3  → baseline accepted, knocked below floor by penalties → NEWLY_REJECTED
    4    → baseline accepted ADD, floor raised by add_score_threshold → NEWLY_REJECTED
    5-6  → baseline rejected, lifted above floor by boost → NEWLY_ACCEPTED
    7    → clean setup, no rule fires → UNCHANGED_ACCEPT
    8    → far below floor, no boosts fire → UNCHANGED_REJECT
    9    → TRIM row, trim_profit_threshold annotation → threshold_utilization ticks
    """

    def _row(row_id: int, symbol: str, **over: Any) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "row_id": row_id,
            "symbol": symbol,
            "opportunity_type": "ENTRY",
            "effective_stance": "BUY",
            "sizing_bucket": "STANDARD",
            "confidence": 0.70,
            "edge_score": 0.55,
            "execution_blocked": False,
            "execution_block_reason": "",
            "post_stop_out_flag": "",
            "spread_bucket": "TIGHT",
            "quote_is_stale": False,
            "execution_risk_flag": "LOW",
            "execution_quality_score": 0.90,
        }
        base.update(over)
        return base

    return [
        # 1 — WIDE spread penalty should push edge 0.55 → 0.25 (< 0.50 floor)
        _row(1, "WIDE_1", spread_bucket="TOO_WIDE"),
        # 2 — STALE quote penalty: edge 0.55 → 0.25
        _row(2, "STALE_1", quote_is_stale=True),
        # 3 — LOW confidence penalty: confidence 0.40 triggers, edge 0.55 → 0.25
        _row(3, "LOWCONF_1", confidence=0.40),
        # 4 — BORDERLINE ADD near floor; add_score_threshold raises floor 0.50 → 0.65.
        #     Edge 0.52 (just above baseline floor) now below simulated floor.
        _row(4, "ADD_BORDER_1", opportunity_type="ADD", edge_score=0.52, spread_bucket="TIGHT"),
        # 5 — BOOST: baseline 0.20 < 0.50 rejected; HIGH_CONVICTION boost 0.35
        #     → simulated 0.55 ≥ 0.50 → NEWLY_ACCEPTED
        _row(5, "BOOST_1", sizing_bucket="HIGH_CONVICTION", edge_score=0.20, confidence=0.80),
        # 6 — BOOST: baseline 0.18 rejected; simulated 0.53 accepted
        _row(6, "BOOST_2", sizing_bucket="HIGH_CONVICTION", edge_score=0.18, confidence=0.80),
        # 7 — UNCHANGED_ACCEPT: no rule fires, edge 0.80 stays above floor
        _row(7, "CLEAN_1", edge_score=0.80),
        # 8 — UNCHANGED_REJECT: edge 0.10, STANDARD bucket (no boost fires)
        _row(8, "WEAK_1", edge_score=0.10),
        # 9 — TRIM annotation: trim_profit_threshold ticks threshold_utilization
        _row(9, "TRIM_1", opportunity_type="TRIM", effective_stance="SELL", edge_score=0.70),
    ]


def _stress_synthetic_instances() -> List[AdjustmentInstance]:
    """Six synthetic active adjustments that collectively exercise every
    effect type in the rule registry (penalty, boost, floor_raise, trim,
    annotate-only)."""

    def _inst(
        target: str,
        delta: float,
        *,
        related_bucket: str = "",
        related_flag: str = "",
        related_style: str = "",
    ) -> AdjustmentInstance:
        rule = RULE_REGISTRY[target]
        return AdjustmentInstance(
            application_id=f"APPLY-stress-{target}",
            proposal_id=f"ADAPT-stress-{target}",
            adaptation_target=target,
            effective_delta=float(delta),
            related_bucket=related_bucket,
            related_flag=related_flag,
            related_style=related_style,
            rule=rule,
        )

    return [
        _inst("wide_spread_entry_penalty", 0.30, related_bucket="TOO_WIDE"),
        _inst("stale_quote_penalty", 0.30, related_flag="STALE"),
        _inst("low_confidence_entry_penalty", 0.30, related_flag="LOW_CONF"),
        _inst("high_conviction_bucket_validation", 0.35),
        _inst("add_score_threshold", 0.15),
        _inst("trim_profit_threshold", -0.04),
    ]


def _stress_build_inputs(
    instances: List[AdjustmentInstance], synthetic_opps_rows: int
) -> SimInputs:
    """Fabricate a `SimInputs` that documents the stress-test data. The
    `applied` DataFrame mirrors the synthetic instances so the summary's
    `source_availability` block is honest about what the engine saw."""
    applied_rows: List[Dict[str, Any]] = []
    for inst in instances:
        applied_rows.append(
            {
                "application_id": inst.application_id,
                "proposal_id": inst.proposal_id,
                "adaptation_target": inst.adaptation_target,
                "effective_delta": inst.effective_delta,
                "related_bucket": inst.related_bucket,
                "related_flag": inst.related_flag,
                "related_style": inst.related_style,
                "status": "APPLIED",
                "active_flag": True,
                "source_file": "stress_test_harness",
                "source_status": "synthetic",
            }
        )
    applied = pd.DataFrame(applied_rows)
    opps = pd.DataFrame({"synthetic_row": [True] * synthetic_opps_rows})
    status = {
        "applied_adjustments_csv": "synthetic",
        "trade_opportunities_csv": "synthetic",
        "execution_intelligence_csv": "synthetic",
        "execution_plan_csv": "synthetic",
    }
    return SimInputs(
        applied=applied,
        opps=opps,
        exec_intel=pd.DataFrame(),
        exec_plan=pd.DataFrame(),
        status=status,
    )


def _write_stress_outputs(summary: Dict[str, Any], result: SimResult) -> Dict[str, str]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sim_df = _df_for_csv(result.rows)
    sim_df.to_csv(STRESS_SIM_CSV, index=False)
    STRESS_SIM_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {
        "stress_simulation_csv": str(STRESS_SIM_CSV),
        "stress_summary_json": str(STRESS_SIM_SUMMARY_JSON),
    }


def run_stress_test(*, verbose: bool = True) -> Dict[str, Any]:
    """Run the live rule engine against synthetic rows + adjustments.

    Returns a dict with:
        summary          → the summary JSON actually written
        result           → the SimResult with per-row classifications
        written          → paths of the stress-only output files
        assertions       → list of (ok, name, detail) tuples
        all_passed       → True iff every assertion passed
    """
    notes: List[str] = [
        "STRESS TEST run — synthetic data only. Live trade_opportunities.csv "
        "and applied_adjustments.csv were NOT read or modified.",
    ]
    contexts = _stress_synthetic_contexts()
    instances = _stress_synthetic_instances()
    inp = _stress_build_inputs(instances, synthetic_opps_rows=len(contexts))

    result = simulate(contexts, instances, score_floor=STRESS_SCORE_FLOOR)
    summary = build_summary(inp, result, score_floor=STRESS_SCORE_FLOOR, notes=notes)
    summary["stress_test"] = True
    written = _write_stress_outputs(summary, result)

    dc = summary["decision_change_counts"]
    tu = summary["threshold_utilization"]
    threshold_total = (
        int(tu.get("add_score_threshold_matches", 0) or 0)
        + int(tu.get("trim_profit_threshold_rows", 0) or 0)
        + int(tu.get("position_cooldown_bias_rows", 0) or 0)
    )
    unchanged_total = int(dc.get("UNCHANGED_ACCEPT", 0) or 0) + int(
        dc.get("UNCHANGED_REJECT", 0) or 0
    )
    assertions: List[Tuple[bool, str, str]] = [
        (dc.get("NEWLY_REJECTED", 0) >= 3, "NEWLY_REJECTED ≥ 3", f"got {dc.get('NEWLY_REJECTED')}"),
        (dc.get("NEWLY_ACCEPTED", 0) >= 2, "NEWLY_ACCEPTED ≥ 2", f"got {dc.get('NEWLY_ACCEPTED')}"),
        (unchanged_total >= 2, "UNCHANGED_ACCEPT + UNCHANGED_REJECT ≥ 2", f"got {unchanged_total}"),
        (
            threshold_total > 0,
            "threshold_utilization > 0",
            f"add={tu.get('add_score_threshold_matches')}, "
            f"trim={tu.get('trim_profit_threshold_rows')}, "
            f"cooldown={tu.get('position_cooldown_bias_rows')}",
        ),
        (
            summary.get("active_adjustments_used_count", 0) > 0,
            "at least one active adjustment actually fired",
            f"used={summary.get('active_adjustments_used_count')}/"
            f"{summary.get('active_adjustments_count')}",
        ),
    ]
    all_passed = all(ok for ok, _, _ in assertions)

    if verbose:
        print(
            "[stress_test] synthetic rows:",
            len(contexts),
            "| synthetic active adjustments:",
            len(instances),
        )
        print("[stress_test] decision_change_counts:", dc)
        print("[stress_test] threshold_utilization:", tu)
        print(
            "[stress_test] baseline:",
            summary["baseline_counts"],
            "| simulated:",
            summary["simulated_counts"],
        )
        print("[stress_test] exposure_delta_estimate:", summary.get("exposure_delta_estimate"))
        for ok, name, detail in assertions:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
        for key, path in written.items():
            print(f"[stress_test] {key}: {path}")
        print("[stress_test]", "ALL PASSED" if all_passed else "SOME FAILED")

    return {
        "summary": summary,
        "result": result,
        "written": written,
        "assertions": assertions,
        "all_passed": all_passed,
    }


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="adaptation_simulation",
        description=(
            "Phase-2 adaptation simulation — read-only. "
            "Writes data/results/adaptation_simulation.* only. "
            "Use --stress-test to run a synthetic adversarial "
            "check that writes only to *.stress_test.* files."
        ),
    )
    p.add_argument(
        "--score-floor",
        type=float,
        default=0.0,
        help="Baseline score floor below which rows are treated as rejected.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress stdout logging.")
    p.add_argument(
        "--stress-test",
        action="store_true",
        help=(
            "Run the synthetic stress-test harness (does NOT "
            "read or modify real simulation inputs or outputs)."
        ),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.stress_test:
        out = run_stress_test(verbose=not args.quiet)
        return 0 if out.get("all_passed") else 2
    out = run_simulation(score_floor=float(args.score_floor), verbose=not args.quiet)
    return 0 if out and out.get("written") else 1


if __name__ == "__main__":
    sys.exit(main())
