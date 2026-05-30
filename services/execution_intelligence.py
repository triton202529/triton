# services/execution_intelligence.py
"""
TRITON — Execution intelligence overlay (annotations only).

This module is a PURE helper layer that adds smart annotations on top of:
  - planned orders (BUY / ADD / TRIM / EXIT),
  - quotes (bid / ask / quote timestamp),
  - liquidity proxies (close * avg volume),
  - partial fills,
  - slippage diagnostics.

It DOES NOT:
  - touch broker integration / account auth code,
  - replace lifecycle decisions or hard risk caps,
  - place, cancel, or modify orders,
  - mutate any DataFrame in place.

Every public helper:
  - is safe to call with missing / NaN / malformed inputs,
  - returns a deterministic dict / dataclass with neutral fallbacks,
  - never raises (exceptions become "neutral fallback" annotations).

Allowed enums (all uppercase strings):
  spread_bucket    : TIGHT | NORMAL | WIDE | TOO_WIDE | UNKNOWN
  execution_style  : PASSIVE_LIMIT | NORMAL_LIMIT | AGGRESSIVE_LIMIT | DEFER | SKIP
  partial_fill_act : KEEP_WORKING | REPRICE | CANCEL | DEFER_REPRICE
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────


@dataclass
class ExecutionIntelligenceConfig:
    """All knobs for the v1 execution intelligence overlay."""

    # Quote freshness
    max_quote_age_sec: float = 120.0

    # Spread thresholds (in basis points of the mid)
    tight_spread_bps: float = 8.0
    normal_spread_bps: float = 20.0
    wide_spread_bps: float = 35.0
    too_wide_spread_bps: float = 75.0

    # Liquidity (order_notional / liquidity_proxy)
    max_notional_vs_liquidity: float = 0.02
    elevated_notional_vs_liquidity: float = 0.005

    # Partial-fill follow-up
    partial_fill_stale_minutes: float = 15.0
    repricing_fill_pct_min: float = 0.10  # consider "barely filled" below this
    mostly_filled_pct: float = 0.85

    # De-risking semantics — never let intelligence skip these lightly.
    de_risking_actions: Tuple[str, ...] = ("TRIM", "EXIT", "SELL")
    entry_actions: Tuple[str, ...] = ("BUY", "ADD", "ENTRY")


# Module-level enum strings (for callers that want symbolic constants).
SPREAD_TIGHT = "TIGHT"
SPREAD_NORMAL = "NORMAL"
SPREAD_WIDE = "WIDE"
SPREAD_TOO_WIDE = "TOO_WIDE"
SPREAD_UNKNOWN = "UNKNOWN"

STYLE_PASSIVE = "PASSIVE_LIMIT"
STYLE_NORMAL = "NORMAL_LIMIT"
STYLE_AGGRESSIVE = "AGGRESSIVE_LIMIT"
STYLE_DEFER = "DEFER"
STYLE_SKIP = "SKIP"

PFA_KEEP = "KEEP_WORKING"
PFA_REPRICE = "REPRICE"
PFA_CANCEL = "CANCEL"
PFA_DEFER = "DEFER_REPRICE"

RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"
RISK_UNKNOWN = "UNKNOWN"

NEUTRAL_FALLBACK_REASON = "neutral fallback"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _safe_pos_float(x: Any) -> Optional[float]:
    v = _safe_float(x, None)
    if v is None or v <= 0:
        return None
    return v


def _parse_ts(ts: Any) -> Optional[datetime]:
    """Best-effort: accept datetime, ISO strings (with/without Z), epoch seconds."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, (int, float)) and math.isfinite(float(ts)):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except Exception:
            return None
    s = str(ts).strip()
    if not s:
        return None
    try:
        # Tolerate trailing Z.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────
# Quote quality
# ─────────────────────────────────────────────────────────────


def evaluate_quote_quality(
    bid: Any,
    ask: Any,
    quote_ts: Any = None,
    cfg: Optional[ExecutionIntelligenceConfig] = None,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Inspect a top-of-book quote.

    Returns a dict with keys:
      mid, spread, spread_pct, spread_bps,
      quote_age_sec, quote_is_stale, spread_bucket, quote_reason
    Neutral fallbacks are emitted whenever inputs are missing / unusable.
    """
    cfg = cfg or ExecutionIntelligenceConfig()
    out: Dict[str, Any] = {
        "mid": None,
        "spread": None,
        "spread_pct": None,
        "spread_bps": None,
        "quote_age_sec": None,
        "quote_is_stale": True,  # be conservative: missing -> treat as stale
        "spread_bucket": SPREAD_UNKNOWN,
        "quote_reason": "no_quote",
    }

    b = _safe_pos_float(bid)
    a = _safe_pos_float(ask)
    if b is None or a is None or a < b:
        # Malformed / crossed / missing quote — leave neutral fallbacks.
        if b is None and a is None:
            out["quote_reason"] = "no_quote"
        elif a is not None and b is not None and a < b:
            out["quote_reason"] = "crossed_quote"
        else:
            out["quote_reason"] = "missing_side"
        return out

    mid = (a + b) / 2.0
    spread = a - b
    spread_pct = spread / mid if mid > 0 else None
    spread_bps = spread_pct * 10_000.0 if spread_pct is not None else None
    out["mid"] = float(mid)
    out["spread"] = float(spread)
    out["spread_pct"] = float(spread_pct) if spread_pct is not None else None
    out["spread_bps"] = float(round(spread_bps, 4)) if spread_bps is not None else None

    # Spread bucket
    if spread_bps is None:
        bucket = SPREAD_UNKNOWN
    elif spread_bps <= cfg.tight_spread_bps:
        bucket = SPREAD_TIGHT
    elif spread_bps <= cfg.normal_spread_bps:
        bucket = SPREAD_NORMAL
    elif spread_bps <= cfg.wide_spread_bps:
        bucket = SPREAD_WIDE
    elif spread_bps <= cfg.too_wide_spread_bps:
        bucket = SPREAD_WIDE  # still wide but not yet "too wide"
    else:
        bucket = SPREAD_TOO_WIDE
    # Anything above too_wide threshold is hard TOO_WIDE; the > wide && <= too_wide
    # band intentionally maps to WIDE so callers can react proportionally.
    if spread_bps is not None and spread_bps > cfg.too_wide_spread_bps:
        bucket = SPREAD_TOO_WIDE
    out["spread_bucket"] = bucket

    # Quote freshness
    qts = _parse_ts(quote_ts)
    if qts is not None:
        n = now or _now_utc()
        age = (n - qts).total_seconds()
        if age < 0:
            age = 0.0
        out["quote_age_sec"] = float(round(age, 3))
        out["quote_is_stale"] = bool(age > cfg.max_quote_age_sec)
        out["quote_reason"] = "fresh" if not out["quote_is_stale"] else "stale_quote"
    else:
        out["quote_reason"] = "no_quote_ts"
        out["quote_is_stale"] = True

    return out


# ─────────────────────────────────────────────────────────────
# Liquidity proxy
# ─────────────────────────────────────────────────────────────


def evaluate_liquidity(
    *,
    close: Any,
    avg_volume: Any = None,
    order_qty: Any = None,
    order_notional: Any = None,
    cfg: Optional[ExecutionIntelligenceConfig] = None,
) -> Dict[str, Any]:
    """
    Estimate liquidity proxy (close * avg_volume) and order_notional.
    Returns dict with: close, avg_volume, liquidity_proxy, order_notional,
    notional_vs_liquidity, liquidity_reason.

    All values default to None / 0 if inputs are missing.
    """
    cfg = cfg or ExecutionIntelligenceConfig()
    c = _safe_pos_float(close)
    v = _safe_pos_float(avg_volume)
    q = _safe_float(order_qty, None)
    notional = _safe_float(order_notional, None)

    liq = (c * v) if (c is not None and v is not None) else None
    if notional is None and q is not None and c is not None:
        notional = float(abs(q) * c)

    nvl: Optional[float] = None
    if liq is not None and liq > 0 and notional is not None and notional >= 0:
        nvl = float(notional / liq)

    reason = "ok"
    if liq is None:
        reason = "no_liquidity_proxy"
    elif notional is None:
        reason = "no_notional"
    elif nvl is not None and nvl > cfg.max_notional_vs_liquidity:
        reason = "large_notional_vs_liquidity"
    elif nvl is not None and nvl > cfg.elevated_notional_vs_liquidity:
        reason = "elevated_notional_vs_liquidity"

    return {
        "close": float(c) if c is not None else None,
        "avg_volume": float(v) if v is not None else None,
        "liquidity_proxy": float(liq) if liq is not None else None,
        "order_notional": float(notional) if notional is not None else None,
        "notional_vs_liquidity": float(round(nvl, 8)) if nvl is not None else None,
        "liquidity_reason": reason,
    }


# ─────────────────────────────────────────────────────────────
# Execution style recommendation
# ─────────────────────────────────────────────────────────────


def _action_class(action: Any, cfg: ExecutionIntelligenceConfig) -> str:
    a = str(action or "").strip().upper()
    if a in cfg.de_risking_actions:
        return "DE_RISKING"
    if a in cfg.entry_actions:
        return "ENTRY"
    return "OTHER"


def recommend_execution_style(
    *,
    action: Any,
    quote: Optional[Dict[str, Any]] = None,
    liquidity: Optional[Dict[str, Any]] = None,
    cfg: Optional[ExecutionIntelligenceConfig] = None,
) -> Dict[str, Any]:
    """
    Combine quote + liquidity + action class into an execution recommendation.

    Returns dict with: execution_style, execution_aggressiveness (1-5),
    execution_reason, execution_skip_flag, execution_skip_reason.
    """
    cfg = cfg or ExecutionIntelligenceConfig()
    quote = quote or {}
    liquidity = liquidity or {}
    klass = _action_class(action, cfg)

    bucket = str(quote.get("spread_bucket") or SPREAD_UNKNOWN).strip().upper()
    # Distinguish "stale live quote" from "no quote was ever provided".
    # Planning-stage callers won't have a quote at all and must emit a
    # neutral NORMAL_LIMIT recommendation, not a DEFER.
    has_quote_ts = quote.get("quote_age_sec") is not None
    has_quote_pricing = quote.get("mid") is not None or quote.get("spread_bps") is not None
    quote_provided = bool(has_quote_ts or has_quote_pricing)
    is_stale = bool(quote.get("quote_is_stale", True)) and quote_provided
    nvl = _safe_float(liquidity.get("notional_vs_liquidity"), None)

    # Defaults — neutral.
    style = STYLE_NORMAL
    aggressiveness = 3
    reasons: list[str] = []
    skip = False
    skip_reason = ""

    # ── Quote quality drives base style ────────────────────────────────────
    if bucket == SPREAD_UNKNOWN:
        style = STYLE_NORMAL
        reasons.append(NEUTRAL_FALLBACK_REASON + ":no_spread")
    elif bucket == SPREAD_TIGHT:
        style = STYLE_AGGRESSIVE
        aggressiveness = 4
        reasons.append("tight_spread")
    elif bucket == SPREAD_NORMAL:
        style = STYLE_NORMAL
        aggressiveness = 3
        reasons.append("normal_spread")
    elif bucket == SPREAD_WIDE:
        style = STYLE_PASSIVE
        aggressiveness = 2
        reasons.append("wide_spread")
    elif bucket == SPREAD_TOO_WIDE:
        if klass == "DE_RISKING":
            # De-risking should still execute — go passive but don't skip.
            style = STYLE_PASSIVE
            aggressiveness = 2
            reasons.append("too_wide_spread_but_de_risking")
        else:
            style = STYLE_DEFER
            aggressiveness = 1
            reasons.append("too_wide_spread")

    # ── Liquidity tilt ─────────────────────────────────────────────────────
    if nvl is not None:
        if nvl > cfg.max_notional_vs_liquidity:
            if klass == "DE_RISKING":
                style = _downgrade_style(style, STYLE_PASSIVE)
                aggressiveness = min(aggressiveness, 2)
                reasons.append("large_notional_vs_liquidity_but_de_risking")
            else:
                style = STYLE_DEFER
                aggressiveness = 1
                reasons.append("large_notional_vs_liquidity")
        elif nvl > cfg.elevated_notional_vs_liquidity:
            style = _downgrade_style(style, STYLE_PASSIVE)
            aggressiveness = min(aggressiveness, 2)
            reasons.append("elevated_notional_vs_liquidity")

    # ── Stale quote handling ───────────────────────────────────────────────
    if is_stale:
        if klass == "DE_RISKING":
            # Don't lightly block exits — annotate but stay PASSIVE/NORMAL.
            style = _downgrade_style(style, STYLE_PASSIVE)
            aggressiveness = min(aggressiveness, 2)
            reasons.append("stale_quote_de_risking_annotated")
        else:
            style = STYLE_DEFER
            aggressiveness = 1
            reasons.append("stale_quote_entry")

    # ── Hard skip cases ────────────────────────────────────────────────────
    if style == STYLE_DEFER and klass != "DE_RISKING":
        # Truly unusable data combined with entry action -> SKIP.
        if bucket == SPREAD_TOO_WIDE and is_stale:
            style = STYLE_SKIP
            skip = True
            skip_reason = "too_wide_spread+stale_quote"

    if not reasons:
        reasons.append(NEUTRAL_FALLBACK_REASON)

    return {
        "execution_style": style,
        "execution_aggressiveness": int(aggressiveness),
        "execution_reason": "+".join(reasons),
        "execution_skip_flag": bool(skip),
        "execution_skip_reason": skip_reason,
    }


def _downgrade_style(current: str, floor_style: str) -> str:
    """Return the more passive of `current` and `floor_style`."""
    order = [STYLE_AGGRESSIVE, STYLE_NORMAL, STYLE_PASSIVE, STYLE_DEFER, STYLE_SKIP]
    try:
        ci = order.index(current)
        fi = order.index(floor_style)
        return order[max(ci, fi)]
    except ValueError:
        return floor_style


# ─────────────────────────────────────────────────────────────
# Partial fill intelligence
# ─────────────────────────────────────────────────────────────


def recommend_partial_fill_action(
    *,
    filled_qty: Any,
    total_qty: Any,
    order_age_minutes: Any,
    quote: Optional[Dict[str, Any]] = None,
    cfg: Optional[ExecutionIntelligenceConfig] = None,
) -> Dict[str, Any]:
    """
    Decide what to do with an open / partially-filled working order.

    Returns dict with: fill_pct, partial_fill_action, partial_fill_reason.
    """
    cfg = cfg or ExecutionIntelligenceConfig()
    quote = quote or {}

    f = _safe_float(filled_qty, 0.0) or 0.0
    t = _safe_float(total_qty, 0.0) or 0.0
    age = _safe_float(order_age_minutes, None)

    if t <= 0:
        return {
            "fill_pct": 0.0,
            "partial_fill_action": PFA_KEEP,
            "partial_fill_reason": NEUTRAL_FALLBACK_REASON + ":no_total_qty",
        }
    fill_pct = max(0.0, min(1.0, float(f) / float(t)))

    bucket = str(quote.get("spread_bucket") or SPREAD_UNKNOWN).strip().upper()
    is_stale = bool(quote.get("quote_is_stale", True))

    # Mostly filled -> just let it work.
    if fill_pct >= cfg.mostly_filled_pct:
        return {
            "fill_pct": float(round(fill_pct, 6)),
            "partial_fill_action": PFA_KEEP,
            "partial_fill_reason": "mostly_filled",
        }

    # Very recent order -> let it breathe.
    if age is not None and age < cfg.partial_fill_stale_minutes:
        return {
            "fill_pct": float(round(fill_pct, 6)),
            "partial_fill_action": PFA_KEEP,
            "partial_fill_reason": "still_within_stale_window",
        }

    # Stale quote -> defer reprice (don't chase ghost prices).
    if is_stale:
        return {
            "fill_pct": float(round(fill_pct, 6)),
            "partial_fill_action": PFA_DEFER,
            "partial_fill_reason": "stale_quote_defer_reprice",
        }

    # Spread far too wide -> defer; chasing wastes edge.
    if bucket == SPREAD_TOO_WIDE:
        return {
            "fill_pct": float(round(fill_pct, 6)),
            "partial_fill_action": PFA_DEFER,
            "partial_fill_reason": "spread_too_wide",
        }

    # Wide spread + barely filled -> keep working rather than chase.
    if bucket == SPREAD_WIDE and fill_pct < cfg.repricing_fill_pct_min:
        return {
            "fill_pct": float(round(fill_pct, 6)),
            "partial_fill_action": PFA_KEEP,
            "partial_fill_reason": "wide_spread_low_fill",
        }

    # Otherwise we're stale + fresh quote + reasonable spread -> reprice.
    return {
        "fill_pct": float(round(fill_pct, 6)),
        "partial_fill_action": PFA_REPRICE,
        "partial_fill_reason": "stale_with_fresh_quote",
    }


# ─────────────────────────────────────────────────────────────
# Slippage diagnostics
# ─────────────────────────────────────────────────────────────


def compute_slippage_diagnostics(
    *,
    side: Any,
    intended_price: Any = None,
    submitted_limit_price: Any = None,
    fill_price: Any = None,
    decision_mid_price: Any = None,
) -> Dict[str, Any]:
    """
    Compute additive slippage fields. Missing inputs -> None (never NaN).

    Returns dict with: side, intended_price, submitted_limit_price, fill_price,
    decision_mid_price, expected_slippage_bps, realized_slippage_bps,
    slippage_reason.
    """
    s = str(side or "").strip().upper()
    sign = 1.0 if s == "BUY" else (-1.0 if s == "SELL" else 0.0)

    ip = _safe_pos_float(intended_price)
    sp = _safe_pos_float(submitted_limit_price)
    fp = _safe_pos_float(fill_price)
    dm = _safe_pos_float(decision_mid_price)

    expected_bps: Optional[float] = None
    realized_bps: Optional[float] = None

    # Expected: limit price vs decision mid; positive bps == paying up (BUY)
    # or selling lower than mid (SELL). Sign convention is "cost to us".
    if dm is not None and sp is not None and dm > 0 and sign != 0.0:
        diff = (sp - dm) * sign
        expected_bps = float(round(diff / dm * 10_000.0, 4))

    if dm is not None and fp is not None and dm > 0 and sign != 0.0:
        diff = (fp - dm) * sign
        realized_bps = float(round(diff / dm * 10_000.0, 4))

    reason = "ok"
    if dm is None:
        reason = "no_decision_mid"
    elif sp is None and fp is None:
        reason = "no_prices"
    elif sign == 0.0:
        reason = "unknown_side"

    return {
        "side": s or "",
        "intended_price": float(ip) if ip is not None else None,
        "submitted_limit_price": float(sp) if sp is not None else None,
        "fill_price": float(fp) if fp is not None else None,
        "decision_mid_price": float(dm) if dm is not None else None,
        "expected_slippage_bps": expected_bps,
        "realized_slippage_bps": realized_bps,
        "slippage_reason": reason,
    }


# ─────────────────────────────────────────────────────────────
# Execution quality score + risk flag (diagnostics only)
# ─────────────────────────────────────────────────────────────

# Penalty tables — kept as module-level constants so callers can introspect.
_SPREAD_PENALTY: Dict[str, float] = {
    SPREAD_TIGHT: 0.00,
    SPREAD_NORMAL: 0.05,
    SPREAD_WIDE: 0.20,
    SPREAD_TOO_WIDE: 0.40,
    SPREAD_UNKNOWN: 0.10,
}

_STYLE_PENALTY: Dict[str, float] = {
    STYLE_AGGRESSIVE: 0.00,
    STYLE_NORMAL: 0.02,
    STYLE_PASSIVE: 0.08,
    STYLE_DEFER: 0.25,
    STYLE_SKIP: 0.40,
}

_STALE_PENALTY = 0.35
_NVL_PENALTY_MILD = 0.20  # > 0.02
_NVL_PENALTY_HEAVY = 0.35  # > 0.05 (replaces, not stacks)
_SKIP_FLAG_PENALTY = 0.20

# Risk thresholds.
_RISK_THRESHOLD_LOW = 0.75
_RISK_THRESHOLD_MEDIUM = 0.45


def compute_execution_quality_score(
    *,
    quote: Optional[Dict[str, Any]] = None,
    liquidity: Optional[Dict[str, Any]] = None,
    style: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute a compact diagnostic execution-quality score in [0, 1].

    Inputs are the same dicts produced by `evaluate_quote_quality`,
    `evaluate_liquidity` and `recommend_execution_style`. Any of them
    can be None / missing — the function never raises and always returns
    a deterministic dict.

    Returns:
        {
          "execution_quality_score": float in [0, 1],
          "execution_quality_reason": "+"-joined penalty tags,
          "execution_quality_used_fallback": bool,
        }
    """
    quote = quote or {}
    liquidity = liquidity or {}
    style = style or {}

    score = 1.0
    reasons: list[str] = []

    # Distinguish "no quote was provided" from "live quote went stale".
    has_quote_ts = quote.get("quote_age_sec") is not None
    has_quote_pricing = quote.get("mid") is not None or quote.get("spread_bps") is not None
    quote_provided = bool(has_quote_ts or has_quote_pricing)

    nvl = _safe_float(liquidity.get("notional_vs_liquidity"), None)
    bucket_raw = quote.get("spread_bucket")
    bucket = str(bucket_raw or SPREAD_UNKNOWN).strip().upper()
    style_raw = style.get("execution_style")
    style_str = str(style_raw or STYLE_NORMAL).strip().upper()
    skip_flag = bool(style.get("execution_skip_flag", False))
    is_stale = bool(quote.get("quote_is_stale", False))

    # ── Fallback shortcut ───────────────────────────────────────────────
    # If we have neither quote info nor liquidity info, the score is not
    # informative; emit a deterministic neutral 0.50.
    if (not quote_provided) and (nvl is None) and (bucket == SPREAD_UNKNOWN):
        return {
            "execution_quality_score": 0.50,
            "execution_quality_reason": NEUTRAL_FALLBACK_REASON,
            "execution_quality_used_fallback": True,
        }

    # ── Stale quote (only if a live quote was actually provided) ────────
    if quote_provided and is_stale:
        score -= _STALE_PENALTY
        reasons.append("stale_quote")

    # ── Spread bucket penalty ───────────────────────────────────────────
    sp = _SPREAD_PENALTY.get(bucket, _SPREAD_PENALTY[SPREAD_UNKNOWN])
    if sp > 0:
        score -= sp
        reasons.append(f"spread_{bucket.lower()}")

    # ── Liquidity pressure (heavy replaces mild — not stacked) ──────────
    if nvl is not None:
        if nvl > 0.05:
            score -= _NVL_PENALTY_HEAVY
            reasons.append("nvl_heavy")
        elif nvl > 0.02:
            score -= _NVL_PENALTY_MILD
            reasons.append("nvl_mild")

    # ── Execution style penalty ─────────────────────────────────────────
    stp = _STYLE_PENALTY.get(style_str, _STYLE_PENALTY[STYLE_NORMAL])
    if stp > 0:
        score -= stp
        reasons.append(f"style_{style_str.lower()}")

    # ── Hard skip flag adds extra penalty ───────────────────────────────
    if skip_flag:
        score -= _SKIP_FLAG_PENALTY
        reasons.append("skip_flag")

    # Clamp.
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    return {
        "execution_quality_score": float(round(score, 4)),
        "execution_quality_reason": "+".join(reasons) if reasons else "clean",
        "execution_quality_used_fallback": False,
    }


def classify_execution_risk_flag(score: Any, *, used_fallback: bool = False) -> str:
    """
    Map an execution-quality score to a categorical risk flag.

    - used_fallback=True (no quote AND no liquidity info) -> UNKNOWN
    - score >= 0.75 -> LOW
    - score >= 0.45 -> MEDIUM
    - score <  0.45 -> HIGH
    - non-numeric / NaN score -> UNKNOWN
    """
    if used_fallback:
        return RISK_UNKNOWN
    s = _safe_float(score, None)
    if s is None:
        return RISK_UNKNOWN
    if s >= _RISK_THRESHOLD_LOW:
        return RISK_LOW
    if s >= _RISK_THRESHOLD_MEDIUM:
        return RISK_MEDIUM
    return RISK_HIGH


# ─────────────────────────────────────────────────────────────
# Convenience: one-shot annotation for a planned/placed order
# ─────────────────────────────────────────────────────────────


def annotate_order(
    *,
    action: Any,
    side: Any = None,
    bid: Any = None,
    ask: Any = None,
    quote_ts: Any = None,
    close: Any = None,
    avg_volume: Any = None,
    order_qty: Any = None,
    order_notional: Any = None,
    intended_price: Any = None,
    submitted_limit_price: Any = None,
    fill_price: Any = None,
    decision_mid_price: Any = None,
    cfg: Optional[ExecutionIntelligenceConfig] = None,
) -> Dict[str, Any]:
    """
    Compute the full execution-intelligence column set for a single order.

    Returns a flat dict ready to merge into a planning / log row. All keys
    are always present (None where data is missing) so downstream writers can
    rely on a stable schema.
    """
    cfg = cfg or ExecutionIntelligenceConfig()

    quote = evaluate_quote_quality(bid, ask, quote_ts, cfg)
    liq = evaluate_liquidity(
        close=close,
        avg_volume=avg_volume,
        order_qty=order_qty,
        order_notional=order_notional,
        cfg=cfg,
    )
    style = recommend_execution_style(action=action, quote=quote, liquidity=liq, cfg=cfg)

    # If decision_mid_price not provided but we have a quote mid, use it.
    if decision_mid_price is None and quote.get("mid") is not None:
        decision_mid_price = quote.get("mid")
    slip = compute_slippage_diagnostics(
        side=side or action,
        intended_price=intended_price,
        submitted_limit_price=submitted_limit_price,
        fill_price=fill_price,
        decision_mid_price=decision_mid_price,
    )
    quality = compute_execution_quality_score(quote=quote, liquidity=liq, style=style)
    risk_flag = classify_execution_risk_flag(
        quality.get("execution_quality_score"),
        used_fallback=bool(quality.get("execution_quality_used_fallback", False)),
    )

    out: Dict[str, Any] = {
        "bid": _safe_float(bid, None),
        "ask": _safe_float(ask, None),
        "quote_mid": quote.get("mid"),
        "spread_pct": quote.get("spread_pct"),
        "spread_bps": quote.get("spread_bps"),
        "quote_age_sec": quote.get("quote_age_sec"),
        "quote_is_stale": quote.get("quote_is_stale"),
        "spread_bucket": quote.get("spread_bucket"),
        "quote_reason": quote.get("quote_reason"),
        "close": liq.get("close"),
        "avg_volume": liq.get("avg_volume"),
        "liquidity_proxy": liq.get("liquidity_proxy"),
        "order_notional": liq.get("order_notional"),
        "notional_vs_liquidity": liq.get("notional_vs_liquidity"),
        "liquidity_reason": liq.get("liquidity_reason"),
        "execution_style": style.get("execution_style"),
        "execution_aggressiveness": style.get("execution_aggressiveness"),
        "execution_reason": style.get("execution_reason"),
        "execution_skip_flag": style.get("execution_skip_flag"),
        "execution_skip_reason": style.get("execution_skip_reason"),
        "intended_price": slip.get("intended_price"),
        "submitted_limit_price": slip.get("submitted_limit_price"),
        "fill_price": slip.get("fill_price"),
        "decision_mid_price": slip.get("decision_mid_price"),
        "expected_slippage_bps": slip.get("expected_slippage_bps"),
        "realized_slippage_bps": slip.get("realized_slippage_bps"),
        "slippage_reason": slip.get("slippage_reason"),
        "execution_quality_score": quality.get("execution_quality_score"),
        "execution_quality_reason": quality.get("execution_quality_reason"),
        "execution_risk_flag": risk_flag,
    }
    return out


# Stable list of keys produced by annotate_order(); useful for CSV writers
# that want a deterministic column order.
ANNOTATE_ORDER_KEYS: Tuple[str, ...] = (
    "bid",
    "ask",
    "quote_mid",
    "spread_pct",
    "spread_bps",
    "quote_age_sec",
    "quote_is_stale",
    "spread_bucket",
    "quote_reason",
    "close",
    "avg_volume",
    "liquidity_proxy",
    "order_notional",
    "notional_vs_liquidity",
    "liquidity_reason",
    "execution_style",
    "execution_aggressiveness",
    "execution_reason",
    "execution_skip_flag",
    "execution_skip_reason",
    "intended_price",
    "submitted_limit_price",
    "fill_price",
    "decision_mid_price",
    "expected_slippage_bps",
    "realized_slippage_bps",
    "slippage_reason",
    "execution_quality_score",
    "execution_quality_reason",
    "execution_risk_flag",
)


# ─────────────────────────────────────────────────────────────
# Symbol-coverage completion
#
# `execution_intelligence.csv` is written as a per-order sidecar log by
# services/place_live_orders.py — it only receives rows for symbols that
# actually had a live order submitted. That leaves most opportunity
# symbols missing from the file, which in turn prevents downstream
# observability layers (build_trade_opportunities, adaptation_simulation)
# from seeing any execution context for them.
#
# This block fixes the COVERAGE problem without touching:
#   - quote / liquidity / style / quality scoring logic above,
#   - any execution thresholds,
#   - any broker / order-placement behavior.
#
# It augments `execution_intelligence.csv` with a "fallback" row for every
# opportunity symbol currently absent from the file. Each fallback row
# carries explicit UNKNOWN markers plus `quote_is_stale=True` and
# `quote_reason="missing_execution_data"` so downstream consumers can
# tell a synthetic placeholder apart from a real order-intelligence row.
#
# Safety properties:
#   * Pure additive — existing rows are never modified or reordered.
#   * Fallback rows are APPENDED to the end of the file so any future
#     real rows (added later by place_live_orders.py) will appear after
#     the fallback and win `drop_duplicates(..., keep="last")` dedup in
#     downstream loaders.
#   * Idempotent — re-running the coverage step does not duplicate
#     fallback rows: a symbol already present in the file (real OR
#     previously-added fallback) is skipped.
#   * Never raises: filesystem / parse errors degrade to a no-op.
#   * Never touches trading behavior, planner, lifecycle, or risk rules.
# ─────────────────────────────────────────────────────────────

try:  # pandas is an optional dep for library users; required for coverage.
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover — library use without pandas still works
    _pd = None  # type: ignore


_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTION_INTELLIGENCE_CSV = _ROOT / "data" / "results" / "execution_intelligence.csv"
DEFAULT_TRADE_OPPORTUNITIES_CSV = _ROOT / "data" / "results" / "trade_opportunities.csv"

COVERAGE_FALLBACK_TAG = "coverage_fallback"
COVERAGE_QUOTE_REASON = "missing_execution_data"

# Columns the fallback row is required to populate. Spec-mandated keys
# plus a couple of already-existing sidecar columns so the row is
# readable without special casing by downstream loaders.
_COVERAGE_REQUIRED_COLS: Tuple[str, ...] = (
    "symbol",
    "spread_bucket",
    "spread_bps",
    "spread_pct",
    "execution_risk_flag",
    "execution_style",
    "quote_is_stale",
    "quote_reason",
    "liquidity_pressure_bucket",
)


def _uc(x: Any) -> str:
    """Safe uppercase-strip; returns '' for None / NaN / blank."""
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    return s.upper()


def _read_opportunity_symbols(opps_csv: Path) -> List[str]:
    """Return the uppercased unique symbol list from trade_opportunities.csv.

    Tolerates: missing file, empty file, unreadable file, or a file with
    neither `symbol` nor `ticker` — in which case returns [] and the
    caller reports zero coverage work to do.
    """
    if _pd is None or not opps_csv.exists() or opps_csv.stat().st_size == 0:
        return []
    try:
        df = _pd.read_csv(opps_csv)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    col = None
    for c in ("symbol", "ticker"):
        if c in df.columns:
            col = c
            break
    if col is None:
        return []
    syms = (
        df[col]
        .astype(str)
        .map(lambda s: s.strip().upper())
        .replace({"": None, "NAN": None, "NONE": None, "NULL": None})
        .dropna()
        .unique()
        .tolist()
    )
    return sorted([s for s in syms if s])


def _read_existing_ei(ei_csv: Path) -> Optional["_pd.DataFrame"]:
    """Read the existing sidecar CSV, returning None when unreadable/empty."""
    if _pd is None or not ei_csv.exists() or ei_csv.stat().st_size == 0:
        return None
    try:
        df = _pd.read_csv(ei_csv)
    except Exception:
        return None
    if df is None or df.empty:
        return df  # empty but readable — caller handles
    return df


def _build_fallback_row(symbol: str, columns: List[str]) -> Dict[str, Any]:
    """Construct one fallback row conforming to the file's column set."""
    row: Dict[str, Any] = {c: "" for c in columns}
    if "timestamp" in columns:
        row["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if "session" in columns:
        row["session"] = COVERAGE_FALLBACK_TAG
    if "action" in columns:
        row["action"] = ""
    row["symbol"] = symbol
    row["spread_bucket"] = SPREAD_UNKNOWN
    row["spread_bps"] = ""
    row["spread_pct"] = ""
    row["execution_risk_flag"] = RISK_UNKNOWN
    row["execution_style"] = "UNKNOWN"
    row["quote_is_stale"] = True
    row["quote_reason"] = COVERAGE_QUOTE_REASON
    row["liquidity_pressure_bucket"] = "UNKNOWN"
    return row


def ensure_symbol_coverage(
    *,
    opps_csv: Optional[Path] = None,
    ei_csv: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Ensure every symbol in `trade_opportunities.csv` has at least one row
    in `execution_intelligence.csv`. Missing symbols get a deterministic
    fallback row with UNKNOWN markers appended to the end of the file.

    Returns a diagnostics dict:
        {
          "total_symbols":    int,  # unique symbols in trade_opportunities
          "covered_symbols":  int,  # present in EI before this call
          "fallback_symbols": int,  # rows added by this call
          "fallback_list":    List[str],
          "ei_csv":           str,
          "opps_csv":         str,
          "status":           "ok" | "skipped:<reason>",
        }

    Never raises. If pandas is unavailable, or either file is unreadable,
    returns a diagnostic dict with an appropriate `status` value and takes
    no action.
    """
    opps_path = Path(opps_csv) if opps_csv else DEFAULT_TRADE_OPPORTUNITIES_CSV
    ei_path = Path(ei_csv) if ei_csv else DEFAULT_EXECUTION_INTELLIGENCE_CSV

    diag: Dict[str, Any] = {
        "total_symbols": 0,
        "covered_symbols": 0,
        "fallback_symbols": 0,
        "fallback_list": [],
        "ei_csv": str(ei_path),
        "opps_csv": str(opps_path),
        "status": "ok",
    }

    if _pd is None:
        diag["status"] = "skipped:pandas_unavailable"
        if verbose:
            _log_coverage(diag)
        return diag

    opp_symbols = _read_opportunity_symbols(opps_path)
    diag["total_symbols"] = len(opp_symbols)
    if not opp_symbols:
        diag["status"] = "skipped:no_opportunities"
        if verbose:
            _log_coverage(diag)
        return diag

    existing = _read_existing_ei(ei_path)
    existing_symbols: set[str] = set()
    existing_cols: List[str] = list(EI_FALLBACK_HEADER)
    if existing is not None and not existing.empty:
        if "symbol" in existing.columns:
            existing_symbols = {_uc(v) for v in existing["symbol"].astype(str).tolist()}
            existing_symbols.discard("")
        # Preserve the existing file's column ordering verbatim, then
        # append any required fallback columns that are missing, at the
        # END (safe for subsequent line-oriented appends by
        # place_live_orders.py which writes a fixed EI_FIELDS schema).
        existing_cols = list(existing.columns)
    for req in _COVERAGE_REQUIRED_COLS:
        if req not in existing_cols:
            existing_cols.append(req)

    diag["covered_symbols"] = len(existing_symbols & set(opp_symbols))

    missing = [s for s in opp_symbols if s not in existing_symbols]
    diag["fallback_symbols"] = len(missing)
    diag["fallback_list"] = list(missing)

    if not missing:
        diag["status"] = "ok:already_fully_covered"
        if verbose:
            _log_coverage(diag)
        return diag

    try:
        fallback_rows = [_build_fallback_row(sym, existing_cols) for sym in missing]
        fallback_df = _pd.DataFrame(fallback_rows, columns=existing_cols)

        if existing is None or existing.empty:
            # Fresh file — write header + fallbacks.
            ei_path.parent.mkdir(parents=True, exist_ok=True)
            fallback_df.to_csv(ei_path, index=False)
        else:
            # Extend existing with any new columns, then append rows at
            # the TAIL so real rows added later win `keep="last"` dedup.
            for c in existing_cols:
                if c not in existing.columns:
                    existing[c] = ""
            existing = existing[existing_cols]
            combined = _pd.concat([existing, fallback_df], ignore_index=True)
            ei_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(ei_path, index=False)
    except Exception as exc:  # pragma: no cover — defensive
        diag["status"] = f"skipped:write_failed:{exc!r}"
        diag["fallback_symbols"] = 0
        diag["fallback_list"] = []
        if verbose:
            _log_coverage(diag)
        return diag

    if verbose:
        _log_coverage(diag)
    return diag


# Minimal header used when execution_intelligence.csv does not exist yet.
# Mirrors the EI_FIELDS shape produced by place_live_orders.py plus the
# coverage-required columns, so the first-ever writeout is well-formed.
EI_FALLBACK_HEADER: Tuple[str, ...] = (
    (
        "timestamp",
        "session",
        "action",
        "symbol",
        "side",
        "qty",
        "order_id",
        "client_order_id",
        "status",
    )
    + ANNOTATE_ORDER_KEYS
    + ("liquidity_pressure_bucket",)
)


def _log_coverage(diag: Dict[str, Any]) -> None:
    total = diag.get("total_symbols", 0)
    covered = diag.get("covered_symbols", 0)
    fallback = diag.get("fallback_symbols", 0)
    status = diag.get("status", "ok")
    # Spec-mandated log format.
    print(
        "[execution_intelligence] coverage: "
        f"total_symbols={total} covered_symbols={covered} "
        f"fallback_symbols={fallback} status={status}"
    )


# ─────────────────────────────────────────────────────────────
# CLI entrypoint — `python -m services.execution_intelligence`
# ─────────────────────────────────────────────────────────────


def _parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="services.execution_intelligence",
        description=(
            "Ensure execution_intelligence.csv carries at least one row "
            "per symbol in trade_opportunities.csv. Missing symbols are "
            "populated with UNKNOWN/missing_execution_data fallback rows "
            "so downstream observability (adaptation_simulation, etc.) "
            "has a stable schema to read from. This script NEVER touches "
            "broker, order, lifecycle, or scoring logic."
        ),
    )
    p.add_argument(
        "--ei-csv",
        type=str,
        default=str(DEFAULT_EXECUTION_INTELLIGENCE_CSV),
        help="Path to execution_intelligence.csv (default: data/results/...).",
    )
    p.add_argument(
        "--opps-csv",
        type=str,
        default=str(DEFAULT_TRADE_OPPORTUNITIES_CSV),
        help="Path to trade_opportunities.csv (default: data/results/...).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress stdout log.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_cli(argv)
    diag = ensure_symbol_coverage(
        opps_csv=Path(args.opps_csv),
        ei_csv=Path(args.ei_csv),
        verbose=not args.quiet,
    )
    return 0 if str(diag.get("status", "")).startswith("ok") else 0


if __name__ == "__main__":
    sys.exit(main())
