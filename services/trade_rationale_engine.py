"""
Trade Rationale & Explainability Engine — Step 8 of the WATCH → DEPLOY funnel.

Reads:
    data/results/portfolio_execution_intents.csv
    data/results/opportunity_persistence_recommendations.csv
    data/results/portfolio_rebalance_plan.csv
    data/results/performance_risk_overlay.csv
    data/results/signals_with_rationale.csv
    data/results/signal_lifecycle_effective.csv

Writes:
    data/results/trade_rationale.csv
    data/results/trade_rationale_summary.json

Purpose
-------
Step 7 emitted an `execution_intent` for every rebalance row. This
engine generates the *human-facing* layer on top of that: one
institutional-quality explanation per ticker, with one-line and
paragraph forms, machine-readable tags, calibrated labels, and an
explanation_score that aggregates how well the recommendation is
substantiated by the underlying evidence.

It answers:

    "Why is Triton recommending this action?"

This is the layer the investment committee reads — every line of every
rationale is fully traceable back to the underlying signal, persistence
metric, lifecycle action, or risk flag that produced it.

Safety
------
* Read-only. No orders, no broker calls, no mutation of execute_trades
  or manage_positions. The output is text only.
* Missing inputs warn and continue (empty enrichment maps).
* Atomic writes via `*.tmp` + `os.replace`.
* main() returns 0 on success, 2 only on output-write failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

# Reuse the canonical ticker→sector mapping for the diversification
# context paragraph — single source of truth for sector labels.
try:
    from services.sector_exposure import get_sector, UNKNOWN_SECTOR_LABEL  # type: ignore
except Exception:  # pragma: no cover
    UNKNOWN_SECTOR_LABEL = "Unknown"

    def get_sector(symbol: str) -> str:  # type: ignore[misc]
        return UNKNOWN_SECTOR_LABEL


# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_INTENTS_CSV = RESULTS_DIR / "portfolio_execution_intents.csv"
DEFAULT_PERSISTENCE_CSV = RESULTS_DIR / "opportunity_persistence_recommendations.csv"
DEFAULT_REBALANCE_CSV = RESULTS_DIR / "portfolio_rebalance_plan.csv"
DEFAULT_RISK_OVERLAY_CSV = RESULTS_DIR / "performance_risk_overlay.csv"
DEFAULT_SIGNALS_CSV = RESULTS_DIR / "signals_with_rationale.csv"
DEFAULT_LIFECYCLE_EFFECTIVE_CSV = RESULTS_DIR / "signal_lifecycle_effective.csv"

DEFAULT_OUTPUT_CSV = RESULTS_DIR / "trade_rationale.csv"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "trade_rationale_summary.json"

# -----------------------------------------------------------
# Labels & tunables
# -----------------------------------------------------------
INTENT_EXECUTE = "EXECUTE_NOW"
INTENT_DELAY = "DELAY"
INTENT_SKIP = "SKIP"
INTENT_BLOCK = "BLOCK"

REBAL_BUY_NEW = "BUY_NEW"
REBAL_ADD = "ADD"
REBAL_TRIM = "TRIM"
REBAL_SELL = "SELL"
REBAL_FULL_EXIT = "FULL_EXIT"
REBAL_HOLD = "HOLD"
REBAL_NO_ACTION = "NO_ACTION"

BUY_DIRECTION_ACTIONS: frozenset = frozenset({REBAL_BUY_NEW, REBAL_ADD})
SELL_DIRECTION_ACTIONS: frozenset = frozenset({REBAL_FULL_EXIT, REBAL_TRIM, REBAL_SELL})

POSITIVE_LABELS: frozenset = frozenset({"BUY", "ADD", "LONG"})
NEGATIVE_LABELS: frozenset = frozenset({"SELL", "EXIT", "REDUCE", "SHORT", "DUMP"})

RISK_FORCE_EXIT = "FORCE_EXIT"
RISK_TRIM_PRIORITY = "TRIM_PRIORITY"
RISK_BLOCK_NEW_BUY = "BLOCK_NEW_BUY"
RISK_OK = "OK"

# Confidence labels (spec §5).
CONF_VERY_HIGH = 0.70
CONF_HIGH = 0.55
CONF_MEDIUM = 0.40

# Conviction labels (spec §6) combine persistence + intent_score.
CONVICTION_INSTITUTIONAL_PERSIST = 0.80
CONVICTION_INSTITUTIONAL_INTENT = 0.70
CONVICTION_STRONG_PERSIST = 0.65
CONVICTION_STRONG_INTENT = 0.60
CONVICTION_MODERATE_PERSIST = 0.50
CONVICTION_MODERATE_INTENT = 0.45

# Trend interpretation thresholds (slope per cycle).
TREND_STRONG = 0.01
TREND_FLAT = 0.001

# Persistence engine decision labels we surface in tags.
PERSISTENCE_PROMOTE_CONFIRMED = "PROMOTE_CONFIRMED_OPEN_NEW"
PERSISTENCE_KEEP_WATCH = "KEEP_WATCH"
PERSISTENCE_DEMOTE = "DEMOTE_WATCH"
PERSISTENCE_REJECT = "REJECT"

OUTPUT_COLUMNS = [
    "ticker",
    "execution_intent",
    "rebalance_action",
    "confidence",
    "persistence_score",
    "delta_pct",
    "lifecycle_action",
    "signal",
    "risk_flag",
    "rationale_short",
    "rationale_long",
    "rationale_tags",
    "confidence_label",
    "conviction_label",
    "explanation_score",
]


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[TRADE_RATIONALE_WARN] {msg}", flush=True)


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


def _opt_fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _opt_pct(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.{digits}f}%"


# -----------------------------------------------------------
# Loaders
# -----------------------------------------------------------
def _load_intents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        _warn("execution intents CSV missing ticker/symbol; no rationale rows")
        return []
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        rows.append(
            {
                "ticker": sym,
                "execution_intent": _norm_upper(r.get("execution_intent")),
                "rebalance_action": _norm_upper(r.get("rebalance_action")),
                "rebalance_amount_usd": _to_float_or_zero(r.get("rebalance_amount_usd")),
                "priority": _to_float_or_zero(r.get("priority")),
                "intent_score": _to_float_or_zero(r.get("intent_score")),
                "execution_ready": _to_bool(r.get("execution_ready")),
                "confidence": _to_float(r.get("confidence")),
                "persistence_score": _to_float(r.get("persistence_score")),
                "delta_pct": _to_float(r.get("delta_pct")),
                "lifecycle_action": _norm_upper(r.get("lifecycle_action")),
                "signal": _norm_upper(r.get("signal")),
                "risk_flag": _norm_upper(r.get("risk_flag")) or RISK_OK,
                "intent_reason": str(r.get("reason") or ""),
            }
        )
    return rows


def _load_persistence_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
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
            "promotion_decision": _norm_upper(r.get("promotion_decision")),
            "persistence_score": _to_float(r.get("persistence_score")),
            "confidence_trend": _to_float(r.get("confidence_trend")),
            "delta_trend": _to_float(r.get("delta_trend")),
            "promotion_score_trend": _to_float(r.get("promotion_score_trend")),
            "consecutive_watch_cycles": int(_to_float_or_zero(r.get("consecutive_watch_cycles"))),
            "signal_consistency": _to_float(r.get("signal_consistency")),
            "lifecycle_consistency": _to_float(r.get("lifecycle_consistency")),
        }
    return out


def _load_rebalance_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
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
            "execution_order": int(_to_float_or_zero(r.get("execution_order"))),
            "delta_weight_pct": _to_float(r.get("delta_weight_pct")),
            "current_value_usd": _to_float(r.get("current_value_usd")),
            "target_value_usd": _to_float(r.get("target_value_usd")),
            "current_weight_pct": _to_float(r.get("current_weight_pct")),
            "target_weight_pct": _to_float(r.get("target_weight_pct")),
            "reason": str(r.get("reason") or ""),
        }
    return out


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
            "rationale": str(r.get("rationale") or r.get("rationale_short") or ""),
        }
    return out


def _load_lifecycle_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
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
            "lifecycle_action": _norm_upper(r.get("lifecycle_action")),
            "effective_stance": _norm_upper(r.get("effective_stance"))
            or _norm_upper(r.get("stance")),
        }
    return out


# -----------------------------------------------------------
# Label derivation
# -----------------------------------------------------------
def _confidence_label(confidence: Optional[float]) -> str:
    if confidence is None:
        return "LOW"
    if confidence >= CONF_VERY_HIGH:
        return "VERY_HIGH"
    if confidence >= CONF_HIGH:
        return "HIGH"
    if confidence >= CONF_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _conviction_label(persistence_score: Optional[float], intent_score: float) -> str:
    p = persistence_score or 0.0
    if p >= CONVICTION_INSTITUTIONAL_PERSIST and intent_score >= CONVICTION_INSTITUTIONAL_INTENT:
        return "INSTITUTIONAL"
    if p >= CONVICTION_STRONG_PERSIST or intent_score >= CONVICTION_STRONG_INTENT:
        return "STRONG"
    if p >= CONVICTION_MODERATE_PERSIST or intent_score >= CONVICTION_MODERATE_INTENT:
        return "MODERATE"
    return "WEAK"


def _trend_word(slope: Optional[float], *, label: str) -> str:
    """Translate a numeric trend slope into 'strengthening / weakening / stable'."""
    if slope is None:
        return f"{label} trend unavailable"
    if slope >= TREND_STRONG:
        return f"strengthening {label}"
    if slope <= -TREND_STRONG:
        return f"weakening {label}"
    return f"stable {label}"


def _alignment_value(
    *,
    rebalance_action: str,
    label_value: str,
    risk_components: List[str],
) -> float:
    """
    Return 1.0 if `label_value` aligns with the trade direction, 0.0 if
    it contradicts, 0.5 if neutral. For risk-protective sells, missing
    or BUY-aligned labels are treated as neutral (the risk flag is the
    real alignment).
    """
    if rebalance_action in BUY_DIRECTION_ACTIONS:
        if label_value in POSITIVE_LABELS:
            return 1.0
        if label_value in NEGATIVE_LABELS:
            return 0.0
        return 0.5
    if rebalance_action in SELL_DIRECTION_ACTIONS:
        # Risk-protective sell with FORCE_EXIT/TRIM_PRIORITY → max alignment.
        if RISK_FORCE_EXIT in risk_components or RISK_TRIM_PRIORITY in risk_components:
            return 1.0
        if label_value in NEGATIVE_LABELS:
            return 1.0
        if label_value in POSITIVE_LABELS:
            return 0.0
        return 0.5
    # HOLD / NO_ACTION → alignment irrelevant.
    return 0.5


def _explanation_score(
    *,
    confidence: Optional[float],
    persistence_score: Optional[float],
    intent_score: float,
    lifecycle_alignment: float,
    signal_alignment: float,
) -> float:
    """0–1 measure of how well the recommendation is *substantiated*."""
    return round(
        _clamp(
            0.25 * _clamp(_to_float_or_zero(confidence), 0.0, 1.0)
            + 0.20 * _clamp(_to_float_or_zero(persistence_score), 0.0, 1.0)
            + 0.25 * _clamp(intent_score, 0.0, 1.0)
            + 0.15 * _clamp(lifecycle_alignment, 0.0, 1.0)
            + 0.15 * _clamp(signal_alignment, 0.0, 1.0),
            0.0,
            1.0,
        ),
        6,
    )


# -----------------------------------------------------------
# Tag generation
# -----------------------------------------------------------
def _build_tags(
    *,
    intent: str,
    rebalance_action: str,
    confidence: Optional[float],
    persistence_score: Optional[float],
    delta_pct: Optional[float],
    signal: str,
    lifecycle_action: str,
    risk_components: List[str],
    persistence_decision: str,
    lifecycle_alignment: float,
    signal_alignment: float,
    intent_reason: str,
) -> List[str]:
    """Discrete machine-filterable tags (spec §4)."""
    tags: List[str] = []

    # Delta direction
    if delta_pct is None:
        tags.append("delta_unavailable")
    elif delta_pct > 0:
        tags.append("positive_delta")
    elif delta_pct < 0:
        tags.append("negative_delta")
    else:
        tags.append("neutral_delta")

    # Confidence band
    if confidence is None:
        tags.append("confidence_unavailable")
    elif confidence >= CONF_VERY_HIGH:
        tags.append("very_high_confidence")
    elif confidence >= CONF_HIGH:
        tags.append("high_confidence")
    elif confidence >= CONF_MEDIUM:
        tags.append("medium_confidence")
    else:
        tags.append("low_confidence")

    if confidence is not None and 0.45 <= confidence < 0.50:
        tags.append("confidence_borderline")

    # Persistence
    if persistence_decision == PERSISTENCE_PROMOTE_CONFIRMED:
        tags.append("persistence_confirmed")
    elif persistence_decision == PERSISTENCE_DEMOTE:
        tags.append("persistence_weakening")
    elif persistence_decision == PERSISTENCE_REJECT:
        tags.append("persistence_rejected")
    elif persistence_score is not None and persistence_score >= 0.60:
        tags.append("persistence_strong")
    elif persistence_score is not None and persistence_score < 0.45:
        tags.append("persistence_weak")

    # Risk
    if RISK_FORCE_EXIT in risk_components:
        tags.append("risk_force_exit")
    if RISK_TRIM_PRIORITY in risk_components:
        tags.append("risk_trim_priority")
    if RISK_BLOCK_NEW_BUY in risk_components:
        tags.append("risk_block_new_buy")

    # Signal / lifecycle alignment
    if signal in NEGATIVE_LABELS:
        tags.append("negative_signal")
    elif signal in POSITIVE_LABELS:
        tags.append("positive_signal")

    if lifecycle_action in NEGATIVE_LABELS:
        tags.append("lifecycle_exit")
    elif lifecycle_action in POSITIVE_LABELS:
        tags.append("lifecycle_buy")

    if lifecycle_alignment <= 0.0 and rebalance_action in BUY_DIRECTION_ACTIONS:
        tags.append("lifecycle_contradiction")
    if signal_alignment <= 0.0 and rebalance_action in BUY_DIRECTION_ACTIONS:
        tags.append("signal_contradiction")
    if lifecycle_alignment >= 1.0 and signal_alignment >= 1.0:
        tags.append("full_alignment")

    # Intent-specific
    if intent == INTENT_EXECUTE:
        if rebalance_action in SELL_DIRECTION_ACTIONS:
            tags.append("risk_protective_sell")
        else:
            tags.append("execute_now_buy")
    elif intent == INTENT_DELAY:
        tags.append("delayed_for_confirmation")
    elif intent == INTENT_BLOCK:
        tags.append("blocked")
        if "blocked_by_risk_overlay" in intent_reason or "blocked_new_buy_risk" in intent_reason:
            tags.append("risk_blocked")
        if "lifecycle_contradiction" in intent_reason:
            tags.append("blocked_by_lifecycle")
        if "negative_delta" in intent_reason:
            tags.append("blocked_negative_delta")
        if "confidence_below_floor" in intent_reason:
            tags.append("blocked_low_confidence")
    elif intent == INTENT_SKIP:
        tags.append("skipped")
        if "below_trade_minimum" in intent_reason:
            tags.append("below_trade_minimum")
        if "non_actionable_row" in intent_reason:
            tags.append("hold_no_trade")

    # Dedupe while preserving order.
    seen = set()
    deduped: List[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


# -----------------------------------------------------------
# Rationale text generation
# -----------------------------------------------------------
def _intent_short_label(intent: str) -> str:
    return {
        INTENT_EXECUTE: "Execute now",
        INTENT_DELAY: "Delayed",
        INTENT_BLOCK: "Blocked",
        INTENT_SKIP: "Skipped",
    }.get(intent, intent.title())


def _rationale_short(
    *,
    intent: str,
    rebalance_action: str,
    conviction_label: str,
    confidence_label: str,
    persistence_decision: str,
    risk_components: List[str],
    intent_reason: str,
) -> str:
    """One-line summary (spec §2)."""
    prefix = _intent_short_label(intent)

    if intent == INTENT_EXECUTE:
        if rebalance_action == REBAL_FULL_EXIT:
            flag = "FORCE_EXIT" if RISK_FORCE_EXIT in risk_components else "risk overlay"
            return f"{prefix}: risk-protective full exit driven by {flag}."
        if rebalance_action == REBAL_TRIM:
            flag = "TRIM_PRIORITY" if RISK_TRIM_PRIORITY in risk_components else "risk overlay"
            return f"{prefix}: protective trim driven by {flag}."
        if rebalance_action == REBAL_SELL:
            return f"{prefix}: discretionary rebalance sell."
        if persistence_decision == PERSISTENCE_PROMOTE_CONFIRMED:
            return f"{prefix}: persistent strengthening signal ({conviction_label.lower()} conviction)."
        return f"{prefix}: {conviction_label.lower()} conviction setup with {confidence_label.lower()} confidence."

    if intent == INTENT_DELAY:
        if "confidence_borderline" in intent_reason:
            return f"{prefix}: confidence borderline -- awaiting strengthening."
        if "persistence_weak" in intent_reason:
            return f"{prefix}: persistence below execution threshold."
        if "persistence_deteriorating" in intent_reason:
            return f"{prefix}: persistence is deteriorating."
        return f"{prefix}: directionally positive but below execution gate."

    if intent == INTENT_BLOCK:
        if (
            "blocked_by_risk_overlay" in intent_reason
            or "blocked_new_buy_risk" in intent_reason
            or "blocked_force_exit_risk" in intent_reason
        ):
            return f"{prefix}: risk overlay restriction prevents new exposure."
        if "lifecycle_contradiction" in intent_reason:
            return f"{prefix}: lifecycle contradicts trade direction."
        if "negative_delta" in intent_reason:
            return f"{prefix}: signal delta turned negative."
        if "persistence_rejected" in intent_reason:
            return f"{prefix}: persistence engine rejected the candidate."
        if "confidence_below_floor" in intent_reason:
            return f"{prefix}: confidence below execution floor."
        return f"{prefix}: cannot execute due to upstream risk."

    if intent == INTENT_SKIP:
        if "below_trade_minimum" in intent_reason:
            return f"{prefix}: trade notional below minimum threshold."
        if "non_actionable_row" in intent_reason:
            return f"{prefix}: position requires no rebalance today."
        if "upstream_not_ready" in intent_reason:
            return f"{prefix}: upstream not ready for execution."
        return f"{prefix}: non-actionable row."

    return f"{prefix}: see detailed reason."


def _rationale_long(
    *,
    ticker: str,
    sector: str,
    intent: str,
    rebalance_action: str,
    rebalance_amount_usd: float,
    current_weight_pct: Optional[float],
    target_weight_pct: Optional[float],
    delta_weight_pct: Optional[float],
    confidence: Optional[float],
    persistence_score: Optional[float],
    delta_pct: Optional[float],
    signal: str,
    lifecycle_action: str,
    risk_flag: str,
    risk_components: List[str],
    persistence_decision: str,
    confidence_trend: Optional[float],
    delta_trend: Optional[float],
    consecutive_watch_cycles: int,
    confidence_label: str,
    conviction_label: str,
    lifecycle_alignment: float,
    signal_alignment: float,
) -> str:
    """Institutional-quality paragraph (spec §3) with full provenance."""
    parts: List[str] = []

    # Opening sentence anchored on intent + sector context.
    if intent == INTENT_EXECUTE and rebalance_action in BUY_DIRECTION_ACTIONS:
        verb = "is recommended for immediate execution"
    elif intent == INTENT_EXECUTE and rebalance_action == REBAL_FULL_EXIT:
        verb = "is recommended for immediate full exit"
    elif intent == INTENT_EXECUTE and rebalance_action == REBAL_TRIM:
        verb = "is recommended for an immediate trim"
    elif intent == INTENT_EXECUTE:
        verb = "is recommended for immediate execution"
    elif intent == INTENT_DELAY:
        verb = "is held back for one cycle"
    elif intent == INTENT_BLOCK:
        verb = "is blocked from execution"
    elif intent == INTENT_SKIP:
        verb = "carries no actionable trade today"
    else:
        verb = "is under review"

    sector_phrase = f" ({sector} exposure)" if sector and sector != UNKNOWN_SECTOR_LABEL else ""
    parts.append(f"{ticker}{sector_phrase} {verb}.")

    # Signal direction + confidence
    sig_phrase: List[str] = []
    if signal:
        if signal in POSITIVE_LABELS:
            sig_phrase.append(f"the underlying signal is {signal}")
        elif signal in NEGATIVE_LABELS:
            sig_phrase.append(f"the underlying signal is {signal}")
        else:
            sig_phrase.append(f"signal is {signal}")
    if confidence is not None:
        sig_phrase.append(
            f"confidence is {confidence_label.lower().replace('_', ' ')} ({confidence:.2f})"
        )
    if sig_phrase:
        parts.append("Signal context: " + ", ".join(sig_phrase) + ".")

    # Trend phrases
    trend_phrases: List[str] = []
    if confidence_trend is not None:
        trend_phrases.append(_trend_word(confidence_trend, label="confidence"))
    if delta_trend is not None:
        trend_phrases.append(_trend_word(delta_trend, label="delta"))
    if delta_pct is not None:
        direction = "positive" if delta_pct > 0 else "negative" if delta_pct < 0 else "flat"
        trend_phrases.append(f"latest delta is {direction} ({_opt_pct(delta_pct)})")
    if trend_phrases:
        parts.append("Trends: " + "; ".join(trend_phrases) + ".")

    # Persistence
    persist_phrases: List[str] = []
    if persistence_score is not None:
        persist_phrases.append(f"persistence score {persistence_score:.2f}")
    if persistence_decision:
        persist_phrases.append(f"persistence engine decision {persistence_decision}")
    if consecutive_watch_cycles > 0:
        persist_phrases.append(
            f"{consecutive_watch_cycles} consecutive WATCH cycle"
            + ("s" if consecutive_watch_cycles != 1 else "")
        )
    if persist_phrases:
        parts.append("Persistence: " + "; ".join(persist_phrases) + ".")

    # Lifecycle alignment
    lc_phrase_bits: List[str] = []
    if lifecycle_action:
        if lifecycle_alignment >= 1.0:
            lc_phrase_bits.append(
                f"lifecycle action {lifecycle_action} aligns with the trade direction"
            )
        elif lifecycle_alignment <= 0.0:
            lc_phrase_bits.append(
                f"lifecycle action {lifecycle_action} contradicts the trade direction"
            )
        else:
            lc_phrase_bits.append(f"lifecycle action is {lifecycle_action} (neutral)")
    if lc_phrase_bits:
        parts.append("Lifecycle alignment: " + "; ".join(lc_phrase_bits) + ".")

    # Risk overlay
    if risk_components:
        parts.append(
            "Risk overlay flags: "
            + ", ".join(risk_components)
            + ". These flags "
            + (
                "drive the protective sell decision."
                if rebalance_action in SELL_DIRECTION_ACTIONS
                else "prevent or constrain new exposure on this name."
            )
        )
    else:
        if rebalance_action in BUY_DIRECTION_ACTIONS:
            parts.append("Risk overlay is clean (no severe flags).")

    # Portfolio / diversification context
    portfolio_bits: List[str] = []
    if current_weight_pct is not None and target_weight_pct is not None:
        portfolio_bits.append(
            f"current weight {current_weight_pct:.2f}% -> target weight {target_weight_pct:.2f}%"
        )
    if delta_weight_pct is not None and abs(delta_weight_pct) > 0.001:
        sign = "+" if delta_weight_pct > 0 else ""
        portfolio_bits.append(f"weight delta {sign}{delta_weight_pct:.2f}%")
    if abs(rebalance_amount_usd) >= 0.01:
        sign = "+" if rebalance_amount_usd > 0 else ""
        portfolio_bits.append(f"notional {sign}${rebalance_amount_usd:,.2f}")
    if portfolio_bits:
        parts.append("Portfolio impact: " + "; ".join(portfolio_bits) + ".")

    # Closing line for executes -- institutional-grade conclusion.
    if intent == INTENT_EXECUTE and rebalance_action in BUY_DIRECTION_ACTIONS:
        parts.append(
            f"Conviction is rated {conviction_label} -- the position improves "
            f"{sector or 'portfolio'} diversification while staying within "
            "concentration limits."
        )

    return " ".join(parts)


# -----------------------------------------------------------
# Pipeline
# -----------------------------------------------------------
def build_rationales(
    *,
    intent_rows: List[Dict[str, Any]],
    persistence_map: Dict[str, Dict[str, Any]],
    rebalance_map: Dict[str, Dict[str, Any]],
    risk_overlay_map: Dict[str, str],
    signals_map: Dict[str, Dict[str, Any]],
    lifecycle_map: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Pure planner — no IO. Returns (rationale_df, summary)."""
    rows: List[Dict[str, Any]] = []

    for r in intent_rows:
        sym = r["ticker"]
        intent = r["execution_intent"]
        rebalance_action = r["rebalance_action"]
        intent_score = float(r.get("intent_score") or 0.0)
        intent_reason = r.get("intent_reason", "")
        rebalance_amount = float(r.get("rebalance_amount_usd") or 0.0)

        # Enrichment with documented precedence: intents → persistence
        # → lifecycle/signals → risk overlay → rebalance.
        persist = persistence_map.get(sym, {})
        reb = rebalance_map.get(sym, {})
        sig = signals_map.get(sym, {})
        lc = lifecycle_map.get(sym, {})

        confidence = (
            r.get("confidence")
            if r.get("confidence") is not None
            else (sig.get("confidence") or persist.get("latest_confidence"))
        )
        persistence_score = (
            r.get("persistence_score")
            if r.get("persistence_score") is not None
            else persist.get("persistence_score")
        )
        delta_pct = r.get("delta_pct") if r.get("delta_pct") is not None else sig.get("delta_pct")
        signal = r.get("signal") or sig.get("signal") or ""
        lifecycle_action = r.get("lifecycle_action") or lc.get("lifecycle_action") or ""
        risk_flag = r.get("risk_flag") or risk_overlay_map.get(sym, RISK_OK)
        risk_comps = _risk_components(risk_flag)

        persistence_decision = persist.get("promotion_decision", "") or ""
        confidence_trend = persist.get("confidence_trend")
        delta_trend = persist.get("delta_trend")
        consecutive_watch_cycles = int(persist.get("consecutive_watch_cycles", 0) or 0)

        # Alignment scores feed both labels and explanation_score.
        lifecycle_alignment = _alignment_value(
            rebalance_action=rebalance_action,
            label_value=lifecycle_action,
            risk_components=risk_comps,
        )
        signal_alignment = _alignment_value(
            rebalance_action=rebalance_action,
            label_value=signal,
            risk_components=risk_comps,
        )

        confidence_label = _confidence_label(confidence)
        conviction_label = _conviction_label(persistence_score, intent_score)

        explanation_score = _explanation_score(
            confidence=confidence,
            persistence_score=persistence_score,
            intent_score=intent_score,
            lifecycle_alignment=lifecycle_alignment,
            signal_alignment=signal_alignment,
        )

        tags = _build_tags(
            intent=intent,
            rebalance_action=rebalance_action,
            confidence=confidence,
            persistence_score=persistence_score,
            delta_pct=delta_pct,
            signal=signal,
            lifecycle_action=lifecycle_action,
            risk_components=risk_comps,
            persistence_decision=persistence_decision,
            lifecycle_alignment=lifecycle_alignment,
            signal_alignment=signal_alignment,
            intent_reason=intent_reason,
        )

        sector = get_sector(sym)

        short = _rationale_short(
            intent=intent,
            rebalance_action=rebalance_action,
            conviction_label=conviction_label,
            confidence_label=confidence_label,
            persistence_decision=persistence_decision,
            risk_components=risk_comps,
            intent_reason=intent_reason,
        )

        long = _rationale_long(
            ticker=sym,
            sector=sector,
            intent=intent,
            rebalance_action=rebalance_action,
            rebalance_amount_usd=rebalance_amount,
            current_weight_pct=reb.get("current_weight_pct"),
            target_weight_pct=reb.get("target_weight_pct"),
            delta_weight_pct=reb.get("delta_weight_pct"),
            confidence=confidence,
            persistence_score=persistence_score,
            delta_pct=delta_pct,
            signal=signal,
            lifecycle_action=lifecycle_action,
            risk_flag=risk_flag,
            risk_components=risk_comps,
            persistence_decision=persistence_decision,
            confidence_trend=confidence_trend,
            delta_trend=delta_trend,
            consecutive_watch_cycles=consecutive_watch_cycles,
            confidence_label=confidence_label,
            conviction_label=conviction_label,
            lifecycle_alignment=lifecycle_alignment,
            signal_alignment=signal_alignment,
        )

        rows.append(
            {
                "ticker": sym,
                "execution_intent": intent,
                "rebalance_action": rebalance_action,
                "confidence": (round(float(confidence), 6) if confidence is not None else None),
                "persistence_score": (
                    round(float(persistence_score), 6) if persistence_score is not None else None
                ),
                "delta_pct": (round(float(delta_pct), 6) if delta_pct is not None else None),
                "lifecycle_action": lifecycle_action,
                "signal": signal,
                "risk_flag": risk_flag,
                "rationale_short": short,
                "rationale_long": long,
                "rationale_tags": "|".join(tags),
                "confidence_label": confidence_label,
                "conviction_label": conviction_label,
                "explanation_score": explanation_score,
                "_execution_order": int(reb.get("execution_order", 0) or 0),
            }
        )

    # Preserve Step 6 execution order (mirrors Step 7's contract).
    rows.sort(key=lambda r: (r["_execution_order"], r["ticker"]))
    for r in rows:
        r.pop("_execution_order", None)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # ── Summary ──────────────────────────────────────────────────
    def _count(intent: str) -> int:
        return int(sum(1 for r in rows if r["execution_intent"] == intent))

    execute_n = _count(INTENT_EXECUTE)
    delay_n = _count(INTENT_DELAY)
    skip_n = _count(INTENT_SKIP)
    block_n = _count(INTENT_BLOCK)

    avg_score = (
        round(
            float(sum(float(r["explanation_score"]) for r in rows) / max(1, len(rows))),
            6,
        )
        if rows
        else 0.0
    )

    # Strongest = top by explanation_score; weakest = bottom (excluding
    # SKIP rows which are intentionally low-scored non-trades).
    score_rows = [r for r in rows if r["execution_intent"] != INTENT_SKIP] or list(rows)
    score_rows_sorted = sorted(
        score_rows,
        key=lambda r: (float(r["explanation_score"]), r["ticker"]),
        reverse=True,
    )

    def _pack(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ticker": r["ticker"],
            "execution_intent": r["execution_intent"],
            "rebalance_action": r["rebalance_action"],
            "explanation_score": float(r["explanation_score"]),
            "confidence_label": r["confidence_label"],
            "conviction_label": r["conviction_label"],
            "rationale_short": r["rationale_short"],
        }

    strongest = [_pack(r) for r in score_rows_sorted[:5]]
    weakest = [_pack(r) for r in score_rows_sorted[-5:][::-1]]

    confidence_label_counts = Counter(r["confidence_label"] for r in rows)
    conviction_label_counts = Counter(r["conviction_label"] for r in rows)

    summary: Dict[str, Any] = {
        "generated_at_utc": _now_iso_utc(),
        "total_explanations": int(len(rows)),
        "execute_now": execute_n,
        "delayed": delay_n,
        "skipped": skip_n,
        "blocked": block_n,
        "avg_explanation_score": avg_score,
        "confidence_label_distribution": dict(confidence_label_counts),
        "conviction_label_distribution": dict(conviction_label_counts),
        "thresholds": {
            "confidence_very_high": CONF_VERY_HIGH,
            "confidence_high": CONF_HIGH,
            "confidence_medium": CONF_MEDIUM,
            "conviction_institutional_persistence": CONVICTION_INSTITUTIONAL_PERSIST,
            "conviction_institutional_intent": CONVICTION_INSTITUTIONAL_INTENT,
            "conviction_strong_persistence": CONVICTION_STRONG_PERSIST,
            "conviction_strong_intent": CONVICTION_STRONG_INTENT,
            "conviction_moderate_persistence": CONVICTION_MODERATE_PERSIST,
            "conviction_moderate_intent": CONVICTION_MODERATE_INTENT,
        },
        "strongest_rationales": strongest,
        "weakest_rationales": weakest,
    }
    return df, summary


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only trade rationale & explainability engine (step 8 of WATCH "
            "funnel). Generates institutional-quality explanations for every "
            "execution intent."
        ),
    )
    p.add_argument("--intents", default=str(DEFAULT_INTENTS_CSV))
    p.add_argument("--persistence", default=str(DEFAULT_PERSISTENCE_CSV))
    p.add_argument("--rebalance", default=str(DEFAULT_REBALANCE_CSV))
    p.add_argument("--risk-overlay", default=str(DEFAULT_RISK_OVERLAY_CSV))
    p.add_argument("--signals", default=str(DEFAULT_SIGNALS_CSV))
    p.add_argument("--lifecycle-effective", default=str(DEFAULT_LIFECYCLE_EFFECTIVE_CSV))
    p.add_argument("--out-csv", default=str(DEFAULT_OUTPUT_CSV))
    p.add_argument("--out-json", default=str(DEFAULT_OUTPUT_JSON))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[TRADE_RATIONALE] starting (read-only explainability engine)", flush=True)

    intents_df = _safe_read_csv(Path(args.intents), label="portfolio_execution_intents.csv")
    persistence_df = _safe_read_csv(
        Path(args.persistence), label="opportunity_persistence_recommendations.csv"
    )
    rebalance_df = _safe_read_csv(Path(args.rebalance), label="portfolio_rebalance_plan.csv")
    risk_df = _safe_read_csv(Path(args.risk_overlay), label="performance_risk_overlay.csv")
    signals_df = _safe_read_csv(Path(args.signals), label="signals_with_rationale.csv")
    lifecycle_df = _safe_read_csv(
        Path(args.lifecycle_effective), label="signal_lifecycle_effective.csv"
    )

    intent_rows = _load_intents(intents_df)
    persistence_map = _load_persistence_map(persistence_df)
    rebalance_map = _load_rebalance_map(rebalance_df)
    risk_overlay_map = _load_risk_overlay_map(risk_df)
    signals_map = _load_signals_map(signals_df)
    lifecycle_map = _load_lifecycle_map(lifecycle_df)

    df, summary = build_rationales(
        intent_rows=intent_rows,
        persistence_map=persistence_map,
        rebalance_map=rebalance_map,
        risk_overlay_map=risk_overlay_map,
        signals_map=signals_map,
        lifecycle_map=lifecycle_map,
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
        "[TRADE_RATIONALE] "
        f"generated={summary['total_explanations']} "
        f"execute={summary['execute_now']} "
        f"delay={summary['delayed']} "
        f"block={summary['blocked']} "
        f"skip={summary['skipped']} "
        f"avg_score={summary['avg_explanation_score']:.3f}",
        flush=True,
    )
    print(
        "[RATIONALE_TOP] symbols=" f"{[r['ticker'] for r in summary['strongest_rationales']]}",
        flush=True,
    )
    print(
        f"[TRADE_RATIONALE_OUT] csv={out_csv.as_posix()} summary={out_json.as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
