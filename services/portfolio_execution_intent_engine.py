"""
Portfolio Execution Intent Engine — Step 7 of the WATCH → DEPLOY funnel.

Reads:
    data/results/portfolio_rebalance_plan.csv
    data/results/portfolio_rebalance_summary.json
    data/results/performance_risk_overlay.csv
    data/results/opportunity_persistence_recommendations.csv
    data/results/signals_with_rationale.csv
    data/results/signal_lifecycle_effective.csv

Writes:
    data/results/portfolio_execution_intents.csv
    data/results/portfolio_execution_summary.json

Purpose
-------
Step 6 produced a sequenced rebalance plan. Step 7 is the *final gate*
before any of those trades would actually be placed.

It answers:

    "Should Triton execute this rebalance action *now*?"

Per row it emits an `execution_intent`:

  * EXECUTE_NOW — passes all freshness/conviction/risk gates
  * DELAY       — directionally correct but borderline; wait one cycle
  * SKIP        — not a trade (HOLD/NO_ACTION), below minimums, or
                  upstream not-ready
  * BLOCK       — severe risk, negative signal, or lifecycle/rebalance
                  contradiction

Sells driven by risk (FULL_EXIT/TRIM) are treated as protective and
execute whenever ready — they do *not* need persistence or signal
confirmation since the risk overlay is the rationale. Buys (BUY_NEW/
ADD) must pass full conviction gates.

Safety
------
* Read-only. No orders, no broker calls, no mutation of execute_trades
  or manage_positions. The output is intent only — execute_trades is
  the sole authority for placing real orders.
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

DEFAULT_REBALANCE_CSV = RESULTS_DIR / "portfolio_rebalance_plan.csv"
DEFAULT_REBALANCE_JSON = RESULTS_DIR / "portfolio_rebalance_summary.json"
DEFAULT_RISK_OVERLAY_CSV = RESULTS_DIR / "performance_risk_overlay.csv"
DEFAULT_PERSISTENCE_CSV = RESULTS_DIR / "opportunity_persistence_recommendations.csv"
DEFAULT_SIGNALS_CSV = RESULTS_DIR / "signals_with_rationale.csv"
DEFAULT_LIFECYCLE_EFFECTIVE_CSV = RESULTS_DIR / "signal_lifecycle_effective.csv"

DEFAULT_OUTPUT_CSV = RESULTS_DIR / "portfolio_execution_intents.csv"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "portfolio_execution_summary.json"

# -----------------------------------------------------------
# Tunables
# -----------------------------------------------------------
INTENT_EXECUTE = "EXECUTE_NOW"
INTENT_DELAY = "DELAY"
INTENT_SKIP = "SKIP"
INTENT_BLOCK = "BLOCK"

# Conviction gates (apply to BUY direction only).
MIN_EXECUTE_CONFIDENCE = 0.50
MIN_EXECUTE_PERSISTENCE = 0.60
MIN_KEEP_CONFIDENCE = 0.45  # below this for a BUY → BLOCK
BORDERLINE_CONFIDENCE_LOW = 0.45  # [low, high) = DELAY band
BORDERLINE_CONFIDENCE_HIGH = MIN_EXECUTE_CONFIDENCE
DELAY_PERSISTENCE_LOW = 0.45  # [low, MIN_EXECUTE_PERSISTENCE) = DELAY band

# Step 11 runtime policy override (optional, additive).
# Default 0.0 disables the intent_score floor entirely -- with no
# override loaded, behaviour is identical to pre-Step-11. When
# runtime_policy.json sets deployment_threshold > 0, BUY-direction
# rows whose intent_score falls below that floor are demoted from
# EXECUTE_NOW to DELAY. SELL/EXIT actions are never affected
# (risk-protective sells must always be able to fire).
DEFAULT_RUNTIME_POLICY_JSON = RESULTS_DIR / "runtime_policy.json"
MIN_EXECUTE_INTENT_SCORE = 0.0

# Persistence engine labels (deterioration signals)
PERSISTENCE_DEMOTE = "DEMOTE_WATCH"
PERSISTENCE_REJECT = "REJECT"

# Risk flag tokens
RISK_FORCE_EXIT = "FORCE_EXIT"
RISK_TRIM_PRIORITY = "TRIM_PRIORITY"
RISK_BLOCK_NEW_BUY = "BLOCK_NEW_BUY"
RISK_OK = "OK"

# Rebalance action labels (mirror Step 6)
REBAL_BUY_NEW = "BUY_NEW"
REBAL_ADD = "ADD"
REBAL_TRIM = "TRIM"
REBAL_SELL = "SELL"
REBAL_FULL_EXIT = "FULL_EXIT"
REBAL_HOLD = "HOLD"
REBAL_NO_ACTION = "NO_ACTION"

BUY_DIRECTION_ACTIONS: frozenset = frozenset({REBAL_BUY_NEW, REBAL_ADD})
SELL_DIRECTION_ACTIONS: frozenset = frozenset({REBAL_FULL_EXIT, REBAL_TRIM, REBAL_SELL})

# Signal / lifecycle direction labels
POSITIVE_LABELS: frozenset = frozenset({"BUY", "ADD", "LONG"})
NEGATIVE_LABELS: frozenset = frozenset({"SELL", "EXIT", "REDUCE", "SHORT", "DUMP"})

OUTPUT_COLUMNS = [
    "ticker",
    "rebalance_action",
    "rebalance_amount_usd",
    "priority",
    "execution_ready",
    "execution_intent",
    "confidence",
    "persistence_score",
    "delta_pct",
    "lifecycle_action",
    "signal",
    "risk_flag",
    "intent_score",
    "reason",
]


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[EXECUTION_INTENT_WARN] {msg}", flush=True)


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
def _load_rebalance_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Pull every row from Step 6's CSV in execution order."""
    if df is None or df.empty:
        return []
    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        _warn("rebalance CSV missing ticker/symbol column; no intents")
        return []
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        rows.append(
            {
                "ticker": sym,
                "rebalance_action": _norm_upper(r.get("rebalance_action")),
                "rebalance_amount_usd": _to_float_or_zero(r.get("rebalance_amount_usd")),
                "priority": _to_float_or_zero(r.get("priority")),
                "execution_order": int(_to_float_or_zero(r.get("execution_order"))),
                "execution_ready": _to_bool(r.get("execution_ready")),
                "rebalance_reason": str(r.get("reason") or ""),
                "delta_weight_pct": _to_float_or_zero(r.get("delta_weight_pct")),
            }
        )
    # Preserve Step 6 sequence by sorting on execution_order.
    rows.sort(key=lambda r: (r["execution_order"], r["ticker"]))
    return rows


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
            "latest_confidence": _to_float(r.get("latest_confidence")),
            "latest_delta_pct": _to_float(r.get("latest_delta_pct")),
            "confidence_trend": _to_float(r.get("confidence_trend")),
            "delta_trend": _to_float(r.get("delta_trend")),
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


def _load_lifecycle_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
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
# Decision logic
# -----------------------------------------------------------
def _decide_intent(
    *,
    rebalance_action: str,
    execution_ready: bool,
    confidence: Optional[float],
    persistence_score: Optional[float],
    delta_pct: Optional[float],
    signal: str,
    lifecycle_action: str,
    risk_components: List[str],
    persistence_decision: str,
    rebalance_reason: str,
) -> Tuple[str, str]:
    """
    Apply precedence: BLOCK → SKIP → DELAY → EXECUTE_NOW.

    Risk-protective sells (FULL_EXIT/TRIM) bypass conviction gates —
    the risk overlay IS the rationale, so they execute whenever ready.
    Buys (BUY_NEW/ADD) must pass freshness + conviction + persistence.
    """
    is_buy = rebalance_action in BUY_DIRECTION_ACTIONS
    is_sell = rebalance_action in SELL_DIRECTION_ACTIONS

    # ── D. BLOCK (highest precedence) ────────────────────────────
    # Buys into a severely-flagged name should never execute, even if
    # Step 5/6 somehow let them through.
    if is_buy:
        if RISK_FORCE_EXIT in risk_components:
            return INTENT_BLOCK, "blocked_force_exit_risk"
        if RISK_BLOCK_NEW_BUY in risk_components:
            return INTENT_BLOCK, "blocked_new_buy_risk"
        # Lifecycle / signal contradiction with the trade direction.
        if signal in NEGATIVE_LABELS or lifecycle_action in NEGATIVE_LABELS:
            return INTENT_BLOCK, "lifecycle_contradiction"
        if delta_pct is not None and delta_pct < 0:
            return INTENT_BLOCK, "negative_delta"
        if confidence is not None and confidence < MIN_KEEP_CONFIDENCE:
            return INTENT_BLOCK, "confidence_below_floor"
        if persistence_decision == PERSISTENCE_REJECT:
            return INTENT_BLOCK, "persistence_rejected"

    # ── C. SKIP ──────────────────────────────────────────────────
    if rebalance_action in {REBAL_HOLD, REBAL_NO_ACTION}:
        return INTENT_SKIP, "non_actionable_row"
    if not execution_ready:
        # Propagate the upstream not-ready reason if we can pluck it out
        # of Step 6's reason string (e.g. below_min_trade).
        upstream = ""
        if "below_min_trade" in rebalance_reason:
            upstream = ":below_trade_minimum"
        elif "severe_risk_lock_for_buy" in rebalance_reason:
            upstream = ":severe_risk_lock_for_buy"
        elif "blocked" in rebalance_reason:
            upstream = ":upstream_blocked"
        return INTENT_SKIP, f"upstream_not_ready{upstream}"

    # ── For risk-protective sells: execute when ready ────────────
    if is_sell:
        # Sells driven by the risk overlay are exactly what the engine
        # *wants* to execute. Confidence/persistence are intentionally
        # not required — a deteriorating position should be sold even
        # if conviction is weak.
        if RISK_FORCE_EXIT in risk_components or rebalance_action == REBAL_FULL_EXIT:
            return INTENT_EXECUTE, "risk_protective_full_exit"
        if RISK_TRIM_PRIORITY in risk_components or rebalance_action == REBAL_TRIM:
            return INTENT_EXECUTE, "risk_protective_trim"
        # Plain SELL not tied to a risk flag (rare; Step 6 produces this
        # when a HOLD has a non-zero delta toward smaller). Treat as
        # ready-to-execute risk-neutral sell.
        return INTENT_EXECUTE, "discretionary_sell"

    # ── A / B for buys ──────────────────────────────────────────
    if is_buy:
        # DELAY: borderline or weakening (positive but not strong enough)
        if (
            persistence_decision == PERSISTENCE_DEMOTE
            or (confidence is not None and confidence < BORDERLINE_CONFIDENCE_HIGH)
            or (persistence_score is not None and persistence_score < MIN_EXECUTE_PERSISTENCE)
        ):
            if (
                confidence is not None
                and BORDERLINE_CONFIDENCE_LOW <= confidence < BORDERLINE_CONFIDENCE_HIGH
            ):
                return INTENT_DELAY, "confidence_borderline"
            if (
                persistence_score is not None
                and DELAY_PERSISTENCE_LOW <= persistence_score < MIN_EXECUTE_PERSISTENCE
            ):
                return INTENT_DELAY, "persistence_weak"
            if persistence_decision == PERSISTENCE_DEMOTE:
                return INTENT_DELAY, "persistence_deteriorating"
            return INTENT_DELAY, "below_execution_threshold"

        # EXECUTE_NOW gates
        if (
            confidence is not None
            and confidence >= MIN_EXECUTE_CONFIDENCE
            and persistence_score is not None
            and persistence_score >= MIN_EXECUTE_PERSISTENCE
            and delta_pct is not None
            and delta_pct > 0
        ):
            return INTENT_EXECUTE, "high_conviction_execute"

        # Defensive fallback (shouldn't reach here normally).
        return INTENT_DELAY, "missing_conviction_data"

    return INTENT_SKIP, "unhandled_action"


def _intent_score(
    *,
    priority: float,
    confidence: Optional[float],
    persistence_score: Optional[float],
    delta_pct: Optional[float],
    signal: str,
    lifecycle_action: str,
    risk_components: List[str],
    rebalance_action: str,
) -> float:
    """
    Convex combination of weighted conviction signals minus risk
    penalty. Returns a [0,1] score where 1.0 means "everything aligned
    to execute right now".

    Sells are weighted more heavily on `priority` (which already encodes
    the risk-tier band from Step 6), and lightly on conviction signals
    so a high-priority FORCE_EXIT scores ≈1.0 even with no positive
    conviction inputs.
    """
    priority_n = _clamp(priority, 0.0, 1.0)
    confidence_n = _clamp(_to_float_or_zero(confidence), 0.0, 1.0)
    persistence_n = _clamp(_to_float_or_zero(persistence_score), 0.0, 1.0)
    # delta in roughly ±5% range; map (+0.005 ~ +0.05) into a [0,1] boost.
    delta_n = _clamp((_to_float_or_zero(delta_pct) * 20.0 + 1.0) * 0.5, 0.0, 1.0)

    signal_confirmation = (
        1.0 if signal in POSITIVE_LABELS else (0.0 if signal in NEGATIVE_LABELS else 0.5)
    )
    lifecycle_confirmation = (
        1.0
        if lifecycle_action in POSITIVE_LABELS
        else (0.0 if lifecycle_action in NEGATIVE_LABELS else 0.5)
    )

    risk_penalty = 0.0
    if rebalance_action in BUY_DIRECTION_ACTIONS:
        if RISK_FORCE_EXIT in risk_components:
            risk_penalty += 0.50
        if RISK_BLOCK_NEW_BUY in risk_components:
            risk_penalty += 0.40
        if RISK_TRIM_PRIORITY in risk_components:
            risk_penalty += 0.15

    if rebalance_action in SELL_DIRECTION_ACTIONS:
        # Sell scoring: priority-dominant, signal direction doesn't help
        # (since selling a "BUY" name is unusual but allowed for risk).
        score = (
            0.60 * priority_n
            + 0.15 * confidence_n
            + 0.10 * persistence_n
            + 0.10 * (1.0 - signal_confirmation)  # selling = aligned with negative signal
            + 0.05 * (1.0 - lifecycle_confirmation)
        )
    elif rebalance_action in BUY_DIRECTION_ACTIONS:
        score = (
            0.30 * priority_n
            + 0.20 * confidence_n
            + 0.20 * persistence_n
            + 0.10 * delta_n
            + 0.10 * signal_confirmation
            + 0.10 * lifecycle_confirmation
            - risk_penalty
        )
    else:
        # HOLD / NO_ACTION — no intent to execute.
        score = 0.0

    return round(_clamp(score, 0.0, 1.0), 6)


# -----------------------------------------------------------
# Pipeline
# -----------------------------------------------------------
def build_execution_intents(
    *,
    rebalance_rows: List[Dict[str, Any]],
    risk_overlay_map: Dict[str, str],
    persistence_map: Dict[str, Dict[str, Any]],
    signals_map: Dict[str, Dict[str, Any]],
    lifecycle_map: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Pure planner — no IO. Returns (intents_df, summary)."""
    rows: List[Dict[str, Any]] = []

    for r in rebalance_rows:
        sym = r["ticker"]
        rebalance_action = r["rebalance_action"]
        execution_ready = bool(r["execution_ready"])
        priority = float(r.get("priority") or 0.0)
        amount = float(r.get("rebalance_amount_usd") or 0.0)
        rebalance_reason = r.get("rebalance_reason", "")

        # ── Enrichment with documented precedence ──────────────────
        persist = persistence_map.get(sym, {})
        lc = lifecycle_map.get(sym, {})
        sig = signals_map.get(sym, {})
        risk_flag = risk_overlay_map.get(sym, RISK_OK) or RISK_OK
        risk_comps = _risk_components(risk_flag)

        confidence = (
            persist.get("latest_confidence")
            if persist.get("latest_confidence") is not None
            else (lc.get("confidence") or sig.get("confidence"))
        )
        delta_pct = (
            persist.get("latest_delta_pct")
            if persist.get("latest_delta_pct") is not None
            else (lc.get("delta_pct") or sig.get("delta_pct"))
        )
        persistence_score = persist.get("persistence_score")
        persistence_decision = persist.get("promotion_decision", "") or ""
        signal = lc.get("signal") or sig.get("signal") or ""
        lifecycle_action = lc.get("lifecycle_action") or ""

        intent, intent_reason = _decide_intent(
            rebalance_action=rebalance_action,
            execution_ready=execution_ready,
            confidence=confidence,
            persistence_score=persistence_score,
            delta_pct=delta_pct,
            signal=signal,
            lifecycle_action=lifecycle_action,
            risk_components=risk_comps,
            persistence_decision=persistence_decision,
            rebalance_reason=rebalance_reason,
        )

        score = _intent_score(
            priority=priority,
            confidence=confidence,
            persistence_score=persistence_score,
            delta_pct=delta_pct,
            signal=signal,
            lifecycle_action=lifecycle_action,
            risk_components=risk_comps,
            rebalance_action=rebalance_action,
        )

        # Step 11: optional intent_score floor from runtime policy.
        # Only demotes BUY-direction EXECUTE_NOW rows. Risk-protective
        # sells are exempt so the deploy gate cannot stop a forced
        # exit. Default floor is 0.0 -> no behaviour change.
        if (
            MIN_EXECUTE_INTENT_SCORE > 0.0
            and intent == INTENT_EXECUTE
            and rebalance_action in BUY_DIRECTION_ACTIONS
            and score < MIN_EXECUTE_INTENT_SCORE
        ):
            intent = INTENT_DELAY
            intent_reason = (
                f"intent_score_below_runtime_floor:" f"{score:.2f}<{MIN_EXECUTE_INTENT_SCORE:.2f}"
            )

        # Build a self-describing reason that pairs the intent decision
        # with the original Step 6 rebalance reason so downstream tools
        # can render full provenance without rejoining files.
        reason_parts = [intent_reason]
        if rebalance_reason:
            reason_parts.append(f"from:{rebalance_reason}")

        rows.append(
            {
                "ticker": sym,
                "rebalance_action": rebalance_action,
                "rebalance_amount_usd": round(amount, 2),
                "priority": round(priority, 6),
                "execution_ready": bool(execution_ready),
                "execution_intent": intent,
                "confidence": (round(float(confidence), 6) if confidence is not None else None),
                "persistence_score": (
                    round(float(persistence_score), 6) if persistence_score is not None else None
                ),
                "delta_pct": (round(float(delta_pct), 6) if delta_pct is not None else None),
                "lifecycle_action": lifecycle_action,
                "signal": signal,
                "risk_flag": risk_flag,
                "intent_score": score,
                "reason": "|".join(reason_parts),
                "_execution_order": int(r.get("execution_order") or 0),
            }
        )

    # Preserve Step 6 sequence (do NOT re-sort by intent_score; the
    # caller — including a future executor — must see the same trade
    # order the rebalance plan produced).
    rows.sort(key=lambda r: (r["_execution_order"], r["ticker"]))
    for r in rows:
        r.pop("_execution_order", None)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    def _count(intent: str) -> int:
        return int(sum(1 for r in rows if r["execution_intent"] == intent))

    execute_n = _count(INTENT_EXECUTE)
    delay_n = _count(INTENT_DELAY)
    skip_n = _count(INTENT_SKIP)
    block_n = _count(INTENT_BLOCK)

    avg_intent_score = (
        round(
            float(sum(float(r["intent_score"]) for r in rows) / max(1, len(rows))),
            6,
        )
        if rows
        else 0.0
    )

    # Top execution candidates: EXECUTE_NOW only, ranked by intent_score
    # desc with priority and amount as tiebreakers. Cap at 10.
    execute_rows = [r for r in rows if r["execution_intent"] == INTENT_EXECUTE]
    execute_rows.sort(
        key=lambda r: (
            float(r["intent_score"]),
            float(r["priority"]),
            abs(float(r["rebalance_amount_usd"])),
        ),
        reverse=True,
    )
    top_execution_candidates: List[Dict[str, Any]] = [
        {
            "ticker": r["ticker"],
            "rebalance_action": r["rebalance_action"],
            "rebalance_amount_usd": float(r["rebalance_amount_usd"]),
            "intent_score": float(r["intent_score"]),
            "priority": float(r["priority"]),
            "confidence": (float(r["confidence"]) if r["confidence"] is not None else None),
            "persistence_score": (
                float(r["persistence_score"]) if r["persistence_score"] is not None else None
            ),
            "reason": r["reason"],
        }
        for r in execute_rows[:10]
    ]

    total_execute_notional = sum(
        abs(float(r["rebalance_amount_usd"]))
        for r in rows
        if r["execution_intent"] == INTENT_EXECUTE
    )

    summary: Dict[str, Any] = {
        "generated_at_utc": _now_iso_utc(),
        "total_actions": int(len(rows)),
        "execute_now": execute_n,
        "delayed": delay_n,
        "skipped": skip_n,
        "blocked": block_n,
        "avg_intent_score": avg_intent_score,
        "execute_now_notional_usd": round(float(total_execute_notional), 2),
        "thresholds": {
            "min_execute_confidence": MIN_EXECUTE_CONFIDENCE,
            "min_execute_persistence": MIN_EXECUTE_PERSISTENCE,
            "min_keep_confidence": MIN_KEEP_CONFIDENCE,
            "borderline_confidence_low": BORDERLINE_CONFIDENCE_LOW,
            "borderline_confidence_high": BORDERLINE_CONFIDENCE_HIGH,
            "delay_persistence_low": DELAY_PERSISTENCE_LOW,
        },
        "top_execution_candidates": top_execution_candidates,
    }
    return df, summary


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only execution intent engine (step 7 of WATCH funnel). "
            "Decides whether each rebalance row should EXECUTE_NOW / DELAY / "
            "SKIP / BLOCK; does NOT place any trades."
        ),
    )
    p.add_argument("--rebalance", default=str(DEFAULT_REBALANCE_CSV))
    p.add_argument("--rebalance-summary", default=str(DEFAULT_REBALANCE_JSON))
    p.add_argument("--risk-overlay", default=str(DEFAULT_RISK_OVERLAY_CSV))
    p.add_argument("--persistence", default=str(DEFAULT_PERSISTENCE_CSV))
    p.add_argument("--signals", default=str(DEFAULT_SIGNALS_CSV))
    p.add_argument("--lifecycle-effective", default=str(DEFAULT_LIFECYCLE_EFFECTIVE_CSV))
    p.add_argument("--out-csv", default=str(DEFAULT_OUTPUT_CSV))
    p.add_argument("--out-json", default=str(DEFAULT_OUTPUT_JSON))
    return p.parse_args(argv)


def _apply_runtime_policy(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Step 11 integration. Reads runtime_policy.json (if present) and
    overrides MIN_EXECUTE_INTENT_SCORE (the intent-score floor for BUY
    executes). Safe to call every cycle -- missing/malformed file
    leaves the floor at 0.0, which is OFF (no behaviour change).
    Path resolves via module attribute at call time so tests can
    monkey-patch ``DEFAULT_RUNTIME_POLICY_JSON``.
    """
    global MIN_EXECUTE_INTENT_SCORE
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
            f"[EXECUTION_INTENT_WARN] runtime_policy.json present but unreadable "
            f"({type(e).__name__}: {e}); keeping defaults",
            flush=True,
        )
        return None
    aliases = rp.get("aliases") or {}
    v = rp.get("deployment_threshold")
    if v is None:
        v = aliases.get("min_execute_intent_score")
    if v is not None:
        try:
            MIN_EXECUTE_INTENT_SCORE = max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            pass
    print(
        "[EXECUTION_INTENT_POLICY] "
        f"regime={rp.get('regime', 'UNKNOWN')} "
        f"min_execute_intent_score={MIN_EXECUTE_INTENT_SCORE:.2f}",
        flush=True,
    )
    return rp


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[EXECUTION_INTENT] starting (read-only final-gate engine)", flush=True)
    _apply_runtime_policy()

    rebalance_df = _safe_read_csv(Path(args.rebalance), label="portfolio_rebalance_plan.csv")
    _ = _safe_read_json(Path(args.rebalance_summary), label="portfolio_rebalance_summary.json")
    risk_df = _safe_read_csv(Path(args.risk_overlay), label="performance_risk_overlay.csv")
    persistence_df = _safe_read_csv(
        Path(args.persistence), label="opportunity_persistence_recommendations.csv"
    )
    signals_df = _safe_read_csv(Path(args.signals), label="signals_with_rationale.csv")
    lifecycle_df = _safe_read_csv(
        Path(args.lifecycle_effective), label="signal_lifecycle_effective.csv"
    )

    rebalance_rows = _load_rebalance_rows(rebalance_df)
    risk_overlay_map = _load_risk_overlay_map(risk_df)
    persistence_map = _load_persistence_map(persistence_df)
    signals_map = _load_signals_map(signals_df)
    lifecycle_map = _load_lifecycle_map(lifecycle_df)

    df, summary = build_execution_intents(
        rebalance_rows=rebalance_rows,
        risk_overlay_map=risk_overlay_map,
        persistence_map=persistence_map,
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
        "[EXECUTION_INTENT] "
        f"actions={summary['total_actions']} "
        f"execute={summary['execute_now']} "
        f"delay={summary['delayed']} "
        f"skip={summary['skipped']} "
        f"block={summary['blocked']} "
        f"avg_intent_score={summary['avg_intent_score']:.3f} "
        f"execute_notional=${summary['execute_now_notional_usd']:.2f}",
        flush=True,
    )
    print(
        "[EXECUTION_TOP] symbols=" f"{[c['ticker'] for c in summary['top_execution_candidates']]}",
        flush=True,
    )
    print(
        f"[EXECUTION_INTENT_OUT] csv={out_csv.as_posix()} summary={out_json.as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
