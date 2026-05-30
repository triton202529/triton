"""
Opportunity Persistence Engine — Step 3 of the WATCH → OPEN_NEW funnel.

Reads:
    data/results/opportunity_promotion_recommendations.csv
    data/results/opportunity_promotion_summary.json
    data/results/portfolio_allocation_recommendations.csv   (informational)
    data/results/signals_with_rationale.csv                 (informational)
    data/results/signal_lifecycle_effective.csv             (informational)

State (append-only):
    data/results/opportunity_watch_history.parquet
    data/results/opportunity_watch_history.csv              (optional export)

Writes:
    data/results/opportunity_persistence_recommendations.csv
    data/results/opportunity_persistence_summary.json

Purpose
-------
Step 2 (services.opportunity_promotion_engine) makes per-cycle WATCH /
PROMOTE / REJECT calls. This engine layers historical persistence on top:
a symbol must demonstrate strengthening conviction across MULTIPLE
cycles before it earns `PROMOTE_CONFIRMED_OPEN_NEW`. This rewards
improving signal quality and damps one-cycle false positives.

Per cycle the engine:
    1. Loads the current promotion recommendations (input).
    2. Appends one row per current WATCH symbol to the append-only
       parquet history file (deduped by (ticker, timestamp_utc)).
    3. Recomputes per-ticker rolling persistence metrics from history.
    4. Applies the persistence-based decision precedence:

       REJECT first      → confidence below keep floor OR no positive
                           signal/lifecycle OR repeated deterioration.
       DEMOTE_WATCH      → delta turned non-positive OR confidence
                           trend falling materially.
       PROMOTE_CONFIRMED → ≥2 consecutive WATCH cycles AND latest
                           confidence ≥ 0.50 AND latest delta > 0 AND
                           promotion_score trend improving AND
                           persistence_score ≥ 0.60 AND signal/lifecycle
                           positive.
       KEEP_WATCH        → positive but at least one persistence gate
                           is not yet strong enough.

    5. Writes outputs (CSV per ticker + JSON summary) atomically.

Safety
------
* Read-only with respect to broker / trading layers. No orders are
  placed; manage_positions and execute_trades are not invoked or
  modified.
* Missing inputs warn and continue (each loader yields an empty df).
* History parquet writes are atomic; a malformed history file is
  detected and the run starts fresh rather than crashing.
* main() returns 0 on success, 2 only on output-write failures.
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

import numpy as np
import pandas as pd

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_PROMOTION_CSV = RESULTS_DIR / "opportunity_promotion_recommendations.csv"
DEFAULT_PROMOTION_JSON = RESULTS_DIR / "opportunity_promotion_summary.json"
DEFAULT_ALLOCATION_CSV = RESULTS_DIR / "portfolio_allocation_recommendations.csv"
DEFAULT_SIGNALS_CSV = RESULTS_DIR / "signals_with_rationale.csv"
DEFAULT_LIFECYCLE_EFFECTIVE_CSV = RESULTS_DIR / "signal_lifecycle_effective.csv"

DEFAULT_HISTORY_PARQUET = RESULTS_DIR / "opportunity_watch_history.parquet"
DEFAULT_HISTORY_CSV = RESULTS_DIR / "opportunity_watch_history.csv"

DEFAULT_OUTPUT_CSV = RESULTS_DIR / "opportunity_persistence_recommendations.csv"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "opportunity_persistence_summary.json"

# -----------------------------------------------------------
# Thresholds / labels (analytics-only)
# -----------------------------------------------------------
DECISION_PROMOTE_CONFIRMED = "PROMOTE_CONFIRMED_OPEN_NEW"
DECISION_KEEP = "KEEP_WATCH"
DECISION_DEMOTE = "DEMOTE_WATCH"
DECISION_REJECT = "REJECT"

# Spec gates.
MIN_CONSECUTIVE_FOR_PROMOTE = 2
MIN_PROMOTE_CONFIDENCE = 0.50
MIN_PROMOTE_PERSISTENCE_SCORE = 0.60

MIN_KEEP_CONFIDENCE = 0.45
CONFIDENCE_FALL_THRESHOLD = -0.05  # confidence_trend below this = falling materially
REPEATED_DETERIORATION_PERSISTENCE = 0.30  # persistence below this w/ consec>=2 = REJECT

# Rolling-window cap for trend math.
HISTORY_LOOKBACK_CYCLES = 30

# Consecutive-cycle saturation point for persistence bonus.
CONSEC_BONUS_SATURATION = 5

# Labels treated as positive intent.
POSITIVE_SIGNAL_LABELS: frozenset = frozenset({"BUY", "ADD"})
LIFECYCLE_ACTION_POSITIVE: frozenset = frozenset({"BUY", "ADD"})

# Promotion-engine decisions that count as a "watch cycle" for the
# consecutive-streak counter. PROMOTE_OPEN_NEW (step 2's promote) also
# counts so a ticker that briefly graduated to PROMOTE then dropped back
# to KEEP_WATCH keeps its streak.
WATCH_STREAK_DECISIONS: frozenset = frozenset(
    {"KEEP_WATCH", "PROMOTE_OPEN_NEW", DECISION_PROMOTE_CONFIRMED}
)

HISTORY_COLUMNS = [
    "timestamp_utc",
    "ticker",
    "allocation_score",
    "promotion_score",
    "confidence",
    "delta_pct",
    "signal",
    "lifecycle_action",
    "effective_stance",
    "promotion_decision",
]

OUTPUT_COLUMNS = [
    "ticker",
    "watch_cycles",
    "consecutive_watch_cycles",
    "avg_confidence",
    "latest_confidence",
    "confidence_trend",
    "avg_delta_pct",
    "latest_delta_pct",
    "delta_trend",
    "avg_promotion_score",
    "latest_promotion_score",
    "promotion_score_trend",
    "signal_consistency",
    "lifecycle_consistency",
    "persistence_score",
    "promotion_decision",
    "reason",
]


# -----------------------------------------------------------
# Safe IO helpers
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[OPPORTUNITY_PERSISTENCE_WARN] {msg}", flush=True)


def _safe_read_csv(path: Path, *, label: str) -> pd.DataFrame:
    """Return a DataFrame; never raises. Missing/empty/unreadable -> empty df."""
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


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    """Return history DataFrame; empty (with HISTORY_COLUMNS) on any failure."""
    try:
        if not path.is_file():
            return pd.DataFrame(columns=HISTORY_COLUMNS)
    except OSError:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        _warn(
            f"failed to read history parquet ({path}): {type(e).__name__}: {e}; "
            f"starting fresh history"
        )
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    for c in HISTORY_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[HISTORY_COLUMNS].copy()


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
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
# History management
# -----------------------------------------------------------
def _current_cycle_rows(promo_df: pd.DataFrame, cycle_ts: str) -> pd.DataFrame:
    """Project the promotion CSV into history-schema rows for this cycle."""
    if promo_df is None or promo_df.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    sym_col = _pick_first_present(promo_df, ("ticker", "symbol"))
    if not sym_col:
        _warn("opportunity_promotion_recommendations.csv missing ticker/symbol column")
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for _, r in promo_df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        rows.append(
            {
                "timestamp_utc": cycle_ts,
                "ticker": sym,
                "allocation_score": _to_float(r.get("allocation_score")),
                "promotion_score": _to_float(r.get("promotion_score")),
                "confidence": _to_float(r.get("confidence")),
                "delta_pct": _to_float(r.get("delta_pct")),
                "signal": _norm_upper(r.get("signal")),
                "lifecycle_action": _norm_upper(r.get("lifecycle_action")),
                "effective_stance": _norm_upper(r.get("effective_stance")),
                "promotion_decision": _norm_upper(r.get("promotion_decision")),
            }
        )
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def _append_cycle_to_history(history_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append current cycle to history. Dedupes on (ticker, timestamp_utc)
    with last-write-wins so a re-run within the same cycle replaces stale
    rows rather than double-counting them.
    """
    if current_df is None or current_df.empty:
        return history_df.copy()

    # Avoid pandas' "concat with empty/all-NA entries" FutureWarning by
    # skipping the concat when history is empty — current rows are already
    # in the canonical history schema.
    if history_df is None or history_df.empty:
        combined = current_df.copy()
    else:
        combined = pd.concat([history_df, current_df], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(
        subset=["ticker", "timestamp_utc"], keep="last"
    ).reset_index(drop=True)
    return combined[HISTORY_COLUMNS].copy()


# -----------------------------------------------------------
# Trend / metric helpers
# -----------------------------------------------------------
def _finite_floats(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        f = _to_float(v)
        if f is not None:
            out.append(f)
    return out


def _mean_or_none(values: Iterable[Any]) -> Optional[float]:
    arr = _finite_floats(values)
    if not arr:
        return None
    return float(np.mean(arr))


def _slope(values: Iterable[Any]) -> float:
    """Linear regression slope across the input. Returns 0.0 when <2 finite values."""
    arr = _finite_floats(values)
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    try:
        slope, _ = np.polyfit(x, np.asarray(arr, dtype=float), 1)
    except Exception:
        return 0.0
    return float(slope) if np.isfinite(slope) else 0.0


def _trend_norm(slope: float, scale: float) -> float:
    """
    Map a regression slope to [0, 1] for the persistence score.
    0.5 means flat; <0.5 falling; >0.5 improving.
    `scale` is chosen so a "meaningful" slope reaches the [0,1] bounds.
    """
    return _clamp(0.5 + slope * scale, 0.0, 1.0)


def _consistency(values: Iterable[Any], positive_set: frozenset) -> float:
    """Fraction of non-empty values that fall in `positive_set` (0..1)."""
    total = 0
    hits = 0
    for v in values:
        s = _norm_upper(v)
        if not s:
            continue
        total += 1
        if s in positive_set:
            hits += 1
    if total == 0:
        return 0.0
    return float(hits) / float(total)


# -----------------------------------------------------------
# Per-ticker metrics
# -----------------------------------------------------------
def _compute_ticker_metrics(
    history_df: pd.DataFrame,
    ticker: str,
    all_cycles_desc: List[str],
) -> Dict[str, Any]:
    """
    Compute the rolling persistence metrics for one ticker.

    `all_cycles_desc` is the global list of distinct cycle timestamps
    sorted descending (most recent first) — used to count consecutive
    cycles from the most recent run.
    """
    sub = history_df[history_df["ticker"] == ticker].copy()
    if sub.empty:
        return {
            "watch_cycles": 0,
            "consecutive_watch_cycles": 0,
            "avg_confidence": None,
            "latest_confidence": None,
            "confidence_trend": 0.0,
            "avg_delta_pct": None,
            "latest_delta_pct": None,
            "delta_trend": 0.0,
            "avg_promotion_score": None,
            "latest_promotion_score": None,
            "promotion_score_trend": 0.0,
            "signal_consistency": 0.0,
            "lifecycle_consistency": 0.0,
            "persistence_score": 0.0,
            "latest_signal": "",
            "latest_lifecycle_action": "",
        }

    sub = sub.sort_values("timestamp_utc")
    if len(sub) > HISTORY_LOOKBACK_CYCLES:
        sub = sub.tail(HISTORY_LOOKBACK_CYCLES).copy()
    latest = sub.iloc[-1]

    # Consecutive: walk the global cycle list from most-recent to oldest,
    # counting how many consecutive cycles this ticker appears with a
    # WATCH-streak decision. Stop at first miss or non-streak decision.
    by_ts: Dict[str, str] = {
        str(r["timestamp_utc"]): _norm_upper(r["promotion_decision"]) for _, r in sub.iterrows()
    }
    consec = 0
    for ts in all_cycles_desc:
        decision = by_ts.get(ts)
        if decision is None:
            break
        if decision not in WATCH_STREAK_DECISIONS:
            break
        consec += 1

    conf_vals = sub["confidence"].tolist()
    delta_vals = sub["delta_pct"].tolist()
    pscore_vals = sub["promotion_score"].tolist()

    avg_conf = _mean_or_none(conf_vals)
    avg_delta = _mean_or_none(delta_vals)
    avg_pscore = _mean_or_none(pscore_vals)
    latest_conf = _to_float(latest.get("confidence"))
    latest_delta = _to_float(latest.get("delta_pct"))
    latest_pscore = _to_float(latest.get("promotion_score"))

    conf_trend = _slope(conf_vals)
    delta_trend = _slope(delta_vals)
    pscore_trend = _slope(pscore_vals)

    sig_consistency = _consistency(sub["signal"].tolist(), POSITIVE_SIGNAL_LABELS)
    lc_consistency = _consistency(sub["lifecycle_action"].tolist(), LIFECYCLE_ACTION_POSITIVE)

    # Normalize trends and combine into persistence_score in [0, 1].
    # Scales: promotion_score deltas ~ 0.1/cycle, confidence ~ 0.05/cycle,
    # delta_pct ~ 0.005/cycle. Pick scale = 1/typical so a "typical full
    # improvement" pushes the term to ~1.0.
    p_trend_norm = _trend_norm(pscore_trend, scale=5.0)
    c_trend_norm = _trend_norm(conf_trend, scale=10.0)
    d_trend_norm = _trend_norm(delta_trend, scale=100.0)
    persistence_bonus = _clamp(consec / float(CONSEC_BONUS_SATURATION), 0.0, 1.0)

    persistence_score = round(
        0.30 * p_trend_norm
        + 0.25 * c_trend_norm
        + 0.20 * d_trend_norm
        + 0.15 * sig_consistency
        + 0.10 * persistence_bonus,
        6,
    )

    return {
        "watch_cycles": int(len(sub)),
        "consecutive_watch_cycles": int(consec),
        "avg_confidence": (round(avg_conf, 6) if avg_conf is not None else None),
        "latest_confidence": (round(latest_conf, 6) if latest_conf is not None else None),
        "confidence_trend": round(conf_trend, 6),
        "avg_delta_pct": (round(avg_delta, 6) if avg_delta is not None else None),
        "latest_delta_pct": (round(latest_delta, 6) if latest_delta is not None else None),
        "delta_trend": round(delta_trend, 6),
        "avg_promotion_score": (round(avg_pscore, 6) if avg_pscore is not None else None),
        "latest_promotion_score": (round(latest_pscore, 6) if latest_pscore is not None else None),
        "promotion_score_trend": round(pscore_trend, 6),
        "signal_consistency": round(sig_consistency, 6),
        "lifecycle_consistency": round(lc_consistency, 6),
        "persistence_score": persistence_score,
        "latest_signal": _norm_upper(latest.get("signal")),
        "latest_lifecycle_action": _norm_upper(latest.get("lifecycle_action")),
    }


# -----------------------------------------------------------
# Decision
# -----------------------------------------------------------
def _decide_persistence(metrics: Dict[str, Any]) -> Tuple[str, str]:
    """
    Persistence-based decision precedence: REJECT → DEMOTE → PROMOTE → KEEP.
    """
    latest_conf = metrics.get("latest_confidence")
    latest_delta = metrics.get("latest_delta_pct")
    latest_sig = metrics.get("latest_signal") or ""
    latest_lc = metrics.get("latest_lifecycle_action") or ""
    consec = int(metrics.get("consecutive_watch_cycles") or 0)
    conf_trend = float(metrics.get("confidence_trend") or 0.0)
    pscore_trend = float(metrics.get("promotion_score_trend") or 0.0)
    persistence = float(metrics.get("persistence_score") or 0.0)

    pos_sig = latest_sig in POSITIVE_SIGNAL_LABELS
    pos_lc = latest_lc in LIFECYCLE_ACTION_POSITIVE
    pos_either = pos_sig or pos_lc

    # ── D: REJECT (highest precedence) ──────────────────────────────
    if latest_conf is None or latest_conf < MIN_KEEP_CONFIDENCE:
        return DECISION_REJECT, "negative_signal:confidence_below_keep_threshold"
    if not pos_either:
        return DECISION_REJECT, "negative_signal:no_positive_signal_or_lifecycle"
    if consec >= MIN_CONSECUTIVE_FOR_PROMOTE and persistence < REPEATED_DETERIORATION_PERSISTENCE:
        return DECISION_REJECT, "repeated_deterioration"

    # ── C: DEMOTE_WATCH (currently weakening) ───────────────────────
    if latest_delta is None or latest_delta <= 0:
        return DECISION_DEMOTE, "weakening_delta"
    if conf_trend < CONFIDENCE_FALL_THRESHOLD:
        return DECISION_DEMOTE, "confidence_not_improving"

    # ── A: PROMOTE_CONFIRMED_OPEN_NEW ───────────────────────────────
    promote_ok = (
        consec >= MIN_CONSECUTIVE_FOR_PROMOTE
        and latest_conf >= MIN_PROMOTE_CONFIDENCE
        and latest_delta > 0
        and pscore_trend > 0
        and persistence >= MIN_PROMOTE_PERSISTENCE_SCORE
        and pos_either
    )
    if promote_ok:
        return DECISION_PROMOTE_CONFIRMED, "strengthening_signal_persistence"

    # ── B: KEEP_WATCH (positive but a persistence gate misses) ─────
    why: List[str] = []
    if consec < MIN_CONSECUTIVE_FOR_PROMOTE:
        why.append(f"insufficient_cycles ({consec}<{MIN_CONSECUTIVE_FOR_PROMOTE})")
    if latest_conf < MIN_PROMOTE_CONFIDENCE:
        why.append(f"conf<{MIN_PROMOTE_CONFIDENCE}")
    if pscore_trend <= 0:
        why.append("confidence_not_improving")
    if persistence < MIN_PROMOTE_PERSISTENCE_SCORE:
        why.append(f"persistence<{MIN_PROMOTE_PERSISTENCE_SCORE}")
    return DECISION_KEEP, "promotion_threshold_not_met:" + (",".join(why) or "insufficient_cycles")


# -----------------------------------------------------------
# Aggregation
# -----------------------------------------------------------
def build_persistence(
    *,
    history_df: pd.DataFrame,
    cycle_ts: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute per-ticker persistence metrics + decisions for every ticker
    present in the CURRENT cycle. (Tickers in history that didn't appear
    this cycle are not reported — they're not actionable right now.)
    """
    if history_df is None or history_df.empty:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty, _empty_summary(cycle_ts, watch_n=0)

    current_tickers = sorted(
        set(
            history_df.loc[history_df["timestamp_utc"] == cycle_ts, "ticker"]
            .astype(str)
            .str.strip()
            .tolist()
        )
    )
    if not current_tickers:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty, _empty_summary(cycle_ts, watch_n=0)

    all_cycles_desc = sorted(
        {str(t) for t in history_df["timestamp_utc"].astype(str).tolist() if t},
        reverse=True,
    )

    rows: List[Dict[str, Any]] = []
    for sym in current_tickers:
        metrics = _compute_ticker_metrics(history_df, sym, all_cycles_desc)
        decision, reason = _decide_persistence(metrics)
        rows.append(
            {
                "ticker": sym,
                "watch_cycles": metrics["watch_cycles"],
                "consecutive_watch_cycles": metrics["consecutive_watch_cycles"],
                "avg_confidence": metrics["avg_confidence"],
                "latest_confidence": metrics["latest_confidence"],
                "confidence_trend": metrics["confidence_trend"],
                "avg_delta_pct": metrics["avg_delta_pct"],
                "latest_delta_pct": metrics["latest_delta_pct"],
                "delta_trend": metrics["delta_trend"],
                "avg_promotion_score": metrics["avg_promotion_score"],
                "latest_promotion_score": metrics["latest_promotion_score"],
                "promotion_score_trend": metrics["promotion_score_trend"],
                "signal_consistency": metrics["signal_consistency"],
                "lifecycle_consistency": metrics["lifecycle_consistency"],
                "persistence_score": metrics["persistence_score"],
                "promotion_decision": decision,
                "reason": reason,
            }
        )

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    def _count(dec: str) -> int:
        if df.empty:
            return 0
        return int((df["promotion_decision"] == dec).sum())

    promoted = _count(DECISION_PROMOTE_CONFIRMED)
    kept = _count(DECISION_KEEP)
    demoted = _count(DECISION_DEMOTE)
    rejected = _count(DECISION_REJECT)

    avg_persistence = (
        float(round(df["persistence_score"].astype(float).mean(), 6)) if not df.empty else 0.0
    )

    # Strongest = highest persistence_score, prefer PROMOTE then KEEP.
    # Weakest = lowest, prefer REJECT then DEMOTE.
    def _ranked(df_in: pd.DataFrame, ascending: bool, classes: List[str]) -> List[Dict[str, Any]]:
        if df_in.empty:
            return []
        d = df_in.copy()
        d["__class_rank"] = d["promotion_decision"].apply(
            lambda x: classes.index(x) if x in classes else len(classes)
        )
        d["__pscore"] = pd.to_numeric(d["persistence_score"], errors="coerce").fillna(
            -1e9 if not ascending else 1e9
        )
        d["__conf"] = pd.to_numeric(d["latest_confidence"], errors="coerce").fillna(
            -1e9 if not ascending else 1e9
        )
        d = d.sort_values(
            by=["__class_rank", "__pscore", "__conf", "ticker"],
            ascending=[True, ascending, ascending, True],
        )
        out: List[Dict[str, Any]] = []
        for _, r in d.head(5).iterrows():
            out.append(
                {
                    "ticker": str(r["ticker"]),
                    "promotion_decision": str(r["promotion_decision"]),
                    "persistence_score": float(r["__pscore"]),
                    "latest_confidence": (
                        float(r["latest_confidence"]) if pd.notna(r["latest_confidence"]) else None
                    ),
                    "latest_delta_pct": (
                        float(r["latest_delta_pct"]) if pd.notna(r["latest_delta_pct"]) else None
                    ),
                    "consecutive_watch_cycles": int(r["consecutive_watch_cycles"]),
                    "reason": str(r["reason"]),
                }
            )
        return out

    strongest = _ranked(
        df,
        ascending=False,
        classes=[DECISION_PROMOTE_CONFIRMED, DECISION_KEEP, DECISION_DEMOTE, DECISION_REJECT],
    )
    weakest = _ranked(
        df,
        ascending=True,
        classes=[DECISION_REJECT, DECISION_DEMOTE, DECISION_KEEP, DECISION_PROMOTE_CONFIRMED],
    )

    summary: Dict[str, Any] = {
        "generated_at_utc": _now_iso_utc(),
        "cycle_timestamp_utc": cycle_ts,
        "watch_candidates": int(len(df)),
        "promoted_confirmed": promoted,
        "kept_watch": kept,
        "demoted": demoted,
        "rejected": rejected,
        "avg_persistence_score": avg_persistence,
        "thresholds": {
            "min_consecutive_for_promote": MIN_CONSECUTIVE_FOR_PROMOTE,
            "min_promote_confidence": MIN_PROMOTE_CONFIDENCE,
            "min_promote_persistence_score": MIN_PROMOTE_PERSISTENCE_SCORE,
            "min_keep_confidence": MIN_KEEP_CONFIDENCE,
            "confidence_fall_threshold": CONFIDENCE_FALL_THRESHOLD,
            "repeated_deterioration_persistence": REPEATED_DETERIORATION_PERSISTENCE,
            "history_lookback_cycles": HISTORY_LOOKBACK_CYCLES,
        },
        "strongest_candidates": strongest,
        "weakest_candidates": weakest,
    }
    return df, summary


def _empty_summary(cycle_ts: str, *, watch_n: int) -> Dict[str, Any]:
    return {
        "generated_at_utc": _now_iso_utc(),
        "cycle_timestamp_utc": cycle_ts,
        "watch_candidates": watch_n,
        "promoted_confirmed": 0,
        "kept_watch": 0,
        "demoted": 0,
        "rejected": 0,
        "avg_persistence_score": 0.0,
        "thresholds": {
            "min_consecutive_for_promote": MIN_CONSECUTIVE_FOR_PROMOTE,
            "min_promote_confidence": MIN_PROMOTE_CONFIDENCE,
            "min_promote_persistence_score": MIN_PROMOTE_PERSISTENCE_SCORE,
            "min_keep_confidence": MIN_KEEP_CONFIDENCE,
            "confidence_fall_threshold": CONFIDENCE_FALL_THRESHOLD,
            "repeated_deterioration_persistence": REPEATED_DETERIORATION_PERSISTENCE,
            "history_lookback_cycles": HISTORY_LOOKBACK_CYCLES,
        },
        "strongest_candidates": [],
        "weakest_candidates": [],
    }


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only opportunity persistence engine (step 3 of WATCH funnel). "
            "Tracks WATCH history and promotes only on multi-cycle strengthening."
        ),
    )
    p.add_argument("--promotion", default=str(DEFAULT_PROMOTION_CSV))
    p.add_argument("--promotion-summary", default=str(DEFAULT_PROMOTION_JSON))
    p.add_argument("--allocation", default=str(DEFAULT_ALLOCATION_CSV))
    p.add_argument("--signals", default=str(DEFAULT_SIGNALS_CSV))
    p.add_argument("--lifecycle-effective", default=str(DEFAULT_LIFECYCLE_EFFECTIVE_CSV))
    p.add_argument("--history-parquet", default=str(DEFAULT_HISTORY_PARQUET))
    p.add_argument(
        "--history-csv",
        default=str(DEFAULT_HISTORY_CSV),
        help="Optional human-readable history export (set to empty string to disable).",
    )
    p.add_argument("--out-csv", default=str(DEFAULT_OUTPUT_CSV))
    p.add_argument("--out-json", default=str(DEFAULT_OUTPUT_JSON))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    print("[OPPORTUNITY_PERSISTENCE] starting (read-only intelligence layer)", flush=True)

    promo_df = _safe_read_csv(
        Path(args.promotion), label="opportunity_promotion_recommendations.csv"
    )
    promo_summary = _safe_read_json(
        Path(args.promotion_summary), label="opportunity_promotion_summary.json"
    )

    # Informational inputs — loaded for warning visibility / future enrichment.
    # We don't consume their fields directly because the promotion engine has
    # already aligned everything we need.
    _safe_read_csv(Path(args.allocation), label="portfolio_allocation_recommendations.csv")
    _safe_read_csv(Path(args.signals), label="signals_with_rationale.csv")
    _safe_read_csv(Path(args.lifecycle_effective), label="signal_lifecycle_effective.csv")

    # Cycle timestamp: prefer the promotion engine's stamp so all three
    # engines (alloc → promote → persistence) refer to the same "cycle".
    cycle_ts = str((promo_summary or {}).get("generated_at_utc") or "").strip()
    if not cycle_ts:
        cycle_ts = _now_iso_utc()

    history_parquet = Path(args.history_parquet)
    history_df = _safe_read_parquet(history_parquet)

    current_rows = _current_cycle_rows(promo_df, cycle_ts)
    history_df = _append_cycle_to_history(history_df, current_rows)

    # Persist the updated history BEFORE computing outputs so a downstream
    # failure (e.g. CSV write) doesn't lose the new cycle's data.
    try:
        _atomic_write_parquet(history_df, history_parquet)
    except Exception as e:
        _warn(
            f"failed to write history parquet ({history_parquet}): "
            f"{type(e).__name__}: {e}; persistence metrics may regress next run"
        )

    history_csv_path = str(args.history_csv or "").strip()
    if history_csv_path:
        try:
            _atomic_write_csv(history_df, Path(history_csv_path))
        except Exception as e:
            _warn(
                f"failed to write history CSV export ({history_csv_path}): "
                f"{type(e).__name__}: {e}"
            )

    df, summary = build_persistence(history_df=history_df, cycle_ts=cycle_ts)

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
        "[OPPORTUNITY_PERSISTENCE] "
        f"watch={summary['watch_candidates']} "
        f"promoted={summary['promoted_confirmed']} "
        f"kept={summary['kept_watch']} "
        f"demoted={summary['demoted']} "
        f"rejected={summary['rejected']}",
        flush=True,
    )
    print(
        "[OPPORTUNITY_STRONGEST] symbols="
        f"{[o['ticker'] for o in summary.get('strongest_candidates', [])]}",
        flush=True,
    )
    print(
        "[OPPORTUNITY_WEAKEST] symbols="
        f"{[o['ticker'] for o in summary.get('weakest_candidates', [])]}",
        flush=True,
    )
    print(
        f"[OPPORTUNITY_PERSISTENCE_OUT] csv={out_csv.as_posix()} "
        f"summary={out_json.as_posix()} history={history_parquet.as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
