"""
Portfolio Memory & Learning Engine — Step 12 (experiential layer).

Reads:
    data/results/trade_rationale.csv
    data/results/portfolio_execution_intents.csv
    data/results/portfolio_execution_summary.json     (optional, for cycle id)
    data/results/trade_rationale_summary.json         (optional, for cycle id)
    data/results/adaptive_regime.json
    data/results/adaptive_policy.json
    data/results/performance_intelligence_by_symbol.csv
    data/results/trade_outcomes.csv                   (optional)
    data/results/trade_log.csv                        (optional)
    data/results/portfolio_history.csv                (optional)

State files (persisted across runs, append-only):
    data/results/portfolio_memory.parquet            (primary, atomic)
    data/results/portfolio_memory.csv                (human-readable mirror)

Writes (cycle outputs):
    data/results/portfolio_memory_insights.json
    data/results/portfolio_learning_adjustments.json

Purpose
-------
Steps 1-11 are *prospective* — they produce recommendations based on
the current state of the world. This engine is the first
*retrospective* layer: it persists every decision Triton makes
together with the regime it was made under and the observable
post-decision PnL, then computes:

    "What historically worked in similar conditions?"

It produces:

    * portfolio_memory_insights.json   — descriptive stats by regime,
      by pattern, by regime×pattern, by ticker
    * portfolio_learning_adjustments.json — *bounded* (±0.05) numeric
      adjustments per regime / pattern / ticker that downstream
      engines could consume as an additive bias in future cycles

Safety
------
* READ ONLY to trading logic. No broker calls, no execution-state
  mutation, no overrides applied directly to any engine.
* Memory writes are append-only with deduplication on
  (ticker, cycle_timestamp_utc, execution_intent) so re-running the
  same cycle never double-counts.
* Learning adjustments are *strictly bounded* by ``MAX_ABS_ADJUSTMENT``
  (default ±0.05) and only emitted when the sample size meets
  ``MIN_SAMPLE_SIZE`` (default 5).
* Atomic writes (``.tmp`` + ``os.replace``) for every output.
* Missing inputs warn-and-continue; the engine degrades gracefully.
* main() returns 0 on success, 2 on output-write failure.

PnL attribution caveat (v1)
---------------------------
Per-trade attribution would require future-cycle PnL we don't yet have.
This v1 uses the symbol's *current cumulative* total_pl from
performance_intelligence_by_symbol.csv as a directional proxy:

    outcome_success = True  if total_pl > 0
                      False if total_pl < 0
                      None  if total_pl == 0 or unavailable

That is biased toward "what positions are working today" rather than
"was this specific decision correct", but it is directionally
actionable and the proper attribution can be layered in later without
schema changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
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

DEFAULT_RATIONALE_CSV = RESULTS_DIR / "trade_rationale.csv"
DEFAULT_INTENTS_CSV = RESULTS_DIR / "portfolio_execution_intents.csv"
DEFAULT_RATIONALE_SUMMARY = RESULTS_DIR / "trade_rationale_summary.json"
DEFAULT_INTENT_SUMMARY = RESULTS_DIR / "portfolio_execution_summary.json"
DEFAULT_REGIME_JSON = RESULTS_DIR / "adaptive_regime.json"
DEFAULT_POLICY_JSON = RESULTS_DIR / "adaptive_policy.json"
DEFAULT_PERF_CSV = RESULTS_DIR / "performance_intelligence_by_symbol.csv"
DEFAULT_TRADE_OUTCOMES_CSV = RESULTS_DIR / "trade_outcomes.csv"
DEFAULT_TRADE_LOG_CSV = RESULTS_DIR / "trade_log.csv"
DEFAULT_PORTFOLIO_HISTORY_CSV = RESULTS_DIR / "portfolio_history.csv"

DEFAULT_MEMORY_PARQUET = RESULTS_DIR / "portfolio_memory.parquet"
DEFAULT_MEMORY_CSV = RESULTS_DIR / "portfolio_memory.csv"

DEFAULT_INSIGHTS_JSON = RESULTS_DIR / "portfolio_memory_insights.json"
DEFAULT_ADJUSTMENTS_JSON = RESULTS_DIR / "portfolio_learning_adjustments.json"

# -----------------------------------------------------------
# Tunables — strictly bounded so a misbehaving adjustment can
# never shift a decision by more than the cap.
# -----------------------------------------------------------
MAX_ABS_ADJUSTMENT = 0.05  # hard ceiling per spec §6
MIN_SAMPLE_SIZE = 5  # smallest bucket that can emit an adjustment
ADJUSTMENT_SCALING = 0.10  # lift -> adjustment slope before clamp
HISTORY_LOOKBACK_ROWS = 5000  # rolling cap on memory size for stats

# Intent / action labels (mirror upstream engines).
INTENT_EXECUTE = "EXECUTE_NOW"
INTENT_DELAY = "DELAY"
INTENT_SKIP = "SKIP"
INTENT_BLOCK = "BLOCK"

BUY_ACTIONS: frozenset = frozenset({"BUY_NEW", "ADD"})
SELL_ACTIONS: frozenset = frozenset({"FULL_EXIT", "TRIM", "SELL"})

MEMORY_COLUMNS: Tuple[str, ...] = (
    "ticker",
    "timestamp_utc",
    "cycle_timestamp_utc",
    "regime",
    "execution_intent",
    "rebalance_action",
    "rationale_tags",
    "confidence",
    "persistence_score",
    "delta_pct",
    "intent_score",
    "explanation_score",
    "conviction_label",
    "confidence_label",
    "signal",
    "lifecycle_action",
    "risk_flag",
    "realized_pl",
    "unrealized_pl",
    "total_pl",
    "outcome_success",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[PORTFOLIO_MEMORY_WARN] {msg}", flush=True)


def _safe_read_json(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        if not path.is_file():
            _warn(f"missing input: {label} ({path}); continuing without it")
            return {}
    except OSError as e:
        _warn(f"stat failed for {label} ({path}): {type(e).__name__}: {e}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        _warn(f"failed to parse {label} ({path}): {type(e).__name__}: {e}")
        return {}


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


def _safe_read_parquet(path: Path, *, label: str) -> pd.DataFrame:
    try:
        if not path.is_file():
            return pd.DataFrame()
    except OSError as e:
        _warn(f"stat failed for {label} ({path}): {type(e).__name__}: {e}")
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        _warn(f"failed to read parquet {label} ({path}): {type(e).__name__}: {e}")
        return pd.DataFrame()


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> bool:
    """Returns True on success, False if parquet engine unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
    except Exception as e:
        _warn(
            f"parquet write unavailable ({type(e).__name__}: {e}); "
            "CSV mirror is still authoritative"
        )
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False
    os.replace(tmp, path)
    return True


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
# Coercion
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


def _norm_upper(x: Any) -> str:
    return str(x or "").strip().upper()


def _norm_symbol(x: Any) -> str:
    return str(x or "").strip().upper()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _pick_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------------------------------------
# Outcome labelling
# -----------------------------------------------------------
def _outcome_success(total_pl: Optional[float]) -> Optional[bool]:
    """v1 proxy: directional label from cumulative symbol PnL."""
    if total_pl is None:
        return None
    if total_pl > 0:
        return True
    if total_pl < 0:
        return False
    return None  # zero PnL → no signal


def _build_perf_lookup(perf_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if perf_df is None or perf_df.empty:
        return out
    sym_col = _pick_first_present(perf_df, ("symbol", "ticker"))
    if not sym_col:
        return out
    for _, r in perf_df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        out[sym] = {
            "realized_pl": _to_float_or_zero(r.get("realized_pl")),
            "unrealized_pl": _to_float_or_zero(r.get("unrealized_pl")),
            "total_pl": _to_float_or_zero(r.get("total_pl")),
        }
    return out


def _build_intent_score_lookup(intents_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if intents_df is None or intents_df.empty:
        return out
    sym_col = _pick_first_present(intents_df, ("ticker", "symbol"))
    if not sym_col or "intent_score" not in intents_df.columns:
        return out
    for _, r in intents_df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if sym:
            out[sym] = _to_float_or_zero(r.get("intent_score"))
    return out


def _resolve_cycle_id(
    *,
    rationale_summary: Dict[str, Any],
    intent_summary: Dict[str, Any],
    fallback_iso: str,
) -> str:
    """
    Stable cycle id used for memory deduplication. Preference order:
        1. trade_rationale_summary.generated_at_utc
        2. portfolio_execution_summary.generated_at_utc
        3. fallback (current wall time)
    """
    cid = (
        rationale_summary.get("generated_at_utc")
        or intent_summary.get("generated_at_utc")
        or fallback_iso
    )
    return str(cid)


def _resolve_regime(regime_json: Dict[str, Any], policy_json: Dict[str, Any]) -> str:
    reg = regime_json.get("regime") or policy_json.get("regime") or "UNKNOWN"
    return str(reg).strip().upper() or "UNKNOWN"


# -----------------------------------------------------------
# Build current-cycle memory rows
# -----------------------------------------------------------
def _build_current_cycle_rows(
    *,
    rationale_df: pd.DataFrame,
    intent_score_map: Dict[str, float],
    perf_map: Dict[str, Dict[str, float]],
    regime: str,
    cycle_ts: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    if rationale_df is None or rationale_df.empty:
        return []
    sym_col = _pick_first_present(rationale_df, ("ticker", "symbol"))
    if not sym_col:
        return []
    rows: List[Dict[str, Any]] = []
    for _, r in rationale_df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        perf = perf_map.get(sym, {})
        realized = perf.get("realized_pl")
        unrealized = perf.get("unrealized_pl")
        total = perf.get("total_pl")
        if total is None and (realized is not None or unrealized is not None):
            total = (realized or 0.0) + (unrealized or 0.0)
        rows.append(
            {
                "ticker": sym,
                "timestamp_utc": now_iso,
                "cycle_timestamp_utc": cycle_ts,
                "regime": regime,
                "execution_intent": _norm_upper(r.get("execution_intent")),
                "rebalance_action": _norm_upper(r.get("rebalance_action")),
                "rationale_tags": str(r.get("rationale_tags") or ""),
                "confidence": _to_float(r.get("confidence")),
                "persistence_score": _to_float(r.get("persistence_score")),
                "delta_pct": _to_float(r.get("delta_pct")),
                "intent_score": float(intent_score_map.get(sym, 0.0)),
                "explanation_score": _to_float_or_zero(r.get("explanation_score")),
                "conviction_label": _norm_upper(r.get("conviction_label")),
                "confidence_label": _norm_upper(r.get("confidence_label")),
                "signal": _norm_upper(r.get("signal")),
                "lifecycle_action": _norm_upper(r.get("lifecycle_action")),
                "risk_flag": _norm_upper(r.get("risk_flag")) or "OK",
                "realized_pl": realized,
                "unrealized_pl": unrealized,
                "total_pl": total,
                "outcome_success": _outcome_success(total),
            }
        )
    return rows


# -----------------------------------------------------------
# Memory persistence
# -----------------------------------------------------------
def _load_existing_memory(parquet_path: Path, csv_path: Path) -> pd.DataFrame:
    """Prefer parquet, fall back to csv, fall back to empty."""
    df = _safe_read_parquet(parquet_path, label="portfolio_memory.parquet")
    if not df.empty:
        return _coerce_memory_schema(df)
    df = _safe_read_csv(csv_path, label="portfolio_memory.csv")
    if not df.empty:
        return _coerce_memory_schema(df)
    return pd.DataFrame(columns=list(MEMORY_COLUMNS))


def _coerce_memory_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column set and types for stable merging."""
    if df is None or df.empty:
        return pd.DataFrame(columns=list(MEMORY_COLUMNS))
    for col in MEMORY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    # Preserve any extra historical columns the user might have added,
    # but ensure the canonical columns lead.
    extras = [c for c in df.columns if c not in MEMORY_COLUMNS]
    df = df[list(MEMORY_COLUMNS) + extras]
    return df


def _append_with_dedupe(
    memory_df: pd.DataFrame,
    new_rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    if not new_rows:
        return memory_df
    new_df = pd.DataFrame(new_rows, columns=list(MEMORY_COLUMNS))
    if memory_df is None or memory_df.empty:
        return new_df
    combined = pd.concat([memory_df, new_df], ignore_index=True)
    # Latest row wins per (ticker, cycle_timestamp_utc, execution_intent).
    combined = combined.drop_duplicates(
        subset=["ticker", "cycle_timestamp_utc", "execution_intent"],
        keep="last",
    )
    return combined.reset_index(drop=True)


def _split_tags(tag_str: str) -> List[str]:
    if not tag_str:
        return []
    return [t.strip() for t in str(tag_str).split("|") if t.strip()]


# -----------------------------------------------------------
# Statistics
# -----------------------------------------------------------
def _stat_record(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(observations)
    labelled = [o for o in observations if o["outcome_success"] is not None]
    n_labelled = len(labelled)
    n_success = sum(1 for o in labelled if o["outcome_success"] is True)
    n_failed = n_labelled - n_success
    sr = (n_success / n_labelled) if n_labelled > 0 else None
    pls = [o["total_pl"] for o in observations if o["total_pl"] is not None]
    avg_pl = (sum(pls) / len(pls)) if pls else None
    realized = [o["realized_pl"] for o in observations if o["realized_pl"] is not None]
    avg_realized = (sum(realized) / len(realized)) if realized else None
    drawdown = min(pls) if pls else None
    return {
        "n_observations": n,
        "n_with_outcome": n_labelled,
        "n_successful": n_success,
        "n_failed": n_failed,
        "success_rate": round(sr, 6) if sr is not None else None,
        "avg_total_pl": round(avg_pl, 4) if avg_pl is not None else None,
        "avg_realized_pl": round(avg_realized, 4) if avg_realized is not None else None,
        "drawdown_total_pl": round(drawdown, 4) if drawdown is not None else None,
    }


def _baseline_success_rate(memory_df: pd.DataFrame) -> Optional[float]:
    if memory_df is None or memory_df.empty:
        return None
    labelled = memory_df[memory_df["outcome_success"].notna()]
    n = int(len(labelled))
    if n == 0:
        return None
    successes = int((labelled["outcome_success"] == True).sum())  # noqa: E712
    return successes / n


def _memory_to_observations(memory_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Materialise once for fast bucketing — avoids pandas group iteration cost."""
    if memory_df is None or memory_df.empty:
        return []
    cap = HISTORY_LOOKBACK_ROWS
    df = memory_df.tail(cap) if len(memory_df) > cap else memory_df
    records: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        records.append(
            {
                "ticker": str(r.get("ticker") or "").upper(),
                "regime": str(r.get("regime") or "UNKNOWN").upper(),
                "execution_intent": str(r.get("execution_intent") or "").upper(),
                "rebalance_action": str(r.get("rebalance_action") or "").upper(),
                "rationale_tags": _split_tags(str(r.get("rationale_tags") or "")),
                "confidence_label": str(r.get("confidence_label") or "").upper(),
                "conviction_label": str(r.get("conviction_label") or "").upper(),
                "total_pl": _to_float(r.get("total_pl")),
                "realized_pl": _to_float(r.get("realized_pl")),
                "outcome_success": (
                    bool(r.get("outcome_success"))
                    if (
                        r.get("outcome_success") is not None
                        and not (
                            isinstance(r.get("outcome_success"), float)
                            and math.isnan(r.get("outcome_success"))
                        )
                        and str(r.get("outcome_success")).strip().lower() not in ("nan", "")
                    )
                    else None
                ),
            }
        )
    return records


def _regime_stats(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in observations:
        buckets[o["regime"]].append(o)
    return {r: _stat_record(rows) for r, rows in buckets.items()}


def _pattern_stats(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in observations:
        # Tag-derived patterns
        for t in o["rationale_tags"]:
            buckets[t].append(o)
        # Synthesised patterns
        if o["execution_intent"] == INTENT_EXECUTE and o["rebalance_action"] in BUY_ACTIONS:
            buckets["executed_buy"].append(o)
        elif o["execution_intent"] == INTENT_EXECUTE and o["rebalance_action"] in SELL_ACTIONS:
            buckets["executed_sell"].append(o)
        elif o["execution_intent"] == INTENT_BLOCK and o["rebalance_action"] in BUY_ACTIONS:
            buckets["blocked_buy"].append(o)
        elif o["execution_intent"] == INTENT_DELAY and o["rebalance_action"] in BUY_ACTIONS:
            buckets["delayed_buy"].append(o)
    return {p: _stat_record(rows) for p, rows in buckets.items()}


def _regime_pattern_stats(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for o in observations:
        for t in o["rationale_tags"]:
            buckets[(o["regime"], t)].append(o)
    return {f"{regime}|{tag}": _stat_record(rows) for (regime, tag), rows in buckets.items()}


def _ticker_stats(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in observations:
        if o["ticker"]:
            buckets[o["ticker"]].append(o)
    return {t: _stat_record(rows) for t, rows in buckets.items()}


# -----------------------------------------------------------
# Bounded learning adjustments
# -----------------------------------------------------------
def _compute_adjustment(
    bucket_stat: Dict[str, Any],
    baseline_success_rate: Optional[float],
) -> Optional[Tuple[float, float]]:
    """
    Return (lift, raw_adjustment) for a single bucket if it qualifies.

    raw_adjustment is unbounded; callers split into typed fields
    (confidence/persistence/deployment/risk) and clamp.
    """
    n = int(bucket_stat.get("n_with_outcome") or 0)
    sr = bucket_stat.get("success_rate")
    if n < MIN_SAMPLE_SIZE or sr is None or baseline_success_rate is None:
        return None
    lift = float(sr) - float(baseline_success_rate)
    raw = lift * (1.0 / ADJUSTMENT_SCALING) * MAX_ABS_ADJUSTMENT
    # lift==0 → 0; lift==+0.1 → +0.05; lift==-0.1 → -0.05; clamped.
    return lift, raw


def _bucket_adjustment_payload(
    bucket_stat: Dict[str, Any],
    baseline_success_rate: Optional[float],
) -> Optional[Dict[str, Any]]:
    res = _compute_adjustment(bucket_stat, baseline_success_rate)
    if res is None:
        return None
    lift, raw = res
    raw_clamped = _clamp(raw, -MAX_ABS_ADJUSTMENT, MAX_ABS_ADJUSTMENT)
    # Spread the same signal across the 4 typed adjustments. Confidence
    # and persistence respond most directly to "did our gates pick
    # winners"; deployment is the intent_score floor; risk_adjustment
    # is the cushion (negative -> tighten risk).
    confidence_adj = round(_clamp(raw, -MAX_ABS_ADJUSTMENT, MAX_ABS_ADJUSTMENT), 4)
    persistence_adj = round(_clamp(raw * 0.80, -MAX_ABS_ADJUSTMENT, MAX_ABS_ADJUSTMENT), 4)
    deployment_adj = round(_clamp(raw * 0.60, -MAX_ABS_ADJUSTMENT, MAX_ABS_ADJUSTMENT), 4)
    risk_adj = round(_clamp(-raw * 0.50, -MAX_ABS_ADJUSTMENT, MAX_ABS_ADJUSTMENT), 4)
    direction = "positive" if raw > 0 else ("negative" if raw < 0 else "neutral")
    reason_bits: List[str] = []
    sr = bucket_stat.get("success_rate")
    n = bucket_stat.get("n_with_outcome")
    reason_bits.append(f"success_rate={sr:.2f}" if sr is not None else "success_rate=n/a")
    reason_bits.append(f"n_with_outcome={n}")
    reason_bits.append(f"lift_over_baseline={lift:+.3f}")
    return {
        "confidence_adjustment": confidence_adj,
        "persistence_adjustment": persistence_adj,
        "deployment_adjustment": deployment_adj,
        "risk_adjustment": risk_adj,
        "raw_adjustment": round(raw_clamped, 4),
        "lift_over_baseline": round(lift, 4),
        "direction": direction,
        "n_observations": int(bucket_stat.get("n_observations") or 0),
        "n_with_outcome": int(n or 0),
        "success_rate": round(float(sr), 4) if sr is not None else None,
        "reason": ", ".join(reason_bits),
    }


def _build_adjustments_bundle(
    *,
    regime_stats: Dict[str, Any],
    pattern_stats: Dict[str, Any],
    ticker_stats: Dict[str, Any],
    baseline_success_rate: Optional[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "regime_adjustments": {},
        "pattern_adjustments": {},
        "ticker_adjustments": {},
    }
    for k, v in regime_stats.items():
        payload = _bucket_adjustment_payload(v, baseline_success_rate)
        if payload is not None:
            out["regime_adjustments"][k] = payload
    for k, v in pattern_stats.items():
        payload = _bucket_adjustment_payload(v, baseline_success_rate)
        if payload is not None:
            out["pattern_adjustments"][k] = payload
    for k, v in ticker_stats.items():
        payload = _bucket_adjustment_payload(v, baseline_success_rate)
        if payload is not None:
            out["ticker_adjustments"][k] = payload
    return out


# -----------------------------------------------------------
# Top-level orchestration
# -----------------------------------------------------------
def build_memory_artefacts(
    *,
    rationale_df: pd.DataFrame,
    intents_df: pd.DataFrame,
    rationale_summary: Dict[str, Any],
    intent_summary: Dict[str, Any],
    regime_json: Dict[str, Any],
    policy_json: Dict[str, Any],
    perf_df: pd.DataFrame,
    memory_parquet_path: Path,
    memory_csv_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Pure orchestrator — returns (updated_memory_df, insights, adjustments).
    Caller is responsible for writing the outputs.
    """
    now_iso = _now_iso_utc()
    cycle_ts = _resolve_cycle_id(
        rationale_summary=rationale_summary,
        intent_summary=intent_summary,
        fallback_iso=now_iso,
    )
    regime = _resolve_regime(regime_json, policy_json)
    perf_map = _build_perf_lookup(perf_df)
    intent_score_map = _build_intent_score_lookup(intents_df)

    new_rows = _build_current_cycle_rows(
        rationale_df=rationale_df,
        intent_score_map=intent_score_map,
        perf_map=perf_map,
        regime=regime,
        cycle_ts=cycle_ts,
        now_iso=now_iso,
    )
    existing = _load_existing_memory(memory_parquet_path, memory_csv_path)
    updated = _append_with_dedupe(existing, new_rows)

    observations = _memory_to_observations(updated)
    baseline = _baseline_success_rate(updated)
    regime_stats = _regime_stats(observations)
    pattern_stats = _pattern_stats(observations)
    rp_stats = _regime_pattern_stats(observations)
    ticker_stats = _ticker_stats(observations)

    successful_total = sum(1 for o in observations if o["outcome_success"] is True)
    failed_total = sum(1 for o in observations if o["outcome_success"] is False)
    labelled_total = successful_total + failed_total

    # Top tickers by success rate (require >=MIN_SAMPLE_SIZE).
    top_tickers = sorted(
        ((t, s) for t, s in ticker_stats.items() if s.get("n_with_outcome", 0) >= MIN_SAMPLE_SIZE),
        key=lambda kv: (kv[1].get("success_rate") or 0.0, kv[1].get("n_observations") or 0),
        reverse=True,
    )[:10]
    bottom_tickers = sorted(
        ((t, s) for t, s in ticker_stats.items() if s.get("n_with_outcome", 0) >= MIN_SAMPLE_SIZE),
        key=lambda kv: (kv[1].get("success_rate") or 0.0, -(kv[1].get("n_observations") or 0)),
    )[:10]

    cycle_counts = Counter(o["execution_intent"] for o in observations)

    insights: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "portfolio_memory_engine",
        "engine_version": 1,
        "cycle_timestamp_utc": cycle_ts,
        "current_regime": regime,
        "memory_size_total": int(len(updated)),
        "memory_size_window": int(len(observations)),
        "memory_size_with_outcome": int(labelled_total),
        "overall_success_rate": (round(baseline, 6) if baseline is not None else None),
        "successful_observations": int(successful_total),
        "failed_observations": int(failed_total),
        "execution_intent_counts": dict(cycle_counts),
        "regime_stats": regime_stats,
        "pattern_stats": pattern_stats,
        "regime_pattern_stats": rp_stats,
        "ticker_stats": ticker_stats,
        "top_tickers": [{"ticker": t, **s} for t, s in top_tickers],
        "bottom_tickers": [{"ticker": t, **s} for t, s in bottom_tickers],
        "thresholds": {
            "history_lookback_rows": HISTORY_LOOKBACK_ROWS,
            "min_sample_size": MIN_SAMPLE_SIZE,
            "max_abs_adjustment": MAX_ABS_ADJUSTMENT,
            "adjustment_scaling": ADJUSTMENT_SCALING,
        },
    }

    adjustments_bundle = _build_adjustments_bundle(
        regime_stats=regime_stats,
        pattern_stats=pattern_stats,
        ticker_stats=ticker_stats,
        baseline_success_rate=baseline,
    )
    positive_n = (
        sum(1 for d in adjustments_bundle["regime_adjustments"].values() if d["raw_adjustment"] > 0)
        + sum(
            1 for d in adjustments_bundle["pattern_adjustments"].values() if d["raw_adjustment"] > 0
        )
        + sum(
            1 for d in adjustments_bundle["ticker_adjustments"].values() if d["raw_adjustment"] > 0
        )
    )
    negative_n = (
        sum(1 for d in adjustments_bundle["regime_adjustments"].values() if d["raw_adjustment"] < 0)
        + sum(
            1 for d in adjustments_bundle["pattern_adjustments"].values() if d["raw_adjustment"] < 0
        )
        + sum(
            1 for d in adjustments_bundle["ticker_adjustments"].values() if d["raw_adjustment"] < 0
        )
    )
    adjustments: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "portfolio_memory_engine",
        "engine_version": 1,
        "cycle_timestamp_utc": cycle_ts,
        "current_regime": regime,
        "baseline_success_rate": (round(baseline, 6) if baseline is not None else None),
        "policy_bounds": {
            "max_abs_adjustment": MAX_ABS_ADJUSTMENT,
            "min_sample_size": MIN_SAMPLE_SIZE,
            "adjustment_scaling": ADJUSTMENT_SCALING,
        },
        "totals": {
            "regime_buckets": len(adjustments_bundle["regime_adjustments"]),
            "pattern_buckets": len(adjustments_bundle["pattern_adjustments"]),
            "ticker_buckets": len(adjustments_bundle["ticker_adjustments"]),
            "positive_adjustments": int(positive_n),
            "negative_adjustments": int(negative_n),
        },
        **adjustments_bundle,
    }
    return updated, insights, adjustments


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only portfolio memory & learning engine (Step 12). "
            "Persists decisions across cycles, computes regime/pattern "
            "success rates, and emits bounded learning adjustments "
            "for future consumption by downstream policy layers."
        ),
    )
    p.add_argument("--rationale", default=str(DEFAULT_RATIONALE_CSV))
    p.add_argument("--intents", default=str(DEFAULT_INTENTS_CSV))
    p.add_argument("--rationale-summary", default=str(DEFAULT_RATIONALE_SUMMARY))
    p.add_argument("--intent-summary", default=str(DEFAULT_INTENT_SUMMARY))
    p.add_argument("--regime", default=str(DEFAULT_REGIME_JSON))
    p.add_argument("--policy", default=str(DEFAULT_POLICY_JSON))
    p.add_argument("--performance", default=str(DEFAULT_PERF_CSV))
    p.add_argument("--memory-parquet", default=str(DEFAULT_MEMORY_PARQUET))
    p.add_argument("--memory-csv", default=str(DEFAULT_MEMORY_CSV))
    p.add_argument("--out-insights", default=str(DEFAULT_INSIGHTS_JSON))
    p.add_argument("--out-adjustments", default=str(DEFAULT_ADJUSTMENTS_JSON))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[PORTFOLIO_MEMORY] starting (read-only learning engine)", flush=True)

    rationale_df = _safe_read_csv(Path(args.rationale), label="trade_rationale.csv")
    intents_df = _safe_read_csv(Path(args.intents), label="portfolio_execution_intents.csv")
    rationale_summary = _safe_read_json(
        Path(args.rationale_summary), label="trade_rationale_summary.json"
    )
    intent_summary = _safe_read_json(
        Path(args.intent_summary), label="portfolio_execution_summary.json"
    )
    regime_json = _safe_read_json(Path(args.regime), label="adaptive_regime.json")
    policy_json = _safe_read_json(Path(args.policy), label="adaptive_policy.json")
    perf_df = _safe_read_csv(Path(args.performance), label="performance_intelligence_by_symbol.csv")

    memory_parquet = Path(args.memory_parquet)
    memory_csv = Path(args.memory_csv)

    updated_memory, insights, adjustments = build_memory_artefacts(
        rationale_df=rationale_df,
        intents_df=intents_df,
        rationale_summary=rationale_summary,
        intent_summary=intent_summary,
        regime_json=regime_json,
        policy_json=policy_json,
        perf_df=perf_df,
        memory_parquet_path=memory_parquet,
        memory_csv_path=memory_csv,
    )

    # Persist memory (parquet primary, CSV mirror always authoritative).
    try:
        _atomic_write_csv(updated_memory, memory_csv)
    except Exception as e:
        _warn(f"failed to write {memory_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(updated_memory, memory_parquet)

    try:
        _atomic_write_json(insights, Path(args.out_insights))
    except Exception as e:
        _warn(f"failed to write {args.out_insights}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(adjustments, Path(args.out_adjustments))
    except Exception as e:
        _warn(f"failed to write {args.out_adjustments}: {type(e).__name__}: {e}")
        return 2

    regimes_seen = sorted(insights["regime_stats"].keys())
    print(
        "[PORTFOLIO_MEMORY] "
        f"observations={insights['memory_size_window']} "
        f"successful={insights['successful_observations']} "
        f"failed={insights['failed_observations']} "
        f"regimes={regimes_seen}",
        flush=True,
    )
    print(
        "[LEARNING_ADJUSTMENTS] "
        f"positive={adjustments['totals']['positive_adjustments']} "
        f"negative={adjustments['totals']['negative_adjustments']} "
        f"baseline_success_rate="
        + (
            f"{insights['overall_success_rate']:.3f}"
            if insights["overall_success_rate"] is not None
            else "n/a"
        ),
        flush=True,
    )
    print(
        f"[PORTFOLIO_MEMORY_OUT] memory_csv={memory_csv.as_posix()} "
        f"insights={Path(args.out_insights).as_posix()} "
        f"adjustments={Path(args.out_adjustments).as_posix()}"
        + ("" if parquet_ok else " [parquet unavailable, CSV only]"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
