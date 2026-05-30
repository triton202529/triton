"""
Autonomous Strategy Diagnostics Engine -- Step 16 (governance audit).

Reads:
    data/results/autonomous_committee_decision.json
    data/results/autonomous_committee_summary.json
    data/results/meta_decision_intelligence.json
    data/results/adaptive_regime.json
    data/results/runtime_policy.json
    data/results/trade_outcomes.csv
    data/results/portfolio_history.csv
    data/results/trade_log.csv

State files (persisted across runs, append-only):
    data/results/autonomous_decision_memory.parquet
    data/results/autonomous_decision_memory.csv

Writes:
    data/results/autonomous_strategy_diagnostics.json
    data/results/autonomous_strategy_summary.json

Purpose
-------
Steps 1-15 *produce* recommendations. Step 12 (portfolio memory)
measures per-trade outcomes. This engine sits one level higher: it
audits the autonomous committee itself.

    "How good are Triton's decisions over time?"

Per cycle it appends a decision row, retroactively backfills the
forward 1d/5d/20d NAV returns for previously-stored rows whose
forward windows have now elapsed, and recomputes:

    * Per-decision hit rate (HOLD, DEPLOY_*, DELEVER, etc.)
    * Per-regime hit rate (DEFENSIVE, OPPORTUNISTIC, ...)
    * Per-trust-level hit rate (LOW / MODERATE / HIGH / VERY_HIGH)
    * Six normalised (0-1) diagnostic scores:
        - alpha_preservation_score      success of DEPLOY_* decisions
        - drawdown_avoidance_score      success of defensive decisions
        - deployment_accuracy_score     fraction of deployments with +ve forward_5d
        - governance_quality_score      defensive triggers correctly preceding draws
        - regime_prediction_score       directional agreement between regime + market
        - trust_quality_score           monotonic ordering of trust -> success rate
    * Text insights summarising the strongest and weakest patterns.

Outcome labelling (v1 heuristic, documented)
--------------------------------------------
For each decision row with a known forward_5d return:
    DEPLOY_AGGRESSIVELY / DEPLOY_SELECTIVELY
        success  iff forward_5d >  0.0
    DELEVER / DEFENSIVE_ROTATION / CAPITAL_PRESERVATION
        success  iff forward_5d <= 0.0    (defensive call validated)
    HOLD
        success  iff |forward_5d| < 0.02  (no large move missed or suffered)

The heuristic biases toward symmetric reward (a defensive call is
"right" when the market actually fell). It is intentionally simple
and only intended as a directional governance signal -- not a P&L
attribution model.

Safety
------
* READ ONLY to trading logic. No broker mutation, no engine state
  mutation. Memory is append-only with deduplication on
  (cycle_timestamp_utc, committee_decision) so re-running the same
  cycle never double-counts.
* Atomic writes (.tmp + os.replace) for every output.
* Missing inputs warn-and-continue; the engine always emits a
  defensible (often empty) diagnostics blob.
* All emitted scores are clamped into [0, 1].
* main() returns 0 on success, 2 on output-write failure.
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

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_COMMITTEE_DECISION = RESULTS_DIR / "autonomous_committee_decision.json"
DEFAULT_COMMITTEE_SUMMARY = RESULTS_DIR / "autonomous_committee_summary.json"
DEFAULT_META_INTEL = RESULTS_DIR / "meta_decision_intelligence.json"
DEFAULT_REGIME = RESULTS_DIR / "adaptive_regime.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy.json"
DEFAULT_TRADE_OUTCOMES = RESULTS_DIR / "trade_outcomes.csv"
DEFAULT_PORTFOLIO_HISTORY = RESULTS_DIR / "portfolio_history.csv"
DEFAULT_TRADE_LOG = RESULTS_DIR / "trade_log.csv"

DEFAULT_MEMORY_PARQUET = RESULTS_DIR / "autonomous_decision_memory.parquet"
DEFAULT_MEMORY_CSV = RESULTS_DIR / "autonomous_decision_memory.csv"

DEFAULT_OUT_DIAGNOSTICS = RESULTS_DIR / "autonomous_strategy_diagnostics.json"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "autonomous_strategy_summary.json"

# -----------------------------------------------------------
# Tunables
# -----------------------------------------------------------
MIN_SAMPLE_SIZE = 5  # smallest bucket that emits a hit-rate
HOLD_TOLERANCE = 0.02  # |forward_5d| <= this  -> HOLD is success
HISTORY_LOOKBACK_ROWS = 5000  # rolling cap when scoring

FORWARD_WINDOWS_DAYS: Tuple[int, ...] = (1, 5, 20)
PRIMARY_FORWARD_WINDOW = 5  # window used for success labelling

DEPLOY_DECISIONS = frozenset({"DEPLOY_SELECTIVELY", "DEPLOY_AGGRESSIVELY"})
DEFENSIVE_DECISIONS = frozenset({"DELEVER", "DEFENSIVE_ROTATION", "CAPITAL_PRESERVATION"})
HOLD_DECISIONS = frozenset({"HOLD"})

MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp_utc",
    "cycle_timestamp_utc",
    "cycle_date",
    "regime",
    "trust_level",
    "self_confidence_score",
    "committee_decision",
    "recommendation_confidence",
    "deployment_pressure",
    "defensive_pressure",
    "portfolio_health_score",
    "governance_score",
    "runtime_policy_snapshot",  # JSON-encoded dict
    "nav_at_decision",
    "realized_return_forward_1d",
    "realized_return_forward_5d",
    "realized_return_forward_20d",
    "outcome_success",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[STRATEGY_DIAGNOSTICS_WARN] {msg}", flush=True)


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


_NUMERIC_MEMORY_COLUMNS: Tuple[str, ...] = (
    "self_confidence_score",
    "recommendation_confidence",
    "deployment_pressure",
    "defensive_pressure",
    "portfolio_health_score",
    "governance_score",
    "nav_at_decision",
    "realized_return_forward_1d",
    "realized_return_forward_5d",
    "realized_return_forward_20d",
)


def _coerce_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric columns coming back through the CSV mirror may end up as
    object dtype with mixed empty-strings / NaN / floats, which pyarrow
    refuses. Coerce them to a clean float column (NaN for missing)
    before handing to to_parquet so the parquet roundtrip never fails
    on memory that was just round-tripped through CSV.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in _NUMERIC_MEMORY_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    # outcome_success: tri-state bool -> nullable boolean
    if "outcome_success" in out.columns:
        out["outcome_success"] = out["outcome_success"].map(lambda v: _to_bool_optional(v))
    return out


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> bool:
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
    if isinstance(x, bool):
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _to_bool_optional(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip().lower()
    if s in ("", "nan", "none", "null"):
        return None
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None


def _norm_upper(x: Any) -> str:
    return str(x or "").strip().upper()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _iso_to_date(s: Any) -> Optional[str]:
    """Extract a YYYY-MM-DD date string from an ISO-8601 timestamp."""
    if not s:
        return None
    txt = str(s).strip()
    if not txt:
        return None
    if "T" in txt:
        return txt.split("T", 1)[0]
    if " " in txt:
        return txt.split(" ", 1)[0]
    if len(txt) >= 10:
        return txt[:10]
    return None


def _add_days(date_str: str, days: int) -> Optional[str]:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None
    from datetime import timedelta

    return (d + timedelta(days=days)).strftime("%Y-%m-%d")


# -----------------------------------------------------------
# Memory schema
# -----------------------------------------------------------
def _coerce_memory_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(MEMORY_COLUMNS))
    for col in MEMORY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    extras = [c for c in df.columns if c not in MEMORY_COLUMNS]
    df = df[list(MEMORY_COLUMNS) + extras]
    return df


def _load_existing_memory(parquet_path: Path, csv_path: Path) -> pd.DataFrame:
    df = _safe_read_parquet(parquet_path, label="autonomous_decision_memory.parquet")
    if not df.empty:
        return _coerce_memory_schema(df)
    df = _safe_read_csv(csv_path, label="autonomous_decision_memory.csv")
    if not df.empty:
        return _coerce_memory_schema(df)
    return pd.DataFrame(columns=list(MEMORY_COLUMNS))


# -----------------------------------------------------------
# Portfolio history helpers
# -----------------------------------------------------------
def _build_nav_lookup(history_df: pd.DataFrame) -> Dict[str, float]:
    """Return {YYYY-MM-DD -> total_value} (or market_value fallback)."""
    out: Dict[str, float] = {}
    if history_df is None or history_df.empty:
        return out
    date_col = None
    for c in ("date", "timestamp", "ts", "cycle_date"):
        if c in history_df.columns:
            date_col = c
            break
    nav_col = None
    for c in ("total_value", "nav", "portfolio_value", "market_value", "total_equity"):
        if c in history_df.columns:
            nav_col = c
            break
    if not date_col or not nav_col:
        return out
    for _, r in history_df.iterrows():
        d = _iso_to_date(r.get(date_col))
        if not d:
            continue
        v = _to_float(r.get(nav_col))
        if v is None:
            continue
        # Keep last write per date (history files are appended).
        out[d] = v
    return out


def _forward_return(
    nav_lookup: Dict[str, float],
    base_date: str,
    days_ahead: int,
) -> Optional[float]:
    """
    Return the simple return from base_date to base_date+days_ahead.

    Looks for an exact NAV match first; if missing, scans forward up
    to 5 calendar days for the nearest available NAV (markets close on
    weekends/holidays). Returns None if either NAV is missing.
    """
    base_nav = nav_lookup.get(base_date)
    if base_nav is None or base_nav <= 0:
        return None
    target = _add_days(base_date, days_ahead)
    if target is None:
        return None
    target_nav = nav_lookup.get(target)
    if target_nav is None:
        # Scan a few days forward in case the target lands on a weekend.
        for offset in range(1, 6):
            candidate = _add_days(target, offset)
            if candidate is None:
                continue
            candidate_nav = nav_lookup.get(candidate)
            if candidate_nav is not None:
                target_nav = candidate_nav
                break
    if target_nav is None or target_nav <= 0:
        return None
    return (target_nav / base_nav) - 1.0


# -----------------------------------------------------------
# Outcome labelling (spec heuristic)
# -----------------------------------------------------------
def _label_outcome(decision: str, forward_5d: Optional[float]) -> Optional[bool]:
    if forward_5d is None:
        return None
    decision_u = _norm_upper(decision)
    if decision_u in DEPLOY_DECISIONS:
        return forward_5d > 0.0
    if decision_u in DEFENSIVE_DECISIONS:
        return forward_5d <= 0.0
    if decision_u in HOLD_DECISIONS:
        return abs(forward_5d) < HOLD_TOLERANCE
    return None


# -----------------------------------------------------------
# Current-cycle row
# -----------------------------------------------------------
def _build_current_row(
    *,
    committee_decision: Dict[str, Any],
    committee_summary: Dict[str, Any],
    meta_intel: Dict[str, Any],
    regime_json: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    nav_lookup: Dict[str, float],
    now_iso: str,
) -> Optional[Dict[str, Any]]:
    if not committee_decision:
        _warn("no autonomous_committee_decision.json; skipping current-cycle row")
        return None

    cycle_ts = str(
        committee_decision.get("generated_at_utc")
        or (committee_summary or {}).get("generated_at_utc")
        or now_iso
    )
    cycle_date = _iso_to_date(cycle_ts) or _iso_to_date(now_iso)
    regime = _norm_upper(
        committee_decision.get("regime")
        or (regime_json or {}).get("regime")
        or (runtime_policy or {}).get("regime")
        or "UNKNOWN"
    )
    decision = _norm_upper(committee_decision.get("decision"))
    trust_level = _norm_upper(
        committee_decision.get("meta_trust_level")
        or (meta_intel or {}).get("trust_level")
        or "MODERATE"
    )
    self_conf = _to_float(
        committee_decision.get("self_confidence_score")
        or (meta_intel or {}).get("self_confidence_score")
    )
    rec_conf = _to_float(committee_decision.get("recommendation_confidence"))
    dep_p = _to_float(committee_decision.get("deployment_pressure"))
    def_p = _to_float(committee_decision.get("defensive_pressure"))
    cs = committee_decision.get("committee_scores") or {}
    ph = _to_float(cs.get("portfolio_health_score"))
    gov = _to_float(cs.get("governance_score"))
    # Compact runtime policy snapshot.
    policy_snap = {
        "regime": _norm_upper((runtime_policy or {}).get("regime") or regime),
        "confidence_threshold": _to_float((runtime_policy or {}).get("confidence_threshold")),
        "persistence_threshold": _to_float((runtime_policy or {}).get("persistence_threshold")),
        "deployment_threshold": _to_float((runtime_policy or {}).get("deployment_threshold")),
        "target_cash_pct": _to_float((runtime_policy or {}).get("target_cash_pct")),
        "max_position_pct": _to_float((runtime_policy or {}).get("max_position_pct")),
        "engine": str((runtime_policy or {}).get("engine") or ""),
    }
    nav_at = nav_lookup.get(cycle_date) if cycle_date else None
    return {
        "timestamp_utc": now_iso,
        "cycle_timestamp_utc": cycle_ts,
        "cycle_date": cycle_date,
        "regime": regime,
        "trust_level": trust_level,
        "self_confidence_score": self_conf,
        "committee_decision": decision,
        "recommendation_confidence": rec_conf,
        "deployment_pressure": dep_p,
        "defensive_pressure": def_p,
        "portfolio_health_score": ph,
        "governance_score": gov,
        "runtime_policy_snapshot": json.dumps(policy_snap, default=_json_safe, sort_keys=True),
        "nav_at_decision": nav_at,
        "realized_return_forward_1d": None,
        "realized_return_forward_5d": None,
        "realized_return_forward_20d": None,
        "outcome_success": None,
    }


def _append_with_dedupe(
    memory_df: pd.DataFrame,
    new_row: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    if new_row is None:
        return memory_df if memory_df is not None else pd.DataFrame(columns=list(MEMORY_COLUMNS))
    new_df = pd.DataFrame([new_row], columns=list(MEMORY_COLUMNS))
    if memory_df is None or memory_df.empty:
        return new_df
    combined = pd.concat([memory_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["cycle_timestamp_utc", "committee_decision"],
        keep="last",
    )
    return combined.reset_index(drop=True)


# -----------------------------------------------------------
# Backfill forward returns
# -----------------------------------------------------------
def _backfill_forward_returns(
    memory_df: pd.DataFrame,
    nav_lookup: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Walk every memory row and (re)compute forward returns where the
    forward window has now elapsed and NAV is available. Re-derive
    outcome_success from the primary window. Returns the mutated df
    and a stats dict with backfill counts.
    """
    stats = {"rows_seen": 0, "rows_backfilled": 0, "rows_labelled": 0}
    if memory_df is None or memory_df.empty or not nav_lookup:
        return memory_df, stats
    for idx in memory_df.index:
        stats["rows_seen"] += 1
        base_date = _iso_to_date(memory_df.at[idx, "cycle_date"]) or _iso_to_date(
            memory_df.at[idx, "cycle_timestamp_utc"]
        )
        if not base_date:
            continue
        any_filled = False
        for days in FORWARD_WINDOWS_DAYS:
            col = f"realized_return_forward_{days}d"
            current = _to_float(memory_df.at[idx, col])
            if current is not None:
                continue
            r = _forward_return(nav_lookup, base_date, days)
            if r is not None:
                memory_df.at[idx, col] = r
                any_filled = True
        if any_filled:
            stats["rows_backfilled"] += 1
        # Re-derive outcome_success from the primary window any time
        # the window value is available. This keeps the label coherent
        # even if upstream definitions evolve.
        forward = _to_float(memory_df.at[idx, f"realized_return_forward_{PRIMARY_FORWARD_WINDOW}d"])
        decision = _norm_upper(memory_df.at[idx, "committee_decision"])
        label = _label_outcome(decision, forward)
        memory_df.at[idx, "outcome_success"] = label
        if label is not None:
            stats["rows_labelled"] += 1
    return memory_df, stats


# -----------------------------------------------------------
# Statistics
# -----------------------------------------------------------
def _bucket_stat(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    labelled = [r for r in rows if r["outcome_success"] is not None]
    n_labelled = len(labelled)
    n_success = sum(1 for r in labelled if r["outcome_success"] is True)
    n_failed = n_labelled - n_success
    sr = (n_success / n_labelled) if n_labelled > 0 else None
    rets = [r["forward_5d"] for r in rows if r["forward_5d"] is not None]
    avg_ret = (sum(rets) / len(rets)) if rets else None
    drawdown = min(rets) if rets else None
    return {
        "n_observations": n,
        "n_with_outcome": n_labelled,
        "n_successful": n_success,
        "n_failed": n_failed,
        "success_rate": round(sr, 6) if sr is not None else None,
        "avg_forward_5d": round(avg_ret, 6) if avg_ret is not None else None,
        "drawdown_5d": round(drawdown, 6) if drawdown is not None else None,
    }


def _materialise(memory_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if memory_df is None or memory_df.empty:
        return []
    cap = HISTORY_LOOKBACK_ROWS
    df = memory_df.tail(cap) if len(memory_df) > cap else memory_df
    out: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        out.append(
            {
                "decision": _norm_upper(r.get("committee_decision")),
                "regime": _norm_upper(r.get("regime")),
                "trust_level": _norm_upper(r.get("trust_level")),
                "self_confidence": _to_float(r.get("self_confidence_score")),
                "recommendation_confidence": _to_float(r.get("recommendation_confidence")),
                "deployment_pressure": _to_float(r.get("deployment_pressure")),
                "defensive_pressure": _to_float(r.get("defensive_pressure")),
                "portfolio_health": _to_float(r.get("portfolio_health_score")),
                "governance": _to_float(r.get("governance_score")),
                "forward_1d": _to_float(r.get("realized_return_forward_1d")),
                "forward_5d": _to_float(r.get("realized_return_forward_5d")),
                "forward_20d": _to_float(r.get("realized_return_forward_20d")),
                "outcome_success": _to_bool_optional(r.get("outcome_success")),
            }
        )
    return out


def _group_stats(
    rows: List[Dict[str, Any]],
    key: str,
) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        k = r[key] or "UNKNOWN"
        buckets[k].append(r)
    return {k: _bucket_stat(v) for k, v in buckets.items()}


def _overall_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _bucket_stat(rows)


def _baseline_success_rate(rows: List[Dict[str, Any]]) -> Optional[float]:
    labelled = [r for r in rows if r["outcome_success"] is not None]
    if not labelled:
        return None
    successes = sum(1 for r in labelled if r["outcome_success"] is True)
    return successes / len(labelled)


# -----------------------------------------------------------
# Diagnostic scores (six normalised 0-1)
# -----------------------------------------------------------
def _score_from_group(
    decision_stats: Dict[str, Any],
    keys: Iterable[str],
) -> Tuple[Optional[float], int]:
    """Pooled success rate across the requested decision keys."""
    total_labelled = 0
    total_success = 0
    for k in keys:
        s = decision_stats.get(k) or {}
        total_labelled += int(s.get("n_with_outcome") or 0)
        total_success += int(s.get("n_successful") or 0)
    if total_labelled == 0:
        return None, 0
    return total_success / total_labelled, total_labelled


def _alpha_preservation(decision_stats: Dict[str, Any]) -> Tuple[float, bool, str]:
    sr, n = _score_from_group(decision_stats, DEPLOY_DECISIONS)
    if sr is None or n < MIN_SAMPLE_SIZE:
        return 0.50, False, f"insufficient_history (n_with_outcome={n})"
    return _clamp(sr, 0.0, 1.0), True, f"deploy_success_rate={sr:.3f} n={n}"


def _drawdown_avoidance(decision_stats: Dict[str, Any]) -> Tuple[float, bool, str]:
    sr, n = _score_from_group(decision_stats, DEFENSIVE_DECISIONS)
    if sr is None or n < MIN_SAMPLE_SIZE:
        return 0.50, False, f"insufficient_history (n_with_outcome={n})"
    return _clamp(sr, 0.0, 1.0), True, f"defensive_success_rate={sr:.3f} n={n}"


def _deployment_accuracy(rows: List[Dict[str, Any]]) -> Tuple[float, bool, str]:
    """Fraction of DEPLOY_* rows that had positive forward_5d."""
    deploys = [r for r in rows if r["decision"] in DEPLOY_DECISIONS and r["forward_5d"] is not None]
    if len(deploys) < MIN_SAMPLE_SIZE:
        return 0.50, False, f"insufficient_history (n={len(deploys)})"
    wins = sum(1 for r in deploys if r["forward_5d"] > 0)
    sr = wins / len(deploys)
    return _clamp(sr, 0.0, 1.0), True, f"positive_forward_5d={wins}/{len(deploys)}"


def _governance_quality(rows: List[Dict[str, Any]]) -> Tuple[float, bool, str]:
    """
    "When the market actually fell, how often had Triton already gone
    defensive?" -- a precision-style metric on the governance gate.
    """
    drawdowns = [r for r in rows if r["forward_5d"] is not None and r["forward_5d"] < 0.0]
    if len(drawdowns) < MIN_SAMPLE_SIZE:
        # Fall back to pooled labelled success rate across all decisions.
        all_labelled = [r for r in rows if r["outcome_success"] is not None]
        if len(all_labelled) < MIN_SAMPLE_SIZE:
            return 0.50, False, "insufficient_history"
        wins = sum(1 for r in all_labelled if r["outcome_success"])
        return (
            _clamp(wins / len(all_labelled), 0.0, 1.0),
            True,
            (f"fallback_overall_success={wins}/{len(all_labelled)}"),
        )
    defensive_during_draw = sum(1 for r in drawdowns if r["decision"] in DEFENSIVE_DECISIONS)
    score = defensive_during_draw / len(drawdowns)
    return (
        _clamp(score, 0.0, 1.0),
        True,
        (f"defensive_when_market_fell={defensive_during_draw}/{len(drawdowns)}"),
    )


def _regime_prediction(regime_stats: Dict[str, Any]) -> Tuple[float, bool, str]:
    """
    Per-regime hit rate, weighted by sample size. High score means each
    regime classification correctly anticipated subsequent behaviour.
    """
    total_n = 0
    weighted = 0.0
    contributing: List[str] = []
    for regime, s in regime_stats.items():
        n = int(s.get("n_with_outcome") or 0)
        sr = s.get("success_rate")
        if n < MIN_SAMPLE_SIZE or sr is None:
            continue
        total_n += n
        weighted += n * float(sr)
        contributing.append(f"{regime}:{n}")
    if total_n == 0:
        return 0.50, False, "insufficient_history"
    score = weighted / total_n
    return (
        _clamp(score, 0.0, 1.0),
        True,
        (f"weighted_hit_rate={score:.3f} over [{', '.join(contributing)}]"),
    )


def _trust_quality(trust_stats: Dict[str, Any]) -> Tuple[float, bool, str]:
    """
    Monotonicity check: do higher trust levels correspond to higher
    realised hit rates? Returns 1.0 for perfect monotonic ordering,
    0.0 for fully inverted. Computed as Spearman-style concordance
    over the trust ordering LOW->VERY_LOW...VERY_HIGH.
    """
    order = ["VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    seen: List[Tuple[int, float, int]] = []  # (rank, sr, n)
    for rank, lvl in enumerate(order):
        s = trust_stats.get(lvl) or {}
        n = int(s.get("n_with_outcome") or 0)
        sr = s.get("success_rate")
        if n >= MIN_SAMPLE_SIZE and sr is not None:
            seen.append((rank, float(sr), n))
    if len(seen) < 2:
        return 0.50, False, f"insufficient_distinct_trust_levels (have {len(seen)})"
    # Concordance: number of monotone-increasing pairs minus inversions,
    # normalised into [0, 1].
    pairs = 0
    concordant = 0
    discordant = 0
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            pairs += 1
            dr = seen[j][1] - seen[i][1]
            if dr > 0:
                concordant += 1
            elif dr < 0:
                discordant += 1
    if pairs == 0:
        return 0.50, False, "no_paired_observations"
    score = 0.5 + 0.5 * ((concordant - discordant) / pairs)
    return (
        _clamp(score, 0.0, 1.0),
        True,
        (f"concordant_pairs={concordant}/{pairs} discordant={discordant}"),
    )


# -----------------------------------------------------------
# Insights
# -----------------------------------------------------------
def _build_insights(
    *,
    decision_stats: Dict[str, Any],
    regime_stats: Dict[str, Any],
    trust_stats: Dict[str, Any],
    decision_x_regime: Dict[str, Any],
    overall: Dict[str, Any],
) -> List[str]:
    out: List[str] = []
    overall_sr = overall.get("success_rate")
    if overall_sr is not None:
        out.append(
            f"Overall decision hit-rate is {overall_sr:.1%} across "
            f"{overall['n_with_outcome']} labelled observation(s)."
        )

    # Per-decision highlights -- pick the best and worst.
    qualifying = [
        (k, s)
        for k, s in decision_stats.items()
        if s.get("n_with_outcome")
        and s["n_with_outcome"] >= MIN_SAMPLE_SIZE
        and s.get("success_rate") is not None
    ]
    if qualifying:
        best = max(qualifying, key=lambda kv: kv[1]["success_rate"])
        worst = min(qualifying, key=lambda kv: kv[1]["success_rate"])
        out.append(
            f"Best-performing decision: {best[0]} at "
            f"{best[1]['success_rate']:.1%} success "
            f"(n={best[1]['n_with_outcome']})."
        )
        if best[0] != worst[0]:
            out.append(
                f"Weakest-performing decision: {worst[0]} at "
                f"{worst[1]['success_rate']:.1%} success "
                f"(n={worst[1]['n_with_outcome']})."
            )

    # Per-regime highlight.
    regime_qual = [
        (k, s)
        for k, s in regime_stats.items()
        if s.get("n_with_outcome")
        and s["n_with_outcome"] >= MIN_SAMPLE_SIZE
        and s.get("success_rate") is not None
    ]
    if regime_qual:
        best_r = max(regime_qual, key=lambda kv: kv[1]["success_rate"])
        out.append(
            f"Strongest regime: {best_r[0]} at "
            f"{best_r[1]['success_rate']:.1%} success "
            f"(n={best_r[1]['n_with_outcome']})."
        )

    # Per-trust monotonicity insight.
    trust_qual = [
        (k, s)
        for k, s in trust_stats.items()
        if s.get("n_with_outcome")
        and s["n_with_outcome"] >= MIN_SAMPLE_SIZE
        and s.get("success_rate") is not None
    ]
    if len(trust_qual) >= 2:
        order = {"VERY_LOW": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3, "VERY_HIGH": 4}
        sorted_q = sorted(trust_qual, key=lambda kv: order.get(kv[0], 99))
        low = sorted_q[0]
        high = sorted_q[-1]
        delta = (high[1]["success_rate"] or 0.0) - (low[1]["success_rate"] or 0.0)
        direction = "rises" if delta > 0 else ("falls" if delta < 0 else "is flat")
        out.append(
            f"Hit-rate {direction} from {low[0]} trust "
            f"({low[1]['success_rate']:.1%}) to {high[0]} trust "
            f"({high[1]['success_rate']:.1%}) -- delta {delta:+.1%}."
        )

    # Decision-x-regime highlight (e.g., "HOLD in DEFENSIVE avoided drawdowns X%").
    interesting: List[Tuple[str, Dict[str, Any]]] = []
    for k, s in decision_x_regime.items():
        if (
            s.get("n_with_outcome")
            and s["n_with_outcome"] >= MIN_SAMPLE_SIZE
            and s.get("success_rate") is not None
        ):
            interesting.append((k, s))
    if interesting:
        best_dr = max(interesting, key=lambda kv: kv[1]["success_rate"])
        out.append(
            f"Combined: {best_dr[0]} historically succeeded "
            f"{best_dr[1]['success_rate']:.1%} of the time "
            f"(n={best_dr[1]['n_with_outcome']})."
        )

    if not out:
        out.append(
            "Insufficient labelled history yet -- diagnostics will activate "
            f"once at least {MIN_SAMPLE_SIZE} cycles have completed their "
            f"{PRIMARY_FORWARD_WINDOW}-day forward window."
        )
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_diagnostics(
    *,
    committee_decision: Dict[str, Any],
    committee_summary: Dict[str, Any],
    meta_intel: Dict[str, Any],
    regime_json: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    portfolio_history: pd.DataFrame,
    memory_parquet_path: Path,
    memory_csv_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, int]]:
    now_iso = _now_iso_utc()
    nav_lookup = _build_nav_lookup(portfolio_history)

    new_row = _build_current_row(
        committee_decision=committee_decision,
        committee_summary=committee_summary,
        meta_intel=meta_intel,
        regime_json=regime_json,
        runtime_policy=runtime_policy,
        nav_lookup=nav_lookup,
        now_iso=now_iso,
    )
    existing = _load_existing_memory(memory_parquet_path, memory_csv_path)
    updated = _append_with_dedupe(existing, new_row)
    updated, backfill_stats = _backfill_forward_returns(updated, nav_lookup)

    rows = _materialise(updated)
    overall = _overall_stats(rows)
    decision_stats = _group_stats(rows, "decision")
    regime_stats = _group_stats(rows, "regime")
    trust_stats = _group_stats(rows, "trust_level")

    # decision x regime joint bucket -- "HOLD|DEFENSIVE" style keys.
    dr_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        dr_buckets[f"{r['decision']}|{r['regime']}"].append(r)
    decision_x_regime = {k: _bucket_stat(v) for k, v in dr_buckets.items()}

    alpha_v, alpha_k, alpha_src = _alpha_preservation(decision_stats)
    draw_v, draw_k, draw_src = _drawdown_avoidance(decision_stats)
    accuracy_v, accuracy_k, accuracy_src = _deployment_accuracy(rows)
    gov_v, gov_k, gov_src = _governance_quality(rows)
    regp_v, regp_k, regp_src = _regime_prediction(regime_stats)
    trust_v, trust_k, trust_src = _trust_quality(trust_stats)

    diagnostic_scores = {
        "alpha_preservation_score": round(alpha_v, 6),
        "drawdown_avoidance_score": round(draw_v, 6),
        "deployment_accuracy_score": round(accuracy_v, 6),
        "governance_quality_score": round(gov_v, 6),
        "regime_prediction_score": round(regp_v, 6),
        "trust_quality_score": round(trust_v, 6),
    }
    scores_known = {
        "alpha_preservation_score": alpha_k,
        "drawdown_avoidance_score": draw_k,
        "deployment_accuracy_score": accuracy_k,
        "governance_quality_score": gov_k,
        "regime_prediction_score": regp_k,
        "trust_quality_score": trust_k,
    }
    score_sources = {
        "alpha_preservation_score": alpha_src,
        "drawdown_avoidance_score": draw_src,
        "deployment_accuracy_score": accuracy_src,
        "governance_quality_score": gov_src,
        "regime_prediction_score": regp_src,
        "trust_quality_score": trust_src,
    }
    # Top-line "decision_quality_score" = mean of the *known* sub-scores.
    known_values = [v for k, v in diagnostic_scores.items() if scores_known[k]]
    if known_values:
        decision_quality = sum(known_values) / len(known_values)
    else:
        decision_quality = 0.50
    decision_quality = round(_clamp(decision_quality, 0.0, 1.0), 6)

    insights = _build_insights(
        decision_stats=decision_stats,
        regime_stats=regime_stats,
        trust_stats=trust_stats,
        decision_x_regime=decision_x_regime,
        overall=overall,
    )

    diagnostics: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_strategy_diagnostics",
        "engine_version": 1,
        "decision_quality_score": decision_quality,
        "scores": diagnostic_scores,
        "scores_known": scores_known,
        "score_sources": score_sources,
        "overall": overall,
        "decision_stats": decision_stats,
        "regime_stats": regime_stats,
        "trust_stats": trust_stats,
        "decision_x_regime_stats": decision_x_regime,
        "insights": insights,
        "memory_size_total": int(len(updated)),
        "memory_size_window": int(len(rows)),
        "memory_size_with_outcome": int(overall.get("n_with_outcome") or 0),
        "backfill_stats": backfill_stats,
        "thresholds": {
            "min_sample_size": MIN_SAMPLE_SIZE,
            "hold_tolerance": HOLD_TOLERANCE,
            "forward_windows_days": list(FORWARD_WINDOWS_DAYS),
            "primary_forward_window_days": PRIMARY_FORWARD_WINDOW,
            "history_lookback_rows": HISTORY_LOOKBACK_ROWS,
        },
        "inputs_seen": {
            "committee_decision": bool(committee_decision),
            "committee_summary": bool(committee_summary),
            "meta_intel": bool(meta_intel),
            "regime_json": bool(regime_json),
            "runtime_policy": bool(runtime_policy),
            "portfolio_history": bool(nav_lookup),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_strategy_diagnostics",
        "engine_version": 1,
        "decision_quality_score": decision_quality,
        "alpha_preservation_score": diagnostic_scores["alpha_preservation_score"],
        "drawdown_avoidance_score": diagnostic_scores["drawdown_avoidance_score"],
        "deployment_accuracy_score": diagnostic_scores["deployment_accuracy_score"],
        "governance_quality_score": diagnostic_scores["governance_quality_score"],
        "regime_prediction_score": diagnostic_scores["regime_prediction_score"],
        "trust_quality_score": diagnostic_scores["trust_quality_score"],
        "memory_size_total": int(len(updated)),
        "memory_size_with_outcome": int(overall.get("n_with_outcome") or 0),
        "overall_success_rate": overall.get("success_rate"),
        "top_insights": insights[:3],
        "regimes_seen": sorted(regime_stats.keys()),
        "decisions_seen": sorted(decision_stats.keys()),
    }
    return updated, diagnostics, summary, backfill_stats


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only autonomous strategy diagnostics engine (Step 16). "
            "Persists Step 15 committee decisions, backfills forward "
            "returns from portfolio_history.csv, and emits six normalised "
            "diagnostic scores plus narrative insights."
        ),
    )
    p.add_argument("--committee-decision", default=str(DEFAULT_COMMITTEE_DECISION))
    p.add_argument("--committee-summary", default=str(DEFAULT_COMMITTEE_SUMMARY))
    p.add_argument("--meta-intel", default=str(DEFAULT_META_INTEL))
    p.add_argument("--regime", default=str(DEFAULT_REGIME))
    p.add_argument("--runtime-policy", default=str(DEFAULT_RUNTIME_POLICY))
    p.add_argument("--portfolio-history", default=str(DEFAULT_PORTFOLIO_HISTORY))
    p.add_argument("--memory-parquet", default=str(DEFAULT_MEMORY_PARQUET))
    p.add_argument("--memory-csv", default=str(DEFAULT_MEMORY_CSV))
    p.add_argument("--out-diagnostics", default=str(DEFAULT_OUT_DIAGNOSTICS))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[STRATEGY_DIAGNOSTICS] starting (read-only governance audit)", flush=True)

    committee_decision = _safe_read_json(
        Path(args.committee_decision), label="autonomous_committee_decision.json"
    )
    committee_summary = _safe_read_json(
        Path(args.committee_summary), label="autonomous_committee_summary.json"
    )
    meta_intel = _safe_read_json(Path(args.meta_intel), label="meta_decision_intelligence.json")
    regime_json = _safe_read_json(Path(args.regime), label="adaptive_regime.json")
    runtime_policy = _safe_read_json(Path(args.runtime_policy), label="runtime_policy.json")
    portfolio_history = _safe_read_csv(Path(args.portfolio_history), label="portfolio_history.csv")

    memory_parquet = Path(args.memory_parquet)
    memory_csv = Path(args.memory_csv)

    updated_memory, diagnostics, summary, backfill = build_diagnostics(
        committee_decision=committee_decision,
        committee_summary=committee_summary,
        meta_intel=meta_intel,
        regime_json=regime_json,
        runtime_policy=runtime_policy,
        portfolio_history=portfolio_history,
        memory_parquet_path=memory_parquet,
        memory_csv_path=memory_csv,
    )

    try:
        _atomic_write_csv(updated_memory, memory_csv)
    except Exception as e:
        _warn(f"failed to write {memory_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(_coerce_for_parquet(updated_memory), memory_parquet)

    try:
        _atomic_write_json(diagnostics, Path(args.out_diagnostics))
    except Exception as e:
        _warn(f"failed to write {args.out_diagnostics}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(summary, Path(args.out_summary))
    except Exception as e:
        _warn(f"failed to write {args.out_summary}: {type(e).__name__}: {e}")
        return 2

    print(
        "[STRATEGY_DIAGNOSTICS] "
        f"observations={summary['memory_size_total']} "
        f"with_outcome={summary['memory_size_with_outcome']} "
        f"decision_quality={summary['decision_quality_score']:.3f} "
        f"governance={summary['governance_quality_score']:.3f} "
        f"trust_quality={summary['trust_quality_score']:.3f}",
        flush=True,
    )
    if backfill.get("rows_backfilled"):
        print(
            f"[STRATEGY_DIAGNOSTICS_BACKFILL] "
            f"seen={backfill['rows_seen']} "
            f"backfilled={backfill['rows_backfilled']} "
            f"labelled={backfill['rows_labelled']}",
            flush=True,
        )
    print(
        f"[STRATEGY_DIAGNOSTICS_OUT] memory_csv={memory_csv.as_posix()} "
        f"diagnostics={Path(args.out_diagnostics).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()}"
        + ("" if parquet_ok else " [parquet unavailable, CSV only]"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
