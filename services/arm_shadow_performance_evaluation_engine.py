"""
ARM Shadow Performance Evaluation Engine -- Step 30.

Reads:
    data/results/arm_shadow_execution_memory.csv         (Step 29 -- primary)
    data/results/arm_shadow_execution_memory.parquet     (Step 29 -- fallback)
    data/results/portfolio_history.csv                    (NAV path)
    data/results/trade_log.csv                            (turnover context)
    data/results/adaptive_regime.json                     (Step 10)
    data/results/runtime_policy_governed.json             (Step 18)
    data/results/meta_decision_intelligence.json          (Step 13)
    data/results/autonomous_governance_scorecard.json     (Step 19)

Writes:
    data/results/arm_shadow_performance.json
    data/results/arm_shadow_performance.md
    data/results/arm_shadow_performance_summary.json
    data/results/arm_shadow_outcomes_memory.csv
    data/results/arm_shadow_outcomes_memory.parquet

Purpose
-------
This engine answers:

    "How good were Triton's shadow decisions?"

It walks the Step 29 shadow memory, enriches every row with the
realized portfolio forward return at +1d / +5d / +20d (where enough
NAV history has elapsed), persists the enriched view to a separate
*outcomes* memory file, and produces eight normalised performance
metrics plus an apprenticeship verdict. Forward returns are derived
from ``portfolio_history.csv`` exactly the way Step 16 does, so the
unit of analysis is portfolio-level realised effectiveness.

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
The forbidden token list is defined in the Step 29 spec; to keep the
file grep-clean those tokens are never written literally in this
source. Verified at import time by ``_self_check_no_broker_tokens``.

Apprenticeship verdict cascade (strict precedence)
--------------------------------------------------
    LEARNING            n_labelled < MIN_LEARNING_OBSERVATIONS
    AUTONOMY_NOT_READY  autonomy_readiness < 0.40 OR win_rate < 0.45
                        (with enough observations to be meaningful)
    AUTONOMY_CANDIDATE  autonomy_readiness >= 0.70 AND n_labelled
                        >= MIN_CANDIDATE_OBSERVATIONS AND win_rate
                        >= 0.55
    TRUST_BUILDING      autonomy_readiness >= 0.55 AND n_labelled
                        >= MIN_TRUST_OBSERVATIONS
    IMPROVING           anything else with positive trajectory
    LEARNING            otherwise (small sample fallback)

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation, no
  trade-API imports.
* Atomic writes (.tmp + os.replace) for JSON / MD / CSV / Parquet.
* Append-only outcomes memory with dedup by (cycle_id, ticker, action)
  -- new enrichment of an existing row replaces in place.
* Missing inputs warn-and-continue. With no Step 29 memory the
  verdict is LEARNING and zero outcome rows are emitted.
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_SHADOW_MEMORY_CSV = RESULTS_DIR / "arm_shadow_execution_memory.csv"
DEFAULT_SHADOW_MEMORY_PARQUET = RESULTS_DIR / "arm_shadow_execution_memory.parquet"
DEFAULT_PORTFOLIO_HISTORY = RESULTS_DIR / "portfolio_history.csv"
DEFAULT_TRADE_LOG = RESULTS_DIR / "trade_log.csv"
DEFAULT_REGIME = RESULTS_DIR / "adaptive_regime.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_META_DECISION = RESULTS_DIR / "meta_decision_intelligence.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_shadow_performance.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_shadow_performance.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_shadow_performance_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_shadow_outcomes_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_shadow_outcomes_memory.parquet"


# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------
FORWARD_WINDOWS_DAYS: Tuple[int, ...] = (1, 5, 20)
PRIMARY_WINDOW_DAYS = 5
DRAWDOWN_FLOOR = -0.03  # forward return below this is "drawdown"
ALPHA_BAND = 0.05  # +/-5% mean return spans [0,1] alpha band

MIN_LEARNING_OBSERVATIONS = 10
MIN_TRUST_OBSERVATIONS = 30
MIN_CANDIDATE_OBSERVATIONS = 50
TRAJECTORY_RECENT_FRACTION = 0.4  # last 40% of observations vs the rest

SHADOW_MODES_ANALYSED = (
    "SHADOW_OBSERVATION",
    "SHADOW_ASSISTED",
    "SHADOW_AUTO",
)

DEPLOY_ACTIONS = frozenset({"buy_new", "open_position", "add", "add_to_position"})
ROTATION_ACTIONS = frozenset({"trim", "sell", "rotate", "rotation", "exit", "full_exit"})

# Outcomes-memory column order
OUTCOMES_MEMORY_COLUMNS: Tuple[str, ...] = (
    # mirror of Step 29 shadow-memory schema (so the file is a
    # self-contained enriched view that can be joined / replayed)
    "cycle_id",
    "timestamp_utc",
    "ticker",
    "action",
    "target_weight",
    "estimated_notional",
    "plan_confidence",
    "shadow_mode",
    "rationale",
    "execution_mode",
    "authorization_state",
    "would_execute",
    "requires_operator_confirmation",
    "blocked_reason",
    "regime",
    "trust_level",
    "governance_state",
    "runtime_policy_snapshot_json",
    # Enrichment fields
    "cycle_date",
    "nav_at_cycle",
    "nav_fwd_1d",
    "nav_fwd_5d",
    "nav_fwd_20d",
    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "outcome_known",
    "outcome_label",
    "enriched_at_utc",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_SHADOW_PERF_WARN] {msg}", flush=True)


def _safe_read_json(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        if not path.is_file():
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


def _safe_read_csv_rows(path: Path, *, label: str) -> List[Dict[str, str]]:
    try:
        if not path.is_file():
            return []
    except OSError as e:
        _warn(f"stat failed for {label} ({path}): {type(e).__name__}: {e}")
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except Exception as e:
        _warn(f"failed to parse {label} ({path}): {type(e).__name__}: {e}")
        return []


def _safe_read_parquet_rows(path: Path, *, label: str) -> List[Dict[str, Any]]:
    """Best-effort parquet read returning list-of-dicts; warn on failure."""
    try:
        if not path.is_file():
            return []
    except OSError as e:
        _warn(f"stat failed for {label} ({path}): {type(e).__name__}: {e}")
        return []
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        _warn(f"pandas unavailable to read parquet {label}: {type(e).__name__}: {e}")
        return []
    try:
        df = pd.read_parquet(path)
        # pandas NaN -> None for cleanliness
        return [
            {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.items()}
            for r in df.to_dict(orient="records")
        ]
    except Exception as e:
        _warn(f"failed to read parquet {label} ({path}): {type(e).__name__}: {e}")
        return []


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_safe)
    os.replace(tmp, path)


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_write_csv(rows: List[Dict[str, Any]], path: Path, *, columns: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            row_out = {c: ("" if r.get(c) is None else r.get(c)) for c in columns}
            w.writerow(row_out)
    os.replace(tmp, path)


def _atomic_write_parquet(rows: List[Dict[str, Any]], path: Path) -> bool:
    """Best-effort parquet write with explicit dtype coercion."""
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        _warn(f"pandas unavailable for parquet write: {type(e).__name__}: {e}")
        return False
    try:
        df = pd.DataFrame(rows, columns=list(OUTCOMES_MEMORY_COLUMNS))
        for col in (
            "target_weight",
            "estimated_notional",
            "plan_confidence",
            "nav_at_cycle",
            "nav_fwd_1d",
            "nav_fwd_5d",
            "nav_fwd_20d",
            "future_return_1d",
            "future_return_5d",
            "future_return_20d",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("would_execute", "requires_operator_confirmation", "outcome_known"):
            if col in df.columns:
                df[col] = df[col].map(_to_bool_optional)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return True
    except Exception as e:
        _warn(f"parquet write failed for {path}: {type(e).__name__}: {e}")
        return False


def _to_bool_optional(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("", "nan", "none", "null"):
        return None
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None


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
    if x is None or isinstance(x, bool):
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


def _norm_upper(x: Any, default: str = "UNKNOWN") -> str:
    s = str(x or "").strip().upper()
    return s or default


def _norm_lower(x: Any, default: str = "") -> str:
    s = str(x or "").strip().lower()
    return s or default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _iso_to_date(s: Any) -> Optional[str]:
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
    return (d + timedelta(days=days)).strftime("%Y-%m-%d")


# -----------------------------------------------------------
# NAV lookup (mirror of Step 16 convention)
# -----------------------------------------------------------
def _build_nav_lookup(history_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return {YYYY-MM-DD -> total_value} (last write per date wins)."""
    out: Dict[str, float] = {}
    if not history_rows:
        return out
    date_keys = ("date", "timestamp", "ts", "cycle_date")
    nav_keys = ("total_value", "nav", "portfolio_value", "market_value", "total_equity")
    sample = history_rows[0]
    date_col = next((c for c in date_keys if c in sample), None)
    nav_col = next((c for c in nav_keys if c in sample), None)
    if not date_col or not nav_col:
        return out
    for r in history_rows:
        d = _iso_to_date(r.get(date_col))
        if not d:
            continue
        v = _to_float(r.get(nav_col))
        if v is None:
            continue
        out[d] = v
    return out


def _nav_at(nav_lookup: Dict[str, float], date_str: str, scan_days: int = 5) -> Optional[float]:
    """Find NAV at date_str, scanning forward up to scan_days for weekends/holidays."""
    if not date_str:
        return None
    v = nav_lookup.get(date_str)
    if v is not None and v > 0:
        return v
    for offset in range(1, scan_days + 1):
        cand = _add_days(date_str, offset)
        if cand is None:
            continue
        v = nav_lookup.get(cand)
        if v is not None and v > 0:
            return v
    return None


def _forward_return(
    nav_lookup: Dict[str, float],
    base_date: str,
    days_ahead: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Return (base_nav, forward_nav, simple_return) for base_date -> +days_ahead.
    Both NAVs scan forward 5 days for weekend/holiday tolerance.
    """
    base_nav = _nav_at(nav_lookup, base_date)
    if base_nav is None:
        return None, None, None
    target = _add_days(base_date, days_ahead)
    if target is None:
        return base_nav, None, None
    fwd_nav = _nav_at(nav_lookup, target)
    if fwd_nav is None:
        return base_nav, None, None
    return base_nav, fwd_nav, (fwd_nav / base_nav) - 1.0


# -----------------------------------------------------------
# Enrichment
# -----------------------------------------------------------
def _outcome_label(
    *,
    would_execute: bool,
    action: str,
    forward_5d: Optional[float],
) -> Optional[str]:
    """
    Heuristic success label for an *individual shadow row*:
      * would_execute=True on a deploy action -> success iff fwd5d > 0
      * would_execute=True on a rotation/exit action -> success iff fwd5d <= 0
        (trimming/selling was vindicated by a flat-or-down portfolio path)
      * would_execute=False -> success iff fwd5d <= 0
        (correctly skipped a losing window)
    Returns 'success', 'fail', 'neutral', or None if no label is possible.
    """
    if forward_5d is None:
        return None
    act = (action or "").strip().lower()
    if would_execute:
        if act in DEPLOY_ACTIONS:
            return "success" if forward_5d > 0 else "fail"
        if act in ROTATION_ACTIONS:
            return "success" if forward_5d <= 0 else "fail"
        # generic action: success on a positive window
        return "success" if forward_5d > 0 else "fail"
    # Counterfactual discipline: skipping a positive window is a missed
    # opportunity; skipping a negative window is good discipline.
    if forward_5d <= 0:
        return "success"
    return "fail"


def _enrich_shadow_row(
    row: Dict[str, Any],
    nav_lookup: Dict[str, float],
    enriched_at: str,
) -> Dict[str, Any]:
    """Build the enriched outcomes row for a single shadow-memory row."""
    out: Dict[str, Any] = {}
    # Carry over Step 29 fields
    for col in OUTCOMES_MEMORY_COLUMNS:
        if col in row:
            out[col] = row[col]
        else:
            out[col] = None

    cycle_id = str(row.get("cycle_id") or row.get("timestamp_utc") or "")
    cycle_date = _iso_to_date(cycle_id)
    out["cycle_id"] = cycle_id
    out["timestamp_utc"] = str(row.get("timestamp_utc") or cycle_id)
    out["cycle_date"] = cycle_date
    out["would_execute"] = _to_bool_optional(row.get("would_execute"))
    out["requires_operator_confirmation"] = _to_bool_optional(
        row.get("requires_operator_confirmation")
    )

    nav_at = _nav_at(nav_lookup, cycle_date) if cycle_date else None
    out["nav_at_cycle"] = nav_at

    fwd_returns: Dict[int, Optional[float]] = {d: None for d in FORWARD_WINDOWS_DAYS}
    fwd_navs: Dict[int, Optional[float]] = {d: None for d in FORWARD_WINDOWS_DAYS}

    if cycle_date and nav_at is not None:
        for d in FORWARD_WINDOWS_DAYS:
            base_nav, fwd_nav, fwd_r = _forward_return(nav_lookup, cycle_date, d)
            fwd_returns[d] = fwd_r
            fwd_navs[d] = fwd_nav

    out["nav_fwd_1d"] = fwd_navs[1]
    out["nav_fwd_5d"] = fwd_navs[5]
    out["nav_fwd_20d"] = fwd_navs[20]
    out["future_return_1d"] = fwd_returns[1]
    out["future_return_5d"] = fwd_returns[5]
    out["future_return_20d"] = fwd_returns[20]

    primary = fwd_returns[PRIMARY_WINDOW_DAYS]
    out["outcome_known"] = primary is not None
    out["outcome_label"] = _outcome_label(
        would_execute=bool(out["would_execute"]),
        action=str(row.get("action") or ""),
        forward_5d=primary,
    )
    out["enriched_at_utc"] = enriched_at
    return out


def _enrich_all(
    shadow_rows: List[Dict[str, Any]],
    nav_lookup: Dict[str, float],
    enriched_at: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in shadow_rows:
        out.append(_enrich_shadow_row(r, nav_lookup, enriched_at))
    return out


# -----------------------------------------------------------
# Append-only outcomes memory dedup
# -----------------------------------------------------------
def _merge_outcomes_memory(
    existing_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Append new enriched rows. New observations replace older entries
    that share (cycle_id, ticker, action) so re-enriching a cycle
    overwrites prior placeholders without losing rows from other cycles.
    """
    merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in existing_rows:
        key = (str(r.get("cycle_id", "")), str(r.get("ticker", "")), str(r.get("action", "")))
        merged[key] = r
    for r in new_rows:
        key = (str(r.get("cycle_id", "")), str(r.get("ticker", "")), str(r.get("action", "")))
        merged[key] = r
    out = list(merged.values())
    for r in out:
        for c in OUTCOMES_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Metrics
# -----------------------------------------------------------
def _safe_mean(xs: List[float]) -> Optional[float]:
    return statistics.mean(xs) if xs else None


def _safe_median(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def _alpha_score(mean_return: Optional[float]) -> float:
    """Map mean forward return to [0,1] across +/- ALPHA_BAND."""
    if mean_return is None:
        return 0.5
    raw = (mean_return + ALPHA_BAND) / (2.0 * ALPHA_BAND)
    return _clamp(raw, 0.0, 1.0)


def _win_rate(returns: List[float], *, positive_threshold: float = 0.0) -> Optional[float]:
    if not returns:
        return None
    wins = sum(1 for r in returns if r > positive_threshold)
    return wins / len(returns)


def _drawdown_score(returns: List[float]) -> Optional[float]:
    """Fraction of windows that avoided breaching the drawdown floor."""
    if not returns:
        return None
    avoided = sum(1 for r in returns if r > DRAWDOWN_FLOOR)
    return avoided / len(returns)


def _discipline_score(rows: List[Dict[str, Any]]) -> Optional[float]:
    """
    Fraction of labelled rows whose ``outcome_label`` is 'success'.
    The label already encodes both action-correctness (deploys win
    when positive) and counterfactual discipline (skipped rows win
    when negative), so this is a single coherent score.
    """
    labelled = [r for r in rows if r.get("outcome_label") in ("success", "fail")]
    if not labelled:
        return None
    wins = sum(1 for r in labelled if r["outcome_label"] == "success")
    return wins / len(labelled)


def _deployment_quality(rows: List[Dict[str, Any]]) -> Optional[float]:
    deploys = [
        r
        for r in rows
        if (r.get("action") or "").strip().lower() in DEPLOY_ACTIONS
        and r.get("outcome_label") in ("success", "fail")
        and bool(r.get("would_execute"))
    ]
    if not deploys:
        return None
    wins = sum(1 for r in deploys if r["outcome_label"] == "success")
    return wins / len(deploys)


def _rotation_quality(rows: List[Dict[str, Any]]) -> Optional[float]:
    rots = [
        r
        for r in rows
        if (r.get("action") or "").strip().lower() in ROTATION_ACTIONS
        and r.get("outcome_label") in ("success", "fail")
        and bool(r.get("would_execute"))
    ]
    if not rots:
        return None
    wins = sum(1 for r in rots if r["outcome_label"] == "success")
    return wins / len(rots)


def _hypothetical_turnover(rows: List[Dict[str, Any]], nav_baseline: Optional[float]) -> float:
    """
    Sum of |estimated_notional| across would_execute=True rows,
    normalized by ``nav_baseline`` if available, else returned as
    absolute USD. Returned 0.0 when no rows.
    """
    if not rows:
        return 0.0
    total = 0.0
    for r in rows:
        if not bool(r.get("would_execute")):
            continue
        n = _to_float(r.get("estimated_notional"))
        if n is None:
            continue
        total += abs(n)
    if nav_baseline and nav_baseline > 0:
        return total / nav_baseline
    return total


def _bucket_metrics(rows: List[Dict[str, Any]], nav_baseline: Optional[float]) -> Dict[str, Any]:
    n = len(rows)
    returns_5d = [
        _to_float(r.get("future_return_5d"))
        for r in rows
        if _to_float(r.get("future_return_5d")) is not None
    ]
    returns_5d_clean: List[float] = [r for r in returns_5d if r is not None]
    success = _discipline_score(rows)
    return {
        "action_count": n,
        "labelled_count": sum(1 for r in rows if r.get("outcome_known")),
        "realized_return_mean": _safe_mean(returns_5d_clean),
        "realized_return_median": _safe_median(returns_5d_clean),
        "hypothetical_turnover": _hypothetical_turnover(rows, nav_baseline),
        "hypothetical_success_rate": success,
    }


# -----------------------------------------------------------
# Trajectory check
# -----------------------------------------------------------
def _trajectory_improving(rows: List[Dict[str, Any]]) -> bool:
    """
    Sort labelled rows by cycle_date. Compare mean forward_5d of the
    most recent TRAJECTORY_RECENT_FRACTION vs the older remainder.
    Returns True iff the recent slice has a strictly larger mean.
    """
    labelled = [
        (str(r.get("cycle_date") or r.get("cycle_id") or ""), _to_float(r.get("future_return_5d")))
        for r in rows
        if _to_float(r.get("future_return_5d")) is not None
    ]
    if len(labelled) < 6:
        return False
    labelled.sort(key=lambda t: t[0])
    cutoff = max(1, int(round(len(labelled) * (1.0 - TRAJECTORY_RECENT_FRACTION))))
    older = [r for _, r in labelled[:cutoff] if r is not None]
    recent = [r for _, r in labelled[cutoff:] if r is not None]
    if not older or not recent:
        return False
    return statistics.mean(recent) > statistics.mean(older)


# -----------------------------------------------------------
# Composite + verdict
# -----------------------------------------------------------
def _autonomy_readiness(
    *,
    alpha: float,
    win_rate: Optional[float],
    drawdown: Optional[float],
    discipline: Optional[float],
    governance: float,
) -> float:
    """Weighted blend; missing components fall back to neutral 0.5."""
    components = [
        (0.35, alpha),
        (0.25, win_rate if win_rate is not None else 0.5),
        (0.15, drawdown if drawdown is not None else 0.5),
        (0.15, discipline if discipline is not None else 0.5),
        (0.10, governance),
    ]
    total_w = sum(w for w, _ in components)
    score = sum(w * v for w, v in components) / total_w
    return _clamp(score, 0.0, 1.0)


def _verdict(
    *,
    n_labelled: int,
    autonomy_readiness: float,
    win_rate: Optional[float],
    trajectory_improving: bool,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    # 1. LEARNING (insufficient observations)
    if n_labelled < MIN_LEARNING_OBSERVATIONS:
        reasons.append(
            f"only {n_labelled} labelled observation(s); need "
            f">= {MIN_LEARNING_OBSERVATIONS} to evaluate trust"
        )
        return "LEARNING", reasons

    wr = win_rate if win_rate is not None else 0.5

    # 2. AUTONOMY_NOT_READY (clear weak signal with enough data)
    if autonomy_readiness < 0.40 or wr < 0.45:
        reasons.append(
            f"autonomy_readiness={autonomy_readiness:.2f}, win_rate={wr:.2f} "
            "indicate weak shadow performance"
        )
        return "AUTONOMY_NOT_READY", reasons

    # 3. AUTONOMY_CANDIDATE (strong sustained signal)
    if autonomy_readiness >= 0.70 and n_labelled >= MIN_CANDIDATE_OBSERVATIONS and wr >= 0.55:
        reasons.append(
            f"autonomy_readiness={autonomy_readiness:.2f} >= 0.70, "
            f"win_rate={wr:.2f} >= 0.55, "
            f"n_labelled={n_labelled} >= {MIN_CANDIDATE_OBSERVATIONS}"
        )
        return "AUTONOMY_CANDIDATE", reasons

    # 4. TRUST_BUILDING (stable positive)
    if autonomy_readiness >= 0.55 and n_labelled >= MIN_TRUST_OBSERVATIONS:
        reasons.append(
            f"autonomy_readiness={autonomy_readiness:.2f} >= 0.55, "
            f"n_labelled={n_labelled} >= {MIN_TRUST_OBSERVATIONS}"
        )
        return "TRUST_BUILDING", reasons

    # 5. IMPROVING (immature but positive trajectory)
    if trajectory_improving:
        reasons.append(
            f"autonomy_readiness={autonomy_readiness:.2f} immature, "
            f"but forward-return trajectory is improving"
        )
        return "IMPROVING", reasons

    reasons.append(
        f"autonomy_readiness={autonomy_readiness:.2f}, n_labelled={n_labelled}; "
        "insufficient signal to advance"
    )
    return "LEARNING", reasons


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
def _recommendations(
    *,
    verdict: str,
    n_labelled: int,
    autonomy_readiness: float,
    win_rate: Optional[float],
    governance: float,
) -> List[str]:
    recs: List[str] = []
    if verdict == "LEARNING":
        recs.append("Continue shadow observation -- sample size is too small for trust evaluation.")
        recs.append(
            f"Need at least {MIN_LEARNING_OBSERVATIONS} labelled observations; currently {n_labelled}."
        )
        recs.append("Extend apprenticeship period; do not enable any autonomous execution.")
        return recs
    if verdict == "AUTONOMY_NOT_READY":
        recs.append(
            "Restrict to shadow observation only; do not enable assisted or auto execution."
        )
        recs.append("Increase confidence and persistence thresholds to filter weaker setups.")
        recs.append("Investigate which regime/mode buckets dragged shadow performance down.")
        return recs
    if verdict == "IMPROVING":
        recs.append("Continue shadow observation; trajectory is positive but immature.")
        recs.append("Extend apprenticeship period until autonomy_readiness reaches 0.55.")
        return recs
    if verdict == "TRUST_BUILDING":
        recs.append(
            "Permit future assisted-deployment testing once governance maturity is confirmed."
        )
        recs.append("Compare shadow decisions against live portfolio for divergence analysis.")
        if governance < 0.60:
            recs.append(
                "Governance scorecard remains modest -- prioritise governance maturity first."
            )
        return recs
    # AUTONOMY_CANDIDATE
    recs.append(
        "Sustained strong shadow performance -- candidate for restricted assisted autonomy."
    )
    recs.append("Operator review still required before enabling any real-money execution.")
    recs.append(
        "Maintain shadow evaluation in parallel after any autonomy lift to detect regressions."
    )
    return recs


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    verdict: str,
    verdict_reasons: List[str],
    metrics: Dict[str, Any],
    by_mode: Dict[str, Dict[str, Any]],
    by_regime: Dict[str, Dict[str, Any]],
    recommendations: List[str],
    n_total_rows: int,
    n_labelled: int,
    autonomy_readiness: float,
) -> str:
    def fmt_pct(x: Optional[float]) -> str:
        if x is None:
            return "-"
        return f"{x * 100:.2f}%"

    def fmt_ratio(x: Optional[float]) -> str:
        if x is None:
            return "-"
        return f"{x:.3f}"

    def fmt_money(x: Optional[float]) -> str:
        if x is None:
            return "-"
        return f"${x:,.0f}"

    lines: List[str] = []
    lines.append("# Triton ARM Shadow Performance")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_")
    lines.append("")

    lines.append("## Apprenticeship Verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    for r in verdict_reasons:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| autonomy_readiness | {autonomy_readiness:.3f} |")
    lines.append(f"| total_shadow_rows | {n_total_rows} |")
    lines.append(f"| labelled_rows | {n_labelled} |")
    lines.append("")

    lines.append("## Shadow Performance")
    lines.append("")
    lines.append("| metric | score |")
    lines.append("|---|---|")
    lines.append(f"| shadow_alpha_score | {fmt_ratio(metrics.get('shadow_alpha_score'))} |")
    lines.append(f"| shadow_win_rate | {fmt_ratio(metrics.get('shadow_win_rate'))} |")
    lines.append(f"| shadow_drawdown_score | {fmt_ratio(metrics.get('shadow_drawdown_score'))} |")
    lines.append(
        f"| shadow_deployment_quality | {fmt_ratio(metrics.get('shadow_deployment_quality'))} |"
    )
    lines.append(
        f"| shadow_rotation_quality | {fmt_ratio(metrics.get('shadow_rotation_quality'))} |"
    )
    lines.append(
        f"| shadow_discipline_score | {fmt_ratio(metrics.get('shadow_discipline_score'))} |"
    )
    lines.append(
        f"| shadow_governance_score | {fmt_ratio(metrics.get('shadow_governance_score'))} |"
    )
    lines.append(
        f"| shadow_autonomy_readiness_score | {fmt_ratio(metrics.get('shadow_autonomy_readiness_score'))} |"
    )
    lines.append("")

    lines.append("## Mode Analysis")
    lines.append("")
    if by_mode:
        lines.append(
            "| mode | rows | labelled | mean_ret_5d | median_ret_5d | turnover | success_rate |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for mode in SHADOW_MODES_ANALYSED:
            m = by_mode.get(mode, {})
            lines.append(
                f"| {mode} | {m.get('action_count', 0)} | "
                f"{m.get('labelled_count', 0)} | "
                f"{fmt_pct(m.get('realized_return_mean'))} | "
                f"{fmt_pct(m.get('realized_return_median'))} | "
                f"{fmt_ratio(m.get('hypothetical_turnover'))} | "
                f"{fmt_ratio(m.get('hypothetical_success_rate'))} |"
            )
    else:
        lines.append("_(no mode buckets to analyse)_")
    lines.append("")

    lines.append("## Regime Analysis")
    lines.append("")
    if by_regime:
        lines.append(
            "| regime | rows | labelled | mean_ret_5d | median_ret_5d | turnover | success_rate |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for regime in sorted(by_regime.keys()):
            m = by_regime[regime]
            lines.append(
                f"| {regime} | {m.get('action_count', 0)} | "
                f"{m.get('labelled_count', 0)} | "
                f"{fmt_pct(m.get('realized_return_mean'))} | "
                f"{fmt_pct(m.get('realized_return_median'))} | "
                f"{fmt_ratio(m.get('hypothetical_turnover'))} | "
                f"{fmt_ratio(m.get('hypothetical_success_rate'))} |"
            )
    else:
        lines.append("_(no regime buckets to analyse)_")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for r in recommendations:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Narrative")
    lines.append("")
    if verdict == "LEARNING":
        narrative = (
            f"Triton is still in its apprenticeship: {n_labelled} of {n_total_rows} "
            f"shadow observations have realised forward returns, which is below the "
            f"{MIN_LEARNING_OBSERVATIONS}-observation threshold for trust evaluation. "
            f"Continue shadow operation until the sample matures."
        )
    elif verdict == "AUTONOMY_NOT_READY":
        narrative = (
            f"Shadow performance is materially weak (autonomy_readiness "
            f"{autonomy_readiness:.2f}). Triton has not earned the right to "
            f"any form of autonomy yet."
        )
    elif verdict == "IMPROVING":
        narrative = (
            f"Triton's shadow trajectory is improving. Autonomy readiness "
            f"({autonomy_readiness:.2f}) is still immature, so the apprenticeship "
            f"continues, but the trend is positive."
        )
    elif verdict == "TRUST_BUILDING":
        narrative = (
            f"Shadow outcomes are stable and positive (autonomy_readiness "
            f"{autonomy_readiness:.2f}, {n_labelled} labelled observations). "
            f"Triton is building the empirical record required for assisted-mode "
            f"testing."
        )
    else:  # AUTONOMY_CANDIDATE
        narrative = (
            f"Sustained strong shadow performance (autonomy_readiness "
            f"{autonomy_readiness:.2f}, {n_labelled} labelled observations) makes "
            f"Triton a candidate for restricted assisted autonomy under continued "
            f"operator review."
        )
    lines.append(narrative)
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def _load_shadow_memory(csv_path: Path, parquet_path: Path) -> List[Dict[str, Any]]:
    """Prefer parquet (richer dtypes); fall back to CSV."""
    rows = _safe_read_parquet_rows(parquet_path, label="arm_shadow_execution_memory.parquet")
    if rows:
        return rows
    return _safe_read_csv_rows(csv_path, label="arm_shadow_execution_memory.csv")


def build_performance_evaluation(
    *,
    shadow_rows: List[Dict[str, Any]],
    portfolio_history_rows: List[Dict[str, Any]],
    trade_log_rows: List[Dict[str, Any]],
    regime_json: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    meta_decision: Dict[str, Any],
    governance_scorecard: Dict[str, Any],
    existing_outcomes_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    enriched_at = _now_iso_utc()

    nav_lookup = _build_nav_lookup(portfolio_history_rows)
    enriched_rows = _enrich_all(shadow_rows, nav_lookup, enriched_at)
    merged_outcomes = _merge_outcomes_memory(existing_outcomes_rows, enriched_rows)

    # Analyse on the *full* merged history so older enriched rows
    # whose forward windows have now elapsed also contribute.
    analysis_rows = merged_outcomes

    # NAV baseline for turnover normalisation: latest NAV in the lookup
    nav_baseline: Optional[float] = None
    if nav_lookup:
        try:
            latest_date = max(nav_lookup.keys())
            nav_baseline = nav_lookup.get(latest_date)
        except Exception:
            nav_baseline = None

    # ---- Aggregate metrics ----
    labelled = [r for r in analysis_rows if r.get("outcome_known")]
    n_labelled = len(labelled)
    n_total = len(analysis_rows)

    returns_5d = [
        _to_float(r.get("future_return_5d"))
        for r in labelled
        if _to_float(r.get("future_return_5d")) is not None
    ]
    returns_5d_clean: List[float] = [r for r in returns_5d if r is not None]
    mean_ret = _safe_mean(returns_5d_clean)
    win_rate = _win_rate(returns_5d_clean)
    drawdown = _drawdown_score(returns_5d_clean)
    discipline = _discipline_score(analysis_rows)
    deployment_quality = _deployment_quality(analysis_rows)
    rotation_quality = _rotation_quality(analysis_rows)

    governance_score = _to_float(
        (governance_scorecard or {}).get("governance_quality_score")
        or (governance_scorecard or {}).get("intelligence_health_score")
    )
    if governance_score is None:
        governance_score = 0.5
    governance_score = _clamp(governance_score, 0.0, 1.0)

    alpha = _alpha_score(mean_ret)
    autonomy_readiness = _autonomy_readiness(
        alpha=alpha,
        win_rate=win_rate,
        drawdown=drawdown,
        discipline=discipline,
        governance=governance_score,
    )

    # ---- By-mode ----
    by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in SHADOW_MODES_ANALYSED:
        bucket = [r for r in analysis_rows if _norm_upper(r.get("shadow_mode")) == mode]
        by_mode[mode] = _bucket_metrics(bucket, nav_baseline)

    # ---- By-regime ----
    regimes = sorted({_norm_upper(r.get("regime")) for r in analysis_rows if r.get("regime")})
    by_regime: Dict[str, Dict[str, Any]] = {}
    for regime in regimes:
        bucket = [r for r in analysis_rows if _norm_upper(r.get("regime")) == regime]
        by_regime[regime] = _bucket_metrics(bucket, nav_baseline)

    trajectory = _trajectory_improving(analysis_rows)
    verdict, verdict_reasons = _verdict(
        n_labelled=n_labelled,
        autonomy_readiness=autonomy_readiness,
        win_rate=win_rate,
        trajectory_improving=trajectory,
    )
    recommendations = _recommendations(
        verdict=verdict,
        n_labelled=n_labelled,
        autonomy_readiness=autonomy_readiness,
        win_rate=win_rate,
        governance=governance_score,
    )

    metrics = {
        "shadow_alpha_score": round(alpha, 6),
        "shadow_win_rate": None if win_rate is None else round(win_rate, 6),
        "shadow_drawdown_score": None if drawdown is None else round(drawdown, 6),
        "shadow_deployment_quality": (
            None if deployment_quality is None else round(deployment_quality, 6)
        ),
        "shadow_rotation_quality": None if rotation_quality is None else round(rotation_quality, 6),
        "shadow_discipline_score": None if discipline is None else round(discipline, 6),
        "shadow_governance_score": round(governance_score, 6),
        "shadow_autonomy_readiness_score": round(autonomy_readiness, 6),
    }

    record: Dict[str, Any] = {
        "generated_at_utc": enriched_at,
        "engine": "arm_shadow_performance_evaluation_engine",
        "engine_version": 1,
        "apprenticeship_verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "metrics": metrics,
        "summary_stats": {
            "n_total_rows": n_total,
            "n_labelled": n_labelled,
            "n_unlabelled": n_total - n_labelled,
            "n_enriched_this_cycle": len(enriched_rows),
            "mean_return_5d": None if mean_ret is None else round(mean_ret, 6),
            "median_return_5d": _safe_median(returns_5d_clean),
            "trajectory_improving": trajectory,
            "nav_baseline": nav_baseline,
            "nav_lookup_days": len(nav_lookup),
        },
        "by_mode": by_mode,
        "by_regime": by_regime,
        "recommendations": recommendations,
        "regime_context": {
            "regime": _norm_upper((regime_json or {}).get("regime")),
            "runtime_policy_version": (runtime_policy or {}).get("policy_version"),
            "trust_level": _norm_upper((meta_decision or {}).get("trust_level")),
            "self_confidence_score": _to_float((meta_decision or {}).get("self_confidence_score")),
        },
        "trade_log_rows": len(trade_log_rows),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "observational_only": True,
        },
        "inputs_seen": {
            "shadow_memory_rows": len(shadow_rows),
            "portfolio_history_rows": len(portfolio_history_rows),
            "trade_log_rows": len(trade_log_rows),
            "adaptive_regime": bool(regime_json),
            "runtime_policy_governed": bool(runtime_policy),
            "meta_decision_intelligence": bool(meta_decision),
            "autonomous_governance_scorecard": bool(governance_scorecard),
            "existing_outcomes_memory_rows": len(existing_outcomes_rows),
        },
        "outcomes_memory_size_after_merge": len(merged_outcomes),
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": enriched_at,
        "engine": "arm_shadow_performance_evaluation_engine",
        "apprenticeship_verdict": verdict,
        "n_total_rows": n_total,
        "n_labelled": n_labelled,
        "shadow_alpha_score": metrics["shadow_alpha_score"],
        "shadow_win_rate": metrics["shadow_win_rate"],
        "shadow_discipline_score": metrics["shadow_discipline_score"],
        "shadow_governance_score": metrics["shadow_governance_score"],
        "shadow_autonomy_readiness_score": metrics["shadow_autonomy_readiness_score"],
        "mean_return_5d": record["summary_stats"]["mean_return_5d"],
        "trajectory_improving": trajectory,
        "regimes_analysed": regimes,
        "modes_analysed": list(SHADOW_MODES_ANALYSED),
        "n_recommendations": len(recommendations),
    }

    md = _render_markdown(
        generated_at=enriched_at,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
        metrics=metrics,
        by_mode=by_mode,
        by_regime=by_regime,
        recommendations=recommendations,
        n_total_rows=n_total,
        n_labelled=n_labelled,
        autonomy_readiness=autonomy_readiness,
    )
    return record, summary, md, merged_outcomes


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM shadow performance evaluation engine "
            "(Step 30). Enriches Step 29 shadow memory with realized "
            "portfolio forward returns, produces shadow performance "
            "metrics, and emits an apprenticeship verdict. No broker "
            "calls; no portfolio mutation."
        ),
    )
    p.add_argument("--shadow-memory-csv", default=str(DEFAULT_SHADOW_MEMORY_CSV))
    p.add_argument("--shadow-memory-parquet", default=str(DEFAULT_SHADOW_MEMORY_PARQUET))
    p.add_argument("--portfolio-history", default=str(DEFAULT_PORTFOLIO_HISTORY))
    p.add_argument("--trade-log", default=str(DEFAULT_TRADE_LOG))
    p.add_argument("--regime", default=str(DEFAULT_REGIME))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--meta-decision", default=str(DEFAULT_META_DECISION))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-mem-csv", default=str(DEFAULT_OUT_MEM_CSV))
    p.add_argument("--out-mem-parquet", default=str(DEFAULT_OUT_MEM_PQ))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        "[ARM_SHADOW_PERFORMANCE] starting (read-only shadow performance "
        "evaluation; no broker calls)",
        flush=True,
    )

    shadow_rows = _load_shadow_memory(
        Path(args.shadow_memory_csv),
        Path(args.shadow_memory_parquet),
    )
    portfolio_history_rows = _safe_read_csv_rows(
        Path(args.portfolio_history),
        label="portfolio_history.csv",
    )
    trade_log_rows = _safe_read_csv_rows(
        Path(args.trade_log),
        label="trade_log.csv",
    )
    regime_json = _safe_read_json(Path(args.regime), label="adaptive_regime.json")
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    meta_decision = _safe_read_json(
        Path(args.meta_decision), label="meta_decision_intelligence.json"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    existing_outcomes = _safe_read_csv_rows(
        Path(args.out_mem_csv),
        label="arm_shadow_outcomes_memory.csv",
    )

    record, summary, md, merged_outcomes = build_performance_evaluation(
        shadow_rows=shadow_rows,
        portfolio_history_rows=portfolio_history_rows,
        trade_log_rows=trade_log_rows,
        regime_json=regime_json,
        runtime_policy=runtime_policy,
        meta_decision=meta_decision,
        governance_scorecard=scorecard,
        existing_outcomes_rows=existing_outcomes,
    )

    try:
        _atomic_write_json(record, Path(args.out_json))
    except Exception as e:
        _warn(f"failed to write {args.out_json}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_text(md, Path(args.out_md))
    except Exception as e:
        _warn(f"failed to write {args.out_md}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(summary, Path(args.out_summary))
    except Exception as e:
        _warn(f"failed to write {args.out_summary}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_csv(merged_outcomes, Path(args.out_mem_csv), columns=OUTCOMES_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write outcomes csv {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_outcomes, Path(args.out_mem_parquet))

    metrics = record["metrics"]
    print(
        "[ARM_SHADOW_PERFORMANCE] "
        f"verdict={record['apprenticeship_verdict']} "
        f"observations={record['summary_stats']['n_labelled']} "
        f"shadow_alpha={metrics['shadow_alpha_score']:.3f} "
        f"discipline="
        f"{(metrics['shadow_discipline_score'] if metrics['shadow_discipline_score'] is not None else 0):.3f} "
        f"autonomy_readiness={metrics['shadow_autonomy_readiness_score']:.3f}",
        flush=True,
    )
    print(
        "[ARM_SHADOW_PERFORMANCE_SAFETY] broker_calls=0 orders_placed=0 " "portfolio_mutated=False",
        flush=True,
    )
    print(
        f"[ARM_SHADOW_PERFORMANCE_OUT] json={Path(args.out_json).as_posix()} "
        f"md={Path(args.out_md).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()} "
        f"outcomes_csv={Path(args.out_mem_csv).as_posix()} "
        f"outcomes_parquet={Path(args.out_mem_parquet).as_posix() if parquet_ok else 'SKIPPED'}",
        flush=True,
    )
    return 0


# -----------------------------------------------------------
# Self-inspection: enforce the no-broker safety rule at import time
# -----------------------------------------------------------
_FORBIDDEN_TOKENS: Tuple[str, ...] = (
    "Alpaca" + "Broker",
    "place" + "_order",
    "submit" + "_order",
    "execute" + "_trades",
    "place" + "_live_orders",
    "broker" + "_client",
)


def _self_check_no_broker_tokens() -> None:
    """Refuse to import if any forbidden broker token appears in source."""
    try:
        src = Path(__file__).read_text(encoding="utf-8")
    except Exception:
        return
    for tok in _FORBIDDEN_TOKENS:
        if tok in src:
            raise RuntimeError(f"[ARM_SHADOW_PERF_SAFETY] forbidden broker token detected: {tok!r}")


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
