# services/feedback_loop.py
"""
TRITON — Feedback Loop (observational + advisory layer).

Purpose
-------
Read trade outcomes, signal/decision context, edge sizing, and execution
intelligence diagnostics, then emit:

  - data/results/feedback_loop_report.csv
        per-trade / per-order feedback observations (one row each).
  - data/results/feedback_loop_summary.json
        aggregate performance grouped by execution-risk / spread / style /
        sizing / signal / liquidity / quote-freshness, plus ranked
        recommendations and source-availability metadata.
  - data/results/feedback_recommendations.csv
        flattened recommendations table (advisory only).

Design principles
-----------------
1. Read-only relative to live trading logic.
   This module never mutates the existing CSVs it reads.

2. Conservative & explainable.
   Recommendations always carry evidence_count / evidence_strength /
   recommendation_confidence and a metric_snapshot string so a human can
   audit why the engine flagged something.

3. Robust to missing data.
   Any input file may be absent / empty / malformed — the loop still
   produces stable outputs (with feedback_quality=LOW for thin records and
   `missing_inputs` recorded in the summary).

4. No hidden self-modification.
   This is Phase 1 — the feedback loop only writes its own report files.
   It does not push thresholds, weights, or styles back into the live
   trading layer.

Run
---
    python -m services.feedback_loop
or
    python services/feedback_loop.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

INPUT_PATHS: Dict[str, Path] = {
    "trade_log": RESULTS / "trade_log.csv",
    "execution_intelligence": RESULTS / "execution_intelligence.csv",
    "execution_plan": RESULTS / "execution_plan.csv",
    "reprice_open_orders": RESULTS / "reprice_open_orders.csv",
    "signal_lifecycle": RESULTS / "signal_lifecycle.csv",
    "trade_opportunities": RESULTS / "trade_opportunities.csv",
    "target_weights": RESULTS / "target_weights.csv",
    "live_orders_log": RESULTS / "live_orders_log.csv",
    "portfolio_history": RESULTS / "portfolio_history.csv",
}

REPORT_CSV = RESULTS / "feedback_loop_report.csv"
SUMMARY_JSON = RESULTS / "feedback_loop_summary.json"
RECOMMENDATIONS_CSV = RESULTS / "feedback_recommendations.csv"

# ─────────────────────────────────────────────────────────────
# Constants & small enums
# ─────────────────────────────────────────────────────────────

EVIDENCE_LOW = "LOW"
EVIDENCE_MEDIUM = "MEDIUM"
EVIDENCE_HIGH = "HIGH"

FEEDBACK_LOW = "LOW"
FEEDBACK_MEDIUM = "MEDIUM"
FEEDBACK_HIGH = "HIGH"

# Liquidity-pressure buckets used when notional_vs_liquidity is present.
NVL_BUCKETS: List[Tuple[str, float]] = [
    ("LOW", 0.005),  # <= 0.005
    ("MEDIUM", 0.02),  # <= 0.02
    ("HIGH", float("inf")),
]


# ─────────────────────────────────────────────────────────────
# Safe IO helpers
# ─────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_csv_safe(path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Best-effort CSV loader.

    Returns (df, status) where status ∈ {"ok", "missing", "empty", "error:<msg>"}.
    Always returns a DataFrame (possibly empty) — never raises.
    """
    try:
        if not path.exists():
            return pd.DataFrame(), "missing"
        try:
            stat = path.stat()
            if stat.st_size == 0:
                return pd.DataFrame(), "empty"
        except OSError:
            pass
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        if df is None:
            return pd.DataFrame(), "empty"
        if df.empty:
            return df, "empty"
        # Strip whitespace from column names for safer matching downstream.
        df.columns = [str(c).strip() for c in df.columns]
        return df, "ok"
    except Exception as e:  # absolute last-resort guard
        return pd.DataFrame(), f"error:{type(e).__name__}:{str(e)[:120]}"


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


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    v = _safe_float(x, None)
    return int(v) if v is not None else default


def _to_dt(x: Any) -> Optional[pd.Timestamp]:
    try:
        ts = pd.to_datetime(x, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────


def normalize_symbol(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    return "" if s in ("NAN", "NONE") else s


def normalize_side(x: Any) -> str:
    """Map various order-side / action conventions to {BUY, SELL, ""}."""
    s = str(x or "").strip().upper()
    if s in ("BUY", "B", "LONG", "ENTRY", "ADD"):
        return "BUY"
    if s in ("SELL", "S", "SHORT", "EXIT", "TRIM", "CLOSE"):
        return "SELL"
    return s if s in ("BUY", "SELL") else ""


def normalize_action(x: Any) -> str:
    """Coarse action class: BUY / ADD / TRIM / EXIT / SELL / HOLD / OTHER."""
    s = str(x or "").strip().upper()
    for known in ("ADD", "TRIM", "EXIT", "BUY", "SELL", "HOLD"):
        if known == s:
            return known
    if "ADD" in s:
        return "ADD"
    if "TRIM" in s:
        return "TRIM"
    if "EXIT" in s:
        return "EXIT"
    if "BUY" in s or "ENTRY" in s:
        return "BUY"
    if "SELL" in s:
        return "SELL"
    return "OTHER"


def _ensure_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a canonical 'symbol' column derived from ticker if needed."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "symbol" not in out.columns and "ticker" in out.columns:
        out["symbol"] = out["ticker"]
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].map(normalize_symbol)
    return out


# ─────────────────────────────────────────────────────────────
# Liquidity bucket helper
# ─────────────────────────────────────────────────────────────


def liquidity_bucket(nvl: Any) -> str:
    v = _safe_float(nvl, None)
    if v is None:
        return "UNKNOWN"
    for label, upper in NVL_BUCKETS:
        if v <= upper:
            return label
    return "HIGH"


# ─────────────────────────────────────────────────────────────
# Source loading + availability tracking
# ─────────────────────────────────────────────────────────────


@dataclass
class FeedbackInputs:
    sources: Dict[str, pd.DataFrame] = field(default_factory=dict)
    status: Dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> pd.DataFrame:
        return self.sources.get(name, pd.DataFrame())

    def has(self, name: str) -> bool:
        df = self.sources.get(name)
        return df is not None and not df.empty

    def missing(self) -> List[str]:
        return [n for n, s in self.status.items() if s != "ok"]


def load_inputs() -> FeedbackInputs:
    inp = FeedbackInputs()
    for name, path in INPUT_PATHS.items():
        df, status = load_csv_safe(path)
        inp.sources[name] = _ensure_symbol(df) if not df.empty else df
        inp.status[name] = status
    return inp


# ─────────────────────────────────────────────────────────────
# Per-record feedback rows
# ─────────────────────────────────────────────────────────────


def _latest_per_symbol(df: pd.DataFrame, time_col_candidates: Tuple[str, ...]) -> pd.DataFrame:
    """Return one row per symbol, preferring the most recent by an available time col."""
    if df is None or df.empty or "symbol" not in df.columns:
        return df.copy() if df is not None else pd.DataFrame()
    out = df.copy()
    tcol = next((c for c in time_col_candidates if c in out.columns), None)
    if tcol is not None:
        out["__ts"] = pd.to_datetime(out[tcol], errors="coerce", utc=True)
        out = out.sort_values(["symbol", "__ts"], na_position="first")
        out = out.drop_duplicates(subset=["symbol"], keep="last")
        out = out.drop(columns=["__ts"])
    else:
        out = out.drop_duplicates(subset=["symbol"], keep="last")
    return out


def _attach_optional(
    left: pd.DataFrame, right: pd.DataFrame, *, cols: List[str], suffix: str = ""
) -> pd.DataFrame:
    """
    Left-join `right` (deduped on 'symbol') into `left` for the given cols.

    Spine columns are the source of truth — we only fill in cols that
    don't already exist (or are entirely empty) in `left`. This avoids
    pandas auto-suffixing collisions like `signal_x` / `signal_y`.
    """
    if right is None or right.empty or "symbol" not in right.columns:
        return left
    if suffix:
        keep = [c for c in cols if c in right.columns]
        if not keep:
            return left
        sub = right[["symbol"] + keep].copy()
        sub = sub.drop_duplicates(subset=["symbol"], keep="last")
        sub = sub.rename(columns={c: f"{c}{suffix}" for c in keep})
        return left.merge(sub, on="symbol", how="left")

    keep: List[str] = []
    for c in cols:
        if c not in right.columns:
            continue
        if c in left.columns:
            try:
                non_empty = left[c].notna() & (left[c].astype(str).str.strip() != "")
                if non_empty.any():
                    continue  # spine already has this populated
            except Exception:
                continue
        keep.append(c)
    if not keep:
        return left

    sub = right[["symbol"] + keep].copy()
    sub = sub.drop_duplicates(subset=["symbol"], keep="last")
    out = left.copy()
    if any(c in out.columns for c in keep):
        # Drop empty pre-existing columns so the merge doesn't auto-suffix.
        drop_cols = [c for c in keep if c in out.columns]
        out = out.drop(columns=drop_cols)
    return out.merge(sub, on="symbol", how="left")


def build_feedback_records(inp: FeedbackInputs) -> pd.DataFrame:
    """
    Build one row per feedback observation.

    Strategy (best-effort, never strict):
      - Spine = trade_log.csv if present (one row per closed trade).
      - Otherwise, spine = the most-recent per-symbol view of
        execution_intelligence.csv / execution_plan.csv / live_orders_log.csv,
        whichever is non-empty first.
      - Attach signal/decision context per symbol from signal_lifecycle.csv.
      - Attach edge / sizing context per symbol from trade_opportunities.csv.
      - Attach execution context per symbol from execution_intelligence.csv,
        execution_plan.csv (fallback), reprice_open_orders.csv.
      - Compute feedback_quality + matched_sources_count from how much of
        each context bucket was found.
    """
    trade_log = inp.get("trade_log")
    ei = inp.get("execution_intelligence")
    plan = inp.get("execution_plan")
    reprice = inp.get("reprice_open_orders")
    lifecycle = inp.get("signal_lifecycle")
    opps = inp.get("trade_opportunities")
    live_log = inp.get("live_orders_log")

    # ── Build spine ────────────────────────────────────────────────────
    if trade_log is not None and not trade_log.empty and "ticker" in trade_log.columns:
        spine = trade_log.copy()
        spine["symbol"] = spine["ticker"].map(normalize_symbol)
        spine_source = "trade_log"
    else:
        spine_candidates = [
            ("execution_intelligence", ei),
            ("execution_plan", plan),
            ("live_orders_log", live_log),
        ]
        spine = pd.DataFrame()
        spine_source = ""
        for label, df in spine_candidates:
            if df is not None and not df.empty and "symbol" in df.columns:
                spine = df.copy()
                spine_source = label
                break
    if spine.empty:
        return pd.DataFrame()

    # Canonicalize identity columns.
    spine["symbol"] = spine["symbol"].map(normalize_symbol)
    if "action" in spine.columns:
        spine["action"] = spine["action"].map(normalize_action)
    elif "stance" in spine.columns:
        spine["action"] = spine["stance"].map(normalize_action)
    else:
        spine["action"] = ""
    if "side" in spine.columns:
        spine["side"] = spine["side"].map(normalize_side)
    else:
        # Best-effort: derive a side from the action class.
        action_to_side = {
            "BUY": "BUY",
            "ADD": "BUY",
            "SELL": "SELL",
            "TRIM": "SELL",
            "EXIT": "SELL",
        }
        spine["side"] = spine["action"].map(lambda a: action_to_side.get(str(a), ""))

    # Best available date column for the spine.
    for cand in ("date", "timestamp", "submitted_at", "generated_at", "created_at"):
        if cand in spine.columns:
            spine["__date"] = pd.to_datetime(spine[cand], errors="coerce", utc=True)
            break
    else:
        spine["__date"] = pd.NaT

    # ── Attach signal/decision context ─────────────────────────────────
    lc_latest = _latest_per_symbol(lifecycle, ("date", "generated_at_utc", "generated_at_local"))
    spine = _attach_optional(
        spine,
        lc_latest,
        cols=[
            "decision_action",
            "decision_reason",
            "signal",
            "confidence",
            "regime",
            "rationale",
        ],
    )

    # ── Attach edge / sizing context ───────────────────────────────────
    op_latest = _latest_per_symbol(opps, ())
    spine = _attach_optional(
        spine,
        op_latest,
        cols=[
            "edge_score",
            "edge_rank",
            "edge_percentile",
            "sizing_bucket",
            "allocation_multiplier",
            "allocation_reason",
            "portfolio_adjustment_factor",
        ],
    )

    # ── Attach execution intelligence context (sidecar preferred) ──────
    ei_latest = _latest_per_symbol(ei, ("timestamp",))
    spine = _attach_optional(
        spine,
        ei_latest,
        cols=[
            "execution_style",
            "execution_aggressiveness",
            "execution_reason",
            "execution_skip_flag",
            "execution_skip_reason",
            "execution_quality_score",
            "execution_quality_reason",
            "execution_risk_flag",
            "spread_bucket",
            "spread_bps",
            "quote_is_stale",
            "quote_age_sec",
            "liquidity_proxy",
            "order_notional",
            "notional_vs_liquidity",
            "expected_slippage_bps",
            "realized_slippage_bps",
        ],
    )
    # Fallback: use execution_plan for any EI cols that are still missing.
    plan_latest = _latest_per_symbol(plan, ())
    plan_cols_to_fill = [
        "execution_style",
        "execution_aggressiveness",
        "execution_reason",
        "execution_skip_flag",
        "execution_skip_reason",
        "execution_quality_score",
        "execution_quality_reason",
        "execution_risk_flag",
        "spread_bucket",
        "spread_bps",
        "quote_is_stale",
        "quote_age_sec",
        "liquidity_proxy",
        "order_notional",
        "notional_vs_liquidity",
        "expected_slippage_bps",
        "realized_slippage_bps",
    ]
    if plan_latest is not None and not plan_latest.empty:
        for col in plan_cols_to_fill:
            if col in plan_latest.columns:
                tmp = plan_latest[["symbol", col]].rename(columns={col: f"__plan_{col}"})
                spine = spine.merge(tmp.drop_duplicates(subset=["symbol"]), on="symbol", how="left")
                if col in spine.columns:
                    spine[col] = spine[col].where(
                        spine[col].notna() & (spine[col].astype(str).str.strip() != ""),
                        spine[f"__plan_{col}"],
                    )
                else:
                    spine[col] = spine[f"__plan_{col}"]
                spine = spine.drop(columns=[f"__plan_{col}"])

    # ── Attach repricing / partial-fill context ────────────────────────
    rep_latest = _latest_per_symbol(reprice, ("timestamp",))
    spine = _attach_optional(
        spine,
        rep_latest,
        cols=[
            "fill_pct",
            "partial_fill_action",
            "partial_fill_reason",
        ],
    )

    # Repricing happened? Detect by presence of a row in reprice_open_orders.
    if rep_latest is not None and not rep_latest.empty and "symbol" in rep_latest.columns:
        repriced_syms = set(rep_latest["symbol"].dropna().unique().tolist())
    else:
        repriced_syms = set()
    spine["repriced_in_session"] = spine["symbol"].isin(repriced_syms)

    # ── Outcome columns from trade_log (always best-effort) ────────────
    if "profit" in spine.columns:
        spine["pnl"] = pd.to_numeric(spine["profit"], errors="coerce")
    elif "pnl" not in spine.columns:
        spine["pnl"] = np.nan

    if "entry_price" in spine.columns and "exit_price" in spine.columns:
        ep = pd.to_numeric(spine["entry_price"], errors="coerce")
        xp = pd.to_numeric(spine["exit_price"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            spine["pnl_pct"] = ((xp - ep) / ep * 100.0).where(ep > 0)
    elif "pnl_pct" not in spine.columns:
        spine["pnl_pct"] = np.nan

    if "holding_days" in spine.columns:
        spine["hold_days"] = pd.to_numeric(spine["holding_days"], errors="coerce")
    else:
        spine["hold_days"] = np.nan

    # ── Was the order filled / skipped ? ───────────────────────────────
    if "exit_price" in spine.columns:
        # closed trade implies a fill occurred at entry
        spine["filled"] = pd.to_numeric(spine["exit_price"], errors="coerce").notna()
    elif "status" in spine.columns:
        st_norm = spine["status"].astype(str).str.lower()
        spine["filled"] = st_norm.isin(["filled", "partially_filled", "partial", "done"])
    else:
        spine["filled"] = pd.NA

    if "execution_skip_flag" in spine.columns:
        spine["skipped"] = (
            spine["execution_skip_flag"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "t"])
        )
    else:
        spine["skipped"] = False

    # Liquidity bucket derived from notional_vs_liquidity.
    if "notional_vs_liquidity" in spine.columns:
        spine["liquidity_pressure_bucket"] = spine["notional_vs_liquidity"].map(liquidity_bucket)
    else:
        spine["liquidity_pressure_bucket"] = "UNKNOWN"

    # ── Feedback quality scoring ───────────────────────────────────────
    matched_cols = ["symbol"]
    quality_groups = {
        "decision": ["decision_action", "signal", "confidence"],
        "edge": ["edge_score", "sizing_bucket", "allocation_multiplier"],
        "execution_ctx": [
            "execution_style",
            "execution_risk_flag",
            "execution_quality_score",
            "spread_bucket",
        ],
        "outcome": ["pnl", "pnl_pct", "filled"],
        "reprice": ["partial_fill_action", "fill_pct"],
    }

    def _has_value(s: pd.Series) -> pd.Series:
        return s.notna() & (s.astype(str).str.strip() != "")

    matched_count = pd.Series(0, index=spine.index, dtype="int")
    for gname, cols in quality_groups.items():
        present_cols = [c for c in cols if c in spine.columns]
        if not present_cols:
            continue
        any_present = pd.Series(False, index=spine.index)
        for c in present_cols:
            any_present = any_present | _has_value(spine[c])
        matched_count = matched_count + any_present.astype(int)
        matched_cols.extend(present_cols)

    def _quality(n: int) -> str:
        if n >= 4:
            return FEEDBACK_HIGH
        if n >= 2:
            return FEEDBACK_MEDIUM
        return FEEDBACK_LOW

    spine["matched_sources_count"] = matched_count
    spine["feedback_quality"] = matched_count.map(_quality)
    spine["spine_source"] = spine_source
    if "session" not in spine.columns:
        spine["session"] = ""

    # ── Final output schema (curated, then any extras) ─────────────────
    preferred = [
        "symbol",
        "__date",
        "session",
        "side",
        "action",
        "signal",
        "decision_action",
        "decision_reason",
        "edge_score",
        "edge_rank",
        "edge_percentile",
        "sizing_bucket",
        "allocation_multiplier",
        "allocation_reason",
        "portfolio_adjustment_factor",
        "execution_style",
        "execution_aggressiveness",
        "execution_quality_score",
        "execution_risk_flag",
        "spread_bucket",
        "spread_bps",
        "quote_is_stale",
        "quote_age_sec",
        "notional_vs_liquidity",
        "liquidity_pressure_bucket",
        "partial_fill_action",
        "partial_fill_reason",
        "fill_pct",
        "expected_slippage_bps",
        "realized_slippage_bps",
        "pnl",
        "pnl_pct",
        "hold_days",
        "filled",
        "skipped",
        "repriced_in_session",
        "feedback_quality",
        "matched_sources_count",
        "spine_source",
    ]
    cols_present = [c for c in preferred if c in spine.columns]
    extras = [
        c
        for c in spine.columns
        if c not in cols_present and not c.startswith("__") and c not in ("ticker",)
    ]
    out = spine[cols_present + extras].copy()
    out = out.rename(columns={"__date": "date"})
    return out


# ─────────────────────────────────────────────────────────────
# Grouped metrics
# ─────────────────────────────────────────────────────────────


def _safe_mean(s: pd.Series) -> Optional[float]:
    try:
        s2 = pd.to_numeric(s, errors="coerce").dropna()
        if s2.empty:
            return None
        v = float(s2.mean())
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _safe_winrate(s: pd.Series) -> Optional[float]:
    try:
        s2 = pd.to_numeric(s, errors="coerce").dropna()
        if s2.empty:
            return None
        return float((s2 > 0).mean())
    except Exception:
        return None


def _safe_fillrate(s: pd.Series) -> Optional[float]:
    try:
        s2 = s.dropna()
        if s2.empty:
            return None
        s3 = s2.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])
        return float(s3.mean())
    except Exception:
        return None


def _group_metrics(df: pd.DataFrame, by: str) -> Dict[str, Dict[str, Any]]:
    """Aggregate outcome metrics by a single categorical column."""
    if df is None or df.empty or by not in df.columns:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    grouped = df.groupby(df[by].fillna("(missing)").astype(str), dropna=False)
    for key, sub in grouped:
        out[str(key)] = {
            "count": int(sub.shape[0]),
            "avg_pnl": _safe_mean(sub["pnl"]) if "pnl" in sub.columns else None,
            "avg_pnl_pct": _safe_mean(sub["pnl_pct"]) if "pnl_pct" in sub.columns else None,
            "win_rate": _safe_winrate(sub["pnl"]) if "pnl" in sub.columns else None,
            "avg_realized_slippage_bps": (
                _safe_mean(sub["realized_slippage_bps"])
                if "realized_slippage_bps" in sub.columns
                else None
            ),
            "avg_expected_slippage_bps": (
                _safe_mean(sub["expected_slippage_bps"])
                if "expected_slippage_bps" in sub.columns
                else None
            ),
            "fill_rate": _safe_fillrate(sub["filled"]) if "filled" in sub.columns else None,
        }
    return out


def compute_grouped_metrics(records: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {
        "by_execution_risk_flag": _group_metrics(records, "execution_risk_flag"),
        "by_spread_bucket": _group_metrics(records, "spread_bucket"),
        "by_execution_style": _group_metrics(records, "execution_style"),
        "by_sizing_bucket": _group_metrics(records, "sizing_bucket"),
        "by_signal": _group_metrics(records, "signal"),
        "by_decision_action": _group_metrics(records, "decision_action"),
        "by_quote_is_stale": _group_metrics(records, "quote_is_stale"),
        "by_liquidity_pressure_bucket": _group_metrics(records, "liquidity_pressure_bucket"),
        "by_partial_fill_action": _group_metrics(records, "partial_fill_action"),
        "by_action": _group_metrics(records, "action"),
    }


# ─────────────────────────────────────────────────────────────
# Evidence + recommendations
# ─────────────────────────────────────────────────────────────


def evidence_strength_for(count: int) -> str:
    if count < 5:
        return EVIDENCE_LOW
    if count < 15:
        return EVIDENCE_MEDIUM
    return EVIDENCE_HIGH


def recommendation_confidence(count: int, effect_size: Optional[float]) -> float:
    """
    Combine sample-size confidence with effect magnitude.
    Returns a value in [0, 1]. Conservative — small samples cap at 0.50.
    """
    if count <= 0:
        return 0.0
    # Sample-size factor: saturates around N=50.
    n_factor = min(1.0, math.log10(count + 1) / math.log10(51))
    # Effect-size factor: tanh of |effect| scaled.
    if effect_size is None or not math.isfinite(effect_size):
        eff_factor = 0.5
    else:
        eff_factor = math.tanh(abs(effect_size))
    base = 0.5 * n_factor + 0.5 * eff_factor
    if count < 5:
        base = min(base, 0.50)
    return float(round(max(0.0, min(1.0, base)), 4))


def _fmt_metric(v: Optional[float], fmt: str = "{:.3f}") -> str:
    if v is None or not math.isfinite(v):
        return "N/A"
    return fmt.format(v)


def _baseline_avg(records: pd.DataFrame, col: str) -> Optional[float]:
    if records is None or records.empty or col not in records.columns:
        return None
    return _safe_mean(records[col])


def _format_metric_snapshot(stats: Dict[str, Any]) -> str:
    parts = [
        f"n={stats.get('count', 0)}",
    ]
    if stats.get("avg_pnl") is not None:
        parts.append(f"avg_pnl={_fmt_metric(stats['avg_pnl'])}")
    if stats.get("win_rate") is not None:
        parts.append(f"win_rate={_fmt_metric(stats['win_rate'])}")
    if stats.get("avg_realized_slippage_bps") is not None:
        parts.append(f"avg_slip_bps={_fmt_metric(stats['avg_realized_slippage_bps'], '{:.2f}')}")
    if stats.get("fill_rate") is not None:
        parts.append(f"fill_rate={_fmt_metric(stats['fill_rate'])}")
    return ", ".join(parts)


def build_recommendations(
    records: pd.DataFrame, grouped: Dict[str, Dict[str, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Apply the documented recommendation rules and emit a list of dicts.

    Each recommendation includes:
      recommendation_type, recommendation_text, evidence_count,
      evidence_strength, recommendation_confidence,
      related_bucket / related_flag / related_style, metric_snapshot.
    """
    recs: List[Dict[str, Any]] = []
    if records is None or records.empty:
        return recs

    baseline_pnl = _baseline_avg(records, "pnl") or 0.0
    baseline_slip = _baseline_avg(records, "realized_slippage_bps") or 0.0
    baseline_winrate = (_safe_winrate(records["pnl"]) if "pnl" in records.columns else None) or 0.5

    def _add(
        rec_type: str,
        text: str,
        *,
        count: int,
        effect: Optional[float],
        related_bucket: str = "",
        related_flag: str = "",
        related_style: str = "",
        metric_snapshot: str = "",
    ) -> None:
        recs.append(
            {
                "recommendation_type": rec_type,
                "recommendation_text": text,
                "evidence_count": int(count),
                "evidence_strength": evidence_strength_for(int(count)),
                "recommendation_confidence": recommendation_confidence(int(count), effect),
                "related_bucket": related_bucket,
                "related_flag": related_flag,
                "related_style": related_style,
                "metric_snapshot": metric_snapshot,
            }
        )

    # 1. EXECUTION_CAUTION — HIGH execution_risk_flag underperforms
    risk_groups = grouped.get("by_execution_risk_flag", {})
    high_risk = risk_groups.get("HIGH") or {}
    if high_risk.get("count", 0) > 0:
        slip = high_risk.get("avg_realized_slippage_bps")
        pnl = high_risk.get("avg_pnl")
        worse_slip = slip is not None and slip > baseline_slip + 5.0
        worse_pnl = pnl is not None and pnl < baseline_pnl
        if worse_slip or worse_pnl:
            effect = max(
                (slip - baseline_slip) / 50.0 if slip is not None else 0.0,
                (
                    (baseline_pnl - pnl) / max(1.0, abs(baseline_pnl) + 1.0)
                    if pnl is not None
                    else 0.0
                ),
            )
            _add(
                "EXECUTION_CAUTION",
                "HIGH execution-risk entries show worse realized slippage / PnL — "
                "consider reducing aggressiveness or trust for these trades.",
                count=high_risk.get("count", 0),
                effect=effect,
                related_flag="HIGH",
                metric_snapshot=_format_metric_snapshot(high_risk),
            )

    # 2. SPREAD_CAUTION — WIDE / TOO_WIDE spread buckets underperform
    spread_groups = grouped.get("by_spread_bucket", {})
    for bucket in ("WIDE", "TOO_WIDE"):
        stats = spread_groups.get(bucket) or {}
        if stats.get("count", 0) <= 0:
            continue
        slip = stats.get("avg_realized_slippage_bps")
        pnl = stats.get("avg_pnl")
        if (slip is not None and slip > baseline_slip + 3.0) or (
            pnl is not None and pnl < baseline_pnl
        ):
            effect = (
                (slip - baseline_slip) / 30.0
                if slip is not None
                else (
                    (baseline_pnl - pnl) / max(1.0, abs(baseline_pnl) + 1.0)
                    if pnl is not None
                    else 0.0
                )
            )
            _add(
                "SPREAD_CAUTION",
                f"{bucket} spread environments underperform — "
                "consider penalizing or deferring entries during wide-spread conditions.",
                count=stats.get("count", 0),
                effect=effect,
                related_bucket=bucket,
                metric_snapshot=_format_metric_snapshot(stats),
            )

    # 3. SIZING_REVIEW / 4. EDGE_VALIDATION — by sizing_bucket
    sizing_groups = grouped.get("by_sizing_bucket", {})
    for bucket, stats in sizing_groups.items():
        if not stats or stats.get("count", 0) <= 0:
            continue
        pnl = stats.get("avg_pnl")
        wr = stats.get("win_rate")
        if pnl is None and wr is None:
            continue
        if (pnl is not None and pnl < baseline_pnl) or (
            wr is not None and wr < baseline_winrate - 0.10
        ):
            effect = max(
                (
                    (baseline_pnl - pnl) / max(1.0, abs(baseline_pnl) + 1.0)
                    if pnl is not None
                    else 0.0
                ),
                (baseline_winrate - wr) if wr is not None else 0.0,
            )
            _add(
                "SIZING_REVIEW",
                f"Sizing bucket '{bucket}' underperforms baseline — reassess allocation policy.",
                count=stats.get("count", 0),
                effect=effect,
                related_bucket=bucket,
                metric_snapshot=_format_metric_snapshot(stats),
            )
        elif "HIGH" in str(bucket).upper() and (
            (pnl is not None and pnl > baseline_pnl)
            or (wr is not None and wr > baseline_winrate + 0.10)
        ):
            effect = max(
                (
                    (pnl - baseline_pnl) / max(1.0, abs(baseline_pnl) + 1.0)
                    if pnl is not None
                    else 0.0
                ),
                (wr - baseline_winrate) if wr is not None else 0.0,
            )
            _add(
                "EDGE_VALIDATION",
                f"High-conviction sizing bucket '{bucket}' consistently outperforms — "
                "maintain or strengthen trust in this bucket.",
                count=stats.get("count", 0),
                effect=effect,
                related_bucket=bucket,
                metric_snapshot=_format_metric_snapshot(stats),
            )

    # 5. QUOTE_FRESHNESS_WARNING — stale quotes correlate with worse outcomes
    stale_groups = grouped.get("by_quote_is_stale", {})
    stale_stats = (
        stale_groups.get("True") or stale_groups.get("TRUE") or stale_groups.get("true") or {}
    )
    if stale_stats.get("count", 0) > 0:
        slip = stale_stats.get("avg_realized_slippage_bps")
        pnl = stale_stats.get("avg_pnl")
        if (slip is not None and slip > baseline_slip + 3.0) or (
            pnl is not None and pnl < baseline_pnl
        ):
            effect = (
                (slip - baseline_slip) / 30.0
                if slip is not None
                else (
                    (baseline_pnl - pnl) / max(1.0, abs(baseline_pnl) + 1.0)
                    if pnl is not None
                    else 0.0
                )
            )
            _add(
                "QUOTE_FRESHNESS_WARNING",
                "Stale-quote orders correlate with worse outcomes — "
                "increase caution when quote freshness is poor.",
                count=stale_stats.get("count", 0),
                effect=effect,
                related_flag="STALE",
                metric_snapshot=_format_metric_snapshot(stale_stats),
            )

    # 6. STYLE_REVIEW — styles that underperform meaningfully
    style_groups = grouped.get("by_execution_style", {})
    for style, stats in style_groups.items():
        if not stats or stats.get("count", 0) <= 0:
            continue
        pnl = stats.get("avg_pnl")
        if pnl is not None and pnl < baseline_pnl * 0.75 and stats.get("count", 0) >= 5:
            effect = (baseline_pnl - pnl) / max(1.0, abs(baseline_pnl) + 1.0)
            _add(
                "STYLE_REVIEW",
                f"Execution style '{style}' underperforms baseline — "
                "review aggressiveness mapping for this style.",
                count=stats.get("count", 0),
                effect=effect,
                related_style=style,
                metric_snapshot=_format_metric_snapshot(stats),
            )

    # 7 + 8. SIGNAL trust boosts and cautions
    sig_groups = grouped.get("by_signal", {})
    for sig, stats in sig_groups.items():
        if not stats or stats.get("count", 0) <= 0:
            continue
        wr = stats.get("win_rate")
        if wr is None:
            continue
        if wr > baseline_winrate + 0.10 and stats.get("count", 0) >= 3:
            _add(
                "SIGNAL_TRUST_BOOST",
                f"Signal '{sig}' shows above-baseline win rate — consider boosting trust.",
                count=stats.get("count", 0),
                effect=(wr - baseline_winrate),
                related_flag=str(sig),
                metric_snapshot=_format_metric_snapshot(stats),
            )
        elif wr < baseline_winrate - 0.15 and stats.get("count", 0) >= 3:
            _add(
                "SIGNAL_CAUTION",
                f"Signal '{sig}' shows below-baseline win rate — consider reducing trust.",
                count=stats.get("count", 0),
                effect=(baseline_winrate - wr),
                related_flag=str(sig),
                metric_snapshot=_format_metric_snapshot(stats),
            )

    # Sort: highest confidence first, ties broken by evidence_count.
    recs.sort(key=lambda r: (r["recommendation_confidence"], r["evidence_count"]), reverse=True)
    return recs


# ─────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────


def _df_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame for stable CSV output.

    - Deduplicate column names (keep first) to avoid `df[col]` returning
      a DataFrame instead of a Series during downstream processing.
    - Cast bool columns to "1"/"0"/"" (no Python True/False / NaN noise).
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated(keep="first")]
    bool_cols = [out.columns[i] for i, dt in enumerate(out.dtypes) if dt == bool]
    for col in bool_cols:
        s = out[col]
        out = out.drop(columns=[col])
        out[col] = s.astype("Int64").astype(str).replace({"<NA>": ""})
    return out


def write_outputs(
    records: pd.DataFrame,
    grouped: Dict[str, Dict[str, Dict[str, Any]]],
    recommendations: List[Dict[str, Any]],
    inp: FeedbackInputs,
    notes: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Write report CSV, recommendations CSV, and summary JSON. Never raises."""
    notes = notes or []
    written: Dict[str, str] = {}

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    try:
        if records is None or records.empty:
            pd.DataFrame(
                columns=[
                    "symbol",
                    "date",
                    "session",
                    "side",
                    "action",
                    "feedback_quality",
                    "matched_sources_count",
                ]
            ).to_csv(REPORT_CSV, index=False)
        else:
            _df_for_csv(records).to_csv(REPORT_CSV, index=False)
        written["report_csv"] = str(REPORT_CSV)
    except Exception as e:
        written["report_csv_error"] = f"{type(e).__name__}:{e}"

    try:
        if recommendations:
            pd.DataFrame(recommendations).to_csv(RECOMMENDATIONS_CSV, index=False)
        else:
            pd.DataFrame(
                columns=[
                    "recommendation_type",
                    "recommendation_text",
                    "evidence_count",
                    "evidence_strength",
                    "recommendation_confidence",
                    "related_bucket",
                    "related_flag",
                    "related_style",
                    "metric_snapshot",
                ]
            ).to_csv(RECOMMENDATIONS_CSV, index=False)
        written["recommendations_csv"] = str(RECOMMENDATIONS_CSV)
    except Exception as e:
        written["recommendations_csv_error"] = f"{type(e).__name__}:{e}"

    feedback_quality_counts: Dict[str, int] = {}
    if records is not None and not records.empty and "feedback_quality" in records.columns:
        try:
            feedback_quality_counts = (
                records["feedback_quality"]
                .fillna("")
                .astype(str)
                .value_counts(dropna=False)
                .to_dict()
            )
        except Exception:
            feedback_quality_counts = {}

    summary: Dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "schema_version": 1,
        "advisory_only": True,
        "source_availability": {
            name: {
                "status": inp.status.get(name, "missing"),
                "rows": (
                    int(inp.sources.get(name, pd.DataFrame()).shape[0])
                    if name in inp.sources
                    else 0
                ),
                "path": str(INPUT_PATHS[name]),
            }
            for name in INPUT_PATHS
        },
        "missing_inputs": inp.missing(),
        "record_counts": {
            "feedback_records": int(records.shape[0]) if records is not None else 0,
            "by_feedback_quality": feedback_quality_counts,
            "spine_source": (
                str(records["spine_source"].iloc[0])
                if records is not None and not records.empty and "spine_source" in records.columns
                else ""
            ),
        },
        "aggregate_performance": grouped,
        "top_recommendations": recommendations[:10],
        "recommendation_counts_by_type": _counts_by_type(recommendations),
        "notes": notes,
    }

    try:
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        written["summary_json"] = str(SUMMARY_JSON)
    except Exception as e:
        written["summary_json_error"] = f"{type(e).__name__}:{e}"

    return written


def _counts_by_type(recs: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in recs:
        t = str(r.get("recommendation_type", ""))
        out[t] = out.get(t, 0) + 1
    return out


# ─────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────


def run_feedback_loop(verbose: bool = True) -> Dict[str, Any]:
    """Run the full pipeline. Always returns a result dict, never raises."""
    inp = load_inputs()
    notes: List[str] = []
    if inp.missing():
        notes.append(
            "Missing or unreadable input(s): "
            + ", ".join(inp.missing())
            + ". Feedback was computed on best-effort partial data."
        )

    try:
        records = build_feedback_records(inp)
    except Exception as e:
        records = pd.DataFrame()
        notes.append(f"build_feedback_records error: {type(e).__name__}: {e}")

    if records.empty:
        notes.append(
            "No feedback rows produced — no usable spine data "
            "(trade_log / execution_intelligence / execution_plan / live_orders_log)."
        )

    try:
        grouped = compute_grouped_metrics(records)
    except Exception as e:
        grouped = {}
        notes.append(f"compute_grouped_metrics error: {type(e).__name__}: {e}")

    try:
        recommendations = build_recommendations(records, grouped)
    except Exception as e:
        recommendations = []
        notes.append(f"build_recommendations error: {type(e).__name__}: {e}")

    if records is not None and not records.empty:
        thin = (
            (records["matched_sources_count"] < 2).mean()
            if "matched_sources_count" in records.columns
            else 0
        )
        if thin > 0.5:
            notes.append(
                "Thin data: more than half of feedback rows had matched_sources_count<2; "
                "treat recommendations as low-confidence."
            )

    written = write_outputs(records, grouped, recommendations, inp, notes=notes)

    if verbose:
        print(f"[feedback_loop] sources_ok={[n for n,s in inp.status.items() if s=='ok']}")
        print(f"[feedback_loop] missing={inp.missing()}")
        print(
            f"[feedback_loop] records={0 if records is None else records.shape[0]} "
            f"recommendations={len(recommendations)}"
        )
        for k, v in written.items():
            print(f"[feedback_loop] {k}: {v}")

    return {
        "records": records,
        "grouped": grouped,
        "recommendations": recommendations,
        "written": written,
        "notes": notes,
        "source_status": inp.status,
    }


def main(argv: Optional[List[str]] = None) -> int:
    _ = argv
    try:
        run_feedback_loop(verbose=True)
        return 0
    except Exception as e:
        print(f"[feedback_loop] FATAL {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
