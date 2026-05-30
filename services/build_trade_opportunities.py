"""
build_trade_opportunities.py
----------------------------
Build execution-ready trade opportunities from signal_lifecycle_effective.csv.

Input:  data/results/signal_lifecycle_effective.csv (read-only)
Output: data/results/trade_opportunities.csv

Classification uses (effective_position_state, lifecycle_action) so broker-truth position
pairs with lifecycle intent — not effective_stance, which may be WAIT/HOLD after reconciliation.

Post-lifecycle drops (only): invalid price, qty=0, hard risk blocks (read-only from state files).
Diagnostics: data/results/trade_opportunity_build_diagnostics.json, trade_opportunity_build_drops.csv

If strict opportunity_count is 0 after the above, optionally inject up to N exploratory FLAT→ENTRY rows
(exploration_flag=True). Exploration pool may apply confidence/delta filters (diagnostic only for that path).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from services.lifecycle_truth import evaluate_lifecycle_gate, summarize_opportunity_build
from services.edge_ranking import (
    EnrichmentSpec,
    enrich_with_edge,
    FILTERED_BUCKET,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_IN = RESULTS_DIR / "signal_lifecycle_effective.csv"
DEFAULT_OUT = RESULTS_DIR / "trade_opportunities.csv"
DIAG_JSON = RESULTS_DIR / "trade_opportunity_build_diagnostics.json"
DROPS_CSV = RESULTS_DIR / "trade_opportunity_build_drops.csv"
# Optional source of feature-component scores produced by services/generate_signals.py.
# Read-only; absence is tolerated.
SIGNALS_WITH_RATIONALE_CSV = RESULTS_DIR / "signals_with_rationale.csv"
# Optional execution-context sources — read-only, written by execute_trades /
# place_live_orders. Absence, emptiness, and missing symbols are all tolerated.
EXECUTION_INTELLIGENCE_CSV = RESULTS_DIR / "execution_intelligence.csv"
EXECUTION_PLAN_CSV = RESULTS_DIR / "execution_plan.csv"
EXPLORATION_TOP_N = 3

# Canonical "no information" marker for execution-context columns. Downstream
# code (adaptation_simulation, dashboards) treats UNKNOWN as "do not count
# toward spread/risk matching" — that's the whole point: honest absence.
EXEC_CTX_UNKNOWN = "UNKNOWN"

# Edge columns appended to trade_opportunities.csv for downstream sizing/observability.
# Downstream code that does not understand them must continue to ignore them safely.
EDGE_OUTPUT_COLS = [
    "edge_score",
    "edge_rank",
    "edge_percentile",
    "sizing_bucket",
    "allocation_multiplier",
    "allocation_reason",
    # Interpretable score components emitted by services/trade_rationale.py
    # so dashboards / diagnostics can see *why* a score is high or low.
    # Downstream code that does not understand them ignores them safely.
    "score",
    "momentum_score",
    "trend_score",
    "breakout_score",
    "volatility_score",
    "delta_conviction",
    "score_model_component",
    "score_rank_component",
    "score_quality_component",
    "score_penalty_component",
    "score_final_preclip",
    "components_present",
]

# Execution-context columns appended to trade_opportunities.csv. These are
# read-only diagnostics sourced from execution_intelligence.csv /
# execution_plan.csv — purely additive, downstream consumers that do not
# understand them ignore them safely. Their purpose is to let
# adaptation_simulation.py and dashboards see spread / liquidity / stale-quote
# context per opportunity row instead of having to join independently.
EXEC_CONTEXT_COLS = [
    "spread_bucket",
    "spread_bps",
    "spread_pct",
    "execution_risk_flag",
    "execution_style",
    "quote_staleness_flag",
    "quote_is_stale",
    "quote_reason",
    "liquidity_bucket",
    "liquidity_pressure_bucket",
    "execution_context_source",
]

# Symbols not executable via typical broker APIs (indices, etc.)
INVALID_SYMBOLS = ["^VIX"]

CONTEXT_COLS = [
    "ticker",
    "effective_stance",
    "effective_position_state",
    "lifecycle_decision_reason",
    "confidence",
    "delta_pct",
    "rationale",
    "healed",
    "heal_reason",
    "reason_code",
    "lifecycle_authoritative_source",
    "lifecycle_consistency",
    "execution_blocked",
    "execution_block_reason",
    "reconciled_with_broker",
    "reconciled_reason",
    "stance_adjustment",
]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_effective(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing effective lifecycle file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("signal_lifecycle_effective.csv must include ticker")
    if "effective_position_state" not in df.columns:
        raise ValueError("signal_lifecycle_effective.csv must include effective_position_state")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df


def _series_lifecycle_action(df: pd.DataFrame) -> pd.Series:
    """Prefer lifecycle_action; else stance (raw lifecycle output before effective_stance overlay)."""
    if "lifecycle_action" in df.columns:
        s = df["lifecycle_action"]
    elif "stance" in df.columns:
        s = df["stance"]
    else:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    return s.fillna("").astype(str).str.strip().str.upper()


def classify_opportunity_from_lifecycle(pos_u: str, lifecycle_action_u: str) -> Optional[str]:
    """
    Map effective position + lifecycle intent → opportunity_type.
    Authoritative: ENTRY = FLAT+BUY; ADD = LONG+ADD only (LONG+BUY must not appear after effective build).
    """
    p = str(pos_u or "").strip().upper()
    a = str(lifecycle_action_u or "").strip().upper()
    if p == "FLAT" and a == "BUY":
        return "ENTRY"
    if p == "LONG" and a == "ADD":
        return "ADD"
    if p == "LONG" and a == "TRIM":
        return "TRIM"
    if p == "LONG" and a == "EXIT":
        return "EXIT"
    return None


def _input_decision_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """
    Per-row lifecycle intent on the effective input: ENTRY/ADD/TRIM/EXIT match
    `classify_opportunity_from_lifecycle`; HOLD counts `lifecycle_action` == HOLD
    (FLAT or LONG), so the mix includes no-trade stances.
    """
    if df is None or df.empty:
        return {"entry": 0, "add": 0, "hold": 0, "trim": 0, "exit": 0}
    pos = df["effective_position_state"].fillna("").astype(str).str.strip().str.upper()
    la = _series_lifecycle_action(df)
    entry = int(((pos == "FLAT") & (la == "BUY")).sum())
    add = int(((pos == "LONG") & (la == "ADD")).sum())
    hold = int((la == "HOLD").sum())
    trim = int(((pos == "LONG") & (la == "TRIM")).sum())
    exit_n = int(((pos == "LONG") & (la == "EXIT")).sum())
    return {
        "entry": entry,
        "add": add,
        "hold": hold,
        "trim": trim,
        "exit": exit_n,
    }


def _intent_stance_for_opportunity(opportunity_type: str) -> str:
    return {"ENTRY": "BUY", "ADD": "ADD", "TRIM": "TRIM", "EXIT": "EXIT"}.get(
        str(opportunity_type or "").strip().upper(), ""
    )


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_hard_risk_flags() -> Tuple[bool, bool]:
    """
    allow_new_orders, allow_new_trades — False means hard block (mirror execute_trades / risk snapshot).
    """
    allow_new_orders = True
    allow_new_trades = True

    rj = _load_json(RESULTS_DIR / "adaptive_risk_state.json")
    if isinstance(rj, dict):
        ctrl = rj.get("controls") if isinstance(rj.get("controls"), dict) else {}
        if "risk_on" in ctrl:
            allow_new_orders = allow_new_orders and bool(ctrl.get("risk_on", True))
        if "allow_new_orders" in ctrl:
            allow_new_orders = allow_new_orders and bool(ctrl.get("allow_new_orders", True))

    for cpm_path in (
        RESULTS_DIR / "capital_preservation_mode.json",
        RESULTS_DIR / "capital_preservation_state.json",
    ):
        cj = _load_json(cpm_path)
        if isinstance(cj, dict) and "allow_new_trades" in cj:
            allow_new_trades = allow_new_trades and bool(cj.get("allow_new_trades", True))
            break

    return allow_new_orders, allow_new_trades


def _row_invalid_price(row: pd.Series) -> bool:
    """True if close is unusable for sizing reference."""
    if "close" not in row.index:
        return False
    c = pd.to_numeric(row.get("close"), errors="coerce")
    if pd.isna(c) or float(c) <= 0:
        return True
    return False


def _row_invalid_qty_zero(row: pd.Series) -> bool:
    """True only when an explicit qty column exists and is numerically zero."""
    for q in ("qty", "target_qty", "planned_qty", "order_qty"):
        if q not in row.index:
            continue
        v = row.get(q)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        try:
            if float(v) == 0.0:
                return True
        except Exception:
            continue
    return False


def _risk_guard_drops_opportunity(
    opportunity_type: str,
    allow_new_orders: bool,
    allow_new_trades: bool,
) -> bool:
    ot = str(opportunity_type or "").strip().upper()
    if not allow_new_trades:
        return True
    if ot in ("ENTRY", "ADD") and not allow_new_orders:
        return True
    return False


def _output_columns(sub: pd.DataFrame) -> list[str]:
    head = ["ticker", "effective_stance", "effective_position_state"]
    optional = [c for c in CONTEXT_COLS if c not in head and c in sub.columns]
    tail = ["opportunity_type", "exploration_flag"]
    edge_tail = [c for c in EDGE_OUTPUT_COLS if c in sub.columns]
    exec_tail = [c for c in EXEC_CONTEXT_COLS if c in sub.columns]
    return [c for c in head if c in sub.columns] + optional + tail + edge_tail + exec_tail


def _load_feature_components_optional() -> Optional[pd.DataFrame]:
    """
    Best-effort load of per-ticker feature-component scores produced by
    services/generate_signals.py. Returns None if unavailable / empty / malformed.

    Looks for the most recent row per ticker in signals_with_rationale.csv and
    returns just the columns relevant to edge scoring. Never raises.
    """
    path = SIGNALS_WITH_RATIONALE_CSV
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df is None or df.empty or "ticker" not in df.columns:
        return None
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    keep_candidates = [
        "ticker",
        "score",
        "momentum_score",
        "trend_score",
        "breakout_score",
        "volatility_score",
        # Interpretability components emitted by services/trade_rationale.py.
        # Optional; best-effort load — absence is tolerated.
        "delta_conviction",
        "score_model_component",
        "score_rank_component",
        "score_quality_component",
        "score_penalty_component",
        "score_final_preclip",
        "components_present",
    ]
    keep = [c for c in keep_candidates if c in df.columns]
    if "ticker" not in keep or len(keep) <= 1:
        # Nothing useful beyond the join key — skip silently.
        return None
    # Pick the latest row per ticker if a date column exists; else last by file order.
    if "date" in df.columns:
        try:
            df["_d"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values(["ticker", "_d"], kind="mergesort").drop(columns=["_d"])
        except Exception:
            pass
    df = df[keep].drop_duplicates(subset=["ticker"], keep="last")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Execution-context enrichment
# ──────────────────────────────────────────────────────────────
#
# Why we do this here (and not inside adaptation_simulation.py):
#
# Historically, trade_opportunities.csv shipped zero execution-context
# fields, which meant adaptation_simulation.py could never match
# spread / liquidity / stale-quote rules on any opportunity row — every
# wide_spread_entry_penalty or execution-based adjustment landed at
# rows_matched = 0, purely because the data plumbing didn't carry
# reality into the row. This module is the natural place to fix that:
# it's the one writer of trade_opportunities.csv, so a left-join here
# means every downstream consumer (simulation, dashboards, audits) gets
# a consistent, honest snapshot of execution conditions per symbol.
#
# Non-goals (strict):
#   * no changes to simulation / execution / lifecycle / broker logic
#   * no fabrication of spread or risk data — missing → UNKNOWN
#   * no crash on missing / empty / malformed source CSVs
#
# Inputs:
#   data/results/execution_intelligence.csv   (primary — richest data)
#   data/results/execution_plan.csv           (fallback — same fields, fewer)
#
# Both are best-effort, read-only. Absence is tolerated silently: the
# output simply fills UNKNOWN for every row.


_EXEC_SOURCE_COLS = (
    "spread_bucket",
    "execution_risk_flag",
    "quote_is_stale",
    "quote_reason",
    "liquidity_reason",
    "liquidity_pressure_bucket",
    "notional_vs_liquidity",
    "execution_style",
    "spread_bps",
    "spread_pct",
)


def _safe_float_or_none(v: Any) -> Optional[float]:
    """Return v as float, or None if not numeric / NaN / blank."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and pd.isna(v):
            return None
    except Exception:
        pass
    try:
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null", "unknown", "na", "n/a"):
            return None
        return float(s)
    except Exception:
        return None


def _norm_upper_or_unknown(val: Any) -> str:
    """Strip + uppercase; convert NaN / blank / 'nan' / 'none' to UNKNOWN."""
    if val is None:
        return EXEC_CTX_UNKNOWN
    try:
        if isinstance(val, float) and pd.isna(val):
            return EXEC_CTX_UNKNOWN
    except Exception:
        pass
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return EXEC_CTX_UNKNOWN
    return s.upper()


def _bucket_from_notional_vs_liquidity(v: Any) -> str:
    """
    Derive a simple liquidity bucket from notional_vs_liquidity (fraction of
    available liquidity consumed). Thresholds are intentionally conservative
    and match the spirit of execution_intelligence.csv — small notional
    versus liquidity = LOW pressure, big notional = HIGH pressure.
    """
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return EXEC_CTX_UNKNOWN
        f = float(v)
    except Exception:
        return EXEC_CTX_UNKNOWN
    if f < 0.01:
        return "LOW"
    if f < 0.05:
        return "NORMAL"
    if f < 0.15:
        return "ELEVATED"
    return "HIGH"


def _load_execution_context_optional() -> Optional[pd.DataFrame]:
    """
    Best-effort latest-per-symbol execution context snapshot.

    Returns a DataFrame keyed on uppercase `ticker`, carrying:
        spread_bucket, execution_risk_flag, quote_is_stale,
        quote_staleness_flag, liquidity_bucket, execution_context_source
    Returns None when neither source file is usable (caller then fills
    UNKNOWN for every row).
    """

    def _read(path: Path, source_tag: str) -> Optional[pd.DataFrame]:
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if df is None or df.empty or "symbol" not in df.columns:
            return None
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        # Preserve file order; later rows for a symbol win (append-ish files).
        df["__order"] = range(len(df))
        df = (
            df.sort_values("__order", kind="mergesort")
            .drop(columns=["__order"])
            .reset_index(drop=True)
        )
        df["ticker"] = df["symbol"].astype(str).str.strip().str.upper()
        keep = ["ticker"] + [c for c in _EXEC_SOURCE_COLS if c in df.columns]
        df = df[keep].drop_duplicates(subset=["ticker"], keep="last")
        df["execution_context_source"] = source_tag
        return df.reset_index(drop=True)

    primary = _read(EXECUTION_INTELLIGENCE_CSV, "execution_intelligence")
    fallback = _read(EXECUTION_PLAN_CSV, "execution_plan")

    if primary is None and fallback is None:
        return None

    # Prefer primary rows; for symbols only present in fallback, pad them in.
    if primary is None:
        combined = fallback
    elif fallback is None:
        combined = primary
    else:
        primary_syms = set(primary["ticker"].astype(str))
        fill = fallback[~fallback["ticker"].astype(str).isin(primary_syms)]
        combined = pd.concat([primary, fill], ignore_index=True, sort=False)

    if combined is None or combined.empty:
        return None

    # Normalize outputs. quote_is_stale is preserved as raw (bool-ish) AND
    # also surfaced as an uppercased string flag so categorical consumers
    # don't have to re-parse it.
    def _stale_bool(v: Any) -> Optional[bool]:
        if v is None:
            return None
        try:
            if isinstance(v, float) and pd.isna(v):
                return None
        except Exception:
            pass
        s = str(v).strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
        return None

    for col in (
        "spread_bucket",
        "execution_risk_flag",
        "quote_is_stale",
        "quote_reason",
        "liquidity_reason",
        "liquidity_pressure_bucket",
        "notional_vs_liquidity",
        "execution_style",
        "spread_bps",
        "spread_pct",
    ):
        if col not in combined.columns:
            combined[col] = pd.Series([None] * len(combined))

    combined["spread_bucket"] = combined["spread_bucket"].apply(_norm_upper_or_unknown)
    combined["execution_risk_flag"] = combined["execution_risk_flag"].apply(_norm_upper_or_unknown)
    combined["execution_style"] = combined["execution_style"].apply(_norm_upper_or_unknown)
    combined["quote_reason"] = combined["quote_reason"].apply(_norm_upper_or_unknown)
    combined["liquidity_pressure_bucket"] = combined["liquidity_pressure_bucket"].apply(
        _norm_upper_or_unknown
    )
    # Numeric passthroughs — preserve None so consumers can detect "missing".
    combined["spread_bps"] = combined["spread_bps"].apply(_safe_float_or_none)
    combined["spread_pct"] = combined["spread_pct"].apply(_safe_float_or_none)
    stale_bools = combined["quote_is_stale"].apply(_stale_bool)
    combined["quote_is_stale"] = stale_bools
    combined["quote_staleness_flag"] = stale_bools.apply(
        lambda b: EXEC_CTX_UNKNOWN if b is None else ("STALE" if b else "FRESH")
    )
    combined["liquidity_bucket"] = combined["notional_vs_liquidity"].apply(
        _bucket_from_notional_vs_liquidity
    )

    out_cols = [
        "ticker",
        "spread_bucket",
        "spread_bps",
        "spread_pct",
        "execution_risk_flag",
        "execution_style",
        "quote_is_stale",
        "quote_staleness_flag",
        "quote_reason",
        "liquidity_bucket",
        "liquidity_pressure_bucket",
        "execution_context_source",
    ]
    combined = combined[out_cols].drop_duplicates(subset=["ticker"], keep="last")
    return combined.reset_index(drop=True)


def _merge_execution_context(
    out_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Left-join execution context onto opportunities on `ticker`.

    Adds spread_bucket, execution_risk_flag, quote_is_stale (bool),
    quote_staleness_flag (STALE/FRESH/UNKNOWN), liquidity_bucket, and
    execution_context_source. Missing rows fall back to UNKNOWN — never
    crash. Returns (df, diagnostics).
    """
    diag: Dict[str, Any] = {
        "available": False,
        "source_files": [],
        "rows_in_context_source": 0,
        "matched_symbols": 0,
        "total_symbols": 0,
        "coverage_ratio": 0.0,
        "unknown_rows": 0,
        "note": "",
    }
    if out_df is None or out_df.empty:
        diag["note"] = "empty_opportunities"
        return out_df, diag

    enriched = out_df.copy()

    ctx = _load_execution_context_optional()
    sources: List[str] = []
    if EXECUTION_INTELLIGENCE_CSV.exists() and EXECUTION_INTELLIGENCE_CSV.stat().st_size > 0:
        sources.append(EXECUTION_INTELLIGENCE_CSV.name)
    if EXECUTION_PLAN_CSV.exists() and EXECUTION_PLAN_CSV.stat().st_size > 0:
        sources.append(EXECUTION_PLAN_CSV.name)
    diag["source_files"] = sources

    if ctx is None or ctx.empty:
        # Safe fallback: fill every row with UNKNOWN so downstream consumers
        # see a stable schema.
        for col in (
            "spread_bucket",
            "execution_risk_flag",
            "execution_style",
            "quote_staleness_flag",
            "quote_reason",
            "liquidity_bucket",
            "liquidity_pressure_bucket",
        ):
            enriched[col] = EXEC_CTX_UNKNOWN
        enriched["quote_is_stale"] = pd.Series([None] * len(enriched))
        enriched["spread_bps"] = pd.Series([None] * len(enriched))
        enriched["spread_pct"] = pd.Series([None] * len(enriched))
        enriched["execution_context_source"] = "missing"
        diag["total_symbols"] = (
            int(enriched["ticker"].nunique()) if "ticker" in enriched.columns else 0
        )
        diag["unknown_rows"] = len(enriched)
        diag["note"] = (
            "execution_intelligence.csv / execution_plan.csv unavailable — "
            "filled UNKNOWN for every opportunity row"
        )
        return enriched, diag

    diag["available"] = True
    diag["rows_in_context_source"] = int(len(ctx))

    # Keep only the context columns we intend to expose; avoid clobbering
    # any same-named column that already exists on the opportunity row.
    ctx_cols_to_add = [c for c in EXEC_CONTEXT_COLS if c in ctx.columns]
    merge_cols = ["ticker"] + [c for c in ctx_cols_to_add if c not in enriched.columns]
    if len(merge_cols) <= 1:
        # Nothing new to add (opportunities already carry these). Still
        # succeed with a zero-coverage diagnostic so we don't silently skip.
        diag["note"] = "all_exec_context_cols_already_present_on_opportunities"
        diag["total_symbols"] = int(enriched["ticker"].nunique())
        return enriched, diag

    try:
        enriched = enriched.merge(ctx[merge_cols], on="ticker", how="left")
    except Exception as exc:  # pragma: no cover — defensive
        diag["note"] = f"merge_failed:{exc!r} — fallback to UNKNOWN"
        for col in EXEC_CONTEXT_COLS:
            if col not in enriched.columns:
                enriched[col] = EXEC_CTX_UNKNOWN
        return enriched, diag

    # Normalize / fill missing after the join. We must apply UNKNOWN fill per
    # column so a symbol that wasn't in the context source gets a stable
    # string value instead of NaN.
    for col in (
        "spread_bucket",
        "execution_risk_flag",
        "execution_style",
        "quote_staleness_flag",
        "quote_reason",
        "liquidity_bucket",
        "liquidity_pressure_bucket",
        "execution_context_source",
    ):
        if col in enriched.columns:
            enriched[col] = enriched[col].apply(_norm_upper_or_unknown)
    # quote_is_stale is a bool (or None) — leave it as-is. NaN rows are
    # honest "don't know" values; downstream (simulation) has a safe
    # reader for that. spread_bps / spread_pct stay numeric-or-None.
    for num_col in ("spread_bps", "spread_pct"):
        if num_col in enriched.columns:
            enriched[num_col] = enriched[num_col].apply(_safe_float_or_none)

    # Coverage: ratio of unique opportunity tickers for which the context
    # source contributed ANY non-UNKNOWN signal (spread / risk / staleness /
    # liquidity). A symbol whose exec-context row exists but is all UNKNOWN
    # is counted as non-matched — honest "we have no usable signal".
    if "ticker" in enriched.columns:
        unique_tickers = enriched["ticker"].astype(str).str.upper().unique().tolist()
        diag["total_symbols"] = len(unique_tickers)
        if len(unique_tickers) > 0:
            cond = pd.Series(False, index=enriched.index)
            for col in (
                "spread_bucket",
                "execution_risk_flag",
                "execution_style",
                "quote_staleness_flag",
                "quote_reason",
                "liquidity_bucket",
                "liquidity_pressure_bucket",
            ):
                if col in enriched.columns:
                    cond = cond | (enriched[col].astype(str).str.upper() != EXEC_CTX_UNKNOWN)
            for num_col in ("spread_bps", "spread_pct"):
                if num_col in enriched.columns:
                    cond = cond | enriched[num_col].notna()
            if "quote_is_stale" in enriched.columns:
                cond = cond | enriched["quote_is_stale"].apply(lambda v: v is True or v is False)
            matched = enriched.loc[cond]["ticker"].astype(str).str.upper().unique().tolist()
            diag["matched_symbols"] = len(matched)
            diag["coverage_ratio"] = round(len(matched) / len(unique_tickers), 6)
    diag["unknown_rows"] = (
        int((enriched["spread_bucket"] == EXEC_CTX_UNKNOWN).sum())
        if "spread_bucket" in enriched.columns
        else 0
    )

    if diag["matched_symbols"] == 0:
        diag["note"] = (
            "execution context loaded but no opportunity symbols overlapped "
            "(rows filled UNKNOWN honestly)"
        )

    return enriched, diag


def _enrich_opportunities_with_edge(
    out_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Add edge / ranking / sizing columns to the opportunities DataFrame.

    Read-only side data: signals_with_rationale.csv (optional). Existing schema
    is preserved; any failure degrades to confidence-only edge sizing.
    """
    diag: Dict[str, Any] = {
        "edge_components_loaded": False,
        "edge_components_columns": [],
        "edge_buckets": {},
        "edge_eligible_rows": 0,
    }
    if out_df is None or out_df.empty:
        return out_df, diag

    enriched = out_df.copy()
    components = _load_feature_components_optional()
    if components is not None and "ticker" in enriched.columns:
        # Avoid clobbering existing columns (e.g. confidence already present).
        merge_cols = [c for c in components.columns if c == "ticker" or c not in enriched.columns]
        if len(merge_cols) > 1:
            try:
                enriched = enriched.merge(components[merge_cols], on="ticker", how="left")
                diag["edge_components_loaded"] = True
                diag["edge_components_columns"] = [c for c in merge_cols if c != "ticker"]
            except Exception:
                # Defensive: never break opportunity build because of optional enrichment.
                enriched = out_df.copy()

    enriched = enrich_with_edge(
        enriched,
        EnrichmentSpec(opportunity_col="opportunity_type"),
    )

    if "sizing_bucket" in enriched.columns:
        diag["edge_buckets"] = (
            enriched["sizing_bucket"].fillna("").astype(str).value_counts().to_dict()
        )
    if "allocation_multiplier" in enriched.columns:
        diag["edge_eligible_rows"] = int((enriched["allocation_multiplier"] > 0.0).sum())

    return enriched, diag


def _exploration_entry_fallback(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """When strict opportunity_count == 0: filtered FLAT rows → top-N exploratory ENTRY (BUY)."""
    base_cols = [c for c in CONTEXT_COLS if c in df.columns] + [
        "opportunity_type",
        "exploration_flag",
    ]
    pos = df["effective_position_state"].fillna("").astype(str).str.strip().str.upper()
    flat = df.loc[pos == "FLAT"].copy()
    if flat.empty:
        print("[EXPLORATION] filtered_candidates=", 0)
        print("[EXPLORATION] selected=", 0)
        return pd.DataFrame(columns=base_cols)

    mask = pd.Series(True, index=flat.index)
    if "confidence" in flat.columns:
        cnum = pd.to_numeric(flat["confidence"], errors="coerce")
        mask = mask & (cnum >= 0.55)
    if "delta_pct" in flat.columns:
        dnum = pd.to_numeric(flat["delta_pct"], errors="coerce")
        mask = mask & (dnum.abs() >= 0.01)

    filtered = flat.loc[mask].copy()
    print("[EXPLORATION] filtered_candidates=", len(filtered))

    if filtered.empty:
        print("[EXPLORATION] selected=", 0)
        return pd.DataFrame(columns=base_cols)

    conf = (
        pd.to_numeric(filtered["confidence"], errors="coerce")
        if "confidence" in filtered.columns
        else None
    )
    if conf is not None and conf.notna().any():
        filtered["_sort"] = conf
    elif "delta_pct" in filtered.columns:
        filtered["_sort"] = pd.to_numeric(filtered["delta_pct"], errors="coerce")
    else:
        filtered["_sort"] = 0.0
    filtered["_sort"] = filtered["_sort"].fillna(float("-inf"))
    filtered = filtered.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    selected = filtered.head(n).copy()
    print("[EXPLORATION] selected=", len(selected))

    selected["effective_stance"] = "BUY"
    selected["opportunity_type"] = "ENTRY"
    selected["exploration_flag"] = True
    if "reason_code" not in selected.columns:
        selected["reason_code"] = "OK"
    else:
        selected["reason_code"] = "OK"

    cols = _output_columns(selected)
    return selected[cols].reset_index(drop=True)


def build_opportunities_clean(
    df: pd.DataFrame,
    *,
    allow_new_orders: bool,
    allow_new_trades: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Returns (opportunities_df, diagnostics_dict, drops_df).
    """
    out_cols = CONTEXT_COLS + ["opportunity_type", "exploration_flag"]
    empty_diag: Dict[str, Any] = {
        "timestamp": _utc_iso(),
        "input_rows": 0,
        "lifecycle_actionable_rows": 0,
        "opportunities_emitted": 0,
        "dropped_after_lifecycle": 0,
        "not_actionable_lifecycle": 0,
        "drop_reason_counts": {},
    }
    if df.empty:
        return pd.DataFrame(columns=out_cols), empty_diag, pd.DataFrame()

    input_rows_total = len(df)
    blocked_due_exec = 0
    if "execution_blocked" in df.columns:
        eb = df["execution_blocked"].fillna(False)
        try:
            eb = eb.astype(bool)
        except Exception:
            eb = eb.apply(lambda x: str(x).strip().lower() in ("1", "true", "yes"))
        blocked_due_exec = int(eb.sum())
        if blocked_due_exec:
            df = df.loc[~eb].copy()
    if df.empty:
        zdiag = {
            **empty_diag,
            "timestamp": _utc_iso(),
            "input_rows": input_rows_total,
            "blocked_execution_rows": blocked_due_exec,
            "note": (
                "no rows left after execution_blocked filter" if blocked_due_exec else "empty input"
            ),
        }
        return pd.DataFrame(columns=out_cols), zdiag, pd.DataFrame()

    pos = df["effective_position_state"].fillna("").astype(str).str.strip().str.upper()
    lc = _series_lifecycle_action(df)

    kept_idx: List[int] = []
    otypes: List[str] = []
    reason_codes: List[str] = []
    drops: List[Dict[str, Any]] = []

    actionable_mask = []
    for i in range(len(df)):
        row = df.iloc[i]
        ot = classify_opportunity_from_lifecycle(str(pos.iloc[i]), str(lc.iloc[i]))
        actionable_mask.append(ot is not None)
        if ot is None:
            continue

        reason = ""
        if _row_invalid_price(row):
            reason = "INVALID_PRICE"
        elif _row_invalid_qty_zero(row):
            reason = "INVALID_QTY"
        elif _risk_guard_drops_opportunity(ot, allow_new_orders, allow_new_trades):
            reason = "RISK_GUARD"
        else:
            kept_idx.append(i)
            otypes.append(ot)
            reason_codes.append("OK")
            continue

        drops.append(
            {
                "ticker": str(row.get("ticker", "")).strip().upper(),
                "effective_position_state": str(pos.iloc[i]),
                "lifecycle_action": str(lc.iloc[i]),
                "opportunity_type_would_be": ot,
                "reason_code": reason,
            }
        )

    lifecycle_actionable = int(sum(actionable_mask))
    dropped_after = len(drops)

    drop_counts: Dict[str, int] = {}
    for d in drops:
        rc = str(d.get("reason_code") or "UNKNOWN_FILTER")
        drop_counts[rc] = drop_counts.get(rc, 0) + 1

    not_actionable = len(df) - lifecycle_actionable

    if not kept_idx:
        empty_df = pd.DataFrame(columns=out_cols)
        diag = {
            "timestamp": _utc_iso(),
            "input_rows": len(df),
            "blocked_execution_rows": blocked_due_exec,
            "lifecycle_actionable_rows": lifecycle_actionable,
            "opportunities_emitted": 0,
            "dropped_after_lifecycle": dropped_after,
            "not_actionable_lifecycle": not_actionable,
            "drop_reason_counts": drop_counts,
            "hard_risk_flags": {
                "allow_new_orders": allow_new_orders,
                "allow_new_trades": allow_new_trades,
            },
        }
        return empty_df, diag, pd.DataFrame(drops)

    sub = df.iloc[kept_idx].copy()
    sub["opportunity_type"] = otypes
    sub["exploration_flag"] = False
    # Execution handoff uses effective_stance when non-empty; set intent so WAIT/HOLD does not override mapped stance.
    sub["effective_stance"] = [_intent_stance_for_opportunity(o) for o in otypes]
    sub["reason_code"] = reason_codes

    cols = _output_columns(sub)
    diag = {
        "timestamp": _utc_iso(),
        "input_rows": len(df),
        "blocked_execution_rows": blocked_due_exec,
        "lifecycle_actionable_rows": lifecycle_actionable,
        "opportunities_emitted": len(sub),
        "dropped_after_lifecycle": dropped_after,
        "not_actionable_lifecycle": not_actionable,
        "drop_reason_counts": drop_counts,
        "hard_risk_flags": {
            "allow_new_orders": allow_new_orders,
            "allow_new_trades": allow_new_trades,
        },
    }
    return sub[cols].reset_index(drop=True), diag, pd.DataFrame(drops)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build trade_opportunities.csv from signal_lifecycle_effective.csv"
    )
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    gate = evaluate_lifecycle_gate(path=args.in_path)
    print(gate.format_block())
    if gate.status == "BLOCKED":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(args.out, index=False)
        diag: Dict[str, Any] = {
            "timestamp": _utc_iso(),
            "lifecycle_gate": gate.to_dict(),
            "opportunities_emitted": 0,
            "blocked_by_lifecycle_gate": True,
        }
        try:
            DIAG_JSON.parent.mkdir(parents=True, exist_ok=True)
            DIAG_JSON.write_text(json.dumps(diag, indent=2), encoding="utf-8")
        except Exception:
            pass
        summarize_opportunity_build(
            entry=0,
            add=0,
            exit_n=0,
            trim=0,
            suppressed=0,
            blocked_due_to_lifecycle=1,
        )
        print("[DECISION_DISTRIBUTION] entry=0 add=0 hold=0 trim=0 exit=0")
        print(
            "[build_trade_opportunities] blocked: lifecycle gate; wrote empty trade_opportunities.csv"
        )
        return 0

    try:
        df = load_effective(args.in_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    allow_new_orders, allow_new_trades = _parse_hard_risk_flags()
    out_df, diag, drops_df = build_opportunities_clean(
        df, allow_new_orders=allow_new_orders, allow_new_trades=allow_new_trades
    )

    if out_df.empty:
        out_df = _exploration_entry_fallback(df, EXPLORATION_TOP_N)
        diag["exploration_fallback_used"] = True
        diag["opportunities_emitted_after_exploration"] = len(out_df)
    else:
        diag["exploration_fallback_used"] = False

    if not out_df.empty and "ticker" in out_df.columns:
        out_df = out_df[~out_df["ticker"].isin(INVALID_SYMBOLS)].copy()
        diag["opportunities_emitted"] = len(out_df)
        if diag.get("exploration_fallback_used"):
            diag["opportunities_emitted_after_exploration"] = len(out_df)

    # Edge-based ranking and sizing enrichment (additive, read-only sources).
    out_df, edge_diag = _enrich_opportunities_with_edge(out_df)
    diag["edge_ranking"] = edge_diag

    # Execution-context enrichment — left-joins spread_bucket /
    # execution_risk_flag / quote_staleness_flag / liquidity_bucket onto each
    # opportunity row. Purely additive and read-only (no trading behavior
    # change); lets adaptation_simulation.py and dashboards see the real
    # execution conditions per symbol.
    out_df, exec_ctx_diag = _merge_execution_context(out_df)
    diag["execution_context"] = exec_ctx_diag

    # Reorder so edge and exec-context columns sit at the tail of the schema.
    if not out_df.empty:
        out_df = out_df[_output_columns(out_df)].reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    try:
        diag["reason_code_semantics"] = {
            "OK": "Included in trade_opportunities.csv (strict path).",
            "INVALID_PRICE": "close missing, NaN, or <= 0 (hard drop).",
            "INVALID_QTY": "explicit qty/target_qty/planned_qty/order_qty column equals 0.",
            "RISK_GUARD": "Hard block from adaptive_risk_state / capital_preservation (no sizing changes).",
            "LOW_CONFIDENCE": "Not used to drop strict lifecycle rows (reserved / exploration diagnostics only).",
            "LOW_DELTA": "Not used to drop strict lifecycle rows (reserved / exploration diagnostics only).",
            "UNKNOWN_FILTER": "Unexpected drop reason (should not occur in strict path).",
        }
        DIAG_JSON.parent.mkdir(parents=True, exist_ok=True)
        DIAG_JSON.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    except Exception:
        pass

    try:
        if not drops_df.empty:
            drops_df.to_csv(DROPS_CSV, index=False)
        else:
            pd.DataFrame(
                columns=[
                    "ticker",
                    "effective_position_state",
                    "lifecycle_action",
                    "opportunity_type_would_be",
                    "reason_code",
                ]
            ).to_csv(DROPS_CSV, index=False)
    except Exception:
        pass

    try:
        from services.signal_pressure_diagnostics import refresh_signal_pressure_diagnostics

        refresh_signal_pressure_diagnostics()
    except Exception:
        pass

    n = len(out_df)
    ot = (
        out_df["opportunity_type"].fillna("").astype(str).str.upper()
        if not out_df.empty
        else pd.Series([], dtype=object)
    )
    summarize_opportunity_build(
        entry=int((ot == "ENTRY").sum()),
        add=int((ot == "ADD").sum()),
        exit_n=int((ot == "EXIT").sum()),
        trim=int((ot == "TRIM").sum()),
        suppressed=int(diag.get("not_actionable_lifecycle", 0) or 0),
        blocked_due_to_lifecycle=int(diag.get("blocked_execution_rows", 0) or 0),
    )
    dmix = _input_decision_distribution(df)
    print(
        f"[DECISION_DISTRIBUTION] entry={dmix['entry']} add={dmix['add']} hold={dmix['hold']} "
        f"trim={dmix['trim']} exit={dmix['exit']}"
    )
    print(f"[build_trade_opportunities] wrote {args.out}")
    print(f"[build_trade_opportunities] total_rows={n} opportunity_count={n}")
    print(
        f"[build_trade_opportunities] dropped_after_lifecycle={diag.get('dropped_after_lifecycle', 0)} "
        f"lifecycle_actionable={diag.get('lifecycle_actionable_rows', 0)} input_rows={diag.get('input_rows', 0)}"
    )
    ctx_diag = diag.get("execution_context", {}) or {}
    print(
        "[build_trade_opportunities] execution context coverage: "
        f"matched_symbols={ctx_diag.get('matched_symbols', 0)} / "
        f"total={ctx_diag.get('total_symbols', 0)} "
        f"(ratio={ctx_diag.get('coverage_ratio', 0.0)}, "
        f"source={'|'.join(ctx_diag.get('source_files', [])) or 'missing'})"
    )
    if n == 0:
        print(
            "[build_trade_opportunities] idle: no strict opportunities; exploration pool empty or failed filters"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
