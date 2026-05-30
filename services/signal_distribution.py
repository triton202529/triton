"""services/signal_distribution.py

Diagnostic (read-only) measurement of the dispersion of `confidence` and
`edge_score` across the signal pipeline.

Why this exists
---------------
Historically Triton's downstream artifacts (signal_lifecycle*.csv,
trade_opportunities.csv) shipped a near-constant ``confidence`` column
(~0.65) because ``services/trade_rationale.py::confidence_score`` only
considered fields that don't exist in the prediction parquets. The fix in
``trade_rationale.py`` replaces that flattening with a deterministic,
feature-based composite. This module measures whether the fix is working
by comparing the observed dispersion against simple quality targets.

This module **never** modifies live trading logic, broker behavior, or the
lifecycle state — it only reads existing CSV outputs and writes two
artifacts:

    - data/results/signal_distribution_summary.json
    - data/results/signal_distribution_sample.csv

Both are safe to ship: they are purely observational and are consumed by
diagnostic tooling / dashboards.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

SUMMARY_JSON = RESULTS_DIR / "signal_distribution_summary.json"
SAMPLE_CSV = RESULTS_DIR / "signal_distribution_sample.csv"

SIGNALS_WITH_RATIONALE_CSV = RESULTS_DIR / "signals_with_rationale.csv"
SIGNALS_CSV = RESULTS_DIR / "signals.csv"
TRADE_OPPORTUNITIES_CSV = RESULTS_DIR / "trade_opportunities.csv"
SIGNAL_LIFECYCLE_EFFECTIVE_CSV = RESULTS_DIR / "signal_lifecycle_effective.csv"


# ─────────────────────────────────────────────────────────────
# Buckets used for grouping confidence / edge_score rows.
# ─────────────────────────────────────────────────────────────
DEFAULT_BUCKETS: List[Dict[str, Any]] = [
    {"name": "very_weak", "lo": 0.00, "hi": 0.35},
    {"name": "weak", "lo": 0.35, "hi": 0.55},
    {"name": "borderline", "lo": 0.55, "hi": 0.65},
    {"name": "strong", "lo": 0.65, "hi": 0.80},
    {"name": "very_strong", "lo": 0.80, "hi": 1.01},  # upper open-ish to include 1.0
]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def _bucket_counts(values: pd.Series, buckets: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if values is None or values.empty:
        for b in buckets:
            out[b["name"]] = 0
        return out
    for b in buckets:
        lo, hi, name = float(b["lo"]), float(b["hi"]), str(b["name"])
        mask = (values >= lo) & (values < hi)
        out[name] = int(mask.sum())
    return out


def _summarize_series(values: pd.Series) -> Dict[str, Any]:
    if values is None or values.empty:
        return {
            "count": 0,
            "unique_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "mode_value": None,
            "pct_rows_same_as_mode": 0.0,
        }
    mode_series = values.mode()
    mode_value = float(mode_series.iloc[0]) if not mode_series.empty else float("nan")
    pct_mode = 0.0
    if math.isfinite(mode_value):
        pct_mode = float((values == mode_value).sum()) / float(len(values))
    return {
        "count": int(len(values)),
        "unique_count": int(values.nunique()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "mode_value": mode_value if math.isfinite(mode_value) else None,
        "pct_rows_same_as_mode": round(pct_mode, 6),
    }


# ─────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────
@dataclass
class SourceFrame:
    name: str
    df: Optional[pd.DataFrame]
    confidence_col: Optional[str]
    edge_col: Optional[str]


def _pick_conf_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("confidence", "confidence_score"):
        if c in df.columns:
            return c
    return None


def _pick_edge_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("edge_score", "score"):
        if c in df.columns:
            return c
    return None


def _collect_sources() -> List[SourceFrame]:
    sources: List[SourceFrame] = []
    for name, path in (
        ("signals_with_rationale", SIGNALS_WITH_RATIONALE_CSV),
        ("signals", SIGNALS_CSV),
        ("signal_lifecycle_effective", SIGNAL_LIFECYCLE_EFFECTIVE_CSV),
        ("trade_opportunities", TRADE_OPPORTUNITIES_CSV),
    ):
        df = _safe_read_csv(path)
        if df is None:
            sources.append(SourceFrame(name=name, df=None, confidence_col=None, edge_col=None))
            continue
        sources.append(
            SourceFrame(
                name=name,
                df=df,
                confidence_col=_pick_conf_col(df),
                edge_col=_pick_edge_col(df),
            )
        )
    return sources


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def build_summary(buckets: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Build the full diagnostic summary. Never raises — returns a structured
    dict even when inputs are missing or empty.
    """
    buckets = buckets or DEFAULT_BUCKETS
    notes: List[str] = []

    sources = _collect_sources()
    per_source: Dict[str, Any] = {}
    for s in sources:
        if s.df is None:
            per_source[s.name] = {"available": False, "reason": "missing_or_empty"}
            continue
        conf_vals = _numeric(s.df[s.confidence_col]) if s.confidence_col else pd.Series(dtype=float)
        edge_vals = _numeric(s.df[s.edge_col]) if s.edge_col else pd.Series(dtype=float)
        per_source[s.name] = {
            "available": True,
            "rows": int(len(s.df)),
            "confidence_column": s.confidence_col,
            "edge_column": s.edge_col,
            "confidence": _summarize_series(conf_vals),
            "edge_score": _summarize_series(edge_vals),
            "confidence_bucket_counts": _bucket_counts(conf_vals, buckets),
            "edge_score_bucket_counts": _bucket_counts(edge_vals, buckets),
        }

    # Primary source for top-level numbers: trade_opportunities if present,
    # else signal_lifecycle_effective, else signals_with_rationale.
    primary_name = None
    for candidate in (
        "trade_opportunities",
        "signal_lifecycle_effective",
        "signals_with_rationale",
    ):
        info = per_source.get(candidate)
        if info and info.get("available") and info.get("rows", 0) > 0:
            primary_name = candidate
            break

    if primary_name is None:
        notes.append("no populated source found for top-level metrics")
        rows_seen = 0
        unique_conf = 0
        unique_edge = 0
        conf_summary: Dict[str, Any] = _summarize_series(pd.Series(dtype=float))
        edge_summary: Dict[str, Any] = _summarize_series(pd.Series(dtype=float))
        conf_buckets: Dict[str, int] = {b["name"]: 0 for b in buckets}
        edge_buckets: Dict[str, int] = {b["name"]: 0 for b in buckets}
    else:
        primary = per_source[primary_name]
        rows_seen = int(primary.get("rows", 0))
        conf_summary = primary.get("confidence", {})
        edge_summary = primary.get("edge_score", {})
        unique_conf = int(conf_summary.get("unique_count", 0) or 0)
        unique_edge = int(edge_summary.get("unique_count", 0) or 0)
        conf_buckets = primary.get("confidence_bucket_counts", {})
        edge_buckets = primary.get("edge_score_bucket_counts", {})

        if rows_seen > 0:
            if unique_conf <= 2:
                notes.append(
                    f"confidence collapsed to {unique_conf} unique values in {primary_name} — "
                    "feature-based composite likely not reaching downstream"
                )
            if unique_edge <= 2 and edge_summary.get("count", 0) > 0:
                notes.append(
                    f"edge_score collapsed to {unique_edge} unique values in {primary_name}"
                )

    summary = {
        "schema_version": 1,
        "generated_at_utc": _utc_now_z(),
        "module": "services.signal_distribution",
        "primary_source": primary_name,
        "rows_seen": rows_seen,
        "unique_confidence_count": unique_conf,
        "unique_edge_score_count": unique_edge,
        "min_confidence": conf_summary.get("min"),
        "max_confidence": conf_summary.get("max"),
        "mean_confidence": conf_summary.get("mean"),
        "median_confidence": conf_summary.get("median"),
        "pct_rows_same_confidence_as_mode": conf_summary.get("pct_rows_same_as_mode", 0.0),
        "min_edge_score": edge_summary.get("min"),
        "max_edge_score": edge_summary.get("max"),
        "mean_edge_score": edge_summary.get("mean"),
        "median_edge_score": edge_summary.get("median"),
        "pct_rows_same_edge_score_as_mode": edge_summary.get("pct_rows_same_as_mode", 0.0),
        "confidence_bucket_counts": conf_buckets,
        "edge_score_bucket_counts": edge_buckets,
        "buckets": buckets,
        "per_source": per_source,
        "quality_targets": {
            "unique_confidence_count_gt_5": unique_conf > 5,
            "unique_edge_score_count_gt_5": unique_edge > 5,
            "pct_rows_same_confidence_lt_0_5": (
                float(conf_summary.get("pct_rows_same_as_mode", 1.0) or 1.0) < 0.5
            ),
            "pct_rows_same_edge_score_lt_0_5": (
                float(edge_summary.get("pct_rows_same_as_mode", 1.0) or 1.0) < 0.5
            ),
            "three_practical_groups": (
                sum(1 for k in ("weak", "borderline", "strong") if int(conf_buckets.get(k, 0)) > 0)
                >= 3
            ),
        },
        "notes": notes,
    }
    return summary


def build_sample(max_rows: int = 200) -> pd.DataFrame:
    """
    Produce a tidy sample CSV for quick eyeballing / dashboard display.
    Prefers `trade_opportunities.csv`, falls back to
    `signal_lifecycle_effective.csv`, then `signals_with_rationale.csv`.
    """
    candidates = [
        TRADE_OPPORTUNITIES_CSV,
        SIGNAL_LIFECYCLE_EFFECTIVE_CSV,
        SIGNALS_WITH_RATIONALE_CSV,
    ]
    for path in candidates:
        df = _safe_read_csv(path)
        if df is None or df.empty:
            continue
        keep_candidates = [
            "ticker",
            "signal",
            "effective_stance",
            "opportunity_type",
            "confidence",
            "confidence_score",
            "edge_score",
            "score",
            "delta_pct",
            "delta_conviction",
            "momentum_score",
            "trend_score",
            "breakout_score",
            "volatility_score",
            "score_model_component",
            "score_rank_component",
            "score_quality_component",
            "score_penalty_component",
            "score_final_preclip",
            "components_present",
            "allocation_multiplier",
            "sizing_bucket",
        ]
        keep = [c for c in keep_candidates if c in df.columns]
        if not keep:
            continue
        sample = df[keep].head(max_rows).copy()
        sample.insert(0, "source_file", path.name)
        return sample.reset_index(drop=True)
    return pd.DataFrame(columns=["source_file", "ticker", "signal", "confidence", "edge_score"])


def write_outputs(summary: Dict[str, Any], sample: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[signal_distribution] failed to write summary JSON: {e}")
    try:
        sample.to_csv(SAMPLE_CSV, index=False)
    except Exception as e:
        print(f"[signal_distribution] failed to write sample CSV: {e}")


def run() -> int:
    summary = build_summary()
    sample = build_sample()
    write_outputs(summary, sample)

    rs = int(summary.get("rows_seen") or 0)
    uc = int(summary.get("unique_confidence_count") or 0)
    ue = int(summary.get("unique_edge_score_count") or 0)
    mn = summary.get("min_confidence")
    mx = summary.get("max_confidence")
    mn_str = f"{float(mn):.3f}" if isinstance(mn, (int, float)) else "n/a"
    mx_str = f"{float(mx):.3f}" if isinstance(mx, (int, float)) else "n/a"
    print(
        "[signal_distribution] "
        f"primary={summary.get('primary_source')} rows={rs} "
        f"unique_confidence={uc} unique_edge_score={ue} "
        f"confidence_range=[{mn_str}, {mx_str}]"
    )
    for n in summary.get("notes", []) or []:
        print(f"[signal_distribution] note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
