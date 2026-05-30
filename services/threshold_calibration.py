"""
Threshold Calibration Analysis Layer — READ-ONLY.

Answers: "Is Triton under-deployed because the thresholds are too strict,
or because the opportunities are genuinely not good enough?"

Inputs (all optional — graceful fallbacks):
    data/results/trade_opportunities.csv    (required for meaningful output)
    data/results/applied_adjustments.csv    (optional, informational only)

Outputs (additive, read-only to the rest of the system):
    data/results/threshold_calibration.csv
    data/results/threshold_calibration_summary.json

HARD SAFETY:
    * Does NOT modify execute_trades, broker, lifecycle, signal generation,
      or any live threshold or config.
    * Writes only to its own output files.
    * No network I/O.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"

TRADE_OPPORTUNITIES_CSV = RESULTS / "trade_opportunities.csv"
APPLIED_ADJUSTMENTS_CSV = RESULTS / "applied_adjustments.csv"

OUT_CSV = RESULTS / "threshold_calibration.csv"
OUT_SUMMARY_JSON = RESULTS / "threshold_calibration_summary.json"

SCHEMA_VERSION = "1.0.0"
MODULE_NAME = "threshold_calibration"


# Decision / zone labels
ZONE_CLEAR_ACCEPT = "CLEAR_ACCEPT"
ZONE_BORDERLINE_ACCEPT = "BORDERLINE_ACCEPT"
ZONE_BORDERLINE_REJECT = "BORDERLINE_REJECT"
ZONE_CLEAR_REJECT = "CLEAR_REJECT"
ZONE_BLOCKED = "EXECUTION_BLOCKED"

DECISION_ACCEPT = "ACCEPT"
DECISION_REJECT = "REJECT"

CONF_ZONE_ABOVE_CLEAR = "ABOVE_PENALTY_CLEAR"
CONF_ZONE_ABOVE_BORDER = "ABOVE_PENALTY_BORDERLINE"
CONF_ZONE_BELOW_BORDER = "BELOW_PENALTY_BORDERLINE"
CONF_ZONE_BELOW_CLEAR = "BELOW_PENALTY_CLEAR"


# Stable output column order for threshold_calibration.csv
OUT_COLUMNS: List[str] = [
    "ticker",
    "opportunity_type",
    "effective_stance",
    "sizing_bucket",
    "edge_score",
    "confidence",
    "execution_blocked",
    "execution_block_reason",
    "base_floor",
    "base_floor_applied",
    "distance_to_base_floor",
    "baseline_decision",
    "threshold_zone",
    "near_threshold_flag",
    "add_floor_offset",
    "is_add_like",
    "distance_to_add_floor",
    "near_add_threshold_flag",
    "confidence_penalty_threshold",
    "distance_to_confidence_penalty",
    "near_confidence_penalty_flag",
    "confidence_zone",
    "edge_rank",
    "edge_percentile",
    "exploration_flag",
    "reason_code",
]


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────


@dataclass
class ThresholdCalibrationConfig:
    """All tunables for the analysis layer. None of these values ever
    mutate live thresholds — they only drive the analysis output."""

    base_floor: float = 0.50
    add_floor_offset: float = 0.15
    confidence_penalty_threshold: float = 0.55
    borderline_band: float = 0.05
    top_n: int = 10

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


# ──────────────────────────────────────────────────────────────
# Safe loaders / helpers
# ──────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_csv_safe(path: Path) -> Tuple[pd.DataFrame, str]:
    """Return (df, status). Status ∈ {ok, missing, empty, error:<msg>}."""
    try:
        if not path.exists():
            return pd.DataFrame(), "missing"
        df = pd.read_csv(path)
        if df.shape[0] == 0:
            return df, "empty"
        return df, "ok"
    except Exception as exc:  # pragma: no cover — defensive
        return pd.DataFrame(), f"error:{type(exc).__name__}:{exc}"


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if val is None:
            return default
        if isinstance(val, float) and pd.isna(val):
            return default
        f = float(val)
        if f != f:  # NaN
            return default
        return f
    except (TypeError, ValueError):
        return default


def _safe_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    try:
        if isinstance(val, float) and pd.isna(val):
            return default
    except Exception:
        pass
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f", "", "nan", "none"):
        return False
    return default


def _safe_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    try:
        if isinstance(val, float) and pd.isna(val):
            return default
    except Exception:
        pass
    s = str(val).strip()
    return s if s.lower() != "nan" else default


def _is_add_like(row: Dict[str, Any]) -> bool:
    for k in ("opportunity_type", "action", "effective_stance"):
        v = _safe_str(row.get(k)).upper()
        if v:
            return v == "ADD"
    return False


def _symbol_of(row: Dict[str, Any]) -> str:
    for k in ("ticker", "symbol"):
        v = _safe_str(row.get(k))
        if v:
            return v
    return ""


# ──────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────


def _classify_edge_zone(
    edge: Optional[float],
    base_floor_applied: float,
    band: float,
    execution_blocked: bool,
) -> Tuple[str, str, Optional[float], bool]:
    """Return (decision, zone, distance, near_flag)."""
    if execution_blocked:
        return DECISION_REJECT, ZONE_BLOCKED, None, False
    if edge is None:
        return DECISION_REJECT, ZONE_CLEAR_REJECT, None, False
    dist = float(edge) - float(base_floor_applied)
    near = abs(dist) <= band + 1e-9
    if dist >= 0:
        zone = ZONE_BORDERLINE_ACCEPT if near else ZONE_CLEAR_ACCEPT
        return DECISION_ACCEPT, zone, dist, near
    zone = ZONE_BORDERLINE_REJECT if near else ZONE_CLEAR_REJECT
    return DECISION_REJECT, zone, dist, near


def _classify_confidence_zone(
    conf: Optional[float], penalty_threshold: float, band: float
) -> Tuple[str, Optional[float], bool]:
    """Return (zone, distance_to_penalty, near_flag). Distance is positive
    when confidence is above the penalty threshold."""
    if conf is None:
        return "", None, False
    dist = float(conf) - float(penalty_threshold)
    near = abs(dist) <= band + 1e-9
    if dist >= 0:
        zone = CONF_ZONE_ABOVE_BORDER if near else CONF_ZONE_ABOVE_CLEAR
    else:
        zone = CONF_ZONE_BELOW_BORDER if near else CONF_ZONE_BELOW_CLEAR
    return zone, dist, near


# ──────────────────────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────────────────────


def analyze(opps: pd.DataFrame, cfg: ThresholdCalibrationConfig) -> pd.DataFrame:
    """Produce a per-opportunity calibration DataFrame.

    Never mutates the input; never raises on malformed rows.
    """
    if opps is None or opps.empty:
        return _empty_out_df()

    out_rows: List[Dict[str, Any]] = []
    for _, r in opps.iterrows():
        row = r.to_dict() if hasattr(r, "to_dict") else dict(r)
        edge = _safe_float(row.get("edge_score"))
        conf = _safe_float(row.get("confidence"))
        exec_blocked = _safe_bool(row.get("execution_blocked"), default=False)
        is_add = _is_add_like(row)

        add_offset = float(cfg.add_floor_offset) if is_add else 0.0
        base_floor_applied = float(cfg.base_floor) + add_offset

        decision, zone, dist, near = _classify_edge_zone(
            edge, base_floor_applied, cfg.borderline_band, exec_blocked
        )
        # Distance to ADD floor relative to the row's edge (always defined,
        # regardless of whether the row is ADD — useful for "how close would
        # this become" questions in the dashboard).
        dist_add = None
        if edge is not None:
            dist_add = float(edge) - (float(cfg.base_floor) + float(cfg.add_floor_offset))
        near_add = (
            is_add
            and not exec_blocked
            and dist_add is not None
            and abs(dist_add) <= cfg.borderline_band + 1e-9
        )

        conf_zone, conf_dist, conf_near = _classify_confidence_zone(
            conf, cfg.confidence_penalty_threshold, cfg.borderline_band
        )

        out_rows.append(
            {
                "ticker": _symbol_of(row),
                "opportunity_type": _safe_str(row.get("opportunity_type")),
                "effective_stance": _safe_str(row.get("effective_stance")),
                "sizing_bucket": _safe_str(row.get("sizing_bucket")),
                "edge_score": edge if edge is not None else "",
                "confidence": conf if conf is not None else "",
                "execution_blocked": bool(exec_blocked),
                "execution_block_reason": _safe_str(row.get("execution_block_reason")),
                "base_floor": float(cfg.base_floor),
                "base_floor_applied": float(base_floor_applied),
                "distance_to_base_floor": (round(dist, 6) if dist is not None else ""),
                "baseline_decision": decision,
                "threshold_zone": zone,
                "near_threshold_flag": bool(near),
                "add_floor_offset": float(cfg.add_floor_offset),
                "is_add_like": bool(is_add),
                "distance_to_add_floor": (round(dist_add, 6) if dist_add is not None else ""),
                "near_add_threshold_flag": bool(near_add),
                "confidence_penalty_threshold": float(cfg.confidence_penalty_threshold),
                "distance_to_confidence_penalty": (
                    round(conf_dist, 6) if conf_dist is not None else ""
                ),
                "near_confidence_penalty_flag": bool(conf_near),
                "confidence_zone": conf_zone,
                "edge_rank": _safe_str(row.get("edge_rank")),
                "edge_percentile": _safe_str(row.get("edge_percentile")),
                "exploration_flag": bool(_safe_bool(row.get("exploration_flag"), default=False)),
                "reason_code": _safe_str(row.get("reason_code")),
            }
        )

    df = pd.DataFrame(out_rows)
    # Enforce stable column order; tolerate any extras by dropping them.
    df = df.reindex(columns=OUT_COLUMNS)
    return df


def _empty_out_df() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in OUT_COLUMNS})


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────


def _value_counts(df: pd.DataFrame, col: str) -> Dict[str, int]:
    if df.empty or col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[col].fillna("").value_counts().items()}


def _top_borderline(
    df: pd.DataFrame, zone: str, top_n: int, *, ascending_dist: bool
) -> List[Dict[str, Any]]:
    if df.empty or "threshold_zone" not in df.columns:
        return []
    sub = df[df["threshold_zone"] == zone].copy()
    if sub.empty:
        return []
    # Coerce distance to numeric for sorting
    sub["_dist_num"] = pd.to_numeric(sub["distance_to_base_floor"], errors="coerce")
    if ascending_dist:
        # BORDERLINE_REJECT → least-negative (closest to floor) first
        sub = sub.sort_values("_dist_num", ascending=False)
    else:
        # BORDERLINE_ACCEPT → smallest positive distance (closest to floor) first
        sub = sub.sort_values("_dist_num", ascending=True)
    sub = sub.head(int(top_n)).drop(columns=["_dist_num"], errors="ignore")
    keep = [
        "ticker",
        "opportunity_type",
        "effective_stance",
        "sizing_bucket",
        "edge_score",
        "confidence",
        "base_floor_applied",
        "distance_to_base_floor",
        "threshold_zone",
        "confidence_zone",
    ]
    keep = [c for c in keep if c in sub.columns]
    # JSON-friendly records
    out: List[Dict[str, Any]] = []
    for rec in sub[keep].to_dict(orient="records"):
        clean: Dict[str, Any] = {}
        for k, v in rec.items():
            if isinstance(v, float) and pd.isna(v):
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
    return out


def _numeric_stats(df: pd.DataFrame, col: str) -> Dict[str, Optional[float]]:
    if df.empty or col not in df.columns:
        return {"avg": None, "min": None, "max": None, "count": 0}
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return {"avg": None, "min": None, "max": None, "count": 0}
    return {
        "avg": round(float(s.mean()), 6),
        "min": round(float(s.min()), 6),
        "max": round(float(s.max()), 6),
        "count": int(s.shape[0]),
    }


def _under_deployment_assessment(
    zone_counts: Dict[str, int],
    edge_stats: Dict[str, Optional[float]],
    cfg: ThresholdCalibrationConfig,
) -> Tuple[str, List[str]]:
    """Heuristic, transparent, and labelled as advisory-only.

    Returns (verdict, explanation_lines).
    """
    total = sum(zone_counts.values()) if zone_counts else 0
    if total == 0:
        return "NO_DATA", ["No opportunities available to analyse."]

    accept_clear = int(zone_counts.get(ZONE_CLEAR_ACCEPT, 0))
    accept_border = int(zone_counts.get(ZONE_BORDERLINE_ACCEPT, 0))
    reject_clear = int(zone_counts.get(ZONE_CLEAR_REJECT, 0))
    reject_border = int(zone_counts.get(ZONE_BORDERLINE_REJECT, 0))
    blocked = int(zone_counts.get(ZONE_BLOCKED, 0))
    analysable = total - blocked
    if analysable <= 0:
        return "ALL_BLOCKED", [
            f"All {total} opportunities are execution_blocked; threshold "
            "calibration cannot be assessed."
        ]

    border_total = accept_border + reject_border
    border_pct = border_total / analysable if analysable else 0.0
    reject_border_pct = reject_border / analysable if analysable else 0.0
    accept_total = accept_clear + accept_border
    accept_pct = accept_total / analysable if analysable else 0.0
    avg_edge = edge_stats.get("avg")
    max_edge = edge_stats.get("max")

    notes: List[str] = []
    notes.append(
        f"{border_total}/{analysable} ({border_pct:.0%}) rows sit within "
        f"±{cfg.borderline_band:g} of the applied floor."
    )
    notes.append(
        f"{reject_border} rows are BORDERLINE_REJECT and would pass with a "
        f"small floor relaxation."
    )
    notes.append(
        f"{accept_total}/{analysable} ({accept_pct:.0%}) rows currently accept "
        f"at base_floor={cfg.base_floor:g}."
    )

    if reject_border >= 3 and reject_border >= accept_clear:
        verdict = "LIKELY_UNDER_DEPLOYED_DUE_TO_THRESHOLDS"
    elif reject_border_pct >= 0.25:
        verdict = "POSSIBLY_UNDER_DEPLOYED_DUE_TO_THRESHOLDS"
    elif accept_pct >= 0.50 and reject_clear + reject_border > 0:
        verdict = "THRESHOLDS_APPEAR_REASONABLE"
    elif max_edge is not None and float(max_edge) < cfg.base_floor - cfg.borderline_band:
        verdict = "OPPORTUNITIES_GENUINELY_WEAK"
    else:
        verdict = "NO_CLEAR_SIGNAL"

    if avg_edge is not None:
        notes.append(f"Average edge_score={avg_edge:g} vs base_floor={cfg.base_floor:g}.")
    return verdict, notes


def build_summary(
    out_df: pd.DataFrame,
    cfg: ThresholdCalibrationConfig,
    sources: Dict[str, Any],
    extra_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    total = int(out_df.shape[0])
    zone_counts_raw = _value_counts(out_df, "threshold_zone")
    decision_counts = _value_counts(out_df, "baseline_decision")
    conf_zone_counts = _value_counts(out_df, "confidence_zone")

    # Fill in zeros for all expected keys so downstream consumers can
    # trust the shape even when the dataset is thin.
    zone_counts: Dict[str, int] = {
        ZONE_CLEAR_ACCEPT: int(zone_counts_raw.get(ZONE_CLEAR_ACCEPT, 0)),
        ZONE_BORDERLINE_ACCEPT: int(zone_counts_raw.get(ZONE_BORDERLINE_ACCEPT, 0)),
        ZONE_BORDERLINE_REJECT: int(zone_counts_raw.get(ZONE_BORDERLINE_REJECT, 0)),
        ZONE_CLEAR_REJECT: int(zone_counts_raw.get(ZONE_CLEAR_REJECT, 0)),
        ZONE_BLOCKED: int(zone_counts_raw.get(ZONE_BLOCKED, 0)),
    }

    borderline_count = zone_counts[ZONE_BORDERLINE_ACCEPT] + zone_counts[ZONE_BORDERLINE_REJECT]
    near_add_count = (
        int(out_df.get("near_add_threshold_flag", pd.Series(dtype=bool)).fillna(False).sum())
        if "near_add_threshold_flag" in out_df.columns
        else 0
    )
    near_conf_count = (
        int(out_df.get("near_confidence_penalty_flag", pd.Series(dtype=bool)).fillna(False).sum())
        if "near_confidence_penalty_flag" in out_df.columns
        else 0
    )

    edge_stats = _numeric_stats(out_df, "edge_score")
    conf_stats = _numeric_stats(out_df, "confidence")

    verdict, verdict_notes = _under_deployment_assessment(zone_counts, edge_stats, cfg)

    notes: List[str] = [
        "Analysis only — no execution, broker, lifecycle, or config changes.",
        f"base_floor={cfg.base_floor:g}, add_floor_offset={cfg.add_floor_offset:g}, "
        f"confidence_penalty_threshold={cfg.confidence_penalty_threshold:g}, "
        f"borderline_band=±{cfg.borderline_band:g}.",
    ]
    notes.extend(verdict_notes)
    if extra_notes:
        notes.extend(extra_notes)

    summary: Dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "module": MODULE_NAME,
        "schema_version": SCHEMA_VERSION,
        "analysis_only": True,
        "advisory_only": True,
        "config": cfg.as_dict(),
        "source_availability": sources,
        "total_opportunities": total,
        "borderline_count": int(borderline_count),
        "borderline_accept_count": int(zone_counts[ZONE_BORDERLINE_ACCEPT]),
        "borderline_reject_count": int(zone_counts[ZONE_BORDERLINE_REJECT]),
        "clear_accept_count": int(zone_counts[ZONE_CLEAR_ACCEPT]),
        "clear_reject_count": int(zone_counts[ZONE_CLEAR_REJECT]),
        "execution_blocked_count": int(zone_counts[ZONE_BLOCKED]),
        "near_add_threshold_count": int(near_add_count),
        "near_confidence_penalty_count": int(near_conf_count),
        "average_edge_score": edge_stats["avg"],
        "min_edge_score": edge_stats["min"],
        "max_edge_score": edge_stats["max"],
        "confidence_stats": conf_stats,
        "decision_counts": decision_counts,
        "threshold_zone_counts": zone_counts,
        "confidence_zone_counts": conf_zone_counts,
        "top_borderline_accepts": _top_borderline(
            out_df, ZONE_BORDERLINE_ACCEPT, cfg.top_n, ascending_dist=False
        ),
        "top_borderline_rejects": _top_borderline(
            out_df, ZONE_BORDERLINE_REJECT, cfg.top_n, ascending_dist=True
        ),
        "under_deployment_assessment": verdict,
        "notes": notes,
    }
    return summary


# ──────────────────────────────────────────────────────────────
# Writers
# ──────────────────────────────────────────────────────────────


def _df_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_out_df()
    out = df.copy()
    # Coerce bool columns to stable int-strings for CSV
    bool_cols = (
        "execution_blocked",
        "near_threshold_flag",
        "is_add_like",
        "near_add_threshold_flag",
        "near_confidence_penalty_flag",
        "exploration_flag",
    )
    for c in bool_cols:
        if c in out.columns:
            s = out[c]
            try:
                out[c] = (
                    s.astype("boolean")
                    .astype("Int64")
                    .astype(str)
                    .replace({"1": "True", "0": "False", "<NA>": ""})
                )
            except Exception:
                out[c] = s.astype(str)
    return out.reindex(columns=OUT_COLUMNS)


def write_outputs(out_df: pd.DataFrame, summary: Dict[str, Any]) -> Dict[str, str]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_df = _df_for_csv(out_df)
    csv_df.to_csv(OUT_CSV, index=False)
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {
        "threshold_calibration_csv": str(OUT_CSV),
        "threshold_calibration_summary_json": str(OUT_SUMMARY_JSON),
    }


# ──────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────


def run_calibration(
    cfg: Optional[ThresholdCalibrationConfig] = None,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load inputs, analyse, and write outputs. Never raises on missing
    or malformed files — always emits stable-schema outputs."""
    cfg = cfg or ThresholdCalibrationConfig()

    opps, opps_status = load_csv_safe(TRADE_OPPORTUNITIES_CSV)
    applied, applied_status = load_csv_safe(APPLIED_ADJUSTMENTS_CSV)

    sources = {
        "trade_opportunities_csv": {
            "status": opps_status,
            "rows": int(opps.shape[0]),
            "path": str(TRADE_OPPORTUNITIES_CSV),
        },
        "applied_adjustments_csv": {
            "status": applied_status,
            "rows": int(applied.shape[0]),
            "path": str(APPLIED_ADJUSTMENTS_CSV),
        },
    }

    extra_notes: List[str] = []
    if opps_status != "ok":
        extra_notes.append(
            f"trade_opportunities.csv status='{opps_status}'; " "analysis emitted with zero rows."
        )

    out_df = analyze(opps, cfg)
    summary = build_summary(out_df, cfg, sources, extra_notes=extra_notes)
    written = write_outputs(out_df, summary)
    summary["written"] = written

    if verbose:
        print(f"[{MODULE_NAME}] opps_status={opps_status} rows={out_df.shape[0]}")
        print(f"[{MODULE_NAME}] zone_counts={summary['threshold_zone_counts']}")
        print(
            f"[{MODULE_NAME}] borderline_accept={summary['borderline_accept_count']} "
            f"borderline_reject={summary['borderline_reject_count']} "
            f"near_add={summary['near_add_threshold_count']} "
            f"near_conf_penalty={summary['near_confidence_penalty_count']}"
        )
        print(
            f"[{MODULE_NAME}] edge avg={summary['average_edge_score']} "
            f"min={summary['min_edge_score']} max={summary['max_edge_score']}"
        )
        print(f"[{MODULE_NAME}] verdict={summary['under_deployment_assessment']}")
        for k, p in written.items():
            print(f"[{MODULE_NAME}] {k}: {p}")

    return summary


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="threshold_calibration",
        description=(
            "Read-only threshold calibration analysis. Writes only "
            "data/results/threshold_calibration.* — does not modify "
            "execution, broker, lifecycle, signal generation, or configs."
        ),
    )
    p.add_argument(
        "--base-floor",
        type=float,
        default=0.50,
        help="Baseline edge_score floor used for the analysis (default: 0.50).",
    )
    p.add_argument(
        "--add-floor-offset",
        type=float,
        default=0.15,
        help="Additional floor offset applied to ADD opportunities (default: 0.15).",
    )
    p.add_argument(
        "--confidence-penalty-threshold",
        type=float,
        default=0.55,
        help="Confidence level below which the low-confidence penalty applies (default: 0.55).",
    )
    p.add_argument(
        "--borderline-band",
        type=float,
        default=0.05,
        help="Absolute distance band used to classify borderline rows (default: 0.05).",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of rows to include in top_borderline_* lists (default: 10).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress stdout logging.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    cfg = ThresholdCalibrationConfig(
        base_floor=float(args.base_floor),
        add_floor_offset=float(args.add_floor_offset),
        confidence_penalty_threshold=float(args.confidence_penalty_threshold),
        borderline_band=float(args.borderline_band),
        top_n=int(args.top_n),
    )
    out = run_calibration(cfg=cfg, verbose=not args.quiet)
    # Success = outputs were written (they always are on a happy path).
    return 0 if out and out.get("written") else 1


if __name__ == "__main__":
    sys.exit(main())
