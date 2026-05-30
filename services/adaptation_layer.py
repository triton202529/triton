# services/adaptation_layer.py
"""
TRITON — Controlled Adaptation Layer (advisory proposal engine).

Purpose
-------
Translate the feedback-loop's observational recommendations into explicit,
bounded, auditable, *reversible* parameter-adjustment proposals.

Inputs (best-effort, never required):
    data/results/feedback_recommendations.csv   (primary)
    data/results/feedback_loop_summary.json     (primary)
    data/results/feedback_loop_report.csv       (primary)
    data/results/execution_plan.csv             (optional)
    data/results/execution_intelligence.csv     (optional)
    data/results/target_weights.csv             (optional)
    data/results/trade_opportunities.csv        (optional)
    data/results/signal_lifecycle.csv           (optional)

Outputs:
    data/results/adaptation_proposals.csv       — every candidate proposal
    data/results/adaptation_summary.json        — availability + counts
    data/results/adaptation_review_queue.csv    — curated review list

Hard contract — Phase 1
-----------------------
1. This module never mutates live configs, thresholds, or execution behavior.
2. Every emitted proposal carries  advisory_only = True  and
                                   auto_apply_allowed = False.
3. Every numeric delta is *clamped* to a per-target safe range. If clamping
   was applied, bounded_change_applied is set to True so a human reviewer
   can see it.
4. Thin-data conditions (low evidence, missing inputs) are flagged
   conservatively and pushed to the bottom of the review queue.
5. The module performs no I/O against broker, lifecycle, signal, execution,
   portfolio, or risk control surfaces. Reads from data/results/ only.

Run
---
    python -m services.adaptation_layer
or
    python services/adaptation_layer.py
"""

from __future__ import annotations

import json
import math
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

INPUT_PATHS: Dict[str, Path] = {
    "feedback_recommendations": RESULTS / "feedback_recommendations.csv",
    "feedback_loop_summary": RESULTS / "feedback_loop_summary.json",
    "feedback_loop_report": RESULTS / "feedback_loop_report.csv",
    "execution_plan": RESULTS / "execution_plan.csv",
    "execution_intelligence": RESULTS / "execution_intelligence.csv",
    "target_weights": RESULTS / "target_weights.csv",
    "trade_opportunities": RESULTS / "trade_opportunities.csv",
    "signal_lifecycle": RESULTS / "signal_lifecycle.csv",
}

PROPOSALS_CSV = RESULTS / "adaptation_proposals.csv"
SUMMARY_JSON = RESULTS / "adaptation_summary.json"
REVIEW_QUEUE_CSV = RESULTS / "adaptation_review_queue.csv"

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

EVIDENCE_LOW = "LOW"
EVIDENCE_MEDIUM = "MEDIUM"
EVIDENCE_HIGH = "HIGH"

PRIORITY_LOW = "LOW"
PRIORITY_MEDIUM = "MEDIUM"
PRIORITY_HIGH = "HIGH"

STATUS_PROPOSED = "PROPOSED"

# Confidence ceilings by evidence strength (Phase 1, conservative).
_CONF_CAP_BY_EVIDENCE: Dict[str, float] = {
    EVIDENCE_LOW: 0.40,
    EVIDENCE_MEDIUM: 0.70,
    EVIDENCE_HIGH: 0.90,
    "": 0.30,  # safe default for unknown
}

# Minimum recommendation_confidence to even consider emitting a proposal.
_MIN_REC_CONFIDENCE = 0.20

# Minimum evidence_count below which we always force review_priority=LOW
# and thin_data_flag=True regardless of confidence.
_THIN_DATA_THRESHOLD = 5


# ─────────────────────────────────────────────────────────────
# Adaptation targets registry (bounded, allow-listed)
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AdaptationTarget:
    """A single tunable surface this layer is allowed to propose changes to."""

    name: str
    proposal_type: str
    direction: str  # "DECREASE" | "INCREASE" | "MAINTAIN_OR_SLIGHTLY_INCREASE"
    delta_min: float  # inclusive lower bound on proposed_delta
    delta_max: float  # inclusive upper bound on proposed_delta
    default_delta: float  # the engine's typical proposal magnitude
    note: str  # human-readable description


# Allow-list. The engine will *only* emit proposals for these targets.
ADAPTATION_TARGETS: Dict[str, AdaptationTarget] = {
    "execution_entry_aggressiveness": AdaptationTarget(
        name="execution_entry_aggressiveness",
        proposal_type="DECREASE_AGGRESSIVENESS",
        direction="DECREASE",
        delta_min=-0.15,
        delta_max=0.0,
        default_delta=-0.05,
        note="Reduce entry aggressiveness in poor execution environments.",
    ),
    "wide_spread_entry_penalty": AdaptationTarget(
        name="wide_spread_entry_penalty",
        proposal_type="INCREASE_PENALTY",
        direction="INCREASE",
        delta_min=0.0,
        delta_max=0.20,
        default_delta=0.10,
        note="Penalize / defer wide-spread entries more strongly.",
    ),
    "stale_quote_entry_caution": AdaptationTarget(
        name="stale_quote_entry_caution",
        proposal_type="INCREASE_CAUTION",
        direction="INCREASE",
        delta_min=0.0,
        delta_max=0.20,
        default_delta=0.10,
        note="Increase caution when entry quotes are stale.",
    ),
    "sizing_bucket_multiplier_adjustment": AdaptationTarget(
        name="sizing_bucket_multiplier_adjustment",
        proposal_type="ADJUST_BUCKET",
        direction="DECREASE",
        delta_min=-0.20,
        delta_max=0.10,
        default_delta=-0.10,
        note="Reduce or slightly boost a sizing bucket based on observed results.",
    ),
    "high_execution_risk_entry_trust": AdaptationTarget(
        name="high_execution_risk_entry_trust",
        proposal_type="DECREASE_TRUST",
        direction="DECREASE",
        delta_min=-0.15,
        delta_max=0.0,
        default_delta=-0.10,
        note="Reduce trust in HIGH execution-risk entry trades.",
    ),
    "high_conviction_bucket_validation": AdaptationTarget(
        name="high_conviction_bucket_validation",
        proposal_type="MAINTAIN_OR_SLIGHTLY_INCREASE",
        direction="MAINTAIN_OR_SLIGHTLY_INCREASE",
        delta_min=0.0,
        delta_max=0.10,
        default_delta=0.05,
        note="Maintain or slightly strengthen trust in high-conviction setups.",
    ),
    "signal_trust_adjustment": AdaptationTarget(
        name="signal_trust_adjustment",
        proposal_type="ADJUST_SIGNAL_TRUST",
        direction="ADJUST",
        delta_min=-0.10,
        delta_max=0.10,
        default_delta=0.05,
        note="Small trust adjustment for individual signals (Phase 1: capped tight).",
    ),
}

# Mapping: feedback recommendation_type → adaptation_target name + base
# direction. For SIZING_REVIEW / SIGNAL_* the direction depends on the
# specific recommendation text, handled in the rule below.
_REC_TO_TARGET: Dict[str, str] = {
    "EXECUTION_CAUTION": "high_execution_risk_entry_trust",
    "SPREAD_CAUTION": "wide_spread_entry_penalty",
    "QUOTE_FRESHNESS_WARNING": "stale_quote_entry_caution",
    "SIZING_REVIEW": "sizing_bucket_multiplier_adjustment",
    "EDGE_VALIDATION": "high_conviction_bucket_validation",
    "STYLE_REVIEW": "execution_entry_aggressiveness",
    "SIGNAL_TRUST_BOOST": "signal_trust_adjustment",
    "SIGNAL_CAUTION": "signal_trust_adjustment",
}


# ─────────────────────────────────────────────────────────────
# Safe IO helpers (mirrors services/feedback_loop.py contracts)
# ─────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_csv_safe(path: Path) -> Tuple[pd.DataFrame, str]:
    """Best-effort CSV loader. Returns (df, status); never raises."""
    try:
        if not path.exists():
            return pd.DataFrame(), "missing"
        try:
            if path.stat().st_size == 0:
                return pd.DataFrame(), "empty"
        except OSError:
            pass
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        if df is None or df.empty:
            return (df if df is not None else pd.DataFrame()), "empty"
        df.columns = [str(c).strip() for c in df.columns]
        return df, "ok"
    except Exception as e:
        return pd.DataFrame(), f"error:{type(e).__name__}:{str(e)[:120]}"


def load_json_safe(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Best-effort JSON loader. Returns (obj_or_None, status); never raises."""
    try:
        if not path.exists():
            return None, "missing"
        try:
            if path.stat().st_size == 0:
                return None, "empty"
        except OSError:
            pass
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None, "empty"
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, "ok"
        return {"value": obj}, "ok"
    except Exception as e:
        return None, f"error:{type(e).__name__}:{str(e)[:120]}"


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


def _safe_int(x: Any, default: int = 0) -> int:
    v = _safe_float(x, None)
    return int(v) if v is not None else default


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("nan", "none") else s


def _clamp(v: float, lo: float, hi: float) -> Tuple[float, bool]:
    """Clamp v into [lo, hi]. Returns (clamped_value, was_bounded)."""
    if v < lo:
        return lo, True
    if v > hi:
        return hi, True
    return v, False


# ─────────────────────────────────────────────────────────────
# Loader bundle
# ─────────────────────────────────────────────────────────────


@dataclass
class AdaptationInputs:
    csvs: Dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: Optional[Dict[str, Any]] = None
    status: Dict[str, str] = field(default_factory=dict)

    def missing(self) -> List[str]:
        return [name for name, s in self.status.items() if s != "ok"]


def load_inputs() -> AdaptationInputs:
    inp = AdaptationInputs()
    for name, path in INPUT_PATHS.items():
        if path.suffix.lower() == ".json":
            obj, status = load_json_safe(path)
            inp.status[name] = status
            if name == "feedback_loop_summary":
                inp.summary = obj
        else:
            df, status = load_csv_safe(path)
            inp.csvs[name] = df
            inp.status[name] = status
    return inp


# ─────────────────────────────────────────────────────────────
# Confidence + evidence helpers
# ─────────────────────────────────────────────────────────────


def _evidence_strength(count: int, fallback: str = "") -> str:
    if count <= 0:
        return fallback or EVIDENCE_LOW
    if count < 5:
        return EVIDENCE_LOW
    if count < 15:
        return EVIDENCE_MEDIUM
    return EVIDENCE_HIGH


def _proposal_confidence(
    rec_conf: Optional[float], evidence_strength: str, effect_size: Optional[float], thin_data: bool
) -> float:
    """
    Combine recommendation_confidence with evidence strength + effect size.
    Always clamped to [0, 1] and capped by per-evidence ceiling. Thin-data
    rows are additionally pushed below 0.50.
    """
    rc = rec_conf if rec_conf is not None and math.isfinite(rec_conf) else 0.0
    rc = max(0.0, min(1.0, rc))

    if effect_size is None or not math.isfinite(effect_size):
        eff = 0.50
    else:
        eff = math.tanh(abs(effect_size))  # 0 → 0; 1 → ~0.76; 3 → ~0.99
        eff = max(0.0, min(1.0, eff))

    base = 0.6 * rc + 0.4 * eff
    cap = _CONF_CAP_BY_EVIDENCE.get(evidence_strength, 0.40)
    conf = min(base, cap)
    if thin_data:
        conf = min(conf, 0.50)
    return float(round(max(0.0, conf), 4))


def _review_priority(proposal_confidence: float, evidence_strength: str, thin_data: bool) -> str:
    if thin_data:
        return PRIORITY_LOW
    if evidence_strength == EVIDENCE_HIGH and proposal_confidence >= 0.70:
        return PRIORITY_HIGH
    if proposal_confidence >= 0.55:
        return PRIORITY_MEDIUM
    return PRIORITY_LOW


# ─────────────────────────────────────────────────────────────
# Recommendation → proposal mapping
# ─────────────────────────────────────────────────────────────


def _direction_for_sizing_review(rec_text: str) -> str:
    """
    SIZING_REVIEW recommendations come in two flavors:
      - underperforming bucket  → direction DECREASE (negative delta)
      - the SIZING_REVIEW emitter never proposes 'increase';
        EDGE_VALIDATION handles the upside case.
    Default: DECREASE.
    """
    text = (rec_text or "").lower()
    if "underperform" in text or "reassess" in text:
        return "DECREASE"
    return "DECREASE"


def _direction_for_signal(rec_type: str) -> str:
    if rec_type == "SIGNAL_TRUST_BOOST":
        return "INCREASE"
    if rec_type == "SIGNAL_CAUTION":
        return "DECREASE"
    return "ADJUST"


def _scaled_default_delta(
    target: AdaptationTarget,
    direction: str,
    rec_conf: Optional[float],
    effect_size: Optional[float],
) -> float:
    """
    Pick a base delta inside the target's allow-listed range.

    Strategy:
      - Start from target.default_delta (already in-range).
      - Scale magnitude by max(rec_conf, effect_factor) so very weak signals
        produce smaller proposed deltas than strong ones.
      - Flip sign if direction conflicts with default_delta sign and the
        target permits both directions (e.g. sizing / signal-trust).
    """
    base = float(target.default_delta)
    rc = rec_conf if rec_conf is not None and math.isfinite(rec_conf) else 0.5
    if effect_size is None or not math.isfinite(effect_size):
        eff = 0.5
    else:
        eff = math.tanh(abs(effect_size))
    scale = max(0.25, min(1.0, max(rc, eff)))
    proposed = base * scale

    if direction == "INCREASE" and proposed < 0:
        proposed = abs(proposed)
    elif direction == "DECREASE" and proposed > 0:
        proposed = -abs(proposed)
    return float(proposed)


def _proposal_id(rec_type: str, target_name: str, related: str) -> str:
    """Stable-ish, human-greppable id with a short uuid suffix."""
    rel = (related or "ANY").replace(" ", "_")
    suf = uuid.uuid4().hex[:8]
    return f"ADAPT-{rec_type}-{target_name}-{rel}-{suf}"


def _row_for_recommendation(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map a single feedback-recommendation row to a proposal row, or None."""
    rec_type = _safe_str(rec.get("recommendation_type")).upper()
    if not rec_type or rec_type not in _REC_TO_TARGET:
        return None
    target_name = _REC_TO_TARGET[rec_type]
    target = ADAPTATION_TARGETS.get(target_name)
    if target is None:
        return None

    rec_text = _safe_str(rec.get("recommendation_text"))
    rec_conf = _safe_float(rec.get("recommendation_confidence"), 0.0) or 0.0
    if rec_conf < _MIN_REC_CONFIDENCE:
        return None

    evidence_count = _safe_int(rec.get("evidence_count"), 0)
    evid_strength_raw = _safe_str(rec.get("evidence_strength")).upper()
    evid_strength = (
        evid_strength_raw
        if evid_strength_raw in (EVIDENCE_LOW, EVIDENCE_MEDIUM, EVIDENCE_HIGH)
        else _evidence_strength(evidence_count)
    )
    thin_data = evidence_count < _THIN_DATA_THRESHOLD

    related_bucket = _safe_str(rec.get("related_bucket"))
    related_flag = _safe_str(rec.get("related_flag"))
    related_style = _safe_str(rec.get("related_style"))
    related = related_bucket or related_flag or related_style

    # Per-recommendation direction overrides
    if rec_type == "SIZING_REVIEW":
        direction = _direction_for_sizing_review(rec_text)
    elif rec_type in ("SIGNAL_TRUST_BOOST", "SIGNAL_CAUTION"):
        direction = _direction_for_signal(rec_type)
    else:
        direction = target.direction

    # The metric_snapshot string from the feedback layer carries enough
    # context that we forward it verbatim into the proposal evidence.
    snap = _safe_str(rec.get("metric_snapshot"))

    # Effect size estimate: prefer recommendation_confidence as proxy
    # (the feedback layer already folded effect-size into it). If we can
    # parse a slip / win_rate hint from the snapshot, use that as a
    # secondary signal — best-effort only, never fatal.
    effect_size = rec_conf
    if "avg_slip_bps=" in snap:
        try:
            tail = snap.split("avg_slip_bps=", 1)[1]
            num = float(tail.split(",", 1)[0])
            effect_size = max(effect_size, min(1.0, num / 50.0))
        except Exception:
            pass

    raw_delta = _scaled_default_delta(target, direction, rec_conf, effect_size)
    proposed_delta, bounded = _clamp(raw_delta, target.delta_min, target.delta_max)

    proposal_strength = "WEAK"
    abs_max = max(abs(target.delta_min), abs(target.delta_max)) or 1.0
    frac = abs(proposed_delta) / abs_max
    if frac >= 0.66:
        proposal_strength = "STRONG"
    elif frac >= 0.33:
        proposal_strength = "MODERATE"

    proposal_conf = _proposal_confidence(rec_conf, evid_strength, effect_size, thin_data)
    priority = _review_priority(proposal_conf, evid_strength, thin_data)

    observed_group = related or rec_type
    observed_metric = "metric_snapshot"
    observed_value = snap

    reason = (
        f"Mapped from feedback recommendation '{rec_type}' "
        f"(rec_conf={rec_conf:.2f}, evidence={evid_strength}, n={evidence_count}); "
        f"target={target.name}; direction={direction}; "
        f"delta={proposed_delta:+.3f} bounded=[{target.delta_min:+.2f},{target.delta_max:+.2f}]"
        + (" [clamped]" if bounded else "")
    )

    return {
        # Identity
        "proposal_id": _proposal_id(rec_type, target.name, related),
        "generated_at_utc": _utc_now_iso(),
        "adaptation_target": target.name,
        "proposal_type": target.proposal_type,
        # Evidence
        "recommendation_type": rec_type,
        "source_recommendation_text": rec_text,
        "evidence_count": int(evidence_count),
        "evidence_strength": evid_strength,
        "recommendation_confidence": float(round(rec_conf, 4)),
        "observed_group": observed_group,
        "observed_metric": observed_metric,
        "observed_value": observed_value,
        "baseline_value": "",
        "effect_direction": direction,
        # Proposed change
        "current_value": "",
        "proposed_value": "",
        "proposed_delta": float(round(proposed_delta, 4)),
        "proposal_direction": direction,
        "proposal_strength": proposal_strength,
        "proposal_confidence": float(round(proposal_conf, 4)),
        # Guardrails
        "min_allowed_value": float(target.delta_min),
        "max_allowed_value": float(target.delta_max),
        "bounded_change_applied": bool(bounded),
        "requires_manual_review": True,
        "auto_apply_allowed": False,
        # Explanation
        "proposal_reason": reason,
        "proposal_note": target.note,
        "advisory_only": True,
        # Status
        "status": STATUS_PROPOSED,
        "review_priority": priority,
        "thin_data_flag": bool(thin_data),
        # Optional context to help the reviewer
        "related_bucket": related_bucket,
        "related_flag": related_flag,
        "related_style": related_style,
    }


def build_proposals(inp: AdaptationInputs) -> pd.DataFrame:
    """Iterate the feedback recommendations DataFrame and emit proposal rows."""
    rec_df = inp.csvs.get("feedback_recommendations", pd.DataFrame())
    if rec_df is None or rec_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, rec in rec_df.iterrows():
        try:
            r = _row_for_recommendation(rec.to_dict())
        except Exception:
            r = None
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Review queue
# ─────────────────────────────────────────────────────────────

_PRIORITY_ORDER: Dict[str, int] = {
    PRIORITY_HIGH: 0,
    PRIORITY_MEDIUM: 1,
    PRIORITY_LOW: 2,
    "": 3,
}


def build_review_queue(proposals: pd.DataFrame, *, top_n: int = 25) -> pd.DataFrame:
    """Curated subset: highest priority + confidence first; thin-data last."""
    if proposals is None or proposals.empty:
        return pd.DataFrame()
    df = proposals.copy()
    df["__pri"] = df.get("review_priority", "").map(
        lambda p: _PRIORITY_ORDER.get(str(p).upper(), 3)
    )
    df["__thin"] = df.get("thin_data_flag", False).astype(bool).map(lambda b: 1 if b else 0)
    df["__conf"] = pd.to_numeric(df.get("proposal_confidence"), errors="coerce").fillna(0.0)
    df["__evid"] = pd.to_numeric(df.get("evidence_count"), errors="coerce").fillna(0)
    df = df.sort_values(
        ["__thin", "__pri", "__conf", "__evid"],
        ascending=[True, True, False, False],
    )
    df = df.drop(columns=["__pri", "__thin", "__conf", "__evid"])
    return df.head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────


def _df_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Stable CSV cast — bool → '1'/'0'/''. Avoid duplicate columns."""
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


def _empty_proposals_df() -> pd.DataFrame:
    cols = [
        "proposal_id",
        "generated_at_utc",
        "adaptation_target",
        "proposal_type",
        "recommendation_type",
        "source_recommendation_text",
        "evidence_count",
        "evidence_strength",
        "recommendation_confidence",
        "observed_group",
        "observed_metric",
        "observed_value",
        "baseline_value",
        "effect_direction",
        "current_value",
        "proposed_value",
        "proposed_delta",
        "proposal_direction",
        "proposal_strength",
        "proposal_confidence",
        "min_allowed_value",
        "max_allowed_value",
        "bounded_change_applied",
        "requires_manual_review",
        "auto_apply_allowed",
        "proposal_reason",
        "proposal_note",
        "advisory_only",
        "status",
        "review_priority",
        "thin_data_flag",
        "related_bucket",
        "related_flag",
        "related_style",
    ]
    return pd.DataFrame(columns=cols)


def write_outputs(
    proposals: pd.DataFrame,
    review_queue: pd.DataFrame,
    inp: AdaptationInputs,
    notes: Optional[List[str]] = None,
) -> Dict[str, str]:
    notes = notes or []
    written: Dict[str, str] = {}

    PROPOSALS_CSV.parent.mkdir(parents=True, exist_ok=True)

    try:
        if proposals is None or proposals.empty:
            _empty_proposals_df().to_csv(PROPOSALS_CSV, index=False)
        else:
            _df_for_csv(proposals).to_csv(PROPOSALS_CSV, index=False)
        written["proposals_csv"] = str(PROPOSALS_CSV)
    except Exception as e:
        written["proposals_csv_error"] = f"{type(e).__name__}:{e}"

    try:
        if review_queue is None or review_queue.empty:
            _empty_proposals_df().to_csv(REVIEW_QUEUE_CSV, index=False)
        else:
            _df_for_csv(review_queue).to_csv(REVIEW_QUEUE_CSV, index=False)
        written["review_queue_csv"] = str(REVIEW_QUEUE_CSV)
    except Exception as e:
        written["review_queue_csv_error"] = f"{type(e).__name__}:{e}"

    summary = _build_summary_obj(proposals, inp, notes=notes)
    try:
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        written["summary_json"] = str(SUMMARY_JSON)
    except Exception as e:
        written["summary_json_error"] = f"{type(e).__name__}:{e}"

    return written


def _vc(df: pd.DataFrame, col: str) -> Dict[str, int]:
    if df is None or df.empty or col not in df.columns:
        return {}
    try:
        return df[col].fillna("").astype(str).value_counts(dropna=False).to_dict()
    except Exception:
        return {}


def _top_proposals_for_summary(df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    sub = df.copy()
    if "review_priority" in sub.columns:
        sub["__pri"] = sub["review_priority"].map(lambda p: _PRIORITY_ORDER.get(str(p).upper(), 3))
    else:
        sub["__pri"] = 3
    sub["__conf"] = pd.to_numeric(sub.get("proposal_confidence"), errors="coerce").fillna(0.0)
    sub["__evid"] = pd.to_numeric(sub.get("evidence_count"), errors="coerce").fillna(0)
    sub["__thin"] = sub.get("thin_data_flag", False).astype(bool).map(lambda b: 1 if b else 0)
    sub = sub.sort_values(
        ["__thin", "__pri", "__conf", "__evid"],
        ascending=[True, True, False, False],
    ).head(top_n)
    keep = [
        c
        for c in (
            "proposal_id",
            "adaptation_target",
            "proposal_type",
            "recommendation_type",
            "evidence_count",
            "evidence_strength",
            "proposed_delta",
            "proposal_direction",
            "proposal_strength",
            "proposal_confidence",
            "review_priority",
            "bounded_change_applied",
            "thin_data_flag",
            "related_bucket",
            "related_flag",
            "related_style",
        )
        if c in sub.columns
    ]
    return sub[keep].to_dict(orient="records") if keep else []


def _build_summary_obj(
    proposals: pd.DataFrame, inp: AdaptationInputs, notes: List[str]
) -> Dict[str, Any]:
    n = 0 if proposals is None else int(proposals.shape[0])
    thin_count = 0
    if proposals is not None and not proposals.empty and "thin_data_flag" in proposals.columns:
        try:
            thin_count = int(proposals["thin_data_flag"].astype(bool).sum())
        except Exception:
            thin_count = 0

    src_rows: Dict[str, Dict[str, Any]] = {}
    for name, path in INPUT_PATHS.items():
        rows = 0
        if path.suffix.lower() == ".csv":
            df = inp.csvs.get(name, pd.DataFrame())
            try:
                rows = int(df.shape[0])
            except Exception:
                rows = 0
        else:
            obj = inp.summary if name == "feedback_loop_summary" else None
            rows = 0 if obj is None else 1
        src_rows[name] = {
            "status": inp.status.get(name, "missing"),
            "rows": rows,
            "path": str(path),
        }

    return {
        "generated_at_utc": _utc_now_iso(),
        "schema_version": 1,
        "advisory_only": True,
        "auto_apply_allowed": False,
        "phase": "1-advisory-proposals-only",
        "source_availability": src_rows,
        "missing_inputs": [n for n, s in inp.status.items() if s != "ok"],
        "proposal_count": n,
        "thin_data_proposal_count": thin_count,
        "proposal_count_by_target": _vc(proposals, "adaptation_target"),
        "proposal_count_by_type": _vc(proposals, "proposal_type"),
        "proposal_count_by_priority": _vc(proposals, "review_priority"),
        "proposal_count_by_evidence_strength": _vc(proposals, "evidence_strength"),
        "top_proposals": _top_proposals_for_summary(proposals),
        "adaptation_targets": [
            {
                "name": t.name,
                "proposal_type": t.proposal_type,
                "direction": t.direction,
                "delta_min": t.delta_min,
                "delta_max": t.delta_max,
                "default_delta": t.default_delta,
                "note": t.note,
            }
            for t in ADAPTATION_TARGETS.values()
        ],
        "notes": notes,
    }


# ─────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────


def run_adaptation_layer(verbose: bool = True) -> Dict[str, Any]:
    """Run the full pipeline. Always returns a result dict, never raises."""
    inp = load_inputs()
    notes: List[str] = []
    if inp.missing():
        notes.append(
            "Missing or unreadable input(s): "
            + ", ".join(inp.missing())
            + ". Adaptation proposals were generated on best-effort partial data."
        )

    rec_df = inp.csvs.get("feedback_recommendations", pd.DataFrame())
    if rec_df is None or rec_df.empty:
        notes.append(
            "No feedback recommendations found — emitting empty proposals "
            "and an empty review queue. Run `python -m services.feedback_loop` "
            "first to populate `data/results/feedback_recommendations.csv`."
        )

    try:
        proposals = build_proposals(inp)
    except Exception as e:
        proposals = pd.DataFrame()
        notes.append(f"build_proposals error: {type(e).__name__}: {e}")

    try:
        review_queue = build_review_queue(proposals)
    except Exception as e:
        review_queue = pd.DataFrame()
        notes.append(f"build_review_queue error: {type(e).__name__}: {e}")

    if proposals is not None and not proposals.empty:
        try:
            thin = (
                proposals["thin_data_flag"].astype(bool).mean()
                if ("thin_data_flag" in proposals.columns)
                else 0
            )
            if thin > 0.5:
                notes.append(
                    "More than half of proposals were marked thin_data_flag=True; "
                    "treat the adaptation slate as low-confidence."
                )
        except Exception:
            pass

    written = write_outputs(proposals, review_queue, inp, notes=notes)

    if verbose:
        print(f"[adaptation_layer] sources_ok={[n for n,s in inp.status.items() if s=='ok']}")
        print(f"[adaptation_layer] missing={inp.missing()}")
        print(
            f"[adaptation_layer] proposals="
            f"{0 if proposals is None else proposals.shape[0]} "
            f"review_queue={0 if review_queue is None else review_queue.shape[0]}"
        )
        for k, v in written.items():
            print(f"[adaptation_layer] {k}: {v}")

    return {
        "proposals": proposals,
        "review_queue": review_queue,
        "written": written,
        "notes": notes,
        "source_status": inp.status,
    }


def main(argv: Optional[List[str]] = None) -> int:
    _ = argv
    try:
        run_adaptation_layer(verbose=True)
        return 0
    except Exception as e:
        print(f"[adaptation_layer] FATAL {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
