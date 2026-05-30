"""
Governance Trust Feedback Engine -- Step 17 (diagnostics -> self-trust).

Reads:
    data/results/autonomous_strategy_diagnostics.json
    data/results/autonomous_strategy_summary.json
    data/results/meta_decision_intelligence.json
    data/results/portfolio_memory_insights.json
    data/results/adaptive_regime.json

Writes:
    data/results/governance_trust_feedback.json
    data/results/governance_runtime_adjustments.json

Purpose
-------
Step 16 (autonomous_strategy_diagnostics) measures whether Triton's
autonomous governance system actually works. Step 17 closes the
feedback loop by translating those measurements into bounded trust
adjustments:

    "Has Triton earned the right to trust itself?"

When the autonomous committee (Step 15) has historically made *good*
decisions -- alpha preserved during deployments, drawdowns avoided
during defensive calls, monotonic trust quality, accurate regime
classification -- this engine emits positive trust feedback so that
future cycles deploy more readily, hold less cash, and let
self-confidence rise.

When the governance system has been *poor* -- decisions fail more
often than expected, defensives fire after the drawdown, trust
levels don't predict success -- the engine emits negative feedback
that tightens confidence requirements, raises cash reserves, and
caps self-confidence growth.

Bounded outputs
---------------
All six deltas are *strictly* clamped to +/-0.05 per spec section 2.
A bad diagnostic blob therefore cannot shift Triton's behaviour by
more than the cap in a single cycle. The deltas accumulate slowly
across cycles, mirroring how human institutional governance evolves.

Trust levels (spec section 3)
-----------------------------
governance_health is the master signal -- a weighted blend of the
six Step 16 scores, renormalised over only the *known* scores so
"insufficient_history" never artificially pulls the score toward
neutral. The classification ladder is:

    COLLAPSED      gov_health < 0.30
    WEAK           gov_health < 0.45
    STABLE         gov_health < 0.60
    STRONG         gov_health < 0.80
    VERY_STRONG    gov_health >= 0.80

When too few Step 16 sub-scores are known (default <2), the engine
declines to emit non-zero deltas and reports trust level STABLE
with rationale_tags=["insufficient_governance_history"]. This is
the safe default until the autonomous diagnostics have accumulated
real labelled history.

Safety
------
* READ ONLY. No broker calls, no engine state mutation. The deltas
  are written to JSON for future downstream consumption only.
* Every delta hard-clamped to MAX_ABS_DELTA (default 0.05).
* Atomic writes (.tmp + os.replace) for both outputs.
* Missing inputs warn-and-continue. With no diagnostics blob the
  engine still emits a defensible zero-delta STABLE feedback so the
  schema invariant holds.
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_DIAGNOSTICS = RESULTS_DIR / "autonomous_strategy_diagnostics.json"
DEFAULT_DIAG_SUMMARY = RESULTS_DIR / "autonomous_strategy_summary.json"
DEFAULT_META_INTEL = RESULTS_DIR / "meta_decision_intelligence.json"
DEFAULT_MEMORY_INSIGHTS = RESULTS_DIR / "portfolio_memory_insights.json"
DEFAULT_REGIME = RESULTS_DIR / "adaptive_regime.json"

DEFAULT_OUT_FEEDBACK = RESULTS_DIR / "governance_trust_feedback.json"
DEFAULT_OUT_ADJUSTMENTS = RESULTS_DIR / "governance_runtime_adjustments.json"

# -----------------------------------------------------------
# Tunables
# -----------------------------------------------------------
MAX_ABS_DELTA = 0.05  # spec section 2 hard cap
MIN_KNOWN_SCORES_FOR_FEEDBACK = 2  # need at least this many is_known scores
# before emitting non-zero deltas
MIN_LABELLED_OBSERVATIONS = 5  # mirrors Step 16's MIN_SAMPLE_SIZE

# Composite weights for governance_health. Sum is normalised over
# the *known* scores so missing inputs never artificially pull the
# blend toward 0.5.
SCORE_WEIGHTS: Dict[str, float] = {
    "decision_quality_score": 0.25,
    "governance_quality_score": 0.20,
    "drawdown_avoidance_score": 0.15,
    "deployment_accuracy_score": 0.15,
    "regime_prediction_score": 0.15,
    "trust_quality_score": 0.10,
}

# Per-field delta scaling. Tuple = (max_abs_field, sign).
#
#   sign = +1  ->  positive when governance is GOOD
#                  (raise trust, deploy more, get more aggressive)
#   sign = -1  ->  negative when governance is GOOD
#                  (relax confidence bar, hold less cash, less skeptical)
#
# At the extreme of governance_health == 1.0 the field saturates at
# its per-field cap (with the correct sign). At gov == 0.5 every
# delta is exactly zero. At gov == 0.0 every field reaches its per-
# field cap with the *opposite* sign. Independently clamped to
# +/-MAX_ABS_DELTA so a future scaling tweak cannot accidentally
# escape the spec cap.
FIELD_SPEC: Dict[str, Tuple[float, int]] = {
    "trust_delta": (0.05, +1),
    "confidence_delta": (0.04, -1),
    "aggressiveness_delta": (0.05, +1),
    "skepticism_delta": (0.04, -1),
    "deployment_delta": (0.05, +1),
    "cash_delta": (0.05, -1),
}

# Trust-level thresholds on governance_health.
TRUST_LEVELS: Tuple[Tuple[float, str], ...] = (
    (0.80, "VERY_STRONG"),
    (0.60, "STRONG"),
    (0.45, "STABLE"),
    (0.30, "WEAK"),
    (0.00, "COLLAPSED"),
)

# Decision-quality bands referenced in rationale_short.
DQ_GOOD_FLOOR = 0.70
DQ_POOR_CEILING = 0.35


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[GOVERNANCE_TRUST_WARN] {msg}", flush=True)


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


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x or "").strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm01(x: Any, default: float = 0.50) -> float:
    v = _to_float(x)
    if v is None:
        return default
    return _clamp(v, 0.0, 1.0)


def _trust_level(score: float) -> str:
    for threshold, label in TRUST_LEVELS:
        if score >= threshold:
            return label
    return "COLLAPSED"


# -----------------------------------------------------------
# Score extraction from Step 16 diagnostics
# -----------------------------------------------------------
def _extract_scores(diagnostics: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """
    Pull the six diagnostic scores and their is_known flags from the
    Step 16 blob. Missing scores default to 0.50 with is_known=False.
    """
    raw_scores = diagnostics.get("scores") or {}
    raw_known = diagnostics.get("scores_known") or {}
    scores: Dict[str, float] = {}
    known: Dict[str, bool] = {}
    for k in SCORE_WEIGHTS:
        if k == "decision_quality_score":
            # Top-level field (not inside `scores`)
            v = _to_float(diagnostics.get("decision_quality_score"))
            scores[k] = 0.50 if v is None else _clamp(v, 0.0, 1.0)
            # decision_quality is "known" iff any of the sub-scores were known
            known[k] = any(bool(b) for b in raw_known.values())
        else:
            scores[k] = _norm01(raw_scores.get(k))
            known[k] = bool(raw_known.get(k, False))
    return scores, known


def _compute_governance_health(
    scores: Dict[str, float],
    known: Dict[str, bool],
) -> Tuple[float, Dict[str, float]]:
    """
    Weighted blend over the *known* sub-scores; weights renormalise.
    Returns (governance_health, contributing_weights) for transparency.
    """
    total_w = 0.0
    weighted = 0.0
    contributing: Dict[str, float] = {}
    for k, w in SCORE_WEIGHTS.items():
        if not known.get(k):
            continue
        v = scores.get(k, 0.50)
        weighted += w * v
        total_w += w
        contributing[k] = w
    if total_w <= 0.0:
        return 0.50, contributing
    return _clamp(weighted / total_w, 0.0, 1.0), contributing


# -----------------------------------------------------------
# Delta builder
# -----------------------------------------------------------
def _build_deltas(governance_health: float, *, active: bool) -> Dict[str, float]:
    """
    Map governance_health (0-1) into the six bounded deltas.

    trust_delta_raw = sign * field_cap * (gov_health - 0.5) * 2

    Then clamped to +/-MAX_ABS_DELTA. When ``active`` is False
    (insufficient labelled history) every delta is exactly zero --
    the trust system never moves on noise.
    """
    out: Dict[str, float] = {}
    if not active:
        return {k: 0.0 for k in FIELD_SPEC}
    delta_norm = _clamp((governance_health - 0.5) * 2.0, -1.0, 1.0)
    for field, (cap, sign) in FIELD_SPEC.items():
        raw = sign * cap * delta_norm
        out[field] = round(_clamp(raw, -MAX_ABS_DELTA, MAX_ABS_DELTA), 4)
    return out


# -----------------------------------------------------------
# Rationale builder
# -----------------------------------------------------------
def _build_rationale(
    *,
    trust_level: str,
    governance_health: float,
    scores: Dict[str, float],
    known: Dict[str, bool],
    deltas: Dict[str, float],
    active: bool,
    observations: int,
    labelled: int,
) -> Tuple[str, str, List[str]]:
    tags: List[str] = []
    if not active:
        if labelled < MIN_LABELLED_OBSERVATIONS:
            tags.append("insufficient_labelled_history")
            reason_short = (
                f"Governance feedback dormant: {labelled} labelled observation(s) "
                f"below the {MIN_LABELLED_OBSERVATIONS}-cycle floor."
            )
        else:
            tags.append("insufficient_governance_history")
            reason_short = (
                f"Governance feedback dormant: only "
                f"{sum(1 for v in known.values() if v)} of {len(SCORE_WEIGHTS)} "
                f"diagnostic sub-scores have enough history to be trusted."
            )
        reason_long = (
            f"{reason_short} All trust deltas held at zero pending more "
            f"diagnostic samples. {observations} cycle(s) on record."
        )
        return reason_short, reason_long, tags

    dq = scores.get("decision_quality_score", 0.50)
    bullets: List[str] = []

    if dq > DQ_GOOD_FLOOR:
        tags.append("decision_quality_strong")
        bullets.append(f"decision_quality {dq:.2f} above good floor ({DQ_GOOD_FLOOR})")
    elif dq < DQ_POOR_CEILING:
        tags.append("decision_quality_poor")
        bullets.append(f"decision_quality {dq:.2f} below poor ceiling ({DQ_POOR_CEILING})")
    else:
        tags.append("decision_quality_neutral")

    for k in (
        "governance_quality_score",
        "drawdown_avoidance_score",
        "deployment_accuracy_score",
        "regime_prediction_score",
        "trust_quality_score",
    ):
        if not known.get(k):
            continue
        v = scores.get(k, 0.50)
        if v < 0.40:
            tag = f"{k.replace('_score', '')}_weak"
            tags.append(tag)
            bullets.append(f"{k.replace('_score', '')} {v:.2f} weak")
        elif v > 0.70:
            tag = f"{k.replace('_score', '')}_strong"
            tags.append(tag)
            bullets.append(f"{k.replace('_score', '')} {v:.2f} strong")

    if not bullets:
        bullets.append("balanced sub-scores around neutral")

    direction = {
        "VERY_STRONG": "expanded autonomy across the trust stack",
        "STRONG": "modestly expanded autonomy",
        "STABLE": "kept the trust stack neutral",
        "WEAK": "tightened deployment and raised skepticism",
        "COLLAPSED": "sharply reined in autonomy and raised cash discipline",
    }.get(trust_level, "kept the trust stack neutral")

    reason_short = f"Governance health {governance_health:.2f} ({trust_level}); " f"{direction}."
    reason_long = (
        f"Triton {direction} because governance health is {governance_health:.2f} "
        f"({trust_level}). Contributing factors: " + ", ".join(bullets) + ". "
        f"Deltas issued -- trust={deltas['trust_delta']:+.3f}, "
        f"confidence={deltas['confidence_delta']:+.3f}, "
        f"aggressiveness={deltas['aggressiveness_delta']:+.3f}, "
        f"skepticism={deltas['skepticism_delta']:+.3f}, "
        f"deployment={deltas['deployment_delta']:+.3f}, "
        f"cash={deltas['cash_delta']:+.3f}."
    )
    return reason_short, reason_long, tags


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_feedback(
    *,
    diagnostics: Dict[str, Any],
    diag_summary: Dict[str, Any],
    meta_intel: Dict[str, Any],
    memory_insights: Dict[str, Any],
    regime_json: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    now_iso = _now_iso_utc()

    scores, known = _extract_scores(diagnostics)
    n_known = sum(1 for v in known.values() if v)
    n_labelled = int(
        diagnostics.get("memory_size_with_outcome")
        or diag_summary.get("memory_size_with_outcome")
        or 0
    )
    observations = int(
        diagnostics.get("memory_size_total") or diag_summary.get("memory_size_total") or 0
    )
    active = n_known >= MIN_KNOWN_SCORES_FOR_FEEDBACK and n_labelled >= MIN_LABELLED_OBSERVATIONS

    governance_health, contributing = _compute_governance_health(scores, known)
    trust_level = _trust_level(governance_health) if active else "STABLE"
    deltas = _build_deltas(governance_health, active=active)
    reason_short, reason_long, tags = _build_rationale(
        trust_level=trust_level,
        governance_health=governance_health,
        scores=scores,
        known=known,
        deltas=deltas,
        active=active,
        observations=observations,
        labelled=n_labelled,
    )

    # Pull meta context for provenance / cross-referencing
    meta_trust = (
        str((meta_intel or {}).get("trust_level") or "MODERATE").strip().upper() or "MODERATE"
    )
    meta_self_conf = _norm01((meta_intel or {}).get("self_confidence_score"))
    regime = (
        str((regime_json or {}).get("regime") or diagnostics.get("regime") or "UNKNOWN")
        .strip()
        .upper()
        or "UNKNOWN"
    )

    feedback: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "governance_trust_feedback_engine",
        "engine_version": 1,
        "regime": regime,
        "governance_trust_level": trust_level,
        "governance_health_score": round(governance_health, 6),
        "active": bool(active),
        "scores": {k: round(v, 6) for k, v in scores.items()},
        "scores_known": known,
        "contributing_weights": contributing,
        "n_known_scores": n_known,
        "n_labelled_observations": n_labelled,
        "memory_size_total": observations,
        "deltas": deltas,
        "rationale_short": reason_short,
        "rationale_long": reason_long,
        "rationale_tags": tags,
        "meta_context": {
            "meta_trust_level": meta_trust,
            "meta_self_confidence_score": round(meta_self_conf, 4),
        },
        "thresholds": {
            "max_abs_delta": MAX_ABS_DELTA,
            "min_known_scores_for_feedback": MIN_KNOWN_SCORES_FOR_FEEDBACK,
            "min_labelled_observations": MIN_LABELLED_OBSERVATIONS,
            "decision_quality_good_floor": DQ_GOOD_FLOOR,
            "decision_quality_poor_ceiling": DQ_POOR_CEILING,
            "trust_level_bands": [{"min_score": t, "label": lbl} for t, lbl in TRUST_LEVELS],
        },
        "inputs_seen": {
            "autonomous_strategy_diagnostics": bool(diagnostics),
            "autonomous_strategy_summary": bool(diag_summary),
            "meta_decision_intelligence": bool(meta_intel),
            "portfolio_memory_insights": bool(memory_insights),
            "adaptive_regime": bool(regime_json),
        },
    }

    adjustments: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "governance_trust_feedback_engine",
        "engine_version": 1,
        "regime": regime,
        "governance_trust_level": trust_level,
        "governance_health_score": round(governance_health, 6),
        "active": bool(active),
        "policy_bounds": {
            "max_abs_delta": MAX_ABS_DELTA,
        },
        "deltas": deltas,
        "field_caps": {f: cap for f, (cap, _) in FIELD_SPEC.items()},
        "field_signs": {
            f: ("HIGH_GOV_POSITIVE" if sign > 0 else "HIGH_GOV_NEGATIVE")
            for f, (_, sign) in FIELD_SPEC.items()
        },
        "rationale_short": reason_short,
        "rationale_tags": tags,
    }
    return feedback, adjustments


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only governance trust feedback engine (Step 17). "
            "Translates Step 16 strategy diagnostics into bounded "
            "trust-system deltas. Stays dormant until enough labelled "
            "history exists to make the feedback statistically meaningful."
        ),
    )
    p.add_argument("--diagnostics", default=str(DEFAULT_DIAGNOSTICS))
    p.add_argument("--diag-summary", default=str(DEFAULT_DIAG_SUMMARY))
    p.add_argument("--meta-intel", default=str(DEFAULT_META_INTEL))
    p.add_argument("--memory-insights", default=str(DEFAULT_MEMORY_INSIGHTS))
    p.add_argument("--regime", default=str(DEFAULT_REGIME))
    p.add_argument("--out-feedback", default=str(DEFAULT_OUT_FEEDBACK))
    p.add_argument("--out-adjustments", default=str(DEFAULT_OUT_ADJUSTMENTS))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[GOVERNANCE_TRUST] starting (read-only diagnostics -> trust feedback)", flush=True)

    diagnostics = _safe_read_json(
        Path(args.diagnostics), label="autonomous_strategy_diagnostics.json"
    )
    diag_summary = _safe_read_json(
        Path(args.diag_summary), label="autonomous_strategy_summary.json"
    )
    meta_intel = _safe_read_json(Path(args.meta_intel), label="meta_decision_intelligence.json")
    memory_insights = _safe_read_json(
        Path(args.memory_insights), label="portfolio_memory_insights.json"
    )
    regime_json = _safe_read_json(Path(args.regime), label="adaptive_regime.json")

    feedback, adjustments = build_feedback(
        diagnostics=diagnostics,
        diag_summary=diag_summary,
        meta_intel=meta_intel,
        memory_insights=memory_insights,
        regime_json=regime_json,
    )

    try:
        _atomic_write_json(feedback, Path(args.out_feedback))
    except Exception as e:
        _warn(f"failed to write {args.out_feedback}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(adjustments, Path(args.out_adjustments))
    except Exception as e:
        _warn(f"failed to write {args.out_adjustments}: {type(e).__name__}: {e}")
        return 2

    print(
        "[GOVERNANCE_TRUST] "
        f"trust={feedback['governance_trust_level']} "
        f"delta={adjustments['deltas']['trust_delta']:+.3f} "
        f"decision_quality={feedback['scores']['decision_quality_score']:.3f} "
        f"governance_quality={feedback['scores']['governance_quality_score']:.3f}"
        + ("" if feedback["active"] else " [DORMANT]"),
        flush=True,
    )
    print(
        f"[GOVERNANCE_TRUST_OUT] feedback={Path(args.out_feedback).as_posix()} "
        f"adjustments={Path(args.out_adjustments).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
