"""
Meta Decision Intelligence Engine -- Step 13 (self-trust layer).

Reads:
    data/results/portfolio_memory_insights.json
    data/results/portfolio_learning_adjustments.json
    data/results/investment_committee_summary.json
    data/results/adaptive_regime.json
    data/results/adaptive_policy.json
    data/results/portfolio_execution_summary.json
    data/results/trade_outcomes.csv

Writes:
    data/results/meta_decision_intelligence.json
    data/results/meta_runtime_adjustments.json

Purpose
-------
Steps 1-11 are prospective (recommendations from current state).
Step 12 is retrospective (memory of what happened).
Step 13 is *introspective* -- it inspects the prior twelve layers
and answers:

    "How much should Triton trust itself today?"

It produces:

    * meta_decision_intelligence.json
        - the diagnostic readout: seven measured scores, a single
          normalised self_confidence_score, a categorical trust_level,
          and a human-readable rationale.
    * meta_runtime_adjustments.json
        - six bounded (+/-0.10) runtime modifiers that downstream
          policy layers can additively apply when self-trust is
          elevated or impaired.

Modifier sign convention
------------------------
A *positive* modifier on each field means:
    confidence_modifier      raise the confidence bar (be stricter)
    persistence_modifier     raise the persistence bar (be stricter)
    deployment_modifier      deploy more capital (be braver)
    execution_modifier       execute more readily (be braver)
    cash_modifier            hold more cash (be defensive)
    aggressiveness_modifier  size up / take more risk (be braver)

Low trust therefore pushes (+confidence, +persistence, +cash) and
(-deployment, -execution, -aggressiveness). High trust pushes the
opposite. Every field is independently clamped at +/-0.10 per spec.

Safety
------
* READ ONLY. No broker mutation, no engine state mutation. Modifiers
  are written to JSON for downstream consumption only.
* Every modifier strictly clamped to MAX_ABS_MODIFIER (default 0.10).
* Atomic writes (.tmp + os.replace) for both outputs.
* Missing inputs warn-and-continue; the engine always produces a
  defensible output even when only a subset of upstream files exist.
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_MEMORY_INSIGHTS = RESULTS_DIR / "portfolio_memory_insights.json"
DEFAULT_LEARNING_ADJ = RESULTS_DIR / "portfolio_learning_adjustments.json"
DEFAULT_COMMITTEE = RESULTS_DIR / "investment_committee_summary.json"
DEFAULT_REGIME = RESULTS_DIR / "adaptive_regime.json"
DEFAULT_POLICY = RESULTS_DIR / "adaptive_policy.json"
DEFAULT_EXEC_SUMMARY = RESULTS_DIR / "portfolio_execution_summary.json"
DEFAULT_TRADE_OUTCOMES = RESULTS_DIR / "trade_outcomes.csv"

DEFAULT_OUT_INTELLIGENCE = RESULTS_DIR / "meta_decision_intelligence.json"
DEFAULT_OUT_RUNTIME = RESULTS_DIR / "meta_runtime_adjustments.json"

# -----------------------------------------------------------
# Tunables
# -----------------------------------------------------------
MAX_ABS_MODIFIER = 0.10  # hard cap per spec section 5
MIN_MEMORY_FOR_TRUST = 5  # below this, recent_hit_rate is "unknown" and weight redistributes

# Component weights for self_confidence_score. Sum is normalised.
SCORE_WEIGHTS: Dict[str, float] = {
    "recent_hit_rate": 0.22,
    "deployment_success_rate": 0.18,
    "portfolio_health_score": 0.15,
    "governance_score": 0.12,
    "regime_stability_score": 0.12,
    "learning_consistency_score": 0.11,
    "contradiction_health": 0.10,  # = 1 - contradiction_rate
}

# Per-field modifier scaling. Sign in {+1, -1}:
#   +1  -> high trust pushes positive  (deploy more / execute more / aggressive)
#   -1  -> high trust pushes negative  (lower the bar / hold less cash)
MODIFIER_SPEC: Dict[str, Tuple[float, int]] = {
    # field                     (max_abs, high_trust_sign)
    "confidence_modifier": (0.05, -1),
    "persistence_modifier": (0.04, -1),
    "deployment_modifier": (0.05, +1),
    "execution_modifier": (0.10, +1),
    "cash_modifier": (0.10, -1),
    "aggressiveness_modifier": (0.06, +1),
}

# Regime stability priors. Used only when adaptive_regime.json does
# not expose a direct stability signal. Higher = more stable.
REGIME_STABILITY_PRIOR: Dict[str, float] = {
    "NEUTRAL": 0.80,
    "OPPORTUNISTIC": 0.75,
    "MOMENTUM": 0.65,
    "AGGRESSIVE": 0.55,
    "DEFENSIVE": 0.60,
    "ROTATION": 0.50,
    "HIGH_VOLATILITY": 0.30,
    "RISK_OFF": 0.40,
    "UNKNOWN": 0.50,
}

# Trust level thresholds on self_confidence_score (0-1).
TRUST_LEVELS: Tuple[Tuple[float, str], ...] = (
    (0.80, "VERY_HIGH"),
    (0.65, "HIGH"),
    (0.45, "MODERATE"),
    (0.30, "LOW"),
    (0.00, "VERY_LOW"),
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[META_DECISION_WARN] {msg}", flush=True)


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
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm01(x: Optional[float], *, default: float = 0.50) -> float:
    """Coerce to a [0, 1] number; out-of-range or None -> default."""
    v = _to_float(x)
    if v is None:
        return default
    return _clamp(v, 0.0, 1.0)


def _trust_level(score: float) -> str:
    for threshold, label in TRUST_LEVELS:
        if score >= threshold:
            return label
    return "VERY_LOW"


# -----------------------------------------------------------
# Metric extraction (every extractor returns
# (value, is_known, source_note))
# -----------------------------------------------------------
Metric = Tuple[float, bool, str]


def _extract_recent_hit_rate(
    memory_insights: Dict[str, Any],
    trade_outcomes: pd.DataFrame,
) -> Metric:
    """Prefer Step 12 memory; fall back to closed trade_outcomes."""
    n_labelled = int(memory_insights.get("memory_size_with_outcome") or 0)
    if n_labelled >= MIN_MEMORY_FOR_TRUST:
        sr = _to_float(memory_insights.get("overall_success_rate"))
        if sr is not None:
            return _clamp(sr, 0.0, 1.0), True, f"memory_insights n={n_labelled}"
    if trade_outcomes is not None and not trade_outcomes.empty:
        df = trade_outcomes.copy()
        if "outcome_type" in df.columns:
            df = df[df["outcome_type"].astype(str).str.upper() == "REALIZED"]
        col = (
            "total_pl"
            if "total_pl" in df.columns
            else ("realized_pl" if "realized_pl" in df.columns else None)
        )
        if col is not None and len(df) >= MIN_MEMORY_FOR_TRUST:
            vals = [_to_float(v) for v in df[col].tolist()]
            vals = [v for v in vals if v is not None and v != 0.0]
            if len(vals) >= MIN_MEMORY_FOR_TRUST:
                wins = sum(1 for v in vals if v > 0)
                return (
                    wins / len(vals),
                    True,
                    f"trade_outcomes n={len(vals)}",
                )
    return 0.50, False, "insufficient_history"


def _extract_deployment_success_rate(memory_insights: Dict[str, Any]) -> Metric:
    pat = (memory_insights or {}).get("pattern_stats") or {}
    eb = pat.get("executed_buy") or {}
    n = int(eb.get("n_with_outcome") or 0)
    sr = _to_float(eb.get("success_rate"))
    if n >= MIN_MEMORY_FOR_TRUST and sr is not None:
        return _clamp(sr, 0.0, 1.0), True, f"executed_buy n={n}"
    # Fallback to overall
    overall = _to_float(memory_insights.get("overall_success_rate"))
    overall_n = int(memory_insights.get("memory_size_with_outcome") or 0)
    if overall is not None and overall_n >= MIN_MEMORY_FOR_TRUST:
        return _clamp(overall, 0.0, 1.0), True, f"overall_fallback n={overall_n}"
    return 0.50, False, "insufficient_history"


def _extract_portfolio_health(committee: Dict[str, Any]) -> Metric:
    scores = (committee or {}).get("scores") or {}
    v = _to_float(scores.get("portfolio_health_score"))
    if v is None:
        return 0.50, False, "missing"
    return _clamp(v, 0.0, 1.0), True, "committee.scores.portfolio_health_score"


def _extract_governance(committee: Dict[str, Any]) -> Metric:
    scores = (committee or {}).get("scores") or {}
    v = _to_float(scores.get("governance_score"))
    if v is None:
        return 0.50, False, "missing"
    return _clamp(v, 0.0, 1.0), True, "committee.scores.governance_score"


def _extract_regime_stability(
    regime_json: Dict[str, Any],
    policy_json: Dict[str, Any],
) -> Metric:
    # Prefer a direct stability/confidence signal if exposed by Step 10.
    candidate_keys = ("stability_score", "regime_stability_score", "confidence")
    src = regime_json or {}
    for k in candidate_keys:
        v = _to_float(src.get(k))
        if v is not None:
            return _clamp(v, 0.0, 1.0), True, f"adaptive_regime.{k}"
    src_scores = (regime_json or {}).get("source_scores") or {}
    for k in candidate_keys:
        v = _to_float(src_scores.get(k))
        if v is not None:
            return _clamp(v, 0.0, 1.0), True, f"adaptive_regime.source_scores.{k}"
    # Fall back to a regime-keyed prior.
    regime = str(
        (regime_json or {}).get("regime") or (policy_json or {}).get("regime") or "UNKNOWN"
    ).upper()
    prior = REGIME_STABILITY_PRIOR.get(regime, 0.50)
    return prior, False, f"regime_prior:{regime}"


def _extract_learning_consistency(learning_adj: Dict[str, Any]) -> Metric:
    """
    Consistency = 1 - dispersion of |raw_adjustment| across all emitted
    buckets. Low dispersion -> the learner is converging on a coherent
    pattern. High dispersion / many clipped extremes -> the learner is
    being whipsawed.
    """
    if not learning_adj:
        return 0.50, False, "missing"
    mags: List[float] = []
    for section in ("regime_adjustments", "pattern_adjustments", "ticker_adjustments"):
        for v in (learning_adj.get(section) or {}).values():
            raw = _to_float(v.get("raw_adjustment"))
            if raw is not None:
                mags.append(abs(raw))
    if len(mags) < 2:
        # Trivially consistent (or nothing learned yet).
        return 0.80 if mags else 0.50, bool(mags), f"n_buckets={len(mags)}"
    spread = statistics.pstdev(mags)
    # Normalise: spread of 0 -> 1.0, spread >= MAX_ABS_MODIFIER -> 0.0
    score = 1.0 - _clamp(spread / MAX_ABS_MODIFIER, 0.0, 1.0)
    return score, True, f"pstdev={spread:.4f} n={len(mags)}"


def _extract_contradiction_rate(
    exec_summary: Dict[str, Any],
    memory_insights: Dict[str, Any],
) -> Metric:
    """
    Contradiction = "Triton blocked or delayed a buy it considered".
    Numerator = blocked + delayed. Denominator = total buy-side actions.
    """
    es = exec_summary or {}
    execute_now = int(es.get("execute_now") or 0)
    blocked = int(es.get("blocked") or 0)
    delayed = int(es.get("delayed") or 0)
    denom = execute_now + blocked + delayed
    if denom > 0:
        rate = (blocked + delayed) / denom
        return (
            _clamp(rate, 0.0, 1.0),
            True,
            f"execution_summary blocked={blocked} delayed={delayed} executed={execute_now}",
        )
    # Fall back to memory pattern stats.
    pat = (memory_insights or {}).get("pattern_stats") or {}
    eb = int((pat.get("executed_buy") or {}).get("n_observations") or 0)
    bb = int((pat.get("blocked_buy") or {}).get("n_observations") or 0)
    db = int((pat.get("delayed_buy") or {}).get("n_observations") or 0)
    denom_mem = eb + bb + db
    if denom_mem >= MIN_MEMORY_FOR_TRUST:
        rate = (bb + db) / denom_mem
        return (
            _clamp(rate, 0.0, 1.0),
            True,
            f"memory blocked={bb} delayed={db} executed={eb}",
        )
    return 0.20, False, "insufficient_history"


# -----------------------------------------------------------
# Self confidence + rationale
# -----------------------------------------------------------
def _compute_self_confidence(metrics: Dict[str, float]) -> float:
    """Weighted blend; weights renormalise over the *known* components."""
    total_w = 0.0
    score = 0.0
    for key, w in SCORE_WEIGHTS.items():
        v = metrics.get(key)
        if v is None:
            continue
        score += w * _clamp(float(v), 0.0, 1.0)
        total_w += w
    if total_w <= 0.0:
        return 0.50
    return _clamp(score / total_w, 0.0, 1.0)


def _build_rationale(
    *,
    trust_level: str,
    self_confidence: float,
    metrics: Dict[str, float],
    contradiction_rate: float,
    modifiers: Dict[str, float],
    metric_known: Dict[str, bool],
) -> Tuple[str, str, List[str]]:
    tags: List[str] = []
    parts: List[str] = []

    hit = metrics.get("recent_hit_rate")
    if hit is not None and metric_known.get("recent_hit_rate"):
        if hit < 0.45:
            tags.append("weak_hit_rate")
            parts.append(f"weak recent hit-rate ({hit:.2f})")
        elif hit >= 0.65:
            tags.append("strong_hit_rate")
            parts.append(f"strong recent hit-rate ({hit:.2f})")

    ds = metrics.get("deployment_success_rate")
    if ds is not None and metric_known.get("deployment_success_rate"):
        if ds < 0.45:
            tags.append("weak_deployment_success")
            parts.append(f"weak deployment success ({ds:.2f})")
        elif ds >= 0.65:
            tags.append("strong_deployment_success")
            parts.append(f"strong deployment success ({ds:.2f})")

    ph = metrics.get("portfolio_health_score")
    if ph is not None and metric_known.get("portfolio_health_score"):
        if ph < 0.45:
            tags.append("declining_portfolio_health")
            parts.append(f"declining portfolio health ({ph:.2f})")
        elif ph >= 0.70:
            tags.append("strong_portfolio_health")

    gov = metrics.get("governance_score")
    if gov is not None and metric_known.get("governance_score"):
        if gov < 0.45:
            tags.append("weak_governance")
            parts.append(f"weak governance ({gov:.2f})")

    rs = metrics.get("regime_stability_score")
    if rs is not None:
        if rs < 0.45:
            tags.append("unstable_regime")
            parts.append(f"unstable regime ({rs:.2f})")
        elif rs >= 0.70:
            tags.append("stable_regime")

    lc = metrics.get("learning_consistency_score")
    if lc is not None and metric_known.get("learning_consistency_score"):
        if lc < 0.45:
            tags.append("inconsistent_learning")
            parts.append(f"inconsistent learning signal ({lc:.2f})")
        elif lc >= 0.70:
            tags.append("consistent_learning")

    if contradiction_rate >= 0.50:
        tags.append("high_contradiction_frequency")
        parts.append(f"high contradiction frequency ({contradiction_rate:.2f})")
    elif contradiction_rate >= 0.30:
        tags.append("elevated_contradiction_frequency")
        parts.append(f"elevated contradiction frequency ({contradiction_rate:.2f})")

    if not parts:
        parts.append("balanced inputs across all measured dimensions")

    direction = {
        "VERY_HIGH": "expanded deployment latitude",
        "HIGH": "modestly expanded deployment latitude",
        "MODERATE": "kept policy thresholds neutral",
        "LOW": "tightened deployment thresholds and raised cash reserve",
        "VERY_LOW": "sharply tightened deployment thresholds and raised cash reserve",
    }.get(trust_level, "kept policy thresholds neutral")

    rationale_short = f"Self-confidence={self_confidence:.2f} ({trust_level}); {direction}."
    rationale_long = (
        f"Triton {direction} because self-confidence is {self_confidence:.2f} "
        f"({trust_level}). Contributing factors: " + ", ".join(parts) + ". "
        f"Runtime modifiers issued -- "
        f"confidence={modifiers['confidence_modifier']:+.3f}, "
        f"persistence={modifiers['persistence_modifier']:+.3f}, "
        f"deployment={modifiers['deployment_modifier']:+.3f}, "
        f"execution={modifiers['execution_modifier']:+.3f}, "
        f"cash={modifiers['cash_modifier']:+.3f}, "
        f"aggressiveness={modifiers['aggressiveness_modifier']:+.3f}."
    )
    return rationale_short, rationale_long, tags


# -----------------------------------------------------------
# Modifier generation
# -----------------------------------------------------------
def _build_modifiers(self_confidence: float) -> Dict[str, float]:
    """
    All modifiers scale linearly with trust deviation from 0.5.

    trust_delta = (self_confidence - 0.5) * 2     # range [-1, +1]

    For each field:
        modifier = max_abs_field * trust_delta * high_trust_sign

    Then clamped at +/-MAX_ABS_MODIFIER (the global hard cap from spec section 5).
    """
    trust_delta = _clamp((self_confidence - 0.5) * 2.0, -1.0, 1.0)
    out: Dict[str, float] = {}
    for field, (max_abs_field, sign) in MODIFIER_SPEC.items():
        raw = max_abs_field * trust_delta * sign
        out[field] = round(_clamp(raw, -MAX_ABS_MODIFIER, MAX_ABS_MODIFIER), 4)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_meta_artefacts(
    *,
    memory_insights: Dict[str, Any],
    learning_adj: Dict[str, Any],
    committee: Dict[str, Any],
    regime_json: Dict[str, Any],
    policy_json: Dict[str, Any],
    exec_summary: Dict[str, Any],
    trade_outcomes: pd.DataFrame,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    now_iso = _now_iso_utc()

    hit, hit_known, hit_src = _extract_recent_hit_rate(memory_insights, trade_outcomes)
    dep, dep_known, dep_src = _extract_deployment_success_rate(memory_insights)
    ph, ph_known, ph_src = _extract_portfolio_health(committee)
    gov, gov_known, gov_src = _extract_governance(committee)
    rs, rs_known, rs_src = _extract_regime_stability(regime_json, policy_json)
    lc, lc_known, lc_src = _extract_learning_consistency(learning_adj)
    cr, cr_known, cr_src = _extract_contradiction_rate(exec_summary, memory_insights)

    metrics: Dict[str, float] = {
        "recent_hit_rate": hit,
        "deployment_success_rate": dep,
        "portfolio_health_score": ph,
        "governance_score": gov,
        "regime_stability_score": rs,
        "learning_consistency_score": lc,
        "contradiction_health": 1.0 - cr,  # blend input -- inverted
    }
    metric_known = {
        "recent_hit_rate": hit_known,
        "deployment_success_rate": dep_known,
        "portfolio_health_score": ph_known,
        "governance_score": gov_known,
        "regime_stability_score": rs_known,
        "learning_consistency_score": lc_known,
        "contradiction_health": cr_known,
    }
    sources = {
        "recent_hit_rate": hit_src,
        "deployment_success_rate": dep_src,
        "portfolio_health_score": ph_src,
        "governance_score": gov_src,
        "regime_stability_score": rs_src,
        "learning_consistency_score": lc_src,
        "contradiction_rate": cr_src,
    }

    self_confidence = _compute_self_confidence(metrics)
    trust_level = _trust_level(self_confidence)
    modifiers = _build_modifiers(self_confidence)

    short, long_, tags = _build_rationale(
        trust_level=trust_level,
        self_confidence=self_confidence,
        metrics=metrics,
        contradiction_rate=cr,
        modifiers=modifiers,
        metric_known=metric_known,
    )

    intelligence: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "meta_decision_engine",
        "engine_version": 1,
        "regime": str(
            (regime_json or {}).get("regime") or (policy_json or {}).get("regime") or "UNKNOWN"
        ).upper(),
        "trust_level": trust_level,
        "self_confidence_score": round(self_confidence, 6),
        "scores": {
            "recent_hit_rate": round(hit, 6),
            "regime_stability_score": round(rs, 6),
            "learning_consistency_score": round(lc, 6),
            "contradiction_rate": round(cr, 6),
            "deployment_success_rate": round(dep, 6),
            "portfolio_health_score": round(ph, 6),
            "governance_score": round(gov, 6),
        },
        "scores_known": metric_known,
        "score_weights": dict(SCORE_WEIGHTS),
        "sources": sources,
        "rationale_short": short,
        "rationale_long": long_,
        "rationale_tags": tags,
        "modifiers": modifiers,
        "thresholds": {
            "max_abs_modifier": MAX_ABS_MODIFIER,
            "min_memory_for_trust": MIN_MEMORY_FOR_TRUST,
            "trust_level_bands": [{"min_score": t, "label": lbl} for t, lbl in TRUST_LEVELS],
        },
        "inputs_seen": {
            "memory_insights": bool(memory_insights),
            "learning_adjustments": bool(learning_adj),
            "investment_committee": bool(committee),
            "adaptive_regime": bool(regime_json),
            "adaptive_policy": bool(policy_json),
            "execution_summary": bool(exec_summary),
            "trade_outcomes": (trade_outcomes is not None and not trade_outcomes.empty),
        },
    }

    runtime: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "meta_decision_engine",
        "engine_version": 1,
        "regime": intelligence["regime"],
        "trust_level": trust_level,
        "self_confidence_score": round(self_confidence, 6),
        "policy_bounds": {
            "max_abs_modifier": MAX_ABS_MODIFIER,
        },
        "modifiers": modifiers,
        "rationale_short": short,
        "rationale_tags": tags,
        "field_caps": {field: round(MODIFIER_SPEC[field][0], 4) for field in MODIFIER_SPEC},
        "field_signs": {
            field: ("HIGH_TRUST_POSITIVE" if sign > 0 else "HIGH_TRUST_NEGATIVE")
            for field, (_, sign) in MODIFIER_SPEC.items()
        },
    }
    return intelligence, runtime


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only meta decision intelligence engine (Step 13). "
            "Reads Steps 7/9/10/12 outputs and emits a self-confidence "
            "score, a categorical trust level, and a bundle of bounded "
            "runtime modifiers downstream policy layers can apply."
        ),
    )
    p.add_argument("--memory-insights", default=str(DEFAULT_MEMORY_INSIGHTS))
    p.add_argument("--learning-adjustments", default=str(DEFAULT_LEARNING_ADJ))
    p.add_argument("--committee", default=str(DEFAULT_COMMITTEE))
    p.add_argument("--regime", default=str(DEFAULT_REGIME))
    p.add_argument("--policy", default=str(DEFAULT_POLICY))
    p.add_argument("--execution-summary", default=str(DEFAULT_EXEC_SUMMARY))
    p.add_argument("--trade-outcomes", default=str(DEFAULT_TRADE_OUTCOMES))
    p.add_argument("--out-intelligence", default=str(DEFAULT_OUT_INTELLIGENCE))
    p.add_argument("--out-runtime", default=str(DEFAULT_OUT_RUNTIME))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[META_DECISION] starting (read-only introspection layer)", flush=True)

    memory_insights = _safe_read_json(
        Path(args.memory_insights), label="portfolio_memory_insights.json"
    )
    learning_adj = _safe_read_json(
        Path(args.learning_adjustments), label="portfolio_learning_adjustments.json"
    )
    committee = _safe_read_json(Path(args.committee), label="investment_committee_summary.json")
    regime_json = _safe_read_json(Path(args.regime), label="adaptive_regime.json")
    policy_json = _safe_read_json(Path(args.policy), label="adaptive_policy.json")
    exec_summary = _safe_read_json(
        Path(args.execution_summary), label="portfolio_execution_summary.json"
    )
    trade_outcomes = _safe_read_csv(Path(args.trade_outcomes), label="trade_outcomes.csv")

    intelligence, runtime = build_meta_artefacts(
        memory_insights=memory_insights,
        learning_adj=learning_adj,
        committee=committee,
        regime_json=regime_json,
        policy_json=policy_json,
        exec_summary=exec_summary,
        trade_outcomes=trade_outcomes,
    )

    try:
        _atomic_write_json(intelligence, Path(args.out_intelligence))
    except Exception as e:
        _warn(f"failed to write {args.out_intelligence}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(runtime, Path(args.out_runtime))
    except Exception as e:
        _warn(f"failed to write {args.out_runtime}: {type(e).__name__}: {e}")
        return 2

    print(
        "[META_DECISION] "
        f"trust={intelligence['trust_level']} "
        f"confidence={intelligence['self_confidence_score']:.3f} "
        f"hit_rate={intelligence['scores']['recent_hit_rate']:.3f} "
        f"contradictions={intelligence['scores']['contradiction_rate']:.3f}",
        flush=True,
    )
    print(
        "[META_RUNTIME] "
        f"cash_modifier={runtime['modifiers']['cash_modifier']:+.3f} "
        f"execution_modifier={runtime['modifiers']['execution_modifier']:+.3f} "
        f"aggressiveness_modifier={runtime['modifiers']['aggressiveness_modifier']:+.3f}",
        flush=True,
    )
    print(
        f"[META_DECISION_OUT] intelligence={Path(args.out_intelligence).as_posix()} "
        f"runtime={Path(args.out_runtime).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
