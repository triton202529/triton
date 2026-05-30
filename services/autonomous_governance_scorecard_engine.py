"""
Autonomous Governance Scorecard Engine -- Step 19.

Reads:
    data/results/autonomous_strategy_diagnostics.json     (Step 16)
    data/results/meta_decision_intelligence.json          (Step 13)
    data/results/governance_trust_feedback.json           (Step 17)
    data/results/runtime_policy_governed.json             (Step 18)
    data/results/investment_committee_summary.json        (Step 9)
    data/results/portfolio_memory_insights.json           (Step 12)
    data/results/adaptive_regime.json                     (Step 10)
    data/results/autonomous_committee_summary.json        (Step 15)

Writes:
    data/results/autonomous_governance_scorecard.json
    data/results/autonomous_governance_scorecard.md
    data/results/autonomous_governance_summary.json

Purpose
-------
Steps 1-18 built Triton's autonomous intelligence stack one engine
at a time. Step 19 stitches their outputs back together into one
read-only portfolio-manager-level scorecard:

    "How healthy is Triton's decision system?"

The scorecard is a *reporting* layer -- it never modifies policy,
never places trades, and never mutates engine state. Its job is
to surface the dominant patterns across the eight category scores
(spec section 1), classify the system into one or more state
labels (spec section 2), and produce a narrative + recommendations
that a human PM can read in 30 seconds.

Category scores (spec section 1)
--------------------------------
Each score is normalised to [0, 1] and tagged with ``is_known``
(False when the underlying inputs lack enough labelled history to
make the number meaningful, in which case the value defaults to
0.50 -- the universal "I don't know yet" marker that matches Steps
16/17).

    intelligence_health_score    Step 13 self_confidence + Step 17 gov_health
                                  + Step 16 decision_quality
    governance_quality_score     Step 16 governance_quality_score (Step 17 fallback)
    trust_quality_score          Step 16 trust_quality + Step 13 trust_level numeric
    deployment_discipline_score  Step 16 deployment_accuracy + Step 9 readiness
    capital_preservation_score   Step 16 alpha_preservation + drawdown_avoidance
    learning_maturity_score      Step 12 observation counts (saturates ~30 samples)
    portfolio_discipline_score   Step 9 portfolio_health + diversification + conviction
    regime_quality_score         Step 16 regime_prediction + regime stability proxy

System state (spec section 2)
-----------------------------
The eight-label state set is *multi-label* -- a healthy system
might be tagged ``[STABLE, GOVERNANCE_STRONG]`` while a fresh
install is ``[EARLY_LEARNING, DEFENSIVE]``. Defensive states
override aggressive ones when both could trigger.

Safety
------
* READ ONLY. No broker calls, no engine state mutation.
* Atomic writes (.tmp + os.replace) for all three outputs.
* Missing inputs warn-and-continue. With zero inputs the scorecard
  still produces a valid blob with all eight scores at the
  unknown-default of 0.50 and the labels set to ``[EARLY_LEARNING]``.
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
DEFAULT_META_INTEL = RESULTS_DIR / "meta_decision_intelligence.json"
DEFAULT_GOV_FEEDBACK = RESULTS_DIR / "governance_trust_feedback.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_IC_SUMMARY = RESULTS_DIR / "investment_committee_summary.json"
DEFAULT_MEMORY_INSIGHTS = RESULTS_DIR / "portfolio_memory_insights.json"
DEFAULT_REGIME = RESULTS_DIR / "adaptive_regime.json"
DEFAULT_COMMITTEE_SUMMARY = RESULTS_DIR / "autonomous_committee_summary.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_OUT_MD = RESULTS_DIR / "autonomous_governance_scorecard.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "autonomous_governance_summary.json"


# -----------------------------------------------------------
# Tunables
# -----------------------------------------------------------
UNKNOWN_DEFAULT = 0.50
LEARNING_SATURATION_OBS = 30  # observations beyond which learning_maturity = 1.0
LEARNING_LABELLED_FLOOR = 5  # mirrors Step 16 MIN_SAMPLE_SIZE

# Trust-level -> numeric for blending into trust_quality_score
TRUST_LEVEL_NUMERIC: Dict[str, float] = {
    "VERY_LOW": 0.10,
    "LOW": 0.30,
    "MODERATE": 0.55,
    "HIGH": 0.78,
    "VERY_HIGH": 0.92,
}

# Score bands for labelling individual category scores in markdown
SCORE_BANDS: Tuple[Tuple[float, str], ...] = (
    (0.80, "STRONG"),
    (0.60, "HEALTHY"),
    (0.45, "STABLE"),
    (0.30, "WEAK"),
    (0.00, "POOR"),
)

# Regimes considered defensive for state labelling
DEFENSIVE_REGIMES = {"DEFENSIVE", "RISK_OFF", "HIGH_VOLATILITY", "CAPITAL_PRESERVATION"}
AGGRESSIVE_REGIMES = {"AGGRESSIVE", "OPPORTUNISTIC", "MOMENTUM"}

# Committee decisions that imply defensive vs aggressive postures
DEFENSIVE_COMMITTEE_DECISIONS = {"CAPITAL_PRESERVATION", "DEFENSIVE_ROTATION", "DELEVER"}
AGGRESSIVE_COMMITTEE_DECISIONS = {"DEPLOY_AGGRESSIVELY", "DEPLOY_SELECTIVELY"}


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[GOVERNANCE_SCORECARD_WARN] {msg}", flush=True)


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


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
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


def _to_int(x: Any) -> int:
    v = _to_float(x)
    if v is None:
        return 0
    try:
        return int(v)
    except Exception:
        return 0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm01(x: Any, default: float = UNKNOWN_DEFAULT) -> float:
    v = _to_float(x)
    if v is None:
        return default
    return _clamp(v, 0.0, 1.0)


def _band(score: float) -> str:
    for threshold, label in SCORE_BANDS:
        if score >= threshold:
            return label
    return "POOR"


# -----------------------------------------------------------
# Category-score builders
# -----------------------------------------------------------
def _blend_known(values: List[Tuple[float, bool, float]]) -> Tuple[float, bool]:
    """
    Weighted blend of (value, is_known, weight) tuples. Renormalises
    over only the known contributors. Returns (score, is_known) where
    is_known is True iff at least one contributor was known.
    """
    total_w = 0.0
    weighted = 0.0
    any_known = False
    for v, k, w in values:
        if not k or w <= 0.0:
            continue
        weighted += w * _clamp(v, 0.0, 1.0)
        total_w += w
        any_known = True
    if total_w <= 0.0 or not any_known:
        return UNKNOWN_DEFAULT, False
    return _clamp(weighted / total_w, 0.0, 1.0), True


def _extract_diag_scores(diagnostics: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, bool]]:
    raw_scores = diagnostics.get("scores") or {}
    raw_known = diagnostics.get("scores_known") or {}
    keys = (
        "alpha_preservation_score",
        "drawdown_avoidance_score",
        "deployment_accuracy_score",
        "governance_quality_score",
        "regime_prediction_score",
        "trust_quality_score",
    )
    scores = {k: _norm01(raw_scores.get(k)) for k in keys}
    known = {k: bool(raw_known.get(k, False)) for k in keys}
    return scores, known


def _intelligence_health_score(
    *,
    meta_intel: Dict[str, Any],
    feedback: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> Tuple[float, bool]:
    meta_conf = _to_float((meta_intel or {}).get("self_confidence_score"))
    gov_health = _to_float((feedback or {}).get("governance_health_score"))
    decision_q = _to_float((diagnostics or {}).get("decision_quality_score"))
    contributors = [
        # (value, is_known, weight)
        (_clamp(meta_conf or UNKNOWN_DEFAULT, 0.0, 1.0), meta_conf is not None, 0.40),
        (
            _clamp(gov_health or UNKNOWN_DEFAULT, 0.0, 1.0),
            bool((feedback or {}).get("active", False)),
            0.35,
        ),
        (
            _clamp(decision_q or UNKNOWN_DEFAULT, 0.0, 1.0),
            decision_q is not None and decision_q != UNKNOWN_DEFAULT,
            0.25,
        ),
    ]
    return _blend_known(contributors)


def _governance_quality_score(
    *,
    diag_scores: Dict[str, float],
    diag_known: Dict[str, bool],
    feedback: Dict[str, Any],
) -> Tuple[float, bool]:
    gov_diag = diag_scores.get("governance_quality_score", UNKNOWN_DEFAULT)
    gov_known = diag_known.get("governance_quality_score", False)
    gov_health_fb = _to_float((feedback or {}).get("governance_health_score"))
    fb_known = bool((feedback or {}).get("active", False)) and gov_health_fb is not None
    contributors = [
        (gov_diag, gov_known, 0.70),
        (gov_health_fb if gov_health_fb is not None else UNKNOWN_DEFAULT, fb_known, 0.30),
    ]
    return _blend_known(contributors)


def _trust_quality_score(
    *,
    diag_scores: Dict[str, float],
    diag_known: Dict[str, bool],
    meta_intel: Dict[str, Any],
) -> Tuple[float, bool]:
    diag_tq = diag_scores.get("trust_quality_score", UNKNOWN_DEFAULT)
    diag_known_tq = diag_known.get("trust_quality_score", False)
    meta_level = str((meta_intel or {}).get("trust_level") or "MODERATE").strip().upper()
    meta_numeric = TRUST_LEVEL_NUMERIC.get(meta_level, UNKNOWN_DEFAULT)
    meta_known = bool(meta_intel)
    contributors = [
        (diag_tq, diag_known_tq, 0.65),
        (meta_numeric, meta_known, 0.35),
    ]
    return _blend_known(contributors)


def _deployment_discipline_score(
    *,
    diag_scores: Dict[str, float],
    diag_known: Dict[str, bool],
    ic_summary: Dict[str, Any],
) -> Tuple[float, bool]:
    diag_dep = diag_scores.get("deployment_accuracy_score", UNKNOWN_DEFAULT)
    diag_known_dep = diag_known.get("deployment_accuracy_score", False)
    readiness = _to_float((ic_summary or {}).get("deployment_readiness_score"))
    readiness_known = readiness is not None
    contributors = [
        (diag_dep, diag_known_dep, 0.65),
        (_clamp(readiness or UNKNOWN_DEFAULT, 0.0, 1.0), readiness_known, 0.35),
    ]
    return _blend_known(contributors)


def _capital_preservation_score(
    *,
    diag_scores: Dict[str, float],
    diag_known: Dict[str, bool],
) -> Tuple[float, bool]:
    alpha = diag_scores.get("alpha_preservation_score", UNKNOWN_DEFAULT)
    drawdown = diag_scores.get("drawdown_avoidance_score", UNKNOWN_DEFAULT)
    alpha_k = diag_known.get("alpha_preservation_score", False)
    dd_k = diag_known.get("drawdown_avoidance_score", False)
    contributors = [
        (alpha, alpha_k, 0.40),
        (drawdown, dd_k, 0.60),
    ]
    return _blend_known(contributors)


def _learning_maturity_score(
    *,
    diagnostics: Dict[str, Any],
    memory_insights: Dict[str, Any],
) -> Tuple[float, bool, int, int]:
    total_obs = max(
        _to_int((diagnostics or {}).get("memory_size_total")),
        _to_int((memory_insights or {}).get("total_observations")),
    )
    labelled_obs = max(
        _to_int((diagnostics or {}).get("memory_size_with_outcome")),
        _to_int((memory_insights or {}).get("labelled_observations")),
    )
    # Maturity = min(labelled / saturation, 1.0) with a small bonus
    # for total volume even if not labelled (early signs of activity).
    labelled_share = _clamp(labelled_obs / float(LEARNING_SATURATION_OBS), 0.0, 1.0)
    activity_share = _clamp(total_obs / float(LEARNING_SATURATION_OBS * 2), 0.0, 1.0)
    score = _clamp(0.80 * labelled_share + 0.20 * activity_share, 0.0, 1.0)
    is_known = total_obs > 0
    return score, is_known, total_obs, labelled_obs


def _portfolio_discipline_score(ic_summary: Dict[str, Any]) -> Tuple[float, bool]:
    health = _to_float((ic_summary or {}).get("portfolio_health_score"))
    diversification = _to_float((ic_summary or {}).get("diversification_score"))
    conviction = _to_float((ic_summary or {}).get("conviction_score"))
    contributors = [
        (_clamp(health or UNKNOWN_DEFAULT, 0.0, 1.0), health is not None, 0.45),
        (_clamp(diversification or UNKNOWN_DEFAULT, 0.0, 1.0), diversification is not None, 0.30),
        (_clamp(conviction or UNKNOWN_DEFAULT, 0.0, 1.0), conviction is not None, 0.25),
    ]
    return _blend_known(contributors)


def _regime_quality_score(
    *,
    diag_scores: Dict[str, float],
    diag_known: Dict[str, bool],
    regime_json: Dict[str, Any],
) -> Tuple[float, bool]:
    diag_rq = diag_scores.get("regime_prediction_score", UNKNOWN_DEFAULT)
    diag_known_rq = diag_known.get("regime_prediction_score", False)
    # Use confidence_score on the regime classification itself, if exposed,
    # as a secondary proxy for "how sure was the classifier".
    regime_conf = _to_float((regime_json or {}).get("confidence_score"))
    regime_known = regime_conf is not None
    contributors = [
        (diag_rq, diag_known_rq, 0.70),
        (_clamp(regime_conf or UNKNOWN_DEFAULT, 0.0, 1.0), regime_known, 0.30),
    ]
    return _blend_known(contributors)


# -----------------------------------------------------------
# System state labelling
# -----------------------------------------------------------
def _classify_system_state(
    *,
    scores: Dict[str, float],
    known: Dict[str, bool],
    total_obs: int,
    labelled_obs: int,
    regime: str,
    committee_decision: str,
    feedback: Dict[str, Any],
) -> List[str]:
    labels: List[str] = []

    # 1. EARLY_LEARNING: not enough samples to trust scores.
    if scores["learning_maturity_score"] < 0.40 or labelled_obs < (
        LEARNING_LABELLED_FLOOR * 2
    ):  # need 10+ labelled before maturity
        labels.append("EARLY_LEARNING")

    # 2. DEFENSIVE / AGGRESSIVE (mutually exclusive; defensive wins).
    is_defensive = (
        regime in DEFENSIVE_REGIMES or committee_decision in DEFENSIVE_COMMITTEE_DECISIONS
    )
    is_aggressive = (
        regime in AGGRESSIVE_REGIMES or committee_decision in AGGRESSIVE_COMMITTEE_DECISIONS
    )
    if is_defensive:
        labels.append("DEFENSIVE")
    elif is_aggressive:
        labels.append("AGGRESSIVE")

    # 3. CAPITAL_PRESERVATION: dominant capital-preservation behaviour.
    if scores["capital_preservation_score"] >= 0.65 or committee_decision == "CAPITAL_PRESERVATION":
        labels.append("CAPITAL_PRESERVATION")

    # 4. GOVERNANCE_WEAK / GOVERNANCE_STRONG (only when measurable).
    if known["governance_quality_score"]:
        if scores["governance_quality_score"] < 0.40:
            labels.append("GOVERNANCE_WEAK")
        elif scores["governance_quality_score"] > 0.65:
            labels.append("GOVERNANCE_STRONG")

    # 5. SELF_CORRECTING: Step 17 is active AND emitting non-trivial deltas
    #    that move thresholds toward the right direction.
    if bool((feedback or {}).get("active", False)):
        deltas = (feedback or {}).get("deltas") or {}
        any_nonzero = any(
            abs(_to_float(deltas.get(k)) or 0.0) >= 0.005
            for k in (
                "trust_delta",
                "confidence_delta",
                "deployment_delta",
                "aggressiveness_delta",
                "cash_delta",
                "skepticism_delta",
            )
        )
        if any_nonzero:
            labels.append("SELF_CORRECTING")

    # 6. STABLE: intelligence health in healthy band, no early-learning,
    #    no governance weakness, no defensive override.
    if (
        "EARLY_LEARNING" not in labels
        and "GOVERNANCE_WEAK" not in labels
        and "DEFENSIVE" not in labels
        and 0.45 <= scores["intelligence_health_score"] <= 0.72
    ):
        labels.append("STABLE")

    if not labels:
        labels.append("EARLY_LEARNING")  # safe default
    return labels


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
def _build_recommendations(
    *,
    scores: Dict[str, float],
    known: Dict[str, bool],
    labels: List[str],
    total_obs: int,
    labelled_obs: int,
    feedback: Dict[str, Any],
    governed_policy: Dict[str, Any],
) -> List[str]:
    recs: List[str] = []

    if "EARLY_LEARNING" in labels:
        recs.append(
            f"Continue conservative deployment -- only {labelled_obs} labelled "
            f"observation(s) on record; governance feedback remains dormant."
        )
        recs.append(
            f"Increase sample size before trusting governance "
            f"(target >= {LEARNING_SATURATION_OBS} labelled cycles for full maturity)."
        )

    if known["governance_quality_score"] and scores["governance_quality_score"] < 0.40:
        recs.append(
            "Investigate governance weakness -- governance_quality_score "
            f"{scores['governance_quality_score']:.2f} is below the 0.40 floor."
        )

    if known["trust_quality_score"] and scores["trust_quality_score"] < 0.45:
        recs.append(
            "Monitor trust calibration -- meta-trust signal is not reliably "
            "correlated with realised outcomes."
        )

    if known["capital_preservation_score"] and scores["capital_preservation_score"] < 0.45:
        recs.append(
            "Tighten capital preservation -- drawdown avoidance and alpha "
            "preservation are below stable thresholds."
        )

    if known["deployment_discipline_score"] and scores["deployment_discipline_score"] < 0.45:
        recs.append(
            "Reduce deployment aggressiveness -- recent deploys are under-"
            "performing on a forward-return basis."
        )

    if "DEFENSIVE" in labels and "GOVERNANCE_STRONG" not in labels:
        recs.append(
            "Maintain elevated cash reserves and minimum-position discipline "
            "while regime stays defensive."
        )

    if scores["portfolio_discipline_score"] < 0.45:
        recs.append(
            "Reduce concentration risk and improve diversification -- "
            f"portfolio_discipline_score {scores['portfolio_discipline_score']:.2f} weak."
        )

    if "GOVERNANCE_STRONG" in labels and "EARLY_LEARNING" not in labels:
        recs.append(
            "Governance has earned the right to widen autonomy -- consider "
            "letting the governance overlay drive deployment thresholds."
        )

    if "SELF_CORRECTING" in labels:
        gd = (feedback or {}).get("deltas") or {}
        trust_d = _to_float(gd.get("trust_delta")) or 0.0
        direction = "raising" if trust_d > 0 else "lowering"
        recs.append(
            f"Continue monitoring -- governance overlay is actively {direction} "
            f"self-trust (trust_delta={trust_d:+.3f})."
        )

    # Cash threshold check from the governed policy: ensure it sits within
    # a reasonable institutional band.
    cash = _to_float((governed_policy or {}).get("target_cash_pct"))
    if cash is not None and cash >= 30.0 and "DEFENSIVE" not in labels:
        recs.append(
            f"Cash reserve elevated ({cash:.1f}%) -- review whether defensive "
            "posture is still warranted given regime."
        )

    if not recs:
        recs.append(
            "No urgent actions -- continue current cycle cadence and gather "
            "more labelled outcomes."
        )
    return recs


# -----------------------------------------------------------
# Narrative
# -----------------------------------------------------------
def _build_narrative(
    *,
    scores: Dict[str, float],
    known: Dict[str, bool],
    labels: List[str],
    regime: str,
    committee_decision: str,
    governance_trust_level: str,
) -> str:
    primary_state = labels[0] if labels else "EARLY_LEARNING"

    def describe(band: str) -> str:
        return {
            "STRONG": "strong",
            "HEALTHY": "healthy",
            "STABLE": "stable",
            "WEAK": "weak",
            "POOR": "poor",
        }.get(band, "neutral")

    health_phrase = describe(_band(scores["intelligence_health_score"]))
    cap_phrase = (
        describe(_band(scores["capital_preservation_score"]))
        if known["capital_preservation_score"]
        else "early-stage"
    )
    gov_phrase = (
        describe(_band(scores["governance_quality_score"]))
        if known["governance_quality_score"]
        else "not-yet-measurable"
    )
    deploy_phrase = (
        describe(_band(scores["deployment_discipline_score"]))
        if known["deployment_discipline_score"]
        else "untested"
    )

    descriptor = ", ".join(labels)
    intelligence_pct = scores["intelligence_health_score"]

    sentence_1 = (
        f"Triton sits in {primary_state} mode "
        f"(state={descriptor}) with {health_phrase} intelligence health "
        f"({intelligence_pct:.2f}), {cap_phrase} capital preservation, "
        f"{gov_phrase} governance quality, and {deploy_phrase} deployment discipline."
    )
    sentence_2 = (
        f"Current regime is {regime}, committee verdict is "
        f"{committee_decision or 'UNKNOWN'}, governance trust level is "
        f"{governance_trust_level}."
    )
    return sentence_1 + " " + sentence_2


# -----------------------------------------------------------
# Markdown report
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    scores: Dict[str, float],
    known: Dict[str, bool],
    labels: List[str],
    narrative: str,
    recommendations: List[str],
    regime: str,
    committee_decision: str,
    governance_trust_level: str,
    total_obs: int,
    labelled_obs: int,
) -> str:
    def _row(key: str, label: str) -> str:
        v = scores[key]
        k = known[key]
        return f"- **{label}**: {v:.3f} ({_band(v)}){'' if k else ' [unknown -- default]'}"

    lines: List[str] = []
    lines.append("# Triton Governance Scorecard")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_  ")
    lines.append(
        f"_Regime: **{regime}** | Committee: **{committee_decision or 'UNKNOWN'}** "
        f"| Governance trust: **{governance_trust_level}**_  "
    )
    lines.append(f"_System state: **{', '.join(labels)}**_  ")
    lines.append(f"_Observations: total={total_obs}, labelled={labelled_obs}_")
    lines.append("")

    lines.append("## Intelligence Health")
    lines.append(_row("intelligence_health_score", "intelligence_health_score"))
    lines.append("")

    lines.append("## Governance Quality")
    lines.append(_row("governance_quality_score", "governance_quality_score"))
    lines.append("")

    lines.append("## Trust Quality")
    lines.append(_row("trust_quality_score", "trust_quality_score"))
    lines.append("")

    lines.append("## Deployment Discipline")
    lines.append(_row("deployment_discipline_score", "deployment_discipline_score"))
    lines.append(_row("capital_preservation_score", "capital_preservation_score"))
    lines.append("")

    lines.append("## Learning Maturity")
    lines.append(_row("learning_maturity_score", "learning_maturity_score"))
    lines.append(f"  - total observations: {total_obs}")
    lines.append(
        f"  - labelled observations: {labelled_obs} "
        f"(maturity floor: {LEARNING_LABELLED_FLOOR}, saturation: {LEARNING_SATURATION_OBS})"
    )
    lines.append("")

    lines.append("## Portfolio Discipline")
    lines.append(_row("portfolio_discipline_score", "portfolio_discipline_score"))
    lines.append(_row("regime_quality_score", "regime_quality_score"))
    lines.append("")

    lines.append("## Recommendations")
    for r in recommendations:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Narrative")
    lines.append(narrative)
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_scorecard(
    *,
    diagnostics: Dict[str, Any],
    meta_intel: Dict[str, Any],
    feedback: Dict[str, Any],
    governed_policy: Dict[str, Any],
    ic_summary: Dict[str, Any],
    memory_insights: Dict[str, Any],
    regime_json: Dict[str, Any],
    committee_summary: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    diag_scores, diag_known = _extract_diag_scores(diagnostics)

    ih_score, ih_known = _intelligence_health_score(
        meta_intel=meta_intel, feedback=feedback, diagnostics=diagnostics
    )
    gov_score, gov_known = _governance_quality_score(
        diag_scores=diag_scores, diag_known=diag_known, feedback=feedback
    )
    trust_score, trust_known = _trust_quality_score(
        diag_scores=diag_scores, diag_known=diag_known, meta_intel=meta_intel
    )
    dep_score, dep_known = _deployment_discipline_score(
        diag_scores=diag_scores, diag_known=diag_known, ic_summary=ic_summary
    )
    cap_score, cap_known = _capital_preservation_score(
        diag_scores=diag_scores, diag_known=diag_known
    )
    learn_score, learn_known, total_obs, labelled_obs = _learning_maturity_score(
        diagnostics=diagnostics, memory_insights=memory_insights
    )
    port_score, port_known = _portfolio_discipline_score(ic_summary)
    regime_score, regime_known = _regime_quality_score(
        diag_scores=diag_scores, diag_known=diag_known, regime_json=regime_json
    )

    scores: Dict[str, float] = {
        "intelligence_health_score": round(ih_score, 6),
        "governance_quality_score": round(gov_score, 6),
        "trust_quality_score": round(trust_score, 6),
        "deployment_discipline_score": round(dep_score, 6),
        "capital_preservation_score": round(cap_score, 6),
        "learning_maturity_score": round(learn_score, 6),
        "portfolio_discipline_score": round(port_score, 6),
        "regime_quality_score": round(regime_score, 6),
    }
    known: Dict[str, bool] = {
        "intelligence_health_score": ih_known,
        "governance_quality_score": gov_known,
        "trust_quality_score": trust_known,
        "deployment_discipline_score": dep_known,
        "capital_preservation_score": cap_known,
        "learning_maturity_score": learn_known,
        "portfolio_discipline_score": port_known,
        "regime_quality_score": regime_known,
    }

    regime = (
        str((regime_json or {}).get("regime") or (diagnostics or {}).get("regime") or "UNKNOWN")
        .strip()
        .upper()
        or "UNKNOWN"
    )
    committee_decision = (
        str(
            (committee_summary or {}).get("committee_decision")
            or (committee_summary or {}).get("decision")
            or ""
        )
        .strip()
        .upper()
    )
    governance_trust_level = (
        str((feedback or {}).get("governance_trust_level") or "STABLE").strip().upper()
    )

    labels = _classify_system_state(
        scores=scores,
        known=known,
        total_obs=total_obs,
        labelled_obs=labelled_obs,
        regime=regime,
        committee_decision=committee_decision,
        feedback=feedback,
    )
    recommendations = _build_recommendations(
        scores=scores,
        known=known,
        labels=labels,
        total_obs=total_obs,
        labelled_obs=labelled_obs,
        feedback=feedback,
        governed_policy=governed_policy,
    )
    narrative = _build_narrative(
        scores=scores,
        known=known,
        labels=labels,
        regime=regime,
        committee_decision=committee_decision,
        governance_trust_level=governance_trust_level,
    )

    now_iso = _now_iso_utc()
    scorecard: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_governance_scorecard_engine",
        "engine_version": 1,
        "regime": regime,
        "committee_decision": committee_decision or None,
        "governance_trust_level": governance_trust_level,
        "system_state": labels,
        "scores": scores,
        "scores_known": known,
        "observations": {
            "total": total_obs,
            "labelled": labelled_obs,
            "labelled_floor": LEARNING_LABELLED_FLOOR,
            "saturation_target": LEARNING_SATURATION_OBS,
        },
        "narrative": narrative,
        "recommendations": recommendations,
        "inputs_seen": {
            "autonomous_strategy_diagnostics": bool(diagnostics),
            "meta_decision_intelligence": bool(meta_intel),
            "governance_trust_feedback": bool(feedback),
            "runtime_policy_governed": bool(governed_policy),
            "investment_committee_summary": bool(ic_summary),
            "portfolio_memory_insights": bool(memory_insights),
            "adaptive_regime": bool(regime_json),
            "autonomous_committee_summary": bool(committee_summary),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_governance_scorecard_engine",
        "regime": regime,
        "committee_decision": committee_decision or None,
        "governance_trust_level": governance_trust_level,
        "system_state": labels,
        "headline_scores": {
            "intelligence_health": scores["intelligence_health_score"],
            "governance_quality": scores["governance_quality_score"],
            "trust_quality": scores["trust_quality_score"],
            "deployment_discipline": scores["deployment_discipline_score"],
            "capital_preservation": scores["capital_preservation_score"],
            "learning_maturity": scores["learning_maturity_score"],
            "portfolio_discipline": scores["portfolio_discipline_score"],
            "regime_quality": scores["regime_quality_score"],
        },
        "n_known_scores": sum(1 for v in known.values() if v),
        "n_recommendations": len(recommendations),
        "observations_total": total_obs,
        "observations_labelled": labelled_obs,
    }

    md = _render_markdown(
        generated_at=now_iso,
        scores=scores,
        known=known,
        labels=labels,
        narrative=narrative,
        recommendations=recommendations,
        regime=regime,
        committee_decision=committee_decision,
        governance_trust_level=governance_trust_level,
        total_obs=total_obs,
        labelled_obs=labelled_obs,
    )
    return scorecard, md, summary


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only autonomous governance scorecard (Step 19). "
            "Synthesises outputs from Steps 9, 10, 12, 13, 15, 16, 17, 18 "
            "into a portfolio-manager-level health report covering eight "
            "category scores, multi-label system state, narrative, and "
            "recommendations."
        ),
    )
    p.add_argument("--diagnostics", default=str(DEFAULT_DIAGNOSTICS))
    p.add_argument("--meta-intel", default=str(DEFAULT_META_INTEL))
    p.add_argument("--gov-feedback", default=str(DEFAULT_GOV_FEEDBACK))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--ic-summary", default=str(DEFAULT_IC_SUMMARY))
    p.add_argument("--memory-insights", default=str(DEFAULT_MEMORY_INSIGHTS))
    p.add_argument("--regime", default=str(DEFAULT_REGIME))
    p.add_argument("--committee-summary", default=str(DEFAULT_COMMITTEE_SUMMARY))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[GOVERNANCE_SCORECARD] starting (synthesis of Steps 9..18)", flush=True)

    diagnostics = _safe_read_json(
        Path(args.diagnostics), label="autonomous_strategy_diagnostics.json"
    )
    meta_intel = _safe_read_json(Path(args.meta_intel), label="meta_decision_intelligence.json")
    feedback = _safe_read_json(Path(args.gov_feedback), label="governance_trust_feedback.json")
    governed_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    ic_summary = _safe_read_json(Path(args.ic_summary), label="investment_committee_summary.json")
    memory_insights = _safe_read_json(
        Path(args.memory_insights), label="portfolio_memory_insights.json"
    )
    regime_json = _safe_read_json(Path(args.regime), label="adaptive_regime.json")
    committee_summary = _safe_read_json(
        Path(args.committee_summary), label="autonomous_committee_summary.json"
    )

    scorecard, md, summary = build_scorecard(
        diagnostics=diagnostics,
        meta_intel=meta_intel,
        feedback=feedback,
        governed_policy=governed_policy,
        ic_summary=ic_summary,
        memory_insights=memory_insights,
        regime_json=regime_json,
        committee_summary=committee_summary,
    )

    try:
        _atomic_write_json(scorecard, Path(args.out_json))
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

    s = scorecard["scores"]
    print(
        "[GOVERNANCE_SCORECARD] "
        f"health={s['intelligence_health_score']:.3f} "
        f"governance={s['governance_quality_score']:.3f} "
        f"trust={s['trust_quality_score']:.3f} "
        f"discipline={s['deployment_discipline_score']:.3f}",
        flush=True,
    )
    print(
        f"[GOVERNANCE_SCORECARD_STATE] state={','.join(scorecard['system_state'])} "
        f"regime={scorecard['regime']} "
        f"committee={scorecard.get('committee_decision') or 'UNKNOWN'} "
        f"obs_labelled={scorecard['observations']['labelled']}",
        flush=True,
    )
    print(
        f"[GOVERNANCE_SCORECARD_OUT] json={Path(args.out_json).as_posix()} "
        f"md={Path(args.out_md).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
