"""
ARM Mode Governance Engine -- Step 28.

Reads:
    data/results/autonomous_execution_certificate_summary.json     (Step 27)
    data/results/autonomous_execution_certificate.json             (Step 27)
    data/results/autonomous_execution_risk_committee_summary.json  (Step 26)
    data/results/autonomous_readiness_summary.json                 (Step 21)
    data/results/autonomous_system_health_summary.json             (Step 20)
    data/results/autonomous_governance_summary.json                (Step 19)
    data/results/meta_decision_intelligence.json                   (Step 13)
    data/results/runtime_policy_governed.json                      (Step 18)

Writes:
    data/results/arm_mode_governance.json
    data/results/arm_mode_governance.md
    data/results/arm_mode_governance_summary.json

Purpose
-------
This engine answers:

    "What autonomy level is Triton allowed today?"

It is a *dormant* governance layer -- no execution code consumes it
yet. Future ARM (auto real-money) runtime/execution systems MUST
obey this layer before any broker activity.

ARM mode states (spec section 1, strict precedence)
---------------------------------------------------
    MANUAL          most restrictive; cert BLOCKED/DENIED, stale
                    system, READ_ONLY readiness, governance weak,
                    operator required
    ASSISTED        execution certified limited (or ESCALATED);
                    governance maturing; operator confirmation
                    required; suggestions allowed
    AUTO_DISABLED   execution certified (full) BUT autonomy
                    explicitly disabled by defensive runtime,
                    elevated uncertainty, low meta-trust, or
                    governance not yet adequate; shadow-mode only
    AUTO_ALLOWED    execution certified, system HEALTHY, readiness
                    READY, governance adequate, trust adequate,
                    runtime fresh; future auto execution allowed

Governance booleans (spec section 2)
------------------------------------
    autonomous_execution_allowed   AUTO_ALLOWED only
    operator_confirmation_required MANUAL or ASSISTED
    shadow_mode_required           AUTO_DISABLED
    manual_override_required       MANUAL
    auto_mode_eligible             AUTO_DISABLED or AUTO_ALLOWED
                                   (could become auto if conditions
                                    improve / explicit enable)
    certificate_required           always True

Autonomy confidence (spec section 3)
------------------------------------
0..1, weighted blend of six contributors:

    execution_certificate_confidence  0.30
    governance_quality                0.20
    readiness_score                   0.15
    system_health                     0.15
    trust_quality                     0.10
    runtime_freshness                 0.10

Safety
------
* READ ONLY. No broker calls, no execution mutation.
* Atomic writes (.tmp + os.replace).
* Missing inputs warn-and-continue. With insufficient evidence the
  default mode is MANUAL.
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

DEFAULT_CERT_SUMMARY = RESULTS_DIR / "autonomous_execution_certificate_summary.json"
DEFAULT_CERT_RECORD = RESULTS_DIR / "autonomous_execution_certificate.json"
DEFAULT_COMMITTEE_SUMMARY = RESULTS_DIR / "autonomous_execution_risk_committee_summary.json"
DEFAULT_READINESS_SUMMARY = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_GOV_SUMMARY = RESULTS_DIR / "autonomous_governance_summary.json"
DEFAULT_META_DECISION = RESULTS_DIR / "meta_decision_intelligence.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_mode_governance.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_mode_governance.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_mode_governance_summary.json"


# -----------------------------------------------------------
# Tunables
# -----------------------------------------------------------
MODE_MANUAL = "MANUAL"
MODE_ASSISTED = "ASSISTED"
MODE_AUTO_DISABLED = "AUTO_DISABLED"
MODE_AUTO_ALLOWED = "AUTO_ALLOWED"

ALL_MODES: Tuple[str, ...] = (
    MODE_MANUAL,
    MODE_ASSISTED,
    MODE_AUTO_DISABLED,
    MODE_AUTO_ALLOWED,
)

# Step 27 certification states
CERT_BLOCKED = "EXECUTION_BLOCKED"
CERT_DENIED = "EXECUTION_DENIED"
CERT_ESCALATED = "EXECUTION_ESCALATED"
CERT_CERTIFIED_LIMITED = "EXECUTION_CERTIFIED_LIMITED"
CERT_CERTIFIED = "EXECUTION_CERTIFIED"

# Step 20 health -> numeric
HEALTH_NUMERIC: Dict[str, float] = {
    "HEALTHY": 1.00,
    "DEGRADED": 0.60,
    "STALE": 0.30,
    "CRITICAL": 0.10,
    "OFFLINE": 0.00,
    "UNKNOWN": 0.50,
}

# Defensive regimes preclude full autonomy regardless of certification
DEFENSIVE_REGIMES = {"DEFENSIVE", "RISK_OFF", "HIGH_VOLATILITY", "CAPITAL_PRESERVATION"}

# Trust gates
LOW_META_TRUST_LEVELS = {"VERY_LOW", "LOW"}
COLLAPSED_GOV_TRUST = {"COLLAPSED"}

# Floors for AUTO_ALLOWED routing
AUTONOMY_CONFIDENCE_FLOOR = 0.65
GOVERNANCE_QUALITY_FLOOR = 0.60
TRUST_QUALITY_FLOOR = 0.55
RUNTIME_FRESHNESS_FLOOR = 0.60

# Confidence weights
CONFIDENCE_WEIGHTS: Dict[str, float] = {
    "execution_certificate_confidence": 0.30,
    "governance_quality": 0.20,
    "readiness_score": 0.15,
    "system_health": 0.15,
    "trust_quality": 0.10,
    "runtime_freshness": 0.10,
}


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_GOVERNANCE_WARN] {msg}", flush=True)


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


def _to_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_upper(x: Any, default: str = "UNKNOWN") -> str:
    s = str(x or "").strip().upper()
    return s or default


# -----------------------------------------------------------
# Contributor extraction
# -----------------------------------------------------------
def _extract_contributors(
    *,
    cert_summary: Dict[str, Any],
    cert_record: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
    gov_summary: Dict[str, Any],
    meta_decision: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    contributors: Dict[str, float] = {}
    known: Dict[str, bool] = {}

    # execution_certificate_confidence
    ec = _to_float(cert_summary.get("certificate_confidence"))
    if ec is None:
        ec = _to_float(cert_record.get("certificate_confidence"))
    contributors["execution_certificate_confidence"] = (
        _clamp(ec, 0.0, 1.0) if ec is not None else 0.0
    )
    known["execution_certificate_confidence"] = ec is not None

    # governance_quality: prefer Step 19 scorecard score
    gov_scores = (gov_summary or {}).get("scores") or {}
    gq = _to_float(gov_scores.get("governance_quality_score"))
    if gq is None:
        gq = _to_float((gov_summary or {}).get("governance_quality_score"))
    contributors["governance_quality"] = _clamp(gq, 0.0, 1.0) if gq is not None else 0.50
    known["governance_quality"] = gq is not None

    # readiness_score
    r = _to_float(readiness_summary.get("readiness_score"))
    contributors["readiness_score"] = _clamp(r, 0.0, 1.0) if r is not None else 0.50
    known["readiness_score"] = r is not None

    # system_health
    health = _norm_upper(health_summary.get("overall_status"), default="UNKNOWN")
    contributors["system_health"] = HEALTH_NUMERIC.get(health, 0.50)
    known["system_health"] = health != "UNKNOWN"

    # trust_quality
    tq = _to_float((meta_decision or {}).get("self_confidence_score"))
    if tq is None:
        tq = _to_float(gov_scores.get("trust_quality_score"))
    contributors["trust_quality"] = _clamp(tq, 0.0, 1.0) if tq is not None else 0.50
    known["trust_quality"] = tq is not None

    # runtime_freshness: lift from certificate's contributor if present
    rf = _to_float(
        ((cert_record or {}).get("certificate_confidence_contributors") or {}).get(
            "runtime_freshness"
        )
    )
    contributors["runtime_freshness"] = _clamp(rf, 0.0, 1.0) if rf is not None else 0.50
    known["runtime_freshness"] = rf is not None

    return contributors, known


def _compute_autonomy_confidence(contributors: Dict[str, float]) -> float:
    total_w = sum(CONFIDENCE_WEIGHTS.values()) or 1.0
    blended = sum(CONFIDENCE_WEIGHTS[k] * contributors[k] for k in CONFIDENCE_WEIGHTS) / total_w
    return _clamp(blended, 0.0, 1.0)


# -----------------------------------------------------------
# Mode classification (strict precedence)
# -----------------------------------------------------------
def _classify_mode(
    *,
    cert_state: str,
    certificate_valid: bool,
    committee_verdict: str,
    readiness_state: str,
    operator_review_required: bool,
    health_status: str,
    gov_trust_level: str,
    meta_trust_level: str,
    regime: str,
    autonomy_confidence: float,
    contributors: Dict[str, float],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    # ============ 1. MANUAL (most restrictive) ============
    if cert_state in (CERT_BLOCKED, CERT_DENIED):
        reasons.append(f"certification_state={cert_state}")
        return MODE_MANUAL, reasons
    if not certificate_valid:
        reasons.append("certificate not valid (certificate_valid=False)")
        return MODE_MANUAL, reasons
    if readiness_state in ("BLOCKED", "READ_ONLY"):
        reasons.append(f"readiness_state={readiness_state}")
        return MODE_MANUAL, reasons
    if health_status in ("CRITICAL", "OFFLINE"):
        reasons.append(f"system_health={health_status}")
        return MODE_MANUAL, reasons
    if gov_trust_level in COLLAPSED_GOV_TRUST:
        reasons.append(f"governance_trust_level={gov_trust_level}")
        return MODE_MANUAL, reasons
    if cert_state == "UNKNOWN":
        reasons.append("certification artefact missing or malformed")
        return MODE_MANUAL, reasons

    # ============ 2. ASSISTED ============
    if cert_state == CERT_ESCALATED:
        reasons.append(f"certification_state={cert_state}")
        return MODE_ASSISTED, reasons
    if cert_state == CERT_CERTIFIED_LIMITED:
        reasons.append(f"certification_state={cert_state}")
        return MODE_ASSISTED, reasons
    if operator_review_required:
        reasons.append("upstream operator_review_required=True")
        return MODE_ASSISTED, reasons
    if committee_verdict == "APPROVED_LIMITED":
        reasons.append(f"committee_verdict={committee_verdict}")
        return MODE_ASSISTED, reasons

    # ============ AUTO_DISABLED vs AUTO_ALLOWED ============
    # We require a fully CERTIFIED certificate at this point.
    if cert_state != CERT_CERTIFIED:
        reasons.append(f"certification_state={cert_state}; defaulting to MANUAL")
        return MODE_MANUAL, reasons

    disables: List[str] = []
    if regime in DEFENSIVE_REGIMES:
        disables.append(f"defensive regime {regime}")
    if meta_trust_level in LOW_META_TRUST_LEVELS:
        disables.append(f"meta_trust_level={meta_trust_level}")
    if autonomy_confidence < AUTONOMY_CONFIDENCE_FLOOR:
        disables.append(
            f"autonomy_confidence {autonomy_confidence:.3f} < " f"{AUTONOMY_CONFIDENCE_FLOOR:.2f}"
        )
    gq = contributors.get("governance_quality", 0.5)
    if gq < GOVERNANCE_QUALITY_FLOOR:
        disables.append(f"governance_quality {gq:.3f} < {GOVERNANCE_QUALITY_FLOOR:.2f}")
    tq = contributors.get("trust_quality", 0.5)
    if tq < TRUST_QUALITY_FLOOR:
        disables.append(f"trust_quality {tq:.3f} < {TRUST_QUALITY_FLOOR:.2f}")
    rf = contributors.get("runtime_freshness", 0.5)
    if rf < RUNTIME_FRESHNESS_FLOOR:
        disables.append(f"runtime_freshness {rf:.3f} < {RUNTIME_FRESHNESS_FLOOR:.2f}")
    if readiness_state != "READY":
        disables.append(f"readiness_state={readiness_state} (expected READY)")
    if health_status != "HEALTHY":
        disables.append(f"system_health={health_status} (expected HEALTHY)")

    if disables:
        reasons.append("execution certified but autonomy explicitly disabled:")
        reasons.extend(disables)
        return MODE_AUTO_DISABLED, reasons

    # ============ 4. AUTO_ALLOWED ============
    reasons.append(
        "certification CERTIFIED + readiness READY + system HEALTHY + "
        "governance adequate + trust adequate + runtime fresh + "
        "regime not defensive"
    )
    return MODE_AUTO_ALLOWED, reasons


# -----------------------------------------------------------
# Governance booleans
# -----------------------------------------------------------
def _governance_booleans(mode: str) -> Dict[str, bool]:
    is_auto = mode == MODE_AUTO_ALLOWED
    is_disabled = mode == MODE_AUTO_DISABLED
    is_assisted = mode == MODE_ASSISTED
    is_manual = mode == MODE_MANUAL
    return {
        "autonomous_execution_allowed": is_auto,
        "operator_confirmation_required": is_manual or is_assisted,
        "shadow_mode_required": is_disabled,
        "manual_override_required": is_manual,
        "auto_mode_eligible": is_disabled or is_auto,
        "certificate_required": True,
    }


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
def _build_recommendations(
    *,
    mode: str,
    reasons: List[str],
    contributors: Dict[str, float],
    regime: str,
    health_status: str,
) -> List[str]:
    recs: List[str] = []
    text = " ".join(reasons).lower()

    if mode == MODE_MANUAL:
        recs.append("Continue manual supervision -- no autonomous behaviour permitted.")
        if "stale" in health_status.lower() or "stale" in text:
            recs.append("Refresh stale pipeline artefacts before any future certification cycle.")
        if "collapsed" in text:
            recs.append("Governance trust has collapsed -- escalate to operator immediately.")
        recs.append("Maintain operator approval for all actions.")
        return recs

    if mode == MODE_ASSISTED:
        recs.append("Permit assisted deployment with operator confirmation on each action.")
        recs.append("Suggestions allowed; no unsupervised execution.")
        if "approved_limited" in text or "certified_limited" in text:
            recs.append(
                "Certificate is limited -- maintain elevated cash posture and defensive constraints."
            )
        if "escalated" in text or "escalate" in text:
            recs.append(
                "Re-run the committee after pipeline refresh before requesting higher autonomy."
            )
        return recs

    if mode == MODE_AUTO_DISABLED:
        recs.append(
            "Use shadow-mode only -- execution permitted in principle but autonomy disabled."
        )
        if regime in DEFENSIVE_REGIMES:
            recs.append(f"Defensive regime {regime} forbids full autonomy until conditions soften.")
        gq = contributors.get("governance_quality", 0.5)
        if gq < GOVERNANCE_QUALITY_FLOOR:
            recs.append(
                f"Governance quality {gq:.2f} below floor "
                f"{GOVERNANCE_QUALITY_FLOOR:.2f} -- continue observation."
            )
        tq = contributors.get("trust_quality", 0.5)
        if tq < TRUST_QUALITY_FLOOR:
            recs.append(
                f"Trust quality {tq:.2f} below floor "
                f"{TRUST_QUALITY_FLOOR:.2f} -- defer autonomy until self-confidence rebuilds."
            )
        rf = contributors.get("runtime_freshness", 0.5)
        if rf < RUNTIME_FRESHNESS_FLOOR:
            recs.append("Refresh runtime artefacts to lift freshness toward the autonomy floor.")
        recs.append("Re-evaluate ARM mode after the next certification cycle.")
        return recs

    # AUTO_ALLOWED
    recs.append("Autonomous execution permitted within the active certificate window.")
    recs.append(
        "Any future ARM execution engine must still validate the live certificate before acting."
    )
    if contributors.get("governance_quality", 0.5) < 0.70:
        recs.append("Governance maturity is still building -- monitor decision quality post-cycle.")
    return recs


# -----------------------------------------------------------
# Narrative
# -----------------------------------------------------------
def _narrative(
    *,
    mode: str,
    cert_state: str,
    readiness_state: str,
    health_status: str,
    regime: str,
    confidence: float,
    contributors: Dict[str, float],
) -> str:
    if mode == MODE_MANUAL:
        return (
            f"Triton is restricted to MANUAL mode. Certification "
            f"{cert_state}, readiness {readiness_state}, system "
            f"{health_status}. No autonomous behaviour is permitted; "
            f"operator supervision required. Autonomy confidence "
            f"{confidence:.2f}."
        )
    if mode == MODE_ASSISTED:
        return (
            f"Triton is restricted to ASSISTED mode because certification "
            f"{cert_state} or operator review is required. Suggestions "
            f"allowed with per-action operator confirmation. Autonomy "
            f"confidence {confidence:.2f}."
        )
    if mode == MODE_AUTO_DISABLED:
        return (
            f"Triton is execution-certified but autonomy is explicitly "
            f"disabled. Regime {regime}, system {health_status}, "
            f"governance quality {contributors.get('governance_quality', 0.5):.2f}. "
            f"Shadow-mode only; future ARM execution remains gated. "
            f"Autonomy confidence {confidence:.2f}."
        )
    return (
        f"Triton is in AUTO_ALLOWED mode -- the highest autonomy level "
        f"currently permitted. Certification {cert_state}, readiness "
        f"{readiness_state}, system {health_status}, regime {regime}. "
        f"Future ARM execution may proceed within the live certificate "
        f"window. Autonomy confidence {confidence:.2f}."
    )


# -----------------------------------------------------------
# Markdown report
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    mode: str,
    booleans: Dict[str, bool],
    confidence: float,
    contributors: Dict[str, float],
    known: Dict[str, bool],
    reasons: List[str],
    recommendations: List[str],
    narrative_text: str,
    upstream_context: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Triton ARM Governance")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_")
    lines.append("")

    lines.append("## ARM Mode")
    lines.append("")
    lines.append(f"**{mode}**")
    lines.append("")
    lines.append("| input | value |")
    lines.append("|---|---|")
    for k in (
        "certification_state",
        "committee_verdict",
        "certificate_valid",
        "readiness_state",
        "system_health",
        "governance_trust_level",
        "meta_trust_level",
        "regime",
    ):
        lines.append(f"| {k} | {upstream_context.get(k, '-')} |")
    lines.append("")

    lines.append("## Governance Confidence")
    lines.append("")
    lines.append(f"**{confidence:.3f}**")
    lines.append("")
    lines.append("| contributor | score | known | weight |")
    lines.append("|---|---|---|---|")
    for k, w in CONFIDENCE_WEIGHTS.items():
        lines.append(f"| {k} | {contributors[k]:.3f} | {known.get(k, False)} | {w:.2f} |")
    lines.append("")
    lines.append("**Autonomy floors:**")
    lines.append("")
    lines.append(f"- autonomy_confidence floor: {AUTONOMY_CONFIDENCE_FLOOR:.2f}")
    lines.append(f"- governance_quality floor:  {GOVERNANCE_QUALITY_FLOOR:.2f}")
    lines.append(f"- trust_quality floor:       {TRUST_QUALITY_FLOOR:.2f}")
    lines.append(f"- runtime_freshness floor:   {RUNTIME_FRESHNESS_FLOOR:.2f}")
    lines.append("")

    lines.append("## Permissions")
    lines.append("")
    lines.append("| boolean | value |")
    lines.append("|---|---|")
    for k in (
        "autonomous_execution_allowed",
        "operator_confirmation_required",
        "shadow_mode_required",
        "manual_override_required",
        "auto_mode_eligible",
        "certificate_required",
    ):
        lines.append(f"| {k} | {booleans[k]} |")
    lines.append("")

    lines.append("## Why")
    lines.append("")
    for r in reasons:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for r in recommendations:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Narrative")
    lines.append("")
    lines.append(narrative_text)
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_arm_governance(
    *,
    cert_summary: Dict[str, Any],
    cert_record: Dict[str, Any],
    committee_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
    gov_summary: Dict[str, Any],
    meta_decision: Dict[str, Any],
    runtime_policy: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    cert_state = _norm_upper(cert_summary.get("certification_state"))
    if cert_state == "UNKNOWN":
        cert_state = _norm_upper(cert_record.get("certification_state"))

    certificate_valid = _to_bool(
        cert_summary.get("certificate_valid", False),
        default=_to_bool(
            (cert_record.get("certificate_booleans") or {}).get("certificate_valid", False),
            default=False,
        ),
    )

    committee_verdict = _norm_upper(committee_summary.get("committee_verdict"))
    readiness_state = _norm_upper(readiness_summary.get("readiness_state"))
    operator_review_required = _to_bool(
        readiness_summary.get("operator_review_required", False),
        default=_to_bool(committee_summary.get("operator_review_required", False), default=False),
    )
    health_status = _norm_upper(health_summary.get("overall_status"))

    # governance_trust_level appears in Step 17 or Step 19 outputs; we
    # accept either via the governance summary blob.
    gov_trust_level = _norm_upper(
        (gov_summary or {}).get("governance_trust_level")
        or ((gov_summary or {}).get("scores") or {}).get("governance_trust_level")
    )
    meta_trust_level = _norm_upper((meta_decision or {}).get("trust_level"))

    regime = _norm_upper(
        (runtime_policy or {}).get("regime")
        or cert_summary.get("regime")
        or committee_summary.get("regime")
    )

    contributors, known = _extract_contributors(
        cert_summary=cert_summary,
        cert_record=cert_record,
        readiness_summary=readiness_summary,
        health_summary=health_summary,
        gov_summary=gov_summary,
        meta_decision=meta_decision,
    )
    autonomy_confidence = _compute_autonomy_confidence(contributors)

    mode, reasons = _classify_mode(
        cert_state=cert_state,
        certificate_valid=certificate_valid,
        committee_verdict=committee_verdict,
        readiness_state=readiness_state,
        operator_review_required=operator_review_required,
        health_status=health_status,
        gov_trust_level=gov_trust_level,
        meta_trust_level=meta_trust_level,
        regime=regime,
        autonomy_confidence=autonomy_confidence,
        contributors=contributors,
    )
    booleans = _governance_booleans(mode)
    recommendations = _build_recommendations(
        mode=mode,
        reasons=reasons,
        contributors=contributors,
        regime=regime,
        health_status=health_status,
    )
    narrative_text = _narrative(
        mode=mode,
        cert_state=cert_state,
        readiness_state=readiness_state,
        health_status=health_status,
        regime=regime,
        confidence=autonomy_confidence,
        contributors=contributors,
    )

    upstream_context = {
        "certification_state": cert_state,
        "certificate_valid": certificate_valid,
        "committee_verdict": committee_verdict,
        "readiness_state": readiness_state,
        "operator_review_required": operator_review_required,
        "system_health": health_status,
        "governance_trust_level": gov_trust_level,
        "meta_trust_level": meta_trust_level,
        "regime": regime,
    }

    now_iso = _now_iso_utc()
    record: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "arm_mode_governance_engine",
        "engine_version": 1,
        "arm_mode": mode,
        "arm_mode_reasons": reasons,
        "governance_booleans": booleans,
        "autonomy_confidence": round(autonomy_confidence, 6),
        "autonomy_confidence_contributors": {k: round(v, 6) for k, v in contributors.items()},
        "autonomy_confidence_known": known,
        "autonomy_confidence_weights": CONFIDENCE_WEIGHTS,
        "thresholds": {
            "autonomy_confidence_floor": AUTONOMY_CONFIDENCE_FLOOR,
            "governance_quality_floor": GOVERNANCE_QUALITY_FLOOR,
            "trust_quality_floor": TRUST_QUALITY_FLOOR,
            "runtime_freshness_floor": RUNTIME_FRESHNESS_FLOOR,
        },
        "recommendations": recommendations,
        "narrative": narrative_text,
        "upstream_context": upstream_context,
        "inputs_seen": {
            "autonomous_execution_certificate_summary": bool(cert_summary),
            "autonomous_execution_certificate": bool(cert_record),
            "autonomous_execution_risk_committee_summary": bool(committee_summary),
            "autonomous_readiness_summary": bool(readiness_summary),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_governance_summary": bool(gov_summary),
            "meta_decision_intelligence": bool(meta_decision),
            "runtime_policy_governed": bool(runtime_policy),
        },
        "note": (
            "This is a DORMANT governance layer. No execution engine "
            "currently consumes this artefact. Any future ARM runtime "
            "or execution code MUST honour this decision before any "
            "broker activity."
        ),
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "arm_mode_governance_engine",
        "arm_mode": mode,
        "autonomy_confidence": round(autonomy_confidence, 6),
        "autonomous_execution_allowed": booleans["autonomous_execution_allowed"],
        "operator_confirmation_required": booleans["operator_confirmation_required"],
        "shadow_mode_required": booleans["shadow_mode_required"],
        "manual_override_required": booleans["manual_override_required"],
        "auto_mode_eligible": booleans["auto_mode_eligible"],
        "certificate_required": booleans["certificate_required"],
        "certification_state": cert_state,
        "certificate_valid": certificate_valid,
        "committee_verdict": committee_verdict,
        "readiness_state": readiness_state,
        "system_health": health_status,
        "regime": regime,
        "n_recommendations": len(recommendations),
    }

    md = _render_markdown(
        generated_at=now_iso,
        mode=mode,
        booleans=booleans,
        confidence=autonomy_confidence,
        contributors=contributors,
        known=known,
        reasons=reasons,
        recommendations=recommendations,
        narrative_text=narrative_text,
        upstream_context=upstream_context,
    )
    return record, summary, md


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM mode governance engine (Step 28). Determines "
            "the highest autonomy level Triton is permitted to operate "
            "under today. This is a dormant governance layer; future "
            "ARM execution systems must obey its output."
        ),
    )
    p.add_argument("--cert-summary", default=str(DEFAULT_CERT_SUMMARY))
    p.add_argument("--cert-record", default=str(DEFAULT_CERT_RECORD))
    p.add_argument("--committee-summary", default=str(DEFAULT_COMMITTEE_SUMMARY))
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUMMARY))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--gov-summary", default=str(DEFAULT_GOV_SUMMARY))
    p.add_argument("--meta-decision", default=str(DEFAULT_META_DECISION))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[ARM_GOVERNANCE] starting (read-only autonomy governance)", flush=True)

    cert_summary = _safe_read_json(
        Path(args.cert_summary), label="autonomous_execution_certificate_summary.json"
    )
    cert_record = _safe_read_json(
        Path(args.cert_record), label="autonomous_execution_certificate.json"
    )
    committee_summary = _safe_read_json(
        Path(args.committee_summary), label="autonomous_execution_risk_committee_summary.json"
    )
    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="autonomous_readiness_summary.json"
    )
    health_summary = _safe_read_json(
        Path(args.health_summary), label="autonomous_system_health_summary.json"
    )
    gov_summary = _safe_read_json(
        Path(args.gov_summary), label="autonomous_governance_summary.json"
    )
    meta_decision = _safe_read_json(
        Path(args.meta_decision), label="meta_decision_intelligence.json"
    )
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")

    record, summary, md = build_arm_governance(
        cert_summary=cert_summary,
        cert_record=cert_record,
        committee_summary=committee_summary,
        readiness_summary=readiness_summary,
        health_summary=health_summary,
        gov_summary=gov_summary,
        meta_decision=meta_decision,
        runtime_policy=runtime_policy,
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

    b = record["governance_booleans"]
    print(
        "[ARM_GOVERNANCE] "
        f"mode={record['arm_mode']} "
        f"auto_allowed={b['autonomous_execution_allowed']} "
        f"shadow={b['shadow_mode_required']} "
        f"operator_required={b['operator_confirmation_required']} "
        f"confidence={record['autonomy_confidence']:.3f}",
        flush=True,
    )
    print(
        f"[ARM_GOVERNANCE_OUT] json={Path(args.out_json).as_posix()} "
        f"md={Path(args.out_md).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
