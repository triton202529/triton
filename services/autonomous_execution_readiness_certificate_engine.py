"""
Autonomous Execution Readiness Certificate Engine -- Step 27.

Reads:
    data/results/autonomous_execution_risk_committee_summary.json (Step 26)
    data/results/autonomous_execution_risk_committee.json         (Step 26)
    data/results/autonomous_execution_simulation_summary.json     (Step 25)
    data/results/autonomous_execution_summary.json                (Step 23)
    data/results/autonomous_readiness_summary.json                (Step 21)
    data/results/autonomous_system_health_summary.json            (Step 20)
    data/results/autonomous_governance_summary.json               (Step 19)
    data/results/runtime_policy_governed.json                     (Step 18)

Writes:
    data/results/autonomous_execution_certificate.json
    data/results/autonomous_execution_certificate.md
    data/results/autonomous_execution_certificate_summary.json

Purpose
-------
Step 26 produced an institutional committee verdict. Step 27 is the
formal *certification* layer that issues a time-bounded
machine-readable execution license. It answers:

    "Is Triton officially certified safe to execute?"

The certificate is the canonical artefact that any future ARM (auto
real-money) execution layer must validate before any broker activity
is even theoretically possible. This engine performs no broker calls,
no order placement, and no portfolio mutation.

Certification states (spec section 1, strict precedence)
--------------------------------------------------------
    EXECUTION_BLOCKED            committee BLOCKED, readiness
                                 BLOCKED/READ_ONLY, system CRITICAL/
                                 OFFLINE, auth BLOCKED/ANALYSIS_ONLY,
                                 missing runtime policy
    EXECUTION_DENIED             committee REJECTED
    EXECUTION_ESCALATED          committee ESCALATE
    EXECUTION_CERTIFIED_LIMITED  committee APPROVED_LIMITED, *or*
                                 committee APPROVED but with at least
                                 one downstream gate not fully clean
    EXECUTION_CERTIFIED          committee APPROVED + readiness READY
                                 + system HEALTHY + auth FULL_AUTONOMY
                                 + simulation SAFE

Cross-validation is the key value-add: even with committee APPROVED,
this engine independently downgrades to CERTIFIED_LIMITED when *any*
upstream gate is not fully clean. The certificate never silently
trusts a single layer.

Certificate booleans (spec section 2)
-------------------------------------
    execution_certified              CERTIFIED or CERTIFIED_LIMITED
    limited_execution_only           CERTIFIED_LIMITED
    operator_review_required         ESCALATED, DENIED, BLOCKED
    defensive_constraints_required   CERTIFIED_LIMITED or defensive regime
    autonomous_execution_allowed     CERTIFIED only (most conservative)
    certificate_valid                CERTIFIED or CERTIFIED_LIMITED

Certificate confidence (spec section 3)
---------------------------------------
0..1, weighted blend of six contributors:

    committee_confidence    0.30  (Step 26 approval_confidence)
    simulation_confidence   0.20  (Step 25 simulation_confidence)
    readiness_score         0.15  (Step 21 readiness_score)
    system_health           0.15  (Step 20 overall_status -> numeric)
    governance_quality      0.10  (Step 19 scorecard score or feedback)
    runtime_freshness       0.10  (Step 18 policy age + health blend)

Certificate metadata (spec section 4)
-------------------------------------
    certificate_id        deterministic EXC-<sha256[:12]> over the
                          issuance tuple (issued_at, state, verdict,
                          policy_version) -- the same inputs yield the
                          same id, so verifiers can match the artefact
                          against a re-emitted run
    issued_at             ISO-8601 UTC issuance timestamp
    expires_at            ISO-8601 UTC expiry timestamp
    validity_minutes      TTL in whole minutes (30..120)
    runtime_policy_version  carried through from Step 18 governance
    governance_state      Step 19 multi-label system state
    readiness_state       Step 21 readiness_state
    committee_verdict     Step 26 committee_verdict

TTL policy
----------
    CERTIFIED              60..120 min, scales with runtime_freshness
    CERTIFIED_LIMITED      45..90  min
    ESCALATED/DENIED/BLOCKED  30 min (forces refresh)

Safety
------
* READ ONLY. No broker calls, no execution mutation.
* Atomic writes (.tmp + os.replace).
* Missing inputs warn-and-continue. With insufficient evidence the
  default certification state is EXECUTION_BLOCKED.
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_COMMITTEE_SUMMARY = RESULTS_DIR / "autonomous_execution_risk_committee_summary.json"
DEFAULT_COMMITTEE_RECORD = RESULTS_DIR / "autonomous_execution_risk_committee.json"
DEFAULT_SIM_SUMMARY = RESULTS_DIR / "autonomous_execution_simulation_summary.json"
DEFAULT_AUTH_SUMMARY = RESULTS_DIR / "autonomous_execution_summary.json"
DEFAULT_READINESS_SUMMARY = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_GOV_SUMMARY = RESULTS_DIR / "autonomous_governance_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "autonomous_execution_certificate.json"
DEFAULT_OUT_MD = RESULTS_DIR / "autonomous_execution_certificate.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "autonomous_execution_certificate_summary.json"


# -----------------------------------------------------------
# Tunables
# -----------------------------------------------------------
STATE_CERTIFIED = "EXECUTION_CERTIFIED"
STATE_CERTIFIED_LIMITED = "EXECUTION_CERTIFIED_LIMITED"
STATE_ESCALATED = "EXECUTION_ESCALATED"
STATE_DENIED = "EXECUTION_DENIED"
STATE_BLOCKED = "EXECUTION_BLOCKED"

ALL_STATES: Tuple[str, ...] = (
    STATE_BLOCKED,
    STATE_DENIED,
    STATE_ESCALATED,
    STATE_CERTIFIED_LIMITED,
    STATE_CERTIFIED,
)

# Step 26 verdicts
COMM_APPROVED = "APPROVED"
COMM_APPROVED_LIMITED = "APPROVED_LIMITED"
COMM_ESCALATE = "ESCALATE"
COMM_REJECTED = "REJECTED"
COMM_BLOCKED = "BLOCKED"

# Step 23 authorization states
AUTH_FULL_AUTONOMY = "FULL_AUTONOMY"
AUTH_SELECTIVE_DEPLOYMENT = "SELECTIVE_DEPLOYMENT"
AUTH_DEFENSIVE_EXECUTION = "DEFENSIVE_EXECUTION"
AUTH_EXIT_ONLY = "EXIT_ONLY"
AUTH_ANALYSIS_ONLY = "ANALYSIS_ONLY"
AUTH_BLOCKED = "BLOCKED"
AUTH_BLOCKING_STATES = {AUTH_BLOCKED, AUTH_ANALYSIS_ONLY}

# Step 25 simulation verdicts
SIM_SAFE = "SAFE"
SIM_SAFE_LIMITED = "SAFE_LIMITED"
SIM_WARNING = "WARNING"
SIM_UNSAFE = "UNSAFE"
SIM_BLOCKED = "BLOCKED"

# Step 20 system-health -> numeric
HEALTH_NUMERIC: Dict[str, float] = {
    "HEALTHY": 1.00,
    "DEGRADED": 0.60,
    "STALE": 0.30,
    "CRITICAL": 0.10,
    "OFFLINE": 0.00,
    "UNKNOWN": 0.50,
}

DEFENSIVE_REGIMES = {"DEFENSIVE", "RISK_OFF", "HIGH_VOLATILITY", "CAPITAL_PRESERVATION"}

# Confidence contributor weights
CONFIDENCE_WEIGHTS: Dict[str, float] = {
    "committee_confidence": 0.30,
    "simulation_confidence": 0.20,
    "readiness_score": 0.15,
    "system_health": 0.15,
    "governance_quality": 0.10,
    "runtime_freshness": 0.10,
}

# TTL band (minutes)
TTL_BLOCKED_MIN = 30
TTL_DENIED_MIN = 30
TTL_ESCALATED_MIN = 30
TTL_LIMITED_LO = 45
TTL_LIMITED_HI = 90
TTL_CERT_LO = 60
TTL_CERT_HI = 120

# Runtime-policy freshness window (minutes)
POLICY_FRESHNESS_GREEN_MIN = 30  # 0..30 min  -> 1.0
POLICY_FRESHNESS_EXPIRE_MIN = 240  # 240+ min   -> 0.0


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[EXECUTION_CERTIFICATE_WARN] {msg}", flush=True)


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


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(x: Any) -> Optional[datetime]:
    if not x:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_upper(x: Any, default: str = "UNKNOWN") -> str:
    s = str(x or "").strip().upper()
    return s or default


# -----------------------------------------------------------
# Freshness
# -----------------------------------------------------------
def _runtime_policy_freshness(
    runtime_policy: Dict[str, Any],
    *,
    now: datetime,
) -> Tuple[float, Optional[float]]:
    """
    Returns (freshness_score in [0,1], age_minutes or None).
    Linear decay between POLICY_FRESHNESS_GREEN_MIN and
    POLICY_FRESHNESS_EXPIRE_MIN.
    """
    gen = _parse_iso(
        (runtime_policy or {}).get("generated_at_utc") or (runtime_policy or {}).get("generated_at")
    )
    if gen is None:
        return 0.50, None
    age_s = (now - gen).total_seconds()
    age_min = age_s / 60.0
    if age_min <= POLICY_FRESHNESS_GREEN_MIN:
        return 1.0, age_min
    if age_min >= POLICY_FRESHNESS_EXPIRE_MIN:
        return 0.0, age_min
    span = POLICY_FRESHNESS_EXPIRE_MIN - POLICY_FRESHNESS_GREEN_MIN
    return _clamp(1.0 - (age_min - POLICY_FRESHNESS_GREEN_MIN) / span, 0.0, 1.0), age_min


# -----------------------------------------------------------
# Contributor extraction
# -----------------------------------------------------------
def _extract_contributors(
    *,
    committee_summary: Dict[str, Any],
    committee_record: Dict[str, Any],
    sim_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
    gov_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    now: datetime,
) -> Tuple[Dict[str, float], Dict[str, bool], float]:
    contributors: Dict[str, float] = {}
    known: Dict[str, bool] = {}

    # committee_confidence
    cc = _to_float(committee_summary.get("approval_confidence"))
    if cc is None:
        cc = _to_float(committee_record.get("approval_confidence"))
    contributors["committee_confidence"] = _clamp(cc, 0.0, 1.0) if cc is not None else 0.0
    known["committee_confidence"] = cc is not None

    # simulation_confidence
    sc = _to_float(sim_summary.get("simulation_confidence"))
    contributors["simulation_confidence"] = _clamp(sc, 0.0, 1.0) if sc is not None else 0.0
    known["simulation_confidence"] = sc is not None

    # readiness_score
    r = _to_float(readiness_summary.get("readiness_score"))
    contributors["readiness_score"] = _clamp(r, 0.0, 1.0) if r is not None else 0.50
    known["readiness_score"] = r is not None

    # system_health (numeric)
    health = _norm_upper(health_summary.get("overall_status"), default="UNKNOWN")
    contributors["system_health"] = HEALTH_NUMERIC.get(health, 0.50)
    known["system_health"] = health != "UNKNOWN"

    # governance_quality: prefer Step 19 score, fall back to neutral
    gov_scores = (gov_summary or {}).get("scores") or {}
    gq = _to_float(gov_scores.get("governance_quality_score"))
    if gq is None:
        gq = _to_float((gov_summary or {}).get("governance_quality_score"))
    contributors["governance_quality"] = _clamp(gq, 0.0, 1.0) if gq is not None else 0.50
    known["governance_quality"] = gq is not None

    # runtime_freshness: blend of policy age and health
    pol_fresh, age_min = _runtime_policy_freshness(runtime_policy, now=now)
    health_score = contributors["system_health"]
    runtime_freshness = (pol_fresh * 0.7) + (health_score * 0.3)
    contributors["runtime_freshness"] = _clamp(runtime_freshness, 0.0, 1.0)
    known["runtime_freshness"] = bool(runtime_policy)

    return contributors, known, (age_min if age_min is not None else float("nan"))


def _compute_certificate_confidence(contributors: Dict[str, float]) -> float:
    total_w = sum(CONFIDENCE_WEIGHTS.values()) or 1.0
    blended = sum(CONFIDENCE_WEIGHTS[k] * contributors[k] for k in CONFIDENCE_WEIGHTS) / total_w
    return _clamp(blended, 0.0, 1.0)


# -----------------------------------------------------------
# State classification (strict precedence + cross-validation)
# -----------------------------------------------------------
def _classify_state(
    *,
    committee_verdict: str,
    auth_state: str,
    readiness_state: str,
    health_status: str,
    sim_verdict: str,
    plan_mode: str,
    has_runtime_policy: bool,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    # ----- 1. BLOCKED (highest precedence) -----
    if committee_verdict == COMM_BLOCKED:
        reasons.append("committee verdict=BLOCKED")
        return STATE_BLOCKED, reasons
    if readiness_state in ("BLOCKED", "READ_ONLY"):
        reasons.append(f"readiness_state={readiness_state}")
        return STATE_BLOCKED, reasons
    if health_status in ("CRITICAL", "OFFLINE"):
        reasons.append(f"system_health overall_status={health_status}")
        return STATE_BLOCKED, reasons
    if auth_state in AUTH_BLOCKING_STATES:
        reasons.append(f"authorization_state={auth_state}")
        return STATE_BLOCKED, reasons
    if plan_mode == "NO_EXECUTION":
        reasons.append("plan_mode=NO_EXECUTION")
        return STATE_BLOCKED, reasons
    if not has_runtime_policy:
        reasons.append("runtime_policy_governed missing")
        return STATE_BLOCKED, reasons
    if committee_verdict == "UNKNOWN":
        reasons.append("committee verdict missing or malformed")
        return STATE_BLOCKED, reasons

    # ----- 2. DENIED -----
    if committee_verdict == COMM_REJECTED:
        reasons.append("committee verdict=REJECTED")
        return STATE_DENIED, reasons

    # ----- 3. ESCALATED -----
    if committee_verdict == COMM_ESCALATE:
        reasons.append("committee verdict=ESCALATE")
        return STATE_ESCALATED, reasons

    # ----- 4. CERTIFIED_LIMITED -----
    if committee_verdict == COMM_APPROVED_LIMITED:
        reasons.append("committee verdict=APPROVED_LIMITED")
        return STATE_CERTIFIED_LIMITED, reasons

    # ----- 5. APPROVED with cross-validation -----
    if committee_verdict == COMM_APPROVED:
        downgrades: List[str] = []
        if readiness_state != "READY":
            downgrades.append(f"readiness_state={readiness_state} (expected READY)")
        if health_status != "HEALTHY":
            downgrades.append(f"system_health={health_status} (expected HEALTHY)")
        if auth_state != AUTH_FULL_AUTONOMY:
            downgrades.append(f"authorization_state={auth_state} (expected {AUTH_FULL_AUTONOMY})")
        if sim_verdict != SIM_SAFE:
            downgrades.append(f"simulation_verdict={sim_verdict} (expected SAFE)")
        if downgrades:
            reasons.append("committee APPROVED but downgraded by cross-validation:")
            reasons.extend(downgrades)
            return STATE_CERTIFIED_LIMITED, reasons
        reasons.append(
            "committee APPROVED + readiness READY + system HEALTHY + "
            f"auth {AUTH_FULL_AUTONOMY} + simulation SAFE"
        )
        return STATE_CERTIFIED, reasons

    # Fallback: any unknown verdict defaults to BLOCKED for safety
    reasons.append(f"unrecognized committee verdict={committee_verdict}")
    return STATE_BLOCKED, reasons


# -----------------------------------------------------------
# Certificate booleans
# -----------------------------------------------------------
def _certificate_booleans(state: str, *, regime: str) -> Dict[str, bool]:
    is_cert = state == STATE_CERTIFIED
    is_cert_limited = state == STATE_CERTIFIED_LIMITED
    return {
        "execution_certified": is_cert or is_cert_limited,
        "limited_execution_only": is_cert_limited,
        "operator_review_required": state in (STATE_ESCALATED, STATE_DENIED, STATE_BLOCKED),
        "defensive_constraints_required": is_cert_limited or (regime in DEFENSIVE_REGIMES),
        "autonomous_execution_allowed": is_cert,
        "certificate_valid": is_cert or is_cert_limited,
    }


# -----------------------------------------------------------
# TTL
# -----------------------------------------------------------
def _compute_ttl_minutes(state: str, runtime_freshness: float) -> int:
    f = _clamp(runtime_freshness, 0.0, 1.0)
    if state == STATE_CERTIFIED:
        return int(round(TTL_CERT_LO + (TTL_CERT_HI - TTL_CERT_LO) * f))
    if state == STATE_CERTIFIED_LIMITED:
        return int(round(TTL_LIMITED_LO + (TTL_LIMITED_HI - TTL_LIMITED_LO) * f))
    if state == STATE_ESCALATED:
        return TTL_ESCALATED_MIN
    if state == STATE_DENIED:
        return TTL_DENIED_MIN
    return TTL_BLOCKED_MIN


# -----------------------------------------------------------
# Certificate ID (deterministic over inputs + issuance time)
# -----------------------------------------------------------
def _make_certificate_id(
    *,
    issued_at: str,
    state: str,
    committee_verdict: str,
    policy_version: str,
) -> str:
    raw = f"{issued_at}|{state}|{committee_verdict}|{policy_version}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:12].upper()
    return f"EXC-{digest}"


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
def _build_recommendations(
    *,
    state: str,
    regime: str,
    runtime_freshness: float,
    ttl_minutes: int,
    contributors: Dict[str, float],
    downgrades: List[str],
) -> List[str]:
    recs: List[str] = []
    if state == STATE_CERTIFIED:
        recs.append(
            "Execution certified -- proceed only after future ARM mode validates this certificate."
        )
        if runtime_freshness < 0.70:
            recs.append(
                "Refresh the pipeline before certificate expiration to maintain certified status."
            )
        gov = contributors.get("governance_quality", 0.5)
        if gov < 0.60:
            recs.append(
                "Governance maturity is still early -- monitor decision quality after the cycle."
            )
        return recs

    if state == STATE_CERTIFIED_LIMITED:
        recs.append("Proceed with defensive deployment only -- limited execution certified.")
        recs.append("Maintain elevated cash posture for this certificate window.")
        if regime in DEFENSIVE_REGIMES:
            recs.append(f"Defensive regime {regime} caps single-cycle deployment scale.")
        if downgrades:
            recs.append("Operator review recommended before scaling beyond limited deployment.")
        if runtime_freshness < 0.50:
            recs.append("Refresh stale runtime artefacts before considering re-certification.")
        return recs

    if state == STATE_ESCALATED:
        recs.append("Execution paused pending operator review -- conflicting upstream signals.")
        recs.append("Refresh the pipeline and rerun the committee before re-certification.")
        return recs

    if state == STATE_DENIED:
        recs.append("Execution denied -- treat plan as advisory only.")
        recs.append("Address the underlying violations before any future certification cycle.")
        return recs

    # BLOCKED
    recs.append("Execution blocked -- remain observational this cycle.")
    recs.append("Refresh the autonomous pipeline before any future certification attempt.")
    if ttl_minutes < 60:
        recs.append("Short TTL active -- re-emit the certificate after the next pipeline refresh.")
    return recs


# -----------------------------------------------------------
# Narrative
# -----------------------------------------------------------
def _narrative(
    *,
    state: str,
    committee_verdict: str,
    sim_verdict: str,
    readiness_state: str,
    health_status: str,
    auth_state: str,
    confidence: float,
    ttl_minutes: int,
    regime: str,
    downgrades: List[str],
) -> str:
    if state == STATE_CERTIFIED:
        return (
            f"Triton is officially certified for autonomous execution. "
            f"Committee {committee_verdict}, simulation {sim_verdict}, "
            f"readiness {readiness_state}, system {health_status}, "
            f"authorization {auth_state}. Certificate valid for "
            f"{ttl_minutes} minutes; confidence {confidence:.2f}."
        )
    if state == STATE_CERTIFIED_LIMITED:
        if downgrades:
            why = "downgraded by cross-validation (" + "; ".join(downgrades) + ")"
        else:
            why = f"committee verdict={committee_verdict}"
        return (
            f"Triton is certified for limited execution only -- {why}. "
            f"Regime {regime}, simulation {sim_verdict}. Certificate valid "
            f"for {ttl_minutes} minutes; confidence {confidence:.2f}."
        )
    if state == STATE_ESCALATED:
        return (
            f"Execution escalated to operator review. Committee "
            f"{committee_verdict}; conflicting signals between simulation "
            f"({sim_verdict}) and readiness ({readiness_state}). "
            f"Re-certify only after refresh; confidence {confidence:.2f}."
        )
    if state == STATE_DENIED:
        return (
            f"Execution denied. Committee {committee_verdict}; treat the "
            f"plan as advisory only and address violations before any "
            f"future certification. Confidence {confidence:.2f}."
        )
    return (
        f"Execution blocked. Committee {committee_verdict}, readiness "
        f"{readiness_state}, system {health_status}. Refresh the "
        f"pipeline before re-certification; confidence {confidence:.2f}."
    )


# -----------------------------------------------------------
# Markdown report
# -----------------------------------------------------------
def _render_markdown(
    *,
    state: str,
    metadata: Dict[str, Any],
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
    lines.append("# Triton Execution Readiness Certificate")
    lines.append("")
    lines.append(f"_Generated at {metadata['issued_at']}_")
    lines.append("")

    lines.append("## Certification State")
    lines.append("")
    lines.append(f"**{state}**")
    lines.append("")
    lines.append("| boolean | value |")
    lines.append("|---|---|")
    for k in (
        "execution_certified",
        "limited_execution_only",
        "operator_review_required",
        "defensive_constraints_required",
        "autonomous_execution_allowed",
        "certificate_valid",
    ):
        lines.append(f"| {k} | {booleans[k]} |")
    lines.append("")

    lines.append("## Certificate Metadata")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    for k in (
        "certificate_id",
        "issued_at",
        "expires_at",
        "validity_minutes",
        "runtime_policy_version",
        "governance_state",
        "readiness_state",
        "committee_verdict",
    ):
        lines.append(f"| {k} | {metadata.get(k, '-')} |")
    lines.append("")

    lines.append("**Certificate confidence:**")
    lines.append("")
    lines.append(f"**{confidence:.3f}**")
    lines.append("")
    lines.append("| contributor | score | known | weight |")
    lines.append("|---|---|---|---|")
    for k, w in CONFIDENCE_WEIGHTS.items():
        lines.append(f"| {k} | {contributors[k]:.3f} | {known.get(k, False)} | {w:.2f} |")
    lines.append("")

    lines.append("## Why")
    lines.append("")
    for r in reasons:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("**Upstream context:**")
    lines.append("")
    lines.append("| input | value |")
    lines.append("|---|---|")
    for k in (
        "committee_verdict",
        "simulation_verdict",
        "readiness_state",
        "system_health",
        "authorization_state",
        "plan_execution_mode",
        "regime",
    ):
        lines.append(f"| {k} | {upstream_context.get(k, '-')} |")
    lines.append("")

    lines.append("## Constraints")
    lines.append("")
    if booleans["execution_certified"]:
        if booleans["limited_execution_only"]:
            lines.append("- Limited deployment only -- defensive constraints active.")
        else:
            lines.append("- Full autonomous execution permitted within this certificate window.")
    else:
        lines.append("- No execution permitted under this certificate.")
    if booleans["operator_review_required"]:
        lines.append("- Operator review required.")
    if booleans["defensive_constraints_required"]:
        lines.append("- Defensive runtime constraints enforced.")
    lines.append(
        f"- Certificate expires at {metadata['expires_at']} (TTL {metadata['validity_minutes']} min)."
    )
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
def build_certificate(
    *,
    committee_summary: Dict[str, Any],
    committee_record: Dict[str, Any],
    sim_summary: Dict[str, Any],
    auth_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
    gov_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    now: Optional[datetime] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)

    committee_verdict = _norm_upper(committee_summary.get("committee_verdict"))
    if committee_verdict == "UNKNOWN":
        committee_verdict = _norm_upper(committee_record.get("committee_verdict"))
    auth_state = _norm_upper(auth_summary.get("authorization_state"))
    readiness_state = _norm_upper(readiness_summary.get("readiness_state"))
    health_status = _norm_upper(health_summary.get("overall_status"))
    sim_verdict = _norm_upper(sim_summary.get("simulation_verdict"))
    plan_mode = _norm_upper(
        (committee_record.get("upstream_context") or {}).get("plan_execution_mode")
        or auth_summary.get("plan_execution_mode")
    )
    regime = _norm_upper(
        (runtime_policy or {}).get("regime")
        or committee_summary.get("regime")
        or sim_summary.get("regime")
    )

    has_runtime_policy = bool(runtime_policy)

    state, reasons = _classify_state(
        committee_verdict=committee_verdict,
        auth_state=auth_state,
        readiness_state=readiness_state,
        health_status=health_status,
        sim_verdict=sim_verdict,
        plan_mode=plan_mode,
        has_runtime_policy=has_runtime_policy,
    )

    contributors, known, policy_age_min = _extract_contributors(
        committee_summary=committee_summary,
        committee_record=committee_record,
        sim_summary=sim_summary,
        readiness_summary=readiness_summary,
        health_summary=health_summary,
        gov_summary=gov_summary,
        runtime_policy=runtime_policy,
        now=now,
    )
    confidence = _compute_certificate_confidence(contributors)
    ttl_minutes = _compute_ttl_minutes(state, contributors["runtime_freshness"])

    issued_at = _iso_utc(now)
    expires_dt = now + timedelta(minutes=ttl_minutes)
    expires_at = _iso_utc(expires_dt)

    policy_version = str(
        (runtime_policy or {}).get("policy_version")
        or (runtime_policy or {}).get("generated_at_utc")
        or (runtime_policy or {}).get("generated_at")
        or "unknown"
    )

    cert_id = _make_certificate_id(
        issued_at=issued_at,
        state=state,
        committee_verdict=committee_verdict,
        policy_version=policy_version,
    )

    governance_state = (
        gov_summary.get("system_state") or gov_summary.get("governance_state") or "UNKNOWN"
    )

    metadata: Dict[str, Any] = {
        "certificate_id": cert_id,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "validity_minutes": ttl_minutes,
        "runtime_policy_version": policy_version,
        "governance_state": governance_state,
        "readiness_state": readiness_state,
        "committee_verdict": committee_verdict,
    }

    booleans = _certificate_booleans(state, regime=regime)

    # Extract downgrades from reasons (those after the "downgraded" marker)
    downgrades: List[str] = []
    capture = False
    for r in reasons:
        if capture and r:
            downgrades.append(r)
        if "downgraded by cross-validation" in r:
            capture = True

    recommendations = _build_recommendations(
        state=state,
        regime=regime,
        runtime_freshness=contributors["runtime_freshness"],
        ttl_minutes=ttl_minutes,
        contributors=contributors,
        downgrades=downgrades,
    )
    narrative_text = _narrative(
        state=state,
        committee_verdict=committee_verdict,
        sim_verdict=sim_verdict,
        readiness_state=readiness_state,
        health_status=health_status,
        auth_state=auth_state,
        confidence=confidence,
        ttl_minutes=ttl_minutes,
        regime=regime,
        downgrades=downgrades,
    )

    upstream_context = {
        "committee_verdict": committee_verdict,
        "simulation_verdict": sim_verdict,
        "readiness_state": readiness_state,
        "system_health": health_status,
        "authorization_state": auth_state,
        "plan_execution_mode": plan_mode,
        "regime": regime,
        "runtime_policy_age_minutes": (
            None if (policy_age_min != policy_age_min) else round(policy_age_min, 2)  # NaN guard
        ),
    }

    record: Dict[str, Any] = {
        "engine": "autonomous_execution_readiness_certificate_engine",
        "engine_version": 1,
        "certification_state": state,
        "certificate_metadata": metadata,
        "certificate_booleans": booleans,
        "certificate_confidence": round(confidence, 6),
        "certificate_confidence_contributors": {k: round(v, 6) for k, v in contributors.items()},
        "certificate_confidence_known": known,
        "certificate_confidence_weights": CONFIDENCE_WEIGHTS,
        "verdict_reasons": reasons,
        "downgrades": downgrades,
        "recommendations": recommendations,
        "narrative": narrative_text,
        "upstream_context": upstream_context,
        "inputs_seen": {
            "autonomous_execution_risk_committee_summary": bool(committee_summary),
            "autonomous_execution_risk_committee": bool(committee_record),
            "autonomous_execution_simulation_summary": bool(sim_summary),
            "autonomous_execution_summary": bool(auth_summary),
            "autonomous_readiness_summary": bool(readiness_summary),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_governance_summary": bool(gov_summary),
            "runtime_policy_governed": bool(runtime_policy),
        },
    }

    summary: Dict[str, Any] = {
        "engine": "autonomous_execution_readiness_certificate_engine",
        "certification_state": state,
        "certificate_id": cert_id,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "validity_minutes": ttl_minutes,
        "certificate_confidence": round(confidence, 6),
        "execution_certified": booleans["execution_certified"],
        "limited_execution_only": booleans["limited_execution_only"],
        "operator_review_required": booleans["operator_review_required"],
        "defensive_constraints_required": booleans["defensive_constraints_required"],
        "autonomous_execution_allowed": booleans["autonomous_execution_allowed"],
        "certificate_valid": booleans["certificate_valid"],
        "committee_verdict": committee_verdict,
        "simulation_verdict": sim_verdict,
        "readiness_state": readiness_state,
        "system_health": health_status,
        "authorization_state": auth_state,
        "regime": regime,
        "n_recommendations": len(recommendations),
    }

    md = _render_markdown(
        state=state,
        metadata=metadata,
        booleans=booleans,
        confidence=confidence,
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
            "Read-only execution readiness certificate engine (Step "
            "27). Issues a formal time-bounded machine-readable "
            "execution license over the upstream pipeline. Places no "
            "orders and mutates no portfolio state."
        ),
    )
    p.add_argument("--committee-summary", default=str(DEFAULT_COMMITTEE_SUMMARY))
    p.add_argument("--committee-record", default=str(DEFAULT_COMMITTEE_RECORD))
    p.add_argument("--sim-summary", default=str(DEFAULT_SIM_SUMMARY))
    p.add_argument("--auth-summary", default=str(DEFAULT_AUTH_SUMMARY))
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUMMARY))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--gov-summary", default=str(DEFAULT_GOV_SUMMARY))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument(
        "--now", default=None, help="Override 'now' as ISO-8601 UTC for deterministic testing"
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[EXECUTION_CERTIFICATE] starting (read-only certification)", flush=True)

    committee_summary = _safe_read_json(
        Path(args.committee_summary), label="autonomous_execution_risk_committee_summary.json"
    )
    committee_record = _safe_read_json(
        Path(args.committee_record), label="autonomous_execution_risk_committee.json"
    )
    sim_summary = _safe_read_json(
        Path(args.sim_summary), label="autonomous_execution_simulation_summary.json"
    )
    auth_summary = _safe_read_json(
        Path(args.auth_summary), label="autonomous_execution_summary.json"
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
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")

    now_dt = _parse_iso(args.now) if args.now else None

    record, summary, md = build_certificate(
        committee_summary=committee_summary,
        committee_record=committee_record,
        sim_summary=sim_summary,
        auth_summary=auth_summary,
        readiness_summary=readiness_summary,
        health_summary=health_summary,
        gov_summary=gov_summary,
        runtime_policy=runtime_policy,
        now=now_dt,
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

    b = record["certificate_booleans"]
    print(
        "[EXECUTION_CERTIFICATE] "
        f"state={record['certification_state']} "
        f"certified={b['execution_certified']} "
        f"limited={b['limited_execution_only']} "
        f"review={b['operator_review_required']} "
        f"confidence={record['certificate_confidence']:.3f} "
        f"ttl={record['certificate_metadata']['validity_minutes']}",
        flush=True,
    )
    print(
        f"[EXECUTION_CERTIFICATE_ID] {record['certificate_metadata']['certificate_id']} "
        f"expires_at={record['certificate_metadata']['expires_at']}",
        flush=True,
    )
    print(
        f"[EXECUTION_CERTIFICATE_OUT] json={Path(args.out_json).as_posix()} "
        f"md={Path(args.out_md).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
