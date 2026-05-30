"""
Autonomous Execution Risk Committee Engine -- Step 26.

Reads:
    data/results/autonomous_execution_simulation.json           (Step 25)
    data/results/autonomous_execution_simulation_summary.json   (Step 25)
    data/results/autonomous_execution_plan_summary.json         (Step 24)
    data/results/runtime_policy_governed.json                   (Step 18)
    data/results/autonomous_governance_scorecard.json           (Step 19)
    data/results/autonomous_system_health_summary.json          (Step 20)
    data/results/autonomous_readiness_summary.json              (Step 21)
    data/results/meta_decision_intelligence.json                (Step 13)
    data/results/governance_trust_feedback.json                 (Step 17)

Writes:
    data/results/autonomous_execution_risk_committee.json
    data/results/autonomous_execution_risk_committee.md
    data/results/autonomous_execution_risk_committee_summary.json

Purpose
-------
Step 25 ran the "what-if" simulation. Step 26 is the final
institutional-style approval layer that answers:

    "Should Triton approve this execution plan?"

It is *strictly* a decision engine. It places no orders, mutates
no portfolio state, and never touches a broker. Its output is a
formal committee verdict, a set of approval booleans, a confidence
score, an enumerated violations list, and operator-facing
recommendations.

Verdict cascade (spec section 1, strict precedence)
---------------------------------------------------
    BLOCKED       sim BLOCKED, readiness BLOCKED/READ_ONLY,
                  system CRITICAL/OFFLINE, plan NO_EXECUTION
    REJECTED      sim UNSAFE, any critical violation,
                  governance trust COLLAPSED
    ESCALATE      sim WARNING, operator-review required by upstream,
                  meta trust VERY_LOW with deployment proposed,
                  governance trust WEAK with deployment proposed,
                  system DEGRADED
    APPROVED_LIMITED  sim SAFE_LIMITED, readiness READY_LIMITED,
                  defensive regime with deployment, system STALE
    APPROVED      sim SAFE, readiness READY, system HEALTHY,
                  governance not weak

Approval booleans (spec section 2)
----------------------------------
    execution_approved              -> APPROVED or APPROVED_LIMITED
    limited_execution_only          -> APPROVED_LIMITED
    operator_review_required        -> ESCALATE, REJECTED, BLOCKED
    defensive_constraints_required  -> APPROVED_LIMITED or defensive regime
    autonomous_execution_allowed    -> APPROVED only (strict reserve)

Approval confidence (spec section 3)
------------------------------------
0..1, weighted blend of six contributors:

    simulation_confidence   0.25
    readiness_score         0.20
    governance_quality      0.15
    trust_quality           0.15
    system_health           0.15
    committee_confidence    0.10

Safety
------
* READ ONLY. No broker calls, no execution mutation.
* Atomic writes (.tmp + os.replace).
* Missing inputs warn-and-continue. With insufficient evidence the
  default verdict is BLOCKED.
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

DEFAULT_SIM_JSON = RESULTS_DIR / "autonomous_execution_simulation.json"
DEFAULT_SIM_SUMMARY = RESULTS_DIR / "autonomous_execution_simulation_summary.json"
DEFAULT_PLAN_SUMMARY = RESULTS_DIR / "autonomous_execution_plan_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS_SUMMARY = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_META_DECISION = RESULTS_DIR / "meta_decision_intelligence.json"
DEFAULT_GOV_FEEDBACK = RESULTS_DIR / "governance_trust_feedback.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "autonomous_execution_risk_committee.json"
DEFAULT_OUT_MD = RESULTS_DIR / "autonomous_execution_risk_committee.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "autonomous_execution_risk_committee_summary.json"


# -----------------------------------------------------------
# Tunables
# -----------------------------------------------------------
VERDICT_APPROVED = "APPROVED"
VERDICT_APPROVED_LIMITED = "APPROVED_LIMITED"
VERDICT_ESCALATE = "ESCALATE"
VERDICT_REJECTED = "REJECTED"
VERDICT_BLOCKED = "BLOCKED"

ALL_VERDICTS: Tuple[str, ...] = (
    VERDICT_BLOCKED,
    VERDICT_REJECTED,
    VERDICT_ESCALATE,
    VERDICT_APPROVED_LIMITED,
    VERDICT_APPROVED,
)

SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"

# Simulation verdict semantics from Step 25
SIM_BLOCKED = "BLOCKED"
SIM_UNSAFE = "UNSAFE"
SIM_WARNING = "WARNING"
SIM_SAFE_LIMITED = "SAFE_LIMITED"
SIM_SAFE = "SAFE"

# Health -> numeric
HEALTH_NUMERIC: Dict[str, float] = {
    "HEALTHY": 1.00,
    "DEGRADED": 0.60,
    "STALE": 0.40,
    "CRITICAL": 0.10,
    "OFFLINE": 0.00,
    "UNKNOWN": 0.50,
}

# Defensive regimes
DEFENSIVE_REGIMES = {"DEFENSIVE", "RISK_OFF", "HIGH_VOLATILITY", "CAPITAL_PRESERVATION"}

# Trust levels
LOW_TRUST_LEVELS = {"VERY_LOW", "LOW"}
WEAK_GOV_TRUST = {"COLLAPSED", "WEAK"}
COLLAPSED_GOV_TRUST = {"COLLAPSED"}

# Approval-confidence contributor weights
CONFIDENCE_WEIGHTS: Dict[str, float] = {
    "simulation_confidence": 0.25,
    "readiness_score": 0.20,
    "governance_quality": 0.15,
    "trust_quality": 0.15,
    "system_health": 0.15,
    "committee_confidence": 0.10,
}

# Deployment-pct thresholds where governance pressure escalates
ESCALATION_DEPLOY_PCT = 5.0


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[EXECUTION_RISK_WARN] {msg}", flush=True)


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
    sim: Dict[str, Any],
    sim_summary: Dict[str, Any],
    plan_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    meta_decision: Dict[str, Any],
    gov_feedback: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """
    Returns:
        contributors: confidence-input scores in [0,1]
        known:        whether each contributor was observed (vs defaulted)
    """
    contributors: Dict[str, float] = {}
    known: Dict[str, bool] = {}

    # simulation_confidence
    sim_conf = _to_float(sim.get("simulation_confidence"))
    if sim_conf is None:
        sim_conf = _to_float(sim_summary.get("simulation_confidence"))
    contributors["simulation_confidence"] = (
        _clamp(sim_conf, 0.0, 1.0) if sim_conf is not None else 0.0
    )
    known["simulation_confidence"] = sim_conf is not None

    # readiness_score
    r = _to_float(readiness_summary.get("readiness_score"))
    contributors["readiness_score"] = _clamp(r, 0.0, 1.0) if r is not None else 0.50
    known["readiness_score"] = r is not None

    # governance_quality: scorecard.scores.governance_quality_score (Step 19)
    # else governance_trust_feedback.governance_health (Step 17)
    gq: Optional[float] = None
    scorecard_scores = (scorecard or {}).get("scores") or {}
    if "governance_quality_score" in scorecard_scores:
        gq = _to_float(scorecard_scores.get("governance_quality_score"))
    if gq is None:
        gq = _to_float((gov_feedback or {}).get("governance_health"))
    contributors["governance_quality"] = _clamp(gq, 0.0, 1.0) if gq is not None else 0.50
    known["governance_quality"] = gq is not None

    # trust_quality: meta_decision.self_confidence_score (Step 13)
    tq = _to_float((meta_decision or {}).get("self_confidence_score"))
    if tq is None:
        # Step 19 scorecard sometimes carries a trust_quality_score
        tq = _to_float(scorecard_scores.get("trust_quality_score"))
    contributors["trust_quality"] = _clamp(tq, 0.0, 1.0) if tq is not None else 0.50
    known["trust_quality"] = tq is not None

    # system_health: convert overall_status -> numeric
    overall = _norm_upper((health_summary or {}).get("overall_status"), default="UNKNOWN")
    contributors["system_health"] = HEALTH_NUMERIC.get(overall, 0.50)
    known["system_health"] = overall != "UNKNOWN"

    # committee_confidence: prefer plan_confidence (Step 24); else
    # the scorecard's deployment-discipline score; else neutral.
    cc = _to_float((plan_summary or {}).get("plan_confidence"))
    if cc is None:
        cc = _to_float(scorecard_scores.get("deployment_discipline_score"))
    contributors["committee_confidence"] = _clamp(cc, 0.0, 1.0) if cc is not None else 0.50
    known["committee_confidence"] = cc is not None

    return contributors, known


def _compute_approval_confidence(contributors: Dict[str, float]) -> float:
    total_w = sum(CONFIDENCE_WEIGHTS.values()) or 1.0
    blended = sum(CONFIDENCE_WEIGHTS[k] * contributors[k] for k in CONFIDENCE_WEIGHTS) / total_w
    return _clamp(blended, 0.0, 1.0)


# -----------------------------------------------------------
# Verdict classification (strict precedence)
# -----------------------------------------------------------
def _classify_verdict(
    *,
    sim: Dict[str, Any],
    sim_summary: Dict[str, Any],
    plan_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
    meta_decision: Dict[str, Any],
    gov_feedback: Dict[str, Any],
    regime: str,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    sim_verdict = _norm_upper(
        sim.get("simulation_verdict") or sim_summary.get("simulation_verdict"),
        default="UNKNOWN",
    )
    readiness_state = _norm_upper(readiness_summary.get("readiness_state"), default="UNKNOWN")
    health_overall = _norm_upper(health_summary.get("overall_status"), default="UNKNOWN")
    plan_mode = _norm_upper(plan_summary.get("execution_mode"), default="UNKNOWN")
    operator_review = bool(readiness_summary.get("operator_review_required"))
    n_critical = int(
        sim.get("n_critical_violations") or sim_summary.get("n_critical_violations") or 0
    )
    deploy_pct = _to_float(
        (sim.get("projected_metrics") or {}).get("projected_deployment_pct") if sim else None
    )
    if deploy_pct is None:
        deploy_pct = _to_float(sim_summary.get("projected_deployment_pct")) or 0.0
    meta_trust = _norm_upper(meta_decision.get("trust_level"), default="UNKNOWN")
    gov_trust = _norm_upper(gov_feedback.get("governance_trust_level"), default="UNKNOWN")

    # ----------- 1. BLOCKED (highest precedence) -----------
    if sim_verdict == SIM_BLOCKED:
        reasons.append("simulation_verdict=BLOCKED (no executable plan)")
        return VERDICT_BLOCKED, reasons
    if plan_mode == "NO_EXECUTION":
        reasons.append("plan execution_mode=NO_EXECUTION")
        return VERDICT_BLOCKED, reasons
    if readiness_state in ("BLOCKED", "READ_ONLY"):
        reasons.append(f"readiness_state={readiness_state}")
        return VERDICT_BLOCKED, reasons
    if health_overall in ("CRITICAL", "OFFLINE"):
        reasons.append(f"system_health overall_status={health_overall}")
        return VERDICT_BLOCKED, reasons
    if sim_verdict == "UNKNOWN":
        reasons.append("simulation artifact missing or malformed")
        return VERDICT_BLOCKED, reasons

    # ----------- 2. REJECTED -----------
    if sim_verdict == SIM_UNSAFE:
        reasons.append("simulation_verdict=UNSAFE")
        return VERDICT_REJECTED, reasons
    if n_critical >= 1:
        reasons.append(f"{n_critical} critical violation(s) from simulation")
        return VERDICT_REJECTED, reasons
    if gov_trust in COLLAPSED_GOV_TRUST:
        reasons.append(f"governance_trust_level={gov_trust}")
        return VERDICT_REJECTED, reasons

    # ----------- 3. ESCALATE -----------
    if sim_verdict == SIM_WARNING:
        reasons.append("simulation_verdict=WARNING (multiple warnings)")
        return VERDICT_ESCALATE, reasons
    if operator_review:
        reasons.append("readiness requires operator_review")
        return VERDICT_ESCALATE, reasons
    if meta_trust in LOW_TRUST_LEVELS and (deploy_pct or 0.0) > 0.0:
        reasons.append(f"meta_trust_level={meta_trust} with deployment proposed")
        return VERDICT_ESCALATE, reasons
    if gov_trust in WEAK_GOV_TRUST and (deploy_pct or 0.0) > ESCALATION_DEPLOY_PCT:
        reasons.append(
            f"governance_trust_level={gov_trust} with deployment "
            f"{deploy_pct:.2f}% > {ESCALATION_DEPLOY_PCT:.2f}%"
        )
        return VERDICT_ESCALATE, reasons
    if health_overall == "DEGRADED":
        reasons.append("system_health=DEGRADED")
        return VERDICT_ESCALATE, reasons

    # ----------- 4. APPROVED_LIMITED -----------
    if sim_verdict == SIM_SAFE_LIMITED:
        reasons.append("simulation_verdict=SAFE_LIMITED")
        return VERDICT_APPROVED_LIMITED, reasons
    if readiness_state == "READY_LIMITED":
        reasons.append("readiness_state=READY_LIMITED")
        return VERDICT_APPROVED_LIMITED, reasons
    if regime in DEFENSIVE_REGIMES and (deploy_pct or 0.0) > 0.0:
        reasons.append(f"defensive regime {regime} with deployment proposed")
        return VERDICT_APPROVED_LIMITED, reasons
    if health_overall == "STALE":
        reasons.append("system_health=STALE (pipeline refresh recommended)")
        return VERDICT_APPROVED_LIMITED, reasons

    # ----------- 5. APPROVED (default after clean checks) -----------
    if sim_verdict == SIM_SAFE:
        reasons.append("simulation_verdict=SAFE; readiness/health/governance clean")
        return VERDICT_APPROVED, reasons

    # Defensive fallback: anything else escalates rather than approves.
    reasons.append("indeterminate signal mix; defaulting to ESCALATE")
    return VERDICT_ESCALATE, reasons


# -----------------------------------------------------------
# Approval booleans
# -----------------------------------------------------------
def _approval_booleans(verdict: str, *, regime: str) -> Dict[str, bool]:
    is_approved = verdict == VERDICT_APPROVED
    is_approved_limited = verdict == VERDICT_APPROVED_LIMITED
    return {
        "execution_approved": is_approved or is_approved_limited,
        "limited_execution_only": is_approved_limited,
        "operator_review_required": verdict
        in (VERDICT_ESCALATE, VERDICT_REJECTED, VERDICT_BLOCKED),
        "defensive_constraints_required": is_approved_limited or (regime in DEFENSIVE_REGIMES),
        "autonomous_execution_allowed": is_approved,
    }


# -----------------------------------------------------------
# Violations enumeration
# -----------------------------------------------------------
# Map Step 25 violation names -> committee-friendly aliases
SIM_VIOLATION_ALIASES: Dict[str, str] = {
    "concentration_risk": "excessive_concentration",
    "insufficient_cash_buffer": "insufficient_cash_buffer",
    "approaching_cash_floor": "low_cash_buffer",
    "excessive_deployment": "excessive_deployment",
    "elevated_turnover": "elevated_turnover",
    "defensive_policy_violation": "defensive_policy_violation",
    "low_diversification": "low_diversification",
}

SIM_CRITICAL_NAMES = {
    "concentration_risk",
    "insufficient_cash_buffer",
    "defensive_policy_violation",
}


def _collect_violations(
    *,
    sim: Dict[str, Any],
    sim_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
    meta_decision: Dict[str, Any],
    gov_feedback: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Merge sim violations + committee-level concerns."""
    out: List[Dict[str, Any]] = []

    # 1. Lift each simulation violation
    sim_violations = sim.get("violations") or []
    if not sim_violations:
        for nm in sim_summary.get("violation_names") or []:
            sim_violations.append(
                {"name": nm, "severity": "warning", "detail": "(from simulation summary)"}
            )
    for v in sim_violations:
        nm = str(v.get("name") or "").strip()
        alias = SIM_VIOLATION_ALIASES.get(nm, nm)
        severity = str(v.get("severity") or "").strip().lower() or (
            SEVERITY_CRITICAL if nm in SIM_CRITICAL_NAMES else SEVERITY_WARNING
        )
        out.append(
            {
                "name": alias,
                "source": "simulation",
                "severity": severity,
                "detail": v.get("detail") or f"upstream simulation violation: {nm}",
            }
        )

    # 2. Governance / trust concerns
    gov_trust = _norm_upper(gov_feedback.get("governance_trust_level"), default="UNKNOWN")
    if gov_trust in COLLAPSED_GOV_TRUST:
        out.append(
            {
                "name": "governance_collapsed",
                "source": "governance_trust_feedback",
                "severity": SEVERITY_CRITICAL,
                "detail": f"governance_trust_level={gov_trust}",
            }
        )
    elif gov_trust in WEAK_GOV_TRUST:
        out.append(
            {
                "name": "governance_uncertainty",
                "source": "governance_trust_feedback",
                "severity": SEVERITY_WARNING,
                "detail": f"governance_trust_level={gov_trust}",
            }
        )

    # 3. Meta-trust concerns
    meta_trust = _norm_upper(meta_decision.get("trust_level"), default="UNKNOWN")
    if meta_trust in LOW_TRUST_LEVELS:
        out.append(
            {
                "name": "low_meta_trust",
                "source": "meta_decision_intelligence",
                "severity": SEVERITY_WARNING,
                "detail": f"meta_trust_level={meta_trust}",
            }
        )

    # 4. Pipeline freshness concerns
    health_overall = _norm_upper(health_summary.get("overall_status"), default="UNKNOWN")
    if health_overall in ("CRITICAL", "OFFLINE"):
        out.append(
            {
                "name": "pipeline_critical",
                "source": "system_health",
                "severity": SEVERITY_CRITICAL,
                "detail": f"overall_status={health_overall}",
            }
        )
    elif health_overall == "STALE":
        out.append(
            {
                "name": "stale_runtime_inputs",
                "source": "system_health",
                "severity": SEVERITY_WARNING,
                "detail": "overall_status=STALE",
            }
        )
    elif health_overall == "DEGRADED":
        out.append(
            {
                "name": "degraded_pipeline",
                "source": "system_health",
                "severity": SEVERITY_WARNING,
                "detail": "overall_status=DEGRADED",
            }
        )

    return out


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
def _build_recommendations(
    *,
    verdict: str,
    violations: List[Dict[str, Any]],
    regime: str,
    contributors: Dict[str, float],
) -> List[str]:
    recs: List[str] = []
    names = {v["name"] for v in violations}

    if verdict == VERDICT_BLOCKED:
        recs.append(
            "Block deployment and remain observational -- there is no "
            "authorised execution plan to evaluate, or the pipeline is "
            "unsafe to consume."
        )
        if "pipeline_critical" in names:
            recs.append("Refresh the pipeline before any future committee cycle.")
        return recs

    if verdict == VERDICT_REJECTED:
        recs.append("Reject deployment and remain observational this cycle.")
        if "excessive_concentration" in names:
            recs.append("Trim concentration risk before any future buys are proposed.")
        if "insufficient_cash_buffer" in names:
            recs.append("Restore the cash buffer to the policy target before re-proposing buys.")
        if "defensive_policy_violation" in names:
            recs.append("Honour the defensive deployment cap until the regime softens.")
        if "governance_collapsed" in names:
            recs.append(
                "Governance trust has collapsed -- escalate to operator review immediately."
            )
        return recs

    if verdict == VERDICT_ESCALATE:
        recs.append("Require operator review before any future execution honours this plan.")
        if "low_meta_trust" in names:
            recs.append("Meta-trust is low -- defer deployment until self-confidence rebuilds.")
        if "governance_uncertainty" in names:
            recs.append("Increase observation sample size before trusting governance signal.")
        if "degraded_pipeline" in names:
            recs.append("Refresh degraded artefacts before re-evaluating.")
        recs.append(
            "Conflicting signals detected -- treat the simulated plan as advisory until resolved."
        )
        return recs

    if verdict == VERDICT_APPROVED_LIMITED:
        recs.append("Proceed with defensive deployment only -- limited execution authorised.")
        recs.append("Maintain elevated cash discipline for this cycle.")
        if regime in DEFENSIVE_REGIMES:
            recs.append(f"Defensive regime {regime} caps single-cycle deployment.")
        if "low_cash_buffer" in names:
            recs.append(
                "Watch cash buffer drift -- avoid further deployment that breaches the target."
            )
        if "stale_runtime_inputs" in names:
            recs.append("Refresh stale runtime artefacts before scaling deployment.")
        if "low_diversification" in names:
            recs.append("Broaden the candidate set before further deployment.")
        return recs

    if verdict == VERDICT_APPROVED:
        recs.append("Execution approved -- plan satisfies all institutional risk checks.")
        recs.append("Proceed only after a future execution engine is authorised and connected.")
        gov = contributors.get("governance_quality", 0.5)
        if gov < 0.60:
            recs.append(
                "Governance maturity is still early -- monitor decision quality post-cycle."
            )
        return recs

    recs.append("No actionable recommendation derived -- treat plan as advisory.")
    return recs


# -----------------------------------------------------------
# Markdown report
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    verdict: str,
    verdict_reasons: List[str],
    approvals: Dict[str, bool],
    confidence: float,
    contributors: Dict[str, float],
    known: Dict[str, bool],
    violations: List[Dict[str, Any]],
    recommendations: List[str],
    regime: str,
    sim_verdict: str,
    readiness_state: str,
    health_overall: str,
    meta_trust: str,
    gov_trust: str,
) -> str:
    lines: List[str] = []
    lines.append("# Triton Execution Risk Committee")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_")
    lines.append("")

    lines.append("## Committee Verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append("| input | value |")
    lines.append("|---|---|")
    lines.append(f"| simulation_verdict | {sim_verdict} |")
    lines.append(f"| readiness_state | {readiness_state} |")
    lines.append(f"| system_health | {health_overall} |")
    lines.append(f"| meta_trust_level | {meta_trust} |")
    lines.append(f"| governance_trust_level | {gov_trust} |")
    lines.append(f"| regime | {regime} |")
    lines.append("")
    lines.append("| approval | value |")
    lines.append("|---|---|")
    for k in (
        "execution_approved",
        "limited_execution_only",
        "operator_review_required",
        "defensive_constraints_required",
        "autonomous_execution_allowed",
    ):
        lines.append(f"| {k} | {approvals[k]} |")
    lines.append("")

    lines.append("## Approval Confidence")
    lines.append("")
    lines.append(f"**{confidence:.3f}**")
    lines.append("")
    lines.append("| contributor | score | known | weight |")
    lines.append("|---|---|---|---|")
    for k, w in CONFIDENCE_WEIGHTS.items():
        lines.append(f"| {k} | {contributors[k]:.3f} | {known.get(k, False)} | {w:.2f} |")
    lines.append("")

    lines.append("## Violations")
    lines.append("")
    if violations:
        lines.append("| name | severity | source | detail |")
        lines.append("|---|---|---|---|")
        for v in violations:
            lines.append(
                f"| {v['name']} | {v['severity']} | {v.get('source', '-')} | "
                f"{v.get('detail', '-')} |"
            )
    else:
        lines.append("_(none)_")
    lines.append("")

    lines.append("## Why")
    lines.append("")
    for r in verdict_reasons:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for r in recommendations:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Narrative")
    lines.append("")
    n_crit = sum(1 for v in violations if v["severity"] == SEVERITY_CRITICAL)
    n_warn = sum(1 for v in violations if v["severity"] == SEVERITY_WARNING)
    if verdict == VERDICT_BLOCKED:
        narrative = (
            f"Execution risk committee BLOCKED. Simulation verdict {sim_verdict}, "
            f"readiness {readiness_state}, system health {health_overall} -- there is "
            f"no executable plan to approve."
        )
    elif verdict == VERDICT_REJECTED:
        narrative = (
            f"Execution REJECTED with {n_crit} critical and {n_warn} warning "
            f"violation(s). Treat the plan as advisory only and address the "
            f"violations before any future committee cycle."
        )
    elif verdict == VERDICT_ESCALATE:
        narrative = (
            f"Execution ESCALATED to operator review. Conflicting signals between "
            f"simulation ({sim_verdict}), readiness ({readiness_state}), and "
            f"trust levels prevent unsupervised approval. Confidence {confidence:.2f}."
        )
    elif verdict == VERDICT_APPROVED_LIMITED:
        narrative = (
            f"Execution APPROVED with limited deployment. Simulation {sim_verdict}, "
            f"regime {regime}. Defensive constraints required -- maintain elevated "
            f"cash discipline and constrain scale. Confidence {confidence:.2f}."
        )
    elif verdict == VERDICT_APPROVED:
        narrative = (
            f"Execution APPROVED. Simulation SAFE, readiness READY, system "
            f"HEALTHY, governance not weak. Confidence {confidence:.2f}. Proceed only "
            f"after a future execution engine is authorised and connected."
        )
    else:
        narrative = (
            f"Indeterminate committee state; defaulting to ESCALATE. "
            f"Confidence {confidence:.2f}."
        )
    lines.append(narrative)
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_risk_committee(
    *,
    sim: Dict[str, Any],
    sim_summary: Dict[str, Any],
    plan_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    meta_decision: Dict[str, Any],
    gov_feedback: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    regime = _norm_upper(
        sim.get("regime") or sim_summary.get("regime") or (runtime_policy or {}).get("regime"),
        default="UNKNOWN",
    )

    contributors, known = _extract_contributors(
        sim=sim,
        sim_summary=sim_summary,
        plan_summary=plan_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        meta_decision=meta_decision,
        gov_feedback=gov_feedback,
    )
    confidence = _compute_approval_confidence(contributors)

    verdict, verdict_reasons = _classify_verdict(
        sim=sim,
        sim_summary=sim_summary,
        plan_summary=plan_summary,
        readiness_summary=readiness_summary,
        health_summary=health_summary,
        meta_decision=meta_decision,
        gov_feedback=gov_feedback,
        regime=regime,
    )
    approvals = _approval_booleans(verdict, regime=regime)
    violations = _collect_violations(
        sim=sim,
        sim_summary=sim_summary,
        health_summary=health_summary,
        meta_decision=meta_decision,
        gov_feedback=gov_feedback,
    )
    recommendations = _build_recommendations(
        verdict=verdict,
        violations=violations,
        regime=regime,
        contributors=contributors,
    )

    sim_verdict = _norm_upper(
        sim.get("simulation_verdict") or sim_summary.get("simulation_verdict")
    )
    readiness_state = _norm_upper(readiness_summary.get("readiness_state"))
    health_overall = _norm_upper(health_summary.get("overall_status"))
    meta_trust = _norm_upper(meta_decision.get("trust_level"))
    gov_trust = _norm_upper(gov_feedback.get("governance_trust_level"))

    now_iso = _now_iso_utc()
    record: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_execution_risk_committee_engine",
        "engine_version": 1,
        "committee_verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "approvals": approvals,
        "approval_confidence": round(confidence, 6),
        "approval_confidence_contributors": {k: round(v, 6) for k, v in contributors.items()},
        "approval_confidence_known": known,
        "approval_confidence_weights": CONFIDENCE_WEIGHTS,
        "violations": violations,
        "n_violations": len(violations),
        "n_critical_violations": sum(1 for v in violations if v["severity"] == SEVERITY_CRITICAL),
        "n_warning_violations": sum(1 for v in violations if v["severity"] == SEVERITY_WARNING),
        "recommendations": recommendations,
        "upstream_context": {
            "simulation_verdict": sim_verdict,
            "readiness_state": readiness_state,
            "system_health": health_overall,
            "meta_trust_level": meta_trust,
            "governance_trust_level": gov_trust,
            "plan_execution_mode": plan_summary.get("execution_mode"),
            "regime": regime,
        },
        "inputs_seen": {
            "autonomous_execution_simulation": bool(sim),
            "autonomous_execution_simulation_summary": bool(sim_summary),
            "autonomous_execution_plan_summary": bool(plan_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(readiness_summary),
            "meta_decision_intelligence": bool(meta_decision),
            "governance_trust_feedback": bool(gov_feedback),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_execution_risk_committee_engine",
        "committee_verdict": verdict,
        "approval_confidence": round(confidence, 6),
        "execution_approved": approvals["execution_approved"],
        "limited_execution_only": approvals["limited_execution_only"],
        "operator_review_required": approvals["operator_review_required"],
        "defensive_constraints_required": approvals["defensive_constraints_required"],
        "autonomous_execution_allowed": approvals["autonomous_execution_allowed"],
        "n_violations": len(violations),
        "n_critical_violations": sum(1 for v in violations if v["severity"] == SEVERITY_CRITICAL),
        "n_warning_violations": sum(1 for v in violations if v["severity"] == SEVERITY_WARNING),
        "violation_names": [v["name"] for v in violations],
        "simulation_verdict": sim_verdict,
        "readiness_state": readiness_state,
        "system_health": health_overall,
        "regime": regime,
        "n_recommendations": len(recommendations),
    }

    md = _render_markdown(
        generated_at=now_iso,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
        approvals=approvals,
        confidence=confidence,
        contributors=contributors,
        known=known,
        violations=violations,
        recommendations=recommendations,
        regime=regime,
        sim_verdict=sim_verdict,
        readiness_state=readiness_state,
        health_overall=health_overall,
        meta_trust=meta_trust,
        gov_trust=gov_trust,
    )
    return record, summary, md


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only execution risk committee (Step 26). Issues a "
            "final institutional approval verdict over Step 25's "
            "simulated execution plan. Places no orders and mutates no "
            "portfolio state."
        ),
    )
    p.add_argument("--sim", default=str(DEFAULT_SIM_JSON))
    p.add_argument("--sim-summary", default=str(DEFAULT_SIM_SUMMARY))
    p.add_argument("--plan-summary", default=str(DEFAULT_PLAN_SUMMARY))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUMMARY))
    p.add_argument("--meta-decision", default=str(DEFAULT_META_DECISION))
    p.add_argument("--gov-feedback", default=str(DEFAULT_GOV_FEEDBACK))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[EXECUTION_RISK_COMMITTEE] starting (read-only final approval)", flush=True)

    sim = _safe_read_json(Path(args.sim), label="autonomous_execution_simulation.json")
    sim_summary = _safe_read_json(
        Path(args.sim_summary), label="autonomous_execution_simulation_summary.json"
    )
    plan_summary = _safe_read_json(
        Path(args.plan_summary), label="autonomous_execution_plan_summary.json"
    )
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    health_summary = _safe_read_json(
        Path(args.health_summary), label="autonomous_system_health_summary.json"
    )
    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="autonomous_readiness_summary.json"
    )
    meta_decision = _safe_read_json(
        Path(args.meta_decision), label="meta_decision_intelligence.json"
    )
    gov_feedback = _safe_read_json(Path(args.gov_feedback), label="governance_trust_feedback.json")

    record, summary, md = build_risk_committee(
        sim=sim,
        sim_summary=sim_summary,
        plan_summary=plan_summary,
        runtime_policy=runtime_policy,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        meta_decision=meta_decision,
        gov_feedback=gov_feedback,
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

    print(
        "[EXECUTION_RISK_COMMITTEE] "
        f"verdict={record['committee_verdict']} "
        f"approved={record['approvals']['execution_approved']} "
        f"limited={record['approvals']['limited_execution_only']} "
        f"review={record['approvals']['operator_review_required']} "
        f"confidence={record['approval_confidence']:.3f}",
        flush=True,
    )
    if record["violations"]:
        names = ",".join(v["name"] for v in record["violations"])
        print(f"[EXECUTION_RISK_VIOLATIONS] {names}", flush=True)
    print(
        f"[EXECUTION_RISK_OUT] json={Path(args.out_json).as_posix()} "
        f"md={Path(args.out_md).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
