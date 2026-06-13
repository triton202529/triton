"""
TRITON Phase 199 — Governance Authorization Intelligence & Alignment.

Explains why governance remains blocked, whether restriction is intentional,
and what evidence path is required for future authorization. Read-only —
does NOT modify governance rules, execution permissions, trading, or risk controls.

Outputs (data/results/):
    governance_authorization_intelligence.json
    governance_blocking_chain.csv
    governance_recovery_requirements.csv
    governance_decision_history.csv

Run:
    python -m services.governance_authorization_intelligence
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from services.concentration_recovery_simulator import CPS_GREEN, CPS_YELLOW

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "governance_authorization_intelligence.json"
OUT_CHAIN = RESULTS / "governance_blocking_chain.csv"
OUT_REQUIREMENTS = RESULTS / "governance_recovery_requirements.csv"
OUT_DECISION_HISTORY = RESULTS / "governance_decision_history.csv"

PERMIT_ESCALATION = {"GREEN", "YELLOW"}


def _log(msg: str) -> None:
    print(f"[governance_authorization_intelligence] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _load_context() -> Dict[str, Any]:
    return {
        "auth": _read_json(RESULTS / "governance_authorization.json"),
        "readiness": _read_json(RESULTS / "execution_readiness.json"),
        "stability": _read_json(RESULTS / "execution_readiness_stability.json"),
        "cpe": _read_json(RESULTS / "capital_preservation_escalation.json"),
        "cpi": _read_json(RESULTS / "capital_preservation_intelligence.json"),
        "gov_risk": _read_json(RESULTS / "governance_risk_summary.json"),
        "timeline": _read_json(RESULTS / "governance_recovery_timeline.json"),
        "thresholds": _read_json(RESULTS / "governance_recovery_thresholds.json"),
        "policy": _read_json(RESULTS / "protective_action_policy.json"),
        "queue": _read_json(RESULTS / "human_approval_queue.json"),
        "watchdog": _read_json(RESULTS / "watchdog_status.json"),
        "trace": _read_json(RESULTS / "governance_authorization_trace.json"),
    }


def _layer(
    name: str,
    *,
    passed: bool,
    confidence: float,
    evidence_source: str,
    timestamp: Optional[str],
    blocking_reason: Optional[str],
    downstream_impact: str,
    detail: Any = None,
) -> Dict[str, Any]:
    return {
        "layer": name,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "confidence": round(confidence, 1),
        "evidence_source": evidence_source,
        "timestamp": timestamp,
        "blocking_reason": blocking_reason,
        "downstream_impact": downstream_impact,
        "detail": detail,
    }


def _build_decision_chain(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    auth = ctx["auth"]
    readiness = ctx["readiness"]
    cpe = ctx["cpe"]
    gov = ctx["gov_risk"]
    policy = ctx["policy"]
    queue = ctx["queue"]
    watchdog = ctx["watchdog"]
    ts = auth.get("generated_at") or readiness.get("generated_at") or _now_iso()

    op_ok = bool(auth.get("operator_authorized"))
    approved = (queue.get("counts") or {}).get("APPROVED", 0)
    pol_ok = bool(auth.get("policy_authorized"))
    exec_ok = bool(auth.get("execution_authorized"))
    gov_ok = bool(auth.get("governance_authorized"))
    overall = bool(auth.get("overall_authorization"))

    cps = cpe.get("capital_preservation_score") or gov.get("capital_preservation_score")
    escalation = cpe.get("escalation_state") or gov.get("escalation_state") or "UNKNOWN"
    risk_ok = escalation in PERMIT_ESCALATION or gov_ok

    chain = [
        _layer(
            "Operator Approval",
            passed=op_ok,
            confidence=95.0 if op_ok else 60.0,
            evidence_source="governance_authorization.json + human_approval_queue.json",
            timestamp=ts,
            blocking_reason=None if op_ok else auth.get("gate_reasons", {}).get("operator"),
            downstream_impact="Required for candidate-level protective actions",
            detail={"approved_count": approved},
        ),
        _layer(
            "Policy Authorization",
            passed=pol_ok,
            confidence=90.0 if pol_ok else 55.0,
            evidence_source="governance_authorization.json + protective_action_policy.json",
            timestamp=ts,
            blocking_reason=None if pol_ok else auth.get("gate_reasons", {}).get("policy"),
            downstream_impact="Enables protective policy execution paths",
            detail={"enabled_policy_count": policy.get("enabled_policy_count")},
        ),
        _layer(
            "Execution Authorization",
            passed=exec_ok,
            confidence=92.0 if exec_ok else 50.0,
            evidence_source="governance_authorization.json",
            timestamp=ts,
            blocking_reason=None if exec_ok else auth.get("gate_reasons", {}).get("execution"),
            downstream_impact="Paper execution gate — live remains blocked",
            detail={
                "paper_execution_permitted": auth.get("paper_execution_permitted"),
                "live_execution_permitted": auth.get("live_execution_permitted"),
            },
        ),
        _layer(
            "Risk Evaluation",
            passed=risk_ok,
            confidence=85.0 if risk_ok else 80.0,
            evidence_source="governance_risk_summary.json + watchdog_status.json",
            timestamp=gov.get("generated_at") or watchdog.get("timestamp"),
            blocking_reason=(
                None
                if risk_ok
                else f"Escalation {escalation} — concentration and execution risk elevated"
            ),
            downstream_impact="Feeds Capital Preservation Review and Governance Decision",
            detail={
                "governance_drivers": gov.get("governance_drivers"),
                "governance_awareness": gov.get("governance_awareness_level"),
            },
        ),
        _layer(
            "Capital Preservation Review",
            passed=escalation in PERMIT_ESCALATION,
            confidence=88.0,
            evidence_source="capital_preservation_escalation.json",
            timestamp=cpe.get("generated_at"),
            blocking_reason=(
                None
                if escalation in PERMIT_ESCALATION
                else "; ".join(
                    cpe.get("escalation_reason_labels") or cpe.get("escalation_reason") or []
                )
            ),
            downstream_impact="Determines whether governance gate may permit authorization",
            detail={
                "cps_score": cps,
                "health_band": cpe.get("health_band"),
                "risk_trend": cpe.get("risk_trend"),
                "escalation_state": escalation,
            },
        ),
        _layer(
            "Governance Decision",
            passed=gov_ok,
            confidence=90.0 if gov_ok else 85.0,
            evidence_source="governance_authorization.json (governance_execution_readiness.py)",
            timestamp=ts,
            blocking_reason=None if gov_ok else auth.get("gate_reasons", {}).get("governance"),
            downstream_impact="Final institutional permission layer before overall authorization",
            detail={"requires_green_governance": True, "current_escalation": escalation},
        ),
        _layer(
            "Overall Authorization",
            passed=overall,
            confidence=95.0 if overall else 70.0,
            evidence_source="governance_authorization.json",
            timestamp=ts,
            blocking_reason=(
                None
                if overall
                else "Composite gate — governance blocked while execution may remain authorized"
            ),
            downstream_impact="Controls end-to-end authorized protective execution",
            detail={"overall_authorization": overall},
        ),
    ]
    return chain


def _classify_blocker(layer: str, reason: str, ctx: Dict[str, Any]) -> str:
    reason_l = (reason or "").lower()
    escalation = (ctx["cpe"].get("escalation_state") or "").upper()

    if layer == "Governance Decision" and "orange" in reason_l:
        return "EXPECTED_PROTECTIVE_BLOCK"
    if layer == "Capital Preservation Review" and escalation == "ORANGE":
        return "EXPECTED_PROTECTIVE_BLOCK"
    if "concentration" in reason_l or "excess_concentration" in reason_l:
        return "EXPECTED_PROTECTIVE_BLOCK"
    if layer == "Operator Approval" and "no approved" in reason_l:
        return "MISSING_EVIDENCE"
    if "policy disabled" in reason_l or "no protective policy" in reason_l:
        return "CONFIGURATION_CONFLICT"
    if "stale" in reason_l or "watchdog" in reason_l:
        return "TEMPORARY_CONDITION"
    if layer == "Risk Evaluation":
        return "EXPECTED_PROTECTIVE_BLOCK"
    return "UNKNOWN"


def _blocking_analysis(chain: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
    blockers = [l for l in chain if not l["pass"]]
    classified: List[Dict[str, Any]] = []
    for i, layer in enumerate(blockers):
        reason = layer.get("blocking_reason") or "Unknown"
        cls = _classify_blocker(layer["layer"], reason, ctx)
        classified.append(
            {
                "sequence": i + 1,
                "layer": layer["layer"],
                "blocking_reason": reason,
                "classification": cls,
                "confidence": layer.get("confidence"),
                "downstream_impact": layer.get("downstream_impact"),
                "is_primary": i == 0,
            }
        )

    primary = classified[0] if classified else None
    secondary = classified[1:] if len(classified) > 1 else []

    hist = _decision_history_rows(ctx)
    gov_fail_rate = None
    if hist:
        gov_fail_rate = sum(1 for h in hist if h.get("governance_authorized") is False) / len(hist)

    timeline = ctx["timeline"].get("executive_report") or {}
    auto_resolve_prob = timeline.get("recovery_probability_score")
    if auto_resolve_prob is None:
        auto_resolve_prob = 75 if (ctx["cpe"].get("risk_trend") == "STABLE") else 40

    return {
        "primary_blocker": primary,
        "secondary_blockers": secondary,
        "root_cause": (
            f"{ctx['cpe'].get('escalation_state')} escalation driven by "
            f"{', '.join(ctx['cpe'].get('escalation_reason_labels') or ['risk elevation'])}"
        ),
        "first_blocking_decision": primary["layer"] if primary else None,
        "historical_persistence_pct": (
            round(gov_fail_rate * 100, 1) if gov_fail_rate is not None else None
        ),
        "automatic_resolution_probability": auto_resolve_prob,
        "blockers": classified,
        "block_intentional": (
            primary["classification"] == "EXPECTED_PROTECTIVE_BLOCK" if primary else False
        ),
    }


def _cps_alignment(ctx: Dict[str, Any]) -> Dict[str, Any]:
    cpe = ctx["cpe"]
    cps = float(
        cpe.get("capital_preservation_score")
        or ctx["gov_risk"].get("capital_preservation_score")
        or 0
    )
    escalation = cpe.get("escalation_state") or "UNKNOWN"
    trend = cpe.get("risk_trend") or "UNKNOWN"

    timeline = ctx["timeline"].get("executive_report") or {}
    thresholds = ctx["thresholds"].get("executive_report") or {}

    dist_yellow = max(0, CPS_YELLOW - cps)
    dist_green = max(0, CPS_GREEN - cps)

    return {
        "cps_score": cps,
        "escalation_state": escalation,
        "health_band": cpe.get("health_band"),
        "risk_trend": trend,
        "trajectory": (
            "IMPROVING"
            if trend == "IMPROVING"
            else "STABLE" if trend == "STABLE" else "DETERIORATING"
        ),
        "distance_to_yellow_cps_points": dist_yellow,
        "distance_to_green_cps_points": dist_green,
        "concentration_drivers": cpe.get("escalation_reason_labels"),
        "diversification_status": timeline.get("current_state", {}).get("bito_status"),
        "expected_yellow_date": timeline.get("expected_yellow_date"),
        "expected_green_date": timeline.get("expected_green_date"),
        "days_to_yellow": timeline.get("days_to_yellow"),
        "days_to_green": timeline.get("days_to_green"),
        "recovery_probability": timeline.get("recovery_probability"),
        "recovery_probability_score": timeline.get("recovery_probability_score"),
        "minimum_yellow_plan": thresholds.get("minimum_yellow_recovery_plan"),
        "recommended_recovery_path": timeline.get("recommended_recovery_path"),
        "projected_governance_recovery": timeline.get("projected_governance_recovery"),
    }


def _recovery_requirements(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    cpe = ctx["cpe"]
    timeline = ctx["timeline"].get("executive_report") or {}
    thresholds = ctx["thresholds"].get("executive_report") or {}
    yellow_plan = thresholds.get("minimum_yellow_recovery_plan") or {}
    reqs: List[Dict[str, Any]] = []

    for label in cpe.get("escalation_reason_labels") or cpe.get("escalation_reason") or []:
        reqs.append(
            {
                "requirement": f"Resolve {label}",
                "priority": "CRITICAL",
                "owner": "Risk Committee / Operator",
                "estimated_resolution_days": timeline.get("days_to_yellow"),
                "impact_on_authorization": "Required for YELLOW escalation and partial governance authorization",
                "evidence_needed": f"Metrics show {label} no longer triggering escalation",
            }
        )

    if yellow_plan:
        reqs.append(
            {
                "requirement": f"Achieve CPS ≥ {CPS_YELLOW} (projected {yellow_plan.get('projected_cps')})",
                "priority": "CRITICAL",
                "owner": "Portfolio / Allocation",
                "estimated_resolution_days": timeline.get("days_to_yellow"),
                "impact_on_authorization": "Enables YELLOW escalation — governance may authorize protective actions",
                "evidence_needed": f"Diversification to {yellow_plan.get('weights')} weights",
            }
        )

    reqs.append(
        {
            "requirement": f"Achieve CPS ≥ {CPS_GREEN} for GREEN governance",
            "priority": "HIGH",
            "owner": "Portfolio / Lifecycle",
            "estimated_resolution_days": timeline.get("days_to_green"),
            "impact_on_authorization": "Full governance authorization restoration",
            "evidence_needed": "BITO promotion and concentration within green thresholds",
        }
    )

    for action in cpe.get("recommended_review_actions") or []:
        reqs.append(
            {
                "requirement": action,
                "priority": "HIGH",
                "owner": "Operator",
                "estimated_resolution_days": None,
                "impact_on_authorization": "Supports escalation downgrade evidence package",
                "evidence_needed": "Documented review completion in accountability registry",
            }
        )

    return reqs


def _authorization_package(ctx: Dict[str, Any], chain: List[Dict[str, Any]]) -> Dict[str, Any]:
    auth = ctx["auth"]
    exec_ok = bool(auth.get("execution_authorized"))
    gov_ok = bool(auth.get("governance_authorized"))

    if gov_ok and exec_ok:
        status = "AUTHORIZED"
    elif exec_ok or auth.get("paper_execution_permitted"):
        status = "PARTIALLY_AUTHORIZED"
    else:
        status = "BLOCKED"

    reqs = _recovery_requirements(ctx)
    return {
        "current_status": status,
        "governance_authorized": gov_ok,
        "execution_authorized": exec_ok,
        "paper_execution_permitted": auth.get("paper_execution_permitted"),
        "required_evidence": reqs,
        "summary": (
            "Execution capability available under paper mode; governance blocked by protective ORANGE escalation"
            if status == "PARTIALLY_AUTHORIZED"
            else "Fully authorized" if status == "AUTHORIZED" else "Blocked at multiple layers"
        ),
    }


def _contradiction_detection(ctx: Dict[str, Any]) -> Dict[str, Any]:
    auth = ctx["auth"]
    readiness = ctx["readiness"]
    stability = ctx["stability"]

    ops_ready = (stability.get("certification_impact") or {}).get("operational_readiness_status")
    if not ops_ready:
        checks = readiness.get("checks") or {}
        ops_ready = (
            "READY"
            if all(
                checks.get(c)
                for c in ("data_freshness", "watchdog", "broker", "lifecycle", "signals")
            )
            else "NOT_READY"
        )

    gov_blocked = not auth.get("governance_authorized")
    exec_ready = readiness.get("readiness_status") in ("READY", "PARTIALLY_READY")
    exec_auth = bool(auth.get("execution_authorized"))

    contradiction = gov_blocked and (ops_ready == "READY" or exec_auth)

    if not contradiction:
        classification = "NONE"
        explanation = "No material contradiction between execution and governance layers"
    elif (
        stability.get("governance_awareness", {}).get("classification")
        == "EXPECTED_GOVERNANCE_RESTRICTION"
    ):
        classification = "HEALTHY_SEPARATION"
        explanation = (
            "Operational stack is READY while governance intentionally withholds authorization "
            "due to ORANGE protective escalation — disciplined separation of capability vs permission"
        )
    elif not exec_auth and gov_blocked:
        classification = "POLICY_CONFLICT"
        explanation = "Both execution and governance blocked — aligned denial"
    elif readiness.get("checks", {}).get("policy") is False:
        classification = "MISCONFIGURATION"
        explanation = "Policy gate misaligned with governance posture"
    else:
        classification = "STALE_DECISION"
        explanation = "Governance block may not reflect current operational readiness — review escalation inputs"

    return {
        "contradiction_detected": contradiction,
        "classification": classification,
        "execution_posture": ops_ready,
        "governance_posture": "BLOCKED" if gov_blocked else "AUTHORIZED",
        "readiness_status": readiness.get("readiness_status"),
        "explanation": explanation,
    }


def _intelligence_scores(
    ctx: Dict[str, Any],
    chain: List[Dict[str, Any]],
    blocking: Dict[str, Any],
    package: Dict[str, Any],
) -> Dict[str, Any]:
    layers_with_reason = sum(1 for l in chain if l.get("blocking_reason") or l["pass"])
    transparency = round(layers_with_reason / len(chain) * 100, 1) if chain else 0.0

    primary = blocking.get("primary_blocker") or {}
    if primary.get("classification") == "EXPECTED_PROTECTIVE_BLOCK":
        auth_confidence = 85.0
    elif primary.get("classification") == "TEMPORARY_CONDITION":
        auth_confidence = 55.0
    else:
        auth_confidence = 65.0

    reqs = package.get("required_evidence") or []
    evidence_complete = sum(1 for r in reqs if r.get("evidence_needed")) / max(len(reqs), 1) * 100

    timeline = ctx["timeline"].get("executive_report") or {}
    recovery_score = float(timeline.get("recovery_probability_score") or 50)

    composite = round(
        transparency * 0.25
        + auth_confidence * 0.25
        + evidence_complete * 0.25
        + recovery_score * 0.25,
        1,
    )

    if composite >= 90:
        band = "INSTITUTIONAL_GRADE"
    elif composite >= 75:
        band = "MATURE"
    elif composite >= 50:
        band = "DEVELOPING"
    else:
        band = "INSUFFICIENT"

    return {
        "governance_transparency_score": transparency,
        "authorization_confidence_score": auth_confidence,
        "evidence_completeness_score": round(evidence_complete, 1),
        "recovery_readiness_score": recovery_score,
        "composite_intelligence_score": composite,
        "classification": band,
    }


def _decision_history_rows(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = RESULTS / "execution_readiness_history.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []

    rows: List[Dict[str, Any]] = []
    prev_gov: Optional[bool] = None
    for _, r in df.tail(500).iterrows():
        gov = bool(r.get("governance")) if pd.notna(r.get("governance")) else None
        if gov is None:
            continue
        changed = prev_gov is not None and gov != prev_gov
        if changed or len(rows) == 0:
            rows.append(
                {
                    "timestamp": r.get("timestamp"),
                    "event": "GOVERNANCE_PASS" if gov else "GOVERNANCE_BLOCK",
                    "governance_authorized": gov,
                    "readiness_status": r.get("readiness_status"),
                    "watchdog_ok": bool(r.get("watchdog")) if pd.notna(r.get("watchdog")) else None,
                    "broker_ok": bool(r.get("broker")) if pd.notna(r.get("broker")) else None,
                }
            )
        prev_gov = gov

    return rows[-50:]


def _executive_report(
    package: Dict[str, Any],
    blocking: Dict[str, Any],
    cps: Dict[str, Any],
    contradiction: Dict[str, Any],
    scores: Dict[str, Any],
    chain: List[Dict[str, Any]],
) -> Dict[str, Any]:
    primary = blocking.get("primary_blocker") or {}
    return {
        "report_type": "GOVERNANCE_AUTHORIZATION_INTELLIGENCE_REPORT",
        "current_authorization_status": package.get("current_status"),
        "block_intentional": blocking.get("block_intentional"),
        "primary_blocking_reason": primary.get("blocking_reason"),
        "governance_decision_chain_summary": [
            {"layer": l["layer"], "status": l["status"]} for l in chain
        ],
        "evidence_missing": [r["requirement"] for r in package.get("required_evidence", [])[:5]],
        "recovery_requirements_count": len(package.get("required_evidence") or []),
        "estimated_path_to_authorization": cps.get("projected_governance_recovery"),
        "authorization_probability_pct": cps.get("recovery_probability_score"),
        "expected_yellow_date": cps.get("expected_yellow_date"),
        "expected_green_date": cps.get("expected_green_date"),
        "contradiction_classification": contradiction.get("classification"),
        "institutional_learning": (
            "Governance block is protective discipline, not infrastructure failure. "
            "ORANGE escalation reflects concentration and execution risk; authorization restores "
            "when CPS trajectory reaches YELLOW/GREEN thresholds with supporting evidence."
        ),
        "executive_questions": {
            "why_blocked": primary.get("blocking_reason") or blocking.get("root_cause"),
            "intentional_or_system_problem": (
                "INTENTIONAL — EXPECTED_PROTECTIVE_BLOCK"
                if blocking.get("block_intentional")
                else "Investigate — may be operational or configuration issue"
            ),
            "evidence_required": [
                r.get("evidence_needed") for r in package.get("required_evidence", [])[:3]
            ],
            "must_change_before_approval": primary.get("blocking_reason"),
            "future_authorization_likelihood": cps.get("recovery_probability"),
            "expected_timeline_days_yellow": cps.get("days_to_yellow"),
        },
        "recommended_actions": [
            r["requirement"]
            for r in package.get("required_evidence", [])
            if r.get("priority") == "CRITICAL"
        ][:4],
    }


def run_analysis() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    ctx = _load_context()

    chain = _build_decision_chain(ctx)
    blocking = _blocking_analysis(chain, ctx)
    cps = _cps_alignment(ctx)
    package = _authorization_package(ctx, chain)
    contradiction = _contradiction_detection(ctx)
    scores = _intelligence_scores(ctx, chain, blocking, package)
    history = _decision_history_rows(ctx)
    executive = _executive_report(package, blocking, cps, contradiction, scores, chain)

    report = {
        "generated_at": ts,
        "phase": "199",
        "mode": "governance_authorization_intelligence",
        "disclaimer": (
            "Governance intelligence only — does NOT modify governance rules, execution "
            "permissions, trading strategy, broker integrations, or risk controls."
        ),
        "authorization_status": package,
        "decision_chain": chain,
        "blocking_analysis": blocking,
        "cps_alignment": cps,
        "contradiction_detection": contradiction,
        "intelligence_scores": scores,
        "executive_recommendation": executive,
        "summary": {
            "authorization_status": package.get("current_status"),
            "block_intentional": blocking.get("block_intentional"),
            "primary_blocker": (blocking.get("primary_blocker") or {}).get("layer"),
            "escalation_state": cps.get("escalation_state"),
            "contradiction": contradiction.get("classification"),
            "intelligence_classification": scores.get("classification"),
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    chain_rows = blocking.get("blockers") or []
    if chain_rows:
        pd.DataFrame(chain_rows).to_csv(OUT_CHAIN, index=False, encoding="utf-8")
    else:
        OUT_CHAIN.write_text("sequence,layer,blocking_reason,classification\n", encoding="utf-8")

    reqs = package.get("required_evidence") or []
    pd.DataFrame(reqs).to_csv(OUT_REQUIREMENTS, index=False, encoding="utf-8")

    if history:
        pd.DataFrame(history).to_csv(OUT_DECISION_HISTORY, index=False, encoding="utf-8")
    else:
        OUT_DECISION_HISTORY.write_text(
            "timestamp,event,governance_authorized,readiness_status\n", encoding="utf-8"
        )

    _log(
        f"status={package.get('current_status')} intentional={blocking.get('block_intentional')} "
        f"escalation={cps.get('escalation_state')} contradiction={contradiction.get('classification')}"
    )
    return report


def main() -> int:
    run_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
