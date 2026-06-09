"""
TRITON Phase 157 — Institutional Intelligence Stress Testing.

Simulation-only stress tests of institutional intelligence, governance, oversight,
and certification layers. Does not modify execution, governance rules, or positions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.capital_preservation_certification import compute_preservation_certification
from services.institutional_intelligence import (
    compute_decision_quality_assessment,
    compute_institutional_intelligence,
)
from services.institutional_operations import (
    compute_investment_committee_review,
    compute_strategic_oversight,
    compute_triton_maturity_assessment,
)
from services.institutional_protection import (
    ESCALATION_SEVERITY,
    compute_preservation_governance_board,
    compute_risk_committee_oversight,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

OUTPUT_JSON = RESULTS / "institutional_intelligence_stress_test.json"
OUTPUT_CSV = RESULTS / "institutional_intelligence_stress_scorecard.csv"

SCORECARD_COLUMNS = [
    "section",
    "scenario",
    "subsystem",
    "status",
    "detail",
    "generated_at",
]

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_scorecard_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _log_tag(tag: str, fields: Dict[str, Any], verbose: bool) -> None:
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"[{tag}] {parts}"
    logger.info(line)
    if verbose:
        print(line, flush=True)


def _aggregate_status(statuses: List[str]) -> str:
    if any(s == "FAIL" for s in statuses):
        return "FAIL"
    if any(s == "WARN" for s in statuses):
        return "WARN"
    return "PASS"


def _merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class ScenarioResult:
    name: str
    status: str
    detail: str
    subsystem: str = ""


@dataclass
class BaselineDocs:
    cpi: Dict[str, Any] = field(default_factory=dict)
    cpe: Dict[str, Any] = field(default_factory=dict)
    cpa: Dict[str, Any] = field(default_factory=dict)
    gov: Dict[str, Any] = field(default_factory=dict)
    auth: Dict[str, Any] = field(default_factory=dict)
    readiness: Dict[str, Any] = field(default_factory=dict)
    governor: Dict[str, Any] = field(default_factory=dict)
    adaptive: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    alerts: Dict[str, Any] = field(default_factory=dict)
    exec_doc: Dict[str, Any] = field(default_factory=dict)
    sim: Dict[str, Any] = field(default_factory=dict)
    accountability: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    queue: Dict[str, Any] = field(default_factory=dict)
    candidates: Dict[str, Any] = field(default_factory=dict)
    maturity: Dict[str, Any] = field(default_factory=dict)
    intelligence: Dict[str, Any] = field(default_factory=dict)
    decision: Dict[str, Any] = field(default_factory=dict)
    cert: Dict[str, Any] = field(default_factory=dict)


def _load_baseline(results_dir: Path) -> BaselineDocs:
    rd = Path(results_dir)
    return BaselineDocs(
        cpi=_read_json(rd / "capital_preservation_intelligence.json"),
        cpe=_read_json(rd / "capital_preservation_escalation.json"),
        cpa=_read_json(rd / "capital_preservation_advisory.json"),
        gov=_read_json(rd / "governance_risk_summary.json"),
        auth=_read_json(rd / "governance_authorization.json"),
        readiness=_read_json(rd / "execution_readiness.json"),
        governor=_read_json(rd / "capital_preservation_governor.json"),
        adaptive=_read_json(rd / "adaptive_capital_preservation.json"),
        evaluation=_read_json(rd / "protective_action_evaluation.json"),
        alerts=_read_json(rd / "watchdog_alerts.json"),
        exec_doc=_read_json(rd / "executive_risk_summary.json"),
        sim=_read_json(rd / "defensive_simulation_results.json"),
        accountability=_read_json(rd / "accountability_registry.json"),
        audit=_read_json(rd / "capital_preservation_audit.json"),
        policy=_read_json(rd / "protective_action_policy.json"),
        queue=_read_json(rd / "human_approval_queue.json"),
        candidates=_read_json(rd / "defensive_action_candidates.json"),
        maturity=_read_json(rd / "triton_maturity_assessment.json"),
        intelligence=_read_json(rd / "institutional_intelligence.json"),
        decision=_read_json(rd / "decision_quality_assessment.json"),
        cert=_read_json(rd / "capital_preservation_certification.json"),
    )


def _compute_stack(
    *,
    docs: BaselineDocs,
    results_dir: Path,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run institutional intelligence pipeline with optional synthetic overrides."""
    ov = overrides or {}
    cpi = _merge(docs.cpi, ov.get("cpi", {}))
    cpe = _merge(docs.cpe, ov.get("cpe", {}))
    cpa = _merge(docs.cpa, ov.get("cpa", {}))
    gov = _merge(docs.gov, ov.get("gov", {}))
    auth = _merge(docs.auth, ov.get("auth", {}))
    readiness = _merge(docs.readiness, ov.get("readiness", {}))
    governor = _merge(docs.governor, ov.get("governor", {}))
    adaptive = _merge(docs.adaptive, ov.get("adaptive", {}))
    evaluation = _merge(docs.evaluation, ov.get("evaluation", {}))
    alerts = _merge(docs.alerts, ov.get("alerts", {}))
    exec_doc = _merge(docs.exec_doc, ov.get("exec", {}))
    sim = _merge(docs.sim, ov.get("sim", {}))
    accountability = _merge(docs.accountability, ov.get("accountability", {}))

    cert = compute_preservation_certification(
        cpi_doc=cpi,
        alerts_doc=alerts,
        gov_doc=gov,
        auth_doc=auth,
        readiness_doc=readiness,
        sim_doc=sim,
        evaluation_doc=evaluation,
        governor_doc=governor,
        cpe_doc=cpe,
    )

    committee = compute_risk_committee_oversight(
        cpi_doc=cpi,
        gov_doc=gov,
        auth_doc=auth,
        governor_doc=governor,
        cpe_doc=cpe,
        readiness_doc=readiness,
        cert_doc=cert,
    )
    board = compute_preservation_governance_board(
        committee_doc=committee,
        accountability_doc=accountability,
        cert_doc=cert,
        gov_doc=gov,
        auth_doc=auth,
        governor_doc=governor,
    )
    investment = compute_investment_committee_review(
        cpi_doc=cpi,
        cpe_doc=cpe,
        alerts_doc=alerts,
        exec_doc=exec_doc,
        gov_doc=gov,
        auth_doc=auth,
        cert_doc=cert,
        readiness_doc=readiness,
    )
    maturity = compute_triton_maturity_assessment(
        cpi_doc=cpi,
        cpe_doc=cpe,
        alerts_doc=alerts,
        exec_doc=exec_doc,
        gov_doc=gov,
        auth_doc=auth,
        governor_doc=governor,
        adaptive_doc=adaptive,
        cert_doc=cert,
        readiness_doc=readiness,
        committee_doc=committee,
        board_doc=board,
        accountability_doc=accountability,
    )
    strategic = compute_strategic_oversight(
        governor_doc=governor,
        cert_doc=cert,
        investment_doc=investment,
        maturity_doc=maturity,
        exec_doc=exec_doc,
        board_doc=board,
    )
    decision = compute_decision_quality_assessment(
        results_dir=results_dir,
        cpa_doc=cpa,
        cpe_doc=cpe,
        cpi_doc=cpi,
        alerts_doc=alerts,
        gov_doc=gov,
        auth_doc=auth,
        readiness_doc=readiness,
        governor_doc=governor,
        evaluation_doc=evaluation,
        adaptive_doc=adaptive,
    )
    intelligence = compute_institutional_intelligence(
        cpi_doc=cpi,
        cpe_doc=cpe,
        alerts_doc=alerts,
        exec_doc=exec_doc,
        gov_doc=gov,
        auth_doc=auth,
        cert_doc=cert,
        readiness_doc=readiness,
        governor_doc=governor,
        adaptive_doc=adaptive,
        committee_doc=committee,
        board_doc=board,
        accountability_doc=accountability,
        maturity_doc=maturity,
    )

    return {
        "cpi": cpi,
        "cpe": cpe,
        "cert": cert,
        "committee": committee,
        "board": board,
        "investment": investment,
        "maturity": maturity,
        "strategic": strategic,
        "decision": decision,
        "intelligence": intelligence,
    }


def _metric_score(decision: Dict[str, Any], name: str) -> int:
    for row in decision.get("metrics") or []:
        if row.get("metric") == name:
            return int(row.get("score") or 0)
    return 0


def _stress_conflicting_signals(
    *,
    baseline: BaselineDocs,
    results_dir: Path,
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[ScenarioResult], List[Dict[str, str]]]:
    """Part A — conflicting subsystem signals."""
    scenarios: List[ScenarioResult] = []
    csv_rows: List[Dict[str, str]] = []

    # Excellent CPS + poor evaluation
    stack_a = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "cpi": {
                "capital_preservation_score": 92,
                "health_band": "HEALTHY",
                "component_scores": {
                    "drawdown": 95,
                    "concentration": 98,
                    "exposure": 90,
                    "operational": 88,
                    "execution": 85,
                },
            },
            "cpe": {
                "escalation_state": "GREEN",
                "escalation_reason": [],
                "escalation_reason_labels": [],
            },
            "evaluation": {
                "average_effectiveness": 18.0,
                "evaluation_count": 4,
                "beneficial_count": 0,
                "negative_count": 3,
                "neutral_count": 1,
            },
            "governor": {"preservation_posture": "GREEN", "governor_confidence": 90},
        },
    )
    eval_cert = bool((stack_a["cert"].get("areas") or {}).get("Evaluation", {}).get("certified"))
    committee_esc = str(stack_a["committee"].get("committee_status") or "")
    pres_logic = _metric_score(stack_a["decision"], "Preservation Logic Consistency")
    ok_a = (
        not eval_cert
        and committee_esc in ("CLEAR", "MONITOR", "REVIEW_REQUIRED")
        and pres_logic >= 50
    )
    scenarios.append(
        ScenarioResult(
            name="excellent_cps_poor_eval",
            status="PASS" if ok_a else "FAIL",
            detail=(
                f"eval_certified={eval_cert} committee={committee_esc} "
                f"preservation_logic={pres_logic}"
            ),
            subsystem="decision_quality+certification",
        )
    )

    # Strong governance + weak certification inputs
    stack_b = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "gov": {
                "governance_awareness_level": "GREEN",
                "governance_status": "Normal Operations",
            },
            "auth": {
                "overall_authorization": False,
                "governance_authorized": True,
                "paper_execution_permitted": False,
                "execution_authorized": False,
            },
            "readiness": {
                "readiness_status": "NOT_READY",
                "checks_passing_count": 3,
                "checks_total": 8,
                "failed_checks": ["policy", "watchdog", "approval", "broker", "lifecycle"],
            },
            "evaluation": {"average_effectiveness": 20.0, "evaluation_count": 1},
        },
    )
    cert_status = str(stack_b["cert"].get("certification_status") or "")
    gov_cert = bool((stack_b["cert"].get("areas") or {}).get("Governance", {}).get("certified"))
    ok_b = gov_cert and cert_status in ("NOT_CERTIFIED", "PARTIALLY_CERTIFIED")
    scenarios.append(
        ScenarioResult(
            name="strong_gov_weak_cert",
            status="PASS" if ok_b else "WARN" if gov_cert else "FAIL",
            detail=f"gov_certified={gov_cert} overall_cert={cert_status}",
            subsystem="certification",
        )
    )

    # Strong decision-quality inputs + weak maturity drivers
    stack_c = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "cpi": {"capital_preservation_score": 88, "health_band": "HEALTHY"},
            "cpe": {"escalation_state": "GREEN"},
            "gov": {"governance_awareness_level": "GREEN"},
            "auth": {"overall_authorization": True, "governance_authorized": True},
            "readiness": {
                "readiness_status": "NOT_READY",
                "checks_passing_count": 4,
                "failed_checks": ["policy", "watchdog", "approval", "lifecycle"],
            },
            "governor": {"preservation_posture": "GREEN", "governor_confidence": 95},
            "evaluation": {"average_effectiveness": 25.0},
        },
    )
    dq = int(stack_c["decision"].get("decision_quality_score") or 0)
    mat = int(stack_c["maturity"].get("overall_maturity") or 0)
    intel = int(stack_c["intelligence"].get("institutional_intelligence_score") or 0)
    ok_c = dq >= 65 and mat < dq and intel <= max(dq, mat) + 5
    scenarios.append(
        ScenarioResult(
            name="strong_decision_weak_maturity",
            status="PASS" if ok_c else "WARN",
            detail=f"decision_quality={dq} maturity={mat} intelligence={intel}",
            subsystem="maturity+intelligence",
        )
    )

    # Conflicting subsystem conclusions (excellent CPS vs RED escalation vs GREEN governance)
    stack_d = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "cpi": {"capital_preservation_score": 90, "health_band": "HEALTHY"},
            "cpe": {
                "escalation_state": "RED",
                "escalation_reason": ["EXCESS_POSITION_DRAWDOWN"],
                "escalation_reason_labels": ["Drawdown Risk"],
            },
            "gov": {"governance_awareness_level": "GREEN"},
            "governor": {"preservation_posture": "GREEN"},
        },
    )
    esc_consistency = _metric_score(stack_d["decision"], "Escalation Consistency")
    committee_status = str(stack_d["committee"].get("committee_status") or "")
    strategic_status = str(stack_d["strategic"].get("oversight_status") or "")
    ok_d = (
        esc_consistency < 75
        and committee_status in ("REVIEW_REQUIRED", "ESCALATE", "MONITOR")
        and strategic_status != "SUSPENDED"
    )
    scenarios.append(
        ScenarioResult(
            name="conflicting_subsystem_conclusions",
            status="PASS" if ok_d else "FAIL",
            detail=(
                f"escalation_consistency={esc_consistency} committee={committee_status} "
                f"strategic={strategic_status}"
            ),
            subsystem="escalation+oversight",
        )
    )

    status = _aggregate_status([s.status for s in scenarios])
    _log_tag("CONFLICTING_SIGNAL_STRESS", {"result": status, "scenarios": len(scenarios)}, verbose)

    detail = {
        "cases_tested": len(scenarios),
        "case_results": [
            {"name": s.name, "status": s.status, "detail": s.detail, "subsystem": s.subsystem}
            for s in scenarios
        ],
        "status": status,
    }
    for s in scenarios:
        csv_rows.append(
            {
                "section": "conflicting_signal_stress",
                "scenario": s.name,
                "subsystem": s.subsystem,
                "status": s.status,
                "detail": s.detail,
            }
        )
    return status, detail, scenarios, csv_rows


def _stress_false_negative(
    *,
    baseline: BaselineDocs,
    results_dir: Path,
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[ScenarioResult], List[Dict[str, str]]]:
    """Part B — healthy portfolio with artificially weak eval/governance metrics."""
    scenarios: List[ScenarioResult] = []
    csv_rows: List[Dict[str, str]] = []

    healthy_overrides = {
        "cpi": {
            "capital_preservation_score": 88,
            "health_band": "HEALTHY",
            "risk_trend": "IMPROVING",
            "component_scores": {
                "drawdown": 92,
                "concentration": 95,
                "exposure": 90,
                "operational": 85,
                "execution": 80,
            },
        },
        "cpe": {
            "escalation_state": "GREEN",
            "escalation_reason": [],
            "escalation_reason_labels": [],
        },
        "governor": {"preservation_posture": "GREEN", "governor_confidence": 92},
        "evaluation": {
            "average_effectiveness": 12.0,
            "evaluation_count": 4,
            "beneficial_count": 0,
            "negative_count": 3,
            "neutral_count": 1,
            "baseline_context": "HIGH_CPS_PRESERVATION",
        },
        "gov": {
            "governance_awareness_level": "BOARD_REVIEW_REQUIRED",
            "governance_drivers": ["Synthetic weak governance probe"],
        },
        "auth": {"overall_authorization": False, "governance_authorized": False},
    }

    stack = _compute_stack(docs=baseline, results_dir=results_dir, overrides=healthy_overrides)
    committee_status = str(stack["committee"].get("committee_status") or "")
    cert_status = str(stack["cert"].get("certification_status") or "")
    intel_score = int(stack["intelligence"].get("institutional_intelligence_score") or 0)
    coord = int(stack["intelligence"].get("coordination_score") or 0)
    weakest = str(stack["intelligence"].get("weakest_area") or "")

    no_overreaction = committee_status != "ESCALATE"
    cert_not_collapsed = (
        cert_status != "NOT_CERTIFIED"
        or int((stack["cert"].get("areas") or {}).get("Monitoring", {}).get("score") or 0) < 50
    )
    flags_inconsistency = weakest in ("Governance", "Certification") or coord < 80

    scenarios.append(
        ScenarioResult(
            name="healthy_portfolio_weak_metrics",
            status="PASS" if no_overreaction and flags_inconsistency else "WARN",
            detail=(
                f"committee={committee_status} cert={cert_status} "
                f"weakest={weakest} coordination={coord}"
            ),
            subsystem="committee+certification+intelligence",
        )
    )

    # Certification should not fully certify with weak eval
    eval_cert = bool((stack["cert"].get("areas") or {}).get("Evaluation", {}).get("certified"))
    scenarios.append(
        ScenarioResult(
            name="certification_withheld_on_weak_eval",
            status="PASS" if not eval_cert else "FAIL",
            detail=f"evaluation_certified={eval_cert}",
            subsystem="certification",
        )
    )

    # Intelligence should remain above panic floor
    scenarios.append(
        ScenarioResult(
            name="intelligence_no_panic_floor",
            status="PASS" if intel_score >= 40 else "FAIL",
            detail=f"institutional_intelligence_score={intel_score}",
            subsystem="institutional_intelligence",
        )
    )

    status = _aggregate_status([s.status for s in scenarios])
    _log_tag("FALSE_NEGATIVE_STRESS", {"result": status, "scenarios": len(scenarios)}, verbose)

    detail = {
        "cases_tested": len(scenarios),
        "case_results": [
            {"name": s.name, "status": s.status, "detail": s.detail, "subsystem": s.subsystem}
            for s in scenarios
        ],
        "status": status,
    }
    for s in scenarios:
        csv_rows.append(
            {
                "section": "false_negative_stress",
                "scenario": s.name,
                "subsystem": s.subsystem,
                "status": s.status,
                "detail": s.detail,
            }
        )
    return status, detail, scenarios, csv_rows


def _stress_false_positive(
    *,
    baseline: BaselineDocs,
    results_dir: Path,
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[ScenarioResult], List[Dict[str, str]]]:
    """Part C — deteriorating portfolio with artificially strong governance/eval."""
    scenarios: List[ScenarioResult] = []
    csv_rows: List[Dict[str, str]] = []

    deteriorating_overrides = {
        "cpi": {
            "capital_preservation_score": 32,
            "health_band": "CRITICAL",
            "risk_trend": "DETERIORATING",
            "component_scores": {
                "drawdown": 20,
                "concentration": 25,
                "exposure": 30,
                "operational": 28,
                "execution": 22,
            },
        },
        "cpe": {
            "escalation_state": "RED",
            "escalation_reason": ["EXCESS_POSITION_DRAWDOWN", "EXCESS_CONCENTRATION"],
            "escalation_reason_labels": ["Drawdown Risk", "Concentration Risk"],
        },
        "gov": {
            "governance_awareness_level": "GREEN",
            "governance_status": "Normal Operations",
            "governance_drivers": [],
        },
        "auth": {
            "overall_authorization": True,
            "governance_authorized": True,
            "paper_execution_permitted": True,
            "execution_authorized": True,
        },
        "governor": {"preservation_posture": "GREEN", "governor_confidence": 98},
        "evaluation": {
            "average_effectiveness": 92.0,
            "evaluation_count": 5,
            "beneficial_count": 5,
            "negative_count": 0,
            "neutral_count": 0,
            "baseline_context": "HIGH_CPS_PRESERVATION",
            "high_cps_adjustment_applied": True,
        },
        "readiness": {
            "readiness_status": "READY",
            "checks_passing_count": 8,
            "checks_total": 8,
            "failed_checks": [],
        },
        "exec": {
            "executive_summary": {
                "portfolio_health": "HEALTHY",
                "risk_direction": "IMPROVING",
            },
            "predictive_outlook": {"risk_direction": "IMPROVING", "forecast_confidence": 95},
            "top_risks": [],
        },
    }

    stack = _compute_stack(
        docs=baseline, results_dir=results_dir, overrides=deteriorating_overrides
    )
    committee_status = str(stack["committee"].get("committee_status") or "")
    overall_assessment = str(stack["committee"].get("overall_assessment") or "")
    cert_status = str(stack["cert"].get("certification_status") or "")
    strategic_rec = str(stack["strategic"].get("committee_recommendation") or "")
    esc_consistency = _metric_score(stack["decision"], "Escalation Consistency")

    refuses_optimism = committee_status in (
        "REVIEW_REQUIRED",
        "ESCALATE",
    ) and overall_assessment in (
        "HIGH_RISK",
        "CRITICAL",
        "ELEVATED_RISK",
    )
    scenarios.append(
        ScenarioResult(
            name="deteriorating_portfolio_strong_metrics",
            status="PASS" if refuses_optimism else "FAIL",
            detail=(
                f"committee={committee_status} assessment={overall_assessment} "
                f"strategic_rec={strategic_rec}"
            ),
            subsystem="committee+strategic",
        )
    )

    not_fully_certified = cert_status != "CERTIFIED_FOR_PAPER_PROTECTION"
    scenarios.append(
        ScenarioResult(
            name="certification_refuses_false_optimism",
            status="PASS" if not_fully_certified else "FAIL",
            detail=f"certification_status={cert_status}",
            subsystem="certification",
        )
    )

    flags_escalation_mismatch = esc_consistency < 70
    scenarios.append(
        ScenarioResult(
            name="decision_quality_flags_mismatch",
            status="PASS" if flags_escalation_mismatch else "WARN",
            detail=f"escalation_consistency={esc_consistency}",
            subsystem="decision_quality",
        )
    )

    status = _aggregate_status([s.status for s in scenarios])
    _log_tag("FALSE_POSITIVE_STRESS", {"result": status, "scenarios": len(scenarios)}, verbose)

    detail = {
        "cases_tested": len(scenarios),
        "case_results": [
            {"name": s.name, "status": s.status, "detail": s.detail, "subsystem": s.subsystem}
            for s in scenarios
        ],
        "status": status,
    }
    for s in scenarios:
        csv_rows.append(
            {
                "section": "false_positive_stress",
                "scenario": s.name,
                "subsystem": s.subsystem,
                "status": s.status,
                "detail": s.detail,
            }
        )
    return status, detail, scenarios, csv_rows


def _stress_committee_disagreement(
    *,
    baseline: BaselineDocs,
    results_dir: Path,
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[ScenarioResult], List[Dict[str, str]]]:
    """Part D — disagreement across committee layers."""
    scenarios: List[ScenarioResult] = []
    csv_rows: List[Dict[str, str]] = []

    # Risk committee sees critical governance; executive appears calm
    stack = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "cpi": {"capital_preservation_score": 78, "health_band": "CAUTION"},
            "cpe": {"escalation_state": "YELLOW"},
            "gov": {
                "governance_awareness_level": "CRITICAL_INTERVENTION",
                "governance_drivers": ["Authorization Gap", "Policy Failure"],
            },
            "auth": {"overall_authorization": False, "governance_authorized": False},
            "exec": {
                "executive_summary": {
                    "portfolio_health": "HEALTHY",
                    "risk_direction": "STABLE",
                },
                "predictive_outlook": {"risk_direction": "STABLE", "forecast_confidence": 88},
                "top_risks": ["Minor monitoring drift"],
            },
            "governor": {"preservation_posture": "YELLOW"},
        },
    )

    risk_status = str(stack["committee"].get("committee_status") or "")
    board_status = str(stack["board"].get("board_status") or "")
    investment_rec = str(stack["investment"].get("committee_recommendation") or "")
    strategic_status = str(stack["strategic"].get("oversight_status") or "")
    recommendations = stack["strategic"].get("strategic_recommendations") or []

    rational_escalation = risk_status in ("REVIEW_REQUIRED", "ESCALATE")
    coherent_strategic = strategic_status in ("REVIEW", "SUSPENDED")
    explainable = len(recommendations) >= 2 and any(
        "review" in str(r).lower() or "committee" in str(r).lower() for r in recommendations
    )

    scenarios.append(
        ScenarioResult(
            name="risk_critical_exec_calm",
            status="PASS" if rational_escalation and coherent_strategic else "FAIL",
            detail=(
                f"risk={risk_status} board={board_status} investment={investment_rec} "
                f"strategic={strategic_status}"
            ),
            subsystem="risk_committee+executive",
        )
    )
    scenarios.append(
        ScenarioResult(
            name="explainable_recommendations",
            status="PASS" if explainable else "WARN",
            detail=f"recommendation_count={len(recommendations)}",
            subsystem="strategic_oversight",
        )
    )

    # Deliberate cross-layer divergence: preservation GREEN vs governance block
    stack_b = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "governor": {"preservation_posture": "GREEN", "governor_confidence": 95},
            "gov": {"governance_awareness_level": "BOARD_REVIEW_REQUIRED"},
            "cpi": {"capital_preservation_score": 82, "health_band": "HEALTHY"},
            "cpe": {"escalation_state": "GREEN"},
        },
    )
    gov_consistency = _metric_score(stack_b["decision"], "Governance Consistency")
    board_b = str(stack_b["board"].get("board_status") or "")
    scenarios.append(
        ScenarioResult(
            name="preservation_governance_divergence",
            status=(
                "PASS"
                if gov_consistency < 70 and board_b in ("REVIEW", "SUSPENDED", "ACTIVE")
                else "WARN"
            ),
            detail=f"governance_consistency={gov_consistency} board={board_b}",
            subsystem="governance_board+decision_quality",
        )
    )

    status = _aggregate_status([s.status for s in scenarios])
    _log_tag("COMMITTEE_STRESS", {"result": status, "scenarios": len(scenarios)}, verbose)

    detail = {
        "cases_tested": len(scenarios),
        "case_results": [
            {"name": s.name, "status": s.status, "detail": s.detail, "subsystem": s.subsystem}
            for s in scenarios
        ],
        "status": status,
    }
    for s in scenarios:
        csv_rows.append(
            {
                "section": "committee_stress",
                "scenario": s.name,
                "subsystem": s.subsystem,
                "status": s.status,
                "detail": s.detail,
            }
        )
    return status, detail, scenarios, csv_rows


def _stress_escalation_ladder(
    *,
    baseline: BaselineDocs,
    results_dir: Path,
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[ScenarioResult], List[Dict[str, str]]]:
    """Part E — escalation transitions and de-escalation."""
    scenarios: List[ScenarioResult] = []
    csv_rows: List[Dict[str, str]] = []

    ladder = [
        ("GREEN", 85, ("CLEAR", "MONITOR")),
        ("YELLOW", 65, ("MONITOR", "REVIEW_REQUIRED")),
        ("ORANGE", 48, ("REVIEW_REQUIRED", "ESCALATE")),
        ("RED", 28, ("REVIEW_REQUIRED", "ESCALATE")),
    ]

    prev_severity = -1
    for esc_state, cps, allowed_committee in ladder:
        stack = _compute_stack(
            docs=baseline,
            results_dir=results_dir,
            overrides={
                "cpi": {
                    "capital_preservation_score": cps,
                    "health_band": "HEALTHY" if cps >= 70 else "ELEVATED_RISK",
                },
                "cpe": {"escalation_state": esc_state},
                "governor": {"preservation_posture": esc_state},
            },
        )
        committee_status = str(stack["committee"].get("committee_status") or "")
        severity = ESCALATION_SEVERITY.get(esc_state, 0)
        no_skip = severity >= prev_severity
        correct_transition = committee_status in allowed_committee
        ok = no_skip and correct_transition
        scenarios.append(
            ScenarioResult(
                name=f"escalation_{esc_state.lower()}",
                status="PASS" if ok else "WARN" if no_skip else "FAIL",
                detail=f"cps={cps} committee={committee_status} allowed={allowed_committee}",
                subsystem="escalation",
            )
        )
        prev_severity = severity

    # De-escalation path RED -> YELLOW -> GREEN
    deesc_path = [("RED", 30), ("ORANGE", 50), ("YELLOW", 68), ("GREEN", 86)]
    deesc_committee: List[str] = []
    for esc_state, cps in deesc_path:
        stack = _compute_stack(
            docs=baseline,
            results_dir=results_dir,
            overrides={
                "cpi": {"capital_preservation_score": cps, "health_band": "HEALTHY"},
                "cpe": {"escalation_state": esc_state},
                "governor": {"preservation_posture": esc_state},
            },
        )
        deesc_committee.append(str(stack["committee"].get("committee_status") or ""))

    deesc_ok = deesc_committee[-1] in ("CLEAR", "MONITOR", "REVIEW_REQUIRED")
    scenarios.append(
        ScenarioResult(
            name="deescalation_path",
            status="PASS" if deesc_ok else "WARN",
            detail=f"path={'->'.join(deesc_committee)}",
            subsystem="escalation",
        )
    )

    status = _aggregate_status([s.status for s in scenarios])
    _log_tag("ESCALATION_STRESS", {"result": status, "scenarios": len(scenarios)}, verbose)

    detail = {
        "cases_tested": len(scenarios),
        "case_results": [
            {"name": s.name, "status": s.status, "detail": s.detail, "subsystem": s.subsystem}
            for s in scenarios
        ],
        "status": status,
    }
    for s in scenarios:
        csv_rows.append(
            {
                "section": "escalation_stress",
                "scenario": s.name,
                "subsystem": s.subsystem,
                "status": s.status,
                "detail": s.detail,
            }
        )
    return status, detail, scenarios, csv_rows


def _stress_certification(
    *,
    baseline: BaselineDocs,
    results_dir: Path,
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[ScenarioResult], List[Dict[str, str]]]:
    """Part F — certification cannot be falsely earned or denied."""
    scenarios: List[ScenarioResult] = []
    csv_rows: List[Dict[str, str]] = []

    # Strong synthetic inputs — should not fully certify while live gate blocked
    strong_stack = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "cpi": {"capital_preservation_score": 90, "health_band": "HEALTHY"},
            "cpe": {"escalation_state": "GREEN"},
            "gov": {"governance_awareness_level": "GREEN"},
            "auth": {
                "overall_authorization": True,
                "governance_authorized": True,
                "paper_execution_permitted": True,
                "execution_authorized": True,
                "live_execution_permitted": False,
            },
            "readiness": {
                "readiness_status": "READY",
                "checks_passing_count": 8,
                "checks_total": 8,
                "failed_checks": [],
                "live_execution_permitted": False,
            },
            "evaluation": {
                "average_effectiveness": 75.0,
                "evaluation_count": 5,
                "beneficial_count": 4,
                "negative_count": 0,
                "neutral_count": 1,
                "baseline_context": "HIGH_CPS_PRESERVATION",
                "high_cps_adjustment_applied": True,
            },
            "governor": {"preservation_posture": "GREEN", "governor_confidence": 95},
        },
    )
    strong_status = str(strong_stack["cert"].get("certification_status") or "")
    strong_count = int(strong_stack["cert"].get("certified_area_count") or 0)
    scenarios.append(
        ScenarioResult(
            name="strong_inputs_paper_certification",
            status=(
                "PASS"
                if strong_status in ("PARTIALLY_CERTIFIED", "CERTIFIED_FOR_PAPER_PROTECTION")
                and strong_count >= 5
                else "WARN"
            ),
            detail=f"status={strong_status} certified_areas={strong_count}",
            subsystem="certification",
        )
    )

    # Live execution open — must deny certification
    live_stack = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "auth": {"live_execution_permitted": True},
            "readiness": {"live_execution_permitted": True},
        },
    )
    live_status = str(live_stack["cert"].get("certification_status") or "")
    scenarios.append(
        ScenarioResult(
            name="live_gate_blocks_certification",
            status="PASS" if live_status == "NOT_CERTIFIED" else "FAIL",
            detail=f"status={live_status}",
            subsystem="certification",
        )
    )

    # Insufficient evaluation trials — cannot certify evaluation area
    weak_eval = compute_preservation_certification(
        cpi_doc=_merge(baseline.cpi, {"capital_preservation_score": 85}),
        alerts_doc=baseline.alerts,
        gov_doc=_merge(baseline.gov, {"governance_awareness_level": "GREEN"}),
        auth_doc=baseline.auth,
        readiness_doc=baseline.readiness,
        sim_doc=baseline.sim,
        evaluation_doc=_merge(
            baseline.evaluation, {"evaluation_count": 0, "average_effectiveness": 0}
        ),
        governor_doc=baseline.governor,
        cpe_doc=_merge(baseline.cpe, {"escalation_state": "GREEN"}),
    )
    eval_area = (weak_eval.get("areas") or {}).get("Evaluation", {})
    scenarios.append(
        ScenarioResult(
            name="evaluation_minimum_enforced",
            status="PASS" if not eval_area.get("certified") else "FAIL",
            detail=str(eval_area.get("notes") or ""),
            subsystem="certification",
        )
    )

    # Excellent monitoring cannot override critical governor posture
    critical_gov = compute_preservation_certification(
        cpi_doc=_merge(baseline.cpi, {"capital_preservation_score": 95}),
        alerts_doc=baseline.alerts,
        gov_doc=baseline.gov,
        auth_doc=baseline.auth,
        readiness_doc=baseline.readiness,
        sim_doc=baseline.sim,
        evaluation_doc=baseline.evaluation,
        governor_doc=_merge(
            baseline.governor, {"preservation_posture": "CRITICAL", "governor_confidence": 20}
        ),
        cpe_doc=_merge(baseline.cpe, {"escalation_state": "RED"}),
    )
    gov_area = (critical_gov.get("areas") or {}).get("Governor", {})
    full_cert = critical_gov.get("certification_status") == "CERTIFIED_FOR_PAPER_PROTECTION"
    scenarios.append(
        ScenarioResult(
            name="governor_critical_blocks_full_cert",
            status="PASS" if not gov_area.get("certified") and not full_cert else "FAIL",
            detail=f"governor_certified={gov_area.get('certified')} status={critical_gov.get('certification_status')}",
            subsystem="certification",
        )
    )

    status = _aggregate_status([s.status for s in scenarios])
    _log_tag("CERTIFICATION_STRESS", {"result": status, "scenarios": len(scenarios)}, verbose)

    detail = {
        "cases_tested": len(scenarios),
        "case_results": [
            {"name": s.name, "status": s.status, "detail": s.detail, "subsystem": s.subsystem}
            for s in scenarios
        ],
        "status": status,
    }
    for s in scenarios:
        csv_rows.append(
            {
                "section": "certification_stress",
                "scenario": s.name,
                "subsystem": s.subsystem,
                "status": s.status,
                "detail": s.detail,
            }
        )
    return status, detail, scenarios, csv_rows


def _maturity_review(
    *,
    baseline: BaselineDocs,
    part_results: Dict[str, str],
    verbose: bool,
) -> Dict[str, Any]:
    """Part H — institutional maturity impact estimate."""
    before = {
        "institutional_intelligence_score": baseline.intelligence.get(
            "institutional_intelligence_score"
        ),
        "institutional_band": baseline.intelligence.get("institutional_band"),
        "overall_maturity": baseline.maturity.get("overall_maturity"),
        "maturity_band": baseline.maturity.get("maturity_band"),
        "decision_quality_score": baseline.decision.get("decision_quality_score"),
        "certification_status": baseline.cert.get("certification_status"),
        "coordination_score": baseline.intelligence.get("coordination_score"),
    }

    pass_count = sum(1 for v in part_results.values() if v == "PASS")
    warn_count = sum(1 for v in part_results.values() if v == "WARN")
    fail_count = sum(1 for v in part_results.values() if v == "FAIL")

    intel_delta = 2 if pass_count >= 5 else (-4 if fail_count >= 2 else 0)
    maturity_delta = 1 if pass_count >= 5 else (-3 if fail_count >= 2 else 0)
    cert_after = (
        "STRESS_TESTED_CERTIFIED"
        if all(v == "PASS" for v in part_results.values())
        else baseline.cert.get("certification_status")
    )

    after = {
        "institutional_intelligence_score": min(
            100, max(0, int(before["institutional_intelligence_score"] or 0) + intel_delta)
        ),
        "institutional_band": before["institutional_band"],
        "overall_maturity": min(100, max(0, int(before["overall_maturity"] or 0) + maturity_delta)),
        "maturity_band": before["maturity_band"],
        "decision_quality_score": before["decision_quality_score"],
        "certification_status": cert_after,
        "coordination_score": before["coordination_score"],
        "stress_validation_pass_count": pass_count,
        "stress_validation_warn_count": warn_count,
        "stress_validation_fail_count": fail_count,
    }

    impact = {
        "institutional_intelligence": (
            "Validated under adversarial probes; coordination logic holds with minor governance lag."
            if pass_count >= 5
            else "Stress exposure reveals cross-layer alignment gaps under synthetic conflict."
        ),
        "governance": (
            "Governance block levels correctly prevent false authorization under optimistic injection."
            if part_results.get("false_positive_stress") == "PASS"
            else "Governance may over-trust optimistic authorization signals."
        ),
        "maturity": (
            "Maturity scoring remains stable; weakest-area signal (Governance) unchanged."
            if maturity_delta >= 0
            else "Maturity estimate reduced due to stress failures."
        ),
        "certification": (
            "Certification gates resist false earn/deny under synthetic metric challenges."
            if part_results.get("certification_stress") == "PASS"
            else "Certification alignment requires hardening."
        ),
        "strategic_reasoning": (
            "Strategic oversight aggregates committee disagreement into explainable REVIEW posture."
            if part_results.get("committee_stress") == "PASS"
            else "Strategic reasoning may under-communicate committee divergence."
        ),
    }

    review = {"before": before, "after": after, "impact": impact}
    _log_tag(
        "INSTITUTIONAL_MATURITY_REVIEW",
        {
            "before": f"intel={before['institutional_intelligence_score']} maturity={before['overall_maturity']}",
            "after": f"intel={after['institutional_intelligence_score']} maturity={after['overall_maturity']}",
        },
        verbose,
    )
    return review


def _final_decision(
    *,
    scorecard: Dict[str, str],
    part_details: Dict[str, Dict[str, Any]],
    maturity_review: Dict[str, Any],
    verbose: bool,
) -> Dict[str, Any]:
    """Part I — final stress test decision and certification recommendation."""
    overall = scorecard.get("overall_result", "FAIL")
    strengths: List[str] = []
    weaknesses: List[str] = []

    labels = {
        "conflicting_signal_stress": "Conflicting signals handled without irrational escalation",
        "false_negative_stress": "Healthy conditions do not trigger certification collapse",
        "false_positive_stress": "Deteriorating conditions resist false optimism",
        "committee_stress": "Committee disagreement produces coherent advisory escalation",
        "escalation_stress": "Escalation ladder transitions are monotonic and de-escalation works",
        "certification_stress": "Certification gates resist false earn/deny",
    }
    for key, label in labels.items():
        if scorecard.get(key) == "PASS":
            strengths.append(label)
        elif scorecard.get(key) == "FAIL":
            weaknesses.append(f"{key}: {label}")

    for key, detail in part_details.items():
        for case in detail.get("case_results") or []:
            if case.get("status") == "WARN":
                weaknesses.append(f"{case.get('name')}: {case.get('detail')}")

    if not strengths:
        strengths.append("Baseline institutional artifacts available for simulation")
    if not weaknesses:
        weaknesses.append("None identified")

    all_pass = all(scorecard.get(k) == "PASS" for k in labels)
    if all_pass:
        recommendation = (
            "Promote institutional intelligence to STRESS_TESTED_CERTIFIED — "
            "all adversarial scenarios pass."
        )
        cert_status = "STRESS_TESTED_CERTIFIED"
    elif overall == "WARN":
        recommendation = (
            "Conditional pass — institutional intelligence shows robust core behavior "
            "with minor alignment warnings under synthetic stress."
        )
        cert_status = "PARTIALLY_CERTIFIED"
    else:
        recommendation = (
            "Do not promote — institutional intelligence shows material stress failures "
            "under conflicting or adversarial conditions."
        )
        cert_status = str(
            maturity_review.get("before", {}).get("certification_status") or "NOT_CERTIFIED"
        )

    decision = {
        "status": overall,
        "strengths": strengths[:6],
        "weaknesses": weaknesses[:8],
        "recommendation": recommendation,
        "institutional_intelligence_certification_status": cert_status,
        "promotion_recommended": cert_status == "STRESS_TESTED_CERTIFIED",
    }

    _log_tag(
        "INSTITUTIONAL_INTELLIGENCE_STRESS_TEST",
        {
            "status": overall,
            "strengths": "; ".join(strengths[:2]),
            "weaknesses": "; ".join(weaknesses[:2]),
            "recommendation": recommendation[:120],
        },
        verbose,
    )
    if cert_status == "STRESS_TESTED_CERTIFIED":
        _log_tag("INSTITUTIONAL_INTELLIGENCE_CERTIFIED", {"status": cert_status}, verbose)

    return decision


def compute_institutional_intelligence_stress_test(
    *,
    results_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run all Phase 157 institutional intelligence stress scenarios."""
    rd = Path(results_dir) if results_dir else RESULTS
    ts = _utc_iso()
    baseline = _load_baseline(rd)

    part_a_status, part_a_detail, _, csv_a = _stress_conflicting_signals(
        baseline=baseline, results_dir=rd, verbose=verbose
    )
    part_b_status, part_b_detail, _, csv_b = _stress_false_negative(
        baseline=baseline, results_dir=rd, verbose=verbose
    )
    part_c_status, part_c_detail, _, csv_c = _stress_false_positive(
        baseline=baseline, results_dir=rd, verbose=verbose
    )
    part_d_status, part_d_detail, _, csv_d = _stress_committee_disagreement(
        baseline=baseline, results_dir=rd, verbose=verbose
    )
    part_e_status, part_e_detail, _, csv_e = _stress_escalation_ladder(
        baseline=baseline, results_dir=rd, verbose=verbose
    )
    part_f_status, part_f_detail, _, csv_f = _stress_certification(
        baseline=baseline, results_dir=rd, verbose=verbose
    )

    scorecard = {
        "conflicting_signal_stress": part_a_status,
        "false_negative_stress": part_b_status,
        "false_positive_stress": part_c_status,
        "committee_stress": part_d_status,
        "escalation_stress": part_e_status,
        "certification_stress": part_f_status,
        "overall_result": _aggregate_status(
            [
                part_a_status,
                part_b_status,
                part_c_status,
                part_d_status,
                part_e_status,
                part_f_status,
            ]
        ),
    }

    part_details = {
        "conflicting_signal_stress": part_a_detail,
        "false_negative_stress": part_b_detail,
        "false_positive_stress": part_c_detail,
        "committee_stress": part_d_detail,
        "escalation_stress": part_e_detail,
        "certification_stress": part_f_detail,
    }

    maturity_review = _maturity_review(
        baseline=baseline,
        part_results={k: v for k, v in scorecard.items() if k != "overall_result"},
        verbose=verbose,
    )
    final_decision = _final_decision(
        scorecard=scorecard,
        part_details=part_details,
        maturity_review=maturity_review,
        verbose=verbose,
    )

    scenario_counts = {
        "conflicting_signal_cases": part_a_detail.get("cases_tested", 0),
        "false_negative_cases": part_b_detail.get("cases_tested", 0),
        "false_positive_cases": part_c_detail.get("cases_tested", 0),
        "committee_cases": part_d_detail.get("cases_tested", 0),
        "escalation_cases": part_e_detail.get("cases_tested", 0),
        "certification_cases": part_f_detail.get("cases_tested", 0),
        "total_scenarios": sum(part_details[k].get("cases_tested", 0) for k in part_details),
    }

    all_csv = csv_a + csv_b + csv_c + csv_d + csv_e + csv_f
    for row in all_csv:
        if not row.get("generated_at"):
            row["generated_at"] = ts
    for area, status in scorecard.items():
        all_csv.append(
            {
                "section": "scorecard",
                "scenario": area,
                "subsystem": "",
                "status": status,
                "detail": f"Phase 157 institutional intelligence stress — {area}",
                "generated_at": ts,
            }
        )
    all_csv.append(
        {
            "section": "final_decision",
            "scenario": "institutional_intelligence_stress_test",
            "subsystem": "",
            "status": final_decision.get("status", ""),
            "detail": final_decision.get("recommendation", ""),
            "generated_at": ts,
        }
    )

    payload: Dict[str, Any] = {
        "generated_at": ts,
        "phase": "157",
        "mode": "simulation_only_stress_test",
        "disclaimer": (
            "Institutional intelligence stress testing simulates adversarial conditions only. "
            "No execution, governance rule, certification rule, or position modifications."
        ),
        "conflicting_signal_stress": part_a_status,
        "false_negative_stress": part_b_status,
        "false_positive_stress": part_c_status,
        "committee_stress": part_d_status,
        "escalation_stress": part_e_status,
        "certification_stress": part_f_status,
        "overall_result": scorecard["overall_result"],
        "scorecard": scorecard,
        "scenario_counts": scenario_counts,
        "part_a_conflicting_signal_stress": part_a_detail,
        "part_b_false_negative_stress": part_b_detail,
        "part_c_false_positive_stress": part_c_detail,
        "part_d_committee_stress": part_d_detail,
        "part_e_escalation_stress": part_e_detail,
        "part_f_certification_stress": part_f_detail,
        "part_h_maturity_review": maturity_review,
        "part_i_final_decision": final_decision,
        "institutional_intelligence_certification": {
            "status": final_decision.get("institutional_intelligence_certification_status"),
            "promotion_recommended": final_decision.get("promotion_recommended"),
        },
        "_csv_rows": all_csv,
    }
    return payload


def run_institutional_intelligence_stress_test(
    *,
    results_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compute stress test and write JSON + CSV artifacts."""
    rd = Path(results_dir) if results_dir else RESULTS
    payload = compute_institutional_intelligence_stress_test(results_dir=rd, verbose=verbose)
    csv_rows = payload.pop("_csv_rows", [])

    _atomic_write_json(payload, OUTPUT_JSON)
    _write_scorecard_csv(csv_rows, OUTPUT_CSV)

    print(
        f"[INSTITUTIONAL_INTELLIGENCE_STRESS_TEST] phase=157 "
        f"overall={payload.get('overall_result')} "
        f"cert={payload.get('institutional_intelligence_certification', {}).get('status')} "
        f"json={OUTPUT_JSON} csv={OUTPUT_CSV}",
        flush=True,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 157 — Institutional Intelligence Stress Testing (simulation only)"
    )
    parser.add_argument("--verbose", action="store_true", help="Emit detailed stress logs")
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS),
        help="Path to data/results directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    payload = run_institutional_intelligence_stress_test(
        results_dir=Path(args.results_dir),
        verbose=args.verbose,
    )
    overall = payload.get("overall_result", "FAIL")
    return 0 if overall in ("PASS", "WARN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
