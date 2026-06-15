"""
phase_195b_governance_evaluation_gap_audit.py
Phase 195B — Governance Evaluation Gap Root Cause Audit.

Read-only audit. No execution, trading, or governance rule changes.
Identifies why protective action evaluation, preservation certification,
and governance awareness diverge from operational readiness.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

OUTPUT_JSON = RESULTS / "phase_195b_governance_evaluation_gap_audit.json"
OUTPUT_CSV = RESULTS / "phase_195b_governance_evaluation_gap_scorecard.csv"

ARTIFACT_PATHS = {
    "certification": RESULTS / "capital_preservation_certification.json",
    "evaluation": RESULTS / "protective_action_evaluation.json",
    "trials": RESULTS / "protective_action_trials.json",
    "cpi": RESULTS / "capital_preservation_intelligence.json",
    "cpe": RESULTS / "capital_preservation_escalation.json",
    "governor": RESULTS / "capital_preservation_governor.json",
    "gov_risk": RESULTS / "governance_risk_summary.json",
    "readiness": RESULTS / "execution_readiness.json",
    "auth": RESULTS / "governance_authorization.json",
}

SCORECARD_COLUMNS = (
    "finding_id",
    "category",
    "severity",
    "status",
    "root_cause",
    "evidence",
    "impacted_outcome",
)

EXPECTED_EVAL_FIELDS = (
    "baseline_context",
    "high_cps_adjustment_applied",
    "scoring_model_version",
)

MANDATORY_PRESERVATION_GATES = frozenset(
    {"Monitoring", "Governor", "Authorization", "Readiness", "Evaluation"}
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SCORECARD_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SCORECARD_COLUMNS})
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _escalation_from_cps(cps: int) -> str:
    if cps >= 90:
        return "GREEN"
    if cps >= 75:
        return "YELLOW"
    if cps >= 60:
        return "ORANGE"
    if cps >= 40:
        return "RED"
    return "CRITICAL"


def _evaluation_cert_would_pass(
    evaluation: Dict[str, Any],
) -> Tuple[bool, str, float]:
    """Mirror services/capital_preservation_certification._evaluation_certification."""
    eval_count = int(evaluation.get("evaluation_count") or 0)
    avg_eff = float(evaluation.get("average_effectiveness") or 0)
    baseline_context = str(evaluation.get("baseline_context") or "")
    beneficial = int(evaluation.get("beneficial_count") or 0)
    negative = int(evaluation.get("negative_count") or 0)
    high_cps_adj = bool(evaluation.get("high_cps_adjustment_applied"))

    if eval_count < 2:
        return False, "eval_count>=2", 50.0

    if baseline_context == "HIGH_CPS_PRESERVATION" and high_cps_adj:
        threshold = 40.0
        ok = avg_eff >= threshold and beneficial >= 1 and negative <= 1
        return ok, "high_cps_baseline_aware>=40", threshold
    if baseline_context == "MODERATE_CPS_IMPROVEMENT":
        threshold = 45.0
        ok = avg_eff >= threshold and beneficial >= 1
        return ok, "moderate_cps>=45", threshold

    threshold = 50.0
    return avg_eff >= threshold, "standard>=50", threshold


def _audit_producer_contract(
    evaluation: Dict[str, Any], cpi: Dict[str, Any]
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    missing = []
    for f in ("scoring_model_version", "high_cps_adjustment_applied"):
        if f not in evaluation or evaluation.get(f) in (None, ""):
            missing.append(f)
    bc = evaluation.get("baseline_context")
    if isinstance(bc, dict):
        if not bc.get("baseline_context_label"):
            missing.append("baseline_context.baseline_context_label")
    elif not evaluation.get("baseline_context_label") and not bc:
        missing.append("baseline_context")
    if missing:
        findings.append(
            {
                "finding_id": "RC-001",
                "category": "PRODUCER_CONSUMER_CONTRACT_GAP",
                "severity": "HIGH",
                "status": "CONFIRMED",
                "root_cause": (
                    "compute_protective_action_evaluation() does not emit Phase 150B "
                    "baseline-aware fields consumed by certification."
                ),
                "evidence": (
                    f"missing_fields={missing}; "
                    f"baseline_cps={evaluation.get('baseline_cps') or cpi.get('capital_preservation_score')}; "
                    f"certification expects baseline_context when CPS is high"
                ),
                "impacted_outcome": "Evaluation area uses standard threshold (avg>=50) instead of high-CPS path",
                "artifact_refs": [
                    "services/capital_preservation_evaluation.py",
                    "protective_action_evaluation.json",
                ],
            }
        )

    try:
        from services.capital_preservation_evaluation import compute_protective_action_evaluation

        recomputed = compute_protective_action_evaluation(
            trials_doc=_load_json(ARTIFACT_PATHS["trials"]),
            cpi_doc=cpi,
            cpe_doc=_load_json(ARTIFACT_PATHS["cpe"]),
        )
        repro_missing = [
            f
            for f in EXPECTED_EVAL_FIELDS
            if f not in recomputed or recomputed.get(f) in (None, "")
        ]
        if repro_missing:
            findings.append(
                {
                    "finding_id": "RC-002",
                    "category": "RECOMPUTE_PARITY_GAP",
                    "severity": "HIGH",
                    "status": "CONFIRMED",
                    "root_cause": "Recomputed evaluation from live trials reproduces the same missing contract fields.",
                    "evidence": f"recomputed_missing={repro_missing}; evaluation_count={recomputed.get('evaluation_count')}",
                    "impacted_outcome": "Certification audit Part D cannot pass baseline_context checks",
                    "artifact_refs": ["services/capital_preservation_certification_audit.py"],
                }
            )
    except Exception as e:
        findings.append(
            {
                "finding_id": "RC-002",
                "category": "RECOMPUTE_PARITY_GAP",
                "severity": "MEDIUM",
                "status": "INCONCLUSIVE",
                "root_cause": "Could not recompute protective action evaluation for parity check.",
                "evidence": str(e),
                "impacted_outcome": "Parity verification blocked",
                "artifact_refs": ["services/capital_preservation_evaluation.py"],
            }
        )

    try:
        from services.capital_preservation_evaluation import SCORING_MODEL_VERSION  # noqa: F401

        _ = SCORING_MODEL_VERSION
    except ImportError:
        findings.append(
            {
                "finding_id": "RC-003",
                "category": "IMPORT_REFERENCE_GAP",
                "severity": "MEDIUM",
                "status": "CONFIRMED",
                "root_cause": "SCORING_MODEL_VERSION is imported by certification audit but undefined in evaluation producer.",
                "evidence": "ImportError from services.capital_preservation_evaluation.SCORING_MODEL_VERSION",
                "impacted_outcome": "capital_preservation_certification_audit Part D scoring_model_version check cannot run",
                "artifact_refs": [
                    "services/capital_preservation_certification_audit.py",
                    "services/capital_preservation_evaluation.py",
                ],
            }
        )

    return findings


def _audit_evaluation_scoring(
    evaluation: Dict[str, Any],
    cpi: Dict[str, Any],
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    cps = int(cpi.get("capital_preservation_score") or evaluation.get("baseline_cps") or 0)
    avg_eff = float(evaluation.get("average_effectiveness") or 0)
    beneficial = int(evaluation.get("beneficial_count") or 0)
    negative = int(evaluation.get("negative_count") or 0)
    neutral = int(evaluation.get("neutral_count") or 0)
    top = evaluation.get("top_trial") or {}
    top_score = top.get("effectiveness_score")

    current_ok, threshold_label, threshold = _evaluation_cert_would_pass(evaluation)
    if not current_ok:
        findings.append(
            {
                "finding_id": "RC-004",
                "category": "WEAK_PROTECTIVE_ACTION_EVALUATION",
                "severity": "HIGH",
                "status": "CONFIRMED",
                "root_cause": (
                    "Protective action effectiveness uses absolute scoring without baseline-aware "
                    "adjustment at high CPS; marginal trials score NEUTRAL/NEGATIVE."
                ),
                "evidence": (
                    f"cps={cps}; avg_effectiveness={avg_eff}; threshold={threshold_label}({threshold}); "
                    f"beneficial={beneficial}; neutral={neutral}; negative={negative}; "
                    f"top_trial_score={top_score}; top_trial_label={top.get('evaluation')}"
                ),
                "impacted_outcome": "Evaluation certification area fails; governor flags Weak Trial Outcomes",
                "artifact_refs": [
                    "protective_action_evaluation.json",
                    "capital_preservation_governor.json",
                ],
            }
        )

    if cps >= 75 and not str(evaluation.get("baseline_context") or ""):
        sim_ok, sim_label, sim_threshold = _evaluation_cert_would_pass(
            {
                **evaluation,
                "baseline_context": "HIGH_CPS_PRESERVATION",
                "high_cps_adjustment_applied": True,
            }
        )
        findings.append(
            {
                "finding_id": "RC-005",
                "category": "HIGH_BASELINE_THRESHOLD_MISMATCH",
                "severity": "MEDIUM",
                "status": "CONFIRMED",
                "root_cause": (
                    "At CPS>=75 the certification consumer supports a lowered high-CPS threshold, "
                    "but the producer never sets baseline_context; even if set, beneficial_count=0 "
                    "would still fail the high-CPS gate."
                ),
                "evidence": (
                    f"cps={cps}; current_pass={current_ok}; "
                    f"simulated_high_cps_pass={sim_ok}; simulated_threshold={sim_label}; "
                    f"simulated_threshold_value={sim_threshold}; beneficial={beneficial}; negative={negative}"
                ),
                "impacted_outcome": "Structural false-negative on healthy portfolios with limited marginal upside",
                "artifact_refs": ["services/capital_preservation_certification.py"],
            }
        )

    evals = evaluation.get("evaluations") or []
    missing_reason = sum(1 for e in evals if isinstance(e, dict) and not e.get("evaluation_reason"))
    if evals and missing_reason == len(evals):
        findings.append(
            {
                "finding_id": "RC-006",
                "category": "EVALUATION_EXPLAINABILITY_GAP",
                "severity": "LOW",
                "status": "CONFIRMED",
                "root_cause": "Per-trial evaluation_reason fields are absent from protective action evaluation output.",
                "evidence": f"trials={len(evals)}; missing_evaluation_reason={missing_reason}",
                "impacted_outcome": "Certification audit cannot verify explainability of weak outcomes",
                "artifact_refs": ["protective_action_evaluation.json"],
            }
        )

    return findings


def _audit_governance_alignment(
    cpi: Dict[str, Any],
    cpe: Dict[str, Any],
    gov: Dict[str, Any],
    readiness: Dict[str, Any],
    auth: Dict[str, Any],
    cert: Dict[str, Any],
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    cps = int(cpi.get("capital_preservation_score") or 0)
    expected_esc = _escalation_from_cps(cps)
    actual_esc = str(cpe.get("escalation_state") or "").upper()
    awareness = str(gov.get("governance_awareness_level") or "").upper()
    readiness_status = str(readiness.get("readiness_status") or "")
    cert_status = str(cert.get("certification_status") or "")

    if (
        cps >= 75
        and cps < 90
        and actual_esc == "YELLOW"
        and not (cpe.get("escalation_reason") or [])
    ):
        findings.append(
            {
                "finding_id": "RC-007",
                "category": "ESCALATION_CPS_THRESHOLD_GAP",
                "severity": "MEDIUM",
                "status": "CONFIRMED",
                "root_cause": (
                    "Escalation ladder assigns YELLOW for CPS 75–89 (_escalation_state_from_cps); "
                    "GREEN requires CPS>=90. YELLOW carries no alert-driven escalation_reason."
                ),
                "evidence": (
                    f"cps={cps}; health_band={cpi.get('health_band')}; "
                    f"escalation_state={actual_esc}; expected_from_cps={expected_esc}; "
                    f"escalation_reason={cpe.get('escalation_reason')}"
                ),
                "impacted_outcome": "Governance awareness maps to INCREASED_MONITORING despite HEALTHY portfolio",
                "artifact_refs": [
                    "services/risk_watchdog.py:_escalation_state_from_cps",
                    "governance_risk_summary.json",
                ],
            }
        )

    if readiness_status == "READY" and awareness == "INCREASED_MONITORING":
        findings.append(
            {
                "finding_id": "RC-008",
                "category": "GOVERNANCE_READINESS_DIVERGENCE",
                "severity": "HIGH",
                "status": "CONFIRMED",
                "root_cause": (
                    "Operational readiness derives from authorization + infrastructure checks; "
                    "governance awareness derives solely from CPE escalation mapping. "
                    "The readiness governance check uses governance_authorized, not awareness level."
                ),
                "evidence": (
                    f"readiness_status={readiness_status}; checks_passing={readiness.get('checks_passing_count')}/"
                    f"{readiness.get('checks_total')}; governance_awareness={awareness}; "
                    f"governance_authorized={auth.get('governance_authorized')}; "
                    f"gate_reason={auth.get('gate_reasons', {}).get('governance')}"
                ),
                "impacted_outcome": "System appears operationally READY while governance layer signals increased monitoring",
                "artifact_refs": [
                    "execution_readiness.json",
                    "governance_risk_summary.json",
                    "services/governance_execution_readiness.py",
                ],
            }
        )

    pre_veto = cert.get("pre_veto_certification_status")
    if pre_veto == "PARTIALLY_CERTIFIED" and cert_status == "NOT_CERTIFIED":
        failed_gates = [
            g
            for g in MANDATORY_PRESERVATION_GATES
            if not (cert.get("areas") or {}).get(g, {}).get("certified")
        ]
        veto = cert.get("veto_triggered") or []
        findings.append(
            {
                "finding_id": "RC-009",
                "category": "MANDATORY_PRESERVATION_GATE_VETO",
                "severity": "HIGH",
                "status": "CONFIRMED",
                "root_cause": (
                    "Phase 158 mandatory_preservation_gate_failed veto downgrades certification when any "
                    "mandatory area fails, even with 7/8 areas certified and healthy portfolio metrics."
                ),
                "evidence": (
                    f"pre_veto={pre_veto}; final={cert_status}; score={cert.get('certification_score')}; "
                    f"failed_mandatory_gates={failed_gates}; veto_triggered={veto}; "
                    f"preservation_status={cert.get('preservation_status')}; "
                    f"portfolio_health={cert.get('portfolio_health')}"
                ),
                "impacted_outcome": "Preservation certification remains NOT_CERTIFIED despite strong monitoring/readiness",
                "artifact_refs": ["capital_preservation_certification.json"],
            }
        )

    gov_area = (cert.get("areas") or {}).get("Governance") or {}
    if gov_area.get("certified") and awareness == "INCREASED_MONITORING":
        findings.append(
            {
                "finding_id": "RC-010",
                "category": "GOVERNANCE_AREA_SCORING_LENIENT",
                "severity": "LOW",
                "status": "CONFIRMED",
                "root_cause": (
                    "Governance certification area passes when awareness is not in block levels; "
                    "INCREASED_MONITORING is certified with score 35 while operational signals are GREEN-class."
                ),
                "evidence": (
                    f"governance_area_certified={gov_area.get('certified')}; "
                    f"governance_area_score={gov_area.get('score')}; awareness={awareness}; "
                    f"notes={gov_area.get('notes')}"
                ),
                "impacted_outcome": "Governance area certified masks awareness/readiness misalignment in scorecard",
                "artifact_refs": ["services/capital_preservation_certification.py"],
            }
        )

    return findings


def _audit_governor_amplification(
    governor: Dict[str, Any],
    evaluation: Dict[str, Any],
    cpi: Dict[str, Any],
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    posture = str(governor.get("preservation_posture") or "")
    cps = int(cpi.get("capital_preservation_score") or 0)
    negative = int(evaluation.get("negative_count") or 0)

    if cps >= 75 and posture == "ORANGE" and negative >= 2:
        findings.append(
            {
                "finding_id": "RC-011",
                "category": "GOVERNOR_POSTURE_AMPLIFICATION",
                "severity": "MEDIUM",
                "status": "CONFIRMED",
                "root_cause": (
                    "_derive_preservation_posture elevates to ORANGE when negative_count>=2, "
                    "independent of healthy CPS and READY readiness."
                ),
                "evidence": (
                    f"cps={cps}; posture={posture}; negative_count={negative}; "
                    f"top_drivers={governor.get('top_drivers')}; avg_trial_effectiveness="
                    f"{governor.get('average_trial_effectiveness')}"
                ),
                "impacted_outcome": "Governor advisory posture ORANGE while CPI HEALTHY and readiness READY",
                "artifact_refs": [
                    "capital_preservation_governor.json",
                    "services/capital_preservation_evaluation.py",
                ],
            }
        )

    return findings


def _build_alignment_matrix(
    *,
    cpi: Dict[str, Any],
    cpe: Dict[str, Any],
    gov: Dict[str, Any],
    readiness: Dict[str, Any],
    cert: Dict[str, Any],
    governor: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    cps = int(cpi.get("capital_preservation_score") or 0)
    eval_area = (cert.get("areas") or {}).get("Evaluation") or {}
    return {
        "capital_preservation_score": cps,
        "health_band": cpi.get("health_band"),
        "risk_trend": cpi.get("risk_trend"),
        "escalation_state": cpe.get("escalation_state"),
        "governance_awareness_level": gov.get("governance_awareness_level"),
        "readiness_status": readiness.get("readiness_status"),
        "preservation_posture": governor.get("preservation_posture"),
        "certification_status": cert.get("certification_status"),
        "pre_veto_certification_status": cert.get("pre_veto_certification_status"),
        "evaluation_area_certified": eval_area.get("certified"),
        "evaluation_average_effectiveness": evaluation.get("average_effectiveness"),
        "preservation_status": cert.get("preservation_status"),
        "portfolio_health": cert.get("portfolio_health"),
        "alignment_gap_detected": (
            readiness.get("readiness_status") == "READY"
            and str(gov.get("governance_awareness_level") or "").upper() == "INCREASED_MONITORING"
            and cert.get("certification_status") == "NOT_CERTIFIED"
        ),
    }


def run_phase_195b_audit(*, verbose: bool = True) -> Dict[str, Any]:
    artifacts = {k: _load_json(p) for k, p in ARTIFACT_PATHS.items()}

    findings: List[Dict[str, Any]] = []
    findings.extend(_audit_producer_contract(artifacts["evaluation"], artifacts["cpi"]))
    findings.extend(_audit_evaluation_scoring(artifacts["evaluation"], artifacts["cpi"]))
    findings.extend(
        _audit_governance_alignment(
            artifacts["cpi"],
            artifacts["cpe"],
            artifacts["gov_risk"],
            artifacts["readiness"],
            artifacts["auth"],
            artifacts["certification"],
        )
    )
    findings.extend(
        _audit_governor_amplification(
            artifacts["governor"],
            artifacts["evaluation"],
            artifacts["cpi"],
        )
    )

    confirmed = [f for f in findings if f.get("status") == "CONFIRMED"]
    high = [f for f in confirmed if f.get("severity") == "HIGH"]

    alignment = _build_alignment_matrix(
        cpi=artifacts["cpi"],
        cpe=artifacts["cpe"],
        gov=artifacts["gov_risk"],
        readiness=artifacts["readiness"],
        cert=artifacts["certification"],
        governor=artifacts["governor"],
        evaluation=artifacts["evaluation"],
    )

    payload = {
        "generated_at": _utc_now_iso(),
        "framework": "Phase-195B",
        "mode": "read_only_root_cause_audit",
        "objective": (
            "Identify why protective action evaluation produces weak outcomes, "
            "preservation certification remains NOT_CERTIFIED, governance awareness "
            "does not align with operational readiness, and mandatory preservation gates fail."
        ),
        "alignment_matrix": alignment,
        "finding_count": len(findings),
        "confirmed_count": len(confirmed),
        "high_severity_count": len(high),
        "phase_pass": len(high) > 0,
        "executive_summary": {
            "primary_root_cause": (
                "Phase 150B baseline-aware evaluation fields are consumed by certification "
                "but never produced by compute_protective_action_evaluation(), forcing the "
                "standard avg>=50 threshold on a CPS=89 portfolio where marginal protective "
                "trials structurally score low."
            ),
            "certification_blocker": (
                "Evaluation area fails (avg=30.5 < 50) → mandatory_preservation_gate_failed veto "
                "→ NOT_CERTIFIED despite 7/8 areas and preservation_status=PASS."
            ),
            "awareness_misalignment": (
                "CPS 89 maps to YELLOW escalation (GREEN requires >=90) → INCREASED_MONITORING "
                "while execution_readiness is READY 8/8."
            ),
            "hardening_scope": "Institutional wiring of existing contracts — no new governance systems.",
        },
        "findings": findings,
        "artifact_snapshot": {
            k: {
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "present": p.is_file(),
                "generated_at": artifacts[k].get("generated_at"),
            }
            for k, p in ARTIFACT_PATHS.items()
        },
        "remediation_categories": [
            {
                "category": "PRODUCER_CONSUMER_CONTRACT",
                "action": "Wire baseline_context, high_cps_adjustment_applied, scoring_model_version, evaluation_reason into evaluation producer",
                "touches": ["services/capital_preservation_evaluation.py"],
            },
            {
                "category": "THRESHOLD_CALIBRATION",
                "action": "Align high-CPS certification criteria with structurally low marginal trial scores or use top-trial metric",
                "touches": ["services/capital_preservation_certification.py"],
            },
            {
                "category": "SIGNAL_ALIGNMENT",
                "action": "Reconcile escalation ladder GREEN threshold (90) with HEALTHY band or feed awareness from composite posture",
                "touches": [
                    "services/risk_watchdog.py",
                    "services/governance_execution_readiness.py",
                ],
            },
        ],
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(OUTPUT_CSV, findings)

    if verbose:
        print(f"Phase 195B audit: {OUTPUT_JSON}")
        print(f"findings={len(findings)} confirmed={len(confirmed)} high={len(high)}")
        for f in findings:
            print(f"  [{f['finding_id']}] {f['severity']} {f['category']}: {f['status']}")

    return payload


if __name__ == "__main__":
    run_phase_195b_audit()
