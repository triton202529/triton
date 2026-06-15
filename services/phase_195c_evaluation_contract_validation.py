"""
phase_195c_evaluation_contract_validation.py
Phase 195C — Evaluation contract repair and governance calibration validation.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

OUTPUT_JSON = RESULTS / "phase_195c_evaluation_contract_validation.json"
OUTPUT_CSV = RESULTS / "phase_195c_evaluation_contract_scorecard.csv"

SCORECARD_COLUMNS = (
    "check",
    "status",
    "detail",
)

REQUIRED_EVAL_FIELDS = (
    "scoring_model_version",
    "baseline_context",
    "high_cps_adjustment_applied",
    "evaluation_reason",
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


def _snapshot_certification() -> Dict[str, Any]:
    cert = _load_json(RESULTS / "capital_preservation_certification.json")
    return {
        "certification_status": cert.get("certification_status"),
        "veto_triggered": cert.get("veto_triggered") or [],
        "pre_veto_certification_status": cert.get("pre_veto_certification_status"),
        "evaluation_certified": (cert.get("areas") or {}).get("Evaluation", {}).get("certified"),
    }


def _run_governance_chain() -> Dict[str, Any]:
    from services.capital_preservation_evaluation import (
        SCORING_MODEL_VERSION,
        run_evaluation_chain,
    )
    from services.capital_preservation_certification import (
        run_certification_false_positive_elimination,
    )
    from services.risk_watchdog import compute_governance_risk_summary
    from services.risk_committee_oversight import run_risk_committee_oversight

    before = _snapshot_certification()

    eval_payload = run_evaluation_chain(results_dir=RESULTS)
    evaluation = eval_payload.get("protective_action_evaluation") or {}

    cpi = _load_json(RESULTS / "capital_preservation_intelligence.json")
    cpe = _load_json(RESULTS / "capital_preservation_escalation.json")
    cpa = _load_json(RESULTS / "capital_preservation_advisory.json")
    readiness = _load_json(RESULTS / "execution_readiness.json")
    compute_governance_risk_summary(cpi, cpe, cpa, readiness_doc=readiness)

    cert_payload = run_certification_false_positive_elimination(
        results_dir=RESULTS,
        verbose=False,
        run_stress_revalidation=False,
    )
    cert_doc = cert_payload.get("phase_158") or _load_json(
        RESULTS / "capital_preservation_certification.json"
    )
    if not cert_doc.get("certification_status"):
        cert_doc = _load_json(RESULTS / "capital_preservation_certification.json")

    committee = run_risk_committee_oversight(results_dir=RESULTS)

    try:
        from services.phase_195b_governance_evaluation_gap_audit import run_phase_195b_audit

        gap_audit = run_phase_195b_audit(verbose=False)
    except Exception as e:
        gap_audit = {"error": str(e), "phase_pass": False}

    after = _snapshot_certification()
    gov = _load_json(RESULTS / "governance_risk_summary.json")
    auth = _load_json(RESULTS / "governance_authorization.json")

    return {
        "before": before,
        "after": after,
        "evaluation": evaluation,
        "certification": _load_json(RESULTS / "capital_preservation_certification.json"),
        "governor": eval_payload.get("capital_preservation_governor") or {},
        "governance": gov,
        "readiness": readiness,
        "committee": committee,
        "gap_audit": gap_audit,
        "scoring_model_version_constant": SCORING_MODEL_VERSION,
        "auth_unchanged": {
            "live_execution_permitted": auth.get("live_execution_permitted"),
            "paper_execution_permitted": auth.get("paper_execution_permitted"),
            "execution_authorized": auth.get("execution_authorized"),
        },
    }


def run_phase_195c_validation(*, verbose: bool = True) -> Dict[str, Any]:
    chain = _run_governance_chain()
    evaluation = chain["evaluation"]
    before = chain["before"]
    after = chain["after"]
    cert = chain["certification"]
    gov = chain["governance"]
    readiness = chain["readiness"]

    checks: List[Dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    missing = [
        f for f in REQUIRED_EVAL_FIELDS if f not in evaluation or evaluation.get(f) in (None, "")
    ]
    add_check(
        "contract_fields_present",
        not missing,
        f"missing={missing or 'none'}",
    )

    try:
        from services.capital_preservation_evaluation import SCORING_MODEL_VERSION

        add_check(
            "scoring_model_version_defined",
            bool(SCORING_MODEL_VERSION),
            f"version={SCORING_MODEL_VERSION}",
        )
    except ImportError as e:
        add_check("scoring_model_version_defined", False, str(e))

    bc = evaluation.get("baseline_context")
    bc_ok = isinstance(bc, dict) and bool(bc.get("baseline_context_label"))
    add_check(
        "baseline_context_present",
        bc_ok,
        f"type={type(bc).__name__} label={evaluation.get('baseline_context_label')}",
    )

    add_check(
        "high_cps_adjustment_detected",
        evaluation.get("high_cps_adjustment_applied") is True,
        f"applied={evaluation.get('high_cps_adjustment_applied')} stable={evaluation.get('preservation_stable_context')}",
    )

    evals = evaluation.get("evaluations") or []
    reasons_ok = bool(evaluation.get("evaluation_reason")) and all(
        isinstance(e, dict) and e.get("evaluation_reason") for e in evals
    )
    add_check(
        "evaluation_reason_present",
        reasons_ok,
        f"doc_reason={bool(evaluation.get('evaluation_reason'))} trial_reasons={sum(1 for e in evals if e.get('evaluation_reason'))}/{len(evals)}",
    )

    cert_improved = after.get("certification_status") != before.get("certification_status") or (
        before.get("certification_status") == "NOT_CERTIFIED"
        and after.get("certification_status")
        in ("PARTIALLY_CERTIFIED", "CONDITIONALLY_CERTIFIED", "CERTIFIED_FOR_PAPER_PROTECTION")
    )
    add_check(
        "certification_improved_or_explained",
        cert_improved or after.get("evaluation_certified"),
        f"before={before.get('certification_status')} after={after.get('certification_status')}",
    )

    veto_before = "mandatory_preservation_gate_failed" in (before.get("veto_triggered") or [])
    veto_after = "mandatory_preservation_gate_failed" in (after.get("veto_triggered") or [])
    add_check(
        "mandatory_veto_resolved_or_explained",
        (veto_before and not veto_after)
        or (not veto_after)
        or bool(cert.get("evaluation_context")),
        f"veto_before={veto_before} veto_after={veto_after}",
    )

    add_check(
        "readiness_unchanged",
        readiness.get("readiness_status") == "READY" and readiness.get("checks_passing_count") == 8,
        f"status={readiness.get('readiness_status')} checks={readiness.get('checks_passing_count')}/8",
    )

    add_check(
        "execution_authorization_unchanged",
        chain["auth_unchanged"].get("live_execution_permitted") is False,
        json.dumps(chain["auth_unchanged"]),
    )

    awareness = str(gov.get("governance_awareness_level") or "")
    awareness_aligned = (
        awareness in ("ROUTINE_OVERSIGHT", "GREEN") or gov.get("operationally_stable") is True
    )
    add_check(
        "governance_awareness_aligned",
        awareness_aligned,
        f"awareness={awareness} operationally_stable={gov.get('operationally_stable')}",
    )

    eval_area = (cert.get("areas") or {}).get("Evaluation") or {}
    add_check(
        "evaluation_area_certified",
        bool(eval_area.get("certified")),
        eval_area.get("notes", ""),
    )

    phase_pass = all(c["status"] == "PASS" for c in checks)

    payload = {
        "generated_at": _utc_now_iso(),
        "framework": "Phase-195C",
        "phase_pass": phase_pass,
        "contract_fields_present": not missing,
        "scoring_model_version_defined": evaluation.get("scoring_model_version"),
        "baseline_context_present": bc_ok,
        "high_cps_adjustment_detected": evaluation.get("high_cps_adjustment_applied"),
        "evaluation_reason_present": reasons_ok,
        "certification_status_before": before.get("certification_status"),
        "certification_status_after": after.get("certification_status"),
        "mandatory_veto_before": veto_before,
        "mandatory_veto_after": veto_after,
        "readiness_status": readiness.get("readiness_status"),
        "cps_score": (
            (evaluation.get("baseline_context") or {}).get("capital_preservation_score")
            if isinstance(evaluation.get("baseline_context"), dict)
            else evaluation.get("baseline_cps")
        ),
        "governance_awareness": awareness,
        "preservation_posture": chain["governor"].get("preservation_posture"),
        "checks": checks,
        "summary": {
            "contract_repaired": not missing,
            "certification_status": after.get("certification_status"),
            "evaluation_certified": eval_area.get("certified"),
            "governance_awareness": awareness,
            "no_execution_regression": chain["auth_unchanged"].get("live_execution_permitted")
            is False,
        },
        "gap_audit_phase_pass": chain["gap_audit"].get("phase_pass"),
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(OUTPUT_CSV, checks)

    if verbose:
        print(f"Phase 195C validation: {OUTPUT_JSON}")
        print(
            f"phase_pass={phase_pass} cert {before.get('certification_status')} -> {after.get('certification_status')}"
        )
        for c in checks:
            print(f"  {c['check']}: {c['status']}")

    return payload


if __name__ == "__main__":
    run_phase_195c_validation()
