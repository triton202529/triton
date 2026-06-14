"""
execution_gate_simulator.py
THP-005 — Scenario simulation laboratory for execution gate authorization (read-only).

All scenarios call evaluate_execution_gate() with record_evidence=False.
No broker calls, order placement, or portfolio mutations.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.triton_execution_gate import EXECUTION_PATHS, evaluate_execution_gate

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"

SIMULATION_JSON = GOVERNANCE_DIR / "execution_gate_simulation_results.json"
SIMULATION_CSV = GOVERNANCE_DIR / "execution_gate_simulation_results.csv"

SIMULATION_FIELDS = (
    "simulation_id",
    "scenario_name",
    "category",
    "execution_path",
    "input_conditions",
    "expected_decision",
    "expected_allowed",
    "actual_decision",
    "actual_allowed",
    "pass_fail",
    "explanation",
    "timestamp",
)

SCENARIO_CATALOG: List[Dict[str, Any]] = [
    {
        "simulation_id": "sim-live-no-governance",
        "scenario_name": "Live execution without governance approval",
        "category": "live_authorization",
        "path": EXECUTION_PATHS["LIVE_EXECUTION"],
        "context": {
            "liveMode": True,
            "liveModeAuthorized": True,
            "governanceApproved": False,
            "lifecycleApproved": True,
            "allocationApproved": True,
            "riskConstraintsSatisfied": True,
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "simulation_id": "sim-live-no-prod-readiness",
        "scenario_name": "Live execution without production readiness",
        "category": "live_authorization",
        "path": EXECUTION_PATHS["LIVE_EXECUTION"],
        "context": {
            "liveMode": True,
            "liveModeAuthorized": True,
            "governanceApproved": True,
            "lifecycleApproved": True,
            "allocationApproved": True,
            "riskConstraintsSatisfied": True,
            "environment": "production",
            "productionReadinessOverride": False,
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "simulation_id": "sim-live-authorized-review",
        "scenario_name": "Live execution with full authorization",
        "category": "live_authorization",
        "path": EXECUTION_PATHS["LIVE_EXECUTION"],
        "context": {
            "liveMode": True,
            "liveModeAuthorized": True,
            "governanceApproved": True,
            "lifecycleApproved": True,
            "allocationApproved": True,
            "riskConstraintsSatisfied": True,
            "environment": "development",
        },
        "expected_decision": "REVIEW_REQUIRED",
        "expected_allowed": False,
    },
    {
        "simulation_id": "sim-paper-enabled",
        "scenario_name": "Paper execution with paper mode enabled",
        "category": "execution",
        "path": EXECUTION_PATHS["PAPER_EXECUTION"],
        "context": {
            "paperMode": True,
            "lifecycleApproved": True,
            "riskConstraintsSatisfied": True,
            "executionAuthorized": True,
        },
        "expected_decision": "ALLOW",
        "expected_allowed": True,
    },
    {
        "simulation_id": "sim-paper-live-separation",
        "scenario_name": "Paper/live environment separation violation",
        "category": "execution",
        "path": EXECUTION_PATHS["PAPER_EXECUTION"],
        "context": {
            "paperMode": False,
            "liveMode": True,
            "lifecycleApproved": True,
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "simulation_id": "sim-broker-unavailable",
        "scenario_name": "Broker submission without connectivity",
        "category": "execution",
        "path": EXECUTION_PATHS["BROKER_SUBMISSION"],
        "context": {
            "lifecycleApproved": True,
            "executionAuthorized": True,
            "brokerConnected": False,
            "executionGuardApproved": True,
            "duplicateOrderClear": True,
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "simulation_id": "sim-duplicate-order",
        "scenario_name": "Broker submission with duplicate order detected",
        "category": "execution",
        "path": EXECUTION_PATHS["BROKER_SUBMISSION"],
        "context": {
            "lifecycleApproved": True,
            "executionAuthorized": True,
            "brokerConnected": True,
            "executionGuardApproved": True,
            "duplicateOrderClear": False,
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "simulation_id": "sim-broker-authorized",
        "scenario_name": "Broker submission fully authorized",
        "category": "execution",
        "path": EXECUTION_PATHS["BROKER_SUBMISSION"],
        "context": {
            "lifecycleApproved": True,
            "executionAuthorized": True,
            "brokerConnected": True,
            "executionGuardApproved": True,
            "duplicateOrderClear": True,
            "environment": "development",
        },
        "expected_decision": "ALLOW",
        "expected_allowed": True,
    },
    {
        "simulation_id": "sim-risk-lockdown",
        "scenario_name": "Capital deployment with risk constraints violated",
        "category": "risk",
        "path": EXECUTION_PATHS["CAPITAL_DEPLOYMENT"],
        "context": {
            "lifecycleApproved": True,
            "allocationApproved": True,
            "riskConstraintsSatisfied": False,
            "environment": "development",
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "simulation_id": "sim-emergency-force-exit",
        "scenario_name": "Emergency FORCE_EXIT under risk emergency",
        "category": "risk",
        "path": EXECUTION_PATHS["RISK_EMERGENCY_ACTION"],
        "context": {
            "riskEmergencyTriggered": True,
            "emergencyAction": "FORCE_EXIT",
        },
        "expected_decision": "ALLOW",
        "expected_allowed": True,
    },
    {
        "simulation_id": "sim-emergency-no-trigger",
        "scenario_name": "Emergency action without risk trigger",
        "category": "risk",
        "path": EXECUTION_PATHS["RISK_EMERGENCY_ACTION"],
        "context": {"riskEmergencyTriggered": False, "emergencyAction": "FORCE_EXIT"},
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "simulation_id": "sim-governance-override-missing-operator",
        "scenario_name": "Governance override without operator identity",
        "category": "operator",
        "path": EXECUTION_PATHS["GOVERNANCE_OVERRIDE"],
        "context": {"operatorAuthorized": False},
        "expected_decision": "REVIEW_REQUIRED",
        "expected_allowed": False,
    },
    {
        "simulation_id": "sim-operator-override-unauthorized",
        "scenario_name": "Manual operator override without authorization",
        "category": "operator",
        "path": EXECUTION_PATHS["MANUAL_OPERATOR_OVERRIDE"],
        "context": {
            "operatorAuthorized": False,
            "overrideReason": "",
            "operationalContext": "",
        },
        "expected_decision": "REVIEW_REQUIRED",
        "expected_allowed": False,
    },
    {
        "simulation_id": "sim-operator-override-documented",
        "scenario_name": "Manual operator override with audit context",
        "category": "operator",
        "path": EXECUTION_PATHS["MANUAL_OPERATOR_OVERRIDE"],
        "context": {
            "operatorAuthorized": True,
            "overrideReason": "Manual review of stale position",
            "operationalContext": "Operator desk simulation THP-005",
            "auditTrailId": "sim-override-001",
        },
        "expected_decision": "REVIEW_REQUIRED",
        "expected_allowed": False,
    },
    {
        "simulation_id": "sim-capital-no-lifecycle",
        "scenario_name": "Capital deployment without lifecycle approval",
        "category": "risk",
        "path": EXECUTION_PATHS["CAPITAL_DEPLOYMENT"],
        "context": {
            "lifecycleApproved": False,
            "allocationApproved": True,
            "riskConstraintsSatisfied": True,
            "environment": "development",
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            if isinstance(flat.get("input_conditions"), dict):
                flat["input_conditions"] = json.dumps(flat["input_conditions"])
            writer.writerow({k: flat.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def run_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    path = scenario["path"]
    context = dict(scenario.get("context") or {})
    evaluation = evaluate_execution_gate(
        path,
        context,
        record_evidence=False,
        source=f"simulator:{scenario.get('simulation_id')}",
    )

    actual_decision = str(evaluation.get("decision") or "")
    actual_allowed = bool(evaluation.get("allowed"))
    expected_decision = str(scenario.get("expected_decision") or "")
    expected_allowed = bool(scenario.get("expected_allowed"))

    passed = (
        actual_decision.upper() == expected_decision.upper() and actual_allowed == expected_allowed
    )
    explanation = evaluation.get("reason") or ""
    if not passed:
        explanation = (
            f"Expected {expected_decision} (allowed={expected_allowed}) but got "
            f"{actual_decision} (allowed={actual_allowed}). Gate reason: {explanation}"
        )

    return {
        "simulation_id": scenario.get("simulation_id"),
        "scenario_name": scenario.get("scenario_name"),
        "category": scenario.get("category"),
        "execution_path": path,
        "input_conditions": context,
        "expected_decision": expected_decision,
        "expected_allowed": expected_allowed,
        "actual_decision": actual_decision,
        "actual_allowed": actual_allowed,
        "pass_fail": "PASS" if passed else "FAIL",
        "explanation": explanation,
        "timestamp": _utc_now_iso(),
    }


def run_simulations(scenarios: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    catalog = scenarios or SCENARIO_CATALOG
    results = [run_scenario(s) for s in catalog]

    passed = sum(1 for r in results if r.get("pass_fail") == "PASS")
    failed = len(results) - passed
    unexpected_allow = sum(
        1
        for r in results
        if r.get("pass_fail") == "FAIL"
        and r.get("actual_allowed")
        and not r.get("expected_allowed")
    )
    unexpected_block = sum(
        1
        for r in results
        if r.get("pass_fail") == "FAIL"
        and not r.get("actual_allowed")
        and r.get("expected_allowed")
    )

    summary = {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "unexpected_authorizations": unexpected_allow,
        "unexpected_blocks": unexpected_block,
    }

    payload = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-005",
        "summary": summary,
        "records": results,
    }
    _atomic_write_json(SIMULATION_JSON, payload)
    _write_csv(SIMULATION_CSV, results, SIMULATION_FIELDS)
    return payload


def load_simulation_results() -> Dict[str, Any]:
    if not SIMULATION_JSON.is_file():
        return {"records": [], "summary": {}}
    return json.loads(SIMULATION_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    out = run_simulations()
    print(json.dumps(out.get("summary", {}), indent=2))
