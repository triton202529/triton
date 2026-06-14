"""
execution_gate_stress_suite.py
THP-006 — Stress test suite for execution gate resilience (read-only).

All tests call evaluate_execution_gate(record_evidence=False).
No broker API calls, order placement, or portfolio mutations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.triton_execution_gate import EXECUTION_PATHS, evaluate_execution_gate

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"

STRESS_JSON = GOVERNANCE_DIR / "execution_gate_stress_results.json"
STRESS_CSV = GOVERNANCE_DIR / "execution_gate_stress_results.csv"

STRESS_FIELDS = (
    "stress_id",
    "category",
    "scenario_name",
    "execution_path",
    "input_summary",
    "expected_decision",
    "expected_allowed",
    "actual_decision",
    "actual_allowed",
    "pass_fail",
    "determinism_class",
    "explanation",
    "timestamp",
)

STRESS_CATALOG: List[Dict[str, Any]] = [
    # 1. Broker failure scenarios
    {
        "stress_id": "str-broker-offline",
        "category": "broker_safety",
        "scenario_name": "Broker offline",
        "path": EXECUTION_PATHS["BROKER_SUBMISSION"],
        "context": {
            "brokerConnected": False,
            "lifecycleApproved": True,
            "executionGuardApproved": True,
            "duplicateOrderClear": True,
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "stress_id": "str-broker-timeout",
        "category": "broker_safety",
        "scenario_name": "Broker timeout (simulated unavailable)",
        "path": EXECUTION_PATHS["BROKER_SUBMISSION"],
        "context": {
            "brokerConnected": False,
            "executionAuthorized": True,
            "executionGuardApproved": False,
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "stress_id": "str-broker-auth-fail",
        "category": "broker_safety",
        "scenario_name": "Broker authentication failure",
        "path": EXECUTION_PATHS["BROKER_SUBMISSION"],
        "context": {
            "brokerConnected": False,
            "executionAuthorized": False,
            "lifecycleApproved": False,
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "stress_id": "str-paper-live-routing",
        "category": "broker_safety",
        "scenario_name": "Paper broker attempting live routing",
        "path": EXECUTION_PATHS["LIVE_EXECUTION"],
        "context": {
            "paperMode": True,
            "liveMode": True,
            "liveModeAuthorized": False,
            "governanceApproved": False,
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    # 2. Governance failure scenarios
    {
        "stress_id": "str-gov-missing",
        "category": "governance_integrity",
        "scenario_name": "Missing governance authorization",
        "path": EXECUTION_PATHS["LIVE_EXECUTION"],
        "context": {
            "liveMode": True,
            "liveModeAuthorized": True,
            "governanceApproved": False,
            "lifecycleApproved": True,
            "allocationApproved": True,
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "stress_id": "str-gov-invalid-chain",
        "category": "governance_integrity",
        "scenario_name": "Invalid approval chain",
        "path": EXECUTION_PATHS["GOVERNANCE_OVERRIDE"],
        "context": {
            "operatorAuthorized": False,
            "overrideReason": "",
            "operationalContext": "",
        },
        "expected_decision": "REVIEW_REQUIRED",
        "expected_allowed": False,
    },
    {
        "stress_id": "str-gov-unauthorized-override",
        "category": "governance_integrity",
        "scenario_name": "Unauthorized override attempt",
        "path": EXECUTION_PATHS["MANUAL_OPERATOR_OVERRIDE"],
        "context": {"operatorAuthorized": False},
        "expected_decision": "REVIEW_REQUIRED",
        "expected_allowed": False,
    },
    # 3. Risk protection failures
    {
        "stress_id": "str-risk-lockdown",
        "category": "risk_protection",
        "scenario_name": "Risk lockdown active",
        "path": EXECUTION_PATHS["CAPITAL_DEPLOYMENT"],
        "context": {
            "lifecycleApproved": True,
            "allocationApproved": True,
            "riskConstraintsSatisfied": False,
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "stress_id": "str-risk-emergency-no-trigger",
        "category": "risk_protection",
        "scenario_name": "Emergency action without trigger",
        "path": EXECUTION_PATHS["RISK_EMERGENCY_ACTION"],
        "context": {"riskEmergencyTriggered": False, "emergencyAction": "RISK_LOCKDOWN"},
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "stress_id": "str-risk-invalid-portfolio",
        "category": "risk_protection",
        "scenario_name": "Invalid portfolio state (allocation missing)",
        "path": EXECUTION_PATHS["POSITION_ENTRY"],
        "context": {
            "lifecycleApproved": True,
            "allocationApproved": False,
            "riskConstraintsSatisfied": True,
            "environment": "development",
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    # 5. Environment boundary testing
    {
        "stress_id": "str-env-live-in-paper",
        "category": "environment_isolation",
        "scenario_name": "Live execution in paper environment",
        "path": EXECUTION_PATHS["PAPER_EXECUTION"],
        "context": {"paperMode": False, "liveMode": True},
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "stress_id": "str-env-paper-live-creds",
        "category": "environment_isolation",
        "scenario_name": "Paper execution with live mode flags",
        "path": EXECUTION_PATHS["PAPER_EXECUTION"],
        "context": {
            "paperMode": False,
            "liveMode": True,
            "executionAuthorized": False,
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "stress_id": "str-env-missing-vars",
        "category": "environment_isolation",
        "scenario_name": "Missing environment variables (paper/live unset)",
        "path": EXECUTION_PATHS["PAPER_EXECUTION"],
        "context": {
            "paperMode": False,
            "liveMode": True,
        },
        "expected_decision": "BLOCK",
        "expected_allowed": False,
    },
    {
        "stress_id": "str-env-invalid-mode",
        "category": "environment_isolation",
        "scenario_name": "Invalid execution mode combination",
        "path": EXECUTION_PATHS["LIVE_EXECUTION"],
        "context": {
            "liveMode": True,
            "liveModeAuthorized": False,
            "governanceApproved": False,
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
            if isinstance(flat.get("input_summary"), dict):
                flat["input_summary"] = json.dumps(flat["input_summary"])
            writer.writerow({k: flat.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _evaluate(path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    return evaluate_execution_gate(path, context, record_evidence=False, source="stress_suite")


def _check_determinism(path: str, context: Dict[str, Any], runs: int = 3) -> str:
    results = []
    for _ in range(runs):
        ev = _evaluate(path, context)
        fingerprint = f"{ev.get('decision')}|{ev.get('allowed')}|{ev.get('reason', '')[:80]}"
        results.append(fingerprint)
    if len(set(results)) == 1:
        return "DETERMINISTIC"
    if len(set(results)) == runs:
        return "NON_DETERMINISTIC"
    return "REVIEW_REQUIRED"


def run_stress_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    path = scenario["path"]
    context = dict(scenario.get("context") or {})
    evaluation = _evaluate(path, context)

    actual_decision = str(evaluation.get("decision") or "")
    actual_allowed = bool(evaluation.get("allowed"))
    expected_decision = str(scenario.get("expected_decision") or "")
    expected_allowed = bool(scenario.get("expected_allowed"))

    passed = (
        actual_decision.upper() == expected_decision.upper() and actual_allowed == expected_allowed
    )
    determinism = _check_determinism(path, context)

    explanation = evaluation.get("reason") or ""
    if not passed:
        explanation = (
            f"Expected {expected_decision}/allowed={expected_allowed}, "
            f"got {actual_decision}/allowed={actual_allowed}. {explanation}"
        )

    return {
        "stress_id": scenario.get("stress_id"),
        "category": scenario.get("category"),
        "scenario_name": scenario.get("scenario_name"),
        "execution_path": path,
        "input_summary": context,
        "expected_decision": expected_decision,
        "expected_allowed": expected_allowed,
        "actual_decision": actual_decision,
        "actual_allowed": actual_allowed,
        "pass_fail": "PASS" if passed else "FAIL",
        "determinism_class": determinism,
        "explanation": explanation,
        "timestamp": _utc_now_iso(),
    }


def run_concurrency_burst(
    *,
    count: int = 100,
    path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Simulate concurrent execution requests — verify deterministic outcomes."""
    path = path or EXECUTION_PATHS["BROKER_SUBMISSION"]
    context = context or {
        "lifecycleApproved": True,
        "executionAuthorized": True,
        "brokerConnected": True,
        "executionGuardApproved": True,
        "duplicateOrderClear": True,
        "environment": "development",
    }

    def _one(_: int) -> str:
        ev = _evaluate(path, context)
        return f"{ev.get('decision')}|{ev.get('allowed')}"

    fingerprints: List[str] = []
    with ThreadPoolExecutor(max_workers=min(20, count)) as pool:
        futures = [pool.submit(_one, i) for i in range(count)]
        for fut in as_completed(futures):
            fingerprints.append(fut.result())

    unique = set(fingerprints)
    deterministic = len(unique) == 1
    no_accidental_allow = all(
        not fp.endswith("|True") or fp.startswith("ALLOW|True")
        for fp in fingerprints
        if "BLOCK" not in fp
    )

    return {
        "stress_id": "str-concurrency-burst",
        "category": "authorization_determinism",
        "scenario_name": f"Concurrent burst ({count} requests)",
        "execution_path": path,
        "input_summary": {"concurrent_requests": count, **context},
        "expected_decision": "DETERMINISTIC",
        "expected_allowed": deterministic,
        "actual_decision": "UNIQUE" if len(unique) == 1 else f"VARIED({len(unique)})",
        "actual_allowed": deterministic,
        "pass_fail": "PASS" if deterministic else "FAIL",
        "determinism_class": "DETERMINISTIC" if deterministic else "NON_DETERMINISTIC",
        "explanation": (
            f"{count} concurrent evaluations produced {len(unique)} unique outcome(s). "
            f"No accidental authorization drift detected: {no_accidental_allow}"
        ),
        "timestamp": _utc_now_iso(),
    }


def run_duplicate_identifier_test() -> Dict[str, Any]:
    """Rapid repeated requests with same execution identifier."""
    path = EXECUTION_PATHS["BROKER_SUBMISSION"]
    context = {
        "lifecycleApproved": True,
        "executionAuthorized": True,
        "brokerConnected": True,
        "executionGuardApproved": True,
        "duplicateOrderClear": False,
        "auditTrailId": "stress-dup-001",
    }
    results = [_evaluate(path, context) for _ in range(10)]
    all_blocked = all(not r.get("allowed") for r in results)
    all_same = len({r.get("decision") for r in results}) == 1

    return {
        "stress_id": "str-duplicate-identifier",
        "category": "authorization_determinism",
        "scenario_name": "Duplicate execution identifier burst",
        "execution_path": path,
        "input_summary": context,
        "expected_decision": "BLOCK",
        "expected_allowed": False,
        "actual_decision": str(results[0].get("decision") if results else ""),
        "actual_allowed": not all_blocked,
        "pass_fail": "PASS" if all_blocked and all_same else "FAIL",
        "determinism_class": "DETERMINISTIC" if all_same else "NON_DETERMINISTIC",
        "explanation": "Duplicate order protection must block all repeated submissions.",
        "timestamp": _utc_now_iso(),
    }


def run_authorization_fingerprint_test() -> Dict[str, Any]:
    """Verify same input produces identical decision fingerprint."""
    path = EXECUTION_PATHS["LIVE_EXECUTION"]
    context = {
        "liveMode": True,
        "liveModeAuthorized": False,
        "governanceApproved": False,
    }
    runs = [_evaluate(path, context) for _ in range(5)]
    fps = [
        hashlib.sha256(
            f"{r.get('decision')}|{r.get('allowed')}|{r.get('reason')}".encode()
        ).hexdigest()[:16]
        for r in runs
    ]
    deterministic = len(set(fps)) == 1

    return {
        "stress_id": "str-auth-fingerprint",
        "category": "authorization_determinism",
        "scenario_name": "Authorization fingerprint consistency",
        "execution_path": path,
        "input_summary": context,
        "expected_decision": "DETERMINISTIC",
        "expected_allowed": True,
        "actual_decision": fps[0] if fps else "",
        "actual_allowed": deterministic,
        "pass_fail": "PASS" if deterministic else "FAIL",
        "determinism_class": "DETERMINISTIC" if deterministic else "NON_DETERMINISTIC",
        "explanation": f"Fingerprint stable across 5 runs: {deterministic}",
        "timestamp": _utc_now_iso(),
    }


def run_stress_suite(*, concurrency_count: int = 100) -> Dict[str, Any]:
    results = [run_stress_scenario(s) for s in STRESS_CATALOG]
    results.append(run_concurrency_burst(count=concurrency_count))
    results.append(run_duplicate_identifier_test())
    results.append(run_authorization_fingerprint_test())

    passed = sum(1 for r in results if r.get("pass_fail") == "PASS")
    failed = len(results) - passed
    categories: Dict[str, Dict[str, int]] = {}
    for row in results:
        cat = str(row.get("category") or "unknown")
        categories.setdefault(cat, {"pass": 0, "fail": 0, "total": 0})
        categories[cat]["total"] += 1
        if row.get("pass_fail") == "PASS":
            categories[cat]["pass"] += 1
        else:
            categories[cat]["fail"] += 1

    payload = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-006",
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "categories": categories,
        },
        "records": results,
    }
    _atomic_write_json(STRESS_JSON, payload)
    _write_csv(STRESS_CSV, results, STRESS_FIELDS)
    return payload


def load_stress_results() -> Dict[str, Any]:
    if not STRESS_JSON.is_file():
        return {"records": [], "summary": {}}
    return json.loads(STRESS_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    out = run_stress_suite()
    print(json.dumps(out.get("summary", {}), indent=2))
