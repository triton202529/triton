"""
execution_gate_chaos_engine.py
THP-006 — Controlled fault injection for execution gate chaos testing (read-only).

Simulates corrupted/missing state via gate context — does not mutate files or evidence.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from services.triton_execution_gate import EXECUTION_PATHS, evaluate_execution_gate

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"

CHAOS_JSON = GOVERNANCE_DIR / "execution_gate_chaos_results.json"
CHAOS_CSV = GOVERNANCE_DIR / "execution_gate_chaos_results.csv"

CHAOS_FIELDS = (
    "chaos_id",
    "fault_type",
    "scenario_name",
    "execution_path",
    "injected_fault",
    "expected_safe",
    "actual_decision",
    "actual_allowed",
    "fail_safe",
    "pass_fail",
    "explanation",
    "timestamp",
)

CHAOS_SCENARIOS: List[Dict[str, Any]] = [
    {
        "chaos_id": "chaos-missing-governance",
        "fault_type": "missing_authorization",
        "scenario_name": "Missing governance artifact (simulated)",
        "path": EXECUTION_PATHS["LIVE_EXECUTION"],
        "context": {"liveMode": True, "governanceApproved": False, "liveModeAuthorized": False},
        "expected_safe": True,
    },
    {
        "chaos_id": "chaos-empty-operator",
        "fault_type": "empty_authorization",
        "scenario_name": "Empty operator authorization record",
        "path": EXECUTION_PATHS["MANUAL_OPERATOR_OVERRIDE"],
        "context": {"operatorAuthorized": False, "operatorId": "", "overrideReason": ""},
        "expected_safe": True,
    },
    {
        "chaos_id": "chaos-invalid-env",
        "fault_type": "invalid_environment",
        "scenario_name": "Invalid environment state",
        "path": EXECUTION_PATHS["LIVE_EXECUTION"],
        "context": {
            "environment": "",
            "liveMode": True,
            "paperMode": True,
            "liveModeAuthorized": False,
        },
        "expected_safe": True,
    },
    {
        "chaos_id": "chaos-unknown-path",
        "fault_type": "unexpected_input",
        "scenario_name": "Unknown execution path",
        "path": "wallet_funding_legacy_removed",
        "context": {},
        "expected_safe": True,
    },
    {
        "chaos_id": "chaos-partial-context",
        "fault_type": "partial_failure",
        "scenario_name": "Partial authorization context",
        "path": EXECUTION_PATHS["CAPITAL_DEPLOYMENT"],
        "context": {"lifecycleApproved": False, "allocationApproved": False},
        "expected_safe": True,
    },
    {
        "chaos_id": "chaos-risk-kill",
        "fault_type": "kill_switch",
        "scenario_name": "Emergency kill switch (simulated lockdown)",
        "path": EXECUTION_PATHS["BROKER_SUBMISSION"],
        "context": {
            "riskConstraintsSatisfied": False,
            "executionGuardApproved": False,
            "brokerConnected": False,
            "lifecycleApproved": False,
        },
        "expected_safe": True,
    },
    {
        "chaos_id": "chaos-type-coercion",
        "fault_type": "unexpected_data_type",
        "scenario_name": "Unexpected data types in context",
        "path": EXECUTION_PATHS["LIVE_EXECUTION"],
        "context": {
            "liveMode": "true",
            "liveModeAuthorized": 0,
            "governanceApproved": None,
            "productionReadinessOverride": "no",
        },
        "expected_safe": True,
    },
    {
        "chaos_id": "chaos-null-context",
        "fault_type": "missing_files",
        "scenario_name": "Minimal empty context",
        "path": EXECUTION_PATHS["BROKER_SUBMISSION"],
        "context": {},
        "expected_safe": True,
    },
    {
        "chaos_id": "chaos-live-bypass-attempt",
        "fault_type": "authorization_bypass",
        "scenario_name": "Live bypass without credentials",
        "path": EXECUTION_PATHS["LIVE_EXECUTION"],
        "context": {
            "liveMode": True,
            "liveModeAuthorized": True,
            "governanceApproved": False,
            "productionReadinessOverride": False,
            "environment": "production",
        },
        "expected_safe": True,
    },
    {
        "chaos_id": "chaos-paper-disabled",
        "fault_type": "invalid_environment",
        "scenario_name": "Missing broker authorization on submission path",
        "path": EXECUTION_PATHS["BROKER_SUBMISSION"],
        "context": {
            "brokerConnected": True,
            "executionAuthorized": False,
            "lifecycleApproved": False,
        },
        "expected_safe": True,
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
            if isinstance(flat.get("injected_fault"), dict):
                flat["injected_fault"] = json.dumps(flat["injected_fault"])
            writer.writerow({k: flat.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def run_chaos_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    path = scenario["path"]
    context = dict(scenario.get("context") or {})

    try:
        evaluation = evaluate_execution_gate(
            path, context, record_evidence=False, source=f"chaos:{scenario.get('chaos_id')}"
        )
    except Exception as exc:
        evaluation = {
            "decision": "BLOCK",
            "allowed": False,
            "reason": f"Gate raised exception (fail-safe): {exc}",
            "path": path,
        }

    actual_allowed = bool(evaluation.get("allowed"))
    actual_decision = str(evaluation.get("decision") or "BLOCK")
    fail_safe = not actual_allowed or actual_decision in (
        "BLOCK",
        "REVIEW_REQUIRED",
        "DISABLED",
        "SANDBOX_ONLY",
    )
    if actual_decision == "ALLOW" and path in (
        EXECUTION_PATHS["LIVE_EXECUTION"],
        EXECUTION_PATHS["BROKER_SUBMISSION"],
        EXECUTION_PATHS["CAPITAL_DEPLOYMENT"],
    ):
        fail_safe = False

    passed = bool(scenario.get("expected_safe")) and fail_safe

    return {
        "chaos_id": scenario.get("chaos_id"),
        "fault_type": scenario.get("fault_type"),
        "scenario_name": scenario.get("scenario_name"),
        "execution_path": path,
        "injected_fault": context,
        "expected_safe": scenario.get("expected_safe"),
        "actual_decision": actual_decision,
        "actual_allowed": actual_allowed,
        "fail_safe": fail_safe,
        "pass_fail": "PASS" if passed else "FAIL",
        "explanation": evaluation.get("reason") or "",
        "timestamp": _utc_now_iso(),
    }


def run_chaos_engine() -> Dict[str, Any]:
    results = [run_chaos_scenario(s) for s in CHAOS_SCENARIOS]
    passed = sum(1 for r in results if r.get("pass_fail") == "PASS")
    accidental_allows = sum(
        1
        for r in results
        if r.get("actual_allowed")
        and r.get("execution_path")
        in (
            EXECUTION_PATHS["LIVE_EXECUTION"],
            EXECUTION_PATHS["BROKER_SUBMISSION"],
            EXECUTION_PATHS["CAPITAL_DEPLOYMENT"],
        )
    )

    payload = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-006",
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "accidental_authorizations": accidental_allows,
            "all_fail_safe": accidental_allows == 0,
        },
        "records": results,
    }
    _atomic_write_json(CHAOS_JSON, payload)
    _write_csv(CHAOS_CSV, results, CHAOS_FIELDS)
    return payload


def load_chaos_results() -> Dict[str, Any]:
    if not CHAOS_JSON.is_file():
        return {"records": [], "summary": {}}
    return json.loads(CHAOS_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    out = run_chaos_engine()
    print(json.dumps(out.get("summary", {}), indent=2))
