"""
execution_gate_failure_investigator.py
THP-005 — Human-readable failure analysis for blocked or drifted execution decisions.

Read-only analysis over replay and simulation artifacts plus decision registry.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.execution_gate_evidence_registry import DECISIONS_JSON
from services.execution_gate_replay_engine import REPLAY_JSON, load_replay_results
from services.execution_gate_simulator import SIMULATION_JSON, load_simulation_results

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"

FAILURE_JSON = GOVERNANCE_DIR / "execution_gate_failure_analysis.json"
FAILURE_CSV = GOVERNANCE_DIR / "execution_gate_failure_analysis.csv"

FAILURE_FIELDS = (
    "analysis_id",
    "source_type",
    "source_id",
    "execution_path",
    "decision",
    "allowed",
    "primary_rejection_reason",
    "failed_stage",
    "missing_evidence",
    "missing_approvals",
    "risk_conflicts",
    "environment_conflicts",
    "human_explanation",
    "timestamp",
)

STAGE_ORDER = (
    "signal",
    "lifecycle",
    "allocation",
    "risk_protection",
    "execution_gate",
    "broker_authorization",
    "execution_outcome",
)


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
            for key in (
                "missing_evidence",
                "missing_approvals",
                "risk_conflicts",
                "environment_conflicts",
            ):
                if isinstance(flat.get(key), list):
                    flat[key] = json.dumps(flat[key])
            writer.writerow({k: flat.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _failed_stage_from_chain(chain: List[Dict[str, Any]]) -> str:
    for step in STAGE_ORDER:
        for item in chain:
            if not isinstance(item, dict) or item.get("step") != step:
                continue
            status = str(item.get("status", "")).lower()
            if status in ("blocked", "violated", "pending", "disabled"):
                return step
    return "execution_gate"


def _failed_stage_from_reason(reason: str, path: str) -> str:
    r = reason.lower()
    if "lifecycle" in r:
        return "lifecycle"
    if "allocation" in r:
        return "allocation"
    if "risk" in r or "concentration" in r or "sizing" in r:
        return "risk_protection"
    if "governance" in r or "operator" in r or "override" in r:
        return "execution_gate"
    if "broker" in r or "connectivity" in r or "guard" in r or "duplicate" in r:
        return "broker_authorization"
    if "paper" in r or "live" in r or "production" in r or "environment" in r:
        return "execution_gate"
    if path in ("live_execution", "paper_execution"):
        return "execution_gate"
    return "execution_gate"


def _missing_approvals(reason: str, path: str) -> List[str]:
    missing: List[str] = []
    r = reason.lower()
    if "governance" in r:
        missing.append("governance_approval")
    if "kyc" in r or "lifecycle" in r:
        missing.append("lifecycle_approval")
    if "allocation" in r:
        missing.append("portfolio_allocation_approval")
    if "live" in r and "authorization" in r:
        missing.append("live_execution_authorization")
    if "production" in r:
        missing.append("production_readiness")
    if "operator" in r:
        missing.append("operator_authorization")
    if path == "live_execution" and "governance" not in r:
        missing.append("live_governance_checkpoint")
    return missing


def _risk_conflicts(reason: str) -> List[str]:
    r = reason.lower()
    conflicts: List[str] = []
    if "risk" in r and "constraint" in r:
        conflicts.append("risk_constraints_not_satisfied")
    if "concentration" in r:
        conflicts.append("concentration_limit")
    if "emergency" in r and "no active" in r:
        conflicts.append("risk_emergency_not_triggered")
    if "drawdown" in r:
        conflicts.append("drawdown_limit")
    return conflicts


def _environment_conflicts(reason: str, record: Dict[str, Any]) -> List[str]:
    r = reason.lower()
    conflicts: List[str] = []
    if "paper" in r and "live" in r:
        conflicts.append("paper_live_separation")
    if "production" in r:
        conflicts.append("production_environment_restriction")
    if record.get("environment") and "environment" in r:
        conflicts.append(f"environment={record.get('environment')}")
    return conflicts


def build_human_explanation(
    *,
    path: str,
    decision: str,
    reason: str,
    failed_stage: str,
    missing_approvals: List[str],
    risk_conflicts: List[str],
) -> str:
    if decision.upper() == "ALLOW":
        return f"Execution on path '{path}' was authorized: {reason}"

    if path == "live_execution" and "governance" in reason.lower():
        return (
            "Live execution denied because governance approval was absent while live "
            "capital deployment requires explicit institutional authorization."
        )
    if "production" in reason.lower() and "disabled" in reason.lower():
        return (
            "Live capital deployment remains disabled until explicit production "
            "authorization is granted through institutional readiness controls."
        )
    if "broker" in reason.lower() and "connectivity" in reason.lower():
        return (
            "Broker submission blocked because active broker connectivity is required "
            "before any order may reach the execution layer."
        )
    if "duplicate order" in reason.lower():
        return (
            "Broker submission blocked by duplicate order protection to prevent "
            "unintentional repeated capital deployment."
        )
    if "paper" in reason.lower() and "live" in reason.lower():
        return (
            "Paper execution blocked because paper and live broker environments "
            "must remain separated."
        )
    if risk_conflicts:
        return (
            f"Execution on '{path}' blocked at {failed_stage} due to risk protection: " f"{reason}"
        )
    if missing_approvals:
        return (
            f"Execution on '{path}' blocked at {failed_stage}. Missing approvals: "
            f"{', '.join(missing_approvals)}. {reason}"
        )
    return f"Execution on '{path}' received {decision} at {failed_stage}: {reason}"


def analyze_decision_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if record.get("allowed"):
        return None

    reason = str(record.get("decision_reason") or record.get("consistency_detail") or "")
    path = str(record.get("execution_path") or "")
    decision = str(
        record.get("execution_gate_decision") or record.get("current_replay_decision") or "BLOCK"
    )
    chain = record.get("authorization_chain") or []
    if isinstance(chain, str):
        try:
            chain = json.loads(chain)
        except json.JSONDecodeError:
            chain = []

    failed_stage = (
        _failed_stage_from_chain(chain) if chain else _failed_stage_from_reason(reason, path)
    )
    missing = _missing_approvals(reason, path)
    risk = _risk_conflicts(reason)
    env = _environment_conflicts(reason, record)

    return {
        "analysis_id": f"inv-{record.get('decision_id') or record.get('replay_id') or record.get('simulation_id')}",
        "source_type": record.get("_source_type", "decision_registry"),
        "source_id": record.get("decision_id")
        or record.get("replay_id")
        or record.get("simulation_id"),
        "execution_path": path,
        "decision": decision,
        "allowed": False,
        "primary_rejection_reason": reason,
        "failed_stage": failed_stage,
        "missing_evidence": [],
        "missing_approvals": missing,
        "risk_conflicts": risk,
        "environment_conflicts": env,
        "human_explanation": build_human_explanation(
            path=path,
            decision=decision,
            reason=reason,
            failed_stage=failed_stage,
            missing_approvals=missing,
            risk_conflicts=risk,
        ),
        "timestamp": _utc_now_iso(),
    }


def run_failure_investigation() -> Dict[str, Any]:
    analyses: List[Dict[str, Any]] = []

    if DECISIONS_JSON.is_file():
        doc = json.loads(DECISIONS_JSON.read_text(encoding="utf-8"))
        for rec in doc.get("records") or []:
            rec = dict(rec)
            rec["_source_type"] = "decision_registry"
            item = analyze_decision_record(rec)
            if item:
                analyses.append(item)

    replay = load_replay_results()
    for rec in replay.get("records") or []:
        if rec.get("classification") not in ("DRIFT_DETECTED", "REQUIRES_REVIEW"):
            continue
        row = dict(rec)
        row["_source_type"] = "replay_engine"
        row["execution_gate_decision"] = rec.get("current_replay_decision")
        row["allowed"] = rec.get("current_replay_allowed")
        row["decision_reason"] = rec.get("consistency_detail")
        item = analyze_decision_record(row)
        if item:
            analyses.append(item)

    sim = load_simulation_results()
    for rec in sim.get("records") or []:
        if rec.get("pass_fail") != "FAIL":
            continue
        row = dict(rec)
        row["_source_type"] = "simulator"
        row["execution_path"] = rec.get("execution_path")
        row["execution_gate_decision"] = rec.get("actual_decision")
        row["allowed"] = rec.get("actual_allowed")
        row["decision_reason"] = rec.get("explanation")
        item = analyze_decision_record(row)
        if item:
            analyses.append(item)

    reason_counts = Counter(a.get("primary_rejection_reason", "")[:120] for a in analyses)
    stage_counts = Counter(a.get("failed_stage", "") for a in analyses)
    risk_failures = sum(1 for a in analyses if a.get("risk_conflicts"))
    gov_failures = sum(
        1
        for a in analyses
        if "governance" in str(a.get("primary_rejection_reason", "")).lower()
        or "governance_approval" in str(a.get("missing_approvals", ""))
    )

    summary = {
        "total_analyses": len(analyses),
        "top_rejection_reasons": dict(reason_counts.most_common(10)),
        "top_failed_stages": dict(stage_counts.most_common()),
        "risk_related_failures": risk_failures,
        "governance_related_failures": gov_failures,
    }

    payload = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-005",
        "summary": summary,
        "records": analyses,
    }
    _atomic_write_json(FAILURE_JSON, payload)
    _write_csv(FAILURE_CSV, analyses, FAILURE_FIELDS)
    return payload


def load_failure_analysis() -> Dict[str, Any]:
    if not FAILURE_JSON.is_file():
        return {"records": [], "summary": {}}
    return json.loads(FAILURE_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    out = run_failure_investigation()
    print(json.dumps(out.get("summary", {}), indent=2))
