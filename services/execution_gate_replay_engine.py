"""
execution_gate_replay_engine.py
THP-005 — Replay historical execution gate decisions (read-only flight recorder).

Re-evaluates stored decisions via evaluate_execution_gate() without mutating history
or recording new governance evidence.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.execution_gate_evidence_registry import (
    DECISIONS_JSON,
    _load_governance_snapshot,
    _summarize_allocation_state,
    _summarize_governance_status,
    _summarize_lifecycle_state,
    _summarize_risk_state,
)
from services.triton_execution_gate import evaluate_execution_gate

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"

REPLAY_JSON = GOVERNANCE_DIR / "execution_gate_replay_results.json"
REPLAY_CSV = GOVERNANCE_DIR / "execution_gate_replay_results.csv"
DRIFT_REPORT_JSON = GOVERNANCE_DIR / "execution_gate_drift_report.json"

REPLAY_FIELDS = (
    "replay_id",
    "decision_id",
    "timestamp_replayed",
    "execution_path",
    "original_decision",
    "original_allowed",
    "historical_replay_decision",
    "historical_replay_allowed",
    "current_replay_decision",
    "current_replay_allowed",
    "classification",
    "consistency_detail",
    "risk_state_original",
    "risk_state_current",
    "governance_status_original",
    "governance_status_current",
    "environment_original",
    "environment_current",
    "changed_conditions",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_decisions() -> List[Dict[str, Any]]:
    if not DECISIONS_JSON.is_file():
        return []
    try:
        data = json.loads(DECISIONS_JSON.read_text(encoding="utf-8"))
        return list(data.get("records") or [])
    except (json.JSONDecodeError, OSError):
        return []


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
            if isinstance(flat.get("changed_conditions"), (list, dict)):
                flat["changed_conditions"] = json.dumps(flat["changed_conditions"])
            writer.writerow({k: flat.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _chain_status(record: Dict[str, Any], step: str, approved_values: tuple[str, ...]) -> bool:
    chain = record.get("authorization_chain") or []
    for item in chain:
        if isinstance(item, dict) and item.get("step") == step:
            return str(item.get("status", "")).lower() in approved_values
    return True


def reconstruct_historical_context(record: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild gate context from a stored decision record."""
    paper_or_live = str(record.get("paper_or_live") or "").lower()
    return {
        "paperMode": paper_or_live == "paper",
        "liveMode": paper_or_live == "live",
        "environment": record.get("environment") or "development",
        "lifecycleApproved": _chain_status(record, "lifecycle", ("approved",)),
        "allocationApproved": _chain_status(record, "allocation", ("approved",)),
        "riskConstraintsSatisfied": _chain_status(
            record, "risk_protection", ("satisfied", "approved")
        ),
        "brokerConnected": _chain_status(record, "broker_authorization", ("ready", "approved")),
        "executionGuardApproved": _chain_status(
            record, "broker_authorization", ("ready", "approved")
        ),
        "duplicateOrderClear": True,
        "operatorId": record.get("operator_identity") or "",
        "liveModeAuthorized": paper_or_live != "live",
        "governanceApproved": str(record.get("governance_status") or "") == "authorized",
        "executionAuthorized": bool(record.get("allowed")),
    }


def build_current_context(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build gate context using today's environment and governance snapshots."""
    snap = _load_governance_snapshot()
    paper_or_live = str(record.get("paper_or_live") or "paper").lower()
    env = os.environ.get("TRITON_ENV", record.get("environment") or "development")
    paper_enabled = os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    live_auth = os.environ.get("TRITON_ENABLE_LIVE_EXECUTION", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    gov_auth = os.environ.get("TRITON_LIVE_GOVERNANCE_APPROVED", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    gov_snap = snap.get("governance") or {}
    return {
        "paperMode": paper_or_live == "paper",
        "liveMode": paper_or_live == "live",
        "environment": env,
        "lifecycleApproved": True,
        "allocationApproved": True,
        "riskConstraintsSatisfied": not (str(_summarize_risk_state(snap)).endswith("risk_off")),
        "brokerConnected": True,
        "executionGuardApproved": True,
        "duplicateOrderClear": True,
        "liveModeAuthorized": live_auth,
        "governanceApproved": gov_auth or gov_snap.get("governance_authorized") is True,
        "executionAuthorized": paper_enabled if paper_or_live == "paper" else live_auth,
    }


def classify_replay(
    *,
    original_decision: str,
    original_allowed: bool,
    historical_eval: Dict[str, Any],
    current_eval: Dict[str, Any],
) -> tuple[str, str]:
    hist_dec = str(historical_eval.get("decision") or "")
    hist_allowed = bool(historical_eval.get("allowed"))
    cur_dec = str(current_eval.get("decision") or "")
    cur_allowed = bool(current_eval.get("allowed"))

    hist_matches = (
        hist_dec.upper() == str(original_decision).upper() and hist_allowed == original_allowed
    )
    cur_matches = (
        cur_dec.upper() == str(original_decision).upper() and cur_allowed == original_allowed
    )

    if not hist_matches:
        return (
            "REQUIRES_REVIEW",
            "Historical replay does not reproduce stored decision — record may predate context model.",
        )
    if cur_matches and hist_matches:
        return "CONSISTENT", "Current gate reproduces original authorization outcome."
    if not cur_matches:
        return (
            "DRIFT_DETECTED",
            f"Current conditions yield {cur_dec} (allowed={cur_allowed}) vs original "
            f"{original_decision} (allowed={original_allowed}).",
        )
    return "REQUIRES_REVIEW", "Authorization outcome ambiguous — manual review recommended."


def _diff_conditions(
    historical_ctx: Dict[str, Any],
    current_ctx: Dict[str, Any],
    record: Dict[str, Any],
) -> List[str]:
    changes: List[str] = []
    for key in (
        "paperMode",
        "liveMode",
        "lifecycleApproved",
        "allocationApproved",
        "riskConstraintsSatisfied",
        "governanceApproved",
        "liveModeAuthorized",
        "executionAuthorized",
        "environment",
    ):
        if historical_ctx.get(key) != current_ctx.get(key):
            changes.append(f"{key}: {historical_ctx.get(key)} -> {current_ctx.get(key)}")
    if record.get("risk_state") != _summarize_risk_state(_load_governance_snapshot()):
        changes.append(
            f"risk_state: {record.get('risk_state')} -> "
            f"{_summarize_risk_state(_load_governance_snapshot())}"
        )
    if record.get("governance_status") != _summarize_governance_status(_load_governance_snapshot()):
        changes.append(
            f"governance_status: {record.get('governance_status')} -> "
            f"{_summarize_governance_status(_load_governance_snapshot())}"
        )
    return changes


def replay_decision(record: Dict[str, Any]) -> Dict[str, Any]:
    path = str(record.get("execution_path") or "")
    historical_ctx = reconstruct_historical_context(record)
    current_ctx = build_current_context(record)

    historical_eval = evaluate_execution_gate(
        path, historical_ctx, record_evidence=False, source="replay_engine:historical"
    )
    current_eval = evaluate_execution_gate(
        path, current_ctx, record_evidence=False, source="replay_engine:current"
    )

    snap = _load_governance_snapshot()
    classification, detail = classify_replay(
        original_decision=str(record.get("execution_gate_decision") or ""),
        original_allowed=bool(record.get("allowed")),
        historical_eval=historical_eval,
        current_eval=current_eval,
    )

    return {
        "replay_id": f"rpl-{record.get('decision_id', 'unknown')}",
        "decision_id": record.get("decision_id"),
        "timestamp_replayed": _utc_now_iso(),
        "execution_path": path,
        "original_decision": record.get("execution_gate_decision"),
        "original_allowed": record.get("allowed"),
        "historical_replay_decision": historical_eval.get("decision"),
        "historical_replay_allowed": historical_eval.get("allowed"),
        "current_replay_decision": current_eval.get("decision"),
        "current_replay_allowed": current_eval.get("allowed"),
        "classification": classification,
        "consistency_detail": detail,
        "risk_state_original": record.get("risk_state"),
        "risk_state_current": _summarize_risk_state(snap),
        "governance_status_original": record.get("governance_status"),
        "governance_status_current": _summarize_governance_status(snap),
        "environment_original": record.get("environment"),
        "environment_current": current_ctx.get("environment"),
        "changed_conditions": _diff_conditions(historical_ctx, current_ctx, record),
        "original_request": {
            "action_requested": record.get("action_requested"),
            "symbol(s)": record.get("symbol(s)"),
            "requesting_service": record.get("requesting_service"),
            "operator_identity": record.get("operator_identity"),
        },
    }


def run_replay(*, limit: Optional[int] = None) -> Dict[str, Any]:
    """Replay all historical decisions and persist results."""
    records = _read_decisions()
    if limit is not None:
        records = records[-limit:]

    results = [replay_decision(r) for r in records]
    summary = {
        "total": len(results),
        "consistent": sum(1 for r in results if r.get("classification") == "CONSISTENT"),
        "drift_detected": sum(1 for r in results if r.get("classification") == "DRIFT_DETECTED"),
        "requires_review": sum(1 for r in results if r.get("classification") == "REQUIRES_REVIEW"),
    }

    payload = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-005",
        "summary": summary,
        "records": results,
    }
    _atomic_write_json(REPLAY_JSON, payload)
    _write_csv(REPLAY_CSV, results, REPLAY_FIELDS)
    build_drift_report(results, summary)
    return payload


def build_drift_report(
    replay_results: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if replay_results is None:
        doc = json.loads(REPLAY_JSON.read_text(encoding="utf-8")) if REPLAY_JSON.is_file() else {}
        replay_results = doc.get("records") or []
        summary = doc.get("summary") or {}

    drift = [r for r in replay_results if r.get("classification") == "DRIFT_DETECTED"]
    review = [r for r in replay_results if r.get("classification") == "REQUIRES_REVIEW"]

    path_drift: Dict[str, int] = {}
    for row in drift:
        p = str(row.get("execution_path") or "unknown")
        path_drift[p] = path_drift.get(p, 0) + 1

    report = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-005",
        "summary": summary,
        "drift_count": len(drift),
        "review_count": len(review),
        "drift_by_path": path_drift,
        "drift_records": drift,
        "review_records": review,
        "policy_observations": [
            "Authorization drift indicates current gate conditions differ from historical decision context.",
            "Review records where historical replay cannot reproduce stored outcomes.",
        ],
    }
    _atomic_write_json(DRIFT_REPORT_JSON, report)
    return report


def load_replay_results() -> Dict[str, Any]:
    if not REPLAY_JSON.is_file():
        return {"records": [], "summary": {}}
    return json.loads(REPLAY_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    out = run_replay()
    print(json.dumps(out.get("summary", {}), indent=2))
