"""
execution_time_machine.py
THP-009 — Immutable execution snapshots for forensic time travel (read-only source).

Captures point-in-time institutional state linked to execution gate decisions.
Never modifies historical evidence in execution_gate_decisions.json or evidence.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.execution_gate_evidence_registry import (
    _load_governance_snapshot,
    load_decisions,
    load_evidence,
)
from services.execution_gate_degradation_manager import load_operational_status
from services.execution_gate_health_monitor import load_health_report
from services.execution_gate_resilience_certification import load_certification
from services.execution_governance_council import determine_authority_level

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"
RESULTS_DIR = ROOT / "data" / "results"

SNAPSHOTS_JSON = GOVERNANCE_DIR / "execution_snapshots.json"
SNAPSHOT_INDEX_CSV = GOVERNANCE_DIR / "execution_snapshot_index.csv"

APPROVAL_REQUESTS_JSON = GOVERNANCE_DIR / "execution_approval_requests.json"
ESCALATIONS_JSON = GOVERNANCE_DIR / "execution_escalations.json"

SNAPSHOT_INDEX_FIELDS = (
    "snapshot_id",
    "decision_id",
    "execution_identifier",
    "timestamp",
    "execution_path",
    "fingerprint",
    "symbol(s)",
    "paper_or_live",
    "decision",
    "allowed",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_index(rows: List[Dict[str, Any]]) -> None:
    GOVERNANCE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SNAPSHOT_INDEX_CSV.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SNAPSHOT_INDEX_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SNAPSHOT_INDEX_FIELDS})
    os.replace(tmp, SNAPSHOT_INDEX_CSV)


def _load_snapshots_doc() -> Dict[str, Any]:
    if not SNAPSHOTS_JSON.is_file():
        return {"records": [], "generated_at": _utc_now_iso(), "framework": "THP-009"}
    try:
        doc = json.loads(SNAPSHOTS_JSON.read_text(encoding="utf-8"))
        doc.setdefault("records", [])
        return doc
    except Exception:
        return {"records": [], "generated_at": _utc_now_iso(), "framework": "THP-009"}


def _linked_evidence(decision_id: str) -> List[Dict[str, Any]]:
    return [e for e in load_evidence() if e.get("decision_id") == decision_id]


def _linked_approvals(execution_identifier: str) -> List[Dict[str, Any]]:
    if not execution_identifier or not APPROVAL_REQUESTS_JSON.is_file():
        return []
    try:
        doc = json.loads(APPROVAL_REQUESTS_JSON.read_text(encoding="utf-8"))
        return [
            r
            for r in doc.get("records") or []
            if r.get("execution_identifier") == execution_identifier
        ]
    except Exception:
        return []


def _linked_escalations(execution_identifier: str) -> List[Dict[str, Any]]:
    if not execution_identifier or not ESCALATIONS_JSON.is_file():
        return []
    try:
        doc = json.loads(ESCALATIONS_JSON.read_text(encoding="utf-8"))
        return [
            r
            for r in doc.get("records") or []
            if r.get("execution_identifier") == execution_identifier
        ]
    except Exception:
        return []


def _load_positions_summary() -> Dict[str, Any]:
    path = RESULTS_DIR / "positions_snapshot.csv"
    if not path.is_file():
        return {}
    try:
        import csv as csvmod

        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csvmod.DictReader(f))
        return {"position_count": len(rows), "positions": rows[:50]}
    except Exception:
        return {}


def _snapshot_id() -> str:
    return f"eg-snap-{uuid.uuid4().hex[:12]}"


def capture_snapshot_from_decision(
    decision: Dict[str, Any],
    *,
    operational_overlay: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build immutable snapshot from a decision record + linked artifacts.
    Source decisions are copied, never mutated.
    """
    decision_id = str(decision.get("decision_id") or "")
    execution_identifier = str(decision.get("execution_identifier") or "")
    intel = operational_overlay or _load_governance_snapshot()
    health = load_health_report()
    certification = load_certification()
    operational = load_operational_status()

    path = str(decision.get("execution_path") or "")
    frozen_decision = json.loads(json.dumps(decision, default=str))
    frozen_evidence = json.loads(json.dumps(_linked_evidence(decision_id), default=str))

    authority_level, triggers = determine_authority_level(
        path=path,
        context={
            "paperMode": str(decision.get("paper_or_live", "")).lower() == "paper",
            "liveMode": str(decision.get("paper_or_live", "")).lower() == "live",
            "governanceApproved": str(decision.get("governance_status")) == "authorized",
            "lifecycleApproved": True,
        },
        evaluation={
            "decision": decision.get("execution_gate_decision"),
            "allowed": decision.get("allowed"),
        },
        capital_exposure=decision.get("capital_exposure"),
    )

    body = {
        "execution_identity": {
            "decision_id": decision_id,
            "execution_identifier": execution_identifier,
            "timestamp": decision.get("timestamp"),
            "requesting_service": decision.get("requesting_service"),
            "execution_path": path,
            "symbol(s)": decision.get("symbol(s)"),
            "action_requested": decision.get("action_requested"),
            "operator_identity": decision.get("operator_identity"),
        },
        "market_intelligence": {
            "signal_state": intel.get("signal") or {},
            "lifecycle_state": decision.get("lifecycle_state"),
            "lifecycle_from_signal": (intel.get("signal") or {}).get("lifecycle_state"),
            "model_outputs": (intel.get("signal") or {}).get("scores"),
            "confidence_scores": (intel.get("signal") or {}).get("confidence"),
        },
        "portfolio_state": {
            "allocation_summary": intel.get("allocation") or {},
            "allocation_state": decision.get("allocation_state"),
            "positions_summary": _load_positions_summary(),
            "capital_exposure": decision.get("capital_exposure"),
        },
        "risk_state": {
            "recorded_risk_state": decision.get("risk_state"),
            "adaptive_risk": intel.get("risk") or {},
            "restrictions": (intel.get("risk") or {}).get("restrictions"),
            "emergency_conditions": (intel.get("risk") or {}).get("lockdown"),
        },
        "governance_state": {
            "governance_status": decision.get("governance_status"),
            "authority_classification": decision.get("authority_classification"),
            "authorization_chain": decision.get("authorization_chain"),
            "authority_level_at_capture": authority_level,
            "authority_triggers_at_capture": triggers,
            "approval_records": _linked_approvals(execution_identifier),
            "escalation_records": _linked_escalations(execution_identifier),
            "health_at_capture": {
                "health_score": health.get("health_score"),
                "health_classification": health.get("health_classification"),
            },
            "certification_at_capture": {
                "certification_level": certification.get("certification_level"),
                "resilience_score": certification.get("resilience_score"),
            },
            "operational_state": operational.get("operational_state"),
        },
        "environment_state": {
            "environment": decision.get("environment"),
            "paper_or_live": decision.get("paper_or_live"),
            "governance_authorization": intel.get("governance") or {},
        },
        "frozen_decision": frozen_decision,
        "frozen_evidence": frozen_evidence,
        "evidence_links": decision.get("evidence_links") or [],
    }

    fingerprint = _fingerprint(body)
    snapshot = {
        "snapshot_id": _snapshot_id(),
        "captured_at": _utc_now_iso(),
        "framework": "THP-009",
        "immutable": True,
        "fingerprint": fingerprint,
        "source_decision_hash": _fingerprint(frozen_decision),
        "decision_timestamp": decision.get("timestamp"),
        **body,
    }
    return snapshot


def verify_snapshot_integrity(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Validate fingerprint without modifying snapshot."""
    stored = snapshot.get("fingerprint")
    body = {
        k: snapshot[k]
        for k in (
            "execution_identity",
            "market_intelligence",
            "portfolio_state",
            "risk_state",
            "governance_state",
            "environment_state",
            "frozen_decision",
            "frozen_evidence",
            "evidence_links",
        )
        if k in snapshot
    }
    computed = _fingerprint(body)
    decision_body = snapshot.get("frozen_decision")
    decision_hash = _fingerprint(decision_body) if decision_body else None
    return {
        "snapshot_id": snapshot.get("snapshot_id"),
        "fingerprint_valid": stored == computed,
        "stored_fingerprint": stored,
        "computed_fingerprint": computed,
        "source_decision_hash_valid": snapshot.get("source_decision_hash") == decision_hash,
        "immutable_flag": snapshot.get("immutable") is True,
    }


def run_time_machine(*, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Capture snapshots for all decisions not yet snapshotted.
    Existing snapshots are never altered.
    """
    decisions = load_decisions()
    doc = _load_snapshots_doc()
    existing_by_decision = {
        s.get("execution_identity", {}).get("decision_id")
        or s.get("frozen_decision", {}).get("decision_id"): s
        for s in doc.get("records") or []
    }

    new_snapshots: List[Dict[str, Any]] = []
    for decision in decisions:
        did = decision.get("decision_id")
        if not did:
            continue
        if did in existing_by_decision and not force_refresh:
            continue
        new_snapshots.append(capture_snapshot_from_decision(decision))

    if force_refresh:
        records = new_snapshots
    else:
        records = list(doc.get("records") or []) + new_snapshots

    doc["generated_at"] = _utc_now_iso()
    doc["framework"] = "THP-009"
    doc["summary"] = {
        "total": len(records),
        "decisions_source": len(decisions),
        "new_captured": len(new_snapshots),
    }
    doc["records"] = records
    _atomic_write_json(SNAPSHOTS_JSON, doc)

    index_rows = []
    for snap in records:
        ident = snap.get("execution_identity") or {}
        frozen = snap.get("frozen_decision") or {}
        index_rows.append(
            {
                "snapshot_id": snap.get("snapshot_id"),
                "decision_id": ident.get("decision_id") or frozen.get("decision_id"),
                "execution_identifier": ident.get("execution_identifier"),
                "timestamp": ident.get("timestamp") or snap.get("decision_timestamp"),
                "execution_path": ident.get("execution_path"),
                "fingerprint": snap.get("fingerprint"),
                "symbol(s)": ident.get("symbol(s)"),
                "paper_or_live": snap.get("environment_state", {}).get("paper_or_live"),
                "decision": frozen.get("execution_gate_decision"),
                "allowed": frozen.get("allowed"),
            }
        )
    _write_index(index_rows)
    return doc


def load_snapshots() -> Dict[str, Any]:
    return _load_snapshots_doc()


def find_snapshot_by_decision_id(decision_id: str) -> Optional[Dict[str, Any]]:
    for snap in load_snapshots().get("records") or []:
        ident = snap.get("execution_identity") or {}
        frozen = snap.get("frozen_decision") or {}
        if ident.get("decision_id") == decision_id or frozen.get("decision_id") == decision_id:
            return snap
    return None


if __name__ == "__main__":
    doc = run_time_machine()
    print(json.dumps(doc.get("summary"), indent=2))
