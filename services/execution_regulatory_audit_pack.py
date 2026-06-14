"""
execution_regulatory_audit_pack.py
THP-009 — Exportable regulatory audit packages for execution decisions.
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

from services.execution_decision_reconstructor import reconstruct_decision
from services.execution_gate_evidence_registry import AUDIT_CSV, load_decisions
from services.execution_time_machine import (
    find_snapshot_by_decision_id,
    run_time_machine,
    verify_snapshot_integrity,
)

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"
RESULTS_DIR = ROOT / "data" / "results"

AUDIT_PACKAGES_JSON = GOVERNANCE_DIR / "execution_audit_packages.json"
AUDIT_PACKAGES_CSV = GOVERNANCE_DIR / "execution_audit_packages.csv"

PACKAGE_FIELDS = (
    "package_id",
    "decision_id",
    "generated_at",
    "execution_timestamp",
    "execution_path",
    "decision_outcome",
    "capital_exposure",
    "approving_authority",
    "integrity_fingerprint",
    "completeness_score",
    "missing_artifact_count",
    "comparison_classification",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_id() -> str:
    return f"eg-aud-{uuid.uuid4().hex[:12]}"


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


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
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _load_packages_doc() -> Dict[str, Any]:
    if not AUDIT_PACKAGES_JSON.is_file():
        return {"records": [], "framework": "THP-009"}
    try:
        return json.loads(AUDIT_PACKAGES_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {"records": [], "framework": "THP-009"}


def _audit_log_excerpt(decision: Dict[str, Any]) -> List[Dict[str, str]]:
    if not AUDIT_CSV.is_file():
        return []
    path = str(decision.get("execution_path") or "")
    ident = str(decision.get("execution_identifier") or "")
    rows: List[Dict[str, str]] = []
    try:
        with AUDIT_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("execution_path") == path or (
                    ident and row.get("execution_identifier") == ident
                ):
                    rows.append(dict(row))
    except Exception:
        pass
    return rows[-10:]


def _completeness_score(
    decision: Dict[str, Any],
    snapshot: Optional[Dict[str, Any]],
    reconstruction: Dict[str, Any],
) -> float:
    checks = 0
    passed = 0
    for cond in (
        bool(decision.get("decision_id")),
        bool(decision.get("authorization_chain")),
        bool(decision.get("evidence_links")),
        bool(decision.get("execution_gate_decision")),
        snapshot is not None,
        (snapshot or {}).get("immutable") is True,
        reconstruction.get("integrity_valid") is True,
        len(reconstruction.get("missing_evidence") or []) == 0,
    ):
        checks += 1
        if cond:
            passed += 1
    return round(100.0 * passed / max(checks, 1), 2)


def _approving_authority(decision: Dict[str, Any], snapshot: Optional[Dict[str, Any]]) -> str:
    auth = str(decision.get("authority_classification") or "")
    if auth == "governance_approval":
        return "Governance Council"
    if auth == "operator_override":
        return decision.get("operator_identity") or "Operator Override"
    if auth == "automatic_approval":
        return "Execution Gate (Automatic)"
    if snapshot:
        approvals = (snapshot.get("governance_state") or {}).get("approval_records") or []
        for ap in approvals:
            if ap.get("approval_status") == "APPROVED":
                return str(ap.get("approving_authority") or "Council Approved")
    return auth or "Unknown"


def _build_timeline(decision: Dict[str, Any]) -> List[Dict[str, str]]:
    chain = decision.get("authorization_chain") or []
    timeline = []
    for item in chain:
        if isinstance(item, dict):
            timeline.append(
                {
                    "step": str(item.get("step") or ""),
                    "status": str(item.get("status") or ""),
                    "detail": str(item.get("detail") or ""),
                }
            )
    if not timeline:
        timeline = [
            {"step": "signal", "status": "unknown", "detail": "No chain recorded"},
            {
                "step": "execution_gate",
                "status": str(decision.get("execution_gate_decision")),
                "detail": decision.get("decision_reason") or "",
            },
            {
                "step": "execution_outcome",
                "status": "authorized" if decision.get("allowed") else "blocked",
                "detail": "",
            },
        ]
    return timeline


def generate_audit_package(decision_id: str) -> Dict[str, Any]:
    decision = next((d for d in load_decisions() if d.get("decision_id") == decision_id), None)
    if not decision:
        raise ValueError(f"Decision not found: {decision_id}")

    run_time_machine()
    snapshot = find_snapshot_by_decision_id(decision_id)
    reconstruction = reconstruct_decision(decision_id)

    completeness = _completeness_score(decision, snapshot, reconstruction)
    missing = list(reconstruction.get("missing_evidence") or [])
    warnings = [
        {"severity": "HIGH" if m == "no_snapshot" else "MEDIUM", "artifact": m} for m in missing
    ]

    executive = {
        "decision_outcome": decision.get("execution_gate_decision"),
        "allowed": decision.get("allowed"),
        "date_time": decision.get("timestamp"),
        "capital_involved": decision.get("capital_exposure") or "not_recorded",
        "symbols": decision.get("symbol(s)"),
        "execution_path": decision.get("execution_path"),
        "approving_authority": _approving_authority(decision, snapshot),
        "environment": decision.get("environment"),
        "paper_or_live": decision.get("paper_or_live"),
    }

    evidence_bundle = {
        "decision_record": decision,
        "snapshot_reference": {
            "snapshot_id": (snapshot or {}).get("snapshot_id"),
            "fingerprint": (snapshot or {}).get("fingerprint"),
        },
        "signals": (snapshot or {}).get("market_intelligence") or {},
        "risk_reports": (snapshot or {}).get("risk_state") or {},
        "allocation_evidence": (snapshot or {}).get("portfolio_state") or {},
        "governance_records": (snapshot or {}).get("governance_state") or {},
        "approval_records": ((snapshot or {}).get("governance_state") or {}).get("approval_records")
        or [],
        "audit_logs": _audit_log_excerpt(decision),
        "reconstruction_summary": reconstruction.get("narrative_summary"),
    }

    cert_body = {
        "executive_summary": executive,
        "decision_timeline": _build_timeline(decision),
        "evidence_bundle_keys": list(evidence_bundle.keys()),
        "snapshot_fingerprint": (snapshot or {}).get("fingerprint"),
    }
    integrity_fingerprint = _fingerprint(cert_body)

    package = {
        "package_id": _package_id(),
        "generated_at": _utc_now_iso(),
        "framework": "THP-009",
        "decision_id": decision_id,
        "executive_summary": executive,
        "decision_timeline": _build_timeline(decision),
        "timeline_flow": "Signal → Lifecycle → Allocation → Risk → Gate → Governance → Outcome",
        "evidence_bundle": evidence_bundle,
        "certification": {
            "integrity_fingerprint": integrity_fingerprint,
            "snapshot_integrity": verify_snapshot_integrity(snapshot) if snapshot else {},
            "completeness_score": completeness,
            "missing_artifact_warnings": warnings,
            "comparison_classification": reconstruction.get("comparison_classification"),
            "present_day_narrative": reconstruction.get("narrative_summary"),
        },
        "reconstruction_id": reconstruction.get("reconstruction_id"),
    }
    return package


def save_audit_package(package: Dict[str, Any]) -> Dict[str, Any]:
    doc = _load_packages_doc()
    records = list(doc.get("records") or [])
    records = [r for r in records if r.get("decision_id") != package.get("decision_id")]
    records.append(package)
    doc["generated_at"] = _utc_now_iso()
    doc["framework"] = "THP-009"
    doc["summary"] = {"total": len(records)}
    doc["records"] = records[-200:]
    _atomic_write_json(AUDIT_PACKAGES_JSON, doc)

    csv_rows = []
    for pkg in doc["records"]:
        exec_sum = pkg.get("executive_summary") or {}
        cert = pkg.get("certification") or {}
        csv_rows.append(
            {
                "package_id": pkg.get("package_id"),
                "decision_id": pkg.get("decision_id"),
                "generated_at": pkg.get("generated_at"),
                "execution_timestamp": exec_sum.get("date_time"),
                "execution_path": exec_sum.get("execution_path"),
                "decision_outcome": exec_sum.get("decision_outcome"),
                "capital_exposure": exec_sum.get("capital_involved"),
                "approving_authority": exec_sum.get("approving_authority"),
                "integrity_fingerprint": cert.get("integrity_fingerprint"),
                "completeness_score": cert.get("completeness_score"),
                "missing_artifact_count": len(cert.get("missing_artifact_warnings") or []),
                "comparison_classification": cert.get("comparison_classification"),
            }
        )
    _write_csv(AUDIT_PACKAGES_CSV, csv_rows, PACKAGE_FIELDS)
    return doc


def run_audit_package_generation(*, limit: Optional[int] = None) -> Dict[str, Any]:
    decisions = load_decisions()
    if limit:
        decisions = decisions[-limit:]
    for decision in decisions:
        did = decision.get("decision_id")
        if did:
            pkg = generate_audit_package(str(did))
            save_audit_package(pkg)
    return load_audit_packages()


def load_audit_packages() -> Dict[str, Any]:
    if not AUDIT_PACKAGES_JSON.is_file():
        return {}
    try:
        return json.loads(AUDIT_PACKAGES_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_audit_package(decision_id: str) -> Optional[Dict[str, Any]]:
    for pkg in load_audit_packages().get("records") or []:
        if pkg.get("decision_id") == decision_id:
            return pkg
    pkg = generate_audit_package(decision_id)
    save_audit_package(pkg)
    return pkg


if __name__ == "__main__":
    doc = run_audit_package_generation()
    print(json.dumps(doc.get("summary"), indent=2))
