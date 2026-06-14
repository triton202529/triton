"""
execution_gate_evidence_registry.py
THP-004 — Institutional governance evidence and decision registry for execution gate.

Read-only enrichment from existing governance artifacts. Never blocks trading execution.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"
RESULTS_DIR = ROOT / "data" / "results"

DECISIONS_JSON = GOVERNANCE_DIR / "execution_gate_decisions.json"
DECISIONS_CSV = GOVERNANCE_DIR / "execution_gate_decisions.csv"
EVIDENCE_JSON = GOVERNANCE_DIR / "execution_gate_evidence.json"
EVIDENCE_CSV = GOVERNANCE_DIR / "execution_gate_evidence.csv"
AUTHORITY_MATRIX_JSON = GOVERNANCE_DIR / "execution_gate_authority_matrix.json"
AUDIT_CSV = RESULTS_DIR / "execution_gate_audit.csv"

DECISION_FIELDS = (
    "decision_id",
    "timestamp",
    "execution_identifier",
    "execution_path",
    "symbol(s)",
    "action_requested",
    "quantity",
    "capital_exposure",
    "environment",
    "paper_or_live",
    "requesting_service",
    "operator_identity",
    "execution_gate_decision",
    "decision_reason",
    "risk_state",
    "lifecycle_state",
    "allocation_state",
    "governance_status",
    "authorization_chain",
    "evidence_links",
    "authority_classification",
    "allowed",
)

EVIDENCE_FIELDS = (
    "evidence_id",
    "decision_id",
    "category",
    "severity",
    "title",
    "description",
    "created_at",
    "source",
    "status",
    "linked_audit_path",
    "execution_path",
    "decision",
    "allowed",
    "framework",
    "tags",
)

AUTHORITY_TYPES = (
    "automatic_approval",
    "governance_approval",
    "operator_override",
    "risk_emergency_action",
    "system_rejection",
    "review_required",
    "disabled",
    "sandbox_only",
)

CHAIN_STEPS = (
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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {"records": [], "generated_at": _utc_now_iso()}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("records", [])
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"records": [], "generated_at": _utc_now_iso()}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _load_governance_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    risk_path = RESULTS_DIR / "adaptive_risk_state.json"
    if risk_path.is_file():
        try:
            snap["risk"] = json.loads(risk_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            snap["risk"] = {}
    gov_path = RESULTS_DIR / "governance_authorization.json"
    if gov_path.is_file():
        try:
            snap["governance"] = json.loads(gov_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            snap["governance"] = {}
    signal_path = RESULTS_DIR / "signal_state.json"
    if signal_path.is_file():
        try:
            snap["signal"] = json.loads(signal_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            snap["signal"] = {}
    alloc_path = RESULTS_DIR / "portfolio_allocation_summary.json"
    if alloc_path.is_file():
        try:
            snap["allocation"] = json.loads(alloc_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            snap["allocation"] = {}
    return snap


def _summarize_risk_state(snap: Dict[str, Any]) -> str:
    risk = snap.get("risk") or {}
    if not risk:
        return "unknown"
    regime = risk.get("regime") or risk.get("risk_regime") or "unknown"
    risk_on = risk.get("risk_on")
    if risk_on is False:
        return f"{regime}:risk_off"
    return str(regime)


def _summarize_lifecycle_state(snap: Dict[str, Any]) -> str:
    signal = snap.get("signal") or {}
    if signal.get("lifecycle_state"):
        return str(signal["lifecycle_state"])
    if signal.get("asof"):
        return f"asof={signal['asof']}"
    return "unknown"


def _summarize_allocation_state(snap: Dict[str, Any]) -> str:
    alloc = snap.get("allocation") or {}
    if alloc.get("status"):
        return str(alloc["status"])
    if alloc.get("recommendation_count") is not None:
        return f"recommendations={alloc['recommendation_count']}"
    return "unknown"


def _summarize_governance_status(snap: Dict[str, Any]) -> str:
    gov = snap.get("governance") or {}
    if gov.get("governance_authorized") is True:
        return "authorized"
    if gov.get("governance_authorized") is False:
        return "not_authorized"
    if gov.get("escalation_state"):
        return str(gov["escalation_state"])
    return "unknown"


def classify_authority(
    evaluation: Dict[str, Any],
    *,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    decision = str(evaluation.get("decision") or "").upper()
    ctx = context or {}
    if decision == "DISABLED":
        return "disabled"
    if decision == "SANDBOX_ONLY":
        return "sandbox_only"
    if decision == "REVIEW_REQUIRED":
        return "review_required"
    if ctx.get("riskEmergencyTriggered") or ctx.get("emergencyAction"):
        return "risk_emergency_action"
    if ctx.get("operatorAuthorized") or ctx.get("overrideReason"):
        return "operator_override"
    if decision == "BLOCK" or not evaluation.get("allowed"):
        return "system_rejection"
    if ctx.get("governanceApproved") or evaluation.get("requires_governance"):
        return "governance_approval"
    return "automatic_approval"


def build_authorization_chain(
    evaluation: Dict[str, Any],
    *,
    context: Optional[Dict[str, Any]] = None,
    allowed: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    ctx = context or {}
    is_allowed = allowed if allowed is not None else bool(evaluation.get("allowed"))
    decision = str(evaluation.get("decision") or "")

    def _step(name: str, status: str, detail: str = "") -> Dict[str, Any]:
        return {"step": name, "status": status, "detail": detail}

    chain = [
        _step("signal", "observed", "signal_generation upstream"),
        _step(
            "lifecycle",
            "approved" if ctx.get("lifecycleApproved", True) else "blocked",
            "lifecycle authorization",
        ),
        _step(
            "allocation",
            "approved" if ctx.get("allocationApproved", True) else "blocked",
            "portfolio allocation",
        ),
        _step(
            "risk_protection",
            "satisfied" if ctx.get("riskConstraintsSatisfied", True) else "violated",
            "risk validation",
        ),
        _step(
            "execution_gate",
            decision.lower() or ("allowed" if is_allowed else "blocked"),
            str(evaluation.get("reason") or ""),
        ),
        _step(
            "broker_authorization",
            (
                "ready"
                if ctx.get("brokerConnected") and ctx.get("executionGuardApproved", True)
                else "pending"
            ),
            "broker connectivity and guard",
        ),
        _step(
            "execution_outcome",
            "authorized" if is_allowed else "blocked",
            "final execution disposition",
        ),
    ]
    return chain


def _make_decision_id(
    evaluation: Dict[str, Any],
    *,
    source: str,
    execution_identifier: Optional[str],
) -> str:
    blob = json.dumps(
        {
            "path": evaluation.get("path"),
            "timestamp": evaluation.get("timestamp"),
            "decision": evaluation.get("decision"),
            "reason": evaluation.get("reason"),
            "source": source,
            "execution_identifier": execution_identifier or "",
            "allowed": evaluation.get("allowed"),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return f"egd-{digest}"


def _build_evidence_links(
    *,
    decision_id: str,
    execution_identifier: Optional[str],
) -> List[str]:
    links = [f"decision:{decision_id}"]
    if AUDIT_CSV.is_file():
        links.append(f"audit:{AUDIT_CSV.as_posix()}")
    if execution_identifier:
        links.append(f"execution_id:{execution_identifier}")
    return links


def _infer_action_requested(evaluation: Dict[str, Any], side: str = "") -> str:
    path = str(evaluation.get("path") or "")
    mapping = {
        "position_entry": "POSITION_ENTRY",
        "position_add": "POSITION_ADD",
        "position_trim": "POSITION_TRIM",
        "position_exit": "POSITION_EXIT",
        "forced_rotation": "FORCED_ROTATION",
        "capital_deployment": "CAPITAL_DEPLOYMENT",
        "broker_submission": "BROKER_SUBMIT",
        "paper_execution": "PAPER_EXECUTE",
        "live_execution": "LIVE_EXECUTE",
        "risk_emergency_action": "RISK_EMERGENCY",
    }
    if path in mapping:
        return mapping[path]
    if side:
        return f"{side.upper()}_ORDER"
    return path.upper()


def record_execution_gate_decision(
    *,
    evaluation: Dict[str, Any],
    source: str,
    broker_mode: str = "",
    symbols: Optional[Sequence[str]] = None,
    execution_identifier: Optional[str] = None,
    action_requested: Optional[str] = None,
    quantity: Optional[Any] = None,
    capital_exposure: Optional[Any] = None,
    operator_identity: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    side: str = "",
) -> Optional[str]:
    """
    Persist institutional governance record for an execution gate evaluation.
    Returns decision_id if recorded, None if deduplicated or skipped.
    """
    if not evaluation:
        return None

    decision_id = _make_decision_id(
        evaluation, source=source, execution_identifier=execution_identifier
    )
    store = _read_json(DECISIONS_JSON)
    existing_ids = {str(r.get("decision_id")) for r in store.get("records", [])}
    if decision_id in existing_ids:
        return None

    snap = _load_governance_snapshot()
    ctx = context or {}
    sym_list = list(symbols or [])
    paper_or_live = broker_mode or (
        "paper" if ctx.get("paperMode") else "live" if ctx.get("liveMode") else "unknown"
    )
    authority = classify_authority(evaluation, context=ctx)
    chain = build_authorization_chain(evaluation, context=ctx)
    evidence_links = _build_evidence_links(
        decision_id=decision_id,
        execution_identifier=execution_identifier,
    )

    record = {
        "decision_id": decision_id,
        "timestamp": evaluation.get("timestamp") or _utc_now_iso(),
        "execution_identifier": execution_identifier or "",
        "execution_path": evaluation.get("path") or "",
        "symbol(s)": ",".join(sym_list),
        "action_requested": action_requested or _infer_action_requested(evaluation, side=side),
        "quantity": quantity if quantity is not None else "",
        "capital_exposure": capital_exposure if capital_exposure is not None else "",
        "environment": evaluation.get("environment") or os.environ.get("TRITON_ENV", "development"),
        "paper_or_live": paper_or_live,
        "requesting_service": source,
        "operator_identity": operator_identity
        or evaluation.get("user_id")
        or ctx.get("operatorId")
        or "",
        "execution_gate_decision": evaluation.get("decision") or "",
        "decision_reason": evaluation.get("reason") or "",
        "risk_state": _summarize_risk_state(snap),
        "lifecycle_state": _summarize_lifecycle_state(snap),
        "allocation_state": _summarize_allocation_state(snap),
        "governance_status": _summarize_governance_status(snap),
        "authorization_chain": chain,
        "evidence_links": evidence_links,
        "authority_classification": authority,
        "allowed": bool(evaluation.get("allowed")),
    }

    store.setdefault("records", []).append(record)
    store["generated_at"] = _utc_now_iso()
    store["record_count"] = len(store["records"])
    _atomic_write_json(DECISIONS_JSON, store)

    evidence_id = f"ege-{decision_id[4:]}"
    severity = "CRITICAL" if not record["allowed"] else "INFO"
    if authority == "risk_emergency_action":
        severity = "HIGH"
    if record["execution_path"] == "live_execution":
        severity = "HIGH"

    evidence = {
        "evidence_id": evidence_id,
        "decision_id": decision_id,
        "category": "EXECUTION_GATE",
        "severity": severity,
        "title": f"Execution gate {record['execution_gate_decision']} — {record['execution_path']}",
        "description": record["decision_reason"],
        "created_at": record["timestamp"],
        "source": source,
        "status": "VERIFIED",
        "linked_audit_path": AUDIT_CSV.as_posix(),
        "execution_path": record["execution_path"],
        "decision": record["execution_gate_decision"],
        "allowed": record["allowed"],
        "framework": "THP-004",
        "tags": f"{authority},{paper_or_live},{record['execution_path']}",
    }

    ev_store = _read_json(EVIDENCE_JSON)
    ev_store.setdefault("records", []).append(evidence)
    ev_store["generated_at"] = _utc_now_iso()
    ev_store["record_count"] = len(ev_store["records"])
    _atomic_write_json(EVIDENCE_JSON, ev_store)

    sync_registry_artifacts()
    return decision_id


def sync_registry_artifacts() -> None:
    """Rewrite CSV mirrors and authority matrix from JSON stores."""
    decisions = _read_json(DECISIONS_JSON).get("records", [])
    evidence = _read_json(EVIDENCE_JSON).get("records", [])

    csv_decisions = []
    for row in decisions:
        flat = dict(row)
        flat["authorization_chain"] = json.dumps(row.get("authorization_chain", []))
        flat["evidence_links"] = json.dumps(row.get("evidence_links", []))
        csv_decisions.append(flat)
    _write_csv(DECISIONS_CSV, csv_decisions, DECISION_FIELDS)
    _write_csv(EVIDENCE_CSV, evidence, EVIDENCE_FIELDS)
    _atomic_write_json(AUTHORITY_MATRIX_JSON, build_authority_matrix(decisions))


def build_authority_matrix(decisions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if decisions is None:
        decisions = _read_json(DECISIONS_JSON).get("records", [])

    by_decision: Dict[str, int] = {}
    by_authority: Dict[str, int] = {}
    by_path: Dict[str, int] = {}
    by_service: Dict[str, int] = {}
    block_reasons: Dict[str, int] = {}

    for row in decisions:
        d = str(row.get("execution_gate_decision") or "UNKNOWN")
        by_decision[d] = by_decision.get(d, 0) + 1
        a = str(row.get("authority_classification") or "unknown")
        by_authority[a] = by_authority.get(a, 0) + 1
        p = str(row.get("execution_path") or "unknown")
        by_path[p] = by_path.get(p, 0) + 1
        s = str(row.get("requesting_service") or "unknown")
        by_service[s] = by_service.get(s, 0) + 1
        if not row.get("allowed"):
            reason = str(row.get("decision_reason") or "unspecified")[:120]
            block_reasons[reason] = block_reasons.get(reason, 0) + 1

    return {
        "generated_at": _utc_now_iso(),
        "framework": "THP-004",
        "decision_types": ["ALLOW", "BLOCK", "REVIEW_REQUIRED", "SANDBOX_ONLY", "DISABLED"],
        "authority_types": list(AUTHORITY_TYPES),
        "authorization_chain_steps": list(CHAIN_STEPS),
        "statistics": {
            "total_decisions": len(decisions),
            "approved": sum(1 for r in decisions if r.get("allowed")),
            "blocked": sum(1 for r in decisions if not r.get("allowed")),
            "by_decision": by_decision,
            "by_authority": by_authority,
            "by_path": by_path,
            "by_service": by_service,
            "top_block_reasons": dict(
                sorted(block_reasons.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        },
    }


def load_decisions() -> List[Dict[str, Any]]:
    return list(_read_json(DECISIONS_JSON).get("records", []))


def load_evidence() -> List[Dict[str, Any]]:
    return list(_read_json(EVIDENCE_JSON).get("records", []))


def initialize_empty_artifacts() -> None:
    """Create empty governance artifacts if missing."""
    GOVERNANCE_DIR.mkdir(parents=True, exist_ok=True)
    if not DECISIONS_JSON.exists():
        _atomic_write_json(DECISIONS_JSON, {"records": [], "generated_at": _utc_now_iso()})
    if not EVIDENCE_JSON.exists():
        _atomic_write_json(EVIDENCE_JSON, {"records": [], "generated_at": _utc_now_iso()})
    sync_registry_artifacts()
