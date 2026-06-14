"""
execution_escalation_engine.py
THP-008 — Automatic escalation for execution governance events.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.execution_gate_degradation_manager import load_operational_status
from services.execution_gate_health_monitor import load_health_report
from services.execution_governance_council import (
    authority_level_name,
    determine_authority_level,
)
from services.triton_execution_gate import EXECUTION_PATHS

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"

ESCALATIONS_JSON = GOVERNANCE_DIR / "execution_escalations.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _escalation_id() -> str:
    return f"eg-esc-{uuid.uuid4().hex[:12]}"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_escalations() -> List[Dict[str, Any]]:
    if not ESCALATIONS_JSON.is_file():
        return []
    try:
        doc = json.loads(ESCALATIONS_JSON.read_text(encoding="utf-8"))
        return list(doc.get("records") or [])
    except Exception:
        return []


def _severity_for(level: int, trigger: str, op_state: str) -> str:
    if level >= 3 or op_state == "LOCKDOWN":
        return "CRITICAL"
    if level >= 2 or trigger in (
        "live_capital_deployment",
        "certification_failure",
        "operational_lockdown",
    ):
        return "HIGH"
    if level >= 1:
        return "MEDIUM"
    return "LOW"


def _recommended_action(level: int, trigger: str) -> str:
    if level >= 3:
        return "Preserve emergency authority; restrict non-emergency capital deployment."
    if level >= 2:
        return "Route to Governance Council for institutional approval before execution."
    if level >= 1:
        return "Assign operator review and document rationale before proceeding."
    return "Continue automated supervision; no escalation required."


def detect_escalations(
    *,
    path: str,
    context: Optional[Dict[str, Any]] = None,
    evaluation: Optional[Dict[str, Any]] = None,
    capital_exposure: Optional[Any] = None,
    action_requested: Optional[str] = None,
    execution_identifier: Optional[str] = None,
) -> List[Dict[str, Any]]:
    level, triggers = determine_authority_level(
        path=path,
        context=context,
        evaluation=evaluation,
        capital_exposure=capital_exposure,
        action_requested=action_requested,
    )
    operational = load_operational_status()
    health = load_health_report()
    op_state = str(operational.get("operational_state") or "MONITOR")
    health_class = str(health.get("health_classification") or "GREEN")

    escalations: List[Dict[str, Any]] = []
    for trigger in triggers:
        escalations.append(
            {
                "escalation_id": _escalation_id(),
                "timestamp": _utc_now_iso(),
                "severity": _severity_for(level, trigger, op_state),
                "triggering_event": trigger,
                "execution_path": path,
                "execution_identifier": execution_identifier or "",
                "required_authority": authority_level_name(level),
                "authority_level": level,
                "recommended_action": _recommended_action(level, trigger),
                "resolution_status": "OPEN",
                "operational_state": op_state,
                "health_classification": health_class,
            }
        )

    if health_class == "RED":
        escalations.append(
            {
                "escalation_id": _escalation_id(),
                "timestamp": _utc_now_iso(),
                "severity": "CRITICAL",
                "triggering_event": "execution_gate_health_red",
                "execution_path": path,
                "execution_identifier": execution_identifier or "",
                "required_authority": "GOVERNANCE_COUNCIL_APPROVAL",
                "authority_level": 2,
                "recommended_action": "Halt non-emergency execution until health restored.",
                "resolution_status": "OPEN",
                "operational_state": op_state,
                "health_classification": health_class,
            }
        )

    if op_state in ("RESTRICT", "LOCKDOWN"):
        escalations.append(
            {
                "escalation_id": _escalation_id(),
                "timestamp": _utc_now_iso(),
                "severity": "HIGH" if op_state == "RESTRICT" else "CRITICAL",
                "triggering_event": f"operational_{op_state.lower()}",
                "execution_path": path,
                "execution_identifier": execution_identifier or "",
                "required_authority": authority_level_name(2 if op_state == "RESTRICT" else 3),
                "authority_level": 2 if op_state == "RESTRICT" else 3,
                "recommended_action": _recommended_action(
                    2 if op_state == "RESTRICT" else 3, op_state
                ),
                "resolution_status": "OPEN",
                "operational_state": op_state,
                "health_classification": health_class,
            }
        )

    if path == EXECUTION_PATHS["LIVE_EXECUTION"] and not (context or {}).get("governanceApproved"):
        if not any(e["triggering_event"] == "live_capital_deployment" for e in escalations):
            escalations.append(
                {
                    "escalation_id": _escalation_id(),
                    "timestamp": _utc_now_iso(),
                    "severity": "HIGH",
                    "triggering_event": "live_execution_without_governance",
                    "execution_path": path,
                    "execution_identifier": execution_identifier or "",
                    "required_authority": "GOVERNANCE_COUNCIL_APPROVAL",
                    "authority_level": 2,
                    "recommended_action": "Obtain Governance Council approval for live deployment.",
                    "resolution_status": "OPEN",
                    "operational_state": op_state,
                    "health_classification": health_class,
                }
            )

    return escalations


def record_escalations(
    new_records: List[Dict[str, Any]], *, dedupe_key: Optional[str] = None
) -> Dict[str, Any]:
    existing = _load_escalations()
    if dedupe_key:
        existing = [e for e in existing if e.get("execution_identifier") != dedupe_key]
    combined = existing + new_records
    open_count = sum(1 for e in combined if e.get("resolution_status") == "OPEN")
    by_severity: Dict[str, int] = {}
    for rec in combined:
        sev = str(rec.get("severity") or "LOW")
        by_severity[sev] = by_severity.get(sev, 0) + 1

    doc = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-008",
        "summary": {
            "total": len(combined),
            "open": open_count,
            "resolved": len(combined) - open_count,
            "by_severity": by_severity,
        },
        "records": combined[-500:],
    }
    _atomic_write_json(ESCALATIONS_JSON, doc)
    return doc


def resolve_escalation(escalation_id: str, resolution: str = "RESOLVED") -> bool:
    records = _load_escalations()
    updated = False
    for rec in records:
        if rec.get("escalation_id") == escalation_id:
            rec["resolution_status"] = resolution
            rec["resolved_at"] = _utc_now_iso()
            updated = True
    if updated:
        open_count = sum(1 for e in records if e.get("resolution_status") == "OPEN")
        by_severity: Dict[str, int] = {}
        for rec in records:
            sev = str(rec.get("severity") or "LOW")
            by_severity[sev] = by_severity.get(sev, 0) + 1
        doc = {
            "generated_at": _utc_now_iso(),
            "framework": "THP-008",
            "summary": {
                "total": len(records),
                "open": open_count,
                "resolved": len(records) - open_count,
                "by_severity": by_severity,
            },
            "records": records[-500:],
        }
        _atomic_write_json(ESCALATIONS_JSON, doc)
    return updated


def run_escalation_scan() -> Dict[str, Any]:
    """Scan operational conditions and record system-level escalations."""
    operational = load_operational_status()
    health = load_health_report()
    records: List[Dict[str, Any]] = []

    if str(health.get("health_classification")) == "RED":
        records.extend(
            detect_escalations(
                path=EXECUTION_PATHS["CAPITAL_DEPLOYMENT"],
                evaluation={"decision": "BLOCK"},
            )
        )

    op_state = str(operational.get("operational_state") or "MONITOR")
    if op_state in ("RESTRICT", "LOCKDOWN"):
        records.extend(
            detect_escalations(
                path=EXECUTION_PATHS["BROKER_SUBMISSION"],
                context={"operationalContext": f"scan:{op_state}"},
            )
        )

    if not records:
        doc = load_escalations()
        if not doc:
            doc = {
                "generated_at": _utc_now_iso(),
                "framework": "THP-008",
                "summary": {"total": 0, "open": 0, "resolved": 0, "by_severity": {}},
                "records": [],
            }
            _atomic_write_json(ESCALATIONS_JSON, doc)
        return doc

    return record_escalations(records)


def load_escalations() -> Dict[str, Any]:
    if not ESCALATIONS_JSON.is_file():
        return {}
    try:
        return json.loads(ESCALATIONS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    doc = run_escalation_scan()
    print(json.dumps(doc.get("summary"), indent=2))
