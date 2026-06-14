"""
execution_gate_degradation_manager.py
THP-007 — Autonomous degradation classification for execution gate supervision.

Read-only governance state machine. Writes operational status and escalation events
without modifying gate logic, trading models, or historical records.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.execution_gate_anomaly_detector import run_anomaly_detection
from services.execution_gate_health_monitor import (
    append_health_history,
    build_monitoring_snapshot,
    run_health_monitor,
)

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"

DEGRADATION_EVENTS_JSON = GOVERNANCE_DIR / "execution_gate_degradation_events.json"
OPERATIONAL_STATUS_JSON = GOVERNANCE_DIR / "execution_gate_operational_status.json"

CAPITAL_DEPLOYMENT_PATHS = {
    "position_entry",
    "position_add",
    "capital_deployment",
    "broker_submission",
    "paper_execution",
    "live_execution",
    "forced_rotation",
}

EMERGENCY_PATHS = {"risk_emergency_action"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_id() -> str:
    return f"eg-deg-{uuid.uuid4().hex[:12]}"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_degradation_history() -> List[Dict[str, Any]]:
    if not DEGRADATION_EVENTS_JSON.is_file():
        return []
    try:
        doc = json.loads(DEGRADATION_EVENTS_JSON.read_text(encoding="utf-8"))
        return list(doc.get("records") or [])
    except Exception:
        return []


def classify_degradation(
    health_report: Dict[str, Any],
    anomalies: List[Dict[str, Any]],
    snapshot: Dict[str, Any],
) -> str:
    """Return MONITOR | WARNING | RESTRICT | LOCKDOWN."""
    severities = {a.get("severity") for a in anomalies}
    health_score = float(health_report.get("health_score") or 0)
    drift_count = int(snapshot.get("drift_count") or 0)

    lockdown_conditions = (
        not (snapshot.get("gate_probe") or {}).get("available")
        or snapshot.get("gate_disabled")
        or not snapshot.get("paper_live_boundary_intact", True)
        or any(
            a.get("severity") == "CRITICAL"
            and a.get("affected_component") in ("environment_isolation", "execution_gate")
            for a in anomalies
        )
    )
    if lockdown_conditions:
        return "LOCKDOWN"

    restrict_conditions = (
        drift_count > 0
        or not _evidence_present(snapshot)
        or health_score < 70
        or "HIGH" in severities
        or any(
            a.get("severity") == "CRITICAL" and a.get("category") == "execution" for a in anomalies
        )
    )
    if restrict_conditions:
        return "RESTRICT"

    warning_conditions = (
        health_score < 85
        or int(snapshot.get("decisions_missing_evidence") or 0) > 0
        or (snapshot.get("evidence_age_hours") or 0) > 72
        or "MEDIUM" in severities
    )
    if warning_conditions:
        return "WARNING"

    return "MONITOR"


def _evidence_present(snapshot: Dict[str, Any]) -> bool:
    from services.execution_gate_evidence_registry import DECISIONS_JSON, EVIDENCE_JSON

    return DECISIONS_JSON.is_file() and EVIDENCE_JSON.is_file()


def build_restrictions(operational_state: str) -> Dict[str, Any]:
    if operational_state == "LOCKDOWN":
        return {
            "capital_deployment_blocked": True,
            "emergency_risk_actions_enabled": True,
            "blocked_paths": sorted(CAPITAL_DEPLOYMENT_PATHS),
            "allowed_paths": sorted(EMERGENCY_PATHS),
            "governance_escalation_required": True,
        }
    if operational_state == "RESTRICT":
        return {
            "capital_deployment_blocked": False,
            "new_capital_deployment_restricted": True,
            "emergency_risk_actions_enabled": True,
            "restricted_paths": sorted(CAPITAL_DEPLOYMENT_PATHS - {"risk_emergency_action"}),
            "governance_escalation_required": True,
        }
    if operational_state == "WARNING":
        return {
            "capital_deployment_blocked": False,
            "emergency_risk_actions_enabled": True,
            "enhanced_monitoring": True,
            "governance_escalation_required": False,
        }
    return {
        "capital_deployment_blocked": False,
        "emergency_risk_actions_enabled": True,
        "governance_escalation_required": False,
    }


def _build_degradation_event(
    *,
    previous_state: str,
    new_state: str,
    health_score: float,
    trigger_anomalies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "event_id": _event_id(),
        "timestamp": _utc_now_iso(),
        "previous_state": previous_state,
        "new_state": new_state,
        "health_score": health_score,
        "trigger_count": len(trigger_anomalies),
        "triggers": [
            {
                "anomaly_id": a.get("anomaly_id"),
                "severity": a.get("severity"),
                "condition": a.get("detected_condition"),
            }
            for a in trigger_anomalies[:5]
        ],
        "recovery_action": _recovery_action(new_state),
    }


def _recovery_action(state: str) -> str:
    return {
        "LOCKDOWN": "Resolve critical anomalies; re-run certification; clear gate bypass.",
        "RESTRICT": "Review drift report; refresh governance evidence; re-run replay.",
        "WARNING": "Monitor failure trends; refresh stale evidence; schedule certification.",
        "MONITOR": "Continue normal supervision cadence.",
    }.get(state, "Review operational status.")


def run_degradation_manager(
    *,
    health_report: Optional[Dict[str, Any]] = None,
    anomaly_report: Optional[Dict[str, Any]] = None,
    snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    snapshot = snapshot or build_monitoring_snapshot()
    health_report = health_report or compute_or_load_health(snapshot)
    anomaly_report = anomaly_report or run_anomaly_detection(snapshot)
    anomalies = list(anomaly_report.get("records") or [])

    previous_status = load_operational_status()
    previous_state = str(previous_status.get("operational_state") or "MONITOR")

    operational_state = classify_degradation(health_report, anomalies, snapshot)
    restrictions = build_restrictions(operational_state)

    history = _load_degradation_history()
    if operational_state != previous_state:
        critical_triggers = [
            a
            for a in anomalies
            if a.get("severity") in ("CRITICAL", "HIGH")
            or operational_state in ("LOCKDOWN", "RESTRICT")
        ]
        history.append(
            _build_degradation_event(
                previous_state=previous_state,
                new_state=operational_state,
                health_score=float(health_report.get("health_score") or 0),
                trigger_anomalies=critical_triggers or anomalies[:3],
            )
        )

    events_doc = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-007",
        "active_state": operational_state,
        "summary": {
            "total_events": len(history),
            "active_restrictions": restrictions.get("capital_deployment_blocked")
            or restrictions.get("new_capital_deployment_restricted"),
        },
        "records": history[-100:],
    }
    _atomic_write_json(DEGRADATION_EVENTS_JSON, events_doc)

    cert = snapshot.get("certification") or {}
    status = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-007",
        "operational_state": operational_state,
        "health_score": health_report.get("health_score"),
        "health_classification": health_report.get("health_classification"),
        "certification_level": cert.get("certification_level"),
        "resilience_score": cert.get("resilience_score"),
        "gate_available": (snapshot.get("gate_probe") or {}).get("available", False),
        "last_successful_validation": cert.get("generated_at"),
        "anomaly_count": len(anomalies),
        "drift_count": snapshot.get("drift_count", 0),
        "paper_live_boundary_intact": snapshot.get("paper_live_boundary_intact", True),
        "restrictions": restrictions,
        "active_anomalies": [
            {
                "anomaly_id": a.get("anomaly_id"),
                "severity": a.get("severity"),
                "category": a.get("category"),
                "condition": a.get("detected_condition"),
            }
            for a in anomalies
            if a.get("severity") in ("CRITICAL", "HIGH")
        ],
    }
    _atomic_write_json(OPERATIONAL_STATUS_JSON, status)

    append_health_history(
        health_score=float(health_report.get("health_score") or 0),
        health_classification=str(health_report.get("health_classification") or ""),
        operational_state=operational_state,
        anomaly_count=len(anomalies),
        gate_available=bool((snapshot.get("gate_probe") or {}).get("available")),
        drift_count=int(snapshot.get("drift_count") or 0),
        certification_level=str(cert.get("certification_level") or ""),
    )

    return status


def compute_or_load_health(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return run_health_monitor(snapshot)


def load_operational_status() -> Dict[str, Any]:
    if not OPERATIONAL_STATUS_JSON.is_file():
        return {}
    try:
        return json.loads(OPERATIONAL_STATUS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_degradation_events() -> Dict[str, Any]:
    if not DEGRADATION_EVENTS_JSON.is_file():
        return {}
    try:
        return json.loads(DEGRADATION_EVENTS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_continuous_supervision() -> Dict[str, Any]:
    """Full THP-007 supervision cycle: health → anomalies → degradation."""
    snapshot = build_monitoring_snapshot()
    health = run_health_monitor(snapshot)
    anomalies = run_anomaly_detection(snapshot)
    status = run_degradation_manager(
        health_report=health,
        anomaly_report=anomalies,
        snapshot=snapshot,
    )
    return {
        "health": health,
        "anomalies": anomalies.get("summary"),
        "operational_status": status,
    }


if __name__ == "__main__":
    result = run_continuous_supervision()
    print(
        json.dumps(
            {
                "health_score": result["health"].get("health_score"),
                "state": result["operational_status"].get("operational_state"),
                "anomalies": result["anomalies"],
            },
            indent=2,
        )
    )
