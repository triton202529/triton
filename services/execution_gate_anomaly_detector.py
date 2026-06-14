"""
execution_gate_anomaly_detector.py
THP-007 — Anomaly detection for execution gate continuous supervision (read-only).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from services.execution_gate_evidence_registry import AUDIT_CSV, DECISIONS_JSON, EVIDENCE_JSON
from services.execution_gate_health_monitor import build_monitoring_snapshot
from services.triton_execution_gate import EXECUTION_PATHS

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"

ANOMALIES_JSON = GOVERNANCE_DIR / "execution_gate_anomalies.json"

KNOWN_PATHS = set(EXECUTION_PATHS.values())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _anomaly_id() -> str:
    return f"eg-anom-{uuid.uuid4().hex[:12]}"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _make_anomaly(
    *,
    category: str,
    severity: str,
    affected_component: str,
    detected_condition: str,
    recommended_action: str,
) -> Dict[str, Any]:
    return {
        "anomaly_id": _anomaly_id(),
        "timestamp": _utc_now_iso(),
        "category": category,
        "severity": severity,
        "affected_component": affected_component,
        "detected_condition": detected_condition,
        "recommended_action": recommended_action,
    }


def detect_execution_anomalies(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    anomalies: List[Dict[str, Any]] = []
    seen_live: Set[Tuple[Any, Any, Any]] = set()
    recent = snapshot.get("recent_audit") or []
    total = len(recent)

    if total >= 5:
        block_rate = snapshot.get("recent_block_count", 0) / total
        if block_rate > 0.75:
            anomalies.append(
                _make_anomaly(
                    category="execution",
                    severity="HIGH" if block_rate > 0.9 else "MEDIUM",
                    affected_component="authorization_decisions",
                    detected_condition=f"Sudden BLOCK increase: {block_rate:.0%} of last 24h decisions",
                    recommended_action="Review recent pipeline failures and gate context construction.",
                )
            )

    for row in snapshot.get("recent_live_allows") or []:
        key = (row.get("execution_path"), row.get("timestamp"), row.get("execution_identifier"))
        if key in seen_live:
            continue
        seen_live.add(key)
        anomalies.append(
            _make_anomaly(
                category="execution",
                severity="CRITICAL",
                affected_component=str(row.get("execution_path", "live_execution")),
                detected_condition=(
                    f"Unexpected ALLOW on capital path: {row.get('execution_path')} "
                    f"({row.get('timestamp')})"
                ),
                recommended_action="Verify paper/live authorization and governance approval chain.",
            )
        )

    for path in snapshot.get("unknown_paths") or []:
        anomalies.append(
            _make_anomaly(
                category="execution",
                severity="HIGH",
                affected_component="execution_paths",
                detected_condition=f"New or unrecognized authorization path: {path}",
                recommended_action="Confirm path is registered in EXECUTION_PATHS registry.",
            )
        )

    if int(snapshot.get("duplicate_decision_ids") or 0) > 0:
        anomalies.append(
            _make_anomaly(
                category="execution",
                severity="MEDIUM",
                affected_component="decision_registry",
                detected_condition=(
                    f"{snapshot['duplicate_decision_ids']} duplicate decision_id records"
                ),
                recommended_action="Audit evidence registry deduplication and ingestion sources.",
            )
        )

    missing = int(snapshot.get("decisions_missing_evidence") or 0)
    if missing > 0:
        anomalies.append(
            _make_anomaly(
                category="execution",
                severity="MEDIUM",
                affected_component="evidence_chain",
                detected_condition=f"{missing} decisions missing linked evidence records",
                recommended_action="Run evidence registry sync and verify _emit_gate_evidence paths.",
            )
        )

    return anomalies


def detect_governance_anomalies(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    anomalies: List[Dict[str, Any]] = []
    age_h = snapshot.get("evidence_age_hours")
    cert_age = snapshot.get("cert_age_days")

    if age_h is not None and age_h > 72:
        anomalies.append(
            _make_anomaly(
                category="governance",
                severity="MEDIUM" if age_h < 168 else "HIGH",
                affected_component="evidence_registry",
                detected_condition=f"Stale authorization records: {age_h:.0f}h since last evidence",
                recommended_action="Verify pipeline is emitting gate evidence during evaluations.",
            )
        )

    if not EVIDENCE_JSON.is_file() or not DECISIONS_JSON.is_file():
        anomalies.append(
            _make_anomaly(
                category="governance",
                severity="HIGH",
                affected_component="governance_artifacts",
                detected_condition="Missing governance JSON artifacts",
                recommended_action="Initialize THP-004 registry artifacts.",
            )
        )

    if not AUDIT_CSV.is_file():
        anomalies.append(
            _make_anomaly(
                category="governance",
                severity="MEDIUM",
                affected_component="audit_trail",
                detected_condition="Missing execution_gate_audit.csv",
                recommended_action="Ensure gate evaluations append audit rows when record_evidence=True.",
            )
        )

    cert = snapshot.get("certification") or {}
    if not cert:
        anomalies.append(
            _make_anomaly(
                category="governance",
                severity="HIGH",
                affected_component="certification",
                detected_condition="THP-006 certification artifact missing",
                recommended_action="Run python -m services.execution_gate_resilience_certification",
            )
        )
    elif cert_age is not None and cert_age > 30:
        anomalies.append(
            _make_anomaly(
                category="governance",
                severity="MEDIUM",
                affected_component="certification",
                detected_condition=f"Certification expired or stale: {cert_age:.0f} days old",
                recommended_action="Re-run full resilience certification suite.",
            )
        )

    drift_count = int(snapshot.get("drift_count") or 0)
    if drift_count > 0:
        anomalies.append(
            _make_anomaly(
                category="governance",
                severity="HIGH",
                affected_component="replay_drift",
                detected_condition=f"{drift_count} authorization drift events detected",
                recommended_action="Review drift report and reconcile policy changes.",
            )
        )

    return anomalies


def detect_operational_anomalies(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    anomalies: List[Dict[str, Any]] = []
    probe = snapshot.get("gate_probe") or {}

    if not probe.get("available"):
        anomalies.append(
            _make_anomaly(
                category="operational",
                severity="CRITICAL",
                affected_component="execution_gate",
                detected_condition=f"Execution Gate unavailable: {probe.get('reason', '')}",
                recommended_action="Verify Node.js gate script and evaluate-execution-gate.mjs.",
            )
        )

    if snapshot.get("gate_disabled"):
        anomalies.append(
            _make_anomaly(
                category="operational",
                severity="CRITICAL",
                affected_component="execution_gate",
                detected_condition="TRITON_DISABLE_EXECUTION_GATE bypass is active",
                recommended_action="Remove bypass env var for production operation.",
            )
        )

    if not snapshot.get("paper_live_boundary_intact", True):
        anomalies.append(
            _make_anomaly(
                category="operational",
                severity="CRITICAL",
                affected_component="environment_isolation",
                detected_condition="Paper/live boundary violation detected in certification",
                recommended_action="Immediate governance review — halt capital deployment.",
            )
        )

    review_count = int(snapshot.get("review_count") or 0)
    if review_count > 0:
        anomalies.append(
            _make_anomaly(
                category="operational",
                severity="MEDIUM",
                affected_component="replay_engine",
                detected_condition=f"{review_count} replay records require manual review",
                recommended_action="Investigate historical decision context reconstruction.",
            )
        )

    return anomalies


def detect_anomalies(snapshot: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    snapshot = snapshot or build_monitoring_snapshot()
    anomalies: List[Dict[str, Any]] = []
    anomalies.extend(detect_execution_anomalies(snapshot))
    anomalies.extend(detect_governance_anomalies(snapshot))
    anomalies.extend(detect_operational_anomalies(snapshot))
    return anomalies


def run_anomaly_detection(snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    snapshot = snapshot or build_monitoring_snapshot()
    records = detect_anomalies(snapshot)
    by_severity: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    for rec in records:
        by_severity[rec["severity"]] = by_severity.get(rec["severity"], 0) + 1
        by_category[rec["category"]] = by_category.get(rec["category"], 0) + 1

    report = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-007",
        "summary": {
            "total": len(records),
            "critical": by_severity.get("CRITICAL", 0),
            "high": by_severity.get("HIGH", 0),
            "medium": by_severity.get("MEDIUM", 0),
            "by_category": by_category,
        },
        "records": records,
    }
    _atomic_write_json(ANOMALIES_JSON, report)
    return report


def load_anomalies() -> Dict[str, Any]:
    if not ANOMALIES_JSON.is_file():
        return {}
    try:
        return json.loads(ANOMALIES_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    result = run_anomaly_detection()
    print(json.dumps(result.get("summary"), indent=2))
