"""
execution_gate_health_monitor.py
THP-007 — Continuous execution gate health scoring and supervision (read-only).

Monitors authorization reliability, drift, governance freshness, and environment
integrity without modifying gate logic or trading behavior.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.execution_gate_evidence_registry import (
    AUDIT_CSV,
    load_decisions,
    load_evidence,
)
from services.execution_gate_replay_engine import DRIFT_REPORT_JSON, load_replay_results
from services.execution_gate_resilience_certification import load_certification
from services.triton_execution_gate import EXECUTION_PATHS, evaluate_execution_gate

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"

HEALTH_REPORT_JSON = GOVERNANCE_DIR / "execution_gate_health_report.json"
HEALTH_HISTORY_CSV = GOVERNANCE_DIR / "execution_gate_health_history.csv"

HEALTH_HISTORY_FIELDS = (
    "timestamp",
    "health_score",
    "health_classification",
    "operational_state",
    "anomaly_count",
    "gate_available",
    "drift_count",
    "certification_level",
)

HEALTH_COMPONENT_WEIGHTS = {
    "authorization_reliability": 0.12,
    "decision_consistency": 0.10,
    "failure_rate": 0.10,
    "replay_consistency": 0.10,
    "drift_status": 0.10,
    "governance_evidence_freshness": 0.12,
    "audit_integrity": 0.10,
    "paper_live_isolation": 0.12,
    "certification_status": 0.08,
    "environment_integrity": 0.06,
}

KNOWN_PATHS = set(EXECUTION_PATHS.values())
CERT_STALE_DAYS = 30
EVIDENCE_STALE_HOURS = 72


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _append_history(row: Dict[str, Any]) -> None:
    GOVERNANCE_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if HEALTH_HISTORY_CSV.is_file():
        with HEALTH_HISTORY_CSV.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows.append({k: str(row.get(k, "")) for k in HEALTH_HISTORY_FIELDS})
    tmp = HEALTH_HISTORY_CSV.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(HEALTH_HISTORY_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, HEALTH_HISTORY_CSV)


def classify_health(score: float) -> str:
    if score >= 95:
        return "GREEN"
    if score >= 85:
        return "YELLOW"
    if score >= 70:
        return "ORANGE"
    return "RED"


def _load_audit_rows() -> List[Dict[str, str]]:
    if not AUDIT_CSV.is_file():
        return []
    with AUDIT_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def probe_gate_availability() -> Dict[str, Any]:
    """Lightweight gate probe — record_evidence=False, no capital deployment."""
    try:
        result = evaluate_execution_gate(
            EXECUTION_PATHS["SIGNAL_GENERATION"],
            {"environment": "development"},
            record_evidence=False,
            source="health_monitor_probe",
        )
        return {
            "available": True,
            "decision": result.get("decision"),
            "allowed": result.get("allowed"),
            "reason": result.get("reason", ""),
        }
    except Exception as exc:
        return {
            "available": False,
            "decision": "UNAVAILABLE",
            "allowed": False,
            "reason": f"Gate probe failed: {exc}",
        }


def _load_drift_report() -> Dict[str, Any]:
    if not DRIFT_REPORT_JSON.is_file():
        return {}
    try:
        return json.loads(DRIFT_REPORT_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_monitoring_snapshot() -> Dict[str, Any]:
    """Collect all upstream artifacts for health, anomaly, and degradation analysis."""
    now = datetime.now(timezone.utc)
    decisions = load_decisions()
    evidence = load_evidence()
    audit_rows = _load_audit_rows()
    certification = load_certification()
    replay = load_replay_results()
    drift = _load_drift_report()
    gate_probe = probe_gate_availability()

    cert_generated = _parse_ts(certification.get("generated_at"))
    cert_age_days = (now - cert_generated).total_seconds() / 86400 if cert_generated else None

    evidence_ts = [_parse_ts(e.get("created_at") or e.get("timestamp")) for e in evidence]
    evidence_ts = [t for t in evidence_ts if t]
    latest_evidence = max(evidence_ts) if evidence_ts else None
    evidence_age_hours = (now - latest_evidence).total_seconds() / 3600 if latest_evidence else None

    decision_ids = [d.get("decision_id") for d in decisions if d.get("decision_id")]
    evidence_decision_ids = {e.get("decision_id") for e in evidence if e.get("decision_id")}

    recent_cutoff = now - timedelta(hours=24)
    recent_audit = [
        r
        for r in audit_rows
        if (_parse_ts(r.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))
        >= recent_cutoff
    ]
    block_count = sum(1 for r in recent_audit if str(r.get("decision", "")).upper() == "BLOCK")
    allow_count = sum(1 for r in recent_audit if str(r.get("allowed", "")).lower() in ("true", "1"))
    live_allows = [
        r
        for r in recent_audit
        if str(r.get("allowed", "")).lower() in ("true", "1")
        and (
            str(r.get("execution_path", "")) == "live_execution"
            or (
                str(r.get("execution_path", "")) == "broker_submission"
                and str(r.get("broker_mode", "")).lower() == "live"
            )
        )
    ]

    paths_seen = {str(r.get("execution_path", "")) for r in audit_rows}
    unknown_paths = sorted(paths_seen - KNOWN_PATHS - {""})

    return {
        "generated_at": _utc_now_iso(),
        "gate_probe": gate_probe,
        "decisions": decisions,
        "evidence": evidence,
        "audit_rows": audit_rows,
        "recent_audit": recent_audit,
        "certification": certification,
        "replay": replay,
        "drift": drift,
        "cert_age_days": cert_age_days,
        "evidence_age_hours": evidence_age_hours,
        "decision_count": len(decisions),
        "evidence_count": len(evidence),
        "duplicate_decision_ids": len(decision_ids) - len(set(decision_ids)),
        "decisions_missing_evidence": sum(
            1 for d in decisions if d.get("decision_id") not in evidence_decision_ids
        ),
        "recent_block_count": block_count,
        "recent_allow_count": allow_count,
        "recent_live_allows": live_allows,
        "unknown_paths": unknown_paths,
        "drift_count": int(drift.get("drift_count") or 0),
        "review_count": int(drift.get("review_count") or 0),
        "replay_consistent": int((replay.get("summary") or {}).get("consistent") or 0),
        "replay_total": int((replay.get("summary") or {}).get("total") or 0),
        "paper_live_boundary_intact": certification.get("paper_live_boundary_intact", True),
        "resilience_score": certification.get("resilience_score"),
        "certification_level": certification.get("certification_level"),
        "gate_disabled": os.environ.get("TRITON_DISABLE_EXECUTION_GATE", "").strip().lower()
        in ("1", "true", "yes", "on"),
    }


def _score_component(name: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    score = 100.0
    notes: List[str] = []

    if name == "authorization_reliability":
        probe = snapshot.get("gate_probe") or {}
        if not probe.get("available"):
            score = 0.0
            notes.append("Gate probe unavailable")
        elif snapshot.get("gate_disabled"):
            score = 50.0
            notes.append("TRITON_DISABLE_EXECUTION_GATE is set")
        elif len(snapshot.get("recent_live_allows") or []) > 0:
            score = max(60.0, 100.0 - 10 * len(snapshot["recent_live_allows"]))
            notes.append(f"{len(snapshot['recent_live_allows'])} recent live/broker ALLOW")

    elif name == "decision_consistency":
        dup = int(snapshot.get("duplicate_decision_ids") or 0)
        if dup > 0:
            score = max(0.0, 100.0 - dup * 15)
            notes.append(f"{dup} duplicate decision records")

    elif name == "failure_rate":
        recent = snapshot.get("recent_audit") or []
        if recent:
            block_rate = snapshot.get("recent_block_count", 0) / len(recent)
            if block_rate > 0.8:
                score = max(50.0, 100.0 - (block_rate - 0.5) * 100)
                notes.append(f"Elevated BLOCK rate: {block_rate:.0%}")
            elif block_rate > 0.5:
                score = 85.0
                notes.append(f"Moderate BLOCK rate: {block_rate:.0%}")

    elif name == "replay_consistency":
        total = int(snapshot.get("replay_total") or 0)
        consistent = int(snapshot.get("replay_consistent") or 0)
        if total > 0:
            score = round(100.0 * consistent / total, 2)
            notes.append(f"Replay {consistent}/{total} consistent")
        review = int(snapshot.get("review_count") or 0)
        if review > 0:
            score = max(0.0, score - review * 10)
            notes.append(f"{review} records require review")

    elif name == "drift_status":
        drift_count = int(snapshot.get("drift_count") or 0)
        if drift_count > 0:
            score = max(0.0, 100.0 - drift_count * 20)
            notes.append(f"{drift_count} drift events detected")

    elif name == "governance_evidence_freshness":
        age_h = snapshot.get("evidence_age_hours")
        if age_h is None and snapshot.get("decision_count", 0) == 0:
            score = 90.0
            notes.append("No governance evidence yet (expected in early deployment)")
        elif age_h is not None and age_h > EVIDENCE_STALE_HOURS:
            score = max(40.0, 100.0 - (age_h - EVIDENCE_STALE_HOURS) * 0.5)
            notes.append(f"Evidence stale: {age_h:.0f}h since last record")
        missing = int(snapshot.get("decisions_missing_evidence") or 0)
        if missing > 0:
            score = min(score, max(50.0, 100.0 - missing * 5))
            notes.append(f"{missing} decisions missing linked evidence")

    elif name == "audit_integrity":
        if not AUDIT_CSV.is_file():
            score = 70.0
            notes.append("Audit CSV not found")
        elif not snapshot.get("audit_rows"):
            score = 80.0
            notes.append("Audit CSV empty")

    elif name == "paper_live_isolation":
        if not snapshot.get("paper_live_boundary_intact", True):
            score = 0.0
            notes.append("Paper/live boundary violation in certification")
        live_allows = snapshot.get("recent_live_allows") or []
        if live_allows:
            score = min(score, 60.0)
            notes.append("Recent live/broker ALLOW in audit trail")

    elif name == "certification_status":
        cert = snapshot.get("certification") or {}
        if not cert:
            score = 60.0
            notes.append("No THP-006 certification artifact")
        else:
            resilience = cert.get("resilience_score")
            if resilience is not None:
                score = float(resilience)
            age = snapshot.get("cert_age_days")
            if age is not None and age > CERT_STALE_DAYS:
                score = min(score, max(50.0, 100.0 - (age - CERT_STALE_DAYS) * 2))
                notes.append(f"Certification age: {age:.0f} days")

    elif name == "environment_integrity":
        if snapshot.get("unknown_paths"):
            score = max(0.0, 100.0 - len(snapshot["unknown_paths"]) * 25)
            notes.append(f"Unknown paths: {snapshot['unknown_paths']}")
        if snapshot.get("gate_disabled"):
            score = min(score, 55.0)
            notes.append("Execution gate bypass env var active")

    return {
        "component": name,
        "score": round(score, 2),
        "weight": HEALTH_COMPONENT_WEIGHTS[name],
        "weighted_score": round(score * HEALTH_COMPONENT_WEIGHTS[name], 2),
        "notes": "; ".join(notes) if notes else "Nominal",
    }


def compute_health_report(snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    snapshot = snapshot or build_monitoring_snapshot()
    components = [_score_component(name, snapshot) for name in HEALTH_COMPONENT_WEIGHTS]
    health_score = round(sum(c["weighted_score"] for c in components), 2)
    classification = classify_health(health_score)

    recent_audit = snapshot.get("recent_audit") or []
    path_counts: Dict[str, int] = {}
    for row in recent_audit:
        path = str(row.get("execution_path") or "unknown")
        path_counts[path] = path_counts.get(path, 0) + 1

    return {
        "generated_at": _utc_now_iso(),
        "framework": "THP-007",
        "health_score": health_score,
        "health_classification": classification,
        "gate_available": (snapshot.get("gate_probe") or {}).get("available", False),
        "gate_probe": snapshot.get("gate_probe"),
        "components": components,
        "decision_analytics": {
            "recent_allow_count": snapshot.get("recent_allow_count", 0),
            "recent_block_count": snapshot.get("recent_block_count", 0),
            "recent_total": len(recent_audit),
            "most_active_paths": sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        },
        "governance_health": {
            "decision_count": snapshot.get("decision_count", 0),
            "evidence_count": snapshot.get("evidence_count", 0),
            "evidence_age_hours": snapshot.get("evidence_age_hours"),
            "drift_count": snapshot.get("drift_count", 0),
            "certification_level": snapshot.get("certification_level"),
            "certification_age_days": snapshot.get("cert_age_days"),
            "resilience_score": snapshot.get("resilience_score"),
        },
        "snapshot_summary": {
            "duplicate_decision_ids": snapshot.get("duplicate_decision_ids", 0),
            "decisions_missing_evidence": snapshot.get("decisions_missing_evidence", 0),
            "unknown_paths": snapshot.get("unknown_paths", []),
            "paper_live_boundary_intact": snapshot.get("paper_live_boundary_intact", True),
        },
    }


def run_health_monitor(snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if snapshot is None:
        snapshot = build_monitoring_snapshot()
    report = compute_health_report(snapshot)
    _atomic_write_json(HEALTH_REPORT_JSON, report)
    return report


def load_health_report() -> Dict[str, Any]:
    if not HEALTH_REPORT_JSON.is_file():
        return {}
    try:
        return json.loads(HEALTH_REPORT_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def append_health_history(
    *,
    health_score: float,
    health_classification: str,
    operational_state: str,
    anomaly_count: int,
    gate_available: bool,
    drift_count: int,
    certification_level: str,
) -> None:
    _append_history(
        {
            "timestamp": _utc_now_iso(),
            "health_score": health_score,
            "health_classification": health_classification,
            "operational_state": operational_state,
            "anomaly_count": anomaly_count,
            "gate_available": gate_available,
            "drift_count": drift_count,
            "certification_level": certification_level or "",
        }
    )


if __name__ == "__main__":
    report = run_health_monitor()
    print(
        json.dumps(
            {
                "health_score": report.get("health_score"),
                "classification": report.get("health_classification"),
            },
            indent=2,
        )
    )
