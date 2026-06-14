"""
execution_decision_reconstructor.py
THP-009 — Forensic decision reconstruction and policy drift analysis.
"""

from __future__ import annotations

import csv
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.execution_approval_workflow import evaluate_governed_execution
from services.execution_gate_evidence_registry import load_decisions
from services.execution_gate_replay_engine import (
    build_current_context,
    classify_replay,
    reconstruct_historical_context,
    replay_decision,
)
from services.execution_governance_council import determine_authority_level
from services.execution_time_machine import (
    find_snapshot_by_decision_id,
    run_time_machine,
    verify_snapshot_integrity,
)
from services.triton_execution_gate import evaluate_execution_gate

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"

RECONSTRUCTION_JSON = GOVERNANCE_DIR / "execution_reconstruction_reports.json"
RECONSTRUCTION_CSV = GOVERNANCE_DIR / "execution_reconstruction_reports.csv"
POLICY_DRIFT_JSON = GOVERNANCE_DIR / "execution_policy_drift_analysis.json"

RECONSTRUCTION_FIELDS = (
    "reconstruction_id",
    "decision_id",
    "timestamp",
    "execution_path",
    "original_decision",
    "original_allowed",
    "current_gate_decision",
    "current_gate_allowed",
    "current_governed_decision",
    "current_governed_allowed",
    "comparison_classification",
    "narrative_summary",
    "integrity_valid",
    "missing_evidence_count",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reconstruction_id() -> str:
    return f"eg-rec-{uuid.uuid4().hex[:12]}"


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


def _find_decision(decision_id: str) -> Optional[Dict[str, Any]]:
    for d in load_decisions():
        if d.get("decision_id") == decision_id:
            return d
    return None


def classify_decision_comparison(
    *,
    original: Dict[str, Any],
    replay_result: Dict[str, Any],
    governed_eval: Dict[str, Any],
    historical_ctx: Dict[str, Any],
    current_ctx: Dict[str, Any],
    authority_level_now: int,
    authority_level_then: int,
) -> Tuple[str, str]:
    orig_dec = str(original.get("execution_gate_decision") or "")
    orig_allowed = bool(original.get("allowed"))
    gate_dec = str(governed_eval.get("decision") or "")
    gate_allowed = bool(governed_eval.get("allowed"))
    replay_class = str(replay_result.get("classification") or "")

    if (
        gate_dec.upper() == orig_dec.upper()
        and gate_allowed == orig_allowed
        and replay_class == "CONSISTENT"
    ):
        return (
            "SAME_DECISION",
            "Current gate and governance model reproduce the original authorization outcome.",
        )

    reasons: List[str] = []
    if authority_level_now > authority_level_then:
        reasons.append("governance")
    if original.get("risk_state") != replay_result.get("risk_state_current"):
        reasons.append("risk")
    if original.get("environment") != replay_result.get("environment_current"):
        reasons.append("environment")
    if replay_class == "DRIFT_DETECTED":
        reasons.append("policy")

    if "policy" in reasons or replay_class == "DRIFT_DETECTED":
        return (
            "POLICY_CHANGED",
            _build_narrative(
                original, governed_eval, authority_level_then, authority_level_now, "policy"
            ),
        )
    if "risk" in reasons:
        return (
            "RISK_CHANGED",
            _build_narrative(
                original, governed_eval, authority_level_then, authority_level_now, "risk"
            ),
        )
    if "governance" in reasons:
        return (
            "GOVERNANCE_CHANGED",
            _build_narrative(
                original, governed_eval, authority_level_then, authority_level_now, "governance"
            ),
        )
    if "environment" in reasons:
        return (
            "ENVIRONMENT_CHANGED",
            _build_narrative(
                original, governed_eval, authority_level_then, authority_level_now, "environment"
            ),
        )

    return (
        "POLICY_CHANGED",
        _build_narrative(
            original, governed_eval, authority_level_then, authority_level_now, "general"
        ),
    )


def _build_narrative(
    original: Dict[str, Any],
    governed: Dict[str, Any],
    level_then: int,
    level_now: int,
    focus: str,
) -> str:
    ts = str(original.get("timestamp") or "")[:10]
    path = original.get("execution_path")
    orig_dec = original.get("execution_gate_decision")
    risk = original.get("risk_state")
    gov = original.get("governance_status")
    cur_dec = governed.get("decision")
    cur_allowed = governed.get("allowed")

    base = (
        f"Execution {ts} on path '{path}' was {orig_dec} because risk was {risk}, "
        f"governance was {gov}, and capital exposure was within recorded limits."
    )
    if level_now > level_then:
        suffix = (
            f" Under today's governance model (Level {level_now} vs Level {level_then} then) "
            f"this execution would require Level {level_now} council approval due to updated "
            f"capital authority thresholds. Current governed outcome: {cur_dec} (allowed={cur_allowed})."
        )
    else:
        suffix = (
            f" Current governed outcome: {cur_dec} (allowed={cur_allowed}). Focus: {focus} drift."
        )

    return base + suffix


def _detect_missing_evidence(
    decision: Dict[str, Any], snapshot: Optional[Dict[str, Any]]
) -> List[str]:
    missing: List[str] = []
    if not decision.get("authorization_chain"):
        missing.append("authorization_chain")
    if not decision.get("evidence_links"):
        missing.append("evidence_links")
    if snapshot:
        integrity = verify_snapshot_integrity(snapshot)
        if not integrity.get("fingerprint_valid"):
            missing.append("snapshot_fingerprint_invalid")
        if not snapshot.get("frozen_evidence"):
            missing.append("frozen_evidence_empty")
    else:
        missing.append("no_snapshot")
    return missing


def reconstruct_decision(decision_id: str) -> Dict[str, Any]:
    decision = _find_decision(decision_id)
    if not decision:
        raise ValueError(f"Decision not found: {decision_id}")

    snapshot = find_snapshot_by_decision_id(decision_id)
    if not snapshot:
        run_time_machine()
        snapshot = find_snapshot_by_decision_id(decision_id)

    replay = replay_decision(decision)
    path = str(decision.get("execution_path") or "")
    historical_ctx = reconstruct_historical_context(decision)
    current_ctx = build_current_context(decision)

    gate_eval = evaluate_execution_gate(
        path, current_ctx, record_evidence=False, source="thp009_reconstruction"
    )
    governed = evaluate_governed_execution(
        path,
        current_ctx,
        record_evidence=False,
        source="thp009_reconstruction",
        execution_identifier=decision.get("execution_identifier"),
        capital_exposure=decision.get("capital_exposure"),
        skip_governance=False,
        record_requests=False,
    )

    level_then, _ = determine_authority_level(
        path=path,
        context=historical_ctx,
        evaluation={
            "decision": decision.get("execution_gate_decision"),
            "allowed": decision.get("allowed"),
        },
        capital_exposure=decision.get("capital_exposure"),
    )
    level_now, triggers_now = determine_authority_level(
        path=path,
        context=current_ctx,
        evaluation=gate_eval,
        capital_exposure=decision.get("capital_exposure"),
    )

    comparison, narrative = classify_decision_comparison(
        original=decision,
        replay_result=replay,
        governed_eval=governed,
        historical_ctx=historical_ctx,
        current_ctx=current_ctx,
        authority_level_now=level_now,
        authority_level_then=level_then,
    )

    missing = _detect_missing_evidence(decision, snapshot)
    integrity = verify_snapshot_integrity(snapshot) if snapshot else {"fingerprint_valid": False}

    report = {
        "reconstruction_id": _reconstruction_id(),
        "generated_at": _utc_now_iso(),
        "framework": "THP-009",
        "decision_id": decision_id,
        "snapshot_id": (snapshot or {}).get("snapshot_id"),
        "original_decision": {
            "decision": decision.get("execution_gate_decision"),
            "allowed": decision.get("allowed"),
            "reason": decision.get("decision_reason"),
            "timestamp": decision.get("timestamp"),
            "execution_path": path,
            "authorization_chain": decision.get("authorization_chain"),
        },
        "current_analysis": {
            "gate_decision": gate_eval.get("decision"),
            "gate_allowed": gate_eval.get("allowed"),
            "governed_decision": governed.get("decision"),
            "governed_allowed": governed.get("allowed"),
            "authority_level_then": level_then,
            "authority_level_now": level_now,
            "authority_triggers_now": triggers_now,
            "replay_classification": replay.get("classification"),
            "changed_conditions": replay.get("changed_conditions"),
        },
        "comparison_classification": comparison,
        "narrative_summary": narrative,
        "integrity": integrity,
        "missing_evidence": missing,
        "missing_evidence_count": len(missing),
        "integrity_valid": integrity.get("fingerprint_valid", False) and len(missing) == 0,
    }
    return report


def run_reconstruction_batch(*, limit: Optional[int] = None) -> Dict[str, Any]:
    run_time_machine()
    decisions = load_decisions()
    if limit:
        decisions = decisions[-limit:]

    reports: List[Dict[str, Any]] = []
    for decision in decisions:
        did = decision.get("decision_id")
        if not did:
            continue
        try:
            reports.append(reconstruct_decision(str(did)))
        except Exception as exc:
            reports.append(
                {
                    "reconstruction_id": _reconstruction_id(),
                    "decision_id": did,
                    "comparison_classification": "REQUIRES_REVIEW",
                    "narrative_summary": f"Reconstruction failed: {exc}",
                    "integrity_valid": False,
                    "missing_evidence_count": -1,
                }
            )

    by_class: Dict[str, int] = {}
    for r in reports:
        c = str(r.get("comparison_classification") or "UNKNOWN")
        by_class[c] = by_class.get(c, 0) + 1

    doc = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-009",
        "summary": {
            "total": len(reports),
            "by_classification": by_class,
            "integrity_valid_count": sum(1 for r in reports if r.get("integrity_valid")),
        },
        "records": reports,
    }
    _atomic_write_json(RECONSTRUCTION_JSON, doc)

    csv_rows = []
    for r in reports:
        orig = r.get("original_decision") or {}
        cur = r.get("current_analysis") or {}
        csv_rows.append(
            {
                "reconstruction_id": r.get("reconstruction_id"),
                "decision_id": r.get("decision_id"),
                "timestamp": orig.get("timestamp"),
                "execution_path": orig.get("execution_path"),
                "original_decision": orig.get("decision"),
                "original_allowed": orig.get("allowed"),
                "current_gate_decision": cur.get("gate_decision"),
                "current_gate_allowed": cur.get("gate_allowed"),
                "current_governed_decision": cur.get("governed_decision"),
                "current_governed_allowed": cur.get("governed_allowed"),
                "comparison_classification": r.get("comparison_classification"),
                "narrative_summary": r.get("narrative_summary"),
                "integrity_valid": r.get("integrity_valid"),
                "missing_evidence_count": r.get("missing_evidence_count"),
            }
        )
    _write_csv(RECONSTRUCTION_CSV, csv_rows, RECONSTRUCTION_FIELDS)

    drift_doc = build_policy_drift_analysis(reports)
    _atomic_write_json(POLICY_DRIFT_JSON, drift_doc)
    return doc


def build_policy_drift_analysis(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    drift_records = []
    for r in reports:
        if r.get("comparison_classification") == "SAME_DECISION":
            continue
        drift_records.append(
            {
                "decision_id": r.get("decision_id"),
                "classification": r.get("comparison_classification"),
                "narrative": r.get("narrative_summary"),
                "replay_classification": (r.get("current_analysis") or {}).get(
                    "replay_classification"
                ),
                "authority_level_then": (r.get("current_analysis") or {}).get(
                    "authority_level_then"
                ),
                "authority_level_now": (r.get("current_analysis") or {}).get("authority_level_now"),
            }
        )

    return {
        "generated_at": _utc_now_iso(),
        "framework": "THP-009",
        "summary": {
            "drift_count": len(drift_records),
            "same_decision_count": sum(
                1 for r in reports if r.get("comparison_classification") == "SAME_DECISION"
            ),
            "by_classification": {
                c: sum(1 for d in drift_records if d.get("classification") == c)
                for c in (
                    "POLICY_CHANGED",
                    "RISK_CHANGED",
                    "GOVERNANCE_CHANGED",
                    "ENVIRONMENT_CHANGED",
                )
            },
        },
        "records": drift_records,
    }


def load_reconstruction_reports() -> Dict[str, Any]:
    if not RECONSTRUCTION_JSON.is_file():
        return {}
    try:
        return json.loads(RECONSTRUCTION_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    doc = run_reconstruction_batch()
    print(json.dumps(doc.get("summary"), indent=2))
