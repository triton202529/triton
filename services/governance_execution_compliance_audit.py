"""
governance_execution_compliance_audit.py
THP-H004 — Verify Governance Council authority on all ACTIVE execution paths.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"
GATE_PATH = ROOT / "services" / "triton_execution_gate.py"
REGISTRY_JSON = GOVERNANCE_DIR / "execution_path_registry.json"

COMPLIANCE_JSON = GOVERNANCE_DIR / "governance_execution_compliance.json"
COMPLIANCE_CSV = GOVERNANCE_DIR / "governance_execution_compliance.csv"

COMPLIANCE_FIELDS = (
    "execution_path",
    "status",
    "uses_execution_gate",
    "uses_governance_council",
    "approval_workflow_status",
    "broker_authorization_status",
    "compliance_result",
    "notes",
)

# Capital deployment surfaces audited in THP-H004
PRIMARY_PATHS = (
    "services/execute_trades.py",
    "services/place_live_orders.py",
    "services/manage_positions.py",
    "place_orders_from_csv.py",
    "services/broker_alpaca.py",
    "services/triton_execution_gate.py",
    "pages/manual_order_desk.py",
    "pages_order_desk.py",
    "services/run_execution_loop.py",
    "services/run_scheduled_paper_cycle.py",
)

DELEGATION_TARGETS = {
    "services/run_execution_loop.py": "services/execute_trades.py",
    "services/run_scheduled_paper_cycle.py": "services/execute_trades.py",
    "pages/manual_order_desk.py": "services/broker_alpaca.py",
    "pages_order_desk.py": "services/broker_alpaca.py",
}


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
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _read_text(rel_path: str) -> str:
    full = ROOT / rel_path.replace("/", os.sep)
    if not full.is_file():
        return ""
    return full.read_text(encoding="utf-8", errors="ignore")


def _load_registry_status() -> Dict[str, str]:
    if not REGISTRY_JSON.is_file():
        return {}
    try:
        doc = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
        return {
            str(p.get("file_name", "")): str(p.get("status", "UNKNOWN")).upper()
            for p in doc.get("paths") or []
        }
    except Exception:
        return {}


def _gate_infrastructure_governed() -> bool:
    text = _read_text("services/triton_execution_gate.py")
    if "def enforce_pipeline_gate" not in text:
        return False
    pipeline_block = re.search(
        r"def enforce_pipeline_gate\([\s\S]*?(?=\ndef enforce_broker_submission_gate)",
        text,
    )
    broker_block = re.search(
        r"def enforce_broker_submission_gate\([\s\S]*?(?=\n(?:def |$))",
        text,
    )
    pipeline_ok = bool(pipeline_block and "evaluate_governed_execution" in pipeline_block.group(0))
    broker_ok = bool(broker_block and "evaluate_governed_execution" in broker_block.group(0))
    return pipeline_ok and broker_ok


def _delegates_to_governed_chain(rel_path: str) -> bool:
    target = DELEGATION_TARGETS.get(rel_path)
    if not target:
        return False
    text = _read_text(target)
    return "enforce_pipeline_gate" in text or "enforce_broker_submission_gate" in text


def _analyze_path(rel_path: str, status: str, infra_governed: bool) -> Dict[str, Any]:
    text = _read_text(rel_path)
    exists = bool(text)

    uses_eval_gate = "evaluate_execution_gate" in text
    uses_governed = "evaluate_governed_execution" in text
    uses_pipeline = "enforce_pipeline_gate" in text
    uses_broker_gate = "enforce_broker_submission_gate" in text
    uses_assert = "assert_execution_allowed" in text

    uses_execution_gate = uses_eval_gate or uses_pipeline or uses_broker_gate or uses_assert
    uses_governance_council = uses_governed or (
        rel_path == "services/triton_execution_gate.py" and infra_governed
    )

    if rel_path == "services/triton_execution_gate.py":
        approval_status = "ENFORCED" if infra_governed else "MISSING"
        broker_status = "ENFORCED" if infra_governed else "MISSING"
    elif uses_pipeline:
        approval_status = "ENFORCED" if infra_governed else "GATE_ONLY"
        broker_status = "DELEGATED" if not uses_broker_gate else "ENFORCED"
    elif uses_broker_gate or rel_path == "services/broker_alpaca.py":
        approval_status = "DELEGATED"
        broker_status = "ENFORCED" if infra_governed else "GATE_ONLY"
    elif rel_path in DELEGATION_TARGETS:
        approval_status = "DELEGATED" if _delegates_to_governed_chain(rel_path) else "MISSING"
        broker_status = "DELEGATED" if _delegates_to_governed_chain(rel_path) else "MISSING"
    else:
        approval_status = "NONE"
        broker_status = "NONE"

    notes: List[str] = []
    compliance = "NON_COMPLIANT"

    if status != "ACTIVE":
        return {
            "execution_path": rel_path,
            "status": status,
            "uses_execution_gate": uses_execution_gate,
            "uses_governance_council": uses_governance_council,
            "approval_workflow_status": "N/A",
            "broker_authorization_status": "N/A",
            "compliance_result": "N/A",
            "notes": "Not an active capital path",
        }

    if not exists:
        compliance = "NON_COMPLIANT"
        notes.append("file missing")
    elif rel_path == "services/triton_execution_gate.py":
        compliance = "FULLY_GOVERNED" if infra_governed else "NON_COMPLIANT"
        if not infra_governed:
            notes.append("enforce_pipeline_gate or enforce_broker_submission_gate not governed")
    elif rel_path in DELEGATION_TARGETS:
        if _delegates_to_governed_chain(rel_path) and infra_governed:
            compliance = "FULLY_GOVERNED"
            notes.append("delegates to governed institutional chain")
        else:
            compliance = "NON_COMPLIANT"
            notes.append("delegation chain not governed")
    elif uses_pipeline and infra_governed:
        compliance = "FULLY_GOVERNED"
    elif (uses_broker_gate or rel_path == "services/broker_alpaca.py") and infra_governed:
        compliance = "FULLY_GOVERNED"
    elif uses_governed and not uses_pipeline and not uses_broker_gate:
        compliance = "PARTIALLY_GOVERNED"
        notes.append("governed evaluation present but no enforce_* wrapper")
    elif uses_execution_gate and not uses_governance_council and not infra_governed:
        compliance = "NON_COMPLIANT"
        notes.append("execution gate without governance council overlay")
    elif uses_execution_gate:
        compliance = "PARTIALLY_GOVERNED"
        notes.append("gate present; governance overlay incomplete")

    return {
        "execution_path": rel_path,
        "status": status,
        "uses_execution_gate": uses_execution_gate,
        "uses_governance_council": uses_governance_council,
        "approval_workflow_status": approval_status,
        "broker_authorization_status": broker_status,
        "compliance_result": compliance,
        "notes": "; ".join(notes),
    }


def run_governance_compliance_audit() -> Dict[str, Any]:
    registry_status = _load_registry_status()
    infra_governed = _gate_infrastructure_governed()

    paths_to_audit = list(dict.fromkeys(list(PRIMARY_PATHS) + list(registry_status.keys())))
    rows: List[Dict[str, Any]] = []
    for rel in paths_to_audit:
        status = registry_status.get(rel, "UNKNOWN")
        if status not in ("ACTIVE", "DEPRECATED", "ARCHIVED"):
            if rel in PRIMARY_PATHS:
                status = "ACTIVE"
            else:
                continue
        row = _analyze_path(rel, status, infra_governed)
        rows.append(row)

    active_rows = [r for r in rows if r.get("status") == "ACTIVE"]
    fully = sum(1 for r in active_rows if r.get("compliance_result") == "FULLY_GOVERNED")
    partial = sum(1 for r in active_rows if r.get("compliance_result") == "PARTIALLY_GOVERNED")
    non = sum(1 for r in active_rows if r.get("compliance_result") == "NON_COMPLIANT")

    summary = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H004",
        "gate_infrastructure_governed": infra_governed,
        "total_paths_audited": len(rows),
        "active_paths_audited": len(active_rows),
        "fully_governed": fully,
        "partially_governed": partial,
        "non_compliant": non,
        "all_active_fully_governed": non == 0 and partial == 0 and fully == len(active_rows),
        "required_chain": [
            "evaluate_governed_execution",
            "Execution Gate Decision",
            "Governance Council Review",
            "Approval Workflow",
            "enforce_broker_submission_gate",
            "Alpaca",
        ],
    }

    payload = {"summary": summary, "records": rows}
    _atomic_write_json(COMPLIANCE_JSON, payload)
    _write_csv(
        COMPLIANCE_CSV,
        [{k: r.get(k, "") for k in COMPLIANCE_FIELDS} for r in rows],
        COMPLIANCE_FIELDS,
    )
    return payload


if __name__ == "__main__":
    result = run_governance_compliance_audit()
    s = result["summary"]
    print(f"Compliance audit: {COMPLIANCE_JSON}")
    print(
        f"Active: {s['active_paths_audited']} | "
        f"FULLY_GOVERNED={s['fully_governed']} | "
        f"PARTIAL={s['partially_governed']} | "
        f"NON_COMPLIANT={s['non_compliant']}"
    )
    print(f"All active fully governed: {s['all_active_fully_governed']}")
