"""
broker_execution_audit.py
THP-H002 — Broker execution path protection inventory and bypass report.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"
BROKER_PATH = ROOT / "services" / "broker_alpaca.py"

PROTECTION_MATRIX_JSON = GOVERNANCE_DIR / "broker_execution_protection_matrix.json"
PROTECTION_MATRIX_CSV = GOVERNANCE_DIR / "broker_execution_protection_matrix.csv"
BYPASS_REPORT_JSON = GOVERNANCE_DIR / "execution_bypass_report.json"

MATRIX_FIELDS = (
    "function_name",
    "order_type",
    "protection_status",
    "gate_chain",
    "broker_final_check",
    "calling_services",
    "risk_severity",
)

# Catalog of AlpacaBroker order-related methods
BROKER_METHOD_CATALOG: List[Dict[str, Any]] = [
    {
        "function_name": "submit_order",
        "order_type": "generic_qty",
        "expected_status": "EXECUTION_PROTECTED",
        "gate_chain": "enforce_broker_submission_gate → evaluate_governed_execution",
    },
    {
        "function_name": "submit_order_notional",
        "order_type": "notional_market",
        "expected_status": "EXECUTION_PROTECTED",
        "gate_chain": "enforce_broker_submission_gate → evaluate_governed_execution",
    },
    {
        "function_name": "submit_bracket_order_notional",
        "order_type": "notional_bracket",
        "expected_status": "EXECUTION_PROTECTED",
        "gate_chain": "enforce_broker_submission_gate → evaluate_governed_execution",
    },
    {
        "function_name": "place_bracket_market",
        "order_type": "bracket_market_qty",
        "expected_status": "EXECUTION_PROTECTED",
        "gate_chain": "submit_order (gated)",
    },
    {
        "function_name": "place_bracket",
        "order_type": "bracket_entry",
        "expected_status": "EXECUTION_PROTECTED",
        "gate_chain": "submit_order (gated)",
    },
    {
        "function_name": "submit_market_order",
        "order_type": "market_qty",
        "expected_status": "EXECUTION_PROTECTED",
        "gate_chain": "submit_order (gated)",
    },
    {
        "function_name": "submit_limit_order",
        "order_type": "limit_qty",
        "expected_status": "EXECUTION_PROTECTED",
        "gate_chain": "submit_order (gated)",
    },
    {
        "function_name": "place_order",
        "order_type": "alias_qty",
        "expected_status": "EXECUTION_PROTECTED",
        "gate_chain": "submit_order (gated)",
    },
    {
        "function_name": "place_order_notional",
        "order_type": "alias_notional",
        "expected_status": "EXECUTION_PROTECTED",
        "gate_chain": "submit_order_notional (gated)",
    },
    {
        "function_name": "cancel_order",
        "order_type": "cancel",
        "expected_status": "EXECUTION_PROTECTED",
        "gate_chain": "N/A — order cancellation (DELETE)",
    },
    {
        "function_name": "cancel_all_orders",
        "order_type": "cancel_all",
        "expected_status": "EXECUTION_PROTECTED",
        "gate_chain": "N/A — order cancellation (DELETE)",
    },
    {
        "function_name": "_post",
        "order_type": "raw_http_post",
        "expected_status": "EXECUTION_PROTECTED",
        "gate_chain": "BROKER FINAL CHECK blocks /v2/orders without auth context",
    },
]

CALLER_MAP: Dict[str, List[str]] = {
    "submit_order": [
        "services/place_live_orders.py",
        "services/manage_open_orders.py",
        "services/reprice_open_orders.py",
        "services/reprice_order_ladder.py",
        "services/position_protector.py",
        "pages/manual_order_desk.py",
    ],
    "place_bracket_market": ["pages_order_desk.py"],
    "submit_order_notional": ["pages_order_desk.py"],
    "submit_bracket_order_notional": ["services/position_protector.py"],
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


def _read_broker_source() -> str:
    if not BROKER_PATH.is_file():
        return ""
    return BROKER_PATH.read_text(encoding="utf-8", errors="ignore")


def _verify_method_protection(name: str, source: str) -> Dict[str, Any]:
    """Static analysis of broker method protection."""
    pattern = rf"def {re.escape(name)}\("
    if not re.search(pattern, source):
        return {"found": False, "protected": False, "direct_post": False}

    # Extract method body up to next def at class level
    match = re.search(pattern + r"[\s\S]*?(?=\n    def |\nclass |\Z)", source)
    body = match.group(0) if match else ""

    direct_post = bool(re.search(r'self\._post\("/v2/orders"', body))
    uses_submit = "self.submit_order" in body or "self.submit_order_notional" in body
    uses_gate = "enforce_broker_submission_gate" in body
    uses_protected = "_submit_order_payload" in body

    if name == "_post":
        protected = "BROKER FINAL CHECK" in source and "_broker_order_authorized" in source
        return {
            "found": True,
            "protected": protected,
            "direct_post": True,
            "broker_final_check": protected,
        }

    if name in ("cancel_order", "cancel_all_orders"):
        return {"found": True, "protected": True, "direct_post": False, "broker_final_check": False}

    protected = (uses_gate and uses_protected) or (uses_submit and not direct_post)
    return {
        "found": True,
        "protected": protected,
        "direct_post": direct_post,
        "uses_gate": uses_gate,
        "uses_protected_submit": uses_protected,
        "delegates_to_gated": uses_submit,
    }


def _find_external_bypass_risks() -> List[Dict[str, Any]]:
    """Known out-of-band order POST paths outside AlpacaBroker protections."""
    from services.execution_path_registry import (
        EXECUTION_PATHS,
        path_has_deprecation_guard,
        path_has_direct_order_post,
    )

    risks: List[Dict[str, Any]] = []
    for path, meta in sorted(EXECUTION_PATHS.items()):
        full = ROOT / path.replace("/", os.sep)
        if not full.is_file():
            continue
        status = str(meta.get("status", "")).upper()
        note = str(meta.get("purpose", ""))
        has_gate = "enforce_broker_submission_gate" in full.read_text(
            encoding="utf-8", errors="ignore"
        ) or "evaluate_governed_execution" in full.read_text(encoding="utf-8", errors="ignore")
        guarded = path_has_deprecation_guard(full)
        direct_post = path_has_direct_order_post(full)

        if status in ("DEPRECATED", "ARCHIVED"):
            risks.append(
                {
                    "path": path,
                    "note": note,
                    "exists": True,
                    "has_gate_reference": has_gate,
                    "deprecation_guard": guarded,
                    "severity": "RESOLVED" if guarded else "HIGH",
                    "status": "DEPRECATED_PROTECTED" if guarded else "BYPASS_RISK",
                }
            )
            continue

        if status == "ACTIVE" and direct_post and path != "services/broker_alpaca.py":
            risks.append(
                {
                    "path": path,
                    "note": note,
                    "exists": True,
                    "has_gate_reference": has_gate,
                    "severity": "MEDIUM",
                    "status": "PARTIAL_PROTECTION" if has_gate else "BYPASS_RISK",
                }
            )
    return risks


def run_broker_execution_audit() -> Dict[str, Any]:
    source = _read_broker_source()
    matrix_rows: List[Dict[str, Any]] = []
    bypass_risks: List[Dict[str, Any]] = []

    for entry in BROKER_METHOD_CATALOG:
        name = entry["function_name"]
        verification = _verify_method_protection(name, source)
        status = entry["expected_status"]
        if verification.get("found"):
            if verification.get("direct_post") and name != "_post":
                status = "BYPASS_RISK"
                bypass_risks.append(
                    {
                        "component": f"broker_alpaca.{name}",
                        "severity": "CRITICAL",
                        "description": f"{name} calls _post(/v2/orders) without gate chain",
                    }
                )
            elif not verification.get("protected"):
                status = "BYPASS_RISK"
                bypass_risks.append(
                    {
                        "component": f"broker_alpaca.{name}",
                        "severity": "HIGH",
                        "description": f"{name} missing expected protection",
                    }
                )

        matrix_rows.append(
            {
                "function_name": name,
                "order_type": entry["order_type"],
                "protection_status": status,
                "gate_chain": entry["gate_chain"],
                "broker_final_check": verification.get("broker_final_check", name == "_post"),
                "calling_services": "; ".join(CALLER_MAP.get(name, [])),
                "risk_severity": "NONE" if status == "EXECUTION_PROTECTED" else "CRITICAL",
            }
        )

    external_risks = _find_external_bypass_risks()
    bypass_risks.extend(
        {
            "component": r["path"],
            "severity": r["severity"],
            "description": r["note"],
        }
        for r in external_risks
        if r["status"] == "BYPASS_RISK"
    )

    protected_count = sum(1 for r in matrix_rows if r["protection_status"] == "EXECUTION_PROTECTED")
    bypass_count = sum(1 for r in matrix_rows if r["protection_status"] == "BYPASS_RISK")

    matrix_doc = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H002",
        "summary": {
            "total_paths_inspected": len(matrix_rows),
            "protected_paths": protected_count,
            "bypass_risks_in_broker": bypass_count,
            "external_bypass_risks": len(
                [r for r in external_risks if r["status"] == "BYPASS_RISK"]
            ),
            "broker_final_check_active": "BROKER FINAL CHECK" in source,
            "governed_gate_active": "evaluate_governed_execution"
            in (ROOT / "services" / "triton_execution_gate.py").read_text(
                encoding="utf-8", errors="ignore"
            ),
        },
        "records": matrix_rows,
    }

    bypass_doc = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H003",
        "summary": {
            "total_bypass_risks": len(
                [b for b in bypass_risks if b.get("severity") not in ("RESOLVED",)]
            ),
            "broker_layer_clean": bypass_count == 0,
            "remaining_external_risks": len(
                [r for r in external_risks if r["status"] == "BYPASS_RISK"]
            ),
            "deprecated_paths_protected": len(
                [r for r in external_risks if r["status"] == "DEPRECATED_PROTECTED"]
            ),
            "highest_severity": (
                "CRITICAL"
                if any(b.get("severity") == "CRITICAL" for b in bypass_risks)
                else "HIGH" if any(b.get("severity") == "HIGH" for b in bypass_risks) else "NONE"
            ),
        },
        "broker_bypass_risks": bypass_risks,
        "external_paths": external_risks,
    }

    _atomic_write_json(PROTECTION_MATRIX_JSON, matrix_doc)
    _write_csv(PROTECTION_MATRIX_CSV, matrix_rows, MATRIX_FIELDS)
    _atomic_write_json(BYPASS_REPORT_JSON, bypass_doc)

    return {"matrix": matrix_doc, "bypass_report": bypass_doc}


if __name__ == "__main__":
    result = run_broker_execution_audit()
    print(json.dumps(result["matrix"].get("summary"), indent=2))
