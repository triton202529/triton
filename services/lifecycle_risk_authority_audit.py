"""
lifecycle_risk_authority_audit.py
THP-H006 — Verify lifecycle and risk authority consolidation on execution consumers.
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

COMPLIANCE_JSON = GOVERNANCE_DIR / "lifecycle_risk_authority_compliance.json"
COMPLIANCE_CSV = GOVERNANCE_DIR / "lifecycle_risk_authority_compliance.csv"

COMPLIANCE_FIELDS = (
    "consumer",
    "domain",
    "role",
    "uses_authoritative_source",
    "uses_deprecated_source",
    "compliance_result",
    "notes",
)

EXECUTION_CONSUMERS = (
    "services/execute_trades.py",
    "services/place_live_orders.py",
    "services/manage_positions.py",
    "services/governance_execution_readiness.py",
    "services/ticker_timeline_builder.py",
    "services/pnl_ledger_builder.py",
)

LIFECYCLE_DEPRECATED_PATTERNS = (
    r"signal_lifecycle\.csv(?!_effective)",
    r"apply_effective_lifecycle_columns\s*\(",
    r'"trade_opps"',
    r"'trade_opps'",
    r"return DEFAULT_LIFECYCLE_RAW",
)

RISK_DEPRECATED_PATTERNS = (
    r'RESULTS\s*/\s*"adaptive_risk_state\.json"',
    r"adaptive_risk_state\.json",
)


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


def _read(rel: str) -> str:
    p = ROOT / rel.replace("/", os.sep)
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")


def _audit_consumer(rel: str) -> List[Dict[str, Any]]:
    text = _read(rel)
    rows: List[Dict[str, Any]] = []

    lifecycle_auth = (
        "signal_lifecycle_effective" in text
        or "decision_authority" in text
        or "validate_authoritative_lifecycle" in text
    )
    lifecycle_dep = any(re.search(p, text) for p in LIFECYCLE_DEPRECATED_PATTERNS)
    # Advisory intelligence reads of adaptive_risk in timeline builder are OK
    risk_dep = False
    if rel in (
        "services/execute_trades.py",
        "services/place_live_orders.py",
        "services/manage_positions.py",
    ):
        risk_dep = "adaptive_risk_state" in text and "decision_authority" not in text
    risk_auth = "decision_authority" in text or "master_execution_gate" in text

    if rel in EXECUTION_CONSUMERS or "execute" in rel:
        rows.append(
            {
                "consumer": rel,
                "domain": "lifecycle_authorization",
                "role": (
                    "EXECUTION"
                    if "execute" in rel or "place_live" in rel or "manage" in rel
                    else "SUPPORTING"
                ),
                "uses_authoritative_source": lifecycle_auth,
                "uses_deprecated_source": lifecycle_dep,
                "compliance_result": (
                    "CONSOLIDATED"
                    if lifecycle_auth and not lifecycle_dep
                    else "PARTIAL" if lifecycle_auth else "NON_COMPLIANT"
                ),
                "notes": "THP-H006 effective lifecycle authority",
            }
        )

    if rel in (
        "services/execute_trades.py",
        "services/place_live_orders.py",
        "services/manage_positions.py",
    ):
        rows.append(
            {
                "consumer": rel,
                "domain": "risk_restriction",
                "role": "EXECUTION",
                "uses_authoritative_source": risk_auth and not risk_dep,
                "uses_deprecated_source": risk_dep,
                "compliance_result": (
                    "CONSOLIDATED"
                    if risk_auth and not risk_dep
                    else "PARTIAL" if risk_auth else "NON_COMPLIANT"
                ),
                "notes": "THP-H006 master_execution_gate.json authority",
            }
        )

    return rows


def run_lifecycle_risk_authority_audit() -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []
    for rel in EXECUTION_CONSUMERS:
        all_rows.extend(_audit_consumer(rel))

    consolidated = sum(1 for r in all_rows if r.get("compliance_result") == "CONSOLIDATED")
    partial = sum(1 for r in all_rows if r.get("compliance_result") == "PARTIAL")
    non = sum(1 for r in all_rows if r.get("compliance_result") == "NON_COMPLIANT")

    high_conflicts_resolved = all(
        r.get("compliance_result") in ("CONSOLIDATED", "PARTIAL")
        for r in all_rows
        if r.get("domain") in ("lifecycle_authorization", "risk_restriction")
        and r.get("role") == "EXECUTION"
    )

    payload = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-H006",
        "authorities": {
            "lifecycle": "data/results/signal_lifecycle_effective.csv",
            "risk_execution": "data/results/master_execution_gate.json",
            "risk_intelligence": "data/results/adaptive_risk_state.json (non-execution)",
        },
        "summary": {
            "consumers_audited": len(all_rows),
            "consolidated": consolidated,
            "partial": partial,
            "non_compliant": non,
            "execution_consumers_consolidated": high_conflicts_resolved,
        },
        "records": all_rows,
        "enforcement": {
            "lifecycle_module": "services/decision_authority.py",
            "risk_reader": "load_risk_execution_controls()",
        },
    }

    _atomic_write_json(COMPLIANCE_JSON, payload)
    _write_csv(COMPLIANCE_CSV, all_rows, COMPLIANCE_FIELDS)
    return payload


if __name__ == "__main__":
    result = run_lifecycle_risk_authority_audit()
    s = result["summary"]
    print(f"Audit: {COMPLIANCE_JSON}")
    print(
        f"CONSOLIDATED={s['consolidated']} PARTIAL={s['partial']} "
        f"NON_COMPLIANT={s['non_compliant']} "
        f"execution_ok={s['execution_consumers_consolidated']}"
    )
