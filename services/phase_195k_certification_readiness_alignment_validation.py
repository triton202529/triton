"""
phase_195k_certification_readiness_alignment_validation.py
Phase 195K — Validate certification artifact alignment with 195H operational freeze.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

OUTPUT_JSON = RESULTS / "phase_195k_certification_readiness_alignment_validation.json"
CERT_PATH = RESULTS / "capital_preservation_certification.json"

PAPER_CERTIFICATION_STATUSES = frozenset(
    {
        "CERTIFIED_FOR_PAPER_PROTECTION",
        "CONDITIONALLY_CERTIFIED",
        "PARTIALLY_CERTIFIED",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def run_phase_195k_validation(*, verbose: bool = True) -> Dict[str, Any]:
    issues: List[str] = []

    cert = _load_json(CERT_PATH)
    readiness = _load_json(RESULTS / "execution_readiness.json")
    gov = _load_json(RESULTS / "governance_authorization.json")

    cert_status = str(cert.get("certification_status") or "").strip()
    cert_source = str(CERT_PATH)
    cert_generated_at = cert.get("generated_at")

    paper = bool(
        gov.get(
            "paper_execution_permitted",
            readiness.get("paper_execution_permitted", False),
        )
    )
    live = bool(
        gov.get("live_execution_permitted", readiness.get("live_execution_permitted", True))
    )

    if not CERT_PATH.is_file():
        issues.append("capital_preservation_certification.json missing")
    if cert_status not in PAPER_CERTIFICATION_STATUSES:
        issues.append(f"certification_status={cert_status or 'missing'}")
    if not paper:
        issues.append("paper_execution_permitted=false")
    if live:
        issues.append("live_execution_permitted must remain false")
    if str(readiness.get("readiness_status") or "") != "READY":
        issues.append(f"readiness_status={readiness.get('readiness_status')}")

    from services.phase_195i_lifecycle_artifact_integrity_validation import (
        run_phase_195i_validation,
    )

    i195 = run_phase_195i_validation(verbose=False)
    if not i195.get("phase_pass"):
        issues.append("phase_195i lifecycle integrity failed")
    elif i195.get("signal_lifecycle_rows") != 64 or i195.get("effective_lifecycle_rows") != 64:
        issues.append(
            f"lifecycle rows {i195.get('signal_lifecycle_rows')}/"
            f"{i195.get('effective_lifecycle_rows')} expected 64/64"
        )

    from services.phase_195h_operational_readiness_freeze import run_phase_195h_freeze

    h195 = run_phase_195h_freeze(verbose=False)
    phase_195h_status = h195.get("operational_freeze_status")
    blocking_issues = list(h195.get("blocking_issues") or [])

    if phase_195h_status == "NOT_READY":
        issues.append(f"phase_195h operational_freeze_status=NOT_READY")
    if any("certification_status=" in str(b) for b in blocking_issues):
        issues.append(f"phase_195h certification blocker: {blocking_issues}")

    open_orders = int(h195.get("open_orders_count") or 0)
    positions = int(h195.get("positions_count") or 0)
    if open_orders != 0:
        issues.append(f"open_orders_count={open_orders}")
    if positions != 0:
        issues.append(f"positions_count={positions}")

    phase_pass = not issues

    payload: Dict[str, Any] = {
        "phase": "195K",
        "timestamp": _utc_now_iso(),
        "phase_pass": phase_pass,
        "certification_status": cert_status,
        "certification_source": cert_source,
        "certification_generated_at": cert_generated_at,
        "phase_195h_status": phase_195h_status,
        "blocking_issues": blocking_issues,
        "paper_execution_permitted": paper,
        "live_execution_permitted": live,
        "issues": issues,
        "lifecycle_rows": {
            "raw": i195.get("signal_lifecycle_rows"),
            "effective": i195.get("effective_lifecycle_rows"),
        },
        "broker_snapshot": {
            "open_orders_count": open_orders,
            "positions_count": positions,
        },
    }

    _atomic_write_json(OUTPUT_JSON, payload)

    if verbose:
        print(json.dumps(payload, indent=2))

    return payload


def main() -> int:
    result = run_phase_195k_validation(verbose=True)
    return 0 if result.get("phase_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
