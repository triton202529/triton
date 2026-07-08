"""
TRITON Execution Readiness — Phase 163A diagnostics enrichment.

Wraps governance readiness evaluation with lifecycle diagnostics attribution.
Pass/fail outcomes are unchanged; this module only adds observability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from services.lifecycle_readiness_diagnostics import (
    build_failure_details,
    evaluate_lifecycle_health,
    log_lifecycle_diagnostics,
    persist_lifecycle_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]


def finalize_readiness_with_lifecycle_diagnostics(
    readiness_doc: Dict[str, Any],
    results_dir: Path,
) -> Dict[str, Any]:
    """
    Run lifecycle diagnostics every readiness cycle and enrich failure_details
    when lifecycle is in failed_checks. Does not change pass/fail outcomes.
    """
    results_dir = Path(results_dir)
    diagnostics = evaluate_lifecycle_health(results_dir=results_dir)
    persist_lifecycle_diagnostics(diagnostics)

    out = dict(readiness_doc)
    failed = list(out.get("failed_checks") or [])
    if "lifecycle" in failed:
        out["failure_details"] = {"lifecycle": build_failure_details(diagnostics)}
        log_lifecycle_diagnostics(diagnostics)
    else:
        out.pop("failure_details", None)

    out["lifecycle_diagnostics_timestamp"] = diagnostics.get("timestamp")
    return out


def main() -> int:
    """CLI: recompute readiness with lifecycle diagnostics."""
    from services.governance_execution_readiness import (
        _iso_utc,
        _read_json,
        compute_governance_authorization,
        _atomic_write_json,
    )

    results_dir = ROOT / "data" / "results"
    policy_doc = _read_json(results_dir / "protective_action_policy.json")
    candidates_doc = _read_json(results_dir / "defensive_action_candidates.json")
    queue_doc = _read_json(results_dir / "human_approval_queue.json")
    gov_doc = _read_json(results_dir / "governance_risk_summary.json")
    cpe_doc = _read_json(results_dir / "capital_preservation_escalation.json")
    cpi_doc = _read_json(results_dir / "capital_preservation_intelligence.json")
    alerts_doc = _read_json(results_dir / "watchdog_alerts.json")
    status_doc = _read_json(results_dir / "watchdog_status.json")

    auth_doc = compute_governance_authorization(
        candidates_doc=candidates_doc,
        queue_doc=queue_doc,
        policy_doc=policy_doc,
        gov_doc=gov_doc,
        cpe_doc=cpe_doc,
    )

    from services.governance_execution_readiness import compute_execution_readiness

    readiness_doc = compute_execution_readiness(
        results_dir=results_dir,
        auth_doc=auth_doc,
        queue_doc=queue_doc,
        policy_doc=policy_doc,
        gov_doc=gov_doc,
        cpe_doc=cpe_doc,
        cpi_doc=cpi_doc,
        watchdog_status=str(status_doc.get("watchdog_status") or "OK"),
        watchdog_ts=str(status_doc.get("timestamp") or _iso_utc()),
        broker_connected=bool(status_doc.get("broker_connected")),
        broker_error=status_doc.get("broker_error"),
        active_alerts=list(alerts_doc.get("active_alerts") or []),
    )
    _atomic_write_json(readiness_doc, results_dir / "execution_readiness.json")
    print(json.dumps({"readiness": readiness_doc}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
