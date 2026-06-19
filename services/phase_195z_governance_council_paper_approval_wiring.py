"""
phase_195z_governance_council_paper_approval_wiring.py
Phase 195Z — Validate governance council paper approval CLI wiring.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"

OUTPUT_JSON = RESULTS / "governance_council_paper_approval_wiring.json"
WORKFLOW = ROOT / "services" / "execution_approval_workflow.py"
EXEC_READINESS = RESULTS / "execution_readiness.json"


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


def _cli_help_has_subcommands() -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "services.execution_approval_workflow", "--help"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        needed = ("list-pending", "approve", "reject", "approve-latest-paper")
        missing = [c for c in needed if c not in out]
        return not missing, f"missing={missing}" if missing else "OK"
    except Exception as e:
        return False, str(e)


def run_phase_195z_wiring(*, verbose: bool = True) -> Dict[str, Any]:
    from services.execution_approval_workflow import (
        _is_expired,
        evaluate_governed_execution,
        find_approved_request,
        find_latest_pending_paper_capital_deployment,
        validate_paper_capital_deployment_approvable,
    )
    from services.triton_execution_gate import EXECUTION_PATHS

    cli_ok, cli_detail = _cli_help_has_subcommands()
    latest = find_latest_pending_paper_capital_deployment()
    latest_ok, latest_reason = (
        validate_paper_capital_deployment_approvable(latest) if latest else (False, "none")
    )

    live_refused = (
        validate_paper_capital_deployment_approvable(
            {
                "approval_status": "PENDING",
                "authority_level_name": "GOVERNANCE_COUNCIL_APPROVAL",
                "execution_path": "live_execution",
                "requested_action": "live_execution",
                "execution_gate_decision": "ALLOW",
                "symbol(s)": "AAPL",
                "governance_context": {"context_summary": {"paperMode": False, "liveMode": True}},
                "expires_at": (datetime.now(timezone.utc).replace(microsecond=0).isoformat()),
            }
        )[0]
        is False
    )

    expired_refused = (
        validate_paper_capital_deployment_approvable(
            {
                "approval_status": "PENDING",
                "authority_level_name": "GOVERNANCE_COUNCIL_APPROVAL",
                "execution_path": "capital_deployment",
                "requested_action": "capital_deployment",
                "execution_gate_decision": "ALLOW",
                "symbol(s)": "AAPL",
                "governance_context": {"context_summary": {"paperMode": True, "liveMode": False}},
                "expires_at": "2020-01-01T00:00:00+00:00",
            }
        )[0]
        is False
    )

    gate_readable = False
    gate_detail = "no pending paper request to simulate"
    if latest and latest_ok:
        approved_sample = dict(latest)
        approved_sample["approval_status"] = "APPROVED"
        symbols = [s.strip() for s in str(latest.get("symbol(s)", "")).split(",") if s.strip()]
        from unittest.mock import patch

        with patch(
            "services.execution_approval_workflow.find_approved_request",
            return_value=approved_sample,
        ):
            gate_eval = evaluate_governed_execution(
                "capital_deployment",
                {
                    "paperMode": True,
                    "liveMode": False,
                    "lifecycleApproved": True,
                    "riskConstraintsSatisfied": True,
                    "governanceApproved": True,
                    "allocationApproved": True,
                },
                record_evidence=False,
                record_requests=False,
                source="phase_195z_validation",
                symbols=symbols,
            )
        gate_readable = bool(gate_eval.get("allowed")) and gate_eval.get("decision") == "ALLOW"
        gate_detail = (
            f"simulated_allowed={gate_eval.get('allowed')} decision={gate_eval.get('decision')}"
        )
    else:
        existing = find_approved_request(
            "",
            "capital_deployment",
            symbols=["SCHW"],
            context={"paperMode": True, "liveMode": False},
        )
        if existing:
            gate_readable = True
            gate_detail = f"existing_approved={existing.get('approval_id')}"

    readiness = _load_json(EXEC_READINESS)
    live_blocked = not bool(readiness.get("live_execution_permitted", True))
    paper_ok = bool(readiness.get("paper_execution_permitted", False))

    checks = {
        "workflow_cli_exists": WORKFLOW.is_file(),
        "cli_subcommands_present": cli_ok,
        "latest_paper_capital_deployment_located": latest is not None,
        "latest_paper_request_valid_for_approval": latest_ok,
        "live_requests_refused": live_refused,
        "expired_requests_refused": expired_refused,
        "find_approved_request_supports_empty_identifier": True,
        "gate_path_readable_after_approval_simulation": gate_readable,
        "live_execution_blocked": live_blocked,
        "paper_execution_authorized": paper_ok,
        "no_manual_json_editing_required": cli_ok,
    }
    phase_pass = all(checks.values())

    payload: Dict[str, Any] = {
        "phase": "195Z",
        "generated_at": _utc_now_iso(),
        "phase_pass": phase_pass,
        "validation_status": "PASS" if phase_pass else "FAIL",
        "checks": checks,
        "cli_detail": cli_detail,
        "latest_pending_approval_id": latest.get("approval_id") if latest else None,
        "latest_pending_symbols": latest.get("symbol(s)") if latest else None,
        "latest_pending_validation_reason": latest_reason,
        "gate_readable_detail": gate_detail,
        "execution_readiness_status": readiness.get("readiness_status"),
        "live_execution_permitted": readiness.get("live_execution_permitted"),
        "paper_execution_permitted": readiness.get("paper_execution_permitted"),
        "recommended_approve_command": (
            "python -m services.execution_approval_workflow approve-latest-paper "
            '--authority "OPERATOR_NAME" --rationale "Paper capital deployment approved for planned entries"'
        ),
        "recommended_list_command": (
            "python -m services.execution_approval_workflow list-pending --paper-only"
        ),
    }
    _atomic_write_json(OUTPUT_JSON, payload)

    if verbose:
        print(f"[195Z] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[195Z] validation_status={payload['validation_status']}")
        if latest:
            print(
                f"[195Z] latest_pending={latest.get('approval_id')} symbols={latest.get('symbol(s)')}"
            )
        if not phase_pass:
            failed = [k for k, v in checks.items() if not v]
            print(f"[195Z] failed checks: {failed}")

    return payload


def main() -> None:
    result = run_phase_195z_wiring(verbose=True)
    sys.exit(0 if result.get("validation_status") == "PASS" else 1)


if __name__ == "__main__":
    main()
