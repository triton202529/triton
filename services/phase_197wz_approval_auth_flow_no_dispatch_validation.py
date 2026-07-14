"""
phase_197wz_approval_auth_flow_no_dispatch_validation.py

No-dispatch validation for the 197W approval + 197Z broker-auth create flow.

Proves:
- 197W defaults to dry-run (no governance write without --create)
- 197W / 197Z require explicit --create + exact operator confirmation
- Approval and broker-auth paths are aligned
- No broker orders are submitted by this validation

Does NOT:
- create persistent approval/auth artifacts by default
- submit broker orders
- enable live execution
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_197wz_approval_auth_flow_no_dispatch_validation.json"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

W197_PHRASE = (
    "I authorize Phase 197W to create the paper-only operator approval artifact "
    "for the current capped defensive batch. No broker orders yet."
)
Z197_PHRASE = (
    "I authorize the paper broker execution phase for the approved capped defensive batch. "
    "Paper only. No live trading."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _run_help(module: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def run_validation(*, verbose: bool = True) -> Dict[str, Any]:
    from services.phase_197w_create_paper_operator_approval_artifact import (
        REAL_APPROVAL_PATH,
        run_phase_197w_create,
    )
    from services.phase_197z_paper_broker_execution_authorization_creator import (
        APPROVAL_PATH as Z_APPROVAL_PATH,
        EXEC_AUTH_PATH,
        run_creator,
    )

    checks: List[Dict[str, Any]] = []
    notes: List[str] = []

    approval_before = APPROVAL_PATH.is_file()
    auth_before = AUTH_PATH.is_file()

    w_help = _run_help("services.phase_197w_create_paper_operator_approval_artifact")
    z_help = _run_help("services.phase_197z_paper_broker_execution_authorization_creator")

    checks.append(
        {
            "name": "197w_help_documents_create",
            "pass": "--create" in w_help and "--operator-confirmation" in w_help,
        }
    )
    checks.append(
        {
            "name": "197w_help_documents_governance_path",
            "pass": "operator_paper_defensive_exit_approval.json" in w_help,
        }
    )
    checks.append(
        {
            "name": "197z_help_documents_create",
            "pass": "--create" in z_help and "--operator-confirmation" in z_help,
        }
    )
    checks.append(
        {
            "name": "197w_197z_approval_path_aligned",
            "pass": REAL_APPROVAL_PATH.resolve() == Z_APPROVAL_PATH.resolve()
            and REAL_APPROVAL_PATH.resolve() == APPROVAL_PATH.resolve(),
        }
    )

    # Default invocation must remain dry-run (no write).
    w_default = run_phase_197w_create(verbose=False, create=False, dry_run=True)
    checks.append(
        {
            "name": "197w_default_is_dry_run",
            "pass": bool(w_default.get("dry_run"))
            and not bool(w_default.get("approval_artifact_created")),
            "final_state": w_default.get("final_state"),
            "status": w_default.get("status"),
        }
    )

    # Create without confirmation must block and not write.
    w_no_confirm = run_phase_197w_create(
        verbose=False,
        create=True,
        operator_confirmation="wrong phrase",
    )
    checks.append(
        {
            "name": "197w_create_without_exact_phrase_blocked",
            "pass": (
                w_no_confirm.get("status") == "PASS_SAFE_BLOCKED"
                and not bool(w_no_confirm.get("approval_artifact_created"))
                and not APPROVAL_PATH.is_file()
                if not approval_before
                else w_no_confirm.get("status") == "PASS_SAFE_BLOCKED"
            ),
            "final_state": w_no_confirm.get("final_state"),
        }
    )

    # 197Z without approval remains blocked for real create.
    z_blocked = run_creator(
        create=True,
        operator_confirmation=Z197_PHRASE,
        verbose=False,
    )
    checks.append(
        {
            "name": "197z_create_blocked_without_197w_approval_or_preconditions",
            "pass": (
                not bool(z_blocked.get("real_created"))
                and z_blocked.get("status") in ("PASS", "PASS_SAFE_BLOCKED")
            ),
            "final_state": z_blocked.get("final_state"),
            "real_created": z_blocked.get("real_created"),
        }
    )

    approval_after = APPROVAL_PATH.is_file()
    auth_after = AUTH_PATH.is_file()
    checks.append(
        {
            "name": "no_new_approval_artifact_from_validation",
            "pass": approval_before == approval_after,
            "before": approval_before,
            "after": approval_after,
        }
    )
    checks.append(
        {
            "name": "no_new_broker_auth_artifact_from_validation",
            "pass": auth_before == auth_after,
            "before": auth_before,
            "after": auth_after,
        }
    )
    checks.append(
        {
            "name": "broker_auth_path_expected",
            "pass": EXEC_AUTH_PATH.resolve() == AUTH_PATH.resolve(),
        }
    )

    failed = [c for c in checks if not c.get("pass")]
    status = "PASS" if not failed else "FAIL"
    payload: Dict[str, Any] = {
        "phase": "197WZ-NO-DISPATCH",
        "generated_at": _utc_now_iso(),
        "status": status,
        "broker_orders_submitted": False,
        "live_execution_allowed": False,
        "approval_path": str(APPROVAL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "broker_auth_path": str(AUTH_PATH.relative_to(ROOT)).replace("\\", "/"),
        "corrected_197w_create_command": (
            "python -m services.phase_197w_create_paper_operator_approval_artifact "
            f'--create --operator-confirmation "{W197_PHRASE}"'
        ),
        "corrected_197z_create_command": (
            "python -m services.phase_197z_paper_broker_execution_authorization_creator "
            f'--create --operator-confirmation "{Z197_PHRASE}"'
        ),
        "checks": checks,
        "failed_checks": [c["name"] for c in failed],
        "notes": notes
        + [
            "This validation does not create persistent approval/auth artifacts.",
            "This validation does not submit broker orders.",
            "Use the corrected create commands only when intentionally opening a paper window.",
        ],
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197WZ-NO-DISPATCH] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[197WZ-NO-DISPATCH] status={status} failed={len(failed)}")
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="No-dispatch validation for 197W/197Z approval-auth create flow"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
