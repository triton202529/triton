"""
phase_197w_fix_expired_approval_cleanup.py
Phase 197W-FIX — Guarded cleanup of expired or invalid paper-only approval artifacts.
Does not submit broker orders or enable live trading.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE = ROOT / "data" / "governance"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
OUTPUT_JSON = ROOT / "data" / "results" / "phase_197w_fix_expired_approval_cleanup.json"

CLEANUP_ALLOWED = "PAPER_APPROVAL_CLEANUP_ALLOWED"
CLEANUP_BLOCKED_SAFETY = "PAPER_APPROVAL_CLEANUP_BLOCKED_SAFETY"
CLEANUP_BLOCKED_VALID_CURRENT = "PAPER_APPROVAL_CLEANUP_BLOCKED_VALID_CURRENT"
CLEANUP_BLOCKED_ABSENT = "PAPER_APPROVAL_CLEANUP_BLOCKED_ABSENT"
CLEANUP_EXECUTED = "PAPER_APPROVAL_CLEANUP_EXECUTED"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_contract() -> Dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        return {}
    try:
        doc = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def evaluate_cleanup_eligibility(
    *, broker_snapshot_ok: bool = True
) -> Tuple[str, str, Dict[str, Any]]:
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_ABSENT,
        PAPER_APPROVAL_EXPIRED,
        PAPER_APPROVAL_VALID_CURRENT,
        INVALID_STATES,
        classify_paper_operator_approval_artifact,
        paper_approval_cleanup_required,
    )

    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=broker_snapshot_ok)
    ap_state, ap_reason, ap_details = classify_paper_operator_approval_artifact(
        broker_snapshot_ok=broker_snapshot_ok
    )
    contract = ap_details.get("contract") or _read_contract()

    details: Dict[str, Any] = {
        "approval_artifact_state": ap_state,
        "approval_reason": ap_reason,
        "artifact_path": (
            str(APPROVAL_PATH.relative_to(ROOT)).replace("\\", "/")
            if APPROVAL_PATH.is_file()
            else None
        ),
        "live_blocked": bool(criteria.get("live_blocked")),
        "open_orders_count": int(criteria.get("open_orders_count") or 0),
        "broker_orders_submitted": False,
        "exits_executed": 0,
        "trims_executed": 0,
    }

    if ap_state == PAPER_APPROVAL_ABSENT:
        return CLEANUP_BLOCKED_ABSENT, "approval_artifact_absent", details

    if ap_state == PAPER_APPROVAL_VALID_CURRENT:
        return CLEANUP_BLOCKED_VALID_CURRENT, "approval_still_valid_current", details

    if not paper_approval_cleanup_required(broker_snapshot_ok=broker_snapshot_ok):
        return CLEANUP_BLOCKED_SAFETY, f"cleanup_not_required state={ap_state}", details

    scope = str(contract.get("approval_scope") or contract.get("approval_mode") or "").upper()
    if scope and scope != "PAPER_ONLY":
        return CLEANUP_BLOCKED_SAFETY, f"approval_scope={scope}", details
    if contract.get("live_authorized") is True:
        return CLEANUP_BLOCKED_SAFETY, "live_authorized=true", details

    if not details["live_blocked"]:
        return CLEANUP_BLOCKED_SAFETY, "live_blocked=false", details
    if details["open_orders_count"] != 0:
        return CLEANUP_BLOCKED_SAFETY, f"open_orders={details['open_orders_count']}", details

    soak_path = ROOT / "data" / "results" / "phase_197t_safe_continuous_observation_soak.json"
    if soak_path.is_file():
        try:
            soak = json.loads(soak_path.read_text(encoding="utf-8"))
            for cycle in soak.get("cycles") or []:
                details["exits_executed"] = max(
                    details["exits_executed"], int(cycle.get("exits_executed") or 0)
                )
                details["trims_executed"] = max(
                    details["trims_executed"], int(cycle.get("trims_executed") or 0)
                )
                if cycle.get("broker_orders_submitted"):
                    details["broker_orders_submitted"] = True
        except Exception:
            pass

    if details["broker_orders_submitted"] or details["exits_executed"] or details["trims_executed"]:
        return CLEANUP_BLOCKED_SAFETY, "execution_detected_in_soak_evidence", details

    if ap_state in {PAPER_APPROVAL_EXPIRED} | INVALID_STATES:
        return CLEANUP_ALLOWED, ap_reason, details

    return CLEANUP_BLOCKED_SAFETY, f"unexpected_state={ap_state}", details


def run_cleanup(*, execute: bool = False, verbose: bool = True) -> Dict[str, Any]:
    state, reason, details = evaluate_cleanup_eligibility()
    removed = False
    notes: List[str] = []

    if execute and state == CLEANUP_ALLOWED and APPROVAL_PATH.is_file():
        APPROVAL_PATH.unlink()
        removed = True
        notes.append("expired_or_invalid_approval_removed")
        state = CLEANUP_EXECUTED
    elif execute and state != CLEANUP_ALLOWED:
        notes.append(f"cleanup_not_executed: {reason}")

    payload: Dict[str, Any] = {
        "phase": "197W-FIX-CLEANUP",
        "generated_at": _utc_now_iso(),
        "status": (
            "PASS"
            if state in (CLEANUP_ALLOWED, CLEANUP_EXECUTED, CLEANUP_BLOCKED_ABSENT)
            else "PASS_SAFE_BLOCKED"
        ),
        "final_state": state,
        "reason": reason,
        "cleanup_executed": removed,
        "execute_requested": execute,
        "cleanup_guidance": (
            "Run with --execute to remove expired/invalid paper-only approval at "
            "data/governance/operator_paper_defensive_exit_approval.json after safety checks pass."
        ),
        **details,
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197W-FIX-CLEANUP] state={state} reason={reason} executed={removed}")
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Guarded cleanup of expired paper approval artifact")
    ap.add_argument("--execute", action="store_true", help="Remove artifact after safety proof")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_cleanup(execute=args.execute, verbose=not args.quiet)
    return 0 if result.get("status") in ("PASS", "PASS_SAFE_BLOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
