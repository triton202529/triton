"""
phase_197o_market_hours_governance_recheck_validation.py
Phase 197O — Validate market-hours governance observation recheck (read-only).
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
OUTPUT_JSON = RESULTS / "phase_197o_market_hours_governance_recheck_validation.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "89c76096"
PREVIOUS_PHASE = "197N"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _git_branch() -> str:
    proc = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (proc.stdout or "").strip()


def _git_clean() -> bool:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return not (proc.stdout or "").strip()


def run_phase_197o_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    status = "PASS"
    git_clean_before = _git_clean()

    from services.defensive_exit_market_hours_governance_recheck import (
        RECHECK_BLOCKED_BY_OPERATOR_APPROVAL,
        RECHECK_BLOCKED_BY_STALE_PACKET,
        RECHECK_GOV,
        RECHECK_GOV_AND_READINESS,
        RECHECK_LIVE_REGRESSION,
        RECHECK_OPEN_ORDERS,
        RECHECK_OPERATOR_APPROVAL_PRESENT_NO_EXECUTION,
        RECHECK_READINESS,
        RECHECK_READY,
        RECHECK_WAITING,
        run_market_hours_governance_recheck,
    )
    from services.defensive_exit_operator_paper_approval import (
        is_valid_paper_only_defensive_exit_approval,
    )
    from services.live_block_classifier import validator_live_block_check

    snapshot_ok = False
    try:
        snap = subprocess.run(
            [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        jstart = (snap.stdout or "").find("{")
        if jstart >= 0:
            snapshot_ok = bool(json.loads(snap.stdout[jstart:]).get("ok"))
    except Exception:
        pass

    recheck = run_market_hours_governance_recheck(
        broker_snapshot_ok=snapshot_ok,
        broker_orders_submitted=False,
    )
    live_block_ok, _ = validator_live_block_check(paper_mode=True)

    state = str(recheck.get("market_hours_recheck_state") or "")
    market_open = bool(recheck.get("market_open"))
    valid_states = {
        RECHECK_WAITING,
        RECHECK_READY,
        RECHECK_GOV,
        RECHECK_READINESS,
        RECHECK_GOV_AND_READINESS,
        RECHECK_BLOCKED_BY_STALE_PACKET,
        RECHECK_BLOCKED_BY_OPERATOR_APPROVAL,
        RECHECK_OPERATOR_APPROVAL_PRESENT_NO_EXECUTION,
        RECHECK_OPEN_ORDERS,
        RECHECK_LIVE_REGRESSION,
    }

    if not live_block_ok or not recheck.get("live_blocked"):
        status = "FAIL"
        notes.append("FAIL_LIVE_BLOCK_REGRESSION")
    if state == RECHECK_LIVE_REGRESSION:
        status = "FAIL"
        notes.append("market hours recheck live regression")
    if REAL_APPROVAL_PATH.is_file():
        valid, reason, _ = is_valid_paper_only_defensive_exit_approval()
        if not valid:
            status = "FAIL"
            notes.append(f"invalid approval artifact: {reason}")
    if recheck.get("eligible_for_execution"):
        status = "FAIL"
        notes.append("eligible_for_execution must be false")
    if int(recheck.get("open_orders_count") or 0) != 0:
        status = "FAIL"
        notes.append("open orders must be 0")
    if not recheck.get("capped_preview_exists"):
        status = "FAIL"
        notes.append("capped preview missing")
    if not recheck.get("operator_packet_exists"):
        status = "FAIL"
        notes.append("operator packet missing")
    if state not in valid_states:
        status = "FAIL"
        notes.append(f"unknown market_hours_recheck_state {state}")

    if not market_open:
        if state != RECHECK_WAITING:
            status = "FAIL"
            notes.append(f"market closed expected {RECHECK_WAITING} got {state}")
    else:
        if not recheck.get("operator_packet_current"):
            expected = RECHECK_BLOCKED_BY_STALE_PACKET
        elif recheck.get("real_approval_artifact_exists"):
            valid, _, _ = is_valid_paper_only_defensive_exit_approval()
            expected = (
                RECHECK_OPERATOR_APPROVAL_PRESENT_NO_EXECUTION
                if valid
                else RECHECK_BLOCKED_BY_OPERATOR_APPROVAL
            )
        elif not recheck.get("real_approval_artifact_exists"):
            expected = RECHECK_BLOCKED_BY_OPERATOR_APPROVAL
        else:
            gov = bool(recheck.get("governance_authorized"))
            readiness = str(recheck.get("readiness_status") or "")
            expected = RECHECK_GOV_AND_READINESS
            if not gov and readiness != "READY":
                expected = RECHECK_GOV_AND_READINESS
            elif not gov and readiness == "READY":
                expected = RECHECK_GOV
            elif gov and readiness != "READY":
                expected = RECHECK_READINESS
            elif (
                gov
                and readiness == "READY"
                and recheck.get("capped_preview_exists")
                and recheck.get("operator_packet_current")
                and not recheck.get("real_approval_artifact_exists")
            ):
                expected = RECHECK_READY
        if state != expected:
            status = "FAIL"
            notes.append(f"market open expected {expected} got {state}")

    payload: Dict[str, Any] = {
        "phase": "197O",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "197N defined governance criteria but TRITON lacked a market-hours observation "
            "gate to classify capped defensive exit state without executing."
        ),
        "fix": (
            "defensive_exit_market_hours_governance_recheck observes governance/readiness "
            "layering during market hours; after hours reports WAITING_FOR_MARKET_OPEN."
        ),
        "market_hours_recheck_state": state,
        "market_hours_recheck_reason": recheck.get("market_hours_recheck_reason"),
        "market_open": market_open,
        "live_blocked": recheck.get("live_blocked"),
        "live_block_reason": recheck.get("live_block_reason"),
        "governance_authorized": recheck.get("governance_authorized"),
        "governance_review_state": recheck.get("governance_review_state"),
        "governance_review_reason": recheck.get("governance_review_reason"),
        "management_review_required": recheck.get("management_review_required"),
        "readiness_status": recheck.get("readiness_status"),
        "paper_entry_gate_status": recheck.get("paper_entry_gate_status"),
        "paper_entry_gate_reason": recheck.get("paper_entry_gate_reason"),
        "master_execution_gate_status": recheck.get("master_execution_gate_status"),
        "broker_snapshot_ok": snapshot_ok,
        "open_orders_count": recheck.get("open_orders_count"),
        "positions_count": recheck.get("positions_count"),
        "capped_preview_exists": recheck.get("capped_preview_exists"),
        "capped_symbols": recheck.get("capped_symbols"),
        "deferred_symbols": recheck.get("deferred_symbols"),
        "capped_notional": recheck.get("capped_notional"),
        "configured_cap": recheck.get("configured_cap"),
        "notional_cap_status": recheck.get("notional_cap_status"),
        "operator_packet_exists": recheck.get("operator_packet_exists"),
        "operator_packet_current": recheck.get("operator_packet_current"),
        "real_approval_artifact_exists": recheck.get("real_approval_artifact_exists"),
        "eligible_for_operator_decision": recheck.get("eligible_for_operator_decision"),
        "eligible_for_execution": recheck.get("eligible_for_execution"),
        "broker_orders_submitted": False,
        "guard_snapshot_age_minutes": recheck.get("guard_snapshot_age_minutes"),
        "guard_snapshot_fresh": recheck.get("guard_snapshot_fresh"),
        "open_orders_snapshot_fresh": recheck.get("open_orders_snapshot_fresh"),
        "gate_freshness_note": recheck.get("gate_freshness_note"),
        "recheck": recheck,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197O-MKT-RECHECK] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197O-MKT-RECHECK] status={status} state={state} market_open={market_open} "
            f"live_blocked={recheck.get('live_blocked')}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 197O market-hours governance recheck validation"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197o_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
