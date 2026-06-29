"""
phase_197n_governance_review_criteria_validation.py
Phase 197N — Validate governance review criteria for capped defensive paper exits.
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
OUTPUT_JSON = RESULTS / "phase_197n_governance_review_criteria_validation.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
PREVIOUS_COMMIT = "415abb78"
PREVIOUS_PHASE = "197M-FIX"


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


def run_phase_197n_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    status = "PASS"
    git_clean_before = _git_clean()

    from services.defensive_exit_governance_review_criteria import (
        GOV_REVIEW_BLOCKED_BY_LIVE_BLOCK_REGRESSION,
        GOV_REVIEW_BLOCKED_BY_MARKET_CLOSED,
        GOV_REVIEW_REQUIRED,
        evaluate_governance_review_criteria,
    )
    from services.live_block_classifier import (
        classify_paper_entry_gate_for_validation,
        defensive_exit_phase_validator_gates,
    )

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

    criteria = evaluate_governance_review_criteria(
        broker_snapshot_ok=snapshot_ok,
        broker_orders_submitted=False,
    )
    gates = defensive_exit_phase_validator_gates(verbose=False)
    gate_classification = classify_paper_entry_gate_for_validation(
        gates["gate_ok"], gates["gate_reason"], gates["gate_details"]
    )

    gov_state = str(criteria.get("governance_review_state") or "")
    safe_market_block = gov_state == GOV_REVIEW_BLOCKED_BY_MARKET_CLOSED
    acceptable_states = {
        GOV_REVIEW_REQUIRED,
        GOV_REVIEW_BLOCKED_BY_MARKET_CLOSED,
        "GOV_REVIEW_BLOCKED_BY_READINESS",
    }

    if not criteria.get("live_blocked"):
        status = "FAIL"
        notes.append("FAIL_LIVE_BLOCK_REGRESSION")
    if gov_state == GOV_REVIEW_BLOCKED_BY_LIVE_BLOCK_REGRESSION:
        status = "FAIL"
        notes.append("governance state live block regression")
    if not criteria.get("capped_preview_exists"):
        status = "FAIL"
        notes.append("capped preview missing")
    if not criteria.get("operator_packet_exists"):
        status = "FAIL"
        notes.append("operator packet missing")
    if REAL_APPROVAL_PATH.is_file():
        status = "FAIL"
        notes.append("real approval artifact exists")
    if criteria.get("eligible_for_execution"):
        status = "FAIL"
        notes.append("eligible_for_execution must be false")
    if not gates.get("live_block_ok"):
        status = "FAIL"
        notes.append("validator live block regression")
    if int(criteria.get("open_orders_count") or 0) != 0:
        status = "FAIL"
        notes.append("open orders must be 0")
    if gov_state not in acceptable_states and not safe_market_block:
        if gov_state not in acceptable_states:
            notes.append(f"governance_review_state={gov_state}")

    payload: Dict[str, Any] = {
        "phase": "197N",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "TRITON blocked capped defensive exits on MANAGEMENT_REVIEW_REQUIRED but lacked "
            "formal auditable criteria defining what evidence governance must inspect before "
            "any future paper-only authorization can be considered."
        ),
        "fix": (
            "defensive_exit_governance_review_criteria defines required evidence, decision "
            "states, and read-only evaluation without approving or executing."
        ),
        "governance_review_state": gov_state,
        "governance_review_reason": criteria.get("governance_review_reason"),
        "live_blocked": criteria.get("live_blocked"),
        "live_block_reason": criteria.get("live_block_reason"),
        "paper_mode": criteria.get("paper_mode"),
        "market_open": criteria.get("market_open"),
        "broker_snapshot_ok": snapshot_ok,
        "open_orders_count": criteria.get("open_orders_count"),
        "positions_count": criteria.get("positions_count"),
        "capped_preview_exists": criteria.get("capped_preview_exists"),
        "capped_symbols": criteria.get("capped_symbols"),
        "deferred_symbols": criteria.get("deferred_symbols"),
        "capped_notional": criteria.get("capped_notional"),
        "configured_cap": criteria.get("configured_cap"),
        "notional_cap_status": criteria.get("notional_cap_status"),
        "operator_packet_exists": criteria.get("operator_packet_exists"),
        "operator_packet_current": criteria.get("operator_packet_current"),
        "source_preview_hash": criteria.get("source_preview_hash"),
        "management_review_required": criteria.get("management_review_required"),
        "readiness_status": criteria.get("readiness_status"),
        "paper_entry_gate_status": gate_classification,
        "paper_entry_gate_reason": gates.get("gate_reason"),
        "real_approval_artifact_exists": criteria.get("real_approval_artifact_exists"),
        "broker_orders_submitted": False,
        "evidence_complete": criteria.get("evidence_complete"),
        "missing_evidence": criteria.get("missing_evidence"),
        "eligible_for_operator_decision": criteria.get("eligible_for_operator_decision"),
        "eligible_for_execution": criteria.get("eligible_for_execution"),
        "criteria": criteria,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197N-GOV-CRITERIA] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197N-GOV-CRITERIA] status={status} state={gov_state} "
            f"live_blocked={criteria.get('live_blocked')} gate={gate_classification}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197N governance review criteria validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197n_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
