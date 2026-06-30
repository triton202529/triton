"""
phase_197w_create_paper_operator_approval_artifact.py
Phase 197W — Create paper-only operator approval artifact (no broker execution).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_197w_create_paper_operator_approval_artifact.json"
V197_JSON = RESULTS / "phase_197v_operator_approval_packet_review.json"
U197_JSON = RESULTS / "phase_197u_paper_runtime_governance_packet_refresh.json"
SOAK_JSON = RESULTS / "phase_197t_safe_continuous_observation_soak.json"
S197_JSON = RESULTS / "phase_197s_post_interlock_runtime_verification.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "9e5e1b1e"
PREVIOUS_PHASE = "197V"

TTL_MINUTES = 30
OPERATOR_AUTHORIZATION_PHRASE = (
    "I authorize Phase 197W to create the paper-only operator approval artifact "
    "for the current capped defensive batch. No broker orders yet."
)
REQUIRED_NEXT_PHASE = "explicit_operator_broker_execution_phase"
EXECUTION_WARNING = "This artifact does not submit orders. Broker execution requires a separate explicit operator phase."

PAPER_OPERATOR_APPROVAL_CREATED_EXECUTION_NOT_RUN = (
    "PAPER_OPERATOR_APPROVAL_CREATED_EXECUTION_NOT_RUN"
)
PAPER_OPERATOR_APPROVAL_VALID_CURRENT_EXECUTION_NOT_RUN = (
    "PAPER_OPERATOR_APPROVAL_VALID_CURRENT_EXECUTION_NOT_RUN"
)
PAPER_OPERATOR_APPROVAL_EXPIRED_CLEANUP_REQUIRED = (
    "PAPER_OPERATOR_APPROVAL_EXPIRED_CLEANUP_REQUIRED"
)
PAPER_OPERATOR_APPROVAL_INVALID_CLEANUP_REQUIRED = (
    "PAPER_OPERATOR_APPROVAL_INVALID_CLEANUP_REQUIRED"
)
PAPER_OPERATOR_APPROVAL_BLOCKED_EXISTING_ARTIFACT = (
    "PAPER_OPERATOR_APPROVAL_BLOCKED_EXISTING_ARTIFACT"
)
PAPER_OPERATOR_APPROVAL_BLOCKED_SAFETY_EVIDENCE_FAILED = (
    "PAPER_OPERATOR_APPROVAL_BLOCKED_SAFETY_EVIDENCE_FAILED"
)
PAPER_OPERATOR_APPROVAL_BLOCKED_PACKET_MISMATCH = "PAPER_OPERATOR_APPROVAL_BLOCKED_PACKET_MISMATCH"
PAPER_OPERATOR_APPROVAL_BLOCKED_OPEN_ORDERS = "PAPER_OPERATOR_APPROVAL_BLOCKED_OPEN_ORDERS"
PAPER_OPERATOR_APPROVAL_BLOCKED_LIVE_NOT_BLOCKED = (
    "PAPER_OPERATOR_APPROVAL_BLOCKED_LIVE_NOT_BLOCKED"
)
PAPER_OPERATOR_APPROVAL_BLOCKED_EXECUTION_DETECTED = (
    "PAPER_OPERATOR_APPROVAL_BLOCKED_EXECUTION_DETECTED"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _norm_symbols(symbols: Any) -> List[str]:
    return sorted({str(s).strip().upper() for s in (symbols or []) if str(s).strip()})


def _soak_execution(soak_doc: Dict[str, Any]) -> Tuple[int, int, bool]:
    exits = trims = 0
    broker_submit = False
    for cycle in soak_doc.get("cycles") or []:
        exits = max(exits, int(cycle.get("exits_executed") or 0))
        trims = max(trims, int(cycle.get("trims_executed") or 0))
        broker_submit = broker_submit or bool(cycle.get("broker_orders_submitted"))
    return exits, trims, broker_submit


def _valid_live_confirm_absent() -> Tuple[bool, str]:
    from services.live_block_classifier import scan_live_confirm_artifacts

    scan = scan_live_confirm_artifacts()
    if int(scan.get("valid_live_confirm_count") or 0) > 0:
        return False, "valid_live_confirm_present"
    return True, ""


def validate_creation_preconditions(
    *,
    v197: Dict[str, Any],
    u197: Dict[str, Any],
    soak: Dict[str, Any],
    s197: Dict[str, Any],
    packet: Dict[str, Any],
    review_symbols: List[str],
    deferred_symbols: List[str],
    batch_notional: float,
    batch_cap: float,
    open_orders_count: int,
    live_blocked: bool,
    broker_orders_submitted: bool,
    exits_executed: int,
    trims_executed: int,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if REAL_APPROVAL_PATH.is_file():
        from services.paper_operator_approval_artifact_state import (
            PAPER_APPROVAL_EXPIRED,
            PAPER_APPROVAL_VALID_CURRENT,
            classify_paper_operator_approval_artifact,
        )

        ap_state, ap_reason, _ = classify_paper_operator_approval_artifact()
        if ap_state == PAPER_APPROVAL_VALID_CURRENT:
            return PAPER_OPERATOR_APPROVAL_VALID_CURRENT_EXECUTION_NOT_RUN, [
                "valid_approval_already_exists"
            ]
        if ap_state == PAPER_APPROVAL_EXPIRED:
            return PAPER_OPERATOR_APPROVAL_EXPIRED_CLEANUP_REQUIRED, [ap_reason]
        return PAPER_OPERATOR_APPROVAL_INVALID_CLEANUP_REQUIRED, [ap_reason]

    if v197.get("status") != "PASS":
        reasons.append(f"197V_status={v197.get('status')}")
    if v197.get("review_state") != "OPERATOR_REVIEW_READY_APPROVAL_REQUIRED":
        reasons.append(f"197V_review_state={v197.get('review_state')}")
    if u197.get("status") != "PASS":
        reasons.append(f"197U_status={u197.get('status')}")
    if v197.get("governance_review_state_197n") not in (
        "GOV_REVIEW_CURRENT_OPERATOR_APPROVAL_REQUIRED",
        None,
    ):
        reasons.append(f"197N_state={v197.get('governance_review_state_197n')}")
    if v197.get("market_hours_recheck_state_197o") not in (
        "MARKET_HOURS_RECHECK_BLOCKED_BY_OPERATOR_APPROVAL",
        None,
    ):
        reasons.append(f"197O_state={v197.get('market_hours_recheck_state_197o')}")
    if soak.get("status") != "PASS":
        reasons.append(f"197T_status={soak.get('status')}")
    if s197.get("status") not in ("PASS", "PASS_SAFE_BLOCK"):
        reasons.append(f"197S_status={s197.get('status')}")

    if not live_blocked:
        return PAPER_OPERATOR_APPROVAL_BLOCKED_LIVE_NOT_BLOCKED, reasons + ["live_blocked=false"]
    if open_orders_count != 0:
        return PAPER_OPERATOR_APPROVAL_BLOCKED_OPEN_ORDERS, reasons + [
            f"open_orders_count={open_orders_count}"
        ]
    if broker_orders_submitted or exits_executed > 0 or trims_executed > 0:
        return PAPER_OPERATOR_APPROVAL_BLOCKED_EXECUTION_DETECTED, reasons + [
            f"broker={broker_orders_submitted} exits={exits_executed} trims={trims_executed}"
        ]

    live_ok, live_reason = _valid_live_confirm_absent()
    if not live_ok:
        reasons.append(live_reason)

    packet_id = str(packet.get("packet_id") or "")
    preview_hash = str(packet.get("source_preview_hash") or "")
    if packet_id != str(v197.get("operator_packet_id") or ""):
        return PAPER_OPERATOR_APPROVAL_BLOCKED_PACKET_MISMATCH, reasons + ["packet_id_mismatch"]
    if preview_hash != str(v197.get("source_preview_hash") or ""):
        return PAPER_OPERATOR_APPROVAL_BLOCKED_PACKET_MISMATCH, reasons + ["preview_hash_mismatch"]

    packet_symbols = _norm_symbols(packet.get("capped_symbols"))
    review_norm = _norm_symbols(review_symbols)
    if packet_symbols != review_norm:
        return PAPER_OPERATOR_APPROVAL_BLOCKED_PACKET_MISMATCH, reasons + [
            "capped_symbols_mismatch"
        ]

    deferred_norm = _norm_symbols(deferred_symbols)
    if set(review_norm) & set(deferred_norm):
        return PAPER_OPERATOR_APPROVAL_BLOCKED_PACKET_MISMATCH, reasons + [
            "deferred_symbols_in_approved_set"
        ]

    if batch_notional > batch_cap:
        return PAPER_OPERATOR_APPROVAL_BLOCKED_PACKET_MISMATCH, reasons + [
            f"batch_notional_exceeds_cap notional={batch_notional} cap={batch_cap}"
        ]

    if reasons:
        return PAPER_OPERATOR_APPROVAL_BLOCKED_SAFETY_EVIDENCE_FAILED, reasons

    return PAPER_OPERATOR_APPROVAL_CREATED_EXECUTION_NOT_RUN, []


def build_paper_approval_artifact(
    *,
    packet: Dict[str, Any],
    v197: Dict[str, Any],
    approved_symbols: List[str],
    deferred_symbols: List[str],
    batch_notional: float,
    batch_cap: float,
    full_handoff_notional: float,
    open_orders_count: int,
    live_blocked: bool,
) -> Dict[str, Any]:
    from services.defensive_exit_operator_paper_approval import OPERATOR_DECISION_APPROVE

    now = _utc_now()
    expires = now + timedelta(minutes=TTL_MINUTES)
    packet_id = str(packet.get("packet_id") or "")
    preview_hash = str(packet.get("source_preview_hash") or "")
    max_notional = min(float(batch_notional), float(batch_cap))

    contract = {
        "approval_id": str(uuid.uuid4()),
        "approved_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "approved_by": "operator_phase_197w",
        "approval_mode": "PAPER_ONLY",
        "approved_symbols": approved_symbols,
        "max_notional": round(max_notional, 2),
        "source_packet_id": packet_id,
        "source_preview_hash": preview_hash,
        "acknowledgement_live_blocked": True,
        "acknowledgement_no_live_execution": True,
        "acknowledgement_paper_only": True,
        "decision": OPERATOR_DECISION_APPROVE,
        "approval_status": "APPROVED",
        "approval_action": OPERATOR_DECISION_APPROVE,
        "approval_scope": "PAPER_ONLY",
        "live_authorized": False,
        "broker_execution_authorized": False,
        "raw_execution_authorized": False,
        "created_at": now.isoformat(),
        "ttl_minutes": TTL_MINUTES,
        "operator_authorization_phrase": OPERATOR_AUTHORIZATION_PHRASE,
        "packet_id": packet_id,
        "review_artifact_path": str(V197_JSON.relative_to(ROOT)).replace("\\", "/"),
        "governance_packet_path": str(OPERATOR_PACKET_JSON.relative_to(ROOT)).replace("\\", "/"),
        "capped_preview_path": str(PREVIEW_JSON.relative_to(ROOT)).replace("\\", "/"),
        "deferred_symbols": deferred_symbols,
        "batch_notional": round(float(batch_notional), 2),
        "batch_notional_cap": round(float(batch_cap), 2),
        "full_handoff_notional": round(float(full_handoff_notional), 2),
        "open_orders_count_at_approval": open_orders_count,
        "live_blocked_at_approval": live_blocked,
        "eligible_for_execution_at_approval": False,
        "required_next_phase_for_execution": REQUIRED_NEXT_PHASE,
        "warning": EXECUTION_WARNING,
        "phase": "197W",
    }
    return contract


def _run_paper_entry_dry_run() -> Tuple[str, str]:
    env = os.environ.copy()
    env["TRITON_ENABLE_PAPER_EXECUTION"] = "1"
    env["TRITON_ENABLE_AUTO_PAPER_ENTRIES"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "services.paper_entry_loop_stage", "--verbose", "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    gate = "UNKNOWN"
    reason = ""
    for line in text.splitlines():
        if "[PAPER_ENTRY_GATE]" in line:
            if "status=" in line:
                gate = line.split("status=", 1)[-1].split()[0]
            if "reason=" in line:
                reason = line.split("reason=", 1)[-1].strip()
        elif "[PAPER_ENTRY_EXECUTION_BLOCKED]" in line and "reason=" in line and not reason:
            reason = line.split("reason=", 1)[-1].strip()
    if reason.endswith(" (SAFE_BLOCK)"):
        reason = reason[: -len(" (SAFE_BLOCK)")]
    return gate, reason


def run_phase_197w_create(*, verbose: bool = True, force: bool = False) -> Dict[str, Any]:
    notes: List[str] = []
    git_clean_before = _git_clean()

    v197 = _read_json(V197_JSON)
    u197 = _read_json(U197_JSON)
    soak = _read_json(SOAK_JSON)
    s197 = _read_json(S197_JSON)
    packet = _read_json(OPERATOR_PACKET_JSON)

    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.defensive_exit_operator_paper_approval import (
        APPROVAL_STATUS_VALID,
        validate_operator_approval_contract,
        write_paper_only_defensive_exit_approval,
    )

    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=True)
    live_blocked = bool(criteria.get("live_blocked"))
    open_orders_count = int(criteria.get("open_orders_count") or 0)
    exits_executed, trims_executed, broker_orders_submitted = _soak_execution(soak)

    approved_symbols = _norm_symbols(v197.get("symbols_to_exit") or packet.get("capped_symbols"))
    deferred_symbols = _norm_symbols(v197.get("deferred_symbols") or packet.get("deferred_symbols"))
    batch_notional = float(v197.get("batch_notional") or packet.get("capped_notional") or 0.0)
    batch_cap = float(v197.get("batch_notional_cap") or packet.get("configured_cap") or 0.0)
    full_handoff = float(v197.get("batch_notional_full") or 0.0)

    final_state, block_reasons = validate_creation_preconditions(
        v197=v197,
        u197=u197,
        soak=soak,
        s197=s197,
        packet=packet,
        review_symbols=approved_symbols,
        deferred_symbols=deferred_symbols,
        batch_notional=batch_notional,
        batch_cap=batch_cap,
        open_orders_count=open_orders_count,
        live_blocked=live_blocked,
        broker_orders_submitted=broker_orders_submitted,
        exits_executed=exits_executed,
        trims_executed=trims_executed,
    )

    approval_created = False
    approval_valid = False
    approval_expired = False
    cleanup_required = False
    approval_artifact_state = None
    approval_status = None
    contract: Dict[str, Any] = {}

    existing_states = {
        PAPER_OPERATOR_APPROVAL_VALID_CURRENT_EXECUTION_NOT_RUN,
        PAPER_OPERATOR_APPROVAL_EXPIRED_CLEANUP_REQUIRED,
        PAPER_OPERATOR_APPROVAL_INVALID_CLEANUP_REQUIRED,
    }

    if final_state == PAPER_OPERATOR_APPROVAL_CREATED_EXECUTION_NOT_RUN:
        contract = build_paper_approval_artifact(
            packet=packet,
            v197=v197,
            approved_symbols=approved_symbols,
            deferred_symbols=deferred_symbols,
            batch_notional=batch_notional,
            batch_cap=batch_cap,
            full_handoff_notional=full_handoff,
            open_orders_count=open_orders_count,
            live_blocked=live_blocked,
        )
        write_paper_only_defensive_exit_approval(contract)
        approval_created = True
        ap_status, ap_reason, _ = validate_operator_approval_contract(broker_snapshot_ok=True)
        approval_valid = ap_status == APPROVAL_STATUS_VALID
        approval_status = ap_status
        if not approval_valid:
            final_state = PAPER_OPERATOR_APPROVAL_BLOCKED_PACKET_MISMATCH
            block_reasons.append(f"post_write_validation_failed={ap_reason}")
            notes.append(ap_reason)
    elif final_state in existing_states:
        from services.paper_operator_approval_artifact_state import (
            classify_paper_operator_approval_artifact,
        )

        approval_artifact_state, _, ap_details = classify_paper_operator_approval_artifact()
        contract = ap_details.get("contract") or _read_json(REAL_APPROVAL_PATH)
        approval_valid = final_state == PAPER_OPERATOR_APPROVAL_VALID_CURRENT_EXECUTION_NOT_RUN
        approval_expired = final_state == PAPER_OPERATOR_APPROVAL_EXPIRED_CLEANUP_REQUIRED
        cleanup_required = final_state in (
            PAPER_OPERATOR_APPROVAL_EXPIRED_CLEANUP_REQUIRED,
            PAPER_OPERATOR_APPROVAL_INVALID_CLEANUP_REQUIRED,
        )
        approval_status = contract.get("approval_status")
        notes.extend(block_reasons)
    else:
        notes.extend(block_reasons)

    paper_gate_status, paper_gate_reason = _run_paper_entry_dry_run()
    criteria_after = evaluate_governance_review_criteria(broker_snapshot_ok=True)

    if approval_artifact_state is None and REAL_APPROVAL_PATH.is_file():
        from services.paper_operator_approval_artifact_state import (
            classify_paper_operator_approval_artifact,
        )

        approval_artifact_state, _, _ = classify_paper_operator_approval_artifact()

    status = "PASS"
    if final_state == PAPER_OPERATOR_APPROVAL_EXPIRED_CLEANUP_REQUIRED:
        status = "PASS_SAFE_BLOCKED"
    elif final_state == PAPER_OPERATOR_APPROVAL_INVALID_CLEANUP_REQUIRED:
        status = "FAIL"
    elif final_state != PAPER_OPERATOR_APPROVAL_CREATED_EXECUTION_NOT_RUN:
        if final_state == PAPER_OPERATOR_APPROVAL_VALID_CURRENT_EXECUTION_NOT_RUN:
            status = "PASS"
        else:
            status = "FAIL"
    if approval_created and not approval_valid:
        status = "FAIL"
        notes.append("approval artifact not valid after write")
    if criteria_after.get("eligible_for_execution"):
        status = "FAIL"
        notes.append("eligible_for_execution must remain false")
    if not live_blocked:
        status = "FAIL"
    if open_orders_count != 0:
        status = "FAIL"
    if broker_orders_submitted or exits_executed != 0 or trims_executed != 0:
        status = "FAIL"

    payload: Dict[str, Any] = {
        "phase": "197W",
        "generated_at": _utc_now_iso(),
        "status": status,
        "final_state": final_state,
        "block_reasons": block_reasons,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "approval_artifact_created": approval_created,
        "approval_artifact_path": (
            str(REAL_APPROVAL_PATH.relative_to(ROOT)).replace("\\", "/")
            if REAL_APPROVAL_PATH.is_file()
            else None
        ),
        "approval_artifact_valid": approval_valid,
        "approval_artifact_state": approval_artifact_state,
        "approval_expired": approval_expired,
        "cleanup_required": cleanup_required,
        "approval_status": approval_status or contract.get("approval_status"),
        "approval_action": contract.get("approval_action"),
        "approval_scope": contract.get("approval_scope"),
        "approved_symbols": contract.get("approved_symbols") or approved_symbols,
        "deferred_symbols": deferred_symbols,
        "packet_id": contract.get("packet_id") or packet.get("packet_id"),
        "source_preview_hash": contract.get("source_preview_hash")
        or packet.get("source_preview_hash"),
        "batch_notional": batch_notional,
        "batch_notional_cap": batch_cap,
        "full_handoff_notional": full_handoff,
        "expires_at": contract.get("expires_at"),
        "ttl_minutes": TTL_MINUTES,
        "open_orders_count": open_orders_count,
        "live_blocked": live_blocked,
        "broker_orders_submitted": broker_orders_submitted,
        "exits_executed": exits_executed,
        "trims_executed": trims_executed,
        "eligible_for_execution": bool(criteria_after.get("eligible_for_execution")),
        "governance_review_state_after": criteria_after.get("governance_review_state"),
        "paper_entry_dry_run_status_after_approval": paper_gate_status,
        "paper_entry_dry_run_reason_after_approval": paper_gate_reason,
        "operator_authorization_phrase": OPERATOR_AUTHORIZATION_PHRASE,
        "required_next_phase_for_execution": REQUIRED_NEXT_PHASE,
        "warning": EXECUTION_WARNING,
        "contract": contract,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197W-APPROVAL] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197W-APPROVAL] status={status} final_state={final_state} "
            f"created={approval_created} valid={approval_valid} symbols={len(approved_symbols)}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197W create paper operator approval artifact")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--force", action="store_true", help="Reserved; creation still requires preconditions"
    )
    args = ap.parse_args(argv)
    result = run_phase_197w_create(verbose=not args.quiet, force=args.force)
    return 0 if result.get("status") in ("PASS", "PASS_SAFE_BLOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
