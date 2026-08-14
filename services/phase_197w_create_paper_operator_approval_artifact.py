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
# Phase 198I-FIX — 197W writes historical evidence only (never hardened authority).
REAL_APPROVAL_PATH = (
    GOVERNANCE / "operator_paper_defensive_exit_approval_legacy_197w.json"
)
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
PAPER_OPERATOR_APPROVAL_BLOCKED_OPERATOR_CONFIRMATION_REQUIRED = (
    "PAPER_OPERATOR_APPROVAL_BLOCKED_OPERATOR_CONFIRMATION_REQUIRED"
)
PAPER_OPERATOR_APPROVAL_BLOCKED_CREATE_FLAG_REQUIRED = (
    "PAPER_OPERATOR_APPROVAL_BLOCKED_CREATE_FLAG_REQUIRED"
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
    partial_mode: bool = False,
    partial_symbols: Optional[List[str]] = None,
    dry_run: bool = False,
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

    allowed_197n = {
        "GOV_REVIEW_CURRENT_OPERATOR_APPROVAL_REQUIRED",
        None,
    }
    allowed_197o = {
        "MARKET_HOURS_RECHECK_BLOCKED_BY_OPERATOR_APPROVAL",
        None,
    }
    # Dry-run / after-hours contract validation may see market-closed review states.
    if dry_run:
        allowed_197n |= {
            "GOV_REVIEW_BLOCKED_BY_MARKET_CLOSED",
            "GOV_REVIEW_CURRENT_OPERATOR_APPROVAL_REQUIRED",
        }
        allowed_197o |= {
            "MARKET_HOURS_RECHECK_WAITING_FOR_MARKET_OPEN",
            "MARKET_HOURS_RECHECK_BLOCKED_BY_OPERATOR_APPROVAL",
        }

    if v197.get("governance_review_state_197n") not in allowed_197n:
        reasons.append(f"197N_state={v197.get('governance_review_state_197n')}")
    if v197.get("market_hours_recheck_state_197o") not in allowed_197o:
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

    review_norm = _norm_symbols(review_symbols)
    deferred_norm = _norm_symbols(deferred_symbols)
    partial_norm = _norm_symbols(partial_symbols or [])

    if partial_mode:
        expected_partial = _norm_symbols(
            packet.get("partial_approved_symbols")
            or list((packet.get("approved_quantities") or {}).keys())
            or partial_norm
        )
        if review_norm != expected_partial:
            return PAPER_OPERATOR_APPROVAL_BLOCKED_PACKET_MISMATCH, reasons + [
                f"partial_symbols_mismatch review={review_norm} expected={expected_partial}"
            ]
        # Full-exit deferred symbols may overlap the partial approval symbol.
        overlap = set(review_norm) & set(deferred_norm)
        if overlap - set(partial_norm or review_norm):
            return PAPER_OPERATOR_APPROVAL_BLOCKED_PACKET_MISMATCH, reasons + [
                "unexpected_deferred_overlap"
            ]
    else:
        packet_symbols = _norm_symbols(packet.get("capped_symbols"))
        if packet_symbols != review_norm:
            return PAPER_OPERATOR_APPROVAL_BLOCKED_PACKET_MISMATCH, reasons + [
                "capped_symbols_mismatch"
            ]
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
    partial_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from services.defensive_exit_operator_paper_approval import OPERATOR_DECISION_APPROVE
    from services.partial_defensive_exit_preview import PREVIEW_TYPE

    now = _utc_now()
    expires = now + timedelta(minutes=TTL_MINUTES)
    packet_id = str(packet.get("packet_id") or "")
    preview_hash = str(packet.get("source_preview_hash") or "")
    max_notional = min(float(batch_notional), float(batch_cap))
    partial_fields = partial_fields or {}

    contract: Dict[str, Any] = {
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
        "paper_only": True,
        "live_authorized": False,
        "live_execution_allowed": False,
        "broker_execution_authorized": False,
        "raw_execution_authorized": False,
        "created_at": now.isoformat(),
        "ttl_minutes": TTL_MINUTES,
        "operator_authorization_phrase": OPERATOR_AUTHORIZATION_PHRASE,
        "operator_confirmation": OPERATOR_AUTHORIZATION_PHRASE,
        "source": "phase_197w",
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
    if partial_fields.get("partial_preview_available"):
        symbol = str(
            partial_fields.get("symbol") or (approved_symbols[0] if approved_symbols else "")
        ).upper()
        authorized_qty = int(partial_fields.get("partial_qty") or 0)
        full_qty = int(partial_fields.get("full_qty") or 0)
        contract.update(
            {
                "preview_type": PREVIEW_TYPE,
                "approved_action": "PARTIAL_EXIT",
                "symbol": symbol,
                "authorized_qty": authorized_qty,
                "approved_quantities": partial_fields.get("approved_quantities")
                or ({symbol: authorized_qty} if symbol else {}),
                "partial_qty": authorized_qty,
                "full_qty": full_qty,
                "remaining_qty_after_preview": partial_fields.get("remaining_qty_after_preview"),
                "full_exit_deferred": True,
                "full_exit_deferred_reason": partial_fields.get("full_exit_deferred_reason"),
                "acknowledgement_partial_exit": True,
                "acknowledgement_full_exit_deferred": True,
                "acknowledgement_no_broker_authorization": True,
                "test_only": False,
            }
        )
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


def run_phase_197w_create(
    *,
    verbose: bool = True,
    force: bool = False,
    dry_run: bool = True,
    create: bool = False,
    operator_confirmation: Optional[str] = None,
) -> Dict[str, Any]:
    notes: List[str] = []
    git_clean_before = _git_clean()

    # Dry-run is the default safety posture. Real write requires --create.
    if create:
        dry_run = False
    else:
        dry_run = True

    if create:
        if not operator_confirmation or operator_confirmation != OPERATOR_AUTHORIZATION_PHRASE:
            payload = {
                "phase": "197W",
                "generated_at": _utc_now_iso(),
                "status": "PASS_SAFE_BLOCKED",
                "final_state": PAPER_OPERATOR_APPROVAL_BLOCKED_OPERATOR_CONFIRMATION_REQUIRED,
                "block_reasons": ["operator_confirmation_phrase_mismatch"],
                "dry_run": False,
                "create_requested": True,
                "approval_artifact_created": False,
                "approval_artifact_path": None,
                "broker_authorization_artifact_created": False,
                "broker_orders_submitted": False,
                "operator_authorization_phrase": OPERATOR_AUTHORIZATION_PHRASE,
                "warning": (
                    "Real approval creation requires --create and the exact "
                    "operator confirmation phrase. Dry-run remains default."
                ),
                "notes": [
                    "Provide --create --operator-confirmation with the exact 197W phrase.",
                    f"Required phrase: {OPERATOR_AUTHORIZATION_PHRASE}",
                ],
            }
            _atomic_write_json(OUTPUT_JSON, payload)
            if verbose:
                print(f"[197W-APPROVAL] wrote {OUTPUT_JSON.relative_to(ROOT)}")
                print(
                    "[197W-APPROVAL] status=PASS_SAFE_BLOCKED "
                    f"final_state={PAPER_OPERATOR_APPROVAL_BLOCKED_OPERATOR_CONFIRMATION_REQUIRED}"
                )
            return payload
    else:
        notes.append("dry_run_default_no_governance_write")

    v197 = _read_json(V197_JSON)
    u197 = _read_json(U197_JSON)
    soak = _read_json(SOAK_JSON)
    s197 = _read_json(S197_JSON)
    packet = _read_json(OPERATOR_PACKET_JSON)
    preview_doc = _read_json(PREVIEW_JSON)

    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.defensive_exit_operator_paper_approval import (
        APPROVAL_STATUS_VALID,
        validate_operator_approval_contract,
        write_paper_only_defensive_exit_approval,
    )
    from services.partial_defensive_exit_preview import extract_partial_approval_contract_fields

    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=True)
    live_blocked = bool(criteria.get("live_blocked"))
    open_orders_count = int(criteria.get("open_orders_count") or 0)
    exits_executed, trims_executed, broker_orders_submitted = _soak_execution(soak)

    partial_fields = extract_partial_approval_contract_fields(preview_doc)
    partial_mode = bool(partial_fields.get("partial_preview_available"))

    if partial_mode:
        approved_symbols = _norm_symbols(partial_fields.get("approved_symbols"))
        deferred_symbols = _norm_symbols(
            list(packet.get("deferred_symbols") or v197.get("deferred_symbols") or [])
            + ([partial_fields.get("symbol")] if partial_fields.get("symbol") else [])
        )
        batch_notional = float(partial_fields.get("estimated_notional") or 0.0)
        batch_cap = float(v197.get("batch_notional_cap") or packet.get("configured_cap") or 0.0)
        full_handoff = float(
            partial_fields.get("full_exit_notional") or v197.get("batch_notional_full") or 0.0
        )
    else:
        approved_symbols = _norm_symbols(
            v197.get("symbols_to_exit") or packet.get("capped_symbols")
        )
        deferred_symbols = _norm_symbols(
            v197.get("deferred_symbols") or packet.get("deferred_symbols")
        )
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
        partial_mode=partial_mode,
        partial_symbols=approved_symbols,
        dry_run=dry_run,
    )

    approval_created = False
    approval_valid = False
    approval_expired = False
    cleanup_required = False
    approval_artifact_state = None
    approval_status = None
    contract: Dict[str, Any] = {}
    dry_run_validated = False
    temp_contract_path: Optional[Path] = None

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
            partial_fields=partial_fields if partial_mode else None,
        )
        if dry_run:
            # Validate via temp fixture path; never leave governance approval artifact.
            RESULTS.mkdir(parents=True, exist_ok=True)
            temp_contract_path = RESULTS / "phase_197w_partial_approval_dry_run_contract.json"
            contract = dict(contract)
            contract["test_only"] = True
            contract["dry_run"] = True
            temp_contract_path.write_text(
                json.dumps(contract, indent=2, default=str), encoding="utf-8"
            )
            ap_status, ap_reason, _ = validate_operator_approval_contract(
                authority_policy="legacy_historical_review",
                broker_snapshot_ok=True,
                contract_path=temp_contract_path,
            )
            approval_valid = ap_status == APPROVAL_STATUS_VALID
            approval_status = ap_status
            dry_run_validated = approval_valid
            approval_created = False
            try:
                temp_contract_path.unlink(missing_ok=True)  # type: ignore[call-arg]
            except TypeError:
                if temp_contract_path.is_file():
                    temp_contract_path.unlink()
            except Exception:
                pass
            if REAL_APPROVAL_PATH.is_file():
                status_note = "dry_run_left_real_approval_present"
                notes.append(status_note)
                final_state = PAPER_OPERATOR_APPROVAL_BLOCKED_EXISTING_ARTIFACT
            if not approval_valid:
                final_state = PAPER_OPERATOR_APPROVAL_BLOCKED_PACKET_MISMATCH
                block_reasons.append(f"dry_run_validation_failed={ap_reason}")
                notes.append(ap_reason)
            else:
                notes.append("dry_run_partial_contract_validated_no_artifact_written")
        else:
            write_paper_only_defensive_exit_approval(contract)
            approval_created = True
            # Historical review validation only — 198I runtime dispatch requires
            # hardened 198H authority and will reject this legacy schema.
            ap_status, ap_reason, _ = validate_operator_approval_contract(
                broker_snapshot_ok=True,
                authority_policy="legacy_historical_review",
            )
            approval_valid = ap_status == APPROVAL_STATUS_VALID
            approval_status = ap_status
            notes.append(
                "198I_note:legacy_197w_artifact_not_runtime_authoritative_use_hardened_sell_arm"
            )
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
    elif (
        dry_run
        and dry_run_validated
        and final_state == PAPER_OPERATOR_APPROVAL_CREATED_EXECUTION_NOT_RUN
    ):
        status = "PASS"
        final_state = "PAPER_OPERATOR_APPROVAL_DRY_RUN_VALIDATED_EXECUTION_NOT_RUN"
    elif final_state != PAPER_OPERATOR_APPROVAL_CREATED_EXECUTION_NOT_RUN:
        if final_state == PAPER_OPERATOR_APPROVAL_VALID_CURRENT_EXECUTION_NOT_RUN:
            status = "PASS"
        else:
            status = "FAIL"
    if approval_created and not approval_valid:
        status = "FAIL"
        notes.append("approval artifact not valid after write")
    if (
        dry_run
        and not dry_run_validated
        and final_state.startswith("PAPER_OPERATOR_APPROVAL_CREATED")
    ):
        status = "FAIL"
    if criteria_after.get("eligible_for_execution"):
        status = "FAIL"
        notes.append("eligible_for_execution must remain false")
    if not live_blocked:
        status = "FAIL"
    if open_orders_count != 0:
        status = "FAIL"
    if broker_orders_submitted or exits_executed != 0 or trims_executed != 0:
        status = "FAIL"
    if dry_run and REAL_APPROVAL_PATH.is_file():
        status = "FAIL"
        notes.append("dry_run_must_not_leave_approval_artifact")

    payload: Dict[str, Any] = {
        "phase": "197W",
        "generated_at": _utc_now_iso(),
        "status": status,
        "final_state": final_state,
        "block_reasons": block_reasons,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "dry_run": dry_run,
        "dry_run_validated": dry_run_validated,
        "create_requested": create,
        "partial_mode": partial_mode,
        "preview_type": contract.get("preview_type") if partial_mode else None,
        "approved_quantities": contract.get("approved_quantities") if partial_mode else None,
        "symbol": contract.get("symbol") if partial_mode else None,
        "authorized_qty": contract.get("authorized_qty") if partial_mode else None,
        "full_qty": contract.get("full_qty") if partial_mode else None,
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
        "full_exit_deferred": bool(partial_mode),
        "expires_at": contract.get("expires_at"),
        "ttl_minutes": TTL_MINUTES,
        "open_orders_count": open_orders_count,
        "live_blocked": live_blocked,
        "broker_orders_submitted": False,
        "exits_executed": exits_executed,
        "trims_executed": trims_executed,
        "eligible_for_execution": bool(criteria_after.get("eligible_for_execution")),
        "governance_review_state_after": criteria_after.get("governance_review_state"),
        "paper_entry_dry_run_status_after_approval": paper_gate_status,
        "paper_entry_dry_run_reason_after_approval": paper_gate_reason,
        "operator_authorization_phrase": OPERATOR_AUTHORIZATION_PHRASE,
        "required_next_phase_for_execution": REQUIRED_NEXT_PHASE,
        "warning": EXECUTION_WARNING,
        "broker_authorization_artifact_created": False,
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
            f"created={approval_created} valid={approval_valid} symbols={len(approved_symbols)} "
            f"dry_run={dry_run} partial_mode={partial_mode}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Phase 197W create paper operator approval artifact. "
            "Default is dry-run (no governance write). "
            "Real creation requires --create and the exact operator confirmation phrase."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Safe default: validate contract only, do not write governance approval\n"
            "  python -m services.phase_197w_create_paper_operator_approval_artifact\n"
            "  python -m services.phase_197w_create_paper_operator_approval_artifact --dry-run\n"
            "\n"
            "  # Real create (writes data/governance/operator_paper_defensive_exit_approval.json):\n"
            "  python -m services.phase_197w_create_paper_operator_approval_artifact --create "
            "--operator-confirmation "
            '"I authorize Phase 197W to create the paper-only operator approval artifact '
            'for the current capped defensive batch. No broker orders yet."\n'
            "\n"
            "Real creation writes ONLY:\n"
            "  data/governance/operator_paper_defensive_exit_approval.json\n"
            "This does NOT create broker authorization and does NOT submit orders."
        ),
    )
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--force", action="store_true", help="Reserved; creation still requires preconditions"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate partial/paper approval contract without writing governance artifact (default)",
    )
    ap.add_argument(
        "--create",
        action="store_true",
        help=(
            "Create real approval artifact at "
            "data/governance/operator_paper_defensive_exit_approval.json "
            "(requires exact --operator-confirmation)"
        ),
    )
    ap.add_argument(
        "--operator-confirmation",
        default="",
        help="Exact 197W operator confirmation phrase (required with --create)",
    )
    args = ap.parse_args(argv)
    if args.create and args.dry_run:
        print(
            "[197W-APPROVAL] refuse: pass either --create or --dry-run, not both",
            file=sys.stderr,
        )
        return 1
    result = run_phase_197w_create(
        verbose=not args.quiet,
        force=args.force,
        dry_run=not args.create,
        create=bool(args.create),
        operator_confirmation=args.operator_confirmation or None,
    )
    return 0 if result.get("status") in ("PASS", "PASS_SAFE_BLOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
