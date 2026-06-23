"""
phase_196d_auto_paper_approval_gate_block_hotfix_validation.py
Phase 196D — Validate auto paper approval gate-block hotfix and diagnostics.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_196d_auto_paper_approval_gate_block_hotfix_validation.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _ready_readiness() -> Dict[str, Any]:
    return {
        "readiness_status": "READY",
        "live_execution_permitted": False,
        "paper_execution_permitted": True,
    }


def _sample_pending_bac() -> Dict[str, Any]:
    exp = (datetime.now(timezone.utc) + timedelta(hours=24)).replace(microsecond=0).isoformat()
    return {
        "approval_id": "eg-apr-test196d0001",
        "approval_status": "PENDING",
        "authority_level_name": "GOVERNANCE_COUNCIL_APPROVAL",
        "execution_path": "capital_deployment",
        "requested_action": "capital_deployment",
        "execution_gate_decision": "ALLOW",
        "symbol(s)": "BAC",
        "governance_context": {
            "context_summary": {
                "paperMode": True,
                "liveMode": False,
                "lifecycleApproved": True,
                "allocationApproved": True,
                "riskConstraintsSatisfied": True,
            },
        },
        "expires_at": exp,
    }


def _gate_allow() -> Dict[str, Any]:
    return {
        "allowed": True,
        "decision": "ALLOW",
        "reason": "Execution path permitted in non-live environment.",
        "path": "capital_deployment",
    }


def _gate_block_allocation() -> Dict[str, Any]:
    return {
        "allowed": False,
        "decision": "BLOCK",
        "reason": "Capital deployment blocked: portfolio allocation approval is required.",
        "path": "capital_deployment",
    }


def _run_with_mocks(
    symbols: List[str],
    *,
    readiness: Dict[str, Any] | None = None,
    market_open: bool = True,
    snap_age: float = 5.0,
    arm_valid: bool = True,
    held: set | None = None,
    open_buy: set | None = None,
    risk_flags: Dict[str, str] | None = None,
    gate_eval: Dict[str, Any] | None = None,
    find_approved_side_effect=None,
    ensure_pending_return=None,
    approve_return=None,
) -> tuple:
    from services.auto_paper_approval import attempt_auto_paper_approval

    base_env = {
        "TRITON_ENABLE_AUTO_PAPER_APPROVAL": "1",
        "TRITON_ENABLE_PAPER_EXECUTION": "1",
        "TRITON_ENABLE_AUTO_PAPER_ENTRIES": "1",
    }
    held = held if held is not None else set()
    open_buy = open_buy if open_buy is not None else set()
    risk_flags = risk_flags if risk_flags is not None else {"BAC": "OK", "SCHW": "BLOCK_NEW_BUY"}

    patches = [
        patch.dict(os.environ, base_env, clear=False),
        patch(
            "services.auto_paper_approval._load_json", return_value=readiness or _ready_readiness()
        ),
        patch("services.guard_auto.is_market_open_et", return_value=market_open),
        patch("services.auto_paper_approval._file_age_minutes", return_value=snap_age),
        patch("services.arm_mode.is_paper_arm_confirm_valid", return_value=arm_valid),
        patch("services.arm_mode.validate_mode", return_value="ASSISTED"),
        patch("services.arm_mode.load_arm_config", return_value={}),
        patch("services.paper_entry_loop_stage._load_open_buy_keys", return_value=open_buy),
        patch("services.paper_entry_loop_stage._load_held_symbols", return_value=held),
        patch("services.auto_paper_approval._load_risk_flags", return_value=risk_flags),
    ]

    if gate_eval is not None:
        patches.append(
            patch(
                "services.auto_paper_approval._evaluate_downstream_gate",
                return_value={
                    "downstream_gate_name": "execution_gate:capital_deployment",
                    "downstream_gate_status": gate_eval.get("decision"),
                    "downstream_gate_reason": gate_eval.get("reason"),
                    "downstream_block_reasons": (
                        gate_eval.get("reason", "") if gate_eval.get("decision") == "BLOCK" else ""
                    ),
                    "gate_evaluation": gate_eval,
                },
            )
        )

    if find_approved_side_effect is not None:
        patches.append(
            patch(
                "services.execution_approval_workflow.find_approved_request",
                side_effect=find_approved_side_effect,
            )
        )

    if ensure_pending_return is not None:
        patches.append(
            patch(
                "services.auto_paper_approval._ensure_pending_request",
                return_value=ensure_pending_return,
            )
        )

    if approve_return is not None:
        patches.append(
            patch(
                "services.execution_approval_workflow.approve_request", return_value=approve_return
            )
        )

    for p in patches:
        p.start()
    try:
        return attempt_auto_paper_approval(symbols, verbose=False)
    finally:
        for p in reversed(patches):
            p.stop()


def run_phase_196d_validation(*, verbose: bool = True) -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    issues: List[str] = []
    details: Dict[str, Any] = {}

    from services.auto_paper_approval import (
        _build_paper_approval_context,
        _ensure_pending_request,
        _evaluate_downstream_gate,
        _format_gate_refusal,
    )
    from services.paper_entry_loop_stage import evaluate_paper_entry_gates

    # Context includes allocationApproved (root-cause fix)
    ctx = _build_paper_approval_context()
    checks["paper_context_has_allocation_approved"] = ctx.get("allocationApproved") is True

    from services.triton_execution_gate import evaluate_execution_gate

    live_gate_raw = evaluate_execution_gate(
        "capital_deployment",
        {
            "paperMode": False,
            "liveMode": True,
            "lifecycleApproved": True,
            "allocationApproved": True,
            "riskConstraintsSatisfied": True,
            "governanceApproved": False,
        },
        record_evidence=False,
    )
    checks["live_context_gate_not_allow"] = live_gate_raw.get("decision") != "ALLOW"

    paper_gate = _evaluate_downstream_gate(["BAC"])
    checks["paper_gate_allows_bac_with_fix"] = paper_gate.get("downstream_gate_status") == "ALLOW"
    details["paper_gate_reason"] = paper_gate.get("downstream_gate_reason")

    # BAC eligible ENTRY produces request_id on PASS
    approved_record = dict(_sample_pending_bac())
    approved_record["approval_status"] = "APPROVED"
    ok, reason, det = _run_with_mocks(
        ["BAC"],
        ensure_pending_return=(
            _sample_pending_bac(),
            "CREATED",
            {
                "downstream_gate_name": "execution_gate:capital_deployment",
                "downstream_gate_status": "ALLOW",
                "downstream_gate_reason": "ok",
                "approval_request_match_status": "CREATED",
                "approval_request_created": True,
            },
        ),
        approve_return=(approved_record, "OK"),
        find_approved_side_effect=[None, approved_record],
    )
    rid = (det.get("audit") or {}).get("request_id") or (det.get("approved") or {}).get(
        "approval_id"
    )
    checks["bac_entry_produces_request_id"] = ok and bool(rid)
    checks["auto_pass_sets_governance_visible"] = (
        det.get("governance_approval_present_after_auto") is True
    )
    details["bac_request_id"] = rid

    # Real _ensure_pending_request creates request when gate allows
    with patch("services.execution_approval_workflow.create_approval_request") as mock_create:
        mock_create.return_value = _sample_pending_bac()
        with patch(
            "services.execution_approval_workflow.find_approved_request",
            return_value=None,
        ):
            with patch("services.auto_paper_approval._find_pending_for_symbols", return_value=None):
                req, status, gate_info = _ensure_pending_request(["BAC"])
    checks["ensure_pending_creates_bac_request"] = (
        req is not None
        and status == "CREATED"
        and gate_info.get("downstream_gate_status") == "ALLOW"
    )

    # SCHW blocked by risk overlay
    ok_s, reason_s, _ = _run_with_mocks(["SCHW"])
    checks["schw_blocked_risk_overlay"] = not ok_s and "RISK_OVERLAY" in reason_s

    # UNG blocked
    ok_u, reason_u, _ = _run_with_mocks(
        ["UNG"],
        risk_flags={"UNG": "BLOCK_NEW_BUY"},
    )
    checks["ung_blocked_risk_overlay"] = not ok_u and "RISK_OVERLAY" in reason_u

    # duplicate held ENTRY
    ok_h, reason_h, _ = _run_with_mocks(["BAC"], held={"BAC"})
    checks["duplicate_held_entry_refused"] = not ok_h and "ALREADY_HELD" in reason_h

    # live_execution_permitted
    ok_l, reason_l, _ = _run_with_mocks(
        ["BAC"],
        readiness={
            "readiness_status": "READY",
            "live_execution_permitted": True,
            "paper_execution_permitted": True,
        },
    )
    checks["live_execution_permitted_refused"] = not ok_l and reason_l == "LIVE_EXECUTION_PERMITTED"

    # readiness NOT_READY
    ok_r, reason_r, _ = _run_with_mocks(
        ["BAC"],
        readiness={"readiness_status": "NOT_READY", "live_execution_permitted": False},
    )
    checks["readiness_not_ready_refused"] = not ok_r and "READINESS" in reason_r

    # stale broker
    ok_b, reason_b, _ = _run_with_mocks(["BAC"], snap_age=200.0)
    checks["stale_broker_refused"] = not ok_b and "BROKER_SNAPSHOT" in reason_b

    # ARM expired
    ok_a, reason_a, _ = _run_with_mocks(["BAC"], arm_valid=False)
    checks["arm_expired_refused"] = not ok_a and "ARM" in reason_a

    # market closed
    ok_m, reason_m, _ = _run_with_mocks(["BAC"], market_open=False)
    checks["market_closed_refused"] = not ok_m and reason_m == "MARKET_CLOSED"

    # empty symbols
    ok_e, reason_e, _ = _run_with_mocks([])
    checks["empty_symbols_refused"] = not ok_e and reason_e == "NO_SYMBOLS"

    # invalid symbol
    ok_i, reason_i, _ = _run_with_mocks(["***"])
    checks["invalid_symbol_refused"] = not ok_i and "INVALID_SYMBOL" in reason_i

    # GATE_BLOCK exposes exact downstream reason
    block_info = {
        "downstream_gate_name": "execution_gate:capital_deployment",
        "downstream_gate_status": "BLOCK",
        "downstream_gate_reason": "Capital deployment blocked: portfolio allocation approval is required.",
        "downstream_block_reasons": "Capital deployment blocked: portfolio allocation approval is required.",
        "gate_evaluation": _gate_block_allocation(),
        "approval_request_match_status": "NOT_CREATED_GATE_BLOCKED",
    }
    refusal = _format_gate_refusal(block_info)
    checks["gate_block_has_exact_reason"] = (
        refusal.startswith("GATE_BLOCK:capital_deployment:")
        and "portfolio allocation approval" in refusal
        and refusal != "GATE_BLOCK"
    )

    ok_g, reason_g, det_g = _run_with_mocks(
        ["BAC"],
        ensure_pending_return=(None, refusal, block_info),
    )
    audit_g = det_g.get("audit") or {}
    checks["gate_block_audit_has_downstream_fields"] = (
        not ok_g
        and "portfolio allocation approval" in str(audit_g.get("downstream_gate_reason"))
        and audit_g.get("downstream_gate_status") == "BLOCK"
        and audit_g.get("approval_request_created") == "false"
    )

    # paper_entry_loop_stage audit downstream + visibility re-check
    approved_record = dict(_sample_pending_bac())
    approved_record["approval_status"] = "APPROVED"
    with patch.dict(
        os.environ,
        {
            "TRITON_ENABLE_PAPER_EXECUTION": "1",
            "TRITON_ENABLE_AUTO_PAPER_ENTRIES": "1",
            "TRITON_ENABLE_AUTO_PAPER_APPROVAL": "1",
        },
        clear=False,
    ):
        with patch("services.paper_entry_loop_stage._load_json", return_value=_ready_readiness()):
            with patch("services.guard_auto.is_market_open_et", return_value=True):
                with patch("services.paper_entry_loop_stage._file_age_minutes", return_value=5.0):
                    with patch(
                        "services.paper_entry_loop_stage._find_execute_trades_running",
                        return_value=(False, []),
                    ):
                        with patch(
                            "services.arm_mode.is_paper_arm_confirm_valid",
                            return_value=True,
                        ):
                            with patch("services.arm_mode.validate_mode", return_value="ASSISTED"):
                                with patch("services.arm_mode.load_arm_config", return_value={}):
                                    with patch(
                                        "services.arm_mode.resolve_arm_mutation_permissions",
                                        return_value=({"execute_trades": True}, []),
                                    ):
                                        with patch(
                                            "services.paper_entry_loop_stage._load_entry_candidates",
                                            return_value=(
                                                [
                                                    {
                                                        "ticker": "BAC",
                                                        "eligible_for_auto_entry": True,
                                                        "block_reasons": "",
                                                        "risk_flag": "OK",
                                                    },
                                                    {
                                                        "ticker": "SCHW",
                                                        "eligible_for_auto_entry": False,
                                                        "block_reasons": "RISK_OVERLAY_BLOCK",
                                                        "risk_flag": "BLOCK_NEW_BUY",
                                                    },
                                                ],
                                                None,
                                            ),
                                        ):
                                            with patch(
                                                "services.execution_approval_workflow.find_approved_request",
                                                side_effect=[None, approved_record],
                                            ):
                                                with patch(
                                                    "services.auto_paper_approval.attempt_auto_paper_approval",
                                                    return_value=(
                                                        True,
                                                        "CREATED",
                                                        {
                                                            "governance_approval_present_after_auto": True,
                                                            "gate": {
                                                                "downstream_gate_name": "execution_gate:capital_deployment",
                                                                "downstream_gate_status": "ALLOW",
                                                                "downstream_gate_reason": "ok",
                                                                "approval_request_created": True,
                                                            },
                                                            "audit": {
                                                                "request_id": "eg-apr-test196d0001",
                                                                "downstream_gate_status": "ALLOW",
                                                            },
                                                        },
                                                    ),
                                                ):
                                                    gate_ok, gate_reason, gate_det = (
                                                        evaluate_paper_entry_gates(verbose=False)
                                                    )
    checks["paper_entry_stage_rechecks_approval"] = (
        gate_ok
        and gate_reason == "OK"
        and gate_det.get("checks", {}).get("governance_approval_present_after_auto") is True
    )
    checks["paper_entry_audit_has_downstream_fields"] = bool(
        gate_det.get("downstream_gate_name") and gate_det.get("approval_request_id")
    )

    # AUTO_APPROVAL_NOT_VISIBLE_TO_EXECUTION_GATE
    with patch.dict(
        os.environ,
        {
            "TRITON_ENABLE_PAPER_EXECUTION": "1",
            "TRITON_ENABLE_AUTO_PAPER_ENTRIES": "1",
            "TRITON_ENABLE_AUTO_PAPER_APPROVAL": "1",
        },
        clear=False,
    ):
        with patch("services.paper_entry_loop_stage._load_json", return_value=_ready_readiness()):
            with patch("services.guard_auto.is_market_open_et", return_value=True):
                with patch("services.paper_entry_loop_stage._file_age_minutes", return_value=5.0):
                    with patch(
                        "services.paper_entry_loop_stage._find_execute_trades_running",
                        return_value=(False, []),
                    ):
                        with patch(
                            "services.arm_mode.is_paper_arm_confirm_valid",
                            return_value=True,
                        ):
                            with patch("services.arm_mode.validate_mode", return_value="ASSISTED"):
                                with patch("services.arm_mode.load_arm_config", return_value={}):
                                    with patch(
                                        "services.arm_mode.resolve_arm_mutation_permissions",
                                        return_value=({"execute_trades": True}, []),
                                    ):
                                        with patch(
                                            "services.paper_entry_loop_stage._load_entry_candidates",
                                            return_value=(
                                                [
                                                    {
                                                        "ticker": "BAC",
                                                        "eligible_for_auto_entry": True,
                                                    }
                                                ],
                                                None,
                                            ),
                                        ):
                                            with patch(
                                                "services.execution_approval_workflow.find_approved_request",
                                                return_value=None,
                                            ):
                                                with patch(
                                                    "services.auto_paper_approval.attempt_auto_paper_approval",
                                                    return_value=(
                                                        True,
                                                        "CREATED",
                                                        {
                                                            "governance_approval_present_after_auto": False,
                                                            "audit": {"request_id": "eg-apr-x"},
                                                        },
                                                    ),
                                                ):
                                                    vis_ok, vis_reason, _ = (
                                                        evaluate_paper_entry_gates(verbose=False)
                                                    )
    checks["not_visible_to_gate_fails_safely"] = (
        not vis_ok and vis_reason == "AUTO_APPROVAL_NOT_VISIBLE_TO_EXECUTION_GATE"
    )

    # live execution remains blocked in repo
    readiness_doc: Dict[str, Any] = {}
    try:
        readiness_doc = json.loads(
            (RESULTS / "execution_readiness.json").read_text(encoding="utf-8")
        )
    except Exception:
        pass
    checks["live_execution_permitted_false"] = not bool(
        readiness_doc.get("live_execution_permitted", True)
    )

    phase_pass = all(checks.values())
    if not phase_pass:
        issues = [k for k, v in checks.items() if not v]

    payload = {
        "phase": "196D",
        "generated_at": _utc_now_iso(),
        "phase_pass": phase_pass,
        "validation_status": "PASS" if phase_pass else "FAIL",
        "checks": checks,
        "issues": issues,
        "details": details,
    }
    _atomic_write_json(OUTPUT_JSON, payload)

    if verbose:
        print(f"[196D] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196D] validation_status={payload['validation_status']}")
        if issues:
            print(f"[196D] issues={issues}")

    return payload


def main() -> int:
    result = run_phase_196d_validation(verbose=True)
    return 0 if result.get("phase_pass") else 1


if __name__ == "__main__":
    sys.exit(main())
