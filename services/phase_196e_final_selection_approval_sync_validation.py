"""
phase_196e_final_selection_approval_sync_validation.py
Phase 196E — Validate planner-final symbol approval sync (not pre-planner eligible set).
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
OUTPUT_JSON = RESULTS / "phase_196e_final_selection_approval_sync_validation.json"


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


def _sample_pending(symbols: str) -> Dict[str, Any]:
    exp = (datetime.now(timezone.utc) + timedelta(hours=24)).replace(microsecond=0).isoformat()
    return {
        "approval_id": "eg-apr-test196e0001",
        "approval_status": "PENDING",
        "authority_level_name": "GOVERNANCE_COUNCIL_APPROVAL",
        "execution_path": "capital_deployment",
        "requested_action": "capital_deployment",
        "execution_gate_decision": "ALLOW",
        "symbol(s)": symbols,
        "governance_context": {
            "context_summary": {
                "paperMode": True,
                "liveMode": False,
                "lifecycleApproved": True,
                "allocationApproved": True,
            },
        },
        "expires_at": exp,
    }


def _gate_pass_mocks():
    return [
        patch.dict(
            os.environ,
            {
                "TRITON_ENABLE_PAPER_EXECUTION": "1",
                "TRITON_ENABLE_AUTO_PAPER_ENTRIES": "1",
                "TRITON_ENABLE_AUTO_PAPER_APPROVAL": "1",
            },
            clear=False,
        ),
        patch("services.paper_entry_loop_stage._load_json", return_value=_ready_readiness()),
        patch("services.guard_auto.is_market_open_et", return_value=True),
        patch("services.paper_entry_loop_stage._file_age_minutes", return_value=5.0),
        patch(
            "services.paper_entry_loop_stage._find_execute_trades_running",
            return_value=(False, []),
        ),
        patch("services.arm_mode.is_paper_arm_confirm_valid", return_value=True),
        patch("services.arm_mode.validate_mode", return_value="ASSISTED"),
        patch("services.arm_mode.load_arm_config", return_value={}),
        patch(
            "services.arm_mode.resolve_arm_mutation_permissions",
            return_value=({"execute_trades": True}, []),
        ),
    ]


def run_phase_196e_validation(*, verbose: bool = True) -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    issues: List[str] = []
    details: Dict[str, Any] = {}

    from services.auto_paper_approval import (
        approval_symbols_exact_match,
        ensure_governance_approval_for_symbols,
        parse_approved_symbols,
    )
    from services.paper_entry_loop_stage import (
        _risk_blocked_in_selection,
        _verify_final_selection_before_execute,
        evaluate_paper_entry_gates,
    )

    approved_bac = dict(_sample_pending("BAC"))
    approved_bac["approval_status"] = "APPROVED"
    approved_ung = dict(_sample_pending("UNG"))
    approved_ung["approval_status"] = "APPROVED"

    # 1. Exact match helper
    checks["exact_match_helper"] = approval_symbols_exact_match(approved_bac, ["BAC"]) and not (
        approval_symbols_exact_match(approved_bac, ["UNG"])
    )

    # 2. Governance sync approves final BAC (not eligible-only mismatch)
    with patch.dict(
        os.environ,
        {
            "TRITON_ENABLE_AUTO_PAPER_APPROVAL": "1",
            "TRITON_ENABLE_PAPER_EXECUTION": "1",
            "TRITON_ENABLE_AUTO_PAPER_ENTRIES": "1",
        },
        clear=False,
    ):
        with patch(
            "services.execution_approval_workflow.find_approved_request",
            side_effect=[None, None, approved_bac, approved_bac],
        ):
            with patch(
                "services.auto_paper_approval._ensure_pending_request",
                return_value=(
                    _sample_pending("BAC"),
                    "CREATED",
                    {"downstream_gate_status": "ALLOW", "approval_request_match_status": "CREATED"},
                ),
            ):
                with patch(
                    "services.execution_approval_workflow.approve_request",
                    return_value=(approved_bac, "OK"),
                ):
                    ok, reason, det = ensure_governance_approval_for_symbols(
                        ["BAC"],
                        context={"paperMode": True, "liveMode": False},
                    )
    checks["ensure_governance_sync_bac"] = ok and det.get("selection_approval_sync")

    # 3. BAC approved does not cover UNG final selection
    checks["bac_approval_does_not_cover_ung"] = not approval_symbols_exact_match(
        approved_bac, ["UNG"]
    )

    # 4. Risk-blocked UNG in final selection detected
    with patch(
        "services.auto_paper_approval._load_risk_flags",
        return_value={"UNG": "BLOCK_NEW_BUY", "BAC": "OK"},
    ):
        blocked = _risk_blocked_in_selection(["UNG"])
    checks["ung_risk_blocked_in_final_selection"] = blocked == ["UNG"]

    # 5. Paper entry gate uses planner preview (BAC final) not eligible-only (BAC+SCHW eligible mismatch)
    candidates = [
        {"ticker": "BAC", "eligible_for_auto_entry": True, "block_reasons": "", "risk_flag": "OK"},
        {
            "ticker": "SCHW",
            "eligible_for_auto_entry": False,
            "block_reasons": "RISK_OVERLAY_BLOCK",
            "risk_flag": "BLOCK_NEW_BUY",
        },
        {
            "ticker": "UNG",
            "eligible_for_auto_entry": False,
            "block_reasons": "RISK_OVERLAY_BLOCK",
            "risk_flag": "BLOCK_NEW_BUY",
        },
    ]
    mocks = _gate_pass_mocks()
    for m in mocks:
        m.start()
    try:
        with patch(
            "services.paper_entry_loop_stage._load_entry_candidates",
            return_value=(candidates, None),
        ):
            with patch(
                "services.paper_entry_loop_stage._preview_final_planned_symbols",
                return_value=(["BAC"], {"preview": True, "symbols": ["BAC"]}),
            ):
                with patch(
                    "services.auto_paper_approval.ensure_governance_approval_for_symbols",
                    return_value=(
                        True,
                        "OK",
                        {
                            "symbols_requested": ["BAC"],
                            "approved_symbols": ["BAC"],
                            "selection_approval_sync": True,
                            "governance_approval_present": True,
                            "approval_id": "eg-apr-test196e0001",
                        },
                    ),
                ):
                    gate_ok, gate_reason, gate_det = evaluate_paper_entry_gates(verbose=False)
    finally:
        for m in reversed(mocks):
            m.stop()
    checks["paper_entry_approves_planner_final_bac"] = (
        gate_ok
        and gate_reason == "OK"
        and gate_det.get("checks", {}).get("final_selected_symbols") == ["BAC"]
    )

    # 6. UNG in planner preview blocked before approval
    mocks = _gate_pass_mocks()
    for m in mocks:
        m.start()
    try:
        with patch(
            "services.paper_entry_loop_stage._load_entry_candidates",
            return_value=(candidates, None),
        ):
            with patch(
                "services.paper_entry_loop_stage._preview_final_planned_symbols",
                return_value=(["UNG"], {"preview": True}),
            ):
                with patch(
                    "services.paper_entry_loop_stage._risk_blocked_in_selection",
                    return_value=["UNG"],
                ):
                    gate_ok, gate_reason, _ = evaluate_paper_entry_gates(verbose=False)
    finally:
        for m in reversed(mocks):
            m.stop()
    checks["ung_final_selection_blocked"] = (
        not gate_ok and "FINAL_SELECTION_RISK_OVERLAY_BLOCK" in gate_reason
    )

    # 7. Pre-execute sync fails when approval mismatches final selection
    with patch(
        "services.paper_entry_loop_stage._preview_final_planned_symbols",
        return_value=(["UNG"], {"preview": True}),
    ):
        with patch("services.paper_entry_loop_stage._risk_blocked_in_selection", return_value=[]):
            with patch(
                "services.execution_approval_workflow.find_approved_request",
                return_value=approved_bac,
            ):
                with patch(
                    "services.auto_paper_approval.ensure_governance_approval_for_symbols",
                    return_value=(
                        False,
                        "FINAL_SELECTION_NOT_APPROVED",
                        {"approved_symbols": ["BAC"], "selection_approval_sync": False},
                    ),
                ):
                    sync_ok, sync_reason, _ = _verify_final_selection_before_execute(
                        ["BAC"],
                        verbose=False,
                    )
    checks["pre_execute_sync_fails_on_mismatch"] = (
        not sync_ok and sync_reason == "FINAL_SELECTION_NOT_APPROVED"
    )

    # 8. Pre-execute sync passes when exact match
    with patch(
        "services.paper_entry_loop_stage._preview_final_planned_symbols",
        return_value=(["BAC"], {"preview": True}),
    ):
        with patch("services.paper_entry_loop_stage._risk_blocked_in_selection", return_value=[]):
            with patch(
                "services.execution_approval_workflow.find_approved_request",
                return_value=approved_bac,
            ):
                sync_ok, sync_reason, sync_det = _verify_final_selection_before_execute(
                    ["BAC"],
                    verbose=False,
                )
    checks["pre_execute_sync_passes_exact_match"] = sync_ok and sync_det.get(
        "selection_approval_sync"
    )

    # 9. parse_approved_symbols
    checks["parse_approved_symbols"] = parse_approved_symbols(approved_bac) == ["BAC"]

    # 10. live execution remains blocked
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
        "phase": "196E",
        "generated_at": _utc_now_iso(),
        "phase_pass": phase_pass,
        "validation_status": "PASS" if phase_pass else "FAIL",
        "checks": checks,
        "issues": issues,
        "details": details,
    }
    _atomic_write_json(OUTPUT_JSON, payload)

    if verbose:
        print(f"[196E] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196E] validation_status={payload['validation_status']}")
        if issues:
            print(f"[196E] issues={issues}")

    return payload


def main() -> int:
    result = run_phase_196e_validation(verbose=True)
    return 0 if result.get("phase_pass") else 1


if __name__ == "__main__":
    sys.exit(main())
