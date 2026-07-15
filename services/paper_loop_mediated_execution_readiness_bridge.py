"""
paper_loop_mediated_execution_readiness_bridge.py
Phase 197Z-RB — Bridge 197W/197Z/197Y paper execution authorization into readiness.
Narrow paper loop-mediated defensive exit path only; does not authorize live trading.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

BRIDGE_ACTIVE = "PAPER_LOOP_MEDIATED_EXECUTION_READINESS_BRIDGE_ACTIVE"
BRIDGE_INACTIVE = "PAPER_LOOP_MEDIATED_EXECUTION_READINESS_BRIDGE_INACTIVE"
DISPATCH_BLOCKED_APPROVAL_REQUIRED = "DISPATCH_BLOCKED_APPROVAL_REQUIRED"
DISPATCH_BLOCKED_BROKER_AUTH_REQUIRED = "DISPATCH_BLOCKED_BROKER_AUTH_REQUIRED"


def _extract_partial_bridge_context() -> Dict[str, Any]:
    """Normalize partial defensive-exit fields for readiness bridge reporting."""
    from pathlib import Path

    from services.defensive_preview_broker_state import (
        TARGET_ALREADY_COMPLETED,
        apply_broker_invalidation_to_partial_fields,
    )
    from services.partial_defensive_exit_preview import (
        PREVIEW_TYPE,
        extract_partial_approval_contract_fields,
    )

    root = Path(__file__).resolve().parents[1]
    results = root / "data" / "results"
    preview_path = results / "defensive_exit_capped_batch_preview.json"
    x197_path = results / "phase_197x_paper_broker_execution_final_dry_run.json"
    y197_path = results / "phase_197y_loop_mediated_paper_execution_gate.json"
    z197_path = results / "phase_197z_paper_broker_execution_authorization_creator.json"

    def _read(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            import json

            doc = json.loads(path.read_text(encoding="utf-8"))
            return doc if isinstance(doc, dict) else {}
        except Exception:
            return {}

    preview_doc = _read(preview_path)
    x197 = _read(x197_path)
    y197 = _read(y197_path)
    z197 = _read(z197_path)
    partial = extract_partial_approval_contract_fields(preview_doc)

    qty_map: Dict[str, Any] = {}
    if isinstance(x197.get("projected_quantities"), dict):
        qty_map = dict(x197.get("projected_quantities") or {})
    elif y197.get("eligible_qty"):
        qty_map = {str(y197.get("symbol") or "").upper(): y197.get("eligible_qty")}
    elif isinstance(z197.get("authorized_quantities"), dict):
        qty_map = dict(z197.get("authorized_quantities") or {})
    elif isinstance(partial.get("approved_quantities"), dict):
        qty_map = dict(partial.get("approved_quantities") or {})
    symbol = str(
        partial.get("symbol")
        or y197.get("symbol")
        or z197.get("symbol")
        or (x197.get("projected_symbols") or [None])[0]
        or ""
    ).upper()
    bridge_qty = int(
        qty_map.get(symbol)
        or y197.get("eligible_qty")
        or z197.get("authorized_qty")
        or partial.get("partial_qty")
        or 0
    )
    partial_for_check = dict(partial)
    if bridge_qty > int(partial_for_check.get("partial_qty") or 0):
        partial_for_check["partial_qty"] = bridge_qty
    partial_for_check = apply_broker_invalidation_to_partial_fields(
        partial_for_check, projected_qty=bridge_qty
    )
    freshness = partial_for_check.get("defensive_preview_broker_state") or {}

    if partial_for_check.get("defensive_preview_invalidated"):
        return {
            "partial_path_detected": False,
            "preview_type": None,
            "symbol": symbol or None,
            "bridge_action": None,
            "bridge_qty": 0,
            "full_qty": int(partial_for_check.get("stale_preview_original_full_qty") or 0),
            "remaining_qty_after_preview": 0,
            "full_exit_deferred": False,
            "full_exit_deferred_reason": None,
            "defensive_preview_state": partial_for_check.get("defensive_preview_state"),
            "defensive_preview_invalidated": True,
            "defensive_preview_broker_state": freshness,
            "stale_preview_original_qty": partial_for_check.get("stale_preview_original_qty"),
            "broker_qty": freshness.get("broker_qty"),
            "invalidation_reason": partial_for_check.get("invalidation_reason"),
            "target_already_completed": (
                partial_for_check.get("defensive_preview_state") == TARGET_ALREADY_COMPLETED
            ),
            "bridge_active_override": False,
        }

    full_qty = int(partial.get("full_qty") or y197.get("full_qty") or z197.get("full_qty") or 0)
    remaining = int(
        partial.get("remaining_qty_after_preview")
        or y197.get("remaining_qty_after_preview")
        or z197.get("remaining_qty_after_authorized_preview")
        or max(0, full_qty - bridge_qty)
    )
    available = bool(partial.get("partial_preview_available")) and bridge_qty > 0
    return {
        "partial_path_detected": available,
        "preview_type": PREVIEW_TYPE if available else None,
        "symbol": symbol if available else None,
        "bridge_action": "PARTIAL_EXIT" if available else None,
        "bridge_qty": bridge_qty if available else 0,
        "full_qty": full_qty if available else 0,
        "remaining_qty_after_preview": remaining if available else 0,
        "full_exit_deferred": bool(available),
        "full_exit_deferred_reason": (
            (
                partial.get("full_exit_deferred_reason")
                or y197.get("full_exit_deferred_reason")
                or z197.get("full_exit_deferred_reason")
            )
            if available
            else None
        ),
        "defensive_preview_state": freshness.get("final_state"),
        "defensive_preview_invalidated": False,
        "defensive_preview_broker_state": freshness,
        "broker_qty": freshness.get("broker_qty"),
        "target_already_completed": False,
    }


def evaluate_paper_loop_mediated_execution_readiness_bridge(
    *,
    mode: str = "paper",
    broker_snapshot_ok: bool = True,
    open_orders_count: Optional[int] = None,
    live_blocked: Optional[bool] = None,
    market_open: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    When 197W + 197Z + 197Y are valid, treat paper loop-mediated defensive execution
    readiness as satisfied without weakening live blocks or MasterExecutionGate rules.
    """
    from pathlib import Path

    from services.defensive_exit_dispatch_interlock import load_paper_approved_execution_scope
    from services.defensive_exit_governance_review_criteria import (
        collect_governance_review_evidence,
        evaluate_governance_review_criteria,
    )
    from services.live_block_classifier import evaluate_live_block_status
    from services.paper_broker_execution_authorization_state import (
        PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT,
        classify_paper_broker_execution_authorization,
    )
    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_VALID_CURRENT,
        classify_paper_operator_approval_artifact,
    )
    from services.phase_197y_loop_mediated_paper_execution_gate import (
        GATE_READY_LOOP_MEDIATED,
        evaluate_loop_mediated_paper_execution_gate,
    )
    from services.portfolio_history_sync import load_live_execution_flag

    root = Path(__file__).resolve().parents[1]
    approval_path = root / "data" / "governance" / "operator_paper_defensive_exit_approval.json"
    broker_auth_path = (
        root / "data" / "governance" / "operator_paper_broker_execution_authorization.json"
    )

    mode_s = str(mode or "paper").lower()
    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=broker_snapshot_ok)
    baseline_readiness = str(criteria.get("readiness_status") or "UNKNOWN")
    partial_ctx = _extract_partial_bridge_context()

    if live_blocked is None:
        live_blocked = bool(evaluate_live_block_status(paper_mode=True).get("live_blocked"))
    evidence = collect_governance_review_evidence(broker_snapshot_ok=broker_snapshot_ok)
    if open_orders_count is None:
        open_orders_count = int(evidence.get("open_orders_count") or 0)
    if market_open is None:
        market_open = bool(evidence.get("market_open"))

    block_reasons: List[str] = []
    bridge_active = False
    bridge_state = BRIDGE_INACTIVE
    missing_requirement = None
    dispatch = DISPATCH_BLOCKED_APPROVAL_REQUIRED
    requires_operator_approval = True
    requires_broker_authorization = True

    if mode_s != "paper":
        block_reasons.append(f"mode={mode_s}")
        missing_requirement = "paper_mode"
    elif not live_blocked:
        block_reasons.append("live_blocked=false")
        missing_requirement = "live_block"
    elif open_orders_count > 0:
        block_reasons.append(f"open_orders_count={open_orders_count}")
        missing_requirement = "open_orders_clear"
    else:
        ap_state, ap_reason, ap_details = classify_paper_operator_approval_artifact(
            broker_snapshot_ok=broker_snapshot_ok
        )
        auth_state, auth_reason, _ = classify_paper_broker_execution_authorization(
            open_orders_count=open_orders_count,
            broker_snapshot_ok=broker_snapshot_ok,
        )
        gate = evaluate_loop_mediated_paper_execution_gate(
            broker_snapshot_ok=broker_snapshot_ok,
            open_orders_count=open_orders_count,
            live_blocked=live_blocked,
            market_open=market_open,
            write_output=False,
        )
        symbols, scope_source, scope_valid = load_paper_approved_execution_scope()

        approval_present = ap_state == PAPER_APPROVAL_VALID_CURRENT and approval_path.is_file()
        broker_auth_present = (
            auth_state == PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT and broker_auth_path.is_file()
        )

        contract = ap_details.get("contract") or {}
        w_paper_only = (
            str(contract.get("approval_scope") or contract.get("approval_mode") or "").upper()
            == "PAPER_ONLY"
        )
        w_no_broker = contract.get("broker_execution_authorized") is not True
        w_no_live = contract.get("live_authorized") is not True

        checks = {
            "paper_mode": mode_s == "paper",
            "live_blocked": bool(live_blocked),
            "open_orders_zero": open_orders_count == 0,
            "197w_valid_current": ap_state == PAPER_APPROVAL_VALID_CURRENT,
            "197w_paper_only": w_paper_only if approval_present else False,
            "197w_broker_execution_not_authorized": w_no_broker if approval_present else True,
            "197w_live_not_authorized": w_no_live if approval_present else True,
            "197z_valid_current": auth_state == PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT,
            "197y_gate_ready": str(gate.get("final_state") or "") == GATE_READY_LOOP_MEDIATED,
            "197y_eligible_for_execution": bool(gate.get("eligible_for_execution")),
            "197x_scope_valid": bool(scope_valid) and len(symbols) > 0,
            "market_open": bool(market_open),
            "approval_artifact_present": approval_present,
            "broker_authorization_artifact_present": broker_auth_present,
        }

        if not approval_present:
            block_reasons.append(
                ap_reason or "operator_approval_required_for_partial_defensive_path"
            )
            missing_requirement = "operator_approval"
            dispatch = DISPATCH_BLOCKED_APPROVAL_REQUIRED
        elif not broker_auth_present:
            block_reasons.append(
                auth_reason or "broker_authorization_required_for_partial_defensive_path"
            )
            missing_requirement = "broker_authorization"
            dispatch = DISPATCH_BLOCKED_BROKER_AUTH_REQUIRED
        else:
            missing_requirement = None
            dispatch = None

        if not checks["197w_valid_current"]:
            block_reasons.append(ap_reason or f"paper_approval_state={ap_state}")
            missing_requirement = missing_requirement or "operator_approval"
            dispatch = dispatch or DISPATCH_BLOCKED_APPROVAL_REQUIRED
        if approval_present and not checks["197w_paper_only"]:
            block_reasons.append("197w_not_paper_only")
        if approval_present and not checks["197w_broker_execution_not_authorized"]:
            block_reasons.append("197w_broker_execution_authorized")
        if approval_present and not checks["197w_live_not_authorized"]:
            block_reasons.append("197w_live_authorized")
        if not checks["197z_valid_current"]:
            block_reasons.append(auth_reason or f"execution_auth_state={auth_state}")
            if approval_present:
                missing_requirement = missing_requirement or "broker_authorization"
                dispatch = dispatch or DISPATCH_BLOCKED_BROKER_AUTH_REQUIRED
        if not checks["197y_gate_ready"]:
            block_reasons.append(f"197y_final_state={gate.get('final_state')}")
        if not checks["197y_eligible_for_execution"]:
            block_reasons.append("197y_eligible_for_execution=false")
        if not checks["197x_scope_valid"]:
            block_reasons.append(f"197x_scope_invalid source={scope_source}")
        if not checks["market_open"]:
            block_reasons.append("market_closed")

        # Bridge is active only when both real artifacts exist and all checks pass.
        bridge_active = all(checks.values()) and approval_present and broker_auth_present
        if partial_ctx.get("defensive_preview_invalidated"):
            bridge_active = False
            block_reasons.append(
                str(
                    partial_ctx.get("invalidation_reason")
                    or partial_ctx.get("defensive_preview_state")
                    or "stale_defensive_preview_invalidated"
                )
            )
            missing_requirement = missing_requirement or "fresh_defensive_preview"
            if partial_ctx.get("target_already_completed"):
                dispatch = dispatch or DISPATCH_BLOCKED_APPROVAL_REQUIRED
        if bridge_active:
            bridge_state = BRIDGE_ACTIVE
            missing_requirement = None
            dispatch = "DISPATCH_READY"
        else:
            bridge_active = False
            bridge_state = BRIDGE_INACTIVE
            if not approval_present:
                dispatch = DISPATCH_BLOCKED_APPROVAL_REQUIRED
                missing_requirement = "operator_approval"
            elif not broker_auth_present:
                dispatch = DISPATCH_BLOCKED_BROKER_AUTH_REQUIRED
                missing_requirement = "broker_authorization"

        return {
            "bridge_state": bridge_state,
            "bridge_active": bridge_active,
            "paper_loop_mediated_only": bridge_active,
            "execution_readiness_bridged": bridge_active,
            "baseline_readiness_status": baseline_readiness,
            "effective_readiness_status": "READY" if bridge_active else baseline_readiness,
            "mode": mode_s,
            "live_blocked": bool(live_blocked),
            "live_execution_permitted": bool(load_live_execution_flag()),
            "open_orders_count": open_orders_count,
            "market_open": bool(market_open),
            "paper_approval_state": ap_state,
            "execution_authorization_state": auth_state,
            "197y_gate_final_state": gate.get("final_state"),
            "197y_eligible_for_execution": bool(gate.get("eligible_for_execution")),
            "approved_scope_symbols": symbols,
            "approved_scope_source": scope_source,
            "bridge_checks": checks,
            "block_reasons": block_reasons,
            "197y_gate": gate,
            "requires_operator_approval": requires_operator_approval,
            "requires_broker_authorization": requires_broker_authorization,
            "missing_requirement": missing_requirement,
            "dispatch": dispatch,
            "broker_orders_submitted": False,
            "approval_artifact_present": approval_present,
            "broker_authorization_artifact_present": broker_auth_present,
            **partial_ctx,
        }

    return {
        "bridge_state": bridge_state,
        "bridge_active": bridge_active,
        "paper_loop_mediated_only": False,
        "execution_readiness_bridged": False,
        "baseline_readiness_status": baseline_readiness,
        "effective_readiness_status": baseline_readiness,
        "mode": mode_s,
        "live_blocked": bool(live_blocked),
        "live_execution_permitted": bool(load_live_execution_flag()),
        "open_orders_count": open_orders_count,
        "market_open": bool(market_open),
        "block_reasons": block_reasons,
        "requires_operator_approval": requires_operator_approval,
        "requires_broker_authorization": requires_broker_authorization,
        "missing_requirement": missing_requirement,
        "dispatch": dispatch,
        "broker_orders_submitted": False,
        "approval_artifact_present": approval_path.is_file(),
        "broker_authorization_artifact_present": broker_auth_path.is_file(),
        **partial_ctx,
    }


def is_paper_loop_mediated_readiness_satisfied(
    readiness_status: str,
    *,
    mode: str = "paper",
    broker_snapshot_ok: bool = True,
    open_orders_count: Optional[int] = None,
    bridge: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when baseline readiness is READY or the 197Z-RB bridge is active for paper mode."""
    if str(readiness_status or "").upper() == "READY":
        return True
    if str(mode or "").lower() != "paper":
        return False
    payload = bridge or evaluate_paper_loop_mediated_execution_readiness_bridge(
        mode=mode,
        broker_snapshot_ok=broker_snapshot_ok,
        open_orders_count=open_orders_count,
    )
    return bool(payload.get("bridge_active"))
