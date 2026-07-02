"""
paper_loop_mediated_execution_readiness_bridge.py
Phase 197Z-RB — Bridge 197W/197Z/197Y paper execution authorization into readiness.
Narrow paper loop-mediated defensive exit path only; does not authorize live trading.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

BRIDGE_ACTIVE = "PAPER_LOOP_MEDIATED_EXECUTION_READINESS_BRIDGE_ACTIVE"
BRIDGE_INACTIVE = "PAPER_LOOP_MEDIATED_EXECUTION_READINESS_BRIDGE_INACTIVE"


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

    mode_s = str(mode or "paper").lower()
    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=broker_snapshot_ok)
    baseline_readiness = str(criteria.get("readiness_status") or "UNKNOWN")

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

    if mode_s != "paper":
        block_reasons.append(f"mode={mode_s}")
    elif not live_blocked:
        block_reasons.append("live_blocked=false")
    elif open_orders_count > 0:
        block_reasons.append(f"open_orders_count={open_orders_count}")
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
            "197w_paper_only": w_paper_only,
            "197w_broker_execution_not_authorized": w_no_broker,
            "197w_live_not_authorized": w_no_live,
            "197z_valid_current": auth_state == PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT,
            "197y_gate_ready": str(gate.get("final_state") or "") == GATE_READY_LOOP_MEDIATED,
            "197y_eligible_for_execution": bool(gate.get("eligible_for_execution")),
            "197x_scope_valid": bool(scope_valid) and len(symbols) > 0,
            "market_open": bool(market_open),
        }

        if not checks["197w_valid_current"]:
            block_reasons.append(ap_reason or f"paper_approval_state={ap_state}")
        if not checks["197w_paper_only"]:
            block_reasons.append("197w_not_paper_only")
        if not checks["197w_broker_execution_not_authorized"]:
            block_reasons.append("197w_broker_execution_authorized")
        if not checks["197w_live_not_authorized"]:
            block_reasons.append("197w_live_authorized")
        if not checks["197z_valid_current"]:
            block_reasons.append(auth_reason or f"execution_auth_state={auth_state}")
        if not checks["197y_gate_ready"]:
            block_reasons.append(f"197y_final_state={gate.get('final_state')}")
        if not checks["197y_eligible_for_execution"]:
            block_reasons.append("197y_eligible_for_execution=false")
        if not checks["197x_scope_valid"]:
            block_reasons.append(f"197x_scope_invalid source={scope_source}")
        if not checks["market_open"]:
            block_reasons.append("market_closed")

        bridge_active = all(checks.values())
        if bridge_active:
            bridge_state = BRIDGE_ACTIVE

        return {
            "bridge_state": bridge_state,
            "bridge_active": bridge_active,
            "paper_loop_mediated_only": bridge_active,
            "execution_readiness_bridged": bridge_active,
            "baseline_readiness_status": baseline_readiness,
            "effective_readiness_status": "READY" if bridge_active else baseline_readiness,
            "mode": mode_s,
            "live_blocked": bool(live_blocked),
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
        "open_orders_count": open_orders_count,
        "market_open": bool(market_open),
        "block_reasons": block_reasons,
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
