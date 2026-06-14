"""
triton_execution_gate.py
THP-003 — Python integration layer for the centralized TRITON execution gate.

Delegates authorization decisions to lib/executionGate.js (single source of truth).
Adds audit tracing without modifying trading strategy, broker, or risk model logic.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = ROOT / "scripts" / "evaluate-execution-gate.mjs"
AUDIT_PATH = ROOT / "data" / "results" / "execution_gate_audit.csv"

EXECUTION_PATHS = {
    "SIGNAL_GENERATION": "signal_generation",
    "LIFECYCLE_AUTHORIZATION": "lifecycle_authorization",
    "PORTFOLIO_ALLOCATION": "portfolio_allocation",
    "POSITION_ENTRY": "position_entry",
    "POSITION_ADD": "position_add",
    "POSITION_TRIM": "position_trim",
    "POSITION_EXIT": "position_exit",
    "FORCED_ROTATION": "forced_rotation",
    "CAPITAL_DEPLOYMENT": "capital_deployment",
    "BROKER_SUBMISSION": "broker_submission",
    "PAPER_EXECUTION": "paper_execution",
    "LIVE_EXECUTION": "live_execution",
    "GOVERNANCE_OVERRIDE": "governance_override",
    "RISK_EMERGENCY_ACTION": "risk_emergency_action",
    "MANUAL_OPERATOR_OVERRIDE": "manual_operator_override",
}


class ExecutionGateBlocked(Exception):
    """Raised when assert_execution_allowed blocks an operation."""

    def __init__(self, evaluation: Dict[str, Any]):
        self.evaluation = evaluation
        super().__init__(str(evaluation.get("reason") or "Execution gate blocked"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _gate_disabled() -> bool:
    return _truthy_env("TRITON_DISABLE_EXECUTION_GATE")


def _paper_execution_enabled() -> bool:
    return _truthy_env("TRITON_ENABLE_PAPER_EXECUTION")


def _live_execution_authorized() -> bool:
    return _truthy_env("TRITON_ENABLE_LIVE_EXECUTION")


def _governance_approved() -> bool:
    return _truthy_env("TRITON_LIVE_GOVERNANCE_APPROVED")


def _broker_connected(broker: Any) -> bool:
    if broker is None:
        return False
    try:
        broker.get_account()
        return True
    except Exception:
        return False


def build_gate_context(
    *,
    broker: Any = None,
    mode: Optional[str] = None,
    source: Optional[str] = None,
    operator_id: Optional[str] = None,
    symbols: Optional[Sequence[str]] = None,
    execution_identifier: Optional[str] = None,
    lifecycle_approved: bool = True,
    allocation_approved: bool = True,
    risk_constraints_satisfied: bool = True,
    execution_authorized: Optional[bool] = None,
    execution_guard_approved: bool = True,
    broker_connected: Optional[bool] = None,
    duplicate_order_clear: bool = True,
    paper_mode: Optional[bool] = None,
    live_mode: Optional[bool] = None,
    live_mode_authorized: Optional[bool] = None,
    governance_approved: Optional[bool] = None,
    override_reason: Optional[str] = None,
    operational_context: Optional[str] = None,
    operator_authorized: Optional[bool] = None,
    risk_emergency_triggered: bool = False,
    emergency_action: Optional[str] = None,
    production_readiness_override: Optional[bool] = None,
    environment: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_mode = (mode or getattr(broker, "mode", "paper") or "paper").lower()
    if paper_mode is None:
        paper_mode = resolved_mode == "paper"
    if live_mode is None:
        live_mode = resolved_mode == "live"
    if execution_authorized is None:
        execution_authorized = (
            _paper_execution_enabled() if paper_mode else _live_execution_authorized()
        )
    if live_mode_authorized is None:
        live_mode_authorized = _live_execution_authorized()
    if governance_approved is None:
        governance_approved = _governance_approved()
    if broker_connected is None:
        broker_connected = _broker_connected(broker)

    ctx: Dict[str, Any] = {
        "paperMode": paper_mode,
        "liveMode": live_mode,
        "liveModeAuthorized": live_mode_authorized,
        "governanceApproved": governance_approved,
        "lifecycleApproved": lifecycle_approved,
        "allocationApproved": allocation_approved,
        "riskConstraintsSatisfied": risk_constraints_satisfied,
        "executionAuthorized": execution_authorized,
        "executionGuardApproved": execution_guard_approved,
        "brokerConnected": broker_connected,
        "duplicateOrderClear": duplicate_order_clear,
    }
    if operator_id:
        ctx["operatorId"] = operator_id
    if override_reason:
        ctx["overrideReason"] = override_reason
    if operational_context:
        ctx["operationalContext"] = operational_context
    if operator_authorized is not None:
        ctx["operatorAuthorized"] = operator_authorized
    if risk_emergency_triggered:
        ctx["riskEmergencyTriggered"] = True
    if emergency_action:
        ctx["emergencyAction"] = emergency_action
    if production_readiness_override is not None:
        ctx["productionReadinessOverride"] = production_readiness_override
    if environment:
        ctx["environment"] = environment
    if source:
        ctx["operationalContext"] = ctx.get("operationalContext") or f"source={source}"
    if symbols:
        ctx["operationalContext"] = (
            f"{ctx.get('operationalContext', '')}; symbols={','.join(symbols)}".strip("; ")
        )
    if execution_identifier:
        ctx["auditTrailId"] = execution_identifier
    return ctx


def _invoke_node_gate(
    path: str, context: Dict[str, Any], *, assert_allowed: bool = False
) -> Dict[str, Any]:
    if _gate_disabled():
        evaluation = {
            "allowed": True,
            "decision": "ALLOW",
            "reason": "Execution gate bypassed via TRITON_DISABLE_EXECUTION_GATE",
            "environment": context.get("environment", ""),
            "user_id": context.get("operatorId", ""),
            "path": path,
            "risk_level": "low",
            "requires_risk_validation": False,
            "requires_governance": False,
            "requires_operator": False,
            "timestamp": _utc_now_iso(),
        }
        return {"ok": True, "evaluation": evaluation, "explanation": evaluation["reason"]}

    if not GATE_SCRIPT.is_file():
        raise ExecutionGateBlocked(
            {
                "allowed": False,
                "decision": "BLOCK",
                "reason": f"Execution gate script missing: {GATE_SCRIPT.as_posix()}",
                "path": path,
                "timestamp": _utc_now_iso(),
            }
        )

    payload = json.dumps({"path": path, "context": context, "assert": assert_allowed})
    proc = subprocess.run(
        ["node", str(GATE_SCRIPT), payload],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if not proc.stdout.strip():
        raise ExecutionGateBlocked(
            {
                "allowed": False,
                "decision": "BLOCK",
                "reason": f"Execution gate returned no output: {proc.stderr.strip()}",
                "path": path,
                "timestamp": _utc_now_iso(),
            }
        )
    result = json.loads(proc.stdout)
    return result


def evaluate_execution_gate(
    path: str,
    context: Optional[Dict[str, Any]] = None,
    *,
    record_evidence: bool = True,
    source: str = "evaluate_execution_gate",
    broker_mode: str = "",
    symbols: Optional[Sequence[str]] = None,
    execution_identifier: Optional[str] = None,
    side: str = "",
    quantity: Optional[Any] = None,
    capital_exposure: Optional[Any] = None,
    operator_identity: Optional[str] = None,
    action_requested: Optional[str] = None,
) -> Dict[str, Any]:
    ctx = context or {}
    result = _invoke_node_gate(path, ctx, assert_allowed=False)
    evaluation = result.get("evaluation") or {}
    if record_evidence:
        _emit_gate_evidence(
            evaluation,
            source=source,
            broker_mode=broker_mode,
            symbols=symbols,
            execution_identifier=execution_identifier,
            context=ctx,
            side=side,
            quantity=quantity,
            capital_exposure=capital_exposure,
            operator_identity=operator_identity,
            action_requested=action_requested,
        )
    return evaluation


def assert_execution_allowed(
    path: str,
    context: Optional[Dict[str, Any]] = None,
    **record_kw: Any,
) -> Dict[str, Any]:
    evaluation = evaluate_execution_gate(path, context, **record_kw)
    if not evaluation.get("allowed"):
        raise ExecutionGateBlocked(evaluation)
    return evaluation


def get_execution_path_status(path: str) -> Dict[str, Any]:
    result = _invoke_node_gate(
        path,
        {"environment": os.environ.get("TRITON_ENV", "development")},
        assert_allowed=False,
    )
    return result.get("evaluation") or {}


def explain_execution_decision(evaluation: Dict[str, Any]) -> str:
    if not evaluation:
        return "No gate evaluation provided."
    lines = [
        f"Decision: {evaluation.get('decision')}",
        f"Allowed: {'yes' if evaluation.get('allowed') else 'no'}",
        f"Path: {evaluation.get('path')}",
        f"Environment: {evaluation.get('environment')}",
        f"Risk level: {evaluation.get('risk_level')}",
        f"Reason: {evaluation.get('reason')}",
    ]
    if evaluation.get("user_id"):
        lines.append(f"Operator: {evaluation.get('user_id')}")
    if evaluation.get("audit"):
        lines.append(f"Audit: {json.dumps(evaluation.get('audit'))}")
    lines.append(f"Evaluated at: {evaluation.get('timestamp')}")
    return "\n".join(lines)


def append_execution_gate_audit(
    *,
    evaluation: Dict[str, Any],
    source: str,
    broker_mode: str = "",
    symbols: Optional[Sequence[str]] = None,
    execution_identifier: Optional[str] = None,
) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": evaluation.get("timestamp") or _utc_now_iso(),
        "execution_path": evaluation.get("path", ""),
        "decision": evaluation.get("decision", ""),
        "allowed": evaluation.get("allowed", False),
        "reason": evaluation.get("reason", ""),
        "environment": evaluation.get("environment", ""),
        "operator_source": source,
        "broker_mode": broker_mode,
        "symbols": ",".join(symbols or []),
        "execution_identifier": execution_identifier or "",
    }
    write_header = not AUDIT_PATH.exists() or AUDIT_PATH.stat().st_size == 0
    with AUDIT_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _emit_gate_evidence(
    evaluation: Dict[str, Any],
    *,
    source: str,
    broker_mode: str = "",
    symbols: Optional[Sequence[str]] = None,
    execution_identifier: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    side: str = "",
    quantity: Optional[Any] = None,
    capital_exposure: Optional[Any] = None,
    operator_identity: Optional[str] = None,
    action_requested: Optional[str] = None,
) -> None:
    """Write operational audit row and institutional governance evidence (THP-004)."""
    append_execution_gate_audit(
        evaluation=evaluation,
        source=source,
        broker_mode=broker_mode,
        symbols=symbols,
        execution_identifier=execution_identifier,
    )
    try:
        from services.execution_gate_evidence_registry import record_execution_gate_decision

        record_execution_gate_decision(
            evaluation=evaluation,
            source=source,
            broker_mode=broker_mode,
            symbols=symbols,
            execution_identifier=execution_identifier,
            context=context,
            side=side,
            quantity=quantity,
            capital_exposure=capital_exposure,
            operator_identity=operator_identity,
            action_requested=action_requested,
        )
    except Exception:
        pass


def _audit_and_block(
    evaluation: Dict[str, Any],
    *,
    source: str,
    broker_mode: str = "",
    symbols: Optional[Sequence[str]] = None,
    execution_identifier: Optional[str] = None,
) -> None:
    _emit_gate_evidence(
        evaluation,
        source=source,
        broker_mode=broker_mode,
        symbols=symbols,
        execution_identifier=execution_identifier,
    )
    if not evaluation.get("allowed"):
        raise ExecutionGateBlocked(evaluation)


def map_side_to_execution_path(side: str, *, force_exit: bool = False) -> str:
    if force_exit:
        return EXECUTION_PATHS["RISK_EMERGENCY_ACTION"]
    if side.lower() == "buy":
        return EXECUTION_PATHS["POSITION_ENTRY"]
    return EXECUTION_PATHS["POSITION_EXIT"]


def map_management_action_to_path(
    action: str,
    *,
    force_exit: bool = False,
    forced_rotation: bool = False,
) -> str:
    action_u = str(action or "").strip().upper()
    if force_exit:
        return EXECUTION_PATHS["RISK_EMERGENCY_ACTION"]
    if forced_rotation or action_u in ("ROTATE_EXIT", "ROTATE", "FORCED_ROTATION"):
        return EXECUTION_PATHS["FORCED_ROTATION"]
    if action_u == "TRIM":
        return EXECUTION_PATHS["POSITION_TRIM"]
    if action_u == "ADD":
        return EXECUTION_PATHS["POSITION_ADD"]
    if action_u == "EXIT":
        return EXECUTION_PATHS["POSITION_EXIT"]
    return EXECUTION_PATHS["POSITION_EXIT"]


def enforce_pipeline_gate(
    *,
    broker: Any,
    mode: str,
    execution_path: str,
    source: str,
    symbols: Optional[Sequence[str]] = None,
    lifecycle_approved: bool = True,
    allocation_approved: bool = True,
    risk_constraints_satisfied: bool = True,
    execution_guard_approved: bool = True,
) -> Dict[str, Any]:
    """Batch-level gate for pipeline stages (execute_trades, manage_positions, place_live_orders)."""
    mode_l = mode.lower()
    pipeline_path = (
        EXECUTION_PATHS["PAPER_EXECUTION"]
        if mode_l == "paper"
        else EXECUTION_PATHS["LIVE_EXECUTION"]
    )
    ctx = build_gate_context(
        broker=broker,
        mode=mode_l,
        source=source,
        symbols=symbols,
        lifecycle_approved=lifecycle_approved,
        allocation_approved=allocation_approved,
        risk_constraints_satisfied=risk_constraints_satisfied,
        execution_guard_approved=execution_guard_approved,
    )
    mode_eval = evaluate_execution_gate(
        pipeline_path,
        ctx,
        source=f"{source}:mode",
        broker_mode=mode_l,
        symbols=symbols,
    )
    if not mode_eval.get("allowed"):
        raise ExecutionGateBlocked(mode_eval)

    stage_eval = evaluate_execution_gate(
        execution_path,
        ctx,
        source=source,
        broker_mode=mode_l,
        symbols=symbols,
    )
    if not stage_eval.get("allowed"):
        raise ExecutionGateBlocked(stage_eval)
    return stage_eval


def enforce_broker_submission_gate(
    *,
    broker: Any,
    symbol: str,
    side: str,
    execution_path: Optional[str] = None,
    client_order_id: Optional[str] = None,
    source: str = "broker_alpaca",
    force_exit: bool = False,
    gate_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Final choke point before any broker HTTP submission."""
    mode = getattr(broker, "mode", "paper")
    position_path = execution_path or map_side_to_execution_path(side, force_exit=force_exit)
    ctx = build_gate_context(
        broker=broker,
        mode=mode,
        source=source,
        symbols=[symbol],
        execution_identifier=client_order_id,
    )
    if gate_extra:
        ctx.update(gate_extra)
    if force_exit:
        ctx["riskEmergencyTriggered"] = True
        ctx["emergencyAction"] = ctx.get("emergencyAction") or "FORCE_EXIT"

    pos_eval = evaluate_execution_gate(
        position_path,
        ctx,
        source=f"{source}:position",
        broker_mode=mode,
        symbols=[symbol],
        execution_identifier=client_order_id,
        side=side,
    )
    if not pos_eval.get("allowed"):
        raise ExecutionGateBlocked(pos_eval)

    broker_eval = evaluate_execution_gate(
        EXECUTION_PATHS["BROKER_SUBMISSION"],
        ctx,
        source=f"{source}:broker",
        broker_mode=mode,
        symbols=[symbol],
        execution_identifier=client_order_id,
        side=side,
        action_requested="BROKER_SUBMIT",
    )
    if not broker_eval.get("allowed"):
        raise ExecutionGateBlocked(broker_eval)
    return broker_eval
