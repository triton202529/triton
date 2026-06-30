"""
runtime_reentry_observation_gate.py
Phase 197Q — Read-only runtime re-entry observation before continuous-loop restart.
Does not start/stop loops, create approvals, or submit orders.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"

REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
WATCHDOG_LOCK = LIVE / "watchdog_loop.lock"
CONTINUOUS_LOCK = LIVE / "continuous_loop.lock"

RESTART_COMMAND = "powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\start_triton_runtime_supervisor.ps1"

RUNTIME_REENTRY_READY_FOR_OBSERVATION = "RUNTIME_REENTRY_READY_FOR_OBSERVATION"
RUNTIME_REENTRY_WAITING_FOR_MARKET_OPEN = "RUNTIME_REENTRY_WAITING_FOR_MARKET_OPEN"
RUNTIME_REENTRY_BLOCKED_BY_LIVE_BLOCK_REGRESSION = (
    "RUNTIME_REENTRY_BLOCKED_BY_LIVE_BLOCK_REGRESSION"
)
RUNTIME_REENTRY_BLOCKED_BY_OPEN_ORDERS = "RUNTIME_REENTRY_BLOCKED_BY_OPEN_ORDERS"
RUNTIME_REENTRY_BLOCKED_BY_REAL_APPROVAL_ARTIFACT = (
    "RUNTIME_REENTRY_BLOCKED_BY_REAL_APPROVAL_ARTIFACT"
)
RUNTIME_REENTRY_BLOCKED_BY_ORDER_GUARD = "RUNTIME_REENTRY_BLOCKED_BY_ORDER_GUARD"
RUNTIME_REENTRY_BLOCKED_BY_GOVERNANCE_RECHECK_FAILURE = (
    "RUNTIME_REENTRY_BLOCKED_BY_GOVERNANCE_RECHECK_FAILURE"
)
RUNTIME_REENTRY_BLOCKED_BY_PAPER_ENTRY_UNSAFE = "RUNTIME_REENTRY_BLOCKED_BY_PAPER_ENTRY_UNSAFE"

RESTART_CONTINUOUS_FOR_OBSERVATION_ONLY = "RESTART_CONTINUOUS_FOR_OBSERVATION_ONLY"
DO_NOT_RESTART_CONTINUOUS = "DO_NOT_RESTART_CONTINUOUS"

SAFE_ORDER_GUARD_STATES = frozenset(
    {
        "ORDER_GUARD_FRESH",
        "ORDER_GUARD_STALE_BUT_BROKER_ZERO_OPEN_ORDERS",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _paper_entry_safe_for_reentry(
    gates: Dict[str, Any],
    *,
    market_open: bool,
) -> Tuple[bool, str]:
    """Paper entry is safe when execution cannot proceed (observation-only posture)."""
    gate_ok = bool(gates.get("gate_ok"))
    details = gates.get("gate_details") or {}
    failed = str(details.get("failed_check") or "")
    reason = str(gates.get("gate_reason") or "")

    if gate_ok:
        return False, "PASS_GATE_OPEN"

    if details.get("safe_block") or failed == "market_open":
        return True, "SAFE_BLOCK_MARKET_CLOSED"

    if failed in ("paper_execution_env", "auto_paper_entries_env"):
        return True, "SAFE_BLOCK_PAPER_ENV_NOT_ARMED"

    if failed == "execution_readiness" or "execution_readiness=" in reason:
        return True, "PASS_SAFE_BLOCKED"

    if "live_execution_permitted" in reason:
        return False, "FAIL_LIVE_BLOCK_REGRESSION"

    if not market_open and "market" in reason.lower():
        return True, "SAFE_BLOCK_MARKET_CLOSED"

    return False, "FAIL_PAPER_ENTRY_GATE"


def _operator_approval_status() -> str:
    packet = _read_json(OPERATOR_PACKET_JSON)
    return str(packet.get("approval_status") or "OPERATOR_APPROVAL_REQUIRED")


def _preview_fields() -> Dict[str, Any]:
    preview = _read_json(PREVIEW_JSON)
    capped = preview.get("capped_symbols") or preview.get("symbols") or []
    deferred = preview.get("deferred_symbols") or []
    if isinstance(capped, str):
        capped = [capped]
    if isinstance(deferred, str):
        deferred = [deferred]
    return {
        "capped_preview_exists": PREVIEW_JSON.is_file(),
        "capped_symbols": list(capped),
        "deferred_symbols": list(deferred),
    }


def detect_runtime_loop_status() -> Dict[str, Any]:
    """Read process and lock status for continuous/watchdog loops."""
    from services.dashboard_status_sync import audit_reconciled_runtime_locks, detect_loop_processes

    loops = detect_loop_processes()
    lock_audit = audit_reconciled_runtime_locks()
    return {
        "continuous_loop_running": bool(loops.get("continuous_process_running")),
        "watchdog_loop_running": bool(loops.get("watchdog_process_running")),
        "continuous_lock_exists": CONTINUOUS_LOCK.is_file(),
        "watchdog_lock_exists": WATCHDOG_LOCK.is_file(),
        "continuous_lock_classification": lock_audit.get("continuous_lock_classification"),
        "watchdog_lock_classification": lock_audit.get("watchdog_lock_classification"),
        "watchdog_lock_healthy": lock_audit.get("watchdog_lock_healthy"),
        "continuous_lock_healthy": lock_audit.get("continuous_lock_healthy"),
    }


def classify_runtime_reentry(
    payload: Dict[str, Any],
) -> Tuple[str, str]:
    """Classify runtime re-entry state. Returns (state, reason)."""
    if payload.get("real_approval_artifact_exists"):
        return (
            RUNTIME_REENTRY_BLOCKED_BY_REAL_APPROVAL_ARTIFACT,
            "real_approval_artifact_present",
        )

    if not payload.get("live_blocked"):
        return (
            RUNTIME_REENTRY_BLOCKED_BY_LIVE_BLOCK_REGRESSION,
            str(payload.get("live_block_reason") or "live_blocked=false"),
        )

    open_orders = int(payload.get("open_orders_count") or 0)
    if open_orders > 0:
        return RUNTIME_REENTRY_BLOCKED_BY_OPEN_ORDERS, f"open_orders_count={open_orders}"

    auth_state = str(payload.get("order_guard_authority_state") or "")
    if auth_state not in SAFE_ORDER_GUARD_STATES:
        return (
            RUNTIME_REENTRY_BLOCKED_BY_ORDER_GUARD,
            str(payload.get("order_guard_authority_reason") or auth_state or "unsafe_order_guard"),
        )

    if not payload.get("paper_entry_gate_safe"):
        return (
            RUNTIME_REENTRY_BLOCKED_BY_PAPER_ENTRY_UNSAFE,
            str(payload.get("paper_entry_gate_status") or "paper_entry_unsafe"),
        )

    recheck_state = str(payload.get("market_hours_recheck_state") or "")
    if recheck_state.endswith("LIVE_BLOCK_REGRESSION"):
        return (
            RUNTIME_REENTRY_BLOCKED_BY_GOVERNANCE_RECHECK_FAILURE,
            recheck_state,
        )

    gov_state = str(payload.get("governance_review_state") or "")
    if gov_state.endswith("LIVE_BLOCK_REGRESSION"):
        return (
            RUNTIME_REENTRY_BLOCKED_BY_GOVERNANCE_RECHECK_FAILURE,
            gov_state,
        )

    if not payload.get("broker_snapshot_ok"):
        return (
            RUNTIME_REENTRY_BLOCKED_BY_ORDER_GUARD,
            "broker_snapshot_not_ok",
        )

    if payload.get("eligible_for_execution"):
        return (
            RUNTIME_REENTRY_BLOCKED_BY_PAPER_ENTRY_UNSAFE,
            "eligible_for_execution=true",
        )

    if not payload.get("market_open"):
        return RUNTIME_REENTRY_WAITING_FOR_MARKET_OPEN, "market_closed_safe_observation"

    return RUNTIME_REENTRY_READY_FOR_OBSERVATION, "safe_for_runtime_observation_not_execution"


def restart_recommendation(state: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    """Return (recommendation, reason). Text only — never starts loops."""
    safe_states = {
        RUNTIME_REENTRY_READY_FOR_OBSERVATION,
        RUNTIME_REENTRY_WAITING_FOR_MARKET_OPEN,
    }
    if state not in safe_states:
        return DO_NOT_RESTART_CONTINUOUS, f"blocked_state={state}"

    if not payload.get("live_blocked"):
        return DO_NOT_RESTART_CONTINUOUS, "live_not_blocked"
    if int(payload.get("open_orders_count") or 0) != 0:
        return DO_NOT_RESTART_CONTINUOUS, "open_orders_present"
    if payload.get("real_approval_artifact_exists"):
        return DO_NOT_RESTART_CONTINUOUS, "real_approval_present"
    if payload.get("eligible_for_execution"):
        return DO_NOT_RESTART_CONTINUOUS, "execution_eligible"

    return (
        RESTART_CONTINUOUS_FOR_OBSERVATION_ONLY,
        "continuous_loop_may_restart_for_observation_only_not_execution",
    )


def run_runtime_reentry_observation(
    *,
    broker_snapshot_ok: bool = True,
    broker_orders_submitted: bool = False,
) -> Dict[str, Any]:
    """Aggregate runtime safety signals for re-entry observation."""
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.defensive_exit_market_hours_governance_recheck import (
        run_market_hours_governance_recheck,
    )
    from services.live_block_classifier import defensive_exit_phase_validator_gates
    from services.order_guard_snapshot_authority import evaluate_order_guard_authority

    runtime = detect_runtime_loop_status()
    criteria = evaluate_governance_review_criteria(
        broker_snapshot_ok=broker_snapshot_ok,
        broker_orders_submitted=broker_orders_submitted,
    )
    recheck = run_market_hours_governance_recheck(
        broker_snapshot_ok=broker_snapshot_ok,
        broker_orders_submitted=broker_orders_submitted,
    )
    authority = evaluate_order_guard_authority(
        broker_snapshot_ok=broker_snapshot_ok,
        open_orders_count=int(criteria.get("open_orders_count") or 0),
        refresh_guard_if_eligible=False,
        mode="paper",
    )
    gates = defensive_exit_phase_validator_gates(verbose=False)
    paper_safe, paper_safe_status = _paper_entry_safe_for_reentry(
        gates,
        market_open=bool(criteria.get("market_open")),
    )
    preview = _preview_fields()

    base = {
        "timestamp": _utc_now_iso(),
        "phase": "197Q",
        "continuous_loop_running": runtime["continuous_loop_running"],
        "watchdog_loop_running": runtime["watchdog_loop_running"],
        "continuous_lock_exists": runtime["continuous_lock_exists"],
        "watchdog_lock_exists": runtime["watchdog_lock_exists"],
        "live_blocked": bool(criteria.get("live_blocked")),
        "live_block_reason": criteria.get("live_block_reason"),
        "order_guard_authority_state": authority.get("order_guard_authority_state"),
        "order_guard_authority_reason": authority.get("order_guard_authority_reason"),
        "broker_snapshot_ok": broker_snapshot_ok,
        "open_orders_count": int(criteria.get("open_orders_count") or 0),
        "positions_count": int(criteria.get("positions_count") or 0),
        "market_open": bool(criteria.get("market_open")),
        "market_hours_recheck_state": recheck.get("market_hours_recheck_state"),
        "governance_review_state": criteria.get("governance_review_state"),
        "governance_authorized": criteria.get("governance_authorized"),
        "readiness_status": criteria.get("readiness_status"),
        "paper_entry_gate_status": paper_safe_status,
        "paper_entry_gate_reason": gates.get("gate_reason"),
        "paper_entry_gate_safe": paper_safe,
        "real_approval_artifact_exists": REAL_APPROVAL_PATH.is_file(),
        "operator_approval_status": _operator_approval_status(),
        "capped_preview_exists": preview["capped_preview_exists"],
        "capped_symbols": preview["capped_symbols"],
        "deferred_symbols": preview["deferred_symbols"],
        "eligible_for_operator_decision": recheck.get("eligible_for_operator_decision"),
        "eligible_for_execution": False,
        "broker_orders_submitted": broker_orders_submitted,
        "watchdog_lock_classification": runtime.get("watchdog_lock_classification"),
        "continuous_lock_classification": runtime.get("continuous_lock_classification"),
    }

    state, reason = classify_runtime_reentry(base)
    restart_rec, restart_reason = restart_recommendation(state, base)

    return {
        **base,
        "runtime_reentry_state": state,
        "runtime_reentry_reason": reason,
        "restart_recommendation": restart_rec,
        "restart_recommendation_reason": restart_reason,
        "restart_command": RESTART_COMMAND,
        "observation_note": (
            "197Q read-only. Does not start loops, approve, or execute. "
            "Restart command is text-only guidance."
        ),
    }
