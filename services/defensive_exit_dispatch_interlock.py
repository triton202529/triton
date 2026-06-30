"""
defensive_exit_dispatch_interlock.py
Phase 197R — Block defensive exit broker dispatch unless all approval gates pass.
Read-only evaluation; never creates approval artifacts or submits orders.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"

DISPATCH_READY = "DISPATCH_READY"
DISPATCH_BLOCKED_APPROVAL_REQUIRED = "DISPATCH_BLOCKED_APPROVAL_REQUIRED"
DISPATCH_BLOCKED_GOVERNANCE = "DISPATCH_BLOCKED_GOVERNANCE"
DISPATCH_BLOCKED_READINESS = "DISPATCH_BLOCKED_READINESS"
DISPATCH_BLOCKED_MARKET_CLOSED = "DISPATCH_BLOCKED_MARKET_CLOSED"
DISPATCH_BLOCKED_BATCH_NOTIONAL_CAP = "DISPATCH_BLOCKED_BATCH_NOTIONAL_CAP"
DISPATCH_BLOCKED_ORDER_GUARD = "DISPATCH_BLOCKED_ORDER_GUARD"
DISPATCH_BLOCKED_LIVE_BLOCK = "DISPATCH_BLOCKED_LIVE_BLOCK"
DISPATCH_BLOCKED_OPEN_ORDERS = "DISPATCH_BLOCKED_OPEN_ORDERS"

SAFE_ORDER_GUARD_STATES = frozenset(
    {
        "ORDER_GUARD_FRESH",
        "ORDER_GUARD_STALE_BUT_BROKER_ZERO_OPEN_ORDERS",
    }
)

DEFENSIVE_SELL_ACTIONS = frozenset({"EXIT", "ROTATE_EXIT", "TRIM", "FORCE_EXIT"})


class DispatchInterlockBlocked(Exception):
    def __init__(self, state: str, reason: str, details: Optional[Dict[str, Any]] = None):
        self.state = state
        self.reason = reason
        self.details = details or {}
        super().__init__(f"{state}: {reason}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_defensive_exit_manage_handoff(planned: Sequence[Any]) -> bool:
    """True when manage_positions planned batch includes sell-side defensive actions."""
    for item in planned:
        action = str(getattr(item, "management_action", "") or "").upper()
        side = str(getattr(item, "side", "") or "").lower()
        if action in DEFENSIVE_SELL_ACTIONS or side == "sell":
            return True
    return False


def evaluate_defensive_exit_dispatch_interlock(
    *,
    mode: str = "paper",
    broker_snapshot_ok: bool = True,
    require_market_open: bool = True,
    planned_count: int = 0,
) -> Dict[str, Any]:
    """Evaluate dispatch interlock for manage_positions defensive exit handoff."""
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.defensive_exit_governance_sync import build_defensive_exit_approval_sync
    from services.defensive_exit_operator_paper_approval import APPROVAL_STATUS_VALID
    from services.live_block_classifier import evaluate_live_block_status
    from services.order_guard_snapshot_authority import evaluate_order_guard_authority

    criteria = evaluate_governance_review_criteria(
        broker_snapshot_ok=broker_snapshot_ok,
        broker_orders_submitted=False,
    )
    sync = build_defensive_exit_approval_sync(cap_context="runtime")
    authority = evaluate_order_guard_authority(
        broker_snapshot_ok=broker_snapshot_ok,
        open_orders_count=int(criteria.get("open_orders_count") or 0),
        refresh_guard_if_eligible=False,
        mode="paper",
    )
    live_payload = evaluate_live_block_status(paper_mode=True)

    market_open = bool(criteria.get("market_open"))
    governance_authorized = bool(criteria.get("governance_authorized"))
    readiness_status = str(criteria.get("readiness_status") or "")
    real_approval_exists = REAL_APPROVAL_PATH.is_file()
    operator_status = str(
        sync.get("operator_paper_approval_status") or "OPERATOR_APPROVAL_REQUIRED"
    )
    open_orders = int(criteria.get("open_orders_count") or 0)
    auth_state = str(authority.get("order_guard_authority_state") or "")

    preview_doc = {}
    if PREVIEW_JSON.is_file():
        try:
            import json

            preview_doc = json.loads(PREVIEW_JSON.read_text(encoding="utf-8"))
        except Exception:
            preview_doc = {}
    batch_notional = float(preview_doc.get("batch_notional_estimate") or 0.0)
    configured_cap = float(
        preview_doc.get("configured_batch_cap_manage_positions")
        or preview_doc.get("configured_batch_cap_effective")
        or 50000.0
    )
    batch_notional_status = "WITHIN_CAP"
    if batch_notional > configured_cap > 0:
        batch_notional_status = "EXCEEDS_CAP"

    base = {
        "timestamp": _utc_now_iso(),
        "phase": "197R",
        "mode": mode,
        "planned_count": planned_count,
        "market_open": market_open,
        "live_blocked": bool(live_payload.get("live_blocked")),
        "live_block_reason": live_payload.get("live_block_reason"),
        "order_guard_authority_state": auth_state,
        "open_orders_count": open_orders,
        "real_approval_artifact_exists": real_approval_exists,
        "operator_approval_status": operator_status,
        "governance_authorized": governance_authorized,
        "readiness_status": readiness_status,
        "batch_notional": round(batch_notional, 2),
        "batch_notional_cap": configured_cap,
        "batch_notional_status": batch_notional_status,
        "eligible_for_execution": False,
        "broker_orders_submitted": False,
    }

    if not live_payload.get("live_blocked"):
        state, reason = DISPATCH_BLOCKED_LIVE_BLOCK, str(
            live_payload.get("live_block_reason") or "live_blocked=false"
        )
    elif open_orders > 0:
        state, reason = DISPATCH_BLOCKED_OPEN_ORDERS, f"open_orders_count={open_orders}"
    elif auth_state not in SAFE_ORDER_GUARD_STATES:
        state, reason = (
            DISPATCH_BLOCKED_ORDER_GUARD,
            str(authority.get("order_guard_authority_reason") or auth_state),
        )
    elif not real_approval_exists or operator_status != APPROVAL_STATUS_VALID:
        state, reason = DISPATCH_BLOCKED_APPROVAL_REQUIRED, (
            "real_approval_artifact_absent"
            if not real_approval_exists
            else f"operator_approval_status={operator_status}"
        )
    elif not governance_authorized:
        state, reason = (
            DISPATCH_BLOCKED_GOVERNANCE,
            str(criteria.get("governance_review_reason") or "governance_authorized=false"),
        )
    elif readiness_status != "READY":
        state, reason = DISPATCH_BLOCKED_READINESS, f"execution_readiness={readiness_status}"
    elif require_market_open and not market_open:
        state, reason = DISPATCH_BLOCKED_MARKET_CLOSED, "market_closed"
    elif batch_notional_status == "EXCEEDS_CAP":
        state, reason = (
            DISPATCH_BLOCKED_BATCH_NOTIONAL_CAP,
            f"batch_notional={batch_notional:.2f} cap={configured_cap:.2f}",
        )
    else:
        state, reason = DISPATCH_READY, "all_dispatch_gates_satisfied"

    dispatch_permitted = state == DISPATCH_READY
    return {
        **base,
        "dispatch_interlock_state": state,
        "dispatch_interlock_reason": reason,
        "dispatch_permitted": dispatch_permitted,
    }


def assert_defensive_exit_dispatch_permitted(
    *,
    mode: str = "paper",
    broker_snapshot_ok: bool = True,
    require_market_open: bool = True,
    planned_count: int = 0,
) -> Dict[str, Any]:
    """Raise DispatchInterlockBlocked when defensive exit dispatch is not permitted."""
    result = evaluate_defensive_exit_dispatch_interlock(
        mode=mode,
        broker_snapshot_ok=broker_snapshot_ok,
        require_market_open=require_market_open,
        planned_count=planned_count,
    )
    if not result.get("dispatch_permitted"):
        raise DispatchInterlockBlocked(
            str(result.get("dispatch_interlock_state") or DISPATCH_BLOCKED_APPROVAL_REQUIRED),
            str(result.get("dispatch_interlock_reason") or ""),
            result,
        )
    return result


def format_dispatch_interlock_block(result: Dict[str, Any]) -> str:
    return (
        f"[DISPATCH_INTERLOCK] status={result.get('dispatch_interlock_state')} "
        f"reason={result.get('dispatch_interlock_reason')} "
        f"real_approval={result.get('real_approval_artifact_exists')} "
        f"operator_status={result.get('operator_approval_status')}"
    )
