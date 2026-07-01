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

SAFE_BLOCK_DISPATCH_APPROVAL_REQUIRED = "SAFE_BLOCK_DISPATCH_APPROVAL_REQUIRED"
SAFE_BLOCK_EXPLICIT_BROKER_EXECUTION_PHASE_REQUIRED = (
    "SAFE_BLOCK_EXPLICIT_BROKER_EXECUTION_PHASE_REQUIRED"
)
EXPLICIT_BROKER_EXECUTION_REASON = "paper_operator_approval_valid_broker_execution_separate"
PROJECTED_ORDERS_CSV = RESULTS / "phase_197x_projected_paper_orders.csv"

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


def is_dispatch_approval_safe_block_state(state: str, *, reason: str = "") -> bool:
    """True when dispatch block is an expected no-execution safe block."""
    state_s = str(state or "")
    reason_s = str(reason or "")
    if state_s == DISPATCH_BLOCKED_APPROVAL_REQUIRED:
        return True
    if state_s == DISPATCH_BLOCKED_GOVERNANCE and reason_s == EXPLICIT_BROKER_EXECUTION_REASON:
        return True
    return False


def is_dispatch_interlock_safe_block(interlock: Dict[str, Any]) -> bool:
    """True when interlock result is an intentional safe-block (no broker submission)."""
    return is_dispatch_approval_safe_block_state(
        str(interlock.get("dispatch_interlock_state") or ""),
        reason=str(interlock.get("dispatch_interlock_reason") or ""),
    )


def resolve_dispatch_safe_block_state(interlock: Dict[str, Any]) -> str:
    reason = str(interlock.get("dispatch_interlock_reason") or "")
    if reason == EXPLICIT_BROKER_EXECUTION_REASON:
        return SAFE_BLOCK_EXPLICIT_BROKER_EXECUTION_PHASE_REQUIRED
    return SAFE_BLOCK_DISPATCH_APPROVAL_REQUIRED


def load_paper_approved_execution_scope() -> Tuple[List[str], str, bool]:
    """
    Approved capped batch symbols for execution scope (197W/197X authority).
    Returns (symbols, source, approval_valid_current).
    """
    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_VALID_CURRENT,
        classify_paper_operator_approval_artifact,
    )

    ap_state, _, ap_details = classify_paper_operator_approval_artifact()
    approval_valid = ap_state == PAPER_APPROVAL_VALID_CURRENT
    if not approval_valid:
        return [], "approval_not_valid_current", False

    contract = ap_details.get("contract") or {}
    approved = sorted(
        {str(s).strip().upper() for s in (contract.get("approved_symbols") or []) if str(s).strip()}
    )
    if PROJECTED_ORDERS_CSV.is_file():
        try:
            import csv

            csv_symbols: List[str] = []
            with PROJECTED_ORDERS_CSV.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    sym = str(row.get("symbol") or "").strip().upper()
                    if sym:
                        csv_symbols.append(sym)
            csv_symbols = sorted(set(csv_symbols))
            if csv_symbols and set(csv_symbols).issubset(set(approved)):
                return csv_symbols, "phase_197x_projected_paper_orders", True
        except Exception:
            pass
    return approved, "operator_paper_defensive_exit_approval", True


def constrain_handoff_to_approved_paper_scope(
    planned: Sequence[Any],
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    When valid paper-only approval exists, restrict sell-side defensive handoff
    to the approved 197W/197X capped batch. Broader candidates remain diagnosed
    upstream; they are excluded from execution-eligible handoff only.
    """
    symbols, source, valid = load_paper_approved_execution_scope()
    meta: Dict[str, Any] = {
        "paper_scope_constrained": False,
        "paper_scope_source": source,
        "paper_scope_approval_valid": valid,
    }
    if not valid or not symbols:
        return list(planned), meta

    approved_set = set(symbols)
    kept: List[Any] = []
    excluded: List[str] = []
    for item in planned:
        sym = str(getattr(item, "symbol", "") or "").strip().upper()
        action = str(getattr(item, "management_action", "") or "").upper()
        side = str(getattr(item, "side", "") or "").lower()
        is_defensive_sell = action in DEFENSIVE_SELL_ACTIONS or side == "sell"
        if is_defensive_sell and sym and sym not in approved_set:
            excluded.append(sym)
            continue
        kept.append(item)

    meta.update(
        {
            "paper_scope_constrained": True,
            "paper_scope_approved_symbols": symbols,
            "paper_scope_excluded_symbols": sorted(set(excluded)),
            "paper_scope_kept_count": len(kept),
            "paper_scope_excluded_count": len(set(excluded)),
        }
    )
    if excluded:
        print(
            f"[MANAGE_SCOPE] excluded {len(set(excluded))} sell(s) outside approved "
            f"197W/197X batch: {sorted(set(excluded))}",
            flush=True,
        )
    return kept, meta


def is_dispatch_approval_safe_block_state_legacy(state: str) -> bool:
    """Backward-compatible state-only check (approval-required safe block)."""
    return str(state or "") == DISPATCH_BLOCKED_APPROVAL_REQUIRED


def print_dispatch_approval_safe_block(interlock: Dict[str, Any]) -> None:
    """Emit stable safe-block markers for expected dispatch no-execution blocks."""
    import sys

    safe_state = resolve_dispatch_safe_block_state(interlock)
    reason = str(interlock.get("dispatch_interlock_reason") or "DISPATCH_APPROVAL_REQUIRED")
    print(f"[SAFE_BLOCK] state={safe_state} reason={reason}", flush=True)
    print("broker_orders_submitted=false", flush=True)
    print("exits_executed=0", flush=True)
    print("trims_executed=0", flush=True)
    print("operator_execution_required=true", flush=True)
    print("explicit_broker_execution_phase_required=true", flush=True)
    print("eligible_for_execution=false", flush=True)
    print("cycle_should_continue=true", flush=True)
    print(
        f"dispatch_interlock_state={interlock.get('dispatch_interlock_state')}",
        flush=True,
    )
    sys.stdout.flush()


def format_dispatch_interlock_block(result: Dict[str, Any], *, stage: Optional[str] = None) -> str:
    parts = [
        f"[DISPATCH_INTERLOCK] status={result.get('dispatch_interlock_state')}",
        f"reason={result.get('dispatch_interlock_reason')}",
    ]
    if stage:
        parts.append(f"stage={stage}")
    parts.extend(
        [
            f"real_approval={result.get('real_approval_artifact_exists')}",
            f"operator_status={result.get('operator_approval_status')}",
        ]
    )
    return " ".join(parts)
