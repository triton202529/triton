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


def _runtime_capped_preview_block(preview_doc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        preview_doc.get("capped_preview_runtime_cap")
        or preview_doc.get("capped_preview")
        or preview_doc.get("capped_preview_cli_cap")
        or {}
    )


def _load_197x_projected_order_rows() -> List[Dict[str, Any]]:
    if not PROJECTED_ORDERS_CSV.is_file():
        return []
    try:
        import csv

        rows: List[Dict[str, Any]] = []
        with PROJECTED_ORDERS_CSV.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                sym = str(row.get("symbol") or "").strip().upper()
                if sym:
                    rows.append(dict(row))
        return rows
    except Exception:
        return []


def compute_dispatch_batch_notional_cap_context(
    *,
    preview_doc: Dict[str, Any],
    use_approved_197x_scope: bool,
) -> Dict[str, Any]:
    """
    Compute batch notional cap context for dispatch interlock.
    When paper loop-mediated execution is authorized, cap applies only to the
    final approved 197X projected order batch, not the wider pre-scope universe.
    """
    runtime_block = _runtime_capped_preview_block(preview_doc)
    pre_scope_planned_notional = float(
        preview_doc.get("batch_notional_estimate")
        or runtime_block.get("batch_notional_full")
        or 0.0
    )
    configured_cap = float(
        preview_doc.get("configured_batch_cap_manage_positions")
        or preview_doc.get("configured_batch_cap_effective")
        or 50000.0
    )

    approved_symbols, scope_source, scope_valid = load_paper_approved_execution_scope()
    approved_set = set(approved_symbols)
    excluded_symbols: List[str] = []
    excluded_outside_197x_scope_notional = 0.0
    approved_scope_notional = float(runtime_block.get("batch_notional_capped_preview") or 0.0)
    cap_basis = "defensive_exit_capped_batch_preview_full"

    preview_orders = list(runtime_block.get("capped_preview_orders") or [])
    if preview_orders and approved_set:
        for order in preview_orders:
            sym = str(order.get("symbol") or "").strip().upper()
            if not sym or sym in approved_set:
                continue
            excluded_symbols.append(sym)
            excluded_outside_197x_scope_notional += float(
                order.get("notional")
                or order.get("notional_estimate")
                or order.get("estimated_notional")
                or 0.0
            )
        excluded_symbols = sorted(set(excluded_symbols))
        excluded_outside_197x_scope_notional = round(excluded_outside_197x_scope_notional, 2)

    projected_rows = _load_197x_projected_order_rows()
    projected_symbols = sorted(
        {str(r.get("symbol") or "").strip().upper() for r in projected_rows if r.get("symbol")}
    )
    projected_notional = round(
        sum(float(r.get("notional_estimate") or 0.0) for r in projected_rows),
        2,
    )

    use_scope_cap = bool(use_approved_197x_scope and scope_valid and projected_rows)
    if use_scope_cap:
        approved_scope_notional = projected_notional
        cap_basis = "phase_197x_projected_paper_orders"
        if (
            not excluded_outside_197x_scope_notional
            and pre_scope_planned_notional > projected_notional
        ):
            excluded_outside_197x_scope_notional = round(
                pre_scope_planned_notional - projected_notional,
                2,
            )
    elif approved_scope_notional <= 0:
        approved_scope_notional = pre_scope_planned_notional

    final_cap_checked_notional = (
        approved_scope_notional if use_scope_cap else pre_scope_planned_notional
    )
    cap_passed = final_cap_checked_notional <= configured_cap if configured_cap > 0 else True
    batch_notional_status = "WITHIN_CAP" if cap_passed else "EXCEEDS_CAP"

    return {
        "cap_basis": cap_basis,
        "use_approved_197x_scope": use_scope_cap,
        "approved_scope_source": scope_source,
        "approved_scope_symbols": projected_symbols if use_scope_cap else approved_symbols,
        "pre_scope_planned_notional": round(pre_scope_planned_notional, 2),
        "excluded_outside_197x_scope_notional": excluded_outside_197x_scope_notional,
        "approved_scope_notional": round(approved_scope_notional, 2),
        "final_cap_checked_notional": round(final_cap_checked_notional, 2),
        "batch_notional_cap": round(configured_cap, 2),
        "cap_passed": cap_passed,
        "batch_notional_status": batch_notional_status,
        "excluded_symbols": excluded_symbols,
    }


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

    cap_context: Dict[str, Any] = {}
    batch_notional = 0.0
    configured_cap = 50000.0
    batch_notional_status = "WITHIN_CAP"

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
        "batch_notional": batch_notional,
        "batch_notional_cap": configured_cap,
        "batch_notional_status": batch_notional_status,
        "batch_notional_cap_diagnostics": cap_context,
        "eligible_for_execution": False,
        "broker_orders_submitted": False,
    }

    state: Optional[str] = None
    reason: Optional[str] = None
    paper_execution_gate: Optional[Dict[str, Any]] = None
    gate_ready = False

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
    else:
        from services.paper_operator_approval_artifact_state import (
            PAPER_APPROVAL_VALID_CURRENT,
            classify_paper_operator_approval_artifact,
        )

        ap_state, _, _ = classify_paper_operator_approval_artifact(
            broker_snapshot_ok=broker_snapshot_ok
        )
        if ap_state == PAPER_APPROVAL_VALID_CURRENT:
            from services.phase_197y_loop_mediated_paper_execution_gate import (
                GATE_READY_LOOP_MEDIATED,
                evaluate_loop_mediated_paper_execution_gate,
                map_gate_state_to_dispatch,
            )

            paper_execution_gate = evaluate_loop_mediated_paper_execution_gate(
                broker_snapshot_ok=broker_snapshot_ok,
                open_orders_count=open_orders,
                live_blocked=bool(live_payload.get("live_blocked")),
                market_open=market_open,
            )
            gate_state = str(paper_execution_gate.get("final_state") or "")
            if gate_state == GATE_READY_LOOP_MEDIATED:
                gate_ready = True
            else:
                state, reason = map_gate_state_to_dispatch(gate_state)
        elif not governance_authorized:
            state, reason = (
                DISPATCH_BLOCKED_GOVERNANCE,
                str(criteria.get("governance_review_reason") or "governance_authorized=false"),
            )

    readiness_bridge: Optional[Dict[str, Any]] = None
    if state is None:
        from services.paper_loop_mediated_execution_readiness_bridge import (
            evaluate_paper_loop_mediated_execution_readiness_bridge,
            is_paper_loop_mediated_readiness_satisfied,
        )

        readiness_bridge = evaluate_paper_loop_mediated_execution_readiness_bridge(
            mode=mode,
            broker_snapshot_ok=broker_snapshot_ok,
            open_orders_count=open_orders,
            live_blocked=bool(live_payload.get("live_blocked")),
            market_open=market_open,
        )
        readiness_satisfied = is_paper_loop_mediated_readiness_satisfied(
            readiness_status,
            mode=mode,
            broker_snapshot_ok=broker_snapshot_ok,
            open_orders_count=open_orders,
            bridge=readiness_bridge,
        )
        bridge_active = bool(readiness_bridge.get("bridge_active"))
        use_scope_cap = bool(gate_ready or bridge_active)
        cap_context = compute_dispatch_batch_notional_cap_context(
            preview_doc=preview_doc,
            use_approved_197x_scope=use_scope_cap,
        )
        batch_notional = float(cap_context.get("final_cap_checked_notional") or 0.0)
        configured_cap = float(cap_context.get("batch_notional_cap") or 50000.0)
        batch_notional_status = str(cap_context.get("batch_notional_status") or "WITHIN_CAP")
        base["batch_notional"] = round(batch_notional, 2)
        base["batch_notional_cap"] = configured_cap
        base["batch_notional_status"] = batch_notional_status
        base["batch_notional_cap_diagnostics"] = cap_context
        if not gate_ready and not governance_authorized:
            state, reason = (
                DISPATCH_BLOCKED_GOVERNANCE,
                str(criteria.get("governance_review_reason") or "governance_authorized=false"),
            )
        elif not readiness_satisfied:
            state, reason = DISPATCH_BLOCKED_READINESS, f"execution_readiness={readiness_status}"
        elif require_market_open and not market_open:
            state, reason = DISPATCH_BLOCKED_MARKET_CLOSED, "market_closed"
        elif batch_notional_status == "EXCEEDS_CAP":
            state, reason = (
                DISPATCH_BLOCKED_BATCH_NOTIONAL_CAP,
                f"batch_notional={batch_notional:.2f} cap={configured_cap:.2f} "
                f"basis={cap_context.get('cap_basis')}",
            )
        else:
            state, reason = DISPATCH_READY, (
                "all_dispatch_gates_satisfied_including_197y"
                if gate_ready
                else "all_dispatch_gates_satisfied"
            )

    dispatch_permitted = state == DISPATCH_READY
    result_payload = {
        **base,
        "dispatch_interlock_state": state,
        "dispatch_interlock_reason": reason,
        "dispatch_permitted": dispatch_permitted,
        "paper_execution_gate_ready": gate_ready,
    }
    if paper_execution_gate is not None:
        result_payload["paper_execution_gate"] = paper_execution_gate
        result_payload["paper_execution_gate_state"] = paper_execution_gate.get("final_state")
    if readiness_bridge is not None:
        result_payload["paper_loop_mediated_readiness_bridge"] = readiness_bridge
        result_payload["execution_readiness_bridged"] = bool(
            readiness_bridge.get("execution_readiness_bridged")
        )
        result_payload["effective_readiness_status"] = readiness_bridge.get(
            "effective_readiness_status"
        )
    return result_payload


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


def load_paper_approved_execution_quantities() -> Tuple[Dict[str, int], str]:
    """
    Approved per-symbol quantities for paper defensive handoff.

    Prefers 197X projected order quantities when they are a subset of the
    approval contract; otherwise uses approval approved_quantities.
    """
    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_VALID_CURRENT,
        classify_paper_operator_approval_artifact,
    )

    ap_state, _, ap_details = classify_paper_operator_approval_artifact()
    if ap_state != PAPER_APPROVAL_VALID_CURRENT:
        return {}, "approval_not_valid_current"

    contract = ap_details.get("contract") or {}
    approval_qty: Dict[str, int] = {}
    raw = contract.get("approved_quantities") or {}
    if isinstance(raw, dict):
        for sym, qty in raw.items():
            try:
                q = int(float(qty))
            except (TypeError, ValueError):
                continue
            s = str(sym).strip().upper()
            if s and q > 0:
                approval_qty[s] = q

    projected_qty: Dict[str, int] = {}
    for row in _load_197x_projected_order_rows():
        sym = str(row.get("symbol") or "").strip().upper()
        try:
            q = int(float(row.get("quantity") or 0))
        except (TypeError, ValueError):
            q = 0
        if sym and q > 0:
            projected_qty[sym] = q

    if projected_qty and (
        not approval_qty or set(projected_qty.keys()).issubset(set(approval_qty.keys()))
    ):
        # When both exist, projected qty must not exceed approval qty.
        if approval_qty:
            for sym, q in list(projected_qty.items()):
                approved = int(approval_qty.get(sym) or 0)
                if approved > 0:
                    projected_qty[sym] = min(q, approved)
        return projected_qty, "phase_197x_projected_paper_orders"
    if approval_qty:
        return approval_qty, "operator_paper_defensive_exit_approval"
    return {}, "no_approved_quantities"


def constrain_handoff_to_approved_paper_scope(
    planned: Sequence[Any],
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    When valid paper-only approval exists, restrict sell-side defensive handoff
    to the approved 197W/197X capped batch (symbols and quantities). Broader
    candidates remain diagnosed upstream; they are excluded from
    execution-eligible handoff only.
    """
    from dataclasses import replace

    symbols, source, valid = load_paper_approved_execution_scope()
    qty_map, qty_source = load_paper_approved_execution_quantities()
    meta: Dict[str, Any] = {
        "paper_scope_constrained": False,
        "paper_scope_source": source,
        "paper_scope_approval_valid": valid,
        "paper_scope_qty_source": qty_source,
        "paper_scope_approved_quantities": dict(qty_map),
        "paper_scope_qty_clamps": [],
    }
    if not valid or not symbols:
        return list(planned), meta

    approved_set = set(symbols)
    kept: List[Any] = []
    excluded: List[str] = []
    qty_clamps: List[str] = []
    for item in planned:
        sym = str(getattr(item, "symbol", "") or "").strip().upper()
        action = str(getattr(item, "management_action", "") or "").upper()
        side = str(getattr(item, "side", "") or "").lower()
        is_defensive_sell = action in DEFENSIVE_SELL_ACTIONS or side == "sell"
        if is_defensive_sell and sym and sym not in approved_set:
            excluded.append(sym)
            continue
        if is_defensive_sell and sym in qty_map:
            approved_qty = int(qty_map[sym])
            cur_qty = int(getattr(item, "qty", 0) or 0)
            pos_qty = float(getattr(item, "current_position_qty", 0) or 0)
            max_by_position = int(pos_qty) if pos_qty > 0 else approved_qty
            new_qty = min(approved_qty, max_by_position)
            if new_qty <= 0:
                excluded.append(sym)
                continue
            if cur_qty != new_qty:
                updates: Dict[str, Any] = {"qty": new_qty}
                planned_notional = float(getattr(item, "planned_notional", 0.0) or 0.0)
                if planned_notional > 0 and cur_qty > 0:
                    updates["planned_notional"] = round(planned_notional * (new_qty / cur_qty), 2)
                try:
                    item = replace(item, **updates)
                except TypeError:
                    try:
                        setattr(item, "qty", new_qty)
                        if "planned_notional" in updates:
                            setattr(item, "planned_notional", updates["planned_notional"])
                    except Exception:
                        pass
                qty_clamps.append(f"{sym}:{cur_qty}->{new_qty}")
        kept.append(item)

    meta.update(
        {
            "paper_scope_constrained": True,
            "paper_scope_approved_symbols": symbols,
            "paper_scope_excluded_symbols": sorted(set(excluded)),
            "paper_scope_kept_count": len(kept),
            "paper_scope_excluded_count": len(set(excluded)),
            "paper_scope_qty_clamps": qty_clamps,
        }
    )
    if excluded:
        print(
            f"[MANAGE_SCOPE] excluded {len(set(excluded))} sell(s) outside approved "
            f"197W/197X batch: {sorted(set(excluded))}",
            flush=True,
        )
    if qty_clamps:
        print(
            f"[MANAGE_SCOPE] clamped defensive qty to approved 197X/197W scope: {qty_clamps}",
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
