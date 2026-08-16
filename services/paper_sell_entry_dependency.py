"""Candidate-level paper entry filter against pending SELL readiness (Phase 198I-ZB-R).

Treats every pending SELL as unexecuted. Credits zero proceeds. Never uses mtime.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.manage_positions_cycle_binding import (
    CLASS_AUTHORITY_READY,
    DECISION_BLOCK,
    DECISION_CANDIDATE_AWARE,
    DECISION_MAY_CONTINUE,
    SCOPE_CANDIDATE_FILTER,
    SCOPE_GLOBAL_SAFETY_BLOCK,
    SCOPE_NONE,
)

ALLOW_EXISTING_GATES = "ALLOW_EXISTING_GATES"
BLOCK_SAME_SYMBOL_PENDING_SELL = "BLOCK_SAME_SYMBOL_PENDING_SELL"
BLOCK_DEPENDS_ON_SELL_PROCEEDS = "BLOCK_DEPENDS_ON_SELL_PROCEEDS"
BLOCK_ROTATION_DEPENDENCY = "BLOCK_ROTATION_DEPENDENCY"
BLOCK_WORSENS_PENDING_RISK_REDUCTION = "BLOCK_WORSENS_PENDING_RISK_REDUCTION"
BLOCK_UNPROVEN_SECTOR_INDEPENDENCE = "BLOCK_UNPROVEN_SECTOR_INDEPENDENCE"
BLOCK_INSUFFICIENT_CURRENT_CASH = "BLOCK_INSUFFICIENT_CURRENT_CASH"
BLOCK_CURRENT_PORTFOLIO_CAP = "BLOCK_CURRENT_PORTFOLIO_CAP"
BLOCK_GLOBAL_SELL_UNCERTAINTY = "BLOCK_GLOBAL_SELL_UNCERTAINTY"
BLOCK_CRITICAL_EXIT_PENDING = "BLOCK_CRITICAL_EXIT_PENDING"

CRITICAL_REASON_MARKERS = ("FORCE_EXIT",)


def _sym(val: Any) -> str:
    return str(val or "").strip().upper()


def _f(val: Any) -> Optional[float]:
    try:
        if val is None or str(val).strip() in ("", "nan", "None"):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def is_critical_pending_sell(row: Dict[str, Any]) -> bool:
    action = _sym(row.get("action") or row.get("final_action") or row.get("management_action"))
    reason = _sym(row.get("reason_code") or row.get("skip_reason"))
    if action == "FORCE_EXIT" or "FORCE_EXIT" in reason:
        return True
    flag = str(row.get("force_exit_override") or row.get("force_exit") or "").lower()
    if flag in {"true", "1", "yes"}:
        return True
    if str(row.get("risk_criticality") or "").upper() == "CRITICAL":
        return True
    return False


def pending_sell_sector(symbol: str) -> str:
    from services.sector_exposure import get_sector

    return str(get_sector(symbol) or "Unknown")


def candidate_notional(candidate: Dict[str, Any]) -> Optional[float]:
    for key in (
        "est_notional",
        "recommended_notional",
        "target_notional",
        "planned_notional",
        "notional",
    ):
        n = _f(candidate.get(key))
        if n is not None:
            return n
    qty = _f(candidate.get("qty") or candidate.get("recommended_qty"))
    px = _f(candidate.get("est_price") or candidate.get("price") or candidate.get("last"))
    if qty is not None and px is not None:
        return qty * px
    return None


SECTOR_REDUCTION_MARKERS = (
    "SECTOR_REDUCTION",
    "REDUCE_SECTOR",
    "SECTOR_CONCENTRATION",
    "CONCENTRATION_REDUCTION",
    "EXPOSURE_REDUC",
    "REDUCE_CONCENTRATION",
)
ROTATION_REASON_MARKERS = ("ROTATION", "ROTATE")
UNKNOWN_SECTORS = {"", "UNKNOWN", "UNMAPPED"}


def _is_sector_reduction_objective(pending: Dict[str, Any]) -> bool:
    if pending.get("sector_reduction") or pending.get("concentration_reduction"):
        return True
    blob = " ".join(
        str(pending.get(k) or "")
        for k in ("reason_code", "risk_objective", "objective", "skip_reason")
    ).upper()
    return any(marker in blob for marker in SECTOR_REDUCTION_MARKERS)


def _rotation_id(obj: Dict[str, Any]) -> str:
    return str(obj.get("rotation_id") or obj.get("rotation_batch_id") or "").strip()


def _rotation_indicated(obj: Dict[str, Any]) -> bool:
    if _rotation_id(obj):
        return True
    if obj.get("rotation_dependent") or obj.get("requires_rotation"):
        return True
    action = _sym(obj.get("action") or obj.get("final_action") or obj.get("allocation_action"))
    if action in {"ROTATE_EXIT", "ROTATE", "ROTATE_IN"}:
        return True
    blob = " ".join(
        str(obj.get(k) or "") for k in ("reason_code", "skip_reason", "funding_source")
    ).upper()
    if any(marker in blob for marker in ROTATION_REASON_MARKERS):
        return True
    if obj.get("paired_exit_symbol") or obj.get("rotation_exit_symbol"):
        return True
    return False


def _dependency_symbols(candidate: Dict[str, Any]) -> List[str]:
    raw = candidate.get("depends_on_symbols") or candidate.get("rotation_from") or []
    if isinstance(raw, str):
        raw = [p.strip() for p in raw.replace("|", ",").split(",") if p.strip()]
    out = [_sym(x) for x in raw]
    paired = _sym(candidate.get("paired_exit_symbol") or candidate.get("rotation_exit_symbol"))
    if paired:
        out.append(paired)
    funding = _sym(candidate.get("funding_source") or candidate.get("capital_source"))
    if funding in {"SELL_PROCEEDS", "PENDING_SELL", "ROTATION_PROCEEDS"}:
        out.append("*PROCEEDS*")
    return [s for s in out if s]


def evaluate_entry_candidate(
    candidate: Dict[str, Any],
    *,
    pending_sells: Sequence[Dict[str, Any]],
    current_cash: Optional[float],
    at_position_cap: bool = False,
    global_scope: str = SCOPE_NONE,
    global_reason: str = "",
) -> Dict[str, Any]:
    """Conservative: pending SELL remains in the book; proceeds = 0."""
    symbol = _sym(candidate.get("ticker") or candidate.get("symbol"))
    action = _sym(candidate.get("allocation_action") or candidate.get("action") or "BUY")
    pending = [p for p in pending_sells if _sym(p.get("symbol"))]
    pending_syms = {_sym(p.get("symbol")) for p in pending}
    rec: Dict[str, Any] = {
        "symbol": symbol,
        "action": action,
        "decision": ALLOW_EXISTING_GATES,
        "proceeds_credited": 0.0,
    }
    if global_scope == SCOPE_GLOBAL_SAFETY_BLOCK:
        if (
            "CRITICAL" in str(global_reason or "").upper()
            or global_reason == BLOCK_CRITICAL_EXIT_PENDING
        ):
            rec["decision"] = BLOCK_CRITICAL_EXIT_PENDING
        else:
            rec["decision"] = BLOCK_GLOBAL_SELL_UNCERTAINTY
        rec["reason"] = global_reason or BLOCK_GLOBAL_SELL_UNCERTAINTY
        return rec
    if not pending:
        rec["decision"] = ALLOW_EXISTING_GATES
        return rec
    if any(is_critical_pending_sell(p) for p in pending):
        rec["decision"] = BLOCK_CRITICAL_EXIT_PENDING
        rec["reason"] = "critical_defensive_exit_pending"
        return rec
    if not symbol:
        rec["decision"] = BLOCK_GLOBAL_SELL_UNCERTAINTY
        rec["reason"] = "entry_symbol_unproven"
        return rec
    if symbol in pending_syms:
        rec["decision"] = BLOCK_SAME_SYMBOL_PENDING_SELL
        rec["reason"] = "same_symbol_pending_sell"
        return rec
    deps = _dependency_symbols(candidate)
    if "*PROCEEDS*" in deps or any(d in pending_syms for d in deps if d != "*PROCEEDS*"):
        if "*PROCEEDS*" in deps or str(candidate.get("funding_source") or "").upper() in {
            "SELL_PROCEEDS",
            "PENDING_SELL",
            "ROTATION_PROCEEDS",
        }:
            rec["decision"] = BLOCK_DEPENDS_ON_SELL_PROCEEDS
            rec["reason"] = "requires_unexecuted_sell_proceeds"
            return rec
        rec["decision"] = BLOCK_ROTATION_DEPENDENCY
        rec["reason"] = "rotation_depends_on_pending_sell"
        return rec
    if bool(candidate.get("rotation_dependent") or candidate.get("requires_sell_proceeds")):
        rec["decision"] = (
            BLOCK_DEPENDS_ON_SELL_PROCEEDS
            if candidate.get("requires_sell_proceeds")
            else BLOCK_ROTATION_DEPENDENCY
        )
        rec["reason"] = "explicit_sell_dependency"
        return rec
    cand_rot = _rotation_id(candidate)
    pending_rot_ids = {_rotation_id(p) for p in pending if _rotation_id(p)}
    if cand_rot and cand_rot in pending_rot_ids:
        rec["decision"] = BLOCK_ROTATION_DEPENDENCY
        rec["reason"] = "shared_rotation_or_batch_id"
        rec["rotation_id"] = cand_rot
        return rec
    pending_rotation_indicated = [p for p in pending if _rotation_indicated(p)]
    if pending_rotation_indicated:
        missing_pending_rot = any(not _rotation_id(p) for p in pending_rotation_indicated)
        cand_rotation_flagged = _rotation_indicated(candidate)
        if cand_rotation_flagged and not cand_rot:
            rec["decision"] = BLOCK_ROTATION_DEPENDENCY
            rec["reason"] = "rotation_metadata_unproven"
            return rec
        if missing_pending_rot and cand_rotation_flagged:
            rec["decision"] = BLOCK_ROTATION_DEPENDENCY
            rec["reason"] = "rotation_metadata_unproven"
            return rec
    cand_sector = _sym(candidate.get("sector") or pending_sell_sector(symbol))
    reducing_sectors = {
        _sym(pending_sell_sector(_sym(p.get("symbol"))))
        for p in pending
        if _sym(p.get("action") or p.get("final_action")) in {"SELL", "TRIM", "EXIT", "ROTATE_EXIT"}
    }
    reducing_sectors -= UNKNOWN_SECTORS
    reducing_sectors.discard("DIVERSIFIED")
    sector_reduction_pending = [p for p in pending if _is_sector_reduction_objective(p)]
    if sector_reduction_pending:
        cand_unresolved = cand_sector in UNKNOWN_SECTORS
        if cand_unresolved:
            rec["decision"] = BLOCK_UNPROVEN_SECTOR_INDEPENDENCE
            rec["reason"] = "sector_independence_unproven"
            rec["sector"] = cand_sector
            return rec
    if (
        cand_sector
        and cand_sector not in {"UNKNOWN", "DIVERSIFIED"}
        and cand_sector in reducing_sectors
    ):
        rec["decision"] = BLOCK_WORSENS_PENDING_RISK_REDUCTION
        rec["reason"] = f"worsens_pending_{cand_sector.lower()}_reduction"
        rec["sector"] = cand_sector
        return rec
    notional = candidate_notional(candidate)
    if current_cash is None:
        rec["decision"] = BLOCK_INSUFFICIENT_CURRENT_CASH
        rec["reason"] = "settled_cash_unproven"
        return rec
    if notional is not None and notional > float(current_cash) + 1e-9:
        rec["decision"] = BLOCK_DEPENDS_ON_SELL_PROCEEDS
        rec["reason"] = "notional_exceeds_current_cash_without_sell_proceeds"
        rec["required_notional"] = notional
        rec["current_cash"] = float(current_cash)
        rec["proceeds_credited"] = 0.0
        return rec
    if at_position_cap and action in {"BUY", "OPEN_NEW"}:
        rec["decision"] = BLOCK_CURRENT_PORTFOLIO_CAP
        rec["reason"] = "position_cap_with_pending_sell_unreleased"
        return rec
    rec["decision"] = ALLOW_EXISTING_GATES
    rec["current_cash"] = float(current_cash)
    rec["required_notional"] = notional
    rec["sector"] = cand_sector
    rec["proceeds_credited"] = 0.0
    return rec


def filter_entry_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    pending_sells: Sequence[Dict[str, Any]],
    current_cash: Optional[float],
    at_position_cap: bool = False,
    global_scope: str = SCOPE_NONE,
    global_reason: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (allowed_candidates, per_symbol_decisions). Does not credit SELL proceeds."""
    decisions: List[Dict[str, Any]] = []
    allowed: List[Dict[str, Any]] = []
    for cand in candidates:
        rec = evaluate_entry_candidate(
            cand,
            pending_sells=pending_sells,
            current_cash=current_cash,
            at_position_cap=at_position_cap,
            global_scope=global_scope,
            global_reason=global_reason,
        )
        decisions.append(rec)
        if rec.get("decision") == ALLOW_EXISTING_GATES:
            allowed.append(dict(cand))
    return allowed, decisions


def context_from_interlock(interlock: Dict[str, Any]) -> Dict[str, Any]:
    scope = str(interlock.get("interlock_scope") or SCOPE_NONE)
    decision = str(interlock.get("decision") or "")
    if decision == DECISION_BLOCK or scope == SCOPE_GLOBAL_SAFETY_BLOCK:
        scope = SCOPE_GLOBAL_SAFETY_BLOCK
    elif (
        decision in {DECISION_CANDIDATE_AWARE, DECISION_MAY_CONTINUE}
        and scope == SCOPE_CANDIDATE_FILTER
    ):
        pass
    elif str(interlock.get("packet_status") or "") == "AUTHORITY_READY":
        scope = SCOPE_CANDIDATE_FILTER
    pending = list(interlock.get("pending_sell_actions") or [])
    if not pending:
        pending = [
            {"symbol": s, "action": "EXIT"} for s in (interlock.get("pending_sell_symbols") or [])
        ]
    return {
        "scope": scope,
        "global_reason": str(interlock.get("global_block_reason") or interlock.get("reason") or ""),
        "pending_sells": pending,
        "proceeds_available_for_entry": 0.0,
        "risk_criticality": str(interlock.get("risk_criticality") or "ORDINARY"),
    }
