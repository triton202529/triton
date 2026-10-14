"""Effective Portfolio action from the sell planner's concentration decision.

The lifecycle model action stays unchanged. This module does not write a
packet, arm authority, call a broker, or read a stale sell packet.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from services.defensive_exit_approval_readiness import (
    CLASS_NOT_SELECTED,
    _f,
    apply_concentration_exit_precedence,
)
from services.paper_sell_arm_authority import _parse_iso
from services.risk_watchdog import DEFAULT_CONCENTRATION_PCT

PENDING_SELL_REVIEW = "Pending fresh paper SELL review"
NO_PERMITTED_TRIM = "No permitted concentration trim"
BASIS_UNPROVEN_HOLD_STATUS = "Basis-unproven hold"
CURRENT_AUTHORITY_STATUS = "Paper SELL authority is current"
REASON_EXCESS_CONCENTRATION = "EXCESS_CONCENTRATION"
REASON_BASIS_UNPROVEN_HOLD = "BASIS_UNPROVEN_HOLD"


def load_sell_authority_view(root: Path) -> Dict[str, Any]:
    """Read the sell-authority file. Missing or unreadable authority is not current."""
    path = Path(root) / "data" / "governance" / "paper_sell_arm_authority.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _blocked_symbols() -> set[str]:
    from services.basis_unproven_hold import (
        HOLD_SYMBOLS,
        evaluate_basis_unproven_hold_for_packet,
    )

    result = evaluate_basis_unproven_hold_for_packet()
    if result.get("ok"):
        return {str(symbol or "").upper().strip() for symbol in result.get("blocked_symbols") or []}
    return {str(symbol).upper() for symbol in HOLD_SYMBOLS}


def _plan_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    symbol = str(row.get("symbol") or "").upper().strip()
    qty = row.get("current_position_qty")
    if qty in (None, ""):
        qty = row.get("qty")
    price = row.get("current_price")
    if price in (None, ""):
        market_value = _f(row.get("market_value") or row.get("value"))
        quantity = _f(qty)
        if market_value is not None and quantity is not None and quantity > 0.0:
            price = market_value / quantity
    return {
        "symbol": symbol,
        "current_position_qty": qty,
        "current_price": price,
    }


def _classified_row(symbol: str, lifecycle_action: str) -> Dict[str, Any]:
    action = str(lifecycle_action or "HOLD").upper().strip() or "HOLD"
    if action in {"", "NONE", "HOLD"}:
        return {
            "symbol": symbol,
            "action": "NONE",
            "classification": CLASS_NOT_SELECTED,
            "reason_code": "NO_ACTION_HOLD",
        }
    return {
        "symbol": symbol,
        "action": action,
        "classification": CLASS_NOT_SELECTED,
        "reason_code": action,
    }


def _authority_is_current_unconsumed(
    authority: Optional[Mapping[str, Any]],
    symbol: str,
    now: datetime,
) -> bool:
    if not isinstance(authority, Mapping) or not authority:
        return False
    status = (
        str(authority.get("consumption_status") or authority.get("status") or "").upper().strip()
    )
    if status in {"CONSUMED", "REVOKED", "EXPIRED"} or authority.get("consumed_at"):
        return False
    expires = _parse_iso(authority.get("expires_at"))
    if expires is None or expires <= now:
        return False
    symbols = [
        str(item).upper().strip()
        for item in (authority.get("approved_symbols") or authority.get("symbols") or [])
        if str(item).strip()
    ]
    return not symbols or symbol in symbols


def resolve_portfolio_effective_actions(
    rows: Sequence[Mapping[str, Any]],
    lifecycle: Optional[Mapping[str, str]] = None,
    *,
    root: Optional[Path] = None,
    blocked_symbols: Optional[Sequence[str]] = None,
    authority: Optional[Mapping[str, Any]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Dict[str, str]]:
    """Return effective action, operator status, reason, and the raw lifecycle action.

    A concentration TRIM is shown only when the sell planner can size a permitted
    trim. An expired or consumed authority does not make that trim executable.
    """
    lifecycle_map = {
        str(symbol).upper().strip(): str(action)
        for symbol, action in (lifecycle or {}).items()
        if str(symbol).strip()
    }
    if blocked_symbols is None:
        blocked = _blocked_symbols()
    else:
        blocked = {str(symbol or "").upper().strip() for symbol in blocked_symbols}
    if authority is None and root is not None:
        authority = load_sell_authority_view(root)
    moment = now or datetime.now(timezone.utc)
    plan_rows = [_plan_row(row) for row in rows if isinstance(row, Mapping)]
    classified = []
    lifecycle_by_symbol: Dict[str, str] = {}
    for plan in plan_rows:
        symbol = plan["symbol"]
        if not symbol:
            continue
        raw = lifecycle_map.get(symbol) or "HOLD"
        lifecycle_by_symbol[symbol] = raw
        classified.append(_classified_row(symbol, raw))
    decision = apply_concentration_exit_precedence(
        classified,
        plan_rows,
        blocked_symbols=sorted(blocked),
    )
    trim_symbols = set()
    if decision.get("status") == "applied":
        trim_symbols = {
            str(item.get("symbol") or "").upper().strip()
            for item in (decision.get("evidence") or {}).get("concentration_trims") or []
        }
    breach_without_trim = decision.get("status") in {
        "unavailable",
        "not_applicable",
    } and decision.get("reason") in {
        "concentration_exit_unavailable",
        "no_measurable_concentration_trim",
    }
    out: Dict[str, Dict[str, str]] = {}
    for symbol, raw in lifecycle_by_symbol.items():
        if symbol in blocked:
            out[symbol] = {
                "effective_action": "HOLD",
                "operator_status": BASIS_UNPROVEN_HOLD_STATUS,
                "reason": REASON_BASIS_UNPROVEN_HOLD,
                "lifecycle_action": raw,
            }
            continue
        if symbol in trim_symbols:
            current = _authority_is_current_unconsumed(authority, symbol, moment)
            out[symbol] = {
                "effective_action": "TRIM",
                "operator_status": (CURRENT_AUTHORITY_STATUS if current else PENDING_SELL_REVIEW),
                "reason": REASON_EXCESS_CONCENTRATION,
                "lifecycle_action": raw,
            }
            continue
        if breach_without_trim and _symbol_is_offender(plan_rows, symbol):
            out[symbol] = {
                "effective_action": (
                    raw if str(raw).upper().strip() not in {"", "NONE"} else "HOLD"
                ),
                "operator_status": NO_PERMITTED_TRIM,
                "reason": REASON_EXCESS_CONCENTRATION,
                "lifecycle_action": raw,
            }
            continue
        shown = raw if str(raw).upper().strip() not in {"", "NONE"} else "HOLD"
        out[symbol] = {
            "effective_action": shown,
            "operator_status": "",
            "reason": "",
            "lifecycle_action": raw,
        }
    return out


def _symbol_is_offender(plan_rows: Sequence[Mapping[str, Any]], symbol: str) -> bool:
    from services.defensive_exit_approval_readiness import (
        _evaluated_position_book,
        _weights_after,
    )

    book = _evaluated_position_book(plan_rows)
    weights = _weights_after(book, {})
    return float(weights.get(symbol) or 0.0) > float(DEFAULT_CONCENTRATION_PCT) + 1e-9
