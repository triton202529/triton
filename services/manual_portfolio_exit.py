"""Post-close manual paper exit. Review writes nothing.

Confirmation creates one ticker-specific, quantity-bounded, notional-bounded
authority in an isolated manual-exit store, then submits through
``broker.submit_order``. That is the existing paper-sell route: permit check,
execution gate, and basis-hold enforcement. This module does not read a
defensive sell packet or reuse an automatic sell authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from services.paper_sell_arm_authority import APPROVAL_CLI_CAP
from services.paper_sell_arm_runtime_enforcement import (
    PaperSellDispatchPermit,
    _ACTIVE_PERMIT,
    _write_durable_permit,
)
from services.paper_sell_arm_store import (
    STORE_SCHEMA_VERSION,
    batch_hash_from_scope,
    canonical_json,
    consume_authority_insert_dispatch,
    write_connection,
)

REASON = "Manual operator exit / profit-taking"
UNAVAILABLE_HOLD = "Basis-unproven hold"
GROSS_LABEL = "Gross provisional gain/loss"
SCHEMA = "MANUAL-PORTFOLIO-EXIT-REQUEST-1"
PAPER_HOST = "paper-api.alpaca.markets"


def _decimal(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError):
        return None
    if not number.is_finite():
        return None
    return number


def _money(number: Decimal) -> str:
    return format(number.quantize(Decimal("0.01")), "f")


def _qty_text(number: Decimal) -> str:
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def blocked_manual_exit_symbols(
    blocked_symbols: Optional[Sequence[str]] = None,
) -> set[str]:
    if blocked_symbols is not None:
        return {str(symbol or "").upper().strip() for symbol in blocked_symbols}
    from services.basis_unproven_hold import (
        HOLD_SYMBOLS,
        evaluate_basis_unproven_hold_for_packet,
    )

    result = evaluate_basis_unproven_hold_for_packet()
    if result.get("ok"):
        return {str(symbol or "").upper().strip() for symbol in result.get("blocked_symbols") or []}
    return {str(symbol).upper() for symbol in HOLD_SYMBOLS}


def _request_id(symbol: str, position_qty: str, price: str, selected_qty: str) -> str:
    material = canonical_json(
        {
            "symbol": symbol,
            "position_qty": position_qty,
            "estimated_price": price,
            "selected_qty": selected_qty,
            "lane": "manual_portfolio_exit",
        }
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _review(
    *,
    symbol: str,
    position_qty: Optional[Decimal],
    price: Optional[Decimal],
    average_cost: Optional[Decimal],
    selected_qty: Optional[Decimal],
    eligible: bool,
    unavailable_reason: str,
) -> Dict[str, Any]:
    position_text = _qty_text(position_qty) if position_qty is not None else ""
    price_text = _money(price) if price is not None else ""
    selected = selected_qty if selected_qty is not None else position_qty
    if isinstance(selected, Decimal):
        selected_text = _qty_text(selected)
    elif selected in (None, ""):
        selected_text = ""
    else:
        selected_text = str(selected).strip()
    notional = None
    if isinstance(selected, Decimal) and price is not None and selected > 0 and price > 0:
        notional = _money(selected * price)
    gross = None
    if (
        eligible
        and isinstance(selected, Decimal)
        and price is not None
        and average_cost is not None
        and selected > 0
    ):
        gross = _money((price - average_cost) * selected)
    return {
        "schema": SCHEMA,
        "request_id": _request_id(symbol, position_text, price_text, selected_text),
        "symbol": symbol,
        "position_qty": position_text,
        "selected_qty": selected_text,
        "estimated_price": price_text,
        "estimated_notional": notional,
        "average_cost": _money(average_cost) if average_cost is not None else None,
        "gross_provisional_gain_loss": gross,
        "gross_provisional_label": GROSS_LABEL,
        "reason": REASON,
        "eligible": eligible,
        "unavailable_reason": unavailable_reason,
    }


def build_manual_exit_review(
    row: Mapping[str, Any],
    *,
    blocked_symbols: Optional[Sequence[str]] = None,
    selected_qty: Any = None,
) -> Dict[str, Any]:
    """Pure review. Does not write authority, packet, audit, broker, or ledger state."""
    symbol = str(row.get("symbol") or "").upper().strip()
    position_qty = _decimal(
        row.get("qty") if row.get("qty") not in (None, "") else row.get("current_position_qty")
    )
    price = _decimal(row.get("current_price"))
    average_cost = _decimal(row.get("avg_entry_price") or row.get("average_cost"))
    if selected_qty is None:
        chosen: Any = position_qty
    else:
        chosen = _decimal(selected_qty)
        if chosen is None:
            chosen = selected_qty
    blocked = symbol in blocked_manual_exit_symbols(blocked_symbols)
    if blocked:
        return _review(
            symbol=symbol,
            position_qty=position_qty,
            price=price,
            average_cost=None,
            selected_qty=chosen,
            eligible=False,
            unavailable_reason=UNAVAILABLE_HOLD,
        )
    return _review(
        symbol=symbol,
        position_qty=position_qty,
        price=price,
        average_cost=average_cost,
        selected_qty=chosen,
        eligible=True,
        unavailable_reason="",
    )


def review_with_quantity(review: Mapping[str, Any], selected_qty: Any) -> Dict[str, Any]:
    """Return a new review for the operator-selected quantity. No writes."""
    row = {
        "symbol": review.get("symbol"),
        "qty": review.get("position_qty"),
        "current_price": review.get("estimated_price"),
        "avg_entry_price": review.get("average_cost"),
    }
    blocked = (
        [str(review.get("symbol"))] if review.get("unavailable_reason") == UNAVAILABLE_HOLD else []
    )
    return build_manual_exit_review(row, blocked_symbols=blocked, selected_qty=selected_qty)


def read_live_execution_permitted(path: Path) -> bool:
    """Fail closed when the readiness file is missing or unreadable."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return bool(payload.get("live_execution_permitted"))


def provisional_gain_loss_text(review: Mapping[str, Any]) -> str:
    amount = review.get("gross_provisional_gain_loss")
    if amount in (None, ""):
        return f"{GROSS_LABEL} unavailable"
    return f"{GROSS_LABEL} {amount}"


def _result(
    *,
    status: str,
    request_id: str,
    symbol: str,
    reason: str,
    submitted: bool = False,
    broker_status: str = "not_submitted",
    broker_order_id: str = "",
    idempotent_replay: bool = False,
    selected_qty: str = "",
) -> Dict[str, Any]:
    return {
        "status": status,
        "request_id": request_id,
        "symbol": symbol,
        "selected_qty": selected_qty,
        "reason": reason,
        "submitted": submitted,
        "broker_status": broker_status,
        "broker_order_id": broker_order_id,
        "idempotent_replay": idempotent_replay,
        "orders_submitted": submitted,
    }


def _request_path(store_root: Path, request_id: str) -> Path:
    return store_root / "requests" / f"{request_id}.json"


def _read_receipt(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["idempotent_replay"] = True
    return payload


def _claim(store_root: Path, request_id: str) -> Optional[Dict[str, Any]]:
    path = _request_path(store_root, request_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return _read_receipt(path)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(
            {"status": "in_progress", "request_id": request_id, "submitted": False},
            handle,
        )
    return None


def _save_receipt(store_root: Path, receipt: Dict[str, Any]) -> None:
    path = _request_path(store_root, str(receipt["request_id"]))
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")


def _paper_broker_ok(broker: Any) -> bool:
    if str(getattr(broker, "mode", "") or "").lower() != "paper":
        return False
    base = str(getattr(broker, "base", "") or "")
    return PAPER_HOST in base


def _broker_position(broker: Any, symbol: str) -> tuple[Optional[Decimal], Optional[Decimal]]:
    rows = broker.get_positions()
    for row in rows or []:
        if str(row.get("symbol") or "").upper().strip() != symbol:
            continue
        return _decimal(row.get("qty")), _decimal(row.get("current_price"))
    return None, None


def _install_permit(
    store_root: Path,
    *,
    symbol: str,
    qty: Decimal,
    notional: Decimal,
    request_id: str,
    now: datetime,
) -> PaperSellDispatchPermit:
    scope = [
        {
            "symbol": symbol,
            "action": "EXIT",
            "max_qty": float(qty),
            "max_notional": float(notional),
        }
    ]
    batch_hash = batch_hash_from_scope(scope)
    plan_hash = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    challenge_id = f"manual-chal-{request_id[:24]}"
    authority_id = f"manual-auth-{request_id[:24]}"
    now_iso = now.isoformat()
    expires = (now + timedelta(minutes=10)).isoformat()
    payload = canonical_json(
        {
            "lane": "manual_portfolio_exit",
            "request_id": request_id,
            "reason": REASON,
            "single_use": True,
        }
    )
    db_path = store_root / "manual_portfolio_exit.sqlite3"
    with write_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """
                INSERT INTO sell_challenges(
                    challenge_id, plan_hash, batch_hash, batch_id, canonical_batch_json,
                    payload_json, created_at, expires_at, state, operator_action_label,
                    schema_version
                ) VALUES (?,?,?,?,?,?,?,?, 'OPEN', ?, ?)
                """,
                (
                    challenge_id,
                    plan_hash,
                    batch_hash,
                    f"manual-batch-{request_id[:16]}",
                    canonical_json(scope),
                    payload,
                    now_iso,
                    expires,
                    "MANUAL_PORTFOLIO_EXIT",
                    STORE_SCHEMA_VERSION,
                ),
            )
            conn.execute(
                """
                INSERT INTO sell_authorities(
                    authority_id, challenge_id, plan_hash, batch_hash, canonical_batch_json,
                    payload_json, created_at, expires_at, single_use, state, schema_version
                ) VALUES (?,?,?,?,?,?,?,?,1,'UNUSED',?)
                """,
                (
                    authority_id,
                    challenge_id,
                    plan_hash,
                    batch_hash,
                    canonical_json(scope),
                    payload,
                    now_iso,
                    expires,
                    STORE_SCHEMA_VERSION,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    committed = consume_authority_insert_dispatch(
        db_path,
        authority_id=authority_id,
        plan_hash=plan_hash,
        batch_hash=batch_hash,
        expected_scope=scope,
        mode="paper",
        live_blocked=True,
        requested=[
            {
                "symbol": symbol,
                "action": "EXIT",
                "qty": float(qty),
                "side": "sell",
            }
        ],
        now=now,
    )
    permit = PaperSellDispatchPermit(
        dispatch_intent_id=str(committed["dispatch_id"]),
        idempotency_key=request_id,
        store_root=store_root,
        consumed_durable=True,
        symbols=(symbol,),
        actions=((symbol, "EXIT", float(qty)),),
        permit_token=f"manual-permit-{uuid.uuid4().hex}",
        sqlite_db_path=db_path,
    )
    _write_durable_permit(permit, now=now, require_full_durability=False)
    _ACTIVE_PERMIT.set(permit)
    return permit


def confirm_manual_paper_exit(
    review: Mapping[str, Any],
    *,
    broker: Any,
    store_root: Path,
    live_execution_permitted: bool = True,
    supplied_authority: Optional[Mapping[str, Any]] = None,
    now: Optional[datetime] = None,
    blocked_symbols: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Confirm one reviewed paper exit. A replay of the same request does not submit again."""
    symbol = str(review.get("symbol") or "").upper().strip()
    request_id = str(review.get("request_id") or "")
    selected_text = str(review.get("selected_qty") or "")
    if supplied_authority is not None:
        return _result(
            status="blocked",
            request_id=request_id,
            symbol=symbol,
            selected_qty=selected_text,
            reason="supplied_authority_cannot_authorize_manual_exit",
        )
    if str(review.get("schema") or "") != SCHEMA or not request_id:
        return _result(
            status="rejected",
            request_id=request_id,
            symbol=symbol,
            selected_qty=selected_text,
            reason="manual_exit_request_missing",
        )
    expected = _request_id(
        symbol,
        str(review.get("position_qty") or ""),
        str(review.get("estimated_price") or ""),
        selected_text,
    )
    if expected != request_id:
        return _result(
            status="rejected",
            request_id=request_id,
            symbol=symbol,
            selected_qty=selected_text,
            reason="manual_exit_request_mismatch",
        )
    existing = _request_path(store_root, request_id)
    if existing.is_file():
        return _read_receipt(existing)
    if (
        symbol in blocked_manual_exit_symbols(blocked_symbols)
        or review.get("unavailable_reason") == UNAVAILABLE_HOLD
    ):
        return _result(
            status="blocked",
            request_id=request_id,
            symbol=symbol,
            selected_qty=selected_text,
            reason=UNAVAILABLE_HOLD,
        )
    selected = _decimal(selected_text)
    held = _decimal(review.get("position_qty"))
    price = _decimal(review.get("estimated_price"))
    if selected is None or selected <= 0 or held is None or held <= 0:
        return _result(
            status="rejected",
            request_id=request_id,
            symbol=symbol,
            selected_qty=selected_text,
            reason="quantity_invalid",
        )
    if selected > held:
        return _result(
            status="rejected",
            request_id=request_id,
            symbol=symbol,
            selected_qty=selected_text,
            reason="quantity_above_holdings",
        )
    if price is None or price <= 0:
        return _result(
            status="rejected",
            request_id=request_id,
            symbol=symbol,
            selected_qty=selected_text,
            reason="estimated_price_unavailable",
        )
    notional = selected * price
    if notional > Decimal(str(APPROVAL_CLI_CAP)):
        return _result(
            status="rejected",
            request_id=request_id,
            symbol=symbol,
            selected_qty=selected_text,
            reason="quantity_exceeds_per_order_sell_notional_cap",
        )
    if live_execution_permitted is not False or not _paper_broker_ok(broker):
        return _result(
            status="blocked",
            request_id=request_id,
            symbol=symbol,
            selected_qty=selected_text,
            reason="paper_only_live_execution_blocked",
        )
    try:
        broker_qty, broker_price = _broker_position(broker, symbol)
    except Exception:
        broker_qty, broker_price = None, None
    if (
        broker_qty is None
        or broker_price is None
        or _qty_text(broker_qty) != str(review.get("position_qty") or "")
        or _money(broker_price) != str(review.get("estimated_price") or "")
    ):
        return _result(
            status="stale",
            request_id=request_id,
            symbol=symbol,
            selected_qty=selected_text,
            reason="position_or_price_changed_reopen_exit",
        )
    replay = _claim(store_root, request_id)
    if replay is not None:
        return replay
    moment = now or datetime.now(timezone.utc)
    client_order_id = f"manual-{symbol}-{request_id[:24]}"
    try:
        _install_permit(
            store_root,
            symbol=symbol,
            qty=selected,
            notional=notional,
            request_id=request_id,
            now=moment,
        )
        order = broker.submit_order(
            symbol=symbol,
            qty=_qty_text(selected),
            side="sell",
            client_order_id=client_order_id,
        )
        broker_status = str((order or {}).get("status") or "accepted")
        receipt = _result(
            status="submitted",
            request_id=request_id,
            symbol=symbol,
            selected_qty=selected_text,
            reason=REASON,
            submitted=True,
            broker_status=broker_status,
            broker_order_id=str((order or {}).get("id") or ""),
        )
    except Exception as exc:
        receipt = _result(
            status="rejected",
            request_id=request_id,
            symbol=symbol,
            selected_qty=selected_text,
            reason="broker_submit_rejected",
            broker_status=type(exc).__name__,
        )
    finally:
        _ACTIVE_PERMIT.set(None)
    _save_receipt(store_root, receipt)
    return receipt
