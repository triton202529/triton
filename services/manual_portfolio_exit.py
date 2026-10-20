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
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from services.paper_sell_arm_authority import APPROVAL_CLI_CAP
from services.paper_sell_arm_runtime_enforcement import (
    DURABLE_PERMIT_NAME,
    PaperSellDispatchPermit,
    _ACTIVE_PERMIT,
    _write_durable_permit,
)
from services.paper_sell_arm_store import (
    STORE_SCHEMA_VERSION,
    PaperSellArmStoreError,
    batch_hash_from_scope,
    canonical_json,
    close_manual_exit_sqlite,
    consume_authority_insert_dispatch,
    write_connection,
)

REASON = "Manual operator exit / profit-taking"
UNAVAILABLE_HOLD = "Basis-unproven hold"
GROSS_LABEL = "Gross provisional gain/loss"
SCHEMA = "MANUAL-PORTFOLIO-EXIT-REQUEST-1"
PAPER_HOST = "paper-api.alpaca.markets"
STORE_DB_NAME = "manual_portfolio_exit.sqlite3"
DISPATCH_TERMINAL_DIR = "dispatch_terminal"
TERMINAL_RECONCILIATION = "terminal_reconciled"
DISPATCH_TERMINAL_SCHEMA = "MANUAL-EXIT-DISPATCH-TERMINAL-1"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in receipt.items() if key != "idempotent_replay"}
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _identity_path(store_root: Path, broker_order_id: str) -> Path:
    return store_root / "order_identities" / f"{broker_order_id}.json"


def _save_order_identity(store_root: Path, identity: Mapping[str, Any]) -> None:
    broker_order_id = str(identity.get("broker_order_id") or "").strip()
    if not broker_order_id:
        return
    path = _identity_path(store_root, broker_order_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(identity), sort_keys=True), encoding="utf-8")


def persist_manual_submission_identity(
    *,
    store_root: Path,
    ledger_root: Path,
    symbol: str,
    qty: Decimal,
    authority_id: str,
    dispatch_id: str,
    broker_order_id: str,
    client_order_id: str,
    event_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Record the broker reference and ACTION_SUBMITTED. Does not call the broker."""
    from services.paper_order_ledger import record_manual_exit_submission
    from services.paper_sell_arm_store import PaperSellArmStoreError, record_dispatch_outcome

    try:
        record_dispatch_outcome(
            store_root / "manual_portfolio_exit.sqlite3",
            dispatch_id=dispatch_id,
            outcome_state="SUBMITTED",
            broker_ref=broker_order_id,
            result_class="SUBMITTED",
            reconciliation_required=False,
        )
    except PaperSellArmStoreError as exc:
        return {"ok": False, "reason": exc.reason}
    recorded = record_manual_exit_submission(
        symbol=symbol,
        qty=qty,
        client_order_id=client_order_id,
        broker_order_id=broker_order_id,
        authority_id=authority_id,
        dispatch_id=dispatch_id,
        event_timestamp=event_timestamp,
        root=ledger_root,
    )
    if not recorded.get("ok"):
        return {"ok": False, "reason": recorded.get("reason") or "submission_event_not_recorded"}
    if recorded.get("appended", 0) == 0 and recorded.get("reason") != "identity_exists":
        return {"ok": False, "reason": recorded.get("reason") or "submission_event_not_recorded"}
    return {"ok": True, "ledger": recorded}


def note_proven_terminal_receipts(
    events: Sequence[Mapping[str, Any]],
    *,
    root: Path,
) -> Dict[str, Any]:
    """Align every durable manual-exit record after a proven terminal event.

    This does not call the broker. An OPEN challenge or a conflicting identity
    is refused with no writes.
    """
    store = Path(root) / "data" / "governance" / "manual_portfolio_exit"
    request_dir = store / "requests"
    if not request_dir.is_dir():
        return {"ok": True, "updated": 0, "reason": "manual_store_absent"}
    updated = 0
    for event in events:
        if str(event.get("event_type") or "") != "ORDER_TERMINAL_RECONCILED":
            continue
        if event.get("terminal_status_proven") is not True:
            continue
        broker_status = _terminal_broker_text(event)
        broker_id = str(event.get("broker_order_id") or "").strip()
        client_id = str(event.get("client_order_id") or "").strip()
        if not broker_status or not (broker_id or client_id):
            continue
        if client_id and not client_id.startswith("manual-"):
            continue
        for path in sorted(request_dir.glob("*.json")):
            payload = _load_json(path)
            if payload is None:
                continue
            saved_client = str(payload.get("client_order_id") or "")
            if not saved_client.startswith("manual-") and not client_id.startswith("manual-"):
                continue
            broker_match = broker_id and str(payload.get("broker_order_id") or "") == broker_id
            client_match = client_id and saved_client == client_id
            if not broker_match and not client_match:
                continue
            outcome = _align_proven_receipt(
                store,
                payload,
                event,
                broker_status=broker_status,
                broker_id=broker_id or str(payload.get("broker_order_id") or ""),
            )
            if outcome.get("refused"):
                return {
                    "ok": False,
                    "updated": updated,
                    "reason": outcome.get("reason") or "manual_exit_terminal_mismatch",
                }
            updated += int(outcome.get("updated") or 0)
    if updated:
        _release_manual_exit_store(store)
    return {"ok": True, "updated": updated}


def _terminal_broker_text(event: Mapping[str, Any]) -> Optional[str]:
    extra = event.get("extra") if isinstance(event.get("extra"), Mapping) else {}
    raw = str((extra or {}).get("raw_broker_status") or "").strip().lower()
    if raw == "cancelled":
        raw = "canceled"
    if raw in {"filled", "canceled", "rejected", "expired", "replaced"}:
        return raw
    normalized = str(event.get("broker_terminal_status") or "").strip().upper()
    return {
        "FILLED": "filled",
        "CANCELLED": "canceled",
        "CANCELED": "canceled",
        "REJECTED": "rejected",
        "EXPIRED": "expired",
        "REPLACED": "replaced",
    }.get(normalized)


def _same_qty(left: Any, right: Any) -> bool:
    first = _decimal(left)
    second = _decimal(right)
    if first is None or second is None:
        return str(left) == str(right)
    return first == second


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _release_manual_exit_store(store_root: Path) -> Dict[str, Any]:
    db_path = Path(store_root) / STORE_DB_NAME
    if not db_path.is_file():
        return {"ok": True, "removed": [], "reason": "database_absent"}
    try:
        return close_manual_exit_sqlite(db_path)
    except (PaperSellArmStoreError, OSError) as exc:
        reason = getattr(exc, "reason", None) or type(exc).__name__
        return {"ok": False, "removed": [], "reason": reason}


def immutable_readonly_sqlite_uri(db_path: Path) -> str:
    """URI used by the integrity gate. It does not create WAL or SHM files."""
    uri = Path(db_path).resolve().as_posix()
    return f"file:{uri}?mode=ro&immutable=1"


def _immutable_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(immutable_readonly_sqlite_uri(db_path), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _terminal_sqlite_snapshot(
    db_path: Path, *, dispatch_id: str, authority_id: str
) -> Dict[str, Any]:
    try:
        conn = _immutable_connection(db_path)
    except sqlite3.Error as exc:
        return {"refused": True, "reason": "manual_exit_sqlite_unreadable", "error": str(exc)}
    try:
        dispatch = conn.execute(
            "SELECT authority_id, broker_ref, intent_json FROM sell_dispatches WHERE dispatch_id=?",
            (dispatch_id,),
        ).fetchone()
        authority = conn.execute(
            "SELECT state, challenge_id, payload_json FROM sell_authorities WHERE authority_id=?",
            (authority_id,),
        ).fetchone()
        if dispatch is None or authority is None:
            return {"refused": True, "reason": "manual_exit_terminal_incomplete"}
        if str(dispatch["authority_id"]) != authority_id:
            return {"refused": True, "reason": "manual_exit_terminal_mismatch"}
        if str(authority["state"]) != "CONSUMED":
            return {"refused": True, "reason": "manual_exit_terminal_incomplete"}
        challenge = conn.execute(
            "SELECT state, consuming_authority_id FROM sell_challenges WHERE challenge_id=?",
            (authority["challenge_id"],),
        ).fetchone()
    finally:
        conn.close()
    if challenge is None:
        return {"refused": True, "reason": "manual_exit_terminal_incomplete"}
    try:
        intent = json.loads(str(dispatch["intent_json"] or "{}"))
        payload = json.loads(str(authority["payload_json"] or "{}"))
    except json.JSONDecodeError:
        return {"refused": True, "reason": "manual_exit_sqlite_unreadable"}
    if not isinstance(intent, dict):
        intent = {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "refused": False,
        "broker_ref": str(dispatch["broker_ref"] or ""),
        "challenge_state": str(challenge["state"] or ""),
        "consuming_authority_id": str(challenge["consuming_authority_id"] or ""),
        "intent_broker_status": str(intent.get("broker_status") or ""),
        "intent_reconciliation_state": str(intent.get("reconciliation_state") or ""),
        "authority_broker_status": str(payload.get("broker_status") or ""),
        "authority_reconciliation_state": str(payload.get("reconciliation_state") or ""),
        "intent": intent,
        "authority_payload": payload,
    }


def _write_terminal_sqlite(
    db_path: Path,
    *,
    dispatch_id: str,
    authority_id: str,
    broker_order_id: str,
    broker_status: str,
) -> None:
    with write_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            dispatch = conn.execute(
                "SELECT authority_id, broker_ref, intent_json FROM sell_dispatches WHERE dispatch_id=?",
                (dispatch_id,),
            ).fetchone()
            authority = conn.execute(
                "SELECT state, challenge_id, payload_json FROM sell_authorities WHERE authority_id=?",
                (authority_id,),
            ).fetchone()
            if dispatch is None or authority is None:
                raise PaperSellArmStoreError("sqlite_terminal_row_missing")
            if str(dispatch["authority_id"]) != authority_id:
                raise PaperSellArmStoreError("sqlite_dispatch_authority_mismatch")
            current_ref = str(dispatch["broker_ref"] or "")
            if current_ref and current_ref != broker_order_id:
                raise PaperSellArmStoreError("sqlite_broker_ref_mismatch")
            if str(authority["state"]) != "CONSUMED":
                raise PaperSellArmStoreError("sqlite_authority_not_consumed")
            challenge = conn.execute(
                "SELECT state, consuming_authority_id FROM sell_challenges WHERE challenge_id=?",
                (authority["challenge_id"],),
            ).fetchone()
            if challenge is None or str(challenge["state"]) != "CONSUMED":
                raise PaperSellArmStoreError("sqlite_challenge_not_consumed")
            bound = str(challenge["consuming_authority_id"] or "")
            if bound not in {"", authority_id}:
                raise PaperSellArmStoreError("sqlite_challenge_authority_mismatch")
            intent = json.loads(str(dispatch["intent_json"] or "{}"))
            payload = json.loads(str(authority["payload_json"] or "{}"))
            if not isinstance(intent, dict):
                intent = {}
            if not isinstance(payload, dict):
                payload = {}
            intent["broker_status"] = broker_status
            intent["reconciliation_state"] = TERMINAL_RECONCILIATION
            intent["broker_order_id"] = broker_order_id
            payload["broker_status"] = broker_status
            payload["reconciliation_state"] = TERMINAL_RECONCILIATION
            payload["broker_order_id"] = broker_order_id
            conn.execute(
                "UPDATE sell_dispatches SET broker_ref=?, intent_json=? WHERE dispatch_id=?",
                (broker_order_id, canonical_json(intent), dispatch_id),
            )
            conn.execute(
                "UPDATE sell_authorities SET payload_json=? WHERE authority_id=?",
                (canonical_json(payload), authority_id),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _terminal_evidence(
    *,
    payload: Mapping[str, Any],
    broker_id: str,
    broker_status: str,
) -> Dict[str, Any]:
    return {
        "schema": DISPATCH_TERMINAL_SCHEMA,
        "dispatch_id": payload.get("dispatch_id"),
        "authority_id": payload.get("authority_id"),
        "request_id": payload.get("request_id"),
        "broker_order_id": broker_id,
        "client_order_id": payload.get("client_order_id"),
        "symbol": payload.get("symbol"),
        "selected_qty": payload.get("selected_qty"),
        "broker_status": broker_status,
        "reconciliation_state": TERMINAL_RECONCILIATION,
        "close_gate_evidence": True,
    }


def _retire_matching_permit(
    store: Path,
    *,
    dispatch_id: str,
    request_id: str,
    broker_id: str,
    broker_status: str,
) -> None:
    path = store / DURABLE_PERMIT_NAME
    if not path.is_file():
        return
    current = _load_json(path)
    if current is None:
        return
    if str(current.get("dispatch_intent_id") or "") != dispatch_id:
        return
    retired = {
        "schema_version": current.get("schema_version"),
        "status": "NON_AUTHORITATIVE",
        "json_role": "NON_AUTHORITATIVE",
        "close_gate_evidence": False,
        "authorizes_submission": False,
        "replay_forbidden": True,
        "dispatch_intent_id": dispatch_id,
        "idempotency_key": request_id,
        "broker_order_id": broker_id,
        "broker_status": broker_status,
        "reconciliation_state": TERMINAL_RECONCILIATION,
    }
    path.write_text(json.dumps(retired, sort_keys=True), encoding="utf-8")


def _permit_is_retired(store: Path, dispatch_id: str) -> bool:
    path = store / DURABLE_PERMIT_NAME
    if not path.is_file():
        return True
    current = _load_json(path)
    if current is None:
        return False
    if str(current.get("dispatch_intent_id") or "") != dispatch_id:
        return True
    return (
        current.get("status") == "NON_AUTHORITATIVE"
        and current.get("json_role") == "NON_AUTHORITATIVE"
        and current.get("close_gate_evidence") is False
        and current.get("authorizes_submission") is False
        and "permit_token" not in current
    )


def _align_proven_receipt(
    store: Path,
    payload: Dict[str, Any],
    event: Mapping[str, Any],
    *,
    broker_status: str,
    broker_id: str,
) -> Dict[str, Any]:
    if not broker_id:
        return {"refused": True, "reason": "manual_exit_terminal_incomplete", "updated": 0}
    identity_path = _identity_path(store, broker_id)
    identity = _load_json(identity_path) if identity_path.is_file() else None
    if identity is not None:
        if str(identity.get("broker_order_id") or "") != broker_id:
            return {"refused": True, "reason": "manual_exit_terminal_mismatch", "updated": 0}
        if str(identity.get("symbol") or "") != str(payload.get("symbol") or ""):
            return {"refused": True, "reason": "manual_exit_terminal_mismatch", "updated": 0}
        if not _same_qty(identity.get("selected_qty"), payload.get("selected_qty")):
            return {"refused": True, "reason": "manual_exit_terminal_mismatch", "updated": 0}
    event_symbol = str(event.get("symbol") or "").upper().strip()
    if event_symbol and event_symbol != str(payload.get("symbol") or "").upper().strip():
        return {"refused": True, "reason": "manual_exit_terminal_mismatch", "updated": 0}
    if event.get("filled_qty") not in (None, "") and not _same_qty(
        event.get("filled_qty"), payload.get("selected_qty")
    ):
        return {"refused": True, "reason": "manual_exit_terminal_mismatch", "updated": 0}
    dispatch_id = str(payload.get("dispatch_id") or "")
    authority_id = str(payload.get("authority_id") or "")
    if not dispatch_id or not authority_id:
        return {"refused": True, "reason": "manual_exit_terminal_incomplete", "updated": 0}
    db_path = store / STORE_DB_NAME
    if not db_path.is_file():
        return {"refused": True, "reason": "manual_exit_terminal_incomplete", "updated": 0}
    snapshot = _terminal_sqlite_snapshot(
        db_path, dispatch_id=dispatch_id, authority_id=authority_id
    )
    if snapshot.get("refused"):
        return {"refused": True, "reason": snapshot.get("reason"), "updated": 0}
    if snapshot["broker_ref"] and snapshot["broker_ref"] != broker_id:
        return {"refused": True, "reason": "manual_exit_terminal_mismatch", "updated": 0}
    if snapshot["challenge_state"] != "CONSUMED":
        return {"refused": True, "reason": "manual_exit_challenge_not_consumed", "updated": 0}
    if snapshot["consuming_authority_id"] not in {"", authority_id}:
        return {"refused": True, "reason": "manual_exit_terminal_mismatch", "updated": 0}
    evidence_path = store / DISPATCH_TERMINAL_DIR / f"{dispatch_id}.json"
    evidence = _load_json(evidence_path) if evidence_path.is_file() else None
    already = (
        str(payload.get("broker_status") or "").lower() == broker_status
        and payload.get("reconciliation_state") == TERMINAL_RECONCILIATION
        and identity is not None
        and str(identity.get("broker_status") or "").lower() == broker_status
        and identity.get("reconciliation_state") == TERMINAL_RECONCILIATION
        and snapshot["intent_broker_status"] == broker_status
        and snapshot["intent_reconciliation_state"] == TERMINAL_RECONCILIATION
        and snapshot["authority_broker_status"] == broker_status
        and snapshot["authority_reconciliation_state"] == TERMINAL_RECONCILIATION
        and evidence is not None
        and evidence.get("broker_status") == broker_status
        and evidence.get("reconciliation_state") == TERMINAL_RECONCILIATION
        and _permit_is_retired(store, dispatch_id)
    )
    if already:
        return {"refused": False, "updated": 0}
    receipt = dict(payload)
    receipt["broker_status"] = broker_status
    receipt["reconciliation_state"] = TERMINAL_RECONCILIATION
    if receipt.get("status") == "repair_needed":
        receipt["status"] = "submitted"
    receipt.pop("idempotent_replay", None)
    saved_identity = dict(identity or receipt)
    saved_identity.update(
        {
            "broker_order_id": broker_id,
            "broker_status": broker_status,
            "reconciliation_state": TERMINAL_RECONCILIATION,
            "symbol": receipt.get("symbol"),
            "selected_qty": receipt.get("selected_qty"),
            "request_id": receipt.get("request_id"),
            "client_order_id": receipt.get("client_order_id"),
            "authority_id": authority_id,
            "dispatch_id": dispatch_id,
        }
    )
    evidence_doc = _terminal_evidence(
        payload=receipt, broker_id=broker_id, broker_status=broker_status
    )
    try:
        _write_terminal_sqlite(
            db_path,
            dispatch_id=dispatch_id,
            authority_id=authority_id,
            broker_order_id=broker_id,
            broker_status=broker_status,
        )
    except PaperSellArmStoreError as exc:
        return {"refused": True, "reason": exc.reason, "updated": 0}
    _save_receipt(store, receipt)
    _save_order_identity(store, saved_identity)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence_doc, sort_keys=True), encoding="utf-8")
    _retire_matching_permit(
        store,
        dispatch_id=dispatch_id,
        request_id=str(receipt.get("request_id") or ""),
        broker_id=broker_id,
        broker_status=broker_status,
    )
    return {"refused": False, "updated": 1}


def _gate_failure(failures: list, path: str, reason: str) -> None:
    failures.append({"path": path, "reason": reason})


def _permit_blocks_close(store: Path, failures: list) -> None:
    path = store / DURABLE_PERMIT_NAME
    if not path.is_file():
        return
    payload = _load_json(path)
    if payload is None:
        _gate_failure(failures, DURABLE_PERMIT_NAME, "mutable_pre_submit_permit")
        return
    retired = (
        payload.get("status") == "NON_AUTHORITATIVE"
        and payload.get("json_role") == "NON_AUTHORITATIVE"
        and payload.get("close_gate_evidence") is False
        and payload.get("authorizes_submission") is False
        and "permit_token" not in payload
        and payload.get("status") != "PERMIT_ISSUED_PRE_SUBMIT"
    )
    if not retired:
        _gate_failure(failures, DURABLE_PERMIT_NAME, "mutable_pre_submit_permit")


def _rows_agree(receipt: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    keys = (
        "broker_order_id",
        "broker_status",
        "reconciliation_state",
        "symbol",
        "selected_qty",
        "dispatch_id",
        "authority_id",
    )
    for key in keys:
        if key == "selected_qty":
            if not _same_qty(receipt.get(key), other.get(key)):
                return False
            continue
        if str(receipt.get(key) or "") != str(other.get(key) or ""):
            return False
    return True


def evaluate_manual_exit_close_gate(store_root: Path) -> Dict[str, Any]:
    """Fail closed unless this store is verified manual-exit evidence.

    The certified-close publisher calls this as a read-only check for the
    exact manual-exit directory. A permit export is tolerated only when it is
    explicitly non-authoritative. WAL sidecars, stale statuses, and unrelated
    children fail. This function does not repair or rewrite the store.
    """
    store = Path(store_root)
    failures: list = []
    if not store.is_dir():
        return {
            "ok": False,
            "reason": "manual_exit_store_absent",
            "failures": [{"path": "", "reason": "manual_exit_store_absent"}],
            "verified_broker_order_ids": [],
        }
    allowed_dirs = {"requests", "order_identities", DISPATCH_TERMINAL_DIR}
    for path in sorted(store.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(store).as_posix()
        if rel.endswith(".sqlite3-wal") or rel.endswith(".sqlite3-shm"):
            _gate_failure(failures, rel, "sqlite_sidecar_unexplained")
            continue
        if rel in {STORE_DB_NAME, DURABLE_PERMIT_NAME}:
            continue
        parts = rel.split("/")
        if len(parts) == 2 and parts[0] in allowed_dirs and parts[1].endswith(".json"):
            continue
        _gate_failure(failures, rel, "unrelated_child")
    _permit_blocks_close(store, failures)
    db_path = store / STORE_DB_NAME
    if not db_path.is_file():
        _gate_failure(failures, STORE_DB_NAME, "incomplete_terminal_evidence")
    verified: list = []
    request_dir = store / "requests"
    requests = sorted(request_dir.glob("*.json")) if request_dir.is_dir() else []
    if not requests:
        _gate_failure(failures, "requests", "incomplete_terminal_evidence")
    for path in requests:
        receipt = _load_json(path)
        rel = path.relative_to(store).as_posix()
        if receipt is None:
            _gate_failure(failures, rel, "incomplete_terminal_evidence")
            continue
        broker_id = str(receipt.get("broker_order_id") or "")
        dispatch_id = str(receipt.get("dispatch_id") or "")
        authority_id = str(receipt.get("authority_id") or "")
        broker_status = str(receipt.get("broker_status") or "").lower()
        if (
            not broker_id
            or not dispatch_id
            or not authority_id
            or receipt.get("reconciliation_state") != TERMINAL_RECONCILIATION
            or broker_status not in {"filled", "canceled", "rejected", "expired", "replaced"}
        ):
            _gate_failure(failures, rel, "incomplete_terminal_evidence")
            continue
        identity_path = _identity_path(store, broker_id)
        identity = _load_json(identity_path) if identity_path.is_file() else None
        evidence_path = store / DISPATCH_TERMINAL_DIR / f"{dispatch_id}.json"
        evidence = _load_json(evidence_path) if evidence_path.is_file() else None
        if identity is None or evidence is None or not _rows_agree(receipt, identity):
            _gate_failure(failures, rel, "stale_or_mismatched_state")
            continue
        if not _rows_agree(receipt, evidence):
            _gate_failure(
                failures,
                evidence_path.relative_to(store).as_posix(),
                "stale_or_mismatched_state",
            )
            continue
        if not db_path.is_file():
            continue
        snapshot = _terminal_sqlite_snapshot(
            db_path, dispatch_id=dispatch_id, authority_id=authority_id
        )
        if snapshot.get("refused") or snapshot.get("challenge_state") != "CONSUMED":
            _gate_failure(failures, STORE_DB_NAME, "stale_or_mismatched_state")
            continue
        if snapshot.get("consuming_authority_id") not in {"", authority_id}:
            _gate_failure(failures, STORE_DB_NAME, "stale_or_mismatched_state")
            continue
        if (
            snapshot.get("broker_ref") != broker_id
            or snapshot.get("intent_broker_status") != broker_status
            or snapshot.get("intent_reconciliation_state") != TERMINAL_RECONCILIATION
            or snapshot.get("authority_broker_status") != broker_status
            or snapshot.get("authority_reconciliation_state") != TERMINAL_RECONCILIATION
        ):
            _gate_failure(failures, STORE_DB_NAME, "stale_or_mismatched_state")
            continue
        verified.append(broker_id)
    ok = not failures and bool(verified)
    return {
        "ok": ok,
        "reason": "manual_exit_store_consistent" if ok else failures[0]["reason"],
        "failures": failures,
        "verified_broker_order_ids": verified,
    }


RESIDUAL_SIDECAR_NAMES = frozenset(
    {
        "manual_portfolio_exit.sqlite3-wal",
        "manual_portfolio_exit.sqlite3-shm",
    }
)


def _residual_sidecar_candidate(store: Path, failure: Mapping[str, Any]) -> bool:
    """True only for the empty WAL or the exact SHM left by a checkpoint."""
    if str(failure.get("reason") or "") != "sqlite_sidecar_unexplained":
        return False
    rel = str(failure.get("path") or "")
    if rel not in RESIDUAL_SIDECAR_NAMES:
        return False
    path = store / rel
    if not path.is_file():
        return False
    if rel.endswith(".sqlite3-wal") and path.stat().st_size != 0:
        return False
    return True


def classify_manual_exit_close_preflight(store_root: Path) -> Dict[str, Any]:
    """Run the full close gate, then defer only residual WAL/SHM filenames.

    Publication still uses ``evaluate_manual_exit_close_gate`` unchanged.
    A zero-byte ``manual_portfolio_exit.sqlite3-wal`` and the exact SHM beside
    it can be deferred. Any other child, a nonzero WAL, stale identity, an
    OPEN challenge, a pre-submit permit, or a failed content check stays
    ineligible. This function does not checkpoint or delete anything.
    """
    gate = evaluate_manual_exit_close_gate(store_root)
    failures = list(gate.get("failures") or [])
    verified = list(gate.get("verified_broker_order_ids") or [])
    if gate.get("ok") is True:
        return {
            **gate,
            "deferred_sidecar_cleanup": False,
            "deferred_sidecar_paths": [],
            "content_ok": True,
        }
    store = Path(store_root)
    deferred = [item for item in failures if _residual_sidecar_candidate(store, item)]
    blocking = [item for item in failures if item not in deferred]
    if blocking or not deferred or not verified:
        reason = str((blocking or failures or [{"reason": gate.get("reason")}])[0].get("reason"))
        return {
            "ok": False,
            "reason": reason,
            "failures": failures,
            "verified_broker_order_ids": verified,
            "deferred_sidecar_cleanup": False,
            "deferred_sidecar_paths": [],
            "content_ok": not blocking and bool(verified),
        }
    return {
        "ok": True,
        "reason": "manual_exit_sidecars_deferred",
        "failures": deferred,
        "verified_broker_order_ids": verified,
        "deferred_sidecar_cleanup": True,
        "deferred_sidecar_paths": [str(item.get("path") or "") for item in deferred],
        "content_ok": True,
    }


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


def _record_accepted_manual_order(
    *,
    store_root: Path,
    ledger_root: Path,
    request_id: str,
    symbol: str,
    selected_text: str,
    selected: Decimal,
    authority_id: str,
    dispatch_id: str,
    client_order_id: str,
    order: Mapping[str, Any],
) -> Dict[str, Any]:
    """Persist the accepted order locally. Never submits another broker order."""
    broker_order_id = str(order.get("id") or "").strip()
    broker_status = str(order.get("status") or "accepted")
    saved_client = str(order.get("client_order_id") or client_order_id)
    repair = _result(
        status="repair_needed",
        request_id=request_id,
        symbol=symbol,
        selected_qty=selected_text,
        reason="local_submission_record_failed",
        submitted=True,
        broker_status=broker_status,
        broker_order_id=broker_order_id,
    )
    repair["reconciliation_state"] = "unreconciled"
    repair["client_order_id"] = saved_client
    repair["authority_id"] = authority_id
    repair["dispatch_id"] = dispatch_id
    _save_receipt(store_root, repair)
    _save_order_identity(store_root, repair)
    if not broker_order_id:
        return repair
    try:
        persisted = persist_manual_submission_identity(
            store_root=store_root,
            ledger_root=ledger_root,
            symbol=symbol,
            qty=selected,
            authority_id=authority_id,
            dispatch_id=dispatch_id,
            broker_order_id=broker_order_id,
            client_order_id=saved_client,
            event_timestamp=str(order.get("submitted_at") or order.get("created_at") or "") or None,
        )
    except Exception:
        return repair
    if not persisted.get("ok"):
        return repair
    submitted = _result(
        status="submitted",
        request_id=request_id,
        symbol=symbol,
        selected_qty=selected_text,
        reason=REASON,
        submitted=True,
        broker_status=broker_status,
        broker_order_id=broker_order_id,
    )
    submitted["reconciliation_state"] = "submission_recorded"
    submitted["client_order_id"] = saved_client
    submitted["authority_id"] = authority_id
    submitted["dispatch_id"] = dispatch_id
    _save_receipt(store_root, submitted)
    _save_order_identity(store_root, submitted)
    return submitted


def confirm_manual_paper_exit(
    review: Mapping[str, Any],
    *,
    broker: Any,
    store_root: Path,
    live_execution_permitted: bool = True,
    supplied_authority: Optional[Mapping[str, Any]] = None,
    now: Optional[datetime] = None,
    blocked_symbols: Optional[Sequence[str]] = None,
    ledger_root: Optional[Path] = None,
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
        _release_manual_exit_store(store_root)
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
        _release_manual_exit_store(store_root)
        return replay
    moment = now or datetime.now(timezone.utc)
    client_order_id = f"manual-{symbol}-{request_id[:24]}"
    authority_id = f"manual-auth-{request_id[:24]}"
    books = Path(ledger_root) if ledger_root is not None else Path(__file__).resolve().parents[1]
    order: Optional[Mapping[str, Any]] = None
    dispatch_id = ""
    try:
        permit = _install_permit(
            store_root,
            symbol=symbol,
            qty=selected,
            notional=notional,
            request_id=request_id,
            now=moment,
        )
        dispatch_id = str(permit.dispatch_intent_id)
        order = broker.submit_order(
            symbol=symbol,
            qty=_qty_text(selected),
            side="sell",
            client_order_id=client_order_id,
        )
    except Exception as exc:
        if order is None:
            receipt = _result(
                status="rejected",
                request_id=request_id,
                symbol=symbol,
                selected_qty=selected_text,
                reason="broker_submit_rejected",
                broker_status=type(exc).__name__,
            )
            _save_receipt(store_root, receipt)
            _release_manual_exit_store(store_root)
            return receipt
    finally:
        _ACTIVE_PERMIT.set(None)
    recorded = _record_accepted_manual_order(
        store_root=store_root,
        ledger_root=books,
        request_id=request_id,
        symbol=symbol,
        selected_text=selected_text,
        selected=selected,
        authority_id=authority_id,
        dispatch_id=dispatch_id,
        client_order_id=client_order_id,
        order=order or {},
    )
    _release_manual_exit_store(store_root)
    return recorded
