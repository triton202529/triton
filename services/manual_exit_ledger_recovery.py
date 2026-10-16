"""Dry-run proposal for the October 15 BITO manual-exit ledger gap.

Write mode is not enabled. Calling it, or passing a write flag, does not
append ledgers, update receipts, or touch ledger.parquet.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from services.paper_order_ledger import (
    EVENT_ACTION_SUBMITTED,
    EVENT_ORDER_TERMINAL_RECONCILED,
    PaperOrderLedgerError,
    ledger_path,
    load_events,
    manual_exit_submission_event,
    status_reconciliation_event,
)
from services.performance_accounting import (
    EVENTS_RELATIVE,
    _fill_event_from_terminal,
)

WRITE_MODE_ENABLED = False
TARGET_BROKER_ORDER_ID = "507112c1-8984-4450-8c67-f9d1bca5a8ab"
TARGET_CLIENT_ORDER_ID = "manual-BITO-0f628c831b6e31d43583ceb6"
TARGET_REQUEST_ID = "0f628c831b6e31d43583ceb610e13db1ed32c1bd727ad32e5856d5e6e0a2215f"
TARGET_AUTHORITY_ID = "manual-auth-0f628c831b6e31d43583ceb6"
TARGET_DISPATCH_ID = "sell-dispatch-f40ccbd63cf54e52"
TARGET_SYMBOL = "BITO"
TARGET_QTY = Decimal("450")
TARGET_FILL_PRICE = Decimal("11.01")
TARGET_CREATED_AT = "2026-10-15T16:14:33.48798992Z"
TARGET_SUBMITTED_AT = "2026-10-15T16:14:33.495870609Z"
TARGET_FILLED_AT = "2026-10-15T16:14:33.914668051Z"
TARGET_UPDATED_AT = "2026-10-15T16:14:33.916512203Z"
TARGET_STATUS = "filled"
GOOGL_BROKER_ORDER_ID = "924938e3-813d-4007-9384-b36fb639cd66"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fail(reason: str, **details: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": False, "wrote": False, "reason": reason, "dry_run": True}
    if details:
        payload["details"] = details
    return payload


def _decimal(value: Any) -> Optional[Decimal]:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError):
        return None
    if not number.is_finite():
        return None
    return number


def _load_target_row(path: Path, broker_order_id: str) -> Dict[str, Any]:
    if not path.is_file():
        return _fail("snapshot_missing", path=str(path))
    matches = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("id") or "").strip() == broker_order_id:
                matches.append(row)
    if not matches:
        return _fail("snapshot_row_missing", broker_order_id=broker_order_id)
    if len(matches) != 1:
        return _fail("snapshot_row_duplicated", count=len(matches))
    return {"ok": True, "row": matches[0]}


def _timestamp_mismatches(row: Mapping[str, Any]) -> Dict[str, str]:
    expected = {
        "created_at": TARGET_CREATED_AT,
        "submitted_at": TARGET_SUBMITTED_AT,
        "filled_at": TARGET_FILLED_AT,
        "updated_at": TARGET_UPDATED_AT,
    }
    mismatches = {}
    for field, wanted in expected.items():
        found = str(row.get(field) or "")
        if found != wanted:
            mismatches[field] = found
    return mismatches


def _identity_mismatches(row: Mapping[str, Any]) -> Dict[str, Any]:
    mismatches: Dict[str, Any] = {}
    if str(row.get("client_order_id") or "") != TARGET_CLIENT_ORDER_ID:
        mismatches["client_order_id"] = row.get("client_order_id")
    if str(row.get("symbol") or "").upper() != TARGET_SYMBOL:
        mismatches["symbol"] = row.get("symbol")
    if str(row.get("status") or "").strip().lower() != TARGET_STATUS:
        mismatches["status"] = row.get("status")
    for field, wanted in (("qty", TARGET_QTY), ("filled_qty", TARGET_QTY)):
        found = _decimal(row.get(field))
        if found != wanted:
            mismatches[field] = row.get(field)
    price = _decimal(row.get("filled_avg_price"))
    if price != TARGET_FILL_PRICE:
        mismatches["filled_avg_price"] = row.get("filled_avg_price")
    mismatches.update(_timestamp_mismatches(row))
    return mismatches


def _existing_target_events(root: Path) -> Dict[str, Any]:
    found = []
    orders = ledger_path(root)
    if orders.is_file():
        for event in load_events(orders):
            if str(event.get("broker_order_id") or "") != TARGET_BROKER_ORDER_ID:
                continue
            if event.get("event_type") in {EVENT_ACTION_SUBMITTED, EVENT_ORDER_TERMINAL_RECONCILED}:
                found.append(str(event.get("event_type")))
    accounting = root / EVENTS_RELATIVE
    if accounting.is_file():
        for line in accounting.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            key = str(event.get("logical_key") or "")
            if key.startswith(f"fill:{TARGET_BROKER_ORDER_ID}:"):
                found.append(str(event.get("event_type") or "FILL_SELL"))
    return {"ok": True, "events": found}


def _proposal(row: Mapping[str, Any]) -> Dict[str, Any]:
    from services.paper_order_ledger import _row_from_recent

    submission = manual_exit_submission_event(
        symbol=TARGET_SYMBOL,
        qty=TARGET_QTY,
        client_order_id=TARGET_CLIENT_ORDER_ID,
        broker_order_id=TARGET_BROKER_ORDER_ID,
        authority_id=TARGET_AUTHORITY_ID,
        dispatch_id=TARGET_DISPATCH_ID,
        event_timestamp=TARGET_SUBMITTED_AT,
    )
    normalized = _row_from_recent(row)
    terminal = status_reconciliation_event(
        rec=normalized,
        match={
            "authority_id": TARGET_AUTHORITY_ID,
            "dispatch_id": TARGET_DISPATCH_ID,
            "action_ordinal": None,
            "trading_date": "2026-10-15",
            "symbol": TARGET_SYMBOL,
            "side": "SELL",
            "selected_qty": 450.0,
            "client_order_id": TARGET_CLIENT_ORDER_ID,
            "broker_order_id": TARGET_BROKER_ORDER_ID,
        },
        method="broker_order_id",
        snapshot_ts=str(row.get("snapshot_ts") or "") or None,
        source_artifact="recent_orders.csv",
    )
    fill = _fill_event_from_terminal(terminal)
    if fill is None:
        return _fail("fill_event_not_proposed")
    return {
        "ok": True,
        "wrote": False,
        "dry_run": True,
        "broker_order_id": TARGET_BROKER_ORDER_ID,
        "action_submitted": submission,
        "order_terminal_reconciled": terminal,
        "fill_sell": fill,
        "proposed_receipt": {
            "request_id": TARGET_REQUEST_ID,
            "status": "submitted",
            "broker_status": TARGET_STATUS,
            "broker_order_id": TARGET_BROKER_ORDER_ID,
            "client_order_id": TARGET_CLIENT_ORDER_ID,
            "reconciliation_state": "terminal_reconciled",
        },
        "proposed_dispatch": {
            "dispatch_id": TARGET_DISPATCH_ID,
            "authority_id": TARGET_AUTHORITY_ID,
            "state": "SUBMITTED",
            "broker_ref": TARGET_BROKER_ORDER_ID,
        },
        "fee": None,
        "commission": None,
        "net_realized_pl": None,
        "realized_pl": None,
        "parquet_modified": False,
    }


def dry_run_october15_bito_recovery(
    broker_order_id: str,
    *,
    recent_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Propose the three ledger identities for this exact order. Writes nothing."""
    root_p = Path(root) if root is not None else _repo_root()
    csv_path = (
        Path(recent_path)
        if recent_path is not None
        else root_p / "data" / "results" / "recent_orders.csv"
    )
    supplied = str(broker_order_id or "").strip()
    if supplied == GOOGL_BROKER_ORDER_ID:
        return _fail("not_recovery_target", broker_order_id=supplied, symbol="GOOGL")
    if supplied != TARGET_BROKER_ORDER_ID:
        return _fail("not_recovery_target", broker_order_id=supplied)
    loaded = _load_target_row(csv_path, supplied)
    if not loaded.get("ok"):
        return loaded
    row = loaded["row"]
    mismatches = _identity_mismatches(row)
    if mismatches:
        return _fail("target_mismatch", fields=sorted(mismatches))
    try:
        present = _existing_target_events(root_p)
    except (OSError, json.JSONDecodeError, PaperOrderLedgerError) as exc:
        return _fail("ledger_unreadable", error=type(exc).__name__)
    if present.get("events"):
        return _fail("target_event_exists", events=present["events"])
    return _proposal(row)


def execute_october15_bito_recovery(
    broker_order_id: str = "",
    *,
    operator_authorization: Optional[str] = None,
    target_confirmation: Optional[str] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    """Refuse every write. A future mode must be enabled separately and confirmed."""
    del broker_order_id, _ignored
    authorized = bool(operator_authorization and target_confirmation)
    return {
        "ok": False,
        "wrote": False,
        "dry_run": True,
        "reason": "write_mode_not_enabled",
        "write_mode_enabled": WRITE_MODE_ENABLED,
        "authorization_present": authorized,
        "required": ["separate_operator_authorization", "target_order_confirmation"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    blocked = [
        arg
        for arg in args
        if arg in {"--write", "--execute", "--apply"} or arg.startswith("--write=")
    ]
    if blocked or "--write" in " ".join(args):
        print(json.dumps(execute_october15_bito_recovery(), sort_keys=True))
        return 2
    parser = argparse.ArgumentParser(
        description="Dry-run the October 15 BITO manual-exit recovery."
    )
    parser.add_argument("--broker-order-id", required=True)
    parser.add_argument("--recent-orders", type=Path)
    parser.add_argument("--root", type=Path)
    parsed = parser.parse_args(args)
    report = dry_run_october15_bito_recovery(
        parsed.broker_order_id,
        recent_path=parsed.recent_orders,
        root=parsed.root,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
