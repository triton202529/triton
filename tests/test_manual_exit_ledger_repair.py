"""Temporary-path coverage for the manual-exit ledger repair and dry-run recovery."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from services.manual_exit_ledger_recovery import (
    GOOGL_BROKER_ORDER_ID,
    TARGET_AUTHORITY_ID,
    TARGET_BROKER_ORDER_ID,
    TARGET_CLIENT_ORDER_ID,
    TARGET_CREATED_AT,
    TARGET_DISPATCH_ID,
    TARGET_FILLED_AT,
    TARGET_FILL_PRICE,
    TARGET_QTY,
    TARGET_REQUEST_ID,
    TARGET_SUBMITTED_AT,
    TARGET_UPDATED_AT,
    dry_run_october15_bito_recovery,
    execute_october15_bito_recovery,
    main,
)
from services.manual_portfolio_exit import (
    UNAVAILABLE_HOLD,
    build_manual_exit_review,
    confirm_manual_paper_exit,
)
from services.paper_order_ledger import (
    EVENT_ACTION_SUBMITTED,
    EVENT_ORDER_TERMINAL_RECONCILED,
    load_events,
    reconcile_from_recent_orders,
)
from services.paper_sell_arm_runtime_enforcement import enforce_paper_sell_broker_submission
from services.basis_unproven_hold import enforce_basis_unproven_hold_on_order_post
from services.performance_accounting import UNAVAILABLE, reduce_events

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = (
    ROOT / "data" / "results" / "paper_order_events.jsonl",
    ROOT / "data" / "results" / "performance_accounting_events.jsonl",
    ROOT / "data" / "ledger" / "ledger.parquet",
    ROOT / "data" / "governance" / "basis_unproven_hold_audit.jsonl",
    ROOT / "data" / "governance" / "basis_unproven_hold.json",
)


def _fingerprint(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


@pytest.fixture(scope="module", autouse=True)
def production_files_unchanged():
    before = {path: _fingerprint(path) for path in PRODUCTION}
    yield
    after = {path: _fingerprint(path) for path in PRODUCTION}
    assert after == before


class Broker:
    def __init__(self, symbol="AAPL", qty="10", price="20", status="pending_new"):
        self.mode = "paper"
        self.base = "https://paper-api.alpaca.markets"
        self.symbol = symbol
        self.positions = [{"symbol": symbol, "qty": qty, "current_price": price}]
        self.orders = []
        self.status = status

    def get_positions(self):
        return list(self.positions)

    def submit_order(self, **kwargs):
        enforce_paper_sell_broker_submission(
            mode=self.mode,
            side=kwargs["side"],
            source="broker_alpaca.submit_order",
            symbol=kwargs["symbol"],
            qty=float(kwargs["qty"]),
            client_order_id=kwargs.get("client_order_id"),
        )
        enforce_basis_unproven_hold_on_order_post(
            {
                "symbol": kwargs["symbol"],
                "side": kwargs["side"],
                "qty": kwargs["qty"],
                "client_order_id": kwargs.get("client_order_id"),
            },
            source="broker_alpaca._post",
        )
        self.orders.append(kwargs)
        return {
            "id": "ord-manual-1",
            "client_order_id": kwargs["client_order_id"],
            "status": self.status,
            "submitted_at": "2026-10-15T16:14:33.495870609Z",
        }


def _store(root: Path) -> Path:
    return root / "data" / "governance" / "manual_portfolio_exit"


def _confirm(root: Path, broker: Broker, review: dict) -> dict:
    return confirm_manual_paper_exit(
        review,
        broker=broker,
        store_root=_store(root),
        ledger_root=root,
        live_execution_permitted=False,
        blocked_symbols=[],
    )


def _review(symbol="AAPL", qty="10", price="20"):
    return build_manual_exit_review(
        {"symbol": symbol, "qty": qty, "current_price": price, "avg_entry_price": "12"},
        blocked_symbols=[],
    )


def _dispatch(root: Path):
    path = _store(root) / "manual_portfolio_exit.sqlite3"
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return con.execute("SELECT state, broker_ref FROM sell_dispatches").fetchone()
    finally:
        con.close()


def _order_events(root: Path):
    path = root / "data" / "results" / "paper_order_events.jsonl"
    return load_events(path) if path.is_file() else []


def _target_row(**overrides) -> dict:
    row = {
        "snapshot_ts": "2026-10-15T23:39:00Z",
        "id": TARGET_BROKER_ORDER_ID,
        "client_order_id": TARGET_CLIENT_ORDER_ID,
        "created_at": TARGET_CREATED_AT,
        "updated_at": TARGET_UPDATED_AT,
        "submitted_at": TARGET_SUBMITTED_AT,
        "filled_at": TARGET_FILLED_AT,
        "symbol": "BITO",
        "qty": "450",
        "filled_qty": "450",
        "filled_avg_price": "11.01",
        "side": "sell",
        "status": "filled",
    }
    row.update(overrides)
    return row


def _write_snapshot(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_manual_submit_records_submission_identity_and_broker_reference(tmp_path):
    broker = Broker()
    receipt = _confirm(tmp_path, broker, _review())
    assert receipt["status"] == "submitted"
    assert receipt["broker_status"] == "pending_new"
    assert receipt["broker_order_id"] == "ord-manual-1"
    assert receipt["reconciliation_state"] == "submission_recorded"
    assert len(broker.orders) == 1
    state, broker_ref = _dispatch(tmp_path)
    assert state == "SUBMITTED"
    assert broker_ref == "ord-manual-1"
    events = _order_events(tmp_path)
    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == EVENT_ACTION_SUBMITTED
    assert event["logical_key"] == "broker:ord-manual-1"
    assert event["broker_order_id"] == "ord-manual-1"
    assert event["client_order_id"] == broker.orders[0]["client_order_id"]
    assert event["side"] == "SELL"
    assert event["terminal_status_proven"] is False
    assert event["authority_id"].startswith("manual-auth-")
    assert event["dispatch_id"]


def test_snapshot_appends_one_terminal_event_and_one_fill(tmp_path):
    broker = Broker()
    review = _review()
    receipt = _confirm(tmp_path, broker, review)
    recent = tmp_path / "recent_orders.csv"
    _write_snapshot(
        recent,
        {
            "snapshot_ts": "2026-10-15T23:39:00Z",
            "id": receipt["broker_order_id"],
            "client_order_id": receipt["client_order_id"],
            "symbol": "AAPL",
            "side": "sell",
            "qty": "10",
            "filled_qty": "10",
            "filled_avg_price": "20.5",
            "status": "filled",
            "filled_at": "2026-10-15T16:20:00Z",
            "submitted_at": "2026-10-15T16:19:00Z",
        },
    )
    first = reconcile_from_recent_orders(root=tmp_path, recent_path=recent)
    second = reconcile_from_recent_orders(root=tmp_path, recent_path=recent)
    events = _order_events(tmp_path)
    terminals = [e for e in events if e["event_type"] == EVENT_ORDER_TERMINAL_RECONCILED]
    assert first["appended"] == 1
    assert len(terminals) == 1
    assert terminals[0]["broker_terminal_status"] == "FILLED"
    assert terminals[0]["terminal_status_proven"] is True
    assert second["appended"] == 0
    accounting = (tmp_path / "data" / "results" / "performance_accounting_events.jsonl").read_text(
        encoding="utf-8"
    )
    fills = [json.loads(line) for line in accounting.splitlines()]
    assert [row["event_type"] for row in fills] == ["FILL_SELL"]
    body = fills[0]["body"]
    assert body["broker_order_id"] == "ord-manual-1"
    assert "realized_pl" not in body
    assert "fee" not in body
    assert "commission" not in body
    reduced = reduce_events(fills)
    sale = reduced["sales"][0]
    assert sale["realized_pl"] == UNAVAILABLE
    assert sale["basis_consumed"] == UNAVAILABLE
    assert sale["fees_allocated"] == UNAVAILABLE
    assert reduced["metrics"]["proven_realized_sale_count"] == 0
    saved = json.loads(
        (_store(tmp_path) / "requests" / f"{receipt['request_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["broker_status"] == "filled"
    assert saved["reconciliation_state"] == "terminal_reconciled"
    assert saved["status"] == "submitted"


def test_retry_does_not_post_again_or_duplicate_the_event(tmp_path):
    broker = Broker()
    review = _review()
    _confirm(tmp_path, broker, review)
    again = _confirm(tmp_path, broker, review)
    assert again["idempotent_replay"] is True
    assert len(broker.orders) == 1
    assert len(_order_events(tmp_path)) == 1


def test_post_submit_write_failure_is_unreconciled_and_blocks_resubmit(tmp_path, monkeypatch):
    def fail_write(**kwargs):
        raise RuntimeError("injected local write failure")

    monkeypatch.setattr(
        "services.manual_portfolio_exit.persist_manual_submission_identity",
        fail_write,
    )
    broker = Broker()
    review = _review()
    first = _confirm(tmp_path, broker, review)
    assert first["status"] == "repair_needed"
    assert first["reconciliation_state"] == "unreconciled"
    assert first["broker_order_id"] == "ord-manual-1"
    assert first["client_order_id"] == broker.orders[0]["client_order_id"]
    identity = json.loads(
        (_store(tmp_path) / "order_identities" / "ord-manual-1.json").read_text(encoding="utf-8")
    )
    assert identity["client_order_id"] == first["client_order_id"]
    assert identity["reconciliation_state"] == "unreconciled"
    second = _confirm(tmp_path, broker, review)
    assert second["status"] == "repair_needed"
    assert second["idempotent_replay"] is True
    assert len(broker.orders) == 1
    assert _order_events(tmp_path) == []


def test_dry_run_recovery_proposes_exact_bito_events_without_writing(tmp_path):
    parquet = tmp_path / "data" / "ledger" / "ledger.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"do-not-touch")
    orders = tmp_path / "data" / "results" / "paper_order_events.jsonl"
    accounting = tmp_path / "data" / "results" / "performance_accounting_events.jsonl"
    recent = tmp_path / "recent_orders.csv"
    _write_snapshot(recent, _target_row())
    before = {
        parquet: parquet.read_bytes(),
        orders: orders.exists(),
        accounting: accounting.exists(),
    }
    report = dry_run_october15_bito_recovery(
        TARGET_BROKER_ORDER_ID, recent_path=recent, root=tmp_path
    )
    assert report["ok"] is True
    assert report["wrote"] is False
    submitted = report["action_submitted"]
    terminal = report["order_terminal_reconciled"]
    fill = report["fill_sell"]
    assert submitted["event_type"] == EVENT_ACTION_SUBMITTED
    assert submitted["logical_key"] == f"broker:{TARGET_BROKER_ORDER_ID}"
    assert submitted["authority_id"] == TARGET_AUTHORITY_ID
    assert submitted["dispatch_id"] == TARGET_DISPATCH_ID
    assert submitted["client_order_id"] == TARGET_CLIENT_ORDER_ID
    assert Decimal(str(submitted["selected_qty"])) == TARGET_QTY
    assert terminal["event_type"] == EVENT_ORDER_TERMINAL_RECONCILED
    assert terminal["logical_key"] == f"broker:{TARGET_BROKER_ORDER_ID}"
    assert terminal["broker_terminal_status"] == "FILLED"
    assert Decimal(str(terminal["filled_qty"])) == TARGET_QTY
    assert Decimal(str(terminal["average_fill_price"])) == TARGET_FILL_PRICE
    assert TARGET_FILLED_AT[:19] in str(terminal["filled_timestamp"])
    assert fill["event_type"] == "FILL_SELL"
    assert fill["logical_key"].startswith(f"fill:{TARGET_BROKER_ORDER_ID}:")
    assert fill["body"]["symbol"] == "BITO"
    assert "realized_pl" not in fill["body"]
    assert report["fee"] is None
    assert report["net_realized_pl"] is None
    assert report["realized_pl"] is None
    assert report["proposed_receipt"]["broker_status"] == "filled"
    assert report["proposed_dispatch"]["state"] == "SUBMITTED"
    assert report["proposed_dispatch"]["broker_ref"] == TARGET_BROKER_ORDER_ID
    assert parquet.read_bytes() == before[parquet]
    assert orders.exists() is False
    assert accounting.exists() is False
    code = main(
        [
            "--broker-order-id",
            TARGET_BROKER_ORDER_ID,
            "--recent-orders",
            str(recent),
            "--root",
            str(tmp_path),
        ]
    )
    assert code == 0
    refused = main(["--write", "--broker-order-id", TARGET_BROKER_ORDER_ID])
    assert refused == 2
    blocked = execute_october15_bito_recovery(
        TARGET_BROKER_ORDER_ID,
        operator_authorization="separate-operator-authorization",
        target_confirmation=TARGET_BROKER_ORDER_ID,
    )
    assert blocked["wrote"] is False
    assert blocked["reason"] == "write_mode_not_enabled"
    assert parquet.read_bytes() == b"do-not-touch"


@pytest.mark.parametrize(
    ("order_id", "overrides"),
    [
        ("other-order", {}),
        (GOOGL_BROKER_ORDER_ID, {}),
        (TARGET_BROKER_ORDER_ID, {"symbol": "GOOGL"}),
        (TARGET_BROKER_ORDER_ID, {"qty": "449", "filled_qty": "449"}),
        (TARGET_BROKER_ORDER_ID, {"filled_at": "2026-10-15T16:14:34Z"}),
    ],
)
def test_recovery_fails_closed_for_wrong_target(tmp_path, order_id, overrides):
    recent = tmp_path / "recent.csv"
    row = _target_row(**overrides)
    if order_id != TARGET_BROKER_ORDER_ID:
        row["id"] = order_id
    _write_snapshot(recent, row)
    report = dry_run_october15_bito_recovery(order_id, recent_path=recent, root=tmp_path)
    assert report["ok"] is False
    assert report["wrote"] is False
    assert report["reason"] in {"not_recovery_target", "target_mismatch"}


def test_recovery_fails_closed_when_a_target_event_exists(tmp_path):
    from services.paper_order_ledger import append_events, manual_exit_submission_event

    recent = tmp_path / "recent.csv"
    _write_snapshot(recent, _target_row())
    event = manual_exit_submission_event(
        symbol="BITO",
        qty=TARGET_QTY,
        client_order_id=TARGET_CLIENT_ORDER_ID,
        broker_order_id=TARGET_BROKER_ORDER_ID,
        authority_id=TARGET_AUTHORITY_ID,
        dispatch_id=TARGET_DISPATCH_ID,
        event_timestamp=TARGET_SUBMITTED_AT,
    )
    append_events([event], root=tmp_path)
    report = dry_run_october15_bito_recovery(
        TARGET_BROKER_ORDER_ID, recent_path=recent, root=tmp_path
    )
    assert report["reason"] == "target_event_exists"
    assert "ACTION_SUBMITTED" in report["details"]["events"]
    assert report["wrote"] is False


def test_voo_and_xlk_remain_blocked(tmp_path):
    broker = Broker(symbol="VOO", qty="2", price="700")
    for symbol in ("VOO", "XLK"):
        review = build_manual_exit_review(
            {"symbol": symbol, "qty": "2", "current_price": "700", "avg_entry_price": "600"},
            blocked_symbols=["VOO", "XLK"],
        )
        result = confirm_manual_paper_exit(
            review,
            broker=broker,
            store_root=_store(tmp_path) / symbol,
            ledger_root=tmp_path / symbol,
            live_execution_permitted=False,
            blocked_symbols=["VOO", "XLK"],
        )
        assert result["status"] == "blocked"
        assert result["reason"] == UNAVAILABLE_HOLD
    assert broker.orders == []
    assert _order_events(tmp_path) == []
