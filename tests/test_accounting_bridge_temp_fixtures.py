"""Temporary-fixture tests for the paper placement → ledger → accounting bridge.

These tests never open a broker, never write the production ledgers, and never
reconstruct October 6 history.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from services.paper_order_ledger import (
    EVENT_ACTION_SUBMITTED,
    SOURCE_PLACEMENT,
    SOURCE_PREPLAN_SELL,
    load_events,
    reconcile_from_recent_orders,
    record_placement_submission,
    record_sell_dispatch_events,
    reduce_ledger,
)
from services.performance_accounting import (
    EVENTS_RELATIVE,
    PerformanceAccountingError,
    load_events as load_accounting_events,
    record_proven_terminal_fills,
    reduce_events as reduce_accounting_events,
)

PRODUCTION = Path(__file__).resolve().parents[1]
PROTECTED = (
    PRODUCTION / "data" / "results" / "paper_order_events.jsonl",
    PRODUCTION / "data" / "results" / "performance_accounting_events.jsonl",
    PRODUCTION / "data" / "governance" / "execution_approval_requests.json",
    PRODUCTION / "data" / "governance" / "paper_sell_arm_authority.json",
    PRODUCTION / "data" / "live" / "paper_sell_arm.sqlite3",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture
def production_hashes():
    before = {str(path): _sha(path) for path in PROTECTED}
    yield before
    after = {str(path): _sha(path) for path in PROTECTED}
    assert after == before


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    (tmp_path / "data" / "results").mkdir(parents=True)
    return tmp_path


def _recent(path: Path, rows: str) -> Path:
    path.write_text(
        "id,client_order_id,symbol,side,qty,filled_qty,filled_avg_price,status,submitted_at,filled_at,snapshot_ts\n"
        + rows,
        encoding="utf-8",
    )
    return path


def _terminal(**overrides):
    row = {
        "event_type": "ORDER_TERMINAL_RECONCILED",
        "terminal_status_proven": True,
        "broker_terminal_status": "FILLED",
        "side": "BUY",
        "symbol": "MSFT",
        "broker_order_id": "broker-1",
        "client_order_id": "client-1",
        "filled_qty": "2",
        "average_fill_price": "10",
        "filled_timestamp": "2026-09-04T15:00:00+00:00",
        "broker_fill_id": "fill-a",
    }
    row.update(overrides)
    return row


def test_buy_submission_is_idempotent_and_sell_writer_stays_separate(tmp_root, production_hashes):
    first = record_placement_submission(
        symbol="msft",
        side="buy",
        qty=2,
        client_order_id="c-msft",
        broker_order_id="b-msft",
        mode="paper",
        root=tmp_root,
    )
    second = record_placement_submission(
        symbol="MSFT",
        side="BUY_TO_OPEN",
        qty=9,
        client_order_id="c-msft",
        broker_order_id="b-other",
        mode="paper",
        root=tmp_root,
    )
    assert first["appended"] == 1
    assert second["reason"] == "identity_exists"
    assert second["appended"] == 0
    events = load_events(tmp_root / "data" / "results" / "paper_order_events.jsonl")
    assert len(events) == 1
    assert events[0]["event_type"] == EVENT_ACTION_SUBMITTED
    assert events[0]["source_pipeline"] == SOURCE_PLACEMENT
    assert events[0]["side"] == "BUY"
    assert events[0]["logical_key"] == "broker:b-msft"

    record_sell_dispatch_events(
        authority={"authority_id": "sell-arm-a", "payload": {"trading_date": "2026-09-04"}},
        dispatch_id="sell-dispatch-a",
        clip_audit=[
            {
                "symbol": "GOOGL",
                "action": "EXIT",
                "authorized_quantity": 3,
                "authorized_max_notional": 100,
                "current_reference_price": 10,
                "runtime_quantity": 3,
                "runtime_estimated_notional": 30,
                "outcome": "UNCHANGED",
                "reason": "authorized_quantity_fits_current_price",
            }
        ],
        submissions=[
            {
                "symbol": "GOOGL",
                "qty": 3,
                "client_order_id": "c-googl",
                "broker_order_id": "b-googl",
                "broker_status": "submitted",
            }
        ],
        root=tmp_root,
    )
    blocked = record_placement_submission(
        symbol="GOOGL",
        side="sell",
        qty=3,
        client_order_id="c-googl",
        broker_order_id="b-googl",
        root=tmp_root,
    )
    assert blocked["reason"] == "identity_exists"
    sell_rows = [
        row
        for row in load_events(tmp_root / "data" / "results" / "paper_order_events.jsonl")
        if row.get("symbol") == "GOOGL"
    ]
    assert len(sell_rows) == 1
    assert sell_rows[0]["source_pipeline"] == SOURCE_PREPLAN_SELL


def test_live_mode_and_missing_identity_write_nothing(tmp_root, production_hashes):
    live = record_placement_submission(
        symbol="MSFT",
        side="buy",
        qty=1,
        client_order_id="c1",
        broker_order_id="b1",
        mode="live",
        root=tmp_root,
    )
    missing = record_placement_submission(symbol="MSFT", side="buy", qty=1, root=tmp_root)
    assert live["reason"] == "not_paper"
    assert missing["reason"] == "missing_order_identity"
    assert not (tmp_root / "data" / "results" / "paper_order_events.jsonl").is_file()


def test_snapshot_reconciliation_stays_update_only(tmp_root, production_hashes):
    record_placement_submission(
        symbol="NVDA",
        side="buy",
        qty=4,
        client_order_id="c-nvda",
        broker_order_id="b-nvda",
        root=tmp_root,
        event_timestamp="2026-09-04T14:00:00+00:00",
    )
    recent = _recent(
        tmp_root / "recent.csv",
        "b-nvda,c-nvda,NVDA,buy,4,4,20.5,filled,2026-09-04T14:00:00Z,2026-09-04T14:00:02Z,2026-09-04T23:00:00Z\n"
        "b-old,c-old,IBM,buy,1,1,9,filled,2026-09-01T14:00:00Z,2026-09-01T14:00:01Z,2026-09-04T23:00:00Z\n",
    )
    first = reconcile_from_recent_orders(root=tmp_root, recent_path=recent)
    second = reconcile_from_recent_orders(root=tmp_root, recent_path=recent)
    assert first["appended"] == 1
    assert first["accounting"]["appended"] == 1
    assert second["appended"] == 0
    assert second["accounting"]["appended"] == 0
    symbols = {row.get("symbol") for row in reduce_ledger(root=tmp_root)}
    assert symbols == {"NVDA"}
    accounting = load_accounting_events(tmp_root / EVENTS_RELATIVE)
    assert len(accounting) == 1
    assert accounting[0]["event_type"] == "FILL_BUY"
    assert accounting[0]["logical_key"].startswith("fill:b-nvda:")
    assert "realized_pl" not in accounting[0]["body"]
    state = reduce_accounting_events(accounting)
    assert state["open_positions"]["NVDA"]["qty"] == "4.000000000000"
    assert state["sales"] == []


def test_fill_sell_realized_stays_unavailable_without_basis(tmp_root, production_hashes):
    result = record_proven_terminal_fills(
        [
            _terminal(
                side="SELL",
                symbol="GOOGL",
                broker_fill_id="",
                filled_qty="3",
                average_fill_price="11",
            )
        ],
        root=tmp_root,
    )
    assert result["appended"] == 1
    event = load_accounting_events(tmp_root / EVENTS_RELATIVE)[0]
    assert event["event_type"] == "FILL_SELL"
    assert event["logical_key"] == ("fill:broker-1:2026-09-04T15:00:00+00:00:3.000000000000")
    assert "realized_pl" not in event["body"]
    sale = reduce_accounting_events([event])["sales"][0]
    assert sale["realized_pl"] == "UNAVAILABLE"
    assert sale["gross_realized_pl"] == "UNAVAILABLE"


def test_duplicate_fill_skips_and_conflict_fails_closed(tmp_root, production_hashes):
    first = record_proven_terminal_fills([_terminal()], root=tmp_root)
    replay = record_proven_terminal_fills([_terminal()], root=tmp_root)
    assert first["appended"] == 1
    assert replay["appended"] == 0
    assert replay["skipped_duplicate"] == 1
    with pytest.raises(PerformanceAccountingError) as raised:
        record_proven_terminal_fills(
            [_terminal(average_fill_price="11")],
            root=tmp_root,
        )
    assert raised.value.reason == "duplicate_logical_key_conflict"
    assert len(load_accounting_events(tmp_root / EVENTS_RELATIVE)) == 1


def test_fee_without_activity_id_writes_nothing(tmp_root, production_hashes):
    with pytest.raises(PerformanceAccountingError) as raised:
        record_proven_terminal_fills(
            [_terminal()],
            root=tmp_root,
            fee_amount="0.04",
            fee_activity_id="",
        )
    assert raised.value.reason == "fee_activity_id_required"
    assert not (tmp_root / EVENTS_RELATIVE).is_file()


def test_partial_fill_does_not_enter_accounting(tmp_root, production_hashes):
    result = record_proven_terminal_fills(
        [
            _terminal(
                event_type="ORDER_STATUS_RECONCILED",
                terminal_status_proven=False,
                broker_terminal_status=None,
                broker_submission_status="PARTIALLY_FILLED",
            )
        ],
        root=tmp_root,
    )
    assert result["reason"] == "no_proven_fills"
    assert not (tmp_root / EVENTS_RELATIVE).is_file()
