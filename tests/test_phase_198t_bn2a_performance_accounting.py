"""Phase 198T-BN2A forward performance-accounting foundation."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from services.performance_accounting import (
    COST_BASIS_METHOD,
    EVENT_ACTION_FAILED,
    EVENT_ACTION_REJECTED,
    EVENT_ACTION_SKIPPED,
    EVENT_BENCHMARK_CLOSE,
    EVENT_CASH_DEPOSIT,
    EVENT_FEE,
    EVENT_FILL_BUY,
    EVENT_FILL_SELL,
    EVENT_PRICE_MARKS,
    EVENT_SESSION_CLOSE,
    EVENTS_RELATIVE,
    PerformanceAccountingError,
    PROPOSED_CUTOVER_CASH,
    PROPOSED_CUTOVER_EQUITY,
    SCHEMA_VERSION,
    UNAVAILABLE,
    accounting_paths,
    append_events,
    build_event,
    build_proposed_cutover_event,
    dry_run_proposed_cutover,
    load_events,
    q_money,
    reduce_events,
    reduce_ledger,
)

ISO = Path(__file__).resolve().parents[1]
PROD = Path(r"C:\Users\akimw\triton")
BN1 = Path(r"C:\Users\akimw\AppData\Local\Temp\phase_198T_BN1_20260909_132114")

pytestmark = pytest.mark.bn2a_isolation


@pytest.fixture
def paths(tmp_path: Path):
    return accounting_paths(tmp_path)


def _cutover():
    return build_proposed_cutover_event(source_hashes={"positions_snapshot.csv": "bn1-frozen"})


def _close(ts: str, session_date: str, equity: str, cash: str, kind: str = "regular_close"):
    mv = str(Decimal(equity) - Decimal(cash))
    return build_event(
        EVENT_SESSION_CLOSE,
        event_timestamp=ts,
        session_date=session_date,
        source="test_certified_close",
        body={
            "session_date": session_date,
            "close_kind": kind,
            "cash": cash,
            "market_value": mv,
            "equity": equity,
            "buying_power": "100000",
            "positions": [
                {
                    "symbol": "GE",
                    "qty": "1",
                    "market_value": mv,
                    "average_entry_price": "280.1925",
                    "unrealized_pl": "0",
                }
            ],
            "source_hashes": {"test": "1"},
        },
    )


def _fill(event_type: str, symbol: str, qty: str, price: str, ts: str, order_id: str):
    return build_event(
        event_type,
        event_timestamp=ts,
        source="paper_order_ledger",
        body={
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "broker_order_id": order_id,
            "broker_fill_id": f"{order_id}-fill",
        },
    )


def test_import_creates_no_files_and_skips_broker():
    import services.performance_accounting as mod

    text = Path(mod.__file__).read_text(encoding="utf-8")
    assert "alpaca" not in text.lower()
    assert "requests." not in text
    assert not (PROD / EVENTS_RELATIVE).exists()
    before = list((ISO / "data" / "results").glob("performance_accounting*"))
    assert before == []


def test_planning_records_rejected():
    with pytest.raises(PerformanceAccountingError) as exc:
        build_event(
            EVENT_FILL_BUY,
            event_timestamp="2026-09-10T14:30:00+00:00",
            source="paper_execution_audit",
            body={
                "symbol": "GE",
                "qty": "1",
                "price": "300",
                "broker_order_id": "x",
                "status": "planned",
            },
        )
    assert exc.value.reason == "planning_record_rejected"


def test_cutover_dry_run_matches_bn1_checkpoint(paths):
    result = dry_run_proposed_cutover(
        events_path=paths.events_path,
        lock_path=paths.lock_path,
        source_hashes={"bn1": "frozen"},
    )
    assert result["opening_equity"] == format(PROPOSED_CUTOVER_EQUITY, "f")
    assert result["opening_cash"] == format(PROPOSED_CUTOVER_CASH, "f")
    assert result["open_position_count"] == 8
    assert set(result["open_positions"]) == {
        "ARKK",
        "BITO",
        "DBA",
        "DIA",
        "GE",
        "GS",
        "MA",
        "TSLA",
    }
    assert result["historical_return"] is None
    assert result["time_weighted_return"] == UNAVAILABLE
    assert result["invented_realized_pl"] == 0
    assert result["fees_status"] == "PENDING"
    assert result["earliest_performance_date"] is None
    assert result["cutover_publication_status"] == "PROPOSED_REQUIRES_FRESH_SNAPSHOT"
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["cost_basis_method"] == COST_BASIS_METHOD


def test_duplicate_event_is_idempotent(paths):
    event = _cutover()
    first = append_events([event], events_path=paths.events_path, lock_path=paths.lock_path)
    second = append_events([event], events_path=paths.events_path, lock_path=paths.lock_path)
    assert first["appended"] == 1
    assert second["appended"] == 0
    assert second["skipped_duplicate"] == 1
    assert len(load_events(paths.events_path)) == 1
    a = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    b = reduce_events(load_events(paths.events_path))
    assert a["open_position_count"] == b["open_position_count"] == 8


def test_duplicate_logical_key_conflict(paths):
    event = _cutover()
    append_events([event], events_path=paths.events_path, lock_path=paths.lock_path)
    conflict = dict(event)
    conflict["body"] = dict(event["body"])
    conflict["body"]["publication_status"] = "TAMPERED"
    conflict["event_id"] = "0" * 64
    with pytest.raises(PerformanceAccountingError) as exc:
        append_events([conflict], events_path=paths.events_path, lock_path=paths.lock_path)
    assert exc.value.reason == "duplicate_logical_key_conflict"


def test_buy_updates_moving_average(paths):
    events = [
        _cutover(),
        _fill(EVENT_FILL_BUY, "GE", "1", "300", "2026-09-10T14:31:00+00:00", "buy-ge-1"),
    ]
    append_events(events, events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    ge = state["open_positions"]["GE"]
    assert ge["qty"] == "2.000000000000"
    expected = q_money(Decimal("280.1925") + Decimal("300"))
    assert Decimal(ge["remaining_basis"]) == expected
    assert Decimal(ge["average_entry"]) == q_money(expected / Decimal("2"))


def test_partial_then_full_close_reconciles_basis(paths):
    events = [
        _cutover(),
        _fill(EVENT_FILL_SELL, "GE", "0.4", "330", "2026-09-10T14:32:00+00:00", "sell-ge-p"),
        build_event(
            EVENT_FEE,
            event_timestamp="2026-09-10T18:00:00+00:00",
            source="broker_activities",
            body={
                "amount": "0.02",
                "activity_id": "fee-ge-p",
                "allocates_to_event_id": None,
                "description": "REG fee by PA3Z3PEK9C5N",
            },
        ),
    ]
    # attach fee after we know the sale event id
    events[1] = _fill(EVENT_FILL_SELL, "GE", "0.4", "330", "2026-09-10T14:32:00+00:00", "sell-ge-p")
    sale_id = events[1]["event_id"]
    events[2] = build_event(
        EVENT_FEE,
        event_timestamp="2026-09-10T18:00:00+00:00",
        source="broker_activities",
        body={
            "amount": "0.02",
            "activity_id": "fee-ge-p",
            "allocates_to_event_id": sale_id,
            "description": "REG fee by PA3Z3PEK9C5N",
        },
    )
    assert "***REDACTED***" in events[2]["body"]["description"]
    assert "PA3Z" not in json.dumps(events[2])
    append_events(events, events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    ge = state["open_positions"]["GE"]
    assert Decimal(ge["qty"]) == Decimal("0.6")
    assert Decimal(ge["remaining_basis"]) == q_money(Decimal("0.6") * Decimal("280.1925"))
    sale = state["sales"][0]
    assert sale["fees_status"] == "ATTRIBUTED"
    proceeds = Decimal("0.4") * Decimal("330")
    consumed = Decimal("0.4") * Decimal("280.1925")
    assert Decimal(sale["realized_pl"]) == q_money(proceeds - consumed - Decimal("0.02"))

    close = _fill(EVENT_FILL_SELL, "GE", "0.6", "331", "2026-09-11T14:32:00+00:00", "sell-ge-f")
    fee = build_event(
        EVENT_FEE,
        event_timestamp="2026-09-11T18:00:00+00:00",
        source="broker_activities",
        body={
            "amount": "0.01",
            "activity_id": "fee-ge-f",
            "allocates_to_event_id": close["event_id"],
        },
    )
    append_events([close, fee], events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert "GE" not in state["open_positions"]
    assert Decimal(state["sales"][1]["qty"]) == Decimal("0.6")


def test_sell_without_fee_leaves_realized_unavailable(paths):
    events = [
        _cutover(),
        _fill(EVENT_FILL_SELL, "GE", "1", "330", "2026-09-10T14:32:00+00:00", "sell-ge-nf"),
    ]
    append_events(events, events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert state["sales"][0]["realized_pl"] == UNAVAILABLE
    assert state["sales"][0]["fees_status"] == "PENDING"
    assert state["metrics"]["proven_realized_sale_count"] == 0


def test_missing_basis_does_not_infer_realized(paths):
    events = [
        _cutover(),
        _fill(EVENT_FILL_SELL, "ZZZ", "1", "10", "2026-09-10T14:32:00+00:00", "sell-zzz"),
    ]
    append_events(events, events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert state["sales"][0]["applied"] is False
    assert state["sales"][0]["realized_pl"] == UNAVAILABLE
    assert state["open_position_count"] == 8


def test_failed_rejected_skipped_do_not_change_basis(paths):
    cut = _cutover()
    extras = [
        build_event(
            EVENT_ACTION_FAILED,
            event_timestamp="2026-09-10T14:00:00+00:00",
            source="paper_order_ledger",
            body={"reason_code": "broker_reject", "symbol": "GE"},
        ),
        build_event(
            EVENT_ACTION_REJECTED,
            event_timestamp="2026-09-10T14:01:00+00:00",
            source="paper_order_ledger",
            body={"reason_code": "gate", "symbol": "GE"},
        ),
        build_event(
            EVENT_ACTION_SKIPPED,
            event_timestamp="2026-09-10T14:02:00+00:00",
            source="paper_order_ledger",
            body={"reason_code": "duplicate", "symbol": "GE"},
        ),
    ]
    append_events([cut, *extras], events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert state["open_positions"]["GE"]["qty"] == "1.000000000000"
    assert state["ignored_non_basis_events"] == 3


def test_external_deposit_excluded_from_return(paths):
    events = [
        _cutover(),
        _close("2026-09-10T20:00:00+00:00", "2026-09-10", "92358.02", "65663.61"),
        build_event(
            EVENT_CASH_DEPOSIT,
            event_timestamp="2026-09-11T14:00:00+00:00",
            source="broker_activities",
            body={"amount": "1000.00", "activity_id": "dep-1"},
        ),
        _close("2026-09-11T20:00:00+00:00", "2026-09-11", "93358.02", "66663.61"),
    ]
    append_events(events, events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    period = state["metrics"]["periods"][0]
    assert period["period_return"] == "0.000000"
    assert state["metrics"]["time_weighted_return"] == "0.000000"
    assert state["session_closes"][1]["external_cash_flow_since_prior"] == "1000.00"


def test_two_closes_required_for_return(paths):
    append_events(
        [_cutover(), _close("2026-09-10T20:00:00+00:00", "2026-09-10", "92000", "65000")],
        events_path=paths.events_path,
        lock_path=paths.lock_path,
    )
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert state["metrics"]["earliest_performance_date"] == "2026-09-10"
    assert state["metrics"]["one_period_return_available"] is False
    assert state["metrics"]["time_weighted_return"] == UNAVAILABLE


def test_missing_benchmark_date_is_unavailable(paths):
    events = [
        _cutover(),
        _close("2026-09-10T20:00:00+00:00", "2026-09-10", "92000", "65000"),
        build_event(
            EVENT_BENCHMARK_CLOSE,
            event_timestamp="2026-09-10T20:05:00+00:00",
            session_date="2026-09-10",
            source="local_bars",
            body={"symbol": "SPY", "session_date": "2026-09-10", "close": "765.96"},
        ),
        _close("2026-09-11T20:00:00+00:00", "2026-09-11", "92100", "65000"),
    ]
    append_events(events, events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert state["metrics"]["periods"][0]["spy_return"] == UNAVAILABLE
    assert state["metrics"]["periods"][0]["qqq_return"] == UNAVAILABLE


def test_same_date_spy_qqq_comparison(paths):
    events = [
        _cutover(),
        _close("2026-09-10T20:00:00+00:00", "2026-09-10", "92000", "65000"),
        _close("2026-09-11T20:00:00+00:00", "2026-09-11", "92920", "65000"),
        build_event(
            EVENT_BENCHMARK_CLOSE,
            event_timestamp="2026-09-10T20:05:00+00:00",
            session_date="2026-09-10",
            source="local_bars",
            body={"symbol": "SPY", "session_date": "2026-09-10", "close": "100"},
        ),
        build_event(
            EVENT_BENCHMARK_CLOSE,
            event_timestamp="2026-09-11T20:05:00+00:00",
            session_date="2026-09-11",
            source="local_bars",
            body={"symbol": "SPY", "session_date": "2026-09-11", "close": "101"},
        ),
        build_event(
            EVENT_BENCHMARK_CLOSE,
            event_timestamp="2026-09-10T20:06:00+00:00",
            session_date="2026-09-10",
            source="local_bars",
            body={"symbol": "QQQ", "session_date": "2026-09-10", "close": "200"},
        ),
        build_event(
            EVENT_BENCHMARK_CLOSE,
            event_timestamp="2026-09-11T20:06:00+00:00",
            session_date="2026-09-11",
            source="local_bars",
            body={"symbol": "QQQ", "session_date": "2026-09-11", "close": "202"},
        ),
    ]
    append_events(events, events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert state["metrics"]["periods"][0]["spy_return"] == "0.010000"
    assert state["metrics"]["periods"][0]["qqq_return"] == "0.010000"
    assert state["metrics"]["periods"][0]["spy_return_kind"] == "PRICE_RETURN"
    assert state["metrics"]["periods"][0]["qqq_return_kind"] == "PRICE_RETURN"
    assert state["metrics"]["periods"][0]["period_return"] == "0.010000"


def test_early_close_is_certified(paths):
    events = [
        _cutover(),
        _close("2026-11-27T18:05:00+00:00", "2026-11-27", "92000", "65000", kind="early_close"),
        _close("2026-11-30T21:05:00+00:00", "2026-11-30", "92100", "65000"),
    ]
    append_events(events, events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert state["session_closes"][0]["close_kind"] == "early_close"
    assert state["metrics"]["one_period_return_available"] is True


def test_missing_decision_price_slippage_unavailable(paths):
    fill = _fill(EVENT_FILL_SELL, "GE", "1", "330", "2026-09-10T14:32:00+00:00", "sell-mark")
    marks = build_event(
        EVENT_PRICE_MARKS,
        event_timestamp="2026-09-10T14:31:50+00:00",
        source="paper_order_ledger",
        body={"broker_order_id": "sell-mark", "submit_mark": "329.50"},
    )
    append_events(
        [_cutover(), marks, fill],
        events_path=paths.events_path,
        lock_path=paths.lock_path,
    )
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert state["metrics"]["submit_to_fill_slippage_notional"] == "-0.500000"
    assert state["metrics"]["decision_to_fill_slippage_notional"] == UNAVAILABLE


def test_interrupted_write_fails_closed(paths):
    event = _cutover()
    append_events([event], events_path=paths.events_path, lock_path=paths.lock_path)
    with paths.events_path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version":"198T-BN2A-PERFORMANCE-ACCOUNTING-1"')
    with pytest.raises(PerformanceAccountingError) as exc:
        load_events(paths.events_path)
    assert exc.value.reason == "ledger_incomplete_final_line"


def test_schema_mismatch_rejected(paths):
    event = _cutover()
    event["schema_version"] = "nope"
    with pytest.raises(PerformanceAccountingError) as exc:
        append_events([event], events_path=paths.events_path, lock_path=paths.lock_path)
    assert exc.value.reason == "schema_mismatch"


def test_bn1_evidence_present_and_isolated():
    assert BN1.is_dir()
    assert (BN1 / "10_classification_summary.json").is_file()
    assert ISO.resolve() != PROD.resolve()
    assert ISO.name == "triton-bn1-performance-isolated"
