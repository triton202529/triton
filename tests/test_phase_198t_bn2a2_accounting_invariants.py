"""Phase 198T-BN2A2 — hardened forward-accounting invariants and isolation."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
import requests

from services.performance_accounting import (
    EVENT_BENCHMARK_CLOSE,
    EVENT_CASH_DEPOSIT,
    EVENT_CASH_WITHDRAWAL,
    EVENT_DIVIDEND,
    EVENT_FEE,
    EVENT_FILL_SELL,
    EVENT_SESSION_CLOSE,
    EVENTS_RELATIVE,
    PerformanceAccountingError,
    UNAVAILABLE,
    accounting_paths,
    append_events,
    build_event,
    build_proposed_cutover_event,
    reduce_events,
    reduce_ledger,
)
from tests.isolation_guard import ISO as GUARDED_ISO
from tests.isolation_guard import ProductionIsolationError
from tests.test_phase_198t_bn2a_performance_accounting import ISO, PROD, _close, _cutover, _fill

pytestmark = pytest.mark.bn2a_isolation


@pytest.fixture
def paths(tmp_path: Path):
    return accounting_paths(tmp_path)


def test_cutover_is_opening_basis_only(paths):
    state = reduce_events([_cutover()])
    cut = state["cutover"]
    assert cut["not_a_return"] is True
    assert cut["not_a_deposit"] is True
    assert cut["not_a_withdrawal"] is True
    assert cut["not_profit_or_loss"] is True
    assert cut["opening_basis_only"] is True
    assert state["metrics"]["historical_return"] is None
    assert state["metrics"]["time_weighted_return"] == UNAVAILABLE
    assert state["external_flows"] == []
    assert state["sales"] == []


def test_session_close_accepted_once_per_xnys_session(paths):
    first = _close("2026-09-10T20:00:00+00:00", "2026-09-10", "92000", "65000")
    second = _close(
        "2026-09-10T20:05:00+00:00",
        "2026-09-10",
        "92000",
        "65000",
        kind="early_close",
    )
    append_events([_cutover(), first], events_path=paths.events_path, lock_path=paths.lock_path)
    with pytest.raises(PerformanceAccountingError) as exc:
        append_events([second], events_path=paths.events_path, lock_path=paths.lock_path)
    assert exc.value.reason == "duplicate_logical_key_conflict"

    forged = dict(second)
    forged["logical_key"] = "session_close:forged-early"
    forged["event_id"] = "a" * 64
    with pytest.raises(PerformanceAccountingError) as red:
        reduce_events([_cutover(), first, forged])
    assert red.value.reason == "duplicate_session_close"


def test_session_close_identity_fails_closed():
    with pytest.raises(PerformanceAccountingError) as exc:
        build_event(
            EVENT_SESSION_CLOSE,
            event_timestamp="2026-09-10T20:00:00+00:00",
            session_date="2026-09-10",
            source="test_certified_close",
            body={
                "session_date": "2026-09-10",
                "close_kind": "regular_close",
                "cash": "65000",
                "market_value": "10000",
                "equity": "92000",
            },
        )
    assert exc.value.reason == "cash_mv_equity_mismatch"


def test_deposits_positive_withdrawals_negative(paths):
    with pytest.raises(PerformanceAccountingError) as dep:
        build_event(
            EVENT_CASH_DEPOSIT,
            event_timestamp="2026-09-11T14:00:00+00:00",
            source="broker_activities",
            body={"amount": "0", "activity_id": "dep-zero"},
        )
    assert dep.value.reason == "deposit_not_positive"
    with pytest.raises(PerformanceAccountingError) as wd:
        build_event(
            EVENT_CASH_WITHDRAWAL,
            event_timestamp="2026-09-11T14:00:00+00:00",
            source="broker_activities",
            body={"amount": "-25", "activity_id": "wd-neg"},
        )
    assert wd.value.reason == "withdrawal_not_positive"

    events = [
        _cutover(),
        build_event(
            EVENT_CASH_DEPOSIT,
            event_timestamp="2026-09-11T14:00:00+00:00",
            source="broker_activities",
            body={"amount": "250.00", "activity_id": "dep-pos"},
        ),
        build_event(
            EVENT_CASH_WITHDRAWAL,
            event_timestamp="2026-09-11T15:00:00+00:00",
            source="broker_activities",
            body={"amount": "40.00", "activity_id": "wd-pos"},
        ),
    ]
    append_events(events, events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    amounts = {row["event_type"]: Decimal(row["amount"]) for row in state["external_flows"]}
    assert amounts[EVENT_CASH_DEPOSIT] == Decimal("250.00")
    assert amounts[EVENT_CASH_WITHDRAWAL] == Decimal("-40.00")


def test_dividends_are_investment_return_not_external_flow(paths):
    events = [
        _cutover(),
        _close("2026-09-10T20:00:00+00:00", "2026-09-10", "92358.02", "65663.61"),
        build_event(
            EVENT_DIVIDEND,
            event_timestamp="2026-09-11T14:00:00+00:00",
            source="broker_activities",
            body={"amount": "12.50", "activity_id": "div-1", "symbol": "GE"},
        ),
        _close("2026-09-11T20:00:00+00:00", "2026-09-11", "92370.52", "65676.11"),
    ]
    append_events(events, events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert state["external_flows"] == []
    assert state["investment_income"][0]["class"] == "investment_return_not_external_flow"
    assert state["session_closes"][1]["external_cash_flow_since_prior"] == "0.00"
    assert state["metrics"]["periods"][0]["period_return"] == "0.000135"


def test_fees_reduce_return_only_once_and_delayed_fee_is_idempotent(paths):
    sale = _fill(EVENT_FILL_SELL, "GE", "1", "330", "2026-09-10T14:32:00+00:00", "sell-fee-once")
    fee = build_event(
        EVENT_FEE,
        event_timestamp="2026-09-10T18:00:00+00:00",
        source="broker_activities",
        body={
            "amount": "0.03",
            "activity_id": "fee-once",
            "allocates_to_event_id": sale["event_id"],
        },
    )
    append_events(
        [_cutover(), sale, fee],
        events_path=paths.events_path,
        lock_path=paths.lock_path,
    )
    again = append_events([fee], events_path=paths.events_path, lock_path=paths.lock_path)
    assert again["skipped_duplicate"] == 1
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    expected = Decimal("330") - Decimal("280.1925") - Decimal("0.03")
    assert Decimal(state["sales"][0]["realized_pl"]) == expected.quantize(Decimal("0.000001"))
    replayed = reduce_events([_cutover(), sale, fee, fee])
    assert replayed["sales"][0]["realized_pl"] == state["sales"][0]["realized_pl"]

    other = build_event(
        EVENT_FEE,
        event_timestamp="2026-09-10T18:05:00+00:00",
        source="broker_activities",
        body={
            "amount": "0.01",
            "activity_id": "fee-second",
            "fee_component": "TAF",
            "allocates_to_event_id": sale["event_id"],
        },
    )
    summed = reduce_events([_cutover(), sale, fee, other])
    expected_sum = Decimal("330") - Decimal("280.1925") - Decimal("0.04")
    assert Decimal(summed["sales"][0]["fees_allocated"]) == Decimal("0.04")
    assert Decimal(summed["sales"][0]["realized_pl"]) == expected_sum.quantize(Decimal("0.000001"))
    assert len(summed["sales"][0]["fee_event_ids"]) == 2


def test_missing_basis_keeps_realized_unavailable(paths):
    sale = _fill(EVENT_FILL_SELL, "ZZZ", "1", "10", "2026-09-10T14:32:00+00:00", "sell-nobasis")
    fee = build_event(
        EVENT_FEE,
        event_timestamp="2026-09-10T18:00:00+00:00",
        source="broker_activities",
        body={
            "amount": "0.01",
            "activity_id": "fee-nobasis",
            "allocates_to_event_id": sale["event_id"],
        },
    )
    state = reduce_events([_cutover(), sale, fee])
    assert state["sales"][0]["realized_pl"] == UNAVAILABLE
    assert state["sales"][0]["applied"] is False


def test_twr_unavailable_until_two_certified_closes(paths):
    append_events(
        [_cutover(), _close("2026-09-10T20:00:00+00:00", "2026-09-10", "92000", "65000")],
        events_path=paths.events_path,
        lock_path=paths.lock_path,
    )
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert state["metrics"]["time_weighted_return"] == UNAVAILABLE
    assert state["metrics"]["exposure_adjusted_return"] == UNAVAILABLE


def test_benchmark_requires_identical_session_dates_and_labels_methodology(paths):
    events = [
        _cutover(),
        _close("2026-09-10T20:00:00+00:00", "2026-09-10", "92000", "65000"),
        _close("2026-09-11T20:00:00+00:00", "2026-09-11", "92100", "65000"),
        build_event(
            EVENT_BENCHMARK_CLOSE,
            event_timestamp="2026-09-10T20:05:00+00:00",
            session_date="2026-09-10",
            source="local_bars",
            body={"symbol": "SPY", "session_date": "2026-09-10", "adj_close": "100"},
        ),
        build_event(
            EVENT_BENCHMARK_CLOSE,
            event_timestamp="2026-09-11T20:05:00+00:00",
            session_date="2026-09-11",
            source="local_bars",
            body={"symbol": "SPY", "session_date": "2026-09-11", "adj_close": "102"},
        ),
    ]
    append_events(events, events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    period = state["metrics"]["periods"][0]
    assert period["spy_return"] == "0.020000"
    assert period["spy_return_kind"] == "TOTAL_RETURN"
    assert state["benchmarks"]["2026-09-10"]["SPY"]["price_field"] == "adj_close"

    mixed = reduce_events(
        [
            _cutover(),
            _close("2026-09-10T20:00:00+00:00", "2026-09-10", "92000", "65000"),
            _close("2026-09-11T20:00:00+00:00", "2026-09-11", "92100", "65000"),
            build_event(
                EVENT_BENCHMARK_CLOSE,
                event_timestamp="2026-09-10T20:05:00+00:00",
                session_date="2026-09-10",
                source="local_bars",
                body={"symbol": "QQQ", "session_date": "2026-09-10", "adj_close": "200"},
            ),
            build_event(
                EVENT_BENCHMARK_CLOSE,
                event_timestamp="2026-09-11T20:06:00+00:00",
                session_date="2026-09-11",
                source="local_bars",
                body={"symbol": "QQQ", "session_date": "2026-09-11", "close": "202"},
            ),
        ]
    )
    assert mixed["metrics"]["periods"][0]["qqq_return"] == UNAVAILABLE
    assert mixed["metrics"]["periods"][0]["qqq_return_kind"] == UNAVAILABLE
    assert mixed["benchmarks"]["2026-09-10"]["QQQ"]["return_kind"] == "TOTAL_RETURN"
    assert mixed["benchmarks"]["2026-09-11"]["QQQ"]["return_kind"] == "PRICE_RETURN"


def test_zero_exposure_adjusted_return_unavailable(paths):
    events = [
        _cutover(),
        _close("2026-09-10T20:00:00+00:00", "2026-09-10", "65000", "65000"),
        _close("2026-09-11T20:00:00+00:00", "2026-09-11", "65100", "65100"),
    ]
    append_events(events, events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert state["session_closes"][0]["gross_exposure"] == "0.000000"
    assert state["metrics"]["time_weighted_return"] != UNAVAILABLE
    assert state["metrics"]["exposure_adjusted_return"] == UNAVAILABLE


def test_account_identifiers_remain_redacted():
    ev = build_event(
        EVENT_CASH_WITHDRAWAL,
        event_timestamp="2026-09-11T15:00:00+00:00",
        source="broker_activities",
        body={
            "amount": "10.00",
            "activity_id": "wd-redact",
            "description": "ACH from PA3Z3PEK9C5N",
            "account_id": "PA3Z3PEK9C5N",
        },
    )
    blob = json.dumps(ev)
    assert "account_id" not in ev["body"]
    assert "***REDACTED***" in ev["body"]["description"]
    assert "PA3Z" not in blob


def test_guards_block_production_artifacts_and_broker():
    ledger = PROD / "data" / "results" / "paper_order_events.jsonl"
    sqlite = PROD / "data" / "live" / "paper_sell_arm.sqlite3"
    iso_ledger = GUARDED_ISO / EVENTS_RELATIVE
    with pytest.raises(ProductionIsolationError):
        ledger.open("r", encoding="utf-8")
    with pytest.raises(ProductionIsolationError):
        ledger.open("a", encoding="utf-8")
    with pytest.raises(ProductionIsolationError):
        sqlite.open("a+b")
    with pytest.raises(ProductionIsolationError):
        iso_ledger.open("w", encoding="utf-8")
    with pytest.raises(ProductionIsolationError, match="forbidden_broker_contact"):
        requests.get("https://paper-api.alpaca.markets/v2/account")
    assert not iso_ledger.exists()


def test_proposed_cutover_event_carries_required_flags():
    event = build_proposed_cutover_event()
    body = event["body"]
    assert body["not_a_deposit"] is True
    assert body["not_a_withdrawal"] is True
    assert body["not_profit_or_loss"] is True
    assert body["not_a_return"] is True
