"""Phase 198T-BN2A3 — isolated accounting semantics review."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
import requests

from services.performance_accounting import (
    ACCOUNT_MODEL,
    EVENT_CASH_ADJUSTMENT,
    EVENT_CASH_DEPOSIT,
    EVENT_CASH_WITHDRAWAL,
    EVENT_FEE,
    EVENT_FILL_SELL,
    EVENT_SESSION_CLOSE,
    EVENTS_RELATIVE,
    GROSS_PROVISIONAL,
    IDENTITY_TOLERANCE,
    PerformanceAccountingError,
    UNAVAILABLE,
    accounting_paths,
    append_events,
    build_event,
    build_external_flow,
    identity_residual,
    reduce_events,
    reduce_ledger,
    signed_external_amount,
)
from tests.isolation_guard import ProductionIsolationError
from tests.test_phase_198t_bn2a_performance_accounting import ISO, PROD, _close, _cutover, _fill


@pytest.fixture
def paths(tmp_path: Path):
    return accounting_paths(tmp_path)


def _fee(sale_id: str, activity_id: str, amount: str, component: str, ts: str):
    return build_event(
        EVENT_FEE,
        event_timestamp=ts,
        source="broker_activities",
        body={
            "amount": amount,
            "activity_id": activity_id,
            "fee_component": component,
            "allocates_to_event_id": sale_id,
        },
    )


@pytest.mark.bn2a_isolation
def test_reg_taf_cat_sum_on_one_sale(paths):
    sale = _fill(EVENT_FILL_SELL, "GE", "1", "330", "2026-09-10T14:32:00+00:00", "sell-multi-fee")
    events = [
        _cutover(),
        sale,
        _fee(sale["event_id"], "act-reg", "0.02", "REG", "2026-09-10T18:00:00+00:00"),
        _fee(sale["event_id"], "act-taf", "0.03", "TAF", "2026-09-10T18:00:01+00:00"),
        _fee(sale["event_id"], "act-cat", "0.01", "CAT", "2026-09-10T18:00:02+00:00"),
    ]
    append_events(events, events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    row = state["sales"][0]
    assert [c["fee_component"] for c in row["fee_components"]] == ["REG", "TAF", "CAT"]
    assert Decimal(row["fees_allocated"]) == Decimal("0.06")
    assert row["logical_key"].startswith("fill:")
    assert {ev["logical_key"] for ev in events[2:]} == {
        "fee:act-reg",
        "fee:act-taf",
        "fee:act-cat",
    }
    net = Decimal("330") - Decimal("280.1925") - Decimal("0.06")
    assert Decimal(row["realized_pl"]) == net.quantize(Decimal("0.000001"))
    assert row["realized_pl_kind"] == "NET"
    assert row["fee_reconciliation_complete"] is True


@pytest.mark.bn2a_isolation
def test_delayed_distinct_fees_sum(paths):
    sale = _fill(EVENT_FILL_SELL, "GE", "1", "330", "2026-09-10T14:32:00+00:00", "sell-delayed")
    first = _fee(sale["event_id"], "act-reg-d", "0.02", "REG", "2026-09-10T18:00:00+00:00")
    append_events(
        [_cutover(), sale, first], events_path=paths.events_path, lock_path=paths.lock_path
    )
    later = _fee(sale["event_id"], "act-taf-d", "0.04", "TAF", "2026-09-11T12:00:00+00:00")
    append_events([later], events_path=paths.events_path, lock_path=paths.lock_path)
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert Decimal(state["sales"][0]["fees_allocated"]) == Decimal("0.06")
    assert len(state["sales"][0]["fee_event_ids"]) == 2


@pytest.mark.bn2a_isolation
def test_duplicate_identical_fee_is_idempotent(paths):
    sale = _fill(EVENT_FILL_SELL, "GE", "1", "330", "2026-09-10T14:32:00+00:00", "sell-dup")
    fee = _fee(sale["event_id"], "act-reg-dup", "0.02", "REG", "2026-09-10T18:00:00+00:00")
    append_events([_cutover(), sale, fee], events_path=paths.events_path, lock_path=paths.lock_path)
    again = append_events([fee], events_path=paths.events_path, lock_path=paths.lock_path)
    assert again["skipped_duplicate"] == 1
    replayed = reduce_events([_cutover(), sale, fee, fee])
    state = reduce_ledger(events_path=paths.events_path, lock_path=paths.lock_path)
    assert Decimal(state["sales"][0]["fees_allocated"]) == Decimal("0.02")
    assert replayed["sales"][0]["realized_pl"] == state["sales"][0]["realized_pl"]
    assert len(replayed["sales"][0]["fee_event_ids"]) == 1


@pytest.mark.bn2a_isolation
def test_conflicting_duplicate_fee_fails_closed(paths):
    sale = _fill(EVENT_FILL_SELL, "GE", "1", "330", "2026-09-10T14:32:00+00:00", "sell-conflict")
    fee = _fee(sale["event_id"], "act-reg-c", "0.02", "REG", "2026-09-10T18:00:00+00:00")
    append_events([_cutover(), sale, fee], events_path=paths.events_path, lock_path=paths.lock_path)
    conflict = dict(fee)
    conflict["body"] = dict(fee["body"])
    conflict["body"]["amount"] = "0.99"
    conflict["event_id"] = "b" * 64
    with pytest.raises(PerformanceAccountingError) as exc:
        append_events([conflict], events_path=paths.events_path, lock_path=paths.lock_path)
    assert exc.value.reason == "duplicate_logical_key_conflict"


@pytest.mark.bn2a_isolation
def test_unattributed_date_level_fee_remains_pending(paths):
    sale = _fill(EVENT_FILL_SELL, "GE", "1", "330", "2026-09-10T14:32:00+00:00", "sell-unattr")
    pending = build_event(
        EVENT_FEE,
        event_timestamp="2026-09-10T18:00:00+00:00",
        source="broker_activities",
        body={
            "amount": "1.25",
            "activity_id": "act-date-level",
            "fee_component": "OTHER",
        },
    )
    state = reduce_events([_cutover(), sale, pending])
    assert state["sales"][0]["fees_status"] == "PENDING"
    assert state["sales"][0]["realized_pl"] == UNAVAILABLE
    assert len(state["pending_fees"]) == 1
    assert state["pending_fees"][0]["activity_id"] == "act-date-level"


@pytest.mark.bn2a_isolation
def test_provisional_gross_versus_final_net_realized_pl(paths):
    sale = _fill(EVENT_FILL_SELL, "GE", "1", "330", "2026-09-10T14:32:00+00:00", "sell-gross")
    before = reduce_events([_cutover(), sale])
    row = before["sales"][0]
    assert row["realized_pl"] == UNAVAILABLE
    assert row["realized_pl_kind"] == UNAVAILABLE
    assert row["gross_realized_pl_label"] == GROSS_PROVISIONAL
    expected_gross = Decimal("330") - Decimal("280.1925")
    assert Decimal(row["gross_realized_pl"]) == expected_gross.quantize(Decimal("0.000001"))
    assert before["metrics"]["proven_realized_sale_count"] == 0

    fee = _fee(sale["event_id"], "act-reg-g", "0.05", "REG", "2026-09-10T18:00:00+00:00")
    after = reduce_events([_cutover(), sale, fee])
    done = after["sales"][0]
    assert done["gross_realized_pl"] == row["gross_realized_pl"]
    assert done["realized_pl_kind"] == "NET"
    assert Decimal(done["realized_pl"]) == (expected_gross - Decimal("0.05")).quantize(
        Decimal("0.000001")
    )
    assert done["gross_realized_pl_label"] == "GROSS"
    assert after["metrics"]["proven_realized_sale_count"] == 1


@pytest.mark.bn2a_isolation
def test_deposit_and_withdrawal_signs_via_public_builder(paths):
    with pytest.raises(PerformanceAccountingError) as zero:
        build_external_flow(
            EVENT_CASH_DEPOSIT,
            amount="0",
            activity_id="dep-0",
            event_timestamp="2026-09-11T14:00:00+00:00",
            source="broker_activities",
        )
    assert zero.value.reason == "flow_amount_not_positive"
    with pytest.raises(PerformanceAccountingError) as neg:
        build_external_flow(
            EVENT_CASH_WITHDRAWAL,
            amount="-10",
            activity_id="wd-neg",
            event_timestamp="2026-09-11T14:00:00+00:00",
            source="broker_activities",
        )
    assert neg.value.reason == "flow_amount_not_positive"

    dep = build_external_flow(
        EVENT_CASH_DEPOSIT,
        amount="250.00",
        activity_id="dep-abs",
        event_timestamp="2026-09-11T14:00:00+00:00",
        source="broker_activities",
    )
    wd = build_external_flow(
        EVENT_CASH_WITHDRAWAL,
        amount="40.00",
        activity_id="wd-abs",
        event_timestamp="2026-09-11T15:00:00+00:00",
        source="broker_activities",
    )
    assert signed_external_amount(EVENT_CASH_DEPOSIT, "250.00") == Decimal("250.00")
    assert signed_external_amount(EVENT_CASH_WITHDRAWAL, "40.00") == Decimal("-40.00")
    state = reduce_events([_cutover(), dep, wd])
    amounts = {row["event_type"]: Decimal(row["amount"]) for row in state["external_flows"]}
    assert amounts[EVENT_CASH_DEPOSIT] == Decimal("250.00")
    assert amounts[EVENT_CASH_WITHDRAWAL] == Decimal("-40.00")


@pytest.mark.bn2a_isolation
def test_adjustment_requires_direction_and_evidence(paths):
    with pytest.raises(PerformanceAccountingError) as missing:
        build_external_flow(
            EVENT_CASH_ADJUSTMENT,
            amount="5.00",
            activity_id="adj-none",
            event_timestamp="2026-09-11T14:00:00+00:00",
            source="broker_activities",
        )
    assert missing.value.reason == "adjustment_direction_required"
    credit = build_external_flow(
        EVENT_CASH_ADJUSTMENT,
        amount="5.00",
        activity_id="adj-cr",
        event_timestamp="2026-09-11T14:00:00+00:00",
        source="broker_activities",
        direction="credit",
        evidence_category="broker_correction",
    )
    debit = build_external_flow(
        EVENT_CASH_ADJUSTMENT,
        amount="2.00",
        activity_id="adj-db",
        event_timestamp="2026-09-11T15:00:00+00:00",
        source="broker_activities",
        direction="debit",
        evidence_category="journal",
    )
    events = [
        _cutover(),
        _close("2026-09-10T20:00:00+00:00", "2026-09-10", "92358.02", "65663.61"),
        credit,
        debit,
        _close("2026-09-11T20:00:00+00:00", "2026-09-11", "92361.02", "65666.61"),
    ]
    state = reduce_events(events)
    flows = {row["activity_id"]: Decimal(row["amount"]) for row in state["external_flows"]}
    assert flows["adj-cr"] == Decimal("5.00")
    assert flows["adj-db"] == Decimal("-2.00")
    assert state["session_closes"][1]["external_cash_flow_since_prior"] == "3.00"
    assert state["metrics"]["periods"][0]["period_return"] == "0.000000"


@pytest.mark.bn2a_isolation
def test_cent_level_account_reconciliation():
    assert IDENTITY_TOLERANCE == Decimal("0.01")
    assert identity_residual("65000.00", "27000.00", "92000.01") == Decimal("0.01")
    event = build_event(
        EVENT_SESSION_CLOSE,
        event_timestamp="2026-09-10T20:00:00+00:00",
        session_date="2026-09-10",
        source="test_certified_close",
        body={
            "session_date": "2026-09-10",
            "close_kind": "regular_close",
            "cash": "65000.00",
            "market_value": "27000.00",
            "equity": "92000.01",
            "account_structure": "long_only_paper",
            "positions": [],
        },
    )
    state = reduce_events([_cutover(), event])
    assert Decimal(state["session_closes"][0]["identity_residual"]) <= IDENTITY_TOLERANCE
    assert state["session_closes"][0]["account_model"] == ACCOUNT_MODEL
    with pytest.raises(PerformanceAccountingError) as exc:
        build_event(
            EVENT_SESSION_CLOSE,
            event_timestamp="2026-09-10T20:00:00+00:00",
            session_date="2026-09-10",
            source="test_certified_close",
            body={
                "session_date": "2026-09-10",
                "close_kind": "regular_close",
                "cash": "65000.00",
                "market_value": "27000.00",
                "equity": "92000.02",
                "positions": [],
            },
        )
    assert exc.value.reason == "cash_mv_equity_mismatch"


@pytest.mark.bn2a_isolation
def test_unsupported_account_structure_rejected():
    with pytest.raises(PerformanceAccountingError) as short:
        build_event(
            EVENT_SESSION_CLOSE,
            event_timestamp="2026-09-10T20:00:00+00:00",
            session_date="2026-09-10",
            source="test_certified_close",
            body={
                "session_date": "2026-09-10",
                "close_kind": "regular_close",
                "cash": "100000.00",
                "market_value": "-8000.00",
                "equity": "92000.00",
                "positions": [
                    {
                        "symbol": "GE",
                        "qty": "-1",
                        "market_value": "-8000",
                        "average_entry_price": "1",
                    }
                ],
            },
        )
    assert short.value.reason == "unsupported_short_position"
    with pytest.raises(PerformanceAccountingError) as margin:
        build_event(
            EVENT_SESSION_CLOSE,
            event_timestamp="2026-09-10T20:00:00+00:00",
            session_date="2026-09-10",
            source="test_certified_close",
            body={
                "session_date": "2026-09-10",
                "close_kind": "regular_close",
                "cash": "65000.00",
                "market_value": "27000.00",
                "equity": "92000.00",
                "account_structure": "margin",
                "positions": [],
            },
        )
    assert margin.value.reason == "unsupported_account_structure"


@pytest.mark.bn2a_isolation
def test_scoped_guard_blocks_production_and_broker():
    ledger = PROD / "data" / "results" / "paper_order_events.jsonl"
    with pytest.raises(ProductionIsolationError):
        ledger.open("r", encoding="utf-8")
    with pytest.raises(ProductionIsolationError, match="forbidden_broker_contact"):
        requests.get("https://paper-api.alpaca.markets/v2/account")
    assert not (ISO / EVENTS_RELATIVE).exists()


def test_unrelated_tests_do_not_activate_isolation_guard(tmp_path: Path):
    """Control: unmarked tests keep real open/request behavior."""
    assert getattr(requests.sessions.Session.request, "__name__", "") != "_block_broker"
    scratch = tmp_path / "unrelated-open-ok.txt"
    scratch.write_text("unrelated-open-ok\n", encoding="utf-8")
    assert scratch.read_text(encoding="utf-8") == "unrelated-open-ok\n"
