"""Temporary-fixture tests for FORWARD_OPENING_BASIS.

These tests do not publish an opening, do not write production ledgers, and do
not turn the October 6 GOOGL sale into realized P&L.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path

import pytest

from services.performance_accounting import (
    EVENT_CUTOVER,
    EVENT_FILL_BUY,
    EVENT_FILL_SELL,
    LABEL_BASIS_UNAVAILABLE,
    LABEL_FILL_HISTORY,
    LABEL_FORWARD_OPENING,
    LABEL_HISTORICAL_CUTOVER,
    LABEL_OPENING_EVENT,
    UNAVAILABLE,
    PerformanceAccountingError,
    append_events,
    append_forward_opening_basis,
    build_event,
    build_forward_opening_basis,
    build_proposed_cutover_event,
    q_money,
    reduce_events,
)
from services.performance_publication_authority import (
    COMMAND_FORWARD_OPENING_BASIS,
    PublicationAuthorityError,
    mint_publication_authority,
    passing_git_identity,
    passing_runtime_state,
)

PROD = Path(r"C:\Users\akimw\triton")
LEDGERS = {
    PROD
    / "data/results/paper_order_events.jsonl": "f471049c339a3fc4569487f6654aa98e353c7e1bb76c8fff82194538f32ec7cf",
    PROD
    / "data/results/performance_accounting_events.jsonl": "2777abc9e838333a946263ee0a18347460c0f5f364565f9f9773c1b8a357488a",
}
EVIDENCE_SHA = "ab" * 32


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(autouse=True)
def production_ledgers_unchanged():
    before = {path: _sha(path) for path in LEDGERS}
    assert before == LEDGERS
    yield
    assert {path: _sha(path) for path in LEDGERS} == before


def _opening(symbol: str, qty: str, basis: str, as_of: str, ref: str = "temp-statement"):
    return build_forward_opening_basis(
        symbol=symbol,
        qty=qty,
        opening_basis=basis,
        as_of_utc=as_of,
        evidence_sha256=EVIDENCE_SHA,
        evidence_ref=ref,
    )


def _fill(event_type: str, symbol: str, qty: str, price: str, timestamp: str, order_id: str):
    return build_event(
        event_type,
        event_timestamp=timestamp,
        source="paper_order_ledger",
        body={
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "broker_order_id": order_id,
            "broker_fill_id": f"{timestamp}:{qty}",
        },
    )


def test_october6_googl_sale_stays_unavailable_after_later_opening():
    sale = _fill(
        EVENT_FILL_SELL,
        "GOOGL",
        "3",
        "345.88",
        "2026-10-06T14:16:36.321483+00:00",
        "924938e3-813d-4007-9384-b36fb639cd66",
    )
    opening = _opening("GOOGL", "1", "360.93", "2026-10-07T16:00:00+00:00")
    state = reduce_events([sale, opening])
    assert state["sales"][0]["realized_pl"] == UNAVAILABLE
    assert state["sales"][0]["basis_consumed"] == UNAVAILABLE
    assert state["sales"][0]["reporting_label"] == LABEL_BASIS_UNAVAILABLE
    assert state["sales"][0]["applied"] is False
    assert len(state["sales"]) == 1
    assert state["forward_openings"][0]["reporting_label"] == LABEL_OPENING_EVENT
    assert all(event["event_type"] != EVENT_FILL_BUY for event in (sale, opening))
    assert "GOOGL" in state["open_positions"]


def test_backdated_opening_is_rejected_and_not_written(tmp_path: Path):
    sale = _fill(
        EVENT_FILL_SELL,
        "GOOGL",
        "3",
        "345.88",
        "2026-10-06T14:16:36.321483+00:00",
        "924938e3-813d-4007-9384-b36fb639cd66",
    )
    opening = _opening("GOOGL", "1", "360.93", "2026-10-06T12:00:00+00:00")
    events_path = tmp_path / "performance_accounting_events.jsonl"
    lock_path = tmp_path / "performance_accounting_events.lock"
    append_events([sale], events_path=events_path, lock_path=lock_path)
    before = events_path.read_bytes()
    with pytest.raises(PerformanceAccountingError) as append_exc:
        append_forward_opening_basis(opening, events_path=events_path, lock_path=lock_path)
    assert append_exc.value.reason == "forward_opening_backdated"
    assert events_path.read_bytes() == before


def test_duplicate_opening_conflicts(tmp_path: Path):
    first = _opening("MSFT", "2", "1000", "2026-10-08T15:00:00+00:00")
    second = _opening("MSFT", "2", "1100", "2026-10-08T16:00:00+00:00", ref="other-statement")
    with pytest.raises(PerformanceAccountingError) as exc:
        reduce_events([first, second])
    assert exc.value.reason == "duplicate_forward_opening"
    events_path = tmp_path / "events.jsonl"
    lock_path = tmp_path / "events.lock"
    append_forward_opening_basis(first, events_path=events_path, lock_path=lock_path)
    before = events_path.read_bytes()
    with pytest.raises(PerformanceAccountingError) as append_exc:
        append_forward_opening_basis(second, events_path=events_path, lock_path=lock_path)
    assert append_exc.value.reason == "duplicate_logical_key_conflict"
    assert events_path.read_bytes() == before
    replay = append_forward_opening_basis(first, events_path=events_path, lock_path=lock_path)
    assert replay["appended"] == 0
    assert replay["skipped_duplicate"] == 1
    assert events_path.read_bytes() == before


def test_opening_rejected_when_fill_buy_already_proved_quantity():
    buy = _fill(
        EVENT_FILL_BUY, "NVDA", "4", "242.38", "2026-10-06T15:02:20.952104+00:00", "nvda-buy"
    )
    opening = _opening("NVDA", "4", "969.52", "2026-10-08T15:00:00+00:00")
    with pytest.raises(PerformanceAccountingError) as exc:
        reduce_events([buy, opening])
    assert exc.value.reason == "forward_opening_position_exists"


def test_sale_after_opening_consumes_only_that_opening_basis():
    opening = _opening("ZZ", "1", "10", "2026-10-08T15:00:00+00:00")
    sale = _fill(EVENT_FILL_SELL, "ZZ", "1", "12", "2026-10-08T16:00:00+00:00", "zz-sell")
    state = reduce_events([opening, sale])
    row = state["sales"][0]
    assert row["basis_consumed"] == "10.000000"
    assert row["gross_realized_pl"] == "2.000000"
    assert row["realized_pl"] == UNAVAILABLE
    assert row["reporting_label"] == LABEL_FORWARD_OPENING
    assert row["basis_origin"] == "FORWARD_OPENING_BASIS"
    assert row["applied"] is True
    assert state["open_positions"] == {}


def test_fill_like_and_broker_average_payloads_emit_nothing(tmp_path: Path):
    with pytest.raises(PerformanceAccountingError) as average_exc:
        build_event(
            "FORWARD_OPENING_BASIS",
            event_timestamp="2026-10-08T15:00:00+00:00",
            source="operator_approved_opening_statement",
            body={
                "symbol": "GOOGL",
                "qty": "1",
                "opening_basis": "360.93",
                "as_of_utc": "2026-10-08T15:00:00+00:00",
                "evidence_kind": "broker_average_cost",
                "evidence_sha256": EVIDENCE_SHA,
                "evidence_ref": "snapshot",
                "not_a_fill": True,
                "not_historical_realized_pl": True,
                "not_profit_or_loss": True,
                "not_a_return": True,
            },
        )
    assert average_exc.value.reason == "forward_opening_evidence_rejected"
    with pytest.raises(PerformanceAccountingError) as fill_exc:
        build_event(
            "FORWARD_OPENING_BASIS",
            event_timestamp="2026-10-08T15:00:00+00:00",
            source="operator_approved_opening_statement",
            body={
                "symbol": "GOOGL",
                "qty": "1",
                "opening_basis": "360.93",
                "price": "360.93",
                "as_of_utc": "2026-10-08T15:00:00+00:00",
                "evidence_kind": "operator_approved_opening_statement",
                "evidence_sha256": EVIDENCE_SHA,
                "evidence_ref": "statement",
                "not_a_fill": True,
                "not_historical_realized_pl": True,
                "not_profit_or_loss": True,
                "not_a_return": True,
            },
        )
    assert fill_exc.value.reason == "forward_opening_fill_like_payload"
    assert list(tmp_path.iterdir()) == []


def test_historical_cutover_math_unchanged_and_second_cutover_still_rejected():
    cutover = build_proposed_cutover_event()
    sell = _fill(EVENT_FILL_SELL, "GE", "1", "300", "2026-09-10T15:00:00+00:00", "ge-sell")
    state = reduce_events([cutover, sell])
    assert state["sales"][0]["basis_consumed"] == format(q_money(Decimal("280.1925")), "f")
    assert state["sales"][0]["reporting_label"] == LABEL_HISTORICAL_CUTOVER
    assert state["cash"] == "65663.61"
    later = _opening("NEW", "1", "10", "2026-10-08T15:00:00+00:00")
    opened = reduce_events([cutover, later])
    assert opened["cash"] == state["cash"]
    assert opened["cutover"]["equity"] == state["cutover"]["equity"]
    assert opened["session_closes"] == []
    assert opened["metrics"]["time_weighted_return"] == UNAVAILABLE
    assert opened["open_positions"]["GE"]["remaining_basis"] == "280.192500"
    buy = _fill(EVENT_FILL_BUY, "GE", "1", "300", "2026-09-10T15:00:00+00:00", "ge-buy")
    blended = reduce_events([cutover, buy])
    assert blended["open_positions"]["GE"]["qty"] == "2.000000000000"
    blended_sale = _fill(
        EVENT_FILL_SELL, "GE", "1", "300", "2026-09-10T16:00:00+00:00", "ge-sell-2"
    )
    blended_state = reduce_events([cutover, buy, blended_sale])
    assert blended_state["sales"][0]["reporting_label"] == LABEL_FILL_HISTORY
    with pytest.raises(PerformanceAccountingError) as exc:
        reduce_events([cutover, build_proposed_cutover_event()])
    assert exc.value.reason == "duplicate_cutover"


def test_publication_command_is_recognized_without_writing(tmp_path: Path):
    events_path = tmp_path / "events.jsonl"
    lock_path = tmp_path / "events.lock"
    authority = mint_publication_authority(
        resolved_root=tmp_path,
        expected_git_head="0" * 40,
        events_path=events_path,
        lock_path=lock_path,
        command=COMMAND_FORWARD_OPENING_BASIS,
        publication_mode="append",
        confirm_phrase="TEMP FIXTURE ONLY",
        payload_sha256=EVIDENCE_SHA,
        existing_ledger_sha256="0" * 64,
        existing_ledger_nbytes=0,
        account_mode="PAPER_ONLY",
        live_execution_permitted=False,
        git_identity=passing_git_identity("0" * 40),
        runtime_state=passing_runtime_state(),
    )
    assert authority.command == "publish-forward-opening-basis"
    assert not events_path.exists()
    with pytest.raises(PublicationAuthorityError) as exc:
        mint_publication_authority(
            resolved_root=tmp_path,
            expected_git_head="0" * 40,
            events_path=events_path,
            lock_path=lock_path,
            command="publish-broker-average",
            publication_mode="append",
            confirm_phrase="TEMP FIXTURE ONLY",
            payload_sha256=EVIDENCE_SHA,
            existing_ledger_sha256="0" * 64,
            existing_ledger_nbytes=0,
            git_identity=passing_git_identity("0" * 40),
            runtime_state=passing_runtime_state(),
        )
    assert exc.value.reason == "publication_command_invalid"
