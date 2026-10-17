"""Activity log clarity: wording, market time, and trade/system split.

Presentation only. These tests do not write ledgers or activity storage.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from dashboard.activity_clarity import (
    FILTER_SCOPE_NOTE,
    NO_TRADE_ACTIVITY,
    activity_primary_line,
    activity_section,
    attach_ledger_structure,
    format_activity_market_time,
    operator_trade_line,
    partition_activity_events,
    present_activity_event,
    trade_activity_notice,
)

ROOT = Path(__file__).resolve().parents[1]


def _order(**overrides):
    event = {
        "event_id": "e1",
        "occurred_at": "2026-10-16T19:15:00+00:00",
        "category": "ORDER",
        "severity": "SUCCESS",
        "title": "Paper order was filled",
        "message": "legacy title must not decide the operator line",
        "source": "paper_order_events",
    }
    event.update(overrides)
    return event


def test_action_status_wording_uses_structured_fields():
    assert (
        operator_trade_line(
            _order(
                symbol="AAPL",
                lifecycle_action="ENTRY",
                filled_qty=4,
                broker_terminal_status="FILLED",
                terminal_status_proven=True,
            )
        )
        == "New AAPL entry — 4 shares filled"
    )
    assert (
        operator_trade_line(
            _order(
                symbol="TSLA",
                opportunity_type="ADD",
                filled_qty=2,
                broker_terminal_status="FILLED",
                terminal_status_proven=True,
            )
        )
        == "Added TSLA — 2 shares filled"
    )
    assert (
        operator_trade_line(
            _order(
                symbol="SPCX",
                action="TRIM",
                filled_qty=8,
                terminal_status="FILLED",
                terminal_status_proven=True,
            )
        )
        == "Trimmed SPCX — 8 shares filled"
    )
    assert (
        operator_trade_line(
            _order(
                symbol="BITO",
                disposition="EXIT",
                filled_qty=400,
                broker_terminal_status="FILLED",
                fill_proven=True,
            )
        )
        == "Exited BITO — 400 shares filled"
    )
    assert (
        operator_trade_line(
            _order(
                symbol="AMD",
                lifecycle_action="EXIT",
                selected_qty=2,
                broker_submission_status="SUBMITTED",
                action_disposition="submitted",
                terminal_status_proven=False,
            )
        )
        == "AMD exit submitted — 2 shares"
    )
    assert (
        operator_trade_line(
            _order(
                symbol="BITO",
                source_pipeline="manual_portfolio_exit",
                side="SELL",
                filled_qty=19,
                broker_terminal_status="FILLED",
                terminal_status_proven=True,
            )
        )
        == "Manual BITO exit — 19 shares filled"
    )
    assert (
        operator_trade_line(
            _order(
                symbol="AMD",
                lifecycle_action="EXIT",
                selected_qty=2,
                broker_submission_status="PENDING",
            )
        )
        == "AMD exit pending — 2 shares"
    )
    assert (
        operator_trade_line(
            _order(
                symbol="AMD",
                lifecycle_action="EXIT",
                selected_qty=2,
                broker_terminal_status="CANCELLED",
            )
        )
        == "AMD exit cancelled — 2 shares"
    )
    assert (
        operator_trade_line(
            _order(
                symbol="AMD",
                lifecycle_action="EXIT",
                selected_qty=2,
                broker_terminal_status="REJECTED",
            )
        )
        == "AMD exit rejected — 2 shares"
    )
    assert (
        operator_trade_line(
            _order(
                symbol="AMD",
                lifecycle_action="EXIT",
                selected_qty=2,
                event_type="ORDER_STATUS_RECONCILED",
            )
        )
        == "AMD exit reconciled — 2 shares"
    )


def test_fractional_quantity_is_preserved():
    line = operator_trade_line(
        _order(
            symbol="AAPL",
            lifecycle_action="ADD",
            filled_qty=2.5,
            selected_qty=4,
            broker_terminal_status="FILLED",
            terminal_status_proven=True,
        )
    )
    assert line == "Added AAPL — 2.5 shares filled"


def test_buy_sell_labels_do_not_become_entry_trim_or_exit():
    buy = operator_trade_line(
        _order(
            symbol="AAPL",
            side="BUY",
            action="BUY",
            selected_qty=4,
            broker_submission_status="SUBMITTED",
            title="Paper order was filled",
        )
    )
    sell = operator_trade_line(
        _order(
            symbol="AMD",
            side="SELL",
            action="SELL",
            selected_qty=2,
            action_disposition="submitted",
            title="Paper order was filled",
        )
    )
    assert buy == "AAPL order submitted — 4 shares"
    assert "entry" not in buy.lower()
    assert "filled" not in buy.lower()
    assert sell == "AMD order submitted — 2 shares"
    assert "exit" not in sell.lower()
    assert "trim" not in sell.lower()
    assert "filled" not in sell.lower()


def test_unproven_fill_is_not_called_filled():
    event = _order(
        symbol="AMD",
        lifecycle_action="EXIT",
        selected_qty=2,
        broker_terminal_status="FILLED",
        status="FILLED",
        terminal_status_proven=False,
        broker_submission_status="SUBMITTED",
        title="Paper order was filled",
        message="Audit row evidences fill status=FILLED",
    )
    line = operator_trade_line(event)
    primary = activity_primary_line(event)
    assert line == "AMD exit submitted — 2 shares"
    assert "filled" not in line.lower()
    unconfirmed = operator_trade_line(
        _order(
            symbol="AAPL",
            lifecycle_action="ENTRY",
            selected_qty=4,
            broker_terminal_status="FILLED",
            terminal_status_proven=False,
        )
    )
    assert unconfirmed == "New AAPL entry pending — 4 shares"
    assert "filled" not in unconfirmed.lower()
    assert "filled" not in primary.lower()
    assert "2026-10-16T19:15:00" not in primary
    assert "+00:00" not in primary
    assert primary.startswith("Oct 16, 2026 · 3:15 PM EDT · ")


def test_title_only_fill_claim_stays_unconfirmed():
    line = operator_trade_line(
        {
            "event_id": "audit-1",
            "category": "ORDER",
            "symbol": "AAPL",
            "title": "Paper order was filled on 2026-10-16",
            "occurred_at": "2026-10-16T19:15:00Z",
        }
    )
    assert line == "AAPL order pending confirmation"
    assert "filled" not in line.lower()


def test_market_time_converts_edt_and_est():
    assert format_activity_market_time("2026-10-16T19:15:00+00:00") == "Oct 16, 2026 · 3:15 PM EDT"
    assert format_activity_market_time("2026-01-15T17:00:00+00:00") == "Jan 15, 2026 · 12:00 PM EST"
    assert format_activity_market_time("2026-03-08T06:59:00+00:00") == "Mar 8, 2026 · 1:59 AM EST"
    assert format_activity_market_time("2026-03-08T07:00:00+00:00") == "Mar 8, 2026 · 3:00 AM EDT"
    assert format_activity_market_time("2026-11-01T05:30:00+00:00") == "Nov 1, 2026 · 1:30 AM EDT"
    assert format_activity_market_time("2026-11-01T06:30:00+00:00") == "Nov 1, 2026 · 1:30 AM EST"
    assert format_activity_market_time("2026-10-16T19:15:00") == "Unavailable"
    assert format_activity_market_time("2026-10-16") == "Oct 16, 2026"


def test_trade_and_system_sections_do_not_duplicate_events():
    trade_event = _order(event_id="trade-1", symbol="AAPL", lifecycle_action="ENTRY")
    system_event = {
        "event_id": "sys-1",
        "occurred_at": "2026-10-16T18:00:00+00:00",
        "category": "ENTRY_GATE",
        "title": "Paper-entry gates were evaluated",
        "source": "paper_entry_stage_audit",
    }
    cycle = {
        "event_id": "sys-2",
        "occurred_at": "2026-10-16T20:00:00+00:00",
        "category": "CYCLE",
        "title": "Continuous cycle 4 started",
        "source": "continuous_loop_log",
    }
    duplicate = dict(trade_event)
    trade, system = partition_activity_events([system_event, trade_event, cycle, duplicate])
    trade_ids = [row["event_id"] for row in trade]
    system_ids = [row["event_id"] for row in system]
    assert trade_ids == ["trade-1"]
    assert system_ids == ["sys-2", "sys-1"]
    assert set(trade_ids).isdisjoint(system_ids)
    assert activity_section(system_event) == "system"
    assert activity_section(trade_event) == "trade"


def test_newest_first_uses_absolute_time_across_dst():
    older = _order(event_id="winter", occurred_at="2026-01-15T17:00:00+00:00", symbol="AAA")
    newer = _order(event_id="summer", occurred_at="2026-10-16T19:15:00+00:00", symbol="BBB")
    trade, system = partition_activity_events([older, newer])
    assert [row["event_id"] for row in trade] == ["summer", "winter"]
    assert system == []


def test_empty_trade_activity_is_plain():
    system_only = [
        {
            "event_id": "hb",
            "occurred_at": "2026-10-16T19:15:00+00:00",
            "category": "DATA",
            "title": "Watchdog reported stale heartbeat",
        }
    ]
    trade, system = partition_activity_events(system_only)
    assert trade == []
    assert len(system) == 1
    assert trade_activity_notice(trade) == NO_TRADE_ACTIVITY
    assert trade_activity_notice(trade) == "No trade activity in this range."
    assert trade_activity_notice([_order()]) is None


def test_presentation_does_not_rewrite_the_event():
    event = _order(
        symbol="BITO",
        lifecycle_action="EXIT",
        selected_qty=19,
        action_disposition="submitted",
        occurred_at="2026-10-16T19:15:00+00:00",
    )
    ledger = {
        "event_id": "e1",
        "source_pipeline": "manual_portfolio_exit",
        "filled_qty": 19,
        "broker_terminal_status": "FILLED",
        "terminal_status_proven": True,
        "selected_qty": 19,
    }
    original = deepcopy(event)
    ledger_original = deepcopy(ledger)
    views = attach_ledger_structure([event], [ledger])
    presented = present_activity_event(views[0])
    assert event == original
    assert ledger == ledger_original
    assert views[0]["source_pipeline"] == "manual_portfolio_exit"
    assert presented["operator_line"] == "Manual BITO exit — 19 shares filled"
    assert presented["technical"]["occurred_at_utc"] == "2026-10-16T19:15:00+00:00"
    assert presented["technical"]["title"] == original["title"]
    assert "+00:00" not in presented["primary_line"]


def test_activity_expanders_match_installed_streamlit_and_omit_key():
    """Render several rows through the installed expander signature. No key=."""
    import inspect
    from unittest.mock import MagicMock, patch

    import dashboard.operator_modern as modern
    from streamlit.elements.layouts import LayoutsMixin

    signature = inspect.signature(LayoutsMixin.expander)
    assert "key" not in signature.parameters

    class _Expander:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    calls = []

    def expander(label, expanded=False, **kwargs):
        calls.append((label, expanded, dict(kwargs)))
        signature.bind(object(), label, expanded, **kwargs)
        return _Expander()

    fake_st = MagicMock()
    fake_st.expander.side_effect = expander
    events = [
        _order(
            event_id="trade-aapl",
            symbol="AAPL",
            lifecycle_action="ENTRY",
            selected_qty=4,
            broker_submission_status="SUBMITTED",
            occurred_at="2026-10-16T19:15:00+00:00",
        ),
        _order(
            event_id="trade-amd",
            symbol="AMD",
            lifecycle_action="EXIT",
            selected_qty=2,
            action_disposition="submitted",
            occurred_at="2026-10-16T18:00:00+00:00",
        ),
        {
            "event_id": "cycle-1",
            "occurred_at": "2026-10-16T17:00:00+00:00",
            "category": "CYCLE",
            "title": "Continuous cycle 4 started",
            "source": "continuous_loop_log",
        },
    ]
    with patch.object(modern, "st", fake_st), patch(
        "services.paper_order_ledger.load_events", return_value=[]
    ):
        modern._render_activity_timeline(events, show_evidence=True)

    assert len(calls) >= 2
    labels = [label for label, _expanded, _kwargs in calls]
    assert len(labels) == len(set(labels))
    assert any(label.startswith("Technical evidence · Trade") for label in labels)
    assert any(label.startswith("Technical evidence · System") for label in labels)
    for _label, _expanded, kwargs in calls:
        assert "key" not in kwargs
        assert kwargs == {}


def test_operator_console_activity_page_uses_the_split():
    text = (ROOT / "dashboard" / "operator_modern.py").read_text(encoding="utf-8")
    assert "Trade activity" in text
    assert "System activity" in text
    assert "FILTER_SCOPE_NOTE" in text
    assert "NO_TRADE_ACTIVITY" in text
    assert "filter_activity_events" in text
    assert "Category, severity, and search apply to both" in FILTER_SCOPE_NOTE
