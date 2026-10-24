"""Closeout and certified-close provenance on the dashboard. Display only."""

from __future__ import annotations

from dashboard.portfolio_history_presentation import render_portfolio_history
from dashboard.status_provenance import (
    ROLE_CURRENT_ERROR,
    ROLE_MISSING,
    classify_closeout_presentation,
    provenance_facts,
    render_closeout_provenance,
    render_status_provenance,
)

APPLICABLE = "2026-10-23"
RUNTIME_AS_OF = "2026-10-24T02:14:00+00:00"
CERTIFIED_SESSION = "2026-10-16"


class _Expander:
    def __init__(self, calls: list, title: str) -> None:
        self._calls = calls
        self._title = title

    def __enter__(self):
        self._calls.append(("expander", self._title))
        return self

    def __exit__(self, *_args) -> bool:
        return False


class _Column:
    def __init__(self, calls: list) -> None:
        self._calls = calls

    def metric(self, label, value, *_args, **_kwargs) -> None:
        self._calls.append(("metric", str(label), str(value)))


class _Streamlit:
    def __init__(self) -> None:
        self.calls: list = []

    def markdown(self, text, *_args, **_kwargs) -> None:
        self.calls.append(("markdown", str(text)))

    def caption(self, text, *_args, **_kwargs) -> None:
        self.calls.append(("caption", str(text)))

    def write(self, text, *_args, **_kwargs) -> None:
        self.calls.append(("write", str(text)))

    def info(self, text, *_args, **_kwargs) -> None:
        self.calls.append(("info", str(text)))

    def warning(self, text, *_args, **_kwargs) -> None:
        self.calls.append(("warning", str(text)))

    def error(self, text, *_args, **_kwargs) -> None:
        self.calls.append(("error", str(text)))

    def success(self, text, *_args, **_kwargs) -> None:
        self.calls.append(("success", str(text)))

    def columns(self, count):
        return [_Column(self.calls) for _ in range(int(count))]

    def expander(self, title, expanded=False):
        return _Expander(self.calls, str(title))

    def selectbox(self, label, options, index=0, key=None, help=None):
        return options[index]

    def dataframe(self, *_args, **_kwargs) -> None:
        return None


def _joined(calls: list) -> str:
    return "\n".join(str(part) for call in calls for part in call)


def _roles(calls: list) -> list:
    return [call[0] for call in calls]


def test_historical_failed_closeout_is_not_a_current_error():
    receipt = {
        "session_date": "2026-10-19",
        "state": "STREAMLIT_STOP_FAILED",
        "reason": "streamlit_pid_still_running",
    }
    presentation = classify_closeout_presentation([receipt], applicable_session=APPLICABLE)
    assert presentation["current_role"] == ROLE_MISSING
    assert presentation["current_receipt"] is None
    assert presentation["historical_receipts"] == [receipt]

    screen = _Streamlit()
    render_closeout_provenance(screen, presentation)
    rendered = _joined(screen.calls)
    assert "error" not in _roles(screen.calls)
    assert "2026-10-19" in rendered
    assert "STREAMLIT_STOP_FAILED" in rendered
    assert "Historical closeout evidence" in rendered


def test_current_session_failed_closeout_stays_a_current_error():
    receipt = {
        "session_date": APPLICABLE,
        "state": "STREAMLIT_STOP_FAILED",
        "reason": "streamlit_pid_still_running",
    }
    presentation = classify_closeout_presentation([receipt], applicable_session=APPLICABLE)
    assert presentation["current_role"] == ROLE_CURRENT_ERROR
    assert presentation["historical_receipts"] == []

    screen = _Streamlit()
    render_closeout_provenance(screen, presentation)
    errors = [call[1] for call in screen.calls if call[0] == "error"]
    assert len(errors) == 1
    assert APPLICABLE in errors[0]
    assert "STREAMLIT_STOP_FAILED" in errors[0]
    assert "expander" not in _roles(screen.calls)


def test_missing_closeout_receipt_has_no_error_banner():
    presentation = classify_closeout_presentation([], applicable_session=APPLICABLE)
    assert presentation["current_role"] == ROLE_MISSING
    assert presentation["historical_receipts"] == []

    screen = _Streamlit()
    render_closeout_provenance(screen, presentation)
    rendered = _joined(screen.calls)
    assert "error" not in _roles(screen.calls)
    assert f"No closeout receipt for trading session {APPLICABLE}." in rendered


def test_certified_close_date_is_independent_of_current_runtime_date():
    facts = provenance_facts(
        runtime_status="STOPPED",
        runtime_as_of=RUNTIME_AS_OF,
        market_hours_outcome="fail",
        market_hours_at="2026-10-23T18:12:40Z",
        certified_session_date=CERTIFIED_SESSION,
    )
    assert facts["current_runtime_as_of"] == RUNTIME_AS_OF
    assert facts["latest_certified_accounting_close"] == CERTIFIED_SESSION
    assert facts["current_runtime_as_of"] != facts["latest_certified_accounting_close"]
    assert facts["certified_close_implies_runtime_failure"] is False

    screen = _Streamlit()
    render_status_provenance(screen, facts)
    rendered = _joined(screen.calls)
    assert "error" not in _roles(screen.calls)
    assert "Current runtime health" in rendered
    assert RUNTIME_AS_OF in rendered
    assert "Latest market-hours cycle" in rendered
    assert "2026-10-23T18:12:40Z" in rendered
    assert "Latest certified accounting close" in rendered
    assert CERTIFIED_SESSION in rendered
    assert "independent of current runtime health" in rendered

    portfolio = _Streamlit()
    render_portfolio_history(
        {
            "certified": {
                "heading": "Certified accounting close",
                "as_of_et": "2026-10-16 19:33:07",
                "session_date": CERTIFIED_SESSION,
                "cash_display": "$75,271.72",
                "market_value_display": "$16,419.91",
                "equity_display": "$91,691.63",
                "evidence": "performance_accounting_events.jsonl session_close:2026-10-16",
            },
            "snapshot": {
                "timestamp_et": "2026-10-23 22:14:00",
                "position_count": 1,
                "open_orders": 0,
                "positions_mtime_et": "2026-10-23 22:14:00 ET",
                "orders_mtime_et": "2026-10-23 22:14:00 ET",
                "open_orders_evidence": "open_orders_snapshot.csv",
            },
            "legacy": {
                "title": "Legacy",
                "status": "complete",
                "cash_note": "note",
                "chart_note": "note",
                "mtime_et": "2026-10-18",
                "exists": False,
            },
            "drawdown": "Unavailable",
        },
        portfolio,
    )
    portfolio_text = _joined(portfolio.calls)
    assert "error" not in _roles(portfolio.calls)
    assert CERTIFIED_SESSION in portfolio_text
    assert "independent of current runtime health" in portfolio_text
    assert "$91,691.63" in portfolio_text
