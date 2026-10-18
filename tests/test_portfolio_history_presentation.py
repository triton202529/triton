"""Presentation-only Portfolio History: certified close stays separate from legacy CSV."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from dashboard.portfolio_history_presentation import (
    DRAWDOWN_UNAVAILABLE,
    LEGACY_CASH_NOTE,
    LEGACY_SECTION_TITLE,
    LEGACY_STATUS_COMPLETE,
    NON_SESSION_LABEL,
    build_portfolio_history_presentation,
    format_et,
    load_latest_certified_close,
    parse_aware_timestamp,
    render_portfolio_history,
)

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (
    "data/results/portfolio_history.csv",
    "data/results/enhanced_portfolio_history.csv",
    "data/results/performance_accounting_events.jsonl",
    "data/results/positions_snapshot.csv",
    "data/results/open_orders_snapshot.csv",
)
WEEKEND_SNAPSHOT = "2026-10-18T02:32:20Z"
IMPUTED_CASH = "78143.71"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _protected_hashes() -> dict:
    return {rel: _sha256(ROOT / rel) for rel in PROTECTED if (ROOT / rel).is_file()}


def _october_close_line() -> str:
    ledger = ROOT / "data" / "results" / "performance_accounting_events.jsonl"
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if '"session_close:2026-10-16"' not in line:
            continue
        event = json.loads(line)
        if event.get("event_type") == "SESSION_CLOSE" and event.get("session_date") == "2026-10-16":
            return line + "\n"
    raise AssertionError("October 16 certified close missing from ledger")


def _write_fixture(root: Path, *, snapshot_ts: str = WEEKEND_SNAPSHOT) -> None:
    results = root / "data" / "results"
    results.mkdir(parents=True)
    (results / "performance_accounting_events.jsonl").write_text(
        _october_close_line(), encoding="utf-8"
    )
    (results / "enhanced_portfolio_history.csv").write_text(
        "date,cash,market_value,total_value,regime\n"
        f"2026-06-18,{IMPUTED_CASH},21856.29,100000.0,\n",
        encoding="utf-8",
    )
    (results / "portfolio_history.csv").write_text(
        "date,cash,market_value,total_value\n"
        f"2026-06-18,{IMPUTED_CASH},21856.29,100000.00\n"
        "2026-07-06,78143.71,132457.56,210601.27\n"
        f"2026-10-18,{IMPUTED_CASH},16473.92,94617.63\n",
        encoding="utf-8",
    )
    (results / "positions_snapshot.csv").write_text(
        "snapshot_ts,date,ticker,symbol,market_value\n"
        f"{snapshot_ts},2026-10-18,AAPL,AAPL,999999.99\n",
        encoding="utf-8",
    )
    (results / "open_orders_snapshot.csv").write_text("snapshot_ts\n", encoding="utf-8")


class _Column:
    def __init__(self, calls: list) -> None:
        self._calls = calls

    def metric(self, label, value, *args, **kwargs) -> None:
        self._calls.append(("metric", str(label), str(value)))


class _Streamlit:
    def __init__(self) -> None:
        self.calls: list = []

    def markdown(self, text, *args, **kwargs) -> None:
        self.calls.append(("markdown", str(text)))

    def caption(self, text, *args, **kwargs) -> None:
        self.calls.append(("caption", str(text)))

    def write(self, text, *args, **kwargs) -> None:
        self.calls.append(("write", str(text)))

    def info(self, text, *args, **kwargs) -> None:
        self.calls.append(("info", str(text)))

    def warning(self, text, *args, **kwargs) -> None:
        self.calls.append(("warning", str(text)))

    def error(self, text, *args, **kwargs) -> None:
        self.calls.append(("error", str(text)))

    def metric(self, label, value, *args, **kwargs) -> None:
        self.calls.append(("metric", str(label), str(value)))

    def columns(self, count):
        return [_Column(self.calls) for _ in range(int(count))]

    def selectbox(self, label, options, index=0, key=None, help=None):
        self.calls.append(("selectbox", str(label), list(options), index, key))
        return options[index]

    def dataframe(self, frame, *args, **kwargs) -> None:
        self.calls.append(("dataframe", len(frame)))

    def plotly_chart(self, fig, *args, **kwargs) -> None:
        self.calls.append(("plotly", True))


def _texts(calls: list) -> str:
    parts = []
    for call in calls:
        parts.extend(str(part) for part in call)
    return "\n".join(parts)


def test_certified_kpis_match_october_16_close():
    before = _protected_hashes()
    certified = load_latest_certified_close(root=ROOT)
    after = _protected_hashes()
    assert after == before
    assert certified["available"] is True
    assert certified["session_date"] == "2026-10-16"
    assert certified["cash"] == "75271.72"
    assert certified["market_value"] == "16419.91"
    assert certified["equity"] == "91691.63"
    assert certified["cash_display"] == "$75,271.72"
    assert certified["market_value_display"] == "$16,419.91"
    assert certified["equity_display"] == "$91,691.63"
    assert certified["uses_portfolio_history"] is False
    assert certified["uses_enhanced_portfolio_history"] is False
    assert IMPUTED_CASH not in {
        certified["cash"],
        certified["market_value"],
        certified["equity"],
        certified["cash_display"],
        certified["market_value_display"],
        certified["equity_display"],
    }


def test_weekend_snapshot_shows_et_timestamp_and_carry_forward_label(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_fixture(root)
        before = _protected_hashes()
        view = build_portfolio_history_presentation(root)
        after = _protected_hashes()
    assert after == before
    assert view["certified"]["cash"] == "75271.72"
    assert view["certified"]["market_value"] == "16419.91"
    assert view["certified"]["equity"] == "91691.63"
    assert view["snapshot"]["timestamp_et"] == "2026-10-17 22:32 ET"
    assert view["snapshot"]["non_session_label"] == NON_SESSION_LABEL
    assert "20:00" not in view["snapshot"]["timestamp_et"]
    assert view["snapshot"]["position_count"] == 1
    assert view["snapshot"]["open_orders"] == 0
    assert IMPUTED_CASH not in json.dumps(
        {key: view["certified"][key] for key in ("cash", "market_value", "equity", "session_date")}
    )
    assert view["drawdown"] == DRAWDOWN_UNAVAILABLE
    assert "210601.27" not in view["drawdown"]
    assert view["legacy"]["status"] == LEGACY_STATUS_COMPLETE
    assert view["legacy"]["title"] == LEGACY_SECTION_TITLE
    assert "imputed/stale cash" in view["legacy"]["cash_note"]
    assert view["writes_performed"] is False
    assert view["broker_called"] is False


def test_date_only_value_is_not_a_snapshot_timestamp():
    assert parse_aware_timestamp("2026-10-18") is None
    assert parse_aware_timestamp(datetime(2026, 10, 18)) is None
    aware = parse_aware_timestamp(WEEKEND_SNAPSHOT)
    assert aware is not None
    assert format_et(aware) == "2026-10-17 22:32 ET"


def test_session_date_after_close_is_not_labeled_carry_forward():
    from dashboard.portfolio_history_presentation import non_session_status

    friday_evening = parse_aware_timestamp("2026-10-16T23:33:07+00:00")
    assert friday_evening is not None
    status = non_session_status(friday_evening)
    assert status["label"] is None
    assert status["non_session"] is False


def test_missing_ledger_is_unavailable_with_evidence():
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        certified = load_latest_certified_close(root=Path(raw))
    assert certified["available"] is False
    assert certified["cash"] == "Unavailable"
    assert certified["equity"] == "Unavailable"
    assert "missing" in certified["evidence"]


def test_drawdown_stays_unavailable_and_csv_rows_are_unchanged():
    import tempfile

    before = _protected_hashes()
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_fixture(root)
        history = root / "data" / "results" / "portfolio_history.csv"
        enhanced = root / "data" / "results" / "enhanced_portfolio_history.csv"
        ledger = root / "data" / "results" / "performance_accounting_events.jsonl"
        hashes = {path.name: _sha256(path) for path in (history, enhanced, ledger)}
        view = build_portfolio_history_presentation(root)
        st = _Streamlit()
        render_portfolio_history(view, st)
        assert {path.name: _sha256(path) for path in (history, enhanced, ledger)} == hashes
    assert _protected_hashes() == before
    rendered = _texts(st.calls)
    assert DRAWDOWN_UNAVAILABLE in rendered
    assert "Drawdown vs Peak" not in rendered
    assert "data quality" not in rendered.lower()
    assert "FULL" not in rendered
    assert LEGACY_SECTION_TITLE in rendered
    assert LEGACY_CASH_NOTE in rendered
    assert NON_SESSION_LABEL in rendered
    assert "$75,271.72" in rendered
    assert "$16,419.91" in rendered
    assert "$91,691.63" in rendered
    assert IMPUTED_CASH not in rendered.split("Legacy portfolio history — non-certified")[0]
    assert view["drawdown"] == DRAWDOWN_UNAVAILABLE


def test_page_source_does_not_mix_legacy_cash_into_certified_kpis():
    text = (ROOT / "view_results.py").read_text(encoding="utf-8")
    body = text.split("def page_portfolio_history", 1)[1].split("def _positions_trace_snapshot", 1)[
        0
    ]
    module = (ROOT / "dashboard" / "portfolio_history_presentation.py").read_text(encoding="utf-8")
    assert "build_portfolio_history_presentation" in body
    assert "render_portfolio_history" in body
    for banned in (
        "resolve_portfolio_dashboard_state",
        "enhanced_portfolio_history",
        "data_quality",
        "Drawdown vs Peak",
        "FULL",
    ):
        assert banned not in body
    for banned in (
        "AlpacaBroker",
        "submit_order",
        "get_account",
        "append_events",
        "to_csv",
        "enhanced_portfolio_history.csv",
        "append_portfolio_history",
        "drawdown_pct",
    ):
        assert banned not in module
    reports = (ROOT / "dashboard" / "performance_presentation.py").read_text(encoding="utf-8")
    assert 'EXPECTED_SESSION_CLOSE_DATE = "2026-09-11"' in reports


def test_production_snapshot_timestamp_comes_from_snapshot_ts():
    positions = (ROOT / "data" / "results" / "positions_snapshot.csv").read_text(encoding="utf-8")
    header, first = positions.splitlines()[:2]
    columns = header.split(",")
    snapshot_ts = first.split(",")[columns.index("snapshot_ts")]
    before = _protected_hashes()
    view = build_portfolio_history_presentation(ROOT)
    st = _Streamlit()
    render_portfolio_history(view, st)
    after = _protected_hashes()
    assert after == before
    expected = format_et(parse_aware_timestamp(snapshot_ts))
    assert view["snapshot"]["timestamp_et"] == expected
    assert expected in _texts(st.calls)
    if snapshot_ts == WEEKEND_SNAPSHOT:
        assert expected == "2026-10-17 22:32 ET"
        assert view["snapshot"]["non_session_label"] == NON_SESSION_LABEL
    assert view["certified"]["cash"] == "75271.72"
    assert view["broker_called"] is False
    assert view["writes_performed"] is False


if __name__ == "__main__":
    test_certified_kpis_match_october_16_close()
    test_weekend_snapshot_shows_et_timestamp_and_carry_forward_label()
    test_date_only_value_is_not_a_snapshot_timestamp()
    test_session_date_after_close_is_not_labeled_carry_forward()
    test_missing_ledger_is_unavailable_with_evidence()
    test_drawdown_stays_unavailable_and_csv_rows_are_unchanged()
    test_page_source_does_not_mix_legacy_cash_into_certified_kpis()
    test_production_snapshot_timestamp_comes_from_snapshot_ts()
    print("PASS")
