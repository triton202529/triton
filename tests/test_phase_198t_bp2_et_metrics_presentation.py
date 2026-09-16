"""BP2 isolated ET and performance-metric presentation. No production writes."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from dashboard.display_labels import profit_factor_text, ratio_percent_text
from dashboard.operator_ui import format_timestamp
from dashboard.performance_presentation import (
    STATUS_AVAILABLE,
    UNAVAILABLE_LABEL,
    load_performance_view,
    metric_text,
)
from dashboard.preview_data import preview_capture_label
from dashboard.time_presentation import (
    NY_TZ,
    TIMEZONE_NAME,
    format_operator_datetime,
    format_raw_utc,
    parse_aware_datetime,
)

ISO_ROOT = Path(__file__).resolve().parents[1]
PROD_ROOT = Path(r"C:\Users\akimw\triton")
BQ2_LEDGER = ISO_ROOT / "tests" / "fixtures" / "bp2" / "performance_accounting_events.jsonl"
BQ2_SHA256 = "450bca40637a244d81715dc1b949ca2dd6921b099b052c23a1ccd4411fb240d7"
PROTECTED = (
    "services/performance_accounting.py",
    "services/performance_evidence_collector.py",
    "services/performance_publication_authority.py",
    "services/market_calendar.py",
    "services/paper_order_ledger.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_bq2(dest_root: Path) -> Path:
    dest = dest_root / "data" / "results" / "performance_accounting_events.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(BQ2_LEDGER.read_bytes())
    return dest


def test_utc_to_et_crosses_calendar_date_boundary():
    assert (
        format_operator_datetime("2026-09-15T03:13:00.063301+00:00")
        == "September 14, 2026 at 11:13 PM ET"
    )


def test_edt_conversion():
    assert (
        format_operator_datetime("2026-09-14T18:26:17.246711Z")
        == "September 14, 2026 at 2:26 PM ET"
    )


def test_est_conversion():
    assert (
        format_operator_datetime("2026-01-15T17:00:00+00:00") == "January 15, 2026 at 12:00 PM ET"
    )


def test_dst_spring_forward():
    before = format_operator_datetime("2026-03-08T06:59:00+00:00")
    after = format_operator_datetime("2026-03-08T07:00:00+00:00")
    assert before == "March 8, 2026 at 1:59 AM ET"
    assert after == "March 8, 2026 at 3:00 AM ET"
    assert parse_aware_datetime("2026-03-08T07:00:00+00:00").astimezone(NY_TZ).dst()


def test_dst_fall_back():
    first = format_operator_datetime("2026-11-01T05:30:00+00:00")
    second = format_operator_datetime("2026-11-01T06:30:00+00:00")
    assert first == "November 1, 2026 at 1:30 AM ET"
    assert second == "November 1, 2026 at 1:30 AM ET"
    first_dt = parse_aware_datetime("2026-11-01T05:30:00+00:00").astimezone(NY_TZ)
    second_dt = parse_aware_datetime("2026-11-01T06:30:00+00:00").astimezone(NY_TZ)
    assert first_dt.utcoffset() != second_dt.utcoffset()


def test_already_offset_timestamp_converts_once():
    offset = "2026-09-14T23:13:00.063301-04:00"
    assert format_operator_datetime(offset) == "September 14, 2026 at 11:13 PM ET"


def test_already_formatted_et_is_not_double_converted():
    text = "September 14, 2026 at 11:13 PM ET"
    assert format_operator_datetime(text) == text
    assert format_timestamp(text) == text


def test_malformed_timestamp_is_unavailable():
    assert format_operator_datetime("not-a-timestamp") == UNAVAILABLE_LABEL
    assert format_operator_datetime("2026/09/14 23:13") == UNAVAILABLE_LABEL


def test_naive_timestamp_is_unavailable():
    assert format_operator_datetime("2026-09-14T23:13:00") == UNAVAILABLE_LABEL
    assert format_operator_datetime(datetime(2026, 9, 14, 23, 13, 0)) == UNAVAILABLE_LABEL


def test_date_only_remains_date_only():
    assert format_operator_datetime("2026-09-14") == "September 14, 2026"
    assert format_operator_datetime(date(2026, 9, 14)) == "September 14, 2026"


def test_raw_utc_retained_for_technical_details():
    raw = "2026-09-15T03:13:00.063301+00:00"
    assert format_raw_utc(raw) == raw
    assert TIMEZONE_NAME == "America/New_York"


def test_format_timestamp_uses_shared_helper():
    assert format_timestamp("2026-09-14T18:26:17.246711Z") == "September 14, 2026 at 2:26 PM ET"
    assert "UTC" not in format_timestamp("2026-09-14T18:26:17.246711Z")
    assert format_timestamp("2026-09-14 18:26:17") == UNAVAILABLE_LABEL


def test_ordinary_pages_do_not_hardcode_utc_clock_format():
    modern = (ISO_ROOT / "dashboard" / "operator_modern.py").read_text(encoding="utf-8")
    page = (ISO_ROOT / "dashboard" / "performance_page.py").read_text(encoding="utf-8")
    ui = (ISO_ROOT / "dashboard" / "operator_ui.py").read_text(encoding="utf-8")
    table = (ISO_ROOT / "dashboard" / "operator_table.py").read_text(encoding="utf-8")
    assert "%Y-%m-%d %H:%M:%S UTC" not in modern
    assert "%Y-%m-%d %H:%M:%S UTC" not in page
    assert "%Y-%m-%d %H:%M:%S UTC" not in ui
    assert "%Y-%m-%d %H:%M:%S UTC" not in table
    assert "format_operator_datetime" in ui
    assert 'st.expander("Technical details", expanded=False)' in page


def test_ratio_metrics_are_percentages_not_currency():
    assert ratio_percent_text("0.001327", signed=True) == "+0.13%"
    assert ratio_percent_text("0.000000", signed=False) == "0.00%"
    assert ratio_percent_text("-0.004462", signed=True) == "-0.45%"
    assert ratio_percent_text("-0.007973", signed=True) == "-0.80%"
    assert ratio_percent_text("0.235580", signed=False) == "23.56%"
    assert ratio_percent_text("1.000000", signed=False) == "100.00%"
    assert profit_factor_text("1.5") == "1.50×"
    assert ratio_percent_text("UNAVAILABLE", signed=True) == UNAVAILABLE_LABEL
    assert profit_factor_text("UNAVAILABLE") == UNAVAILABLE_LABEL
    assert metric_text("0.001327", kind="signed_ratio_pct") == "+0.13%"
    assert metric_text("0.001327") != "+0.13%"
    assert "$" not in metric_text("0.001327", kind="signed_ratio_pct")


def test_currency_metrics_remain_currency():
    assert metric_text("92128.09") == "$92,128.09"
    assert metric_text("26.193528", kind="signed") == "+$26.19"
    assert metric_text("-20.680000", kind="signed") == "-$20.68"


def test_bq2_two_close_exact_display_values(tmp_path: Path):
    assert _sha256(BQ2_LEDGER) == BQ2_SHA256
    _copy_bq2(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=BQ2_SHA256)
    assert view["certified_close_count"] == 2
    assert view["accounting_status"] == STATUS_AVAILABLE
    assert view["latest_equity"] == "$92,128.09"
    assert view["cash"] == "$70,424.55"
    assert view["market_value"] == "$21,703.54"
    assert view["external_flow"] == "$0.00"
    assert view["time_weighted_return"] == "+0.13%"
    assert view["maximum_drawdown"] == "0.00%"
    assert view["spy_comparison"] == "-0.45%"
    assert view["qqq_comparison"] == "-0.80%"
    assert view["average_market_exposure"] == "23.56%"
    assert view["win_rate"] == "100.00%"
    assert view["spy_return_kind"] == "TOTAL_RETURN"
    assert view["qqq_return_kind"] == "TOTAL_RETURN"
    assert view["spy_comparison_label"] == "SPY return · Total return"
    assert view["qqq_comparison_label"] == "QQQ return · Total return"
    assert view["comparison_period"] == "2026-09-11 to 2026-09-14"
    assert view["exposure_adjusted_return"] == UNAVAILABLE_LABEL
    assert view["exposure_adjusted_note"] == "Unavailable because evidence is insufficient"
    assert view["decision_slippage"] == UNAVAILABLE_LABEL
    assert view["fees"] == "Pending fee reconciliation"
    assert view["uses_portfolio_history"] is False
    assert view["checkpoints"][2]["timestamp"] == "2026-09-15T03:13:00.063301+00:00"
    assert view["checkpoints"][2]["timestamp_et"] == "September 14, 2026 at 11:13 PM ET"
    assert view["technical"]["close_timestamp"] == "2026-09-15T03:13:00.063301+00:00"
    monday = {row["symbol"]: row for row in view["activity"]}
    assert monday["GE"]["gross_label"] == "Gross provisional"
    assert monday["GE"]["net_realized_pl"] == UNAVAILABLE_LABEL
    assert monday["MA"]["gross_label"] == "Gross provisional"
    assert view["net_realized_pl"] == "+$26.19"
    for bench in view["benchmarks"]:
        assert bench["methodology"] == "Total return"
        assert bench["comparison"] != UNAVAILABLE_LABEL
        assert bench["comparison_period"] == "2026-09-11 to 2026-09-14"


def test_one_close_ratios_stay_unavailable(tmp_path: Path):
    from dashboard.performance_presentation import EXPECTED_ACCOUNTING_SHA256

    src = ISO_ROOT / "tests" / "fixtures" / "bp1" / "performance_accounting_events.jsonl"
    dest = tmp_path / "data" / "results" / "performance_accounting_events.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    for key in (
        "time_weighted_return",
        "maximum_drawdown",
        "spy_comparison",
        "qqq_comparison",
        "win_rate",
        "exposure_adjusted_return",
        "decision_slippage",
        "profit_factor",
    ):
        assert view[key] == UNAVAILABLE_LABEL
        assert view[key] not in {"$0.00", "0.00%", "0", ""}


def test_no_portfolio_history_fallback(tmp_path: Path):
    _copy_bq2(tmp_path)
    history = tmp_path / "data" / "results" / "portfolio_history.csv"
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text("this,is,not,authority\n", encoding="utf-8")
    view = load_performance_view(root=tmp_path, expected_sha256=BQ2_SHA256)
    assert view["uses_portfolio_history"] is False
    assert view["latest_equity"] == "$92,128.09"
    assert view["time_weighted_return"] == "+0.13%"


def test_import_performs_no_writes_or_broker_calls():
    def _fail(*_a, **_k):
        raise AssertionError("broker or write invoked")

    with (
        patch("urllib.request.urlopen", side_effect=_fail),
        patch("requests.sessions.Session.request", side_effect=_fail, create=True),
    ):
        import dashboard.performance_presentation as pres
        import dashboard.time_presentation as times
        import dashboard.performance_page as page

        del pres, times, page


def test_protected_controls_and_services_unchanged():
    sell = (ISO_ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    buy = (ISO_ROOT / "dashboard" / "paper_buy_add_arm_controls.py").read_text(encoding="utf-8")
    modern = (ISO_ROOT / "dashboard" / "operator_modern.py").read_text(encoding="utf-8")
    assert 'key="paper_sell_arm_reviewed"' in sell
    assert 'key="paper_sell_arm_confirm"' in sell
    assert "CONFIRM PAPER SELLS" in sell
    assert 'key="paper_buy_add_arm_submit"' in buy
    assert "ARM PAPER ENTRIES" in buy
    assert "REVOKE PAPER ENTRIES" in buy
    assert "_render_start_controls" in modern
    for rel in PROTECTED:
        iso = (ISO_ROOT / rel).read_bytes()
        prod = (PROD_ROOT / rel).read_bytes()
        assert iso == prod


def test_preview_banner_bq2_close_is_et():
    label = preview_capture_label("2026-09-15T03:13:00.063301+00:00")
    assert label == "Preview data captured at September 14, 2026 at 11:13 PM ET"
