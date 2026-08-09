"""
UIR-002 — reconciled portfolio reporting engine contracts.

Read-only. Uses fixtures under tests/fixtures/uir002.
Does not call broker, mutate artifacts, or invoke runtime controls.
"""

from __future__ import annotations

import ast
import inspect
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import services.reconciled_portfolio_reporting as rpr
from services.reconciled_portfolio_reporting import (
    DEFAULT_PERIOD,
    PERIOD_TYPES,
    ReportRequest,
    build_reconciled_report,
    format_money,
    resolve_period_bounds,
    tone_for_report_status,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "uir002"
ROOT = Path(__file__).resolve().parents[1]


def _report(fixture: str, **kwargs):
    req = kwargs.pop("request", None) or ReportRequest(**kwargs.pop("req_kwargs", {}))
    now = kwargs.pop("now", datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc))
    return build_reconciled_report(req, root=FIX / fixture, now=now)


def test_period_types_order_and_default():
    assert DEFAULT_PERIOD == "Daily"
    assert list(PERIOD_TYPES) == [
        "Daily",
        "Weekly",
        "Monthly",
        "Quarterly",
        "Half-year",
        "Yearly",
        "Custom",
    ]


def test_period_boundaries_deterministic():
    hist = [{"date": date(2026, 8, 7)}]
    now = datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc)
    daily = resolve_period_bounds(
        ReportRequest(period_type="Daily", anchor_date=date(2026, 8, 7)),
        now=now,
        history=hist,
    )
    assert daily["start_date"] == date(2026, 8, 7) == daily["end_date"]

    weekly = resolve_period_bounds(
        ReportRequest(period_type="Weekly", anchor_date=date(2026, 8, 7)),
        now=now,
        history=hist,
    )
    assert weekly["start_date"] == date(2026, 8, 3)
    assert weekly["end_date"] == date(2026, 8, 9)

    q2 = resolve_period_bounds(
        ReportRequest(period_type="Quarterly", anchor_date=date(2026, 5, 1)),
        now=now,
        history=hist,
    )
    assert q2["start_date"] == date(2026, 4, 1)
    assert q2["end_date"] == date(2026, 6, 30)

    h1 = resolve_period_bounds(
        ReportRequest(period_type="Half-year", anchor_date=date(2026, 3, 1)),
        now=now,
        history=hist,
    )
    assert h1["label"] == "2026 H1"
    assert h1["start_date"] == date(2026, 1, 1)


def test_in_progress_only_while_valid_session_incomplete():
    hist = [{"date": date(2026, 8, 7)}]
    # 15:00 America/New_York on Friday 2026-08-07 is 19:00 UTC (EDT).
    now = datetime(2026, 8, 7, 19, 0, tzinfo=timezone.utc)
    bounds = resolve_period_bounds(
        ReportRequest(period_type="Daily", anchor_date=date(2026, 8, 7)),
        now=now,
        history=hist,
    )
    assert bounds["is_market_session"] is True
    assert bounds["completeness"] == "IN_PROGRESS"

    # After close the same session day must not remain IN_PROGRESS.
    after_close = datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc)  # 17:00 ET
    done = resolve_period_bounds(
        ReportRequest(period_type="Daily", anchor_date=date(2026, 8, 7)),
        now=after_close,
        history=hist,
    )
    assert done["completeness"] != "IN_PROGRESS"
    assert done["completeness"] == "PARTIAL_COVERAGE"


def test_saturday_defaults_to_latest_eligible_trading_session():
    now = datetime(2026, 8, 8, 17, 0, tzinfo=timezone.utc)  # Saturday
    report = build_reconciled_report(
        ReportRequest(period_type="Daily"),
        root=FIX / "weekend_history",
        now=now,
    )
    assert report["ok"] is True
    assert report["end_date"] == "2026-08-07"  # calendar, not hardcoded path
    assert report["completeness"] != "IN_PROGRESS"
    assert report["effective_coverage"]["is_market_session"] is True
    assert "2026-08-08" in (
        report["effective_coverage"].get("non_session_history_observations") or []
    )


def test_sunday_defaults_same_as_saturday():
    now = datetime(2026, 8, 9, 15, 0, tzinfo=timezone.utc)  # Sunday
    report = build_reconciled_report(
        ReportRequest(period_type="Daily"),
        root=FIX / "weekend_history",
        now=now,
    )
    assert report["end_date"] == "2026-08-07"
    assert report["completeness"] != "IN_PROGRESS"


def test_holiday_is_not_active_session():
    # 2026-01-01 is New Year's Day (not a NYSE session).
    assert rpr.is_nyse_trading_session(date(2026, 1, 1)) is False
    now = datetime(2026, 1, 1, 17, 0, tzinfo=timezone.utc)
    bounds = resolve_period_bounds(
        ReportRequest(period_type="Daily", anchor_date=date(2026, 1, 1)),
        now=now,
        history=[],
    )
    assert bounds["is_market_session"] is True  # rolled back to prior session
    assert bounds["completeness"] != "IN_PROGRESS"
    assert bounds["end_date"] < date(2026, 1, 1)


def test_weekend_history_row_is_not_a_market_session():
    assert rpr.is_nyse_trading_session(date(2026, 8, 8)) is False
    now = datetime(2026, 8, 8, 17, 0, tzinfo=timezone.utc)
    report = build_reconciled_report(
        ReportRequest(period_type="Daily"),
        root=FIX / "weekend_history",
        now=now,
    )
    assert report["end_date"] != "2026-08-08"
    assert report["completeness"] != "IN_PROGRESS"


def test_portfolio_history_not_labeled_official_open_or_close():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    m = report["metrics"]
    assert m["opening_equity"]["availability"] == "UNAVAILABLE"
    assert m["closing_equity"]["availability"] == "UNAVAILABLE"
    assert m["net_equity_change"]["availability"] == "UNAVAILABLE"
    assert m["opening_equity"]["value"] is None
    assert (
        "session-open" in (m["opening_equity"].get("reason") or "").lower()
        or "authoritative" in (m["opening_equity"].get("reason") or "").lower()
    )
    assert (
        m["previous_recorded_equity"]["method"] == "previous_recorded_portfolio_history_total_value"
    )
    assert m["latest_recorded_equity"]["method"] == "latest_recorded_portfolio_history_total_value"
    assert "official" not in (m["previous_recorded_equity"].get("method") or "").lower()
    assert "session_close" not in (m["latest_recorded_equity"].get("method") or "").lower()
    assert "prior_session" not in (m["previous_recorded_equity"].get("method") or "")
    assert m["recorded_observation_change"]["value"] == "200"
    reason = (m["recorded_observation_change"].get("reason") or "").lower()
    assert "period p/l" in reason
    assert "investment return" in reason
    assert "session return" in reason


def test_complete_daily_withholds_boundaries_keeps_recorded_observations():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    m = report["metrics"]
    assert format_money(m["opening_equity"]) == "Unavailable"
    assert format_money(m["closing_equity"]) == "Unavailable"
    assert format_money(m["net_equity_change"]) == "Unavailable"
    assert m["previous_recorded_equity"]["value"] == "15000"
    assert m["latest_recorded_equity"]["value"] == "15200"
    assert Decimal(m["recorded_observation_change"]["value"]) == Decimal("200")
    assert m["period_pl"]["availability"] == "UNAVAILABLE"
    assert m["period_return"]["availability"] == "UNAVAILABLE"
    assert m["realized_pl"]["availability"] == "UNAVAILABLE"
    assert m["realized_pl"]["value"] is None
    assert format_money(m["realized_pl"]) == "Unavailable"
    assert report["availability"] == "PARTIAL"
    assert report["completeness"] != "IN_PROGRESS"
    assert report["writes_performed"] is False
    assert report["network_calls_performed"] is False


def test_multi_day_recorded_observation_change_exact_decimal():
    report = build_reconciled_report(
        ReportRequest(
            period_type="Custom",
            custom_start=date(2026, 8, 6),
            custom_end=date(2026, 8, 7),
        ),
        root=FIX / "complete_daily",
        now=datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc),
    )
    assert report["metrics"]["opening_equity"]["availability"] == "UNAVAILABLE"
    assert report["metrics"]["closing_equity"]["availability"] == "UNAVAILABLE"
    assert Decimal(report["metrics"]["recorded_observation_change"]["value"]) == Decimal("200")
    assert (
        "return"
        not in (report["metrics"]["recorded_observation_change"].get("method") or "").lower()
    )


def test_no_data_unavailable_not_zero():
    report = _report(
        "no_data", req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)}
    )
    assert report["availability"] == "UNAVAILABLE"
    assert report["reconciliation"]["status"] != "PASS"
    for key in ("opening_equity", "closing_equity", "net_equity_change", "period_pl"):
        assert report["metrics"][key]["value"] is None
        assert format_money(report["metrics"][key]) == "Unavailable"


def test_missing_opening_for_custom_range_partial():
    report = build_reconciled_report(
        ReportRequest(
            period_type="Custom",
            custom_start=date(2026, 7, 1),
            custom_end=date(2026, 7, 2),
        ),
        root=FIX / "partial_coverage",
        now=datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc),
    )
    # History only on 2026-08-01 — range has no coverage.
    assert report["availability"] in ("UNAVAILABLE", "PARTIAL")
    assert report["metrics"]["opening_equity"]["availability"] == "UNAVAILABLE"
    assert report["metrics"]["closing_equity"]["availability"] == "UNAVAILABLE"


def test_freshness_independent_of_completeness_and_reconciliation():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
        now=datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc),
    )
    # Freshness may be FRESH/STALE/UNKNOWN independently of COMPLETE/PASS.
    assert report["freshness"]["status"] in ("FRESH", "STALE", "UNKNOWN")
    assert report["completeness"] != "COMPLETE"
    assert report["reconciliation"]["status"] != "PASS"
    assert not (
        report["freshness"]["status"] == "FRESH"
        and report["completeness"] == "COMPLETE"
        and report["reconciliation"]["status"] == "PASS"
    )


def test_missing_cash_flow_and_unsupported_realized_when_absent():
    report = _report(
        "partial_coverage",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 1)},
    )
    assert report["metrics"]["net_external_cash_flows"]["availability"] == "UNAVAILABLE"
    assert report["metrics"]["realized_pl"]["availability"] == "UNAVAILABLE"
    assert report["metrics"]["realized_pl"]["value"] is None


def test_conflicting_account_totals_fail_reconciliation():
    report = _report(
        "conflict_totals",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    assert report["reconciliation"]["status"] == "FAIL"
    discs = report["reconciliation"]["discrepancies"]
    assert discs
    assert discs[0]["check"] == "recorded_observation_equity_identity"
    assert discs[0]["expected"] == "200"
    assert discs[0]["observed"] == "250"
    assert discs[0]["difference"] == "50"
    assert report["availability"] in ("CONFLICTED", "AVAILABLE", "PARTIAL")


def test_orders_filled_partial_cancel_reject_and_unlinked():
    report = _report(
        "orders_mix",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    counts = report["activity"]["counts"]
    assert counts["filled"] == 2
    assert counts["partially_filled"] == 1
    assert counts["canceled"] == 1
    assert counts["rejected"] == 1
    assert counts["trades"] == 3  # filled + partial only
    assert counts["unlinked_broker_activity"] >= 1
    partial = next(r for r in report["activity"]["broker_orders"] if r["order_id"] == "p1")
    assert partial["requested_qty"] == "10"
    assert partial["filled_qty"] == "4"
    assert partial["remaining_qty"] == "6"
    unlinked = next(r for r in report["activity"]["broker_orders"] if r["order_id"] == "u1")
    assert unlinked["attribution"] == "UNLINKED_BROKER_ACTIVITY"


def test_order_coverage_is_partial_never_complete_zero_claim():
    missing = _report(
        "partial_coverage",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 1)},
    )
    assert missing["activity"]["availability"] == "UNAVAILABLE"
    assert missing["metrics"]["order_count"]["value"] is None
    assert (
        "unavailable or incomplete"
        in (missing["activity"].get("no_activity_message") or "").lower()
    )

    present = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 6)},
    )
    # recent_orders is a limited window — never complete period totals / never "No activity recorded".
    assert present["activity"]["availability"] == "PARTIAL"
    assert present["activity"]["coverage_complete"] is False
    assert present["activity"]["counts_are_complete_period_totals"] is False
    assert present["activity"]["no_activity_message"] != "No activity recorded"
    assert "partial coverage" in (present["activity"].get("no_activity_message") or "").lower()
    assert present["metrics"]["order_count"]["availability"] == "PARTIAL"
    assert (
        "complete period total" in (present["metrics"]["order_count"].get("reason") or "").lower()
    )


def test_holdings_missing_beginning_not_zero():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    row = report["holdings"]["rows"][0]
    assert row["Beginning quantity"] is None
    assert row.get("Net quantity change") is None
    assert "Ending quantity" not in row
    assert "Ending unrealized P/L" not in row
    assert row["Latest snapshot quantity"] == "10"
    assert row["Latest snapshot unrealized P/L"] == "200"
    assert report["holdings"]["is_period_ending_boundary"] is False
    assert report["holdings"]["period_relationship"] == "latest_non_boundary_snapshot"


def test_stale_and_unknown_freshness_never_green():
    report = _report(
        "stale_sources",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 1)},
        now=datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc),
    )
    assert report["freshness"]["status"] in ("STALE", "UNKNOWN")
    assert tone_for_report_status(report["freshness"]["status"]) != "ok"
    assert tone_for_report_status("UNKNOWN") == "muted"
    assert tone_for_report_status("UNAVAILABLE") == "muted"


def test_custom_invalid_dates_rejected():
    report = build_reconciled_report(
        ReportRequest(
            period_type="Custom",
            custom_start=date(2026, 8, 10),
            custom_end=date(2026, 8, 1),
        ),
        root=FIX / "complete_daily",
        now=datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc),
    )
    assert report["ok"] is False
    assert "earlier" in (report.get("error") or "").lower()


def test_provenance_attached_to_metrics():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    for key, metric in report["metrics"].items():
        assert "availability" in metric
        assert "sources" in metric
        assert isinstance(metric["sources"], list)


def test_engine_module_has_no_network_or_write_calls():
    src = Path(rpr.__file__).read_text(encoding="utf-8")
    banned = (
        "requests.",
        "urllib.request",
        "http.client",
        "alpaca",
        "TradeClient",
        "to_csv(",
        'open(path, "w"',
        "open(path,'w'",
        "write_text(",
        "Path.write_text",
    )
    lower = src.lower()
    for token in banned:
        assert token.lower() not in lower, token
    # AST: no calls to known filesystem mutators
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"write_text", "to_csv", "unlink", "rename"}


def test_missing_evidence_never_pass_reconciliation_alone():
    report = _report(
        "no_data", req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)}
    )
    assert report["reconciliation"]["status"] != "PASS"


# --- UIR-002C: conservative freshness + calendar dependency ---


def test_timezone_aware_source_aged_exactly_no_negative_clamp():
    as_of = datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc)
    src = datetime(2026, 8, 7, 20, 0, tzinfo=timezone.utc)
    row = rpr.classify_source_freshness(
        as_of=as_of,
        source_name="positions_snapshot",
        source_path="positions_snapshot.csv",
        content_timestamp=src,
    )
    assert row["status"] == "FRESH"
    assert Decimal(row["age_minutes"]) == Decimal("60.0")
    assert row["post_as_of"] is False
    # Exact age preserved — never clamped from a negative future delta.
    assert Decimal(row["age_minutes"]) > 0


def test_future_source_timestamp_never_fresh_and_discloses_delta():
    as_of = datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc)
    future = datetime(2026, 8, 7, 22, 30, tzinfo=timezone.utc)
    row = rpr.classify_source_freshness(
        as_of=as_of,
        source_name="positions_snapshot",
        source_path="positions_snapshot.csv",
        content_timestamp=future,
    )
    assert row["status"] == "UNKNOWN"
    assert row["post_as_of"] is True
    assert row["excluded_from_as_of_freshness"] is True
    assert row["age_minutes"] is None
    assert Decimal(row["minutes_after_as_of"]) == Decimal("90.0")
    assert "90.0" in (row.get("reason") or "")
    # Negative age must not be clamped to zero as FRESH.
    status, age, _ = rpr._classify_freshness(source_ts=future, now=as_of)
    assert status == "UNKNOWN"
    assert age is None


def test_date_only_history_never_gets_synthetic_time():
    as_of = datetime(2026, 8, 8, 17, 0, tzinfo=timezone.utc)
    row = rpr.classify_source_freshness(
        as_of=as_of,
        source_name="portfolio_history",
        source_path="portfolio_history.csv",
        content_timestamp=None,
        date_only_value=date(2026, 8, 7),
    )
    assert row["status"] == "UNKNOWN"
    assert row["date_only"] is True
    assert row["source_timestamp"] == "2026-08-07"
    assert "20:00" not in (row.get("reason") or "")
    assert "synthetic" in (row.get("reason") or "").lower()
    src = Path(rpr.__file__).read_text(encoding="utf-8")
    assert "20, 0," not in src and "hour=20" not in src


def test_mtime_cannot_produce_fresh():
    as_of = datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc)
    mtime = datetime(2026, 8, 7, 20, 55, tzinfo=timezone.utc)
    row = rpr.classify_source_freshness(
        as_of=as_of,
        source_name="positions_snapshot",
        source_path="positions_snapshot.csv",
        content_timestamp=None,
        filesystem_mtime=mtime,
    )
    assert row["status"] == "UNKNOWN"
    assert row["filesystem_mtime_advisory"] == mtime.isoformat()
    assert "cannot establish fresh" in (row.get("reason") or "").lower()


def test_fresh_cannot_mask_unknown_or_stale():
    fresh = {
        "status": "FRESH",
        "age_minutes": "10.0",
        "source": "positions_snapshot",
    }
    unknown = {"status": "UNKNOWN", "age_minutes": None, "source": "portfolio_history"}
    stale = {"status": "STALE", "age_minutes": "200.0", "source": "recent_orders"}
    agg_u = rpr.aggregate_report_freshness([fresh, unknown])
    assert agg_u["status"] == "UNKNOWN"
    agg_s = rpr.aggregate_report_freshness([fresh, stale])
    assert agg_s["status"] == "STALE"
    all_fresh = rpr.aggregate_report_freshness(
        [
            {"status": "FRESH", "age_minutes": "5.0", "source": "a"},
            {"status": "FRESH", "age_minutes": "8.0", "source": "b"},
        ]
    )
    assert all_fresh["status"] == "FRESH"


def test_report_freshness_unknown_with_date_only_history_and_per_source_visible():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
        now=datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc),
    )
    assert report["freshness"]["status"] == "UNKNOWN"
    sources = report["freshness"]["sources"]
    assert sources
    by_name = {s["source"]: s for s in sources}
    assert by_name["portfolio_history"]["status"] == "UNKNOWN"
    assert by_name["portfolio_history"]["date_only"] is True
    assert "sources" in report["freshness"]
    assert (report.get("evidence") or {}).get("source_freshness")


def test_saturday_snapshot_not_friday_period_evidence():
    now = datetime(2026, 8, 8, 17, 0, tzinfo=timezone.utc)
    report = build_reconciled_report(
        ReportRequest(period_type="Daily"),
        root=FIX / "weekend_history",
        now=now,
    )
    assert report["end_date"] == "2026-08-07"
    assert report["completeness"] != "IN_PROGRESS"
    # Fixture snapshot is dated 2026-07-01 — not Friday close evidence.
    assert report["holdings"]["as_of_compatible"] is False
    assert report["metrics"]["unrealized_pl"]["availability"] == "PARTIAL"
    reason = (report["metrics"]["unrealized_pl"].get("reason") or "").lower()
    assert "not a historical period-end" in reason or "selected session" in reason


def test_missing_calendar_does_not_weekday_fallback(monkeypatch):
    monkeypatch.setattr(
        rpr,
        "market_calendar_available",
        lambda: (False, "ImportError: simulated missing exchange_calendars"),
    )
    # Friday must not become a session via weekday-only logic.
    assert rpr.is_nyse_trading_session(date(2026, 8, 7)) is False
    bounds = resolve_period_bounds(
        ReportRequest(period_type="Daily"),
        now=datetime(2026, 8, 8, 17, 0, tzinfo=timezone.utc),
        history=[{"date": date(2026, 8, 7)}],
    )
    assert bounds["ok"] is False
    assert bounds.get("calendar_available") is False
    assert "weekday" in (bounds.get("error") or "").lower()
    assert (
        rpr.latest_eligible_trading_session(now=datetime(2026, 8, 8, 17, 0, tzinfo=timezone.utc))
        is None
    )


def test_exchange_calendars_declared_in_production_requirements():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "exchange-calendars==4.12" in req


def test_live_saturday_daily_status_axes_conservative_freshness():
    """Selected session remains Friday; freshness is conservative UNKNOWN with date-only history."""
    report = build_reconciled_report(
        ReportRequest(period_type="Daily"),
        root=ROOT,
        now=datetime(2026, 8, 8, 21, 16, 59, tzinfo=timezone.utc),
    )
    assert report["ok"] is True
    assert report["end_date"] == "2026-08-07"
    assert report["availability"] == "PARTIAL"
    assert report["completeness"] == "PARTIAL_COVERAGE"
    assert report["reconciliation"]["status"] == "PARTIAL"
    assert report["freshness"]["status"] == "UNKNOWN"
    assert report["metrics"]["opening_equity"]["availability"] == "UNAVAILABLE"
    assert report["metrics"]["closing_equity"]["availability"] == "UNAVAILABLE"
    assert report["metrics"]["period_pl"]["availability"] == "UNAVAILABLE"
    assert report["metrics"]["realized_pl"]["availability"] == "UNAVAILABLE"
    assert report["completeness"] != "IN_PROGRESS"


# --- UIR-002D: holdings + reconciliation semantic correction ---


def test_uir002d_non_boundary_snapshot_not_labeled_ending_holdings():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    holdings = report["holdings"]
    cols = holdings["columns"]
    banned = (
        "Ending quantity",
        "Ending market value",
        "Ending cost basis",
        "Ending unrealized P/L",
    )
    for label in banned:
        assert label not in cols
        for row in holdings["rows"]:
            assert label not in row
    assert holdings["is_period_ending_boundary"] is False
    assert "Latest snapshot quantity" in cols
    assert "Latest snapshot market value" in cols
    assert "Latest snapshot unrealized P/L" in cols
    assert holdings["section_title"] == "Latest position snapshot"
    assert holdings["availability"] == "PARTIAL"


def test_uir002d_non_certified_snapshot_cannot_populate_period_ending_fields():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    holdings = report["holdings"]
    assert holdings.get("as_of_compatible") is False
    assert holdings["period_relationship"] == "latest_non_boundary_snapshot"
    disclosure = (holdings.get("disclosure") or "").lower()
    assert "not a certified" in disclosure or "cannot establish" in disclosure
    assert "ending holdings" in disclosure
    # Values remain under latest-snapshot keys only.
    row = holdings["rows"][0]
    assert row["Latest snapshot quantity"] == "10"
    assert row["Latest snapshot market value"] == "5200"
    assert row.get("Ending quantity") is None or "Ending quantity" not in row


def test_uir002d_latest_snapshot_provenance_and_period_relationship():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    holdings = report["holdings"]
    assert holdings["snapshot_ts"] == "2026-08-07T20:00:00+00:00" or holdings[
        "snapshot_ts"
    ].startswith("2026-08-07T20:00:00")
    assert holdings["snapshot_date"] == "2026-08-07"
    assert holdings["selected_period_end"] == "2026-08-07"
    assert holdings["period_relationship"] == "latest_non_boundary_snapshot"
    assert holdings["snapshot_ts"]
    assert "Position snapshot timestamp" in (holdings.get("disclosure") or "")


def test_uir002d_missing_open_end_holdings_block_net_qty_change():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    for row in report["holdings"]["rows"]:
        assert row["Beginning quantity"] is None
        assert row["Net quantity change"] is None
    checks = {c["check"]: c for c in report["reconciliation"]["checks"]}
    assert checks["position_quantity_chain"]["status"] == "UNAVAILABLE"


def test_uir002d_ending_equity_identity_cannot_pass_without_boundary():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    checks = {c["check"]: c for c in report["reconciliation"]["checks"]}
    assert "ending_equity_identity" in checks
    assert checks["ending_equity_identity"]["status"] == "UNAVAILABLE"
    assert checks["ending_equity_identity"]["status"] != "PASS"


def test_uir002d_recorded_observation_identity_pass_precise_name_only():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    checks = {c["check"]: c for c in report["reconciliation"]["checks"]}
    obs = checks["recorded_observation_equity_identity"]
    assert obs["status"] == "PASS"
    assert obs["observation_date"] == "2026-08-07"
    assert obs["expected"] == "15200"
    assert obs["observed"] == "15200"
    assert obs["difference"] == "0"
    assert obs["sources"]
    assert "portfolio_history" in str(obs["sources"][0]).replace("\\", "/")
    expl = (obs.get("explanation") or "").lower()
    assert "does not establish" in expl
    assert "session-close" in expl or "session close" in expl
    assert obs["establishes_period_ending_equity"] is False
    assert obs["establishes_complete_boundary_coverage"] is False
    # Must not use the authoritative ending name for a PASS.
    assert checks["ending_equity_identity"]["status"] != "PASS"


def test_uir002d_recorded_observation_pass_does_not_make_closing_equity_available():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    checks = {c["check"]: c for c in report["reconciliation"]["checks"]}
    assert checks["recorded_observation_equity_identity"]["status"] == "PASS"
    assert report["metrics"]["closing_equity"]["availability"] == "UNAVAILABLE"
    assert report["metrics"]["closing_equity"]["value"] is None
    assert format_money(report["metrics"]["closing_equity"]) == "Unavailable"


def test_uir002d_recorded_observation_pass_does_not_make_coverage_complete():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    checks = {c["check"]: c for c in report["reconciliation"]["checks"]}
    assert checks["recorded_observation_equity_identity"]["status"] == "PASS"
    assert report["completeness"] != "COMPLETE"
    assert report["completeness"] == "PARTIAL_COVERAGE"
    assert report["reconciliation"]["status"] != "PASS"
    assert report["reconciliation"]["status"] == "PARTIAL"


def test_uir002d_partial_activity_qualified_empty_state_wording():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    activity = report["activity"]
    assert activity["availability"] == "PARTIAL"
    assert activity["coverage_complete"] is False
    empty_t = activity.get("empty_state_triton") or ""
    empty_b = activity.get("empty_state_broker") or ""
    assert "No TRITON decision-audit rows in period." not in empty_t
    assert "available local evidence" in empty_t.lower()
    assert "No activity recorded" not in (activity.get("no_activity_message") or "")
    assert "available local evidence" in empty_b.lower()
    ui_src = (ROOT / "dashboard" / "operator_reports.py").read_text(encoding="utf-8")
    assert "No TRITON decision-audit rows in period." not in ui_src
    assert "available local evidence" in ui_src


def test_uir002d_limited_window_zeros_not_complete_period_totals():
    report = _report(
        "complete_daily",
        req_kwargs={"period_type": "Daily", "anchor_date": date(2026, 8, 7)},
    )
    activity = report["activity"]
    assert activity["counts_are_complete_period_totals"] is False
    assert activity["coverage_kind"] == "LIMITED_RECENT_WINDOW"
    msg = (activity.get("no_activity_message") or "").lower()
    note = (activity.get("note") or "").lower()
    assert "complete period" in msg or "complete period" in note
    assert "limited" in msg or "limited" in note
    # Zero counts remain observations, not definitive period totals.
    for key, val in activity["counts"].items():
        if val == 0:
            assert activity["counts_are_complete_period_totals"] is False
