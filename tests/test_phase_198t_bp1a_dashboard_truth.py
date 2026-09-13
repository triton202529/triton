"""BP1A isolated dashboard truth alignment. Presentation only. No production writes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from dashboard.freshness_presentation import (
    CAPTURED_SOURCE_STATUS_LABEL,
    FROZEN_CAPTURE_NOTE,
    FROZEN_PREVIEW_LABEL,
    PRODUCTION_AUTHORITY,
    freshness_locations_agree,
    isolated_status_details_freshness,
    shared_freshness_view,
)
from dashboard.market_session_card import xnys_adjacent_session_times
from dashboard.operator_modern import _certified_holdings_table, _legacy_portfolio_history_section
from dashboard.order_presentation import is_historical_plan_row
from dashboard.paper_sell_arm_controls import COMPLETED_SELL_MESSAGE
from dashboard.performance_presentation import (
    EXPECTED_ACCOUNTING_SHA256,
    KIND_CERTIFIED,
    KIND_OPENING,
    MORE_METRICS_NOTE,
    SECOND_CLOSE_NOTE,
    UNAVAILABLE_LABEL,
    certified_page_totals,
    checkpoint_axis_label,
    load_performance_view,
    try_certified_page_totals,
)
from services.market_calendar import (
    SESSION_CLOSED,
    is_regular_session_open_for_entries,
    regular_session_state,
)
from services.paper_order_ledger import (
    consumed_authority_message,
    dispatch_visibility,
    load_events,
    reduce_ledger,
)

ISO_ROOT = Path(__file__).resolve().parents[1]
PROD_ROOT = Path(r"C:\\Users\\akimw\\triton")
FROZEN_LEDGER = ISO_ROOT / "tests" / "fixtures" / "bp1" / "performance_accounting_events.jsonl"
LEGACY_FORBIDDEN = ("103714", "103,714", "78143.71", "78,143.71", "3618.67", "3,618.67")


def _copy_ledger(dest_root: Path) -> Path:
    dest = dest_root / "data" / "results" / "performance_accounting_events.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(FROZEN_LEDGER.read_bytes())
    return dest


def test_overview_and_performance_share_certified_totals(tmp_path: Path):
    _copy_ledger(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    totals = certified_page_totals(view)
    assert totals["portfolio_value"] == view["latest_equity"] == "$92,006.00"
    assert totals["cash"] == view["cash"] == "$66,447.22"
    assert totals["market_value"] == view["market_value"] == "$25,558.78"
    assert totals["unrealized_pl"] == view["unrealized_pl"] == "+$3,607.22"
    assert totals["period_change"] == UNAVAILABLE_LABEL
    assert totals["period_change_note"] == SECOND_CLOSE_NOTE
    assert totals["source"] == "certified_accounting_close"
    assert totals["uses_portfolio_history"] is False
    assert totals["certified_close_count"] == 1
    assert totals["accounting_status"] == "Building history"
    assert view["ledger_sha256"] == EXPECTED_ACCOUNTING_SHA256
    assert view["technical"]["event_count"] == 14
    assert view["technical"]["cutover_timestamp"] == "2026-09-10T23:22:50+00:00"
    assert view["technical"]["close_timestamp"] == "2026-09-11T22:47:19+00:00"
    for forbidden in LEGACY_FORBIDDEN:
        assert forbidden not in totals["portfolio_value"]
        assert forbidden not in totals["cash"]
        assert forbidden not in totals["unrealized_pl"]


def test_portfolio_current_totals_use_certified_accounting(tmp_path: Path):
    _copy_ledger(tmp_path)
    history = tmp_path / "data" / "results" / "portfolio_history.csv"
    history.write_text(
        "date,total_value\n2026-09-09,103714.00\n2026-09-11,107000.00\n",
        encoding="utf-8",
    )
    totals = try_certified_page_totals(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    assert totals is not None
    assert totals["portfolio_value"] == "$92,006.00"
    assert totals["uses_portfolio_history"] is False
    assert "103714" not in totals["portfolio_value"]
    modern = (ISO_ROOT / "dashboard" / "operator_modern.py").read_text(encoding="utf-8")
    assert 'st.expander("Legacy portfolio history", expanded=False)' in modern
    assert "Historical operational records are not used for verified performance calculations." in (
        modern
    )
    assert "_certified_holdings_table" in modern


def test_legacy_history_cannot_override_certified_totals(tmp_path: Path):
    _copy_ledger(tmp_path)
    (tmp_path / "data" / "results" / "portfolio_history.csv").write_text(
        "date,total_value\n2026-09-11,103714.00\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "results" / "positions_snapshot.csv").write_text(
        "symbol,qty,market_value,unrealized_pl\nDBA,39,1128.27,3618.67\n",
        encoding="utf-8",
    )
    totals = try_certified_page_totals(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    assert totals["portfolio_value"] == "$92,006.00"
    assert totals["unrealized_pl"] == "+$3,607.22"
    assert totals["period_change"] == UNAVAILABLE_LABEL


def test_period_change_remains_unavailable(tmp_path: Path):
    _copy_ledger(tmp_path)
    totals = try_certified_page_totals(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    assert totals["period_change"] == UNAVAILABLE_LABEL
    assert totals["period_change"] != "$0.00"
    assert totals["period_change_note"] == SECOND_CLOSE_NOTE


def test_top_and_sidebar_freshness_cannot_disagree():
    status = {
        "data_freshness": {
            "class": "STALE",
            "heartbeat_timestamp": "2026-09-11T22:47:19+00:00",
        }
    }
    top = shared_freshness_view(status, isolated=True)
    sidebar = shared_freshness_view(status, isolated=True)
    assert freshness_locations_agree(top, sidebar)
    assert top["value"] == FROZEN_PREVIEW_LABEL
    assert sidebar["value"] == FROZEN_PREVIEW_LABEL
    assert top["note"] == FROZEN_CAPTURE_NOTE
    assert top["captured_source_status"] == CAPTURED_SOURCE_STATUS_LABEL
    assert "Fresh" not in top["value"]
    assert "Stale" not in top["value"]
    details = isolated_status_details_freshness("STALE")
    assert details["value"] == FROZEN_PREVIEW_LABEL
    assert details["captured_source_status"] == CAPTURED_SOURCE_STATUS_LABEL
    assert "STALE" not in details["value"]
    assert details["captured_source_status"].startswith("Captured source status:")


def test_frozen_preview_renders_neutral_status():
    status = {"data_freshness": {"class": "FRESH", "heartbeat_timestamp": "now"}}
    view = shared_freshness_view(status, isolated=True)
    assert view["value"] == FROZEN_PREVIEW_LABEL
    assert view["note"] == FROZEN_CAPTURE_NOTE
    assert view["tone"] == "info"
    modern = (ISO_ROOT / "dashboard" / "operator_modern.py").read_text(encoding="utf-8")
    rail = (ISO_ROOT / "view_results.py").read_text(encoding="utf-8")
    assert "shared_freshness_view" in modern
    assert "shared_freshness_view" in rail


def test_production_freshness_uses_one_shared_authority():
    status = {
        "data_freshness": {
            "class": "STALE",
            "heartbeat_timestamp": "2026-09-11T22:47:19+00:00",
        }
    }
    top = shared_freshness_view(status, isolated=False)
    sidebar = shared_freshness_view(status, isolated=False)
    assert freshness_locations_agree(top, sidebar)
    assert top["authority"] == PRODUCTION_AUTHORITY
    assert top["evaluated_at"] == "2026-09-11T22:47:19+00:00"
    assert top["value"] == sidebar["value"]
    assert top["value"] == "Stale"


def test_chart_shows_only_two_session_checkpoint_labels(tmp_path: Path):
    _copy_ledger(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    labels = view["checkpoint_axis_labels"]
    assert labels == ["Sep 10 · Opening checkpoint", "Sep 11 · Certified close"]
    joined = " ".join(labels)
    for tick in ("03:00", "06:00", "09:00", "12:00"):
        assert tick not in joined
    assert checkpoint_axis_label("2026-09-10", KIND_OPENING) == labels[0]
    assert checkpoint_axis_label("2026-09-11", KIND_CERTIFIED) == labels[1]


def test_exact_timestamps_remain_in_details(tmp_path: Path):
    _copy_ledger(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    assert view["checkpoints"][0]["timestamp"] == "2026-09-10T23:22:50+00:00"
    assert view["checkpoints"][1]["timestamp"] == "2026-09-11T22:47:19+00:00"
    assert view["technical"]["checkpoint_timestamps"] == [
        "2026-09-10T23:22:50+00:00",
        "2026-09-11T22:47:19+00:00",
    ]
    assert view["technical"]["cutover_timestamp"] == "2026-09-10T23:22:50+00:00"
    assert view["technical"]["close_timestamp"] == "2026-09-11T22:47:19+00:00"


def test_unavailable_metrics_are_grouped_without_becoming_zero(tmp_path: Path):
    _copy_ledger(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    assert view["primary_metrics"] == [
        "time_weighted_return",
        "maximum_drawdown",
        "spy_comparison",
        "qqq_comparison",
    ]
    assert "net_realized_pl" in view["more_metrics"]
    assert "fees" in view["more_metrics"]
    for key in view["primary_metrics"] + [k for k in view["more_metrics"] if k != "fees"]:
        assert view[key] == UNAVAILABLE_LABEL
        assert view[key] not in {"0", "0.00", "$0.00"}
    assert view["grouped_metrics_note"] == MORE_METRICS_NOTE
    assert view["header_badges"][0]["text"] == KIND_OPENING
    assert view["header_badges"][1]["text"] == "1 certified close"
    assert view["header_badges"][2]["text"] == "Building history"


def test_dba_gross_net_distinction_remains(tmp_path: Path):
    _copy_ledger(tmp_path)
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    dba = next(row for row in view["activity"] if row["symbol"] == "DBA")
    assert dba["quantity"] == "12"
    assert dba["fill_price"] == "$29.03"
    assert dba["proceeds"] == "$348.36"
    assert dba["proceeds_is_not_profit"] is True
    assert dba["position"] == "51 → 39"
    assert dba["gross_result"] == "$26.22"
    assert dba["gross_label"] == "Gross provisional"
    assert dba["net_realized_pl"] == UNAVAILABLE_LABEL
    assert view["net_realized_pl"] == UNAVAILABLE_LABEL


def test_core_service_calendar_authority_unchanged():
    now = datetime(2026, 8, 13, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    before = regular_session_state(now=now)
    neighbors = xnys_adjacent_session_times(now=now)
    after = regular_session_state(now=now)
    assert before["session_state"] == after["session_state"]
    assert before["reason"] == after["reason"]
    assert before["calendar"] == "XNYS"
    assert neighbors["calendar"] == "XNYS"
    assert neighbors["source"].endswith("xnys_adjacent_session_times")
    assert is_regular_session_open_for_entries(now=now) is (before["session_state"] == "OPEN")
    weekend = datetime(2026, 8, 15, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    assert regular_session_state(now=weekend)["session_state"] == SESSION_CLOSED
    src = (ISO_ROOT / "services" / "market_calendar.py").read_text(encoding="utf-8")
    assert "def xnys_adjacent_session_times" not in src
    assert "def is_regular_session_open_for_entries" in src
    card_src = (ISO_ROOT / "dashboard" / "market_session_card.py").read_text(encoding="utf-8")
    assert "Does not grant OPEN" in card_src


def test_core_service_ledger_identity_and_write_surface_unchanged(tmp_path: Path):
    fixture = (
        ISO_ROOT
        / "tests"
        / "fixtures"
        / "bm5"
        / "frozen"
        / "data"
        / "results"
        / "paper_order_events.jsonl"
    )
    if fixture.is_file():
        dest = tmp_path / "data" / "results" / "paper_order_events.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fixture.read_bytes())
        events = load_events(dest)
        rows = reduce_ledger(root=tmp_path)
        broker = [row for row in rows if not is_historical_plan_row(row)]
        vis = dispatch_visibility(broker)
        assert vis["submitted"] == len(
            [
                r
                for r in broker
                if r.get("action_disposition") == "submitted"
                or (r.get("kind") == "order" and r.get("broker_order_id"))
            ]
        )
        assert all(row.get("broker_order_id") for row in broker if row.get("kind") == "order")
    msg = consumed_authority_message({"submitted": 1, "filled": 1, "failed": 0, "skipped": 0})
    assert msg != COMPLETED_SELL_MESSAGE
    ledger_src = (ISO_ROOT / "services" / "paper_order_ledger.py").read_text(encoding="utf-8")
    assert "def append_event" in ledger_src or "def record_" in ledger_src
    assert "is_historical_plan_row" not in ledger_src
    assert "COMPLETED_SELL_MESSAGE" not in ledger_src
    modern = (ISO_ROOT / "dashboard" / "order_presentation.py").read_text(encoding="utf-8")
    assert "is_historical_plan_row" in modern


def test_render_helpers_create_no_files_or_broker_calls(tmp_path: Path):
    before_prod = {
        p: p.stat().st_mtime for p in PROD_ROOT.joinpath("data").rglob("*") if p.is_file()
    }
    iso_data = ISO_ROOT / "data"
    before_iso = (
        {p: p.stat().st_mtime for p in iso_data.rglob("*") if p.is_file()}
        if iso_data.is_dir()
        else {}
    )

    def _fail(*_a, **_k):
        raise AssertionError("broker or write invoked")

    with patch("urllib.request.urlopen", side_effect=_fail):
        _copy_ledger(tmp_path)
        totals = try_certified_page_totals(
            root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256
        )
        assert totals["portfolio_value"] == "$92,006.00"
        shared_freshness_view(
            {"data_freshness": {"class": "STALE", "heartbeat_timestamp": "t"}},
            isolated=True,
        )
    after_prod = {
        p: p.stat().st_mtime for p in PROD_ROOT.joinpath("data").rglob("*") if p.is_file()
    }
    after_iso = (
        {p: p.stat().st_mtime for p in iso_data.rglob("*") if p.is_file()}
        if iso_data.is_dir()
        else {}
    )
    assert before_prod == after_prod
    assert before_iso == after_iso
    assert _certified_holdings_table.__name__ == "_certified_holdings_table"
    assert _legacy_portfolio_history_section.__name__ == "_legacy_portfolio_history_section"


def test_preview_does_not_read_production_paths(tmp_path: Path):
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "ui_a_isolation_marker.txt").write_text(
        "DO_NOT_WRITE_PRODUCTION\n", encoding="utf-8"
    )
    from dashboard.performance_presentation import PerformanceDataUnavailable

    with pytest.raises(PerformanceDataUnavailable):
        load_performance_view(
            root=tmp_path,
            events_path=PROD_ROOT / "data" / "results" / "performance_accounting_events.jsonl",
            expected_sha256=EXPECTED_ACCOUNTING_SHA256,
        )
    from dashboard.preview_data import assert_not_production_path

    try:
        assert_not_production_path(
            PROD_ROOT / "data" / "results" / "portfolio_history.csv",
            root=tmp_path,
        )
        raised = False
    except RuntimeError:
        raised = True
    assert raised
