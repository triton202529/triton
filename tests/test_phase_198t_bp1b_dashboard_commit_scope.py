"""BP1B isolated dashboard acceptance. Presentation only. No production writes."""

from __future__ import annotations

import hashlib
from pathlib import Path

from dashboard.freshness_presentation import (
    CAPTURED_SOURCE_STATUS_LABEL,
    FROZEN_CAPTURE_NOTE,
    FROZEN_PREVIEW_LABEL,
    isolated_status_details_freshness,
    shared_freshness_view,
)
from dashboard.performance_presentation import (
    EXPECTED_ACCOUNTING_SHA256,
    STATUS_BUILDING,
    UNAVAILABLE_LABEL,
    load_performance_view,
    try_certified_page_totals,
)
from services.performance_accounting import load_events

ISO_ROOT = Path(__file__).resolve().parents[1]
PROD_ROOT = Path(r"C:\Users\akimw\triton")
PROD_CALENDAR = PROD_ROOT / "services" / "market_calendar.py"
PROD_LEDGER = PROD_ROOT / "services" / "paper_order_ledger.py"
ISO_CALENDAR = ISO_ROOT / "services" / "market_calendar.py"
ISO_LEDGER = ISO_ROOT / "services" / "paper_order_ledger.py"
FROZEN_LEDGER = ISO_ROOT / "tests" / "fixtures" / "bp1" / "performance_accounting_events.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_accounting_fixture_matches_certified_close():
    assert FROZEN_LEDGER.is_file()
    assert _sha256(FROZEN_LEDGER) == EXPECTED_ACCOUNTING_SHA256
    events = load_events(FROZEN_LEDGER)
    assert len(events) == 14
    types = [str(event.get("event_type") or "") for event in events]
    assert "ACCOUNTING_CUTOVER" in types
    assert "SESSION_CLOSE" in types
    cutover = next(event for event in events if event.get("event_type") == "ACCOUNTING_CUTOVER")
    close = next(event for event in events if event.get("event_type") == "SESSION_CLOSE")
    assert str(cutover.get("event_timestamp") or "") == "2026-09-10T23:22:50+00:00"
    assert str(close.get("event_timestamp") or "") == "2026-09-11T22:47:19+00:00"


def test_overview_portfolio_performance_share_certified_close(tmp_path: Path):
    dest = tmp_path / "data" / "results" / "performance_accounting_events.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(FROZEN_LEDGER.read_bytes())
    view = load_performance_view(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    totals = try_certified_page_totals(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256)
    assert totals is not None
    assert totals["portfolio_value"] == view["latest_equity"] == "$92,006.00"
    assert totals["cash"] == view["cash"] == "$66,447.22"
    assert totals["market_value"] == view["market_value"] == "$25,558.78"
    assert totals["unrealized_pl"] == view["unrealized_pl"] == "+$3,607.22"
    assert totals["certified_close_count"] == view["certified_close_count"] == 1
    assert totals["period_change"] == UNAVAILABLE_LABEL
    assert totals["accounting_status"] == view["accounting_status"] == STATUS_BUILDING
    assert view["ledger_sha256"] == EXPECTED_ACCOUNTING_SHA256
    assert view["technical"]["event_count"] == 14


def test_missing_ledger_does_not_use_synthetic_fallback(tmp_path: Path):
    assert try_certified_page_totals(root=tmp_path, expected_sha256=EXPECTED_ACCOUNTING_SHA256) is (
        None
    )


def test_core_services_match_production_head():
    assert ISO_CALENDAR.read_bytes() == PROD_CALENDAR.read_bytes()
    assert ISO_LEDGER.read_bytes() == PROD_LEDGER.read_bytes()
    assert (
        _sha256(ISO_CALENDAR) == "338ba5b09572312a4486fa75f28e22bd181dbef59dc897279b25f8938526a0b6"
    )
    assert _sha256(ISO_LEDGER) == "3b52d05e01571aad698a0ef93a16ec4f56694595df4afc77004c5bb4d30abb3d"
    calendar_src = ISO_CALENDAR.read_text(encoding="utf-8")
    ledger_src = ISO_LEDGER.read_text(encoding="utf-8")
    assert "def xnys_adjacent_session_times" not in calendar_src
    assert "is_historical_plan_row" not in ledger_src
    assert "COMPLETED_SELL_MESSAGE" not in ledger_src


def test_status_details_presents_frozen_preview_not_stale():
    details = isolated_status_details_freshness("STALE")
    assert details["value"] == FROZEN_PREVIEW_LABEL
    assert details["note"] == FROZEN_CAPTURE_NOTE
    assert details["captured_source_status"] == CAPTURED_SOURCE_STATUS_LABEL
    assert details["value"] != "STALE"
    assert details["value"] != "Stale"
    assert "STALE" not in details["value"]
    rail = (ISO_ROOT / "view_results.py").read_text(encoding="utf-8")
    assert "isolated_status_details_freshness" in rail
    assert 'fresh.get("captured_source_status")' in rail
    assert CAPTURED_SOURCE_STATUS_LABEL.startswith("Captured source status:")
    status = {"data_freshness": {"class": "STALE", "heartbeat_timestamp": "t"}}
    view = shared_freshness_view(status, isolated=True)
    assert view["value"] == FROZEN_PREVIEW_LABEL
    assert view["captured_source_status"] == CAPTURED_SOURCE_STATUS_LABEL
