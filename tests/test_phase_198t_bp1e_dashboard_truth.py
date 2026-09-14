"""BP1E isolated production-dashboard truth repair. Presentation only."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from dashboard.freshness_presentation import (
    AGING_AFTER_CLOSE_LABEL,
    FROZEN_CAPTURE_NOTE,
    FROZEN_PREVIEW_LABEL,
    PRODUCTION_AUTHORITY,
    TECHNICAL_RAW_LABEL,
    bind_freshness_presentation,
    current_freshness_presentation,
    freshness_for_render,
    freshness_locations_agree,
    reset_freshness_presentation,
    shared_freshness_view,
    technical_freshness_evidence,
)
from dashboard.operator_modern import _order_badge_fields
from dashboard.order_presentation import (
    AuthoritativeLedgerError,
    display_side,
    load_authoritative_presentation,
    presentation_order_counts,
)
from dashboard.preview_data import (
    EXPECTED_LEDGER_SHA256,
    isolation_active,
    preview_data_root,
)
from dashboard.preview_data import expected_ledger_sha256 as preview_expected_hash
from services.paper_order_ledger import ledger_path, load_events
from tests.fixtures.bm5.authoritative import (
    EXPECTED_BROKER_ORDERS,
    EXPECTED_FILLED,
    EXPECTED_LEDGER_SHA256 as BM5_SHA256,
    materialize_authoritative_fixture,
)

ISO_ROOT = Path(__file__).resolve().parents[1]
PROD_ROOT = Path(r"C:\Users\akimw\triton")
BM5_PREFIX = "6ce03a05"
PROD_LEDGER_SHA256 = "aa8ab8a92ef2eb999549ccfb0821427687b98f64689c7525c4da073a3fda17f6"
EVIDENCE_LEDGER = (
    Path(r"C:\Users\akimw\AppData\Local\Temp")
    / "phase_198T_BP1E_20260913_evidence"
    / "paper_order_events.jsonl"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_bm5(tmp_path: Path) -> Path:
    materialize_authoritative_fixture(tmp_path)
    return ledger_path(tmp_path)


def _write_ledger(tmp_path: Path, text: str) -> Path:
    dest = tmp_path / "data" / "results" / "paper_order_events.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest


def test_production_orders_accepts_valid_stable_ledger_without_bm5_pin(tmp_path: Path):
    _copy_bm5(tmp_path)
    counts = load_authoritative_presentation(root=tmp_path, expected_sha256=None)
    assert counts["ledger_sha256"] == BM5_SHA256
    assert counts["broker_orders"] == EXPECTED_BROKER_ORDERS
    assert counts["filled"] == EXPECTED_FILLED
    assert preview_expected_hash(tmp_path) is None
    assert BM5_PREFIX not in str(preview_expected_hash(tmp_path) or "")


def test_production_orders_does_not_pin_bm5_hash():
    modern = (ISO_ROOT / "dashboard" / "operator_modern.py").read_text(encoding="utf-8")
    preview = (ISO_ROOT / "dashboard" / "preview_data.py").read_text(encoding="utf-8")
    assert "if isolated else None" in modern
    assert "require_expected_hash=isolated" in modern
    assert "if not isolation_active(root):" in preview
    assert "return None" in preview


def test_displayed_orders_derive_from_supplied_ledger(tmp_path: Path):
    ledger = _copy_bm5(tmp_path)
    events = load_events(ledger)
    counts = load_authoritative_presentation(root=tmp_path)
    event_ids = {event.get("broker_order_id") for event in events if event.get("broker_order_id")}
    assert counts["broker_rows"]
    for row in counts["broker_rows"]:
        if row.get("broker_order_id"):
            assert row["broker_order_id"] in event_ids
        assert display_side(row.get("side")) != "Plan"
        assert str(display_side(row.get("side"))).upper() != "PLAN"


def test_historical_plans_do_not_change_broker_counters(tmp_path: Path):
    _copy_bm5(tmp_path)
    counts = load_authoritative_presentation(root=tmp_path, expected_sha256=BM5_SHA256)
    plans = [
        {
            "kind": "plan",
            "side": "PLAN",
            "symbol": "FAKEPLAN",
            "action_disposition": "planned",
            "source_pipeline": "fplan_management",
        }
    ]
    mixed = presentation_order_counts(
        list(counts["broker_rows"]) + list(counts["plan_rows"]) + plans
    )
    assert mixed["broker_orders"] == counts["broker_orders"]
    assert mixed["filled"] == counts["filled"]
    assert mixed["open"] == counts["open"]
    assert mixed["historical_plans"] == counts["historical_plans"] + 1


def test_plan_is_never_an_order_side():
    assert display_side("PLAN") == "Historical plan"
    assert display_side("SELL") == "SELL"
    assert display_side("BUY") == "BUY"


def test_submitted_is_not_filled():
    rows = [
        {
            "kind": "order",
            "side": "SELL",
            "symbol": "GLD",
            "action_disposition": "submitted",
            "broker_order_id": "open-1",
            "terminal_status": None,
            "terminal_status_proven": False,
        },
        {
            "kind": "order",
            "side": "SELL",
            "symbol": "GE",
            "action_disposition": "submitted",
            "broker_order_id": "fill-1",
            "terminal_status": "FILLED",
            "terminal_status_proven": True,
        },
    ]
    counts = presentation_order_counts(rows)
    assert counts["broker_orders"] == 2
    assert counts["filled"] == 1
    assert counts["open"] == 1
    assert counts["filled"] != counts["broker_orders"]
    fields = _order_badge_fields(rows[0])
    assert "Fill" not in fields["Status"]


def test_malformed_ledger_fails_closed(tmp_path: Path):
    _write_ledger(tmp_path, "{not-json\n")
    with pytest.raises(AuthoritativeLedgerError):
        load_authoritative_presentation(root=tmp_path)
    with pytest.raises(AuthoritativeLedgerError):
        load_authoritative_presentation(root=tmp_path, expected_sha256=None)


def test_truncated_ledger_fails_closed(tmp_path: Path):
    src = _copy_bm5(tmp_path).read_bytes()
    _write_ledger(tmp_path, src.decode("utf-8").rstrip("\n"))
    with pytest.raises(AuthoritativeLedgerError):
        load_authoritative_presentation(root=tmp_path)


def test_repeated_hash_instability_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _copy_bm5(tmp_path)
    hashes = iter(["aaa", "bbb"])
    monkeypatch.setattr(
        "dashboard.order_presentation.file_sha256",
        lambda _path: next(hashes),
    )
    with pytest.raises(AuthoritativeLedgerError, match="unstable"):
        load_authoritative_presentation(root=tmp_path)


def test_missing_ledger_fails_closed(tmp_path: Path):
    with pytest.raises(AuthoritativeLedgerError, match="missing"):
        load_authoritative_presentation(root=tmp_path)


def test_no_synthetic_fallback_on_failure(tmp_path: Path):
    _write_ledger(tmp_path, '{"event_id":"x","symbol":"FAKE"}\n')
    with pytest.raises(AuthoritativeLedgerError):
        load_authoritative_presentation(root=tmp_path)


def test_frozen_fixture_remains_hash_pinned(tmp_path: Path):
    _copy_bm5(tmp_path)
    counts = load_authoritative_presentation(
        root=tmp_path,
        expected_sha256=BM5_SHA256,
        require_expected_hash=True,
    )
    assert counts["ledger_sha256"] == BM5_SHA256
    ledger = ledger_path(tmp_path)
    ledger.write_bytes(ledger.read_bytes() + b" ")
    with pytest.raises(AuthoritativeLedgerError, match="hash_mismatch"):
        load_authoritative_presentation(
            root=tmp_path,
            expected_sha256=BM5_SHA256,
            require_expected_hash=True,
        )


def test_missing_expected_preview_hash_fails_closed(tmp_path: Path):
    _copy_bm5(tmp_path)
    with pytest.raises(AuthoritativeLedgerError, match="preview_expected_hash_missing"):
        load_authoritative_presentation(
            root=tmp_path,
            expected_sha256=None,
            require_expected_hash=True,
        )


def test_production_mode_never_falls_back_to_test_fixtures(tmp_path: Path):
    assert isolation_active(tmp_path) is False
    assert preview_data_root(tmp_path) == tmp_path
    fixture = ISO_ROOT / "tmp" / "phase_198T_BP1_fixtures" / "frozen"
    assert preview_data_root(tmp_path) != fixture


def test_top_sidebar_and_status_details_share_one_freshness_result():
    reset_freshness_presentation()
    status = {
        "data_freshness": {
            "class": "AGING_AFTER_CLOSE",
            "heartbeat_timestamp": "2026-09-13T21:44:00+00:00",
        }
    }
    bound = bind_freshness_presentation(status, isolated=False)
    top = freshness_for_render(status, isolated=False)
    sidebar = freshness_for_render(
        {"data_freshness": {"class": "FRESH", "heartbeat_timestamp": "other"}},
        isolated=False,
    )
    details = current_freshness_presentation()
    assert details is bound
    assert top is bound
    assert sidebar is bound
    assert freshness_locations_agree(top, sidebar)
    assert top["value"] == AGING_AFTER_CLOSE_LABEL
    assert sidebar["value"] == AGING_AFTER_CLOSE_LABEL
    assert details["value"] == AGING_AFTER_CLOSE_LABEL
    assert top["tone"] == "info"
    assert top["evaluated_at"] == "2026-09-13T21:44:00+00:00"
    assert top["evaluated_at"] == sidebar["evaluated_at"] == details["evaluated_at"]
    assert top["authority"] == PRODUCTION_AUTHORITY
    assert top["value"] != "Fresh"
    assert "AGING_AFTER_CLOSE" not in str(top["value"])


def test_only_one_evaluation_timestamp_per_render():
    reset_freshness_presentation()
    status = {
        "data_freshness": {
            "class": "STALE",
            "heartbeat_timestamp": "ts-1",
        }
    }
    first = bind_freshness_presentation(status, isolated=False)
    second = bind_freshness_presentation(
        {"data_freshness": {"class": "FRESH", "heartbeat_timestamp": "ts-2"}},
        isolated=False,
    )
    assert first["evaluated_at"] == "ts-1"
    assert second["evaluated_at"] == "ts-1"
    assert first is second


def test_raw_freshness_codes_remain_under_technical_details():
    reset_freshness_presentation()
    status = {
        "data_freshness": {
            "class": "FRESH",
            "heartbeat_timestamp": "ts",
        }
    }
    view = bind_freshness_presentation(status, isolated=False)
    raw = technical_freshness_evidence(view, guard_data_freshness="AGING_AFTER_CLOSE")
    assert raw["label"] == TECHNICAL_RAW_LABEL
    assert raw["raw_class"] == "FRESH"
    assert raw["guard_data_freshness"] == "AGING_AFTER_CLOSE"
    assert view["value"] == "Fresh"
    rail = (ISO_ROOT / "view_results.py").read_text(encoding="utf-8")
    assert "technical_freshness_evidence" in rail
    assert "Raw freshness evidence" in rail


def test_production_never_shows_frozen_preview():
    reset_freshness_presentation()
    status = {"data_freshness": {"class": "STALE", "heartbeat_timestamp": "t"}}
    view = shared_freshness_view(status, isolated=False)
    assert view["value"] != FROZEN_PREVIEW_LABEL
    assert "Frozen preview" not in str(view["value"])
    assert view["note"] != FROZEN_CAPTURE_NOTE


def test_isolated_preview_keeps_frozen_preview_label():
    reset_freshness_presentation()
    status = {"data_freshness": {"class": "FRESH", "heartbeat_timestamp": "t"}}
    view = shared_freshness_view(status, isolated=True)
    assert view["value"] == FROZEN_PREVIEW_LABEL
    assert view["note"] == FROZEN_CAPTURE_NOTE
    assert view["tone"] == "info"


def test_preview_never_silently_switches_to_production(tmp_path: Path):
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "ui_a_isolation_marker.txt").write_text(
        "DO_NOT_WRITE_PRODUCTION\n", encoding="utf-8"
    )
    from dashboard.preview_data import assert_not_production_path

    with pytest.raises(RuntimeError):
        assert_not_production_path(
            PROD_ROOT / "data" / "results" / "paper_order_events.jsonl",
            root=tmp_path,
        )
    assert isolation_active(tmp_path) is True
    pin = preview_expected_hash(tmp_path)
    assert pin == EXPECTED_LEDGER_SHA256


def test_render_helpers_perform_zero_writes_or_broker_calls(tmp_path: Path):
    _copy_bm5(tmp_path)
    before_prod = {
        path: path.stat().st_mtime
        for path in (
            PROD_ROOT / "data" / "results" / "paper_order_events.jsonl",
            PROD_ROOT / "data" / "results" / "paper_order_events.lock",
            PROD_ROOT / "data" / "results" / "performance_accounting_events.jsonl",
        )
        if path.exists()
    }

    def _fail(*_a, **_k):
        raise AssertionError("broker write or lock invoked")

    with (
        patch("urllib.request.urlopen", side_effect=_fail),
        patch("services.paper_order_ledger.append_events", side_effect=_fail),
        patch("services.paper_order_ledger.record_sell_dispatch_events", side_effect=_fail),
    ):
        counts = load_authoritative_presentation(root=tmp_path)
        reset_freshness_presentation()
        shared_freshness_view(
            {"data_freshness": {"class": "AGING_AFTER_CLOSE", "heartbeat_timestamp": "t"}},
            isolated=False,
        )
        assert counts["broker_orders"] == EXPECTED_BROKER_ORDERS
    after_prod = {
        path: path.stat().st_mtime
        for path in (
            PROD_ROOT / "data" / "results" / "paper_order_events.jsonl",
            PROD_ROOT / "data" / "results" / "paper_order_events.lock",
            PROD_ROOT / "data" / "results" / "performance_accounting_events.jsonl",
        )
        if path.exists()
    }
    assert before_prod == after_prod


def test_protected_controls_remain_unchanged():
    buy = (ISO_ROOT / "dashboard" / "paper_buy_add_arm_controls.py").read_text(encoding="utf-8")
    sell = (ISO_ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    calendar = (ISO_ROOT / "services" / "market_calendar.py").read_bytes()
    ledger = (ISO_ROOT / "services" / "paper_order_ledger.py").read_bytes()
    assert 'key="paper_buy_add_arm_submit"' in buy
    assert "ARM PAPER ENTRIES" in buy
    assert 'key="paper_sell_arm_confirm"' in sell
    assert "CONFIRM PAPER SELLS" in sell
    assert hashlib.sha256(calendar).hexdigest() == (
        "338ba5b09572312a4486fa75f28e22bd181dbef59dc897279b25f8938526a0b6"
    )
    assert hashlib.sha256(ledger).hexdigest() == (
        "3b52d05e01571aad698a0ef93a16ec4f56694595df4afc77004c5bb4d30abb3d"
    )


def test_copied_production_ledger_derived_counters(tmp_path: Path):
    assert EVIDENCE_LEDGER.is_file()
    assert _sha256(EVIDENCE_LEDGER) == PROD_LEDGER_SHA256
    dest = tmp_path / "data" / "results" / "paper_order_events.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(EVIDENCE_LEDGER.read_bytes())
    counts = load_authoritative_presentation(root=tmp_path, expected_sha256=None)
    assert counts["ledger_sha256"] == PROD_LEDGER_SHA256
    assert counts["ledger_sha256"] != BM5_SHA256
    events = load_events(dest)
    event_broker_ids = {
        event.get("broker_order_id") for event in events if event.get("broker_order_id")
    }
    bm5_ge_id = "32e66fe5-abea-4807-96ea-9abb81b243ea"
    for row in counts["broker_rows"]:
        assert display_side(row.get("side")) != "Plan"
        assert row.get("symbol") != "FAKE"
        if row.get("broker_order_id"):
            assert row["broker_order_id"] in event_broker_ids
    if bm5_ge_id not in event_broker_ids:
        assert all(row.get("broker_order_id") != bm5_ge_id for row in counts["broker_rows"])
    # Actual counters are derived, not hardcoded.
    assert isinstance(counts["skipped"], int)
    assert isinstance(counts["failed_or_rejected"], int)
    assert isinstance(counts["historical_plans"], int)
    assert isinstance(counts["open"], int)
