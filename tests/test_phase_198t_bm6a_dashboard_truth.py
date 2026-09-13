"""BM6A presentation-only dashboard truth repair. No production writes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dashboard.display_labels import display_label, display_status
from dashboard.operator_modern import _order_badge_fields
from dashboard.order_presentation import (
    AuthoritativeLedgerError,
    load_authoritative_presentation,
    presentation_order_counts,
)
from dashboard.paper_sell_arm_controls import (
    COMPLETED_NEXT_ACTION,
    COMPLETED_SELL_MESSAGE,
    LABEL_COMPLETED,
    LABEL_UNAVAILABLE,
    completed_sell_next_action,
    map_sell_arm_visual_status,
)
from dashboard.preview_data import preview_capture_label
from services.paper_order_ledger import (
    consumed_authority_message,
    dispatch_visibility,
    filter_unified_rows,
    ledger_path,
    load_events,
    reduce_ledger,
)
from tests.fixtures.bm5.authoritative import (
    EXPECTED_BROKER_ORDERS,
    EXPECTED_FAILED,
    EXPECTED_FILLED,
    EXPECTED_LEDGER_EVENTS,
    EXPECTED_LEDGER_SHA256,
    EXPECTED_OPEN,
    EXPECTED_SKIPPED,
    GE_AUTHORITY_ID,
    GE_BROKER_FILLED,
    GE_BROKER_ORDER_ID,
    GE_BROKER_SUBMITTED,
    GE_FILL_PRICE,
    GE_QTY,
    GE_SYMBOL,
    SAME_CYCLE,
    AuthoritativeLedgerError as FreezeLedgerError,
    file_sha256,
    materialize_authoritative_fixture,
)
from tests.fixtures.bm5.build_fixture import materialize_bm5_fixture, write_malformed_packet

ISO_ROOT = Path(__file__).resolve().parents[1]
PROD_ROOT = Path(r"C:\Users\akimw\triton")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_bm5_fixture_order_counters(tmp_path: Path):
    materialize_bm5_fixture(tmp_path)
    counts = load_authoritative_presentation(root=tmp_path, expected_sha256=EXPECTED_LEDGER_SHA256)
    assert counts["ledger_events"] == EXPECTED_LEDGER_EVENTS
    assert counts["ledger_sha256"] == EXPECTED_LEDGER_SHA256
    assert counts["broker_orders"] == EXPECTED_BROKER_ORDERS
    assert counts["filled"] == EXPECTED_FILLED
    assert counts["open"] == EXPECTED_OPEN
    assert counts["skipped"] == EXPECTED_SKIPPED
    assert counts["failed"] == EXPECTED_FAILED
    assert counts["rejected"] == 0
    vis = dispatch_visibility(counts["broker_rows"])
    assert vis["submitted"] == EXPECTED_BROKER_ORDERS
    assert vis["filled"] == EXPECTED_FILLED
    assert vis["open"] == EXPECTED_OPEN
    assert vis["skipped"] == EXPECTED_SKIPPED
    assert vis["failed"] == EXPECTED_FAILED


def test_displayed_orders_come_only_from_copied_ledger(tmp_path: Path):
    materialize_bm5_fixture(tmp_path)
    ledger = ledger_path(tmp_path)
    assert _sha256(ledger) == EXPECTED_LEDGER_SHA256
    events = load_events(ledger)
    assert len(events) == EXPECTED_LEDGER_EVENTS
    counts = load_authoritative_presentation(root=tmp_path, expected_sha256=EXPECTED_LEDGER_SHA256)
    event_broker_ids = {e.get("broker_order_id") for e in events if e.get("broker_order_id")}
    for row in counts["broker_rows"]:
        if row.get("broker_order_id"):
            assert row["broker_order_id"] in event_broker_ids
        fields = _order_badge_fields(row)
        assert fields["Side"] != "Plan"
        assert str(fields["Side"]).upper() != "PLAN"
    ge = next(r for r in counts["broker_rows"] if r.get("symbol") == GE_SYMBOL)
    assert ge.get("side") == "SELL"
    assert float(ge.get("selected_qty") or ge.get("filled_qty") or 0) == GE_QTY
    assert float(ge.get("average_fill_price") or 0) == GE_FILL_PRICE
    assert ge.get("broker_order_id") == GE_BROKER_ORDER_ID
    filled = str(ge.get("filled_timestamp") or "")
    assert filled.startswith("2026-09-08T15:18:04.609")
    events = load_events(ledger)
    ge_events = [e for e in events if e.get("symbol") == GE_SYMBOL]
    assert any(e.get("event_type") == "ACTION_SUBMITTED" for e in ge_events)
    assert any(
        str(e.get("filled_timestamp") or "").startswith("2026-09-08T15:18:04.609")
        for e in ge_events
    )
    recent = (tmp_path / "data" / "results" / "recent_orders.csv").read_text(encoding="utf-8")
    assert "2026-09-08T15:18:03.782" in recent
    assert "2026-09-08T15:18:04.609911" in recent


def test_synthetic_fallback_cannot_replace_authoritative_ledger(tmp_path: Path):
    materialize_bm5_fixture(tmp_path)
    ledger = ledger_path(tmp_path)
    ledger.write_text(
        '{"event_id":"synthetic","event_type":"ACTION_SUBMITTED","symbol":"FAKE"}\n',
        encoding="utf-8",
    )
    assert _sha256(ledger) != EXPECTED_LEDGER_SHA256
    with pytest.raises(AuthoritativeLedgerError):
        load_authoritative_presentation(root=tmp_path, expected_sha256=EXPECTED_LEDGER_SHA256)
    rematerialized = materialize_authoritative_fixture(tmp_path)
    assert rematerialized["ledger_sha256"] == EXPECTED_LEDGER_SHA256
    assert _sha256(ledger) == EXPECTED_LEDGER_SHA256
    counts = load_authoritative_presentation(root=tmp_path, expected_sha256=EXPECTED_LEDGER_SHA256)
    assert all(r.get("symbol") != "FAKE" for r in counts["broker_rows"])
    assert counts["broker_orders"] == EXPECTED_BROKER_ORDERS


def test_missing_frozen_ledger_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.fixtures.bm5 import authoritative as auth

    monkeypatch.setattr(auth, "FROZEN_DIR", tmp_path / "empty_frozen")
    with pytest.raises(FreezeLedgerError):
        materialize_authoritative_fixture(tmp_path / "dest")


def test_audit_rows_cannot_become_displayed_orders(tmp_path: Path):
    materialize_bm5_fixture(tmp_path)
    ledger_rows = [r for r in reduce_ledger(root=tmp_path) if r.get("kind") != "plan"]
    counts = load_authoritative_presentation(root=tmp_path, expected_sha256=EXPECTED_LEDGER_SHA256)
    ledger_keys = {
        (r.get("broker_order_id"), r.get("symbol"), r.get("action_disposition"))
        for r in ledger_rows
    }
    for row in counts["broker_rows"]:
        key = (row.get("broker_order_id"), row.get("symbol"), row.get("action_disposition"))
        assert key in ledger_keys
    mixed = presentation_order_counts(counts["broker_rows"] + counts["plan_rows"])
    assert mixed["broker_orders"] == counts["broker_orders"]


def test_plan_is_never_an_order_side(tmp_path: Path):
    materialize_bm5_fixture(tmp_path)
    counts = load_authoritative_presentation(root=tmp_path, expected_sha256=EXPECTED_LEDGER_SHA256)
    for row in counts["broker_rows"]:
        fields = _order_badge_fields(row)
        assert str(fields["Side"]).upper() != "PLAN"
        assert fields["Side"] != "Plan"
    shown = filter_unified_rows(counts["broker_rows"], side="SELL")
    assert all(str(r.get("side") or "").upper() != "PLAN" for r in shown)
    from dashboard.order_presentation import display_side

    assert display_side("PLAN") == "Historical plan"
    assert display_side("SELL") == "SELL"


def test_ge_consumed_same_cycle_is_sell_completed():
    vis = {
        "submitted": 1,
        "filled": 1,
        "skipped": 0,
        "failed": 0,
        "has_dispatch": True,
        "dispatch_ids": [GE_AUTHORITY_ID],
    }
    visual = map_sell_arm_visual_status(
        {
            "freshness": "MALFORMED",
            "challenge_state": "CONSUMED",
            "authority_state": "CONSUMED",
            "arm_deny_reason": "malformed_packet",
            "plan_hash": "",
            "batch_hash": "",
            "packet": {"status": SAME_CYCLE, "packet_status": SAME_CYCLE},
            "snapshot": {
                "latest_authority": {"state": "CONSUMED", "authority_id": GE_AUTHORITY_ID}
            },
            "last_dispatch_visibility": vis,
            "last_dispatch_message": consumed_authority_message(vis),
        }
    )
    assert visual["label"] == LABEL_COMPLETED == "SELL completed"
    assert visual["code"] == "COMPLETED"
    assert visual["reason"] == COMPLETED_SELL_MESSAGE
    assert completed_sell_next_action(visual, "CONSUMED") == COMPLETED_NEXT_ACTION
    assert COMPLETED_NEXT_ACTION == "Wait for a fresh cycle."
    assert consumed_authority_message(vis) != COMPLETED_SELL_MESSAGE


def test_completed_sell_wording_is_not_repeated():
    visual = {"code": "COMPLETED", "label": LABEL_COMPLETED, "reason": COMPLETED_SELL_MESSAGE}
    assert completed_sell_next_action(visual, "CONSUMED") == "Wait for a fresh cycle."
    assert completed_sell_next_action(visual, "CONSUMED") != COMPLETED_SELL_MESSAGE
    sell = (ISO_ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    assert 'st.write(f"**Next action:** {next_action}")' in sell
    assert "Wait for a fresh cycle." in sell
    assert 'visual.get("code") != "COMPLETED"' in sell


def test_genuine_malformed_packet_stays_fail_closed():
    visual = map_sell_arm_visual_status(
        {
            "freshness": "MALFORMED",
            "challenge_state": "ABSENT",
            "authority_state": "ABSENT",
            "arm_deny_reason": "missing_generated_at",
            "plan_hash": "",
            "batch_hash": "",
            "packet": {"status": "BROKEN"},
            "snapshot": {"latest_authority": {"state": "ABSENT"}},
        }
    )
    assert visual["label"] == LABEL_UNAVAILABLE
    assert visual["code"] == "UNAVAILABLE"
    assert visual["label"] != LABEL_COMPLETED

    consumed_no_evidence = map_sell_arm_visual_status(
        {
            "freshness": "MALFORMED",
            "challenge_state": "ABSENT",
            "authority_state": "CONSUMED",
            "arm_deny_reason": "missing_generated_at",
            "plan_hash": "",
            "batch_hash": "",
            "packet": {"generated_at": None},
            "snapshot": {"latest_authority": {"state": "CONSUMED"}},
            "last_dispatch_visibility": {"submitted": 0, "filled": 0, "has_dispatch": False},
        }
    )
    assert consumed_no_evidence["code"] != "COMPLETED"
    assert consumed_no_evidence["label"] != "SELL ARM — ARMED"


def test_humanized_labels_hide_underscores():
    assert "_" not in display_label("paper_entry_armed")
    assert "_" not in display_label("live_execution_permitted")
    assert display_label("UNAVAILABLE_SAME_CYCLE_DISPATCH_RECONCILIATION") == (
        "Same-cycle dispatch reconciliation"
    )
    assert display_status("SKIPPED_ZERO") == "Skipped — Zero Quantity"
    uuid = "73dee0a3-0e86-4f2e-984e-9946a74a03b5"
    assert display_label(uuid) == uuid
    assert preview_capture_label("2026-09-08T15:50:00+00:00").startswith(
        "Preview data captured at "
    )


def test_button_keys_and_phrases_unchanged():
    sell = (ISO_ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    buy = (ISO_ROOT / "dashboard" / "paper_buy_add_arm_controls.py").read_text(encoding="utf-8")
    assert 'key="paper_sell_arm_reviewed"' in sell
    assert 'key="paper_sell_arm_confirm"' in sell
    assert "CONFIRM PAPER SELLS" in sell
    assert 'key="paper_buy_add_arm_submit"' in buy
    assert "ARM PAPER ENTRIES" in buy
    assert "REVOKE PAPER ENTRIES" in buy


def test_isolation_does_not_touch_production(tmp_path: Path):
    prod_paths = [
        PROD_ROOT / "data" / "results" / "paper_order_events.jsonl",
        PROD_ROOT / "data" / "live" / "paper_sell_arm.sqlite3",
        PROD_ROOT / "data" / "governance" / "paper_sell_arm_authority.json",
    ]
    before = {str(p): (p.stat().st_mtime, _sha256(p) if p.exists() else None) for p in prod_paths}
    materialize_bm5_fixture(tmp_path, include_sqlite=True)
    write_malformed_packet(tmp_path / "data" / "results" / "broken.json")
    load_authoritative_presentation(root=tmp_path, expected_sha256=EXPECTED_LEDGER_SHA256)
    after = {str(p): (p.stat().st_mtime, _sha256(p) if p.exists() else None) for p in prod_paths}
    assert before == after
    assert ISO_ROOT.resolve() != PROD_ROOT.resolve()
    assert file_sha256(ledger_path(tmp_path)) == EXPECTED_LEDGER_SHA256
