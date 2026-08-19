from __future__ import annotations

import copy
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from dashboard.paper_sell_arm_controls import (
    CONFIRM_BUTTON_LABEL,
    LABEL_ARMED,
    SESSION_PHRASE_CLEAR_PENDING_KEY,
    map_sell_arm_visual_status,
)
from services.paper_sell_arm_console import CONFIRM_PAPER_SELLS_PHRASE, build_presentation
from services.paper_sell_arm_store import db_path_for_store_root, inspect_sqlite_arm_snapshot

from tests._phase_198i_sell_fakes import block_real_network, freeze_dispatch_clock
from tests.test_phase_198n_c_sell_cap_slice import (
    NOW,
    READINESS_PACKET_NAME,
    _SEVEN_ACTION_F2,
    _db,
    _load_prod_copy,
    _write_packet,
)


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    # Keep the UI and underlying services inside tmp only.
    block_real_network(monkeypatch)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    monkeypatch.delenv("TRITON_PAPER_SELL_ARM_STORE", raising=False)
    monkeypatch.delenv("TRITON_PAPER_SELL_ARM_STORE_ROOT", raising=False)
    monkeypatch.delenv("TRITON_PAPER_SELL_ARM_RESULTS_ROOT", raising=False)
    monkeypatch.delenv("TRITON_PAPER_SELL_ARM_DB", raising=False)
    monkeypatch.delenv("RESULTS_ROOT", raising=False)

    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)

    # Ensure authorization timestamps align with sqlite inspection.
    import services.paper_sell_arm_console as console_mod

    console_mod._utc_now = lambda: NOW

    yield


def _seed_fresh_packet(tmp_path: Path, *, generated_at: datetime) -> Dict[str, Any]:
    pkt = _load_prod_copy(generated_at=generated_at.isoformat())
    _write_packet(tmp_path, pkt)
    return pkt


def _make_app(tmp_path: Path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PHASE_198O_TMP_ROOT", str(tmp_path))

    def _app():
        # AppTest executes the function in a temp module; ensure imports are
        # resolved in that module scope.
        import os
        from pathlib import Path

        tmp = Path(os.environ["PHASE_198O_TMP_ROOT"])
        from dashboard.paper_sell_arm_controls import render_paper_sell_arm_controls

        render_paper_sell_arm_controls(
            db_path=tmp / "paper_sell_arm.sqlite3",
            results_root=tmp,
            audit_root=tmp,
            paper_mode=True,
            live_blocked=True,
        )

    return AppTest.from_function(_app, default_timeout=30)


def _arm_and_open_challenge(at, tmp_path: Path):
    # Input phrase is empty → Confirm should be disabled.
    confirm = [b for b in at.button if b.label == CONFIRM_BUTTON_LABEL][0]
    assert confirm.disabled is True

    arm = [b for b in at.button if b.label == "Arm Reviewed Sells"][0]
    at = arm.click().run()

    snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
    assert snap["challenge_count"] == 1
    assert snap["authority_count"] == 0
    assert snap["dispatch_count"] == 0
    assert snap["latest_challenge"]["state"] == "OPEN"
    return at


def _confirm_phrase_and_assert_ok(at, tmp_path: Path):
    # Seed exact phrase and confirm.
    at.text_input[0].input(CONFIRM_PAPER_SELLS_PHRASE).run()
    at = [b for b in at.button if b.label == CONFIRM_BUTTON_LABEL][0].click().run()
    assert not at.exception

    snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
    assert snap["challenge_count"] == 1
    assert snap["authority_count"] == 1
    assert snap["dispatch_count"] == 0
    assert snap["latest_challenge"]["state"] == "CONSUMED"
    assert snap["latest_authority"]["state"] == "UNUSED"
    return snap


def _db_update_latest_challenge_expires_at(tmp_path: Path, expires_at: datetime) -> None:
    db = db_path_for_store_root(tmp_path)
    con = sqlite3.connect(db)
    try:
        con.execute(
            """
            UPDATE sell_challenges
            SET expires_at = ?
            WHERE challenge_id = (
                SELECT challenge_id FROM sell_challenges
                ORDER BY created_at DESC
                LIMIT 1
            )
            """,
            (expires_at.isoformat(),),
        )
        con.commit()
    finally:
        con.close()


@pytest.mark.parametrize(
    "phrase,expect_authority",
    [
        ("", 0),  # missing phrase
        ("WRONG PHRASE", 0),  # wrong phrase
    ],
)
def test_198o_missing_or_wrong_phrase_creates_no_authority(
    tmp_path: Path, monkeypatch, phrase, expect_authority
):
    freeze_dispatch_clock(monkeypatch, NOW)
    _seed_fresh_packet(tmp_path, generated_at=NOW)
    at = _make_app(tmp_path, monkeypatch)
    at.run()
    at = _arm_and_open_challenge(at, tmp_path)

    # Optional: try to click even when disabled; it must not create authority.
    if phrase:
        at.text_input[0].input(phrase).run()
    at = [b for b in at.button if b.label == CONFIRM_BUTTON_LABEL][0].click().run()

    snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
    assert snap["authority_count"] == expect_authority
    assert snap["dispatch_count"] == 0
    assert snap["latest_challenge"]["state"] == "OPEN"


def test_198o_exact_confirmation_creates_one_unused_authority_and_consumes_challenge(
    tmp_path: Path, monkeypatch
):
    freeze_dispatch_clock(monkeypatch, NOW)
    _seed_fresh_packet(tmp_path, generated_at=NOW)
    at = _make_app(tmp_path, monkeypatch)
    at.run()

    at = _arm_and_open_challenge(at, tmp_path)
    snap = _confirm_phrase_and_assert_ok(at, tmp_path)

    # Successful confirmation transitions into ARMED/green.
    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        confirmation_phrase="",
        session_challenge_id=None,
        db_path=db_path_for_store_root(tmp_path),
    )
    visual = map_sell_arm_visual_status(pres)
    assert visual["label"] == LABEL_ARMED


def test_198o_rerun_clears_phrase_and_does_not_duplicate_authority(tmp_path: Path, monkeypatch):
    freeze_dispatch_clock(monkeypatch, NOW)
    _seed_fresh_packet(tmp_path, generated_at=NOW)
    at = _make_app(tmp_path, monkeypatch)
    at.run()
    at = _arm_and_open_challenge(at, tmp_path)
    _ = _confirm_phrase_and_assert_ok(at, tmp_path)

    # Confirm that the text_input is cleared on the next valid render.
    # AppTest represents Streamlit text_input state; if the backing attribute name
    # changes across Streamlit versions, the assertions below still remain robust.
    ti = at.text_input[0]
    cleared = getattr(ti, "value", None) == "" or getattr(ti, "input_value", None) == ""
    assert cleared is True

    snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
    assert snap["authority_count"] == 1

    # Post-rerun visual should be ARMED (green).
    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        confirmation_phrase="",
        session_challenge_id=None,
        db_path=db_path_for_store_root(tmp_path),
    )
    visual = map_sell_arm_visual_status(pres)
    assert visual["label"] == LABEL_ARMED

    # Repeated confirmation must not create a second authority (consume-once).
    at = [b for b in at.button if b.label == CONFIRM_BUTTON_LABEL][0].click().run()
    snap2 = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
    assert snap2["authority_count"] == 1
    assert snap2["dispatch_count"] == 0


def test_198o_packet_change_fail_closed(tmp_path: Path, monkeypatch):
    freeze_dispatch_clock(monkeypatch, NOW)
    _seed_fresh_packet(tmp_path, generated_at=NOW)
    at = _make_app(tmp_path, monkeypatch)
    at.run()
    at = _arm_and_open_challenge(at, tmp_path)

    # Mutate packet identity after the challenge is opened.
    changed = copy.deepcopy(_load_prod_copy(generated_at=(NOW + timedelta(seconds=9)).isoformat()))
    _write_packet(tmp_path, changed)

    at.text_input[0].input(CONFIRM_PAPER_SELLS_PHRASE).run()
    at = [b for b in at.button if b.label == CONFIRM_BUTTON_LABEL][0].click().run()
    assert not at.exception

    snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
    assert snap["authority_count"] == 0
    assert snap["dispatch_count"] == 0
    assert snap["latest_challenge"]["state"] == "OPEN"


def test_198o_expired_challenge_fail_closed(tmp_path: Path, monkeypatch):
    freeze_dispatch_clock(monkeypatch, NOW)
    _seed_fresh_packet(tmp_path, generated_at=NOW)
    at = _make_app(tmp_path, monkeypatch)
    at.run()
    at = _arm_and_open_challenge(at, tmp_path)

    # Expire the open challenge in sqlite before confirmation.
    _db_update_latest_challenge_expires_at(tmp_path, NOW - timedelta(minutes=1))

    at.text_input[0].input(CONFIRM_PAPER_SELLS_PHRASE).run()
    at = [b for b in at.button if b.label == CONFIRM_BUTTON_LABEL][0].click().run()
    assert not at.exception

    snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
    assert snap["authority_count"] == 0
    assert snap["dispatch_count"] == 0
    assert snap["latest_challenge"]["state"] == "OPEN"


def test_198o_mismatched_slice_fail_closed(tmp_path: Path, monkeypatch):
    freeze_dispatch_clock(monkeypatch, NOW)
    _seed_fresh_packet(tmp_path, generated_at=NOW)
    at = _make_app(tmp_path, monkeypatch)
    at.run()
    at = _arm_and_open_challenge(at, tmp_path)

    # Mutate one action's notional/qty while keeping freshness. Slice/hash mismatch
    # should fail closed at confirmation time.
    doc = copy.deepcopy(_load_prod_copy(generated_at=NOW.isoformat()))
    actions = doc.get("approval_ready_actions") or []
    for row in actions:
        if str(row.get("symbol")) == "GLD":
            row["qty"] = float(row.get("qty") or 0) + 1.0
            # Keep est_notional consistent with est_price*qty.
            est_price = float(row.get("est_price") or 0) or (
                float(row.get("est_notional") or 0) / float(row.get("qty") or 1)
            )
            row["est_notional"] = float(est_price) * float(row["qty"])
    _write_packet(tmp_path, doc)

    at.text_input[0].input(CONFIRM_PAPER_SELLS_PHRASE).run()
    at = [b for b in at.button if b.label == CONFIRM_BUTTON_LABEL][0].click().run()
    assert not at.exception

    snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
    assert snap["authority_count"] == 0
    assert snap["dispatch_count"] == 0
    assert snap["latest_challenge"]["state"] == "OPEN"


def test_198o_live_execution_remains_blocked_and_no_dispatch(tmp_path: Path, monkeypatch):
    # If the UI confirmation ever started dispatch/broker logic, this would fail.
    import services.paper_sell_arm_runtime_enforcement as enf

    def _blocked(*_a, **_k):
        raise AssertionError("dispatch called from UI confirmation")

    monkeypatch.setattr(enf, "begin_paper_sell_dispatch", _blocked)

    freeze_dispatch_clock(monkeypatch, NOW)
    _seed_fresh_packet(tmp_path, generated_at=NOW)
    at = _make_app(tmp_path, monkeypatch)
    at.run()
    at = _arm_and_open_challenge(at, tmp_path)
    snap = _confirm_phrase_and_assert_ok(at, tmp_path)

    assert snap["latest_challenge"]["payload"].get("live_execution_state") == "BLOCKED"
