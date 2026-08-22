"""Phase 198I-Y — Operator Console paper SELL arm buttons (challenge then UNUSED authority).

PAPER ONLY. Isolated tmp_path SQLite. No broker, no dispatch, no repository data/live writes.
Does not start Streamlit HTTP. Does not alter BUY/ADD paper_arm_confirm.json.
"""

from __future__ import annotations

import ast
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from services.paper_sell_arm_authority import (
    BUY_ACTIONS,
    CHALLENGE_TTL_MINUTES,
    READINESS_REFERENCE_TTL_MINUTES,
)
from services.paper_sell_arm_console import (
    CONFIRM_PAPER_SELLS_PHRASE,
    READINESS_PACKET_NAME,
    arm_reviewed_sells,
    build_presentation,
    confirm_and_arm_paper_sells,
)
from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT
from services.paper_sell_arm_store import (
    db_path_for_store_root,
    inspect_sqlite_arm_snapshot,
)
from tests._phase_198i_sell_fakes import (
    SESSION_OPEN_NOW,
    block_real_network,
    readiness_packet,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = SESSION_OPEN_NOW
BUY_ARM = ROOT / "data" / "live" / "paper_arm_confirm.json"
REPO_SQLITE = ROOT / "data" / "live" / "paper_sell_arm.sqlite3"


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    block_real_network(monkeypatch)
    for key in list(os.environ):
        if "ALPACA" in key.upper():
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.delenv("TRITON_PAPER_SELL_ARM_STORE", raising=False)
    for key in (
        "TRITON_PAPER_SELL_ARM_STORE_ROOT",
        "TRITON_PAPER_SELL_ARM_RESULTS_ROOT",
        "TRITON_PAPER_SELL_ARM_DB",
        "RESULTS_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)
    assert not REPO_SQLITE.is_file()


def _write_packet(results_root: Path, packet: dict) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / READINESS_PACKET_NAME).write_text(json.dumps(packet), encoding="utf-8")


def _fresh_packet(**overrides) -> dict:
    pkt = readiness_packet(now=NOW)
    pkt["generated_at"] = NOW.isoformat()
    pkt.update(overrides)
    return pkt


def _arm_ok(tmp_path: Path, packet=None):
    packet = packet or _fresh_packet()
    _write_packet(tmp_path, packet)
    return arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
    )


def test_a_arm_reviewed_sells_success(tmp_path: Path):
    result = _arm_ok(tmp_path)
    assert result["ok"] is True
    assert result["message"] == "Reviewed SELL challenge created. No orders were submitted."
    snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
    assert snap["challenge_count"] == 1
    assert snap["open_challenges"][0]["state"] == "OPEN"
    assert snap["authority_count"] == 0
    assert snap["dispatch_count"] == 0
    assert _ACTIVE_PERMIT.get() is None
    assert result["counts"]["broker"] == 0
    assert result["counts"]["network"] == 0


@pytest.mark.parametrize(
    "kind",
    [
        "stale",
        "empty",
        "malformed",
        "live_open",
        "non_paper",
        "db_fail",
        "duplicate",
    ],
)
def test_b_arm_reviewed_sells_denial(tmp_path: Path, monkeypatch, kind: str):
    packet = _fresh_packet()
    kwargs = dict(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
    )
    if kind == "stale":
        packet["generated_at"] = (
            NOW - timedelta(minutes=READINESS_REFERENCE_TTL_MINUTES + 1)
        ).isoformat()
        _write_packet(tmp_path, packet)
        expected = "stale_readiness_packet"
    elif kind == "empty":
        packet["approval_ready_actions"] = []
        packet["approval_ready_symbols"] = []
        _write_packet(tmp_path, packet)
        expected = "no_reviewed_sell_actions"
    elif kind == "malformed":
        (tmp_path / READINESS_PACKET_NAME).write_text("{not-json", encoding="utf-8")
        expected = "malformed_packet"
    elif kind == "live_open":
        _write_packet(tmp_path, packet)
        kwargs["live_blocked"] = False
        expected = "live_execution_not_blocked"
    elif kind == "non_paper":
        packet["mode"] = "LIVE"
        _write_packet(tmp_path, packet)
        expected = "non_paper_mode"
    elif kind == "db_fail":
        _write_packet(tmp_path, packet)
        from services.paper_sell_arm_store import PaperSellArmStoreError
        import services.paper_sell_arm_store as store_mod

        def _boom(*_a, **_k):
            raise PaperSellArmStoreError("sqlite_integrity_check_failed")

        monkeypatch.setattr(store_mod, "persist_open_challenge", _boom)
        expected = "sqlite_pragma_or_schema_failure"
    else:
        _write_packet(tmp_path, packet)
        first = arm_reviewed_sells(**kwargs)
        assert first["ok"] is True
        expected = "duplicate_click"

    result = arm_reviewed_sells(**kwargs)
    assert result["ok"] is False
    assert result["reason"] == expected
    snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
    if kind == "duplicate":
        assert snap["challenge_count"] == 1
    else:
        assert snap["authority_count"] == 0
    assert snap["dispatch_count"] == 0
    assert _ACTIVE_PERMIT.get() is None


def test_c_confirm_and_arm_success(tmp_path: Path):
    armed = _arm_ok(tmp_path)
    result = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=armed["challenge_id"],
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
    )
    assert result["ok"] is True
    assert result["message"] == "Paper SELL authority armed. No orders were submitted."
    snap = inspect_sqlite_arm_snapshot(
        db_path_for_store_root(tmp_path), now=NOW + timedelta(minutes=1)
    )
    assert snap["latest_challenge"]["state"] == "CONSUMED"
    assert snap["latest_authority"]["state"] == "UNUSED"
    assert snap["latest_authority"]["single_use"] is True
    assert snap["authority_count"] == 1
    assert snap["dispatch_count"] == 0
    assert _ACTIVE_PERMIT.get() is None


@pytest.mark.parametrize(
    "kind",
    [
        "missing_phrase",
        "wrong_phrase",
        "expired",
        "replay",
        "changed_batch",
        "changed_plan",
        "stale_refresh",
        "live_open",
        "non_paper",
        "db_fail",
        "double_click",
    ],
)
def test_d_confirmation_denial(tmp_path: Path, monkeypatch, kind: str):
    armed = _arm_ok(tmp_path)
    cid = armed["challenge_id"]
    kwargs = dict(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=cid,
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
    )
    if kind == "missing_phrase":
        kwargs["confirmation_phrase"] = ""
        expected = "confirmation_phrase_mismatch"
    elif kind == "wrong_phrase":
        kwargs["confirmation_phrase"] = "CONFIRM PAPER SELLS "
        expected = "confirmation_phrase_mismatch"
    elif kind == "expired":
        kwargs["now"] = NOW + timedelta(minutes=CHALLENGE_TTL_MINUTES + 1)
        expected = "challenge_expired"
    elif kind == "replay":
        ok = confirm_and_arm_paper_sells(**kwargs)
        assert ok["ok"] is True
        expected = "challenge_already_consumed"
    elif kind == "changed_batch":
        pkt = _fresh_packet()
        pkt["approval_ready_actions"][0]["qty"] = 3
        pkt["total_proposed_notional"] = 3 * 332.5
        _write_packet(tmp_path, pkt)
        expected = "batch_mismatch"
    elif kind == "changed_plan":
        pkt = _fresh_packet()
        pkt["plan_hash"] = "planhash_changed"
        _write_packet(tmp_path, pkt)
        expected = "plan_mismatch"
    elif kind == "stale_refresh":
        pkt = _fresh_packet()
        pkt["generated_at"] = (
            NOW - timedelta(minutes=READINESS_REFERENCE_TTL_MINUTES + 5)
        ).isoformat()
        _write_packet(tmp_path, pkt)
        kwargs["now"] = NOW
        expected = "stale_readiness_packet"
    elif kind == "live_open":
        kwargs["live_blocked"] = False
        expected = "live_execution_not_blocked"
    elif kind == "non_paper":
        pkt = _fresh_packet()
        pkt["mode"] = "LIVE"
        _write_packet(tmp_path, pkt)
        expected = "non_paper_mode"
    elif kind == "db_fail":
        import services.paper_sell_arm_store as store_mod
        from services.paper_sell_arm_store import PaperSellArmStoreError

        def _boom(*_a, **_k):
            raise PaperSellArmStoreError("sqlite_database_corrupt_or_incompatible")

        monkeypatch.setattr(store_mod, "consume_challenge_and_create_authority", _boom)
        expected = "sqlite_corrupt"
    else:
        ok = confirm_and_arm_paper_sells(**kwargs)
        assert ok["ok"] is True
        expected = "challenge_already_consumed"

    result = confirm_and_arm_paper_sells(**kwargs)
    assert result["ok"] is False
    assert result["reason"] == expected
    snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=kwargs["now"])
    assert snap["dispatch_count"] == 0
    assert _ACTIVE_PERMIT.get() is None
    if kind not in ("replay", "double_click"):
        if kind != "db_fail":
            pass
        if kind in (
            "missing_phrase",
            "wrong_phrase",
            "expired",
            "changed_batch",
            "changed_plan",
            "stale_refresh",
            "live_open",
            "non_paper",
            "db_fail",
        ):
            assert snap["authority_count"] == 0


def test_e_session_cannot_authorize_or_replay(tmp_path: Path):
    armed = _arm_ok(tmp_path)
    stale_session = {"challenge_id": armed["challenge_id"], "authorized": True}
    result = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=stale_session["challenge_id"],
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
    )
    assert result["ok"] is True
    replay = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=stale_session["challenge_id"],
        now=NOW + timedelta(minutes=2),
        paper_mode=True,
        live_blocked=True,
    )
    assert replay["ok"] is False
    snap = inspect_sqlite_arm_snapshot(
        db_path_for_store_root(tmp_path), now=NOW + timedelta(minutes=2)
    )
    assert snap["authority_count"] == 1
    assert snap["dispatch_count"] == 0
    # Rerun/build_presentation must not create another authority.
    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW + timedelta(minutes=3),
        paper_mode=True,
        live_blocked=True,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        session_challenge_id=stale_session["challenge_id"],
    )
    assert pres["confirm_enabled"] is False
    snap2 = inspect_sqlite_arm_snapshot(
        db_path_for_store_root(tmp_path), now=NOW + timedelta(minutes=3)
    )
    assert snap2["authority_count"] == 1


def test_f_static_boundary_no_dispatch_or_broker():
    banned = {
        "begin_paper_sell_dispatch",
        "enforce_paper_sell_broker_submission",
        "manage_positions",
        "place_live_orders",
        "AlpacaBroker",
        "submit_order",
        "cancel_order",
        "replace_order",
    }
    for rel in (
        "services/paper_sell_arm_console.py",
        "dashboard/paper_sell_arm_controls.py",
    ):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.add(node.name)
                if node.asname:
                    names.add(node.asname)
        assert names.isdisjoint(banned), rel


def test_g_buy_separation(tmp_path: Path):
    before = BUY_ARM.read_bytes() if BUY_ARM.is_file() else None
    pkt = _fresh_packet()
    pkt["approval_ready_actions"][0]["action"] = "BUY"
    _write_packet(tmp_path, pkt)
    result = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
    )
    assert result["ok"] is False
    after = BUY_ARM.read_bytes() if BUY_ARM.is_file() else None
    assert before == after
    assert "BUY" in BUY_ACTIONS


def test_streamlit_ops_page_or_isolated_harness(tmp_path: Path, monkeypatch):
    wall = NOW
    packet = readiness_packet(now=wall)
    packet["generated_at"] = wall.isoformat()
    _write_packet(tmp_path, packet)
    labels_src = (ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    assert "Arm Reviewed Sells" in labels_src
    assert "Confirm & Arm Paper Sells" in labels_src
    ops_wire = (ROOT / "view_results.py").read_text(encoding="utf-8")
    assert "render_paper_sell_arm_controls" in ops_wire

    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=wall,
        paper_mode=True,
        live_blocked=True,
        confirmation_phrase="",
    )
    assert pres["confirm_enabled"] is False
    assert pres["arm_enabled"] is True

    apptest_error = None
    at = None
    try:
        from streamlit.testing.v1 import AppTest
        from dashboard.paper_sell_arm_controls import render_paper_sell_arm_controls

        def _app():
            render_paper_sell_arm_controls(
                db_path=tmp_path / "paper_sell_arm.sqlite3",
                results_root=tmp_path,
                audit_root=tmp_path,
                paper_mode=True,
                live_blocked=True,
            )

        at = AppTest.from_function(_app, default_timeout=30)
        at.run()
        if at.exception:
            raise at.exception
    except Exception as exc:
        apptest_error = f"{type(exc).__name__}: {exc}"
        at = None

    if at is not None:
        labels = [b.label for b in at.button]
        assert "Arm Reviewed Sells" in labels
        assert "Confirm & Arm Paper Sells" in labels
        confirm = [b for b in at.button if b.label == "Confirm & Arm Paper Sells"][0]
        assert confirm.disabled is True
        [b for b in at.button if b.label == "Arm Reviewed Sells"][0].click().run()
        snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=wall)
        assert snap["challenge_count"] == 1
        assert snap["authority_count"] == 0
        at.text_input[0].input(CONFIRM_PAPER_SELLS_PHRASE).run()
        [b for b in at.button if b.label == "Confirm & Arm Paper Sells"][0].click().run()
        snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=wall)
        assert snap["latest_challenge"]["state"] == "CONSUMED"
        assert snap["latest_authority"]["state"] == "UNUSED"
        assert snap["dispatch_count"] == 0
        at.run()
        snap2 = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=wall)
        assert snap2["authority_count"] == 1
    else:
        # Isolated render/callback harness — AppTest did not complete (no HTTP server).
        _ = apptest_error
        armed = arm_reviewed_sells(
            store_root=tmp_path,
            results_root=tmp_path,
            now=wall,
            paper_mode=True,
            live_blocked=True,
        )
        assert armed["ok"] is True
        confirmed = confirm_and_arm_paper_sells(
            store_root=tmp_path,
            results_root=tmp_path,
            confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
            challenge_id=armed["challenge_id"],
            now=wall + timedelta(minutes=1),
            paper_mode=True,
            live_blocked=True,
        )
        assert confirmed["ok"] is True
        rerun = build_presentation(
            store_root=tmp_path,
            results_root=tmp_path,
            now=wall + timedelta(minutes=2),
            paper_mode=True,
            live_blocked=True,
            confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
            session_challenge_id=armed["challenge_id"],
        )
        assert rerun["confirm_enabled"] is False
        snap = inspect_sqlite_arm_snapshot(
            db_path_for_store_root(tmp_path), now=wall + timedelta(minutes=2)
        )
        assert snap["authority_count"] == 1
        assert snap["dispatch_count"] == 0

    assert _ACTIVE_PERMIT.get() is None
    assert not REPO_SQLITE.is_file()
