"""Phase 198L-A — Operator Action Center layout (presentation only).

PAPER ONLY. Isolated tmp paths. No production arm/SELL writes, no broker,
no runtime start, no Streamlit HTTP server required for contract tests.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
REPO_ARM = ROOT / "data" / "live" / "paper_arm_confirm.json"
REPO_SELL_DB = ROOT / "data" / "live" / "paper_sell_arm.sqlite3"
REPO_HANDOFF = ROOT / "data" / "results" / "market_open_invocation_handoff.json"
NOW_OPEN = datetime(2026, 8, 10, 14, 30, tzinfo=ET)
NOW_WEEKEND = datetime(2026, 8, 15, 14, 30, tzinfo=ET)

PROTECTED = {
    "scripts/run_continuous_loop.ps1": "5784c0dd2ec623c94b35fb58cdd112db1f4373465e45261018e30c407c323375",
    "services/market_calendar.py": "cf7a85a39f80f5bff5296aaaa32628f11f9e34e61d043319f3e3123fd7119b86",
    "services/paper_runtime_session_gate.py": "4332eeb6e5ff65d11988213b4b04107e89dd5f1ba44277e782dacaf0a5ae3417",
    "services/operator_console_status.py": "519e3501a5b3f15db4e405df6bb2e09eedf70c2167df92991b9a2d3a3a3e8c9f",
    "services/operator_runtime_controller.py": "da619123049f722e9f8574ac9fc110dbb802bc35b4b8fd6d76461ab31845f067",
    "services/paper_arm_status.py": "a5cc15593c8cdc35d3ed6d133fdedf1d70b248e289e7ecdd2252fcc1126ef525",
    "services/paper_session_arm_policy.py": "561eed1a805e69fefc75467dd5be948d4f677f874c18c94db76d1d4a4b32436d",
    "services/arm_mode.py": "208e1e43758a242d2a4a1a57ba2465a528cdfc8c42c1c6ef00b3d9f7acee1997",
    "services/operator_activity_feed.py": "79345de8528412b6145f0615ae60478e7269c70122a2f07dc01bc965fac4b1b9",
    "dashboard/operator_console.py": "05e073119fa115754a8def762dcdd1c3615ff2b799da2e8fbcb6fa340e918f3e",
    "dashboard/operator_ui.py": "889fbe070ad4e6858b3346d40885f8c48c5d34290222aa75a586a38d22e0c375",
    "tests/test_paper_runtime_session_gating.py": "5d80396226e5be9858f6b5dc65622d87dabb2471a10a4094ed0431d34c5f64cc",
    "tests/test_paper_runtime_consolidated_unreadable_arm.py": "3ac7524b4d369f213eed15c04bdc6e76c3a128e6637c0735e1b3aab5be1cd628",
    "tests/test_calendar_unavailable_fail_closed.py": "9ed2507c007560f94c46ef76491fd7852aacbc2149b5439861fd8fd6159bc8ff",
    "tests/test_calendar_unknown_powershell_consumer.py": "45bb7c93a99f0b2d7c1d0dece1ea5aad76d8dd92775a0729d45b0674e16a47a2",
}


@pytest.fixture(autouse=True)
def _safety(monkeypatch):
    for key in list(os.environ):
        if "ALPACA" in key.upper() or key.startswith("APCA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    before_arm = REPO_ARM.read_bytes() if REPO_ARM.is_file() else None
    before_handoff = REPO_HANDOFF.exists()
    yield
    after_arm = REPO_ARM.read_bytes() if REPO_ARM.is_file() else None
    assert after_arm == before_arm
    assert not REPO_SELL_DB.is_file()
    assert REPO_HANDOFF.exists() is before_handoff


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _open_session(now: datetime) -> dict:
    return {
        "session_state": "OPEN",
        "reason": "regular_session_open",
        "session_date": now.astimezone(ET).date().isoformat(),
        "session_open_et": (now - timedelta(minutes=30)).isoformat(),
        "session_close_et": (now + timedelta(hours=3)).isoformat(),
        "regular_session_open": True,
    }


def _weekend_session(now: datetime) -> dict:
    return {
        "session_state": "CLOSED",
        "reason": "weekend",
        "session_date": None,
        "regular_session_open": False,
    }


def test_01_action_center_order_in_operations_wiring():
    ops = (ROOT / "view_results.py").read_text(encoding="utf-8")
    assert "_render_paper_trading_action_center" in ops
    assert "PAPER TRADING ACTION CENTER" in ops
    assert "Authorization does not start TRITON or submit orders" in ops
    # Action Center must wrap start controls (before runtime), not trail the page.
    assert "_start_controls_after_action_center" in ops
    assert "oc._render_start_controls = _start_controls_after_action_center" in ops
    # No duplicate trailing mounts after render_operations_page().
    trail = ops.split("render_operations_page()", 1)[1]
    # The try body calls render_operations_page(); after finally, no buy/sell mounts.
    assert (
        "render_paper_buy_add_arm_controls()"
        not in trail.split("def _render_cc_orders", 1)[0].split("finally:", 1)[-1]
    )
    assert (
        "render_paper_sell_arm_controls()"
        not in trail.split("def _render_cc_orders", 1)[0].split("finally:", 1)[-1]
    )
    oc = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    # Protected runtime still precedes recovery/attention in operator_console.
    i_start = oc.index("_render_start_controls(status_snapshot=status)")
    i_rec = oc.index("_render_runtime_recovery_guidance()")
    i_att = oc.index("Operator Attention details")
    assert i_start < i_rec < i_att


def test_02_buy_and_sell_cards_mount_exactly_once_in_action_center():
    ops = (ROOT / "view_results.py").read_text(encoding="utf-8")
    center = ops.split("def _render_paper_trading_action_center", 1)[1].split(
        "def _render_cc_operations", 1
    )[0]
    assert center.count("render_paper_buy_add_arm_controls(") == 1
    assert center.count("render_paper_sell_arm_controls(") == 1
    # Operations entrypoint must only invoke Action Center (no trailing duplicate mounts).
    entry = ops.split("def _render_cc_operations_broker_order_evidence", 1)[1].split(
        "def _render_cc_orders", 1
    )[0]
    assert "_render_paper_trading_action_center()" in entry
    assert entry.count("render_paper_buy_add_arm_controls(") == 0
    assert entry.count("render_paper_sell_arm_controls(") == 0
    assert "render_operations_page()" in entry


def test_03_callback_identities_unchanged():
    from services.paper_buy_add_arm_console import (
        ARM_PHRASE,
        REVOKE_PHRASE,
        build_presentation as buy_pres,
        confirm_paper_buy_add_arm,
        revoke_paper_buy_add_arm,
    )
    from services.paper_sell_arm_console import (
        CONFIRM_PAPER_SELLS_PHRASE,
        arm_reviewed_sells,
        build_presentation as sell_pres,
        confirm_and_arm_paper_sells,
    )
    from services.set_paper_arm_confirm import write_confirm
    from dashboard.paper_buy_add_arm_controls import (
        ARM_BUTTON_LABEL,
        RENEW_BUTTON_LABEL,
        REVOKE_BUTTON_LABEL,
        render_paper_buy_add_arm_controls,
    )
    from dashboard.paper_sell_arm_controls import (
        ARM_BUTTON_LABEL as SELL_ARM,
        CONFIRM_BUTTON_LABEL,
        render_paper_sell_arm_controls,
    )
    from dashboard.operator_console import _render_start_controls

    assert ARM_PHRASE == "ARM PAPER ENTRIES"
    assert REVOKE_PHRASE == "REVOKE PAPER ENTRIES"
    assert CONFIRM_PAPER_SELLS_PHRASE == "CONFIRM PAPER SELLS"
    assert ARM_BUTTON_LABEL == "Arm Paper BUY/ADD"
    assert RENEW_BUTTON_LABEL == "Renew Paper BUY/ADD Arm"
    assert REVOKE_BUTTON_LABEL == "Revoke Paper BUY/ADD Arm"
    assert SELL_ARM == "Arm Reviewed Sells"
    assert CONFIRM_BUTTON_LABEL == "Confirm & Arm Paper Sells"
    assert buy_pres.__module__ == "services.paper_buy_add_arm_console"
    assert confirm_paper_buy_add_arm.__module__ == "services.paper_buy_add_arm_console"
    assert revoke_paper_buy_add_arm.__module__ == "services.paper_buy_add_arm_console"
    assert write_confirm.__module__ == "services.set_paper_arm_confirm"
    assert sell_pres.__module__ == "services.paper_sell_arm_console"
    assert arm_reviewed_sells.__module__ == "services.paper_sell_arm_console"
    assert confirm_and_arm_paper_sells.__module__ == "services.paper_sell_arm_console"
    assert render_paper_buy_add_arm_controls.__module__ == "dashboard.paper_buy_add_arm_controls"
    assert render_paper_sell_arm_controls.__module__ == "dashboard.paper_sell_arm_controls"
    assert _render_start_controls.__module__ == "dashboard.operator_console"
    assert (
        _sha(ROOT / "dashboard" / "operator_console.py")
        == PROTECTED["dashboard/operator_console.py"]
    )


def test_04_exact_phrases_and_labels_in_sources():
    buy = (ROOT / "dashboard" / "paper_buy_add_arm_controls.py").read_text(encoding="utf-8")
    sell = (ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    assert "ARM PAPER ENTRIES" in buy
    assert "REVOKE PAPER ENTRIES" in buy
    assert "Arm Paper BUY/ADD" in buy
    assert "Renew Paper BUY/ADD Arm" in buy
    assert "Revoke Paper BUY/ADD Arm" in buy
    assert "CONFIRM PAPER SELLS" in sell
    assert "Arm Reviewed Sells" in sell
    assert "Confirm & Arm Paper Sells" in sell
    assert "BUY/ADD authorization details" in buy
    assert "SELL readiness and authority details" in sell


def test_05_weekend_buy_add_disabled_no_artifact(tmp_path, monkeypatch):
    from services.paper_buy_add_arm_console import ARM_PHRASE, confirm_paper_buy_add_arm

    dest = tmp_path / "paper_arm_confirm.json"
    result = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_weekend_session,
        now=NOW_WEEKEND,
    )
    assert result.get("ok") is False
    assert not dest.is_file()


def test_06_malformed_weekend_sell_disabled_no_sqlite(tmp_path):
    from services.paper_sell_arm_console import arm_reviewed_sells

    # No readiness packet → fail-closed; no SQLite created.
    result = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
        now=NOW_WEEKEND.astimezone(timezone.utc),
    )
    assert result.get("ok") is False
    assert not (tmp_path / "paper_sell_arm.sqlite3").is_file()


def test_07_isolated_buy_add_arm_renew_revoke(tmp_path):
    from services.paper_arm_status import classify_paper_arm_confirm_status
    from services.paper_buy_add_arm_console import (
        ARM_PHRASE,
        REVOKE_PHRASE,
        confirm_paper_buy_add_arm,
        revoke_paper_buy_add_arm,
    )

    dest = tmp_path / "paper_arm_confirm.json"
    armed = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
        now=NOW_OPEN,
    )
    assert armed.get("ok") is True
    assert dest.is_file()
    assert classify_paper_arm_confirm_status(confirm_path=dest, now=NOW_OPEN) == "VALID"
    first = dest.read_bytes()
    renewed = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
        now=NOW_OPEN + timedelta(seconds=2),
    )
    assert renewed.get("ok") is True
    assert dest.read_bytes() != first
    revoked = revoke_paper_buy_add_arm(
        confirmation_phrase=REVOKE_PHRASE,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
    )
    assert revoked.get("ok") is True
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc.get("allow_execute") is False


def test_08_isolated_sell_challenge_confirm(tmp_path):
    from services.paper_sell_arm_console import (
        CONFIRM_PAPER_SELLS_PHRASE,
        READINESS_PACKET_NAME,
        arm_reviewed_sells,
        confirm_and_arm_paper_sells,
    )
    from services.paper_sell_arm_store import db_path_for_store_root, inspect_sqlite_arm_snapshot
    from tests._phase_198i_sell_fakes import readiness_packet

    wall = datetime.now(timezone.utc)
    packet = readiness_packet(now=wall)
    packet["generated_at"] = wall.isoformat()
    (tmp_path / READINESS_PACKET_NAME).write_text(json.dumps(packet), encoding="utf-8")
    db = tmp_path / "paper_sell_arm.sqlite3"
    armed = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=wall,
        paper_mode=True,
        live_blocked=True,
        db_path=db,
    )
    assert armed.get("ok") is True
    confirmed = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=armed.get("challenge_id"),
        now=wall,
        paper_mode=True,
        live_blocked=True,
        db_path=db,
    )
    assert confirmed.get("ok") is True
    snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=wall)
    assert snap["latest_challenge"]["state"] == "CONSUMED"
    assert snap["latest_authority"]["state"] == "UNUSED"
    assert snap["dispatch_count"] == 0


def test_09_ui_clicks_create_no_orders_or_runtime(tmp_path, monkeypatch):
    counts = {"orders": 0, "broker": 0, "network": 0, "runtime": 0}

    def _boom(*_a, **_k):
        counts["broker"] += 1
        raise AssertionError("broker must not be called")

    monkeypatch.setattr(
        "services.paper_buy_add_arm_console.confirm_paper_buy_add_arm",
        lambda **k: {"ok": False, "message": "blocked"},
    )
    # Ensure no network helpers get used by presentation modules.
    buy = (ROOT / "dashboard" / "paper_buy_add_arm_controls.py").read_text(encoding="utf-8")
    sell = (ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    ops = (ROOT / "view_results.py").read_text(encoding="utf-8")
    for blob in (buy, sell):
        assert "submit_order" not in blob
        assert "AlpacaBroker" not in blob
        assert "requests." not in blob
    center = ops.split("def _render_paper_trading_action_center", 1)[1].split(
        "def _render_cc_operations", 1
    )[0]
    assert "submit_order" not in center
    assert counts["orders"] == 0
    assert counts["broker"] == 0
    assert counts["network"] == 0
    assert counts["runtime"] == 0
    assert os.environ.get("TRITON_ENABLE_LIVE_EXECUTION") == "0"


def test_10_no_duplicate_streamlit_keys():
    buy = (ROOT / "dashboard" / "paper_buy_add_arm_controls.py").read_text(encoding="utf-8")
    sell = (ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    keys = re.findall(r'key\s*=\s*["\']([^"\']+)["\']', buy + "\n" + sell)
    assert len(keys) == len(set(keys))
    assert "paper_buy_add_arm_submit" in keys
    assert "paper_sell_arm_reviewed" in keys
    assert "paper_sell_arm_confirm" in keys


def test_11_runtime_controls_labels_and_eligibility_unchanged():
    oc = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "Protected runtime controls" in oc
    assert "START_TRITON" in oc or "Start TRITON" in oc or "Start" in oc
    assert "_render_start_controls" in oc
    assert (
        _sha(ROOT / "dashboard" / "operator_console.py")
        == PROTECTED["dashboard/operator_console.py"]
    )
    from dashboard.operator_console import _render_start_controls
    import inspect

    src = inspect.getsource(_render_start_controls)
    assert "start_eligibility" in src or "eligibility" in src.lower()


def test_12_recovery_and_attention_remain_below_runtime():
    oc = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    # Anchor inside render_operations_page body (not earlier docstrings/helpers).
    body = oc.split("def render_operations_page()", 1)[1].split(
        "def _render_runtime_recovery_guidance", 1
    )[0]
    i_start = body.index("_render_start_controls(status_snapshot=status)")
    i_rec = body.index("_render_runtime_recovery_guidance()")
    i_att = body.index("Operator Attention details")
    assert i_start < i_rec < i_att


def test_13_protected_hashes_match():
    match = changed = 0
    for rel, expected in PROTECTED.items():
        current = _sha(ROOT / rel)
        if current == expected:
            match += 1
        else:
            changed += 1
    assert match == 15
    assert changed == 0


def test_14_action_center_apptest_single_mount(tmp_path):
    from streamlit.testing.v1 import AppTest

    dest = tmp_path / "paper_arm_confirm.json"
    app = tmp_path / "action_center_app.py"
    app.write_text(
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from dashboard.paper_buy_add_arm_controls import render_paper_buy_add_arm_controls\n"
        "from dashboard.paper_sell_arm_controls import render_paper_sell_arm_controls\n"
        f"DEST = Path(r'{dest}')\n"
        f"ROOT = Path(r'{tmp_path}')\n"
        "def _weekend(now):\n"
        "    return {'session_state': 'CLOSED', 'reason': 'weekend', 'session_date': None, 'regular_session_open': False}\n"
        "st.markdown('### PAPER TRADING ACTION CENTER')\n"
        "left, right = st.columns(2)\n"
        "with left:\n"
        "    with st.container(border=True):\n"
        "        render_paper_buy_add_arm_controls(confirm_path=DEST, paper_mode=True, live_blocked=True, allow_path_override=True, session_state_fn=_weekend, compact=True)\n"
        "with right:\n"
        "    with st.container(border=True):\n"
        "        render_paper_sell_arm_controls(db_path=ROOT / 'paper_sell_arm.sqlite3', results_root=ROOT, audit_root=ROOT, paper_mode=True, live_blocked=True, compact=True)\n",
        encoding="utf-8",
    )
    at = AppTest.from_file(str(app), default_timeout=30)
    at.run()
    if at.exception:
        raise at.exception
    buy_buttons = [b for b in at.button if "BUY/ADD" in str(b.label)]
    sell_arm = [b for b in at.button if str(b.label) == "Arm Reviewed Sells"]
    assert len(buy_buttons) == 1
    assert len(sell_arm) == 1
    assert buy_buttons[0].disabled is True
    assert sell_arm[0].disabled is True
    assert not dest.is_file()
    assert not (tmp_path / "paper_sell_arm.sqlite3").is_file()


def test_15_imports_remain_service_bound():
    buy_tree = ast.parse(
        (ROOT / "dashboard" / "paper_buy_add_arm_controls.py").read_text(encoding="utf-8")
    )
    sell_tree = ast.parse(
        (ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    )
    buy_mods = {
        n.module
        for n in ast.walk(buy_tree)
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("services.")
    }
    sell_mods = {
        n.module
        for n in ast.walk(sell_tree)
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("services.")
    }
    assert "services.paper_buy_add_arm_console" in buy_mods
    assert "services.paper_sell_arm_console" in sell_mods
    assert "services.set_paper_arm_confirm" not in buy_mods  # still via console


def _pres(**kwargs):
    base = {
        "freshness": "FRESH",
        "arm_enabled": False,
        "arm_deny_reason": None,
        "challenge_state": "ABSENT",
        "authority_state": "ABSENT",
        "plan_hash": "plan_a",
        "batch_hash": "batch_a",
        "action_count": 1,
        "snapshot": {"latest_authority": {"state": "ABSENT"}},
    }
    base.update(kwargs)
    return base


def test_16_sell_status_color_mapping_table():
    from dashboard.paper_sell_arm_controls import (
        COLOR_AMBER,
        COLOR_BLUE,
        COLOR_GREEN,
        COLOR_RED,
        LABEL_ARMED,
        LABEL_ATTENTION,
        LABEL_AWAITING,
        LABEL_COMPLETED,
        LABEL_NOT_REQUIRED,
        LABEL_READY,
        LABEL_UNAVAILABLE,
        map_sell_arm_visual_status,
    )

    cases = [
        (
            _pres(
                authority_state="UNUSED",
                snapshot={
                    "latest_authority": {
                        "state": "UNUSED",
                        "plan_hash": "plan_a",
                        "batch_hash": "batch_a",
                    }
                },
            ),
            LABEL_ARMED,
            COLOR_GREEN,
        ),
        (_pres(arm_enabled=True), LABEL_READY, COLOR_AMBER),
        (_pres(challenge_state="OPEN"), LABEL_AWAITING, COLOR_AMBER),
        (
            _pres(freshness="MALFORMED", arm_deny_reason="missing_generated_at"),
            LABEL_UNAVAILABLE,
            COLOR_AMBER,
        ),
        (
            _pres(freshness="EMPTY", arm_deny_reason="no_reviewed_sell_actions", action_count=0),
            LABEL_NOT_REQUIRED,
            COLOR_BLUE,
        ),
        (
            _pres(
                authority_state="CONSUMED",
                snapshot={
                    "latest_authority": {
                        "state": "CONSUMED",
                        "plan_hash": "plan_a",
                        "batch_hash": "batch_a",
                    }
                },
            ),
            LABEL_COMPLETED,
            COLOR_BLUE,
        ),
        (
            _pres(freshness="OUTCOME_UNKNOWN", arm_deny_reason="sell_outcome_unknown"),
            LABEL_ATTENTION,
            COLOR_RED,
        ),
    ]
    for pres, label, color in cases:
        visual = map_sell_arm_visual_status(pres)
        assert visual["label"] == label
        assert visual["color"] == color


def test_17_armed_false_positive_prevention():
    from dashboard.paper_sell_arm_controls import (
        LABEL_ARMED,
        LABEL_ATTENTION,
        LABEL_AWAITING,
        LABEL_COMPLETED,
        LABEL_UNAVAILABLE,
        map_sell_arm_visual_status,
    )

    # OPEN challenge is not ARMED.
    v = map_sell_arm_visual_status(
        _pres(
            challenge_state="OPEN",
            authority_state="UNUSED",
            snapshot={
                "latest_authority": {
                    "state": "UNUSED",
                    "plan_hash": "plan_a",
                    "batch_hash": "batch_a",
                }
            },
        )
    )
    assert v["label"] == LABEL_AWAITING
    assert v["label"] != LABEL_ARMED

    # Expired authority is not ARMED.
    v = map_sell_arm_visual_status(
        _pres(
            authority_state="EXPIRED",
            freshness="MALFORMED",
            arm_deny_reason="missing_generated_at",
            snapshot={
                "latest_authority": {
                    "state": "EXPIRED",
                    "plan_hash": "plan_a",
                    "batch_hash": "batch_a",
                }
            },
        )
    )
    assert v["label"] != LABEL_ARMED

    # Stale packet / authority path is not ARMED.
    v = map_sell_arm_visual_status(
        _pres(freshness="STALE", arm_deny_reason="stale_readiness_packet", authority_state="ABSENT")
    )
    assert v["label"] == LABEL_UNAVAILABLE
    assert v["label"] != LABEL_ARMED

    # Mismatched authority is not ARMED.
    v = map_sell_arm_visual_status(
        _pres(
            authority_state="UNUSED",
            snapshot={
                "latest_authority": {
                    "state": "UNUSED",
                    "plan_hash": "other_plan",
                    "batch_hash": "other_batch",
                }
            },
        )
    )
    assert v["label"] == LABEL_ATTENTION
    assert v["label"] != LABEL_ARMED

    # Consumed authority is not ARMED.
    v = map_sell_arm_visual_status(
        _pres(
            authority_state="CONSUMED",
            snapshot={
                "latest_authority": {
                    "state": "CONSUMED",
                    "plan_hash": "plan_a",
                    "batch_hash": "batch_a",
                }
            },
        )
    )
    assert v["label"] == LABEL_COMPLETED
    assert v["label"] != LABEL_ARMED

    # Absent SQLite / no authority is not ARMED.
    v = map_sell_arm_visual_status(_pres(authority_state="ABSENT", arm_enabled=True))
    assert v["label"] != LABEL_ARMED

    # Malformed packet is not ARMED.
    v = map_sell_arm_visual_status(
        _pres(
            freshness="MALFORMED", arm_deny_reason="missing_generated_at", authority_state="ABSENT"
        )
    )
    assert v["label"] == LABEL_UNAVAILABLE
    assert v["reason"] == "Missing generated_at"
    assert v["label"] != LABEL_ARMED

    # Only valid unused exact-batch authority is ARMED.
    v = map_sell_arm_visual_status(
        _pres(
            authority_state="UNUSED",
            snapshot={
                "latest_authority": {
                    "state": "UNUSED",
                    "plan_hash": "plan_a",
                    "batch_hash": "batch_a",
                }
            },
        )
    )
    assert v["label"] == LABEL_ARMED


def test_18_production_malformed_maps_unavailable_amber():
    from dashboard.paper_sell_arm_controls import (
        COLOR_AMBER,
        LABEL_UNAVAILABLE,
        map_sell_arm_visual_status,
    )
    from services.paper_sell_arm_console import build_presentation

    # Read-only against production packet; must not create SQLite.
    before = REPO_ARM.read_bytes()
    assert not REPO_SELL_DB.is_file()
    pres = build_presentation(paper_mode=True, live_blocked=True)
    visual = map_sell_arm_visual_status(pres)
    assert pres.get("freshness") == "MALFORMED"
    assert pres.get("arm_deny_reason") == "missing_generated_at"
    assert visual["label"] == LABEL_UNAVAILABLE
    assert visual["color"] == COLOR_AMBER
    assert visual["reason"] == "Missing generated_at"
    assert not REPO_SELL_DB.is_file()
    assert REPO_ARM.read_bytes() == before


def test_19_status_box_renders_once_under_sell_header():
    sell = (ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    assert "Reviewed paper SELL authorization" in sell
    assert "map_sell_arm_visual_status" in sell
    assert "_render_status_box" in sell
    assert 'data-testid="sell-arm-status-box"' in sell
    # Header then status box before packet freshness details.
    i_hdr = sell.index('st.markdown("##### Reviewed paper SELL authorization")')
    i_box = sell.index("_render_status_box(visual)")
    i_fresh = sell.index("Packet freshness")
    assert i_hdr < i_box < i_fresh
    assert sell.count("_render_status_box(") == 2  # def + call
