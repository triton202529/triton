"""Phase 198J-A — Operator Console paper BUY/ADD arm control.

PAPER ONLY. Isolated tmp paths. No repository arm writes, no Streamlit HTTP server,
no broker, no runtime start.
"""

from __future__ import annotations

import ast
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from services.paper_arm_status import classify_paper_arm_confirm_status
from services.paper_buy_add_arm_console import (
    ARM_PHRASE,
    REVOKE_PHRASE,
    build_presentation,
    confirm_paper_buy_add_arm,
    evaluate_eligibility,
    revoke_paper_buy_add_arm,
)
from services.set_paper_arm_confirm import write_confirm

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
REPO_ARM = ROOT / "data" / "live" / "paper_arm_confirm.json"
REPO_SELL_DB = ROOT / "data" / "live" / "paper_sell_arm.sqlite3"
NOW_OPEN = datetime(2026, 8, 10, 14, 30, tzinfo=ET)  # Monday session
NOW_WEEKEND = datetime(2026, 8, 15, 14, 30, tzinfo=ET)  # Saturday


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    for key in list(os.environ):
        if "ALPACA" in key.upper() or key.startswith("APCA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    for key in (
        "TRITON_PAPER_ARM_CONFIRM_PATH",
        "TRITON_PAPER_ARM_CONFIRM_ROOT",
        "TRITON_TEST_PAPER_ARM_CONFIRM",
    ):
        monkeypatch.delenv(key, raising=False)
    before = REPO_ARM.read_bytes() if REPO_ARM.is_file() else None
    yield
    after = REPO_ARM.read_bytes() if REPO_ARM.is_file() else None
    assert after == before
    assert not REPO_SELL_DB.is_file()


def _open_session(now: datetime) -> dict:
    close = now + timedelta(hours=3)
    open_ = now - timedelta(minutes=30)
    return {
        "session_state": "OPEN",
        "reason": "regular_session_open",
        "session_date": now.astimezone(ET).date().isoformat(),
        "session_open_et": open_.isoformat(),
        "session_close_et": close.isoformat(),
        "regular_session_open": True,
    }


def _bounded_session(now: datetime) -> dict:
    now_et = now.astimezone(ET)
    close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    open_ = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    return {
        "session_state": "OPEN",
        "reason": "regular_session_open",
        "session_date": now_et.date().isoformat(),
        "session_open_et": open_.isoformat(),
        "session_close_et": close.isoformat(),
        "regular_session_open": True,
    }


def _weekend_session(now: datetime) -> dict:
    return {
        "session_state": "CLOSED",
        "reason": "weekend",
        "session_date": None,
        "regular_session_open": False,
    }


def _unknown_session(_now: datetime) -> dict:
    return {
        "session_state": "UNKNOWN",
        "reason": "calendar_import_unavailable",
        "session_date": None,
        "regular_session_open": False,
    }


def _arm_path(tmp_path: Path) -> Path:
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True, exist_ok=True)
    return live / "paper_arm_confirm.json"


def test_a_valid_paper_session_arm(tmp_path):
    dest = _arm_path(tmp_path)
    result = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
    )
    assert result["ok"] is True
    assert result["counts"]["writes"] == 1
    assert result["counts"]["orders"] == 0
    assert result["counts"]["runtime_starts"] == 0
    assert dest.is_file()
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert set(doc) >= {"timestamp", "mode", "expires_at", "allow_execute"}
    assert doc["mode"] == "ASSISTED"
    assert doc["allow_execute"] is True
    assert classify_paper_arm_confirm_status(confirm_path=dest, now=NOW_OPEN) == "VALID"
    assert result["counts"]["broker"] == 0


def test_b_wrong_or_empty_phrase_writes_nothing(tmp_path):
    dest = _arm_path(tmp_path)
    for phrase in ("", "arm paper entries", "CONFIRM PAPER SELLS"):
        result = confirm_paper_buy_add_arm(
            confirmation_phrase=phrase,
            now=NOW_OPEN,
            paper_mode=True,
            live_blocked=True,
            confirm_path=dest,
            allow_path_override=True,
            session_state_fn=_open_session,
        )
        assert result["ok"] is False
        assert result["wrote"] is False
    assert not dest.is_file()


def test_c_weekend_disabled_no_artifact(tmp_path):
    dest = _arm_path(tmp_path)
    elig = evaluate_eligibility(
        now=NOW_WEEKEND,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_weekend_session,
    )
    assert elig["arm_enabled"] is False
    assert elig["reason"] == "weekend"
    result = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=NOW_WEEKEND,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_weekend_session,
    )
    assert result["ok"] is False
    assert not dest.is_file()


def test_d_calendar_unknown_disabled_no_artifact(tmp_path):
    dest = _arm_path(tmp_path)
    result = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_unknown_session,
    )
    assert result["ok"] is False
    assert result["reason"] in {"calendar_unavailable", "calendar_unknown"}
    assert not dest.is_file()


def test_e_live_not_blocked(tmp_path):
    dest = _arm_path(tmp_path)
    result = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=False,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
    )
    assert result["ok"] is False
    assert result["reason"] == "live_execution_not_blocked"
    assert not dest.is_file()


def test_f_wrong_mode(tmp_path):
    dest = _arm_path(tmp_path)
    result = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=NOW_OPEN,
        paper_mode=False,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
    )
    assert result["ok"] is False
    assert result["reason"] == "non_paper_mode"
    assert not dest.is_file()


def test_g_expiry_does_not_cross_session_boundary(tmp_path):
    dest = _arm_path(tmp_path)
    result = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_bounded_session,
    )
    assert result["ok"] is True
    exp = datetime.fromisoformat(str(result["expires_at"]).replace("Z", "+00:00"))
    close = NOW_OPEN.replace(hour=16, minute=0, second=0, microsecond=0)
    limit = close.astimezone(timezone.utc) + timedelta(minutes=60)
    assert exp <= limit + timedelta(seconds=1)


def test_h_renewal_requires_phrase_and_replaces(tmp_path):
    dest = _arm_path(tmp_path)
    first = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
    )
    assert first["ok"] is True
    doc1 = json.loads(dest.read_text(encoding="utf-8"))
    later = NOW_OPEN + timedelta(minutes=5)
    denied = confirm_paper_buy_add_arm(
        confirmation_phrase="",
        now=later,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
    )
    assert denied["ok"] is False
    assert json.loads(dest.read_text(encoding="utf-8")) == doc1
    second = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=later,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
    )
    assert second["ok"] is True
    assert second["counts"]["writes"] == 1
    doc2 = json.loads(dest.read_text(encoding="utf-8"))
    assert doc2["timestamp"] != doc1["timestamp"]
    assert classify_paper_arm_confirm_status(confirm_path=dest, now=later) == "VALID"


def test_i_stale_session_state_does_not_authorize(tmp_path):
    dest = _arm_path(tmp_path)
    pres = build_presentation(
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
    )
    assert pres["arm_enabled"] is True
    assert not dest.is_file()
    result = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
        idempotency_key="click-1",
        last_idempotency_key="click-1",
    )
    assert result["ok"] is False
    assert not dest.is_file()


def test_j_path_override_poisoning(tmp_path, monkeypatch):
    dest = _arm_path(tmp_path)
    override = tmp_path / "override" / "paper_arm_confirm.json"
    monkeypatch.setenv("TRITON_PAPER_ARM_CONFIRM_PATH", str(override))
    production = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=None,
        allow_path_override=False,
        session_state_fn=_open_session,
    )
    assert production["ok"] is False
    assert production["reason"] == "configuration_blocked"
    injected_denied = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=override,
        allow_path_override=False,
        session_state_fn=_open_session,
    )
    assert injected_denied["ok"] is False
    assert not dest.is_file()
    assert not override.is_file()
    monkeypatch.delenv("TRITON_PAPER_ARM_CONFIRM_PATH", raising=False)
    allowed = confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
    )
    assert allowed["ok"] is True
    assert dest.is_file()


def test_k_buy_sell_separation(tmp_path):
    dest = _arm_path(tmp_path)
    confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
    )
    assert not (tmp_path / "paper_sell_arm.sqlite3").is_file()
    assert not REPO_SELL_DB.is_file()
    from services.paper_sell_arm_authority import BUY_ACTIONS, SELL_ACTIONS
    from services.paper_sell_arm_console import CONFIRM_PAPER_SELLS_PHRASE

    assert ARM_PHRASE != CONFIRM_PAPER_SELLS_PHRASE
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["allow_execute"] is True
    assert "sell" not in json.dumps(doc).lower()
    assert SELL_ACTIONS
    assert "BUY" in BUY_ACTIONS or "OPEN_NEW" in BUY_ACTIONS


def test_l_static_boundary():
    banned = {
        "paper_entry_loop_stage",
        "run_market_open",
        "run_continuous_loop",
        "manage_positions",
        "place_live_orders",
        "begin_paper_sell_dispatch",
        "AlpacaBroker",
        "execute_start_triton",
        "submit_order",
    }
    files = [
        ROOT / "services" / "paper_buy_add_arm_console.py",
        ROOT / "dashboard" / "paper_buy_add_arm_controls.py",
        ROOT / "services" / "set_paper_arm_confirm.py",
    ]
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.add(node.name)
        assert names.isdisjoint(banned), path.name


def test_revoke_existing_disarm_boundary(tmp_path):
    dest = _arm_path(tmp_path)
    confirm_paper_buy_add_arm(
        confirmation_phrase=ARM_PHRASE,
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
    )
    denied = revoke_paper_buy_add_arm(
        confirmation_phrase="nope",
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
    )
    assert denied["ok"] is False
    revoked = revoke_paper_buy_add_arm(
        confirmation_phrase=REVOKE_PHRASE,
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
    )
    assert revoked["ok"] is True
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["allow_execute"] is False
    assert classify_paper_arm_confirm_status(confirm_path=dest, now=NOW_OPEN) == "INVALID"
    pres = build_presentation(
        now=NOW_OPEN,
        paper_mode=True,
        live_blocked=True,
        confirm_path=dest,
        allow_path_override=True,
        session_state_fn=_open_session,
    )
    assert pres["status"] == "REVOKED"


def test_cli_writer_is_shared():
    src = (ROOT / "services" / "set_paper_arm_confirm.py").read_text(encoding="utf-8")
    assert "write_confirm(" in src
    assert "def write_confirm" in src
    console = (ROOT / "services" / "paper_buy_add_arm_console.py").read_text(encoding="utf-8")
    assert "from services.set_paper_arm_confirm import" in console
    assert "write_confirm" in console
    cli = (ROOT / "services" / "set_paper_arm_confirm.py").read_text(encoding="utf-8")
    assert "p = write_confirm(args.ttl, allow_execute=not args.no_execute)" in cli


def test_streamlit_weekend_disabled(tmp_path):
    dest = _arm_path(tmp_path)
    app = tmp_path / "weekend_app.py"
    app.write_text(
        "from pathlib import Path\n"
        "from datetime import datetime\n"
        "from zoneinfo import ZoneInfo\n"
        "from dashboard.paper_buy_add_arm_controls import render_paper_buy_add_arm_controls\n"
        f"DEST = Path({str(dest).replace(chr(92), '/')!r})\n"
        "def _weekend(now):\n"
        "    return {'session_state': 'CLOSED', 'reason': 'weekend', 'session_date': None, 'regular_session_open': False}\n"
        "render_paper_buy_add_arm_controls(confirm_path=DEST, paper_mode=True, live_blocked=True, allow_path_override=True, session_state_fn=_weekend)\n",
        encoding="utf-8",
    )
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(app), default_timeout=30)
    at.run()
    if at.exception:
        raise at.exception
    labels = " ".join(str(b.label) for b in at.button)
    assert "Arm Paper BUY/ADD" in labels
    arm_btn = [b for b in at.button if str(b.label) == "Arm Paper BUY/ADD"][0]
    assert arm_btn.disabled is True
    assert not dest.is_file()


def test_streamlit_controlled_session_arm_and_renewal(tmp_path):
    dest = _arm_path(tmp_path)
    app = tmp_path / "open_app.py"
    app.write_text(
        "from pathlib import Path\n"
        "from datetime import timedelta\n"
        "from dashboard.paper_buy_add_arm_controls import render_paper_buy_add_arm_controls\n"
        f"DEST = Path({str(dest).replace(chr(92), '/')!r})\n"
        "def _open(now):\n"
        "    return {'session_state': 'OPEN', 'reason': 'regular_session_open', 'session_date': '2026-08-10', 'session_open_et': (now - timedelta(minutes=30)).isoformat(), 'session_close_et': (now + timedelta(hours=3)).isoformat(), 'regular_session_open': True}\n"
        "render_paper_buy_add_arm_controls(confirm_path=DEST, paper_mode=True, live_blocked=True, allow_path_override=True, session_state_fn=_open)\n",
        encoding="utf-8",
    )
    from streamlit.testing.v1 import AppTest
    from services.paper_buy_add_arm_console import ARM_PHRASE

    at = AppTest.from_file(str(app), default_timeout=30)
    at.run()
    if at.exception:
        raise at.exception
    at.text_input[0].input(ARM_PHRASE).run()
    armed = [b for b in at.button if str(b.label) == "Arm Paper BUY/ADD"]
    assert armed
    armed[0].click().run()
    if at.exception:
        raise at.exception
    assert dest.is_file()
    first = dest.read_bytes()
    import time

    time.sleep(1.1)
    at.text_input[0].input(ARM_PHRASE).run()
    renew = [b for b in at.button if str(b.label) == "Renew Paper BUY/ADD Arm"]
    assert renew
    renew[0].click().run()
    if at.exception:
        raise at.exception
    second = dest.read_bytes()
    assert second != first
    at.run()
    assert dest.read_bytes() == second
    assert not REPO_SELL_DB.is_file()


def test_ops_page_wires_buy_add_control():
    ops = (ROOT / "view_results.py").read_text(encoding="utf-8")
    assert "render_paper_buy_add_arm_controls" in ops
    assert "render_paper_sell_arm_controls" in ops
    dash = (ROOT / "dashboard" / "paper_buy_add_arm_controls.py").read_text(encoding="utf-8")
    assert ARM_PHRASE in dash
    assert "Arm Paper BUY/ADD" in dash
    assert "Renew Paper BUY/ADD Arm" in dash
    assert "Revoke Paper BUY/ADD Arm" in dash
    assert "The PowerShell command remains an operator fallback." in dash
    console = (ROOT / "services" / "paper_buy_add_arm_console.py").read_text(encoding="utf-8")
    assert "Arming does not start TRITON, start the continuous loop, or submit orders." in console


def test_write_confirm_still_the_cli_schema(tmp_path):
    dest = _arm_path(tmp_path)
    write_confirm(30, allow_execute=True, confirm_path=dest, now=NOW_OPEN)
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert list(doc.keys()) == ["timestamp", "mode", "expires_at", "allow_execute"]
