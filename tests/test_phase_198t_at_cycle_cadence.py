"""Phase 198T-AT — start-to-start cadence contract (TEMP isolated tree).

PAPER ONLY. Fake clocks. No production writes, no broker, no network.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from services.command_center_status_coherence import cycle_deadline_utc
from services.continuous_loop_cadence import (
    DEFAULT_INTERVAL_SECONDS,
    PHASE_ACTIVE_CYCLE,
    PHASE_INTERRUPTIBLE_WAIT,
    authoritative_next_due_utc,
    build_cadence_record,
    catch_up_cycles_permitted,
    compute_next_due,
    ensure_utc,
    iso_et,
    iso_utc,
    overdue_missed_slots,
    remaining_wait_seconds,
    write_cadence_artifact,
)
from services.operator_cooperative_shutdown import (
    ACTION,
    CONTINUOUS_WAIT_CHECK_SEC,
    atomic_write_json,
    interruptible_sleep_seconds,
)
from services.paper_runtime_session_gate import (
    ACTION_INVOKE_ENTRY,
    ACTION_SKIP_CLOSED,
    ACTION_SKIP_NEAR_CLOSE,
    MINIMUM_RUNWAY_SECONDS,
    decide_paper_entry_cycle_action,
)
from services.paper_sell_arm_authority import (
    AUTHORITY_TTL_MINUTES,
    READINESS_REFERENCE_TTL_MINUTES,
)

ROOT = Path(__file__).resolve().parents[1]
LOOP_SCRIPT = ROOT / "scripts" / "run_continuous_loop.ps1"
ET = ZoneInfo("America/New_York")
INTERVAL = DEFAULT_INTERVAL_SECONDS
START = datetime(2026, 9, 1, 13, 38, 13, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _safety(monkeypatch):
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    for key in list(os.environ):
        if "ALPACA" in key.upper() or key.startswith("APCA_"):
            monkeypatch.delenv(key, raising=False)


def _et(year, month, day, hour, minute, second=0):
    return datetime(year, month, day, hour, minute, second, tzinfo=ET)


def test_a_start_to_start_wait_table():
    cases = (
        (0, 2700),
        (182, 2518),
        (600, 2100),
        (2699, 1),
        (2700, 0),
        (2701, 0),
        (4000, 0),
    )
    for duration, expected in cases:
        now = START + timedelta(seconds=duration)
        wait = remaining_wait_seconds(START, now=now, interval_seconds=INTERVAL)
        assert wait == expected, (duration, wait, expected)
        due = compute_next_due(START, interval_seconds=INTERVAL)
        assert due == START + timedelta(seconds=2700)
        assert catch_up_cycles_permitted() == 1


def test_a_overdue_one_next_cycle_only():
    now = START + timedelta(seconds=4000)
    assert remaining_wait_seconds(START, now=now) == 0
    assert overdue_missed_slots(START, now=now) >= 1
    assert catch_up_cycles_permitted() == 1


def test_b_timer_delay_107s_one_cycle_new_deadline():
    complete = START + timedelta(seconds=163)
    wake = complete + timedelta(seconds=2700 + 107)
    # Residual from the original start is already zero; the late wake is one cycle.
    assert remaining_wait_seconds(START, now=wake) == 0
    assert catch_up_cycles_permitted() == 1
    actual_start = wake
    rec = build_cadence_record(
        cycle_started_at=actual_start,
        interval_seconds=INTERVAL,
        now=actual_start,
        phase=PHASE_ACTIVE_CYCLE,
    )
    assert rec["next_due_at_utc"] == iso_utc(actual_start + timedelta(seconds=2700))
    assert rec["catch_up_cycles_permitted"] == 1


def test_c_host_suspend_resume_rechecks_session(tmp_path):
    resume = START + timedelta(seconds=INTERVAL * 3 + 100)
    assert remaining_wait_seconds(START, now=resume) == 0
    assert overdue_missed_slots(START, now=resume) >= 3
    assert catch_up_cycles_permitted() == 1
    closed = decide_paper_entry_cycle_action(now=_et(2026, 9, 1, 18, 0, 0))
    assert closed["action"] == ACTION_SKIP_CLOSED
    assert closed["broker_orders_submitted"] is False
    open_sess = decide_paper_entry_cycle_action(now=_et(2026, 9, 1, 11, 0, 0))
    assert open_sess["action"] == ACTION_INVOKE_ENTRY
    rec = build_cadence_record(cycle_started_at=resume, now=resume)
    write_cadence_artifact(rec, root=tmp_path)
    assert authoritative_next_due_utc(tmp_path) == resume + timedelta(seconds=2700)


def test_d_cooperative_stop_during_shortened_wait(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    wait = remaining_wait_seconds(START, now=START + timedelta(seconds=182))
    assert wait == 2518
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "at-cadence-stop",
            "action": ACTION,
            "expires_at": future,
            "target_components": ["continuous"],
            "expected_instances": {"continuous": {"expected_pid": 29364}},
        },
    )
    t0 = time.time()
    with patch("services.operator_cooperative_shutdown.time.sleep", return_value=None):
        result = interruptible_sleep_seconds(
            wait,
            component="continuous",
            actual_pid=29364,
            script_name="run_continuous_loop.ps1",
            check_interval_sec=CONTINUOUS_WAIT_CHECK_SEC,
            root=tmp_path,
        )
    assert result["shutdown"] is True
    assert time.time() - t0 < 2.0
    src = LOOP_SCRIPT.read_text(encoding="utf-8")
    assert "Remove-OwnLoopLock" in src
    assert "taskkill" not in src.lower()
    assert "Stop-Process" not in src
    assert "Invoke-InterruptibleSleep -TotalSeconds $waitSeconds" in src
    assert "[CADENCE_WAIT]" in src


def test_e_near_close_181_boundary_unchanged():
    assert MINIMUM_RUNWAY_SECONDS == 181
    d182 = decide_paper_entry_cycle_action(now=_et(2026, 8, 26, 15, 56, 58))
    assert d182["action"] == ACTION_INVOKE_ENTRY
    d181 = decide_paper_entry_cycle_action(now=_et(2026, 8, 26, 15, 56, 59))
    assert d181["action"] == ACTION_INVOKE_ENTRY
    d180 = decide_paper_entry_cycle_action(now=_et(2026, 8, 26, 15, 57, 0))
    assert d180["action"] == ACTION_SKIP_NEAR_CLOSE


def test_f_dashboard_deadline_matches_loop_artifact(tmp_path):
    complete = START + timedelta(seconds=163)
    rec = build_cadence_record(
        cycle_started_at=START,
        interval_seconds=INTERVAL,
        now=complete,
        phase=PHASE_INTERRUPTIBLE_WAIT,
    )
    write_cadence_artifact(rec, root=tmp_path)
    due = authoritative_next_due_utc(tmp_path)
    assert due == datetime(2026, 9, 1, 14, 23, 13, tzinfo=timezone.utc)
    assert cycle_deadline_utc(None, root=tmp_path) == due
    assert rec["timezone"] == "UTC"
    assert rec["next_due_at_utc"].endswith("Z")
    assert "T" in rec["next_due_at_et"]
    assert iso_utc(due) == rec["next_due_at_utc"]
    assert iso_et(due) == rec["next_due_at_et"]
    with pytest.raises(ValueError):
        ensure_utc(datetime(2026, 9, 1, 14, 23, 13))


def test_ttl_constants_unchanged():
    assert READINESS_REFERENCE_TTL_MINUTES == 50
    assert AUTHORITY_TTL_MINUTES == 60
    auth_src = (ROOT / "services" / "paper_sell_arm_authority.py").read_text(encoding="utf-8")
    assert "READINESS_REFERENCE_TTL_MINUTES = 50" in auth_src
    assert "AUTHORITY_TTL_MINUTES = 60" in auth_src


def test_loop_uses_shared_remaining_wait_not_full_post_completion_interval():
    src = LOOP_SCRIPT.read_text(encoding="utf-8")
    assert "services.continuous_loop_cadence" in src
    assert "--record-start" in src
    assert "--remaining-wait" in src
    assert "Invoke-InterruptibleSleep -TotalSeconds ($IntervalMinutes * 60)" not in src
    assert recency_wait_fallback_is_explicit(src)


def recency_wait_fallback_is_explicit(src: str) -> bool:
    return "CADENCE_WAIT_PARSE" in src and "$waitSeconds = $intervalSeconds" in src
