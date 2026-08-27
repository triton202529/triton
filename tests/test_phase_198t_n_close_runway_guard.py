"""Phase 198T-O — 181-second regular-session close runway guard.

Isolated. No broker. Uses authoritative market calendar close, not a hardcoded 16:00.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from services.paper_runtime_session_gate import (
    ACTION_INVOKE_ENTRY,
    ACTION_SKIP_CLOSED,
    ACTION_SKIP_NEAR_CLOSE,
    ACTION_BLOCK_UNKNOWN,
    CYCLE_SKIPPED_NEAR_CLOSE,
    MINIMUM_RUNWAY_SECONDS,
    PIPELINE_HEALTHY,
    PIPELINE_INFORMATIONAL_CLOSED,
    decide_paper_entry_cycle_action,
    record_cycle_outcome,
)

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]


def _et(year, month, day, hour, minute, second=0, microsecond=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=ET)


def test_01_remaining_182_invokes():
    now = _et(2026, 8, 26, 15, 56, 58)
    d = decide_paper_entry_cycle_action(now=now)
    assert d["remaining_runway_seconds"] >= 182
    assert d["action"] == ACTION_INVOKE_ENTRY


def test_02_remaining_exactly_181_invokes():
    now = _et(2026, 8, 26, 15, 56, 59)
    d = decide_paper_entry_cycle_action(now=now)
    assert abs(float(d["remaining_runway_seconds"]) - 181.0) < 0.01
    assert d["action"] == ACTION_INVOKE_ENTRY
    assert d["required_runway_seconds"] == MINIMUM_RUNWAY_SECONDS == 181


def test_03_remaining_180_informational_skip():
    now = _et(2026, 8, 26, 15, 57, 0)
    d = decide_paper_entry_cycle_action(now=now)
    assert float(d["remaining_runway_seconds"]) < 181
    assert abs(float(d["remaining_runway_seconds"]) - 180.0) < 0.01
    assert d["action"] == ACTION_SKIP_NEAR_CLOSE
    assert d["cycle_event"] == CYCLE_SKIPPED_NEAR_CLOSE


def test_04_remaining_1_second_skip():
    now = _et(2026, 8, 26, 15, 59, 59)
    d = decide_paper_entry_cycle_action(now=now)
    assert d["action"] == ACTION_SKIP_NEAR_CLOSE
    assert d["broker_orders_submitted"] is False
    assert d["live_execution_permitted"] is False


def test_05_cycle8_replay_5_618_seconds():
    now = _et(2026, 8, 26, 15, 59, 54, 381529)
    d = decide_paper_entry_cycle_action(now=now)
    assert d["action"] == ACTION_SKIP_NEAR_CLOSE
    rem = float(d["remaining_runway_seconds"])
    assert 5.0 < rem < 6.0
    assert abs(rem - 5.618471) < 0.01
    assert d["session"]["session_close_et"]
    assert "16:00:00" in str(d["session"]["session_close_et"])
    assert (d.get("session_date") or d["session"].get("session_date")) == "2026-08-26"


def test_06_market_already_closed():
    now = _et(2026, 8, 26, 16, 0, 0)
    d = decide_paper_entry_cycle_action(now=now)
    assert d["action"] == ACTION_SKIP_CLOSED


def test_07_early_close_uses_calendar_close():
    invoke_ok = _et(2026, 11, 27, 12, 56, 59)
    near = _et(2026, 11, 27, 12, 59, 59)
    closed = _et(2026, 11, 27, 13, 0, 0)
    ok = decide_paper_entry_cycle_action(now=invoke_ok)
    assert ok["action"] == ACTION_INVOKE_ENTRY
    assert "13:00:00" in str(ok["session"].get("session_close_et"))
    skip = decide_paper_entry_cycle_action(now=near)
    assert skip["action"] == ACTION_SKIP_NEAR_CLOSE
    done = decide_paper_entry_cycle_action(now=closed)
    assert done["action"] == ACTION_SKIP_CLOSED
    assert done["session"]["reason"] == "early_close"


def test_08_holiday_existing_closed():
    now = _et(2026, 1, 1, 10, 0, 0)
    d = decide_paper_entry_cycle_action(now=now)
    assert d["action"] == ACTION_SKIP_CLOSED
    assert d["session"]["reason"] == "holiday"


def test_09_weekend_existing_closed():
    now = _et(2026, 8, 15, 10, 0, 0)
    d = decide_paper_entry_cycle_action(now=now)
    assert d["action"] == ACTION_SKIP_CLOSED
    assert d["session"]["reason"] == "weekend"


def test_10_calendar_lookup_failure_unknown(monkeypatch):
    import services.paper_runtime_session_gate as gate
    import services.market_calendar as cal

    def _boom(*, now=None):
        return {
            "session_state": cal.SESSION_UNKNOWN,
            "reason": "schedule_retrieval_failed",
            "session_close_et": None,
        }

    monkeypatch.setattr(cal, "regular_session_state", _boom)
    monkeypatch.setattr(
        "services.market_calendar.regular_session_state",
        _boom,
    )
    d = gate.decide_paper_entry_cycle_action(now=_et(2026, 8, 26, 12, 0, 0))
    assert d["action"] == ACTION_BLOCK_UNKNOWN


def test_11_timezone_aware_utc_et():
    utc_near = datetime(2026, 8, 26, 19, 59, 54, tzinfo=timezone.utc)
    d = decide_paper_entry_cycle_action(now=utc_near)
    assert d["action"] == ACTION_SKIP_NEAR_CLOSE
    et = utc_near.astimezone(ET)
    assert et.hour == 15
    assert et.tzname() == "EDT"


def test_12_to_16_no_invocation_signals_packet_authority_or_cycle_fail():
    src = (ROOT / "scripts" / "run_continuous_loop.ps1").read_text(encoding="utf-8")
    near = src.index("[PAPER_ENTRY_SKIPPED_NEAR_CLOSE]")
    invoke = src.index("run_market_open.ps1")
    assert near < invoke
    block = src[near:invoke]
    assert "generate_signals" not in block
    assert "defensive_exit_approval_readiness" not in block
    assert "CYCLE_FAIL" not in block
    assert "skipped_near_close" in src
    assert "[CYCLE_SKIPPED_NEAR_CLOSE]" in src


def test_17_watchdog_status_informational(tmp_path: Path):
    now = _et(2026, 8, 26, 15, 59, 54)
    rec = record_cycle_outcome(
        outcome="skipped_near_close",
        reason=CYCLE_SKIPPED_NEAR_CLOSE,
        session_open=True,
        root=tmp_path,
        now=now.astimezone(timezone.utc),
    )
    assert rec["last_outcome"] != "fail"
    assert rec["trading_pipeline_status"] in {PIPELINE_HEALTHY, PIPELINE_INFORMATIONAL_CLOSED}
    assert rec["broker_orders_submitted"] is False
    assert rec["live_execution_permitted"] is False
    assert rec["paper_entry_invoked"] is False


def test_18_live_execution_blocked_on_decision():
    d = decide_paper_entry_cycle_action(now=_et(2026, 8, 26, 15, 59, 54))
    assert d["live_execution_permitted"] is False
    assert d["broker_orders_submitted"] is False


def test_19_normal_intraday_unchanged():
    now = _et(2026, 8, 26, 11, 19, 24)
    d = decide_paper_entry_cycle_action(now=now)
    assert d["action"] == ACTION_INVOKE_ENTRY
    assert d["session_state"] == "OPEN"
    assert float(d["remaining_runway_seconds"]) > 181


def test_20_fixed_45_minute_cadence_unchanged():
    src = (ROOT / "scripts" / "run_continuous_loop.ps1").read_text(encoding="utf-8")
    assert "[int]$IntervalMinutes = 45" in src
    assert "Invoke-InterruptibleSleep -TotalSeconds ($IntervalMinutes * 60)" in src
    assert "generate_signals.py" not in src


def test_required_skip_fields_present():
    now = _et(2026, 8, 26, 15, 59, 54, 381529)
    d = decide_paper_entry_cycle_action(now=now)
    assert d.get("as_of")
    assert d["session"].get("session_date") == "2026-08-26"
    assert d["session"].get("session_close_et")
    assert d["remaining_runway_seconds"] is not None
    assert d["required_runway_seconds"] == 181
    assert d["reason"] == CYCLE_SKIPPED_NEAR_CLOSE
