"""Watchdog liveness: a finished one-shot child is not a dead parent loop."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from services.governance_execution_readiness import _check_watchdog_health
from services.risk_watchdog import (
    HANG_BUDGET_CEILING_MINUTES,
    WatchdogAlertEngine,
    bounded_hang_budget_minutes,
    classify_watchdog_liveness,
    watchdog_liveness_blocks_entries,
)

NOW = datetime(2026, 10, 22, 21, 0, tzinfo=timezone.utc)
RESIDUAL_ALERT = {
    "alert_id": "STALE_HEARTBEAT:global",
    "alert_type": "STALE_HEARTBEAT",
    "status": "ACTIVE",
    "severity": "HIGH",
    "first_seen": "2026-10-22T14:28:43Z",
    "details": {"liveness": "DEAD"},
}


def _iso(when: datetime) -> str:
    return when.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _classify(**overrides: object) -> str:
    started = NOW - timedelta(minutes=10)
    completed = NOW - timedelta(minutes=1)
    payload = {
        "cycle_started_at": _iso(started),
        "last_completed_at": _iso(completed),
        "owner_pid_alive": False,
        "lock_present": True,
        "lock_owner_alive": True,
        "now": NOW,
        "hang_budget_minutes": bounded_hang_budget_minutes(),
    }
    payload.update(overrides)
    return classify_watchdog_liveness(**payload)


def test_hang_budget_exceeds_measured_pass_and_has_a_ceiling():
    measured = 1.0
    budget = bounded_hang_budget_minutes(measured)
    assert budget > measured
    assert budget <= HANG_BUDGET_CEILING_MINUTES
    assert bounded_hang_budget_minutes(100.0) == HANG_BUDGET_CEILING_MINUTES


def test_completed_oneshot_child_with_live_parent_is_ok():
    classification = _classify()
    assert classification == "OK"
    assert watchdog_liveness_blocks_entries(classification) is False


def test_in_progress_child_below_hang_budget_is_not_dead():
    classification = _classify(
        cycle_started_at=_iso(NOW - timedelta(minutes=1)),
        last_completed_at=_iso(NOW - timedelta(minutes=30)),
        owner_pid_alive=True,
    )
    assert classification == "IN_PROGRESS"
    assert classification != "DEAD"
    assert watchdog_liveness_blocks_entries(classification) is False


def test_hung_child_beyond_budget_is_blocking_and_raises_stale_alert(tmp_path, monkeypatch):
    classification = _classify(
        cycle_started_at=_iso(NOW - timedelta(minutes=30)),
        last_completed_at=_iso(NOW - timedelta(minutes=40)),
        owner_pid_alive=True,
    )
    assert classification == "HUNG"
    assert watchdog_liveness_blocks_entries(classification) is True

    import services.risk_watchdog as watchdog

    live = tmp_path / "live"
    results = tmp_path / "results"
    live.mkdir()
    results.mkdir()
    state = live / "watchdog_alert_state.json"
    alerts = results / "watchdog_alerts.json"
    state.write_text("{}", encoding="utf-8")
    alerts.write_text(
        json.dumps({"active_alerts": [], "resolved_alerts": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(watchdog, "STATE_PATH", state)
    monkeypatch.setattr(watchdog, "ALERTS_PATH", alerts)
    engine = WatchdogAlertEngine(
        drawdown_pct=-10.0,
        concentration_pct=25.0,
        order_age_minutes=60.0,
        stale_heartbeat_minutes=2.0,
    )
    engine.evaluate_stale_heartbeat(None, liveness=classification)
    alert = engine.active["STALE_HEARTBEAT:global"]
    assert alert["details"]["liveness"] == "HUNG"
    assert not alerts.read_text(encoding="utf-8").count("STALE_HEARTBEAT:global")

    gone = _classify(
        cycle_started_at=_iso(NOW - timedelta(minutes=30)),
        last_completed_at=_iso(NOW - timedelta(minutes=40)),
        owner_pid_alive=False,
    )
    assert gone == "DEAD"


def test_dead_parent_owner_is_dead():
    classification = _classify(lock_owner_alive=False, owner_pid_alive=True)
    assert classification == "DEAD"
    assert watchdog_liveness_blocks_entries(classification) is True


def test_stale_lock_without_a_live_process_is_dead():
    classification = _classify(
        owner_pid_alive=False,
        lock_present=True,
        lock_owner_alive=False,
    )
    assert classification == "DEAD"


def test_long_healthy_pass_does_not_create_a_readiness_block():
    classification = _classify(
        cycle_started_at=_iso(NOW - timedelta(minutes=12)),
        last_completed_at=_iso(NOW - timedelta(seconds=20)),
        owner_pid_alive=False,
        lock_owner_alive=True,
    )
    assert classification == "OK"
    ok, reason = _check_watchdog_health("OK", [RESIDUAL_ALERT], liveness=classification)
    assert ok is True
    assert reason == "Watchdog monitoring healthy"
    blocked, blocked_reason = _check_watchdog_health("OK", [], liveness="DEAD")
    assert blocked is False
    assert blocked_reason == "Watchdog liveness DEAD"


@pytest.mark.parametrize("liveness", ["HUNG", "STALE_LOCK"])
def test_blocking_liveness_still_fails_closed(liveness: str):
    ok, reason = _check_watchdog_health("OK", [], liveness=liveness)
    assert ok is False
    assert reason == f"Watchdog liveness {liveness}"
