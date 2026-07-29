"""
DO-004 focused loop regression — continuous/watchdog PS1 contracts.

Uses source inspection + isolated Python helpers only.
Does not start/stop real loops or touch real data/live runtime artifacts.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from services.operator_cooperative_shutdown import (
    ACTION,
    CONTINUOUS_WAIT_CHECK_SEC,
    WATCHDOG_WAIT_CHECK_SEC,
    atomic_write_json,
    interruptible_sleep_seconds,
    poll_shutdown_for_loop,
)

ROOT = Path(__file__).resolve().parents[1]
CONT = ROOT / "scripts" / "run_continuous_loop.ps1"
WD = ROOT / "scripts" / "run_watchdog_loop.ps1"


def _param_default(src: str, name: str) -> int:
    m = re.search(rf"\[int\]\${name}\s*=\s*(\d+)", src)
    assert m, f"missing default for {name}"
    return int(m.group(1))


def test_continuous_default_cadence_is_45_minutes():
    src = CONT.read_text(encoding="utf-8")
    assert _param_default(src, "IntervalMinutes") == 45
    assert "($IntervalMinutes * 60)" in src
    assert "Invoke-InterruptibleSleep -TotalSeconds ($IntervalMinutes * 60)" in src


def test_watchdog_default_cadence_is_one_minute():
    src = WD.read_text(encoding="utf-8")
    assert _param_default(src, "IntervalMinutes") == 1
    assert "Invoke-InterruptibleSleep -TotalSeconds ($IntervalMinutes * 60)" in src


def test_continuous_no_request_behavior_unchanged_structure():
    src = CONT.read_text(encoding="utf-8")
    # Core cycle work still present
    assert "run_market_open.ps1" in src
    assert "paper_entry_loop_stage" in src
    assert "Update-LoopLockMetadata" in src
    assert "Clear-OrphanLoopLock" in src
    # Sleep still driven by IntervalMinutes (not shortened cadence)
    assert "IntervalMinutes" in src


def test_watchdog_no_request_behavior_unchanged_structure():
    src = WD.read_text(encoding="utf-8")
    assert "services.risk_watchdog" in src
    assert "Update-LoopLockMetadata" in src
    assert "Clear-OrphanLoopLock" in src
    assert '$lockFile = "data\\live\\watchdog_loop.lock"' in src or "watchdog_loop.lock" in src


def test_cooperative_wait_checks_do_not_create_additional_work_cycles():
    for path in (CONT, WD):
        src = path.read_text(encoding="utf-8")
        # Work cycle markers only in the main while body once per iteration
        assert src.count("[CYCLE_START]") == 1
        # Interruptible sleep invokes Python helper — must not call market_open / risk_watchdog
        sleep_fn = src[
            src.index("function Invoke-InterruptibleSleep") : src.index(
                "function Remove-OwnLoopLock"
            )
        ]
        assert "run_market_open" not in sleep_fn
        assert "risk_watchdog" not in sleep_fn
        assert "paper_entry_loop_stage" not in sleep_fn
        assert "operator_cooperative_shutdown" in sleep_fn


def test_continuous_releases_only_own_lock_contract():
    src = CONT.read_text(encoding="utf-8")
    assert "function Remove-OwnLoopLock" in src
    assert "continuous_loop.lock" in src
    assert "watchdog_loop.lock" not in src
    assert "Matches[1]" in src
    assert "finally" in src
    assert "Remove-OwnLoopLock" in src.split("finally")[1]


def test_watchdog_releases_only_own_lock_contract():
    src = WD.read_text(encoding="utf-8")
    assert "function Remove-OwnLoopLock" in src
    assert "watchdog_loop.lock" in src
    assert "continuous_loop.lock" not in src
    assert "Matches[1]" in src
    assert "Remove-OwnLoopLock" in src.split("finally")[1]


def test_malformed_expired_mismatched_requests_do_not_stop_loop(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    # Malformed
    (live / "operator_shutdown_request.json").write_text("{bad", encoding="utf-8")
    assert (
        poll_shutdown_for_loop(
            component="continuous",
            actual_pid=1,
            script_name="run_continuous_loop.ps1",
            root=tmp_path,
        )["action"]
        == "CONTINUE_REFUSED"
    )
    # Expired
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "exp",
            "action": ACTION,
            "expires_at": past,
            "target_components": ["watchdog"],
            "expected_instances": {"watchdog": {"expected_pid": 2}},
        },
    )
    assert (
        poll_shutdown_for_loop(
            component="watchdog",
            actual_pid=2,
            script_name="run_watchdog_loop.ps1",
            root=tmp_path,
        )["action"]
        == "CONTINUE_REFUSED"
    )
    # Mismatched PID
    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "mm",
            "action": ACTION,
            "expires_at": future,
            "target_components": ["continuous"],
            "expected_instances": {"continuous": {"expected_pid": 10}},
        },
    )
    assert (
        poll_shutdown_for_loop(
            component="continuous",
            actual_pid=99,
            script_name="run_continuous_loop.ps1",
            root=tmp_path,
        )["action"]
        == "CONTINUE_REFUSED"
    )


def test_no_forced_termination_in_loop_scripts():
    for path in (CONT, WD):
        text = path.read_text(encoding="utf-8")
        assert "taskkill" not in text.lower()
        assert "Stop-Process" not in text
        assert "TerminateProcess" not in text
        assert "SIGKILL" not in text
        assert "SIGTERM" not in text
        assert "Kill(" not in text


def test_interruptible_check_intervals_match_targets():
    assert CONTINUOUS_WAIT_CHECK_SEC == 10
    assert WATCHDOG_WAIT_CHECK_SEC == 5
    cont = CONT.read_text(encoding="utf-8")
    wd = WD.read_text(encoding="utf-8")
    assert "check every ~10s" in cont
    assert "check every ~5s" in wd


def test_no_request_interruptible_sleep_preserves_total_wait(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    sleeps: list[int] = []

    def fake_sleep(s):
        sleeps.append(int(s))

    with patch("services.operator_cooperative_shutdown.time.sleep", side_effect=fake_sleep):
        result = interruptible_sleep_seconds(
            45 * 60,
            component="continuous",
            actual_pid=1,
            script_name="run_continuous_loop.ps1",
            check_interval_sec=CONTINUOUS_WAIT_CHECK_SEC,
            root=tmp_path,
        )
    assert result["shutdown"] is False
    assert sum(sleeps) == 45 * 60

    sleeps.clear()
    with patch("services.operator_cooperative_shutdown.time.sleep", side_effect=fake_sleep):
        result = interruptible_sleep_seconds(
            60,
            component="watchdog",
            actual_pid=2,
            script_name="run_watchdog_loop.ps1",
            check_interval_sec=WATCHDOG_WAIT_CHECK_SEC,
            root=tmp_path,
        )
    assert result["shutdown"] is False
    assert sum(sleeps) == 60


def test_tests_never_mutate_real_runtime_artifacts(tmp_path):
    """Fixtures use tmp_path only; this suite does not write under the real repo data/live."""
    real_live = (ROOT / "data" / "live").resolve()
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    target = live / "operator_shutdown_request.json"
    target.write_text("{}", encoding="utf-8")
    assert target.is_file()
    assert real_live not in target.resolve().parents
    assert not str(target.resolve()).startswith(str(real_live))
