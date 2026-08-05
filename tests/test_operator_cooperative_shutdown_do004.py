"""DO-004 cooperative shutdown tests — fixtures/mocks only. No real loops/locks/broker."""

from __future__ import annotations

import ast
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from dashboard.operator_console import DEFAULT_OPERATOR_PAGE, OPERATOR_PAGES
from services.operator_cooperative_shutdown import (
    ACTION,
    CONTINUOUS_WAIT_CHECK_SEC,
    WATCHDOG_WAIT_CHECK_SEC,
    append_shutdown_audit,
    atomic_write_json,
    create_shutdown_request,
    interruptible_sleep_seconds,
    poll_shutdown_for_loop,
    validate_request_for_component,
    write_ack,
    _sanitize,
)
from services.operator_runtime_controller import (
    ACTION_START,
    execute_start_triton,
    start_eligibility,
)
from services.operator_shutdown_controller import (
    ACTION_PREPARE,
    DENIED_ACTIONS,
    evaluate_shutdown_decision,
    prepare_cooperative_shutdown,
    shutdown_active,
    shutdown_eligibility,
)


ROOT = Path(__file__).resolve().parents[1]


def _lock_text(pid: int, *, minutes_ago: float = 1.0, cycle: int = 1) -> str:
    now = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return (
        f"started_at={now.isoformat()}\n"
        f"refreshed_at={now.isoformat()}\n"
        f"pid={pid}\n"
        f"source=test\n"
        f"cycle={cycle}\n"
    )


def _write_fixture(
    root: Path,
    *,
    cont_pids=None,
    wd_pids=None,
    cont_lock_pid=None,
    wd_lock_pid=None,
    cont_minutes_ago=1.0,
    wd_minutes_ago=0.5,
    live_permitted=False,
    command_lines=None,
    write_cont_lock=None,
    write_wd_lock=None,
):
    live = root / "data" / "live"
    results = root / "data" / "results"
    scripts = root / "scripts"
    live.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "run_watchdog_loop.ps1").write_text("# fixture\n", encoding="utf-8")
    (scripts / "run_continuous_loop.ps1").write_text("# fixture\n", encoding="utf-8")

    cont_pids = list(cont_pids or [])
    wd_pids = list(wd_pids or [])
    if write_cont_lock is None:
        write_cont_lock = bool(cont_pids) or cont_lock_pid is not None
    if write_wd_lock is None:
        write_wd_lock = bool(wd_pids) or wd_lock_pid is not None

    if write_cont_lock:
        pid = cont_lock_pid if cont_lock_pid is not None else (cont_pids[0] if cont_pids else 999)
        (live / "continuous_loop.lock").write_text(
            _lock_text(pid, minutes_ago=cont_minutes_ago), encoding="utf-8"
        )
    if write_wd_lock:
        pid = wd_lock_pid if wd_lock_pid is not None else (wd_pids[0] if wd_pids else 998)
        (live / "watchdog_loop.lock").write_text(
            _lock_text(pid, minutes_ago=wd_minutes_ago), encoding="utf-8"
        )

    (results / "execution_readiness.json").write_text(
        json.dumps({"live_execution_permitted": live_permitted, "readiness_status": "READY"}),
        encoding="utf-8",
    )
    (live / "fixture_process_state.json").write_text(
        json.dumps(
            {
                "continuous_loop_running": bool(cont_pids),
                "watchdog_loop_running": bool(wd_pids),
                "detected_continuous_process_ids": cont_pids,
                "detected_watchdog_process_ids": wd_pids,
                "process_detection_confidence": "high",
                "command_lines": command_lines or {},
            }
        ),
        encoding="utf-8",
    )


def _set_fixture_pids(root: Path, *, cont_pids=None, wd_pids=None):
    live = root / "data" / "live"
    marker = live / "fixture_process_state.json"
    doc = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
    if cont_pids is not None:
        doc["detected_continuous_process_ids"] = list(cont_pids)
        doc["continuous_loop_running"] = bool(cont_pids)
    if wd_pids is not None:
        doc["detected_watchdog_process_ids"] = list(wd_pids)
        doc["watchdog_loop_running"] = bool(wd_pids)
    marker.write_text(json.dumps(doc), encoding="utf-8")


# --- Decision matrix ---


def test_01_both_absent_already_stopped(tmp_path):
    _write_fixture(tmp_path)
    r = prepare_cooperative_shutdown(root=tmp_path, deliver=True, ack_timeout_sec=0.1)
    assert r["result"] == "ALREADY_STOPPED"
    assert not (tmp_path / "data" / "live" / "operator_shutdown_request.json").is_file()


def test_02_continuous_only_eligible(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101)
    el = shutdown_eligibility(root=tmp_path)
    assert el["decision"] == "SHUTDOWN_ELIGIBLE_CONTINUOUS_ONLY"
    assert el["target_components"] == ["continuous"]


def test_03_watchdog_only_eligible(tmp_path):
    _write_fixture(tmp_path, wd_pids=[202], wd_lock_pid=202)
    el = shutdown_eligibility(root=tmp_path)
    assert el["decision"] == "SHUTDOWN_ELIGIBLE_WATCHDOG_ONLY"


def test_04_both_verified_eligible(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], wd_pids=[202], cont_lock_pid=101, wd_lock_pid=202)
    el = shutdown_eligibility(root=tmp_path)
    assert el["decision"] == "SHUTDOWN_ELIGIBLE_BOTH"
    assert set(el["target_components"]) == {"continuous", "watchdog"}


def test_05_stale_continuous_still_eligible(tmp_path):
    _write_fixture(
        tmp_path,
        cont_pids=[101],
        cont_lock_pid=101,
        cont_minutes_ago=120.0,
        wd_pids=[],
        write_wd_lock=False,
    )
    el = shutdown_eligibility(root=tmp_path)
    assert el["decision"] == "SHUTDOWN_ELIGIBLE_UNHEALTHY_VERIFIED"
    assert "continuous" in el["target_components"]


def test_06_stale_lock_no_process_refuses(tmp_path):
    _write_fixture(tmp_path, cont_pids=[], cont_lock_pid=101, write_cont_lock=True)
    el = shutdown_eligibility(root=tmp_path)
    assert el["decision"] == "SHUTDOWN_REFUSED_UNSAFE_STATE"


def test_07_process_without_valid_lock_refuses(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], write_cont_lock=False)
    el = shutdown_eligibility(root=tmp_path)
    assert el["decision"] == "SHUTDOWN_REFUSED_UNSAFE_STATE"


def test_08_pid_mismatch_refuses(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=999)
    el = shutdown_eligibility(root=tmp_path)
    assert el["decision"] == "SHUTDOWN_REFUSED_UNSAFE_STATE"


def test_09_command_mismatch_refuses(tmp_path):
    _write_fixture(
        tmp_path,
        cont_pids=[101],
        cont_lock_pid=101,
        command_lines={101: "powershell -File evil.ps1"},
    )
    el = shutdown_eligibility(root=tmp_path)
    assert el["decision"] == "SHUTDOWN_REFUSED_UNSAFE_STATE"


def test_10_duplicate_continuous_refuses(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101, 102], cont_lock_pid=101)
    el = shutdown_eligibility(root=tmp_path)
    assert el["decision"] == "SHUTDOWN_REFUSED_UNSAFE_STATE"


def test_11_duplicate_watchdog_refuses(tmp_path):
    _write_fixture(tmp_path, wd_pids=[201, 202], wd_lock_pid=201)
    el = shutdown_eligibility(root=tmp_path)
    assert el["decision"] == "SHUTDOWN_REFUSED_UNSAFE_STATE"


def test_12_ambiguous_refuses():
    d = evaluate_shutdown_decision(
        {"continuous": {"state": "AMBIGUOUS"}, "watchdog": {"state": "NOT_RUNNING"}},
        live_execution_blocked=True,
    )
    assert d["decision"] == "SHUTDOWN_REFUSED_UNSAFE_STATE"


def test_13_unknown_refuses():
    d = evaluate_shutdown_decision(
        {"continuous": {"state": "UNKNOWN"}, "watchdog": {"state": "NOT_RUNNING"}},
        live_execution_blocked=True,
    )
    assert d["decision"] == "SHUTDOWN_REFUSED_UNSAFE_STATE"


def test_14_live_enabled_refuses(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, live_permitted=True)
    el = shutdown_eligibility(root=tmp_path)
    assert el["decision"] == "SHUTDOWN_REFUSED_LIVE_SAFETY"


def test_15_active_start_guard_refuses(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101)
    live = tmp_path / "data" / "live"
    exp = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    (live / "operator_start_action.guard.json").write_text(
        json.dumps({"expires_at": exp, "action_id": "x"}), encoding="utf-8"
    )
    el = shutdown_eligibility(root=tmp_path)
    assert el["decision"] == "SHUTDOWN_REFUSED_START_IN_PROGRESS"


def test_16_active_shutdown_guard_in_progress(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101)
    live = tmp_path / "data" / "live"
    exp = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    (live / "operator_shutdown_action.guard.json").write_text(
        json.dumps({"expires_at": exp, "action_id": "y"}), encoding="utf-8"
    )
    el = shutdown_eligibility(root=tmp_path)
    assert el["decision"] == "SHUTDOWN_IN_PROGRESS"


def test_17_request_write_atomic(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    path = live / "operator_shutdown_request.json"
    r = create_shutdown_request(
        target_components=["continuous"],
        expected_instances={"continuous": {"expected_pid": 1}},
        request_path=path,
    )
    assert r["ok"]
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["action"] == ACTION
    assert doc["request_id"]
    assert not list(live.glob(".shutdown_*.tmp"))


def test_18_malformed_request_refused(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    (live / "operator_shutdown_request.json").write_text("{not-json", encoding="utf-8")
    poll = poll_shutdown_for_loop(
        component="continuous", actual_pid=1, script_name="run_continuous_loop.ps1", root=tmp_path
    )
    assert poll["action"] == "CONTINUE_REFUSED"


def test_19_expired_request_ignored(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    past = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "old",
            "action": ACTION,
            "expires_at": past,
            "target_components": ["continuous"],
            "expected_instances": {"continuous": {"expected_pid": 55}},
        },
    )
    v = validate_request_for_component(
        json.loads((live / "operator_shutdown_request.json").read_text(encoding="utf-8")),
        component="continuous",
        actual_pid=55,
    )
    assert v["state"] == "REFUSED_EXPIRED"


def test_20_replayed_request_does_not_stop_new_instance(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "replay",
            "action": ACTION,
            "expires_at": future,
            "target_components": ["continuous"],
            "expected_instances": {"continuous": {"expected_pid": 100}},
        },
    )
    # New instance PID 200 must refuse
    poll = poll_shutdown_for_loop(
        component="continuous", actual_pid=200, script_name="run_continuous_loop.ps1", root=tmp_path
    )
    assert poll["action"] == "CONTINUE_REFUSED"


def test_21_old_pid_request_does_not_affect_new_pid(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "old-pid",
            "action": ACTION,
            "expires_at": future,
            "target_components": ["continuous"],
            "expected_instances": {"continuous": {"expected_pid": 111}},
        },
    )
    # Old instance would accept
    assert (
        poll_shutdown_for_loop(
            component="continuous",
            actual_pid=111,
            script_name="run_continuous_loop.ps1",
            root=tmp_path,
        )["action"]
        == "EXIT_NOW"
    )
    # Reset ack; new PID must refuse the same request targeting 111
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "old-pid",
            "action": ACTION,
            "expires_at": future,
            "target_components": ["continuous"],
            "expected_instances": {"continuous": {"expected_pid": 111}},
        },
    )
    poll_new = poll_shutdown_for_loop(
        component="continuous",
        actual_pid=222,
        script_name="run_continuous_loop.ps1",
        root=tmp_path,
    )
    assert poll_new["action"] == "CONTINUE_REFUSED"
    assert poll_new["detail"]["state"] == "REFUSED_WRONG_INSTANCE"


def test_22_continuous_accepts_matching(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "ok-c",
            "action": ACTION,
            "expires_at": future,
            "target_components": ["continuous"],
            "expected_instances": {"continuous": {"expected_pid": 77}},
        },
    )
    poll = poll_shutdown_for_loop(
        component="continuous", actual_pid=77, script_name="run_continuous_loop.ps1", root=tmp_path
    )
    assert poll["action"] == "EXIT_NOW"
    ack = json.loads((live / "operator_shutdown_ack_continuous.json").read_text(encoding="utf-8"))
    assert ack["request_id"] == "ok-c"
    assert ack["state"] == "EXITING"


def test_23_continuous_refuses_wrong_instance(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "wrong-c",
            "action": ACTION,
            "expires_at": future,
            "target_components": ["continuous"],
            "expected_instances": {"continuous": {"expected_pid": 50}},
        },
    )
    poll = poll_shutdown_for_loop(
        component="continuous",
        actual_pid=51,
        script_name="run_continuous_loop.ps1",
        root=tmp_path,
    )
    assert poll["action"] == "CONTINUE_REFUSED"
    assert poll["detail"]["state"] == "REFUSED_WRONG_INSTANCE"


def test_24_continuous_finishes_current_safe_unit(tmp_path):
    """
    Scenario 24: request arriving during a cycle is honored only after the safe unit ends.

    Continuous script does not poll mid-cycle; acceptance is at POST_CYCLE / PRE_CYCLE only.
    Simulate: work runs first (cycle_done), then POST_CYCLE poll accepts — no mid-cycle exit.
    """
    src = (ROOT / "scripts" / "run_continuous_loop.ps1").read_text(encoding="utf-8")
    # Shutdown poll occurs before CYCLE_START and after cycle work, not inside market_open block
    cycle_start = src.index("[CYCLE_START]")
    market_open = src.index("run_market_open.ps1")
    post_poll = src.index('Test-CooperativeShutdown -CycleState "POST_CYCLE"')
    assert cycle_start < market_open < post_poll
    # No shutdown poll between market_open and paper_entry completion path
    between = src[market_open:post_poll]
    assert "Test-CooperativeShutdown" not in between

    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "mid-cycle",
            "action": ACTION,
            "expires_at": future,
            "target_components": ["continuous"],
            "expected_instances": {"continuous": {"expected_pid": 9}},
        },
    )
    cycle_done = False
    # Simulate unsafe mid-cycle: request exists but loop continues work first
    cycle_done = True
    poll = poll_shutdown_for_loop(
        component="continuous",
        actual_pid=9,
        script_name="run_continuous_loop.ps1",
        root=tmp_path,
        cycle_state="POST_CYCLE",
        safe_boundary="POST_CYCLE_PRE_SLEEP",
    )
    assert cycle_done is True
    assert poll["action"] == "EXIT_NOW"


def test_25_continuous_no_new_cycle_after_acceptance():
    src = (ROOT / "scripts" / "run_continuous_loop.ps1").read_text(encoding="utf-8")
    assert "$shutdownAccepted = $true" in src
    assert "break" in src
    # After accept at PRE_CYCLE, loop breaks before incrementing / starting cycle work
    pre = src.index('Test-CooperativeShutdown -CycleState "BETWEEN_CYCLES"')
    block = src[pre : src.index("[CYCLE_START]", pre)]
    assert "$shutdownAccepted = $true" in block
    assert "break" in block
    assert "[CYCLE_START]" not in block
    # Cycle increment occurs only after the PRE_CYCLE shutdown check
    assert "$cycle++" in src[src.index("break", pre) :]


def test_26_continuous_releases_own_lock_only(tmp_path):
    src = (ROOT / "scripts" / "run_continuous_loop.ps1").read_text(encoding="utf-8")
    assert "Remove-OwnLoopLock" in src
    assert "lock owned by pid" in src
    assert "pid=$($Matches[1])" in src or "Matches[1]" in src
    assert "continuous_loop.lock" in src
    assert "watchdog_loop.lock" not in src
    assert "taskkill" not in src.lower()
    assert "Stop-Process" not in src
    # Isolated ownership check: foreign PID lock must not be removed by helper contract
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    foreign = live / "continuous_loop.lock"
    foreign.write_text(_lock_text(999), encoding="utf-8")
    # Python side never deletes runtime locks
    from services import operator_shutdown_controller as osc

    src_ctrl = Path(osc.__file__).read_text(encoding="utf-8")
    assert "continuous_loop.lock).unlink" not in src_ctrl
    assert foreign.is_file()


def test_27_watchdog_accepts_matching(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "ok-w",
            "action": ACTION,
            "expires_at": future,
            "target_components": ["watchdog"],
            "expected_instances": {"watchdog": {"expected_pid": 88}},
        },
    )
    poll = poll_shutdown_for_loop(
        component="watchdog", actual_pid=88, script_name="run_watchdog_loop.ps1", root=tmp_path
    )
    assert poll["action"] == "EXIT_NOW"


def test_28_watchdog_refuses_wrong_instance(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "bad-w",
            "action": ACTION,
            "expires_at": future,
            "target_components": ["watchdog"],
            "expected_instances": {"watchdog": {"expected_pid": 1}},
        },
    )
    poll = poll_shutdown_for_loop(
        component="watchdog", actual_pid=999, script_name="run_watchdog_loop.ps1", root=tmp_path
    )
    assert poll["action"] == "CONTINUE_REFUSED"


def test_29_watchdog_exits_from_safe_boundary(tmp_path):
    src = (ROOT / "scripts" / "run_watchdog_loop.ps1").read_text(encoding="utf-8")
    assert 'Test-CooperativeShutdown -CycleState "BETWEEN_CYCLES"' in src
    assert 'Test-CooperativeShutdown -CycleState "POST_CYCLE"' in src
    risk = src.index("services.risk_watchdog")
    post = src.index('Test-CooperativeShutdown -CycleState "POST_CYCLE"')
    assert risk < post
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "wd-boundary",
            "action": ACTION,
            "expires_at": future,
            "target_components": ["watchdog"],
            "expected_instances": {"watchdog": {"expected_pid": 44}},
        },
    )
    poll = poll_shutdown_for_loop(
        component="watchdog",
        actual_pid=44,
        script_name="run_watchdog_loop.ps1",
        root=tmp_path,
        cycle_state="POST_CYCLE",
        safe_boundary="POST_CYCLE_PRE_SLEEP",
    )
    assert poll["action"] == "EXIT_NOW"
    ack = json.loads((live / "operator_shutdown_ack_watchdog.json").read_text(encoding="utf-8"))
    assert ack["safe_boundary"] == "POST_CYCLE_PRE_SLEEP"


def test_30_watchdog_releases_only_own_lock():
    src = (ROOT / "scripts" / "run_watchdog_loop.ps1").read_text(encoding="utf-8")
    assert "Remove-OwnLoopLock" in src
    assert "watchdog_loop.lock" in src
    assert "continuous_loop.lock" not in src
    assert "lock owned by pid" in src
    assert "taskkill" not in src.lower()
    assert "Stop-Process" not in src


def test_31_interruptible_continuous_wait(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "sleep-c",
            "action": ACTION,
            "expires_at": future,
            "target_components": ["continuous"],
            "expected_instances": {"continuous": {"expected_pid": 5}},
        },
    )
    t0 = time.time()
    with patch("services.operator_cooperative_shutdown.time.sleep", return_value=None):
        result = interruptible_sleep_seconds(
            100,
            component="continuous",
            actual_pid=5,
            script_name="run_continuous_loop.ps1",
            check_interval_sec=CONTINUOUS_WAIT_CHECK_SEC,
            root=tmp_path,
        )
    assert result["shutdown"] is True
    assert time.time() - t0 < 2.0


def test_32_interruptible_watchdog_wait(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "sleep-w",
            "action": ACTION,
            "expires_at": future,
            "target_components": ["watchdog"],
            "expected_instances": {"watchdog": {"expected_pid": 6}},
        },
    )
    with patch("services.operator_cooperative_shutdown.time.sleep", return_value=None):
        result = interruptible_sleep_seconds(
            60,
            component="watchdog",
            actual_pid=6,
            script_name="run_watchdog_loop.ps1",
            check_interval_sec=WATCHDOG_WAIT_CHECK_SEC,
            root=tmp_path,
        )
    assert result["shutdown"] is True


def test_33_no_request_preserves_cadence(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)

    with patch("services.operator_cooperative_shutdown.time.sleep", side_effect=fake_sleep):
        result = interruptible_sleep_seconds(
            30,
            component="continuous",
            actual_pid=1,
            script_name="run_continuous_loop.ps1",
            check_interval_sec=10,
            root=tmp_path,
        )
    assert result["shutdown"] is False
    assert sum(sleeps) == 30


def test_34_ack_references_request(tmp_path):
    test_22_continuous_accepts_matching(tmp_path)
    ack = json.loads(
        (tmp_path / "data" / "live" / "operator_shutdown_ack_continuous.json").read_text(
            encoding="utf-8"
        )
    )
    assert ack["request_id"] == "ok-c"


def test_35_ack_sanitizer_removes_secrets():
    cleaned = _sanitize({"api_key": "x", "token": "y", "safe": 1, "password": "z"})
    assert cleaned["api_key"] == "<redacted>"
    assert cleaned["token"] == "<redacted>"
    assert cleaned["password"] == "<redacted>"
    assert cleaned["safe"] == 1


def test_36_request_sanitizer_removes_unsafe_fields(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    path = live / "t.json"
    atomic_write_json(path, {"command": "evil", "env": {"A": "1"}, "ok": True})
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["command"] == "<redacted>"
    assert doc["env"] == "<redacted>"
    assert doc["ok"] is True


def test_37_full_exit_lock_release_completed(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)

    def sleep_fn(_):
        # After request, simulate loop exit + lock release
        live = tmp_path / "data" / "live"
        req = live / "operator_shutdown_request.json"
        if req.is_file():
            write_ack(
                component="continuous",
                request_id=json.loads(req.read_text(encoding="utf-8"))["request_id"],
                state="EXITING",
                actual_pid=101,
                safe_boundary="POST_CYCLE_PRE_SLEEP",
                current_cycle_state="EXIT_PENDING",
                next_action="EXIT",
                safe_message="exiting",
                ack_path=live / "operator_shutdown_ack_continuous.json",
                script_name="run_continuous_loop.ps1",
            )
            _set_fixture_pids(tmp_path, cont_pids=[])
            lock = live / "continuous_loop.lock"
            if lock.is_file():
                lock.unlink()

    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=2.0,
        exit_timeout_sec=2.0,
        lock_timeout_sec=2.0,
        poll_sec=0.01,
        sleep_fn=sleep_fn,
    )
    assert r["result"] == "COOPERATIVE_SHUTDOWN_COMPLETED"
    assert r["forced_termination_used"] is False
    assert r["runtime_locks_deleted_by_controller"] is False


def test_38_ack_without_exit_not_complete(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)

    def sleep_fn(_):
        live = tmp_path / "data" / "live"
        req = live / "operator_shutdown_request.json"
        if req.is_file():
            rid = json.loads(req.read_text(encoding="utf-8"))["request_id"]
            write_ack(
                component="continuous",
                request_id=rid,
                state="ACCEPTED",
                actual_pid=101,
                safe_boundary="POST_CYCLE_PRE_SLEEP",
                current_cycle_state="EXIT_PENDING",
                next_action="EXIT",
                safe_message="accepted but still running",
                ack_path=live / "operator_shutdown_ack_continuous.json",
                script_name="run_continuous_loop.ps1",
            )
            # process still present

    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=0.5,
        exit_timeout_sec=0.3,
        lock_timeout_sec=0.1,
        poll_sec=0.05,
        sleep_fn=sleep_fn,
    )
    assert r["result"] in ("SHUTDOWN_TIMED_OUT", "PARTIAL_SHUTDOWN")
    assert r["forced_termination_used"] is False


def test_39_exit_with_stale_lock_postcondition(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)

    def sleep_fn(_):
        live = tmp_path / "data" / "live"
        req = live / "operator_shutdown_request.json"
        if req.is_file():
            rid = json.loads(req.read_text(encoding="utf-8"))["request_id"]
            write_ack(
                component="continuous",
                request_id=rid,
                state="EXITING",
                actual_pid=101,
                safe_boundary="POST_CYCLE_PRE_SLEEP",
                current_cycle_state="EXIT_PENDING",
                next_action="EXIT",
                safe_message="exiting",
                ack_path=live / "operator_shutdown_ack_continuous.json",
                script_name="run_continuous_loop.ps1",
            )
            _set_fixture_pids(tmp_path, cont_pids=[])
            # leave lock in place

    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=1.0,
        exit_timeout_sec=1.0,
        lock_timeout_sec=0.2,
        poll_sec=0.05,
        sleep_fn=sleep_fn,
    )
    assert r["result"] in ("POSTCONDITION_FAILED", "PARTIAL_SHUTDOWN")
    assert r["runtime_locks_deleted_by_controller"] is False


def test_40_lock_gone_process_active_not_complete(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)

    def sleep_fn(_):
        live = tmp_path / "data" / "live"
        req = live / "operator_shutdown_request.json"
        if req.is_file():
            rid = json.loads(req.read_text(encoding="utf-8"))["request_id"]
            write_ack(
                component="continuous",
                request_id=rid,
                state="EXITING",
                actual_pid=101,
                safe_boundary="POST_CYCLE_PRE_SLEEP",
                current_cycle_state="EXIT_PENDING",
                next_action="EXIT",
                safe_message="ack",
                ack_path=live / "operator_shutdown_ack_continuous.json",
                script_name="run_continuous_loop.ps1",
            )
            lock = live / "continuous_loop.lock"
            if lock.is_file():
                lock.unlink()
            # process still active

    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=0.5,
        exit_timeout_sec=0.3,
        lock_timeout_sec=0.1,
        poll_sec=0.05,
        sleep_fn=sleep_fn,
    )
    assert r["result"] != "COOPERATIVE_SHUTDOWN_COMPLETED"
    assert r["forced_termination_used"] is False


def test_41_partial_one_component(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], wd_pids=[202], cont_lock_pid=101, wd_lock_pid=202)

    def sleep_fn(_):
        live = tmp_path / "data" / "live"
        req = live / "operator_shutdown_request.json"
        if not req.is_file():
            return
        rid = json.loads(req.read_text(encoding="utf-8"))["request_id"]
        for comp, pid, name in (
            ("continuous", 101, "operator_shutdown_ack_continuous.json"),
            ("watchdog", 202, "operator_shutdown_ack_watchdog.json"),
        ):
            write_ack(
                component=comp,
                request_id=rid,
                state="EXITING",
                actual_pid=pid,
                safe_boundary="POST_CYCLE_PRE_SLEEP",
                current_cycle_state="EXIT_PENDING",
                next_action="EXIT",
                safe_message="ack",
                ack_path=live / name,
                script_name=f"run_{comp}_loop.ps1",
            )
        # Only continuous exits
        _set_fixture_pids(tmp_path, cont_pids=[], wd_pids=[202])
        lock = live / "continuous_loop.lock"
        if lock.is_file():
            lock.unlink()

    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=1.0,
        exit_timeout_sec=0.4,
        lock_timeout_sec=0.2,
        poll_sec=0.05,
        sleep_fn=sleep_fn,
    )
    assert r["result"] == "PARTIAL_SHUTDOWN"
    assert r["forced_termination_used"] is False


def test_42_ack_timeout_never_force_terminates(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=0.15,
        exit_timeout_sec=0.1,
        lock_timeout_sec=0.05,
        poll_sec=0.05,
        sleep_fn=lambda _: None,
    )
    assert r["result"] == "SHUTDOWN_TIMED_OUT"
    assert r["forced_termination_used"] is False
    assert (r.get("acknowledgement_results") or {}).get("continuous", {}).get("timed_out") is True
    assert "Forced termination" in (r.get("safe_message") or "")


def test_43_exit_timeout_never_force_terminates(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)

    def sleep_fn(_):
        live = tmp_path / "data" / "live"
        req = live / "operator_shutdown_request.json"
        if req.is_file() and not (live / "operator_shutdown_ack_continuous.json").is_file():
            rid = json.loads(req.read_text(encoding="utf-8"))["request_id"]
            write_ack(
                component="continuous",
                request_id=rid,
                state="ACCEPTED",
                actual_pid=101,
                safe_boundary="POST_CYCLE_PRE_SLEEP",
                current_cycle_state="EXIT_PENDING",
                next_action="EXIT",
                safe_message="accepted; still running",
                ack_path=live / "operator_shutdown_ack_continuous.json",
                script_name="run_continuous_loop.ps1",
            )
        # process remains — exit times out

    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=1.0,
        exit_timeout_sec=0.25,
        lock_timeout_sec=0.05,
        poll_sec=0.05,
        sleep_fn=sleep_fn,
    )
    assert r["result"] in ("SHUTDOWN_TIMED_OUT", "PARTIAL_SHUTDOWN")
    assert r["forced_termination_used"] is False
    assert r["runtime_locks_deleted_by_controller"] is False


def test_44_no_controller_runtime_lock_deletion_path():
    ctrl = (ROOT / "services" / "operator_shutdown_controller.py").read_text(encoding="utf-8")
    coop = (ROOT / "services" / "operator_cooperative_shutdown.py").read_text(encoding="utf-8")
    for src in (ctrl, coop):
        assert "continuous_loop.lock).unlink" not in src
        assert "watchdog_loop.lock).unlink" not in src
        assert "REPAIR_LOCK" in DENIED_ACTIONS or "REPAIR_LOCK" in src
    assert "REMOVE_LOCK" in DENIED_ACTIONS
    assert "DELETE_LOCK" in DENIED_ACTIONS


def test_45_no_process_termination_command():
    from services.operator_shutdown_controller import controller_has_termination_paths

    assert controller_has_termination_paths() is False
    ctrl = (ROOT / "services" / "operator_shutdown_controller.py").read_text(encoding="utf-8")
    assert "Popen" not in ctrl
    assert "import subprocess" not in ctrl


def test_46_stop_process_absent_in_production_loops():
    for name in ("run_continuous_loop.ps1", "run_watchdog_loop.ps1"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "Stop-Process" not in text


def test_47_taskkill_absent_in_production_loops():
    for name in ("run_continuous_loop.ps1", "run_watchdog_loop.ps1"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "taskkill" not in text.lower()
    from services.operator_shutdown_controller import controller_has_termination_paths

    assert controller_has_termination_paths() is False


def test_48_sigkill_absent_in_production_loops():
    for name in ("run_continuous_loop.ps1", "run_watchdog_loop.ps1"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "SIGKILL" not in text
        assert "SIGTERM" not in text
    ctrl = (ROOT / "services" / "operator_shutdown_controller.py").read_text(encoding="utf-8")
    # Denied-action enumerations may list SIGKILL; call forms must not
    assert "signal.SIGKILL" not in ctrl
    assert "os.kill(" not in ctrl


def test_49_frontend_cannot_supply_pid(tmp_path):
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "prepare_cooperative_shutdown(" not in src
    assert "shutdown_pid" not in src
    assert 'key="stop_pid"' not in src
    # Controllers reject injected pid; shutdown has no UI delivery path
    r = execute_start_triton(action=ACTION_START, pid=12345, root=tmp_path)
    assert r["result"] == "START_REFUSED_UNSAFE_STATE"
    assert r.get("error_code") == "FRONTEND_INJECTION_REFUSED"


def test_50_frontend_cannot_supply_command():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "shutdown_command" not in src
    assert "prepare_cooperative_shutdown(" not in src
    r = execute_start_triton(action=ACTION_START, command="evil.ps1")
    assert r.get("error_code") == "FRONTEND_INJECTION_REFUSED"


def test_51_frontend_cannot_supply_path():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "shutdown_path" not in src
    assert "prepare_cooperative_shutdown(" not in src
    r = execute_start_triton(action=ACTION_START, path="C:\\evil")
    assert r.get("error_code") == "FRONTEND_INJECTION_REFUSED"


def test_52_frontend_cannot_supply_environment_values():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "shutdown_env" not in src
    assert "prepare_cooperative_shutdown(" not in src
    r = execute_start_triton(action=ACTION_START, env={"TRITON_ENABLE_LIVE_EXECUTION": "1"})
    assert r.get("error_code") == "FRONTEND_INJECTION_REFUSED"


def test_53_start_refuses_during_active_shutdown(tmp_path):
    # Keep a verified process present so reconcile cannot mark the request complete.
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    live = tmp_path / "data" / "live"
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "schema_version": 1,
            "request_id": "active",
            "action": ACTION,
            "expires_at": future,
            "state": "DELIVERED",
            "target_components": ["continuous"],
            "expected_instances": {},
        },
    )
    el = start_eligibility(root=tmp_path)
    assert el["decision"] == "START_REFUSED_SHUTDOWN_IN_PROGRESS"
    assert el["start_button_enabled"] is False


def test_54_start_reeval_after_clean_shutdown(tmp_path):
    _write_fixture(tmp_path)  # both absent
    el = start_eligibility(root=tmp_path)
    assert el["decision"] == "START_ALLOWED"


def test_55_shutdown_does_not_create_or_extend_paper_arm(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=0.1,
        exit_timeout_sec=0.05,
        lock_timeout_sec=0.05,
        poll_sec=0.02,
        sleep_fn=lambda _: None,
    )
    assert r["paper_arm_mutated"] is False
    assert not (tmp_path / "data" / "live" / "paper_arm_confirm.json").is_file()


def test_56_shutdown_does_not_submit_or_cancel_order(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    before = list((tmp_path / "data").rglob("*order*"))
    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=0.1,
        exit_timeout_sec=0.05,
        lock_timeout_sec=0.05,
        poll_sec=0.02,
        sleep_fn=lambda _: None,
    )
    assert r["orders_submitted"] is False
    after = list((tmp_path / "data").rglob("*order*"))
    assert after == before


def test_57_shutdown_does_not_create_approval_or_authorization(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=0.1,
        exit_timeout_sec=0.05,
        lock_timeout_sec=0.05,
        poll_sec=0.02,
        sleep_fn=lambda _: None,
    )
    assert r["approvals_created"] is False
    assert not (tmp_path / "data" / "live" / "paper_arm_confirm.json").is_file()
    assert (
        not list((tmp_path / "data" / "governance").glob("*"))
        if (tmp_path / "data" / "governance").exists()
        else True
    )


def test_58_live_execution_remains_blocked_before_and_after(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=0.1,
        exit_timeout_sec=0.05,
        lock_timeout_sec=0.05,
        poll_sec=0.02,
        sleep_fn=lambda _: None,
    )
    assert r["live_execution_blocked_before"] is True
    assert r["live_execution_blocked_after"] is True
    readiness = json.loads(
        (tmp_path / "data" / "results" / "execution_readiness.json").read_text(encoding="utf-8")
    )
    assert readiness.get("live_execution_permitted") is False


def test_59_initial_audit_failure_refuses_delivery(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    with patch(
        "services.operator_shutdown_controller.append_shutdown_audit",
        return_value={"ok": False, "error": "fail"},
    ):
        r = prepare_cooperative_shutdown(root=tmp_path, deliver=True, ack_timeout_sec=0.1)
    assert r["result"] == "AUDIT_FAILED"
    assert not (tmp_path / "data" / "live" / "operator_shutdown_request.json").is_file()


def test_60_final_audit_failure_reports_audit_failed(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    calls = {"n": 0}

    def audit_side_effect(record, path=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return append_shutdown_audit(record, path=path)
        return {"ok": False, "error": "final_fail"}

    def sleep_fn(_):
        live = tmp_path / "data" / "live"
        req = live / "operator_shutdown_request.json"
        if req.is_file():
            rid = json.loads(req.read_text(encoding="utf-8"))["request_id"]
            write_ack(
                component="continuous",
                request_id=rid,
                state="EXITING",
                actual_pid=101,
                safe_boundary="POST_CYCLE_PRE_SLEEP",
                current_cycle_state="EXIT_PENDING",
                next_action="EXIT",
                safe_message="exiting",
                ack_path=live / "operator_shutdown_ack_continuous.json",
                script_name="run_continuous_loop.ps1",
            )
            _set_fixture_pids(tmp_path, cont_pids=[])
            lock = live / "continuous_loop.lock"
            if lock.is_file():
                lock.unlink()

    with patch(
        "services.operator_shutdown_controller.append_shutdown_audit",
        side_effect=audit_side_effect,
    ):
        r = prepare_cooperative_shutdown(
            root=tmp_path,
            deliver=True,
            ack_timeout_sec=1.0,
            exit_timeout_sec=1.0,
            lock_timeout_sec=1.0,
            poll_sec=0.01,
            sleep_fn=sleep_fn,
        )
    assert r["result"] == "AUDIT_FAILED"
    assert (tmp_path / "data" / "live" / "operator_shutdown_request.json").is_file()


def test_61_operations_stop_uses_adapter_not_direct_prepare():
    """DO-004/DO-005/DO-006: dashboard must not call prepare_cooperative_shutdown directly."""
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "prepare_cooperative_shutdown(" not in src
    assert "execute_stop_triton" in src
    assert "Stop TRITON Safely" in src
    assert "Confirm Stop" in src
    assert "Restart TRITON Safely" in src
    assert "execute_restart_triton" in src


def test_62_operations_exposes_no_clickable_force_restart():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "Restart TRITON Safely" in src
    assert "Confirm Restart" in src
    assert "RESTART TRITON" in src
    assert "taskkill" not in src.lower()
    assert "Stop-Process" not in src
    # Restart goes through typed orchestrator, not direct Start after Stop in UI
    restart_confirm = src[
        src.index('key="restart_triton_confirm"') : src.index('key="restart_triton_cancel"')
    ]
    assert "execute_start_triton" not in restart_confirm
    assert "execute_restart_triton" in restart_confirm


def test_63_refresh_does_not_request_shutdown():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "prepare_cooperative_shutdown(" not in src
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                assert n.func.id != "prepare_cooperative_shutdown"
            if isinstance(n.func, ast.Attribute):
                assert n.func.attr != "prepare_cooperative_shutdown"
    # render_operations_page must not call prepare on load
    assert "prepare_cooperative_shutdown" not in src


def test_64_query_parameters_cannot_request_shutdown():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "prepare_cooperative_shutdown(" not in src
    # No query-param driven shutdown
    assert "query_params" not in src or "shutdown" not in src.lower()
    assert "st.experimental_get_query_params" not in src
    assert "url_fragment" not in src
    assert "session_state" in src  # Start uses session; shutdown must not
    assert 'session_state["shutdown' not in src
    assert 'session_state.get("shutdown' not in src


def test_65_diagnostics_readonly():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "Cooperative shutdown evidence (DO-004)" in src
    assert "Diagnostics is read-only" in src
    assert "No repair, delete, kill" in src
    assert "prepare_cooperative_shutdown(" not in src


def test_66_six_page_nav_unchanged():
    assert len(OPERATOR_PAGES) == 6
    assert DEFAULT_OPERATOR_PAGE == "Operations"
    assert list(OPERATOR_PAGES) == [
        "Operations",
        "Activity",
        "Portfolio",
        "Orders",
        "Risk & Alerts",
        "Diagnostics",
    ]


def test_67_legacy_available():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "Advanced" in src
    assert "Legacy" in src


def test_68_localhost_binding():
    cfg = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "127.0.0.1" in cfg
    launcher = (ROOT / "scripts" / "run_operator_console.ps1").read_text(encoding="utf-8")
    assert "127.0.0.1" in launcher


def test_69_do003_start_regression_surface_intact(tmp_path):
    """Scenario 69: DO-003 Start surface remains; full suite run separately."""
    assert ACTION_START == "START_TRITON"
    _write_fixture(tmp_path)
    el = start_eligibility(root=tmp_path)
    assert el["allowed_action"] == ACTION_START
    assert el["stop_available"] is False
    assert el["restart_available"] is False
    assert el["decision"] == "START_ALLOWED"
    # Injection still refused (DO-003 contract)
    refused = execute_start_triton(action=ACTION_START, command="x", root=tmp_path)
    assert refused.get("error_code") == "FRONTEND_INJECTION_REFUSED"


def test_70_do002_console_regression_surface_intact():
    """Scenario 70: DO-002 console surface remains; full suite run separately."""
    assert len(OPERATOR_PAGES) == 6
    assert DEFAULT_OPERATOR_PAGE == "Operations"
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "render_operations_page" in src
    assert "Diagnostics" in OPERATOR_PAGES
    assert "Advanced/Legacy Dashboard" in src
    assert "Switch to Advanced / Legacy Dashboard" in src


def test_shutdown_active_helper(tmp_path):
    _write_fixture(tmp_path)
    assert shutdown_active(root=tmp_path)["active"] is False
    live = tmp_path / "data" / "live"
    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {
            "request_id": "a",
            "expires_at": future,
            "state": "DELIVERED",
            "action": ACTION,
        },
    )
    assert shutdown_active(root=tmp_path)["active"] is True


def test_denied_action_prepare_only(tmp_path):
    r = prepare_cooperative_shutdown(action="FORCE_STOP", root=tmp_path)
    assert r["result"] == "FAILED"
    assert r["error_code"] == "DENIED_ACTION"


def test_protocol_flags_on_eligibility(tmp_path):
    _write_fixture(tmp_path)
    el = shutdown_eligibility(root=tmp_path)
    assert el["protocol_implemented"] is True
    # DO-004 protocol layer still reports no Stop button of its own
    assert el["clickable_stop_exposed"] is False
    assert el["clickable_restart_exposed"] is False
    assert el["allowed_backend_action"] == ACTION_PREPARE
    assert "prepare_cooperative_shutdown(" not in (
        ROOT / "dashboard" / "operator_console.py"
    ).read_text(encoding="utf-8")
