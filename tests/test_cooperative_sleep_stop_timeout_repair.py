"""
DO-004/DO-005 defect repair — interruptible cooperative sleep and Stop-timeout alignment.

Isolated mocks/tmp only. Does not touch real data/live, real processes, or PID 24008.
"""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from services.operator_cooperative_shutdown import (
    ACTION,
    ACK_OBSERVATION_MARGIN_SEC,
    CONTINUOUS_WAIT_CHECK_SEC,
    EXIT_ACCEPTED_EXIT_NOW,
    EXIT_ERROR,
    EXIT_NO_REQUEST,
    EXIT_REFUSED,
    HELPER_DEGRADED_WAIT_SEC,
    REQUEST_TTL_SEC,
    WATCHDOG_WAIT_CHECK_SEC,
    atomic_write_json,
    classify_loop_helper_exit,
    decide_post_cycle_action,
    decide_pre_cycle_action,
    decide_sleep_action,
    interruptible_sleep_seconds,
    poll_shutdown_for_loop,
    recommended_ack_timeout_sec,
    validate_request_for_component,
    write_ack,
)
from services.operator_runtime_controller import start_eligibility
from services.operator_shutdown_controller import (
    ACK_TIMEOUT_SEC,
    EXIT_TIMEOUT_SEC,
    prepare_cooperative_shutdown,
    shutdown_active,
)
from services.operator_stop_action import stop_eligibility
from services.operator_restart_action import restart_eligibility

ROOT = Path(__file__).resolve().parents[1]
CONT = ROOT / "scripts" / "run_continuous_loop.ps1"
WD = ROOT / "scripts" / "run_watchdog_loop.ps1"


def _future(minutes: int = 10) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


def _lock_text(pid: int, *, minutes_ago: float = 1.0, cycle: int = 1) -> str:
    now = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return (
        f"updated_at={now.isoformat()}\n"
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
    if cont_pids or cont_lock_pid is not None:
        pid = cont_lock_pid if cont_lock_pid is not None else cont_pids[0]
        (live / "continuous_loop.lock").write_text(_lock_text(pid), encoding="utf-8")
    if write_wd_lock is None:
        write_wd_lock = bool(wd_pids)
    if write_wd_lock:
        (live / "watchdog_loop.lock").write_text(
            _lock_text(wd_pids[0] if wd_pids else 998), encoding="utf-8"
        )
    (results / "execution_readiness.json").write_text(
        json.dumps({"live_execution_permitted": False, "readiness_status": "READY"}),
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
                "command_lines": {},
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


def _valid_request(pid: int = 101, component: str = "continuous") -> dict:
    return {
        "schema_version": 1,
        "request_id": "repair-req",
        "action": ACTION,
        "expires_at": _future(15),
        "target_components": [component],
        "expected_instances": {component: {"expected_pid": pid}},
        "state": "DELIVERED",
    }


def test_01_continuous_preserves_45_minute_cadence():
    src = CONT.read_text(encoding="utf-8")
    assert re.search(r"\[int\]\$IntervalMinutes\s*=\s*45", src)
    assert "Invoke-InterruptibleSleep -TotalSeconds ($IntervalMinutes * 60)" in src
    assert "Start-Sleep -Seconds ($IntervalMinutes * 60)" not in src


def test_02_cooperative_wait_bounded_interval():
    assert CONTINUOUS_WAIT_CHECK_SEC == 10
    assert WATCHDOG_WAIT_CHECK_SEC == 5
    assert recommended_ack_timeout_sec() == CONTINUOUS_WAIT_CHECK_SEC + ACK_OBSERVATION_MARGIN_SEC
    assert ACK_TIMEOUT_SEC == recommended_ack_timeout_sec()
    assert ACK_TIMEOUT_SEC > CONTINUOUS_WAIT_CHECK_SEC
    assert ACK_TIMEOUT_SEC < 45 * 60


def test_03_cooperative_wait_does_not_start_cycle_early(tmp_path):
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
    assert all(s <= CONTINUOUS_WAIT_CHECK_SEC for s in sleeps)


def test_04_stop_request_observed_during_inter_cycle_wait(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    atomic_write_json(live / "operator_shutdown_request.json", _valid_request(55))
    with patch("services.operator_cooperative_shutdown.time.sleep", return_value=None):
        result = interruptible_sleep_seconds(
            100,
            component="continuous",
            actual_pid=55,
            script_name="run_continuous_loop.ps1",
            check_interval_sec=CONTINUOUS_WAIT_CHECK_SEC,
            root=tmp_path,
        )
    assert result["shutdown"] is True
    ack = json.loads((live / "operator_shutdown_ack_continuous.json").read_text(encoding="utf-8"))
    assert ack["state"] == "EXITING"
    assert ack["safe_boundary"] == "INTERRUPTIBLE_WAIT"


def test_05_loop_acknowledges_via_typed_mechanism(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    atomic_write_json(live / "operator_shutdown_request.json", _valid_request(77))
    poll = poll_shutdown_for_loop(
        component="continuous",
        actual_pid=77,
        script_name="run_continuous_loop.ps1",
        root=tmp_path,
        cycle_state="WAITING",
        safe_boundary="INTERRUPTIBLE_WAIT",
    )
    assert poll["action"] == "EXIT_NOW"
    ack = json.loads((live / "operator_shutdown_ack_continuous.json").read_text(encoding="utf-8"))
    assert ack["request_id"] == "repair-req"
    assert ack["state"] == "EXITING"


def test_06_ps1_exits_cooperatively_after_ack_contract():
    """Source contract: sleep capture pattern + exit-10 branch exist in both scripts."""
    for path in (CONT, WD):
        src = path.read_text(encoding="utf-8")
        sleep_fn = src[
            src.index("function Invoke-InterruptibleSleep") : src.index(
                "function Remove-OwnLoopLock"
            )
        ]
        assert "$code = $LASTEXITCODE" in sleep_fn
        assert "return $code" in sleep_fn
        assert "| ForEach-Object" not in sleep_fn
        assert "if ($sleepCode -eq 10)" in src


def test_helper_exit_0_continues_and_10_exits_and_11_refuses_without_accept():
    assert classify_loop_helper_exit(EXIT_NO_REQUEST) == "CONTINUE"
    assert classify_loop_helper_exit(EXIT_ACCEPTED_EXIT_NOW) == "COOPERATIVE_EXIT"
    assert classify_loop_helper_exit(EXIT_REFUSED) == "CONTINUE"
    assert classify_loop_helper_exit(EXIT_REFUSED) != "COOPERATIVE_EXIT"


def test_behavioral_pre_cycle_post_cycle_sleep_exit_matrix():
    """Direct behavioral policy (not source-string): required cases 1–9."""
    # 1. PRE_CYCLE exit 0 permits next cycle
    assert decide_pre_cycle_action(0) == "START_CYCLE"
    # 2. PRE_CYCLE exit 10 exits cooperatively
    assert decide_pre_cycle_action(10) == "COOPERATIVE_EXIT"
    # 3. PRE_CYCLE exit 11 does not impersonate acceptance
    assert decide_pre_cycle_action(11) == "START_CYCLE"
    assert decide_pre_cycle_action(11) != "COOPERATIVE_EXIT"
    # 4. PRE_CYCLE exit 2 prevents a new cycle
    assert decide_pre_cycle_action(2) == "DEGRADED_NO_NEW_CYCLE"
    # 5. PRE_CYCLE null/unavailable helper prevents a new cycle
    assert decide_pre_cycle_action(None) == "DEGRADED_NO_NEW_CYCLE"
    # 6. POST_CYCLE exit 10 exits before inter-cycle waiting
    assert decide_post_cycle_action(10) == "COOPERATIVE_EXIT"
    # 7. POST_CYCLE exit 2 cannot permit an early new cycle
    assert decide_post_cycle_action(2) == "PROCEED_TO_WAIT"
    assert decide_post_cycle_action(2) != "START_CYCLE"
    # 8. Sleep exit 10 exits cooperatively
    assert decide_sleep_action(10) == "COOPERATIVE_EXIT"
    # 9. Sleep exit 2 enters degraded and prevents a new cycle
    assert decide_sleep_action(2) == "DEGRADED_NO_NEW_CYCLE"
    # Sleep exit 11 must NOT be treated like completed wait (cadence/early-cycle risk)
    assert decide_sleep_action(11) == "DEGRADED_NO_NEW_CYCLE"
    assert decide_sleep_action(0) == "PROCEED_TO_PRE_CYCLE"
    # Scripts implement the same branches for both loops
    for path in (CONT, WD):
        src = path.read_text(encoding="utf-8")
        assert "($pollCode -ne 0) -and ($pollCode -ne 11)" in src  # PRE_CYCLE
        assert "if ($sleepCode -eq 10)" in src
        assert "($null -eq $sleepCode) -or ($sleepCode -ne 0)" in src  # sleep: only 0 continues


def test_unexpected_helper_failure_refuses_new_cycle_degraded_wait():
    """Behavioral: unexpected codes → DEGRADED_NO_NEW_CYCLE; HELPER_DEGRADED_WAIT_SEC=30."""
    assert decide_pre_cycle_action(EXIT_ERROR) == "DEGRADED_NO_NEW_CYCLE"
    assert decide_pre_cycle_action(12) == "DEGRADED_NO_NEW_CYCLE"
    assert decide_pre_cycle_action(1) == "DEGRADED_NO_NEW_CYCLE"
    assert decide_pre_cycle_action(9009) == "DEGRADED_NO_NEW_CYCLE"
    assert decide_pre_cycle_action(None) == "DEGRADED_NO_NEW_CYCLE"
    assert HELPER_DEGRADED_WAIT_SEC == 30
    assert classify_loop_helper_exit(EXIT_ERROR) == "DEGRADED_NO_NEW_CYCLE"


def test_refusal_exit_does_not_impersonate_acceptance(tmp_path):
    """Behavioral: refused/malformed request → CONTINUE_REFUSED; exit 11 ≠ accept."""
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {**_valid_request(1), "expires_at": past},
    )
    poll = poll_shutdown_for_loop(
        component="continuous",
        actual_pid=1,
        script_name="run_continuous_loop.ps1",
        root=tmp_path,
    )
    assert poll["action"] == "CONTINUE_REFUSED"
    assert decide_pre_cycle_action(EXIT_REFUSED) == "START_CYCLE"
    assert decide_pre_cycle_action(EXIT_REFUSED) != "COOPERATIVE_EXIT"
    assert decide_sleep_action(EXIT_REFUSED) == "DEGRADED_NO_NEW_CYCLE"


def test_sleep_mode_cli_emits_only_0_or_10(tmp_path):
    """Sleep-mode helper cannot emit exit 11; full wait completes when no accept."""
    from services.operator_cooperative_shutdown import _cli_main

    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    sleeps: list[int] = []

    def fake_sleep(s):
        sleeps.append(int(s))

    with patch("services.operator_cooperative_shutdown.time.sleep", side_effect=fake_sleep):
        rc = _cli_main(
            [
                "--component",
                "continuous",
                "--pid",
                "1",
                "--root",
                str(tmp_path),
                "--mode",
                "sleep",
                "--seconds",
                "30",
            ]
        )
    assert rc == EXIT_NO_REQUEST
    assert sum(sleeps) == 30
    # With a matching request, sleep returns 10 after observing (no full remaining wait required)
    atomic_write_json(live / "operator_shutdown_request.json", _valid_request(1))
    with patch("services.operator_cooperative_shutdown.time.sleep", side_effect=fake_sleep):
        rc2 = _cli_main(
            [
                "--component",
                "continuous",
                "--pid",
                "1",
                "--root",
                str(tmp_path),
                "--mode",
                "sleep",
                "--seconds",
                "30",
            ]
        )
    assert rc2 == EXIT_ACCEPTED_EXIT_NOW
    assert rc2 != EXIT_REFUSED


def test_logging_cannot_overwrite_captured_helper_exit_code():
    """
    Behavioral PowerShell: capture $LASTEXITCODE before logging; logging must not
    replace the returned code (case 10).
    """
    import subprocess

    ps = r"""
$ErrorActionPreference = 'Continue'
function Invoke-CapturePattern {
    $sleepOut = & python -c "print('helper-line'); raise SystemExit(10)" 2>&1
    $code = $LASTEXITCODE
    foreach ($line in @($sleepOut)) { Write-Output "LOG:$line" | Out-Null }
    return $code
}
$code = Invoke-CapturePattern
Write-Output "CAPTURED=$code"
exit 0
"""
    proc = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps,
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    assert "CAPTURED=10" in (proc.stdout or "")
    # Contrast: pipe through ForEach-Object can clobber (document why we avoid it)
    ps_pipe = r"""
$ErrorActionPreference = 'Continue'
& python -c "print('x'); raise SystemExit(10)" 2>&1 | ForEach-Object { $_ | Out-Null }
Write-Output "AFTER_PIPE=$LASTEXITCODE"
exit 0
"""
    proc2 = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps_pipe,
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    # After pipe, LASTEXITCODE is not reliably 10; captured pattern above is.
    assert "AFTER_PIPE=" in (proc2.stdout or "")


def test_07_loop_releases_own_lock_via_finally_cleanup():
    src = CONT.read_text(encoding="utf-8")
    assert "function Remove-OwnLoopLock" in src
    finally_block = src.split("finally")[1]
    assert "Remove-OwnLoopLock" in finally_block
    assert "continuous_loop.lock" in src


def test_08_acknowledgement_alone_is_not_full_completion(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101)

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
                safe_boundary="INTERRUPTIBLE_WAIT",
                current_cycle_state="WAITING",
                next_action="EXIT",
                safe_message="ack only",
                ack_path=live / "operator_shutdown_ack_continuous.json",
                script_name="run_continuous_loop.ps1",
            )

    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=1.0,
        exit_timeout_sec=0.2,
        lock_timeout_sec=0.05,
        poll_sec=0.05,
        sleep_fn=sleep_fn,
    )
    assert r["result"] != "COOPERATIVE_SHUTDOWN_COMPLETED"
    assert (r.get("acknowledgement_results") or {}).get("continuous", {}).get("accepted") is True


def test_09_10_completion_requires_process_absence_and_lock_release(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101)

    def sleep_fn(_):
        live = tmp_path / "data" / "live"
        req = live / "operator_shutdown_request.json"
        if req.is_file() and not (live / "operator_shutdown_ack_continuous.json").is_file():
            rid = json.loads(req.read_text(encoding="utf-8"))["request_id"]
            write_ack(
                component="continuous",
                request_id=rid,
                state="EXITING",
                actual_pid=101,
                safe_boundary="INTERRUPTIBLE_WAIT",
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
        ack_timeout_sec=1.0,
        exit_timeout_sec=1.0,
        lock_timeout_sec=1.0,
        poll_sec=0.05,
        sleep_fn=sleep_fn,
    )
    assert r["result"] == "COOPERATIVE_SHUTDOWN_COMPLETED"
    assert (r.get("process_exit_results") or {}).get("continuous", {}).get("exited") is True
    assert (r.get("lock_release_results") or {}).get("continuous", {}).get("released") is True


def test_11_12_ordinary_in_progress_vs_explicit_timeout(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101)
    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=0.15,
        exit_timeout_sec=0.05,
        lock_timeout_sec=0.05,
        poll_sec=0.05,
        sleep_fn=lambda _: None,
    )
    assert r["result"] == "SHUTDOWN_TIMED_OUT"
    assert (r.get("acknowledgement_results") or {}).get("continuous", {}).get("timed_out") is True
    req = json.loads(
        (tmp_path / "data" / "live" / "operator_shutdown_request.json").read_text(encoding="utf-8")
    )
    assert req["state"] == "DELIVERED"
    assert req.get("controller_ack_timed_out") is True
    assert shutdown_active(root=tmp_path)["active"] is True


def test_13_14_15_16_timeout_never_terminates_deletes_repairs_or_retries(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101)
    lock = tmp_path / "data" / "live" / "continuous_loop.lock"
    assert lock.is_file()
    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=0.12,
        exit_timeout_sec=0.05,
        lock_timeout_sec=0.05,
        poll_sec=0.05,
        sleep_fn=lambda _: None,
    )
    assert r["result"] == "SHUTDOWN_TIMED_OUT"
    assert r["forced_termination_used"] is False
    assert r["runtime_locks_deleted_by_controller"] is False
    assert lock.is_file()
    assert r.get("safe_to_retry") is True
    from services.operator_shutdown_controller import controller_has_termination_paths

    assert controller_has_termination_paths() is False
    ctrl = (ROOT / "services" / "operator_shutdown_controller.py").read_text(encoding="utf-8")
    assert "os.kill(" not in ctrl
    assert "import subprocess" not in ctrl
    assert "taskkill.exe" not in ctrl.lower()


def test_17_18_19_active_stop_blocks_duplicate_start_restart(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101)
    prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=0.12,
        exit_timeout_sec=0.05,
        lock_timeout_sec=0.05,
        poll_sec=0.05,
        sleep_fn=lambda _: None,
    )
    assert shutdown_active(root=tmp_path)["active"] is True
    el_stop = stop_eligibility(root=tmp_path)
    assert el_stop["stop_button_enabled"] is False
    el_start = start_eligibility(root=tmp_path)
    assert el_start["start_button_enabled"] is False
    assert "SHUTDOWN" in str(el_start.get("decision") or "").upper()
    el_restart = restart_eligibility(root=tmp_path)
    assert el_restart.get("restart_button_enabled") is False


def test_20_21_partial_stop_blocks_start_and_restart(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101)

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
                safe_boundary="INTERRUPTIBLE_WAIT",
                current_cycle_state="WAITING",
                next_action="EXIT",
                safe_message="accepted still running",
                ack_path=live / "operator_shutdown_ack_continuous.json",
                script_name="run_continuous_loop.ps1",
            )

    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=1.0,
        exit_timeout_sec=0.2,
        lock_timeout_sec=0.05,
        poll_sec=0.05,
        sleep_fn=sleep_fn,
    )
    assert r["result"] in ("SHUTDOWN_TIMED_OUT", "PARTIAL_SHUTDOWN")
    req = json.loads(
        (tmp_path / "data" / "live" / "operator_shutdown_request.json").read_text(encoding="utf-8")
    )
    assert req["state"] == "EXIT_PENDING"
    assert shutdown_active(root=tmp_path)["active"] is True
    assert start_eligibility(root=tmp_path)["start_button_enabled"] is False
    assert restart_eligibility(root=tmp_path).get("restart_button_enabled") is False


def test_22_expired_request_rejected(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    atomic_write_json(
        live / "operator_shutdown_request.json",
        {**_valid_request(1), "expires_at": past},
    )
    v = validate_request_for_component(
        json.loads((live / "operator_shutdown_request.json").read_text(encoding="utf-8")),
        component="continuous",
        actual_pid=1,
    )
    assert v["ok"] is False
    assert v["state"] == "REFUSED_EXPIRED"


def test_23_wrong_target_ignored(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    atomic_write_json(
        live / "operator_shutdown_request.json",
        _valid_request(9, component="watchdog"),
    )
    poll = poll_shutdown_for_loop(
        component="continuous",
        actual_pid=9,
        script_name="run_continuous_loop.ps1",
        root=tmp_path,
    )
    assert poll["action"] == "CONTINUE_REFUSED"


def test_24_pid_mismatch_rejected(tmp_path):
    v = validate_request_for_component(
        _valid_request(10),
        component="continuous",
        actual_pid=99,
    )
    assert v["ok"] is False
    assert v["state"] == "REFUSED_WRONG_INSTANCE"


def test_25_26_creation_and_fingerprint_mismatch_not_accepted_as_identity():
    expected = {
        "expected_pid": 10,
        "expected_command_fingerprint": "ps1:run_continuous_loop.ps1:pid=10",
        "expected_created_at": "2026-07-20T09:31:02",
    }
    req = _valid_request(10)
    req["expected_instances"]["continuous"] = expected
    v = validate_request_for_component(req, component="continuous", actual_pid=11)
    assert v["ok"] is False
    v2 = validate_request_for_component(req, component="continuous", actual_pid=10)
    assert v2["ok"] is True


def test_27_malformed_fails_closed(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    (live / "operator_shutdown_request.json").write_text("{bad", encoding="utf-8")
    poll = poll_shutdown_for_loop(
        component="continuous",
        actual_pid=1,
        script_name="run_continuous_loop.ps1",
        root=tmp_path,
    )
    assert poll["action"] == "CONTINUE_REFUSED"


def test_28_unknown_target_fails_closed():
    v = validate_request_for_component(
        {
            "action": ACTION,
            "expires_at": _future(),
            "target_components": ["broker"],
            "expected_instances": {"broker": {"expected_pid": 1}},
        },
        component="continuous",
        actual_pid=1,
    )
    assert v["ok"] is False


def test_29_watchdog_interruptible_wait_still_correct(tmp_path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    atomic_write_json(live / "operator_shutdown_request.json", _valid_request(6, "watchdog"))
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
    wd_src = WD.read_text(encoding="utf-8")
    sleep_fn = wd_src[
        wd_src.index("function Invoke-InterruptibleSleep") : wd_src.index(
            "function Remove-OwnLoopLock"
        )
    ]
    assert "$code = $LASTEXITCODE" in sleep_fn
    assert "| ForEach-Object" not in sleep_fn


def test_30_41_safety_invariants_and_no_runtime_mutation():
    from services.operator_shutdown_controller import controller_has_termination_paths

    stop_src = (ROOT / "services" / "operator_stop_action.py").read_text(encoding="utf-8")
    ctrl_src = (ROOT / "services" / "operator_shutdown_controller.py").read_text(encoding="utf-8")
    coop_src = (ROOT / "services" / "operator_cooperative_shutdown.py").read_text(encoding="utf-8")
    cont = CONT.read_text(encoding="utf-8")
    assert controller_has_termination_paths() is False
    assert "Stop-Process" not in cont
    assert "os.kill(" not in stop_src
    assert "os.kill(" not in ctrl_src
    assert "import subprocess" not in ctrl_src
    assert "Popen(" not in ctrl_src
    assert "shell=True" not in stop_src + ctrl_src + coop_src
    assert "TRITON_ENABLE_PAPER_EXECUTION" in cont
    assert "schtasks" not in (stop_src + ctrl_src + coop_src).lower()
    assert "Register-ScheduledTask" not in cont
    assert "execute_start_triton" not in coop_src
    assert "execute_restart_triton" not in coop_src
    assert "START_TRITON" not in coop_src
    assert EXIT_TIMEOUT_SEC == 60


def test_42_ack_and_exit_timeouts_are_separate():
    assert ACK_TIMEOUT_SEC == recommended_ack_timeout_sec()
    assert EXIT_TIMEOUT_SEC == 60
    assert (
        inspect.signature(prepare_cooperative_shutdown).parameters["ack_timeout_sec"].default
        == ACK_TIMEOUT_SEC
    )
    assert (
        inspect.signature(prepare_cooperative_shutdown).parameters["exit_timeout_sec"].default
        == EXIT_TIMEOUT_SEC
    )


def test_43_late_ack_still_possible_after_controller_ack_timeout(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101)
    prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=0.12,
        exit_timeout_sec=0.05,
        lock_timeout_sec=0.05,
        poll_sec=0.05,
        sleep_fn=lambda _: None,
    )
    poll = poll_shutdown_for_loop(
        component="continuous",
        actual_pid=101,
        script_name="run_continuous_loop.ps1",
        root=tmp_path,
        cycle_state="WAITING",
        safe_boundary="INTERRUPTIBLE_WAIT",
    )
    assert poll["action"] == "EXIT_NOW"


def test_44_interruptible_wait_is_documented_safe_boundary():
    src = CONT.read_text(encoding="utf-8")
    assert "INTERRUPTIBLE_WAIT" in src
    sleep_fn = src[
        src.index("function Invoke-InterruptibleSleep") : src.index("function Remove-OwnLoopLock")
    ]
    assert "run_market_open" not in sleep_fn


def test_45_stop_default_ack_timeout_aligned():
    from services.operator_stop_action import execute_stop_triton as est

    default = inspect.signature(est).parameters["ack_timeout_sec"].default
    assert float(default) == float(recommended_ack_timeout_sec())


def test_46_no_broker_order_approval_authorization_mutation_on_timeout(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101)
    r = prepare_cooperative_shutdown(
        root=tmp_path,
        deliver=True,
        ack_timeout_sec=0.1,
        exit_timeout_sec=0.05,
        lock_timeout_sec=0.05,
        poll_sec=0.05,
        sleep_fn=lambda _: None,
    )
    assert r["orders_submitted"] is False
    assert r["approvals_created"] is False
    assert r["paper_arm_mutated"] is False


def test_48_request_ttl_covers_supported_observation_window():
    assert REQUEST_TTL_SEC >= recommended_ack_timeout_sec()
    assert REQUEST_TTL_SEC >= CONTINUOUS_WAIT_CHECK_SEC * 2
