"""Controller polls must not reuse the dashboard process-scan memo.

The scanner is injected. No test launches a process or writes a production file.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from services import operator_runtime_classifier as classifier
from services import runtime_process_detector as detector
from services.operator_restart_action import (
    _verify_start_postconditions,
    _verify_stop_postconditions,
)
from services.operator_runtime_controller import (
    POSTCONDITION_POLL_SEC,
    POSTCONDITION_TIMEOUT_SEC,
    execute_start_triton,
)
from services.operator_shutdown_controller import _poll_completion
from services.runtime_process_detector import (
    CONTINUOUS_SCRIPT,
    WATCHDOG_SCRIPT,
    _find_loop_processes,
    open_process_scan_scope,
)


def _lock_text(pid: int) -> str:
    now = datetime.now(timezone.utc)
    return (
        f"started_at={now.isoformat()}\n"
        f"refreshed_at={now.isoformat()}\n"
        f"pid={pid}\n"
        f"source=test\n"
        f"cycle=1\n"
    )


def _arm(root: Path) -> None:
    live = root / "data" / "live"
    live.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    (live / "paper_arm_confirm.json").write_text(
        json.dumps(
            {
                "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "expires_at": (now + timedelta(minutes=120)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mode": "ASSISTED",
                "allow_execute": True,
                "confirmed": True,
            }
        ),
        encoding="utf-8",
    )


def _root(tmp_path: Path) -> Path:
    live = tmp_path / "data" / "live"
    results = tmp_path / "data" / "results"
    scripts = tmp_path / "scripts"
    live.mkdir(parents=True)
    results.mkdir(parents=True)
    scripts.mkdir(parents=True)
    (scripts / "run_watchdog_loop.ps1").write_text("# fixture\n", encoding="utf-8")
    (scripts / "run_continuous_loop.ps1").write_text("# fixture\n", encoding="utf-8")
    (results / "execution_readiness.json").write_text(
        json.dumps({"live_execution_permitted": False, "readiness_status": "READY"}),
        encoding="utf-8",
    )
    return tmp_path


def _bind(monkeypatch, root: Path, state: dict, scans: dict) -> None:
    monkeypatch.setattr(classifier, "ROOT", root)

    def fake_known():
        scans["n"] += 1
        method = "win32_process_commandline_file_invocation"
        return {
            WATCHDOG_SCRIPT: (list(state["watchdog"]), method, []),
            CONTINUOUS_SCRIPT: (list(state["continuous"]), method, []),
        }

    def forbid_real_scan(*_args, **_kwargs):
        raise AssertionError("real process scan")

    monkeypatch.setattr(detector, "_scan_known_loop_processes", fake_known)
    monkeypatch.setattr(detector.subprocess, "run", forbid_real_scan)


def _write_lock(root: Path, name: str, pid: int) -> None:
    (root / "data" / "live" / name).write_text(_lock_text(pid), encoding="utf-8")


def test_start_poll_sees_launches_inside_the_same_dashboard_render(tmp_path: Path, monkeypatch):
    root = _root(tmp_path)
    _arm(root)
    state = {"watchdog": [], "continuous": []}
    scans = {"n": 0}
    _bind(monkeypatch, root, state, scans)
    launched = []

    def launch(component):
        launched.append(component)
        if component == "watchdog":
            state["watchdog"] = [6776]
            _write_lock(root, "watchdog_loop.lock", 6776)
        else:
            state["continuous"] = [7001]
            _write_lock(root, "continuous_loop.lock", 7001)
        return {"ok": True, "component": component}

    scope = open_process_scan_scope("render-start")
    try:
        assert _find_loop_processes(WATCHDOG_SCRIPT)[0] == []
        assert _find_loop_processes(CONTINUOUS_SCRIPT)[0] == []
        assert scans["n"] == 1
        rec = execute_start_triton(
            root=root,
            launch_fn=launch,
            audit_path=root / "data" / "results" / "operator_runtime_actions.jsonl",
            guard_path=root / "data" / "live" / "operator_start_action.guard.json",
        )
        assert launched == ["watchdog", "continuous"]
        assert rec["result"] == "STARTED"
        assert rec["component_results"]["watchdog"]["postcondition"]["state"] == "HEALTHY_RUNNING"
        assert rec["component_results"]["continuous"]["postcondition"]["state"] == "HEALTHY_RUNNING"
        assert scans["n"] > 1
        cached = scans["n"]
        assert _find_loop_processes(WATCHDOG_SCRIPT)[0] == []
        assert _find_loop_processes(CONTINUOUS_SCRIPT)[0] == []
        assert scans["n"] == cached
    finally:
        scope.close()


def test_lock_alone_does_not_satisfy_start_postcondition(tmp_path: Path, monkeypatch):
    root = _root(tmp_path)
    _arm(root)
    state = {"watchdog": [], "continuous": []}
    scans = {"n": 0}
    _bind(monkeypatch, root, state, scans)
    monkeypatch.setattr(
        "services.operator_runtime_controller.POSTCONDITION_TIMEOUT_SEC",
        0.05,
    )
    monkeypatch.setattr(
        "services.operator_runtime_controller.POSTCONDITION_POLL_SEC",
        0.01,
    )
    launched = []

    def launch(component):
        launched.append(component)
        _write_lock(root, "watchdog_loop.lock", 6776)
        return {"ok": True, "component": component}

    scope = open_process_scan_scope("render-lock")
    try:
        _find_loop_processes(WATCHDOG_SCRIPT)
        rec = execute_start_triton(
            root=root,
            launch_fn=launch,
            audit_path=root / "data" / "results" / "operator_runtime_actions.jsonl",
            guard_path=root / "data" / "live" / "operator_start_action.guard.json",
        )
    finally:
        scope.close()
    assert launched == ["watchdog"]
    assert rec["result"] == "START_FAILED"
    assert (
        rec["component_results"]["watchdog"]["postcondition"]["reason"] == "postcondition_timeout"
    )
    assert rec["component_results"]["watchdog"]["postcondition"]["state"] == "STALE_LOCK_NO_PROCESS"
    assert POSTCONDITION_TIMEOUT_SEC == 25
    assert POSTCONDITION_POLL_SEC == 1.0


def test_stop_and_restart_postconditions_use_a_fresh_scan(tmp_path: Path, monkeypatch):
    root = _root(tmp_path)
    state = {"watchdog": [6776], "continuous": [7001]}
    scans = {"n": 0}
    _bind(monkeypatch, root, state, scans)
    _write_lock(root, "watchdog_loop.lock", 6776)
    _write_lock(root, "continuous_loop.lock", 7001)
    scope = open_process_scan_scope("render-stop")
    try:
        assert _find_loop_processes(WATCHDOG_SCRIPT)[0] == [6776]
        assert scans["n"] == 1
        state["watchdog"] = []
        state["continuous"] = []
        (root / "data" / "live" / "watchdog_loop.lock").unlink()
        (root / "data" / "live" / "continuous_loop.lock").unlink()
        assert _find_loop_processes(WATCHDOG_SCRIPT)[0] == [6776]
        assert scans["n"] == 1

        ack = {
            "request_id": "req-1",
            "state": "EXITING",
            "actual_pid": 6776,
            "safe_boundary": "INTERRUPTIBLE_WAIT",
        }
        (root / "data" / "live" / "operator_shutdown_ack_watchdog.json").write_text(
            json.dumps(ack),
            encoding="utf-8",
        )
        (root / "data" / "live" / "operator_shutdown_ack_continuous.json").write_text(
            json.dumps({**ack, "actual_pid": 7001}),
            encoding="utf-8",
        )
        stop = _poll_completion(
            targets=["watchdog", "continuous"],
            expected_instances={
                "watchdog": {"expected_pid": 6776},
                "continuous": {"expected_pid": 7001},
            },
            classify_fn=lambda: classifier.classify_runtime_loops(root=root),
            root=root,
            ack_timeout=1,
            exit_timeout=1,
            lock_timeout=1,
            poll_sec=0.01,
            sleep_fn=lambda _seconds: None,
            request_id="req-1",
        )
        assert stop["result"] == "COOPERATIVE_SHUTDOWN_COMPLETED"
        assert scans["n"] > 1

        restart_stop = _verify_stop_postconditions(
            {
                "result": "STOP_COMPLETED",
                "forced_termination_used": False,
                "runtime_locks_deleted_by_controller": False,
                "runtime_locks_repaired_by_controller": False,
                "acknowledgement_results": {
                    "watchdog": {"accepted": True},
                    "continuous": {"accepted": True},
                },
                "process_exit_results": {
                    "watchdog": {"exited": True},
                    "continuous": {"exited": True},
                },
                "lock_release_results": {
                    "watchdog": {"released": True},
                    "continuous": {"released": True},
                },
            },
            stop_required=True,
            root=root,
        )
        assert restart_stop["ok"] is True

        state["watchdog"] = [6800]
        state["continuous"] = [6801]
        _write_lock(root, "watchdog_loop.lock", 6800)
        _write_lock(root, "continuous_loop.lock", 6801)
        fresh = classifier.classify_runtime_loops(root=root)
        assert fresh["watchdog"]["state"] == "HEALTHY_RUNNING"
        assert fresh["continuous"]["state"] == "HEALTHY_RUNNING"
        start_post = _verify_start_postconditions(
            {
                "result": "STARTED",
                "launched_components": ["watchdog", "continuous"],
            },
            fresh,
        )
        assert start_post["ok"] is True
        cached = scans["n"]
        assert _find_loop_processes(WATCHDOG_SCRIPT)[0] == [6776]
        assert scans["n"] == cached
    finally:
        scope.close()

    scope = open_process_scan_scope("render-next")
    try:
        before = scans["n"]
        _find_loop_processes(CONTINUOUS_SCRIPT)
        assert scans["n"] == before + 1
    finally:
        scope.close()
