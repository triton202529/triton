"""DO-003 protected START_TRITON tests — fixtures/mocks only. No real loops/locks/broker."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dashboard.operator_console import DEFAULT_OPERATOR_PAGE, OPERATOR_PAGES
from services.operator_runtime_classifier import (
    classify_loop,
    classify_runtime_loops,
    evaluate_start_decision,
)
from services.operator_runtime_controller import (
    ACTION_START,
    DENIED_ACTIONS,
    append_runtime_action_audit,
    execute_start_triton,
    live_execution_appears_enabled,
    read_recent_runtime_actions,
    start_eligibility,
    verify_fixed_paths,
)


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
    live.mkdir(parents=True)
    results.mkdir(parents=True)
    scripts.mkdir(parents=True)
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


def test_operator_nav_unchanged():
    assert OPERATOR_PAGES == [
        "Operations",
        "Activity",
        "Portfolio",
        "Orders",
        "Risk & Alerts",
        "Diagnostics",
    ]
    assert DEFAULT_OPERATOR_PAGE == "Operations"


def test_both_absent_permits_start(tmp_path: Path):
    _write_fixture(tmp_path, cont_pids=[], wd_pids=[], write_cont_lock=False, write_wd_lock=False)
    cls = classify_runtime_loops(root=tmp_path)
    d = evaluate_start_decision(cls, live_execution_blocked=True)
    assert d["decision"] == "START_ALLOWED"
    assert set(d["components_to_start"]) == {"watchdog", "continuous"}


def test_both_healthy_already_running(tmp_path: Path):
    _write_fixture(tmp_path, cont_pids=[111], wd_pids=[222])
    cls = classify_runtime_loops(root=tmp_path)
    assert cls["continuous"]["state"] == "HEALTHY_RUNNING"
    assert cls["watchdog"]["state"] == "HEALTHY_RUNNING"
    d = evaluate_start_decision(cls, live_execution_blocked=True)
    assert d["decision"] == "ALREADY_RUNNING"


def test_partial_watchdog_only(tmp_path: Path):
    _write_fixture(tmp_path, cont_pids=[111], wd_pids=[], write_wd_lock=False)
    d = evaluate_start_decision(classify_runtime_loops(root=tmp_path), live_execution_blocked=True)
    assert d["decision"] == "PARTIAL_START_ALLOWED"
    assert d["components_to_start"] == ["watchdog"]
    assert d["preserved_components"] == ["continuous"]


def test_partial_continuous_only(tmp_path: Path):
    _write_fixture(tmp_path, cont_pids=[], wd_pids=[222], write_cont_lock=False)
    d = evaluate_start_decision(classify_runtime_loops(root=tmp_path), live_execution_blocked=True)
    assert d["decision"] == "PARTIAL_START_ALLOWED"
    assert d["components_to_start"] == ["continuous"]


def test_stale_lock_no_process_refuses(tmp_path: Path):
    _write_fixture(
        tmp_path,
        cont_pids=[],
        wd_pids=[],
        write_cont_lock=True,
        write_wd_lock=True,
        cont_lock_pid=111,
        wd_lock_pid=222,
    )
    cls = classify_runtime_loops(root=tmp_path)
    assert cls["continuous"]["state"] == "STALE_LOCK_NO_PROCESS"
    d = evaluate_start_decision(cls, live_execution_blocked=True)
    assert d["decision"] == "START_REFUSED_UNSAFE_STATE"


def test_process_without_lock_refuses(tmp_path: Path):
    _write_fixture(
        tmp_path, cont_pids=[111], wd_pids=[222], write_cont_lock=False, write_wd_lock=False
    )
    cls = classify_runtime_loops(root=tmp_path)
    assert cls["continuous"]["state"] == "PROCESS_WITHOUT_VALID_LOCK"
    assert evaluate_start_decision(cls, live_execution_blocked=True)["decision"] == (
        "START_REFUSED_UNSAFE_STATE"
    )


def test_pid_mismatch_refuses(tmp_path: Path):
    _write_fixture(tmp_path, cont_pids=[111], wd_pids=[222], cont_lock_pid=999, wd_lock_pid=222)
    cls = classify_runtime_loops(root=tmp_path)
    assert cls["continuous"]["state"] == "PID_MISMATCH"
    assert evaluate_start_decision(cls, live_execution_blocked=True)["decision"] == (
        "START_REFUSED_UNSAFE_STATE"
    )


def test_command_mismatch_refuses(tmp_path: Path):
    _write_fixture(
        tmp_path,
        cont_pids=[111],
        wd_pids=[222],
        command_lines={111: "powershell.exe -File other.ps1", 222: "run_watchdog_loop.ps1"},
    )
    cls = classify_runtime_loops(root=tmp_path)
    assert cls["continuous"]["state"] == "COMMAND_MISMATCH"
    assert evaluate_start_decision(cls, live_execution_blocked=True)["decision"] == (
        "START_REFUSED_UNSAFE_STATE"
    )


def test_duplicate_continuous_refuses(tmp_path: Path):
    _write_fixture(
        tmp_path, cont_pids=[111, 112], wd_pids=[222], write_cont_lock=True, cont_lock_pid=111
    )
    cls = classify_runtime_loops(root=tmp_path)
    assert cls["continuous"]["state"] == "DUPLICATE_PROCESSES"
    assert evaluate_start_decision(cls, live_execution_blocked=True)["decision"] == (
        "START_REFUSED_UNSAFE_STATE"
    )


def test_duplicate_watchdog_refuses(tmp_path: Path):
    _write_fixture(tmp_path, cont_pids=[111], wd_pids=[221, 222], wd_lock_pid=221)
    assert classify_runtime_loops(root=tmp_path)["watchdog"]["state"] == "DUPLICATE_PROCESSES"


def test_stale_overdue_refuses_duplicate(tmp_path: Path):
    _write_fixture(tmp_path, cont_pids=[111], wd_pids=[222], cont_minutes_ago=200.0)
    cls = classify_runtime_loops(root=tmp_path)
    assert cls["continuous"]["state"] == "STALE_OR_OVERDUE"
    d = evaluate_start_decision(cls, live_execution_blocked=True)
    assert d["decision"] == "START_REFUSED_EXISTING_UNHEALTHY_PROCESS"


def test_unknown_detection_refuses(tmp_path: Path):
    _write_fixture(tmp_path, cont_pids=[], wd_pids=[], write_cont_lock=False, write_wd_lock=False)
    live = tmp_path / "data" / "live"
    (live / "fixture_process_state.json").write_text(
        json.dumps(
            {
                "detected_continuous_process_ids": [],
                "detected_watchdog_process_ids": [],
                "process_detection_confidence": "low",
            }
        ),
        encoding="utf-8",
    )
    cls = classify_runtime_loops(root=tmp_path)
    assert cls["continuous"]["state"] == "UNKNOWN"
    assert evaluate_start_decision(cls, live_execution_blocked=True)["decision"] == (
        "START_REFUSED_UNSAFE_STATE"
    )


def test_live_enabled_refuses(tmp_path: Path):
    _write_fixture(
        tmp_path,
        cont_pids=[],
        wd_pids=[],
        write_cont_lock=False,
        write_wd_lock=False,
        live_permitted=True,
    )
    assert live_execution_appears_enabled(root=tmp_path, env={}) is True
    d = evaluate_start_decision(classify_runtime_loops(root=tmp_path), live_execution_blocked=False)
    assert d["decision"] == "START_REFUSED_LIVE_SAFETY"


def test_missing_script_refuses(tmp_path: Path):
    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    (tmp_path / "scripts" / "run_continuous_loop.ps1").unlink()
    paths = verify_fixed_paths(root=tmp_path)
    assert paths["ok"] is False
    elig = start_eligibility(root=tmp_path)
    assert elig["decision"] == "START_REFUSED_UNSAFE_STATE"


def test_frontend_cannot_supply_command_pid_path(tmp_path: Path):
    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    audit = tmp_path / "data" / "results" / "operator_runtime_actions.jsonl"
    for kwargs in (
        {"command": "calc.exe"},
        {"pid": 1234},
        {"path": "C:\\evil"},
        {"env": {"FOO": "1"}},
        {"shell_argument": "-Command hack"},
    ):
        rec = execute_start_triton(
            root=tmp_path,
            audit_path=audit,
            guard_path=tmp_path / "data" / "live" / "g.json",
            skip_postcondition_wait=True,
            launch_fn=lambda c: (_ for _ in ()).throw(AssertionError("must not launch")),
            **kwargs,
        )
        assert rec["result"] == "START_REFUSED_UNSAFE_STATE"
        assert rec["error_code"] == "FRONTEND_INJECTION_REFUSED"


def test_denied_actions_refused(tmp_path: Path):
    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    audit = tmp_path / "data" / "results" / "operator_runtime_actions.jsonl"
    for action in ("STOP_TRITON", "RESTART_TRITON", "PAPER_ARM", "ENABLE_LIVE", "REMOVE_LOCK"):
        assert action in DENIED_ACTIONS
        rec = execute_start_triton(
            action=action,
            root=tmp_path,
            audit_path=audit,
            guard_path=tmp_path / "data" / "live" / f"g_{action}.json",
            skip_postcondition_wait=True,
        )
        assert rec["result"] == "START_REFUSED_UNSAFE_STATE"
        assert rec["error_code"] == "ACTION_NOT_ALLOWED"


def test_already_running_execute(tmp_path: Path):
    _write_fixture(tmp_path, cont_pids=[111], wd_pids=[222])
    launched = []
    rec = execute_start_triton(
        root=tmp_path,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        guard_path=tmp_path / "data" / "live" / "operator_start_action.guard.json",
        skip_postcondition_wait=True,
        launch_fn=lambda c: launched.append(c) or {"ok": True},
    )
    assert rec["result"] == "ALREADY_RUNNING"
    assert launched == []
    assert rec["paper_arm_mutated"] is False
    assert rec["runtime_locks_deleted"] is False
    assert rec["processes_terminated"] is False


def test_start_launches_both_with_mocks(tmp_path: Path):
    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    state = {"cont": [], "wd": []}
    launched = []

    def snapshot():
        return {
            "continuous_pids": list(state["cont"]),
            "watchdog_pids": list(state["wd"]),
            "detection_confidence": "high",
            "command_lines": {},
        }

    def launch(component):
        launched.append(component)
        if component == "watchdog":
            state["wd"] = [222]
            (tmp_path / "data" / "live" / "watchdog_loop.lock").write_text(
                _lock_text(222), encoding="utf-8"
            )
        else:
            state["cont"] = [111]
            (tmp_path / "data" / "live" / "continuous_loop.lock").write_text(
                _lock_text(111), encoding="utf-8"
            )
        return {"ok": True, "component": component}

    rec = execute_start_triton(
        root=tmp_path,
        process_snapshot=snapshot,
        launch_fn=launch,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        guard_path=tmp_path / "data" / "live" / "operator_start_action.guard.json",
        skip_postcondition_wait=False,
    )
    assert launched == ["watchdog", "continuous"]
    assert rec["result"] == "STARTED"
    assert rec["live_execution_blocked_after"] is True
    assert set(rec["launched_components"]) == {"watchdog", "continuous"}


def test_partial_start_preserves_healthy(tmp_path: Path):
    _write_fixture(tmp_path, cont_pids=[111], wd_pids=[], write_wd_lock=False)
    state = {"cont": [111], "wd": []}
    launched = []

    def snapshot():
        return {
            "continuous_pids": list(state["cont"]),
            "watchdog_pids": list(state["wd"]),
            "detection_confidence": "high",
            "command_lines": {},
        }

    def launch(component):
        launched.append(component)
        assert component == "watchdog"
        state["wd"] = [222]
        (tmp_path / "data" / "live" / "watchdog_loop.lock").write_text(
            _lock_text(222), encoding="utf-8"
        )
        return {"ok": True}

    rec = execute_start_triton(
        root=tmp_path,
        process_snapshot=snapshot,
        launch_fn=launch,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        guard_path=tmp_path / "data" / "live" / "operator_start_action.guard.json",
    )
    assert launched == ["watchdog"]
    assert "continuous" in (rec.get("preserved_components") or [])
    assert rec["result"] == "STARTED"


def test_idempotency_no_relaunch(tmp_path: Path):
    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    state = {"cont": [], "wd": []}
    launched = []

    def snapshot():
        return {
            "continuous_pids": list(state["cont"]),
            "watchdog_pids": list(state["wd"]),
            "detection_confidence": "high",
            "command_lines": {},
        }

    def launch(component):
        launched.append(component)
        if component == "watchdog":
            state["wd"] = [222]
            (tmp_path / "data" / "live" / "watchdog_loop.lock").write_text(
                _lock_text(222), encoding="utf-8"
            )
        else:
            state["cont"] = [111]
            (tmp_path / "data" / "live" / "continuous_loop.lock").write_text(
                _lock_text(111), encoding="utf-8"
            )
        return {"ok": True}

    audit = tmp_path / "data" / "results" / "operator_runtime_actions.jsonl"
    guard = tmp_path / "data" / "live" / "operator_start_action.guard.json"
    key = "idem-key-1"
    r1 = execute_start_triton(
        idempotency_key=key,
        root=tmp_path,
        process_snapshot=snapshot,
        launch_fn=launch,
        audit_path=audit,
        guard_path=guard,
    )
    assert r1["result"] == "STARTED"
    count1 = len(launched)
    r2 = execute_start_triton(
        idempotency_key=key,
        root=tmp_path,
        process_snapshot=snapshot,
        launch_fn=lambda c: launched.append(c) or {"ok": True},
        audit_path=audit,
        guard_path=guard,
    )
    assert r2.get("replayed") is True
    assert len(launched) == count1


def test_concurrent_start_in_progress(tmp_path: Path):
    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    guard = tmp_path / "data" / "live" / "operator_start_action.guard.json"
    guard.write_text(
        json.dumps(
            {
                "action_id": "other",
                "idempotency_key": "other-key",
                "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    rec = execute_start_triton(
        root=tmp_path,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        guard_path=guard,
        skip_postcondition_wait=True,
        launch_fn=lambda c: (_ for _ in ()).throw(AssertionError("no launch")),
    )
    assert rec["result"] == "START_IN_PROGRESS"


def test_scheduled_task_race_preserves(tmp_path: Path):
    """If loop becomes healthy before launch, do not duplicate."""
    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    state = {"cont": [], "wd": []}
    calls = {"n": 0}

    def snapshot():
        calls["n"] += 1
        # After first classify (preflight), simulate race: both become healthy
        if calls["n"] > 1:
            state["cont"] = [111]
            state["wd"] = [222]
            (tmp_path / "data" / "live" / "continuous_loop.lock").write_text(
                _lock_text(111), encoding="utf-8"
            )
            (tmp_path / "data" / "live" / "watchdog_loop.lock").write_text(
                _lock_text(222), encoding="utf-8"
            )
        return {
            "continuous_pids": list(state["cont"]),
            "watchdog_pids": list(state["wd"]),
            "detection_confidence": "high",
            "command_lines": {},
        }

    launched = []
    rec = execute_start_triton(
        root=tmp_path,
        process_snapshot=snapshot,
        launch_fn=lambda c: launched.append(c) or {"ok": True},
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        guard_path=tmp_path / "data" / "live" / "operator_start_action.guard.json",
        skip_postcondition_wait=True,
    )
    assert launched == []
    assert rec["result"] in ("STARTED", "ALREADY_RUNNING")


def test_postcondition_fail_without_lock(tmp_path: Path):
    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    state = {"cont": [], "wd": []}

    def snapshot():
        return {
            "continuous_pids": list(state["cont"]),
            "watchdog_pids": list(state["wd"]),
            "detection_confidence": "high",
            "command_lines": {},
        }

    def launch(component):
        # Process appears but no lock written → PROCESS_WITHOUT_VALID_LOCK
        if component == "watchdog":
            state["wd"] = [222]
        else:
            state["cont"] = [111]
        return {"ok": True}

    rec = execute_start_triton(
        root=tmp_path,
        process_snapshot=snapshot,
        launch_fn=launch,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        guard_path=tmp_path / "data" / "live" / "operator_start_action.guard.json",
    )
    assert rec["result"] in ("START_FAILED", "POSTCONDITION_FAILED", "PARTIAL_START")
    assert rec["processes_terminated"] is False


def test_partial_start_on_second_failure(tmp_path: Path):
    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    state = {"cont": [], "wd": []}

    def snapshot():
        return {
            "continuous_pids": list(state["cont"]),
            "watchdog_pids": list(state["wd"]),
            "detection_confidence": "high",
            "command_lines": {},
        }

    def launch(component):
        if component == "watchdog":
            state["wd"] = [222]
            (tmp_path / "data" / "live" / "watchdog_loop.lock").write_text(
                _lock_text(222), encoding="utf-8"
            )
            return {"ok": True}
        return {"ok": False, "error": "launch_failed"}

    rec = execute_start_triton(
        root=tmp_path,
        process_snapshot=snapshot,
        launch_fn=launch,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        guard_path=tmp_path / "data" / "live" / "operator_start_action.guard.json",
    )
    assert rec["result"] == "PARTIAL_START"
    assert "watchdog" in (rec.get("launched_components") or [])


def test_audit_records_and_sanitizes(tmp_path: Path):
    audit = tmp_path / "a.jsonl"
    append_runtime_action_audit(
        {
            "result": "START_REFUSED_UNSAFE_STATE",
            "api_key": "SECRET",
            "token": "SECRET",
            "env": {"TRITON_ENABLE_LIVE_EXECUTION": "0", "API_KEY": "x"},
            "safe_message": "refused",
        },
        path=audit,
    )
    rows = read_recent_runtime_actions(path=audit, limit=5)
    assert rows
    assert rows[0]["api_key"] == "<redacted>"
    assert rows[0]["token"] == "<redacted>"
    assert rows[0]["env"] == "<redacted>"


def test_audit_failure_refuses_launch(tmp_path: Path, monkeypatch):
    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    bad_audit = tmp_path / "missing_dir_not_created" / "nope.jsonl"
    # Make parent a file so mkdir/write fails
    parent = bad_audit.parent
    parent.write_text("not-a-dir", encoding="utf-8")

    launched = []
    rec = execute_start_triton(
        root=tmp_path,
        audit_path=bad_audit,
        guard_path=tmp_path / "data" / "live" / "operator_start_action.guard.json",
        skip_postcondition_wait=True,
        launch_fn=lambda c: launched.append(c) or {"ok": True},
    )
    assert rec["result"] == "START_REFUSED_UNSAFE_STATE"
    assert rec["error_code"] == "AUDIT_UNAVAILABLE"
    assert launched == []


def test_no_termination_or_lock_delete_in_controller_source():
    import inspect
    from services import operator_runtime_controller as c

    src = inspect.getsource(c)
    for token in (
        "taskkill",
        "Stop-Process",
        "terminate(",
        "kill(",
        "Clear-OrphanLoopLock",
        "unlink(CONTINUOUS",
        "unlink(WATCHDOG",
    ):
        assert token not in src
    assert "shell=True" not in src or "shell=False" in src


def test_localhost_config_still_present():
    cfg = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    assert 'address = "127.0.0.1"' in cfg.read_text(encoding="utf-8")


def test_eligibility_disables_unsafe(tmp_path: Path):
    _write_fixture(tmp_path, cont_pids=[], wd_pids=[], write_cont_lock=True, cont_lock_pid=1)
    elig = start_eligibility(root=tmp_path)
    assert elig["start_button_enabled"] is False
    assert elig["stop_available"] is False
    assert elig["restart_available"] is False


def test_classify_loop_not_running():
    c = classify_loop(
        loop="continuous",
        script_name="run_continuous_loop.ps1",
        lock_path=Path("/no/such/lock"),
        pids=[],
        overdue_threshold_minutes=60,
    )
    assert c.state == "NOT_RUNNING"


def test_default_launch_uses_breakaway_job_flags(tmp_path: Path, monkeypatch):
    """Children must break away from Streamlit/agent Job Objects or st.rerun kills them."""
    import subprocess

    from services.operator_runtime_controller import default_launch_loop

    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    captured = {}

    class _FakeProc:
        pid = 424242

    def fake_popen(*args, **kwargs):
        captured["args"] = args[0] if args else kwargs.get("args")
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    rec = default_launch_loop("watchdog", root=tmp_path)
    assert rec["ok"] is True
    flags = int(captured["kwargs"].get("creationflags") or 0)
    breakaway = int(getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000))
    no_window = int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    new_group = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))
    assert flags & breakaway, "CREATE_BREAKAWAY_FROM_JOB required for Streamlit survival"
    assert flags & no_window
    assert flags & new_group
    assert rec.get("breakaway_from_job") is True
    assert captured["kwargs"].get("cwd") == str(tmp_path)
    assert captured["kwargs"].get("shell") is False
    argv = captured["args"]
    assert str(tmp_path / "scripts" / "run_watchdog_loop.ps1") in argv
    env = captured["kwargs"].get("env") or {}
    assert env.get("TRITON_ENABLE_LIVE_EXECUTION") == "0"
    assert env.get("TRITON_LIVE_EXECUTION") == "0"
    assert env.get("TRITON_ENABLE_PAPER_EXECUTION") == "1"


def test_default_launch_continuous_repo_rooted_script(tmp_path: Path, monkeypatch):
    import subprocess

    from services.operator_runtime_controller import default_launch_loop

    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    captured = {}

    class _FakeProc:
        pid = 424243

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *a, **k: captured.update(args=a[0], kwargs=k) or _FakeProc(),
    )
    rec = default_launch_loop("continuous", root=tmp_path)
    assert rec["ok"] is True
    assert str(tmp_path / "scripts" / "run_continuous_loop.ps1") in captured["args"]
    assert captured["kwargs"]["cwd"] == str(tmp_path)


def test_ui_start_call_site_rejects_operational_kwargs():
    """Command Center must call execute_start_triton without command/path/pid/env."""
    import inspect

    from dashboard import operator_console as oc

    src = inspect.getsource(oc._render_start_controls)
    assert "execute_start_triton(" in src
    assert "command=" not in src
    assert "shell_argument=" not in src
    # path=/pid=/env= must not be passed as launch inputs (path= may appear in unrelated text)
    assert "execute_start_triton(\n" in src or "execute_start_triton(" in src
    call = src.split("execute_start_triton(")[1].split(")", 1)[0]
    assert "command" not in call
    assert "pid" not in call
    assert "shell_argument" not in call
    assert "env=" not in call


def test_expired_shutdown_request_does_not_block_start(tmp_path: Path):
    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    live = tmp_path / "data" / "live"
    past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    earlier = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    (live / "operator_shutdown_request.json").write_text(
        json.dumps(
            {
                "request_id": "hist-1",
                "state": "EXIT_PENDING",
                "requested_at": earlier,
                "expires_at": past,
                "target": "BOTH",
                "target_components": ["continuous", "watchdog"],
                "expected_instances": {
                    "watchdog": {"expected_pid": 1},
                    "continuous": {"expected_pid": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    launched = []
    rec = execute_start_triton(
        root=tmp_path,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        guard_path=tmp_path / "data" / "live" / "operator_start_action.guard.json",
        skip_postcondition_wait=True,
        launch_fn=lambda c: launched.append(c) or {"ok": True, "child_pid": 1000 + len(launched)},
        classify_fn=lambda: {
            "overall_state": "FULLY_STOPPED",
            "watchdog": {"state": "NOT_RUNNING"},
            "continuous": {"state": "NOT_RUNNING"},
        },
    )
    assert rec["result"] != "START_REFUSED_SHUTDOWN_IN_PROGRESS"
    assert set(launched) == {"watchdog", "continuous"}


def test_active_unexpired_shutdown_blocks_start(tmp_path: Path):
    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    live = tmp_path / "data" / "live"
    future = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    now = datetime.now(timezone.utc).isoformat()
    (live / "operator_shutdown_request.json").write_text(
        json.dumps(
            {
                "request_id": "active-1",
                "state": "EXIT_PENDING",
                "requested_at": now,
                "expires_at": future,
                "target": "BOTH",
                "target_components": ["continuous", "watchdog"],
                "action": "REQUEST_COOPERATIVE_SHUTDOWN",
                "expected_instances": {
                    "watchdog": {"expected_pid": 1},
                    "continuous": {"expected_pid": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    launched = []
    rec = execute_start_triton(
        root=tmp_path,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        guard_path=tmp_path / "data" / "live" / "operator_start_action.guard.json",
        skip_postcondition_wait=True,
        launch_fn=lambda c: launched.append(c) or {"ok": True},
    )
    assert rec["result"] == "START_REFUSED_SHUTDOWN_IN_PROGRESS"
    assert launched == []


def test_launch_failure_recorded_sanitized(tmp_path: Path):
    _write_fixture(tmp_path, write_cont_lock=False, write_wd_lock=False)
    audit = tmp_path / "data" / "results" / "operator_runtime_actions.jsonl"
    rec = execute_start_triton(
        root=tmp_path,
        audit_path=audit,
        guard_path=tmp_path / "data" / "live" / "operator_start_action.guard.json",
        skip_postcondition_wait=True,
        launch_fn=lambda c: {
            "ok": False,
            "error": "launch_failed:simulated",
            "component": c,
            "api_key": "SECRET",
        },
        classify_fn=lambda: {
            "overall_state": "FULLY_STOPPED",
            "watchdog": {"state": "NOT_RUNNING"},
            "continuous": {"state": "NOT_RUNNING"},
        },
    )
    assert rec["result"] in ("PARTIAL_START", "START_FAILED", "POSTCONDITION_FAILED")
    assert rec.get("paper_arm_mutated") is False
    assert rec.get("live_execution_blocked_after") is True
    rows = read_recent_runtime_actions(path=audit, limit=10)
    blob = json.dumps(rows)
    assert "SECRET" not in blob
    assert "launch_failed:simulated" in blob or rec.get("component_results")


def test_importing_controller_has_no_side_effect_guard():
    import importlib

    from services import operator_runtime_controller as mod

    before = list((Path(mod.ROOT) / "data" / "live").glob("operator_start_action.guard.json"))
    importlib.reload(mod)
    after = list((Path(mod.ROOT) / "data" / "live").glob("operator_start_action.guard.json"))
    assert before == after
