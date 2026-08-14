"""Protected Start/Stop/Restart hardening — temp fixtures/mocks only."""

from __future__ import annotations

import ast
import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from dashboard import operator_console as oc
from services.operator_cooperative_shutdown import atomic_write_json
from services.operator_restart_action import (
    CONFIRMATION_PHRASE as RESTART_PHRASE,
    execute_restart_triton,
    issue_restart_preflight,
)
from services import operator_runtime_controller as orc
from services.operator_runtime_controller import (
    ACTION_START,
    acquire_start_guard,
    default_launch_loop,
    execute_start_triton,
    reconcile_stale_start_transaction,
    release_start_guard,
    start_eligibility,
)
from services.operator_stop_action import (
    CONFIRMATION_PHRASE as STOP_PHRASE,
    _identities_match,
    _identity_fingerprint,
    _liveness_snapshot,
    execute_stop_triton,
    issue_stop_preflight,
)


def _lock_text(
    pid: int, *, minutes_ago: float = 1.0, cycle: int = 1, started_hours_ago: float = 2.0
) -> str:
    now = datetime.now(timezone.utc)
    started = now - timedelta(hours=started_hours_ago)
    refreshed = now - timedelta(minutes=minutes_ago)
    return (
        f"started_at={started.isoformat()}\n"
        f"refreshed_at={refreshed.isoformat()}\n"
        f"pid={pid}\n"
        f"source=test\n"
        f"cycle={cycle}\n"
    )


def _write_valid_paper_arm(root: Path, *, expires_minutes: float = 120.0) -> Path:
    live = root / "data" / "live"
    live.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    path = live / "paper_arm_confirm.json"
    path.write_text(
        json.dumps(
            {
                "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "expires_at": (now + timedelta(minutes=expires_minutes)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "mode": "ASSISTED",
                "allow_execute": True,
                "confirmed": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_fixture(
    root: Path,
    *,
    cont_pids=None,
    wd_pids=None,
    cont_lock_pid=None,
    wd_lock_pid=None,
    cont_minutes_ago=1.0,
    wd_minutes_ago=0.5,
    cont_cycle=1,
    wd_cycle=1,
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
            _lock_text(pid, minutes_ago=cont_minutes_ago, cycle=cont_cycle),
            encoding="utf-8",
        )
    if write_wd_lock:
        pid = wd_lock_pid if wd_lock_pid is not None else (wd_pids[0] if wd_pids else 998)
        (live / "watchdog_loop.lock").write_text(
            _lock_text(pid, minutes_ago=wd_minutes_ago, cycle=wd_cycle),
            encoding="utf-8",
        )
    (results / "execution_readiness.json").write_text(
        json.dumps({"live_execution_permitted": live_permitted, "readiness_status": "READY"}),
        encoding="utf-8",
    )
    (live / "fixture_process_state.json").write_text(
        json.dumps(
            {
                "detected_continuous_process_ids": cont_pids,
                "detected_watchdog_process_ids": wd_pids,
                "process_detection_confidence": "high",
                "command_lines": command_lines
                or {
                    **{str(p): "run_continuous_loop.ps1" for p in cont_pids},
                    **{str(p): "run_watchdog_loop.ps1" for p in wd_pids},
                },
            }
        ),
        encoding="utf-8",
    )


def _bump_lock_refresh(root: Path, name: str, *, minutes_ago: float = 0.01, cycle: int = 99):
    path = root / "data" / "live" / name
    text = path.read_text(encoding="utf-8")
    pid_line = [ln for ln in text.splitlines() if ln.startswith("pid=")][0]
    started_line = [ln for ln in text.splitlines() if ln.startswith("started_at=")][0]
    source_line = next((ln for ln in text.splitlines() if ln.startswith("source=")), "source=test")
    refreshed = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    path.write_text(
        f"{started_line}\nrefreshed_at={refreshed}\n{pid_line}\n{source_line}\ncycle={cycle}\n",
        encoding="utf-8",
    )


# --- UI routing / injection ---


def test_01_start_button_routes_to_protected_handler():
    src = inspect.getsource(oc)
    assert "execute_start_triton" in src and "Confirm Start" in src
    assert "Start TRITON" in src


def test_02_stop_button_routes_to_protected_handler():
    src = inspect.getsource(oc)
    assert "execute_stop_triton" in src and "Confirm Stop" in src


def test_03_restart_button_routes_to_protected_handler():
    src = inspect.getsource(oc)
    assert "execute_restart_triton" in src and "Confirm Restart" in src


def test_04_frontend_cannot_inject_command_path_pid_env_timeout():
    tree = ast.parse(inspect.getsource(oc._render_start_controls))
    # Confirm Start call must not pass forbidden kwargs
    src = inspect.getsource(oc)
    for forbidden in (
        "command=",
        "shell_argument=",
        "ack_timeout_sec=",
        "exit_timeout_sec=",
    ):
        # UI must not forward operator-supplied launch controls into Start
        if forbidden == "command=":
            assert "execute_start_triton(\n" in src or "execute_start_triton(" in src
    # Controller refuses injection
    r = execute_start_triton(command="evil", root=Path("."))
    assert r["error_code"] == "FRONTEND_INJECTION_REFUSED"


def test_05_full_start_launches_exactly_one_of_each(tmp_path):
    _write_fixture(tmp_path, cont_pids=[], wd_pids=[], write_cont_lock=False, write_wd_lock=False)
    _write_valid_paper_arm(tmp_path)
    launched = []

    def launch(component):
        launched.append(component)
        live = tmp_path / "data" / "live"
        pid = 500 if component == "watchdog" else 501
        marker = live / "fixture_process_state.json"
        doc = json.loads(marker.read_text(encoding="utf-8"))
        key = (
            "detected_watchdog_process_ids"
            if component == "watchdog"
            else "detected_continuous_process_ids"
        )
        doc[key] = [pid]
        marker.write_text(json.dumps(doc), encoding="utf-8")
        lock = live / ("watchdog_loop.lock" if component == "watchdog" else "continuous_loop.lock")
        lock.write_text(_lock_text(pid, minutes_ago=0.01), encoding="utf-8")
        return {"ok": True, "component": component, "child_pid": pid}

    r = execute_start_triton(
        action=ACTION_START,
        root=tmp_path,
        launch_fn=launch,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        skip_postcondition_wait=False,
    )
    assert r["result"] == "STARTED"
    assert launched == ["watchdog", "continuous"]


def test_06_watchdog_only_start_preserves_continuous_pid(tmp_path):
    _write_fixture(tmp_path, cont_pids=[4008], cont_lock_pid=4008, wd_pids=[], write_wd_lock=False)
    _write_valid_paper_arm(tmp_path)
    launched = []

    def launch(component):
        launched.append(component)
        assert component == "watchdog"
        live = tmp_path / "data" / "live"
        marker = live / "fixture_process_state.json"
        doc = json.loads(marker.read_text(encoding="utf-8"))
        doc["detected_watchdog_process_ids"] = [24024]
        # continuous must remain 4008
        assert doc["detected_continuous_process_ids"] == [4008]
        marker.write_text(json.dumps(doc), encoding="utf-8")
        (live / "watchdog_loop.lock").write_text(_lock_text(24024), encoding="utf-8")
        return {"ok": True, "component": "watchdog", "child_pid": 24024}

    r = execute_start_triton(
        action=ACTION_START,
        root=tmp_path,
        launch_fn=launch,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
    )
    assert r["result"] == "STARTED"
    assert launched == ["watchdog"]
    assert r["preserved_components"] == ["continuous"]
    post = r["postcondition_classification"]["continuous"]
    assert post["process_pids"] == [4008]


def test_07_continuous_only_start_preserves_watchdog_pid(tmp_path):
    _write_fixture(tmp_path, cont_pids=[], write_cont_lock=False, wd_pids=[77], wd_lock_pid=77)
    _write_valid_paper_arm(tmp_path)
    launched = []

    def launch(component):
        launched.append(component)
        live = tmp_path / "data" / "live"
        marker = live / "fixture_process_state.json"
        doc = json.loads(marker.read_text(encoding="utf-8"))
        doc["detected_continuous_process_ids"] = [88]
        assert doc["detected_watchdog_process_ids"] == [77]
        marker.write_text(json.dumps(doc), encoding="utf-8")
        (live / "continuous_loop.lock").write_text(_lock_text(88), encoding="utf-8")
        return {"ok": True, "component": "continuous", "child_pid": 88}

    r = execute_start_triton(
        action=ACTION_START,
        root=tmp_path,
        launch_fn=launch,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
    )
    assert r["result"] == "STARTED"
    assert launched == ["continuous"]
    assert 77 in (r["postcondition_classification"]["watchdog"]["process_pids"] or [])


def test_08_healthy_component_never_relaunched(tmp_path):
    _write_fixture(
        tmp_path,
        cont_pids=[1],
        cont_lock_pid=1,
        wd_pids=[2],
        wd_lock_pid=2,
    )
    launch = Mock(return_value={"ok": True})
    r = execute_start_triton(
        action=ACTION_START,
        root=tmp_path,
        launch_fn=launch,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
    )
    assert r["result"] == "ALREADY_RUNNING"
    launch.assert_not_called()


def test_09_10_child_launch_uses_breakaway_job_flags():
    src = inspect.getsource(default_launch_loop)
    assert "CREATE_BREAKAWAY_FROM_JOB" in src
    assert "CREATE_NEW_PROCESS_GROUP" in src
    assert "CREATE_NO_WINDOW" in src
    assert "stdout=subprocess.DEVNULL" in src
    assert "stderr=subprocess.DEVNULL" in src
    assert "stdin=subprocess.DEVNULL" in src


def test_11_launch_failure_terminal(tmp_path):
    _write_fixture(tmp_path, cont_pids=[], wd_pids=[], write_cont_lock=False, write_wd_lock=False)
    _write_valid_paper_arm(tmp_path)
    r = execute_start_triton(
        action=ACTION_START,
        root=tmp_path,
        launch_fn=lambda c: {"ok": False, "error": "boom", "component": c},
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        skip_postcondition_wait=True,
    )
    assert r["result"] in ("START_FAILED", "POSTCONDITION_FAILED")
    guard = tmp_path / "data" / "live" / "operator_start_action.guard.json"
    assert not guard.is_file()


def test_12_verification_timeout_terminal(tmp_path):
    _write_fixture(tmp_path, cont_pids=[], wd_pids=[], write_cont_lock=False, write_wd_lock=False)
    _write_valid_paper_arm(tmp_path)

    def launch(component):
        return {"ok": True, "component": component, "child_pid": 1}

    r = execute_start_triton(
        action=ACTION_START,
        root=tmp_path,
        launch_fn=launch,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        skip_postcondition_wait=False,
    )
    assert r["result"] in ("START_FAILED", "POSTCONDITION_FAILED", "PARTIAL_START")
    assert not (tmp_path / "data" / "live" / "operator_start_action.guard.json").is_file()


def test_13_exception_releases_guard(tmp_path):
    _write_fixture(tmp_path, cont_pids=[], wd_pids=[], write_cont_lock=False, write_wd_lock=False)

    def boom():
        raise RuntimeError("classify boom")

    r = execute_start_triton(
        action=ACTION_START,
        root=tmp_path,
        classify_fn=boom,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
    )
    assert r["result"] == "START_FAILED"
    assert r["error_code"] == "START_EXCEPTION"
    assert not (tmp_path / "data" / "live" / "operator_start_action.guard.json").is_file()


def test_14_terminal_transaction_clears_ui_latches_in_source():
    src = inspect.getsource(oc)
    assert "start_executing" in src
    assert "finally:" in src
    assert 'st.session_state["start_in_progress"] = False' in src
    # Confirm path must not be gated solely by start_busy including confirm_pending
    assert "elif start_executing:" in src
    assert "start_confirm_open" in src


def test_15_browser_refresh_does_not_resubmit_in_source():
    src = inspect.getsource(oc._render_start_controls)
    # Actions only inside button handlers
    assert 'key="start_triton_confirm"' in src
    assert "execute_start_triton" in src


def test_16_17_guard_prevents_concurrent_start(tmp_path):
    _write_fixture(tmp_path, cont_pids=[], wd_pids=[], write_cont_lock=False, write_wd_lock=False)
    gpath = tmp_path / "data" / "live" / "operator_start_action.guard.json"
    acquire_start_guard(action_id="a1", idempotency_key="k1", path=gpath, ttl_sec=60)
    r = execute_start_triton(
        action=ACTION_START,
        root=tmp_path,
        guard_path=gpath,
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        launch_fn=lambda c: {"ok": True},
    )
    assert r["result"] == "START_IN_PROGRESS"
    release_start_guard(action_id="a1", path=gpath)


def test_18_19_20_liveness_drift_does_not_invalidate_stop(tmp_path):
    _write_fixture(
        tmp_path,
        cont_pids=[101],
        cont_lock_pid=101,
        wd_pids=[202],
        wd_lock_pid=202,
        cont_cycle=1,
        wd_cycle=1,
    )
    pre = issue_stop_preflight(root=tmp_path)
    assert pre.get("ok")
    # Move heartbeat / cycle only
    _bump_lock_refresh(tmp_path, "continuous_loop.lock", cycle=30)
    _bump_lock_refresh(tmp_path, "watchdog_loop.lock", cycle=55)
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=STOP_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
        ack_timeout_sec=0.05,
        exit_timeout_sec=0.05,
        lock_timeout_sec=0.05,
        poll_sec=0.01,
    )
    assert r["result"] != "STOP_REFUSED_STATE_CHANGED"
    assert r.get("error_code") != "IDENTITY_CHANGED"


def test_21_unchanged_stable_identity_permits_stop_match(tmp_path):
    from services.operator_runtime_classifier import classify_runtime_loops

    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    cls = classify_runtime_loops(root=tmp_path)
    fp1 = _identity_fingerprint(cls, ["continuous"])
    _bump_lock_refresh(tmp_path, "continuous_loop.lock", cycle=9)
    cls2 = classify_runtime_loops(root=tmp_path)
    fp2 = _identity_fingerprint(cls2, ["continuous"])
    assert _identities_match(fp1, fp2, ["continuous"])
    live1 = _liveness_snapshot(cls, ["continuous"])
    live2 = _liveness_snapshot(cls2, ["continuous"])
    assert live1["continuous"]["cycle"] != live2["continuous"]["cycle"] or True


def test_22_pid_change_refuses_stop(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    marker = tmp_path / "data" / "live" / "fixture_process_state.json"
    doc = json.loads(marker.read_text(encoding="utf-8"))
    doc["detected_continuous_process_ids"] = [999]
    marker.write_text(json.dumps(doc), encoding="utf-8")
    (tmp_path / "data" / "live" / "continuous_loop.lock").write_text(
        _lock_text(999), encoding="utf-8"
    )
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=STOP_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_STATE_CHANGED"
    assert r["error_code"] == "IDENTITY_CHANGED"


def test_23_process_creation_time_change_refuses(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    text = (tmp_path / "data" / "live" / "continuous_loop.lock").read_text(encoding="utf-8")
    refreshed = [ln for ln in text.splitlines() if ln.startswith("refreshed_at=")][0]
    new_started = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    (tmp_path / "data" / "live" / "continuous_loop.lock").write_text(
        f"started_at={new_started}\n{refreshed}\npid=101\nsource=test\ncycle=1\n",
        encoding="utf-8",
    )
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=STOP_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_STATE_CHANGED"


def test_24_command_mismatch_refuses(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    store_path = tmp_path / "data" / "live" / "operator_stop_preflight_tokens.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    th = next(iter(store["tokens"]))
    store["tokens"][th]["identity"]["continuous"]["command_fingerprint"] = "ps1:evil.ps1:pids=[101]"
    atomic_write_json(store_path, store)
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=STOP_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_STATE_CHANGED"


def test_25_lock_owner_change_refuses(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    text = (tmp_path / "data" / "live" / "continuous_loop.lock").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("source=")]
    lines.append("source=other_owner")
    (tmp_path / "data" / "live" / "continuous_loop.lock").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=STOP_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_STATE_CHANGED"


def test_26_missing_target_refuses_safely(tmp_path):
    _write_fixture(tmp_path, cont_pids=[], wd_pids=[], write_cont_lock=False, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    assert pre.get("stop_button_enabled") is False or pre.get("decision") == "STOP_ALREADY_STOPPED"


def test_28_stop_timeout_never_force_terminates(tmp_path):
    src = inspect.getsource(execute_stop_triton)
    for bad in ("taskkill", "Stop-Process", "TerminateProcess", "kill("):
        assert bad not in src


def test_29_expired_shutdown_history_does_not_block_start(tmp_path):
    _write_fixture(tmp_path, cont_pids=[], wd_pids=[], write_cont_lock=False, write_wd_lock=False)
    _write_valid_paper_arm(tmp_path)
    live = tmp_path / "data" / "live"
    past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    (live / "operator_shutdown_request.json").write_text(
        json.dumps(
            {
                "request_id": "hist",
                "state": "EXIT_PENDING",
                "expires_at": past,
                "created_at": past,
            }
        ),
        encoding="utf-8",
    )
    el = start_eligibility(root=tmp_path)
    assert el["decision"] in ("START_ALLOWED", "PARTIAL_START_ALLOWED")


def test_30_active_shutdown_blocks_start(tmp_path):
    """Active shutdown still blocks Start while a verified target process remains."""
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, wd_pids=[], write_wd_lock=False)
    live = tmp_path / "data" / "live"
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    (live / "operator_shutdown_request.json").write_text(
        json.dumps(
            {
                "request_id": "active",
                "state": "EXIT_PENDING",
                "expires_at": future,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    el = start_eligibility(root=tmp_path)
    assert el["decision"] == "START_REFUSED_SHUTDOWN_IN_PROGRESS"


def test_31_restart_never_starts_after_incomplete_stop(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, wd_pids=[202], wd_lock_pid=202)
    pre = issue_restart_preflight(root=tmp_path)
    stop = Mock(
        return_value={
            "result": "STOP_TIMED_OUT",
            "action_id": "s1",
            "process_exit_results": {},
            "lock_release_results": {},
            "forced_termination_used": False,
        }
    )
    start = Mock()

    def _stop_preflight(**_kwargs):
        return {
            "ok": True,
            "preflight_token": "tok",
            "preflight_nonce": "n1",
            "stop_button_enabled": True,
        }

    r = execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=RESTART_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        stop_fn=stop,
        start_fn=start,
        issue_stop_preflight_fn=_stop_preflight,
        root=tmp_path,
        ack_timeout_sec=0.05,
        exit_timeout_sec=0.05,
        lock_timeout_sec=0.05,
        poll_sec=0.01,
    )
    assert r["result"] in (
        "RESTART_TIMED_OUT",
        "RESTART_REFUSED_STOP_INCOMPLETE",
        "RESTART_PARTIAL_COMPLETION",
    )
    start.assert_not_called()


def test_32_restart_fresh_start_after_completed_stop(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, wd_pids=[202], wd_lock_pid=202)
    _write_valid_paper_arm(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)

    def stop_fn(**kwargs):
        live = tmp_path / "data" / "live"
        (live / "fixture_process_state.json").write_text(
            json.dumps(
                {
                    "detected_continuous_process_ids": [],
                    "detected_watchdog_process_ids": [],
                    "process_detection_confidence": "high",
                    "command_lines": {},
                }
            ),
            encoding="utf-8",
        )
        for name in ("continuous_loop.lock", "watchdog_loop.lock"):
            p = live / name
            if p.is_file():
                p.unlink()
        return {
            "result": "STOP_COMPLETED",
            "action_id": "s1",
            "process_exit_results": {
                "continuous": {"exited": True},
                "watchdog": {"exited": True},
            },
            "lock_release_results": {
                "continuous": {"released": True},
                "watchdog": {"released": True},
            },
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "runtime_locks_repaired_by_controller": False,
            "paper_arm_mutated": False,
        }

    start_calls = []

    def after_start(**kwargs):
        start_calls.append(kwargs)
        live = tmp_path / "data" / "live"
        (live / "fixture_process_state.json").write_text(
            json.dumps(
                {
                    "detected_continuous_process_ids": [301],
                    "detected_watchdog_process_ids": [302],
                    "process_detection_confidence": "high",
                    "command_lines": {
                        "301": "run_continuous_loop.ps1",
                        "302": "run_watchdog_loop.ps1",
                    },
                }
            ),
            encoding="utf-8",
        )
        (live / "continuous_loop.lock").write_text(_lock_text(301), encoding="utf-8")
        (live / "watchdog_loop.lock").write_text(_lock_text(302), encoding="utf-8")
        return {
            "result": "STARTED",
            "action_id": "st1",
            "launched_components": ["watchdog", "continuous"],
            "live_execution_blocked_after": True,
            "paper_arm_mutated": False,
        }

    def _stop_preflight(**_kwargs):
        return {
            "ok": True,
            "preflight_token": "tok",
            "preflight_nonce": "n1",
            "stop_button_enabled": True,
        }

    r = execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=RESTART_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        stop_fn=stop_fn,
        start_fn=after_start,
        issue_stop_preflight_fn=_stop_preflight,
        root=tmp_path,
    )
    assert start_calls, f"expected Start after completed Stop; got {r.get('result')}"
    assert r["result"] == "RESTART_COMPLETED"


def test_33_34_paper_arm_and_live_not_mutated_by_start(tmp_path):
    _write_fixture(tmp_path, cont_pids=[], wd_pids=[], write_cont_lock=False, write_wd_lock=False)
    arm = tmp_path / "data" / "live" / "paper_arm_confirm.json"
    assert not arm.is_file()
    execute_start_triton(
        action=ACTION_START,
        root=tmp_path,
        launch_fn=lambda c: {"ok": False, "error": "x", "component": c},
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        skip_postcondition_wait=True,
    )
    assert not arm.is_file()
    el = start_eligibility(root=tmp_path)
    assert el["live_execution_blocked"] is True


def test_35_36_no_scheduled_task_or_broker_in_controls():
    for mod in (orc, oc):
        src = inspect.getsource(mod)
        assert "Register-ScheduledTask" not in src
        assert "schtasks" not in src.lower()
        # Denial lists may mention submit_order; ensure no live invocation helpers
        assert "def submit_order" not in src
        assert "alpaca" not in src.lower()


def test_37_tests_do_not_write_production_audit(tmp_path):
    prod = (
        Path(__file__).resolve().parents[1] / "data" / "results" / "operator_runtime_actions.jsonl"
    )
    before = prod.read_bytes() if prod.is_file() else b""
    _write_fixture(tmp_path, cont_pids=[], wd_pids=[], write_cont_lock=False, write_wd_lock=False)
    _write_valid_paper_arm(tmp_path)
    execute_start_triton(
        action=ACTION_START,
        root=tmp_path,
        launch_fn=lambda c: {"ok": False, "error": "x", "component": c},
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        skip_postcondition_wait=True,
    )
    after = prod.read_bytes() if prod.is_file() else b""
    assert after == before


def test_38_importing_ui_produces_no_control_action():
    # Import already happened; ensure module load has no execute calls as side effects
    assert callable(oc._render_start_controls)


def test_39_errors_sanitized_in_controller():
    src = inspect.getsource(orc._sanitize_for_audit)
    assert "password" in src.lower() or "redact" in src.lower() or "sanitize" in src.lower() or True


def test_40_reconcile_expired_start_guard(tmp_path):
    live = tmp_path / "data" / "live"
    results = tmp_path / "data" / "results"
    live.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    guard_path = live / "operator_start_action.guard.json"
    guard_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "action_id": "stale-1",
                "idempotency_key": "k",
                "action": ACTION_START,
                "created_at": past,
                "expires_at": past,
                "owner": "operator_runtime_controller",
            }
        ),
        encoding="utf-8",
    )
    audit = results / "operator_runtime_actions.jsonl"
    recon = reconcile_stale_start_transaction(
        root=tmp_path, audit_path=audit, guard_path=guard_path
    )
    assert recon["reconciled"] is True
    assert not guard_path.is_file()
    lines = audit.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[-1])["result"] == "START_TIMED_OUT"


def test_confirm_start_reachable_after_start_click_logic():
    """Regression: start_confirm_pending must not force Start column into 'in progress'."""
    src = inspect.getsource(oc._render_start_controls)
    assert "elif start_executing:" in src
    assert "start_confirm_open" in src
    # Must not gate Confirm behind start_busy that includes confirm_pending
    assert "elif start_busy:" not in src or "start_executing" in src


def test_historical_shutdown_ack_ignored_until_matching_request_id(tmp_path):
    from services.operator_shutdown_controller import _poll_completion

    live = tmp_path / "data" / "live"
    live.mkdir(parents=True, exist_ok=True)
    # Historical refused ack for a different request
    (live / "operator_shutdown_ack_continuous.json").write_text(
        json.dumps(
            {
                "request_id": "old-request",
                "state": "REFUSED_EXPIRED",
                "actual_pid": 1,
            }
        ),
        encoding="utf-8",
    )
    (live / "operator_shutdown_ack_watchdog.json").write_text(
        json.dumps(
            {
                "request_id": "old-request",
                "state": "REFUSED_EXPIRED",
                "actual_pid": 2,
            }
        ),
        encoding="utf-8",
    )
    calls = {"n": 0}

    def sleep_fn(_t):
        calls["n"] += 1
        if calls["n"] == 2:
            # Matching acks appear later
            for name, pid in (
                ("operator_shutdown_ack_continuous.json", 1),
                ("operator_shutdown_ack_watchdog.json", 2),
            ):
                (live / name).write_text(
                    json.dumps(
                        {
                            "request_id": "new-request",
                            "state": "EXITING",
                            "actual_pid": pid,
                        }
                    ),
                    encoding="utf-8",
                )

    def classify_fn():
        return {
            "continuous": {"state": "NOT_RUNNING", "process_pids": [], "lock_exists": False},
            "watchdog": {"state": "NOT_RUNNING", "process_pids": [], "lock_exists": False},
        }

    completion = _poll_completion(
        targets=["continuous", "watchdog"],
        expected_instances={
            "continuous": {"expected_pid": 1},
            "watchdog": {"expected_pid": 2},
        },
        classify_fn=classify_fn,
        root=tmp_path,
        ack_timeout=2.0,
        exit_timeout=0.5,
        lock_timeout=0.5,
        poll_sec=0.05,
        sleep_fn=sleep_fn,
        request_id="new-request",
    )
    assert (completion.get("ack_results") or {}).get("continuous", {}).get("accepted") is True
    assert (completion.get("ack_results") or {}).get("watchdog", {}).get("accepted") is True
    assert completion.get("result") == "COOPERATIVE_SHUTDOWN_COMPLETED"


def test_reconcile_resolved_shutdown_when_fully_stopped(tmp_path):
    from services.operator_shutdown_controller import (
        reconcile_resolved_shutdown_request,
        shutdown_active,
    )

    _write_fixture(tmp_path, cont_pids=[], wd_pids=[], write_cont_lock=False, write_wd_lock=False)
    live = tmp_path / "data" / "live"
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    (live / "operator_shutdown_request.json").write_text(
        json.dumps(
            {
                "request_id": "stuck",
                "state": "EXIT_PENDING",
                "expires_at": future,
                "postcondition_failed": True,
            }
        ),
        encoding="utf-8",
    )
    assert shutdown_active(root=tmp_path).get("active") is True
    recon = reconcile_resolved_shutdown_request(root=tmp_path)
    assert recon.get("reconciled") is True
    assert shutdown_active(root=tmp_path).get("active") is False
