"""DO-006 protected RESTART_TRITON tests — fixtures/mocks only; no real processes."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dashboard.operator_console import DEFAULT_OPERATOR_PAGE, OPERATOR_PAGES
from services.operator_cooperative_shutdown import atomic_write_json
from services.operator_restart_action import (
    ACTION_RESTART,
    CONFIRMATION_PHRASE,
    _hash_token,
    _load_token_store,
    append_restart_audit,
    controller_has_termination_paths,
    evaluate_restart_decision,
    execute_restart_triton,
    issue_restart_preflight,
    read_recent_restart_actions,
    restart_eligibility,
)

ROOT = Path(__file__).resolve().parents[1]


def _lock_text(pid: int, *, minutes_ago: float = 1.0, cycle: int = 1) -> str:
    now = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return (
        f"started_at={now.isoformat()}\n"
        f"refreshed_at={now.isoformat()}\n"
        f"pid={pid}\n"
        "source=test\n"
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
    now = datetime.now(timezone.utc)
    (live / "paper_arm_confirm.json").write_text(
        json.dumps(
            {
                "timestamp": now.isoformat(),
                "mode": "ASSISTED",
                "expires_at": (now + timedelta(hours=2)).isoformat(),
                "allow_execute": True,
            }
        ),
        encoding="utf-8",
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


def _set_pids(root: Path, *, cont_pids=None, wd_pids=None):
    marker = root / "data" / "live" / "fixture_process_state.json"
    doc = json.loads(marker.read_text(encoding="utf-8"))
    if cont_pids is not None:
        doc["detected_continuous_process_ids"] = list(cont_pids)
        doc["continuous_loop_running"] = bool(cont_pids)
    if wd_pids is not None:
        doc["detected_watchdog_process_ids"] = list(wd_pids)
        doc["watchdog_loop_running"] = bool(wd_pids)
    marker.write_text(json.dumps(doc), encoding="utf-8")


def _started(**_kwargs):
    return {
        "result": "STARTED",
        "action_id": "start-1",
        "launched_components": ["watchdog", "continuous"],
        "paper_arm_mutated": False,
        "live_execution_blocked_before": True,
        "live_execution_blocked_after": True,
        "safe_message": "started with live execution blocked",
    }


def _started_verified(root: Path):
    """Mock Start that plants verified process+lock ownership (DO-003 postconditions)."""

    def start(**_kwargs):
        _write_fixture(root, cont_pids=[501], wd_pids=[502], cont_lock_pid=501, wd_lock_pid=502)
        return _started(**_kwargs)

    return start


def _stop_result(result="STOP_COMPLETED", **extra):
    return {
        "result": result,
        "action_id": "stop-1",
        "request_id": "shutdown-1",
        "forced_termination_used": False,
        "runtime_locks_deleted_by_controller": False,
        "runtime_locks_repaired_by_controller": False,
        "paper_arm_mutated": False,
        "orders_submitted": False,
        "orders_cancelled": False,
        "approvals_created": False,
        "authorizations_created": False,
        "scheduled_task_modified": False,
        "governance_modified": False,
        "lifecycle_modified": False,
        "allocation_modified": False,
        "risk_modified": False,
        **extra,
    }


def _completed_stop(root: Path):
    def stop(**_kwargs):
        _set_pids(root, cont_pids=[], wd_pids=[])
        for name in ("continuous_loop.lock", "watchdog_loop.lock"):
            lock = root / "data" / "live" / name
            if lock.is_file():
                lock.unlink()  # fixture simulates loop-owned lock release
        return _stop_result(
            process_exit_results={
                "continuous": {"exited": True},
                "watchdog": {"exited": True},
            },
            lock_release_results={
                "continuous": {"released": True},
                "watchdog": {"released": True},
            },
        )

    return stop


def _stop_preflight(**_kwargs):
    return {"ok": True, "preflight_token": "stop-token", "preflight_nonce": "stop-nonce"}


def _execute(
    root: Path,
    *,
    stop_fn=None,
    start_fn=None,
    issue_stop_preflight_fn=_stop_preflight,
):
    if start_fn is None:
        start_fn = _started_verified(root)
    pre = issue_restart_preflight(root=root)
    assert pre["ok"]
    return execute_restart_triton(
        action=ACTION_RESTART,
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        stop_fn=stop_fn,
        start_fn=start_fn,
        issue_stop_preflight_fn=issue_stop_preflight_fn,
        root=root,
    )


def _running(root: Path):
    _write_fixture(root, cont_pids=[101], wd_pids=[202], cont_lock_pid=101, wd_lock_pid=202)


def _restart_src() -> str:
    return (ROOT / "services" / "operator_restart_action.py").read_text(encoding="utf-8")


def _dashboard_src() -> str:
    return (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")


def _injection_result(tmp_path: Path, **kwargs):
    _write_fixture(tmp_path)
    return execute_restart_triton(
        action=ACTION_RESTART,
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token="x",
        root=tmp_path,
        **kwargs,
    )


def test_01_unsafe_classifier_disables_restart():
    d = evaluate_restart_decision({"decision": "STOP_REFUSED_UNSAFE_STATE"})
    assert d["decision"] == "RESTART_REFUSED_UNSAFE_STATE" and not d["restart_button_enabled"]


def test_02_pid_mismatch_disables_restart(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=999, write_wd_lock=False)
    assert restart_eligibility(root=tmp_path)["restart_button_enabled"] is False


def test_03_command_mismatch_disables_restart(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, command_lines={101: "evil.ps1"})
    assert restart_eligibility(root=tmp_path)["decision"] == "RESTART_REFUSED_UNSAFE_STATE"


def test_04_process_without_lock_disables_restart(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], write_cont_lock=False, write_wd_lock=False)
    assert restart_eligibility(root=tmp_path)["restart_button_enabled"] is False


def test_05_stale_lock_without_process_disables_restart(tmp_path):
    _write_fixture(tmp_path, cont_lock_pid=101, write_cont_lock=True, write_wd_lock=False)
    assert restart_eligibility(root=tmp_path)["decision"] == "RESTART_REFUSED_UNSAFE_STATE"


def test_06_duplicate_process_disables_restart(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101, 102], cont_lock_pid=101, write_wd_lock=False)
    assert restart_eligibility(root=tmp_path)["restart_button_enabled"] is False


def test_07_ambiguous_classifier_disables_restart():
    d = evaluate_restart_decision(
        {"decision": "STOP_REFUSED_UNSAFE_STATE", "explanation": "ambiguous"}
    )
    assert not d["restart_button_enabled"] and "ambiguous" in d["explanation"]


def test_08_unknown_classifier_disables_restart():
    d = evaluate_restart_decision({"decision": "UNKNOWN"})
    assert d["decision"] == "RESTART_REFUSED_UNSAFE_STATE"


def test_09_live_execution_enabled_disables_restart(tmp_path):
    _write_fixture(tmp_path, live_permitted=True)
    assert restart_eligibility(root=tmp_path)["decision"] == "RESTART_REFUSED_LIVE_SAFETY"


def test_10_start_guard_disables_restart(tmp_path):
    _write_fixture(tmp_path)
    atomic_write_json(
        tmp_path / "data/live/operator_start_action.guard.json",
        {"expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()},
    )
    assert restart_eligibility(root=tmp_path)["decision"] == "RESTART_REFUSED_START_CONFLICT"


def test_11_shutdown_guard_disables_restart(tmp_path):
    _running(tmp_path)
    atomic_write_json(
        tmp_path / "data/live/operator_shutdown_action.guard.json",
        {"expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()},
    )
    assert restart_eligibility(root=tmp_path)["restart_button_enabled"] is False


def test_12_restart_guard_reports_in_progress(tmp_path):
    _write_fixture(tmp_path)
    atomic_write_json(
        tmp_path / "data/live/operator_restart_action.guard.json",
        {"expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()},
    )
    assert restart_eligibility(root=tmp_path)["decision"] == "RESTART_IN_PROGRESS"


def test_13_continuous_only_is_restart_eligible(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    assert restart_eligibility(root=tmp_path)["decision"] == "RESTART_ELIGIBLE_CONTINUOUS_ONLY"


def test_14_watchdog_only_is_restart_eligible(tmp_path):
    _write_fixture(tmp_path, wd_pids=[202], wd_lock_pid=202, write_cont_lock=False)
    assert restart_eligibility(root=tmp_path)["decision"] == "RESTART_ELIGIBLE_WATCHDOG_ONLY"


def test_15_both_running_are_restart_eligible(tmp_path):
    _running(tmp_path)
    assert restart_eligibility(root=tmp_path)["decision"] == "RESTART_ELIGIBLE_RUNNING"


def test_16_verified_unhealthy_is_restart_eligible(tmp_path):
    _write_fixture(
        tmp_path, cont_pids=[101], cont_lock_pid=101, cont_minutes_ago=120, write_wd_lock=False
    )
    el = restart_eligibility(root=tmp_path)
    assert el["restart_button_enabled"] and el["unhealthy_verified"]


def test_17_already_stopped_skips_stop_and_starts(tmp_path):
    _write_fixture(tmp_path)
    stop = Mock()
    r = _execute(tmp_path, stop_fn=stop)
    assert r["result"] == "RESTART_COMPLETED" and r["stop_required"] is False
    stop.assert_not_called()


def test_18_already_stopped_requires_fresh_start_eligibility(tmp_path):
    _write_fixture(tmp_path)
    with patch(
        "services.operator_restart_action.start_eligibility",
        return_value={"decision": "START_REFUSED_UNSAFE_STATE", "start_button_enabled": False},
    ):
        r = _execute(tmp_path, stop_fn=Mock())
    assert r["result"] == "RESTART_REFUSED_START_INELIGIBLE" and r["start_eligible"] is False


def test_19_restart_confirmation_checkbox_required(tmp_path):
    _write_fixture(tmp_path)
    r = execute_restart_triton(
        confirmation_checked=False, confirmation_phrase=CONFIRMATION_PHRASE, root=tmp_path
    )
    assert r["error_code"] == "CHECKBOX_REQUIRED"


def test_20_restart_exact_confirmation_phrase_required(tmp_path):
    _write_fixture(tmp_path)
    r = execute_restart_triton(
        confirmation_checked=True, confirmation_phrase="RESTART TRITON!", root=tmp_path
    )
    assert r["error_code"] == "PHRASE_MISMATCH"


def test_21_restart_lowercase_phrase_refused(tmp_path):
    _write_fixture(tmp_path)
    r = execute_restart_triton(
        confirmation_checked=True, confirmation_phrase="restart triton", root=tmp_path
    )
    assert r["result"] == "RESTART_REFUSED_CONFIRMATION"


def test_22_restart_whitespace_phrase_refused(tmp_path):
    _write_fixture(tmp_path)
    r = execute_restart_triton(
        confirmation_checked=True, confirmation_phrase=" RESTART TRITON", root=tmp_path
    )
    assert r["error_code"] == "PHRASE_MISMATCH"


def test_23_partial_phrase_is_refused(tmp_path):
    """Scenario 23: partial phrase is refused."""
    _write_fixture(tmp_path)
    r = execute_restart_triton(
        confirmation_checked=True, confirmation_phrase="RESTART", root=tmp_path
    )
    assert r["result"] == "RESTART_REFUSED_CONFIRMATION"


def test_23b_stop_phrase_cannot_confirm_restart(tmp_path):
    _write_fixture(tmp_path)
    r = execute_restart_triton(
        confirmation_checked=True, confirmation_phrase="STOP TRITON", root=tmp_path
    )
    assert r["result"] == "RESTART_REFUSED_CONFIRMATION"


def test_24_disabled_control_cannot_submit():
    """Scenario 24: Confirm Restart stays disabled until checkbox + exact phrase + token."""
    src = _dashboard_src()
    restart_fn = "def _render_restart_control_column"
    block = src.split(restart_fn, 1)[1].split("\ndef ", 1)[0]
    assert "can_submit = (" in block
    assert "checked" in block
    assert "phrase == CONFIRMATION_PHRASE" in block
    assert "restart_preflight_token" in block
    assert "disabled=not can_submit" in block
    assert CONFIRMATION_PHRASE == "RESTART TRITON" and ACTION_RESTART == "RESTART_TRITON"


def test_25_frontend_pid_injection_refused(tmp_path):
    assert _injection_result(tmp_path, pid=1)["error_code"] == "FRONTEND_INJECTION_REFUSED"


def test_26_frontend_targets_injection_refused(tmp_path):
    """Scenario 26: frontend cannot supply targets."""
    assert _injection_result(tmp_path, target="BOTH")["error_code"] == "FRONTEND_INJECTION_REFUSED"
    assert (
        _injection_result(tmp_path, targets=["continuous"])["error_code"]
        == "FRONTEND_INJECTION_REFUSED"
    )


def test_27_frontend_path_injection_refused(tmp_path):
    """Scenario 27: frontend cannot supply paths."""
    assert _injection_result(tmp_path, path="C:\\x")["error_code"] == "FRONTEND_INJECTION_REFUSED"


def test_28_frontend_command_injection_refused(tmp_path):
    """Scenario 28: frontend cannot supply commands."""
    assert _injection_result(tmp_path, command="evil")["error_code"] == "FRONTEND_INJECTION_REFUSED"


def test_29_frontend_arguments_injection_refused(tmp_path):
    """Scenario 29: frontend cannot supply arguments."""
    assert (
        _injection_result(tmp_path, arguments=["x"])["error_code"] == "FRONTEND_INJECTION_REFUSED"
    )


def test_30_frontend_environment_injection_refused(tmp_path):
    """Scenario 30: frontend cannot supply environment values."""
    assert (
        _injection_result(tmp_path, environment={"X": "1"})["error_code"]
        == "FRONTEND_INJECTION_REFUSED"
    )


def test_31_frontend_timeout_injection_refused(tmp_path):
    """Scenario 31: frontend cannot supply timeout values."""
    assert _injection_result(tmp_path, timeout=1)["error_code"] == "FRONTEND_INJECTION_REFUSED"


def test_32_frontend_cannot_select_controller_operations(tmp_path):
    """Scenario 32: frontend cannot choose Stop/Start controller operations or targets."""
    for kwargs in (
        {"operation": "STOP_TRITON"},
        {"controller_operation": "START_TRITON"},
        {"stop_operation": "PREPARE_COOPERATIVE_SHUTDOWN"},
        {"start_operation": "START_TRITON"},
        {"selected_operation": "RESTART_TRITON"},
    ):
        r = _injection_result(tmp_path, **kwargs)
        assert r["error_code"] == "FRONTEND_INJECTION_REFUSED", kwargs
    src = _dashboard_src()
    restart_fn = "def _render_restart_control_column"
    block = src.split(restart_fn, 1)[1].split("\ndef ", 1)[0]
    assert "st.selectbox" not in block
    assert "st.multiselect" not in block


def test_33_unknown_frontend_field_refused(tmp_path):
    assert _injection_result(tmp_path, arbitrary=True)["error_code"] == "FRONTEND_INJECTION_REFUSED"


def test_34_restart_preflight_token_is_server_generated(tmp_path):
    _write_fixture(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    assert pre["ok"] and len(pre["preflight_token"]) >= 16


def test_35_restart_token_expires(tmp_path):
    """Scenario 35: token expires."""
    _write_fixture(tmp_path)
    assert issue_restart_preflight(root=tmp_path, ttl_sec=1)["expires_at"]


def test_35b_restart_token_is_stored_only_as_hash(tmp_path):
    _write_fixture(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    store = _load_token_store(tmp_path / "data/live/operator_restart_preflight_tokens.json")
    assert _hash_token(pre["preflight_token"]) in store["tokens"]
    assert pre["preflight_token"] not in json.dumps(store)


def test_36_expired_restart_token_is_refused(tmp_path):
    """Scenario 36: expired token is refused."""
    _write_fixture(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    path = tmp_path / "data/live/operator_restart_preflight_tokens.json"
    store = _load_token_store(path)
    store["tokens"][_hash_token(pre["preflight_token"])]["expires_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    ).isoformat()
    atomic_write_json(path, store)
    r = execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
    )
    assert r["result"] == "RESTART_REFUSED_CONFIRMATION_EXPIRED"


def test_37_restart_token_is_single_use(tmp_path):
    """Scenario 37: token is single-use."""
    _write_fixture(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    r1 = execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce="first",
        start_fn=_started_verified(tmp_path),
        root=tmp_path,
    )
    r2 = execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce="second",
        root=tmp_path,
    )
    assert r1["result"] == "RESTART_COMPLETED" and r2["result"] == "RESTART_REFUSED_REPLAY"


def test_38_replayed_token_is_refused(tmp_path):
    """Scenario 38: replayed token is refused."""
    _write_fixture(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce="n1",
        start_fn=_started_verified(tmp_path),
        root=tmp_path,
    )
    r = execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce="n2",
        root=tmp_path,
    )
    assert r["result"] == "RESTART_REFUSED_REPLAY"


def test_39_restart_token_is_bound_to_restart_action(tmp_path):
    """Scenario 39: token is bound to RESTART_TRITON."""
    _write_fixture(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    path = tmp_path / "data/live/operator_restart_preflight_tokens.json"
    store = _load_token_store(path)
    store["tokens"][_hash_token(pre["preflight_token"])]["action"] = "START_TRITON"
    atomic_write_json(path, store)
    r = execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        root=tmp_path,
    )
    assert r["result"] == "RESTART_REFUSED_CONFIRMATION"


def test_40_restart_token_binds_verified_identities(tmp_path):
    """Scenario 40: token is bound to runtime identities."""
    _running(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    rec = _load_token_store(tmp_path / "data/live/operator_restart_preflight_tokens.json")[
        "tokens"
    ][_hash_token(pre["preflight_token"])]
    assert rec["identity"]["continuous"]["pids"] == [101]
    assert rec["identity"]["watchdog"]["lock_pid"] == 202


def test_41_old_token_cannot_target_replacement_pid(tmp_path):
    """Scenario 41: old token cannot target a replacement PID."""
    _running(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    stop = Mock()
    start = Mock()
    _set_pids(tmp_path, cont_pids=[999], wd_pids=[202])
    (tmp_path / "data/live/continuous_loop.lock").write_text(_lock_text(999), encoding="utf-8")
    r = execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        stop_fn=stop,
        start_fn=start,
        issue_stop_preflight_fn=_stop_preflight,
        root=tmp_path,
    )
    assert r["result"] == "RESTART_REFUSED_STATE_CHANGED"
    stop.assert_not_called()
    start.assert_not_called()


def test_41b_same_nonce_returns_idempotent_result(tmp_path):
    _write_fixture(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    kwargs = dict(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce="same",
        start_fn=_started_verified(tmp_path),
        root=tmp_path,
    )
    r1 = execute_restart_triton(**kwargs)
    r2 = execute_restart_triton(**kwargs)
    assert r2["result"] == r1["result"] and r2["replayed"] is True


def test_42_changed_process_creation_time_refuses(tmp_path):
    """Scenario 42: only process creation time changes; PID/command/lock_pid preserved."""
    _running(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    stop = Mock()
    start = Mock()
    # Preserve PID 101 / lock_pid 101 and command fingerprint; change only started_at.
    old = (tmp_path / "data/live/continuous_loop.lock").read_text(encoding="utf-8")
    assert "pid=101" in old
    refreshed_line = [ln for ln in old.splitlines() if ln.startswith("refreshed_at=")][0]
    new_started = (datetime.now(timezone.utc) - timedelta(hours=9)).isoformat()
    (tmp_path / "data/live/continuous_loop.lock").write_text(
        f"started_at={new_started}\n{refreshed_line}\npid=101\nsource=test\ncycle=1\n",
        encoding="utf-8",
    )
    r = execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        stop_fn=stop,
        start_fn=start,
        issue_stop_preflight_fn=_stop_preflight,
        root=tmp_path,
    )
    assert r["result"] == "RESTART_REFUSED_STATE_CHANGED"
    stop.assert_not_called()
    start.assert_not_called()


def test_43_changed_command_fingerprint_is_refused(tmp_path):
    """Scenario 43: changed command fingerprint refuses Restart."""
    _running(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    stop = Mock()
    start = Mock()
    marker = tmp_path / "data/live/fixture_process_state.json"
    doc = json.loads(marker.read_text(encoding="utf-8"))
    doc["command_lines"] = {"101": "evil.ps1", "202": "run_watchdog_loop.ps1"}
    marker.write_text(json.dumps(doc), encoding="utf-8")
    r = execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        stop_fn=stop,
        start_fn=start,
        root=tmp_path,
    )
    assert r["result"] == "RESTART_REFUSED_STATE_CHANGED"
    stop.assert_not_called()
    start.assert_not_called()


def test_44_changed_lock_identity_is_refused(tmp_path):
    """Scenario 44: changed lock identity refuses Restart."""
    _running(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    stop = Mock()
    start = Mock()
    (tmp_path / "data/live/continuous_loop.lock").write_text(_lock_text(777), encoding="utf-8")
    r = execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        stop_fn=stop,
        start_fn=start,
        root=tmp_path,
    )
    assert r["result"] == "RESTART_REFUSED_STATE_CHANGED"
    stop.assert_not_called()
    start.assert_not_called()


def test_41b_old_token_cannot_target_new_pid(tmp_path):
    """Scenario 41: old token cannot target a replacement PID."""
    _running(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    stop = Mock()
    start = Mock()
    _set_pids(tmp_path, cont_pids=[999])
    (tmp_path / "data/live/continuous_loop.lock").write_text(_lock_text(999), encoding="utf-8")
    r = execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        stop_fn=stop,
        start_fn=start,
        root=tmp_path,
    )
    assert r["result"] == "RESTART_REFUSED_STATE_CHANGED"
    stop.assert_not_called()
    start.assert_not_called()


def test_45_running_restart_invokes_only_protected_stop(tmp_path):
    _running(tmp_path)
    stop = Mock(side_effect=_completed_stop(tmp_path))
    _execute(tmp_path, stop_fn=stop)
    assert stop.call_count == 1 and stop.call_args.kwargs["action"] == "STOP_TRITON"


def test_46_running_restart_issues_fresh_stop_preflight(tmp_path):
    _running(tmp_path)
    issue = Mock(side_effect=_stop_preflight)
    _execute(tmp_path, stop_fn=_completed_stop(tmp_path), issue_stop_preflight_fn=issue)
    issue.assert_called_once()


def test_47_stop_confirmation_uses_do005_phrase(tmp_path):
    _running(tmp_path)
    stop = Mock(side_effect=_completed_stop(tmp_path))
    _execute(tmp_path, stop_fn=stop)
    assert stop.call_args.kwargs["confirmation_phrase"] == "STOP TRITON"


def test_48_restart_cannot_invoke_arbitrary_action(tmp_path):
    _write_fixture(tmp_path)
    r = execute_restart_triton(
        action="FORCE_RESTART",
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token="x",
        root=tmp_path,
    )
    assert r["error_code"] == "ACTION_NOT_ALLOWED"


def test_49_restart_stop_path_is_do005_only():
    src = _restart_src()
    assert "execute_stop_triton" in src and "issue_stop_preflight" in src
    assert "prepare_cooperative_shutdown" not in src


def test_50_double_click_executes_one_restart(tmp_path):
    _write_fixture(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    start = Mock(side_effect=_started_verified(tmp_path))
    kwargs = dict(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce="double",
        start_fn=start,
        root=tmp_path,
    )
    execute_restart_triton(**kwargs)
    r = execute_restart_triton(**kwargs)
    assert start.call_count == 1 and r["replayed"] is True


def test_51_concurrent_tabs_at_most_one_restart_transaction(tmp_path):
    """Scenario 51: two tabs submit valid Restart; at most one Stop and one Start."""
    import threading

    _running(tmp_path)
    pre_a = issue_restart_preflight(root=tmp_path)
    pre_b = issue_restart_preflight(root=tmp_path)
    assert pre_a["preflight_token"] != pre_b["preflight_token"]

    stop_entered = threading.Event()
    release_stop = threading.Event()
    stop = Mock()
    start = Mock()

    def stop_fn(**kwargs):
        stop(kwargs)
        stop_entered.set()
        assert release_stop.wait(timeout=5)
        return _completed_stop(tmp_path)(**kwargs)

    def start_fn(**kwargs):
        start(kwargs)
        return _started_verified(tmp_path)(**kwargs)

    results: dict = {}

    def tab_a():
        results["a"] = execute_restart_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre_a["preflight_token"],
            request_nonce=pre_a["preflight_nonce"],
            stop_fn=stop_fn,
            start_fn=start_fn,
            issue_stop_preflight_fn=_stop_preflight,
            root=tmp_path,
        )

    t = threading.Thread(target=tab_a)
    t.start()
    assert stop_entered.wait(timeout=5)
    results["b"] = execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre_b["preflight_token"],
        request_nonce=pre_b["preflight_nonce"],
        stop_fn=stop_fn,
        start_fn=start_fn,
        issue_stop_preflight_fn=_stop_preflight,
        root=tmp_path,
    )
    release_stop.set()
    t.join(timeout=10)
    assert results["a"]["result"] == "RESTART_COMPLETED"
    assert results["b"]["result"] in (
        "RESTART_IN_PROGRESS",
        "RESTART_REFUSED_UNRESOLVED_TRANSACTION",
        "RESTART_REFUSED_STOP_CONFLICT",
        "RESTART_REFUSED_STATE_CHANGED",
    )
    assert stop.call_count == 1
    assert start.call_count == 1


def test_52_network_retry_is_idempotent(tmp_path):
    _write_fixture(tmp_path)
    pre = issue_restart_preflight(root=tmp_path)
    kwargs = dict(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce="retry",
        start_fn=_started_verified(tmp_path),
        root=tmp_path,
    )
    assert execute_restart_triton(**kwargs)["result"] == execute_restart_triton(**kwargs)["result"]


def test_53_page_refresh_does_not_submit_restart():
    src = _dashboard_src()
    assert src.count("execute_restart_triton(") == 1 and "on_refresh" not in src


def test_54_query_parameters_cannot_submit_restart():
    src = _dashboard_src()
    assert "query_params" not in src
    assert not any(
        isinstance(n, ast.Attribute) and n.attr == "query_params" for n in ast.walk(ast.parse(src))
    )


def test_55_url_fragments_cannot_submit_restart():
    src = _dashboard_src()
    assert "url_fragment" not in src and "location.hash" not in src


def test_56_session_restoration_cannot_auto_submit_restart():
    src = _dashboard_src()
    assert 'session_state["submit_restart"]' not in src
    assert src.count("execute_restart_triton(") == 1


def test_57_page_load_cannot_submit_restart():
    block = _dashboard_src().split("def _render_restart_control_column", 1)[0]
    assert "execute_restart_triton(" not in block


def test_58_diagnostics_polling_cannot_submit_restart():
    src = _dashboard_src()
    block = src[src.index("def render_diagnostics_page") :]
    assert "execute_restart_triton(" not in block


def test_59_ack_alone_cannot_advance_to_start(tmp_path):
    """Scenario 59: Stop acknowledgement alone never advances to Start."""
    _running(tmp_path)
    start = Mock()
    eligibility = Mock()

    def stop_ack_only(**_kwargs):
        # Request acknowledged, but process still active and owner lock remains.
        return _stop_result(
            "STOP_COMPLETED",
            acknowledgement_results={"continuous": {"accepted": True}},
            process_exit_results={"continuous": {"exited": False, "pids": [101]}},
            lock_release_results={"continuous": {"released": False}},
        )

    with patch("services.operator_restart_action.start_eligibility", side_effect=eligibility):
        r = _execute(tmp_path, stop_fn=stop_ack_only, start_fn=start)
    assert r["result"] == "RESTART_REFUSED_STOP_INCOMPLETE"
    start.assert_not_called()
    eligibility.assert_not_called()


def test_60_process_exit_with_stale_lock_cannot_advance(tmp_path):
    """Scenario 60: process exit with stale lock cannot advance to Start."""
    _running(tmp_path)
    start = Mock()
    r = _execute(
        tmp_path,
        stop_fn=lambda **_: _stop_result(
            "STOP_COMPLETED",
            process_exit_results={"continuous": {"exited": True}},
            lock_release_results={"continuous": {"released": False}},
        ),
        start_fn=start,
    )
    assert r["result"] == "RESTART_REFUSED_STOP_INCOMPLETE"
    start.assert_not_called()


def test_61_lock_absence_with_active_process_cannot_advance(tmp_path):
    """Scenario 61: lock absence with active process cannot advance to Start."""
    _running(tmp_path)
    start = Mock()
    r = _execute(
        tmp_path,
        stop_fn=lambda **_: _stop_result(
            "STOP_COMPLETED",
            process_exit_results={"continuous": {"exited": False, "pids": [101]}},
            lock_release_results={"continuous": {"released": True}},
        ),
        start_fn=start,
    )
    assert r["result"] == "RESTART_REFUSED_STOP_INCOMPLETE"
    start.assert_not_called()


def test_62_partial_stop_never_advances_to_start(tmp_path):
    """Scenario 62: partial Stop completion cannot advance to Start."""
    _running(tmp_path)
    start = Mock()
    r = _execute(
        tmp_path, stop_fn=lambda **_: _stop_result("STOP_PARTIAL_COMPLETION"), start_fn=start
    )
    assert r["result"] == "RESTART_PARTIAL_COMPLETION"
    start.assert_not_called()


def test_63_stop_timeout_cannot_advance_to_start(tmp_path):
    """Scenario 63: Stop timeout cannot advance to Start."""
    _running(tmp_path)
    start = Mock()
    r = _execute(tmp_path, stop_fn=lambda **_: _stop_result("STOP_TIMED_OUT"), start_fn=start)
    assert r["result"] == "RESTART_TIMED_OUT"
    start.assert_not_called()


def test_64_stop_audit_failure_cannot_advance_to_start(tmp_path):
    """Scenario 64: Stop audit failure cannot advance to Start."""
    _running(tmp_path)
    start = Mock()
    r = _execute(
        tmp_path,
        stop_fn=lambda **_: _stop_result("STOP_AUDIT_FAILED"),
        start_fn=start,
    )
    assert r["result"] == "RESTART_AUDIT_FAILED"
    assert start.call_count == 0


def test_64b_forced_termination_evidence_blocks_start(tmp_path):
    _running(tmp_path)
    start = Mock()
    stop = lambda **_: _stop_result("STOP_COMPLETED", forced_termination_used=True)
    r = _execute(tmp_path, stop_fn=stop, start_fn=start)
    assert r["result"] == "RESTART_REFUSED_STOP_INCOMPLETE"
    start.assert_not_called()


def test_64c_controller_lock_deletion_evidence_blocks_start(tmp_path):
    _running(tmp_path)
    start = Mock()
    stop = lambda **_: _stop_result("STOP_COMPLETED", runtime_locks_deleted_by_controller=True)
    r = _execute(tmp_path, stop_fn=stop, start_fn=start)
    assert r["result"] == "RESTART_REFUSED_STOP_INCOMPLETE"
    start.assert_not_called()


def test_64d_failed_stop_never_advances_to_start(tmp_path):
    _running(tmp_path)
    start = Mock()
    r = _execute(tmp_path, stop_fn=lambda **_: _stop_result("STOP_REFUSED"), start_fn=start)
    assert r["result"] == "RESTART_REFUSED_STOP_INCOMPLETE"
    start.assert_not_called()


def test_64e_active_process_after_stop_blocks_start(tmp_path):
    _running(tmp_path)
    start = Mock()
    r = _execute(tmp_path, stop_fn=lambda **_: _stop_result(), start_fn=start)
    assert r["result"] == "RESTART_REFUSED_STOP_INCOMPLETE"
    start.assert_not_called()


def _clear_restart_coordination(root: Path) -> None:
    live = root / "data" / "live"
    for name in (
        "operator_restart_action.guard.json",
        "operator_restart_transaction.json",
        "operator_restart_preflight_tokens.json",
        "operator_restart_idempotency.json",
    ):
        path = live / name
        if path.is_file():
            path.unlink()


def test_65_full_stop_requires_exit_and_owner_lock_release(tmp_path):
    """Scenario 65: Start impossible unless both process exit and lock release complete."""
    start = Mock()

    # Exit incomplete, lock released → no Start
    _running(tmp_path)
    r1 = _execute(
        tmp_path,
        stop_fn=lambda **_: _stop_result(
            "STOP_COMPLETED",
            process_exit_results={"continuous": {"exited": False}},
            lock_release_results={"continuous": {"released": True}},
        ),
        start_fn=start,
    )
    assert r1["result"] == "RESTART_REFUSED_STOP_INCOMPLETE"
    start.assert_not_called()

    # Exit complete, lock not released → no Start
    _clear_restart_coordination(tmp_path)
    _running(tmp_path)
    r2 = _execute(
        tmp_path,
        stop_fn=lambda **_: _stop_result(
            "STOP_COMPLETED",
            process_exit_results={"continuous": {"exited": True}},
            lock_release_results={"continuous": {"released": False}},
        ),
        start_fn=start,
    )
    assert r2["result"] == "RESTART_REFUSED_STOP_INCOMPLETE"
    start.assert_not_called()

    # Both complete + owner-cleaned runtime → Start may proceed
    _clear_restart_coordination(tmp_path)
    _running(tmp_path)
    r3 = _execute(tmp_path, stop_fn=_completed_stop(tmp_path), start_fn=_started_verified(tmp_path))
    assert r3["result"] == "RESTART_COMPLETED"


def test_65b_stop_preflight_failure_blocks_start(tmp_path):
    _running(tmp_path)
    start = Mock()
    r = _execute(
        tmp_path,
        stop_fn=Mock(),
        start_fn=start,
        issue_stop_preflight_fn=lambda **_: {"ok": False},
    )
    assert r["result"] == "RESTART_REFUSED_STOP_INCOMPLETE"
    start.assert_not_called()


def test_66_start_uses_fresh_eligibility_after_stop(tmp_path):
    _running(tmp_path)
    with patch(
        "services.operator_restart_action.start_eligibility",
        wraps=__import__(
            "services.operator_restart_action", fromlist=["start_eligibility"]
        ).start_eligibility,
    ) as fresh:
        _execute(tmp_path, stop_fn=_completed_stop(tmp_path))
    assert fresh.call_count == 1


def test_67_start_receives_fresh_idempotency_key(tmp_path):
    _write_fixture(tmp_path)
    start = Mock(side_effect=_started_verified(tmp_path))
    pre = issue_restart_preflight(root=tmp_path)
    execute_restart_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce="restart-nonce",
        start_fn=start,
        root=tmp_path,
    )
    assert start.call_args.kwargs["idempotency_key"] != "restart-nonce"


def test_68_restart_token_is_not_reused_for_start(tmp_path):
    _write_fixture(tmp_path)
    r = _execute(tmp_path)
    assert r["restart_token_reused_for_start"] is False


def test_69_start_action_is_fixed_to_start_triton(tmp_path):
    _write_fixture(tmp_path)
    start = Mock(side_effect=_started_verified(tmp_path))
    _execute(tmp_path, start_fn=start)
    assert start.call_args.kwargs["action"] == "START_TRITON"


def test_70_started_maps_to_restart_completed(tmp_path):
    _write_fixture(tmp_path)
    assert _execute(tmp_path)["result"] == "RESTART_COMPLETED"


def test_71_already_running_maps_to_restart_completed(tmp_path):
    _write_fixture(tmp_path)
    r = _execute(
        tmp_path, start_fn=lambda **_: {"result": "ALREADY_RUNNING", "paper_arm_mutated": False}
    )
    assert r["result"] == "RESTART_COMPLETED"


def test_72_partial_start_maps_to_restart_partial(tmp_path):
    _write_fixture(tmp_path)
    r = _execute(
        tmp_path, start_fn=lambda **_: {"result": "PARTIAL_START", "paper_arm_mutated": False}
    )
    assert r["result"] == "RESTART_PARTIAL_COMPLETION"


def test_73_start_postcondition_failure_maps_to_partial(tmp_path):
    _write_fixture(tmp_path)
    r = _execute(tmp_path, start_fn=lambda **_: {"result": "POSTCONDITION_FAILED"})
    assert r["result"] == "RESTART_PARTIAL_COMPLETION"


def test_74_start_refusal_maps_to_start_ineligible(tmp_path):
    _write_fixture(tmp_path)
    r = _execute(tmp_path, start_fn=lambda **_: {"result": "START_REFUSED_UNSAFE_STATE"})
    assert r["result"] == "RESTART_REFUSED_START_INELIGIBLE"


def test_75_replacement_process_identity_is_verified(tmp_path):
    """Scenario 75: STARTED without a replacement process is not RESTART_COMPLETED."""
    _write_fixture(tmp_path)
    start = Mock()

    def missing_process_start(**_kwargs):
        # Claim STARTED but leave fixture stopped — no replacement PIDs.
        start(_kwargs)
        return {
            "result": "STARTED",
            "action_id": "start-no-proc",
            "launched_components": ["continuous", "watchdog"],
            "paper_arm_mutated": False,
        }

    r = _execute(tmp_path, start_fn=missing_process_start)
    assert r["result"] != "RESTART_COMPLETED"
    assert r["result"] == "RESTART_PARTIAL_COMPLETION"
    assert r.get("start_postcondition_ok") is False
    assert "process_missing" in str(r.get("start_postcondition_reason") or "")
    assert start.call_count == 1


def test_75b_start_failure_maps_to_restart_refused(tmp_path):
    _write_fixture(tmp_path)
    r = _execute(tmp_path, start_fn=lambda **_: {"result": "START_FAILED"})
    assert r["result"] == "RESTART_REFUSED"


def test_76_replacement_lock_ownership_verified(tmp_path):
    """Scenario 76: replacement process with wrong lock ownership is not RESTART_COMPLETED."""
    _write_fixture(tmp_path)
    start = Mock()

    def bad_lock_start(**_kwargs):
        # Process exists, but lock identifies another PID — DO-003 ownership fails.
        _set_pids(tmp_path, cont_pids=[501], wd_pids=[502])
        (tmp_path / "data/live/continuous_loop.lock").write_text(_lock_text(999), encoding="utf-8")
        (tmp_path / "data/live/watchdog_loop.lock").write_text(_lock_text(502), encoding="utf-8")
        start(_kwargs)
        return {
            "result": "STARTED",
            "action_id": "start-bad-lock",
            "launched_components": ["continuous", "watchdog"],
            "paper_arm_mutated": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
        }

    r = _execute(tmp_path, start_fn=bad_lock_start)
    assert r["result"] != "RESTART_COMPLETED"
    assert r["result"] == "RESTART_PARTIAL_COMPLETION"
    assert r.get("start_postcondition_ok") is False
    assert "lock_pid_mismatch" in str(r.get("start_postcondition_reason") or "")
    assert start.call_count == 1  # no duplicate Start retry


def test_76b_running_restart_completes_stop_then_start(tmp_path):
    _running(tmp_path)
    order = []
    stop_impl = _completed_stop(tmp_path)

    def stop(**kwargs):
        order.append("stop")
        return stop_impl(**kwargs)

    def start(**kwargs):
        order.append("start")
        return _started_verified(tmp_path)(**kwargs)

    r = _execute(tmp_path, stop_fn=stop, start_fn=start)
    assert r["result"] == "RESTART_COMPLETED" and order == ["stop", "start"]


def test_77_duplicate_replacement_process_prevents_completion(tmp_path):
    """Scenario 77: duplicate replacement PIDs prevent RESTART_COMPLETED."""
    _write_fixture(tmp_path)
    start = Mock()

    def duplicate_start(**_kwargs):
        _set_pids(tmp_path, cont_pids=[501, 502], wd_pids=[601])
        (tmp_path / "data/live/continuous_loop.lock").write_text(_lock_text(501), encoding="utf-8")
        (tmp_path / "data/live/watchdog_loop.lock").write_text(_lock_text(601), encoding="utf-8")
        start(_kwargs)
        return {
            "result": "STARTED",
            "action_id": "start-dup",
            "launched_components": ["continuous", "watchdog"],
            "paper_arm_mutated": False,
        }

    r = _execute(tmp_path, start_fn=duplicate_start)
    # Duplicate continuous PIDs are unsafe — must not claim restart complete.
    assert r["result"] != "RESTART_COMPLETED"
    assert r["result"] == "RESTART_PARTIAL_COMPLETION"
    assert r.get("start_postcondition_ok") is False
    assert "duplicate_replacement" in str(r.get("start_postcondition_reason") or "")
    assert start.call_count == 1


def test_77b_completion_records_both_transaction_ids(tmp_path):
    _running(tmp_path)
    r = _execute(tmp_path, stop_fn=_completed_stop(tmp_path))
    assert r["stop_action_id"] == "stop-1" and r["start_transaction_id"] == "start-1"


def test_78_completion_records_stage_progression(tmp_path):
    _running(tmp_path)
    r = _execute(tmp_path, stop_fn=_completed_stop(tmp_path))
    stages = [s["stage"] for s in r["stages"]]
    assert "STOP_COMPLETED" in stages and "START_ELIGIBILITY_CONFIRMED" in stages


def test_79_restart_controller_has_no_force_kill_path():
    assert controller_has_termination_paths() is False


def test_80_restart_source_has_no_subprocess_launch():
    src = _restart_src()
    assert "import subprocess" not in src and "Popen(" not in src


def test_81_explicit_start_timeout_maps_to_restart_timed_out(tmp_path):
    """Scenario 81: explicit Start timeout evidence → RESTART_TIMED_OUT; Start called once."""
    _write_fixture(tmp_path)
    start = Mock(
        side_effect=lambda **_: {
            "result": "POSTCONDITION_FAILED",
            "action_id": "start-timeout",
            "error_code": "POSTCONDITION_TIMEOUT",
            "timed_out": True,
            "safe_message": ("Start timed out waiting for verified process/lock postconditions."),
            "launched_components": ["continuous"],
            "component_results": {
                "continuous": {
                    "postcondition": {
                        "ok": False,
                        "reason": "postcondition_timeout",
                        "timed_out": True,
                    }
                }
            },
            "paper_arm_mutated": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
        }
    )
    r = _execute(tmp_path, start_fn=start)
    assert r["result"] != "RESTART_COMPLETED"
    assert r["result"] == "RESTART_TIMED_OUT"
    assert r.get("start_result") == "POSTCONDITION_FAILED"
    assert start.call_count == 1  # typed Start controller called at most once; no auto-retry
    assert r.get("safe_to_retry") is True
    assert r.get("forced_termination_used") is False
    assert r.get("start_transaction_id") == "start-timeout"


def test_81b_ordinary_start_in_progress_is_not_timed_out(tmp_path):
    """Ordinary non-expired START_IN_PROGRESS must not be classified as RESTART_TIMED_OUT."""
    _write_fixture(tmp_path)
    start = Mock(
        side_effect=lambda **_: {
            "result": "START_IN_PROGRESS",
            "action_id": "start-busy",
            "error_code": "START_IN_PROGRESS",
            "safe_message": "A Start action is already in progress. Wait for it to finish.",
            "launched_components": [],
            "paper_arm_mutated": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
            "timed_out": False,
        }
    )
    r = _execute(tmp_path, start_fn=start)
    assert r["result"] != "RESTART_COMPLETED"
    assert r["result"] != "RESTART_TIMED_OUT"
    assert r["result"] == "RESTART_PARTIAL_COMPLETION"
    assert r.get("start_result") == "START_IN_PROGRESS"
    assert start.call_count == 1
    assert r.get("forced_termination_used") is False


def test_81c_restart_source_has_no_os_kill():
    assert "os.kill(" not in _restart_src()


def test_82_restart_never_deletes_runtime_locks(tmp_path):
    """Scenario 82: Restart never deletes runtime locks."""
    src = _restart_src()
    assert ".unlink(" not in src and "os.remove(" not in src and "rmtree" not in src
    _write_fixture(tmp_path)
    r = _execute(tmp_path)
    assert r.get("runtime_locks_deleted_by_controller") is False


def test_82b_restart_source_has_no_taskkill():
    assert "taskkill.exe" not in _restart_src().lower()


def test_83_restart_never_repairs_runtime_locks():
    """Scenario 83: Restart never repairs runtime locks."""
    assert "repair_lock" not in _restart_src()
    assert "Clear-OrphanLoopLock" not in _restart_src()


def test_83b_restart_source_has_no_stop_process():
    assert "Stop-Process -" not in _restart_src()


def test_84_restart_never_invokes_scheduled_task():
    """Scenario 84: Restart never invokes a Scheduled Task."""
    src = _restart_src() + _dashboard_src()
    forbidden = (
        "schtasks",
        "Register-ScheduledTask",
        "Start-ScheduledTask",
        "ScheduledTask",
        "Task Scheduler",
    )
    for term in forbidden:
        assert term not in src
    assert "9:31" not in _restart_src()


def test_85_restart_does_not_create_or_extend_paper_authorization(tmp_path):
    """Scenario 85: Restart does not create or extend paper authorization."""
    _write_fixture(tmp_path)
    r = _execute(tmp_path)
    assert r["paper_arm_mutated"] is False


def test_86_restart_does_not_enable_live_execution(tmp_path):
    """Scenario 86: Restart does not enable live execution."""
    _write_fixture(tmp_path)
    r = _execute(tmp_path)
    assert r["live_execution_blocked_before"] is True
    assert r["live_execution_blocked_after"] is True
    assert r["forced_termination_used"] is False


def test_87_restart_does_not_mutate_paper_arm(tmp_path):
    _write_fixture(tmp_path)
    assert _execute(tmp_path)["paper_arm_mutated"] is False


def test_88_restart_does_not_submit_orders(tmp_path):
    _write_fixture(tmp_path)
    assert _execute(tmp_path)["orders_submitted"] is False


def test_89_restart_does_not_cancel_orders(tmp_path):
    _write_fixture(tmp_path)
    assert _execute(tmp_path)["orders_cancelled"] is False


def test_90_restart_does_not_create_approvals(tmp_path):
    _write_fixture(tmp_path)
    assert _execute(tmp_path)["approvals_created"] is False


def test_91_restart_does_not_create_authorizations(tmp_path):
    _write_fixture(tmp_path)
    assert _execute(tmp_path)["authorizations_created"] is False


def test_92_restart_does_not_modify_governance(tmp_path):
    _write_fixture(tmp_path)
    assert _execute(tmp_path)["governance_modified"] is False


def test_93_restart_does_not_modify_lifecycle_or_allocation(tmp_path):
    _write_fixture(tmp_path)
    r = _execute(tmp_path)
    assert r["lifecycle_modified"] is False and r["allocation_modified"] is False


def test_94_restart_does_not_modify_risk_or_tasks(tmp_path):
    _write_fixture(tmp_path)
    r = _execute(tmp_path)
    assert r["risk_modified"] is False and r["scheduled_task_modified"] is False


def test_95_start_is_disabled_while_restart_active():
    src = _dashboard_src()
    assert "Start unavailable while Stop or Restart is active or unresolved." in src


def test_96_stop_is_disabled_while_restart_active():
    src = _dashboard_src()
    assert "restart_busy" in src and "stop_busy" in src


def test_97_restart_is_disabled_while_start_or_stop_active():
    assert "Restart unavailable while Start or Stop is active." in _dashboard_src()


def test_98_initial_restart_audit_failure_blocks_execution(tmp_path):
    _write_fixture(tmp_path)
    start = Mock()
    with patch(
        "services.operator_restart_action.append_restart_audit",
        return_value={"ok": False, "error": "fail"},
    ):
        r = _execute(tmp_path, start_fn=start)
    assert r["result"] == "RESTART_AUDIT_FAILED"
    start.assert_not_called()


def test_99_mid_restart_audit_failure_blocks_start(tmp_path):
    _running(tmp_path)
    calls = {"n": 0}

    def audit(_record, path=None):
        calls["n"] += 1
        return {"ok": calls["n"] != 2, "path": str(path)}

    start = Mock()
    with patch("services.operator_restart_action.append_restart_audit", side_effect=audit):
        r = _execute(tmp_path, stop_fn=_completed_stop(tmp_path), start_fn=start)
    assert r["result"] == "RESTART_AUDIT_FAILED"
    start.assert_not_called()


def test_100_final_restart_audit_failure_is_reported(tmp_path):
    _write_fixture(tmp_path)
    calls = {"n": 0}

    def audit(_record, path=None):
        calls["n"] += 1
        return {"ok": calls["n"] == 1, "path": str(path)}

    with patch("services.operator_restart_action.append_restart_audit", side_effect=audit):
        r = _execute(tmp_path)
    assert r["result"] == "RESTART_AUDIT_FAILED" and r["error_code"] == "FINAL_AUDIT_FAILED"


def test_101_restart_audit_redacts_secrets(tmp_path):
    path = tmp_path / "restart.jsonl"
    assert append_restart_audit(
        {"preflight_token": "secret", "confirmation_phrase": CONFIRMATION_PHRASE}, path=path
    )["ok"]
    row = read_recent_restart_actions(path=path, limit=1)[0]
    assert row["preflight_token"] == row["confirmation_phrase"] == "<redacted>"


def test_102_operations_explains_two_transaction_consequence():
    """Scenario 102: Operations explains the two-transaction consequence."""
    src = _dashboard_src()
    assert "Restart TRITON Safely" in src and "Confirm Restart" in src
    assert "two separately verified transactions" in src
    assert "cooperative Stop followed" in src
    assert "fresh protected Start" in src


def test_103_operations_shows_stop_and_start_stages_separately():
    """Scenario 103: Operations UI renders Stop and Start stages as separate captions."""
    src = _dashboard_src()
    assert (
        "_render_start_controls("
        in src.split("def render_operations_page", 1)[1].split("\ndef ", 1)[0]
    )
    controls = src.split("def _render_start_controls", 1)[1].split(
        "\ndef _render_stop_control_column", 1
    )[0]
    assert "Restart stages (Stop and Start are separate verified transactions)" in controls
    assert "Stop-stage progress:" in controls
    assert "Verified Stop completion:" in controls
    assert "Fresh Start eligibility:" in controls
    assert "Start-stage progress:" in controls
    assert "Final runtime verification:" in controls
    assert "not claimed complete until Start postconditions pass" in controls


def test_103b_operations_uses_exact_restart_phrase():
    assert "Type exactly: RESTART TRITON" in _dashboard_src()


def test_104_operations_never_claims_immediate_restart():
    """Scenario 104: Operations copy describes cooperative Stop then fresh Start, not immediate restart."""
    src = _dashboard_src()
    ops_page = src.split("def render_operations_page", 1)[1].split("\ndef ", 1)[0]
    controls = src.split("def _render_start_controls", 1)[1].split(
        "\ndef _render_cooperative_shutdown_status", 1
    )[0]
    restart_col = src.split("def _render_restart_control_column", 1)[1].split("\ndef ", 1)[0]
    combined = ops_page + controls + restart_col
    assert "cooperative Stop" in combined or "verified cooperative Stop" in combined
    assert "fresh protected Start" in combined or "fresh Start" in combined
    assert "safe boundary" in combined
    # Restart prose must not claim immediacy or guaranteed completion
    restart_prose = "\n".join(
        ln
        for ln in (controls + restart_col).splitlines()
        if "Restart" in ln or "restart" in ln or "safe boundary" in ln
    )
    assert "immediate restart" not in restart_prose.lower()
    assert "immediately restart" not in restart_prose.lower()
    assert "guaranteed" not in restart_prose.lower()
    assert "does not force-terminate" in combined or "will not force-terminate" in combined
    assert "delete locks" in combined  # as a negative promise
    assert "enable live execution" in combined  # as a negative promise
    assert "force termination" not in restart_prose.lower()
    assert "lock repair" not in combined.lower()
    # Restart control prose must not claim it creates/extends paper authorization.
    # Operations may display paper-authorization *status* as a read-only safety field.
    assert "paper authorization" not in restart_prose.lower()
    assert "create or extend paper" in controls.lower() or "arm paper" in combined
    assert "live enablement" not in combined.lower()
    assert "Repair" not in restart_col
    assert "arm paper" in combined  # negative promise wording


def test_104b_operations_invokes_restart_exactly_once():
    assert _dashboard_src().count("execute_restart_triton(") == 1


def test_105_diagnostics_is_read_only():
    src = _dashboard_src()
    assert "Protected Restart evidence (DO-006)" in src and "Diagnostics is read-only" in src


def test_106_six_page_navigation_remains_intact():
    assert len(OPERATOR_PAGES) == 7 and DEFAULT_OPERATOR_PAGE == "Operations"
    assert "Reports" in OPERATOR_PAGES


def test_107_advanced_legacy_navigation_remains_available():
    assert "Advanced/Legacy Dashboard" in _dashboard_src()


def test_108_operator_console_remains_localhost_only():
    assert "127.0.0.1" in (ROOT / ".streamlit/config.toml").read_text(encoding="utf-8")


def test_109_do005_stop_protections_remain_intact():
    """Scenario 109: DO-005 Stop protections remain intact."""
    from services.operator_stop_action import ACTION_STOP, CONFIRMATION_PHRASE as STOP_PHRASE

    assert ACTION_STOP == "STOP_TRITON" and STOP_PHRASE == "STOP TRITON"
    src = _dashboard_src()
    assert "execute_stop_triton" in src and "Confirm Stop" in src


def test_110_do004_cooperative_shutdown_protections_remain_intact():
    """Scenario 110: DO-004 cooperative-shutdown protections remain intact."""
    from services.operator_cooperative_shutdown import ACTION

    assert ACTION == "REQUEST_COOPERATIVE_SHUTDOWN"
    assert (ROOT / "tests/test_loop_cooperative_shutdown_regression_do004.py").is_file()


def test_111_do003_start_protections_remain_intact():
    """Scenario 111: DO-003 Start protections remain intact."""
    src = _dashboard_src()
    assert "execute_start_triton" in src and "Confirm Start" in src


def test_112_do002_console_protections_remain_intact():
    """Scenario 112: DO-002 console protections remain intact."""
    assert OPERATOR_PAGES[0] == "Operations" and OPERATOR_PAGES[-1] == "Diagnostics"


def test_113_restart_isolated_from_real_runtime_data(tmp_path):
    _write_fixture(tmp_path)
    r = _execute(tmp_path)
    assert str(tmp_path) in str(tmp_path / "data/live/operator_restart_transaction.json")
    assert r["source"] == "operator_console"


def test_114_restart_uses_only_mocked_process_transitions(tmp_path):
    _running(tmp_path)
    r = _execute(tmp_path, stop_fn=_completed_stop(tmp_path), start_fn=_started_verified(tmp_path))
    assert r["result"] == "RESTART_COMPLETED"
    assert controller_has_termination_paths() is False


def test_115_live_execution_remains_blocked_through_restart(tmp_path):
    _write_fixture(tmp_path)
    r = _execute(tmp_path)
    assert r["live_execution_blocked_before"] is True
    assert r["live_execution_blocked_after"] is True


def test_116_final_completion_requires_safe_post_start_state(tmp_path):
    _write_fixture(tmp_path)

    def safe_start(**_kwargs):
        _set_pids(tmp_path, cont_pids=[301], wd_pids=[302])
        live = tmp_path / "data/live"
        (live / "continuous_loop.lock").write_text(_lock_text(301), encoding="utf-8")
        (live / "watchdog_loop.lock").write_text(_lock_text(302), encoding="utf-8")
        return _started()

    r = _execute(tmp_path, start_fn=safe_start)
    states = {c: r["final_runtime_classification"][c]["state"] for c in ("continuous", "watchdog")}
    assert r["result"] == "RESTART_COMPLETED"
    assert states == {"continuous": "HEALTHY_RUNNING", "watchdog": "HEALTHY_RUNNING"}
    assert r["live_execution_blocked_after"] is True
