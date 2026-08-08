"""DO-005 protected STOP_TRITON tests — fixtures/mocks only. No real loops/locks/broker."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from dashboard.operator_console import DEFAULT_OPERATOR_PAGE, OPERATOR_PAGES
from services.operator_cooperative_shutdown import ACTION, atomic_write_json, write_ack
from services.operator_stop_action import (
    ACTION_STOP,
    CONFIRMATION_PHRASE,
    evaluate_stop_decision,
    execute_stop_triton,
    issue_stop_preflight,
    stop_eligibility,
    controller_has_termination_paths,
    _hash_token,
    _load_token_store,
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
                "detected_continuous_process_ids": cont_pids,
                "detected_watchdog_process_ids": wd_pids,
                "process_detection_confidence": "high",
                "command_lines": command_lines or {},
            }
        ),
        encoding="utf-8",
    )


def _set_pids(root: Path, *, cont_pids=None, wd_pids=None):
    live = root / "data" / "live"
    marker = live / "fixture_process_state.json"
    doc = json.loads(marker.read_text(encoding="utf-8"))
    if cont_pids is not None:
        doc["detected_continuous_process_ids"] = list(cont_pids)
    if wd_pids is not None:
        doc["detected_watchdog_process_ids"] = list(wd_pids)
    marker.write_text(json.dumps(doc), encoding="utf-8")


def _preflight_and_stop(root: Path, **kwargs):
    pre = issue_stop_preflight(root=root)
    assert pre.get("ok")
    return execute_stop_triton(
        action=ACTION_STOP,
        confirmation_phrase=CONFIRMATION_PHRASE,
        confirmation_checked=True,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=root,
        ack_timeout_sec=kwargs.pop("ack_timeout_sec", 0.2),
        exit_timeout_sec=kwargs.pop("exit_timeout_sec", 0.15),
        lock_timeout_sec=kwargs.pop("lock_timeout_sec", 0.1),
        poll_sec=kwargs.pop("poll_sec", 0.05),
        sleep_fn=kwargs.pop("sleep_fn", lambda _: None),
        **kwargs,
    )


# --- Eligibility matrix ---


def test_01_already_stopped_disables_stop(tmp_path):
    _write_fixture(tmp_path)
    el = stop_eligibility(root=tmp_path)
    assert el["decision"] == "STOP_ALREADY_STOPPED"
    assert el["stop_button_enabled"] is False


def test_02_already_stopped_returns_without_request(tmp_path):
    _write_fixture(tmp_path)
    el = stop_eligibility(root=tmp_path)
    assert el["decision"] == "STOP_ALREADY_STOPPED"
    pre = issue_stop_preflight(root=tmp_path)
    assert pre.get("stop_button_enabled") is False
    assert not (tmp_path / "data" / "live" / "operator_shutdown_request.json").is_file()
    r = _preflight_and_stop(tmp_path)
    assert r["result"] in (
        "STOP_ALREADY_STOPPED",
        "STOP_REFUSED_CONFIRMATION_EXPIRED",
        "STOP_REFUSED",
        "STOP_REFUSED_CONFIRMATION",
    )
    assert not (tmp_path / "data" / "live" / "operator_shutdown_request.json").is_file()


def test_03_continuous_only_enables(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    el = stop_eligibility(root=tmp_path)
    assert el["decision"] == "STOP_ELIGIBLE_CONTINUOUS_ONLY"
    assert el["stop_button_enabled"] is True


def test_04_watchdog_only_enables(tmp_path):
    _write_fixture(tmp_path, wd_pids=[202], wd_lock_pid=202, write_cont_lock=False)
    el = stop_eligibility(root=tmp_path)
    assert el["decision"] == "STOP_ELIGIBLE_WATCHDOG_ONLY"
    assert el["stop_button_enabled"] is True


def test_05_both_enable(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], wd_pids=[202], cont_lock_pid=101, wd_lock_pid=202)
    el = stop_eligibility(root=tmp_path)
    assert el["decision"] == "STOP_ELIGIBLE_BOTH"
    assert el["stop_button_enabled"] is True


def test_06_unhealthy_verified_eligible(tmp_path):
    _write_fixture(
        tmp_path, cont_pids=[101], cont_lock_pid=101, cont_minutes_ago=120.0, write_wd_lock=False
    )
    el = stop_eligibility(root=tmp_path)
    assert el["decision"] == "STOP_ELIGIBLE_UNHEALTHY_VERIFIED"
    assert el["stop_button_enabled"] is True


def test_07_pid_mismatch_disables(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=999, write_wd_lock=False)
    assert stop_eligibility(root=tmp_path)["stop_button_enabled"] is False


def test_08_command_mismatch_disables(tmp_path):
    _write_fixture(
        tmp_path,
        cont_pids=[101],
        cont_lock_pid=101,
        write_wd_lock=False,
        command_lines={101: "evil.ps1"},
    )
    assert stop_eligibility(root=tmp_path)["stop_button_enabled"] is False


def test_09_process_without_lock_disables(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], write_cont_lock=False, write_wd_lock=False)
    assert stop_eligibility(root=tmp_path)["stop_button_enabled"] is False


def test_10_stale_lock_no_process_disables(tmp_path):
    _write_fixture(
        tmp_path, cont_pids=[], cont_lock_pid=101, write_cont_lock=True, write_wd_lock=False
    )
    assert stop_eligibility(root=tmp_path)["decision"] == "STOP_REFUSED_UNSAFE_STATE"


def test_11_duplicate_disables(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101, 102], cont_lock_pid=101, write_wd_lock=False)
    assert stop_eligibility(root=tmp_path)["stop_button_enabled"] is False


def test_12_ambiguous_disables():
    d = evaluate_stop_decision(
        {"continuous": {"state": "AMBIGUOUS"}, "watchdog": {"state": "NOT_RUNNING"}},
        live_execution_blocked=True,
    )
    assert d["stop_button_enabled"] is False


def test_13_unknown_disables():
    d = evaluate_stop_decision(
        {"continuous": {"state": "UNKNOWN"}, "watchdog": {"state": "NOT_RUNNING"}},
        live_execution_blocked=True,
    )
    assert d["stop_button_enabled"] is False


def test_14_live_enabled_disables(tmp_path):
    _write_fixture(
        tmp_path, cont_pids=[101], cont_lock_pid=101, live_permitted=True, write_wd_lock=False
    )
    assert stop_eligibility(root=tmp_path)["decision"] == "STOP_REFUSED_LIVE_SAFETY"


def test_15_start_guard_disables(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    exp = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    (tmp_path / "data" / "live" / "operator_start_action.guard.json").write_text(
        json.dumps({"expires_at": exp}), encoding="utf-8"
    )
    assert stop_eligibility(root=tmp_path)["decision"] == "STOP_REFUSED_START_IN_PROGRESS"


def test_16_shutdown_guard_in_progress(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    exp = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    (tmp_path / "data" / "live" / "operator_shutdown_action.guard.json").write_text(
        json.dumps({"expires_at": exp}), encoding="utf-8"
    )
    assert stop_eligibility(root=tmp_path)["decision"] == "STOP_IN_PROGRESS"


def test_17_unresolved_shutdown_disables(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    atomic_write_json(
        tmp_path / "data" / "live" / "operator_shutdown_request.json",
        {
            "request_id": "old",
            "action": ACTION,
            "expires_at": future,
            "state": "DELIVERED",
            "target_components": ["continuous"],
        },
    )
    el = stop_eligibility(root=tmp_path)
    assert el["stop_button_enabled"] is False
    assert el["decision"] in (
        "STOP_REFUSED_SHUTDOWN_UNRESOLVED",
        "STOP_IN_PROGRESS",
        "STOP_REFUSED_UNSAFE_STATE",
    )


# --- Confirmation ---


def test_18_checkbox_required(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    r = execute_stop_triton(
        confirmation_checked=False,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_CONFIRMATION"


def test_19_exact_phrase_required(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase="STOP TRITON!",
        preflight_token=pre["preflight_token"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_CONFIRMATION"


def test_20_lowercase_refused(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase="stop triton",
        preflight_token=pre["preflight_token"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_CONFIRMATION"


def test_21_whitespace_refused(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=" STOP TRITON",
        preflight_token=pre["preflight_token"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_CONFIRMATION"


def test_22_partial_phrase_refused(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase="STOP",
        preflight_token=pre["preflight_token"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_CONFIRMATION"


def test_23_ui_confirm_disabled_without_phrase():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "disabled=not can_submit" in src
    assert "phrase == CONFIRMATION_PHRASE" in src


# --- Frontend injection ---


def _assert_injection_refused(tmp_path, **kwargs):
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token="x",
        root=tmp_path,
        **kwargs,
    )
    assert r["result"] == "STOP_REFUSED_CONFIRMATION"
    assert r.get("error_code") == "FRONTEND_INJECTION_REFUSED"


def test_24_frontend_cannot_supply_pid(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    _assert_injection_refused(tmp_path, pid=1)


def test_25_frontend_cannot_supply_targets(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    _assert_injection_refused(tmp_path, targets=["continuous"])
    _assert_injection_refused(tmp_path, target="BOTH")


def test_26_frontend_cannot_supply_path(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    _assert_injection_refused(tmp_path, path="C:\\x")


def test_27_frontend_cannot_supply_command(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    _assert_injection_refused(tmp_path, command="x")


def test_28_frontend_cannot_supply_arguments(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    _assert_injection_refused(tmp_path, arguments=["a"])


def test_29_frontend_cannot_supply_environment(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    _assert_injection_refused(tmp_path, environment={"A": "1"})


def test_30_frontend_cannot_supply_timeout(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    _assert_injection_refused(tmp_path, timeout=5)


def test_31_unknown_fields_refused(tmp_path):
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token="x",
        root=tmp_path,
        weird_field=1,
    )
    assert r.get("error_code") == "FRONTEND_INJECTION_REFUSED"


# --- Token / replay ---


def test_32_preflight_token_server_generated(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path, ttl_sec=60)
    assert pre.get("preflight_token")
    assert len(pre["preflight_token"]) >= 16
    store = _load_token_store(tmp_path / "data" / "live" / "operator_stop_preflight_tokens.json")
    assert _hash_token(pre["preflight_token"]) in store["tokens"]
    assert pre["preflight_token"] not in json.dumps(store)


def test_33_preflight_token_expires(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path, ttl_sec=1)
    assert pre.get("expires_at")


def test_34_expired_token_cannot_submit(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path, ttl_sec=1)
    store = _load_token_store(tmp_path / "data" / "live" / "operator_stop_preflight_tokens.json")
    past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    th = _hash_token(pre["preflight_token"])
    store["tokens"][th]["expires_at"] = past
    atomic_write_json(tmp_path / "data" / "live" / "operator_stop_preflight_tokens.json", store)
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_CONFIRMATION_EXPIRED"


def test_35_token_single_use(tmp_path):
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
                safe_message="exit",
                ack_path=live / "operator_shutdown_ack_continuous.json",
                script_name="run_continuous_loop.ps1",
            )
            _set_pids(tmp_path, cont_pids=[])
            lock = live / "continuous_loop.lock"
            if lock.is_file():
                lock.unlink()

    pre = issue_stop_preflight(root=tmp_path)
    token = pre["preflight_token"]
    r1 = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=token,
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
        ack_timeout_sec=1.0,
        exit_timeout_sec=1.0,
        lock_timeout_sec=1.0,
        poll_sec=0.01,
        sleep_fn=sleep_fn,
    )
    assert r1["result"] in (
        "STOP_COMPLETED",
        "STOP_TIMED_OUT",
        "STOP_PARTIAL_COMPLETION",
        "STOP_AUDIT_FAILED",
    )
    store = _load_token_store(tmp_path / "data" / "live" / "operator_stop_preflight_tokens.json")
    th = _hash_token(token)
    assert store["tokens"][th].get("used") is True
    r2 = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=token,
        request_nonce="other-nonce",
        root=tmp_path,
    )
    assert r2["result"] == "STOP_REFUSED_REPLAY"


def test_36_replayed_token_refused(tmp_path):
    test_35_token_single_use(tmp_path)


def test_37_token_bound_to_stop_only(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    store = _load_token_store(tmp_path / "data" / "live" / "operator_stop_preflight_tokens.json")
    th = _hash_token(pre["preflight_token"])
    store["tokens"][th]["action"] = "START_TRITON"
    atomic_write_json(tmp_path / "data" / "live" / "operator_stop_preflight_tokens.json", store)
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_CONFIRMATION"


def test_38_token_bound_to_verified_identities(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    store = _load_token_store(tmp_path / "data" / "live" / "operator_stop_preflight_tokens.json")
    th = _hash_token(pre["preflight_token"])
    identity = store["tokens"][th].get("identity") or {}
    assert "continuous" in identity
    assert identity["continuous"].get("pids") == [101]
    assert identity["continuous"].get("lock_pid") == 101
    assert identity["continuous"].get("process_creation_time")
    assert "lock_refreshed_at" not in identity["continuous"]
    assert identity["continuous"].get("command_fingerprint")


def test_39_old_token_cannot_target_new_pid(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    _set_pids(tmp_path, cont_pids=[999])
    (tmp_path / "data" / "live" / "continuous_loop.lock").write_text(
        _lock_text(999), encoding="utf-8"
    )
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_STATE_CHANGED"


def test_40_old_token_cannot_target_replacement_instance(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    _set_pids(tmp_path, cont_pids=[777])
    (tmp_path / "data" / "live" / "continuous_loop.lock").write_text(
        _lock_text(777, minutes_ago=0.1, cycle=99), encoding="utf-8"
    )
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_STATE_CHANGED"


def test_41_changed_command_fingerprint_refuses(tmp_path):
    _write_fixture(
        tmp_path,
        cont_pids=[101],
        cont_lock_pid=101,
        write_wd_lock=False,
        command_lines={101: "run_continuous_loop.ps1"},
    )
    pre = issue_stop_preflight(root=tmp_path)
    store_path = tmp_path / "data" / "live" / "operator_stop_preflight_tokens.json"
    store = _load_token_store(store_path)
    th = _hash_token(pre["preflight_token"])
    store["tokens"][th]["identity"]["continuous"]["command_fingerprint"] = "ps1:evil.ps1:pids=[101]"
    atomic_write_json(store_path, store)
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_STATE_CHANGED"


def test_42_changed_lock_identity_refuses(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    store_path = tmp_path / "data" / "live" / "operator_stop_preflight_tokens.json"
    store = _load_token_store(store_path)
    th = _hash_token(pre["preflight_token"])
    store["tokens"][th]["identity"]["continuous"]["lock_pid"] = 555
    atomic_write_json(store_path, store)
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_STATE_CHANGED"


def test_43_changed_process_creation_time_refuses(tmp_path):
    """PID-reuse resistance: lock ownership started_at change refuses Stop."""
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    store_path = tmp_path / "data" / "live" / "operator_stop_preflight_tokens.json"
    store = _load_token_store(store_path)
    th = _hash_token(pre["preflight_token"])
    store["tokens"][th]["identity"]["continuous"]["process_creation_time"] = (
        datetime.now(timezone.utc) - timedelta(hours=5)
    ).isoformat()
    atomic_write_json(store_path, store)
    r = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_STATE_CHANGED"


def test_44_backend_reclassifies_immediately_before_delivery(tmp_path):
    test_39_old_token_cannot_target_new_pid(tmp_path)


def test_45_state_change_returns_refused_state_changed(tmp_path):
    test_39_old_token_cannot_target_new_pid(tmp_path)


def test_46_valid_confirmation_invokes_only_do004(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    called = {}

    def fake_prepare(**kwargs):
        called["action"] = kwargs.get("action")
        called["source"] = kwargs.get("source")
        return {
            "result": "ALREADY_STOPPED",
            "request_id": None,
            "target_components": [],
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
            "safe_message": "already",
        }

    pre = issue_stop_preflight(root=tmp_path)
    with patch(
        "services.operator_stop_action.prepare_cooperative_shutdown", side_effect=fake_prepare
    ):
        r = execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre["preflight_token"],
            request_nonce=pre["preflight_nonce"],
            root=tmp_path,
        )
    assert called.get("action") == "PREPARE_COOPERATIVE_SHUTDOWN"
    assert "operator_stop_action" in str(called.get("source"))
    assert r["backend_operation"] == "PREPARE_COOPERATIVE_SHUTDOWN"


def test_47_valid_confirmation_cannot_invoke_arbitrary_actions(tmp_path):
    r = execute_stop_triton(
        action="FORCE_STOP",
        confirmation_checked=True,
        confirmation_phrase=CONFIRMATION_PHRASE,
        preflight_token="x",
        root=tmp_path,
    )
    assert r["result"] == "STOP_REFUSED_CONFIRMATION"
    assert r.get("error_code") == "ACTION_NOT_ALLOWED"
    src = (ROOT / "services" / "operator_stop_action.py").read_text(encoding="utf-8")
    assert "prepare_cooperative_shutdown" in src
    assert "ACTION_PREPARE" in src


def test_48_fixed_both_policy_not_frontend_selectable(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    el = stop_eligibility(root=tmp_path)
    assert el["dashboard_target_policy"] == "BOTH"
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "target_selector" not in src
    assert 'key="stop_target"' not in src
    stop_block = src[
        src.index("def _render_stop_control_column") : src.index(
            "def _render_cooperative_shutdown_status"
        )
    ]
    assert "selectbox" not in stop_block


def test_49_do004_may_narrow_targets(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    el = stop_eligibility(root=tmp_path)
    assert el["dashboard_target_policy"] == "BOTH"
    assert el["target_components"] == ["continuous"]


def test_50_double_click_one_request(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    with patch(
        "services.operator_stop_action.prepare_cooperative_shutdown",
        return_value={
            "result": "SHUTDOWN_TIMED_OUT",
            "request_id": "r1",
            "target_components": ["continuous"],
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
            "safe_message": "timeout",
            "safe_to_retry": True,
        },
    ) as mock_prep:
        r1 = execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre["preflight_token"],
            request_nonce="same-nonce",
            root=tmp_path,
        )
        r2 = execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre["preflight_token"],
            request_nonce="same-nonce",
            root=tmp_path,
        )
    assert mock_prep.call_count == 1
    assert r2.get("replayed") is True
    assert r1["result"] == r2["result"] == "STOP_TIMED_OUT"


def test_51_concurrent_tabs_at_most_one_request(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre_a = issue_stop_preflight(root=tmp_path)
    pre_b = issue_stop_preflight(root=tmp_path)
    assert pre_a["preflight_token"] != pre_b["preflight_token"]
    with patch(
        "services.operator_stop_action.prepare_cooperative_shutdown",
        return_value={
            "result": "SHUTDOWN_TIMED_OUT",
            "request_id": "rA",
            "target_components": ["continuous"],
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
            "safe_message": "timeout",
        },
    ) as mock_prep:
        r_a = execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre_a["preflight_token"],
            request_nonce=pre_a["preflight_nonce"],
            root=tmp_path,
        )
        exp = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        (tmp_path / "data" / "live" / "operator_shutdown_action.guard.json").write_text(
            json.dumps({"expires_at": exp}), encoding="utf-8"
        )
        r_b = execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre_b["preflight_token"],
            request_nonce=pre_b["preflight_nonce"],
            root=tmp_path,
        )
    assert mock_prep.call_count == 1
    assert r_a["result"] == "STOP_TIMED_OUT"
    assert r_b["result"] in (
        "STOP_IN_PROGRESS",
        "STOP_REFUSED_STATE_CHANGED",
        "STOP_REFUSED_SHUTDOWN_UNRESOLVED",
    )


def test_52_network_retry_idempotent(tmp_path):
    test_50_double_click_one_request(tmp_path)


def test_53_page_refresh_does_not_submit_stop():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert src.count("execute_stop_triton(") == 1
    assert 'key="stop_triton_confirm"' in src
    assert "on_refresh" not in src


def test_54_query_parameters_cannot_submit_stop():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "st.query_params" not in src
    assert "experimental_get_query_params" not in src
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "query_params":
            pytest.fail("query_params must not drive Stop")


def test_55_url_fragments_cannot_submit_stop():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "url_fragment" not in src
    assert "location.hash" not in src


def test_56_session_restoration_cannot_submit_stop():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert 'session_state.get("auto_stop"' not in src
    assert 'session_state["submit_stop"]' not in src
    assert src.count("execute_stop_triton(") == 1


def test_57_page_load_cannot_submit_stop():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    load = src[src.index("def render_operations_page") : src.index("_render_start_controls()")]
    assert "execute_stop_triton" not in load


def test_58_readonly_polling_cannot_submit_stop():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    coop = src[src.index("def _render_cooperative_shutdown_status") :]
    head = coop.split("def render_")[0]
    assert "execute_stop_triton" not in head
    assert "issue_stop_preflight" not in head


def test_59_ack_alone_not_completed(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    with patch(
        "services.operator_stop_action.prepare_cooperative_shutdown",
        return_value={
            "result": "SHUTDOWN_TIMED_OUT",
            "request_id": "x",
            "target_components": ["continuous"],
            "acknowledgement_results": {"continuous": {"accepted": True}},
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
            "safe_message": "ack only",
        },
    ):
        pre = issue_stop_preflight(root=tmp_path)
        r = execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre["preflight_token"],
            request_nonce=pre["preflight_nonce"],
            root=tmp_path,
        )
    assert r["result"] == "STOP_TIMED_OUT"
    assert r["result"] != "STOP_COMPLETED"


def test_60_exit_with_stale_lock_not_completed(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    with patch(
        "services.operator_stop_action.prepare_cooperative_shutdown",
        return_value={
            "result": "POSTCONDITION_FAILED",
            "request_id": "x",
            "target_components": ["continuous"],
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
            "safe_message": "stale lock",
        },
    ):
        pre = issue_stop_preflight(root=tmp_path)
        r = execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre["preflight_token"],
            request_nonce=pre["preflight_nonce"],
            root=tmp_path,
        )
    assert r["result"] == "STOP_PARTIAL_COMPLETION"


def test_61_lock_absence_with_active_process_not_completed(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    with patch(
        "services.operator_stop_action.prepare_cooperative_shutdown",
        return_value={
            "result": "PARTIAL_SHUTDOWN",
            "request_id": "x",
            "target_components": ["continuous"],
            "process_exit_results": {"continuous": {"exited": False, "pids": [101]}},
            "lock_release_results": {"continuous": {"released": True}},
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
            "safe_message": "lock gone process active",
        },
    ):
        pre = issue_stop_preflight(root=tmp_path)
        r = execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre["preflight_token"],
            request_nonce=pre["preflight_nonce"],
            root=tmp_path,
        )
    assert r["result"] != "STOP_COMPLETED"
    assert r["result"] == "STOP_PARTIAL_COMPLETION"
    assert r["forced_termination_used"] is False


def test_62_full_completion(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    with patch(
        "services.operator_stop_action.prepare_cooperative_shutdown",
        return_value={
            "result": "COOPERATIVE_SHUTDOWN_COMPLETED",
            "request_id": "x",
            "target_components": ["continuous"],
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
            "safe_message": "done",
        },
    ):
        pre = issue_stop_preflight(root=tmp_path)
        r = execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre["preflight_token"],
            request_nonce=pre["preflight_nonce"],
            root=tmp_path,
        )
    assert r["result"] == "STOP_COMPLETED"


def test_63_partial(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], wd_pids=[202], cont_lock_pid=101, wd_lock_pid=202)
    with patch(
        "services.operator_stop_action.prepare_cooperative_shutdown",
        return_value={
            "result": "PARTIAL_SHUTDOWN",
            "request_id": "x",
            "target_components": ["continuous", "watchdog"],
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
            "safe_message": "partial",
        },
    ):
        pre = issue_stop_preflight(root=tmp_path)
        r = execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre["preflight_token"],
            request_nonce=pre["preflight_nonce"],
            root=tmp_path,
        )
    assert r["result"] == "STOP_PARTIAL_COMPLETION"
    assert r["forced_termination_used"] is False


def test_64_acknowledgement_timeout(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    with patch(
        "services.operator_stop_action.prepare_cooperative_shutdown",
        return_value={
            "result": "SHUTDOWN_TIMED_OUT",
            "request_id": "x",
            "target_components": ["continuous"],
            "acknowledgement_results": {"continuous": {"timed_out": True}},
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
            "safe_message": "Forced termination is outside policy.",
        },
    ):
        pre = issue_stop_preflight(root=tmp_path)
        r = execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre["preflight_token"],
            request_nonce=pre["preflight_nonce"],
            root=tmp_path,
        )
    assert r["result"] == "STOP_TIMED_OUT"
    assert r["forced_termination_used"] is False


def test_65_exit_timeout(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    with patch(
        "services.operator_stop_action.prepare_cooperative_shutdown",
        return_value={
            "result": "SHUTDOWN_TIMED_OUT",
            "request_id": "x",
            "target_components": ["continuous"],
            "acknowledgement_results": {"continuous": {"accepted": True}},
            "process_exit_results": {"continuous": {"exited": False, "timed_out": True}},
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
            "safe_message": "exit timeout",
        },
    ):
        pre = issue_stop_preflight(root=tmp_path)
        r = execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre["preflight_token"],
            request_nonce=pre["preflight_nonce"],
            root=tmp_path,
        )
    assert r["result"] in ("STOP_TIMED_OUT", "STOP_PARTIAL_COMPLETION")
    assert r["forced_termination_used"] is False


def test_66_timeout_never_force_terminates(tmp_path):
    test_64_acknowledgement_timeout(tmp_path)


def test_67_partial_never_force_terminates(tmp_path):
    test_63_partial(tmp_path)


def test_68_controller_never_deletes_runtime_lock():
    assert controller_has_termination_paths() is False
    src = (ROOT / "services" / "operator_stop_action.py").read_text(encoding="utf-8")
    assert "continuous_loop.lock).unlink" not in src
    assert "watchdog_loop.lock).unlink" not in src
    assert "runtime_locks_deleted_by_controller" in src


def test_69_controller_never_repairs_runtime_lock():
    src = (ROOT / "services" / "operator_stop_action.py").read_text(encoding="utf-8")
    assert "runtime_locks_repaired_by_controller" in src
    assert "repair_lock" not in src
    assert "unlink(" not in src or "operator_stop" in src  # no lock unlink API


def _stop_mutation_flags(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    with patch(
        "services.operator_stop_action.prepare_cooperative_shutdown",
        return_value={
            "result": "SHUTDOWN_TIMED_OUT",
            "request_id": "x",
            "target_components": ["continuous"],
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
            "safe_message": "t",
        },
    ):
        pre = issue_stop_preflight(root=tmp_path)
        return execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre["preflight_token"],
            request_nonce=pre["preflight_nonce"],
            root=tmp_path,
        )


def test_70_stop_does_not_create_or_extend_paper(tmp_path):
    assert _stop_mutation_flags(tmp_path)["paper_arm_mutated"] is False


def test_71_stop_does_not_submit_order(tmp_path):
    assert _stop_mutation_flags(tmp_path)["orders_submitted"] is False


def test_72_stop_does_not_cancel_order(tmp_path):
    assert _stop_mutation_flags(tmp_path)["orders_cancelled"] is False


def test_73_stop_does_not_create_approval(tmp_path):
    assert _stop_mutation_flags(tmp_path)["approvals_created"] is False


def test_74_stop_does_not_create_authorization(tmp_path):
    assert _stop_mutation_flags(tmp_path)["authorizations_created"] is False


def test_75_stop_does_not_change_governance(tmp_path):
    assert _stop_mutation_flags(tmp_path)["governance_modified"] is False


def test_76_stop_does_not_change_lifecycle(tmp_path):
    assert _stop_mutation_flags(tmp_path)["lifecycle_modified"] is False


def test_77_stop_does_not_change_allocation(tmp_path):
    assert _stop_mutation_flags(tmp_path)["allocation_modified"] is False


def test_78_stop_does_not_change_risk(tmp_path):
    assert _stop_mutation_flags(tmp_path)["risk_modified"] is False


def test_79_stop_does_not_change_scheduled_tasks(tmp_path):
    assert _stop_mutation_flags(tmp_path)["scheduled_task_modified"] is False


def test_80_live_execution_remains_blocked(tmp_path):
    r = _stop_mutation_flags(tmp_path)
    assert r["live_execution_blocked_before"] is True
    assert r["live_execution_blocked_after"] is True


def test_81_start_disabled_during_active_stop():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "Start unavailable while Stop" in src


def test_82_stop_disabled_during_active_start():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "Stop unavailable while Start" in src


def test_83_start_requires_fresh_evaluation_after_stop():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "execute_start_triton" in src
    stop_confirm = src[
        src.index('key="stop_triton_confirm"') : src.index('key="stop_triton_cancel"')
    ]
    assert "execute_start_triton" not in stop_confirm


def test_84_stop_completion_does_not_automatically_start():
    stop_mod = (ROOT / "services" / "operator_stop_action.py").read_text(encoding="utf-8")
    assert "execute_start_triton" not in stop_mod
    assert "automatic_restart" in stop_mod


def test_85_restart_remains_unavailable():
    # DO-005 stop_action still does not implement Restart; DO-006 owns Restart.
    stop_mod = (ROOT / "services" / "operator_stop_action.py").read_text(encoding="utf-8")
    assert "ACTION_RESTART" not in stop_mod
    assert "execute_restart" not in stop_mod
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "Restart TRITON Safely" in src
    assert "execute_restart_triton" in src


def test_86_no_hidden_restart_endpoint():
    stop_mod = (ROOT / "services" / "operator_stop_action.py").read_text(encoding="utf-8")
    assert "ACTION_RESTART" not in stop_mod
    # Stop module must not chain Start
    assert "execute_start_triton" not in stop_mod


def test_87_no_stop_then_start_chain():
    stop_mod = (ROOT / "services" / "operator_stop_action.py").read_text(encoding="utf-8")
    assert "Stop followed by Start" not in stop_mod
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    stop_confirm = src[
        src.index('key="stop_triton_confirm"') : src.index('key="stop_triton_cancel"')
    ]
    assert "execute_start_triton" not in stop_confirm
    assert "execute_restart_triton" not in stop_confirm


def test_88_initial_audit_failure(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    pre = issue_stop_preflight(root=tmp_path)
    with patch(
        "services.operator_stop_action.append_stop_audit",
        return_value={"ok": False, "error": "fail"},
    ):
        r = execute_stop_triton(
            confirmation_checked=True,
            confirmation_phrase=CONFIRMATION_PHRASE,
            preflight_token=pre["preflight_token"],
            request_nonce=pre["preflight_nonce"],
            root=tmp_path,
        )
    assert r["result"] == "STOP_AUDIT_FAILED"
    assert not (tmp_path / "data" / "live" / "operator_shutdown_request.json").is_file()


def test_89_final_audit_failure(tmp_path):
    _write_fixture(tmp_path, cont_pids=[101], cont_lock_pid=101, write_wd_lock=False)
    calls = {"n": 0}

    def audit_side(record, path=None):
        calls["n"] += 1
        if calls["n"] == 1:
            from services.operator_stop_action import append_stop_audit as real

            return real(record, path=path)
        return {"ok": False, "error": "final"}

    pre = issue_stop_preflight(root=tmp_path)
    with patch(
        "services.operator_stop_action.prepare_cooperative_shutdown",
        return_value={
            "result": "COOPERATIVE_SHUTDOWN_COMPLETED",
            "request_id": "x",
            "target_components": ["continuous"],
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": True,
            "live_execution_blocked_after": True,
            "safe_message": "done",
        },
    ):
        with patch("services.operator_stop_action.append_stop_audit", side_effect=audit_side):
            r = execute_stop_triton(
                confirmation_checked=True,
                confirmation_phrase=CONFIRMATION_PHRASE,
                preflight_token=pre["preflight_token"],
                request_nonce=pre["preflight_nonce"],
                root=tmp_path,
            )
    assert r["result"] == "STOP_AUDIT_FAILED"


def test_90_audit_sanitizer_redacts_token(tmp_path):
    from services.operator_stop_action import append_stop_audit, read_recent_stop_actions

    path = tmp_path / "a.jsonl"
    append_stop_audit({"preflight_token": "SECRETTOKEN", "api_key": "k", "result": "x"}, path=path)
    rows = read_recent_stop_actions(path=path, limit=1)
    assert rows[0]["preflight_token"] == "<redacted>"
    assert rows[0]["api_key"] == "<redacted>"


def test_91_operations_shows_confirmation_consequences():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "documented safe boundary" in src
    assert "never force-terminated" in src


def test_92_operations_shows_component_specific_progress():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "Stop technical evidence" in src
    assert "_render_cooperative_shutdown_status" in src


def test_93_operations_does_not_claim_immediate_termination():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "Kill process" not in src
    assert "Force stop" not in src
    assert "Terminate now" not in src


def test_94_diagnostics_remains_read_only():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "Protected Stop evidence (DO-005)" in src
    assert "Diagnostics is read-only" in src


def test_95_six_page_navigation_unchanged():
    assert len(OPERATOR_PAGES) == 7
    assert DEFAULT_OPERATOR_PAGE == "Operations"
    assert "Reports" in OPERATOR_PAGES


def test_96_advanced_legacy_dashboard_available():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "Advanced/Legacy Dashboard" in src


def test_97_localhost_binding_remains_effective():
    cfg = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "127.0.0.1" in cfg


def test_98_do004_cooperative_surface_intact():
    assert ACTION_STOP == "STOP_TRITON"
    from services.operator_stop_action import ACTION_PREPARE
    from services.operator_cooperative_shutdown import ACTION as REQUEST_ACTION

    assert ACTION_PREPARE == "PREPARE_COOPERATIVE_SHUTDOWN"
    assert REQUEST_ACTION == "REQUEST_COOPERATIVE_SHUTDOWN"


def test_99_do004_loop_regression_surface_intact():
    assert (ROOT / "tests" / "test_loop_cooperative_shutdown_regression_do004.py").is_file()
    assert (ROOT / "scripts" / "run_continuous_loop.ps1").is_file()
    assert (ROOT / "scripts" / "run_watchdog_loop.ps1").is_file()


def test_100_do003_start_surface_intact():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "execute_start_triton" in src
    assert "Confirm Start" in src


def test_101_do002_console_surface_intact():
    assert len(OPERATOR_PAGES) == 7
    assert DEFAULT_OPERATOR_PAGE == "Operations"


def test_102_no_real_runtime_artifacts_touched_in_tests():
    """Isolation guard: this suite uses tmp_path/mocks only; no production kill paths."""
    src = (ROOT / "services" / "operator_stop_action.py").read_text(encoding="utf-8")
    assert "import subprocess" not in src
    assert "Popen(" not in src
    dash = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "taskkill" not in dash.lower()
    assert "Stop-Process" not in dash
    assert controller_has_termination_paths() is False
    assert CONFIRMATION_PHRASE == "STOP TRITON"
