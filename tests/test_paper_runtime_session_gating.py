"""Isolated paper-runtime session-gating tests. Temp roots, synthetic time, no broker."""

from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from services.market_calendar import regular_session_state
from services.operator_activity_feed import _events_from_paper_execution_audit
from services.operator_console_status import build_operator_status, map_paper_arm_display
from services.operator_runtime_controller import (
    classify_paper_arm_for_start,
    execute_start_triton,
    paper_arm_start_refusal,
    start_eligibility,
)
from services.paper_arm_status import classify_paper_arm_confirm_status
from services.paper_runtime_session_gate import (
    PIPELINE_DEGRADED,
    PIPELINE_INFORMATIONAL_CLOSED,
    decide_paper_entry_cycle_action,
    record_cycle_outcome,
)

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_network_and_credentials(monkeypatch):
    for key in (
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "APCA_API_BASE_URL",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")

    def _blocked(*_a, **_k):
        raise RuntimeError("network blocked in paper_runtime_session_gating tests")

    monkeypatch.setattr(socket, "create_connection", _blocked)


def _et(year, month, day, hour, minute, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=ET)


def _lock_text(pid: int, *, minutes_ago: float = 1.0) -> str:
    now = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return (
        f"started_at={now.isoformat()}\n"
        f"refreshed_at={now.isoformat()}\n"
        f"pid={pid}\n"
        f"source=test\n"
        f"cycle=1\n"
    )


def _write_runtime_fixture(root: Path) -> None:
    live = root / "data" / "live"
    results = root / "data" / "results"
    scripts = root / "scripts"
    live.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "run_watchdog_loop.ps1").write_text("# fixture\n", encoding="utf-8")
    (scripts / "run_continuous_loop.ps1").write_text("# fixture\n", encoding="utf-8")
    (results / "execution_readiness.json").write_text(
        json.dumps({"live_execution_permitted": False, "readiness_status": "READY"}),
        encoding="utf-8",
    )
    (live / "fixture_process_state.json").write_text(
        json.dumps(
            {
                "continuous_loop_running": False,
                "watchdog_loop_running": False,
                "detected_continuous_process_ids": [],
                "detected_watchdog_process_ids": [],
                "process_detection_confidence": "high",
                "command_lines": {},
            }
        ),
        encoding="utf-8",
    )


def _write_arm(root: Path, *, status: str = "valid", now: datetime | None = None) -> Path:
    live = root / "data" / "live"
    live.mkdir(parents=True, exist_ok=True)
    path = live / "paper_arm_confirm.json"
    clock = now or datetime.now(timezone.utc)
    if status == "absent":
        if path.is_file():
            path.unlink()
        return path
    if status == "malformed":
        path.write_text("{not-json", encoding="utf-8")
        return path
    if status == "expired":
        exp = clock - timedelta(minutes=5)
        ts = clock - timedelta(hours=8)
    else:
        ts = clock
        exp = clock + timedelta(hours=2)
    path.write_text(
        json.dumps(
            {
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "expires_at": exp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mode": "ASSISTED",
                "allow_execute": True,
                "confirmed": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def _start(root: Path, launched: list, *, classify_arm=None, **kwargs):
    state = {"cont": [], "wd": []}

    def snapshot():
        return {
            "continuous_pids": list(state["cont"]),
            "watchdog_pids": list(state["wd"]),
            "detection_confidence": "high",
            "command_lines": {},
        }

    def launch(component):
        launched.append(component)
        pid = 222 if component == "watchdog" else 111
        if component == "watchdog":
            state["wd"] = [pid]
            (root / "data" / "live" / "watchdog_loop.lock").write_text(
                _lock_text(pid), encoding="utf-8"
            )
        else:
            state["cont"] = [pid]
            (root / "data" / "live" / "continuous_loop.lock").write_text(
                _lock_text(pid), encoding="utf-8"
            )
        return {"ok": True, "component": component}

    kw = dict(
        root=root,
        process_snapshot=snapshot,
        launch_fn=launch,
        audit_path=root / "data" / "results" / "operator_runtime_actions.jsonl",
        guard_path=root / "data" / "live" / "operator_start_action.guard.json",
        skip_postcondition_wait=True,
    )
    kw.update(kwargs)
    return execute_start_triton(**kw)


def test_01_valid_arm_permits_process_launch_path(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="valid")
    launched: list = []
    rec = _start(tmp_path, launched)
    assert rec["result"] == "STARTED"
    assert launched == ["watchdog", "continuous"]
    el = start_eligibility(
        root=tmp_path,
        process_snapshot=lambda: {
            "continuous_pids": [111],
            "watchdog_pids": [222],
            "detection_confidence": "high",
            "command_lines": {},
        },
    )
    assert el["paper_arm_canonical"] == "VALID"


def test_02_absent_arm_refuses_start(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    launched: list = []
    rec = _start(tmp_path, launched)
    assert rec["error_code"] == "PAPER_ARM_REQUIRED"
    assert rec["result"] == "START_REFUSED_PAPER_ARM_REQUIRED"
    assert launched == []
    el = start_eligibility(root=tmp_path)
    assert el["decision"] == "START_REFUSED_PAPER_ARM_REQUIRED"
    assert el["start_button_enabled"] is False
    assert "arm paper execution first" in (el["explanation"] or "").lower()


def test_03_expired_arm_refuses_start(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="expired")
    launched: list = []
    rec = _start(tmp_path, launched)
    assert rec["error_code"] == "PAPER_ARM_EXPIRED"
    assert launched == []
    assert classify_paper_arm_confirm_status(root=tmp_path) == "EXPIRED"


def test_04_malformed_arm_refuses_start(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="malformed")
    launched: list = []
    rec = _start(tmp_path, launched)
    assert rec["error_code"] == "PAPER_ARM_MALFORMED"
    assert launched == []
    assert classify_paper_arm_confirm_status(root=tmp_path) == "INVALID"


def test_05_unknown_classifier_state_refuses_start(tmp_path: Path, monkeypatch):
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="valid")
    monkeypatch.setattr(
        "services.operator_runtime_controller.classify_paper_arm_for_start",
        lambda **_k: "FROBNITZ",
    )
    launched: list = []
    rec = _start(tmp_path, launched)
    assert rec["error_code"] == "PAPER_ARM_MALFORMED"
    assert launched == []
    refusal = paper_arm_start_refusal("FROBNITZ")
    assert refusal is not None
    assert refusal["error_code"] == "PAPER_ARM_MALFORMED"


def test_06_invalid_arm_launches_zero_continuous_loops(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="expired")
    launched: list = []
    rec = _start(tmp_path, launched)
    assert "continuous" not in launched
    assert rec["launched_components"] == []


def test_07_invalid_arm_launches_zero_watchdogs(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="expired")
    launched: list = []
    rec = _start(tmp_path, launched)
    assert "watchdog" not in launched
    assert rec["launched_components"] == []


def test_08_invalid_arm_creates_zero_runtime_locks(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="expired")
    launched: list = []
    _start(tmp_path, launched)
    assert not (tmp_path / "data" / "live" / "continuous_loop.lock").is_file()
    assert not (tmp_path / "data" / "live" / "watchdog_loop.lock").is_file()
    guard = tmp_path / "data" / "live" / "operator_start_action.guard.json"
    assert not guard.is_file()


def test_09_invalid_arm_causes_zero_broker_initialization(tmp_path: Path, monkeypatch):
    hits = []

    class Boom:
        def __init__(self, *a, **k):
            hits.append("init")
            raise AssertionError("broker client initialized")

    monkeypatch.setattr("alpaca_client.AlpacaClient", Boom, raising=False)
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="expired")
    launched: list = []
    rec = _start(tmp_path, launched)
    assert rec["error_code"] == "PAPER_ARM_EXPIRED"
    assert hits == []
    assert launched == []


def test_10_invalid_arm_does_not_create_or_renew_authorization(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    path = _write_arm(tmp_path, status="expired")
    before = path.read_bytes()
    launched: list = []
    rec = _start(tmp_path, launched)
    assert rec["paper_arm_mutated"] is False
    assert path.read_bytes() == before


def test_11_arm_valid_after_refusal_does_not_retroactively_succeed(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    launched: list = []
    rec = _start(tmp_path, launched)
    assert rec["error_code"] == "PAPER_ARM_REQUIRED"
    _write_arm(tmp_path, status="valid")
    assert rec["result"] == "START_REFUSED_PAPER_ARM_REQUIRED"
    assert rec["launched_components"] == []


def test_12_valid_arm_invalid_immediately_before_launch_fails_closed(tmp_path: Path, monkeypatch):
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="valid")
    calls = {"n": 0}

    def flaky(*, root=None, now=None):
        calls["n"] += 1
        return "VALID" if calls["n"] == 1 else "EXPIRED"

    monkeypatch.setattr("services.operator_runtime_controller.classify_paper_arm_for_start", flaky)
    launched: list = []
    rec = _start(tmp_path, launched)
    assert rec["error_code"] == "PAPER_ARM_EXPIRED"
    assert launched == []
    assert rec["evidence"]["paper_arm_gate"] == "pre_launch"


def test_13_pre_open_skips_paper_entry_without_cycle_failure():
    now = _et(2026, 8, 13, 9, 29, 59)
    decision = decide_paper_entry_cycle_action(now=now)
    assert decision["action"] == "skip_market_closed"
    assert decision["log_event"] == "PAPER_ENTRY_SKIPPED_MARKET_CLOSED"
    assert decision["session"]["reason"] == "pre_open"


def test_14_exact_open_permits_normal_entry_stage_path():
    now = _et(2026, 8, 13, 9, 30, 0)
    decision = decide_paper_entry_cycle_action(now=now)
    assert decision["action"] == "invoke_entry_sequence"
    assert decision["regular_session_open"] is True


def test_15_one_second_before_close_skips_near_close():
    now = _et(2026, 8, 13, 15, 59, 59)
    decision = decide_paper_entry_cycle_action(now=now)
    assert decision["action"] == "skip_near_close"
    assert decision["cycle_event"] == "CYCLE_SKIPPED_NEAR_CLOSE"
    assert decision["remaining_runway_seconds"] < 181


def test_16_exact_regular_close_skips_paper_entry_without_cycle_failure():
    now = _et(2026, 8, 13, 16, 0, 0)
    decision = decide_paper_entry_cycle_action(now=now)
    assert decision["action"] == "skip_market_closed"
    assert decision["session"]["reason"] == "regular_close"


def test_17_post_close_expired_arm_produces_informational_skip(tmp_path: Path):
    now = _et(2026, 8, 12, 16, 18, 0)
    decision = decide_paper_entry_cycle_action(now=now)
    assert decision["action"] == "skip_market_closed"
    rec = record_cycle_outcome(
        outcome="skipped_market_closed",
        reason=decision["session"]["reason"],
        session_open=False,
        root=tmp_path,
        now=now.astimezone(timezone.utc),
    )
    assert rec["trading_pipeline_status"] == PIPELINE_INFORMATIONAL_CLOSED
    assert rec["last_outcome"] != "fail"


def test_18_weekend_produces_informational_skip():
    now = _et(2026, 8, 15, 10, 0, 0)
    decision = decide_paper_entry_cycle_action(now=now)
    assert decision["action"] == "skip_market_closed"
    assert decision["session"]["reason"] == "weekend"


def test_19_full_holiday_produces_informational_skip():
    now = _et(2026, 1, 1, 10, 0, 0)
    decision = decide_paper_entry_cycle_action(now=now)
    assert decision["action"] == "skip_market_closed"
    assert decision["session"]["reason"] == "holiday"


def test_20_early_close_boundary_is_respected():
    invoke_ok = _et(2026, 11, 27, 12, 56, 59)
    near = _et(2026, 11, 27, 12, 59, 59)
    closed_at = _et(2026, 11, 27, 13, 0, 0)
    assert decide_paper_entry_cycle_action(now=invoke_ok)["action"] == "invoke_entry_sequence"
    near_dec = decide_paper_entry_cycle_action(now=near)
    assert near_dec["action"] == "skip_near_close"
    closed = decide_paper_entry_cycle_action(now=closed_at)
    assert closed["action"] == "skip_market_closed"
    assert closed["session"]["reason"] == "early_close"


def test_21_utc_to_eastern_conversion_correct_during_dst():
    utc_open = datetime(2026, 8, 13, 13, 30, 0, tzinfo=timezone.utc)
    utc_pre = datetime(2026, 8, 13, 13, 29, 59, tzinfo=timezone.utc)
    assert regular_session_state(utc_open)["regular_session_open"] is True
    assert regular_session_state(utc_pre)["regular_session_open"] is False
    assert utc_open.astimezone(ET).hour == 9
    assert utc_open.astimezone(ET).tzname() == "EDT"


def test_22_utc_to_eastern_conversion_correct_during_standard_time():
    utc_open = datetime(2026, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    utc_pre = datetime(2026, 1, 15, 14, 29, 59, tzinfo=timezone.utc)
    assert utc_open.astimezone(ET).hour == 9
    assert utc_open.astimezone(ET).tzname() == "EST"
    assert regular_session_state(utc_open)["regular_session_open"] is True
    assert regular_session_state(utc_pre)["regular_session_open"] is False


def test_23_market_hours_invalid_arm_remains_blocked_degraded(tmp_path: Path):
    now = _et(2026, 8, 13, 15, 46, 15)
    assert decide_paper_entry_cycle_action(now=now)["action"] == "invoke_entry_sequence"
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="expired")
    launched: list = []
    rec = _start(tmp_path, launched)
    assert rec["error_code"] == "PAPER_ARM_EXPIRED"
    health = record_cycle_outcome(
        outcome="fail",
        reason="blocked_arm_expired",
        session_open=True,
        root=tmp_path,
        now=now.astimezone(timezone.utc),
    )
    assert health["trading_pipeline_status"] == PIPELINE_DEGRADED


def test_24_market_closed_path_makes_zero_broker_attempts(tmp_path: Path, monkeypatch):
    hits = []

    class Boom:
        def __init__(self, *a, **k):
            hits.append("init")
            raise AssertionError("broker")

    monkeypatch.setattr("alpaca_client.AlpacaClient", Boom, raising=False)
    now = _et(2026, 8, 13, 16, 18, 0)
    decision = decide_paper_entry_cycle_action(now=now)
    assert decision["action"] == "skip_market_closed"
    rec = record_cycle_outcome(
        outcome="skipped_market_closed",
        reason=decision["session"]["reason"],
        session_open=False,
        root=tmp_path,
        now=now.astimezone(timezone.utc),
    )
    assert rec["broker_attempts"] == 0
    assert hits == []


def test_25_market_closed_path_creates_no_execution_plan_or_authorization(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    arm = _write_arm(tmp_path, status="expired")
    before = arm.read_bytes()
    now = _et(2026, 8, 13, 16, 18, 0)
    rec = record_cycle_outcome(
        outcome="skipped_market_closed",
        reason="regular_close",
        session_open=False,
        root=tmp_path,
        now=now.astimezone(timezone.utc),
    )
    assert rec["authorization_mutated"] is False
    assert arm.read_bytes() == before
    assert not (tmp_path / "data" / "live" / "paper_entry_dispatch_contract.json").is_file()


def _running_detector():
    return {
        "continuous_loop_running": True,
        "watchdog_loop_running": True,
        "detected_continuous_process_ids": [11],
        "detected_watchdog_process_ids": [22],
        "process_detection_confidence": "high",
        "continuous_detection_method": "fixture",
        "watchdog_detection_method": "fixture",
        "continuous_lock_classification": "LOCK_HEALTHY_ACTIVE",
        "watchdog_lock_classification": "LOCK_HEALTHY_ACTIVE",
        "continuous_lock_healthy": True,
        "watchdog_lock_healthy": True,
        "continuous_lock_exists": True,
        "watchdog_lock_exists": True,
    }


def test_26_latest_market_hours_cycle_fail_produces_degraded_trading_health(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    record_cycle_outcome(
        outcome="fail", reason="blocked_arm_expired", session_open=True, root=tmp_path
    )
    status = build_operator_status(root=tmp_path, process_detector=_running_detector)
    assert status["trading_pipeline"]["status"] == PIPELINE_DEGRADED
    assert status["overall_status"] == "DEGRADED"
    assert status["process_overall_status"] == "RUNNING"


def test_27_healthy_watchdog_does_not_erase_degraded_trading_health(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    (tmp_path / "data" / "results" / "watchdog_status.json").write_text(
        json.dumps({"status": "OK", "health_band": "HEALTHY", "broker_connected": True}),
        encoding="utf-8",
    )
    record_cycle_outcome(
        outcome="fail", reason="blocked_arm_expired", session_open=True, root=tmp_path
    )
    status = build_operator_status(root=tmp_path, process_detector=_running_detector)
    assert status["trading_pipeline"]["status"] == PIPELINE_DEGRADED
    assert status["overall_status"] == "DEGRADED"
    assert status["watchdog"]["process_running"] is True
    assert status["trading_pipeline"]["watchdog_does_not_override"] is True


def test_28_later_market_hours_cycle_ok_clears_degradation(tmp_path: Path):
    record_cycle_outcome(
        outcome="fail", reason="blocked_arm_expired", session_open=True, root=tmp_path
    )
    rec = record_cycle_outcome(
        outcome="ok", reason="market_hours_cycle_ok", session_open=True, root=tmp_path
    )
    assert rec["trading_pipeline_status"] == "HEALTHY"
    assert rec["last_market_hours_outcome"] == "ok"


def test_29_after_hours_skip_does_not_clear_unresolved_market_hours_failure(tmp_path: Path):
    record_cycle_outcome(
        outcome="fail", reason="blocked_arm_expired", session_open=True, root=tmp_path
    )
    rec = record_cycle_outcome(
        outcome="skipped_market_closed",
        reason="regular_close",
        session_open=False,
        root=tmp_path,
    )
    assert rec["trading_pipeline_status"] == PIPELINE_DEGRADED
    assert rec["last_market_hours_outcome"] == "fail"


def test_30_after_hours_skip_alone_does_not_create_degradation(tmp_path: Path):
    rec = record_cycle_outcome(
        outcome="skipped_market_closed",
        reason="regular_close",
        session_open=False,
        root=tmp_path,
    )
    assert rec["trading_pipeline_status"] == PIPELINE_INFORMATIONAL_CLOSED
    _write_runtime_fixture(tmp_path)
    status = build_operator_status(root=tmp_path)
    assert (
        status["overall_status"] != "DEGRADED"
        or status["trading_pipeline"]["status"] != PIPELINE_DEGRADED
    )
    assert status["trading_pipeline"]["status"] == PIPELINE_INFORMATIONAL_CLOSED


def test_31_current_valid_arm_plus_prior_failed_cycle_displays_both(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="valid")
    record_cycle_outcome(
        outcome="fail", reason="blocked_arm_expired", session_open=True, root=tmp_path
    )
    status = build_operator_status(root=tmp_path, process_detector=_running_detector)
    assert status["paper_arm"]["display_status"] == "ARMED"
    assert status["paper_arm"]["currently_valid"] is True
    assert status["trading_pipeline"]["status"] == PIPELINE_DEGRADED
    assert status["overall_status"] == "DEGRADED"
    codes = {a.get("classification_code") for a in (status.get("attention") or [])}
    assert "MARKET_HOURS_CYCLE_FAIL" in codes


def test_32_expired_after_hours_arm_displays_informational_status(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="expired")
    record_cycle_outcome(
        outcome="skipped_market_closed", reason="regular_close", session_open=False, root=tmp_path
    )
    status = build_operator_status(root=tmp_path)
    assert status["paper_arm"]["display_status"] == "EXPIRED"
    assert status["paper_arm"]["currently_valid"] is False
    assert status["trading_pipeline"]["status"] == PIPELINE_INFORMATIONAL_CLOSED
    assert map_paper_arm_display("EXPIRED") == "EXPIRED"


def test_33_historical_order_event_displays_its_actual_date(tmp_path: Path):
    csv_path = tmp_path / "paper_execution_audit.csv"
    csv_path.write_text(
        "timestamp,symbol,submitted_order_id,status\n" "2026-08-11T20:15:00Z,HD,oid-hd,SUBMITTED\n",
        encoding="utf-8",
    )
    events = _events_from_paper_execution_audit(csv_path, set())
    assert len(events) == 1
    assert events[0].occurred_date == "2026-08-11"
    assert "2026-08-11" in events[0].title
    assert "2026-08-13" not in events[0].title


def test_source_continuous_loop_skips_market_open_when_closed():
    src = (ROOT / "scripts" / "run_continuous_loop.ps1").read_text(encoding="utf-8")
    assert "paper_runtime_session_gate" in src
    assert "[PAPER_ENTRY_SKIPPED_MARKET_CLOSED]" in src
    assert "[CYCLE_SKIPPED_MARKET_CLOSED]" in src
    assert "[CYCLE_SKIPPED_NEAR_CLOSE]" in src
    assert "skipped_near_close" in src
    session_idx = src.index("paper_runtime_session_gate --check")
    market_idx = src.index("run_market_open.ps1")
    near_idx = src.index("[PAPER_ENTRY_SKIPPED_NEAR_CLOSE]")
    assert session_idx < market_idx
    assert near_idx < market_idx
    skip_idx = src.index("[PAPER_ENTRY_SKIPPED_MARKET_CLOSED]")
    assert skip_idx < market_idx
    fail_idx = src.index("[CYCLE_FAIL]")
    assert "skipped_market_closed" in src[skip_idx:fail_idx]
    assert "CYCLE_FAIL" not in src[src.index("if (-not $sessionOpen)") : src.index("} else {")]
    assert "$IntervalMinutes = 45" in src or "[int]$IntervalMinutes = 45" in src


def test_canonical_classifier_is_shared():
    src = (ROOT / "services" / "operator_runtime_controller.py").read_text(encoding="utf-8")
    assert "classify_paper_arm_confirm_status" in src
    assert "PAPER_ARM_EXPIRED" in src
    policy = (ROOT / "services" / "paper_session_arm_policy.py").read_text(encoding="utf-8")
    assert "classify_paper_arm_confirm_status" in policy
    assert classify_paper_arm_for_start.__doc__
    assert "same as execution step 0" in (classify_paper_arm_for_start.__doc__ or "")
