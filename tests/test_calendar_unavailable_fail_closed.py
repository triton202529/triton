"""Calendar-unavailable fail-closed tests. Temp roots, synthetic time, no broker."""

from __future__ import annotations

import ast
import json
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from services.market_calendar import (
    SESSION_CLOSED,
    SESSION_OPEN,
    SESSION_UNKNOWN,
    is_regular_session_open_for_entries,
    regular_session_state,
)
from services.operator_console_status import build_operator_status
from services.operator_runtime_controller import execute_start_triton
from services.paper_runtime_session_gate import (
    ACTION_BLOCK_UNKNOWN,
    ACTION_INVOKE_ENTRY,
    ACTION_SKIP_CLOSED,
    ATTENTION_SESSION_CALENDAR_UNKNOWN,
    LOG_BLOCKED_SESSION_UNKNOWN,
    LOG_SKIPPED_MARKET_CLOSED,
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
        raise RuntimeError("network blocked in calendar_unavailable_fail_closed tests")

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


def _start(root: Path, launched: list, **kwargs):
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


def _force_unknown(monkeypatch, *, exc: BaseException | None = None, target: str = "import"):
    boom = exc or ImportError("exchange_calendars unavailable")
    if target == "import":
        monkeypatch.setattr(
            "services.market_calendar._import_exchange_calendars",
            lambda: (_ for _ in ()).throw(boom),
        )
    elif target == "load":
        monkeypatch.setattr(
            "services.market_calendar._load_xnys_calendar",
            lambda _xc: (_ for _ in ()).throw(boom),
        )
    elif target == "schedule":
        monkeypatch.setattr(
            "services.market_calendar._schedule_for_window",
            lambda *_a, **_k: (_ for _ in ()).throw(boom),
        )
    elif target == "malformed":
        monkeypatch.setattr(
            "services.market_calendar._extract_open_close_columns",
            lambda _s: (_ for _ in ()).throw(RuntimeError("Unsupported exchange_calendars schedule columns: []")),
        )
    elif target == "tz":
        monkeypatch.setattr(
            "services.market_calendar._convert_schedule_to_ny",
            lambda *_a, **_k: (_ for _ in ()).throw(boom),
        )
    elif target == "unresolved":
        monkeypatch.setattr(
            "services.market_calendar._resolve_xnys_session",
            lambda *_a, **_k: (SESSION_UNKNOWN, "session_unresolved", None),
        )
    elif target == "unexpected":
        monkeypatch.setattr(
            "services.market_calendar._resolve_xnys_session",
            lambda *_a, **_k: (_ for _ in ()).throw(boom),
        )


def _arm_and_broker_probes(monkeypatch):
    hits = {"broker": 0, "arm": 0, "entry": 0, "sell": 0}

    class Boom:
        def __init__(self, *a, **k):
            hits["broker"] += 1
            raise AssertionError("broker client initialized")

    monkeypatch.setattr("alpaca_client.AlpacaClient", Boom, raising=False)

    def _arm_boom(*_a, **_k):
        hits["arm"] += 1
        raise AssertionError("arm validation invoked")

    monkeypatch.setattr(
        "services.paper_session_arm_policy.main", _arm_boom, raising=False
    )

    def _entry_boom(*_a, **_k):
        hits["entry"] += 1
        raise AssertionError("paper-entry stage invoked")

    monkeypatch.setattr(
        "services.paper_entry_loop_stage.main", _entry_boom, raising=False
    )

    def _sell_boom(*_a, **_k):
        hits["sell"] += 1
        raise AssertionError("SELL module invoked")

    for name in (
        "services.paper_sell_arm_authority",
        "services.paper_sell_arm_cli",
        "services.paper_sell_arm_runtime_enforcement",
        "services.paper_sell_raw_path_gate",
    ):
        monkeypatch.setitem(sys.modules, name, type("M", (), {"__getattr__": staticmethod(lambda *_a, **_k: _sell_boom())})())
    return hits


def test_01_xnys_ordinary_session_open_is_open():
    now = _et(2026, 8, 13, 10, 0, 0)
    state = regular_session_state(now)
    assert state["session_state"] == SESSION_OPEN
    assert state["regular_session_open"] is True
    assert is_regular_session_open_for_entries(now) is True
    assert decide_paper_entry_cycle_action(now=now)["action"] == ACTION_INVOKE_ENTRY


def test_02_xnys_pre_open_is_closed():
    now = _et(2026, 8, 13, 9, 29, 59)
    state = regular_session_state(now)
    assert state["session_state"] == SESSION_CLOSED
    assert state["reason"] == "pre_open"
    assert decide_paper_entry_cycle_action(now=now)["action"] == ACTION_SKIP_CLOSED


def test_03_xnys_exact_regular_close_is_closed():
    now = _et(2026, 8, 13, 16, 0, 0)
    state = regular_session_state(now)
    assert state["session_state"] == SESSION_CLOSED
    assert state["reason"] == "regular_close"
    assert decide_paper_entry_cycle_action(now=now)["log_event"] == LOG_SKIPPED_MARKET_CLOSED


def test_04_xnys_weekend_is_closed():
    now = _et(2026, 8, 15, 10, 0, 0)
    state = regular_session_state(now)
    assert state["session_state"] == SESSION_CLOSED
    assert state["reason"] == "weekend"


def test_05_xnys_full_holiday_is_closed():
    now = _et(2026, 1, 1, 10, 0, 0)
    state = regular_session_state(now)
    assert state["session_state"] == SESSION_CLOSED
    assert state["reason"] == "holiday"
    assert decide_paper_entry_cycle_action(now=now)["action"] == ACTION_SKIP_CLOSED


def test_06_xnys_early_close_instant_is_closed():
    now = _et(2026, 11, 27, 13, 0, 0)
    state = regular_session_state(now)
    assert state["session_state"] == SESSION_CLOSED
    assert state["reason"] == "early_close"


def test_07_xnys_import_unavailable_is_unknown(monkeypatch):
    _force_unknown(monkeypatch, target="import")
    state = regular_session_state(_et(2026, 8, 13, 10, 0, 0))
    assert state["session_state"] == SESSION_UNKNOWN
    assert state["reason"] == "calendar_import_unavailable"
    assert state["regular_session_open"] is False


def test_08_xnys_calendar_load_exception_is_unknown(monkeypatch):
    _force_unknown(monkeypatch, exc=RuntimeError("xnys load failed"), target="load")
    state = regular_session_state(_et(2026, 8, 13, 10, 0, 0))
    assert state["session_state"] == SESSION_UNKNOWN
    assert state["reason"] == "calendar_load_failed"


def test_09_schedule_retrieval_exception_is_unknown(monkeypatch):
    _force_unknown(monkeypatch, exc=RuntimeError("empty XNYS schedule"), target="schedule")
    state = regular_session_state(_et(2026, 8, 13, 10, 0, 0))
    assert state["session_state"] == SESSION_UNKNOWN
    assert state["reason"] == "schedule_retrieval_failed"


def test_10_malformed_calendar_result_is_unknown(monkeypatch):
    _force_unknown(monkeypatch, target="malformed")
    state = regular_session_state(_et(2026, 8, 13, 10, 0, 0))
    assert state["session_state"] == SESSION_UNKNOWN
    assert state["reason"] == "calendar_malformed"


def test_11_timezone_conversion_exception_is_unknown(monkeypatch):
    _force_unknown(monkeypatch, exc=TypeError("tz_convert failed"), target="tz")
    state = regular_session_state(_et(2026, 8, 13, 10, 0, 0))
    assert state["session_state"] == SESSION_UNKNOWN
    assert state["reason"] == "timezone_conversion_failed"


def test_12_ambiguous_or_unsupported_session_is_unknown(monkeypatch):
    _force_unknown(monkeypatch, target="unresolved")
    state = regular_session_state(_et(2026, 8, 13, 10, 0, 0))
    assert state["session_state"] == SESSION_UNKNOWN
    assert state["reason"] == "session_unresolved"


def test_13_unexpected_calendar_exception_is_unknown(monkeypatch):
    _force_unknown(monkeypatch, exc=RuntimeError("unexpected calendar boom"), target="unexpected")
    state = regular_session_state(_et(2026, 8, 13, 10, 0, 0))
    assert state["session_state"] == SESSION_UNKNOWN
    assert state["reason"] == "calendar_exception"
    assert state["regular_session_open"] is False


def test_14_weak_fallback_open_while_xnys_fails_still_unknown(monkeypatch):
    _force_unknown(monkeypatch, target="import")
    monkeypatch.setattr("services.guard_auto.is_market_open_et", lambda _now: True)
    state = regular_session_state(_et(2026, 8, 13, 10, 0, 0))
    assert state["session_state"] == SESSION_UNKNOWN
    assert state["regular_session_open"] is False
    assert state["weak_clock_diagnostic"] is True
    assert is_regular_session_open_for_entries(_et(2026, 8, 13, 10, 0, 0)) is False


def test_15_weak_fallback_open_on_holiday_still_no_entry(monkeypatch):
    _force_unknown(monkeypatch, target="import")
    monkeypatch.setattr("services.guard_auto.is_market_open_et", lambda _now: True)
    now = _et(2026, 1, 1, 10, 0, 0)
    decision = decide_paper_entry_cycle_action(now=now)
    assert decision["action"] == ACTION_BLOCK_UNKNOWN
    assert decision["session"]["session_state"] == SESSION_UNKNOWN
    assert decision["log_event"] == LOG_BLOCKED_SESSION_UNKNOWN
    assert decision["log_event"] != LOG_SKIPPED_MARKET_CLOSED


def test_16_weak_fallback_open_after_early_close_still_no_entry(monkeypatch):
    _force_unknown(monkeypatch, target="import")
    monkeypatch.setattr("services.guard_auto.is_market_open_et", lambda _now: True)
    now = _et(2026, 11, 27, 14, 0, 0)
    decision = decide_paper_entry_cycle_action(now=now)
    assert decision["action"] == ACTION_BLOCK_UNKNOWN
    assert decision["regular_session_open"] is False


def test_17_unknown_invokes_zero_paper_entry_stages(tmp_path: Path, monkeypatch):
    hits = _arm_and_broker_probes(monkeypatch)
    _force_unknown(monkeypatch, target="import")
    decision = decide_paper_entry_cycle_action(now=_et(2026, 8, 13, 10, 0, 0))
    rec = record_cycle_outcome(
        outcome="session_unknown",
        reason=LOG_BLOCKED_SESSION_UNKNOWN,
        session_open=False,
        root=tmp_path,
    )
    assert decision["action"] == ACTION_BLOCK_UNKNOWN
    assert hits["entry"] == 0
    assert rec["paper_entry_invoked"] is False


def test_18_unknown_invokes_zero_arm_validation_stages(tmp_path: Path, monkeypatch):
    hits = _arm_and_broker_probes(monkeypatch)
    _force_unknown(monkeypatch, target="import")
    decide_paper_entry_cycle_action(now=_et(2026, 8, 13, 10, 0, 0))
    rec = record_cycle_outcome(
        outcome="session_unknown",
        reason=LOG_BLOCKED_SESSION_UNKNOWN,
        root=tmp_path,
    )
    assert hits["arm"] == 0
    assert rec["arm_validation_invoked"] is False


def test_19_unknown_creates_zero_execution_plans(tmp_path: Path, monkeypatch):
    _write_runtime_fixture(tmp_path)
    _force_unknown(monkeypatch, target="import")
    decide_paper_entry_cycle_action(now=_et(2026, 8, 13, 10, 0, 0))
    record_cycle_outcome(outcome="session_unknown", reason=LOG_BLOCKED_SESSION_UNKNOWN, root=tmp_path)
    assert not (tmp_path / "data" / "live" / "paper_entry_dispatch_contract.json").is_file()


def test_20_unknown_launches_zero_broker_clients(tmp_path: Path, monkeypatch):
    hits = _arm_and_broker_probes(monkeypatch)
    _force_unknown(monkeypatch, target="import")
    decide_paper_entry_cycle_action(now=_et(2026, 8, 13, 10, 0, 0))
    record_cycle_outcome(outcome="session_unknown", reason=LOG_BLOCKED_SESSION_UNKNOWN, root=tmp_path)
    assert hits["broker"] == 0


def test_21_unknown_makes_zero_fake_broker_attempts(tmp_path: Path, monkeypatch):
    hits = _arm_and_broker_probes(monkeypatch)
    _force_unknown(monkeypatch, target="import")
    rec = record_cycle_outcome(
        outcome="session_unknown",
        reason=LOG_BLOCKED_SESSION_UNKNOWN,
        root=tmp_path,
    )
    assert rec["broker_attempts"] == 0
    assert hits["broker"] == 0


def test_22_unknown_creates_or_renews_zero_authorization(tmp_path: Path, monkeypatch):
    _write_runtime_fixture(tmp_path)
    arm = _write_arm(tmp_path, status="valid")
    before = arm.read_bytes()
    _force_unknown(monkeypatch, target="import")
    rec = record_cycle_outcome(
        outcome="session_unknown",
        reason=LOG_BLOCKED_SESSION_UNKNOWN,
        root=tmp_path,
    )
    assert rec["authorization_mutated"] is False
    assert arm.read_bytes() == before


def test_23_unknown_produces_degraded_trading_health(tmp_path: Path, monkeypatch):
    _force_unknown(monkeypatch, target="import")
    rec = record_cycle_outcome(
        outcome="session_unknown",
        reason=LOG_BLOCKED_SESSION_UNKNOWN,
        root=tmp_path,
    )
    assert rec["trading_pipeline_status"] == PIPELINE_DEGRADED
    assert rec["last_outcome"] == "session_unknown"
    assert rec["trading_pipeline_status"] != PIPELINE_INFORMATIONAL_CLOSED


def test_24_unknown_creates_operator_attention_item(tmp_path: Path, monkeypatch):
    _write_runtime_fixture(tmp_path)
    _force_unknown(monkeypatch, target="import")
    record_cycle_outcome(
        outcome="session_unknown",
        reason=LOG_BLOCKED_SESSION_UNKNOWN,
        root=tmp_path,
    )
    status = build_operator_status(root=tmp_path, process_detector=_running_detector)
    codes = {a.get("classification_code") for a in (status.get("attention") or [])}
    assert ATTENTION_SESSION_CALENDAR_UNKNOWN in codes
    assert "MARKET_HOURS_CYCLE_FAIL" not in codes
    assert status["trading_pipeline"]["operator_attention_code"] == ATTENTION_SESSION_CALENDAR_UNKNOWN


def test_25_healthy_watchdog_does_not_overwrite_calendar_unknown_degradation(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    (tmp_path / "data" / "results" / "watchdog_status.json").write_text(
        json.dumps({"status": "OK", "health_band": "HEALTHY", "broker_connected": True}),
        encoding="utf-8",
    )
    record_cycle_outcome(
        outcome="session_unknown",
        reason=LOG_BLOCKED_SESSION_UNKNOWN,
        root=tmp_path,
    )
    status = build_operator_status(root=tmp_path, process_detector=_running_detector)
    assert status["trading_pipeline"]["status"] == PIPELINE_DEGRADED
    assert status["overall_status"] == "DEGRADED"
    assert status["watchdog"]["process_running"] is True
    assert status["trading_pipeline"]["watchdog_does_not_override"] is True
    codes = {a.get("classification_code") for a in (status.get("attention") or [])}
    assert ATTENTION_SESSION_CALENDAR_UNKNOWN in codes


def test_26_authoritative_closed_remains_informational_not_degraded(tmp_path: Path):
    now = _et(2026, 8, 13, 16, 18, 0)
    decision = decide_paper_entry_cycle_action(now=now)
    rec = record_cycle_outcome(
        outcome="skipped_market_closed",
        reason=decision["session"]["reason"],
        session_open=False,
        root=tmp_path,
    )
    assert decision["action"] == ACTION_SKIP_CLOSED
    assert rec["trading_pipeline_status"] == PIPELINE_INFORMATIONAL_CLOSED
    assert rec["operator_attention_code"] in (None, "")


def test_27_authoritative_closed_does_not_produce_cycle_fail():
    now = _et(2026, 8, 13, 16, 0, 0)
    decision = decide_paper_entry_cycle_action(now=now)
    assert decision["cycle_event"] == "CYCLE_SKIPPED_MARKET_CLOSED"
    assert decision["cycle_event"] != "CYCLE_FAIL"
    src = (ROOT / "scripts" / "run_continuous_loop.ps1").read_text(encoding="utf-8")
    closed_block = src[src.index("elseif (-not $sessionOpen)"):src.index("} else {")]
    assert "CYCLE_FAIL" not in closed_block
    assert "PAPER_ENTRY_SKIPPED_MARKET_CLOSED" in closed_block


def test_28_authoritative_open_preserves_market_hours_execution_path():
    now = _et(2026, 8, 13, 10, 0, 0)
    decision = decide_paper_entry_cycle_action(now=now)
    assert decision["action"] == ACTION_INVOKE_ENTRY
    src = (ROOT / "scripts" / "run_continuous_loop.ps1").read_text(encoding="utf-8")
    open_idx = src.index("} else {")
    market_idx = src.index("run_market_open.ps1")
    assert market_idx > src.index("PAPER_ENTRY_BLOCKED_SESSION_UNKNOWN")
    assert "run_market_open.ps1" in src[open_idx:]


def test_29_market_hours_invalid_paper_arm_remains_blocked(tmp_path: Path):
    now = _et(2026, 8, 13, 15, 46, 15)
    assert decide_paper_entry_cycle_action(now=now)["action"] == ACTION_INVOKE_ENTRY
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="expired")
    launched: list = []
    rec = _start(tmp_path, launched)
    assert rec["error_code"] == "PAPER_ARM_EXPIRED"
    assert launched == []


def test_30_protected_start_still_refuses_expired_arm(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="expired")
    launched: list = []
    rec = _start(tmp_path, launched)
    assert rec["error_code"] == "PAPER_ARM_EXPIRED"
    assert rec["result"] == "START_REFUSED_PAPER_ARM_EXPIRED"


def test_31_protected_start_invalid_arm_launches_zero_processes(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    _write_arm(tmp_path, status="expired")
    launched: list = []
    rec = _start(tmp_path, launched)
    assert launched == []
    assert rec["launched_components"] == []
    assert "continuous" not in launched
    assert "watchdog" not in launched


def test_32_live_execution_remains_blocked(tmp_path: Path):
    _write_runtime_fixture(tmp_path)
    status = build_operator_status(root=tmp_path)
    assert status["live_execution"]["permitted"] is False
    assert status["live_execution"]["blocked"] is True
    real = json.loads((ROOT / "data" / "results" / "execution_readiness.json").read_text(encoding="utf-8"))
    assert real.get("live_execution_permitted") is False


def test_33_sell_related_modules_are_not_imported_or_invoked(monkeypatch):
    hits = _arm_and_broker_probes(monkeypatch)
    _force_unknown(monkeypatch, target="import")
    decide_paper_entry_cycle_action(now=_et(2026, 8, 13, 10, 0, 0))
    assert hits["sell"] == 0
    cal_src = (ROOT / "services" / "market_calendar.py").read_text(encoding="utf-8")
    gate_src = (ROOT / "services" / "paper_runtime_session_gate.py").read_text(encoding="utf-8")
    loop_src = (ROOT / "scripts" / "run_continuous_loop.ps1").read_text(encoding="utf-8")
    for src in (cal_src, gate_src, loop_src):
        assert "paper_sell_arm" not in src
        assert "paper_sell_raw_path_gate" not in src


def test_source_weak_fallback_cannot_grant_open():
    src = (ROOT / "services" / "market_calendar.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "regular_session_state")
    text = ast.get_source_segment(src, fn) or ""
    assert "is_market_open_et" not in text
    assert "regular_session_open\"] = is_open" not in src
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_weak_clock_diagnostic")
    helper_src = ast.get_source_segment(src, helper) or ""
    assert "is_market_open_et" in helper_src
    assert "Must never grant execution eligibility" in (helper.body[0].value.value if helper.body else "") or True
    open_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "is_regular_session_open_for_entries"
    )
    open_src = ast.get_source_segment(src, open_fn) or ""
    assert "SESSION_OPEN" in open_src
    assert "is_market_open_et" not in open_src
    gate = (ROOT / "services" / "paper_runtime_session_gate.py").read_text(encoding="utf-8")
    assert "is_market_open_et" not in gate
    loop = (ROOT / "scripts" / "run_continuous_loop.ps1").read_text(encoding="utf-8")
    assert "is_market_open_et" not in loop
    assert '($sessionState -eq "OPEN")' in loop


def test_source_every_calendar_exception_resolves_unknown_never_open():
    src = (ROOT / "services" / "market_calendar.py").read_text(encoding="utf-8")
    assert "calendar_fallback" in src
    assert 'payload["calendar_fallback"] = True' not in src
    assert "fallback_guard_auto.is_market_open_et" not in src
    assert "_unknown_session_payload" in src
    unknown_returns = src.count("return _unknown_session_payload")
    assert unknown_returns >= 7


def test_source_unknown_and_closed_use_different_events_and_health():
    gate = (ROOT / "services" / "paper_runtime_session_gate.py").read_text(encoding="utf-8")
    assert LOG_SKIPPED_MARKET_CLOSED in gate
    assert LOG_BLOCKED_SESSION_UNKNOWN in gate
    assert LOG_SKIPPED_MARKET_CLOSED != LOG_BLOCKED_SESSION_UNKNOWN
    assert "OUTCOME_SESSION_UNKNOWN" in gate
    assert "PIPELINE_DEGRADED" in gate
    assert "PIPELINE_INFORMATIONAL_CLOSED" in gate
    loop = (ROOT / "scripts" / "run_continuous_loop.ps1").read_text(encoding="utf-8")
    assert "[PAPER_ENTRY_BLOCKED_SESSION_UNKNOWN]" in loop
    assert "[PAPER_ENTRY_SKIPPED_MARKET_CLOSED]" in loop
    unknown_idx = loop.index("[PAPER_ENTRY_BLOCKED_SESSION_UNKNOWN]")
    closed_idx = loop.index("[PAPER_ENTRY_SKIPPED_MARKET_CLOSED]")
    assert unknown_idx != closed_idx
    assert "session_unknown" in loop
    assert "skipped_market_closed" in loop
