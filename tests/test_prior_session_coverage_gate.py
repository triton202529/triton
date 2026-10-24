"""Prior-session coverage must block protected start. Temporary stores only."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from services.operator_runtime_controller import execute_start_triton, start_eligibility
from services.operator_stop_action import execute_stop_triton, issue_stop_preflight
from services.nyse_signal_session import required_signal_session
from services.prior_session_coverage import (
    REFUSAL_CODE,
    REFUSAL_DECISION,
    assess_prior_session_coverage,
)
from tests.prior_session_coverage_support import (
    index_symbols,
    install_ticker_universe,
    seed_prior_session_coverage,
    tradable_symbols,
    write_non_coverage_artifacts,
    write_processed_bars,
    write_unreadable_processed_file,
)

ET = ZoneInfo("America/New_York")
PRE_OPEN = datetime(2026, 10, 23, 9, 0, tzinfo=ET)


def _layout(root: Path) -> None:
    live = root / "data" / "live"
    results = root / "data" / "results"
    scripts = root / "scripts"
    for path in (live, results, scripts):
        path.mkdir(parents=True, exist_ok=True)
    (scripts / "run_watchdog_loop.ps1").write_text("# fixture\n", encoding="utf-8")
    (scripts / "run_continuous_loop.ps1").write_text("# fixture\n", encoding="utf-8")
    (results / "execution_readiness.json").write_text(
        json.dumps({"live_execution_permitted": False, "readiness_status": "READY"}),
        encoding="utf-8",
    )
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


def _arm(root: Path, when: datetime) -> None:
    clock = when.astimezone(timezone.utc)
    (root / "data" / "live" / "paper_arm_confirm.json").write_text(
        json.dumps(
            {
                "timestamp": clock.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "expires_at": (clock + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mode": "ASSISTED",
                "allow_execute": True,
                "confirmed": True,
            }
        ),
        encoding="utf-8",
    )


def _stopped_evening(root: Path) -> dict:
    _layout(root)
    pre = issue_stop_preflight(root=root)
    assert pre["ok"] is True
    stopped = execute_stop_triton(
        confirmation_checked=True,
        confirmation_phrase="STOP TRITON",
        preflight_token=pre["preflight_token"],
        request_nonce=pre["preflight_nonce"],
        root=root,
    )
    assert stopped["result"] == "STOP_ALREADY_STOPPED"
    assert stopped["forced_termination_used"] is False
    assert stopped["runtime_locks_deleted_by_controller"] is False
    assert not (root / "data" / "live" / "continuous_loop.lock").exists()
    assert not (root / "data" / "live" / "watchdog_loop.lock").exists()
    return stopped


def _launch_kwargs(root: Path, launched: list):
    state = {"continuous_pids": [], "watchdog_pids": [], "command_lines": {}}

    def snapshot():
        return {
            "continuous_pids": list(state["continuous_pids"]),
            "watchdog_pids": list(state["watchdog_pids"]),
            "detection_confidence": "high",
            "command_lines": dict(state["command_lines"]),
        }

    def launch(component):
        launched.append(component)
        live = root / "data" / "live"
        pid = 222 if component == "watchdog" else 111
        script = "run_watchdog_loop.ps1" if component == "watchdog" else "run_continuous_loop.ps1"
        now = datetime.now(timezone.utc).isoformat()
        (live / f"{component}_loop.lock").write_text(
            f"started_at={now}\nrefreshed_at={now}\npid={pid}\nsource=test\ncycle=1\n",
            encoding="utf-8",
        )
        key = "watchdog_pids" if component == "watchdog" else "continuous_pids"
        state[key] = [pid]
        state["command_lines"][pid] = script
        return {"ok": True, "component": component}

    return dict(
        root=root,
        now=PRE_OPEN,
        launch_fn=launch,
        audit_path=root / "data" / "results" / "operator_runtime_actions.jsonl",
        guard_path=root / "data" / "live" / "operator_start_action.guard.json",
        skip_postcondition_wait=True,
        process_snapshot=snapshot,
    )


def _assert_refusal(eligibility: dict, *, cause: str) -> dict:
    coverage = eligibility["prior_session_coverage"]
    assert eligibility["decision"] == REFUSAL_DECISION
    assert eligibility["error_code"] == REFUSAL_CODE
    assert eligibility["components_to_start"] == []
    assert eligibility["start_button_enabled"] is False
    assert coverage["ok"] is False
    assert coverage["cause"] == cause
    assert coverage["threshold"] == 0.90
    assert coverage["required_count"] == 63
    assert coverage["minimum_covered"] == 57
    assert coverage["index_excluded_count"] == 4
    assert coverage["expected_session"] == "2026-10-22"
    assert "2026-10-22" in eligibility["explanation"]
    assert "57" in eligibility["explanation"]
    assert cause in eligibility["explanation"]
    return coverage


def test_group1_protected_stop_evening_then_preopen_missing_bars_refuses(tmp_path: Path):
    _stopped_evening(tmp_path)
    _arm(tmp_path, PRE_OPEN)
    install_ticker_universe(tmp_path)
    assert required_signal_session(PRE_OPEN) == date(2026, 10, 22)
    eligibility = start_eligibility(root=tmp_path, now=PRE_OPEN)
    coverage = _assert_refusal(eligibility, cause="processed_store_missing")
    assert coverage["covered_count"] == 0
    assert len(coverage["missing_symbols"]) == 63
    launched: list = []
    rec = execute_start_triton(**_launch_kwargs(tmp_path, launched))
    assert rec["result"] == REFUSAL_DECISION
    assert rec["error_code"] == REFUSAL_CODE
    assert rec["launched_components"] == []
    assert launched == []


def test_group2_same_preopen_with_coverage_allows_one_start(tmp_path: Path):
    _stopped_evening(tmp_path)
    _arm(tmp_path, PRE_OPEN)
    session = seed_prior_session_coverage(tmp_path, covered=57, now=PRE_OPEN)
    assert session == date(2026, 10, 22)
    eligibility = start_eligibility(root=tmp_path, now=PRE_OPEN)
    assert eligibility["decision"] == "START_ALLOWED"
    assert eligibility["start_button_enabled"] is True
    assert set(eligibility["components_to_start"]) == {"watchdog", "continuous"}
    coverage = eligibility["prior_session_coverage"]
    assert coverage["ok"] is True
    assert coverage["covered_count"] == 57
    assert coverage["required_count"] == 63
    assert coverage["expected_session"] == "2026-10-22"
    launched: list = []
    rec = execute_start_triton(**_launch_kwargs(tmp_path, launched))
    assert rec["result"] == "STARTED"
    assert launched == ["watchdog", "continuous"]
    assert rec["evidence"]["prior_session_coverage"]["covered_count"] == 57


def test_group3_threshold_is_57_of_63_and_invalid_bars_do_not_count(tmp_path: Path):
    _layout(tmp_path)
    _arm(tmp_path, PRE_OPEN)
    names = tradable_symbols()
    assert len(names) == 63
    session = seed_prior_session_coverage(tmp_path, covered=56, now=PRE_OPEN)
    short = start_eligibility(root=tmp_path, now=PRE_OPEN)
    assert short["decision"] == REFUSAL_DECISION
    assert short["prior_session_coverage"]["covered_count"] == 56
    assert short["start_button_enabled"] is False

    write_processed_bars(tmp_path, names[56], session)
    allowed = start_eligibility(root=tmp_path, now=PRE_OPEN)
    assert allowed["decision"] == "START_ALLOWED"
    assert allowed["prior_session_coverage"]["covered_count"] == 57

    write_processed_bars(tmp_path, names[56], session, rows=5)
    history = assess_prior_session_coverage(root=tmp_path, now=PRE_OPEN)
    assert history["covered_count"] == 56
    assert names[56] in history["invalid_symbols"]
    assert start_eligibility(root=tmp_path, now=PRE_OPEN)["decision"] == REFUSAL_DECISION

    write_processed_bars(tmp_path, names[56], session, close_px="nan")
    finite = assess_prior_session_coverage(root=tmp_path, now=PRE_OPEN)
    assert finite["covered_count"] == 56
    assert names[56] in finite["missing_symbols"]

    write_processed_bars(tmp_path, names[56], session, high_px=1.0, low_px=9.0)
    ohlc = assess_prior_session_coverage(root=tmp_path, now=PRE_OPEN)
    assert ohlc["covered_count"] == 56
    assert names[56] in ohlc["invalid_symbols"]
    assert start_eligibility(root=tmp_path, now=PRE_OPEN)["start_button_enabled"] is False


def test_group4_calendar_unreadable_and_receipts_fail_closed(tmp_path: Path, monkeypatch):
    _layout(tmp_path)
    _arm(tmp_path, PRE_OPEN)
    install_ticker_universe(tmp_path)
    session = required_signal_session(PRE_OPEN)
    write_non_coverage_artifacts(tmp_path, session)
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    receipts = start_eligibility(root=tmp_path, now=PRE_OPEN)
    coverage = _assert_refusal(receipts, cause="coverage_below_threshold")
    assert coverage["covered_count"] == 0
    assert (tmp_path / "data" / "results" / "signals.csv").is_file()
    assert (tmp_path / "data" / "results" / "daily_refresh_fetch_receipt.json").is_file()
    assert (tmp_path / "data" / "results" / "data_freshness_refresh_audit.json").is_file()

    processed = tmp_path / "data" / "processed"
    for child in processed.iterdir():
        child.unlink()
    processed.rmdir()
    missing = start_eligibility(root=tmp_path, now=PRE_OPEN)
    assert missing["prior_session_coverage"]["cause"] == "processed_store_missing"

    processed.mkdir()
    for symbol in tradable_symbols():
        write_unreadable_processed_file(tmp_path, symbol)
    unreadable = start_eligibility(root=tmp_path, now=PRE_OPEN)
    assert unreadable["decision"] == REFUSAL_DECISION
    assert unreadable["prior_session_coverage"]["cause"] == "processed_unreadable"
    assert len(unreadable["prior_session_coverage"]["unreadable_symbols"]) == 63
    launched: list = []
    rec = execute_start_triton(**_launch_kwargs(tmp_path, launched))
    assert rec["result"] == REFUSAL_DECISION
    assert launched == []

    seed_prior_session_coverage(tmp_path, covered=63, now=PRE_OPEN)

    def _boom(now=None):
        raise RuntimeError("XNYS calendar unavailable")

    monkeypatch.setattr("services.nyse_signal_session.required_signal_session", _boom)
    calendar = start_eligibility(root=tmp_path, now=PRE_OPEN)
    assert calendar["decision"] == REFUSAL_DECISION
    assert calendar["prior_session_coverage"]["cause"] == "calendar_unavailable"
    assert calendar["start_button_enabled"] is False
    launched_calendar: list = []
    blocked = execute_start_triton(**_launch_kwargs(tmp_path, launched_calendar))
    assert blocked["result"] == REFUSAL_DECISION
    assert blocked["evidence"]["prior_session_coverage"]["cause"] == "calendar_unavailable"
    assert launched_calendar == []


def test_group5_index_symbols_are_outside_the_denominator(tmp_path: Path):
    _layout(tmp_path)
    _arm(tmp_path, PRE_OPEN)
    assert index_symbols() == ["^DJI", "^GSPC", "^IXIC", "^VIX"]
    session = seed_prior_session_coverage(tmp_path, covered=56, now=PRE_OPEN, include_indexes=True)
    refused = assess_prior_session_coverage(root=tmp_path, now=PRE_OPEN)
    assert refused["required_count"] == 63
    assert refused["covered_count"] == 56
    assert refused["index_excluded_count"] == 4
    assert refused["ok"] is False
    assert start_eligibility(root=tmp_path, now=PRE_OPEN)["decision"] == REFUSAL_DECISION

    write_processed_bars(tmp_path, tradable_symbols()[56], session)
    allowed = assess_prior_session_coverage(root=tmp_path, now=PRE_OPEN)
    assert allowed["covered_count"] == 57
    assert allowed["ok"] is True
    assert "^DJI" not in allowed["symbol_reasons"]
    assert start_eligibility(root=tmp_path, now=PRE_OPEN)["decision"] == "START_ALLOWED"


def test_group6_executor_bypass_refuses_before_launch(tmp_path: Path):
    _layout(tmp_path)
    _arm(tmp_path, PRE_OPEN)
    install_ticker_universe(tmp_path)
    launched: list = []
    rec = execute_start_triton(**_launch_kwargs(tmp_path, launched))
    assert rec["result"] == REFUSAL_DECISION
    assert rec["error_code"] == REFUSAL_CODE
    assert rec["launched_components"] == []
    assert rec["evidence"]["prior_session_coverage"]["expected_session"] == "2026-10-22"
    assert rec["evidence"]["prior_session_coverage"]["covered_count"] == 0
    assert rec["evidence"]["prior_session_coverage"]["required_count"] == 63
    assert launched == []


def test_already_running_is_unchanged_when_coverage_is_absent(tmp_path: Path):
    _layout(tmp_path)
    live = tmp_path / "data" / "live"
    now = datetime.now(timezone.utc)
    for name, pid in (("continuous_loop.lock", 111), ("watchdog_loop.lock", 222)):
        (live / name).write_text(
            f"started_at={now.isoformat()}\nrefreshed_at={now.isoformat()}\npid={pid}\n"
            "source=test\ncycle=1\n",
            encoding="utf-8",
        )
    (live / "fixture_process_state.json").write_text(
        json.dumps(
            {
                "detected_continuous_process_ids": [111],
                "detected_watchdog_process_ids": [222],
                "process_detection_confidence": "high",
                "command_lines": {
                    "111": "run_continuous_loop.ps1",
                    "222": "run_watchdog_loop.ps1",
                },
            }
        ),
        encoding="utf-8",
    )
    eligibility = start_eligibility(root=tmp_path, now=PRE_OPEN)
    assert eligibility["decision"] == "ALREADY_RUNNING"
    assert eligibility["prior_session_coverage"] is None
    assert eligibility["components_to_start"] == []
    launched: list = []
    rec = execute_start_triton(
        root=tmp_path,
        now=PRE_OPEN,
        launch_fn=lambda component: launched.append(component) or {"ok": True},
        audit_path=tmp_path / "data" / "results" / "operator_runtime_actions.jsonl",
        guard_path=tmp_path / "data" / "live" / "operator_start_action.guard.json",
        skip_postcondition_wait=True,
    )
    assert rec["result"] == "ALREADY_RUNNING"
    assert launched == []


def test_arm_and_live_safety_still_precede_coverage(tmp_path: Path):
    _layout(tmp_path)
    install_ticker_universe(tmp_path)
    missing_arm = start_eligibility(root=tmp_path, now=PRE_OPEN)
    assert missing_arm["decision"] == "START_REFUSED_PAPER_ARM_REQUIRED"
    assert missing_arm["prior_session_coverage"] is None
    (tmp_path / "data" / "results" / "execution_readiness.json").write_text(
        json.dumps({"live_execution_permitted": True}),
        encoding="utf-8",
    )
    _arm(tmp_path, PRE_OPEN)
    seed_prior_session_coverage(tmp_path, covered=63, now=PRE_OPEN)
    live = start_eligibility(root=tmp_path, now=PRE_OPEN)
    assert live["decision"] == "START_REFUSED_LIVE_SAFETY"
    assert live["prior_session_coverage"] is None
    launched: list = []
    rec = execute_start_triton(**_launch_kwargs(tmp_path, launched))
    assert rec["result"] == "START_REFUSED_LIVE_SAFETY"
    assert launched == []
