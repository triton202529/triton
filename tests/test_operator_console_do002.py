"""Tests for Phase DO-002 operator console (fixtures only — no real loops/broker)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dashboard.operator_console import (
    DEFAULT_OPERATOR_PAGE,
    OPERATOR_PAGES,
    OPERATOR_RENDERERS,
    is_simulation_nav_label,
)
from services.operator_activity_feed import (
    collect_activity_events,
    filter_activity_events,
    _parse_continuous_log,
    _events_from_paper_execution_audit,
)
from services.operator_console_status import (
    classify_overall_status,
    map_paper_arm_display,
    normalize_order_status,
    build_operator_status,
    build_portfolio_summary,
    build_orders_summary,
)


def test_operator_navigation_exactly_six_pages():
    assert len(OPERATOR_PAGES) == 6
    assert OPERATOR_PAGES == [
        "Operations",
        "Activity",
        "Portfolio",
        "Orders",
        "Risk & Alerts",
        "Diagnostics",
    ]
    assert set(OPERATOR_RENDERERS.keys()) == set(OPERATOR_PAGES)


def test_operations_is_default():
    assert DEFAULT_OPERATOR_PAGE == "Operations"
    assert OPERATOR_PAGES[0] == "Operations"


def test_simulation_pages_not_in_operator_nav():
    for page in OPERATOR_PAGES:
        assert not is_simulation_nav_label(page)
    assert is_simulation_nav_label("🧪 Defensive Simulation Lab")
    assert is_simulation_nav_label("Lifecycle Rule Simulation Lab")


def test_legacy_pages_remain_in_registry():
    # Import after path setup — view_results is heavy; check PAGE_REGISTRY via exec of constants only
    import ast

    root = Path(__file__).resolve().parents[1]
    text = (root / "view_results.py").read_text(encoding="utf-8", errors="replace")
    assert "PAGE_REGISTRY" in text
    assert "SECTIONS" in text
    assert "Advanced / Legacy Dashboard" in text
    assert "dashboard_nav_mode" in text
    # Simulation label still present in legacy SECTIONS body
    assert "Simulation Lab" in text or "Simulation" in text


def test_classify_healthy_running():
    status = classify_overall_status(
        continuous_running=True,
        watchdog_running=True,
        continuous_lock_healthy=True,
        watchdog_lock_healthy=True,
        continuous_lock_class="LOCK_HEALTHY_ACTIVE",
        watchdog_lock_class="LOCK_HEALTHY_ACTIVE",
        continuous_cycle_age_minutes=10.0,
    )
    assert status == "RUNNING"


def test_lock_without_process_not_running():
    status = classify_overall_status(
        continuous_running=False,
        watchdog_running=False,
        continuous_lock_healthy=False,
        watchdog_lock_healthy=False,
        continuous_lock_class="STALE_ORPHAN_LOCK",
        watchdog_lock_class="STALE_ORPHAN_LOCK",
        continuous_cycle_age_minutes=None,
    )
    assert status != "RUNNING"
    assert status == "DEGRADED"


def test_process_lock_mismatch_degraded():
    status = classify_overall_status(
        continuous_running=True,
        watchdog_running=False,
        continuous_lock_healthy=True,
        watchdog_lock_healthy=False,
        continuous_lock_class="LOCK_HEALTHY_ACTIVE",
        watchdog_lock_class="NO_PROCESS_NO_LOCK",
        continuous_cycle_age_minutes=5.0,
    )
    assert status == "DEGRADED"


def test_stale_cycle_classification():
    status = classify_overall_status(
        continuous_running=True,
        watchdog_running=True,
        continuous_lock_healthy=True,
        watchdog_lock_healthy=True,
        continuous_lock_class="LOCK_HEALTHY_ACTIVE",
        watchdog_lock_class="LOCK_HEALTHY_ACTIVE",
        continuous_cycle_age_minutes=200.0,
    )
    assert status == "STALE"


def test_stopped_when_no_process_no_lock():
    status = classify_overall_status(
        continuous_running=False,
        watchdog_running=False,
        continuous_lock_healthy=False,
        watchdog_lock_healthy=False,
        continuous_lock_class="NO_PROCESS_NO_LOCK",
        watchdog_lock_class="NO_PROCESS_NO_LOCK",
        continuous_cycle_age_minutes=None,
    )
    assert status == "STOPPED"


def test_paper_arm_mapping():
    assert map_paper_arm_display("VALID") == "ARMED"
    assert map_paper_arm_display("EXPIRED") == "EXPIRED"
    assert map_paper_arm_display("MISSING") == "MISSING"
    assert map_paper_arm_display("INVALID") == "INVALID"
    assert map_paper_arm_display("") == "UNKNOWN"


def test_order_status_normalization():
    assert normalize_order_status("filled") == "FILLED"
    assert normalize_order_status("NEW") == "SUBMITTED"
    assert normalize_order_status("CANCELED") == "CANCELLED"
    assert normalize_order_status("partial") == "PARTIALLY_FILLED"
    assert normalize_order_status("weird") == "UNKNOWN"


def test_missing_and_malformed_artifacts(tmp_path: Path):
    root = tmp_path
    (root / "data" / "live").mkdir(parents=True)
    (root / "data" / "results").mkdir(parents=True)
    (root / "data" / "results" / "execution_readiness.json").write_text(
        "{not-json", encoding="utf-8"
    )
    (root / "data" / "live" / "fixture_process_state.json").write_text(
        json.dumps(
            {
                "watchdog_loop_running": False,
                "continuous_loop_running": False,
                "detected_watchdog_process_ids": [],
                "detected_continuous_process_ids": [],
            }
        ),
        encoding="utf-8",
    )
    status = build_operator_status(root=root)
    assert status["overall_status"] in ("STOPPED", "UNKNOWN", "DEGRADED")
    assert status["runtime_controls_exposed"] is False
    assert status["live_execution_control_exposed"] is False
    assert status["live_execution"]["blocked"] is True
    assert "LIVE EXECUTION BLOCKED" in status["live_execution"]["display"]


def test_healthy_fixture_status(tmp_path: Path):
    root = tmp_path
    live = root / "data" / "live"
    results = root / "data" / "results"
    live.mkdir(parents=True)
    results.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    lock_body = (
        f"started_at={now.isoformat()}\n"
        f"refreshed_at={now.isoformat()}\n"
        "pid={pid}\n"
        "source=test\n"
        "cycle={cycle}\n"
    )
    (live / "continuous_loop.lock").write_text(
        lock_body.format(pid=111, cycle=114), encoding="utf-8"
    )
    (live / "watchdog_loop.lock").write_text(lock_body.format(pid=222, cycle=50), encoding="utf-8")
    (live / "paper_arm_confirm.json").write_text(
        json.dumps(
            {
                "allow_execute": True,
                "expires_at": (now + timedelta(hours=1)).isoformat(),
                "confirmed_at": now.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    (results / "execution_readiness.json").write_text(
        json.dumps({"live_execution_permitted": False, "readiness_status": "READY"}),
        encoding="utf-8",
    )
    (live / "fixture_process_state.json").write_text(
        json.dumps(
            {
                "watchdog_loop_running": True,
                "continuous_loop_running": True,
                "detected_watchdog_process_ids": [222],
                "detected_continuous_process_ids": [111],
            }
        ),
        encoding="utf-8",
    )
    (results / "watchdog_status.json").write_text(
        json.dumps({"heartbeat": {"timestamp": now.isoformat()}, "broker_connected": True}),
        encoding="utf-8",
    )
    (results / "heartbeat.json").write_text(
        json.dumps({"timestamp": now.isoformat()}),
        encoding="utf-8",
    )

    status = build_operator_status(root=root)
    assert status["continuous"]["process_running"] is True
    assert status["watchdog"]["process_running"] is True
    assert status["paper_arm"]["display_status"] == "ARMED"
    assert status["live_execution"]["blocked"] is True
    assert status["overall_status"] == "RUNNING"
    assert status["cycle"]["last_cycle_number"] == 114
    assert status["runtime_controls_exposed"] is False


def test_activity_exit_code_not_order_event(tmp_path: Path):
    log = tmp_path / "continuous_loop_test.log"
    log.write_text(
        "2026-07-28 10:00:00 stage finished exit_code=0\n"
        "2026-07-28 10:01:00 [CYCLE_START] cycle=114\n"
        "2026-07-28 10:02:00 [CYCLE_OK] cycle=114\n",
        encoding="utf-8",
    )
    seen: set[str] = set()
    events = _parse_continuous_log(log, seen)
    cats = {e.category for e in events}
    assert "ORDER" not in cats
    titles = [e.title for e in events]
    assert any("Stage completed successfully" in t for t in titles)
    assert any("started" in t.lower() for t in titles)
    assert any("completed successfully" in t.lower() for t in titles)
    # start and complete are distinct
    assert any("started" in e.title.lower() and e.category == "CYCLE" for e in events)
    assert any("completed successfully" in e.title.lower() for e in events)


def test_submitted_not_filled(tmp_path: Path):
    csv_path = tmp_path / "paper_execution_audit.csv"
    csv_path.write_text(
        "timestamp,symbol,submitted_order_id,status\n" "2026-07-28T10:00:00Z,AAA,oid-1,SUBMITTED\n",
        encoding="utf-8",
    )
    seen: set[str] = set()
    events = _events_from_paper_execution_audit(csv_path, seen)
    assert len(events) == 1
    assert events[0].title == "Paper order was submitted"
    assert "filled" not in events[0].title.lower()


def test_activity_dedup_and_sort(tmp_path: Path):
    root = tmp_path
    results = root / "data" / "results"
    (results / "continuous_loop").mkdir(parents=True)
    log = results / "continuous_loop" / "continuous_loop_20260728_100000.log"
    # Duplicate identical lines should dedupe
    log.write_text(
        "2026-07-28 10:00:00 [CYCLE_OK] cycle=10\n"
        "2026-07-28 10:00:00 [CYCLE_OK] cycle=10\n"
        "2026-07-28 11:00:00 [CYCLE_OK] cycle=11\n",
        encoding="utf-8",
    )
    (results / "execution_readiness.json").write_text(
        json.dumps(
            {"live_execution_permitted": False, "generated_at": "2026-07-28T12:00:00+00:00"}
        ),
        encoding="utf-8",
    )
    payload = collect_activity_events(root=root, since_hours=720)
    cycle_ok = [
        e for e in payload["events"] if e["title"].startswith("Cycle") and "completed" in e["title"]
    ]
    # Deduped identical line
    assert len([e for e in cycle_ok if e.get("cycle_number") == 10]) == 1
    # Newest first
    times = [e["occurred_at"] for e in payload["events"]]
    assert times == sorted(times, reverse=True)
    filtered = filter_activity_events(payload, category="SAFETY")
    assert all(e["category"] == "SAFETY" for e in filtered)


def test_localhost_config_present():
    root = Path(__file__).resolve().parents[1]
    cfg = root / ".streamlit" / "config.toml"
    assert cfg.is_file()
    text = cfg.read_text(encoding="utf-8")
    assert 'address = "127.0.0.1"' in text
    assert "enableXsrfProtection = true" in text
    launcher = root / "scripts" / "run_operator_console.ps1"
    assert launcher.is_file()
    assert "127.0.0.1" in launcher.read_text(encoding="utf-8")


def test_no_mutating_runtime_action_exposed():
    from dashboard import operator_console as oc
    import inspect

    src = inspect.getsource(oc)
    forbidden = [
        "Start TRITON",
        "Stop TRITON",
        "Restart TRITON",
        "set_paper_arm_confirm",
        "taskkill",
        "subprocess.run",
        "Clear-OrphanLoopLock",
    ]
    for token in forbidden:
        assert token not in src


def test_portfolio_and_orders_missing_root(tmp_path: Path):
    (tmp_path / "data" / "results").mkdir(parents=True)
    port = build_portfolio_summary(root=tmp_path)
    assert port["available"] is False
    orders = build_orders_summary(root=tmp_path)
    assert orders["orders"] == []
    assert orders.get("by_status", {}).get("UNKNOWN", 0) == 0
