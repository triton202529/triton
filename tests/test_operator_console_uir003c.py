"""
UIR-003C — Always-visible runtime instance counts and state-consistent action messages.

Presentation-only. Does not start/stop runtime, call broker, or mutate live evidence.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

from dashboard import operator_ui as oui
from dashboard.operator_console import render_operations_page
from services.operator_console_status import _build_attention, build_operator_status

ROOT = Path(__file__).resolve().parents[1]


def _card(
    *,
    count: Optional[int],
    hb: str = "FRESH",
    age: Any = None,
    discovery: bool = True,
    running: Optional[bool] = None,
) -> Dict[str, Any]:
    return oui.loop_process_card_state(
        process_running=running if running is not None else (True if count == 1 else False if count == 0 else None),
        process_count=count,
        discovery_available=discovery,
        heartbeat_class=hb,
        heartbeat_age_minutes=age,
    )


def test_zero_zero_stopped_cards():
    cont = _card(count=0)
    wd = _card(count=0)
    assert cont["label"] == "STOPPED"
    assert wd["label"] == "STOPPED"
    assert cont["display"] == "STOPPED · 0 / 1 INSTANCE"
    assert wd["display"] == "STOPPED · 0 / 1 INSTANCE"
    assert cont["tone"] == "crit"
    assert wd["tone"] == "crit"


def test_one_one_running_fresh():
    cont = _card(count=1, hb="FRESH")
    wd = _card(count=1, hb="FRESH")
    assert cont["display"] == "RUNNING · 1 / 1 INSTANCE"
    assert wd["display"] == "RUNNING · 1 / 1 INSTANCE"
    assert cont["tone"] == "ok"
    assert wd["tone"] == "ok"


def test_continuous_stale_not_stopped():
    cont = _card(count=1, hb="STALE", age=95)
    assert cont["label"] == "STALE"
    assert cont["display"] == "STALE · 1 / 1 INSTANCE"
    assert cont["tone"] == "warn"
    assert cont["label"] != "STOPPED"
    assert "95" in (cont.get("note") or "")


def test_watchdog_stale_not_stopped():
    wd = _card(count=1, hb="STALE", age=120)
    assert wd["label"] == "STALE"
    assert wd["display"] == "STALE · 1 / 1 INSTANCE"
    assert wd["tone"] == "warn"


def test_continuous_duplicate_runtime():
    cont = _card(count=2)
    assert cont["label"] == "DUPLICATE RUNTIME"
    assert cont["display"] == "DUPLICATE RUNTIME · 2 / 1 INSTANCES"
    assert cont["tone"] == "crit"


def test_watchdog_duplicate_runtime():
    wd = _card(count=2)
    assert wd["label"] == "DUPLICATE RUNTIME"
    assert wd["display"] == "DUPLICATE RUNTIME · 2 / 1 INSTANCES"
    assert wd["tone"] == "crit"


def test_discovery_unavailable_not_false_zero():
    card = _card(count=None, discovery=False, running=None)
    assert card["label"] == "UNKNOWN"
    assert "COUNT UNAVAILABLE" in card["display"]
    assert card["display"] != "STOPPED · 0 / 1 INSTANCE"
    assert card["tone"] == "muted"


def test_orphaned_lock_count_remains_zero():
    card = _card(count=0)
    assert card["display"] == "STOPPED · 0 / 1 INSTANCE"
    # Lock is not an instance — presentation count stays 0.
    assert card["process_count"] == 0


def test_process_without_fresh_heartbeat_keeps_count():
    card = _card(count=1, hb="UNKNOWN", age=None)
    assert card["process_count"] == 1
    assert card["display"] == "RUNNING · 1 / 1 INSTANCE"


def test_duplicate_attention_is_high_priority():
    parts = {
        "overall_status": "DEGRADED",
        "continuous": {
            "process_running": True,
            "process_count": 2,
            "pids": [1, 2],
            "process_discovery_available": True,
            "lock_exists": True,
            "lock_classification": "HEALTHY_ACTIVE_LOCK",
            "lock_path": "data/live/continuous_loop.lock",
        },
        "watchdog": {
            "process_running": True,
            "process_count": 1,
            "pids": [3],
            "process_discovery_available": True,
            "lock_exists": True,
            "lock_classification": "HEALTHY_ACTIVE_LOCK",
            "lock_path": "data/live/watchdog_loop.lock",
        },
        "paper_arm": {"display_status": "ARMED"},
        "live_execution": {"blocked": True, "permitted": False},
        "market": {"status": "CLOSED", "market_open": False, "session_phase": "AFTER_HOURS"},
        "broker": {},
        "data_freshness": {"class": "FRESH"},
        "alerts": {"critical_count": 0, "active_count": 0},
        "governance": {},
        "cycle": {"overdue": False},
        "orders_positions": {"open_orders": 0},
    }
    items = [a.to_dict() for a in _build_attention(parts)]
    titles = [i["title"] for i in items]
    assert "Duplicate continuous runtime" in titles
    dup = next(i for i in items if i["title"] == "Duplicate continuous runtime")
    assert dup["severity"] == "CRITICAL"


def test_orphaned_lock_attention_separate_from_count():
    parts = {
        "overall_status": "STOPPED",
        "continuous": {
            "process_running": False,
            "process_count": 0,
            "pids": [],
            "process_discovery_available": True,
            "lock_exists": True,
            "lock_classification": "STALE_ORPHAN_LOCK",
            "lock_path": "data/live/continuous_loop.lock",
        },
        "watchdog": {
            "process_running": False,
            "process_count": 0,
            "pids": [],
            "process_discovery_available": True,
            "lock_exists": False,
            "lock_classification": "NO_LOCK",
            "lock_path": "data/live/watchdog_loop.lock",
        },
        "paper_arm": {"display_status": "ARMED"},
        "live_execution": {"blocked": True, "permitted": False},
        "market": {
            "status": "CLOSED",
            "market_open": False,
            "session_phase": "AFTER_HOURS",
        },
        "broker": {},
        "data_freshness": {"class": "FRESH"},
        "alerts": {"critical_count": 0, "active_count": 0},
        "governance": {},
        "cycle": {"overdue": False},
        "orders_positions": {"open_orders": 0},
    }
    items = [a.to_dict() for a in _build_attention(parts)]
    titles = [i["title"] for i in items]
    assert "Continuous stale lock" in titles
    # Orphan refuses clean after-hours stop; do not also spam missing-runtime HIGH.
    assert not any(i.get("severity") == "HIGH" and "missing" in i.get("title", "").lower() for i in items)
    assert "TRITON is safely stopped after market close" not in titles


def test_start_then_stop_clears_start_as_current():
    start = {
        "result": "STARTED",
        "safe_message": (
            "TRITON started successfully. One continuous loop and one watchdog are running. "
            "Live execution remains blocked."
        ),
        "action_id": "start-1",
    }
    stop = {
        "result": "STOP_COMPLETED",
        "safe_message": "All targeted components exited and released their own locks.",
        "action_id": "stop-1",
    }
    view = oui.present_current_runtime_action_message(
        overall_status="STOPPED",
        continuous_process_count=0,
        watchdog_process_count=0,
        continuous_lock_exists=False,
        watchdog_lock_exists=False,
        live_execution_blocked=True,
        start_last_result=start,
        stop_last_result=stop,
    )
    assert view["start_suppressed"] is True
    current = view["current"]
    assert current is not None
    assert "started successfully" not in str(current["message"]).lower()
    assert "not running" in str(current["message"]).lower()
    assert any(h.get("source") == "start" for h in view["historical"])


def test_refresh_while_stopped_no_stale_running_banner():
    start = {
        "result": "STARTED",
        "safe_message": (
            "TRITON started successfully. One continuous loop and one watchdog are running. "
            "Live execution remains blocked."
        ),
    }
    view = oui.present_current_runtime_action_message(
        overall_status="STOPPED",
        continuous_process_count=0,
        watchdog_process_count=0,
        continuous_lock_exists=False,
        watchdog_lock_exists=False,
        live_execution_blocked=True,
        start_last_result=start,
        stop_last_result=None,
    )
    msg = str((view.get("current") or {}).get("message") or "").lower()
    assert "one continuous loop and one watchdog are running" not in msg
    assert "triton is stopped" in msg
    assert view["start_suppressed"] is True


def test_current_stopped_screenshot_agreement():
    """Cards, sidebar lines, overall STOPPED, and action message agree."""
    status = {
        "overall_status": "STOPPED",
        "continuous": {
            "process_running": False,
            "process_count": 0,
            "pids": [],
            "process_discovery_available": True,
            "lock_exists": False,
        },
        "watchdog": {
            "process_running": False,
            "process_count": 0,
            "pids": [],
            "process_discovery_available": True,
            "lock_exists": False,
        },
        "live_execution": {"blocked": True},
        "paper_arm": {"display_status": "ARMED"},
        "data_freshness": {"class": "STALE"},
    }
    lines = oui.compact_system_status_lines(status)
    cont = oui.loop_process_card_state(
        process_count=0, process_running=False, discovery_available=True
    )
    wd = oui.loop_process_card_state(
        process_count=0, process_running=False, discovery_available=True
    )
    action = oui.present_current_runtime_action_message(
        overall_status="STOPPED",
        continuous_process_count=0,
        watchdog_process_count=0,
        continuous_lock_exists=False,
        watchdog_lock_exists=False,
        live_execution_blocked=True,
        start_last_result={
            "result": "STARTED",
            "safe_message": (
                "TRITON started successfully. One continuous loop and one watchdog are running. "
                "Live execution remains blocked."
            ),
        },
        stop_last_result={
            "result": "STOP_COMPLETED",
            "safe_message": "All targeted components exited and released their own locks.",
        },
    )
    assert lines["overall"] == "STOPPED"
    assert lines["continuous"] == "OFF · 0"
    assert lines["watchdog"] == "OFF · 0"
    assert cont["display"] == "STOPPED · 0 / 1 INSTANCE"
    assert wd["display"] == "STOPPED · 0 / 1 INSTANCE"
    assert "not running" in str(action["current"]["message"]).lower()
    assert "started successfully" not in str(action["current"]["message"]).lower()


def test_operations_page_wires_process_count_into_cards():
    src = inspect.getsource(render_operations_page)
    assert "process_count=cont.get(\"process_count\")" in src or "process_count=cont.get('process_count')" in src
    assert "process_count=wd.get(\"process_count\")" in src or "process_count=wd.get('process_count')" in src
    assert "process_discovery_available" in src
    assert 'cont_card.get("display")' in src or "cont_card.get('display')" in src


def test_stop_clears_start_latch_in_console_source():
    src = Path(ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "start_result_history" in src
    assert 'st.session_state["start_last_result"] = None' in src
    assert "present_current_runtime_action_message" in src


def test_build_operator_status_exposes_process_count_fields():
    def detector() -> Dict[str, Any]:
        return {
            "watchdog_loop_running": False,
            "continuous_loop_running": False,
            "detected_watchdog_process_ids": [],
            "detected_continuous_process_ids": [],
            "watchdog_detection_method": "win32_process_commandline_file_invocation",
            "continuous_detection_method": "win32_process_commandline_file_invocation",
            "process_detection_confidence": "high",
            "process_detection_warnings": [],
            "watchdog_lock_exists": False,
            "continuous_lock_exists": False,
            "watchdog_lock_classification": "NO_LOCK",
            "continuous_lock_classification": "NO_LOCK",
            "watchdog_lock_healthy": True,
            "continuous_lock_healthy": True,
        }

    snap = build_operator_status(process_detector=detector)
    assert snap["continuous"]["process_count"] == 0
    assert snap["watchdog"]["process_count"] == 0
    assert snap["continuous"]["process_discovery_available"] is True
    assert snap["watchdog"]["process_discovery_available"] is True
    assert snap["overall_status"] == "STOPPED"


def test_discovery_failed_marks_unavailable():
    def detector() -> Dict[str, Any]:
        return {
            "watchdog_loop_running": False,
            "continuous_loop_running": False,
            "detected_watchdog_process_ids": [],
            "detected_continuous_process_ids": [],
            "watchdog_detection_method": "failed",
            "continuous_detection_method": "failed",
            "process_detection_confidence": "low",
            "process_detection_warnings": ["process_scan_failed=boom"],
            "watchdog_lock_exists": False,
            "continuous_lock_exists": False,
            "watchdog_lock_classification": "NO_LOCK",
            "continuous_lock_classification": "NO_LOCK",
            "watchdog_lock_healthy": True,
            "continuous_lock_healthy": True,
        }

    snap = build_operator_status(process_detector=detector)
    assert snap["continuous"]["process_discovery_available"] is False
    assert snap["continuous"]["process_count"] is None
    card = oui.loop_process_card_state(
        process_count=snap["continuous"]["process_count"],
        discovery_available=False,
    )
    assert "COUNT UNAVAILABLE" in card["display"]


def test_responsive_safety_strip_css_has_narrow_breakpoints():
    src = inspect.getsource(oui.inject_operator_theme)
    assert "2048" in src or "1920" in src
    assert "1280" in src or "minmax" in src
    assert "overflow-wrap: anywhere" in src
    assert ".triton-safety-strip" in src


def test_sidebar_compact_lines_include_loop_counts():
    lines = oui.compact_system_status_lines(
        {
            "overall_status": "STOPPED",
            "live_execution": {"blocked": True},
            "paper_arm": {"display_status": "ARMED"},
            "data_freshness": {"class": "FRESH"},
            "continuous": {"process_count": 0, "process_discovery_available": True},
            "watchdog": {"process_count": 0, "process_discovery_available": True},
        }
    )
    assert lines["continuous"] == "OFF · 0"
    assert lines["watchdog"] == "OFF · 0"
    lines_on = oui.compact_system_status_lines(
        {
            "overall_status": "RUNNING",
            "live_execution": {"blocked": True},
            "paper_arm": {"display_status": "ARMED"},
            "data_freshness": {"class": "FRESH"},
            "continuous": {"process_count": 1, "process_discovery_available": True},
            "watchdog": {"process_count": 1, "process_discovery_available": True},
        }
    )
    assert lines_on["continuous"] == "ON · 1"
    assert lines_on["watchdog"] == "ON · 1"
