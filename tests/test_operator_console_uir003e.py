"""
UIR-003E — Clean after-hours heartbeat and pipeline presentation.

Presentation-only. Does not start/stop runtime, renew paper arm, or call brokers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from dashboard import operator_ui as oui
from services.dashboard_status_sync import (
    HEARTBEAT_INACTIVE_STOPPED_LABEL,
    SIDEBAR_EXPECTED_AFTER_HOURS_LABEL,
    choose_sidebar_label,
    classify_heartbeat_banner_presentation,
    resolve_expected_after_hours_stop,
)
from services.operator_console_status import (
    EXPECTED_AFTER_HOURS_STOP,
    SESSION_AFTER_HOURS,
    SESSION_MARKET_OPEN,
    SESSION_UNKNOWN,
    classify_expected_after_hours_stop,
)
from tests.test_operator_console_uir003d import _base


def _sidebar(
    *,
    expected: bool,
    runtime_pipeline_health: str = "FAIL",
    data_freshness: str = "AGING_AFTER_CLOSE",
    market_open: bool = False,
    live_permitted: bool = False,
) -> tuple[str, str]:
    return choose_sidebar_label(
        data_freshness=data_freshness,
        runtime_pipeline_health=runtime_pipeline_health,
        market_open=market_open,
        paper_operational=False,
        paper_permitted=True,
        readiness_status="NOT_READY",
        live_permitted=live_permitted,
        execution_blocked_reasons=["market_closed"],
        expected_after_hours_stop=expected,
    )


def _banner(
    *,
    age: Optional[float],
    process_count: Optional[int],
    expected: bool,
    lock_held: bool = False,
    discovery_available: bool = True,
    timestamp_present: bool = True,
    timestamp_malformed: bool = False,
) -> Dict[str, Any]:
    return classify_heartbeat_banner_presentation(
        age_minutes=age,
        timestamp_present=timestamp_present,
        process_count=process_count,
        discovery_available=discovery_available,
        lock_held=lock_held,
        expected_after_hours_stop=expected,
        timestamp_malformed=timestamp_malformed,
    )


def test_g1_expected_stop_old_watchdog_no_actionable_banner():
    parts = _base(market_open=False, phase=SESSION_AFTER_HOURS, paper="EXPIRED")
    meta = classify_expected_after_hours_stop(parts)
    assert meta["expected"] is True
    banner = _banner(age=317.0, process_count=0, expected=True)
    assert banner["show_actionable_warning"] is False
    assert banner["presentation_label"] == HEARTBEAT_INACTIVE_STOPPED_LABEL
    hb = oui.present_watchdog_heartbeat_label(
        {"timestamp": "x", "age_minutes": 317.0, "path": "watchdog_status.json", "exists": True},
        process_count=0,
        expected_after_hours_stop=True,
        lock_held=False,
    )
    assert hb["label"] == HEARTBEAT_INACTIVE_STOPPED_LABEL
    assert hb["actionable_stale_banner"] is False
    assert hb["age_note"] is not None
    assert "watchdog_status.json" in hb["source"]


def test_g2_expected_stop_old_continuous_pipeline_no_actionable_banner():
    banner = _banner(age=200.0, process_count=0, expected=True)
    assert banner["show_actionable_warning"] is False
    assert banner["suppress_stale_for_expected_stop"] is True
    pipe = oui.present_pipeline_heartbeat_label(
        freshness_class="STALE",
        process_count=0,
        expected_after_hours_stop=True,
        lock_held=False,
        age_minutes=200.0,
    )
    assert pipe["label"] == HEARTBEAT_INACTIVE_STOPPED_LABEL
    assert pipe["actionable_stale_banner"] is False


def test_g3_sidebar_expected_stop_not_degraded():
    label, level = _sidebar(expected=True, runtime_pipeline_health="FAIL")
    assert "DEGRADED" not in label
    assert label == SIDEBAR_EXPECTED_AFTER_HOURS_LABEL
    assert level == "info"


def test_g4_refresh_expected_stop_banner_stays_suppressed():
    # Pure function: recompute from same evidence must remain suppressed.
    first = _banner(age=400.0, process_count=0, expected=True)
    second = _banner(age=400.0, process_count=0, expected=True)
    assert first["show_actionable_warning"] is False
    assert second["show_actionable_warning"] is False
    assert first == second


def test_g5_running_stale_watchdog_after_hours_remains():
    banner = _banner(age=90.0, process_count=1, expected=False)
    assert banner["show_actionable_warning"] is True
    assert banner["banner_kind"] == "stale"
    label, level = _sidebar(expected=False, runtime_pipeline_health="DEGRADED")
    assert "PIPELINE DEGRADED" in label
    assert level == "error"
    hb = oui.present_watchdog_heartbeat_label(
        {"timestamp": "x", "age_minutes": 90.0, "exists": True},
        process_count=1,
        expected_after_hours_stop=False,
    )
    assert hb["label"] == "STALE"
    assert hb["actionable_stale_banner"] is True


def test_g6_running_stale_continuous_after_hours_remains():
    banner = _banner(age=120.0, process_count=1, expected=False)
    assert banner["show_actionable_warning"] is True
    pipe = oui.present_pipeline_heartbeat_label(
        freshness_class="STALE",
        process_count=1,
        expected_after_hours_stop=False,
    )
    assert pipe["label"] == "STALE"
    assert pipe["actionable_stale_banner"] is True


def test_g7_market_open_count_zero_not_expected_stop():
    parts = _base(market_open=True, phase=SESSION_MARKET_OPEN, paper="ARMED")
    meta = classify_expected_after_hours_stop(parts)
    assert meta["expected"] is False
    label, _ = _sidebar(expected=False, runtime_pipeline_health="FAIL", market_open=True)
    assert "EXPECTED AFTER-HOURS" not in label
    banner = _banner(age=90.0, process_count=0, expected=False)
    # Not expected-stop suppression; missing/stale while stopped during open stays actionable
    # when age is stale (process absent is handled by attention HIGH separately).
    assert banner["presentation_label"] != HEARTBEAT_INACTIVE_STOPPED_LABEL


def test_g8_market_open_stale_running_remains_degraded():
    banner = _banner(age=100.0, process_count=1, expected=False)
    assert banner["show_actionable_warning"] is True
    label, level = _sidebar(
        expected=False, runtime_pipeline_health="DEGRADED", market_open=True, data_freshness="STALE"
    )
    assert "PIPELINE DEGRADED" in label
    assert level == "error"


def test_g9_orphan_lock_refuses_expected_and_warns():
    parts = _base(
        market_open=False,
        phase=SESSION_AFTER_HOURS,
        cont_lock=True,
        cont_lock_class="STALE_ORPHAN_LOCK",
        paper="EXPIRED",
    )
    meta = classify_expected_after_hours_stop(parts)
    assert meta["expected"] is False
    banner = _banner(age=200.0, process_count=0, expected=False, lock_held=True)
    assert banner["show_actionable_warning"] is True
    assert banner["banner_kind"] == "orphaned_lock"
    label, _ = _sidebar(expected=False, runtime_pipeline_health="FAIL")
    assert "EXPECTED AFTER-HOURS" not in label


def test_g10_discovery_unavailable_unknown_not_suppressed():
    banner = _banner(
        age=200.0,
        process_count=None,
        expected=False,
        discovery_available=False,
    )
    assert banner["presentation_label"] == "UNKNOWN"
    assert banner["show_actionable_warning"] is True
    parts = _base(
        market_open=False,
        phase=SESSION_AFTER_HOURS,
        discovery_ok=False,
        paper="EXPIRED",
    )
    assert classify_expected_after_hours_stop(parts)["expected"] is False


def test_g11_session_unknown_refuses_expected_wording():
    parts = _base(market_open=None, phase=SESSION_UNKNOWN, paper="EXPIRED")
    assert classify_expected_after_hours_stop(parts)["expected"] is False
    label, _ = _sidebar(expected=False, runtime_pipeline_health="FAIL")
    assert "EXPECTED AFTER-HOURS" not in label


def test_g12_duplicate_continuous_critical_path_not_expected():
    parts = _base(market_open=False, phase=SESSION_AFTER_HOURS, cont_n=2, wd_n=0, paper="EXPIRED")
    meta = classify_expected_after_hours_stop(parts)
    assert meta["expected"] is False
    label, level = _sidebar(expected=False, runtime_pipeline_health="DEGRADED")
    assert "PIPELINE DEGRADED" in label
    assert level == "error"


def test_g13_duplicate_watchdog_critical_path_not_expected():
    parts = _base(market_open=False, phase=SESSION_AFTER_HOURS, cont_n=0, wd_n=2, paper="EXPIRED")
    assert classify_expected_after_hours_stop(parts)["expected"] is False


def test_g14_unexpected_live_authority_not_suppressed():
    label, level = _sidebar(expected=False, live_permitted=True)
    assert label == "LIVE RUNTIME ENABLED"
    assert level == "error"
    # Even if someone wrongly passes expected=True, live still wins.
    label2, level2 = _sidebar(expected=True, live_permitted=True)
    assert label2 == "LIVE RUNTIME ENABLED"
    assert level2 == "error"


def test_g15_active_order_attention_path_untouched_by_banner_helper():
    # Banner helper must not claim expected-stop when caller passes expected=False
    # (order intervention remains an attention/order surface concern).
    banner = _banner(age=10.0, process_count=0, expected=False)
    assert banner["suppress_stale_for_expected_stop"] is False


def test_g16_malformed_heartbeat_integrity_warning():
    banner = _banner(
        age=None,
        process_count=0,
        expected=True,
        timestamp_present=False,
        timestamp_malformed=True,
    )
    assert banner["show_actionable_warning"] is True
    assert banner["banner_kind"] == "evidence_integrity"
    hb = oui.present_watchdog_heartbeat_label(
        {"exists": True, "timestamp": None, "age_minutes": None},
        process_count=0,
        expected_after_hours_stop=True,
    )
    assert hb["label"] == "EVIDENCE INTEGRITY"
    assert hb["actionable_stale_banner"] is True


def test_g17_future_dated_heartbeat_integrity_warning():
    banner = _banner(age=-30.0, process_count=0, expected=True)
    assert banner["show_actionable_warning"] is True
    assert banner["banner_kind"] == "evidence_integrity"


def test_g18_stopped_during_market_open_no_inactive_expected():
    banner = _banner(age=200.0, process_count=0, expected=False)
    assert banner["presentation_label"] != HEARTBEAT_INACTIVE_STOPPED_LABEL
    pipe = oui.present_pipeline_heartbeat_label(
        freshness_class="STALE",
        process_count=0,
        expected_after_hours_stop=False,
        age_minutes=200.0,
    )
    assert pipe["label"] != HEARTBEAT_INACTIVE_STOPPED_LABEL


def test_g19_clean_stop_expired_arm_no_heartbeat_emergency():
    parts = _base(market_open=False, phase=SESSION_AFTER_HOURS, paper="EXPIRED")
    assert classify_expected_after_hours_stop(parts)["classification"] == EXPECTED_AFTER_HOURS_STOP
    banner = _banner(age=500.0, process_count=0, expected=True)
    assert banner["show_actionable_warning"] is False


def test_g20_validated_screenshot_state_mapping():
    parts = _base(market_open=False, phase=SESSION_AFTER_HOURS, paper="EXPIRED")
    meta = classify_expected_after_hours_stop(parts)
    assert meta["expected"] is True
    cont = oui.loop_process_card_state(process_count=0, discovery_available=True)
    wd = oui.loop_process_card_state(process_count=0, discovery_available=True)
    assert "STOPPED · 0 / 1" in cont["display"]
    assert "STOPPED · 0 / 1" in wd["display"]
    label, _ = _sidebar(expected=True)
    assert label == SIDEBAR_EXPECTED_AFTER_HOURS_LABEL
    assert _banner(age=317.0, process_count=0, expected=True)["show_actionable_warning"] is False


def test_g21_transition_open_running_to_after_hours_clean_stop():
    open_label, _ = _sidebar(
        expected=False, runtime_pipeline_health="HEALTHY", market_open=True, data_freshness="FRESH"
    )
    assert "EXPECTED AFTER-HOURS" not in open_label
    stop_label, stop_level = _sidebar(expected=True, runtime_pipeline_health="FAIL")
    assert stop_label == SIDEBAR_EXPECTED_AFTER_HOURS_LABEL
    assert stop_level == "info"
    assert _banner(age=200.0, process_count=1, expected=False)["show_actionable_warning"] is True
    assert _banner(age=200.0, process_count=0, expected=True)["show_actionable_warning"] is False


def test_g22_transition_clean_stop_to_premarket_wording():
    # Pre-market is not EXPECTED_AFTER_HOURS_STOP; sidebar must not use after-hours inactive label.
    from services.operator_console_status import SESSION_PRE_MARKET

    parts = _base(market_open=False, phase=SESSION_PRE_MARKET, paper="EXPIRED")
    assert classify_expected_after_hours_stop(parts)["expected"] is False
    label, _ = _sidebar(expected=False, runtime_pipeline_health="FAIL")
    assert "EXPECTED AFTER-HOURS" not in label


def test_g23_transition_clean_stop_to_running_stale_returns_warning():
    assert _banner(age=200.0, process_count=0, expected=True)["show_actionable_warning"] is False
    revived = _banner(age=200.0, process_count=1, expected=False)
    assert revived["show_actionable_warning"] is True
    assert revived["banner_kind"] == "stale"


def test_g24_unrelated_live_banner_unchanged():
    label, level = choose_sidebar_label(
        data_freshness="FRESH",
        runtime_pipeline_health="HEALTHY",
        market_open=True,
        paper_operational=True,
        paper_permitted=True,
        readiness_status="READY",
        live_permitted=True,
        execution_blocked_reasons=[],
        expected_after_hours_stop=True,
    )
    assert label == "LIVE RUNTIME ENABLED"
    assert level == "error"


def test_resolve_expected_stop_uses_uir003d_classifier():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    loops = {
        "watchdog_process_running": False,
        "continuous_process_running": False,
        "watchdog_lock_present": False,
        "continuous_lock_present": False,
        "detected_watchdog_process_ids": [],
        "detected_continuous_process_ids": [],
        "process_detection_confidence": "high",
    }
    # Pin after-hours ET so the resolver is not flaky across session phases.
    after_hours = datetime(2026, 8, 10, 21, 58, tzinfo=ZoneInfo("America/New_York"))
    meta = resolve_expected_after_hours_stop(
        loops=loops,
        lock_audit={
            "watchdog_lock_classification": "NO_PROCESS_NO_LOCK",
            "continuous_lock_classification": "NO_PROCESS_NO_LOCK",
        },
        market={"status": "CLOSED", "market_open": False, "session_phase": SESSION_AFTER_HOURS},
        live_permitted=False,
        live_blocked=True,
        now=after_hours,
    )
    assert meta["expected"] is True
    assert meta["classification"] == EXPECTED_AFTER_HOURS_STOP
