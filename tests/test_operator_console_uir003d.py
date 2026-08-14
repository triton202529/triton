"""
UIR-003D — Session-aware Operator Attention and expected after-hours state.

Presentation-only. Does not start/stop runtime, renew paper arm, or call brokers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from dashboard import operator_ui as oui
from services.operator_console_status import (
    BUCKET_ACTION_NOW,
    BUCKET_BEFORE_OPEN,
    BUCKET_INFO,
    EXPECTED_AFTER_HOURS_STOP,
    SESSION_AFTER_HOURS,
    SESSION_MARKET_OPEN,
    SESSION_PRE_MARKET,
    SESSION_UNKNOWN,
    _build_attention,
    classify_expected_after_hours_stop,
    classify_operator_session_context,
    group_attention_items,
)

ET = ZoneInfo("America/New_York")


def _base(
    *,
    market_open: Optional[bool],
    phase: str,
    cont_n: int = 0,
    wd_n: int = 0,
    cont_lock: bool = False,
    wd_lock: bool = False,
    cont_lock_class: str = "NO_PROCESS_NO_LOCK",
    wd_lock_class: str = "NO_PROCESS_NO_LOCK",
    paper: str = "EXPIRED",
    live_blocked: bool = True,
    live_permitted: bool = False,
    discovery_ok: bool = True,
    freshness: str = "FRESH",
    open_orders: Optional[int] = 0,
    expires_at: Any = None,
) -> Dict[str, Any]:
    return {
        "overall_status": "STOPPED" if cont_n == 0 and wd_n == 0 else "RUNNING",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "continuous": {
            "process_running": cont_n > 0 if discovery_ok else None,
            "process_count": cont_n if discovery_ok else None,
            "pids": list(range(cont_n)) if discovery_ok else [],
            "process_discovery_available": discovery_ok,
            "lock_exists": cont_lock,
            "lock_classification": cont_lock_class,
            "lock_path": "data/live/continuous_loop.lock",
        },
        "watchdog": {
            "process_running": wd_n > 0 if discovery_ok else None,
            "process_count": wd_n if discovery_ok else None,
            "pids": list(range(100, 100 + wd_n)) if discovery_ok else [],
            "process_discovery_available": discovery_ok,
            "lock_exists": wd_lock,
            "lock_classification": wd_lock_class,
            "lock_path": "data/live/watchdog_loop.lock",
        },
        "paper_arm": {
            "display_status": paper,
            "raw_status": paper,
            "expires_at": expires_at,
            "evidence_path": "data/live/paper_arm_confirm.json",
        },
        "live_execution": {
            "blocked": live_blocked,
            "permitted": live_permitted,
            "evidence_path": "data/results/execution_readiness.json",
        },
        "market": {
            "status": "OPEN" if market_open else ("CLOSED" if market_open is False else "UNKNOWN"),
            "market_open": market_open,
            "session_phase": phase,
        },
        "broker": {},
        "data_freshness": {"class": freshness, "evidence_path": "data/results/heartbeat.json"},
        "alerts": {"critical_count": 0, "active_count": 0},
        "governance": {},
        "cycle": {"overdue": False},
        "orders_positions": {"open_orders": open_orders, "open_orders_path": "data/results/open_orders_snapshot.csv"},
    }


def _items(parts: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [a.to_dict() for a in _build_attention(parts)]


def test_after_hours_clean_stop_expected_no_high_missing():
    parts = _base(market_open=False, phase=SESSION_AFTER_HOURS, paper="EXPIRED")
    meta = classify_expected_after_hours_stop(parts)
    assert meta["expected"] is True
    assert meta["classification"] == EXPECTED_AFTER_HOURS_STOP
    items = _items(parts)
    assert any(i.get("classification_code") == EXPECTED_AFTER_HOURS_STOP for i in items)
    highs = [i for i in items if i.get("severity") == "HIGH"]
    assert not any("missing" in i.get("title", "").lower() for i in highs)
    groups = group_attention_items(items)
    assert groups["action_required_now_count"] == 0


def test_after_hours_paper_arm_expired_is_before_open_not_high():
    parts = _base(market_open=False, phase=SESSION_AFTER_HOURS, paper="EXPIRED")
    items = _items(parts)
    arm = next(i for i in items if "Paper arm expired" in i["title"])
    assert arm["severity"] == "INFO"
    assert arm["bucket"] == BUCKET_BEFORE_OPEN
    assert arm["classification_code"] == "PAPER_ARM_EXPIRED_AFTER_HOURS"


def test_market_open_both_stopped_high():
    parts = _base(market_open=True, phase=SESSION_MARKET_OPEN, paper="ARMED")
    items = _items(parts)
    titles = [i["title"] for i in items]
    assert "Continuous loop missing" in titles
    assert "Watchdog missing" in titles
    assert all(
        i["severity"] == "HIGH" and i["bucket"] == BUCKET_ACTION_NOW
        for i in items
        if "missing" in i["title"].lower()
    )


def test_market_open_paper_arm_expired_high():
    parts = _base(market_open=True, phase=SESSION_MARKET_OPEN, paper="EXPIRED")
    items = _items(parts)
    arm = next(i for i in items if i["title"] == "Paper arm expired")
    assert arm["severity"] == "HIGH"
    assert arm["bucket"] == BUCKET_ACTION_NOW


def test_premarket_arm_not_through_close_readiness():
    # Armed but expires before coming close.
    parts = _base(
        market_open=False,
        phase=SESSION_PRE_MARKET,
        paper="ARMED",
        expires_at="2026-08-11T12:00:00Z",  # 08:00 ET — before 16:00 ET close
    )
    # Force now to pre-market Monday-like via generated_at morning ET
    parts["generated_at"] = "2026-08-11T12:00:00+00:00"  # 08:00 ET
    items = _items(parts)
    codes = [i.get("classification_code") for i in items]
    assert "PAPER_ARM_NOT_THROUGH_CLOSE" in codes
    item = next(i for i in items if i.get("classification_code") == "PAPER_ARM_NOT_THROUGH_CLOSE")
    assert item["bucket"] == BUCKET_BEFORE_OPEN
    assert item["severity"] in ("MEDIUM", "INFO", "HIGH")
    assert not any(i.get("classification_code") == EXPECTED_AFTER_HOURS_STOP for i in items)


def test_after_hours_one_component_running_not_falsely_stopped():
    parts = _base(
        market_open=False,
        phase=SESSION_AFTER_HOURS,
        cont_n=1,
        wd_n=0,
        cont_lock=True,
        cont_lock_class="HEALTHY_ACTIVE_LOCK",
        paper="ARMED",
    )
    parts["overall_status"] = "DEGRADED"
    meta = classify_expected_after_hours_stop(parts)
    assert meta["expected"] is False
    items = _items(parts)
    assert any("still running after hours" in i["title"].lower() for i in items)
    assert not any(i.get("classification_code") == EXPECTED_AFTER_HOURS_STOP for i in items)


def test_running_stale_remains_actionable_after_hours():
    parts = _base(
        market_open=False,
        phase=SESSION_AFTER_HOURS,
        cont_n=1,
        wd_n=1,
        cont_lock=True,
        wd_lock=True,
        cont_lock_class="HEALTHY_ACTIVE_LOCK",
        wd_lock_class="HEALTHY_ACTIVE_LOCK",
        paper="ARMED",
        freshness="STALE",
    )
    parts["overall_status"] = "STALE"
    items = _items(parts)
    assert any(i.get("classification_code") == "RUNNING_PROCESS_STALE_HEALTH" for i in items)
    stale = next(i for i in items if i.get("classification_code") == "RUNNING_PROCESS_STALE_HEALTH")
    assert stale["severity"] == "HIGH"
    assert stale["bucket"] == BUCKET_ACTION_NOW


def test_orphaned_lock_not_clean_expected_stop():
    parts = _base(
        market_open=False,
        phase=SESSION_AFTER_HOURS,
        cont_lock=True,
        cont_lock_class="STALE_ORPHAN_LOCK",
        paper="ARMED",
    )
    meta = classify_expected_after_hours_stop(parts)
    assert meta["expected"] is False
    items = _items(parts)
    assert any(i.get("classification_code") == "ORPHANED_LOCK" for i in items)


def test_duplicate_continuous_critical_every_session():
    for phase, open_ in (
        (SESSION_AFTER_HOURS, False),
        (SESSION_MARKET_OPEN, True),
        (SESSION_PRE_MARKET, False),
    ):
        parts = _base(market_open=open_, phase=phase, cont_n=2, wd_n=1, paper="ARMED")
        items = _items(parts)
        dup = next(i for i in items if i["title"] == "Duplicate continuous runtime")
        assert dup["severity"] == "CRITICAL"


def test_duplicate_watchdog_critical_every_session():
    parts = _base(market_open=False, phase=SESSION_AFTER_HOURS, cont_n=0, wd_n=2, paper="ARMED")
    items = _items(parts)
    dup = next(i for i in items if i["title"] == "Duplicate watchdog runtime")
    assert dup["severity"] == "CRITICAL"


def test_discovery_unavailable_unknown_refuses_safe_stop():
    parts = _base(
        market_open=False,
        phase=SESSION_AFTER_HOURS,
        discovery_ok=False,
        paper="EXPIRED",
    )
    meta = classify_expected_after_hours_stop(parts)
    assert meta["expected"] is False
    items = _items(parts)
    assert any(i.get("classification_code") == "PROCESS_DISCOVERY_UNAVAILABLE" for i in items)
    card = oui.loop_process_card_state(discovery_available=False)
    assert "COUNT UNAVAILABLE" in card["display"]


def test_session_unknown_conservative():
    parts = _base(market_open=None, phase=SESSION_UNKNOWN, paper="EXPIRED")
    meta = classify_expected_after_hours_stop(parts)
    assert meta["expected"] is False
    items = _items(parts)
    assert any(
        i.get("classification_code") in ("SESSION_EVIDENCE_UNKNOWN", "RUNTIME_MISSING_SESSION_UNKNOWN")
        for i in items
    )


def test_live_permitted_critical_any_session():
    parts = _base(
        market_open=False,
        phase=SESSION_AFTER_HOURS,
        paper="EXPIRED",
        live_blocked=False,
        live_permitted=True,
    )
    items = _items(parts)
    live = next(i for i in items if "Live execution" in i["title"])
    assert live["severity"] == "CRITICAL"
    assert live["bucket"] == BUCKET_ACTION_NOW


def test_open_orders_after_hours_remain_visible():
    parts = _base(
        market_open=False,
        phase=SESSION_AFTER_HOURS,
        paper="EXPIRED",
        open_orders=2,
    )
    items = _items(parts)
    oo = next(i for i in items if i.get("classification_code") == "OPEN_ORDERS_REQUIRE_ATTENTION")
    assert oo["severity"] == "HIGH"
    assert oo["bucket"] == BUCKET_ACTION_NOW


def test_current_validated_after_hours_screenshot_state():
    parts = _base(market_open=False, phase=SESSION_AFTER_HOURS, paper="EXPIRED")
    items = _items(parts)
    groups = group_attention_items(items)
    assert groups["action_required_now_count"] == 0
    assert groups["before_next_market_open_count"] >= 1
    assert any("safely stopped" in i["title"].lower() for i in groups["informational"])
    cont = oui.loop_process_card_state(process_count=0, process_running=False)
    wd = oui.loop_process_card_state(process_count=0, process_running=False)
    assert cont["display"] == "STOPPED · 0 / 1 INSTANCE"
    assert wd["display"] == "STOPPED · 0 / 1 INSTANCE"
    summary = oui.summarize_attention_items(items)
    assert summary["action_required_now_count"] == 0


def test_transition_open_running_to_after_hours_stop_clears_high_missing():
    open_parts = _base(market_open=True, phase=SESSION_MARKET_OPEN, cont_n=0, wd_n=0, paper="ARMED")
    open_items = _items(open_parts)
    assert any(i["severity"] == "HIGH" and "missing" in i["title"].lower() for i in open_items)

    closed_parts = _base(market_open=False, phase=SESSION_AFTER_HOURS, paper="EXPIRED")
    closed_items = _items(closed_parts)
    assert not any(i["severity"] == "HIGH" and "missing" in i["title"].lower() for i in closed_items)


def test_transition_after_hours_to_premarket_shows_readiness():
    parts = _base(market_open=False, phase=SESSION_PRE_MARKET, paper="EXPIRED")
    items = _items(parts)
    groups = group_attention_items(items)
    assert groups["before_next_market_open_count"] >= 1
    assert any(
        i.get("classification_code") in ("PAPER_ARM_EXPIRED_AFTER_HOURS", "NEXT_SESSION_START_PREP", "PREMARKET_RUNTIME_STOPPED", "EXPECTED_PREMARKET_STOP")
        or i.get("bucket") == BUCKET_BEFORE_OPEN
        for i in items
    )


def test_no_duplicate_urgent_warnings_for_same_expected_stop():
    parts = _base(market_open=False, phase=SESSION_AFTER_HOURS, paper="EXPIRED", freshness="STALE")
    items = _items(parts)
    missing = [i for i in items if "missing" in i["title"].lower()]
    assert missing == []
    expected = [i for i in items if i.get("classification_code") == EXPECTED_AFTER_HOURS_STOP]
    assert len(expected) == 1


def test_session_context_from_et_clock_after_hours():
    # Monday 21:00 ET
    now = datetime(2026, 8, 10, 21, 0, tzinfo=ET).astimezone(timezone.utc)
    ctx = classify_operator_session_context(now=now, market_open=False)
    assert ctx["session_phase"] == SESSION_AFTER_HOURS
    assert "ET" in ctx["as_of_et"]


def test_session_context_premarket():
    now = datetime(2026, 8, 11, 8, 0, tzinfo=ET).astimezone(timezone.utc)
    ctx = classify_operator_session_context(now=now, market_open=False)
    assert ctx["session_phase"] == SESSION_PRE_MARKET


def test_responsive_attention_grouping_present_in_console_source():
    from dashboard.operator_console import render_operations_page
    import inspect

    src = inspect.getsource(render_operations_page)
    assert "Action required now" in src
    assert "Before next market open" in src
    assert "Informational / expected" in src
    assert "Session context" in src
