"""Focused tests for Command Center status coherence (read-only)."""

from __future__ import annotations

import hashlib
import importlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from services import command_center_status_coherence as coherence
from services.command_center_status_coherence import (
    CONTINUOUS_CYCLE_OVERDUE_MINUTES,
    CONTINUOUS_INTERVAL_MINUTES,
    LOCK_TIMESTAMP_SOURCE,
    age_minutes,
    cycle_deadline_utc,
    cycle_is_overdue,
    format_live_permission_display,
    format_paper_certification_display,
    parse_lock_timestamp,
    parse_status_timestamp,
    parse_utc_timestamp,
    resolve_market_status,
    resolve_shutdown_presentation,
)

ROOT = Path(__file__).resolve().parents[1]
FIXED_NOW = datetime(2026, 8, 3, 19, 30, 0, tzinfo=timezone.utc)
LOCAL_TZ = ZoneInfo("America/New_York")

REAL_ARTIFACTS = [
    ROOT / "data" / "live" / "paper_arm_confirm.json",
    ROOT / "data" / "results" / "governance_authorization.json",
    ROOT / "data" / "results" / "capital_preservation_certification.json",
    ROOT / "data" / "results" / "execution_readiness.json",
    ROOT / "data" / "results" / "heartbeat.json",
    ROOT / "data" / "results" / "watchdog_status.json",
    ROOT / "data" / "live" / "operator_shutdown_request.json",
    ROOT / "data" / "live" / "operator_shutdown_ack_continuous.json",
    ROOT / "data" / "live" / "operator_shutdown_ack_watchdog.json",
    ROOT / "data" / "live" / "continuous_loop.lock",
    ROOT / "data" / "live" / "watchdog_loop.lock",
]


def _file_hash(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def real_hashes_before():
    return {str(p): _file_hash(p) for p in REAL_ARTIFACTS}


def test_z_timestamps_parse_as_utc():
    dt = parse_utc_timestamp("2026-08-03T17:50:41Z")
    assert dt == datetime(2026, 8, 3, 17, 50, 41, tzinfo=timezone.utc)


def test_offset_aware_timestamps_preserve_instant():
    dt = parse_utc_timestamp("2026-08-03T13:50:41-04:00")
    assert dt == datetime(2026, 8, 3, 17, 50, 41, tzinfo=timezone.utc)


def test_naive_local_lock_timestamp_converts_correctly():
    naive = "2026-08-03T15:22:24"
    dt = parse_lock_timestamp(naive, naive_tz=LOCAL_TZ)
    assert dt == datetime(2026, 8, 3, 19, 22, 24, tzinfo=timezone.utc)
    # Must not treat naive lock time as UTC
    wrong = parse_status_timestamp(naive, source="utc_json")
    assert wrong != dt


def test_fresh_heartbeats_remain_fresh():
    from services.runtime_lock_classifier import classify_runtime_lock

    lock = {
        "exists": True,
        "pid": 111,
        "started_at": "2026-08-03T10:00:00",  # old ownership time (naive local)
        "refreshed_at": "2026-08-03T15:00:00",
    }
    cls = classify_runtime_lock(
        lock,
        True,
        [111],
        loop_kind="continuous",
        grace_minutes=67.5,
        heartbeat_age_minutes=2.0,
    )
    assert cls == "LOCK_HEALTHY_ACTIVE"

    wd = classify_runtime_lock(
        {"exists": True, "pid": 222, "started_at": "2026-08-03T10:00:00"},
        True,
        [222],
        loop_kind="watchdog",
        grace_minutes=5.0,
        heartbeat_age_minutes=1.0,
    )
    assert wd == "LOCK_HEALTHY_ACTIVE"


def test_cycle_not_overdue_before_deadline():
    refreshed = FIXED_NOW - timedelta(minutes=30)
    assert cycle_is_overdue(refreshed, now=FIXED_NOW) is False
    deadline = cycle_deadline_utc(refreshed)
    assert deadline == FIXED_NOW + timedelta(minutes=15)
    assert age_minutes(refreshed, FIXED_NOW) == 30.0
    assert CONTINUOUS_INTERVAL_MINUTES == 45.0


def test_cycle_overdue_only_after_deadline_and_grace():
    # Just past interval (46m) but before overdue threshold — not overdue
    almost = FIXED_NOW - timedelta(minutes=46)
    assert almost < cycle_deadline_utc(almost)  # sanity
    assert cycle_is_overdue(almost, now=FIXED_NOW) is False
    # Past interval + grace (overdue threshold)
    late = FIXED_NOW - timedelta(minutes=CONTINUOUS_CYCLE_OVERDUE_MINUTES + 1)
    assert cycle_is_overdue(late, now=FIXED_NOW) is True


def test_static_lock_start_age_not_false_stale(tmp_path: Path, monkeypatch):
    from services import operator_console_status as ocs
    from services.runtime_lock_classifier import classify_runtime_lock

    # Old started_at, recent refreshed_at as naive local; matching live owner + fresh HB
    started_local = (
        (FIXED_NOW - timedelta(hours=5)).astimezone(LOCAL_TZ).strftime("%Y-%m-%dT%H:%M:%S")
    )
    refreshed_local = (
        (FIXED_NOW - timedelta(minutes=10)).astimezone(LOCAL_TZ).strftime("%Y-%m-%dT%H:%M:%S")
    )
    lock = {
        "exists": True,
        "pid": 111,
        "started_at": started_local,
        "refreshed_at": refreshed_local,
        # Deliberately wrong precomputed age as if naive-as-UTC inflated it
        "lock_freshness_age_minutes": 250.0,
    }
    cls = classify_runtime_lock(
        lock,
        True,
        [111],
        loop_kind="continuous",
        grace_minutes=67.5,
        heartbeat_age_minutes=3.0,
    )
    assert cls == "LOCK_HEALTHY_ACTIVE"

    monkeypatch.setattr(coherence, "local_tzinfo", lambda: LOCAL_TZ)
    monkeypatch.setattr(
        ocs, "parse_lock_timestamp", lambda v, **kw: parse_lock_timestamp(v, naive_tz=LOCAL_TZ)
    )
    monkeypatch.setattr(ocs, "_parse_lock_dt", lambda v: parse_lock_timestamp(v, naive_tz=LOCAL_TZ))

    root = tmp_path
    live = root / "data" / "live"
    results = root / "data" / "results"
    live.mkdir(parents=True)
    results.mkdir(parents=True)
    (live / "continuous_loop.lock").write_text(
        f"started_at={started_local}\nrefreshed_at={refreshed_local}\npid=111\nsource=t\ncycle=5\n",
        encoding="utf-8",
    )
    (live / "watchdog_loop.lock").write_text(
        f"started_at={refreshed_local}\nrefreshed_at={refreshed_local}\npid=222\nsource=t\ncycle=1\n",
        encoding="utf-8",
    )
    (results / "execution_readiness.json").write_text(
        json.dumps({"live_execution_permitted": False, "readiness_status": "READY"}),
        encoding="utf-8",
    )
    (results / "heartbeat.json").write_text(
        json.dumps({"timestamp": FIXED_NOW.isoformat()}), encoding="utf-8"
    )
    (results / "watchdog_status.json").write_text(
        json.dumps({"heartbeat": {"timestamp": FIXED_NOW.isoformat()}, "broker_connected": True}),
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
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "services.runtime_lock_classifier._heartbeat_age_minutes",
            lambda *_a, **_k: 2.0,
        )
        mp.setattr(
            "services.runtime_lock_classifier._effective_heartbeat_age_minutes",
            lambda *_a, **_k: 2.0,
        )
        status = ocs.build_operator_status(root=root, now=FIXED_NOW)
    assert status["overall_status"] == "RUNNING"
    assert status["cycle"]["overdue"] is False
    assert status["cycle"]["started_at_is_heartbeat"] is False


def test_dead_or_mismatched_owner_unhealthy():
    from services.runtime_lock_classifier import classify_runtime_lock

    orphan = classify_runtime_lock(
        {"exists": True, "pid": 999, "refreshed_at": FIXED_NOW.isoformat()},
        False,
        [],
        loop_kind="continuous",
        heartbeat_age_minutes=1.0,
    )
    assert orphan == "STALE_ORPHAN_LOCK"

    mismatch = classify_runtime_lock(
        {"exists": True, "pid": 999, "refreshed_at": FIXED_NOW.isoformat()},
        True,
        [111],
        loop_kind="continuous",
        heartbeat_age_minutes=1.0,
    )
    # process active but pid mismatch → still classified via process_active path
    assert mismatch in (
        "LOCK_HEALTHY_ACTIVE",
        "STALE_ACTIVE_LOCK",
        "MISSING_ACTIVE_LOCK",
        "LOCK_TIMESTAMP_STALE_BUT_PROCESS_ACTIVE",
    )


def test_runtime_health_does_not_override_readiness_governance(tmp_path: Path):
    from services.operator_console_status import build_operator_status

    root = tmp_path
    live = root / "data" / "live"
    results = root / "data" / "results"
    live.mkdir(parents=True)
    results.mkdir(parents=True)
    now = FIXED_NOW
    body = (
        f"started_at={now.isoformat()}\nrefreshed_at={now.isoformat()}\n"
        "pid={pid}\nsource=t\ncycle=1\n"
    )
    (live / "continuous_loop.lock").write_text(body.format(pid=111), encoding="utf-8")
    (live / "watchdog_loop.lock").write_text(body.format(pid=222), encoding="utf-8")
    (results / "execution_readiness.json").write_text(
        json.dumps({"live_execution_permitted": False, "readiness_status": "NOT_READY"}),
        encoding="utf-8",
    )
    (results / "governance_authorization.json").write_text(
        json.dumps({"overall_authorization": False, "paper_execution_permitted": True}),
        encoding="utf-8",
    )
    (results / "heartbeat.json").write_text(
        json.dumps({"timestamp": now.isoformat()}), encoding="utf-8"
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
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "services.runtime_lock_classifier._heartbeat_age_minutes",
            lambda *_a, **_k: 1.0,
        )
        mp.setattr(
            "services.runtime_lock_classifier._effective_heartbeat_age_minutes",
            lambda *_a, **_k: 1.0,
        )
        status = build_operator_status(root=root, now=now)
    assert status["overall_status"] == "RUNNING"
    assert status["governance"]["execution_readiness_status"] == "NOT_READY"
    assert status["governance"]["blocked"] is True
    assert status["live_execution"]["blocked"] is True


def test_main_and_sidebar_market_identical():
    def _open(_now):
        return False

    main = resolve_market_status(now=FIXED_NOW, market_open_fn=_open)
    side = resolve_market_status(now=FIXED_NOW, market_open_fn=_open)
    assert main["status"] == side["status"] == "CLOSED"
    assert main["source"] == side["source"]
    assert main["as_of"] == side["as_of"]


def test_missing_market_evidence_unknown_both_surfaces():
    def _fail(_now):
        raise RuntimeError("no calendar")

    main = resolve_market_status(now=FIXED_NOW, market_open_fn=_fail)
    side = resolve_market_status(now=FIXED_NOW, market_open_fn=_fail)
    assert main["status"] == side["status"] == "UNKNOWN"
    assert main["as_of"] == side["as_of"]


def test_paper_cert_qualified_while_live_blocked():
    paper = format_paper_certification_display("CERTIFIED_FOR_PAPER_PROTECTION")
    live = format_live_permission_display(live_permitted=False)
    assert paper["display_status"] == "PAPER:CERTIFIED_FOR_PAPER_PROTECTION"
    assert paper["paper_certified"] is True
    assert live["display_status"] == "LIVE:BLOCKED"
    assert live["live_execution_blocked"] is True


def test_missing_cert_does_not_fabricate():
    paper = format_paper_certification_display("")
    assert paper["paper_certified"] is False
    assert paper["display_status"] == "PAPER:UNKNOWN"


def test_live_block_not_failed_paper_cert():
    from services.dashboard_status_sync import derive_execution_block_reasons

    reasons = derive_execution_block_reasons(
        market_open=True,
        gate_bits={"CONFIRM": True},
        certification_status="CERTIFIED_FOR_PAPER_PROTECTION",
        runtime_pipeline_health="HEALTHY",
        data_freshness="FRESH",
        live_permitted=False,
    )
    assert "NOT_CERTIFIED" not in reasons


def test_july_30_expired_shutdown_inactive(tmp_path: Path):
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    req = {
        "request_id": "297ce786-july30",
        "state": "EXIT_PENDING",
        "expires_at": "2026-07-30T20:00:00Z",
        "requested_at": "2026-07-30T19:00:00Z",
    }
    (live / "operator_shutdown_request.json").write_text(json.dumps(req), encoding="utf-8")
    (live / "operator_shutdown_ack_continuous.json").write_text(
        json.dumps({"request_id": "297ce786-july30", "state": "ACKNOWLEDGED"}),
        encoding="utf-8",
    )
    presentation = resolve_shutdown_presentation(root=tmp_path, now=FIXED_NOW)
    assert presentation["active"] is False
    assert presentation["expired"] is True
    assert presentation["affects_current_health"] is False
    assert presentation["historical"] is True
    assert (
        "INACTIVE" in presentation["display_status"] or "HISTORY" in presentation["display_status"]
    )


def test_expired_acks_do_not_affect_health(tmp_path: Path):
    presentation = resolve_shutdown_presentation(
        root=tmp_path,
        now=FIXED_NOW,
        request={
            "request_id": "old",
            "state": "EXIT_PENDING",
            "expires_at": "2026-07-30T20:00:00+00:00",
        },
        ack_continuous={"request_id": "old", "state": "ACKNOWLEDGED"},
        ack_watchdog={"request_id": "old", "state": "ACKNOWLEDGED"},
    )
    assert presentation["affects_current_health"] is False
    assert presentation["ack_continuous_attributed"] is True  # id matches, but inactive


def test_fresh_unexpired_request_recognized():
    presentation = resolve_shutdown_presentation(
        root=Path("."),
        now=FIXED_NOW,
        request={
            "request_id": "active-1",
            "state": "EXIT_PENDING",
            "expires_at": (FIXED_NOW + timedelta(minutes=30)).isoformat(),
        },
        ack_continuous={},
        ack_watchdog={},
    )
    assert presentation["active"] is True
    assert presentation["pending"] is True
    assert presentation["affects_current_health"] is True


def test_mismatched_ack_request_ids_rejected():
    presentation = resolve_shutdown_presentation(
        root=Path("."),
        now=FIXED_NOW,
        request={
            "request_id": "req-A",
            "state": "EXIT_PENDING",
            "expires_at": (FIXED_NOW + timedelta(minutes=10)).isoformat(),
        },
        ack_continuous={"request_id": "req-B", "state": "ACKNOWLEDGED"},
        ack_watchdog={"request_id": "req-C", "state": "ACKNOWLEDGED"},
    )
    assert presentation["ack_continuous_attributed"] is False
    assert presentation["ack_watchdog_attributed"] is False


def test_historical_shutdown_display_not_pending():
    presentation = resolve_shutdown_presentation(
        root=Path("."),
        now=FIXED_NOW,
        request={
            "request_id": "hist",
            "state": "COMPLETED",
            "expires_at": (FIXED_NOW - timedelta(hours=1)).isoformat(),
        },
        ack_continuous={"request_id": "hist"},
        ack_watchdog={"request_id": "hist"},
    )
    assert presentation["pending"] is False
    assert presentation["historical"] is True


def test_build_operator_status_market_and_cert(tmp_path: Path, monkeypatch):
    from services.operator_console_status import build_operator_status

    root = tmp_path
    live = root / "data" / "live"
    results = root / "data" / "results"
    live.mkdir(parents=True)
    results.mkdir(parents=True)
    (results / "execution_readiness.json").write_text(
        json.dumps({"live_execution_permitted": False, "readiness_status": "READY"}),
        encoding="utf-8",
    )
    (results / "capital_preservation_certification.json").write_text(
        json.dumps({"certification_status": "CERTIFIED_FOR_PAPER_PROTECTION"}),
        encoding="utf-8",
    )
    (results / "governance_authorization.json").write_text(
        json.dumps({"overall_authorization": True, "paper_execution_permitted": True}),
        encoding="utf-8",
    )
    (live / "fixture_process_state.json").write_text(
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
    monkeypatch.setattr(
        "services.operator_console_status.resolve_market_status",
        lambda now=None, market_open_fn=None: {
            "status": "CLOSED",
            "market_open": False,
            "source": "guard_auto.is_market_open_et",
            "as_of": FIXED_NOW.isoformat(),
            "note": "test",
        },
    )
    status = build_operator_status(root=root, now=FIXED_NOW)
    assert status["market"]["status"] == "CLOSED"
    assert status["market"]["source"] == "guard_auto.is_market_open_et"
    assert status["paper_certification"]["display_status"] == (
        "PAPER:CERTIFIED_FOR_PAPER_PROTECTION"
    )
    assert status["live_permission"]["display_status"] == "LIVE:BLOCKED"
    assert status["governance"]["blocked"] is False  # boolean True auth not blocked
    assert status["live_execution"]["blocked"] is True


def test_import_does_not_mutate_artifacts(real_hashes_before):
    importlib.reload(importlib.import_module("services.command_center_status_coherence"))
    importlib.reload(importlib.import_module("services.operator_console_status"))
    importlib.reload(importlib.import_module("services.runtime_lock_classifier"))
    importlib.reload(importlib.import_module("services.operator_runtime_classifier"))
    importlib.reload(importlib.import_module("services.dashboard_status_sync"))
    after = {str(p): _file_hash(p) for p in REAL_ARTIFACTS}
    assert after == real_hashes_before


def test_real_artifacts_unchanged_after_suite(real_hashes_before):
    after = {str(p): _file_hash(p) for p in REAL_ARTIFACTS}
    assert after == real_hashes_before


def test_no_broker_or_runtime_handlers_in_coherence_module():
    src = (ROOT / "services" / "command_center_status_coherence.py").read_text(encoding="utf-8")
    banned = (
        "submit_order",
        "cancel_order",
        "write_confirm",
        "prepare_cooperative_shutdown",
        "requests.post",
    )
    lower = src.lower()
    for b in banned:
        assert b not in lower
    assert "write_text" not in src
    assert LOCK_TIMESTAMP_SOURCE in src
