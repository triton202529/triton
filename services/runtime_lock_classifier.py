"""
runtime_lock_classifier.py
Process-aware lock health classification for loops (Phase 196B).

Uses lock metadata, matching processes, and heartbeat age — not lock mtime alone.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"

WATCHDOG_LOCK = LIVE / "watchdog_loop.lock"
CONTINUOUS_LOCK = LIVE / "continuous_loop.lock"
WATCHDOG_STATUS = RESULTS / "watchdog_status.json"
WATCHDOG_STATE = LIVE / "watchdog_alert_state.json"

WATCHDOG_GRACE_MINUTES = 5.0
CONTINUOUS_INTERVAL_MINUTES = 45.0
CONTINUOUS_GRACE_MINUTES = max(60.0, CONTINUOUS_INTERVAL_MINUTES * 1.5)

CLASSIFICATIONS = (
    "LOCK_HEALTHY_ACTIVE",
    "LOCK_TIMESTAMP_STALE_BUT_PROCESS_ACTIVE",
    "STALE_ACTIVE_LOCK",
    "STALE_ORPHAN_LOCK",
    "NO_LOCK_RUNNING",
    "NO_PROCESS_NO_LOCK",
    "CORRUPT_LOCK",
    "MISSING_ACTIVE_LOCK",
)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse UTC/Z/offset JSON timestamps (heartbeat / status artifacts)."""
    from services.command_center_status_coherence import parse_utc_timestamp

    return parse_utc_timestamp(value)


def _parse_lock_ts(value: Any) -> Optional[datetime]:
    """Parse lock started_at/refreshed_at (naive local from loop_lock_utils.ps1)."""
    from services.command_center_status_coherence import parse_lock_timestamp

    return parse_lock_timestamp(value)


def _minutes_since(dt: Optional[datetime], now: Optional[datetime] = None) -> Optional[float]:
    if dt is None:
        return None
    now = now or datetime.now(timezone.utc)
    return (now - dt).total_seconds() / 60.0


def _heartbeat_age_minutes(loop_kind: str) -> Optional[float]:
    now = datetime.now(timezone.utc)
    if loop_kind == "watchdog":
        wd = _load_json(WATCHDOG_STATUS)
        hb = wd.get("heartbeat") if isinstance(wd.get("heartbeat"), dict) else {}
        ts = _parse_ts(hb.get("timestamp") or wd.get("timestamp"))
        return _minutes_since(ts, now)
    state = _load_json(WATCHDOG_STATE)
    ts = _parse_ts(state.get("heartbeat_refresh_ts") or state.get("last_cycle_ts"))
    return _minutes_since(ts, now)


def _pipeline_heartbeat_age_minutes() -> Optional[float]:
    hb_path = RESULTS / "heartbeat.json"
    if not hb_path.is_file():
        return None
    try:
        data = json.loads(hb_path.read_text(encoding="utf-8"))
        ts = _parse_ts(
            data.get("timestamp") or data.get("ts") or data.get("updated_at") or data.get("time")
        )
        return _minutes_since(ts)
    except Exception:
        return None


def _effective_heartbeat_age_minutes(
    loop_kind: str,
    heartbeat_age_minutes: Optional[float],
) -> Optional[float]:
    """Prefer freshest evidence: loop heartbeat, pipeline heartbeat, runtime watchdog."""
    ages: List[float] = []
    if heartbeat_age_minutes is not None:
        ages.append(float(heartbeat_age_minutes))
    pipeline_age = _pipeline_heartbeat_age_minutes()
    if pipeline_age is not None:
        ages.append(float(pipeline_age))
    if loop_kind == "continuous":
        wd = _load_json(WATCHDOG_STATUS)
        hb = wd.get("heartbeat") if isinstance(wd.get("heartbeat"), dict) else {}
        rt_age = _minutes_since(_parse_ts(hb.get("timestamp") or wd.get("timestamp")))
        if rt_age is not None:
            ages.append(float(rt_age))
    return min(ages) if ages else None


def classify_runtime_lock(
    lock: Dict[str, Any],
    process_running: bool,
    process_pids: Optional[List[int]] = None,
    *,
    loop_kind: str = "continuous",
    grace_minutes: Optional[float] = None,
    heartbeat_age_minutes: Optional[float] = None,
) -> str:
    """
    Classify lock health using process + heartbeat awareness.

    LOCK_HEALTHY_ACTIVE: lock exists, matching process, heartbeat inside grace.
    STALE_ACTIVE_LOCK: process exists but heartbeat beyond grace.
    STALE_ORPHAN_LOCK: lock exists, no matching process, stale.
    NO_LOCK_RUNNING: no lock file but process running.
    NO_PROCESS_NO_LOCK: neither lock nor process.
    """
    if lock.get("corrupt"):
        return "CORRUPT_LOCK"

    pids = process_pids or []
    lock_exists = bool(lock.get("exists"))
    process_active = bool(process_running and pids)
    lock_pid = lock.get("pid")
    pid_match = lock_pid is not None and lock_pid in pids

    if grace_minutes is None:
        grace_minutes = (
            WATCHDOG_GRACE_MINUTES if loop_kind == "watchdog" else CONTINUOUS_GRACE_MINUTES
        )

    if heartbeat_age_minutes is None:
        heartbeat_age_minutes = _heartbeat_age_minutes(loop_kind)
    effective_hb_age = _effective_heartbeat_age_minutes(loop_kind, heartbeat_age_minutes)

    if not lock_exists and not process_active:
        return "NO_PROCESS_NO_LOCK"
    if not lock_exists and process_active:
        return "NO_LOCK_RUNNING"
    if lock_exists and not process_active:
        return "STALE_ORPHAN_LOCK"

    # lock exists and process active
    # Lock started_at/refreshed_at ages are ownership/cycle metadata — not heartbeats.
    # Recompute lock age with source-aware local parsing when raw timestamps exist.
    lock_ts_age = lock.get("lock_freshness_age_minutes") or lock.get("lock_age_minutes")
    raw_refresh = lock.get("refreshed_at") or lock.get("started_at")
    if raw_refresh:
        recomputed = _minutes_since(_parse_lock_ts(raw_refresh))
        if recomputed is not None:
            lock_ts_age = recomputed
    hb_fresh = effective_hb_age is not None and effective_hb_age <= grace_minutes

    # Matching live owner + fresh heartbeat => healthy; static lock age is informational.
    if hb_fresh and (pid_match or process_active):
        return "LOCK_HEALTHY_ACTIVE"

    if pid_match and effective_hb_age is not None and effective_hb_age <= grace_minutes:
        return "LOCK_HEALTHY_ACTIVE"
    if process_active and (pid_match or not lock_pid):
        if effective_hb_age is None or effective_hb_age <= grace_minutes:
            return "LOCK_HEALTHY_ACTIVE"
        # Preserve classification label for diagnostics, but only when HB itself is aging
        # and lock metadata (correctly parsed) is also old — never for fresh HB.
        if (
            lock_ts_age is not None
            and float(lock_ts_age) > grace_minutes
            and effective_hb_age is not None
            and effective_hb_age <= grace_minutes * 2
            and loop_kind == "continuous"
            and not hb_fresh
        ):
            return "LOCK_TIMESTAMP_STALE_BUT_PROCESS_ACTIVE"
        return "STALE_ACTIVE_LOCK"
    if lock_exists and process_active:
        return "STALE_ACTIVE_LOCK"
    return "MISSING_ACTIVE_LOCK"


def is_lock_healthy(classification: str) -> bool:
    return classification in (
        "LOCK_HEALTHY_ACTIVE",
        "LOCK_TIMESTAMP_STALE_BUT_PROCESS_ACTIVE",
        "FRESH_ACTIVE_LOCK",
    )


def audit_runtime_locks(
    *,
    watchdog_lock: Optional[Dict[str, Any]] = None,
    continuous_lock: Optional[Dict[str, Any]] = None,
    watchdog_running: bool = False,
    continuous_running: bool = False,
    watchdog_pids: Optional[List[int]] = None,
    continuous_pids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Build runtime lock audit payload for dashboard and validators."""
    from services.phase_195h_operational_readiness_freeze import _parse_lock

    wd_lock = watchdog_lock if watchdog_lock is not None else _parse_lock(WATCHDOG_LOCK)
    cl_lock = continuous_lock if continuous_lock is not None else _parse_lock(CONTINUOUS_LOCK)
    wd_hb = _heartbeat_age_minutes("watchdog")
    cl_hb = _heartbeat_age_minutes("continuous")

    wd_class = classify_runtime_lock(
        wd_lock,
        watchdog_running,
        watchdog_pids or [],
        loop_kind="watchdog",
        heartbeat_age_minutes=wd_hb,
    )
    cl_class = classify_runtime_lock(
        cl_lock,
        continuous_running,
        continuous_pids or [],
        loop_kind="continuous",
        heartbeat_age_minutes=cl_hb,
    )

    return {
        "watchdog_process_running": watchdog_running,
        "continuous_process_running": continuous_running,
        "watchdog_heartbeat_age_minutes": wd_hb,
        "continuous_heartbeat_age_minutes": cl_hb,
        "watchdog_lock_classification": wd_class,
        "continuous_lock_classification": cl_class,
        "watchdog_lock_healthy": is_lock_healthy(wd_class),
        "continuous_lock_healthy": is_lock_healthy(cl_class),
        "loops_runtime_healthy": is_lock_healthy(wd_class)
        and is_lock_healthy(cl_class)
        and watchdog_running
        and continuous_running,
        "watchdog_grace_minutes": WATCHDOG_GRACE_MINUTES,
        "continuous_grace_minutes": CONTINUOUS_GRACE_MINUTES,
    }
