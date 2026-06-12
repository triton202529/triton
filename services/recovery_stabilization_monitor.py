"""
TRITON Phase 197 — Recovery Stabilization Verification & Intelligent Health Confirmation.

Replaces fixed short-timeout recovery checks with adaptive operational readiness
assessment. Does NOT modify trading, governance, lifecycle, allocation, execution,
broker, or risk logic.

Outputs (data/results/):
    recovery_stabilization_status.json
    recovery_stabilization_history.csv
    recovery_verification_events.csv

Run:
    python -m services.recovery_stabilization_monitor
    python -m services.recovery_stabilization_monitor --observe-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from services.continuous_loop_recovery_manager import (
    DEFAULT_INTERVAL_MINUTES,
    LOCK_PATH,
    LOOP_LOG_DIR,
    WATCHDOG_STATE,
    _heartbeat_snapshot,
    _is_loop_process_running,
    _latest_loop_log,
    _minutes_since,
    _now_iso,
    _parse_lock,
    _parse_loop_log,
    _parse_ts,
    _read_json,
)

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "recovery_stabilization_status.json"
OUT_HISTORY = RESULTS / "recovery_stabilization_history.csv"
OUT_EVENTS = RESULTS / "recovery_verification_events.csv"
ACTIONS_CSV = RESULTS / "hung_loop_recovery_actions.csv"

POLL_INTERVAL_SECONDS = 5
MIN_OBSERVATION_SECONDS = 30
MAX_OBSERVATION_SECONDS = 60
STARTUP_TIMEOUT_SECONDS = 60
HEALTHY_CHECKS_REQUIRED = 3
INIT_WINDOW_SECONDS = 15

StartupState = str  # STARTING | WARMING_UP | HEALTHY | DEGRADED | FAILED
ReadinessClass = str  # UNHEALTHY | RECOVERING | STABLE | PRODUCTION_READY


def _log(msg: str) -> None:
    print(f"[recovery_stabilization_monitor] {msg}", flush=True)


def _snapshot(
    *,
    before_refresh: Optional[str] = None,
    baseline_log_mtime: Optional[float] = None,
    baseline_cycle_count: int = 0,
) -> Dict[str, Any]:
    running, pids = _is_loop_process_running()
    lock = _parse_lock(LOCK_PATH)
    state = _read_json(WATCHDOG_STATE)
    hb = _heartbeat_snapshot()
    log_path = _latest_loop_log()
    log_info = _parse_loop_log(log_path)
    refresh = state.get("heartbeat_refresh_ts")
    log_mtime = log_path.stat().st_mtime if log_path and log_path.is_file() else None
    log_progressed = (
        log_mtime is not None and baseline_log_mtime is not None and log_mtime > baseline_log_mtime
    ) or log_info.get("cycle_count", 0) > baseline_cycle_count

    hb_advanced = bool(
        refresh and before_refresh and refresh != before_refresh or (refresh and not before_refresh)
    )
    if before_refresh and refresh == before_refresh:
        hb_advanced = False

    return {
        "timestamp": _now_iso(),
        "process_running": running,
        "pids": pids,
        "lock_exists": lock.get("exists"),
        "lock_started_at": lock.get("started_at"),
        "heartbeat_refresh_ts": refresh,
        "heartbeat_advanced": hb_advanced,
        "heartbeat_age_minutes": hb.get("continuous_loop_refresh_age_minutes"),
        "last_cycle_ts": state.get("last_cycle_ts"),
        "log_path": str(log_path) if log_path else None,
        "log_progressed": log_progressed,
        "cycle_count": log_info.get("cycle_count", 0),
        "consecutive_ok": log_info.get("consecutive_ok", 0),
        "last_cycle_ok": log_info.get("last_cycle_ok"),
        "log_has_banner": _log_has_startup_banner(log_path),
        "log_has_cycle_start": log_info.get("cycle_count", 0) > 0,
    }


def _log_has_startup_banner(log_path: Optional[Path]) -> bool:
    if not log_path or not log_path.is_file():
        return False
    try:
        head = log_path.read_text(encoding="utf-8", errors="replace")[:4000]
        return "CONTINUOUS LOOP START" in head
    except OSError:
        return False


def _classify_startup_state(
    snap: Dict[str, Any],
    *,
    elapsed_seconds: float,
    stable_pid_checks: int,
    heartbeat_progressions: int,
) -> StartupState:
    if not snap.get("process_running"):
        if elapsed_seconds >= STARTUP_TIMEOUT_SECONDS:
            return "FAILED"
        return "FAILED" if elapsed_seconds > INIT_WINDOW_SECONDS else "STARTING"

    if (
        snap.get("lock_exists")
        and (snap.get("heartbeat_advanced") or snap.get("heartbeat_refresh_ts"))
        and (snap.get("log_progressed") or snap.get("log_has_banner"))
    ):
        if (
            stable_pid_checks >= HEALTHY_CHECKS_REQUIRED
            and heartbeat_progressions >= 1
            and (snap.get("consecutive_ok", 0) >= 1 or snap.get("log_has_cycle_start"))
        ):
            return "HEALTHY"
        return "WARMING_UP"

    if elapsed_seconds > STARTUP_TIMEOUT_SECONDS:
        if snap.get("process_running") and not snap.get("heartbeat_refresh_ts"):
            return "DEGRADED"
        return "FAILED"

    if snap.get("process_running") and elapsed_seconds <= INIT_WINDOW_SECONDS:
        return "STARTING"

    if snap.get("process_running"):
        if snap.get("lock_exists") or snap.get("log_has_banner"):
            return "WARMING_UP"
        return "DEGRADED"

    return "STARTING"


def _compute_scores(
    state: StartupState,
    snaps: List[Dict[str, Any]],
    *,
    elapsed_seconds: float,
    heartbeat_progressions: int,
    stable_pid_checks: int,
) -> Dict[str, Any]:
    last = snaps[-1] if snaps else {}
    process_score = 100.0 if last.get("process_running") else 0.0
    if stable_pid_checks >= HEALTHY_CHECKS_REQUIRED:
        process_score = min(100.0, process_score)
    elif last.get("process_running"):
        process_score = 60.0 + min(30.0, stable_pid_checks * 10.0)

    hb_score = 0.0
    if last.get("heartbeat_refresh_ts"):
        hb_score = 40.0
    if heartbeat_progressions >= 1:
        hb_score = 70.0
    if heartbeat_progressions >= 2:
        hb_score = 90.0
    age = last.get("heartbeat_age_minutes")
    if age is not None and age < 5:
        hb_score = min(100.0, hb_score + 10.0)

    startup_conf = 20.0
    if state == "STARTING":
        startup_conf = 35.0
    elif state == "WARMING_UP":
        startup_conf = 55.0 + min(25.0, stable_pid_checks * 8.0)
    elif state == "HEALTHY":
        startup_conf = 90.0
    elif state == "DEGRADED":
        startup_conf = 40.0
    elif state == "FAILED":
        startup_conf = 10.0

    cycle_score = 0.0
    if last.get("log_has_cycle_start"):
        cycle_score = 50.0
    if last.get("consecutive_ok", 0) >= 1:
        cycle_score = 80.0
    if last.get("consecutive_ok", 0) >= 2:
        cycle_score = 100.0

    recovery_conf = round(
        (process_score * 0.3 + hb_score * 0.35 + startup_conf * 0.2 + cycle_score * 0.15), 1
    )
    operational = round((process_score * 0.25 + hb_score * 0.3 + cycle_score * 0.45), 1)

    readiness: ReadinessClass = "UNHEALTHY"
    if state in ("STARTING", "WARMING_UP"):
        readiness = "RECOVERING"
    elif state == "HEALTHY" and recovery_conf >= 75:
        readiness = "STABLE"
    elif state == "HEALTHY" and recovery_conf >= 85 and cycle_score >= 80:
        readiness = "PRODUCTION_READY"
    elif state == "DEGRADED":
        readiness = "RECOVERING"
    elif state == "FAILED":
        readiness = "UNHEALTHY"

    return {
        "process_health_score": round(process_score, 1),
        "startup_confidence": round(startup_conf, 1),
        "recovery_confidence": recovery_conf,
        "operational_readiness_score": operational,
        "readiness_classification": readiness,
    }


def observe_stabilization(
    *,
    before_refresh: Optional[str] = None,
    min_seconds: int = MIN_OBSERVATION_SECONDS,
    max_seconds: int = MAX_OBSERVATION_SECONDS,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
) -> Dict[str, Any]:
    """Adaptive readiness monitoring — poll until HEALTHY or timeout."""
    started = time.monotonic()
    restart_ts = _now_iso()
    log_path = _latest_loop_log()
    baseline_mtime = log_path.stat().st_mtime if log_path and log_path.is_file() else None
    baseline_log = _parse_loop_log(log_path)
    baseline_cycles = baseline_log.get("cycle_count", 0)

    snaps: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    last_refresh = before_refresh
    heartbeat_progressions = 0
    stable_pid_checks = 0
    last_pids: Optional[List[int]] = None
    final_state: StartupState = "STARTING"
    first_hb_ts: Optional[str] = None
    first_lock_ts: Optional[str] = None
    first_cycle_ok: Optional[str] = None

    events.append(
        {
            "timestamp": restart_ts,
            "event_type": "OBSERVATION_START",
            "state": "STARTING",
            "detail": {"before_refresh": before_refresh, "max_seconds": max_seconds},
        }
    )

    while True:
        elapsed = time.monotonic() - started
        snap = _snapshot(
            before_refresh=before_refresh,
            baseline_log_mtime=baseline_mtime,
            baseline_cycle_count=baseline_cycles,
        )
        snaps.append(snap)

        refresh = snap.get("heartbeat_refresh_ts")
        if refresh and refresh != last_refresh:
            heartbeat_progressions += 1
            last_refresh = refresh
            if not first_hb_ts:
                first_hb_ts = refresh
            events.append(
                {
                    "timestamp": snap["timestamp"],
                    "event_type": "HEARTBEAT_PROGRESSION",
                    "state": final_state,
                    "detail": {"heartbeat_refresh_ts": refresh},
                }
            )

        if snap.get("lock_exists") and not first_lock_ts:
            first_lock_ts = snap.get("lock_started_at") or snap["timestamp"]
            events.append(
                {
                    "timestamp": snap["timestamp"],
                    "event_type": "LOCK_CREATED",
                    "state": final_state,
                    "detail": {"lock_started_at": first_lock_ts},
                }
            )

        if snap.get("last_cycle_ok") and not first_cycle_ok:
            first_cycle_ok = snap.get("last_cycle_ok")
            events.append(
                {
                    "timestamp": snap["timestamp"],
                    "event_type": "FIRST_CYCLE_OK",
                    "state": final_state,
                    "detail": {"last_cycle_ok": first_cycle_ok},
                }
            )

        pids = snap.get("pids") or []
        if pids and pids == last_pids:
            stable_pid_checks += 1
        else:
            stable_pid_checks = 1 if pids else 0
        last_pids = list(pids) if pids else None

        final_state = _classify_startup_state(
            snap,
            elapsed_seconds=elapsed,
            stable_pid_checks=stable_pid_checks,
            heartbeat_progressions=heartbeat_progressions,
        )

        events.append(
            {
                "timestamp": snap["timestamp"],
                "event_type": "STATE_CHECK",
                "state": final_state,
                "detail": {
                    "elapsed_seconds": round(elapsed, 1),
                    "pids": pids,
                    "lock_exists": snap.get("lock_exists"),
                },
            }
        )

        if final_state == "HEALTHY" and elapsed >= min_seconds:
            break
        if final_state == "FAILED":
            break
        if elapsed >= max_seconds:
            if final_state in ("WARMING_UP", "DEGRADED") and snap.get("process_running"):
                final_state = "DEGRADED"
            break

        time.sleep(POLL_INTERVAL_SECONDS)

    elapsed_total = round(time.monotonic() - started, 1)
    scores = _compute_scores(
        final_state,
        snaps,
        elapsed_seconds=elapsed_total,
        heartbeat_progressions=heartbeat_progressions,
        stable_pid_checks=stable_pid_checks,
    )

    last = snaps[-1] if snaps else {}
    warm_up_complete = (
        last.get("process_running")
        and last.get("lock_exists")
        and heartbeat_progressions >= 1
        and elapsed_total >= min_seconds
        and stable_pid_checks >= 2
    )

    recovery_successful = (
        warm_up_complete
        and stable_pid_checks >= HEALTHY_CHECKS_REQUIRED
        and final_state in ("HEALTHY", "WARMING_UP")
        and scores["readiness_classification"] in ("STABLE", "PRODUCTION_READY", "RECOVERING")
    )

    hb_confirmed = heartbeat_progressions >= 1

    return {
        "restart_timestamp": restart_ts,
        "observation_seconds": elapsed_total,
        "final_state": final_state,
        "startup_timeline": {
            "restart": restart_ts,
            "first_lock": first_lock_ts,
            "first_heartbeat": first_hb_ts,
            "first_cycle_ok": first_cycle_ok,
            "stabilization_complete": snap["timestamp"] if final_state == "HEALTHY" else None,
        },
        "snapshots": snaps,
        "events": events,
        "scores": scores,
        "warm_up_complete": warm_up_complete,
        "recovery_successful": recovery_successful,
        "process_running": last.get("process_running", False),
        "pids": last.get("pids") or [],
        "lock_exists": last.get("lock_exists", False),
        "heartbeat_refresh_confirmed": hb_confirmed,
        "heartbeat_refresh_ts": last.get("heartbeat_refresh_ts"),
        "heartbeat_progressions": heartbeat_progressions,
        "stable_pid_checks": stable_pid_checks,
        "cycle_advanced": (last.get("cycle_count", 0) > baseline_cycles),
        "interval_minutes": interval_minutes,
        "forensics": _build_forensics(last, elapsed_total, before_refresh),
    }


def _build_forensics(
    snap: Dict[str, Any], elapsed: float, before_refresh: Optional[str]
) -> Dict[str, Any]:
    return {
        "restart_timestamp": snap.get("timestamp"),
        "new_pids": snap.get("pids"),
        "startup_duration_seconds": elapsed,
        "heartbeat_creation_time": snap.get("heartbeat_refresh_ts"),
        "first_successful_cycle": snap.get("last_cycle_ok"),
        "lock_creation_time": snap.get("lock_started_at"),
        "log_path": snap.get("log_path"),
        "log_progressed": snap.get("log_progressed"),
        "environment_note": (
            "TRITON_ENABLE_PAPER_EXECUTION=1 set inside run_continuous_loop.ps1 at startup"
        ),
        "before_refresh": before_refresh,
        "stderr_stdout_note": "Recovery restart still uses DEVNULL — capture recommended",
    }


def adaptive_verify_recovery(
    before_refresh: Optional[str],
    *,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    min_seconds: int = MIN_OBSERVATION_SECONDS,
    max_seconds: int = MAX_OBSERVATION_SECONDS,
) -> Dict[str, Any]:
    """Drop-in replacement for fixed-timeout _verify_recovery."""
    result = observe_stabilization(
        before_refresh=before_refresh,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        interval_minutes=interval_minutes,
    )
    return {
        "process_running": result["process_running"],
        "pids": result["pids"],
        "lock_exists": result["lock_exists"],
        "heartbeat_refresh_confirmed": result["heartbeat_refresh_confirmed"],
        "heartbeat_refresh_ts": result["heartbeat_refresh_ts"],
        "recovery_successful": result["recovery_successful"],
        "stabilization_state": result["final_state"],
        "observation_seconds": result["observation_seconds"],
        "warm_up_complete": result["warm_up_complete"],
        "readiness_classification": result["scores"]["readiness_classification"],
        "recovery_confidence": result["scores"]["recovery_confidence"],
        "heartbeat_progressions": result["heartbeat_progressions"],
        "cycle_advanced": result["cycle_advanced"],
        "startup_timeline": result["startup_timeline"],
    }


def assess_current_loop(interval_minutes: int = DEFAULT_INTERVAL_MINUTES) -> Dict[str, Any]:
    """Assess live loop readiness without blocking observation."""
    snap = _snapshot()
    elapsed = 0.0
    state = _classify_startup_state(
        snap,
        elapsed_seconds=elapsed,
        stable_pid_checks=HEALTHY_CHECKS_REQUIRED if snap.get("process_running") else 0,
        heartbeat_progressions=1 if snap.get("heartbeat_refresh_ts") else 0,
    )
    log_info = _parse_loop_log(_latest_loop_log())
    cycle_age = _minutes_since(log_info.get("last_cycle_ok_dt"))
    if snap.get("process_running") and cycle_age is not None and cycle_age > interval_minutes + 15:
        state = "DEGRADED"

    scores = _compute_scores(
        state,
        [snap],
        elapsed_seconds=0,
        heartbeat_progressions=1 if snap.get("heartbeat_refresh_ts") else 0,
        stable_pid_checks=HEALTHY_CHECKS_REQUIRED if snap.get("process_running") else 0,
    )
    return {
        "assessed_at": _now_iso(),
        "current_state": state,
        "snapshot": snap,
        "scores": scores,
        "cycle_age_minutes": cycle_age,
    }


def _load_last_recovery() -> Dict[str, Any]:
    if not ACTIONS_CSV.is_file():
        return {}
    df = pd.read_csv(ACTIONS_CSV)
    if df.empty:
        return {}
    row = df.iloc[-1].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def _success_rate() -> Dict[str, Any]:
    if not OUT_HISTORY.is_file():
        if not ACTIONS_CSV.is_file():
            return {"attempts": 0, "successes": 0, "failures": 0, "success_pct": None}
        df = pd.read_csv(ACTIONS_CSV)
        attempts = len(df[df["action_type"].isin(["RECOVERY_ATTEMPT", "RECOVERY_SUCCESS"])])
        successes = len(df[df["result"] == "RECOVERY_SUCCESS"])
        failures = len(df[df["result"] == "RECOVERY_FAILED"])
        pct = round(successes / attempts * 100, 1) if attempts else None
        return {
            "attempts": attempts,
            "successes": successes,
            "failures": failures,
            "success_pct": pct,
        }

    df = pd.read_csv(OUT_HISTORY)
    attempts = len(df)
    successes = len(df[df.get("recovery_success", False) == True])  # noqa: E712
    failures = len(df[df.get("recovery_success", False) == False])  # noqa: E712
    pct = round(successes / attempts * 100, 1) if attempts else None
    return {"attempts": attempts, "successes": successes, "failures": failures, "success_pct": pct}


def _executive_report(
    current: Dict[str, Any],
    last_recovery: Dict[str, Any],
    observation: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    state = current.get("current_state", "UNKNOWN")
    scores = current.get("scores") or {}
    readiness = scores.get("readiness_classification", "UNHEALTHY")
    obs = observation or {}

    truly_succeeded = (
        bool(obs.get("recovery_successful")) or last_recovery.get("result") == "RECOVERY_SUCCESS"
    )
    warming = state in ("STARTING", "WARMING_UP") or (
        obs.get("final_state") in ("STARTING", "WARMING_UP")
    )
    stable = state == "HEALTHY" and readiness in ("STABLE", "PRODUCTION_READY")

    return {
        "report_type": "RECOVERY_STABILIZATION_REPORT",
        "recovery_truly_succeeded": truly_succeeded,
        "still_warming_up": warming,
        "loop_stable": stable,
        "initialization_seconds": obs.get("observation_seconds")
        or current.get("snapshot", {}).get("lock_age_minutes"),
        "safe_for_autonomous_operation": stable and readiness == "PRODUCTION_READY",
        "evidence": {
            "current_state": state,
            "readiness_classification": readiness,
            "recovery_confidence": scores.get("recovery_confidence"),
            "last_recovery_result": last_recovery.get("result"),
            "observation_final_state": obs.get("final_state"),
            "heartbeat_progressions": obs.get("heartbeat_progressions"),
            "pids": current.get("snapshot", {}).get("pids"),
        },
        "executive_questions": {
            "has_loop_actually_recovered": truly_succeeded or stable,
            "running_vs_operational": (
                "OPERATIONAL"
                if stable
                else (
                    "WARMING_UP"
                    if warming
                    else (
                        "RUNNING_ONLY"
                        if current.get("snapshot", {}).get("process_running")
                        else "NOT_RUNNING"
                    )
                )
            ),
            "healthy_startup_duration_seconds": f"{MIN_OBSERVATION_SECONDS}-{MAX_OBSERVATION_SECONDS}",
            "startup_failure_cause": (
                obs.get("final_state") if obs.get("final_state") == "FAILED" else None
            ),
            "autonomous_recovery_certifiable": stable
            and scores.get("recovery_confidence", 0) >= 75,
        },
    }


def _append_history(row: Dict[str, Any]) -> None:
    if OUT_HISTORY.is_file():
        df = pd.read_csv(OUT_HISTORY)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(OUT_HISTORY, index=False, encoding="utf-8")


def _append_events(events: List[Dict[str, Any]]) -> None:
    if not events:
        return
    rows = []
    for ev in events:
        rows.append(
            {
                "timestamp": ev.get("timestamp"),
                "event_type": ev.get("event_type"),
                "state": ev.get("state"),
                "detail": json.dumps(ev.get("detail") or {}, default=str),
            }
        )
    new_df = pd.DataFrame(rows)
    if OUT_EVENTS.is_file():
        existing = pd.read_csv(OUT_EVENTS)
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(OUT_EVENTS, index=False, encoding="utf-8")


def run_monitor(*, observe: bool = False, observe_seconds: Optional[int] = None) -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    last_recovery = _load_last_recovery()
    current = assess_current_loop()

    observation: Optional[Dict[str, Any]] = None
    if observe:
        max_s = observe_seconds or MAX_OBSERVATION_SECONDS
        before = _read_json(WATCHDOG_STATE).get("heartbeat_refresh_ts")
        observation = observe_stabilization(before_refresh=before, max_seconds=max_s)
        _append_events(observation.get("events") or [])
    _append_history(
        {
            "timestamp": ts,
            "mode": "live_observation" if observation else "snapshot",
            "final_state": (
                observation.get("final_state") if observation else current.get("current_state")
            ),
            "readiness_classification": (
                (observation.get("scores") or {}).get("readiness_classification")
                if observation
                else (current.get("scores") or {}).get("readiness_classification")
            ),
            "recovery_confidence": (
                (observation.get("scores") or {}).get("recovery_confidence")
                if observation
                else (current.get("scores") or {}).get("recovery_confidence")
            ),
            "recovery_success": (observation or {}).get("recovery_successful"),
            "observation_seconds": (observation or {}).get("observation_seconds"),
            "pids": ",".join(
                str(p)
                for p in (
                    (observation or {}).get("pids")
                    or (current.get("snapshot") or {}).get("pids")
                    or []
                )
            ),
        }
    )

    stats = _success_rate()
    executive = _executive_report(current, last_recovery, observation)

    report = {
        "generated_at": ts,
        "phase": "197",
        "mode": "recovery_stabilization_monitor",
        "disclaimer": (
            "Intelligent recovery readiness assessment only — does NOT modify governance, "
            "lifecycle, allocation, execution strategy, broker logic, or risk controls."
        ),
        "current_recovery_state": current.get("current_state"),
        "current_assessment": current,
        "last_recovery_attempt": last_recovery,
        "live_observation": observation,
        "startup_timeline": (observation or {}).get("startup_timeline") or {},
        "warm_up_progress": {
            "final_state": (observation or {}).get("final_state") or current.get("current_state"),
            "warm_up_complete": (observation or {}).get("warm_up_complete"),
            "heartbeat_progressions": (observation or {}).get("heartbeat_progressions"),
            "observation_seconds": (observation or {}).get("observation_seconds"),
        },
        "confidence_scores": current.get("scores") or (observation or {}).get("scores"),
        "startup_forensics": (observation or {}).get("forensics")
        or _build_forensics(current.get("snapshot") or {}, 0, None),
        "recovery_success_rate": stats,
        "executive_recommendation": executive,
        "summary": {
            "current_state": current.get("current_state"),
            "readiness_classification": (current.get("scores") or {}).get(
                "readiness_classification"
            ),
            "recovery_truly_succeeded": executive.get("recovery_truly_succeeded"),
            "loop_stable": executive.get("loop_stable"),
            "autonomous_recovery_certifiable": executive.get("executive_questions", {}).get(
                "autonomous_recovery_certifiable"
            ),
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(
        f"state={current.get('current_state')} readiness="
        f"{(current.get('scores') or {}).get('readiness_classification')} "
        f"observe={observe}"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TRITON recovery stabilization verification monitor"
    )
    parser.add_argument(
        "--observe",
        action="store_true",
        help=f"Run adaptive {MIN_OBSERVATION_SECONDS}-{MAX_OBSERVATION_SECONDS}s observation window",
    )
    parser.add_argument(
        "--observe-seconds",
        type=int,
        default=None,
        help="Override max observation seconds (default 60)",
    )
    args = parser.parse_args()
    run_monitor(observe=args.observe, observe_seconds=args.observe_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
