"""Shared start-to-start continuous-loop cadence contract (Phase 198T-AT).

Authoritative rule:
    next_due = cycle_started_at + interval_seconds
    remaining_wait = max(0, next_due - now)

IntervalMinutes=45 means 2700 seconds from one cycle start to the next
cycle start. Post-completion sleep is the residual, never a second full
interval. Overdue loops run at most one next eligible cycle (no catch-up
burst). Host suspend/resume uses current wall-clock remaining_wait.

Does not change SELL packet TTL, SELL authority TTL, or live-execution
policy. Artifact timestamps are UTC; ET labels are derived explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERVAL_SECONDS = 2700
SCHEMA_VERSION = "198T-AT-CADENCE-1"
CADENCE_RELATIVE = Path("data") / "results" / "continuous_loop_cadence.json"
ET = ZoneInfo("America/New_York")

PHASE_ACTIVE_CYCLE = "ACTIVE_CYCLE"
PHASE_INTERRUPTIBLE_WAIT = "INTERRUPTIBLE_WAIT"
PHASE_BETWEEN_CYCLES = "BETWEEN_CYCLES"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("timestamp_must_be_timezone_aware")
    return ts.astimezone(timezone.utc)


def iso_utc(ts: datetime) -> str:
    return ensure_utc(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_et(ts: datetime) -> str:
    return ensure_utc(ts).astimezone(ET).isoformat()


def compute_next_due(
    cycle_started_at: datetime,
    *,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> datetime:
    start = ensure_utc(cycle_started_at)
    return start + timedelta(seconds=int(interval_seconds))


def remaining_wait_seconds(
    cycle_started_at: datetime,
    *,
    now: Optional[datetime] = None,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> int:
    due = compute_next_due(cycle_started_at, interval_seconds=interval_seconds)
    clock = ensure_utc(now or utc_now())
    leftover = (due - clock).total_seconds()
    if leftover <= 0:
        return 0
    return int(leftover)


def overdue_missed_slots(
    cycle_started_at: datetime,
    *,
    now: Optional[datetime] = None,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> int:
    """How many interval slots were missed. Catch-up still runs at most one cycle."""
    due = compute_next_due(cycle_started_at, interval_seconds=interval_seconds)
    clock = ensure_utc(now or utc_now())
    if clock <= due:
        return 0
    lag = (clock - due).total_seconds()
    return int(lag // int(interval_seconds)) + 1


def catch_up_cycles_permitted() -> int:
    """Hard cap: never replay a queue of missed historical cycles."""
    return 1


def cadence_artifact_path(root: Optional[Path] = None) -> Path:
    return (root or ROOT) / CADENCE_RELATIVE


def build_cadence_record(
    *,
    cycle_started_at: datetime,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    cycle: int = 0,
    phase: str = PHASE_ACTIVE_CYCLE,
    now: Optional[datetime] = None,
    remaining: Optional[int] = None,
) -> Dict[str, Any]:
    start = ensure_utc(cycle_started_at)
    due = compute_next_due(start, interval_seconds=interval_seconds)
    clock = ensure_utc(now or utc_now())
    wait = (
        remaining
        if remaining is not None
        else remaining_wait_seconds(start, now=clock, interval_seconds=interval_seconds)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "timezone": "UTC",
        "cycle": int(cycle),
        "phase": str(phase),
        "heartbeat_phase": str(phase),
        "interval_seconds": int(interval_seconds),
        "interval_minutes": int(interval_seconds) / 60.0,
        "cycle_started_at_utc": iso_utc(start),
        "cycle_started_at_et": iso_et(start),
        "next_due_at_utc": iso_utc(due),
        "next_due_at_et": iso_et(due),
        "remaining_wait_seconds": int(wait),
        "recorded_at_utc": iso_utc(clock),
        "recorded_at_et": iso_et(clock),
        "authoritative": True,
        "catch_up_cycles_permitted": catch_up_cycles_permitted(),
        "live_execution_permitted": False,
    }


def write_cadence_artifact(record: Dict[str, Any], *, root: Optional[Path] = None) -> Path:
    path = cadence_artifact_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_cadence_artifact(root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = cadence_artifact_path(root)
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    return doc


def parse_utc_field(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def authoritative_next_due_utc(root: Optional[Path] = None) -> Optional[datetime]:
    doc = load_cadence_artifact(root)
    if not doc:
        return None
    return parse_utc_field(doc.get("next_due_at_utc"))


def _cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="198T-AT continuous loop cadence helper")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--record-start", action="store_true")
    parser.add_argument("--remaining-wait", action="store_true")
    parser.add_argument("--phase", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root)
    now = utc_now()
    if args.record_start:
        rec = build_cadence_record(
            cycle_started_at=now,
            interval_seconds=args.interval_seconds,
            cycle=args.cycle,
            phase=args.phase or PHASE_ACTIVE_CYCLE,
            now=now,
        )
        write_cadence_artifact(rec, root=root)
        print(json.dumps(rec, sort_keys=True))
        return 0
    if args.remaining_wait:
        doc = load_cadence_artifact(root) or {}
        started = parse_utc_field(doc.get("cycle_started_at_utc"))
        interval = int(doc.get("interval_seconds") or args.interval_seconds)
        if started is None:
            wait = int(args.interval_seconds)
            due = now + timedelta(seconds=wait)
            payload = {
                "remaining_wait_seconds": wait,
                "next_due_at_utc": iso_utc(due),
                "timezone": "UTC",
                "reason": "cadence_artifact_missing_fallback_full_interval",
            }
            print(json.dumps(payload, sort_keys=True))
            return 0
        wait = remaining_wait_seconds(started, now=now, interval_seconds=interval)
        rec = build_cadence_record(
            cycle_started_at=started,
            interval_seconds=interval,
            cycle=int(doc.get("cycle") or args.cycle or 0),
            phase=args.phase or PHASE_INTERRUPTIBLE_WAIT,
            now=now,
            remaining=wait,
        )
        write_cadence_artifact(rec, root=root)
        print(json.dumps(rec, sort_keys=True))
        return 0
    parser.error("specify --record-start or --remaining-wait")
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
