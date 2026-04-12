# services/runtime_state.py
from __future__ import annotations

import os
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from services.runtime_io import atomic_write_json, safe_read_json


RUNTIME_DIR = Path("data/runtime")
STATE_PATH = RUNTIME_DIR / "runtime_state.json"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _mins_ago(ts_iso: Optional[str]) -> Optional[int]:
    if not ts_iso:
        return None
    try:
        s = ts_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return int((_utcnow() - dt).total_seconds() // 60)
    except Exception:
        return None


def _default_identity() -> Dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "user": os.getenv("USERNAME") or os.getenv("USER") or None,
        "pid": os.getpid(),
    }


def load_runtime_state() -> Dict[str, Any]:
    return safe_read_json(STATE_PATH)


def write_heartbeat(
    *,
    automation_enabled: bool,
    mode: str = "NORMAL",
    phase: str = "1.5",
    note: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Writes a heartbeat without implying a full cycle. Safe to call frequently (e.g., every minute).
    """
    prev = load_runtime_state()
    state = {
        **prev,
        "ts_utc": _iso(_utcnow()),
        "automation_enabled": bool(automation_enabled),
        "mode": str(mode).upper(),
        "phase": str(phase),
        "note": note,
        "identity": _default_identity(),
    }
    if extra:
        state["extra"] = extra
    atomic_write_json(STATE_PATH, state)
    return state


@dataclass
class CycleContext:
    cycle_id: str
    started_ts_utc: str


def cycle_start(
    *,
    automation_enabled: bool,
    mode: str = "NORMAL",
    reason: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> CycleContext:
    """
    Mark the start of an automation cycle. Returns a context with cycle_id.
    """
    cid = str(uuid.uuid4())
    now = _utcnow()
    prev = load_runtime_state()

    state = {
        **prev,
        "ts_utc": _iso(now),
        "automation_enabled": bool(automation_enabled),
        "mode": str(mode).upper(),
        "cycle": {
            "id": cid,
            "status": "RUNNING",
            "started_ts_utc": _iso(now),
            "ended_ts_utc": None,
            "duration_sec": None,
            "success": None,
            "reason": reason,
        },
        "identity": _default_identity(),
    }
    if extra:
        state["cycle"]["extra"] = extra

    atomic_write_json(STATE_PATH, state)
    return CycleContext(cycle_id=cid, started_ts_utc=_iso(now) or "")


def cycle_end(
    ctx: CycleContext,
    *,
    success: bool,
    reason: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Mark end of cycle. Keeps last cycle state in runtime_state.json.
    """
    now = _utcnow()
    prev = load_runtime_state()
    prev_cycle = (prev.get("cycle") or {}) if isinstance(prev.get("cycle"), dict) else {}

    started_iso = prev_cycle.get("started_ts_utc") or ctx.started_ts_utc
    started_dt = None
    try:
        if started_iso:
            started_dt = datetime.fromisoformat(started_iso.replace("Z", "+00:00"))
    except Exception:
        started_dt = None

    duration = None
    if started_dt:
        duration = int((now - started_dt).total_seconds())

    state = {
        **prev,
        "ts_utc": _iso(now),
        "cycle": {
            "id": prev_cycle.get("id") or ctx.cycle_id,
            "status": "DONE",
            "started_ts_utc": started_iso,
            "ended_ts_utc": _iso(now),
            "duration_sec": duration,
            "success": bool(success),
            "reason": reason,
        },
        "identity": _default_identity(),
    }
    if extra:
        state["cycle"]["extra"] = extra

    atomic_write_json(STATE_PATH, state)
    return state


# ----------------------------
# Optional CLI (handy)
# ----------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Triton runtime heartbeat/state writer")
    p.add_argument("--on", action="store_true", help="Set automation_enabled=True")
    p.add_argument("--off", action="store_true", help="Set automation_enabled=False")
    p.add_argument("--mode", default="NORMAL", help="NORMAL/DEFENSIVE/LOCKDOWN")
    p.add_argument("--note", default="", help="Optional note")
    args = p.parse_args()

    enabled = True
    if args.off:
        enabled = False
    if args.on:
        enabled = True

    st = write_heartbeat(automation_enabled=enabled, mode=args.mode, note=args.note)
    print(
        f"[runtime_state] wrote {STATE_PATH.as_posix()} automation_enabled={st.get('automation_enabled')} mode={st.get('mode')}"
    )
