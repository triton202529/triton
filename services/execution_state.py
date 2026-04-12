# services/execution_state.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# Project root = .../services/../
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXEC_STATE_PATH = RESULTS_DIR / "execution_state.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: Any) -> Optional[datetime]:
    if not s:
        return None
    try:
        ss = str(s).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


DEFAULT_STATE: Dict[str, Any] = {
    # Hard stop (non-negotiable)
    "kill_switch": True,
    "kill_reason": "Maintenance / freeze trading",
    # Time-box arming (permission expires automatically)
    "armed_until_utc": "",
    # Capital preservation caps
    "max_notional_usd": 500.0,  # safest default
    "max_qty": 25,  # safest default
    "max_orders_per_run": 20,  # for CSV runs
    # Limit sanity: reject limits too far from latest price
    # Example: 0.20 => limit must be within ±20% of latest
    "limit_price_guard_pct": 0.20,
    "updated_at_utc": "",
}


def load_state() -> Dict[str, Any]:
    if not EXEC_STATE_PATH.exists() or EXEC_STATE_PATH.stat().st_size == 0:
        st = dict(DEFAULT_STATE)
        st["updated_at_utc"] = _iso(_utc_now())
        save_state(st)
        return st

    try:
        data = json.loads(EXEC_STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("execution_state.json must be an object")
    except Exception:
        # If corrupted, fall back safely (kill switch ON)
        data = dict(DEFAULT_STATE)

    # Fill missing defaults
    out = dict(DEFAULT_STATE)
    out.update(data)
    return out


def save_state(state: Dict[str, Any]) -> None:
    s = dict(state)
    s["updated_at_utc"] = _iso(_utc_now())
    EXEC_STATE_PATH.write_text(json.dumps(s, indent=2), encoding="utf-8")


def set_kill_switch(on: bool, reason: str = "") -> Dict[str, Any]:
    st = load_state()
    st["kill_switch"] = bool(on)
    if reason is not None:
        st["kill_reason"] = str(reason).strip()
    save_state(st)
    return st


def disarm(reason: str = "") -> Dict[str, Any]:
    st = load_state()
    st["armed_until_utc"] = ""
    if reason:
        st["kill_reason"] = str(reason).strip()
    save_state(st)
    return st


def arm(minutes: int, reason: str = "") -> Dict[str, Any]:
    """
    Arms trading for N minutes (time-boxed). Does NOT automatically disable kill switch.
    You must explicitly switch kill_switch off as a second action.
    """
    st = load_state()
    mins = max(1, int(minutes))
    until = _utc_now() + timedelta(minutes=mins)
    st["armed_until_utc"] = _iso(until)
    if reason:
        st["kill_reason"] = str(reason).strip()
    save_state(st)
    return st


def is_armed(state: Optional[Dict[str, Any]] = None) -> bool:
    st = state or load_state()
    until = _parse_iso(st.get("armed_until_utc"))
    if not until:
        return False
    return _utc_now() < until


def gate(state: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Returns: (blocked, reason, state)
    blocked True if kill switch ON or not armed.
    """
    st = state or load_state()

    if bool(st.get("kill_switch", True)):
        reason = str(st.get("kill_reason") or "Kill switch enabled").strip()
        return True, f"BLOCKED [KILL_SWITCH]: {reason}", st

    if not is_armed(st):
        until = st.get("armed_until_utc") or ""
        msg = "Trading is DISARMED (no active arming window)."
        if until:
            msg = f"Trading arming window expired at {until}."
        return True, f"BLOCKED [DISARMED]: {msg}", st

    until = st.get("armed_until_utc") or ""
    return False, f"ARMED until {until}", st


def status(state: Optional[Dict[str, Any]] = None) -> str:
    st = state or load_state()
    if bool(st.get("kill_switch", True)):
        return "🔴 KILL_SWITCH"
    if is_armed(st):
        return "🟢 ARMED"
    return "🟡 DISARMED"
