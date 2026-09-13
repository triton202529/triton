"""Presentation mapper for the Overview Market session card.

Uses services.market_calendar.regular_session_state only. Does not invent
weekday/clock eligibility, contact a broker, or change session gating.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional
from zoneinfo import ZoneInfo

from services import market_calendar as _market_calendar
from services.market_calendar import (
    NY_TZ,
    SESSION_CLOSED,
    SESSION_OPEN,
    SESSION_UNKNOWN,
    regular_session_state,
)

NY_TZ = ZoneInfo("America/New_York")
TIMEZONE_NAME = "America/New_York"

LABEL_OPEN = "Open"
LABEL_PRE_MARKET = "Pre-market"
LABEL_AFTER_HOURS = "After hours"
LABEL_CLOSED = "Closed"
LABEL_HOLIDAY = "Holiday closed"
LABEL_UNAVAILABLE = "Unavailable"

_CALENDAR_FAIL_REASONS = frozenset(
    {
        "calendar_unavailable",
        "calendar_import_unavailable",
        "calendar_load_failed",
        "schedule_retrieval_failed",
        "calendar_malformed",
        "timezone_conversion_failed",
        "calendar_exception",
        "session_unresolved",
    }
)


def xnys_adjacent_session_times(now: Optional[datetime] = None) -> dict:
    """Read-only next XNYS open/close for session-card notes.

    Uses the same private calendar loaders as regular_session_state.
    Does not grant OPEN, change gating, or invent weekday/clock eligibility.
    """
    now_utc = _market_calendar._aware_utc(now)
    now_ny = now_utc.astimezone(NY_TZ)
    out = {
        "next_open_et": None,
        "next_close_et": None,
        "calendar": "XNYS",
        "source": "dashboard.market_session_card.xnys_adjacent_session_times",
        "as_of_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_et": now_ny.isoformat(),
    }
    try:
        xc, pd = _market_calendar._import_exchange_calendars()
        cal = _market_calendar._load_xnys_calendar(xc)
        start = (now_ny.date() - timedelta(days=7)).isoformat()
        end = (now_ny.date() + timedelta(days=14)).isoformat()
        sched = _market_calendar._schedule_for_window(cal, start, end)
        open_col, close_col = _market_calendar._extract_open_close_columns(sched)
        sched = _market_calendar._convert_schedule_to_ny(sched, pd, open_col, close_col)
        now_ts = pd.Timestamp(now_ny)
        for _, row in sched.iterrows():
            open_ny = row["open_ny"]
            close_ny = row["close_ny"]
            if pd.isna(open_ny) or pd.isna(close_ny):
                continue
            if now_ts < open_ny:
                out["next_open_et"] = open_ny.isoformat()
                out["next_close_et"] = close_ny.isoformat()
                break
            if open_ny <= now_ts < close_ny:
                out["next_close_et"] = close_ny.isoformat()
                break
    except Exception:
        return out
    return out


def _parse_et(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(NY_TZ)


def human_et_stamp(value: Any, *, as_of: Any = None) -> str:
    dt = _parse_et(value)
    if dt is None:
        return "Unavailable"
    clock = dt.strftime("%I:%M %p").lstrip("0") + " ET"
    as_of_dt = _parse_et(as_of)
    if as_of_dt is not None and as_of_dt.date() == dt.date():
        return clock
    return f"{dt.strftime('%B')} {dt.day} at {clock}"


def market_session_card(
    *,
    now: Optional[datetime] = None,
    session: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = dict(session) if session is not None else regular_session_state(now=now)
    state = str(payload.get("session_state") or SESSION_UNKNOWN).upper()
    reason = str(payload.get("reason") or "")
    as_of = payload.get("as_of_et") or payload.get("as_of_utc")
    neighbors = xnys_adjacent_session_times(now=now)
    next_open = (
        payload.get("session_open_et") if reason == "pre_open" else neighbors.get("next_open_et")
    )
    next_close = payload.get("session_close_et") or neighbors.get("next_close_et")

    if state == SESSION_UNKNOWN or reason in _CALENDAR_FAIL_REASONS:
        label = LABEL_UNAVAILABLE
        tone = "crit"
        code = "UNAVAILABLE"
        note = None
    elif state == SESSION_OPEN:
        label = LABEL_OPEN
        tone = "ok"
        code = "OPEN"
        note = f"Closes at {human_et_stamp(next_close, as_of=as_of)}" if next_close else None
    elif reason == "pre_open":
        label = LABEL_PRE_MARKET
        tone = "warn"
        code = "PRE_MARKET"
        note = (
            f"Opens at {human_et_stamp(next_open or payload.get('session_open_et'), as_of=as_of)}"
        )
    elif reason == "holiday":
        label = LABEL_HOLIDAY
        tone = "info"
        code = "HOLIDAY_CLOSED"
        note = f"Opens at {human_et_stamp(next_open, as_of=as_of)}" if next_open else None
    elif reason in {"regular_close", "early_close"}:
        label = LABEL_AFTER_HOURS
        tone = "info"
        code = "AFTER_HOURS"
        note = f"Opens at {human_et_stamp(next_open, as_of=as_of)}" if next_open else None
    elif state == SESSION_CLOSED:
        label = LABEL_CLOSED
        tone = "info"
        code = "CLOSED"
        note = f"Opens at {human_et_stamp(next_open, as_of=as_of)}" if next_open else None
    else:
        label = LABEL_UNAVAILABLE
        tone = "crit"
        code = "UNAVAILABLE"
        note = None

    if note and "Unavailable" in note:
        note = None

    return {
        "label": "Market session",
        "value": label,
        "tone": tone,
        "note": note,
        "code": code,
        "session_state": state,
        "reason": reason,
        "calendar": payload.get("calendar") or "XNYS",
        "source": payload.get("source") or "services.market_calendar.regular_session_state",
        "timezone": TIMEZONE_NAME,
        "as_of_utc": payload.get("as_of_utc"),
        "as_of_et": payload.get("as_of_et"),
        "session_open_et": payload.get("session_open_et"),
        "session_close_et": payload.get("session_close_et"),
        "next_open_et": next_open,
        "next_close_et": next_close,
        "technical": {
            "session_state": state,
            "reason": reason,
            "calendar": payload.get("calendar") or "XNYS",
            "source": payload.get("source") or "services.market_calendar.regular_session_state",
            "timezone": TIMEZONE_NAME,
            "as_of_utc": payload.get("as_of_utc"),
            "as_of_et": payload.get("as_of_et"),
            "session_open_et": payload.get("session_open_et"),
            "session_close_et": payload.get("session_close_et"),
            "next_open_et": next_open,
            "next_close_et": next_close,
        },
    }
