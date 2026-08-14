# services/market_calendar.py
"""
NYSE market calendar helpers.

Goal:
- Provide a deterministic "last completed NYSE session" date.
- During market hours (or before the close), this returns the PREVIOUS trading day.
- Only AFTER the NYSE close has passed does it return today's session date.

This prevents AS_OF drift that breaks preprocessing, training, and AS_OF contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SessionInfo:
    session_date: date
    market_open: datetime
    market_close: datetime


def _get_schedule_df(cal, start_date: str, end_date: str):
    """
    exchange_calendars compatibility:
    - Older versions: cal.schedule(start_date=..., end_date=...) -> DataFrame
    - Newer versions: cal.schedule is a DataFrame (property)
    """
    sched = getattr(cal, "schedule", None)
    if sched is None:
        return None

    # callable -> old API
    if callable(sched):
        try:
            return sched(start_date=start_date, end_date=end_date)
        except TypeError:
            # some versions accept positional
            return sched(start_date, end_date)

    # property DataFrame -> new API
    try:
        df = sched
        # schedule index is session labels (timestamps)
        return df.loc[start_date:end_date]
    except Exception:
        return None


def _extract_open_close_columns(sched) -> tuple[str, str]:
    """
    exchange_calendars schedule column name compatibility.
    Most common:
      - market_open / market_close
    Older or alternate:
      - open / close
    """
    cols = {str(c) for c in getattr(sched, "columns", [])}

    if "market_open" in cols and "market_close" in cols:
        return "market_open", "market_close"
    if "open" in cols and "close" in cols:
        return "open", "close"

    # Try common variants (just in case)
    open_candidates = ["market_open", "open", "session_open"]
    close_candidates = ["market_close", "close", "session_close"]

    open_col = next((c for c in open_candidates if c in cols), None)
    close_col = next((c for c in close_candidates if c in cols), None)

    if not open_col or not close_col:
        raise RuntimeError(f"Unsupported exchange_calendars schedule columns: {sorted(cols)}")

    return open_col, close_col


def _try_exchange_calendars_session_info(now_ny: datetime) -> Optional[SessionInfo]:
    """
    Uses exchange_calendars if installed (recommended).
    Returns the most recent session whose CLOSE is <= now_ny.
    """
    try:
        import exchange_calendars as xc  # type: ignore
        import pandas as pd  # type: ignore
    except Exception:
        return None

    cal = xc.get_calendar("XNYS")

    # Small window around 'now' (covers weekends/holidays)
    start = (now_ny.date() - timedelta(days=14)).isoformat()
    end = (now_ny.date() + timedelta(days=2)).isoformat()

    sched = _get_schedule_df(cal, start, end)
    if sched is None or len(sched) == 0:
        return None

    sched = sched.copy()

    open_col, close_col = _extract_open_close_columns(sched)

    # Convert opens/closes to NY time for correct comparisons with now_ny.
    # They are typically tz-aware UTC timestamps.
    try:
        sched["open_ny"] = sched[open_col].dt.tz_convert(NY_TZ)
        sched["close_ny"] = sched[close_col].dt.tz_convert(NY_TZ)
    except Exception:
        # If for some reason they aren't tz-aware, try forcing as UTC then convert.
        sched[open_col] = pd.to_datetime(sched[open_col], errors="coerce", utc=True)
        sched[close_col] = pd.to_datetime(sched[close_col], errors="coerce", utc=True)
        sched["open_ny"] = sched[open_col].dt.tz_convert(NY_TZ)
        sched["close_ny"] = sched[close_col].dt.tz_convert(NY_TZ)

    now_ts = pd.Timestamp(now_ny)

    # Sessions that have fully closed
    closed = sched[sched["close_ny"] <= now_ts]
    if closed.empty:
        # Very early morning after holiday/weekend, etc.
        closed = sched[sched["open_ny"].dt.date < now_ny.date()]
        if closed.empty:
            return None

    row = closed.iloc[-1]
    session_label = row.name  # Timestamp-like session label
    try:
        session_date = session_label.date()
    except Exception:
        session_date = pd.Timestamp(session_label).date()

    return SessionInfo(
        session_date=session_date,
        market_open=row["open_ny"].to_pydatetime(),
        market_close=row["close_ny"].to_pydatetime(),
    )


def last_completed_nyse_session(now: Optional[datetime] = None) -> date:
    """
    Returns the last NYSE session that has COMPLETED (market close has passed).

    This is the AS_OF you should use for:
      - preprocessing strict writes
      - training/predictions AS_OF tagging
      - generate_signals "session_as_of_date"
      - asof_contract expected AS_OF
    """
    now_ny = (now or datetime.now(tz=NY_TZ)).astimezone(NY_TZ)

    # Best: exchange_calendars (handles holidays, early closes, DST)
    info = _try_exchange_calendars_session_info(now_ny)
    if info is not None:
        return info.session_date

    # Fallback (no calendar lib): simple weekday rule.
    # Fixes the "today before close" bug (but not holiday-aware).
    d = now_ny.date()

    # weekend rollbacks
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d - timedelta(days=2)

    # before 4pm NY time -> yesterday (or last weekday)
    if now_ny.hour < 16:
        d2 = d - timedelta(days=1)
        while d2.weekday() >= 5:
            d2 -= timedelta(days=1)
        return d2

    # after close -> today
    return d


# Backwards-compatible helper name (some parts of the codebase may import this)
def last_completed_session_date(now: Optional[datetime] = None) -> str:
    return last_completed_nyse_session(now=now).isoformat()


def _aware_utc(now: Optional[datetime] = None) -> datetime:
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        return clock.replace(tzinfo=timezone.utc)
    return clock.astimezone(timezone.utc)


SESSION_OPEN = "OPEN"
SESSION_CLOSED = "CLOSED"
SESSION_UNKNOWN = "UNKNOWN"

ATTENTION_SESSION_CALENDAR_UNKNOWN = "SESSION_CALENDAR_UNKNOWN"


def _import_exchange_calendars():
    import exchange_calendars as xc  # type: ignore
    import pandas as pd  # type: ignore

    return xc, pd


def _load_xnys_calendar(xc):
    return xc.get_calendar("XNYS")


def _schedule_for_window(cal, start: str, end: str):
    sched = _get_schedule_df(cal, start, end)
    if sched is None or len(sched) == 0:
        raise RuntimeError("empty XNYS schedule")
    return sched


def _convert_schedule_to_ny(sched, pd, open_col: str, close_col: str):
    converted = sched.copy()
    try:
        converted["open_ny"] = converted[open_col].dt.tz_convert(NY_TZ)
        converted["close_ny"] = converted[close_col].dt.tz_convert(NY_TZ)
    except Exception:
        converted[open_col] = pd.to_datetime(converted[open_col], errors="coerce", utc=True)
        converted[close_col] = pd.to_datetime(converted[close_col], errors="coerce", utc=True)
        converted["open_ny"] = converted[open_col].dt.tz_convert(NY_TZ)
        converted["close_ny"] = converted[close_col].dt.tz_convert(NY_TZ)
    return converted


def _weak_clock_diagnostic(now_utc: datetime) -> Optional[bool]:
    """Descriptive weekday/clock hint only. Must never grant execution eligibility."""
    try:
        from services.guard_auto import is_market_open_et

        return bool(is_market_open_et(now_utc))
    except Exception:
        return None


def _unknown_session_payload(payload: dict, reason: str, exc: Optional[BaseException], now_utc: datetime) -> dict:
    payload["session_state"] = SESSION_UNKNOWN
    payload["regular_session_open"] = False
    payload["reason"] = reason
    payload["calendar_fallback"] = False
    payload["source"] = "services.market_calendar.regular_session_state"
    payload["calendar_error"] = (
        f"{type(exc).__name__}: {exc}" if exc is not None else None
    )
    payload["weak_clock_diagnostic"] = _weak_clock_diagnostic(now_utc)
    return payload


def _resolve_xnys_session(sched, now_ny: datetime, pd) -> tuple[str, str, Optional[object]]:
    """Return (OPEN|CLOSED|UNKNOWN, reason, row_or_none) from an already-converted XNYS schedule."""
    from datetime import time as dt_time

    now_ts = pd.Timestamp(now_ny)
    open_row = None
    today_row = None
    usable_rows = 0
    for _, row in sched.iterrows():
        open_ny = row["open_ny"]
        close_ny = row["close_ny"]
        if pd.isna(open_ny) or pd.isna(close_ny):
            continue
        try:
            if open_ny >= close_ny:
                return SESSION_UNKNOWN, "session_unresolved", None
        except Exception:
            return SESSION_UNKNOWN, "session_unresolved", None
        usable_rows += 1
        try:
            session_date = open_ny.date()
        except Exception:
            session_date = pd.Timestamp(open_ny).date()
        if session_date == now_ny.date():
            today_row = row
        if open_ny <= now_ts < close_ny:
            open_row = row
            break

    if usable_rows == 0:
        return SESSION_UNKNOWN, "calendar_malformed", None
    if open_row is not None:
        return SESSION_OPEN, "regular_session_open", open_row
    if now_ny.weekday() >= 5:
        return SESSION_CLOSED, "weekend", today_row
    if today_row is None:
        return SESSION_CLOSED, "holiday", None
    open_ny = today_row["open_ny"]
    close_ny = today_row["close_ny"]
    if now_ts < open_ny:
        return SESSION_CLOSED, "pre_open", today_row
    close_time = close_ny.timetz() if hasattr(close_ny, "timetz") else close_ny.time()
    regular_close = dt_time(16, 0)
    try:
        close_clock = close_ny.tz_convert(NY_TZ).time()
    except Exception:
        close_clock = close_time
    if close_clock < regular_close:
        return SESSION_CLOSED, "early_close", today_row
    return SESSION_CLOSED, "regular_close", today_row


def regular_session_state(now: Optional[datetime] = None) -> dict:
    """Authoritative NYSE regular-session tri-state for new paper entries.

    OPEN is returned only when XNYS successfully proves the instant is inside
    [session_open, session_close). CLOSED is returned only when XNYS successfully
    proves the instant is outside that interval. Every import, load, schedule,
    conversion, or resolution failure is UNKNOWN and fail-closed.

    The weekday 09:30–16:00 ET helper is diagnostic only and cannot grant OPEN.
    """
    now_utc = _aware_utc(now)
    now_ny = now_utc.astimezone(NY_TZ)
    payload: dict = {
        "session_state": SESSION_UNKNOWN,
        "regular_session_open": False,
        "reason": "calendar_unavailable",
        "as_of_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_et": now_ny.isoformat(),
        "session_date": None,
        "session_open_et": None,
        "session_close_et": None,
        "calendar": "XNYS",
        "source": "services.market_calendar.regular_session_state",
        "calendar_fallback": False,
        "calendar_error": None,
        "weak_clock_diagnostic": None,
    }

    try:
        try:
            xc, pd = _import_exchange_calendars()
        except Exception as exc:
            return _unknown_session_payload(payload, "calendar_import_unavailable", exc, now_utc)

        try:
            cal = _load_xnys_calendar(xc)
        except Exception as exc:
            return _unknown_session_payload(payload, "calendar_load_failed", exc, now_utc)

        start = (now_ny.date() - timedelta(days=7)).isoformat()
        end = (now_ny.date() + timedelta(days=2)).isoformat()
        try:
            sched = _schedule_for_window(cal, start, end)
        except Exception as exc:
            return _unknown_session_payload(payload, "schedule_retrieval_failed", exc, now_utc)

        try:
            open_col, close_col = _extract_open_close_columns(sched)
        except Exception as exc:
            return _unknown_session_payload(payload, "calendar_malformed", exc, now_utc)

        try:
            sched = _convert_schedule_to_ny(sched, pd, open_col, close_col)
        except Exception as exc:
            return _unknown_session_payload(payload, "timezone_conversion_failed", exc, now_utc)

        try:
            state, reason, row = _resolve_xnys_session(sched, now_ny, pd)
        except Exception as exc:
            return _unknown_session_payload(payload, "calendar_exception", exc, now_utc)

        if state == SESSION_UNKNOWN:
            return _unknown_session_payload(payload, reason, None, now_utc)

        if row is not None:
            open_ny = row["open_ny"]
            close_ny = row["close_ny"]
            payload["session_date"] = open_ny.date().isoformat()
            payload["session_open_et"] = open_ny.isoformat()
            payload["session_close_et"] = close_ny.isoformat()

        payload["session_state"] = state
        payload["regular_session_open"] = state == SESSION_OPEN
        payload["reason"] = reason
        payload["calendar_fallback"] = False
        payload["source"] = "services.market_calendar.regular_session_state"
        return payload
    except Exception as exc:
        return _unknown_session_payload(payload, "calendar_exception", exc, now_utc)


def is_regular_session_open_for_entries(now: Optional[datetime] = None) -> bool:
    """True only when XNYS authoritatively proves OPEN. UNKNOWN is fail-closed."""
    return regular_session_state(now=now).get("session_state") == SESSION_OPEN
