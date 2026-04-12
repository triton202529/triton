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
from datetime import date, datetime, timedelta
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
