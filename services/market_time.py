# services/market_time.py
"""
Market-time helpers for TRITON.

Goal:
- Provide a single, deterministic definition of "last completed session date"
  using an exchange calendar (NYSE) instead of fragile timezone heuristics.

Definition:
- "Last completed session" is the most recent trading day whose official NYSE
  market_close timestamp is <= now_utc.

This makes AS_OF stable during the trading day:
- Before close -> AS_OF is previous session
- After close  -> AS_OF becomes today (if today is a session)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta, date
from typing import Optional

import pandas as pd

try:
    import pandas_market_calendars as mcal  # type: ignore
except Exception as e:  # pragma: no cover
    mcal = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_date(x) -> Optional[date]:
    try:
        ts = pd.Timestamp(x)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.date()
    except Exception:
        return None


def last_completed_session_date(calendar=None, now_utc: Optional[datetime] = None) -> str:
    """
    Return YYYY-MM-DD string for the last completed NYSE session.

    Uses pandas_market_calendars schedule and compares market_close vs now_utc.
    """
    if now_utc is None:
        now_utc = _utc_now()

    # Ensure tz-aware UTC
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    if mcal is None and calendar is None:
        # Hard fallback (shouldn't happen in your environment since pmc is installed)
        # Return "yesterday weekday" behavior as a last resort.
        d = now_utc.date() - timedelta(days=1)
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
        return d.isoformat()

    if calendar is None:
        calendar = mcal.get_calendar("NYSE")

    # We only need a small window around now
    start = (now_utc - timedelta(days=14)).date().isoformat()
    end = (now_utc + timedelta(days=2)).date().isoformat()

    sched = calendar.schedule(start_date=start, end_date=end)

    # Ensure schedule columns exist
    if sched is None or sched.empty or "market_close" not in sched.columns:
        # Fallback to "previous weekday"
        d = now_utc.date() - timedelta(days=1)
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
        return d.isoformat()

    # market_close is tz-aware (UTC). Find rows with close <= now_utc
    closes = pd.to_datetime(sched["market_close"], utc=True, errors="coerce")
    closes = closes.dropna()

    if closes.empty:
        d = now_utc.date() - timedelta(days=1)
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
        return d.isoformat()

    mask = closes <= pd.Timestamp(now_utc)
    if not mask.any():
        # We are before the first close in window, use previous session day
        # (take earliest session in sched, then go one session back)
        idx = sched.index
        if len(idx) >= 2:
            return pd.Timestamp(idx[0]).date().isoformat()  # conservative
        d = now_utc.date() - timedelta(days=1)
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
        return d.isoformat()

    # last close <= now
    last_ts = closes[mask].iloc[-1]
    # The schedule index is session dates normalized (midnight). Use that index row.
    # Find the session whose market_close equals last_ts (safe lookup)
    # (If duplicates, take last)
    session_rows = sched[
        pd.to_datetime(sched["market_close"], utc=True, errors="coerce") == last_ts
    ]
    if not session_rows.empty:
        session_date = pd.Timestamp(session_rows.index[-1]).date().isoformat()
        return session_date

    # If for some reason lookup fails, just use last_ts date in NYSE close (UTC date is fine for session label)
    d = _to_date(last_ts)
    return d.isoformat() if d else now_utc.date().isoformat()
