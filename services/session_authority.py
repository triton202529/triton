# services/session_authority.py
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd


def last_completed_session_date_nyse() -> date:
    """
    Returns the last COMPLETED NYSE session date (ET), holiday-aware.
    Uses pandas_market_calendars (already installed in your venv).
    """
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")
    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    sched = nyse.schedule(
        start_date=(now_utc - pd.Timedelta(days=10)).date(),
        end_date=(now_utc + pd.Timedelta(days=2)).date(),
    )

    closes = pd.to_datetime(sched["market_close"], utc=True)
    last = sched[closes <= now_utc].iloc[-1]
    return last.name.date()


def fetch_end_exclusive_for_daily(as_of: Optional[date] = None) -> date:
    """
    Many daily-bar APIs use an EXCLUSIVE end date. To include AS_OF bar, end = AS_OF + 1 day.
    """
    as_of = as_of or last_completed_session_date_nyse()
    return as_of + timedelta(days=1)
