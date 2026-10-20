"""Required signal session from the XNYS calendar.

The signal as-of date is the latest NYSE session whose close is already at or
before the clock. Weekends, holidays, and early closes use that session.
Calendar-yesterday and weekday rollback are not used.
"""

from __future__ import annotations

from datetime import date, datetime

from services.market_calendar import NY_TZ, _try_exchange_calendars_session_info


def required_signal_session(now: datetime | None = None) -> date:
    """Return the latest completed XNYS session date.

    Raises if the exchange calendar cannot be read. A missing calendar must
    not fall through to yesterday's civil date.
    """
    clock = now or datetime.now(tz=NY_TZ)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=NY_TZ)
    now_ny = clock.astimezone(NY_TZ)
    info = _try_exchange_calendars_session_info(now_ny)
    if info is None:
        raise RuntimeError(
            "XNYS calendar unavailable; calendar-yesterday fallback is forbidden "
            "for the signal session."
        )
    return info.session_date
