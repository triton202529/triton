"""Shared operator-facing Eastern Time presentation.

Presentation only. Does not rewrite stored timestamps, contact a broker, or
invent a clock reading. Ordinary pages use America/New_York via zoneinfo.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
TIMEZONE_NAME = "America/New_York"
OPERATOR_TZ_LABEL = "ET"
UNAVAILABLE_LABEL = "Unavailable"

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_OPERATOR_ET_RE = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4} at \d{1,2}:\d{2} (AM|PM) ET$")
_HAS_OFFSET_RE = re.compile(r"(Z|[+-]\d{2}:\d{2})$", re.IGNORECASE)


def is_operator_et_text(value: Any) -> bool:
    return bool(_OPERATOR_ET_RE.match(str(value or "").strip()))


def format_date_only(value: Any) -> str:
    """Human date without inventing a time-of-day."""
    if isinstance(value, datetime):
        day = value.date()
    elif isinstance(value, date):
        day = value
    else:
        text = str(value or "").strip()
        if not _DATE_ONLY_RE.match(text):
            return UNAVAILABLE_LABEL
        try:
            day = date.fromisoformat(text)
        except ValueError:
            return UNAVAILABLE_LABEL
    return f"{day.strftime('%B')} {day.day}, {day.year}"


def parse_aware_datetime(value: Any) -> Optional[datetime]:
    """Parse a timezone-aware datetime. Naive or malformed values return None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return None
        return value
    if isinstance(value, date) and not isinstance(value, datetime):
        return None
    text = str(value).strip()
    if not text or is_operator_et_text(text):
        return None
    if _DATE_ONLY_RE.match(text):
        return None
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    if not _HAS_OFFSET_RE.search(text):
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def format_operator_datetime(value: Any) -> str:
    """Render an evidenced timestamp in Eastern Time, labeled ET.

    Already-offset values are converted once through ZoneInfo. Naive, malformed,
    or ambiguous values are Unavailable. Date-only evidence stays date-only.
    """
    if value is None or value == "":
        return UNAVAILABLE_LABEL
    if is_operator_et_text(value):
        return str(value).strip()
    if isinstance(value, date) and not isinstance(value, datetime):
        return format_date_only(value)
    text = str(value).strip()
    if _DATE_ONLY_RE.match(text):
        return format_date_only(text)
    parsed = parse_aware_datetime(value)
    if parsed is None:
        return UNAVAILABLE_LABEL
    local = parsed.astimezone(NY_TZ)
    hour = local.strftime("%I").lstrip("0") or "12"
    return (
        f"{local.strftime('%B')} {local.day}, {local.year} "
        f"at {hour}:{local.strftime('%M')} {local.strftime('%p')} {OPERATOR_TZ_LABEL}"
    )


def format_raw_utc(value: Any) -> str:
    """Preserve a raw UTC/ISO reading for Technical details only."""
    if value is None or value == "":
        return UNAVAILABLE_LABEL
    if isinstance(value, str) and value.strip():
        return value.strip()
    parsed = parse_aware_datetime(value)
    if parsed is None:
        return UNAVAILABLE_LABEL
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
