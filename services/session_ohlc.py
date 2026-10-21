"""Shared session-bar OHLC contract.

Prices are compared as stored. This module does not round, clamp, or rewrite them.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

OHLC_ABSOLUTE_TOLERANCE = 1e-6

_INEQUALITIES = (
    ("low_gt_open", "low", "open"),
    ("low_gt_close", "low", "close"),
    ("high_lt_open", "open", "high"),
    ("high_lt_close", "close", "high"),
    ("high_lt_low", "low", "high"),
)


def _finite(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def session_ohlc_violation(
    *,
    open: Any,
    high: Any,
    low: Any,
    close: Any,
    symbol: str,
    session_date: str,
    source_layer: str,
) -> Optional[dict]:
    """Return a diagnostic when the session bar breaks the shared OHLC contract."""
    raw_values = {"open": open, "high": high, "low": low, "close": close}
    parsed = {name: _finite(value) for name, value in raw_values.items()}
    diagnostic = {
        "symbol": str(symbol or "").upper(),
        "session_date": str(session_date or ""),
        "relationship": "",
        "values": {
            name: parsed[name] if parsed[name] is not None else raw_values[name]
            for name in ("open", "high", "low", "close")
        },
        "tolerance": OHLC_ABSOLUTE_TOLERANCE,
        "source_layer": source_layer,
    }
    if any(number is None for number in parsed.values()):
        diagnostic["relationship"] = "non_finite_ohlc"
        return diagnostic
    numbers = {name: parsed[name] for name in parsed}
    for relationship, left_name, right_name in _INEQUALITIES:
        left = numbers[left_name]
        right = numbers[right_name]
        assert left is not None and right is not None
        if (left - right) > OHLC_ABSOLUTE_TOLERANCE:
            diagnostic["relationship"] = relationship
            diagnostic["values"] = numbers
            return diagnostic
    return None


def violation_from_mapping(
    row: Mapping[str, Any],
    *,
    symbol: str,
    session_date: str,
    source_layer: str,
) -> Optional[dict]:
    return session_ohlc_violation(
        open=row.get("open"),
        high=row.get("high"),
        low=row.get("low"),
        close=row.get("close"),
        symbol=symbol,
        session_date=session_date,
        source_layer=source_layer,
    )
