"""BM6A Market session card. Presentation only. No production writes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dashboard.market_session_card import (
    LABEL_AFTER_HOURS,
    LABEL_CLOSED,
    LABEL_HOLIDAY,
    LABEL_OPEN,
    LABEL_PRE_MARKET,
    LABEL_UNAVAILABLE,
    market_session_card,
)
from services.market_calendar import SESSION_CLOSED, SESSION_OPEN, SESSION_UNKNOWN

ET = ZoneInfo("America/New_York")
ISO_ROOT = Path(__file__).resolve().parents[1]
PROD_ROOT = Path(r"C:\Users\akimw\triton")


def _et(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=ET)


def test_regular_session_open_is_green():
    card = market_session_card(now=_et(2026, 8, 13, 10, 0))
    assert card["value"] == LABEL_OPEN
    assert card["tone"] == "ok"
    assert card["session_state"] == SESSION_OPEN
    assert card["reason"] == "regular_session_open"
    assert card["note"] and card["note"].startswith("Closes at ")
    assert card["note"].endswith("ET")
    assert "4:00 PM ET" in card["note"]
    assert card["technical"]["calendar"] == "XNYS"
    assert card["technical"]["source"] == "services.market_calendar.regular_session_state"
    assert card["technical"]["timezone"] == "America/New_York"
    assert card["technical"]["as_of_utc"]
    assert card["technical"]["as_of_et"]


def test_pre_market_is_amber():
    card = market_session_card(now=_et(2026, 8, 13, 8, 15))
    assert card["value"] == LABEL_PRE_MARKET
    assert card["tone"] == "warn"
    assert card["session_state"] == SESSION_CLOSED
    assert card["reason"] == "pre_open"
    assert card["note"] and card["note"].startswith("Opens at ")
    assert "9:30 AM ET" in card["note"]


def test_after_hours_frozen_preview_september_8():
    card = market_session_card(now=_et(2026, 9, 8, 21, 4, 11))
    assert card["value"] == LABEL_AFTER_HOURS
    assert card["tone"] == "info"
    assert card["session_state"] == SESSION_CLOSED
    assert card["reason"] == "regular_close"
    assert card["note"] == "Opens at September 9 at 9:30 AM ET"
    assert card["technical"]["session_state"] == SESSION_CLOSED
    assert card["technical"]["reason"] == "regular_close"
    assert card["technical"]["calendar"] == "XNYS"


def test_holiday_closed_is_neutral():
    card = market_session_card(now=_et(2026, 1, 1, 10, 0))
    assert card["value"] == LABEL_HOLIDAY
    assert card["tone"] == "info"
    assert card["session_state"] == SESSION_CLOSED
    assert card["reason"] == "holiday"
    assert card["note"] and card["note"].startswith("Opens at ")
    assert card["tone"] != "crit"


def test_weekend_closed_is_neutral():
    card = market_session_card(now=_et(2026, 8, 15, 10, 0))
    assert card["value"] == LABEL_CLOSED
    assert card["tone"] == "info"
    assert card["reason"] == "weekend"
    assert card["note"] and card["note"].startswith("Opens at ")


def test_early_close_is_after_hours_not_an_error():
    card = market_session_card(now=_et(2026, 11, 27, 14, 0))
    assert card["value"] == LABEL_AFTER_HOURS
    assert card["tone"] == "info"
    assert card["session_state"] == SESSION_CLOSED
    assert card["reason"] == "early_close"
    assert card["note"] and card["note"].startswith("Opens at ")


def test_unavailable_is_fail_closed():
    card = market_session_card(
        session={
            "session_state": SESSION_UNKNOWN,
            "reason": "calendar_unavailable",
            "calendar": "XNYS",
            "source": "services.market_calendar.regular_session_state",
            "as_of_utc": "2026-09-08T15:50:00Z",
            "as_of_et": "2026-09-08T11:50:00-04:00",
        }
    )
    assert card["value"] == LABEL_UNAVAILABLE
    assert card["tone"] == "crit"
    assert card["code"] == "UNAVAILABLE"
    assert card["note"] is None
    assert card["technical"]["session_state"] == SESSION_UNKNOWN
    assert card["technical"]["reason"] == "calendar_unavailable"


def test_status_bar_places_market_session_after_trading_mode():
    text = (ISO_ROOT / "dashboard" / "operator_modern.py").read_text(encoding="utf-8")
    trading = text.index('"label": "Trading mode"')
    market = text.index('"label": "Market session"')
    paper = text.index('"label": "Paper execution"')
    assert trading < market < paper


def test_session_card_does_not_contact_broker_or_add_weekday_logic():
    card_src = (ISO_ROOT / "dashboard" / "market_session_card.py").read_text(encoding="utf-8")
    assert "regular_session_state" in card_src
    assert "alpaca" not in card_src.lower()
    assert ".weekday(" not in card_src
    assert "time(9, 30)" not in card_src
    assert ISO_ROOT.resolve() != PROD_ROOT.resolve()
