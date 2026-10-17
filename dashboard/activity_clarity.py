"""Presentation-only wording for the Operator Console activity feed.

Does not write logs, ledgers, receipts, or timestamps. Callers pass event
mappings; this module returns new display values and leaves inputs unchanged.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from dashboard.time_presentation import (
    UNAVAILABLE_LABEL,
    format_raw_utc,
    parse_aware_datetime,
)

NY_TZ = ZoneInfo("America/New_York")

NO_TRADE_ACTIVITY = "No trade activity in this range."
NO_SYSTEM_ACTIVITY = "No system activity in this range."
FILTER_SCOPE_NOTE = (
    "Category, severity, and search apply to both Trade activity and System activity. "
    "Each section lists newest events first."
)

_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FILLED_WORD = re.compile(r"\bfilled\b", re.IGNORECASE)

_ACTION_FIELDS = (
    "trade_action",
    "portfolio_action",
    "lifecycle_action",
    "opportunity_type",
    "action",
    "disposition",
)
_STRUCTURE_FIELDS = _ACTION_FIELDS + (
    "action_disposition",
    "event_type",
    "broker_terminal_status",
    "terminal_status",
    "broker_submission_status",
    "submission_status",
    "order_status",
    "status",
    "selected_qty",
    "filled_qty",
    "qty",
    "quantity",
    "submitted_qty",
    "source_pipeline",
    "terminal_status_proven",
    "fill_proven",
)
_LEDGER_COPY_FIELDS = _STRUCTURE_FIELDS + ("extra", "symbol", "side")

_ACTION_MAP = {
    "ENTRY": "entry",
    "OPEN_NEW": "entry",
    "NEW_ENTRY": "entry",
    "ADD": "add",
    "TRIM": "trim",
    "EXIT": "exit",
    "MANUAL_EXIT": "manual_exit",
    "MANUAL_PORTFOLIO_EXIT": "manual_exit",
}
_STATUS_TOKENS = {
    "SUBMITTED",
    "PENDING",
    "PENDING_NEW",
    "PENDING_CANCEL",
    "ACCEPTED",
    "NEW",
    "OPEN",
    "WORKING",
    "FILLED",
    "PARTIAL",
    "PARTIALLY_FILLED",
    "CANCELLED",
    "CANCELED",
    "REJECTED",
    "EXPIRED",
    "RECONCILED",
    "FAILED",
    "EXPLICITLY_SKIPPED",
    "SKIPPED",
    "SKIPPED_ZERO",
    "UNKNOWN",
    "PLANNED",
    "AUTHORIZED",
    "PLAN",
}
_TRADE_EVENT_TYPES = {
    "ACTION_SUBMITTED",
    "ACTION_FAILED",
    "ACTION_SKIPPED",
    "ORDER_STATUS_RECONCILED",
    "ORDER_TERMINAL_RECONCILED",
}


def _token(value: Any) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str) and value.strip().lower() in {"true", "yes", "1"}:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 1:
        return True
    return False


def _action_token(value: Any) -> Optional[str]:
    token = _token(value)
    if not token or token in _STATUS_TOKENS:
        return None
    return _ACTION_MAP.get(token)


def _action_kind(event: Mapping[str, Any]) -> Optional[str]:
    """Structured entry/add/trim/exit/manual exit. BUY and SELL are not actions."""
    if _token(event.get("source_pipeline")) == "MANUAL_PORTFOLIO_EXIT":
        return "manual_exit"
    for key in _ACTION_FIELDS:
        kind = _action_token(event.get(key))
        if kind:
            return kind
    extra = event.get("extra")
    if isinstance(extra, Mapping):
        for key in _ACTION_FIELDS:
            kind = _action_token(extra.get(key))
            if kind:
                return kind
        if _token(extra.get("source_pipeline")) == "MANUAL_PORTFOLIO_EXIT":
            return "manual_exit"
    return _action_token(event.get("action_disposition"))


def fill_proven(event: Mapping[str, Any]) -> bool:
    """True only when terminal broker or ledger evidence marks a fill proven."""
    if not (_is_true(event.get("terminal_status_proven")) or _is_true(event.get("fill_proven"))):
        return False
    terminal = _token(event.get("broker_terminal_status") or event.get("terminal_status"))
    return terminal in {"FILLED", "PARTIAL", "PARTIALLY_FILLED"}


def _status_word(event: Mapping[str, Any]) -> str:
    terminal = _token(event.get("broker_terminal_status") or event.get("terminal_status"))
    proven = _is_true(event.get("terminal_status_proven")) or _is_true(event.get("fill_proven"))
    submission = _token(
        event.get("broker_submission_status")
        or event.get("submission_status")
        or event.get("order_status")
        or event.get("status")
    )
    disposition = _token(event.get("action_disposition"))
    event_type = _token(event.get("event_type"))

    if proven and terminal == "FILLED":
        return "filled"
    if proven and terminal in {"PARTIAL", "PARTIALLY_FILLED"}:
        return "partially filled"
    if not proven:
        if terminal in {"FILLED", "PARTIAL", "PARTIALLY_FILLED"}:
            terminal = ""
        if submission in {"FILLED", "PARTIAL", "PARTIALLY_FILLED"}:
            submission = ""

    if terminal in {"CANCELLED", "CANCELED"} or submission in {"CANCELLED", "CANCELED"}:
        return "cancelled"
    if terminal == "REJECTED" or submission == "REJECTED" or disposition == "REJECTED":
        return "rejected"
    if terminal == "EXPIRED" or submission == "EXPIRED":
        return "expired"
    if disposition == "FAILED" or event_type == "ACTION_FAILED":
        return "failed"
    if (
        disposition in {"EXPLICITLY_SKIPPED", "SKIPPED", "SKIPPED_ZERO"}
        or event_type == "ACTION_SKIPPED"
    ):
        return "skipped"
    if submission in {
        "PENDING",
        "PENDING_NEW",
        "ACCEPTED",
        "NEW",
        "OPEN",
        "WORKING",
        "PENDING_CANCEL",
    }:
        return "pending"
    if event_type in {"ORDER_STATUS_RECONCILED", "ORDER_TERMINAL_RECONCILED"}:
        return "reconciled"
    if disposition == "SUBMITTED" or submission == "SUBMITTED" or event_type == "ACTION_SUBMITTED":
        return "submitted"
    claimed = _token(
        event.get("broker_terminal_status")
        or event.get("terminal_status")
        or event.get("status")
        or event.get("order_status")
    )
    if claimed in {"FILLED", "PARTIAL", "PARTIALLY_FILLED"}:
        return "pending"
    return ""


def _format_qty(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num < 0 or num != num:
        return None
    if num.is_integer():
        return str(int(num))
    return f"{num:.8f}".rstrip("0").rstrip(".")


def _quantity_text(event: Mapping[str, Any], status: str) -> Optional[str]:
    if status in {"filled", "partially filled"}:
        raw = event.get("filled_qty")
        if raw in (None, ""):
            raw = event.get("selected_qty")
        if raw in (None, ""):
            raw = event.get("qty", event.get("quantity"))
        return _format_qty(raw)
    for key in ("selected_qty", "qty", "quantity", "submitted_qty"):
        if event.get(key) not in (None, ""):
            return _format_qty(event.get(key))
    return None


def _share_phrase(qty: Optional[str]) -> Optional[str]:
    if qty is None:
        return None
    unit = "share" if qty == "1" else "shares"
    return f"{qty} {unit}"


def _symbol(event: Mapping[str, Any]) -> str:
    raw = str(event.get("symbol") or "").strip().upper()
    return raw or "Order"


def _trade_label(symbol: str, kind: Optional[str]) -> str:
    if kind == "entry":
        return f"New {symbol} entry"
    if kind == "add":
        return f"{symbol} add"
    if kind == "trim":
        return f"{symbol} trim"
    if kind == "exit":
        return f"{symbol} exit"
    if kind == "manual_exit":
        return f"Manual {symbol} exit"
    if symbol == "Order":
        return "Order"
    return f"{symbol} order"


def _has_trade_structure(event: Mapping[str, Any]) -> bool:
    for key in _STRUCTURE_FIELDS:
        if key in event and event.get(key) not in (None, ""):
            return True
    extra = event.get("extra")
    if isinstance(extra, Mapping):
        for key in _ACTION_FIELDS:
            if extra.get(key) not in (None, ""):
                return True
    return False


def _filled_line(symbol: str, kind: Optional[str], qty_bit: Optional[str]) -> str:
    if kind == "entry":
        head = f"New {symbol} entry"
    elif kind == "add":
        head = f"Added {symbol}"
    elif kind == "trim":
        head = f"Trimmed {symbol}"
    elif kind == "exit":
        head = f"Exited {symbol}"
    elif kind == "manual_exit":
        head = f"Manual {symbol} exit"
    elif symbol == "Order":
        head = "Order"
    else:
        head = f"{symbol} order"
    if qty_bit:
        return f"{head} — {qty_bit} filled"
    return f"{head} filled"


def _legacy_trade_line(event: Mapping[str, Any]) -> str:
    title = str(event.get("title") or "Order event").strip() or "Order event"
    if not fill_proven(event) and _FILLED_WORD.search(title):
        symbol = str(event.get("symbol") or "").strip().upper()
        if symbol:
            return f"{symbol} order pending confirmation"
        return "Order pending confirmation"
    lowered = title.lower()
    if "batch" in lowered:
        if "completion" in lowered or "summary" in lowered:
            return "Paper sell batch summary"
        return "Paper sell batch dispatched"
    return title


def operator_trade_line(event: Mapping[str, Any]) -> str:
    """Plain operator sentence. Never says filled without terminal proof."""
    if not _has_trade_structure(event):
        return _legacy_trade_line(event)
    symbol = _symbol(event)
    kind = _action_kind(event)
    status = _status_word(event)
    qty_bit = _share_phrase(_quantity_text(event, status))
    if status == "filled":
        return _filled_line(symbol, kind, qty_bit)
    label = _trade_label(symbol, kind)
    if status and qty_bit:
        return f"{label} {status} — {qty_bit}"
    if status:
        return f"{label} {status}"
    if qty_bit:
        return f"{label} — {qty_bit}"
    return label


def operator_system_line(event: Mapping[str, Any]) -> str:
    title = str(event.get("title") or "System event").strip() or "System event"
    if not fill_proven(event):
        title = re.sub(r"\bpartially filled\b", "updated", title, flags=re.IGNORECASE)
        title = _FILLED_WORD.sub("updated", title)
    return title


def activity_section(event: Mapping[str, Any]) -> str:
    """Trade lifecycle versus system runtime. One event has one section."""
    if _token(event.get("category")) == "ORDER":
        return "trade"
    if _token(event.get("event_type")) in _TRADE_EVENT_TYPES:
        return "trade"
    if _action_kind(event):
        return "trade"
    return "system"


def _format_market_date(day: date) -> str:
    return f"{_MONTHS[day.month - 1]} {day.day}, {day.year}"


def format_activity_market_time(value: Any) -> str:
    """America/New_York clock for the primary line. Naive values stay Unavailable."""
    if value is None or value == "":
        return UNAVAILABLE_LABEL
    if isinstance(value, datetime):
        parsed = parse_aware_datetime(value)
        if parsed is None:
            return UNAVAILABLE_LABEL
    elif isinstance(value, date):
        return _format_market_date(value)
    else:
        text = str(value).strip()
        if _DATE_ONLY.match(text):
            try:
                return _format_market_date(date.fromisoformat(text))
            except ValueError:
                return UNAVAILABLE_LABEL
        parsed = parse_aware_datetime(value)
        if parsed is None:
            return UNAVAILABLE_LABEL
    local = parsed.astimezone(NY_TZ)
    hour = local.strftime("%I").lstrip("0") or "12"
    abbr = local.tzname() or ""
    if abbr not in {"EDT", "EST"}:
        offset = local.utcoffset()
        hours = int(offset.total_seconds() // 3600) if offset is not None else 0
        abbr = "EDT" if hours == -4 else "EST" if hours == -5 else (abbr or "ET")
    return (
        f"{_MONTHS[local.month - 1]} {local.day}, {local.year} · "
        f"{hour}:{local.strftime('%M')} {local.strftime('%p')} {abbr}"
    )


def operator_line(event: Mapping[str, Any], *, section: Optional[str] = None) -> str:
    chosen = section or activity_section(event)
    if chosen == "trade":
        return operator_trade_line(event)
    return operator_system_line(event)


def activity_primary_line(event: Mapping[str, Any], *, section: Optional[str] = None) -> str:
    """Market time plus the operator sentence. Raw UTC stays out of this line."""
    body = operator_line(event, section=section)
    when = format_activity_market_time(event.get("occurred_at"))
    if when == UNAVAILABLE_LABEL:
        return body
    return f"{when} · {body}"


def activity_technical_details(event: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy of evidence for an expander. Does not rewrite the source event."""
    details: Dict[str, Any] = {
        "occurred_at_utc": format_raw_utc(event.get("occurred_at")),
        "event_id": event.get("event_id"),
        "source": event.get("source"),
        "category": event.get("category"),
        "severity": event.get("severity"),
        "title": event.get("title"),
        "message": event.get("message"),
        "evidence_path": event.get("evidence_path"),
        "raw_reference": event.get("raw_reference"),
        "symbol": event.get("symbol"),
        "order_id": event.get("order_id"),
        "cycle_number": event.get("cycle_number"),
    }
    for key in _LEDGER_COPY_FIELDS:
        if key in event:
            details[key] = event.get(key)
    return details


def present_activity_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    section = activity_section(event)
    return {
        "section": section,
        "market_time": format_activity_market_time(event.get("occurred_at")),
        "operator_line": operator_line(event, section=section),
        "primary_line": activity_primary_line(event, section=section),
        "technical": activity_technical_details(event),
    }


def _newest_key(event: Mapping[str, Any]) -> Tuple[int, float, str]:
    parsed = parse_aware_datetime(event.get("occurred_at"))
    if parsed is None:
        return (0, 0.0, str(event.get("event_id") or ""))
    return (1, parsed.timestamp(), str(event.get("event_id") or ""))


def partition_activity_events(
    events: Sequence[Mapping[str, Any]],
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """Newest-first trade and system lists. An event id appears in only one."""
    ordered = sorted(events, key=_newest_key, reverse=True)
    trade: List[Mapping[str, Any]] = []
    system: List[Mapping[str, Any]] = []
    seen: set[str] = set()
    for event in ordered:
        event_id = str(event.get("event_id") or "")
        if event_id:
            if event_id in seen:
                continue
            seen.add(event_id)
        if activity_section(event) == "trade":
            trade.append(event)
        else:
            system.append(event)
    return trade, system


def trade_activity_notice(trade_events: Sequence[Mapping[str, Any]]) -> Optional[str]:
    if not trade_events:
        return NO_TRADE_ACTIVITY
    return None


def attach_ledger_structure(
    events: Sequence[Mapping[str, Any]],
    ledger_events: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Read-only display copies. Missing structured fields are filled from the ledger."""
    by_id = {str(row.get("event_id")): row for row in ledger_events if row.get("event_id")}
    views: List[Dict[str, Any]] = []
    for event in events:
        view = dict(event)
        source = by_id.get(str(event.get("event_id") or ""))
        if source:
            for key in _LEDGER_COPY_FIELDS:
                if view.get(key) not in (None, "") or key not in source:
                    continue
                value = source.get(key)
                if value in (None, ""):
                    continue
                view[key] = dict(value) if key == "extra" and isinstance(value, Mapping) else value
        views.append(view)
    return views
