"""
Centralized display-label formatter (UI-A).

Presentation only. Does not rename stored fields, schema keys, CSV headers,
database columns, or execution contracts. Technical expanders may still show
raw identifiers unchanged.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

# Meaningful abbreviations kept as-is (case-insensitive match on tokens).
PRESERVED_ACRONYMS = frozenset(
    {
        "BUY",
        "SELL",
        "ADD",
        "HOLD",
        "TRIM",
        "EXIT",
        "ETF",
        "ID",
        "UTC",
        "API",
        "P/L",
        "PNL",
        "TTL",
        "CSV",
        "JSON",
        "REST",
        "SQLite".upper(),
        "BUY/ADD",
        "P&L",
    }
)

# Explicit operator-facing replacements. Keys compared case-insensitively.
KNOWN_LABELS: Dict[str, str] = {
    "paper_entry_armed": "Paper Entry Armed",
    "live_execution_permitted": "Live Execution Permitted",
    "paper_execution_permitted": "Paper Execution Permitted",
    "continuous_loop": "Continuous Loop",
    "skipped_zero": "Skipped — Zero Quantity",
    "trim_priority": "Trim Priority",
    "block_new_buy": "Block New Buys",
    "generated_at": "Generated At",
    "generated_at_utc": "Generated At (UTC)",
    "filled_avg_price": "Average Fill Price",
    "average_fill_price": "Average Fill Price",
    "avg_fill": "Average Fill Price",
    "avg_entry_price": "Average Cost",
    "market_value": "Market Value",
    "unrealized_pl": "Unrealized P/L",
    "unrealized_pnl": "Unrealized P/L",
    "unrealized_plpc": "Unrealized P/L %",
    "realized_pl": "Realized P/L",
    "realized_pnl": "Realized P/L",
    "total_value": "Portfolio Value",
    "total_portfolio_value": "Portfolio Value",
    "last_refresh_time": "Last Refresh",
    "authority_id": "Authority ID",
    "dispatch_id": "Dispatch ID",
    "broker_order_id": "Broker Order ID",
    "client_order_id": "Client Order ID",
    "trading_date": "Trading Date",
    "event_timestamp": "Event Time",
    "submitted_at": "Submitted At",
    "filled_at": "Filled At",
    "selected_qty": "Quantity",
    "filled_qty": "Filled Quantity",
    "submission_status": "Submission Status",
    "terminal_status": "Terminal Status",
    "action_disposition": "Action Outcome",
    "source_pipeline": "Source Pipeline",
    "paper_arm_status": "Paper Authorization",
    "execution_readiness_status": "Execution Readiness",
    "readiness_status": "Readiness",
    "overall_status": "Overall Status",
    "process_count": "Process Count",
    "lock_exists": "Lock Present",
    "data_freshness": "Data Freshness",
    "heartbeat_timestamp": "Heartbeat Time",
    "heartbeat_age_minutes": "Heartbeat Age (min)",
    "next_expected_at": "Next Expected Cycle",
    "last_completed_at": "Last Completed Cycle",
    "live_execution": "Live Execution",
    "paper_only": "Paper Only",
    "paper_mode": "Paper Mode",
    "force_exit": "Force Exit",
    "excess_concentration": "Excess Concentration",
    "excess_position_drawdown": "Excess Position Drawdown",
    "partially_ready": "Partially Ready",
    "partially_filled": "Partially Filled",
    "action_submitted": "Action Submitted",
    "action_skipped": "Action Skipped",
    "action_failed": "Action Failed",
    "order_status_reconciled": "Order Status Reconciled",
    "order_terminal_reconciled": "Order Reached Terminal Status",
    "explicitly_skipped": "Explicitly Skipped",
    "lifecycle_action": "Lifecycle Action",
    "current_price": "Current Price",
    "cost_basis": "Cost Basis",
    "qty_available": "Quantity Available",
    "snapshot_ts": "Snapshot Time",
    "authorized_session_date": "Authorized Session Date",
    "expires_at": "Expires At",
    "created_at": "Created At",
    "remaining_minutes": "Minutes Remaining",
    "consumer_status": "Consumer Status",
    "fees_commissions": "Fees And Commissions",
    "proceeds": "Proceeds",
    "open_orders": "Open Orders",
    "position_count": "Position Count",
    "capital_preservation_posture": "Capital Preservation",
    "risk_increasing_actions_blocked": "Risk-Increasing Actions Blocked",
    "governance_mode": "Governance Mode",
    "orange_degraded": "Orange — Degraded",
    "continuous_loop_log": "Continuous Loop Log",
    "watchdog_loop_log": "Watchdog Loop Log",
    "runtime_autostart": "Runtime Autostart",
    "paper_entry_stage_audit": "Paper Entry Stage Audit",
    "paper_execution_audit": "Paper Execution Audit",
    "watchdog_alerts": "Watchdog Alerts",
    "data_freshness_audit": "Data Freshness Audit",
    "execution_readiness": "Execution Readiness",
    "governance_authorization": "Governance Authorization",
    "paper_order_events": "Paper Order Events",
    "preplan_authorized_sell": "Preplan Authorized SELL",
    "recent_orders_snapshot": "Recent Orders Snapshot",
    "fplan_management": "F-Plan Management",
    "is_market_session": "Market Session",
    "previous_recorded_date": "Previous Recorded Date",
    "latest_recorded_date": "Latest Recorded Date",
    "non_session_history_observations": "Non-Session History Observations",
    "history_dates_in_range": "History Dates In Range",
    "history_dates": "History Dates",
    "portfolio_history_disclosure": "Portfolio History Disclosure",
    "coverage_kind": "Coverage",
    "counts_are_complete_period_totals": "Counts Are Complete Period Totals",
    "limited_recent_window": "Limited Recent Window",
    "post_as_of": "Post As-Of",
    "green_fresh": "Shown As Fresh",
    "age_minutes": "Age (min)",
    "minutes_after_as_of": "Minutes After As-Of",
    "date_only": "Date Only",
    "mtime_advisory_only": "Filesystem Time Advisory Only",
    "source_timestamp": "Source Time",
    "period_relationship": "Period Relationship",
    "snapshot_date": "Snapshot Date",
    "selected_period_end": "Selected Period End",
    "period_type": "Period",
    "start_date": "Start Date",
    "end_date": "End Date",
}

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_HEX_ID_RE = re.compile(r"^[0-9a-f]{12,}$", re.IGNORECASE)
_AUTHORITY_RE = re.compile(r"^(sell-arm|sell-dispatch|eg-apr|buy-arm)[-_]", re.IGNORECASE)
_PATH_RE = re.compile(r"[\\/]|\.[a-z]{2,4}$", re.IGNORECASE)


def is_raw_identifier(value: Any) -> bool:
    """True for paths, hashes, UUIDs, authority/dispatch IDs, and similar keys."""
    text = str(value or "").strip()
    if not text:
        return False
    if _PATH_RE.search(text) and (
        "/" in text or "\\" in text or text.endswith((".json", ".csv", ".jsonl", ".py", ".lock"))
    ):
        return True
    if _UUID_RE.match(text):
        return True
    if _HEX_ID_RE.match(text):
        return True
    if _AUTHORITY_RE.match(text):
        return True
    if text.startswith("data/") or text.startswith("data\\"):
        return True
    return False


def _title_token(token: str) -> str:
    raw = token.strip()
    if not raw:
        return raw
    upper = raw.upper()
    if upper in PRESERVED_ACRONYMS or upper in {"BUY", "SELL", "ADD", "HOLD", "TRIM", "EXIT"}:
        if upper == "PNL":
            return "P/L"
        if upper == "SQLITE":
            return "SQLite"
        return upper
    if raw.lower() in {"id", "utc", "api", "etf"}:
        return raw.upper()
    if raw.lower() == "pnl":
        return "P/L"
    return raw[:1].upper() + raw[1:].lower()


def display_label(value: Any, *, technical: bool = False) -> str:
    """Human-readable label for operator views.

    When technical=True, raw identifiers and unknown machine tokens are left unchanged.
    """
    if value is None:
        return "Unavailable"
    text = str(value).strip()
    if not text:
        return "Unavailable"
    if technical and is_raw_identifier(text):
        return text
    key = text.lower().replace(" ", "_")
    if key in KNOWN_LABELS:
        return KNOWN_LABELS[key]
    compact = text.replace("-", "_")
    if compact.lower() in KNOWN_LABELS:
        return KNOWN_LABELS[compact.lower()]
    if is_raw_identifier(text):
        return text
    if "_" not in text and " " not in text and text.isupper() and len(text) > 2:
        known = KNOWN_LABELS.get(text.lower())
        if known:
            return known
        parts = text.split("_") if "_" in text else [text]
        if len(parts) == 1 and not technical:
            return _title_token(text)
        return " ".join(_title_token(p) for p in re.split(r"[_\s]+", text) if p)
    if "_" in text or (text.isupper() and "-" in text):
        return " ".join(_title_token(p) for p in re.split(r"[_\-]+", text) if p)
    if text.islower() and " " not in text and not is_raw_identifier(text):
        return _title_token(text)
    return text


def display_status(value: Any) -> str:
    """Format a status / enum for badges without inventing a new meaning."""
    if value is None or str(value).strip() == "":
        return "Unknown"
    text = str(value).strip()
    if is_raw_identifier(text):
        return text
    return display_label(text)


def format_column_label(name: Any) -> str:
    return display_label(name)


def format_row_for_display(
    row: Mapping[str, Any],
    *,
    visible: Optional[Sequence[str]] = None,
    technical_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return a copy with human-readable keys for visible columns."""
    tech = set(technical_keys or ())
    keys = list(visible) if visible is not None else list(row.keys())
    out: Dict[str, Any] = {}
    for key in keys:
        if key not in row:
            continue
        label = key if key in tech else format_column_label(key)
        val = row.get(key)
        if key in tech:
            out[label] = val
        elif isinstance(val, str) and not is_raw_identifier(val):
            out[label] = display_label(val) if "_" in val or val.isupper() else val
        else:
            out[label] = val
    return out


def technical_payload(row: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    """Preserve raw identifiers for expanders — no reformatting."""
    return {k: row.get(k) for k in keys if k in row}


def signed_text(value: Any, *, currency: bool = True) -> str:
    if value is None or value == "":
        return "Unavailable"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    prefix = "$" if currency else ""
    if num > 0:
        return f"+{prefix}{num:,.2f}"
    if num < 0:
        return f"-{prefix}{abs(num):,.2f}"
    return f"{prefix}{num:,.2f}"


def money_text(value: Any) -> str:
    if value is None or value == "":
        return "Unavailable"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def quantity_text(value: Any) -> str:
    if value is None or value == "":
        return "Unavailable"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num.is_integer():
        return f"{int(num):,}"
    return f"{num:,.4g}"


def percent_text(value: Any, *, signed: bool = False) -> str:
    if value is None or value == "":
        return "Unavailable"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if signed:
        return f"{num:+.2f}%"
    return f"{num:.2f}%"
