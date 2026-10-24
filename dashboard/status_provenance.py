"""Dashboard provenance labels. Presentation only.

Health classifications and alert records stay where they are. This module decides
whether a closeout receipt is current-session evidence or older-session evidence,
and it labels runtime health, the latest market-hours cycle, and the latest
certified accounting close as separate facts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from services.operator_closeout_control import COMPLETED_STATES, STATE_PUBLISHING

ROLE_CURRENT_ERROR = "current_error"
ROLE_CURRENT_SUCCESS = "current_success"
ROLE_CURRENT_WARNING = "current_warning"
ROLE_MISSING = "missing"

HISTORICAL_HEADING = "Historical closeout evidence"
HISTORICAL_NOTE = (
    "Prior-session closeout evidence, kept with its own session date. "
    "These records are historical evidence."
)
CURRENT_RUNTIME_LABEL = "Current runtime health"
MARKET_HOURS_LABEL = "Latest market-hours cycle"
CERTIFIED_CLOSE_LABEL = "Latest certified accounting close"

_PROVEN_SESSION_REASONS = frozenset(
    {"regular_session_open", "pre_open", "regular_close", "early_close"}
)


def applicable_trading_session(now: Optional[Any] = None) -> Optional[str]:
    """Session a closeout receipt must match before it is shown as current.

    Uses the calendar session date while that session is the active trading day.
    On a weekend or holiday, uses the latest completed session. An unproven
    calendar yields no applicable session, so an older receipt stays historical.
    """
    try:
        from services.market_calendar import regular_session_state

        state = regular_session_state(now)
    except Exception:
        return None
    if str(state.get("session_state") or "") == "UNKNOWN":
        return None
    reason = str(state.get("reason") or "")
    session_date = str(state.get("session_date") or "")[:10]
    if reason in _PROVEN_SESSION_REASONS and len(session_date) == 10:
        return session_date
    try:
        from services.nyse_signal_session import required_signal_session

        completed = required_signal_session(now)
    except Exception:
        return None
    text = completed.isoformat() if hasattr(completed, "isoformat") else str(completed)
    text = text[:10]
    return text if len(text) == 10 else None


def classify_closeout_presentation(
    receipts: Optional[List[Mapping[str, Any]]],
    *,
    applicable_session: Optional[str],
) -> Dict[str, Any]:
    """Split receipts into the applicable session and older evidence.

    A failed receipt is a current error only when its session date is the
    applicable trading session. Every other receipt, including older failures,
    stays in the historical list.
    """
    applicable = str(applicable_session or "")[:10] or None
    current: Optional[Mapping[str, Any]] = None
    historical: List[Mapping[str, Any]] = []
    for receipt in receipts or []:
        if not isinstance(receipt, Mapping):
            continue
        session = str(receipt.get("session_date") or "")[:10]
        if applicable and session == applicable:
            if current is not None:
                historical.append(current)
            current = receipt
        else:
            historical.append(receipt)
    historical.sort(key=lambda item: str(item.get("session_date") or ""))
    if current is None:
        role = ROLE_MISSING
    else:
        state = str(current.get("state") or "")
        if state in COMPLETED_STATES:
            role = ROLE_CURRENT_SUCCESS
        elif state == STATE_PUBLISHING:
            role = ROLE_CURRENT_WARNING
        else:
            role = ROLE_CURRENT_ERROR
    return {
        "applicable_session": applicable,
        "current_receipt": None if current is None else dict(current),
        "current_role": role,
        "historical_receipts": [dict(item) for item in historical],
    }


def current_closeout_message(receipt: Mapping[str, Any]) -> str:
    session = receipt.get("session_date") or "unavailable"
    state = receipt.get("state") or "UNKNOWN"
    reason = receipt.get("reason") or "blocked"
    return f"Current closeout {state} for session {session}: {reason}"


def historical_closeout_message(receipt: Mapping[str, Any]) -> str:
    session = receipt.get("session_date") or "unavailable"
    state = receipt.get("state") or "unavailable"
    reason = receipt.get("reason") or "no reason recorded"
    return (
        f"Session {session} · {state} · {reason}. " "Historical closeout evidence for that session."
    )


def missing_closeout_message(applicable_session: Optional[str]) -> str:
    if not applicable_session:
        return (
            "No applicable trading session is proven. "
            "No closeout receipt is shown as a current runtime error."
        )
    return f"No closeout receipt for trading session {applicable_session}."


def render_closeout_provenance(st: Any, presentation: Mapping[str, Any]) -> None:
    """Render current-session status and the historical evidence section."""
    role = str(presentation.get("current_role") or ROLE_MISSING)
    current = presentation.get("current_receipt")
    if role == ROLE_CURRENT_ERROR and isinstance(current, Mapping):
        st.error(current_closeout_message(current))
    elif role == ROLE_CURRENT_WARNING and isinstance(current, Mapping):
        st.warning(current_closeout_message(current))
    elif role == ROLE_CURRENT_SUCCESS and isinstance(current, Mapping):
        st.success(current_closeout_message(current))
    else:
        st.caption(missing_closeout_message(presentation.get("applicable_session")))
    historical = list(presentation.get("historical_receipts") or [])
    if not historical:
        return
    with st.expander(HISTORICAL_HEADING, expanded=False):
        st.caption(HISTORICAL_NOTE)
        for receipt in historical:
            if isinstance(receipt, Mapping):
                st.markdown(historical_closeout_message(receipt))


def provenance_facts(
    *,
    runtime_status: Any,
    runtime_as_of: Any,
    market_hours_outcome: Any,
    market_hours_at: Any,
    certified_session_date: Any,
) -> Dict[str, Any]:
    """Three labeled facts. The certified-close date is not a runtime failure."""
    runtime_date = str(runtime_as_of or "") or None
    certified_date = str(certified_session_date or "") or None
    return {
        "current_runtime_health": str(runtime_status or "UNKNOWN"),
        "current_runtime_as_of": runtime_date,
        "latest_market_hours_outcome": str(market_hours_outcome or "") or None,
        "latest_market_hours_at": str(market_hours_at or "") or None,
        "latest_certified_accounting_close": certified_date,
        "certified_close_implies_runtime_failure": False,
    }


def certified_close_caption(session_date: Any) -> str:
    shown = str(session_date or "") or "unavailable"
    return (
        f"Certified accounting close session {shown}. "
        "This session date and these totals are accounting evidence, "
        "independent of current runtime health."
    )


def render_status_provenance(st: Any, facts: Mapping[str, Any]) -> None:
    """Show runtime health, the market-hours cycle, and the certified close apart."""
    runtime_as_of = facts.get("current_runtime_as_of") or "unavailable"
    cycle_outcome = facts.get("latest_market_hours_outcome") or "unavailable"
    cycle_at = facts.get("latest_market_hours_at") or "unavailable"
    st.markdown(f"**{CURRENT_RUNTIME_LABEL}**")
    st.caption(f"{facts.get('current_runtime_health') or 'UNKNOWN'} as of {runtime_as_of}")
    st.markdown(f"**{MARKET_HOURS_LABEL}**")
    st.caption(f"{cycle_outcome} at {cycle_at}")
    st.markdown(f"**{CERTIFIED_CLOSE_LABEL}**")
    st.caption(certified_close_caption(facts.get("latest_certified_accounting_close")))
