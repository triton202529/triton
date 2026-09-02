"""
command_center_status_coherence.py

Pure read-only helpers for Command Center status coherence.

- Source-aware timestamp parsing (Z / offset / naive-local lock times)
- Unified market-status resolution (no broker calls)
- Paper certification display vs live permission separation
- Shutdown request/ack presentation (expiry + request-id correlation)

Does not write artifacts, call brokers, or invoke protected controls.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]

# Lock writers (scripts/loop_lock_utils.ps1) emit naive local timestamps via Get-Date -Format s.
LOCK_TIMESTAMP_SOURCE = "loop_lock_utils_naive_local"
UTC_JSON_SOURCE = "utc_json"

CONTINUOUS_INTERVAL_MINUTES = 45.0
# Proven grace matches runtime classifiers: max(60, interval * 1.5) total age threshold.
CONTINUOUS_CYCLE_OVERDUE_MINUTES = max(60.0, CONTINUOUS_INTERVAL_MINUTES * 1.5)
CONTINUOUS_CYCLE_GRACE_AFTER_DEADLINE_MINUTES = (
    CONTINUOUS_CYCLE_OVERDUE_MINUTES - CONTINUOUS_INTERVAL_MINUTES
)

PAPER_CERTIFICATION_STATUSES = frozenset(
    {
        "CERTIFIED_FOR_PAPER_PROTECTION",
        "CONDITIONALLY_CERTIFIED",
        "PARTIALLY_CERTIFIED",
    }
)

ClockFn = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def local_tzinfo():
    """Host local timezone used by PowerShell Get-Date -Format s lock writers."""
    return datetime.now().astimezone().tzinfo or timezone.utc


def parse_status_timestamp(
    value: Any,
    *,
    source: str = UTC_JSON_SOURCE,
    naive_tz: Optional[Any] = None,
) -> Optional[datetime]:
    """Parse a status timestamp into timezone-aware UTC.

    - Z / +00:00 / explicit offsets: preserve the stated instant.
    - Naive values: interpret using source semantics (lock → local; JSON → UTC).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            if source == LOCK_TIMESTAMP_SOURCE:
                dt = dt.replace(tzinfo=naive_tz or local_tzinfo())
            else:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    try:
        s = str(value).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        if " " in s and "T" not in s:
            s = s.replace(" ", "T", 1)
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            if source == LOCK_TIMESTAMP_SOURCE:
                dt = dt.replace(tzinfo=naive_tz or local_tzinfo())
            else:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def parse_lock_timestamp(value: Any, *, naive_tz: Optional[Any] = None) -> Optional[datetime]:
    return parse_status_timestamp(value, source=LOCK_TIMESTAMP_SOURCE, naive_tz=naive_tz)


def parse_utc_timestamp(value: Any) -> Optional[datetime]:
    return parse_status_timestamp(value, source=UTC_JSON_SOURCE)


def age_minutes(ts: Optional[datetime], now: Optional[datetime] = None) -> Optional[float]:
    if ts is None:
        return None
    now_utc = (now or utc_now()).astimezone(timezone.utc)
    return (now_utc - ts.astimezone(timezone.utc)).total_seconds() / 60.0


def cycle_deadline_utc(
    refreshed_at: Optional[datetime],
    *,
    interval_minutes: float = CONTINUOUS_INTERVAL_MINUTES,
    root: Optional[Path] = None,
) -> Optional[datetime]:
    """Authoritative next-due: cadence artifact when present, else start+interval."""
    try:
        from services.continuous_loop_cadence import authoritative_next_due_utc

        artifact_due = authoritative_next_due_utc(root)
        if artifact_due is not None:
            return artifact_due
    except Exception:
        pass
    if refreshed_at is None:
        return None
    return refreshed_at.astimezone(timezone.utc) + timedelta(minutes=float(interval_minutes))


def cycle_is_overdue(
    refreshed_at: Optional[datetime],
    *,
    now: Optional[datetime] = None,
    interval_minutes: float = CONTINUOUS_INTERVAL_MINUTES,
    overdue_minutes: float = CONTINUOUS_CYCLE_OVERDUE_MINUTES,
) -> bool:
    """True only after interval + grace (overdue_minutes total age)."""
    age = age_minutes(refreshed_at, now)
    if age is None:
        return False
    return age > float(overdue_minutes)


def resolve_market_status(
    *,
    now: Optional[datetime] = None,
    market_open_fn: Optional[Callable[[datetime], bool]] = None,
) -> Dict[str, Any]:
    """Single read-only market resolver for main panel and sidebar. No broker calls."""
    now_utc = (now or utc_now()).astimezone(timezone.utc)
    open_fn = market_open_fn
    if open_fn is None:
        try:
            from services.guard_auto import is_market_open_et

            open_fn = is_market_open_et
        except Exception:
            open_fn = None

    if open_fn is None:
        return {
            "status": "UNKNOWN",
            "market_open": None,
            "source": "guard_auto.is_market_open_et",
            "as_of": now_utc.isoformat(),
            "note": "Market session classifier unavailable",
        }

    try:
        is_open = bool(open_fn(now_utc))
    except Exception:
        return {
            "status": "UNKNOWN",
            "market_open": None,
            "source": "guard_auto.is_market_open_et",
            "as_of": now_utc.isoformat(),
            "note": "Market session classification failed",
        }

    return {
        "status": "OPEN" if is_open else "CLOSED",
        "market_open": is_open,
        "source": "guard_auto.is_market_open_et",
        "as_of": now_utc.isoformat(),
        "note": "US equities session (ET); informational only",
    }


def format_paper_certification_display(raw: Any) -> Dict[str, Any]:
    """Qualified PAPER certification display; never implies live permission."""
    status = str(raw or "").strip()
    if status in PAPER_CERTIFICATION_STATUSES:
        return {
            "raw_status": status,
            "display_status": f"PAPER:{status}",
            "paper_certified": True,
            "mode": "PAPER",
        }
    if status == "NOT_CERTIFIED":
        return {
            "raw_status": status,
            "display_status": "PAPER:NOT_CERTIFIED",
            "paper_certified": False,
            "mode": "PAPER",
        }
    if status:
        return {
            "raw_status": status,
            "display_status": f"PAPER:REVIEW ({status})",
            "paper_certified": False,
            "mode": "PAPER",
        }
    return {
        "raw_status": "",
        "display_status": "PAPER:UNKNOWN",
        "paper_certified": False,
        "mode": "PAPER",
    }


def format_live_permission_display(*, live_permitted: bool) -> Dict[str, Any]:
    blocked = not bool(live_permitted)
    return {
        "live_execution_permitted": bool(live_permitted),
        "live_execution_blocked": blocked,
        "display_status": "LIVE:BLOCKED" if blocked else "LIVE:PERMITTED",
        "mode": "LIVE",
    }


def resolve_shutdown_presentation(
    *,
    root: Optional[Path] = None,
    now: Optional[datetime] = None,
    request: Optional[Dict[str, Any]] = None,
    ack_continuous: Optional[Dict[str, Any]] = None,
    ack_watchdog: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read-only shutdown presentation: expiry + request-id correlation.

    Expired requests are inactive history and must not degrade current health.
    """
    root = Path(root) if root is not None else ROOT
    live = root / "data" / "live"
    now_utc = (now or utc_now()).astimezone(timezone.utc)

    def _read(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            import json

            doc = json.loads(path.read_text(encoding="utf-8"))
            return doc if isinstance(doc, dict) else {}
        except Exception:
            return {"__malformed__": True}

    req = request if request is not None else _read(live / "operator_shutdown_request.json")
    ack_c = (
        ack_continuous
        if ack_continuous is not None
        else _read(live / "operator_shutdown_ack_continuous.json")
    )
    ack_w = (
        ack_watchdog
        if ack_watchdog is not None
        else _read(live / "operator_shutdown_ack_watchdog.json")
    )

    if not req or req.get("__malformed__"):
        return {
            "active": False,
            "pending": False,
            "expired": False,
            "historical": False,
            "request_id": None,
            "request_state": None,
            "display_status": "NONE",
            "affects_current_health": False,
            "ack_continuous_attributed": False,
            "ack_watchdog_attributed": False,
            "ack_continuous_request_id": None,
            "ack_watchdog_request_id": None,
            "presentation": "No cooperative shutdown request on file.",
        }

    req_id = str(req.get("request_id") or "").strip() or None
    state = str(req.get("state") or "CREATED")
    exp = parse_utc_timestamp(req.get("expires_at"))
    terminal = {
        "COMPLETED",
        "PARTIAL_COMPLETION",
        "EXPIRED",
        "SUPERSEDED",
        "TIMED_OUT",
        "FAILED",
        "REFUSED_IDENTITY_MISMATCH",
        "REFUSED_UNSAFE_STATE",
    }
    expired = bool(exp is not None and exp <= now_utc) or state == "EXPIRED"
    active = bool(exp is not None and exp > now_utc and state not in terminal)

    def _ack_matches(ack: Dict[str, Any]) -> bool:
        if not ack or ack.get("__malformed__"):
            return False
        ack_rid = str(ack.get("request_id") or "").strip() or None
        return bool(req_id and ack_rid and ack_rid == req_id)

    ack_c_ok = _ack_matches(ack_c)
    ack_w_ok = _ack_matches(ack_w)

    if active:
        display = f"ACTIVE ({state})"
        presentation = (
            f"Active cooperative shutdown request {req_id} state={state}; "
            "acks attributed only when request_id matches."
        )
    elif expired:
        display = f"INACTIVE_HISTORY (expired; was {state})"
        presentation = (
            f"Expired shutdown request {req_id} is inactive history and does not "
            "affect current runtime health."
        )
    elif state in terminal:
        display = f"INACTIVE_HISTORY ({state})"
        presentation = (
            f"Terminal shutdown request {req_id} is inactive history and does not "
            "affect current runtime health."
        )
    else:
        display = f"INACTIVE ({state})"
        presentation = f"Shutdown request {req_id} is not currently active."

    return {
        "active": active,
        "pending": active,
        "expired": expired,
        "historical": bool(expired or state in terminal),
        "request_id": req_id,
        "request_state": state,
        "expires_at": exp.isoformat() if exp else None,
        "display_status": display,
        "affects_current_health": active,
        "ack_continuous_attributed": ack_c_ok,
        "ack_watchdog_attributed": ack_w_ok,
        "ack_continuous_request_id": (
            str(ack_c.get("request_id") or "").strip() or None if ack_c else None
        ),
        "ack_watchdog_request_id": (
            str(ack_w.get("request_id") or "").strip() or None if ack_w else None
        ),
        "presentation": presentation,
    }


def authorization_status_text(value: Any) -> str:
    """Normalize governance authorization for display (booleans are not 'True' blocks)."""
    if value is True:
        return "AUTHORIZED"
    if value is False:
        return "DENIED"
    return str(value or "").strip()


# Keep ZoneInfo import referenced for environments that prefer named zones in tests.
_ET_FALLBACK = ZoneInfo("America/New_York")
