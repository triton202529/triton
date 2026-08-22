"""Production writer for paper SELL dispatch session evidence.

Writes data/governance/independent_dispatch_bindings.json (or an injected
store_root) from services.market_calendar.regular_session_state. Does not
authorize dispatch by itself; runtime enforcement independently re-derives
the calendar and reconciles this bound evidence.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from services.market_calendar import (
    SESSION_CLOSED,
    SESSION_OPEN,
    SESSION_UNKNOWN,
    regular_session_state,
)
from services.paper_sell_arm_authority import (
    LIVE_GOVERNANCE,
    _atomic_write_json,
    _utc_now,
)

SCHEMA_VERSION = "198R-J-DISPATCH-SESSION-EVIDENCE-1"
BINDINGS_FILENAME = "independent_dispatch_bindings.json"
AUTHORITATIVE_SOURCE = "services.market_calendar.regular_session_state"
MARKET_SESSION_TIMEZONE = "America/New_York"
INVOCATION_ENV = "TRITON_MARKET_OPEN_INVOCATION_ID"


class SessionEvidenceError(Exception):
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


def bindings_path(store_root: Optional[Path] = None) -> Path:
    return Path(store_root or LIVE_GOVERNANCE) / BINDINGS_FILENAME


def _aware_utc(now: Optional[datetime]) -> datetime:
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise SessionEvidenceError("market_session_naive_timestamp", {"now": str(now)})
    return now.astimezone(timezone.utc)


def resolve_current_invocation_id(invocation_id: Optional[str] = None) -> str:
    if invocation_id and str(invocation_id).strip():
        return str(invocation_id).strip()
    env = str(os.environ.get(INVOCATION_ENV) or "").strip()
    if env:
        return env
    return ""


def load_dispatch_session_evidence(store_root: Optional[Path] = None) -> Dict[str, Any]:
    path = bindings_path(store_root)
    if not path.is_file():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        if raw.startswith("\ufeff"):
            raise SessionEvidenceError("session_evidence_utf8_bom", {"path": str(path)})
        import json

        doc = json.loads(raw)
    except SessionEvidenceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SessionEvidenceError("session_evidence_unreadable", {"error": str(exc)}) from exc
    if not isinstance(doc, dict):
        raise SessionEvidenceError("session_evidence_malformed", {"type": type(doc).__name__})
    return doc


def calendar_payload_to_session_fields(payload: Mapping[str, Any]) -> Dict[str, Any]:
    state = str(payload.get("session_state") or SESSION_UNKNOWN).upper()
    if state == SESSION_OPEN:
        market_session = "MARKET_OPEN"
    elif state == SESSION_CLOSED:
        market_session = "CLOSED"
    else:
        market_session = "UNKNOWN"
    return {
        "session_state": state,
        "market_session": market_session,
        "session_date": payload.get("session_date"),
        "session_open": payload.get("session_open_et"),
        "session_close": payload.get("session_close_et"),
        "reason": payload.get("reason"),
        "source": payload.get("source") or AUTHORITATIVE_SOURCE,
        "as_of_utc": payload.get("as_of_utc"),
        "as_of_et": payload.get("as_of_et"),
    }


def build_dispatch_session_evidence(
    *,
    now: datetime,
    invocation_id: str,
    calendar_payload: Mapping[str, Any],
    live_execution_blocked: bool = True,
    mode: str = "PAPER",
) -> Dict[str, Any]:
    fields = calendar_payload_to_session_fields(calendar_payload)
    observed = _aware_utc(now)
    return {
        "schema_version": SCHEMA_VERSION,
        "market_open_invocation_id": str(invocation_id or "").strip(),
        "session_date": fields.get("session_date"),
        "market_session": fields.get("market_session"),
        "market_session_observed_at": observed.isoformat(),
        "authoritative_source": AUTHORITATIVE_SOURCE,
        "market_session_timezone": MARKET_SESSION_TIMEZONE,
        "session_open": fields.get("session_open"),
        "session_close": fields.get("session_close"),
        "mode": str(mode or "PAPER").upper(),
        "live_execution_blocked": bool(live_execution_blocked),
        "calendar_reason": fields.get("reason"),
        "calendar_as_of_utc": fields.get("as_of_utc"),
        "calendar_as_of_et": fields.get("as_of_et"),
    }


def validate_dispatch_session_evidence_document(
    doc: Mapping[str, Any],
    *,
    now: datetime,
    expected_invocation_id: Optional[str] = None,
    calendar_payload: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Fail-closed structural checks for a 198R-J evidence document."""
    details: Dict[str, Any] = {"schema_version": doc.get("schema_version")}
    if str(doc.get("schema_version") or "") != SCHEMA_VERSION:
        return False, "session_evidence_schema_mismatch", details
    inv = str(doc.get("market_open_invocation_id") or "").strip()
    if not inv:
        return False, "session_evidence_invocation_missing", details
    expected = str(expected_invocation_id or "").strip()
    if expected and expected != inv:
        details["expected_invocation_id"] = expected
        details["evidence_invocation_id"] = inv
        return False, "session_evidence_invocation_mismatch", details
    if str(doc.get("mode") or "").upper() != "PAPER":
        return False, "session_evidence_mode_not_paper", details
    if doc.get("live_execution_blocked") is not True:
        return False, "session_evidence_live_not_blocked", details
    if str(doc.get("authoritative_source") or "") != AUTHORITATIVE_SOURCE:
        return False, "session_evidence_source_mismatch", details
    if str(doc.get("market_session_timezone") or "") != MARKET_SESSION_TIMEZONE:
        return False, "session_evidence_invalid_timezone", details
    if str(doc.get("market_session") or "").upper() not in {"MARKET_OPEN", "OPEN"}:
        return False, "session_evidence_not_open", details
    if not doc.get("session_date"):
        return False, "session_evidence_session_date_missing", details
    if not doc.get("session_open") or not doc.get("session_close"):
        return False, "session_evidence_session_bounds_missing", details
    if calendar_payload is not None:
        derived = calendar_payload_to_session_fields(calendar_payload)
        if str(derived.get("session_state") or "") == SESSION_UNKNOWN:
            return False, "market_session_derivation_unknown", {**details, **derived}
        if str(derived.get("session_state") or "") != SESSION_OPEN:
            return False, "market_session_not_open", {**details, **derived}
        if str(derived.get("session_date") or "") != str(doc.get("session_date") or ""):
            return False, "session_evidence_session_date_mismatch", details
        if str(derived.get("market_session") or "") not in {"MARKET_OPEN", "OPEN"}:
            return False, "session_evidence_derived_not_open", details
    return True, "ok", details


def write_dispatch_session_evidence(
    *,
    store_root: Optional[Path] = None,
    now: Optional[datetime] = None,
    invocation_id: Optional[str] = None,
    live_execution_blocked: Optional[bool] = None,
    calendar_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Atomically write OPEN session evidence or raise SessionEvidenceError.

    Never writes OPEN evidence for UNKNOWN/CLOSED/malformed calendar state.
    """
    now_utc = _aware_utc(now if now is not None else _utc_now())
    inv = resolve_current_invocation_id(invocation_id)
    if not inv:
        raise SessionEvidenceError("session_evidence_invocation_missing", {})
    payload = (
        dict(calendar_payload)
        if calendar_payload is not None
        else regular_session_state(now=now_utc)
    )
    fields = calendar_payload_to_session_fields(payload)
    state = str(fields.get("session_state") or SESSION_UNKNOWN)
    if state == SESSION_UNKNOWN:
        raise SessionEvidenceError("market_session_derivation_unknown", dict(fields))
    if state == SESSION_CLOSED:
        raise SessionEvidenceError("market_session_not_open", dict(fields))
    if state != SESSION_OPEN:
        raise SessionEvidenceError("market_session_not_open", dict(fields))

    blocked = live_execution_blocked
    if blocked is None:
        from services.live_block_classifier import evaluate_live_block_status

        blocked = bool(evaluate_live_block_status(paper_mode=True).get("live_blocked"))
    if not blocked:
        raise SessionEvidenceError(
            "live_execution_not_blocked", {"live_execution_blocked": blocked}
        )

    doc = build_dispatch_session_evidence(
        now=now_utc,
        invocation_id=inv,
        calendar_payload=payload,
        live_execution_blocked=True,
        mode="PAPER",
    )
    ok, reason, details = validate_dispatch_session_evidence_document(
        doc,
        now=now_utc,
        expected_invocation_id=inv,
        calendar_payload=payload,
    )
    if not ok:
        raise SessionEvidenceError(reason, details)

    root = Path(store_root or LIVE_GOVERNANCE)
    path = root / BINDINGS_FILENAME
    try:
        from services.paper_sell_arm_authority import DispatchDurabilityError

        _atomic_write_json(path, doc, require_full_durability=False)
    except (OSError, DispatchDurabilityError) as exc:
        raise SessionEvidenceError(
            "session_evidence_atomic_write_failed", {"error": type(exc).__name__}
        ) from exc
    # Re-read to refuse BOM / partial writes before callers attempt dispatch.
    loaded = load_dispatch_session_evidence(root)
    ok, reason, details = validate_dispatch_session_evidence_document(
        loaded,
        now=now_utc,
        expected_invocation_id=inv,
        calendar_payload=payload,
    )
    if not ok:
        raise SessionEvidenceError(reason, details)
    return loaded
