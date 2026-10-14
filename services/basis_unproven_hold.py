"""Permanent broker-boundary hold for symbols whose opening basis is unproven.

The policy file names the held symbols and binds the pending partial statement.
Code constants are the fail-closed floor: a missing or mismatched file cannot
release VOO or XLK, and it cannot authorize an order for either symbol.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

SCHEMA = "BASIS-UNPROVEN-HOLD-1"
HOLD_SYMBOLS = frozenset({"VOO", "XLK"})
APPROVAL_IDENTITY = "Akim Wallace — Authorized Paper Trading Operator"
REPORTING_LABEL = "Basis-unproven hold — order submission blocked"
CLASSIFICATION = "BASIS_UNPROVEN_HOLD"
POSITION_DISPOSITION = "MAY_REMAIN_HELD"

POLICY_RELATIVE = "data/governance/basis_unproven_hold.json"
AUDIT_RELATIVE = "data/governance/basis_unproven_hold_audit.jsonl"
EVENTS_RELATIVE = "data/results/performance_accounting_events.jsonl"

_OVERRIDE_FIELDS = (
    "symbol",
    "approval_identity",
    "authorized_at_utc",
    "reason",
    "authorization_reference",
)

_policy_path: ContextVar[Optional[str]] = ContextVar(
    "basis_unproven_hold_policy_path", default=None
)
_audit_path: ContextVar[Optional[str]] = ContextVar("basis_unproven_hold_audit_path", default=None)
_events_path: ContextVar[Optional[str]] = ContextVar(
    "basis_unproven_hold_events_path", default=None
)
_STATEMENT_HASH = re.compile(r"[0-9a-f]{64}")


class BasisUnprovenHoldBlocked(RuntimeError):
    """Raised before broker HTTP when a basis-unproven hold symbol would be submitted."""

    def __init__(self, record: Mapping[str, Any]):
        self.record = dict(record)
        self.reason = str(self.record.get("reason") or "")
        super().__init__(str(self.record.get("reporting_label") or self.reason))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def policy_path() -> Path:
    override = _policy_path.get()
    if override:
        return Path(override)
    return repo_root() / POLICY_RELATIVE


def audit_path() -> Path:
    override = _audit_path.get()
    if override:
        return Path(override)
    return repo_root() / AUDIT_RELATIVE


def events_path() -> Path:
    override = _events_path.get()
    if override:
        return Path(override)
    return repo_root() / EVENTS_RELATIVE


def canonical_override_sha256(record: Mapping[str, Any]) -> str:
    body = {field: record.get(field) for field in _OVERRIDE_FIELDS}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _valid_statement_hash(document: Mapping[str, Any]) -> Optional[str]:
    bound = str(document.get("bound_partial_statement_sha256") or "").strip().lower()
    if _STATEMENT_HASH.fullmatch(bound):
        return bound
    return None


def _committed_policy_bytes() -> Optional[bytes]:
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{POLICY_RELATIVE}"],
            cwd=repo_root(),
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_policy() -> Dict[str, Any]:
    path = policy_path()
    if not path.is_file():
        return {"ok": False, "reason": "basis_unproven_hold_policy_missing", "policy_sha256": None}
    try:
        raw = path.read_bytes()
    except OSError:
        return {
            "ok": False,
            "reason": "basis_unproven_hold_policy_unreadable",
            "policy_sha256": None,
        }
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {
            "ok": False,
            "reason": "basis_unproven_hold_policy_malformed",
            "policy_sha256": None,
        }
    if not isinstance(document, dict):
        return {
            "ok": False,
            "reason": "basis_unproven_hold_policy_malformed",
            "policy_sha256": None,
        }
    digest = _sha256_bytes(raw)
    bound = _valid_statement_hash(document)
    structure = _policy_structure(document)
    if structure == "malformed":
        return {
            "ok": False,
            "reason": "basis_unproven_hold_policy_malformed",
            "policy_sha256": digest,
            "bound_statement_sha256": bound,
        }
    if structure == "invalid_hash":
        return {
            "ok": False,
            "reason": "basis_unproven_hold_policy_invalid_hash",
            "policy_sha256": digest,
            "bound_statement_sha256": None,
        }
    if structure == "mismatched":
        return {
            "ok": False,
            "reason": "basis_unproven_hold_policy_mismatched",
            "policy_sha256": digest,
            "bound_statement_sha256": bound,
        }
    if not _policy_path.get():
        committed = _committed_policy_bytes()
        if committed is None or committed != raw:
            return {
                "ok": False,
                "reason": "basis_unproven_hold_policy_mismatched",
                "policy_sha256": digest,
                "bound_statement_sha256": bound,
            }
    return {
        "ok": True,
        "document": document,
        "policy_sha256": digest,
        "bound_statement_sha256": bound,
    }


def _policy_structure(document: Mapping[str, Any]) -> str:
    rows = document.get("symbols")
    overrides = document.get("operator_overrides")
    if not isinstance(rows, list) or not isinstance(overrides, list):
        return "malformed"
    if any(not isinstance(row, dict) for row in rows):
        return "malformed"
    if any(not isinstance(row, dict) for row in overrides):
        return "malformed"
    if _valid_statement_hash(document) is None:
        return "invalid_hash"
    if not _policy_matches_constants(document):
        return "mismatched"
    return "ok"


def _policy_matches_constants(document: Mapping[str, Any]) -> bool:
    if document.get("schema") != SCHEMA:
        return False
    if document.get("status") != "ACTIVE":
        return False
    if document.get("approval_identity") != APPROVAL_IDENTITY:
        return False
    if not str(document.get("created_at_utc") or "").strip():
        return False
    rows = document.get("symbols")
    found = set()
    for row in rows:
        if row.get("classification") != CLASSIFICATION:
            return False
        if row.get("position_disposition") != POSITION_DISPOSITION:
            return False
        found.add(_symbol(row.get("symbol")))
    return found == HOLD_SYMBOLS


def _override_releases(document: Mapping[str, Any], symbol: str) -> bool:
    for row in document.get("operator_overrides") or []:
        if _symbol(row.get("symbol")) != symbol:
            continue
        if str(row.get("approval_identity") or "") != APPROVAL_IDENTITY:
            continue
        if any(not str(row.get(field) or "").strip() for field in _OVERRIDE_FIELDS):
            continue
        digest = str(row.get("override_sha256") or "").strip().lower()
        if digest and digest == canonical_override_sha256(row):
            return True
    return False


def _published_opening_releases(symbol: str) -> bool:
    from services.performance_accounting import (
        EVENT_FORWARD_OPENING_BASIS,
        PerformanceAccountingError,
        _reject_backdated_forward_openings,
        canonical_symbol,
        load_events,
    )

    try:
        events = load_events(events_path())
        _reject_backdated_forward_openings(events)
    except (OSError, PerformanceAccountingError):
        return False
    key = f"forward_opening_basis:{symbol}"
    for event in events:
        if event.get("event_type") != EVENT_FORWARD_OPENING_BASIS:
            continue
        if str(event.get("logical_key") or "") != key:
            continue
        if canonical_symbol((event.get("body") or {}).get("symbol")) != symbol:
            continue
        return True
    return False


def _record(
    *,
    reason: str,
    symbol: str,
    side: str,
    source: str,
    policy_sha256: Optional[str],
    bound_statement_sha256: Optional[str],
    client_order_id: Optional[str],
) -> Dict[str, Any]:
    return {
        "result": "BLOCKED",
        "reason": reason,
        "reporting_label": REPORTING_LABEL,
        "symbol": symbol or None,
        "side": side or None,
        "source": source,
        "policy_sha256": policy_sha256,
        "bound_statement_sha256": bound_statement_sha256,
        "client_order_id": client_order_id,
        "broker_called": False,
    }


def _append_audit(record: Mapping[str, Any]) -> None:
    line = json.dumps(dict(record), sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    fd = os.open(audit_path(), flags)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def _block(
    *,
    reason: str,
    symbol: str,
    side: str,
    source: str,
    policy_sha256: Optional[str],
    bound_statement_sha256: Optional[str],
    client_order_id: Optional[str],
) -> None:
    record = _record(
        reason=reason,
        symbol=symbol,
        side=side,
        source=source,
        policy_sha256=policy_sha256,
        bound_statement_sha256=bound_statement_sha256,
        client_order_id=client_order_id,
    )
    try:
        _append_audit(record)
    except OSError:
        record = _record(
            reason="basis_unproven_hold_audit_unwritable",
            symbol=symbol,
            side=side,
            source=source,
            policy_sha256=policy_sha256,
            bound_statement_sha256=bound_statement_sha256,
            client_order_id=client_order_id,
        )
    raise BasisUnprovenHoldBlocked(record)


def evaluate_basis_unproven_hold_for_packet() -> Dict[str, Any]:
    """Read-only hold decision for packet generation and arming.

    Uses the same policy load and release rules as the broker boundary.
    Does not append the hold audit, authorize an order, or call a broker.
    Policy validation failure fails closed: every hold symbol stays blocked.
    """
    loaded = _load_policy()
    if not loaded.get("ok"):
        return {
            "ok": False,
            "reason": str(loaded.get("reason") or "basis_unproven_hold_policy_unreadable"),
            "blocked_symbols": sorted(HOLD_SYMBOLS),
            "policy_sha256": loaded.get("policy_sha256"),
            "bound_statement_sha256": loaded.get("bound_statement_sha256"),
        }
    document = loaded["document"]
    blocked: List[str] = []
    for symbol in sorted(HOLD_SYMBOLS):
        if _override_releases(document, symbol) or _published_opening_releases(symbol):
            continue
        blocked.append(symbol)
    return {
        "ok": True,
        "reason": "ok",
        "blocked_symbols": blocked,
        "policy_sha256": loaded.get("policy_sha256"),
        "bound_statement_sha256": loaded.get("bound_statement_sha256"),
    }


def enforce_basis_unproven_hold_on_order_post(
    payload: Mapping[str, Any],
    *,
    source: str = "broker_alpaca._post",
) -> None:
    """Block /v2/orders for a basis-unproven hold symbol. Does not perform HTTP."""
    symbol = _symbol(payload.get("symbol"))
    side = str(payload.get("side") or "").strip().lower()
    client_order_id = payload.get("client_order_id")
    client = str(client_order_id).strip() if client_order_id not in (None, "") else None
    if symbol not in HOLD_SYMBOLS:
        if symbol:
            return
        loaded = _load_policy()
        _block(
            reason="basis_unproven_hold_symbol_unreadable",
            symbol=symbol,
            side=side,
            source=source,
            policy_sha256=loaded.get("policy_sha256"),
            bound_statement_sha256=loaded.get("bound_statement_sha256"),
            client_order_id=client,
        )
    loaded = _load_policy()
    if not loaded.get("ok"):
        _block(
            reason=str(loaded.get("reason") or "basis_unproven_hold_policy_unreadable"),
            symbol=symbol,
            side=side,
            source=source,
            policy_sha256=loaded.get("policy_sha256"),
            bound_statement_sha256=loaded.get("bound_statement_sha256"),
            client_order_id=client,
        )
    document = loaded["document"]
    if _override_releases(document, symbol) or _published_opening_releases(symbol):
        return
    _block(
        reason="basis_unproven_hold",
        symbol=symbol,
        side=side,
        source=source,
        policy_sha256=loaded.get("policy_sha256"),
        bound_statement_sha256=loaded.get("bound_statement_sha256"),
        client_order_id=client,
    )
