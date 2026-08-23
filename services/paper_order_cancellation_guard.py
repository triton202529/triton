"""
paper_order_cancellation_guard.py
Phase 198T-B — Pure paper order-cancellation authorization.

Cancellation is risk-reducing and is NOT a reviewed-SELL dispatch.
This guard never consumes SELL authority, never creates a dispatch row,
and never authorizes POST/PATCH/submit/replace.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

PURE_CANCEL = "PURE_CANCEL"
CANCEL_AND_REPLACE = "CANCEL_AND_REPLACE"

CAP_MONITOR = "monitor_open_orders.cancel_order"
CAP_RUNTIME_VERIFY = "runtime_verify.really_cancel"
CAP_BROKER_CANCEL = "broker_alpaca.pure_cancel"
CAP_BROKER_CANCEL_ALL = "broker_alpaca.cancel_all"
CAP_CSV_CANCEL = "place_orders_from_csv.cancel_order"

ALLOWED_CAPABILITIES = frozenset(
    {
        CAP_MONITOR,
        CAP_RUNTIME_VERIFY,
        CAP_BROKER_CANCEL,
        CAP_BROKER_CANCEL_ALL,
        CAP_CSV_CANCEL,
    }
)

_TERMINAL_STATUSES = frozenset(
    {
        "canceled",
        "cancelled",
        "filled",
        "expired",
        "rejected",
        "done_for_day",
        "replaced",
    }
)

_ORDER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SECRET_KEYS = (
    "key",
    "secret",
    "token",
    "password",
    "authorization",
    "apca-api-key-id",
    "apca-api-secret-key",
    "account_number",
)


class PaperOrderCancelBlocked(RuntimeError):
    def __init__(self, reason: str, details: Optional[Mapping[str, Any]] = None):
        super().__init__(reason)
        self.reason = str(reason)
        self.details = dict(details or {})


@dataclass(frozen=True)
class CancellationRequest:
    broker_mode: str
    live_execution_enabled: bool
    route_id: str
    cancellation_capability: str
    broker_order_id: str
    order_side: Optional[str] = None
    order_status: Optional[str] = None
    intent: str = PURE_CANCEL
    inspected_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    broker_base_url: Optional[str] = None
    expected_environment: str = "paper"


@dataclass
class CancellationDecision:
    allow_delete: bool
    reason: str
    classification: str
    idempotent_terminal: bool = False
    terminal_status: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def raise_if_blocked(self) -> None:
        if self.allow_delete or self.idempotent_terminal:
            return
        raise PaperOrderCancelBlocked(self.reason, self.details)


def live_execution_is_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    src = env if env is not None else os.environ
    for key in ("TRITON_ENABLE_LIVE_EXECUTION", "TRITON_LIVE_EXECUTION"):
        val = str(src.get(key) or "").strip().lower()
        if val in {"1", "true", "yes", "on"}:
            return True
    return False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sanitize_diagnostics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        lk = str(key).lower()
        if any(part in lk for part in _SECRET_KEYS):
            out[key] = "[redacted]"
        elif isinstance(value, dict):
            out[key] = sanitize_diagnostics(value)
        else:
            out[key] = value
    return out


def _paper_base_url(url: Optional[str]) -> bool:
    text = str(url or "").strip().lower()
    if not text:
        return True
    if "paper-api.alpaca.markets" in text:
        return True
    if "api.alpaca.markets" in text and "paper-api" not in text:
        return False
    if "live" in text and "paper" not in text:
        return False
    return True


def evaluate_paper_order_cancellation(
    request: CancellationRequest,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> CancellationDecision:
    """Authorize a pure DELETE. Never authorizes submit/replace."""
    inspected = request.inspected_at or _utc_now()
    details = sanitize_diagnostics(
        {
            "route_id": request.route_id,
            "capability": request.cancellation_capability,
            "intent": request.intent,
            "broker_mode": request.broker_mode,
            "inspected_at": inspected.isoformat(),
            "correlation_id": request.correlation_id,
        }
    )
    intent = str(request.intent or "").strip().upper()
    if intent not in {PURE_CANCEL, CANCEL_AND_REPLACE}:
        return CancellationDecision(
            False,
            "cancel_intent_invalid",
            "BLOCKED",
            details={**details, "intent": intent},
        )
    if intent in {"SUBMIT", "REPLACE", "CONVERT", "INCREASE"}:
        return CancellationDecision(
            False, "cancel_must_not_submit_or_replace", "BLOCKED", details=details
        )

    cap = str(request.cancellation_capability or "").strip()
    if cap not in ALLOWED_CAPABILITIES:
        return CancellationDecision(
            False,
            "cancellation_capability_missing_or_unrecognized",
            "BLOCKED",
            details=details,
        )

    if live_execution_is_enabled(env) or bool(request.live_execution_enabled):
        return CancellationDecision(
            False, "live_execution_blocked_for_cancellation", "BLOCKED", details=details
        )

    mode = str(request.broker_mode or "").strip().lower()
    expected = str(request.expected_environment or "paper").strip().lower()
    if expected != "paper" or mode not in {"paper", "paper-only", "paper_trading"}:
        return CancellationDecision(
            False, "paper_only_cancellation_required", "BLOCKED", details=details
        )
    if not _paper_base_url(request.broker_base_url):
        return CancellationDecision(
            False, "broker_environment_mismatch", "BLOCKED", details=details
        )

    oid = str(request.broker_order_id or "").strip()
    if cap == CAP_BROKER_CANCEL_ALL:
        marker = oid.upper() if oid else "*"
        if marker not in {"", "*", "ALL"}:
            return CancellationDecision(
                False, "cancel_all_must_use_batch_marker", "BLOCKED", details=details
            )
    elif not oid or not _ORDER_ID_RE.match(oid):
        return CancellationDecision(False, "broker_order_id_malformed", "BLOCKED", details=details)

    status = str(request.order_status or "").strip().lower()
    if status in _TERMINAL_STATUSES:
        return CancellationDecision(
            False,
            "already_terminal_idempotent",
            "IDEMPOTENT_TERMINAL",
            idempotent_terminal=True,
            terminal_status=status,
            details=details,
        )

    side = str(request.order_side or "").strip().lower()
    if cap != CAP_BROKER_CANCEL_ALL and not side:
        return CancellationDecision(
            False,
            "order_side_unknown",
            "BLOCKED",
            details={**details, "policy": "unknown_side_fail_closed_no_submit"},
        )
    if side and side not in {"buy", "sell"}:
        return CancellationDecision(False, "order_side_malformed", "BLOCKED", details=details)

    return CancellationDecision(
        True,
        "pure_cancel_allowed",
        "ALLOW",
        details={**details, "order_side": side or None, "does_not_consume_sell_authority": True},
    )


def classify_delete_http_outcome(
    status_code: Optional[int],
    *,
    timeout: bool = False,
    exception: Optional[BaseException] = None,
) -> CancellationDecision:
    if timeout or (exception is not None and "timeout" in str(exception).lower()):
        return CancellationDecision(
            False,
            "cancel_outcome_unknown_timeout",
            "OUTCOME_UNKNOWN",
            details={"retry": False, "requires_reconciliation": True},
        )
    if exception is not None and status_code is None:
        return CancellationDecision(
            False,
            "cancel_outcome_unknown_exception",
            "OUTCOME_UNKNOWN",
            details={"retry": False, "requires_reconciliation": True},
        )
    code = int(status_code or 0)
    if code in {200, 204}:
        return CancellationDecision(
            False, "cancelled", "CANCELLED", idempotent_terminal=True, terminal_status="canceled"
        )
    if code in {404, 422}:
        return CancellationDecision(
            False,
            "already_gone_or_not_found",
            "IDEMPOTENT_TERMINAL",
            idempotent_terminal=True,
            terminal_status="not_found",
        )
    if code in {401, 403, 409}:
        return CancellationDecision(
            False, "cancel_rejected", "REJECTED", details={"http_status": code}
        )
    return CancellationDecision(
        False,
        "cancel_outcome_unknown",
        "OUTCOME_UNKNOWN",
        details={"http_status": code, "retry": False, "requires_reconciliation": True},
    )
