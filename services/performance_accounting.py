"""Phase 198T-BN2A — append-only forward performance-accounting foundation.

Isolated module. Import creates no files, contacts no broker, and does not
change BUY/SELL decisions. All output paths are injectable.

Cost-basis method: MOVING_WEIGHTED_AVERAGE (performance only, not tax).
Never mix with reconstructed FIFO history. Never treat SELL proceeds as profit.
Never assume fees are zero. Never invent missing decision prices or basis.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from services.performance_publication_authority import (
    COMMAND_CUTOVER,
    COMMAND_SESSION_CLOSE,
    PublicationAuthorityError,
    destination_requires_authority,
    preflight_publication_authority,
    require_publication_authority,
    sha256_bytes,
)

SCHEMA_VERSION = "198T-BN2A-PERFORMANCE-ACCOUNTING-1"
COST_BASIS_METHOD = "MOVING_WEIGHTED_AVERAGE"
TIMEZONE_NAME = "America/New_York"
NY_TZ = ZoneInfo(TIMEZONE_NAME)

EVENTS_RELATIVE = "data/results/performance_accounting_events.jsonl"
LOCK_RELATIVE = "data/results/performance_accounting_events.lock"

PROPOSED_CUTOVER_UTC = "2026-09-09T17:07:00+00:00"
PROPOSED_CUTOVER_EQUITY = Decimal("92358.02")
PROPOSED_CUTOVER_CASH = Decimal("65663.61")

EVENT_CUTOVER = "ACCOUNTING_CUTOVER"
EVENT_SESSION_CLOSE = "SESSION_CLOSE"
EVENT_POSITION_SNAPSHOT = "POSITION_SNAPSHOT"
EVENT_FILL_BUY = "FILL_BUY"
EVENT_FILL_SELL = "FILL_SELL"
EVENT_CASH_DEPOSIT = "CASH_DEPOSIT"
EVENT_CASH_WITHDRAWAL = "CASH_WITHDRAWAL"
EVENT_CASH_ADJUSTMENT = "CASH_ADJUSTMENT"
EVENT_DIVIDEND = "DIVIDEND"
EVENT_FEE = "FEE"
EVENT_PRICE_MARKS = "PRICE_MARKS"
EVENT_BENCHMARK_CLOSE = "BENCHMARK_CLOSE"
EVENT_ACTION_FAILED = "ACTION_FAILED"
EVENT_ACTION_REJECTED = "ACTION_REJECTED"
EVENT_ACTION_SKIPPED = "ACTION_SKIPPED"

FILL_EVENTS = frozenset({EVENT_FILL_BUY, EVENT_FILL_SELL})
EXTERNAL_FLOW_EVENTS = frozenset({EVENT_CASH_DEPOSIT, EVENT_CASH_WITHDRAWAL, EVENT_CASH_ADJUSTMENT})
NO_BASIS_EVENTS = frozenset({EVENT_ACTION_FAILED, EVENT_ACTION_REJECTED, EVENT_ACTION_SKIPPED})
CERTIFIED_CLOSE_KINDS = frozenset({"regular_close", "early_close"})
PLANNING_MARKERS = frozenset(
    {
        "planned",
        "plan",
        "paper_execution_audit",
        "legacy_plan",
        "manage_positions_plan",
    }
)
ACCOUNT_ID_RE = re.compile(r"\bPA[A-Z0-9]{6,}\b", re.IGNORECASE)

QTY_Q = Decimal("0.000000000001")
MONEY_Q = Decimal("0.000001")
CASH_Q = Decimal("0.01")
# Session-close / cutover identity: |cash + market_value - equity| <= one cent.
# Never rewrite cash, market_value, or equity to force the identity.
IDENTITY_TOLERANCE = Decimal("0.01")
ACCOUNT_MODEL = "LONG_ONLY_PAPER"
SUPPORTED_ACCOUNT_STRUCTURES = frozenset({"long_only_paper", "LONG_ONLY_PAPER"})
FEE_COMPONENTS = frozenset({"REG", "TAF", "CAT", "COMMISSION", "OTHER"})
ADJUSTMENT_DIRECTIONS = frozenset({"credit", "debit"})
ADJUSTMENT_EVIDENCE_CATEGORIES = frozenset(
    {"broker_correction", "journal", "transfer_adjustment", "other_evidenced"}
)
UNAVAILABLE = "UNAVAILABLE"
GROSS_PROVISIONAL = "GROSS_PROVISIONAL"

_PROCESS_LOCK = threading.RLock()


class PerformanceAccountingError(Exception):
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


@dataclass(frozen=True)
class AccountingPaths:
    events_path: Path
    lock_path: Path


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def accounting_paths(
    root: Optional[Path] = None,
    *,
    events_path: Optional[Path] = None,
    lock_path: Optional[Path] = None,
) -> AccountingPaths:
    base = Path(root) if root is not None else default_repo_root()
    return AccountingPaths(
        events_path=Path(events_path) if events_path is not None else base / EVENTS_RELATIVE,
        lock_path=Path(lock_path) if lock_path is not None else base / LOCK_RELATIVE,
    )


def D(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if value is None or value == "":
        raise PerformanceAccountingError("decimal_missing", {"value": value})
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise PerformanceAccountingError("decimal_invalid", {"value": value}) from exc


def q_money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_Q, rounding=ROUND_HALF_EVEN)


def q_cash(value: Decimal) -> Decimal:
    return value.quantize(CASH_Q, rounding=ROUND_HALF_EVEN)


def q_qty(value: Decimal) -> Decimal:
    return value.quantize(QTY_Q, rounding=ROUND_HALF_EVEN)


def dec_or_none(value: Any) -> Optional[Decimal]:
    if value is None or value == "" or value == UNAVAILABLE:
        return None
    return D(value)


def identity_residual(cash: Any, market_value: Any, equity: Any) -> Decimal:
    return abs(q_cash(D(cash)) + q_cash(D(market_value)) - q_cash(D(equity)))


def _assert_identity(cash: Any, market_value: Any, equity: Any, where: str) -> None:
    if (
        dec_or_none(cash) is None
        or dec_or_none(market_value) is None
        or dec_or_none(equity) is None
    ):
        raise PerformanceAccountingError("identity_fields_missing", {"where": where})
    residual = identity_residual(cash, market_value, equity)
    if residual > IDENTITY_TOLERANCE:
        raise PerformanceAccountingError(
            "cash_mv_equity_mismatch",
            {
                "where": where,
                "cash": str(cash),
                "market_value": str(market_value),
                "equity": str(equity),
                "residual": format(residual, "f"),
                "tolerance": format(IDENTITY_TOLERANCE, "f"),
                "account_model": ACCOUNT_MODEL,
            },
        )


def _assert_long_only_paper(body: Mapping[str, Any], where: str) -> None:
    structure = str(body.get("account_structure") or "long_only_paper").strip()
    if structure not in SUPPORTED_ACCOUNT_STRUCTURES:
        raise PerformanceAccountingError(
            "unsupported_account_structure",
            {"where": where, "account_structure": structure, "supported": ACCOUNT_MODEL},
        )
    if dec_or_none(body.get("cash")) is not None and D(body["cash"]) < 0:
        raise PerformanceAccountingError(
            "unsupported_account_structure",
            {"where": where, "reason": "negative_cash_not_supported"},
        )
    short_mv = dec_or_none(body.get("short_market_value"))
    if short_mv is not None and short_mv != 0:
        raise PerformanceAccountingError(
            "unsupported_account_structure",
            {"where": where, "reason": "short_market_value_not_supported"},
        )
    if body.get("margin") is True or str(body.get("account_type") or "").lower() == "margin":
        raise PerformanceAccountingError(
            "unsupported_account_structure",
            {"where": where, "reason": "margin_not_supported"},
        )
    for row in body.get("positions") or []:
        qty = dec_or_none(row.get("qty"))
        if qty is not None and qty < 0:
            raise PerformanceAccountingError(
                "unsupported_short_position",
                {"where": where, "symbol": row.get("symbol"), "qty": str(row.get("qty"))},
            )


def _benchmark_price(body: Mapping[str, Any]) -> Tuple[str, str, Optional[Decimal]]:
    adj = dec_or_none(body.get("adj_close"))
    raw = dec_or_none(body.get("close"))
    declared = str(body.get("return_kind") or "").strip().upper()
    if adj is not None:
        if declared and declared not in {"TOTAL_RETURN", ""}:
            raise PerformanceAccountingError(
                "benchmark_return_kind_conflict", {"declared": declared}
            )
        return "TOTAL_RETURN", "adj_close", adj
    if raw is not None:
        if declared == "TOTAL_RETURN":
            raise PerformanceAccountingError("benchmark_total_return_requires_adj_close")
        return "PRICE_RETURN", "close", raw
    return "UNAVAILABLE", "missing", None


def parse_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    text = str(value or "").strip()
    if not text:
        raise PerformanceAccountingError("timestamp_missing")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def utc_iso(value: Any) -> str:
    return parse_utc(value).isoformat()


def session_date_et(value: Any) -> str:
    return parse_utc(value).astimezone(NY_TZ).date().isoformat()


def redact_account_ids(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    return ACCOUNT_ID_RE.sub("***REDACTED***", text)


def event_id_for(payload: Mapping[str, Any]) -> str:
    material = {
        "schema_version": payload.get("schema_version"),
        "event_type": payload.get("event_type"),
        "logical_key": payload.get("logical_key"),
        "event_timestamp": payload.get("event_timestamp"),
        "session_date": payload.get("session_date"),
        "body": payload.get("body"),
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def _dump(event: Mapping[str, Any]) -> str:
    return json.dumps(event, separators=(",", ":"), default=_json_default)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_planning(raw: Mapping[str, Any]) -> bool:
    parts = [
        str(raw.get("source") or ""),
        str(raw.get("status") or ""),
        str((raw.get("body") or {}).get("source") or ""),
        str((raw.get("body") or {}).get("record_class") or ""),
        str((raw.get("body") or {}).get("status") or ""),
    ]
    blob = " ".join(parts).lower()
    return any(marker in blob for marker in PLANNING_MARKERS)


def logical_key_for(event_type: str, body: Mapping[str, Any], timestamp: str) -> str:
    if event_type == EVENT_CUTOVER:
        return f"cutover:{body.get('as_of_utc')}"
    if event_type == EVENT_SESSION_CLOSE:
        return f"session_close:{body.get('session_date')}"
    if event_type == EVENT_POSITION_SNAPSHOT:
        return f"position_snap:{body.get('as_of_utc')}:{canonical_symbol(body.get('symbol'))}"
    if event_type in FILL_EVENTS:
        fill_id = str(body.get("broker_fill_id") or body.get("activity_id") or "").strip()
        order_id = str(body.get("broker_order_id") or "").strip()
        if order_id and fill_id:
            return f"fill:{order_id}:{fill_id}"
        if order_id:
            return f"fill:{order_id}:{timestamp}"
        raise PerformanceAccountingError("logical_key_unresolvable", {"event_type": event_type})
    if event_type == EVENT_FEE:
        activity_id = str(body.get("activity_id") or "").strip()
        if not activity_id:
            raise PerformanceAccountingError("fee_activity_id_required")
        return f"fee:{activity_id}"
    if event_type in EXTERNAL_FLOW_EVENTS or event_type == EVENT_DIVIDEND:
        activity_id = str(body.get("activity_id") or "").strip()
        if activity_id:
            return f"{event_type.lower()}:{activity_id}"
        return f"{event_type.lower()}:{timestamp}:{body.get('amount')}:{body.get('symbol') or ''}"
    if event_type == EVENT_PRICE_MARKS:
        order_id = str(body.get("broker_order_id") or "").strip()
        if not order_id:
            raise PerformanceAccountingError("logical_key_unresolvable", {"event_type": event_type})
        return f"marks:{order_id}"
    if event_type == EVENT_BENCHMARK_CLOSE:
        return f"benchmark:{canonical_symbol(body.get('symbol'))}:{body.get('session_date')}"
    if event_type in NO_BASIS_EVENTS:
        return (
            f"{event_type.lower()}:{body.get('reason_code')}:{timestamp}:{body.get('symbol') or ''}"
        )
    raise PerformanceAccountingError("unknown_event_type", {"event_type": event_type})


def canonical_symbol(raw: Any) -> str:
    return str(raw or "").strip().upper()


def build_event(
    event_type: str,
    *,
    event_timestamp: Any,
    body: Mapping[str, Any],
    source: str,
    session_date: Optional[str] = None,
) -> Dict[str, Any]:
    if _is_planning({"source": source, "body": body}):
        raise PerformanceAccountingError(
            "planning_record_rejected",
            {"source": source, "event_type": event_type},
        )
    ts = utc_iso(event_timestamp)
    sess = session_date or session_date_et(ts)
    cleaned = _sanitize_body(dict(body))
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_type": event_type,
        "event_timestamp": ts,
        "session_date": sess,
        "timezone": TIMEZONE_NAME,
        "source": source,
        "cost_basis_method": COST_BASIS_METHOD,
        "body": cleaned,
    }
    event["logical_key"] = logical_key_for(event_type, cleaned, ts)
    event["event_id"] = event_id_for(event)
    validate_event(event)
    return event


def signed_external_amount(
    event_type: str,
    amount: Any,
    *,
    direction: Optional[str] = None,
) -> Decimal:
    """Public builder input is a positive absolute amount; storage is signed."""
    absolute = D(amount)
    if absolute <= 0:
        raise PerformanceAccountingError("flow_amount_not_positive", {"amount": str(amount)})
    if event_type == EVENT_CASH_DEPOSIT:
        return q_cash(absolute)
    if event_type == EVENT_CASH_WITHDRAWAL:
        return q_cash(-absolute)
    if event_type == EVENT_CASH_ADJUSTMENT:
        if direction not in ADJUSTMENT_DIRECTIONS:
            raise PerformanceAccountingError("adjustment_direction_required")
        return q_cash(absolute if direction == "credit" else -absolute)
    raise PerformanceAccountingError("not_an_external_flow", {"event_type": event_type})


def build_external_flow(
    event_type: str,
    *,
    amount: Any,
    activity_id: str,
    event_timestamp: Any,
    source: str,
    direction: Optional[str] = None,
    evidence_category: Optional[str] = None,
    session_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Accept a positive absolute amount. Deposit +, withdrawal -, adjustment signed by direction."""
    if event_type not in EXTERNAL_FLOW_EVENTS:
        raise PerformanceAccountingError("not_an_external_flow", {"event_type": event_type})
    signed_external_amount(event_type, amount, direction=direction)
    body: Dict[str, Any] = {
        "amount": format(D(amount), "f"),
        "activity_id": activity_id,
    }
    if event_type == EVENT_CASH_ADJUSTMENT:
        body["direction"] = direction
        body["evidence_category"] = evidence_category
    return build_event(
        event_type,
        event_timestamp=event_timestamp,
        source=source,
        session_date=session_date,
        body=body,
    )


def _sanitize_body(body: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in body.items():
        if key in {"account_id", "account_number"}:
            continue
        if isinstance(value, str):
            out[key] = redact_account_ids(value)
        elif isinstance(value, dict):
            out[key] = _sanitize_body(value)
        elif isinstance(value, list):
            out[key] = [
                (
                    _sanitize_body(item)
                    if isinstance(item, dict)
                    else redact_account_ids(item) if isinstance(item, str) else item
                )
                for item in value
            ]
        elif isinstance(value, Decimal):
            out[key] = format(value, "f")
        else:
            out[key] = value
    return out


def validate_event(event: Mapping[str, Any]) -> None:
    if event.get("schema_version") != SCHEMA_VERSION:
        raise PerformanceAccountingError("schema_mismatch", {"event_id": event.get("event_id")})
    if event.get("cost_basis_method") != COST_BASIS_METHOD:
        raise PerformanceAccountingError("cost_basis_method_mismatch")
    if event.get("timezone") != TIMEZONE_NAME:
        raise PerformanceAccountingError("timezone_mismatch")
    if _is_planning(event):
        raise PerformanceAccountingError("planning_record_rejected")
    event_type = str(event.get("event_type") or "")
    if not event_type:
        raise PerformanceAccountingError("event_type_missing")
    if not event.get("event_id") or not event.get("logical_key"):
        raise PerformanceAccountingError("identity_missing")
    if session_date_et(event["event_timestamp"]) != str(event.get("session_date")):
        # Allow explicit session_date override only for benchmark/session close
        # of a calendar session that differs from the write timestamp's ET date.
        if event_type not in {EVENT_SESSION_CLOSE, EVENT_BENCHMARK_CLOSE, EVENT_CUTOVER}:
            raise PerformanceAccountingError(
                "session_date_mismatch",
                {
                    "event_type": event_type,
                    "session_date": event.get("session_date"),
                    "derived": session_date_et(event["event_timestamp"]),
                },
            )
    body = event.get("body") or {}
    if event_type == EVENT_CUTOVER:
        _assert_identity(body.get("cash"), body.get("market_value"), body.get("equity"), "cutover")
        _assert_long_only_paper(body, "cutover")
        for flag in ("not_a_deposit", "not_a_withdrawal", "not_profit_or_loss", "not_a_return"):
            if body.get(flag) is not True:
                raise PerformanceAccountingError("cutover_flag_missing", {"flag": flag})
    if event_type == EVENT_SESSION_CLOSE:
        if str(body.get("close_kind") or "") not in CERTIFIED_CLOSE_KINDS:
            raise PerformanceAccountingError("session_close_kind_invalid", {"body": body})
        for required in ("cash", "market_value", "equity", "session_date"):
            if body.get(required) in (None, ""):
                raise PerformanceAccountingError("session_close_field_missing", {"field": required})
        _assert_identity(
            body.get("cash"), body.get("market_value"), body.get("equity"), "session_close"
        )
        _assert_long_only_paper(body, "session_close")
    if event_type == EVENT_CASH_DEPOSIT and D(body.get("amount")) <= 0:
        raise PerformanceAccountingError("deposit_not_positive")
    if event_type == EVENT_CASH_WITHDRAWAL and D(body.get("amount")) <= 0:
        raise PerformanceAccountingError("withdrawal_not_positive")
    if event_type == EVENT_CASH_ADJUSTMENT:
        if D(body.get("amount")) <= 0:
            raise PerformanceAccountingError("adjustment_not_positive")
        if str(body.get("direction") or "") not in ADJUSTMENT_DIRECTIONS:
            raise PerformanceAccountingError("adjustment_direction_required")
        if str(body.get("evidence_category") or "") not in ADJUSTMENT_EVIDENCE_CATEGORIES:
            raise PerformanceAccountingError("adjustment_evidence_category_required")
    if event_type == EVENT_DIVIDEND and D(body.get("amount")) <= 0:
        raise PerformanceAccountingError("dividend_not_positive")
    if event_type == EVENT_FEE:
        if not str(body.get("activity_id") or "").strip():
            raise PerformanceAccountingError("fee_activity_id_required")
        if D(body.get("amount")) <= 0:
            raise PerformanceAccountingError("fee_not_positive")
        component = str(body.get("fee_component") or "OTHER").strip().upper()
        if component not in FEE_COMPONENTS:
            raise PerformanceAccountingError("fee_component_invalid", {"fee_component": component})
        body["fee_component"] = component
    if event_type in FILL_EVENTS:
        if not body.get("broker_order_id"):
            raise PerformanceAccountingError("fill_missing_broker_order_id")
        if dec_or_none(body.get("qty")) is None or dec_or_none(body.get("price")) is None:
            raise PerformanceAccountingError("fill_missing_qty_or_price")
        if D(body["qty"]) <= 0 or D(body["price"]) <= 0:
            raise PerformanceAccountingError("fill_qty_or_price_not_positive")
    if event_type == EVENT_BENCHMARK_CLOSE:
        if canonical_symbol(body.get("symbol")) not in {"SPY", "QQQ"}:
            raise PerformanceAccountingError("benchmark_symbol_invalid")
        kind, field, price = _benchmark_price(body)
        if price is None:
            raise PerformanceAccountingError("benchmark_close_missing")
        body["return_kind"] = kind
        body["price_field"] = field
        body["price"] = format(q_money(price), "f")


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _PROCESS_LOCK:
        handle = None
        last_err: Optional[BaseException] = None
        for _ in range(80):
            try:
                handle = open(path, "a+b")
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    if handle.read(1) == b"":
                        handle.write(b"0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                last_err = None
                break
            except OSError as exc:
                last_err = exc
                if handle is not None:
                    handle.close()
                    handle = None
                time.sleep(0.025)
        if handle is None:
            raise PerformanceAccountingError(
                "accounting_lock_unavailable",
                {"error": type(last_err).__name__ if last_err else "unknown"},
            )
        try:
            yield handle
        finally:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def load_events(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise PerformanceAccountingError("ledger_bom_present", {"path": str(path)})
    text = raw.decode("utf-8")
    if text and not text.endswith("\n"):
        raise PerformanceAccountingError("ledger_incomplete_final_line", {"path": str(path)})
    events: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for idx, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise PerformanceAccountingError("ledger_blank_line", {"line": idx})
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PerformanceAccountingError(
                "ledger_malformed_line", {"line": idx, "error": str(exc)}
            ) from exc
        if not isinstance(rec, dict):
            raise PerformanceAccountingError("ledger_non_object", {"line": idx})
        validate_event(rec)
        eid = str(rec["event_id"])
        if eid in seen:
            raise PerformanceAccountingError("ledger_duplicate_event_id", {"event_id": eid})
        seen.add(eid)
        events.append(rec)
    return events


def append_events(
    events: Sequence[Mapping[str, Any]],
    *,
    root: Optional[Path] = None,
    events_path: Optional[Path] = None,
    lock_path: Optional[Path] = None,
) -> Dict[str, Any]:
    paths = accounting_paths(root, events_path=events_path, lock_path=lock_path)
    appended = 0
    skipped = 0
    with _exclusive_lock(paths.lock_path):
        existing = load_events(paths.events_path) if paths.events_path.is_file() else []
        by_id = {str(e["event_id"]): e for e in existing}
        by_key = {str(e["logical_key"]): e for e in existing}
        to_write: List[Dict[str, Any]] = []
        for raw in events:
            rec = dict(raw)
            validate_event(rec)
            eid = str(rec["event_id"])
            key = str(rec["logical_key"])
            if eid in by_id:
                if _dump(by_id[eid]) != _dump(rec) and by_id[eid].get("logical_key") == key:
                    # identical id already stored — skip
                    skipped += 1
                    continue
                skipped += 1
                continue
            if key in by_key and _dump(by_key[key]) != _dump(rec):
                raise PerformanceAccountingError(
                    "duplicate_logical_key_conflict",
                    {"logical_key": key, "event_id": eid},
                )
            if key in by_key:
                skipped += 1
                continue
            by_id[eid] = rec
            by_key[key] = rec
            to_write.append(rec)
        if to_write:
            paths.events_path.parent.mkdir(parents=True, exist_ok=True)
            payload = "".join(_dump(item) + "\n" for item in to_write)
            with paths.events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            appended = len(to_write)
    return {
        "ok": True,
        "appended": appended,
        "skipped_duplicate": skipped,
        "path": str(paths.events_path),
    }


def encode_ledger_suffix(events: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(_dump(dict(item)) + "\n" for item in events).encode("utf-8")


def plan_ledger_writes(
    events: Sequence[Mapping[str, Any]],
    existing: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    by_id = {str(item["event_id"]): item for item in existing}
    by_key = {str(item["logical_key"]): item for item in existing}
    to_write: List[Dict[str, Any]] = []
    skipped = 0
    for raw in events:
        rec = dict(raw)
        validate_event(rec)
        eid = str(rec["event_id"])
        key = str(rec["logical_key"])
        if eid in by_id or key in by_key:
            if key in by_key and eid not in by_id and _dump(by_key[key]) != _dump(rec):
                raise PerformanceAccountingError(
                    "duplicate_logical_key_conflict",
                    {"logical_key": key, "event_id": eid},
                )
            skipped += 1
            continue
        by_id[eid] = rec
        by_key[key] = rec
        to_write.append(rec)
    return to_write, skipped


def _is_known_accounting_ledger(events: Sequence[Mapping[str, Any]]) -> bool:
    if not events:
        return False
    if not any(event.get("event_type") == EVENT_CUTOVER for event in events):
        return False
    return all(event.get("schema_version") == SCHEMA_VERSION for event in events)


def publish_ledger_atomic(
    events: Sequence[Mapping[str, Any]],
    *,
    events_path: Path,
    lock_path: Path,
    mode: str,
    fail_hook: Optional[Callable[[str, Path, bytes], None]] = None,
    publication_authority: Any = None,
    production_root: Optional[Path] = None,
    expected_git_head: Optional[str] = None,
    now: Optional[datetime] = None,
    publication_command: Optional[str] = None,
    confirm: str = "",
) -> Dict[str, Any]:
    """Create-only or append-only durable publication.

    Writes a sibling temporary file, flush+fsync, verifies the payload hash,
    then atomically replaces the destination. A failure before rename leaves
    an existing ledger unchanged and never creates a destination on first
    publication. Unknown existing ledgers are refused — never overwritten,
    truncated, repaired, or merged.

    Destinations under production require a collector-minted in-memory
    publication authority. TEMP destinations keep working without one.
    Authority bindings are verified before lock, directory, temp-file, or
    writable opens.
    """
    if mode not in {"create", "append"}:
        raise PerformanceAccountingError("publication_mode_invalid", {"mode": mode})
    dest = Path(events_path)
    lock = Path(lock_path)
    command = publication_command or (
        COMMAND_CUTOVER if mode == "create" else COMMAND_SESSION_CLOSE
    )
    protected = destination_requires_authority(dest, lock, production_root)
    if protected or publication_authority is not None:
        if publication_authority is None or production_root is None:
            raise PerformanceAccountingError(
                "missing_publication_authority",
                {"reason": "production_root_or_authority_missing"},
            )
        try:
            preflight_publication_authority(
                publication_authority,
                events_path=dest,
                lock_path=lock,
                production_root=production_root,
                expected_git_head=str(expected_git_head or ""),
                publication_mode=mode,
                command=command,
                confirm_phrase=str(confirm or ""),
                now=now,
            )
        except PublicationAuthorityError as exc:
            raise PerformanceAccountingError(exc.reason, exc.details) from exc
        prior = dest.read_bytes() if dest.is_file() else b""
        if prior:
            try:
                existing_preview = load_events(dest)
            except PerformanceAccountingError:
                existing_preview = []
        else:
            existing_preview = []
        planned_records, _skipped = plan_ledger_writes(events, existing_preview)
        planned_payload = prior + encode_ledger_suffix(planned_records)
        try:
            require_publication_authority(
                publication_authority,
                events_path=dest,
                lock_path=lock,
                production_root=production_root,
                expected_git_head=str(expected_git_head or ""),
                publication_mode=mode,
                command=command,
                confirm_phrase=str(confirm or ""),
                planned_payload=planned_payload,
                prior=prior,
                now=now,
            )
        except PublicationAuthorityError as exc:
            raise PerformanceAccountingError(exc.reason, exc.details) from exc
    tmp = dest.with_name(dest.name + ".tmp")
    with _exclusive_lock(lock):
        existed = dest.is_file()
        if mode == "create":
            if existed:
                raise PerformanceAccountingError("existing_ledger_refused", {"path": str(dest)})
            existing: List[Dict[str, Any]] = []
            prior = b""
        else:
            if not existed:
                raise PerformanceAccountingError("ledger_missing_for_append", {"path": str(dest)})
            try:
                existing = load_events(dest)
            except PerformanceAccountingError as exc:
                raise PerformanceAccountingError(
                    "unknown_existing_ledger",
                    {"cause": exc.reason, "path": str(dest)},
                ) from exc
            if not _is_known_accounting_ledger(existing):
                raise PerformanceAccountingError("unknown_existing_ledger", {"path": str(dest)})
            prior = dest.read_bytes()
        to_write, skipped = plan_ledger_writes(events, existing)
        if mode == "create":
            if len(to_write) != 1 or to_write[0].get("event_type") != EVENT_CUTOVER:
                raise PerformanceAccountingError(
                    "cutover_batch_invalid",
                    {"count": len(to_write)},
                )
        if not to_write:
            return {
                "ok": True,
                "appended": 0,
                "skipped_duplicate": skipped,
                "writes_performed": False,
                "path": str(dest),
            }
        payload = prior + encode_ledger_suffix(to_write)
        expected_hash = sha256_bytes(payload)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tmp.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if fail_hook is not None:
                fail_hook("after_fsync", tmp, payload)
            actual_hash = sha256_file(tmp)
            if actual_hash != expected_hash:
                raise PerformanceAccountingError(
                    "publication_hash_mismatch",
                    {"expected": expected_hash, "actual": actual_hash},
                )
            if fail_hook is not None:
                fail_hook("before_rename", tmp, payload)
            os.replace(tmp, dest)
        except Exception:
            if tmp.is_file():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise
    return {
        "ok": True,
        "appended": len(to_write),
        "skipped_duplicate": skipped,
        "writes_performed": True,
        "payload_hash": expected_hash,
        "path": str(dest),
    }


def _pos(qty: Decimal, basis: Decimal) -> Dict[str, Any]:
    qty = q_qty(qty)
    basis = q_money(basis)
    avg = q_money(basis / qty) if qty > 0 else Decimal("0")
    return {
        "qty": qty,
        "remaining_basis": basis,
        "average_entry": avg,
        "basis_proven": True,
    }


def _match_sale_for_fee(
    sales: Sequence[Dict[str, Any]], body: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    target_event = str(body.get("allocates_to_event_id") or "").strip()
    if target_event:
        return next((s for s in sales if s["event_id"] == target_event), None)
    target_order = str(
        body.get("allocates_to_order_id") or body.get("broker_order_id") or ""
    ).strip()
    if target_order:
        return next((s for s in sales if s.get("broker_order_id") == target_order), None)
    return None


def reduce_events(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(
        events,
        key=lambda e: (str(e.get("event_timestamp") or ""), str(e.get("event_id") or "")),
    )
    cash = Decimal("0")
    cash_proven = False
    positions: Dict[str, Dict[str, Any]] = {}
    closes: List[Dict[str, Any]] = []
    sales: List[Dict[str, Any]] = []
    pending_fees: List[Dict[str, Any]] = []
    external_flows: List[Dict[str, Any]] = []
    investment_income: List[Dict[str, Any]] = []
    marks: Dict[str, Dict[str, Any]] = {}
    benchmarks: Dict[str, Dict[str, Dict[str, str]]] = {}
    cutover: Optional[Dict[str, Any]] = None
    ignored = 0
    applied_fee_ids: set[str] = set()

    for ev in ordered:
        validate_event(ev)
        et = str(ev["event_type"])
        body = ev.get("body") or {}
        if et in NO_BASIS_EVENTS:
            ignored += 1
            continue
        if et == EVENT_CUTOVER:
            if cutover is not None:
                raise PerformanceAccountingError("duplicate_cutover")
            cash = q_cash(D(body["cash"]))
            cash_proven = True
            cutover = {
                "as_of_utc": body.get("as_of_utc"),
                "equity": format(q_cash(D(body["equity"])), "f"),
                "cash": format(cash, "f"),
                "market_value": format(
                    q_cash(D(body.get("market_value") or D(body["equity"]) - cash)), "f"
                ),
                "position_count": len(body.get("positions") or []),
                "not_a_return": True,
                "not_a_deposit": True,
                "not_a_withdrawal": True,
                "not_profit_or_loss": True,
                "opening_basis_only": True,
                "publication_status": body.get("publication_status") or "PROPOSED",
            }
            for row in body.get("positions") or []:
                symbol = canonical_symbol(row.get("symbol"))
                positions[symbol] = _pos(D(row["qty"]), D(row["cost_basis"]))
            continue
        if et == EVENT_FILL_BUY:
            symbol = canonical_symbol(body["symbol"])
            qty = D(body["qty"])
            price = D(body["price"])
            add = q_money(qty * price)
            cur = positions.get(symbol) or _pos(Decimal("0"), Decimal("0"))
            positions[symbol] = _pos(cur["qty"] + qty, cur["remaining_basis"] + add)
            continue
        if et == EVENT_FILL_SELL:
            symbol = canonical_symbol(body["symbol"])
            qty = D(body["qty"])
            price = D(body["price"])
            proceeds = q_money(qty * price)
            cur = positions.get(symbol)
            sale = {
                "event_id": ev["event_id"],
                "logical_key": ev["logical_key"],
                "symbol": symbol,
                "session_date": ev["session_date"],
                "broker_order_id": str(body.get("broker_order_id") or ""),
                "qty": format(q_qty(qty), "f"),
                "price": format(q_money(price), "f"),
                "proceeds": format(proceeds, "f"),
                "basis_consumed": UNAVAILABLE,
                "fees_allocated": UNAVAILABLE,
                "fee_event_ids": [],
                "fee_components": [],
                "gross_realized_pl": UNAVAILABLE,
                "gross_realized_pl_label": UNAVAILABLE,
                "realized_pl": UNAVAILABLE,
                "realized_pl_kind": UNAVAILABLE,
                "fees_status": "PENDING",
                "fee_reconciliation_complete": False,
                "applied": False,
            }
            if cur is None or not cur.get("basis_proven") or cur["qty"] <= 0:
                sales.append(sale)
                continue
            if qty - cur["qty"] > QTY_Q:
                raise PerformanceAccountingError(
                    "sell_exceeds_quantity",
                    {"symbol": symbol, "qty": str(qty), "available": str(cur["qty"])},
                )
            consumed = q_money(qty * cur["average_entry"])
            remaining_qty = q_qty(cur["qty"] - qty)
            if remaining_qty <= 0:
                positions[symbol] = _pos(Decimal("0"), Decimal("0"))
            else:
                positions[symbol] = _pos(remaining_qty, cur["remaining_basis"] - consumed)
            sale.update(
                {
                    "basis_consumed": format(consumed, "f"),
                    "gross_realized_pl": format(q_money(proceeds - consumed), "f"),
                    "gross_realized_pl_label": GROSS_PROVISIONAL,
                    "applied": True,
                }
            )
            sales.append(sale)
            continue
        if et in EXTERNAL_FLOW_EVENTS:
            signed = signed_external_amount(
                et, body["amount"], direction=str(body.get("direction") or "") or None
            )
            if cash_proven:
                cash = q_cash(cash + signed)
            external_flows.append(
                {
                    "event_id": ev["event_id"],
                    "event_type": et,
                    "session_date": ev["session_date"],
                    "event_timestamp": ev["event_timestamp"],
                    "amount": format(signed, "f"),
                    "activity_id": body.get("activity_id"),
                    "direction": body.get("direction"),
                    "evidence_category": body.get("evidence_category"),
                }
            )
            continue
        if et == EVENT_DIVIDEND:
            amount = q_cash(D(body["amount"]))
            if cash_proven:
                cash = q_cash(cash + amount)
            investment_income.append(
                {
                    "event_id": ev["event_id"],
                    "session_date": ev["session_date"],
                    "amount": format(amount, "f"),
                    "class": "investment_return_not_external_flow",
                }
            )
            continue
        if et == EVENT_FEE:
            if ev["event_id"] in applied_fee_ids:
                continue
            amount = q_cash(abs(D(body["amount"])))
            component = str(body.get("fee_component") or "OTHER").strip().upper()
            matched = _match_sale_for_fee(sales, body)
            if matched and matched.get("basis_consumed") != UNAVAILABLE:
                if ev["event_id"] in matched["fee_event_ids"]:
                    applied_fee_ids.add(ev["event_id"])
                    continue
                prior = (
                    D(matched["fees_allocated"])
                    if matched.get("fees_allocated") not in (None, UNAVAILABLE)
                    else Decimal("0")
                )
                total_fees = q_cash(prior + amount)
                matched["fees_allocated"] = format(total_fees, "f")
                matched["fee_event_ids"].append(ev["event_id"])
                matched["fee_components"].append(
                    {
                        "event_id": ev["event_id"],
                        "activity_id": body.get("activity_id"),
                        "fee_component": component,
                        "amount": format(amount, "f"),
                    }
                )
                matched["fees_status"] = "ATTRIBUTED"
                matched["fee_reconciliation_complete"] = True
                matched["realized_pl"] = format(
                    q_money(D(matched["proceeds"]) - D(matched["basis_consumed"]) - total_fees),
                    "f",
                )
                matched["realized_pl_kind"] = "NET"
                matched["gross_realized_pl_label"] = "GROSS"
                if cash_proven:
                    cash = q_cash(cash - amount)
                applied_fee_ids.add(ev["event_id"])
            else:
                pending_fees.append(
                    {
                        "event_id": ev["event_id"],
                        "amount": format(amount, "f"),
                        "activity_id": body.get("activity_id"),
                        "fee_component": component,
                        "description": body.get("description"),
                        "session_date": ev.get("session_date"),
                    }
                )
                if cash_proven:
                    cash = q_cash(cash - amount)
                applied_fee_ids.add(ev["event_id"])
            continue
        if et == EVENT_PRICE_MARKS:
            marks[str(body.get("broker_order_id"))] = {
                "decision_price": body.get("decision_price"),
                "submit_mark": body.get("submit_mark"),
                "bid": body.get("bid"),
                "ask": body.get("ask"),
            }
            continue
        if et == EVENT_BENCHMARK_CLOSE:
            sess = str(body.get("session_date") or ev["session_date"])
            kind, field, price = _benchmark_price(body)
            if price is None:
                raise PerformanceAccountingError("benchmark_close_missing")
            benchmarks.setdefault(sess, {})
            benchmarks[sess][canonical_symbol(body.get("symbol"))] = {
                "price": format(q_money(price), "f"),
                "return_kind": kind,
                "price_field": field,
            }
            continue
        if et == EVENT_POSITION_SNAPSHOT:
            continue
        if et == EVENT_SESSION_CLOSE:
            if any(item["session_date"] == ev["session_date"] for item in closes):
                raise PerformanceAccountingError(
                    "duplicate_session_close", {"session_date": ev["session_date"]}
                )
            close_positions = []
            invested = Decimal("0")
            for row in body.get("positions") or []:
                mv = D(row.get("market_value") or 0)
                invested += mv
                close_positions.append(
                    {
                        "symbol": canonical_symbol(row.get("symbol")),
                        "qty": format(D(row["qty"]), "f"),
                        "market_value": format(q_money(mv), "f"),
                        "average_entry_price": format(D(row["average_entry_price"]), "f"),
                        "unrealized_pl": (
                            format(D(row["unrealized_pl"]), "f")
                            if row.get("unrealized_pl") not in (None, "")
                            else UNAVAILABLE
                        ),
                    }
                )
            equity = q_cash(D(body["equity"]))
            market_value = q_cash(D(body["market_value"]))
            close_cash = q_cash(D(body["cash"]))
            residual = identity_residual(close_cash, market_value, equity)
            cash = close_cash
            cash_proven = True
            prior = closes[-1] if closes else None
            flow_sum = Decimal("0")
            if prior is not None:
                prior_ts = prior["evidence_timestamp"]
                for flow in external_flows:
                    if prior_ts < flow["event_timestamp"] <= ev["event_timestamp"]:
                        flow_sum += D(flow["amount"])
            flow_since_prior = q_cash(flow_sum) if prior is not None else Decimal("0")
            gross = q_money(market_value / equity) if equity > 0 else UNAVAILABLE
            net = gross
            closes.append(
                {
                    "event_id": ev["event_id"],
                    "session_date": ev["session_date"],
                    "close_kind": body.get("close_kind"),
                    "cash": format(close_cash, "f"),
                    "market_value": format(market_value, "f"),
                    "equity": format(equity, "f"),
                    "buying_power": body.get("buying_power"),
                    "positions": close_positions,
                    "external_cash_flow_since_prior": format(flow_since_prior, "f"),
                    "gross_exposure": format(gross, "f") if gross != UNAVAILABLE else UNAVAILABLE,
                    "net_exposure": format(net, "f") if net != UNAVAILABLE else UNAVAILABLE,
                    "evidence_timestamp": ev["event_timestamp"],
                    "source_hashes": body.get("source_hashes") or {},
                    "identity_residual": format(residual, "f"),
                    "identity_tolerance": format(IDENTITY_TOLERANCE, "f"),
                    "account_model": ACCOUNT_MODEL,
                }
            )
            continue
        raise PerformanceAccountingError("unhandled_event_type", {"event_type": et})

    metrics = calculate_metrics(
        cutover=cutover,
        closes=closes,
        sales=sales,
        fills=[e for e in ordered if e["event_type"] in FILL_EVENTS],
        marks=marks,
        benchmarks=benchmarks,
    )
    open_positions = {
        symbol: {
            "qty": format(row["qty"], "f"),
            "remaining_basis": format(row["remaining_basis"], "f"),
            "average_entry": format(row["average_entry"], "f"),
        }
        for symbol, row in positions.items()
        if row["qty"] > 0
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "cost_basis_method": COST_BASIS_METHOD,
        "timezone": TIMEZONE_NAME,
        "cutover": cutover,
        "cash": format(cash, "f") if cash_proven else UNAVAILABLE,
        "open_positions": open_positions,
        "open_position_count": len(open_positions),
        "session_closes": closes,
        "sales": sales,
        "pending_fees": pending_fees,
        "external_flows": external_flows,
        "investment_income": investment_income,
        "marks": marks,
        "benchmarks": benchmarks,
        "ignored_non_basis_events": ignored,
        "metrics": metrics,
    }


def _side_signed_slip(side: str, fill: Decimal, mark: Optional[Decimal], qty: Decimal) -> Any:
    if mark is None:
        return UNAVAILABLE
    signed = (fill - mark) if side == "BUY" else (mark - fill)
    return format(q_money(signed * qty), "f")


def calculate_metrics(
    *,
    cutover: Optional[Mapping[str, Any]],
    closes: Sequence[Mapping[str, Any]],
    sales: Sequence[Mapping[str, Any]],
    fills: Sequence[Mapping[str, Any]],
    marks: Mapping[str, Mapping[str, Any]],
    benchmarks: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    earliest = closes[0]["session_date"] if closes else None
    periods: List[Dict[str, Any]] = []
    if len(closes) >= 2:
        wealth = Decimal("1")
        peak = Decimal("1")
        max_dd = Decimal("0")
        exposures: List[Decimal] = []
        for prev, cur in zip(closes, closes[1:]):
            e0 = D(prev["equity"])
            e1 = D(cur["equity"])
            flow = D(cur["external_cash_flow_since_prior"])
            if e0 <= 0:
                r = UNAVAILABLE
            else:
                r = q_money((e1 - e0 - flow) / e0)
                wealth *= 1 + D(r)
                if wealth > peak:
                    peak = wealth
                dd = (wealth / peak) - 1
                if dd < max_dd:
                    max_dd = dd
            if cur.get("gross_exposure") not in (None, UNAVAILABLE):
                exposures.append(D(cur["gross_exposure"]))
            spy, spy_kind = _same_date_benchmark(prev, cur, benchmarks, "SPY")
            qqq, qqq_kind = _same_date_benchmark(prev, cur, benchmarks, "QQQ")
            periods.append(
                {
                    "from_session": prev["session_date"],
                    "to_session": cur["session_date"],
                    "period_return": format(r, "f") if r != UNAVAILABLE else UNAVAILABLE,
                    "spy_return": spy,
                    "spy_return_kind": spy_kind,
                    "qqq_return": qqq,
                    "qqq_return_kind": qqq_kind,
                }
            )
        twr = (
            format(q_money(wealth - 1), "f")
            if periods and all(p["period_return"] != UNAVAILABLE for p in periods)
            else UNAVAILABLE
        )
        avg_exp = (
            format(q_money(sum(exposures) / Decimal(len(exposures))), "f")
            if exposures
            else UNAVAILABLE
        )
        exp_adj = (
            format(q_money(D(twr) / D(avg_exp)), "f")
            if (
                twr != UNAVAILABLE
                and avg_exp != UNAVAILABLE
                and D(avg_exp) > 0
                and len(exposures) >= 2
                and all(item > 0 for item in exposures)
            )
            else UNAVAILABLE
        )
        mdd = format(q_money(max_dd), "f") if twr != UNAVAILABLE else UNAVAILABLE
    else:
        twr = UNAVAILABLE
        mdd = UNAVAILABLE
        avg_exp = UNAVAILABLE
        exp_adj = UNAVAILABLE

    proven_sales = [s for s in sales if s.get("realized_pl") not in (None, UNAVAILABLE)]
    wins = [D(s["realized_pl"]) for s in proven_sales if D(s["realized_pl"]) > 0]
    losses = [D(s["realized_pl"]) for s in proven_sales if D(s["realized_pl"]) < 0]
    win_rate = (
        format(q_money(Decimal(len(wins)) / Decimal(len(proven_sales))), "f")
        if proven_sales
        else UNAVAILABLE
    )
    avg_win = format(q_money(sum(wins) / Decimal(len(wins))), "f") if wins else UNAVAILABLE
    avg_loss = format(q_money(sum(losses) / Decimal(len(losses))), "f") if losses else UNAVAILABLE
    profit_factor = (
        format(q_money(sum(wins) / abs(sum(losses))), "f") if wins and losses else UNAVAILABLE
    )

    slip_submit: List[str] = []
    slip_decision: List[str] = []
    for fill in fills:
        body = fill.get("body") or {}
        order_id = str(body.get("broker_order_id") or "")
        mark = marks.get(order_id) or {}
        side = "BUY" if fill["event_type"] == EVENT_FILL_BUY else "SELL"
        qty = D(body["qty"])
        price = D(body["price"])
        submit = _side_signed_slip(side, price, dec_or_none(mark.get("submit_mark")), qty)
        decision = _side_signed_slip(side, price, dec_or_none(mark.get("decision_price")), qty)
        if submit != UNAVAILABLE:
            slip_submit.append(submit)
        if decision != UNAVAILABLE:
            slip_decision.append(decision)

    return {
        "formulas": FORMULAS,
        "earliest_performance_date": earliest,
        "one_period_return_available": len(closes) >= 2,
        "time_weighted_return": twr if len(closes) >= 2 else UNAVAILABLE,
        "maximum_drawdown": mdd if len(closes) >= 2 else UNAVAILABLE,
        "average_exposure": avg_exp if closes else UNAVAILABLE,
        "exposure_adjusted_return": exp_adj if len(closes) >= 2 else UNAVAILABLE,
        "periods": periods,
        "win_rate": win_rate,
        "average_winner": avg_win,
        "average_loser": avg_loss,
        "profit_factor": profit_factor,
        "submit_to_fill_slippage_notional": (
            format(q_money(sum(D(x) for x in slip_submit)), "f") if slip_submit else UNAVAILABLE
        ),
        "decision_to_fill_slippage_notional": (
            format(q_money(sum(D(x) for x in slip_decision)), "f") if slip_decision else UNAVAILABLE
        ),
        "proven_realized_sale_count": len(proven_sales),
        "cutover_is_not_a_return": True,
        "historical_return": None,
    }


def _benchmark_point(raw: Any) -> Optional[Dict[str, str]]:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        price = raw.get("price")
        kind = str(raw.get("return_kind") or UNAVAILABLE)
        price_field = str(raw.get("price_field") or "")
        if price is None:
            return None
        return {"price": str(price), "return_kind": kind, "price_field": price_field}
    return {"price": str(raw), "return_kind": "PRICE_RETURN", "price_field": "close"}


def _same_date_benchmark(
    prev: Mapping[str, Any],
    cur: Mapping[str, Any],
    benchmarks: Mapping[str, Mapping[str, Any]],
    symbol: str,
) -> Tuple[Any, str]:
    a = _benchmark_point((benchmarks.get(str(prev["session_date"])) or {}).get(symbol))
    b = _benchmark_point((benchmarks.get(str(cur["session_date"])) or {}).get(symbol))
    if a is None or b is None:
        return UNAVAILABLE, UNAVAILABLE
    if a["return_kind"] != b["return_kind"] or a["return_kind"] not in {
        "TOTAL_RETURN",
        "PRICE_RETURN",
    }:
        return UNAVAILABLE, UNAVAILABLE
    if D(a["price"]) <= 0:
        return UNAVAILABLE, UNAVAILABLE
    return format(q_money((D(b["price"]) - D(a["price"])) / D(a["price"])), "f"), a["return_kind"]


FORMULAS = {
    "cost_basis_method": COST_BASIS_METHOD,
    "buy_average": "new_avg = (remaining_basis + qty*price) / (qty_old + qty)",
    "sell_basis_consumed": "qty * existing_average_entry",
    "realized_pl": "NET = proceeds - basis_consumed - sum(distinct attributed fees); UNAVAILABLE until required fee reconciliation is complete and basis is proven",
    "gross_realized_pl": "proceeds - basis_consumed; labeled GROSS_PROVISIONAL until fee reconciliation completes; never treated as net",
    "fees": "each broker activity_id is a distinct fee; REG/TAF/CAT/COMMISSION/OTHER sum on one sale; same event_id is idempotent; conflicting activity_id fails closed",
    "remaining_basis_after_partial": "prior_remaining_basis - basis_consumed",
    "full_close": "qty == 0 and remaining_basis == 0",
    "external_flow": "public builder takes a positive absolute amount; deposit stores +amount, withdrawal stores -amount; adjustment requires direction+evidence; TWR uses the signed F_i",
    "session_close_identity": "fail closed if |cash + market_value - equity| > 0.01 (one cent); never rewrite the three fields to force equality; long-only paper only",
    "one_period_return": "r_i = (E_i - E_{i-1} - F_i) / E_{i-1} after two certified session closes; F_i is the signed external-flow sum",
    "time_weighted_return": "Π(1 + r_i) - 1",
    "maximum_drawdown": "min(wealth_t / running_peak - 1) on the TWR wealth index",
    "spy_comparison": "(SPY_end/SPY_start - 1) on the exact same session dates; adj_close => TOTAL_RETURN, close-only => PRICE_RETURN; mixed kinds => UNAVAILABLE",
    "qqq_comparison": "same as SPY with QQQ; never silently mix TOTAL_RETURN and PRICE_RETURN",
    "average_exposure": "mean(market_value / equity) over certified closes",
    "exposure_adjusted_return": "twr / average_exposure; UNAVAILABLE if exposure is 0, missing, or fewer than two closes",
    "win_rate": "count(realized_pl > 0) / count(sales with proven realized_pl)",
    "average_winner": "mean(realized_pl | > 0)",
    "average_loser": "mean(realized_pl | < 0)",
    "profit_factor": "sum(wins) / abs(sum(losses))",
    "submit_slippage": "BUY:(fill-submit)*qty; SELL:(submit-fill)*qty; missing submit => UNAVAILABLE",
    "decision_slippage": "same using decision_price; missing decision => UNAVAILABLE, never 0",
}


def reduce_ledger(
    *,
    root: Optional[Path] = None,
    events_path: Optional[Path] = None,
    lock_path: Optional[Path] = None,
) -> Dict[str, Any]:
    paths = accounting_paths(root, events_path=events_path, lock_path=lock_path)
    return reduce_events(load_events(paths.events_path))


def proposed_cutover_positions() -> List[Dict[str, Any]]:
    return [
        {
            "symbol": "ARKK",
            "qty": "32",
            "average_entry_price": "72.082375",
            "cost_basis": "2306.636",
            "market_value": "2717.44",
            "unrealized_pl": "410.804",
        },
        {
            "symbol": "BITO",
            "qty": "1326",
            "average_entry_price": "8.729638",
            "cost_basis": "11575.499988",
            "market_value": "14044.0638",
            "unrealized_pl": "2468.563812",
        },
        {
            "symbol": "DBA",
            "qty": "66",
            "average_entry_price": "26.86314",
            "cost_basis": "1772.96724",
            "market_value": "1916.97",
            "unrealized_pl": "144.00276",
        },
        {
            "symbol": "DIA",
            "qty": "5",
            "average_entry_price": "464.903333",
            "cost_basis": "2324.516665",
            "market_value": "2623.9825",
            "unrealized_pl": "299.465835",
        },
        {
            "symbol": "GE",
            "qty": "1",
            "average_entry_price": "280.1925",
            "cost_basis": "280.1925",
            "market_value": "325.14",
            "unrealized_pl": "44.9475",
        },
        {
            "symbol": "GS",
            "qty": "1",
            "average_entry_price": "893.42",
            "cost_basis": "893.42",
            "market_value": "1032.235",
            "unrealized_pl": "138.815",
        },
        {
            "symbol": "MA",
            "qty": "7",
            "average_entry_price": "496.378571",
            "cost_basis": "3474.649997",
            "market_value": "3986.15",
            "unrealized_pl": "511.500003",
        },
        {
            "symbol": "TSLA",
            "qty": "0.137141758",
            "average_entry_price": "362.435",
            "cost_basis": "49.704973",
            "market_value": "50.550109",
            "unrealized_pl": "0.845136",
        },
    ]


def build_proposed_cutover_event(
    *,
    source_hashes: Optional[Mapping[str, str]] = None,
    buying_power: Optional[str] = "158021.63",
) -> Dict[str, Any]:
    positions = proposed_cutover_positions()
    market_value = q_cash(PROPOSED_CUTOVER_EQUITY - PROPOSED_CUTOVER_CASH)
    return build_event(
        EVENT_CUTOVER,
        event_timestamp=PROPOSED_CUTOVER_UTC,
        session_date="2026-09-09",
        source="phase_198t_bn1_frozen_checkpoint",
        body={
            "as_of_utc": PROPOSED_CUTOVER_UTC,
            "publication_status": "PROPOSED_REQUIRES_FRESH_SNAPSHOT",
            "cash": format(PROPOSED_CUTOVER_CASH, "f"),
            "equity": format(PROPOSED_CUTOVER_EQUITY, "f"),
            "market_value": format(market_value, "f"),
            "buying_power": buying_power,
            "positions": positions,
            "position_count": len(positions),
            "not_a_deposit": True,
            "not_a_withdrawal": True,
            "not_profit_or_loss": True,
            "not_a_return": True,
            "account_structure": "long_only_paper",
            "historical_return": None,
            "source_hashes": dict(source_hashes or {}),
        },
    )


INTEGRATION_MAP = {
    "do_not_change": [
        "services.decide_paper_entry_cycle_action",
        "services.paper_sell_arm_runtime_enforcement",
        "BUY/SELL quantity clipping and dispatch permits",
        "Start/Stop/Restart cooperative controls",
    ],
    "future_passive_capture": [
        {
            "hook": "after paper_order_ledger terminal FILLED reduce",
            "events": ["FILL_BUY", "FILL_SELL", "PRICE_MARKS"],
            "note": "Observe proven fills only. Planning rows stay rejected.",
        },
        {
            "hook": "after data-freshness session-close snapshot, idle + lock absent",
            "events": ["SESSION_CLOSE", "POSITION_SNAPSHOT"],
            "note": "Certified regular or early close only. Do not rewrite history files.",
        },
        {
            "hook": "after read-only GET /v2/account/activities",
            "events": ["FEE", "CASH_DEPOSIT", "CASH_WITHDRAWAL", "CASH_ADJUSTMENT", "DIVIDEND"],
            "note": "Redact account IDs. Attribute fees with a new event.",
        },
        {
            "hook": "after local SPY/QQQ daily bar write",
            "events": ["BENCHMARK_CLOSE"],
            "note": "Same America/New_York session date. Never fill missing dates.",
        },
    ],
    "runtime_integration": "not_in_this_phase",
}


def dry_run_proposed_cutover(
    *,
    events_path: Path,
    lock_path: Path,
    source_hashes: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    event = build_proposed_cutover_event(source_hashes=source_hashes)
    append_events([event], events_path=events_path, lock_path=lock_path)
    state = reduce_ledger(events_path=events_path, lock_path=lock_path)
    metrics = state["metrics"]
    return {
        "classification_target": "PHASE_198T_BN2A_FORWARD_ACCOUNTING_FOUNDATION_READY_FOR_REVIEW",
        "opening_equity": state["cutover"]["equity"],
        "opening_cash": state["cutover"]["cash"],
        "open_position_count": state["open_position_count"],
        "open_positions": state["open_positions"],
        "historical_return": metrics["historical_return"],
        "time_weighted_return": metrics["time_weighted_return"],
        "invented_realized_pl": metrics["proven_realized_sale_count"],
        "pending_fees": state["pending_fees"],
        "fees_status": (
            "PENDING"
            if not any(s.get("fees_status") == "ATTRIBUTED" for s in state["sales"])
            else "MIXED"
        ),
        "earliest_performance_date": metrics["earliest_performance_date"],
        "cutover_publication_status": state["cutover"]["publication_status"],
        "schema_version": SCHEMA_VERSION,
        "cost_basis_method": COST_BASIS_METHOD,
        "event_id": event["event_id"],
    }
