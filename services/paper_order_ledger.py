"""Canonical append-only paper-order event ledger (Phase 198T-BG).

Authoritative path: data/results/paper_order_events.jsonl
Reducer produces one logical record per order/action. CSV snapshots remain
non-authoritative. Import does not write, contact a broker, or create paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "198T-BG-PAPER-ORDER-LEDGER-1"
LEDGER_RELATIVE = "data/results/paper_order_events.jsonl"
LOCK_RELATIVE = "data/results/paper_order_events.lock"

EVENT_ACTION_SUBMITTED = "ACTION_SUBMITTED"
EVENT_ACTION_SKIPPED = "ACTION_SKIPPED"
EVENT_ACTION_FAILED = "ACTION_FAILED"
EVENT_ORDER_STATUS_RECONCILED = "ORDER_STATUS_RECONCILED"
EVENT_ORDER_TERMINAL_RECONCILED = "ORDER_TERMINAL_RECONCILED"

SOURCE_PREPLAN_SELL = "preplan_authorized_sell"
SOURCE_PLACEMENT = "place_live_orders"
SOURCE_RECENT_ORDERS = "recent_orders_snapshot"
SOURCE_LEGACY_AUDIT = "paper_execution_audit"
SOURCE_FPLAN = "fplan_management"
_PROCESS_LOCK = threading.RLock()

TERMINAL_STATUSES = frozenset({"FILLED", "REJECTED", "CANCELLED", "EXPIRED", "REPLACED"})
PROGRESS_RANK = {
    "SUBMITTED": 10,
    "ACCEPTED": 20,
    "NEW": 20,
    "PARTIALLY_FILLED": 30,
    "FILLED": 40,
    "REJECTED": 50,
    "CANCELLED": 50,
    "EXPIRED": 50,
    "REPLACED": 50,
}

_STATUS_ALIASES = {
    "NEW": "SUBMITTED",
    "PENDING_NEW": "SUBMITTED",
    "PENDING": "SUBMITTED",
    "OPEN": "ACCEPTED",
    "PARTIALLYFILLED": "PARTIALLY_FILLED",
    "PARTIAL": "PARTIALLY_FILLED",
    "PARTIAL_FILL": "PARTIALLY_FILLED",
    "DONE": "FILLED",
    "COMPLETE": "FILLED",
    "COMPLETED": "FILLED",
    "CANCELED": "CANCELLED",
    "CANCEL": "CANCELLED",
    "REJECT": "REJECTED",
    "FAILED": "REJECTED",
}


class PaperOrderLedgerError(Exception):
    """Fail-closed ledger condition. Never silently skipped."""

    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ledger_path(root: Optional[Path] = None) -> Path:
    return (Path(root) if root is not None else default_repo_root()) / LEDGER_RELATIVE


def lock_path(root: Optional[Path] = None) -> Path:
    return (Path(root) if root is not None else default_repo_root()) / LOCK_RELATIVE


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc_iso(value: Any) -> str:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    text = str(value or "").strip()
    if not text:
        return utc_now().isoformat()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def trading_date_of(ts: str) -> str:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc).date().isoformat()


def money(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_broker_status(raw: Any) -> Optional[str]:
    if raw is None or str(raw).strip() == "":
        return None
    s = str(raw).strip().upper().replace(" ", "_")
    s = _STATUS_ALIASES.get(s, s)
    if s in PROGRESS_RANK:
        return s
    return s


def canonical_symbol(raw: Any) -> str:
    return str(raw or "").strip().upper()


def logical_key(
    *,
    broker_order_id: Optional[str] = None,
    client_order_id: Optional[str] = None,
    dispatch_id: Optional[str] = None,
    action_ordinal: Optional[int] = None,
) -> str:
    broker = str(broker_order_id or "").strip()
    if broker:
        return f"broker:{broker}"
    client = str(client_order_id or "").strip()
    if client:
        return f"client:{client}"
    disp = str(dispatch_id or "").strip()
    if disp and action_ordinal is not None:
        return f"dispatch:{disp}:{int(action_ordinal)}"
    raise PaperOrderLedgerError(
        "logical_key_unresolvable",
        {"broker_order_id": broker_order_id, "client_order_id": client_order_id},
    )


def event_id_for(payload: Mapping[str, Any]) -> str:
    material = {
        "schema_version": SCHEMA_VERSION,
        "event_type": payload.get("event_type"),
        "logical_key": payload.get("logical_key"),
        "authority_id": payload.get("authority_id"),
        "dispatch_id": payload.get("dispatch_id"),
        "action_ordinal": payload.get("action_ordinal"),
        "symbol": payload.get("symbol"),
        "disposition": payload.get("action_disposition"),
        "client_order_id": payload.get("client_order_id"),
        "broker_order_id": payload.get("broker_order_id"),
        "broker_terminal_status": payload.get("broker_terminal_status"),
        "filled_qty": payload.get("filled_qty"),
        "average_fill_price": payload.get("average_fill_price"),
        "filled_timestamp": payload.get("filled_timestamp"),
        "reason_code": payload.get("reason_code"),
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def calculate_proceeds(
    filled_qty: Optional[float],
    average_fill_price: Optional[float],
    *,
    terminal_status_proven: bool,
) -> Optional[float]:
    if not terminal_status_proven:
        return None
    if filled_qty is None or average_fill_price is None:
        return None
    if filled_qty <= 0 or average_fill_price <= 0:
        return None
    return round(float(filled_qty) * float(average_fill_price), 6)


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
            raise PaperOrderLedgerError(
                "ledger_lock_unavailable", {"error": type(last_err).__name__}
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


class _Lock:
    def __init__(self, path: Path):
        self.path = path
        self._cm = _exclusive_lock(path)
        self.handle = None

    def __enter__(self):
        self.handle = self._cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._cm.__exit__(exc_type, exc, tb)


def load_events(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise PaperOrderLedgerError("ledger_bom_present", {"path": str(path)})
    text = raw.decode("utf-8")
    if text and not text.endswith("\n"):
        raise PaperOrderLedgerError("ledger_incomplete_final_line", {"path": str(path)})
    events: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for idx, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise PaperOrderLedgerError("ledger_blank_line", {"line": idx})
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PaperOrderLedgerError(
                "ledger_malformed_line", {"line": idx, "error": str(exc)}
            ) from exc
        if not isinstance(rec, dict) or not rec.get("event_id"):
            raise PaperOrderLedgerError("ledger_malformed_record", {"line": idx})
        eid = str(rec["event_id"])
        if eid in seen:
            continue
        seen.add(eid)
        events.append(rec)
    return events


def existing_event_ids(path: Path) -> set[str]:
    return {str(e["event_id"]) for e in load_events(path)}


def build_event(
    *,
    event_type: str,
    event_timestamp: Optional[str] = None,
    trading_date: Optional[str] = None,
    mode: str = "paper",
    source_pipeline: str,
    source_invocation: Optional[str] = None,
    cycle_id: Optional[str] = None,
    authority_id: Optional[str] = None,
    dispatch_id: Optional[str] = None,
    action_ordinal: Optional[int] = None,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    authorized_max_qty: Optional[float] = None,
    authorized_max_notional: Optional[float] = None,
    selected_qty: Optional[float] = None,
    submitted_reference_price: Optional[float] = None,
    submitted_notional: Optional[float] = None,
    action_disposition: Optional[str] = None,
    reason_code: Optional[str] = None,
    client_order_id: Optional[str] = None,
    broker_order_id: Optional[str] = None,
    broker_submission_status: Optional[str] = None,
    broker_terminal_status: Optional[str] = None,
    filled_qty: Optional[float] = None,
    average_fill_price: Optional[float] = None,
    filled_timestamp: Optional[str] = None,
    source_artifact: Optional[str] = None,
    terminal_status_proven: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if str(mode).lower() != "paper":
        raise PaperOrderLedgerError("live_order_refused", {"mode": mode})
    ts = as_utc_iso(event_timestamp or utc_now())
    key = logical_key(
        broker_order_id=broker_order_id,
        client_order_id=client_order_id,
        dispatch_id=dispatch_id,
        action_ordinal=action_ordinal,
    )
    proceeds = calculate_proceeds(
        filled_qty, average_fill_price, terminal_status_proven=terminal_status_proven
    )
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_type": event_type,
        "event_timestamp": ts,
        "trading_date": trading_date or trading_date_of(ts),
        "mode": "paper",
        "source_pipeline": source_pipeline,
        "source_invocation": source_invocation,
        "cycle_id": cycle_id,
        "authority_id": authority_id,
        "dispatch_id": dispatch_id,
        "action_ordinal": action_ordinal,
        "symbol": canonical_symbol(symbol) if symbol else None,
        "side": str(side or "").upper() or None,
        "authorized_max_qty": authorized_max_qty,
        "authorized_max_notional": authorized_max_notional,
        "selected_qty": selected_qty,
        "submitted_reference_price": submitted_reference_price,
        "submitted_notional": submitted_notional,
        "action_disposition": action_disposition,
        "reason_code": reason_code,
        "client_order_id": client_order_id or None,
        "broker_order_id": broker_order_id or None,
        "broker_submission_status": normalize_broker_status(broker_submission_status),
        "broker_terminal_status": normalize_broker_status(broker_terminal_status),
        "filled_qty": filled_qty,
        "average_fill_price": average_fill_price,
        "filled_timestamp": as_utc_iso(filled_timestamp) if filled_timestamp else None,
        "proceeds": proceeds,
        "source_artifact": source_artifact,
        "terminal_status_proven": bool(terminal_status_proven),
        "logical_key": key,
    }
    if extra:
        event["extra"] = dict(extra)
    event["event_id"] = event_id_for(event)
    return event


def append_events(
    events: Sequence[Mapping[str, Any]],
    *,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    path = ledger_path(root)
    lock = lock_path(root)
    appended = 0
    skipped = 0
    with _Lock(lock):
        existing = existing_event_ids(path) if path.is_file() else set()
        to_write: List[Dict[str, Any]] = []
        for raw in events:
            rec = dict(raw)
            if rec.get("schema_version") != SCHEMA_VERSION:
                raise PaperOrderLedgerError("schema_mismatch", {"event": rec.get("event_id")})
            eid = str(rec.get("event_id") or "")
            if not eid:
                raise PaperOrderLedgerError("missing_event_id")
            if eid in existing:
                skipped += 1
                continue
            existing.add(eid)
            to_write.append(rec)
        if to_write:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = "".join(
                json.dumps(e, separators=(",", ":"), default=str) + "\n" for e in to_write
            )
            with path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            appended = len(to_write)
    return {
        "ok": True,
        "appended": appended,
        "skipped_duplicate": skipped,
        "path": str(path),
        "written_events": to_write,
    }


def reduce_events(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(
        events,
        key=lambda e: (str(e.get("event_timestamp") or ""), str(e.get("event_id") or "")),
    )
    rows: Dict[str, Dict[str, Any]] = {}
    aliases: Dict[str, str] = {}

    def resolve(key: str) -> str:
        seen = set()
        cur = key
        while cur in aliases and cur not in seen:
            seen.add(cur)
            cur = aliases[cur]
        return cur

    for ev in ordered:
        raw_key = str(ev.get("logical_key") or "")
        if not raw_key:
            continue
        broker = str(ev.get("broker_order_id") or "").strip()
        client = str(ev.get("client_order_id") or "").strip()
        candidates = [raw_key]
        if broker:
            candidates.append(f"broker:{broker}")
        if client:
            candidates.append(f"client:{client}")
        key = None
        for cand in candidates:
            resolved = resolve(cand)
            if resolved in rows:
                key = resolved
                break
        if key is None:
            key = resolve(raw_key)
        for cand in candidates:
            aliases[cand] = key
        row = rows.get(key)
        if row is None:
            row = {
                "logical_key": key,
                "kind": (
                    "order"
                    if ev.get("event_type") == EVENT_ACTION_SUBMITTED or ev.get("broker_order_id")
                    else "action"
                ),
                "authority_id": ev.get("authority_id"),
                "dispatch_id": ev.get("dispatch_id"),
                "action_ordinal": ev.get("action_ordinal"),
                "symbol": ev.get("symbol"),
                "side": ev.get("side"),
                "trading_date": ev.get("trading_date"),
                "source_pipeline": ev.get("source_pipeline"),
                "cycle_id": ev.get("cycle_id"),
                "authorized_max_qty": ev.get("authorized_max_qty"),
                "authorized_max_notional": ev.get("authorized_max_notional"),
                "selected_qty": ev.get("selected_qty"),
                "submitted_notional": ev.get("submitted_notional"),
                "submitted_reference_price": ev.get("submitted_reference_price"),
                "action_disposition": ev.get("action_disposition"),
                "reason_code": ev.get("reason_code"),
                "client_order_id": ev.get("client_order_id"),
                "broker_order_id": ev.get("broker_order_id"),
                "submission_status": ev.get("broker_submission_status"),
                "terminal_status": ev.get("broker_terminal_status"),
                "filled_qty": ev.get("filled_qty"),
                "average_fill_price": ev.get("average_fill_price"),
                "filled_timestamp": ev.get("filled_timestamp"),
                "event_timestamp": ev.get("event_timestamp"),
                "terminal_status_proven": False,
                "proceeds": None,
                "invalid_regression": None,
                "source_artifact": ev.get("source_artifact"),
                "fees_commissions": "UNPROVEN",
            }
            rows[key] = row
        else:
            for field in (
                "authority_id",
                "dispatch_id",
                "action_ordinal",
                "symbol",
                "side",
                "trading_date",
                "client_order_id",
                "broker_order_id",
                "cycle_id",
                "authorized_max_qty",
                "authorized_max_notional",
                "selected_qty",
                "submitted_notional",
                "submitted_reference_price",
            ):
                if ev.get(field) and not row.get(field):
                    row[field] = ev.get(field)
        et = str(ev.get("event_type") or "")
        if et == EVENT_ACTION_SKIPPED:
            row["kind"] = "action"
            row["action_disposition"] = "explicitly_skipped"
            row["reason_code"] = ev.get("reason_code") or row.get("reason_code")
            row["submission_status"] = None
            row["terminal_status"] = None
            row["proceeds"] = None
            row["terminal_status_proven"] = False
        elif et == EVENT_ACTION_FAILED:
            row["kind"] = "action"
            row["action_disposition"] = "failed"
            row["reason_code"] = ev.get("reason_code") or row.get("reason_code")
            row["proceeds"] = None
        elif et == EVENT_ACTION_SUBMITTED:
            row["kind"] = "order"
            row["action_disposition"] = "submitted"
            row["submission_status"] = ev.get("broker_submission_status") or "SUBMITTED"
            if not row.get("terminal_status_proven"):
                row["terminal_status"] = None
        elif et in {EVENT_ORDER_STATUS_RECONCILED, EVENT_ORDER_TERMINAL_RECONCILED}:
            incoming = ev.get("broker_terminal_status") or ev.get("broker_submission_status")
            incoming_n = normalize_broker_status(incoming)
            current = row.get("terminal_status") or row.get("submission_status")
            cur_rank = PROGRESS_RANK.get(str(current or ""), 0)
            new_rank = PROGRESS_RANK.get(str(incoming_n or ""), 0)
            if current in TERMINAL_STATUSES and incoming_n and incoming_n not in TERMINAL_STATUSES:
                if new_rank < cur_rank:
                    row["invalid_regression"] = f"{current}->{incoming_n}"
                    continue
            if incoming_n and new_rank >= cur_rank:
                if incoming_n in TERMINAL_STATUSES and ev.get("terminal_status_proven"):
                    row["terminal_status"] = incoming_n
                    row["terminal_status_proven"] = True
                    row["filled_qty"] = ev.get("filled_qty")
                    row["average_fill_price"] = ev.get("average_fill_price")
                    row["filled_timestamp"] = ev.get("filled_timestamp")
                    row["proceeds"] = calculate_proceeds(
                        ev.get("filled_qty"),
                        ev.get("average_fill_price"),
                        terminal_status_proven=True,
                    )
                else:
                    row["submission_status"] = incoming_n
        if ev.get("event_timestamp"):
            row["last_event_timestamp"] = ev.get("event_timestamp")
    return list(rows.values())


def reduce_ledger(*, root: Optional[Path] = None) -> List[Dict[str, Any]]:
    return reduce_events(load_events(ledger_path(root)))


def record_sell_dispatch_events(
    *,
    authority: Mapping[str, Any],
    dispatch_id: str,
    clip_audit: Sequence[Mapping[str, Any]],
    submissions: Sequence[Mapping[str, Any]],
    failures: Optional[Sequence[Mapping[str, Any]]] = None,
    cycle_id: Optional[str] = None,
    source_invocation: Optional[str] = None,
    event_timestamp: Optional[str] = None,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = authority.get("payload") if isinstance(authority.get("payload"), dict) else authority
    trading_date = str(payload.get("trading_date") or "") or None
    aid = str(authority.get("authority_id") or payload.get("authority_id") or "")
    by_symbol_sub = {}
    for s in submissions:
        by_symbol_sub.setdefault(canonical_symbol(s.get("symbol")), []).append(dict(s))
    by_symbol_fail = {}
    for f in failures or []:
        by_symbol_fail.setdefault(canonical_symbol(f.get("symbol")), []).append(dict(f))
    events: List[Dict[str, Any]] = []
    submitted = skipped = failed = 0
    for idx, clip in enumerate(clip_audit, start=1):
        sym = canonical_symbol(clip.get("symbol"))
        qty = money(clip.get("runtime_quantity"))
        outcome = str(clip.get("outcome") or "")
        sub = (by_symbol_sub.get(sym) or [None]).pop(0) if by_symbol_sub.get(sym) else None
        fail = (by_symbol_fail.get(sym) or [None]).pop(0) if by_symbol_fail.get(sym) else None
        if outcome == "SKIPPED_ZERO" or (qty is not None and qty <= 0 and not sub):
            skipped += 1
            events.append(
                build_event(
                    event_type=EVENT_ACTION_SKIPPED,
                    event_timestamp=event_timestamp,
                    trading_date=trading_date,
                    source_pipeline=SOURCE_PREPLAN_SELL,
                    source_invocation=source_invocation,
                    cycle_id=cycle_id,
                    authority_id=aid,
                    dispatch_id=dispatch_id,
                    action_ordinal=idx,
                    symbol=sym,
                    side="SELL",
                    authorized_max_qty=money(clip.get("authorized_quantity")),
                    authorized_max_notional=money(clip.get("authorized_max_notional")),
                    selected_qty=qty or 0.0,
                    submitted_reference_price=money(clip.get("current_reference_price")),
                    submitted_notional=0.0,
                    action_disposition="explicitly_skipped",
                    reason_code=str(clip.get("reason") or outcome or "SKIPPED_ZERO"),
                    source_artifact="sell_dispatches.intent_json.clip_audit",
                )
            )
            continue
        if fail and not sub:
            failed += 1
            events.append(
                build_event(
                    event_type=EVENT_ACTION_FAILED,
                    event_timestamp=event_timestamp,
                    trading_date=trading_date,
                    source_pipeline=SOURCE_PREPLAN_SELL,
                    source_invocation=source_invocation,
                    cycle_id=cycle_id,
                    authority_id=aid,
                    dispatch_id=dispatch_id,
                    action_ordinal=idx,
                    symbol=sym,
                    side="SELL",
                    authorized_max_qty=money(clip.get("authorized_quantity")),
                    authorized_max_notional=money(clip.get("authorized_max_notional")),
                    selected_qty=qty,
                    action_disposition="failed",
                    reason_code=str(fail.get("reason") or "broker_submit_failed"),
                    source_artifact="place_live_orders.preplan_sell",
                )
            )
            continue
        if sub:
            submitted += 1
            events.append(
                build_event(
                    event_type=EVENT_ACTION_SUBMITTED,
                    event_timestamp=event_timestamp,
                    trading_date=trading_date,
                    source_pipeline=SOURCE_PREPLAN_SELL,
                    source_invocation=source_invocation,
                    cycle_id=cycle_id,
                    authority_id=aid,
                    dispatch_id=dispatch_id,
                    action_ordinal=idx,
                    symbol=sym,
                    side="SELL",
                    authorized_max_qty=money(clip.get("authorized_quantity")),
                    authorized_max_notional=money(clip.get("authorized_max_notional")),
                    selected_qty=money(sub.get("qty")) if sub.get("qty") is not None else qty,
                    submitted_reference_price=money(clip.get("current_reference_price")),
                    submitted_notional=money(clip.get("runtime_estimated_notional")),
                    action_disposition="submitted",
                    reason_code=str(clip.get("reason") or ""),
                    client_order_id=str(sub.get("client_order_id") or "") or None,
                    broker_order_id=str(sub.get("broker_order_id") or "") or None,
                    broker_submission_status=sub.get("broker_status") or "SUBMITTED",
                    terminal_status_proven=False,
                    source_artifact="place_live_orders.preplan_sell",
                )
            )
            continue
        failed += 1
        events.append(
            build_event(
                event_type=EVENT_ACTION_FAILED,
                event_timestamp=event_timestamp,
                trading_date=trading_date,
                source_pipeline=SOURCE_PREPLAN_SELL,
                source_invocation=source_invocation,
                cycle_id=cycle_id,
                authority_id=aid,
                dispatch_id=dispatch_id,
                action_ordinal=idx,
                symbol=sym,
                side="SELL",
                authorized_max_qty=money(clip.get("authorized_quantity")),
                authorized_max_notional=money(clip.get("authorized_max_notional")),
                selected_qty=qty,
                action_disposition="failed",
                reason_code="missing_broker_submission",
                source_artifact="place_live_orders.preplan_sell",
            )
        )
    authorized = len(list(clip_audit))
    if submitted + skipped + failed != authorized:
        raise PaperOrderLedgerError(
            "dispatch_count_mismatch",
            {
                "submitted": submitted,
                "skipped": skipped,
                "failed": failed,
                "authorized": authorized,
            },
        )
    try:
        result = append_events(events, root=root)
    except PaperOrderLedgerError as exc:
        return {
            "ok": False,
            "reason": exc.reason,
            "details": exc.details,
            "authority_reusable": False,
            "replay_forbidden": True,
            "submitted": submitted,
            "skipped": skipped,
            "failed": failed,
        }
    result.update(
        {
            "submitted": submitted,
            "skipped": skipped,
            "failed": failed,
            "authorized": authorized,
            "replay_forbidden": True,
        }
    )
    return result


def _row_from_recent(row: Mapping[str, Any]) -> Dict[str, Any]:
    status = normalize_broker_status(row.get("status") or row.get("order_status"))
    filled_qty = money(row.get("filled_qty"))
    avg = money(row.get("filled_avg_price"))
    proven = status in TERMINAL_STATUSES
    if status == "PARTIALLY_FILLED":
        proven = False
    return {
        "id": str(row.get("id") or row.get("order_id") or "").strip() or None,
        "client_order_id": str(row.get("client_order_id") or "").strip() or None,
        "symbol": canonical_symbol(row.get("symbol") or row.get("ticker")),
        "side": str(row.get("side") or "").upper() or None,
        "qty": money(row.get("qty") or row.get("quantity")),
        "status": status,
        "filled_qty": filled_qty,
        "average_fill_price": avg,
        "submitted_at": row.get("submitted_at") or row.get("created_at"),
        "filled_at": row.get("filled_at"),
        "snapshot_ts": row.get("snapshot_ts"),
    }


def _placement_side(side: Any) -> Optional[str]:
    text = str(side or "").strip().upper().replace(" ", "_")
    if text in {"BUY", "BUY_TO_OPEN"}:
        return "BUY"
    if text in {"SELL", "SELL_TO_CLOSE"}:
        return "SELL"
    return None


def _placement_identity_exists(
    root: Optional[Path],
    broker_order_id: Optional[str],
    client_order_id: Optional[str],
) -> bool:
    """True when either alias is already a reduced ledger identity."""
    broker = str(broker_order_id or "").strip()
    client = str(client_order_id or "").strip()
    if not broker and not client:
        return False
    for row in reduce_ledger(root=root):
        row_broker = str(row.get("broker_order_id") or "").strip()
        row_client = str(row.get("client_order_id") or "").strip()
        row_key = str(row.get("logical_key") or "").strip()
        if broker and (row_broker == broker or row_key == f"broker:{broker}"):
            return True
        if client and (row_client == client or row_key == f"client:{client}"):
            return True
    return False


def record_placement_submission(
    *,
    symbol: Any,
    side: Any,
    qty: Any = None,
    client_order_id: Optional[str] = None,
    broker_order_id: Optional[str] = None,
    mode: str = "paper",
    event_timestamp: Optional[str] = None,
    limit_price: Any = None,
    root: Optional[Path] = None,
    source_artifact: str = "place_live_orders.submit",
) -> Dict[str, Any]:
    """Record one paper submission identity. Existing broker or client alias writes nothing.

    The pre-plan authorized SELL writer is a separate function and is not called here.
    """
    if str(mode).lower() != "paper":
        return {"ok": True, "appended": 0, "skipped_duplicate": 0, "reason": "not_paper"}
    side_n = _placement_side(side)
    if side_n is None:
        return {"ok": False, "appended": 0, "skipped_duplicate": 0, "reason": "side_unsupported"}
    broker = str(broker_order_id or "").strip() or None
    client = str(client_order_id or "").strip() or None
    if not broker and not client:
        return {
            "ok": False,
            "appended": 0,
            "skipped_duplicate": 0,
            "reason": "missing_order_identity",
        }
    if _placement_identity_exists(root, broker, client):
        return {
            "ok": True,
            "appended": 0,
            "skipped_duplicate": 1,
            "reason": "identity_exists",
        }
    event = build_event(
        event_type=EVENT_ACTION_SUBMITTED,
        event_timestamp=event_timestamp,
        mode="paper",
        source_pipeline=SOURCE_PLACEMENT,
        symbol=canonical_symbol(symbol),
        side=side_n,
        selected_qty=money(qty),
        submitted_reference_price=money(limit_price),
        action_disposition="submitted",
        client_order_id=client,
        broker_order_id=broker,
        broker_submission_status="SUBMITTED",
        terminal_status_proven=False,
        source_artifact=source_artifact,
    )
    result = append_events([event], root=root)
    result["reason"] = "submitted"
    return result


def safe_record_placement_submission(**kwargs: Any) -> Dict[str, Any]:
    """Never raise into the generic submit loop after the broker has accepted an order."""
    try:
        return record_placement_submission(**kwargs)
    except PaperOrderLedgerError as exc:
        return {
            "ok": False,
            "appended": 0,
            "skipped_duplicate": 0,
            "reason": exc.reason,
            "details": exc.details,
        }


def load_recent_order_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    import csv

    with path.open(encoding="utf-8", newline="") as fh:
        return [_row_from_recent(r) for r in csv.DictReader(fh)]


def reconcile_from_recent_orders(
    *,
    root: Optional[Path] = None,
    recent_path: Optional[Path] = None,
    snapshot_ts: Optional[str] = None,
) -> Dict[str, Any]:
    root_p = Path(root) if root is not None else default_repo_root()
    csv_path = (
        Path(recent_path)
        if recent_path is not None
        else root_p / "data" / "results" / "recent_orders.csv"
    )
    if not csv_path.is_file():
        return {"ok": True, "appended": 0, "reason": "recent_orders_absent"}
    existing = reduce_ledger(root=root_p)
    by_broker = {r["broker_order_id"]: r for r in existing if r.get("broker_order_id")}
    by_client = {r["client_order_id"]: r for r in existing if r.get("client_order_id")}
    events: List[Dict[str, Any]] = []
    for rec in load_recent_order_rows(csv_path):
        if rec.get("side") and rec["side"] not in {"SELL", "SELL_TO_CLOSE", "BUY", "BUY_TO_OPEN"}:
            continue
        match = None
        method = None
        if rec.get("id") and rec["id"] in by_broker:
            match = by_broker[rec["id"]]
            method = "broker_order_id"
        elif rec.get("client_order_id") and rec["client_order_id"] in by_client:
            match = by_client[rec["client_order_id"]]
            method = "client_order_id"
        if match is None:
            continue
        status = rec.get("status")
        if not status:
            continue
        current_term = match.get("terminal_status")
        if current_term == status and match.get("filled_qty") == rec.get("filled_qty"):
            continue
        if current_term in TERMINAL_STATUSES and status not in TERMINAL_STATUSES:
            if PROGRESS_RANK.get(status, 0) < PROGRESS_RANK.get(str(current_term), 0):
                continue
        proven = status in TERMINAL_STATUSES
        et = EVENT_ORDER_TERMINAL_RECONCILED if proven else EVENT_ORDER_STATUS_RECONCILED
        events.append(
            build_event(
                event_type=et,
                event_timestamp=snapshot_ts or rec.get("snapshot_ts") or utc_now(),
                trading_date=match.get("trading_date"),
                source_pipeline=SOURCE_RECENT_ORDERS,
                authority_id=match.get("authority_id"),
                dispatch_id=match.get("dispatch_id"),
                action_ordinal=match.get("action_ordinal"),
                symbol=rec.get("symbol") or match.get("symbol"),
                side=match.get("side") or rec.get("side"),
                selected_qty=rec.get("qty") or match.get("selected_qty"),
                client_order_id=rec.get("client_order_id") or match.get("client_order_id"),
                broker_order_id=rec.get("id") or match.get("broker_order_id"),
                broker_submission_status=None if proven else status,
                broker_terminal_status=status if proven else None,
                filled_qty=rec.get("filled_qty"),
                average_fill_price=rec.get("average_fill_price"),
                filled_timestamp=rec.get("filled_at"),
                terminal_status_proven=proven,
                source_artifact=str(csv_path),
                extra={"correlation": method},
            )
        )
    if not events:
        return {"ok": True, "appended": 0, "matched": 0, "accounting": {"appended": 0}}
    result = append_events(events, root=root_p)
    result["matched"] = len(events)
    written = list(result.get("written_events") or [])
    if written:
        from services.performance_accounting import record_proven_terminal_fills

        result["accounting"] = record_proven_terminal_fills(written, root=root_p)
    else:
        result["accounting"] = {"ok": True, "appended": 0, "skipped_duplicate": 0}
    return result


def safe_reconcile_after_snapshot(
    *,
    root: Optional[Path] = None,
    recent_path: Optional[Path] = None,
    snapshot_ts: Optional[str] = None,
) -> Dict[str, Any]:
    """Never raise into the snapshot writer. Never contacts a broker."""
    try:
        return reconcile_from_recent_orders(
            root=root, recent_path=recent_path, snapshot_ts=snapshot_ts
        )
    except PaperOrderLedgerError as exc:
        return {"ok": False, "reason": exc.reason, "details": exc.details}
    except Exception as exc:
        return {
            "ok": False,
            "reason": "reconcile_exception",
            "details": {"error": type(exc).__name__},
        }


def safe_record_sell_dispatch(**kwargs: Any) -> Dict[str, Any]:
    try:
        return record_sell_dispatch_events(**kwargs)
    except PaperOrderLedgerError as exc:
        return {
            "ok": False,
            "reason": exc.reason,
            "details": exc.details,
            "authority_reusable": False,
            "replay_forbidden": True,
        }


def merge_legacy_audit_rows(
    reduced: Sequence[Mapping[str, Any]],
    *,
    root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Read-only merge of paper_execution_audit for BUY/F-plan visibility."""
    root_p = Path(root) if root is not None else default_repo_root()
    audit = root_p / "data" / "results" / "paper_execution_audit.csv"
    if not audit.is_file():
        return list(reduced)
    import csv

    known = set()
    out = [dict(r) for r in reduced]
    for r in out:
        if r.get("broker_order_id"):
            known.add(f"broker:{r['broker_order_id']}")
        if r.get("client_order_id"):
            known.add(f"client:{r['client_order_id']}")
    with audit.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            oid = str(row.get("submitted_order_id") or "").strip()
            if not oid:
                continue
            key = f"broker:{oid}"
            if key in known:
                continue
            known.add(key)
            status = normalize_broker_status(row.get("status"))
            out.append(
                {
                    "logical_key": key,
                    "kind": "order",
                    "authority_id": None,
                    "dispatch_id": None,
                    "symbol": canonical_symbol(row.get("symbol")),
                    "side": str(row.get("action") or row.get("signal") or "").upper() or None,
                    "trading_date": str(row.get("timestamp") or "")[:10],
                    "source_pipeline": SOURCE_LEGACY_AUDIT,
                    "selected_qty": money(row.get("qty")),
                    "submission_status": status if status not in TERMINAL_STATUSES else "SUBMITTED",
                    "terminal_status": status if status in TERMINAL_STATUSES else None,
                    "terminal_status_proven": status in TERMINAL_STATUSES,
                    "broker_order_id": oid,
                    "client_order_id": None,
                    "filled_qty": None,
                    "average_fill_price": None,
                    "proceeds": None,
                    "event_timestamp": row.get("timestamp"),
                    "fees_commissions": "UNPROVEN",
                    "source_artifact": str(audit),
                    "action_disposition": row.get("status") or row.get("action"),
                }
            )
    return out


def unified_order_rows(*, root: Optional[Path] = None) -> List[Dict[str, Any]]:
    return merge_legacy_audit_rows(reduce_ledger(root=root), root=root)


def filter_unified_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    trading_date: Optional[str] = None,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    status: Optional[str] = None,
    authority_id: Optional[str] = None,
    dispatch_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out = []
    want_side = str(side or "").upper()
    want_status = str(status or "").upper()
    for r in rows:
        if trading_date and str(r.get("trading_date") or "") != trading_date:
            continue
        if symbol and canonical_symbol(r.get("symbol")) != canonical_symbol(symbol):
            continue
        if want_side in {"BUY", "SELL"}:
            rs = str(r.get("side") or "")
            if (
                want_side == "SELL"
                and "SELL" not in rs
                and r.get("source_pipeline") != SOURCE_PREPLAN_SELL
            ):
                if r.get("action_disposition") not in {"submitted", "explicitly_skipped", "failed"}:
                    continue
            if want_side == "BUY" and "BUY" not in rs and "ADD" not in rs:
                continue
        if want_status:
            term = str(
                r.get("terminal_status")
                or r.get("submission_status")
                or r.get("action_disposition")
                or ""
            ).upper()
            aliases = {
                "OPEN": {"SUBMITTED", "ACCEPTED", "NEW"},
                "SUBMITTED": {"SUBMITTED", "ACCEPTED"},
                "PARTIAL": {"PARTIALLY_FILLED"},
                "FILLED": {"FILLED"},
                "REJECTED": {"REJECTED", "FAILED"},
                "CANCELED": {"CANCELLED"},
                "CANCELLED": {"CANCELLED"},
                "EXPIRED": {"EXPIRED"},
            }
            allowed = aliases.get(want_status, {want_status})
            if term not in allowed:
                continue
        if authority_id and str(r.get("authority_id") or "") != authority_id:
            continue
        if dispatch_id and str(r.get("dispatch_id") or "") != dispatch_id:
            continue
        out.append(dict(r))
    return out


def dispatch_visibility(
    rows: Sequence[Mapping[str, Any]], *, authority_id: Optional[str] = None
) -> Dict[str, Any]:
    scoped = [r for r in rows if not authority_id or r.get("authority_id") == authority_id]
    submitted = [
        r
        for r in scoped
        if r.get("action_disposition") == "submitted"
        or r.get("kind") == "order"
        and r.get("broker_order_id")
    ]
    skipped = [r for r in scoped if r.get("action_disposition") == "explicitly_skipped"]
    failed = [r for r in scoped if r.get("action_disposition") == "failed"]
    filled = [
        r
        for r in submitted
        if r.get("terminal_status") == "FILLED" and r.get("terminal_status_proven")
    ]
    open_rows = [r for r in submitted if not r.get("terminal_status_proven")]
    rejected = [
        r for r in submitted if r.get("terminal_status") in {"REJECTED", "CANCELLED", "EXPIRED"}
    ]
    proceeds = sum(float(r["proceeds"]) for r in filled if r.get("proceeds") is not None)
    return {
        "authority_id": authority_id,
        "submitted": (
            len(submitted) if submitted else len([r for r in scoped if r.get("kind") == "order"])
        ),
        "skipped": len(skipped),
        "failed": len(failed),
        "filled": len(filled),
        "open": len(open_rows),
        "rejected": len(rejected),
        "filled_proceeds": round(proceeds, 6) if filled else None,
        "has_dispatch": bool(scoped),
        "dispatch_ids": sorted({str(r.get("dispatch_id")) for r in scoped if r.get("dispatch_id")}),
    }


def consumed_authority_message(vis: Mapping[str, Any]) -> str:
    submitted = int(vis.get("submitted") or 0)
    skipped = int(vis.get("skipped") or 0)
    failed = int(vis.get("failed") or 0)
    filled = vis.get("filled")
    if submitted <= 0 and skipped <= 0 and failed <= 0:
        return "Paper SELL authority consumed. Execution evidence is incomplete. Open Orders or Activity."
    msg = (
        f"Paper SELL authority consumed: {submitted} submitted, {skipped} skipped, "
        f"{failed} failed. Open Orders or Activity for execution status."
    )
    if vis.get("filled") and vis.get("filled_proceeds") is not None:
        msg += f" Terminal snapshot proves {int(filled)} filled."
    return msg


def activity_events_from_ledger(*, root: Optional[Path] = None) -> List[Dict[str, Any]]:
    events = load_events(ledger_path(root))
    out: List[Dict[str, Any]] = []
    by_dispatch: Dict[str, List[Dict[str, Any]]] = {}
    for ev in events:
        did = str(ev.get("dispatch_id") or "")
        if did:
            by_dispatch.setdefault(did, []).append(ev)
        et = ev.get("event_type")
        title = {
            EVENT_ACTION_SUBMITTED: "Paper order was submitted",
            EVENT_ACTION_SKIPPED: "Authorized SELL action was skipped",
            EVENT_ACTION_FAILED: "Paper order submission failed",
            EVENT_ORDER_STATUS_RECONCILED: "Paper order status advanced",
            EVENT_ORDER_TERMINAL_RECONCILED: "Paper order reached terminal broker status",
        }.get(str(et), "Paper order event")
        term = ev.get("broker_terminal_status")
        if et == EVENT_ORDER_TERMINAL_RECONCILED and term == "FILLED":
            title = "Paper order was filled"
        elif et == EVENT_ORDER_TERMINAL_RECONCILED and term == "PARTIALLY_FILLED":
            title = "Paper order was partially filled"
        elif et == EVENT_ORDER_TERMINAL_RECONCILED and term in {"REJECTED", "CANCELLED", "EXPIRED"}:
            title = f"Paper order was {str(term).lower()}"
        out.append(
            {
                "event_id": ev.get("event_id"),
                "occurred_at": ev.get("event_timestamp"),
                "occurred_date": ev.get("trading_date"),
                "source": "paper_order_events",
                "category": "ORDER",
                "severity": "SUCCESS" if et != EVENT_ACTION_FAILED else "WARN",
                "title": title,
                "message": (
                    f"{ev.get('symbol') or ''} {ev.get('side') or ''} "
                    f"disposition={ev.get('action_disposition') or ev.get('broker_terminal_status') or ev.get('event_type')} "
                    f"authority={ev.get('authority_id') or 'n/a'} dispatch={ev.get('dispatch_id') or 'n/a'}"
                ).strip(),
                "evidence_path": ev.get("source_artifact"),
                "symbol": ev.get("symbol"),
                "order_id": ev.get("broker_order_id") or ev.get("client_order_id"),
                "raw_reference": json.dumps(
                    {
                        "authority_id": ev.get("authority_id"),
                        "dispatch_id": ev.get("dispatch_id"),
                        "event_type": ev.get("event_type"),
                    }
                ),
            }
        )
    for did, group in by_dispatch.items():
        if not any(e.get("event_type") == EVENT_ACTION_SUBMITTED for e in group):
            if not any(
                e.get("event_type") in {EVENT_ACTION_SKIPPED, EVENT_ACTION_FAILED} for e in group
            ):
                continue
        first = min(group, key=lambda e: str(e.get("event_timestamp") or ""))
        submitted = sum(1 for e in group if e.get("event_type") == EVENT_ACTION_SUBMITTED)
        skipped = sum(1 for e in group if e.get("event_type") == EVENT_ACTION_SKIPPED)
        failed = sum(1 for e in group if e.get("event_type") == EVENT_ACTION_FAILED)
        filled = sum(
            1
            for e in group
            if e.get("event_type") == EVENT_ORDER_TERMINAL_RECONCILED
            and e.get("broker_terminal_status") == "FILLED"
        )
        out.append(
            {
                "event_id": hashlib.sha256(
                    f"batch:{did}:{submitted}:{skipped}:{failed}".encode()
                ).hexdigest(),
                "occurred_at": first.get("event_timestamp"),
                "occurred_date": first.get("trading_date"),
                "source": "paper_order_events",
                "category": "ORDER",
                "severity": "SUCCESS",
                "title": "Paper SELL batch dispatched",
                "message": (
                    f"Dispatch {did} authority {first.get('authority_id')}: "
                    f"{submitted} submitted, {skipped} skipped, {failed} failed"
                    + (f", {filled} filled" if filled else "")
                    + "."
                ),
                "evidence_path": LEDGER_RELATIVE,
                "symbol": None,
                "order_id": did,
                "raw_reference": json.dumps(
                    {"dispatch_id": did, "authority_id": first.get("authority_id")}
                ),
            }
        )
        out.append(
            {
                "event_id": hashlib.sha256(
                    f"batch-complete:{did}:{submitted}:{skipped}:{failed}:{filled}".encode()
                ).hexdigest(),
                "occurred_at": max(str(e.get("event_timestamp") or "") for e in group),
                "occurred_date": first.get("trading_date"),
                "source": "paper_order_events",
                "category": "ORDER",
                "severity": "SUCCESS",
                "title": "Paper SELL batch completion summary",
                "message": (
                    f"Dispatch {did} summary: {submitted} submitted, {skipped} skipped, "
                    f"{failed} failed, {filled} terminal filled."
                ),
                "evidence_path": LEDGER_RELATIVE,
                "symbol": None,
                "order_id": did,
                "raw_reference": json.dumps({"dispatch_id": did}),
            }
        )
    seen: set[str] = set()
    unique = []
    for ev in out:
        eid = str(ev.get("event_id") or "")
        if eid in seen:
            continue
        seen.add(eid)
        unique.append(ev)
    unique.sort(key=lambda e: str(e.get("occurred_at") or ""), reverse=True)
    return unique


def merge_activity_payload(
    payload: Mapping[str, Any], *, root: Optional[Path] = None
) -> Dict[str, Any]:
    merged = dict(payload)
    extra = activity_events_from_ledger(root=root)
    events = list(merged.get("events") or [])
    seen = {str(e.get("event_id") or "") for e in events}
    for ev in extra:
        if str(ev.get("event_id") or "") in seen:
            continue
        events.append(ev)
        seen.add(str(ev.get("event_id") or ""))
    events.sort(key=lambda e: str(e.get("occurred_at") or ""), reverse=True)
    merged["events"] = events
    merged["event_count"] = len(events)
    sources = list(merged.get("sources_used") or [])
    path = str(ledger_path(root))
    if path not in sources and extra:
        sources.append(path)
    merged["sources_used"] = sources
    return merged


def proceeds_summary(
    *, root: Optional[Path] = None, trading_date: Optional[str] = None
) -> Dict[str, Any]:
    rows = [
        r
        for r in reduce_ledger(root=root)
        if r.get("kind") == "order"
        and r.get("terminal_status") == "FILLED"
        and r.get("terminal_status_proven")
        and (not trading_date or r.get("trading_date") == trading_date)
    ]
    total = round(sum(float(r["proceeds"]) for r in rows if r.get("proceeds") is not None), 6)
    return {
        "filled_count": len(rows),
        "filled_proceeds": total,
        "fees_commissions": "UNPROVEN",
        "realized_pl": None,
        "realized_pl_availability": "UNAVAILABLE",
        "realized_pl_reason": "Sale proceeds are not a cost-basis realized P/L figure.",
        "authoritative_for_cost_basis": "existing_portfolio_accounting",
    }
