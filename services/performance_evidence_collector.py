"""Phase 198T-BN2B/BN2C/BN2F — isolated performance evidence collector + publication.

Standalone observer around services.performance_accounting. Import creates no
files and contacts no broker. BUY/SELL decision and submission paths are never
imported or invoked. All readers and output roots are injected. Default mode
is dry-run (zero writes). Production writes are allowed only through
--publish-cutover / --publish-session-close after the complete gate mints an
in-memory publication authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

from services.performance_publication_authority import (
    COMMAND_CUTOVER,
    COMMAND_SESSION_CLOSE,
    EMPTY_LEDGER_SHA256,
    REQUIRED_BRANCH,
    inspect_git_identity,
    inspect_runtime_state,
    is_real_production_path,
    mint_publication_authority,
    sha256_bytes,
)
from services.performance_accounting import (
    ACCOUNT_MODEL,
    EVENT_BENCHMARK_CLOSE,
    EVENT_CASH_ADJUSTMENT,
    EVENT_CASH_DEPOSIT,
    EVENT_CASH_WITHDRAWAL,
    EVENT_CUTOVER,
    EVENT_DIVIDEND,
    EVENT_FEE,
    EVENT_FILL_BUY,
    EVENT_FILL_SELL,
    EVENT_POSITION_SNAPSHOT,
    EVENT_PRICE_MARKS,
    EVENT_SESSION_CLOSE,
    EVENTS_RELATIVE,
    FEE_COMPONENTS,
    IDENTITY_TOLERANCE,
    LOCK_RELATIVE,
    PerformanceAccountingError,
    PROPOSED_CUTOVER_CASH,
    PROPOSED_CUTOVER_EQUITY,
    PROPOSED_CUTOVER_UTC,
    UNAVAILABLE,
    append_events,
    build_event,
    build_external_flow,
    encode_ledger_suffix,
    identity_residual,
    load_events,
    plan_ledger_writes,
    publish_ledger_atomic,
    q_cash,
    redact_account_ids,
    reduce_events,
    reduce_ledger,
    session_date_et,
    utc_iso,
)

COLLECTOR_SCHEMA = "198T-BN2B-PERFORMANCE-EVIDENCE-COLLECTOR-1"
PUBLICATION_SCHEMA = "198T-BN2C-PROTECTED-ACCOUNTING-PUBLICATION-1"
PRODUCTION_PUBLICATION_SCHEMA = "198T-BN2F-PRODUCTION-PUBLICATION-GATE-1"
CANONICAL_EVENTS_RELATIVE = EVENTS_RELATIVE
CANONICAL_LOCK_RELATIVE = LOCK_RELATIVE
CERTIFIED_CLOSE_REASONS = frozenset({"regular_close", "early_close"})
STALE_AFTER_SECONDS = 36 * 60 * 60
NO_PUBLICATION_REASON = "isolated_phase_no_publication_mode"
PORTFOLIO_HISTORY_NAME = "portfolio_history.csv"
CONFIRM_CUTOVER = "CREATE PERFORMANCE ACCOUNTING CUTOVER"
CONFIRM_SESSION_CLOSE = "APPEND CERTIFIED PERFORMANCE CLOSE"
LEVERAGE_MARGIN_CAPABLE_UNBORROWED = "MARGIN_CAPABLE_UNBORROWED"
LEVERAGE_MARGIN_IN_USE = "MARGIN_IN_USE"
LEVERAGE_SHORT_EXPOSURE = "SHORT_EXPOSURE"
LEVERAGE_UNDETERMINED = "UNDETERMINED"
REQUIRED_PUBLICATION_HASH_KEYS = ("account", "positions", "orders", "activities", "SPY", "QQQ")
SECRET_KEYS = frozenset(
    {
        "account_id",
        "account_number",
        "api_key",
        "secret_key",
        "apca_api_key_id",
        "apca_api_secret_key",
        "authorization",
        "token",
    }
)
ORDER_MUTATION_METHODS = (
    "submit_order",
    "cancel_order",
    "replace_order",
    "close_position",
    "place_order",
)

PROD = Path(r"C:\Users\akimw\triton").resolve()
ISO = Path(r"C:\Users\akimw\triton-bn1-performance-isolated").resolve()
BM6A = Path(r"C:\Users\akimw\triton-bm6a-ui-isolated").resolve()


class CollectorError(Exception):
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


class EvidenceSource(Protocol):
    def read_account_snapshot(self) -> Mapping[str, Any]: ...

    def read_fills_since(self, since_utc: Optional[str]) -> Sequence[Mapping[str, Any]]: ...

    def read_activities_since(self, since_utc: Optional[str]) -> Sequence[Mapping[str, Any]]: ...

    def read_orders(self) -> Sequence[Mapping[str, Any]]: ...

    def read_benchmark(self, symbol: str, session_date: str) -> Optional[Mapping[str, Any]]: ...

    def read_marks(self, broker_order_id: str) -> Optional[Mapping[str, Any]]: ...

    def source_hashes(self) -> Mapping[str, str]: ...


@dataclass(frozen=True)
class GetOnlyBrokerAdapter:
    """Injected GET-only adapter. No order-mutation capability."""

    adapter_mode: str = "GET_ONLY"

    @property
    def can_mutate_orders(self) -> bool:
        return False


@dataclass
class PublicationContext:
    mode: str = "PAPER_ONLY"
    live_execution_permitted: bool = False
    broker_adapter_mode: str = "GET_ONLY"
    can_mutate_orders: bool = False
    writers_idle: bool = True
    freshness_lock_present: bool = False
    broker_adapter: Any = field(default_factory=GetOnlyBrokerAdapter)
    fail_hook: Optional[Callable[[str, Path, bytes], None]] = None
    confirm: str = ""
    production_root: Optional[Path] = None
    expected_git_head: Optional[str] = None
    git_identity: Optional[Mapping[str, Any]] = None
    runtime_state: Optional[Mapping[str, Any]] = None
    authority_ttl_seconds: float = 120.0
    now: Optional[datetime] = None
    git_inspector: Optional[Callable[..., Mapping[str, Any]]] = None
    runtime_inspector: Optional[Callable[..., Mapping[str, Any]]] = None
    session_inspector: Optional[Callable[..., Mapping[str, Any]]] = None
    evidence_inspector: Optional[Callable[..., Mapping[str, Any]]] = None


def redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in SECRET_KEYS:
                continue
            out[key] = redact_payload(item)
        return out
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    if isinstance(value, str):
        return redact_account_ids(value)
    return value


def classify_leverage(snap: Mapping[str, Any]) -> str:
    cash = _dec(snap.get("cash"))
    short_mv = _dec(snap.get("short_market_value"))
    margin_debit = _dec(snap.get("margin_debit"))
    borrowed = _dec(snap.get("borrowed"))
    used_margin = _dec(snap.get("used_margin"))
    has_short = short_mv is not None and short_mv != 0
    position_qty_unknown = False
    for row in snap.get("positions") or []:
        qty = _dec(row.get("qty"))
        side = str(row.get("side") or "long").lower()
        if qty is None:
            position_qty_unknown = True
            continue
        if qty < 0 or side == "short":
            has_short = True
    if has_short:
        return LEVERAGE_SHORT_EXPOSURE
    if any(value is not None and value > 0 for value in (margin_debit, borrowed, used_margin)):
        return LEVERAGE_MARGIN_IN_USE
    if cash is not None and cash < 0:
        return LEVERAGE_MARGIN_IN_USE
    if cash is None or cash <= 0 or position_qty_unknown:
        return LEVERAGE_UNDETERMINED
    if snap.get("borrowed_exposure") not in (None, "", False, 0, "0"):
        if snap.get("borrowed_exposure") is True or _dec(snap.get("borrowed_exposure")) not in (
            None,
            Decimal("0"),
        ):
            return LEVERAGE_MARGIN_IN_USE
    return LEVERAGE_MARGIN_CAPABLE_UNBORROWED


class SessionGate(Protocol):
    def regular_session_state(self, now: Optional[datetime] = None) -> Mapping[str, Any]: ...


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _under(path: Path, root: Path) -> bool:
    text = str(_resolve(path)).lower()
    prefix = str(root).lower().rstrip("\\/") + os.sep
    return text.startswith(prefix)


def is_temp_path(path: Path) -> bool:
    resolved = str(_resolve(path)).lower()
    roots = [
        os.environ.get("TEMP", ""),
        os.environ.get("TMP", ""),
        r"c:\users\akimw\appdata\local\temp",
        r"c:\users\akimw\appdata\local\pytest-of-akimw",
    ]
    return any(
        resolved.startswith(str(root).lower().rstrip("\\/") + os.sep) for root in roots if root
    )


def assert_isolated_path(path: Path, *, role: str) -> Path:
    resolved = _resolve(Path(path))
    if resolved.name.lower() == PORTFOLIO_HISTORY_NAME:
        raise CollectorError("portfolio_history_rewrite_forbidden", {"path": str(resolved)})
    forbidden = (
        _under(resolved, PROD)
        or _under(resolved, BM6A)
        or _under(resolved, ISO / "data")
        or _under(resolved, PROD / "data")
    )
    if forbidden and not is_temp_path(resolved):
        raise CollectorError(
            "forbidden_production_path",
            {"role": role, "path": str(resolved)},
        )
    return resolved


def _dec(value: Any) -> Optional[Decimal]:
    if value in (None, "", UNAVAILABLE):
        return None
    return Decimal(str(value))


def _missing(row: Mapping[str, Any], keys: Sequence[str]) -> List[str]:
    return [key for key in keys if row.get(key) in (None, "")]


@dataclass
class MemoryEvidenceSource:
    """Injected test/freeze adapter. Never contacts a broker."""

    snapshot: Mapping[str, Any]
    fills: Sequence[Mapping[str, Any]] = field(default_factory=list)
    activities: Sequence[Mapping[str, Any]] = field(default_factory=list)
    orders: Sequence[Mapping[str, Any]] = field(default_factory=list)
    benchmarks: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    marks: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    hashes: Mapping[str, str] = field(default_factory=dict)
    hashes_next: Optional[Mapping[str, str]] = None
    _hash_calls: int = field(default=0, init=False, repr=False)

    def read_account_snapshot(self) -> Mapping[str, Any]:
        return dict(self.snapshot)

    def read_fills_since(self, since_utc: Optional[str]) -> Sequence[Mapping[str, Any]]:
        if not since_utc:
            return list(self.fills)
        return [row for row in self.fills if str(row.get("event_timestamp") or "") > since_utc]

    def read_activities_since(self, since_utc: Optional[str]) -> Sequence[Mapping[str, Any]]:
        if not since_utc:
            return list(self.activities)
        return [row for row in self.activities if str(row.get("event_timestamp") or "") > since_utc]

    def read_orders(self) -> Sequence[Mapping[str, Any]]:
        return list(self.orders)

    def read_benchmark(self, symbol: str, session_date: str) -> Optional[Mapping[str, Any]]:
        return self.benchmarks.get(f"{symbol}:{session_date}")

    def read_marks(self, broker_order_id: str) -> Optional[Mapping[str, Any]]:
        return self.marks.get(broker_order_id)

    def source_hashes(self) -> Mapping[str, str]:
        self._hash_calls += 1
        if self._hash_calls >= 2 and self.hashes_next is not None:
            return dict(self.hashes_next)
        return dict(self.hashes)


@dataclass
class InjectedSessionGate:
    payload: Mapping[str, Any]

    def regular_session_state(self, now: Optional[datetime] = None) -> Mapping[str, Any]:
        return dict(self.payload)


@dataclass
class FrozenBn1EvidenceSource:
    """Read-only BN1 freeze. TEMP evidence only. No broker I/O."""

    freeze_root: Path
    account: Mapping[str, Any]

    def __post_init__(self) -> None:
        assert_isolated_path(self.freeze_root, role="evidence_root")

    def _positions_path(self) -> Path:
        return Path(self.freeze_root) / "frozen" / "data" / "results" / "positions_snapshot.csv"

    def read_account_snapshot(self) -> Mapping[str, Any]:
        path = self._positions_path()
        assert_isolated_path(path, role="frozen_positions")
        positions = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                positions.append(
                    {
                        "symbol": row.get("symbol") or row.get("ticker"),
                        "qty": row.get("qty"),
                        "market_value": row.get("market_value"),
                        "average_entry_price": row.get("avg_entry_price"),
                        "unrealized_pl": row.get("unrealized_pl"),
                        "cost_basis": row.get("cost_basis"),
                        "side": row.get("side") or "long",
                    }
                )
        snap = dict(self.account)
        snap.setdefault("positions", positions)
        snap.setdefault("as_of_utc", snap.get("as_of_utc") or PROPOSED_CUTOVER_UTC)
        return snap

    def read_fills_since(self, since_utc: Optional[str]) -> Sequence[Mapping[str, Any]]:
        return []

    def read_activities_since(self, since_utc: Optional[str]) -> Sequence[Mapping[str, Any]]:
        return []

    def read_orders(self) -> Sequence[Mapping[str, Any]]:
        return []

    def read_benchmark(self, symbol: str, session_date: str) -> Optional[Mapping[str, Any]]:
        path = Path(self.freeze_root) / "frozen" / "data" / "raw" / f"{symbol}.csv"
        if not path.is_file():
            return None
        assert_isolated_path(path, role="frozen_benchmark")
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("date") or "") == session_date:
                    return {
                        "symbol": symbol,
                        "session_date": session_date,
                        "close": row.get("close"),
                        "adj_close": row.get("adj_close"),
                    }
        return None

    def read_marks(self, broker_order_id: str) -> Optional[Mapping[str, Any]]:
        return None

    def source_hashes(self) -> Mapping[str, str]:
        path = self._positions_path()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"positions_snapshot.csv": digest.hexdigest()}


class PerformanceEvidenceCollector:
    def __init__(
        self,
        *,
        events_path: Path,
        lock_path: Path,
        source: EvidenceSource,
        session_gate: SessionGate,
        persist: bool = False,
        allow_production_destination: bool = False,
        production_root: Optional[Path] = None,
    ) -> None:
        if allow_production_destination:
            if persist:
                raise CollectorError("production_persist_forbidden")
            if production_root is None:
                raise CollectorError("production_root_required")
            root = _resolve(Path(production_root))
            ev = _resolve(Path(events_path))
            lk = _resolve(Path(lock_path))
            if (
                ev.name.lower() == PORTFOLIO_HISTORY_NAME
                or lk.name.lower() == PORTFOLIO_HISTORY_NAME
            ):
                raise CollectorError("portfolio_history_rewrite_forbidden")
            if not (_under(ev, root) and _under(lk, root)):
                raise CollectorError(
                    "destination_not_under_production_root",
                    {"events_path": str(ev), "lock_path": str(lk), "production_root": str(root)},
                )
            self.events_path = ev
            self.lock_path = lk
            self.production_root = root
            self.allow_production_destination = True
        else:
            self.events_path = assert_isolated_path(Path(events_path), role="events_path")
            self.lock_path = assert_isolated_path(Path(lock_path), role="lock_path")
            self.production_root = None
            self.allow_production_destination = False
        self.source = source
        self.session_gate = session_gate
        self.persist = bool(persist)

    def _existing(self) -> List[Dict[str, Any]]:
        if not self.events_path.is_file():
            return []
        return load_events(self.events_path)

    def _since(self) -> Optional[str]:
        existing = self._existing()
        if not existing:
            return None
        return max(str(e.get("event_timestamp") or "") for e in existing)

    def _report(
        self,
        *,
        mode: str,
        events: Sequence[Mapping[str, Any]],
        missing: Sequence[str],
        diffs: Sequence[Mapping[str, Any]],
        blocked_reason: Optional[str],
        idempotency: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
        publishing: bool = False,
        source_hashes: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        checks_passed = blocked_reason is None
        if publishing and checks_passed:
            publication_permitted = True
            reason = None
            schema = PUBLICATION_SCHEMA
        else:
            publication_permitted = False
            reason = blocked_reason or NO_PUBLICATION_REASON
            schema = COLLECTOR_SCHEMA
        payload = {
            "schema_version": schema,
            "mode": mode,
            "writes_performed": False,
            "persist_requested": self.persist,
            "checks_passed": checks_passed,
            "publication_permitted": publication_permitted,
            "would_permit_publication": checks_passed,
            "blocked_reason": reason,
            "proposed_events": [
                {
                    "event_type": ev.get("event_type"),
                    "logical_key": ev.get("logical_key"),
                    "event_id": ev.get("event_id"),
                    "event_timestamp": ev.get("event_timestamp"),
                    "session_date": ev.get("session_date"),
                }
                for ev in events
            ],
            "logical_keys": [ev.get("logical_key") for ev in events],
            "source_hashes": dict(
                source_hashes if source_hashes is not None else self.source.source_hashes()
            ),
            "reconciliation_differences": list(diffs),
            "missing_evidence": list(missing),
            "duplicate_idempotency": dict(idempotency or {"appended": 0, "skipped_duplicate": 0}),
            "cutover_is_not_a_return": True,
            "cutover_is_not_profit_or_loss": True,
            "cutover_is_not_external_flow": True,
            "portfolio_history_rewritten": False,
            "account_model": ACCOUNT_MODEL,
            "canonical_events_path": CANONICAL_EVENTS_RELATIVE,
            "canonical_lock_path": CANONICAL_LOCK_RELATIVE,
        }
        if extra:
            payload.update(extra)
        return redact_payload(payload)

    def _maybe_persist(self, events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not self.persist:
            return {"appended": 0, "skipped_duplicate": 0, "writes_performed": False}
        result = append_events(events, events_path=self.events_path, lock_path=self.lock_path)
        result["writes_performed"] = result.get("appended", 0) > 0
        return result

    def _validate_snapshot(self, snap: Mapping[str, Any], *, where: str) -> List[Dict[str, Any]]:
        diffs: List[Dict[str, Any]] = []
        required = ("cash", "market_value", "equity", "as_of_utc")
        missing_fields = _missing(snap, required)
        if missing_fields:
            raise CollectorError("incomplete_snapshot", {"missing": missing_fields, "where": where})
        residual = identity_residual(snap["cash"], snap["market_value"], snap["equity"])
        if residual > IDENTITY_TOLERANCE:
            diffs.append(
                {
                    "kind": "cash_mv_equity_mismatch",
                    "residual": format(residual, "f"),
                    "tolerance": format(IDENTITY_TOLERANCE, "f"),
                }
            )
            raise CollectorError("reconciliation_failure", {"residual": format(residual, "f")})
        positions = list(snap.get("positions") or [])
        if not positions:
            raise CollectorError("incomplete_positions", {"where": where, "reason": "empty"})
        for row in positions:
            absent = _missing(row, ("symbol", "qty", "market_value", "average_entry_price"))
            if absent:
                raise CollectorError(
                    "incomplete_positions", {"symbol": row.get("symbol"), "missing": absent}
                )
        leverage = classify_leverage(snap)
        if leverage != LEVERAGE_MARGIN_CAPABLE_UNBORROWED:
            raise CollectorError(leverage, {"where": where, "leverage_class": leverage})
        return diffs

    def _position_bodies(self, snap: Mapping[str, Any]) -> List[Dict[str, Any]]:
        bodies = []
        for row in snap.get("positions") or []:
            upl = row.get("unrealized_pl")
            bodies.append(
                {
                    "symbol": row.get("symbol"),
                    "qty": row.get("qty"),
                    "market_value": row.get("market_value"),
                    "average_entry_price": row.get("average_entry_price"),
                    "unrealized_pl": upl if upl not in (None, "") else UNAVAILABLE,
                    "cost_basis": row.get("cost_basis"),
                    "as_of_utc": snap.get("as_of_utc"),
                }
            )
        return bodies

    def safe_propose_cutover(self) -> Dict[str, Any]:
        try:
            return self.propose_cutover()
        except (CollectorError, PerformanceAccountingError) as exc:
            return _blocked_report("dry-run-cutover", exc)

    def propose_cutover(self) -> Dict[str, Any]:
        snap = self.source.read_account_snapshot()
        diffs = self._validate_snapshot(snap, where="cutover")
        positions = self._position_bodies(snap)
        body = {
            "as_of_utc": snap["as_of_utc"],
            "publication_status": "PROPOSED_REQUIRES_FRESH_SNAPSHOT",
            "cash": snap["cash"],
            "equity": snap["equity"],
            "market_value": snap["market_value"],
            "buying_power": (
                snap.get("buying_power")
                if snap.get("buying_power") not in (None, "")
                else UNAVAILABLE
            ),
            "positions": [
                {
                    "symbol": p["symbol"],
                    "qty": p["qty"],
                    "average_entry_price": p["average_entry_price"],
                    "cost_basis": p.get("cost_basis"),
                    "market_value": p["market_value"],
                    "unrealized_pl": p["unrealized_pl"],
                }
                for p in positions
            ],
            "position_count": len(positions),
            "not_a_deposit": True,
            "not_a_withdrawal": True,
            "not_profit_or_loss": True,
            "not_a_return": True,
            "account_structure": "long_only_paper",
            "historical_return": None,
            "source_hashes": dict(self.source.source_hashes()),
        }
        event = build_event(
            EVENT_CUTOVER,
            event_timestamp=snap["as_of_utc"],
            session_date=str(snap.get("session_date") or session_date_et(snap["as_of_utc"])),
            source="phase_198t_bn2b_passive_collector",
            body=body,
        )
        idem = self._maybe_persist([event])
        report = self._report(
            mode="dry-run-cutover",
            events=[event],
            missing=[],
            diffs=diffs,
            blocked_reason=None,
            idempotency=idem,
            extra={"buying_power": body["buying_power"], "long_only": True},
        )
        report["writes_performed"] = bool(idem.get("writes_performed"))
        return report

    def _assert_certified_close(self, now: Optional[datetime]) -> Mapping[str, Any]:
        state = self.session_gate.regular_session_state(now)
        session_state = str(state.get("session_state") or "")
        reason = str(state.get("reason") or "")
        if session_state == "OPEN":
            raise CollectorError("session_open", {"state": state})
        if reason in {"pre_open", "premarket"}:
            raise CollectorError("premarket", {"state": state})
        if "holiday" in reason:
            raise CollectorError("holiday", {"state": state})
        if session_state != "CLOSED" or reason not in CERTIFIED_CLOSE_REASONS:
            raise CollectorError("session_not_certified_closed", {"state": state})
        if not state.get("session_date"):
            raise CollectorError("incomplete_session", {"state": state})
        return state

    def _assert_fresh(self, snap: Mapping[str, Any], session: Mapping[str, Any]) -> None:
        as_of = utc_iso(snap["as_of_utc"])
        session_date = str(session.get("session_date"))
        if str(snap.get("session_date") or session_date_et(as_of)) != session_date:
            raise CollectorError("stale_snapshot", {"as_of": as_of, "session_date": session_date})
        close_et = session.get("session_close_et")
        if close_et:
            if datetime.fromisoformat(as_of.replace("Z", "+00:00")) < datetime.fromisoformat(
                str(close_et)
            ).astimezone(timezone.utc):
                raise CollectorError("stale_snapshot", {"reason": "snapshot_before_session_close"})
        age = (
            datetime.now(timezone.utc) - datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        ).total_seconds()
        if age > STALE_AFTER_SECONDS:
            raise CollectorError("stale_snapshot", {"age_seconds": age})

    def _fill_events(
        self, fills: Sequence[Mapping[str, Any]], missing: List[str]
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for row in fills:
            qty = _dec(row.get("qty"))
            price = _dec(row.get("price"))
            order_id = str(row.get("broker_order_id") or "").strip()
            if qty is None or price is None or qty <= 0 or price <= 0 or not order_id:
                raise CollectorError("reconciliation_failure", {"fill": dict(row)})
            side = str(row.get("side") or "").upper()
            event_type = EVENT_FILL_BUY if side == "BUY" else EVENT_FILL_SELL
            events.append(
                build_event(
                    event_type,
                    event_timestamp=row.get("event_timestamp"),
                    source="paper_order_ledger",
                    body={
                        "symbol": row.get("symbol"),
                        "qty": row.get("qty"),
                        "price": row.get("price"),
                        "broker_order_id": order_id,
                        "broker_fill_id": row.get("broker_fill_id") or f"{order_id}-fill",
                    },
                )
            )
            marks = self.source.read_marks(order_id)
            if not marks:
                missing.append(f"marks:{order_id}")
                continue
            mark_body = {"broker_order_id": order_id}
            for key in ("decision_price", "submit_mark", "bid", "ask"):
                mark_body[key] = marks.get(key) if marks.get(key) not in (None, "") else UNAVAILABLE
            if all(
                mark_body[key] == UNAVAILABLE
                for key in ("decision_price", "submit_mark", "bid", "ask")
            ):
                missing.append(f"marks:{order_id}")
                continue
            events.append(
                build_event(
                    EVENT_PRICE_MARKS,
                    event_timestamp=row.get("event_timestamp"),
                    source="paper_order_ledger",
                    body=mark_body,
                )
            )
        return events

    def _activity_events(self, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for row in rows:
            kind = str(row.get("event_type") or "").upper()
            ts = row.get("event_timestamp")
            if kind == EVENT_FEE:
                component = str(row.get("fee_component") or "OTHER").upper()
                if component not in FEE_COMPONENTS:
                    component = "OTHER"
                events.append(
                    build_event(
                        EVENT_FEE,
                        event_timestamp=ts,
                        source="broker_activities",
                        body={
                            "amount": row.get("amount"),
                            "activity_id": row.get("activity_id"),
                            "fee_component": component,
                            "allocates_to_event_id": row.get("allocates_to_event_id"),
                            "allocates_to_order_id": row.get("allocates_to_order_id"),
                        },
                    )
                )
            elif kind == EVENT_DIVIDEND:
                events.append(
                    build_event(
                        EVENT_DIVIDEND,
                        event_timestamp=ts,
                        source="broker_activities",
                        body={
                            "amount": row.get("amount"),
                            "activity_id": row.get("activity_id"),
                            "symbol": row.get("symbol"),
                        },
                    )
                )
            elif kind in {EVENT_CASH_DEPOSIT, EVENT_CASH_WITHDRAWAL, EVENT_CASH_ADJUSTMENT}:
                events.append(
                    build_external_flow(
                        kind,
                        amount=row.get("amount"),
                        activity_id=str(row.get("activity_id")),
                        event_timestamp=ts,
                        source="broker_activities",
                        direction=row.get("direction"),
                        evidence_category=row.get("evidence_category"),
                    )
                )
        return events

    def safe_collect_session_close(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        try:
            return self.collect_session_close(now=now)
        except (CollectorError, PerformanceAccountingError) as exc:
            return _blocked_report("dry-run-session-close", exc)

    def collect_session_close(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        session = self._assert_certified_close(now)
        snap = self.source.read_account_snapshot()
        diffs = self._validate_snapshot(snap, where="session_close")
        self._assert_fresh(snap, session)
        session_date = str(session["session_date"])
        close_kind = str(session["reason"])
        missing: List[str] = []
        events: List[Dict[str, Any]] = []
        positions = self._position_bodies(snap)
        events.append(
            build_event(
                EVENT_SESSION_CLOSE,
                event_timestamp=snap["as_of_utc"],
                session_date=session_date,
                source="phase_198t_bn2b_passive_collector",
                body={
                    "session_date": session_date,
                    "close_kind": close_kind,
                    "cash": snap["cash"],
                    "market_value": snap["market_value"],
                    "equity": snap["equity"],
                    "buying_power": (
                        snap.get("buying_power")
                        if snap.get("buying_power") not in (None, "")
                        else UNAVAILABLE
                    ),
                    "positions": positions,
                    "account_structure": "long_only_paper",
                    "source_hashes": dict(self.source.source_hashes()),
                },
            )
        )
        for pos in positions:
            events.append(
                build_event(
                    EVENT_POSITION_SNAPSHOT,
                    event_timestamp=snap["as_of_utc"],
                    session_date=session_date,
                    source="phase_198t_bn2b_passive_collector",
                    body=pos,
                )
            )
        since = self._since()
        events.extend(self._fill_events(self.source.read_fills_since(since), missing))
        events.extend(self._activity_events(self.source.read_activities_since(since)))
        for symbol in ("SPY", "QQQ"):
            bar = self.source.read_benchmark(symbol, session_date)
            if not bar or (bar.get("adj_close") in (None, "") and bar.get("close") in (None, "")):
                missing.append(f"benchmark:{symbol}:{session_date}")
                continue
            events.append(
                build_event(
                    EVENT_BENCHMARK_CLOSE,
                    event_timestamp=snap["as_of_utc"],
                    session_date=session_date,
                    source="local_bars",
                    body={
                        "symbol": symbol,
                        "session_date": session_date,
                        "close": bar.get("close"),
                        "adj_close": bar.get("adj_close"),
                    },
                )
            )
        idem = self._maybe_persist(events)
        report = self._report(
            mode="dry-run-session-close",
            events=events,
            missing=missing,
            diffs=diffs,
            blocked_reason=None,
            idempotency=idem,
            extra={
                "close_kind": close_kind,
                "session_date": session_date,
                "buying_power": snap.get("buying_power") or UNAVAILABLE,
                "long_only": True,
            },
        )
        report["writes_performed"] = bool(idem.get("writes_performed"))
        return report

    def validate_ledger(self) -> Dict[str, Any]:
        assert_isolated_path(self.events_path, role="events_path")
        events = self._existing()
        state = (
            reduce_events(events)
            if events
            else reduce_ledger(events_path=self.events_path, lock_path=self.lock_path)
        )
        return self._report(
            mode="validate-ledger",
            events=events,
            missing=[],
            diffs=[],
            blocked_reason=None,
            extra={
                "reduced": {
                    "open_position_count": state.get("open_position_count"),
                    "cash": state.get("cash"),
                }
            },
        )

    def _assert_confirmation(self, confirm: str, expected: str) -> None:
        if str(confirm or "") != expected:
            raise CollectorError(
                "wrong_confirmation_phrase",
                {"expected": expected},
            )

    def _assert_publication_gates(self, context: PublicationContext) -> None:
        if str(context.mode or "") != "PAPER_ONLY" or context.live_execution_permitted:
            raise CollectorError(
                "production_or_live_mode",
                {
                    "mode": context.mode,
                    "live_execution_permitted": context.live_execution_permitted,
                },
            )
        if str(context.broker_adapter_mode or "") != "GET_ONLY":
            raise CollectorError(
                "broker_not_get_only", {"broker_adapter_mode": context.broker_adapter_mode}
            )
        if context.can_mutate_orders:
            raise CollectorError("order_mutation_capability")
        adapter = context.broker_adapter
        if adapter is not None:
            if getattr(adapter, "can_mutate_orders", False):
                raise CollectorError("order_mutation_capability")
            if any(callable(getattr(adapter, name, None)) for name in ORDER_MUTATION_METHODS):
                raise CollectorError("order_mutation_capability")
            adapter_mode = getattr(adapter, "adapter_mode", context.broker_adapter_mode)
            if str(adapter_mode or "") != "GET_ONLY":
                raise CollectorError("broker_not_get_only", {"adapter_mode": adapter_mode})
        if not context.writers_idle:
            raise CollectorError("writers_active")
        if context.freshness_lock_present:
            raise CollectorError("freshness_lock_present")

    def _destination_is_production(self, context: PublicationContext) -> bool:
        if self.allow_production_destination:
            return True
        if context.production_root is not None and _under(
            self.events_path, Path(context.production_root)
        ):
            return True
        return is_real_production_path(self.events_path) or is_real_production_path(self.lock_path)

    def _assert_xnys_closed(self, session: Mapping[str, Any]) -> None:
        calendar = str(session.get("calendar") or "").upper()
        if calendar != "XNYS":
            raise CollectorError("session_not_xnys", {"calendar": session.get("calendar")})

    def _assert_production_publication_gates(
        self,
        context: PublicationContext,
        session: Mapping[str, Any],
    ) -> Path:
        root = context.production_root or self.production_root
        if root is None:
            raise CollectorError("production_root_required")
        root = _resolve(Path(root))
        if not (_under(self.events_path, root) and _under(self.lock_path, root)):
            raise CollectorError(
                "destination_not_under_production_root",
                {"production_root": str(root)},
            )
        expected_head = str(context.expected_git_head or "")
        if not expected_head:
            raise CollectorError("expected_git_head_required")
        identity = (
            dict(context.git_identity)
            if context.git_identity is not None
            else inspect_git_identity(root)
        )
        if str(identity.get("head") or "") != expected_head:
            raise CollectorError(
                "git_head_mismatch",
                {"expected": expected_head, "actual": identity.get("head")},
            )
        if str(identity.get("branch") or "") != REQUIRED_BRANCH:
            raise CollectorError(
                "git_branch_not_main",
                {"branch": identity.get("branch")},
            )
        if identity.get("index_empty") is not True:
            raise CollectorError("git_index_not_empty")
        state = (
            dict(context.runtime_state)
            if context.runtime_state is not None
            else inspect_runtime_state(root)
        )
        if not state.get("runtime_stopped"):
            raise CollectorError("runtime_not_stopped")
        if not state.get("watchdog_stopped"):
            raise CollectorError("watchdog_not_stopped")
        if not state.get("streamlit_stopped"):
            raise CollectorError("streamlit_not_stopped")
        if not state.get("ports_closed"):
            raise CollectorError("production_ports_open")
        writers_idle = state.get("freshness_writer_idle", context.writers_idle)
        if not writers_idle or not context.writers_idle:
            raise CollectorError("writers_active")
        lock_present = state.get("freshness_lock_present", context.freshness_lock_present)
        if lock_present or context.freshness_lock_present:
            raise CollectorError("freshness_lock_present")
        self._assert_xnys_closed(session)
        return root

    def _mint_authority_if_required(
        self,
        context: PublicationContext,
        session: Mapping[str, Any],
        *,
        events: Sequence[Mapping[str, Any]],
        mode: str,
        command: str,
        confirm: str,
        evidence_hashes: Mapping[str, str],
    ) -> tuple[Any, Optional[Path], Optional[str]]:
        if not self._destination_is_production(context):
            return None, None, None
        root = self._assert_production_publication_gates(context, session)
        if mode == "create":
            prior = b""
            existing: List[Dict[str, Any]] = []
        elif self.events_path.is_file():
            prior = self.events_path.read_bytes()
            existing = self._existing()
        else:
            prior = b""
            existing = []
        planned, _skipped = plan_ledger_writes(events, existing)
        payload = prior + encode_ledger_suffix(planned)
        identity = (
            dict(context.git_identity)
            if context.git_identity is not None
            else inspect_git_identity(root)
        )
        runtime = (
            dict(context.runtime_state)
            if context.runtime_state is not None
            else inspect_runtime_state(root)
        )
        authority = mint_publication_authority(
            resolved_root=root,
            expected_git_head=str(context.expected_git_head),
            events_path=self.events_path,
            lock_path=self.lock_path,
            command=command,
            publication_mode=mode,
            confirm_phrase=confirm,
            payload_sha256=sha256_bytes(payload),
            existing_ledger_sha256=sha256_bytes(prior) if prior else EMPTY_LEDGER_SHA256,
            existing_ledger_nbytes=len(prior),
            required_prefix=prior if mode == "append" else b"",
            account_mode=str(context.mode or "PAPER_ONLY"),
            live_execution_permitted=bool(context.live_execution_permitted),
            git_identity=identity,
            runtime_state=runtime,
            session=dict(session),
            evidence_hashes=dict(evidence_hashes),
            git_inspector=context.git_inspector,
            runtime_inspector=context.runtime_inspector,
            session_inspector=context.session_inspector,
            evidence_inspector=context.evidence_inspector,
            ttl_seconds=context.authority_ttl_seconds,
            now=context.now,
        )
        return authority, root, str(context.expected_git_head)

    def _stable_publication_hashes(self) -> Mapping[str, str]:
        first = dict(self.source.source_hashes())
        second = dict(self.source.source_hashes())
        if first != second:
            raise CollectorError("changing_evidence", {"first": first, "second": second})
        aliases = dict(first)
        if "positions" not in aliases and aliases.get("positions_snapshot.csv"):
            aliases["positions"] = aliases["positions_snapshot.csv"]
        missing = [key for key in REQUIRED_PUBLICATION_HASH_KEYS if not aliases.get(key)]
        if missing:
            raise CollectorError("incomplete_evidence", {"missing_hashes": missing})
        return first

    def _assert_complete_publication_evidence(self, session_date: str) -> Mapping[str, str]:
        hashes = self._stable_publication_hashes()
        self.source.read_orders()
        self.source.read_activities_since(None)
        self.source.read_account_snapshot()
        for symbol in ("SPY", "QQQ"):
            bar = self.source.read_benchmark(symbol, session_date)
            if not bar or (bar.get("adj_close") in (None, "") and bar.get("close") in (None, "")):
                raise CollectorError(
                    "incomplete_evidence", {"missing": f"benchmark:{symbol}:{session_date}"}
                )
        return hashes

    def _published_cutover_event(
        self,
        snap: Mapping[str, Any],
        *,
        leverage_class: str,
        source_hashes: Mapping[str, str],
    ) -> Dict[str, Any]:
        positions = self._position_bodies(snap)
        body = {
            "as_of_utc": snap["as_of_utc"],
            "publication_status": "PUBLISHED_OPENING_CHECKPOINT",
            "opening_checkpoint": True,
            "opening_basis_only": True,
            "measurement_boundary": "earliest_future",
            "cash": snap["cash"],
            "equity": snap["equity"],
            "market_value": snap["market_value"],
            "buying_power": (
                snap.get("buying_power")
                if snap.get("buying_power") not in (None, "")
                else UNAVAILABLE
            ),
            "positions": [
                {
                    "symbol": p["symbol"],
                    "qty": p["qty"],
                    "average_entry_price": p["average_entry_price"],
                    "cost_basis": p.get("cost_basis"),
                    "market_value": p["market_value"],
                    "unrealized_pl": p["unrealized_pl"],
                }
                for p in positions
            ],
            "position_count": len(positions),
            "not_a_deposit": True,
            "not_a_withdrawal": True,
            "not_profit_or_loss": True,
            "not_a_return": True,
            "historical_return": None,
            "realized_pl": None,
            "deposit": None,
            "withdrawal": None,
            "account_structure": "long_only_paper",
            "leverage_class": leverage_class,
            "source_hashes": dict(source_hashes),
        }
        return build_event(
            EVENT_CUTOVER,
            event_timestamp=snap["as_of_utc"],
            session_date=str(snap.get("session_date") or session_date_et(snap["as_of_utc"])),
            source="phase_198t_bn2c_protected_publication",
            body=body,
        )

    def safe_publish_cutover(
        self,
        *,
        confirm: str,
        context: Optional[PublicationContext] = None,
    ) -> Dict[str, Any]:
        try:
            return self.publish_cutover(confirm=confirm, context=context)
        except (CollectorError, PerformanceAccountingError) as exc:
            return _blocked_report("publish-cutover", exc)

    def publish_cutover(
        self,
        *,
        confirm: str,
        context: Optional[PublicationContext] = None,
    ) -> Dict[str, Any]:
        ctx = context or PublicationContext()
        self._assert_confirmation(confirm, CONFIRM_CUTOVER)
        self._assert_publication_gates(ctx)
        session = self._assert_certified_close(None)
        snap = self.source.read_account_snapshot()
        diffs = self._validate_snapshot(snap, where="publish-cutover")
        self._assert_fresh(snap, session)
        session_date = str(session.get("session_date") or snap.get("session_date") or "")
        hashes = self._assert_complete_publication_evidence(session_date)
        leverage = classify_leverage(snap)
        event = self._published_cutover_event(snap, leverage_class=leverage, source_hashes=hashes)
        authority, prod_root, expected_head = self._mint_authority_if_required(
            ctx,
            session,
            events=[event],
            mode="create",
            command=COMMAND_CUTOVER,
            confirm=confirm,
            evidence_hashes=hashes,
        )
        result = publish_ledger_atomic(
            [event],
            events_path=self.events_path,
            lock_path=self.lock_path,
            mode="create",
            fail_hook=ctx.fail_hook,
            publication_authority=authority,
            production_root=prod_root,
            expected_git_head=expected_head,
            now=ctx.now,
            publication_command=COMMAND_CUTOVER,
            confirm=confirm,
        )
        report = self._report(
            mode="publish-cutover",
            events=[event],
            missing=[],
            diffs=diffs,
            blocked_reason=None,
            idempotency=result,
            publishing=True,
            source_hashes=hashes,
            extra={
                "publication_schema": (
                    PRODUCTION_PUBLICATION_SCHEMA if authority is not None else PUBLICATION_SCHEMA
                ),
                "leverage_class": leverage,
                "opening_checkpoint": True,
                "opening_basis_only": True,
                "creates_historical_return": False,
                "creates_realized_pl": False,
                "creates_deposit": False,
                "creates_withdrawal": False,
                "measurement_boundary": "earliest_future",
                "payload_hash": result.get("payload_hash"),
                "buying_power": snap.get("buying_power") or UNAVAILABLE,
                "publication_authority_issued": authority is not None,
            },
        )
        report["writes_performed"] = bool(result.get("writes_performed"))
        return report

    def safe_publish_session_close(
        self,
        *,
        confirm: str,
        context: Optional[PublicationContext] = None,
        now: Optional[datetime] = None,
        plan_only: bool = False,
    ) -> Dict[str, Any]:
        try:
            return self.publish_session_close(
                confirm=confirm, context=context, now=now, plan_only=plan_only
            )
        except (CollectorError, PerformanceAccountingError) as exc:
            return _blocked_report("publish-session-close", exc)

    def publish_session_close(
        self,
        *,
        confirm: str,
        context: Optional[PublicationContext] = None,
        now: Optional[datetime] = None,
        plan_only: bool = False,
    ) -> Dict[str, Any]:
        ctx = context or PublicationContext()
        self._assert_confirmation(confirm, CONFIRM_SESSION_CLOSE)
        self._assert_publication_gates(ctx)
        session = self._assert_certified_close(now)
        snap = self.source.read_account_snapshot()
        diffs = self._validate_snapshot(snap, where="publish-session-close")
        self._assert_fresh(snap, session)
        session_date = str(session["session_date"])
        hashes = self._assert_complete_publication_evidence(session_date)
        leverage = classify_leverage(snap)
        events: List[Dict[str, Any]] = []
        missing: List[str] = []
        positions = self._position_bodies(snap)
        close_kind = str(session["reason"])
        events.append(
            build_event(
                EVENT_SESSION_CLOSE,
                event_timestamp=snap["as_of_utc"],
                session_date=session_date,
                source="phase_198t_bn2c_protected_publication",
                body={
                    "session_date": session_date,
                    "close_kind": close_kind,
                    "cash": snap["cash"],
                    "market_value": snap["market_value"],
                    "equity": snap["equity"],
                    "buying_power": (
                        snap.get("buying_power")
                        if snap.get("buying_power") not in (None, "")
                        else UNAVAILABLE
                    ),
                    "positions": positions,
                    "account_structure": "long_only_paper",
                    "leverage_class": leverage,
                    "source_hashes": dict(hashes),
                },
            )
        )
        for pos in positions:
            events.append(
                build_event(
                    EVENT_POSITION_SNAPSHOT,
                    event_timestamp=snap["as_of_utc"],
                    session_date=session_date,
                    source="phase_198t_bn2c_protected_publication",
                    body=pos,
                )
            )
        since = self._since()
        events.extend(self._fill_events(self.source.read_fills_since(since), missing))
        events.extend(self._activity_events(self.source.read_activities_since(since)))
        for symbol in ("SPY", "QQQ"):
            bar = self.source.read_benchmark(symbol, session_date)
            events.append(
                build_event(
                    EVENT_BENCHMARK_CLOSE,
                    event_timestamp=snap["as_of_utc"],
                    session_date=session_date,
                    source="local_bars",
                    body={
                        "symbol": symbol,
                        "session_date": session_date,
                        "close": bar.get("close") if bar else UNAVAILABLE,
                        "adj_close": bar.get("adj_close") if bar else UNAVAILABLE,
                    },
                )
            )
        if missing:
            raise CollectorError("incomplete_evidence", {"missing": list(missing)})
        existing_keys = {str(ev.get("logical_key")) for ev in self._existing()}
        close_key = f"session_close:{session_date}"
        planned = sum(1 for ev in events if str(ev.get("logical_key")) not in existing_keys)
        if plan_only:
            report = self._report(
                mode="plan-session-close",
                events=events,
                missing=[],
                diffs=diffs,
                blocked_reason=None,
                idempotency={"appended": 0, "skipped_duplicate": 0},
                publishing=False,
                source_hashes=hashes,
                extra={
                    "leverage_class": leverage,
                    "close_kind": close_kind,
                    "session_date": session_date,
                    "planned_append_count": planned,
                    "plan_only": True,
                    "publication_schema": PUBLICATION_SCHEMA,
                },
            )
            report["writes_performed"] = False
            report["would_permit_publication"] = True
            return report
        if close_key in existing_keys:
            report = self._report(
                mode="publish-session-close",
                events=events,
                missing=[],
                diffs=diffs,
                blocked_reason="duplicate_session_close",
                idempotency={"appended": 0, "skipped_duplicate": 1},
                publishing=False,
                source_hashes=hashes,
                extra={
                    "leverage_class": leverage,
                    "close_kind": close_kind,
                    "session_date": session_date,
                    "planned_append_count": 0,
                    "writes_performed": False,
                },
            )
            report["writes_performed"] = False
            return report
        authority, prod_root, expected_head = self._mint_authority_if_required(
            ctx,
            session,
            events=events,
            mode="append",
            command=COMMAND_SESSION_CLOSE,
            confirm=confirm,
            evidence_hashes=hashes,
        )
        result = publish_ledger_atomic(
            events,
            events_path=self.events_path,
            lock_path=self.lock_path,
            mode="append",
            fail_hook=ctx.fail_hook,
            publication_authority=authority,
            production_root=prod_root,
            expected_git_head=expected_head,
            now=ctx.now,
            publication_command=COMMAND_SESSION_CLOSE,
            confirm=confirm,
        )
        report = self._report(
            mode="publish-session-close",
            events=events,
            missing=[],
            diffs=diffs,
            blocked_reason=None,
            idempotency=result,
            publishing=True,
            source_hashes=hashes,
            extra={
                "publication_schema": (
                    PRODUCTION_PUBLICATION_SCHEMA if authority is not None else PUBLICATION_SCHEMA
                ),
                "leverage_class": leverage,
                "close_kind": close_kind,
                "session_date": session_date,
                "payload_hash": result.get("payload_hash"),
                "publication_authority_issued": authority is not None,
            },
        )
        report["writes_performed"] = bool(result.get("writes_performed"))
        return report


def observe_without_affecting_execution(
    collect: Callable[[], Any],
    execute: Callable[[], Any],
) -> Dict[str, Any]:
    """Collection failures never change the execution result."""
    collection: Dict[str, Any]
    try:
        raw = collect()
        collection = raw if isinstance(raw, dict) else {"ok": True, "result": raw}
    except Exception as exc:
        collection = {
            "ok": False,
            "blocked_reason": "collector_failure",
            "error_type": type(exc).__name__,
            "error": str(getattr(exc, "reason", exc)),
        }
    execution = execute()
    return {"execution": execution, "collection": collection}


def collect_safe(collector_call: Callable[[], Any], execution_result: Any) -> Any:
    observe_without_affecting_execution(collector_call, lambda: execution_result)
    return execution_result


def _blocked_report(mode: str, exc: Exception) -> Dict[str, Any]:
    reason = getattr(exc, "reason", type(exc).__name__)
    details = getattr(exc, "details", {})
    return redact_payload(
        {
            "schema_version": COLLECTOR_SCHEMA,
            "mode": mode,
            "writes_performed": False,
            "persist_requested": False,
            "checks_passed": False,
            "publication_permitted": False,
            "would_permit_publication": False,
            "blocked_reason": reason,
            "details": details,
            "proposed_events": [],
            "logical_keys": [],
            "source_hashes": {},
            "reconciliation_differences": [],
            "missing_evidence": [],
            "duplicate_idempotency": {"appended": 0, "skipped_duplicate": 0},
            "cutover_is_not_a_return": True,
            "portfolio_history_rewritten": False,
        }
    )


def _cli_mode_name(args: argparse.Namespace) -> str:
    if args.validate_ledger:
        return "validate-ledger"
    if args.dry_run_cutover:
        return "dry-run-cutover"
    if args.dry_run_session_close:
        return "dry-run-session-close"
    if args.publish_cutover:
        return "publish-cutover"
    if args.publish_session_close:
        return "publish-session-close"
    return "unknown"


def _cli_publication_context(args: argparse.Namespace) -> PublicationContext:
    return PublicationContext(
        mode=str(args.account_mode or "PAPER_ONLY"),
        live_execution_permitted=bool(args.live_execution_permitted),
        broker_adapter_mode="GET_ONLY",
        can_mutate_orders=False,
        writers_idle=True,
        freshness_lock_present=False,
        production_root=Path(args.production_root) if args.production_root else None,
        expected_git_head=str(args.expected_git_head or "") or None,
    )


def _cli_memory_source(args: argparse.Namespace) -> MemoryEvidenceSource:
    if not args.snapshot_json:
        raise CollectorError("snapshot_json_required")
    snapshot = json.loads(Path(args.snapshot_json).read_text(encoding="utf-8"))
    hashes = (
        json.loads(Path(args.hashes_json).read_text(encoding="utf-8"))
        if args.hashes_json
        else {
            "account": "cli-account",
            "positions": "cli-positions",
            "orders": "cli-orders",
            "activities": "cli-activities",
            "SPY": "cli-spy",
            "QQQ": "cli-qqq",
        }
    )
    benchmarks = (
        json.loads(Path(args.benchmarks_json).read_text(encoding="utf-8"))
        if args.benchmarks_json
        else {}
    )
    return MemoryEvidenceSource(snapshot=snapshot, hashes=hashes, benchmarks=benchmarks)


def run_cli(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Isolated BN2B/BN2C performance-evidence collector"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run-cutover", action="store_true")
    mode.add_argument("--dry-run-session-close", action="store_true")
    mode.add_argument("--validate-ledger", action="store_true")
    mode.add_argument("--publish-cutover", action="store_true")
    mode.add_argument("--publish-session-close", action="store_true")
    parser.add_argument("--events-path", required=True)
    parser.add_argument("--lock-path", required=True)
    parser.add_argument("--evidence-root")
    parser.add_argument("--account-json")
    parser.add_argument("--snapshot-json")
    parser.add_argument("--session-json")
    parser.add_argument("--hashes-json")
    parser.add_argument("--benchmarks-json")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--account-mode", default="PAPER_ONLY")
    parser.add_argument("--live-execution-permitted", action="store_true")
    parser.add_argument("--production-root")
    parser.add_argument("--expected-git-head", default="")
    args = parser.parse_args(argv)

    events_path = Path(args.events_path)
    lock_path = Path(args.lock_path)
    production_publish = bool(
        (args.publish_cutover or args.publish_session_close) and args.production_root
    )
    try:
        if not production_publish:
            assert_isolated_path(events_path, role="events_path")
            assert_isolated_path(lock_path, role="lock_path")
        if args.validate_ledger:
            collector = PerformanceEvidenceCollector(
                events_path=events_path,
                lock_path=lock_path,
                source=MemoryEvidenceSource(
                    snapshot={
                        "as_of_utc": PROPOSED_CUTOVER_UTC,
                        "cash": "0",
                        "market_value": "0",
                        "equity": "0",
                        "positions": [
                            {
                                "symbol": "X",
                                "qty": "1",
                                "market_value": "0",
                                "average_entry_price": "1",
                            }
                        ],
                    }
                ),
                session_gate=InjectedSessionGate(
                    {
                        "session_state": "CLOSED",
                        "reason": "regular_close",
                        "session_date": "2026-09-09",
                    }
                ),
                persist=False,
            )
            return collector.validate_ledger()
        if args.dry_run_cutover:
            if args.evidence_root:
                account = (
                    json.loads(Path(args.account_json).read_text(encoding="utf-8"))
                    if args.account_json
                    else {
                        "as_of_utc": PROPOSED_CUTOVER_UTC,
                        "cash": format(PROPOSED_CUTOVER_CASH, "f"),
                        "equity": format(PROPOSED_CUTOVER_EQUITY, "f"),
                        "market_value": format(
                            q_cash(PROPOSED_CUTOVER_EQUITY - PROPOSED_CUTOVER_CASH), "f"
                        ),
                        "buying_power": "158021.63",
                        "session_date": "2026-09-09",
                        "account_structure": "long_only_paper",
                    }
                )
                source: EvidenceSource = FrozenBn1EvidenceSource(Path(args.evidence_root), account)
            else:
                source = _cli_memory_source(args)
            collector = PerformanceEvidenceCollector(
                events_path=events_path,
                lock_path=lock_path,
                source=source,
                session_gate=InjectedSessionGate(
                    {
                        "session_state": "CLOSED",
                        "reason": "regular_close",
                        "session_date": "2026-09-09",
                    }
                ),
                persist=False,
            )
            return collector.propose_cutover()
        if args.publish_cutover or args.publish_session_close:
            if not args.snapshot_json or not args.session_json:
                raise CollectorError("session_and_snapshot_json_required")
            source = _cli_memory_source(args)
            gate = InjectedSessionGate(
                json.loads(Path(args.session_json).read_text(encoding="utf-8"))
            )
            collector = PerformanceEvidenceCollector(
                events_path=events_path,
                lock_path=lock_path,
                source=source,
                session_gate=gate,
                persist=False,
                allow_production_destination=production_publish,
                production_root=Path(args.production_root) if production_publish else None,
            )
            context = _cli_publication_context(args)
            if args.publish_cutover:
                return collector.publish_cutover(confirm=args.confirm, context=context)
            return collector.publish_session_close(confirm=args.confirm, context=context)
        if not args.snapshot_json or not args.session_json:
            raise CollectorError("session_and_snapshot_json_required")
        source = MemoryEvidenceSource(
            snapshot=json.loads(Path(args.snapshot_json).read_text(encoding="utf-8"))
        )
        gate = InjectedSessionGate(json.loads(Path(args.session_json).read_text(encoding="utf-8")))
        collector = PerformanceEvidenceCollector(
            events_path=events_path,
            lock_path=lock_path,
            source=source,
            session_gate=gate,
            persist=False,
        )
        return collector.collect_session_close()
    except (CollectorError, PerformanceAccountingError) as exc:
        return _blocked_report(_cli_mode_name(args), exc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    report = redact_payload(run_cli(argv))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("checks_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
