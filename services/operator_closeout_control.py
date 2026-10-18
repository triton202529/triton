"""Protected PAPER_ONLY dashboard closeout.

Reviews eligibility, drains the continuous loop and watchdog through the
existing cooperative shutdown controller, then asks a detached non-Streamlit
worker to stop the dashboard and call the certified-close publisher.

This module never submits, cancels, replaces, or otherwise modifies a broker
order. It does not delete or repair runtime locks and it does not force-kill
a TRITON loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from services.operator_cooperative_shutdown import atomic_write_json, is_guard_active, read_json
from services.operator_runtime_classifier import UNSAFE_STATES
from services.performance_evidence_collector import (
    CONFIRM_SESSION_CLOSE,
    ORDER_MUTATION_METHODS,
    STALE_AFTER_SECONDS,
)

ROOT = Path(__file__).resolve().parents[1]
NY_TZ = ZoneInfo("America/New_York")
SCHEMA_VERSION = "CLOSEOUT-1"
CONFIRMATION_PHRASE = "CLOSE OUT AND STOP TRITON"
PAPER_HOST = "paper-api.alpaca.markets"
IDENTITY_TOLERANCE = Decimal("0.01")
RECEIPT_DIRNAME = "operator_closeout"

STATE_PREFLIGHT = "PREFLIGHT"
STATE_DRAINING = "DRAINING"
STATE_STOPPING = "STOPPING"
STATE_RUNTIME_STOPPED = "RUNTIME_STOPPED"
STATE_RECONCILING = "RECONCILING"
STATE_PUBLISHING = "PUBLISHING"
STATE_CERTIFIED = "CERTIFIED"
STATE_REFUSED = "REFUSED"
STATE_WAITING = "WAITING_FOR_DAILY_REFRESH"
STATE_DRAIN_FAILED = "DRAIN_FAILED"
STATE_STOP_FAILED = "STOP_FAILED"
STATE_BROKER_READ_FAILED = "BROKER_READ_FAILED"
STATE_RECONCILE_FAILED = "RECONCILE_FAILED"
STATE_STREAMLIT_STOP_FAILED = "STREAMLIT_STOP_FAILED"
STATE_PUBLICATION_FAILED = "PUBLICATION_FAILED"
STATE_ALREADY_CERTIFIED = "ALREADY_CERTIFIED"

HAPPY_PATH = (
    STATE_PREFLIGHT,
    STATE_DRAINING,
    STATE_STOPPING,
    STATE_RUNTIME_STOPPED,
    STATE_RECONCILING,
    STATE_PUBLISHING,
    STATE_CERTIFIED,
)
TERMINAL_STATES = frozenset(
    {
        STATE_CERTIFIED,
        STATE_REFUSED,
        STATE_WAITING,
        STATE_DRAIN_FAILED,
        STATE_STOP_FAILED,
        STATE_BROKER_READ_FAILED,
        STATE_RECONCILE_FAILED,
        STATE_STREAMLIT_STOP_FAILED,
        STATE_PUBLICATION_FAILED,
        STATE_ALREADY_CERTIFIED,
    }
)
COMPLETED_STATES = frozenset({STATE_CERTIFIED, STATE_ALREADY_CERTIFIED})
ACTIVE_STATES = frozenset(
    {
        STATE_DRAINING,
        STATE_STOPPING,
        STATE_RUNTIME_STOPPED,
        STATE_RECONCILING,
        STATE_PUBLISHING,
    }
)
RETRYABLE_STATES = TERMINAL_STATES - COMPLETED_STATES
CERTIFIED_SESSION_REASONS = frozenset({"regular_close", "early_close"})
SHUTDOWN_SUCCESS = frozenset({"COOPERATIVE_SHUTDOWN_COMPLETED", "ALREADY_STOPPED"})
LOOP_MARKERS = (
    "run_continuous_loop.ps1",
    "run_watchdog_loop.ps1",
    "run_triton_runtime",
    "start_triton_runtime",
    "run_daily_refresh.ps1",
    "run_data_freshness_refresh.ps1",
)
READ_ONLY_BROKER_METHODS = ("get_account", "get_positions", "get_open_orders")


class CloseoutError(Exception):
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


class ReadOnlyBroker:
    """Exposes account, position, and open-order reads. Mutation methods are absent."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def mode(self) -> str:
        return str(getattr(self._inner, "mode", "") or "")

    @property
    def base(self) -> str:
        return str(getattr(self._inner, "base", "") or "")

    def get_account(self) -> Mapping[str, Any]:
        return self._inner.get_account()

    def get_positions(self) -> Sequence[Mapping[str, Any]]:
        return list(self._inner.get_positions() or [])

    def get_open_orders(self) -> Sequence[Mapping[str, Any]]:
        if not callable(getattr(self._inner, "get_open_orders", None)):
            raise CloseoutError("open_orders_unreadable")
        return list(self._inner.get_open_orders() or [])

    def __getattr__(self, name: str) -> Any:
        if name in ORDER_MUTATION_METHODS:
            return None
        raise AttributeError(name)


def receipt_path(root: Path, session_date: str) -> Path:
    return Path(root) / "data" / "live" / RECEIPT_DIRNAME / f"{session_date}.json"


def load_receipt(path: Path) -> Dict[str, Any]:
    doc = read_json(path)
    if not doc or doc.get("__malformed__"):
        return {}
    return doc


def _order_flags() -> Dict[str, bool]:
    return {
        "orders_submitted": False,
        "orders_cancelled": False,
        "orders_replaced": False,
        "orders_modified": False,
    }


def _base_receipt(session_date: str, *, closeout_id: Optional[str] = None) -> Dict[str, Any]:
    now = _iso(datetime.now(timezone.utc))
    return {
        "schema_version": SCHEMA_VERSION,
        "closeout_id": closeout_id or str(uuid.uuid4()),
        "session_date": session_date,
        "state": STATE_PREFLIGHT,
        "states_visited": [STATE_PREFLIGHT],
        "reason": None,
        "blockers": [],
        "created_at": now,
        "updated_at": now,
        "forced_termination_used": False,
        "runtime_locks_deleted_by_controller": False,
        "runtime_locks_repaired_by_controller": False,
        "paper_arm_mutated": False,
        "live_execution_blocked": True,
        "account_mode": "PAPER_ONLY",
        "writes_performed": False,
        "logical_key": f"session_close:{session_date}",
        **_order_flags(),
    }


def _save(path: Path, receipt: Dict[str, Any]) -> Dict[str, Any]:
    receipt["updated_at"] = _iso(datetime.now(timezone.utc))
    receipt.update(_order_flags())
    written = atomic_write_json(path, receipt)
    if not written.get("ok"):
        raise CloseoutError("receipt_write_failed", written)
    return receipt


def _advance(receipt: Dict[str, Any], state: str, *, reason: Optional[str] = None) -> None:
    receipt["state"] = state
    visited = list(receipt.get("states_visited") or [])
    if not visited or visited[-1] != state:
        visited.append(state)
    receipt["states_visited"] = visited
    if reason is not None:
        receipt["reason"] = reason


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _decimal(value: Any) -> Optional[Decimal]:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def format_close_et(value: Optional[str]) -> str:
    if not value:
        return "Unavailable"
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return "Unavailable"
    if parsed.tzinfo is None:
        return "Unavailable"
    local = parsed.astimezone(NY_TZ)
    return local.strftime("%Y-%m-%d %H:%M ET")


def session_close_key(session_date: str) -> str:
    return f"session_close:{session_date}"


def ledger_has_session_close(events_path: Path, session_date: str) -> bool:
    key = session_close_key(session_date)
    if not events_path.is_file():
        return False
    needle = f'"logical_key": "{key}"'
    loose = f'"logical_key":"{key}"'
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if needle in line or loose in line:
                return True
    return False


def list_closeout_receipts(root: Path) -> List[Dict[str, Any]]:
    folder = Path(root) / "data" / "live" / RECEIPT_DIRNAME
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob("*.json")):
        doc = load_receipt(path)
        if doc:
            doc["_path"] = str(path)
            found.append(doc)
    return found


def active_closeout(
    root: Path, *, except_session: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    for doc in list_closeout_receipts(root):
        if except_session and doc.get("session_date") == except_session:
            continue
        if doc.get("state") in ACTIVE_STATES:
            return doc
    return None


def dashboard_stop_argv(pid: int) -> List[str]:
    """Graceful stop of one reviewed dashboard PID. No /F flag."""
    return ["taskkill", "/PID", str(int(pid))]


def classify_dashboard_processes(
    processes: Sequence[Mapping[str, Any]], *, own_pid: Optional[int] = None
) -> Dict[str, Any]:
    dashboards: List[Dict[str, Any]] = []
    unsafe: List[Dict[str, Any]] = []
    for proc in processes:
        cmd = str(proc.get("command_line") or "")
        lowered = cmd.lower()
        pid = proc.get("pid")
        if own_pid is not None and pid == own_pid:
            continue
        loop = any(marker.lower() in lowered for marker in LOOP_MARKERS)
        streamlit = "streamlit" in lowered
        if loop and streamlit:
            unsafe.append({"pid": pid, "reason": "loop_and_streamlit"})
            continue
        if loop:
            continue
        if streamlit:
            dashboards.append({"pid": pid, "command_line": cmd})
    return {
        "dashboard_pids": [row["pid"] for row in dashboards if row.get("pid") is not None],
        "unsafe": unsafe,
        "ok": not unsafe,
    }


def current_process_command() -> str:
    return " ".join(sys.argv).lower()


def exit_current_dashboard(
    *, command: Optional[str] = None, exit_fn: Optional[Callable[[int], None]] = None
) -> Dict[str, Any]:
    """Stop only this process, and only when it is the Streamlit dashboard."""
    cmd = current_process_command() if command is None else command.lower()
    if "pytest" in cmd:
        return {"ok": False, "reason": "refused_test_process"}
    if any(marker.lower() in cmd for marker in LOOP_MARKERS):
        return {"ok": False, "reason": "refused_loop_process"}
    if "streamlit" not in cmd:
        return {"ok": False, "reason": "not_dashboard_process"}
    if exit_fn is None:
        os._exit(0)
    exit_fn(0)
    return {"ok": True, "reason": "exit_requested"}


def _condition(code: str, ok: bool, detail: str) -> Dict[str, Any]:
    return {"code": code, "ok": bool(ok), "detail": detail}


def _paper_broker_ok(broker: Any) -> bool:
    mode = str(getattr(broker, "mode", "") or "").lower()
    base = str(getattr(broker, "base", "") or "").lower()
    return mode == "paper" and PAPER_HOST in base


def _normalize_positions(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    positions = []
    for row in rows:
        qty = _decimal(row.get("qty"))
        mv = _decimal(row.get("market_value"))
        avg = _decimal(row.get("avg_entry_price") or row.get("average_entry_price"))
        positions.append(
            {
                "symbol": str(row.get("symbol") or "").upper(),
                "qty": None if qty is None else format(qty, "f"),
                "market_value": None if mv is None else format(mv, "f"),
                "average_entry_price": None if avg is None else format(avg, "f"),
                "unrealized_pl": row.get("unrealized_pl"),
                "side": str(row.get("side") or "long").lower(),
            }
        )
    return positions


def _account_view(
    account: Mapping[str, Any], positions: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    cash = _decimal(account.get("cash"))
    equity = _decimal(account.get("equity"))
    market = _decimal(account.get("long_market_value") or account.get("market_value"))
    position_sum = Decimal("0")
    shorts = False
    for row in positions:
        qty = _decimal(row.get("qty"))
        side = str(row.get("side") or "long").lower()
        if qty is not None and qty < 0 or side == "short":
            shorts = True
        mv = _decimal(row.get("market_value"))
        if mv is not None:
            position_sum += mv
    residual = None
    position_gap = None
    if cash is not None and market is not None and equity is not None:
        residual = abs(cash + market - equity)
    if market is not None:
        position_gap = abs(position_sum - market)
    account_id = str(account.get("id") or account.get("account_id") or "")
    fingerprint = hashlib.sha256(account_id.encode("utf-8")).hexdigest() if account_id else None
    return {
        "cash": None if cash is None else format(cash, "f"),
        "market_value": None if market is None else format(market, "f"),
        "equity": None if equity is None else format(equity, "f"),
        "residual": None if residual is None else format(residual, "f"),
        "position_gap": None if position_gap is None else format(position_gap, "f"),
        "shorts": shorts,
        "account_fingerprint": fingerprint,
        "positions": _normalize_positions(positions),
        "buying_power": account.get("buying_power"),
    }


def _read_broker(broker: ReadOnlyBroker) -> Dict[str, Any]:
    try:
        account = broker.get_account()
        positions = broker.get_positions()
        orders = broker.get_open_orders()
    except CloseoutError:
        raise
    except Exception as exc:
        raise CloseoutError("broker_read_failed", {"error": type(exc).__name__}) from exc
    view = _account_view(account, positions)
    view["open_order_count"] = len(list(orders))
    return view


def _daily_refresh_status(
    root: Path,
    session_date: str,
    *,
    receipt_doc: Optional[Mapping[str, Any]] = None,
    refresh_running: bool = False,
) -> Dict[str, Any]:
    if receipt_doc is None:
        path = Path(root) / "data" / "results" / "daily_refresh_fetch_receipt.json"
        receipt_doc = read_json(path) if path.is_file() else {}
    lock_present = (Path(root) / "data" / "live" / "daily_refresh.lock").is_file() or (
        Path(root) / "data" / "daily_refresh.lock"
    ).is_file()
    target = str((receipt_doc or {}).get("target_session") or "")
    passed = (receipt_doc or {}).get("passed") is True and not (receipt_doc or {}).get("failure")
    ok = passed and target == session_date and not lock_present and not refresh_running
    if refresh_running or lock_present:
        detail = "daily_refresh_still_running"
    elif not receipt_doc:
        detail = "daily_refresh_receipt_missing"
    elif target != session_date:
        detail = "daily_refresh_session_mismatch"
    elif not passed:
        detail = "daily_refresh_not_passed"
    else:
        detail = "daily_refresh_complete"
    return {"ok": ok, "detail": detail, "target_session": target}


def _benchmark_ok(bars: Mapping[str, Any], symbol: str, session_date: str) -> bool:
    row = bars.get(symbol) or bars.get(f"{symbol}:{session_date}") or {}
    if str(row.get("session_date") or session_date) != session_date:
        return False
    return row.get("close") not in (None, "") or row.get("adj_close") not in (None, "")


def evaluate_eligibility(
    *,
    root: Path,
    session: Mapping[str, Any],
    broker: Optional[Any] = None,
    classifications: Optional[Mapping[str, Any]] = None,
    live_execution_blocked: bool = True,
    start_guard_active: bool = False,
    shutdown_guard_active: bool = False,
    daily_refresh: Optional[Mapping[str, Any]] = None,
    refresh_running: bool = False,
    benchmarks: Optional[Mapping[str, Any]] = None,
    manual_gate: Optional[Mapping[str, Any]] = None,
    duplicate_close: Optional[bool] = None,
    git_identity: Optional[Mapping[str, Any]] = None,
    freshness_idle: bool = True,
    freshness_lock_present: bool = False,
    as_of: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Read-only eligibility. Performs broker reads when a broker is supplied."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    as_of = as_of or now
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    reason = str(session.get("reason") or "")
    state = str(session.get("session_state") or "")
    session_date = str(session.get("session_date") or "")
    close_raw = session.get("session_close_et")
    calendar = str(session.get("calendar") or "")
    session_ok = (
        calendar.upper() == "XNYS"
        and state == "CLOSED"
        and reason in CERTIFIED_SESSION_REASONS
        and bool(session_date)
    )
    conditions = [
        _condition(
            "session_regular_or_early_close",
            session_ok,
            reason if reason else state or "session_unavailable",
        )
    ]
    terminal = None
    terminal_reason = None
    if not session_ok:
        terminal = STATE_REFUSED
        terminal_reason = "session_not_certified_close"
    elif session_date:
        if duplicate_close is None:
            events_path = Path(root) / "data" / "results" / "performance_accounting_events.jsonl"
            duplicate_close = ledger_has_session_close(events_path, session_date)
        if duplicate_close:
            terminal = STATE_ALREADY_CERTIFIED
            terminal_reason = "duplicate_session_close"

    classifications = classifications or {}
    cont = str((classifications.get("continuous") or {}).get("state") or "UNKNOWN")
    wd = str((classifications.get("watchdog") or {}).get("state") or "UNKNOWN")
    stale = cont == "STALE_LOCK_NO_PROCESS" or wd == "STALE_LOCK_NO_PROCESS"
    unsafe = cont in UNSAFE_STATES or wd in UNSAFE_STATES
    other_active = active_closeout(root, except_session=session_date or None)
    same = load_receipt(receipt_path(root, session_date)) if session_date else {}
    same_active = same.get("state") in ACTIVE_STATES
    conditions.append(
        _condition(
            "paper_only_live_blocked",
            live_execution_blocked,
            "PAPER_ONLY" if live_execution_blocked else "live_execution_not_blocked",
        )
    )
    if terminal is None and not live_execution_blocked:
        terminal, terminal_reason = STATE_REFUSED, "live_execution_not_blocked"

    paper_ok = broker is not None and _paper_broker_ok(broker)
    conditions.append(
        _condition("paper_broker", paper_ok, "paper" if paper_ok else "paper_broker_required")
    )
    if terminal is None and not paper_ok:
        terminal, terminal_reason = STATE_REFUSED, "paper_broker_required"

    conditions.append(
        _condition(
            "no_start_guard",
            not start_guard_active,
            "clear" if not start_guard_active else "start_guard_active",
        )
    )
    conditions.append(
        _condition(
            "no_shutdown_guard",
            not shutdown_guard_active,
            "clear" if not shutdown_guard_active else "shutdown_guard_active",
        )
    )
    conditions.append(
        _condition(
            "no_active_closeout",
            not other_active and not same_active,
            "clear" if not other_active and not same_active else "closeout_already_active",
        )
    )
    conditions.append(
        _condition("no_stale_lock", not stale, "clear" if not stale else "stale_lock")
    )
    conditions.append(
        _condition(
            "no_unsafe_process",
            not unsafe,
            f"continuous={cont},watchdog={wd}",
        )
    )
    if terminal is None and (other_active or same_active):
        terminal, terminal_reason = STATE_REFUSED, "closeout_already_active"
    if terminal is None and start_guard_active:
        terminal, terminal_reason = STATE_REFUSED, "start_guard_active"
    if terminal is None and shutdown_guard_active:
        terminal, terminal_reason = STATE_REFUSED, "shutdown_guard_active"
    if terminal is None and stale:
        terminal, terminal_reason = STATE_REFUSED, "stale_lock"
    if terminal is None and unsafe:
        terminal, terminal_reason = STATE_REFUSED, "unsafe_process"

    broker_view: Dict[str, Any] = {}
    broker_error = None
    if broker is not None and paper_ok and terminal is None:
        try:
            broker_view = _read_broker(ReadOnlyBroker(broker))
        except CloseoutError as exc:
            broker_error = exc.reason
    open_count = broker_view.get("open_order_count")
    if broker_view or broker_error:
        conditions.append(
            _condition(
                "zero_open_orders",
                open_count == 0,
                "unreadable" if broker_error else f"open_order_count={open_count}",
            )
        )
    else:
        conditions.append(_condition("zero_open_orders", True, "not_read"))
    if terminal is None and broker_error:
        terminal, terminal_reason = STATE_BROKER_READ_FAILED, broker_error
    if terminal is None and open_count not in (0,):
        terminal, terminal_reason = STATE_REFUSED, "open_orders"

    identity_ok = False
    identity_detail = "not_read"
    if broker_view:
        residual = _decimal(broker_view.get("residual"))
        gap = _decimal(broker_view.get("position_gap"))
        cash = _decimal(broker_view.get("cash"))
        identity_ok = (
            residual is not None
            and residual <= IDENTITY_TOLERANCE
            and gap is not None
            and gap <= IDENTITY_TOLERANCE
            and cash is not None
            and cash > 0
            and bool(broker_view.get("positions"))
            and not broker_view.get("shorts")
        )
        identity_detail = f"residual={broker_view.get('residual')}"
    conditions.append(_condition("account_identity", identity_ok, identity_detail))
    conditions.append(
        _condition(
            "no_short_structure",
            bool(broker_view) and not broker_view.get("shorts"),
            "short_structure" if broker_view.get("shorts") else "long_only",
        )
    )
    if terminal is None and broker_view and (not identity_ok or broker_view.get("shorts")):
        terminal, terminal_reason = STATE_RECONCILE_FAILED, (
            "short_structure" if broker_view.get("shorts") else "account_identity"
        )

    snapshot_ok = False
    snapshot_detail = "not_checked"
    close_dt = None
    if close_raw:
        try:
            close_dt = datetime.fromisoformat(str(close_raw))
            if close_dt.tzinfo is None:
                close_dt = None
        except ValueError:
            close_dt = None
    if close_dt is not None:
        age = (now - as_of).total_seconds()
        snapshot_ok = as_of >= close_dt and 0 <= age <= STALE_AFTER_SECONDS
        if as_of < close_dt:
            snapshot_detail = "snapshot_before_session_close"
        elif age > STALE_AFTER_SECONDS:
            snapshot_detail = "snapshot_older_than_36h"
        else:
            snapshot_detail = "fresh"
    conditions.append(_condition("snapshot_fresh", snapshot_ok, snapshot_detail))
    if terminal is None and not snapshot_ok:
        terminal, terminal_reason = STATE_REFUSED, snapshot_detail

    refresh = _daily_refresh_status(
        root, session_date, receipt_doc=daily_refresh, refresh_running=refresh_running
    )
    conditions.append(_condition("daily_refresh_complete", refresh["ok"], refresh["detail"]))
    if terminal is None and not refresh["ok"]:
        terminal, terminal_reason = STATE_WAITING, refresh["detail"]

    bars = benchmarks or {}
    spy_ok = _benchmark_ok(bars, "SPY", session_date)
    qqq_ok = _benchmark_ok(bars, "QQQ", session_date)
    conditions.append(
        _condition("benchmark_spy", spy_ok, "present" if spy_ok else "missing_spy_close")
    )
    conditions.append(
        _condition("benchmark_qqq", qqq_ok, "present" if qqq_ok else "missing_qqq_close")
    )
    if terminal is None and not (spy_ok and qqq_ok):
        terminal, terminal_reason = STATE_REFUSED, "missing_benchmark"

    if manual_gate is None:
        store = Path(root) / "data" / "governance" / "manual_portfolio_exit"
        if store.is_dir():
            from services.manual_portfolio_exit import evaluate_manual_exit_close_gate

            manual_gate = evaluate_manual_exit_close_gate(store)
        else:
            manual_gate = {"ok": True, "reason": "store_absent"}
    manual_ok = manual_gate.get("ok") is True
    conditions.append(
        _condition(
            "manual_exit_gate", manual_ok, str(manual_gate.get("reason") or "manual_exit_gate")
        )
    )
    if terminal is None and not manual_ok:
        terminal, terminal_reason = STATE_REFUSED, "manual_exit_gate"

    duplicate_close = bool(duplicate_close)
    conditions.append(
        _condition(
            "no_duplicate_session_close",
            not duplicate_close,
            "already_present" if duplicate_close else "absent",
        )
    )

    git_ok = (
        bool(git_identity)
        and git_identity.get("git_identity_valid") is True
        and git_identity.get("index_empty") is True
    )
    conditions.append(
        _condition(
            "git_identity",
            git_ok,
            str(
                (git_identity or {}).get("failure_reason")
                or ("valid" if git_ok else "git_identity_unavailable")
            ),
        )
    )
    if terminal is None and not git_ok:
        terminal, terminal_reason = STATE_REFUSED, "git_identity"

    fresh_ok = freshness_idle and not freshness_lock_present
    conditions.append(
        _condition("freshness_idle", fresh_ok, "idle" if fresh_ok else "freshness_writer_active")
    )
    if terminal is None and not fresh_ok:
        terminal, terminal_reason = STATE_REFUSED, "freshness_writer_active"

    return {
        "ok": terminal is None,
        "terminal": terminal,
        "terminal_reason": terminal_reason,
        "session_date": session_date or None,
        "session_close_et": close_raw,
        "session_close_display": format_close_et(str(close_raw) if close_raw else None),
        "close_kind": reason if reason in CERTIFIED_SESSION_REASONS else None,
        "conditions": conditions,
        "blockers": [row for row in conditions if not row["ok"]],
        "broker": {
            "cash": broker_view.get("cash"),
            "market_value": broker_view.get("market_value"),
            "equity": broker_view.get("equity"),
            "residual": broker_view.get("residual"),
            "open_order_count": broker_view.get("open_order_count"),
            "account_fingerprint": broker_view.get("account_fingerprint"),
            "positions": broker_view.get("positions") or [],
            "buying_power": broker_view.get("buying_power"),
        },
        "as_of_utc": _iso(as_of),
        **_order_flags(),
    }


def review_closeout(**kwargs: Any) -> Dict[str, Any]:
    """Eligibility for the review screen. Does not write a receipt or stop anything."""
    report = evaluate_eligibility(**kwargs)
    report["confirmation_phrase"] = CONFIRMATION_PHRASE
    report["writes_performed"] = False
    return report


def _seal(
    receipt: Dict[str, Any],
    review: Mapping[str, Any],
    session: Mapping[str, Any],
    benchmarks: Mapping[str, Any],
) -> None:
    broker = review.get("broker") or {}
    receipt["session_close_et"] = review.get("session_close_et")
    receipt["session_close_display"] = review.get("session_close_display")
    receipt["close_kind"] = review.get("close_kind")
    receipt["cash"] = broker.get("cash")
    receipt["market_value"] = broker.get("market_value")
    receipt["equity"] = broker.get("equity")
    receipt["identity_residual"] = broker.get("residual")
    receipt["open_order_count"] = broker.get("open_order_count")
    receipt["account_fingerprint"] = broker.get("account_fingerprint")
    receipt["as_of_utc"] = review.get("as_of_utc")
    receipt["sealed_snapshot"] = {
        "cash": broker.get("cash"),
        "market_value": broker.get("market_value"),
        "equity": broker.get("equity"),
        "buying_power": broker.get("buying_power") or "UNAVAILABLE",
        "as_of_utc": review.get("as_of_utc"),
        "session_date": review.get("session_date"),
        "account_structure": "long_only_paper",
        "positions": broker.get("positions") or [],
    }
    receipt["benchmarks"] = {
        "SPY": dict(benchmarks.get("SPY") or {}),
        "QQQ": dict(benchmarks.get("QQQ") or {}),
    }
    receipt["source_hashes"] = {
        "account": hashlib.sha256(str(receipt.get("cash")).encode("utf-8")).hexdigest(),
        "positions": hashlib.sha256(
            json.dumps(receipt["sealed_snapshot"]["positions"], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "orders": hashlib.sha256(str(receipt.get("open_order_count")).encode("utf-8")).hexdigest(),
        "activities": hashlib.sha256(b"closeout-no-activity-delta").hexdigest(),
        "SPY": hashlib.sha256(
            json.dumps(receipt["benchmarks"]["SPY"], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "QQQ": hashlib.sha256(
            json.dumps(receipt["benchmarks"]["QQQ"], sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    receipt["calendar"] = session.get("calendar")
    receipt["blockers"] = []


def _fail(path: Path, receipt: Dict[str, Any], state: str, reason: str) -> Dict[str, Any]:
    _advance(receipt, state, reason=reason)
    return _save(path, receipt)


def execute_closeout(
    *,
    confirmation_phrase: str,
    root: Path,
    session: Mapping[str, Any],
    broker: Any,
    benchmarks: Mapping[str, Any],
    classifications: Mapping[str, Any],
    live_execution_blocked: bool = True,
    start_guard_active: bool = False,
    shutdown_guard_active: bool = False,
    daily_refresh: Optional[Mapping[str, Any]] = None,
    refresh_running: bool = False,
    manual_gate: Optional[Mapping[str, Any]] = None,
    duplicate_close: Optional[bool] = None,
    git_identity: Optional[Mapping[str, Any]] = None,
    freshness_idle: bool = True,
    freshness_lock_present: bool = False,
    as_of: Optional[datetime] = None,
    now: Optional[datetime] = None,
    shutdown_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    worker_runner: Optional[Callable[[Path], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the closeout through PUBLISHING, then the injected or detached worker."""
    session_date = str(session.get("session_date") or "")
    path = receipt_path(root, session_date) if session_date else None
    existing = load_receipt(path) if path and path.is_file() else {}
    if existing.get("state") in COMPLETED_STATES:
        existing["replayed"] = True
        return existing
    if existing.get("state") in ACTIVE_STATES:
        existing["replayed"] = False
        existing["reason"] = existing.get("reason") or "closeout_already_active"
        return existing
    if confirmation_phrase != CONFIRMATION_PHRASE:
        return {
            "state": STATE_REFUSED,
            "reason": "phrase_mismatch",
            "writes_performed": False,
            "replayed": False,
            **_order_flags(),
        }
    review = evaluate_eligibility(
        root=root,
        session=session,
        broker=broker,
        classifications=classifications,
        live_execution_blocked=live_execution_blocked,
        start_guard_active=start_guard_active,
        shutdown_guard_active=shutdown_guard_active,
        daily_refresh=daily_refresh,
        refresh_running=refresh_running,
        benchmarks=benchmarks,
        manual_gate=manual_gate,
        duplicate_close=duplicate_close,
        git_identity=git_identity,
        freshness_idle=freshness_idle,
        freshness_lock_present=freshness_lock_present,
        as_of=as_of,
        now=now,
    )
    if not session_date or path is None:
        return {
            "state": STATE_REFUSED,
            "reason": review.get("terminal_reason") or "session_not_certified_close",
            "conditions": review.get("conditions"),
            "blockers": review.get("blockers"),
            "writes_performed": False,
            **_order_flags(),
        }
    if existing.get("state") in RETRYABLE_STATES:
        receipt = _base_receipt(session_date)
        receipt["prior_state"] = existing.get("state")
        receipt["prior_reason"] = existing.get("reason")
    else:
        receipt = _base_receipt(session_date)
    receipt["session_close_et"] = review.get("session_close_et")
    receipt["session_close_display"] = review.get("session_close_display")
    receipt["close_kind"] = review.get("close_kind")
    receipt["blockers"] = review.get("blockers")
    if review.get("terminal"):
        _advance(receipt, str(review["terminal"]), reason=str(review.get("terminal_reason")))
        if review.get("broker"):
            receipt["cash"] = review["broker"].get("cash")
            receipt["market_value"] = review["broker"].get("market_value")
            receipt["equity"] = review["broker"].get("equity")
            receipt["open_order_count"] = review["broker"].get("open_order_count")
        return _save(path, receipt)

    _save(path, receipt)
    _advance(receipt, STATE_DRAINING)
    _save(path, receipt)
    shutdown = shutdown_fn or _production_shutdown
    try:
        stopped = shutdown(root=root, classifications=classifications)
    except Exception as exc:
        receipt["shutdown_error"] = type(exc).__name__
        return _fail(path, receipt, STATE_DRAIN_FAILED, "shutdown_exception")
    receipt["shutdown_result"] = stopped.get("result")
    receipt["forced_termination_used"] = bool(stopped.get("forced_termination_used"))
    receipt["runtime_locks_deleted_by_controller"] = bool(
        stopped.get("runtime_locks_deleted_by_controller")
    )
    if receipt["forced_termination_used"] or receipt["runtime_locks_deleted_by_controller"]:
        return _fail(path, receipt, STATE_STOP_FAILED, "shutdown_boundary_violated")
    if stopped.get("result") not in SHUTDOWN_SUCCESS:
        failed = (
            STATE_STOP_FAILED
            if "TIME" in str(stopped.get("result"))
            or "PARTIAL" in str(stopped.get("result"))
            or "POSTCONDITION" in str(stopped.get("result"))
            else STATE_DRAIN_FAILED
        )
        return _fail(path, receipt, failed, str(stopped.get("result") or "shutdown_failed"))
    _advance(receipt, STATE_STOPPING)
    _advance(receipt, STATE_RUNTIME_STOPPED)
    _save(path, receipt)

    _advance(receipt, STATE_RECONCILING)
    try:
        sealed = _read_broker(ReadOnlyBroker(broker))
    except CloseoutError as exc:
        return _fail(path, receipt, STATE_BROKER_READ_FAILED, exc.reason)
    review["broker"] = {
        "cash": sealed.get("cash"),
        "market_value": sealed.get("market_value"),
        "equity": sealed.get("equity"),
        "residual": sealed.get("residual"),
        "open_order_count": sealed.get("open_order_count"),
        "account_fingerprint": sealed.get("account_fingerprint"),
        "positions": sealed.get("positions") or [],
        "buying_power": sealed.get("buying_power"),
    }
    if sealed.get("open_order_count") != 0:
        return _fail(path, receipt, STATE_REFUSED, "open_orders")
    if sealed.get("shorts") or _decimal(sealed.get("residual") or "1") > IDENTITY_TOLERANCE:
        return _fail(path, receipt, STATE_RECONCILE_FAILED, "account_identity")
    _seal(receipt, review, session, benchmarks)
    _advance(receipt, STATE_PUBLISHING)
    _save(path, receipt)
    if worker_runner is not None:
        worker_runner(path)
        return load_receipt(path)
    launch_detached_worker(root, path)
    latest = load_receipt(path)
    latest["dashboard_exit_required"] = True
    return latest


def _production_shutdown(*, root: Path, classifications: Mapping[str, Any]) -> Dict[str, Any]:
    from services.operator_shutdown_controller import prepare_cooperative_shutdown

    del classifications
    return prepare_cooperative_shutdown(
        root=root,
        source="operator_closeout",
        reason_code="CLOSEOUT",
    )


def launch_detached_worker(
    root: Path, receipt: Path, *, popen: Optional[Callable[..., Any]] = None
) -> Dict[str, Any]:
    executable = sys.executable
    argv = [
        executable,
        "-m",
        "services.operator_closeout_control",
        "--worker",
        "--root",
        str(root),
        "--receipt",
        str(receipt),
    ]
    if any("streamlit" in str(part).lower() for part in argv[1:]):
        raise CloseoutError("worker_command_refused")
    flags = 0
    if os.name == "nt":
        flags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    launcher = popen or subprocess.Popen
    proc = launcher(
        argv,
        creationflags=flags,
        close_fds=True,
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {"ok": True, "pid": getattr(proc, "pid", None), "argv": argv, "creationflags": flags}


def stop_dashboard_last(
    processes: Sequence[Mapping[str, Any]],
    *,
    killer: Callable[[int], None],
    own_pid: Optional[int] = None,
) -> Dict[str, Any]:
    plan = classify_dashboard_processes(processes, own_pid=own_pid)
    if not plan["ok"]:
        return {"ok": False, "reason": "unsafe_dashboard_target", "plan": plan}
    for pid in plan["dashboard_pids"]:
        argv = dashboard_stop_argv(int(pid))
        if "/F" in argv or "/f" in argv:
            return {"ok": False, "reason": "force_flag_refused", "plan": plan}
        killer(int(pid))
    return {
        "ok": True,
        "reason": "dashboard_stop_requested",
        "plan": plan,
        "pids": plan["dashboard_pids"],
    }


def run_publishing_worker(
    *,
    root: Path,
    receipt_path: Path,
    runtime_inspector: Callable[[Path], Mapping[str, Any]],
    git_inspector: Callable[[Path], Mapping[str, Any]],
    processes: Sequence[Mapping[str, Any]],
    killer: Callable[[int], None],
    publisher: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]],
    own_pid: Optional[int] = None,
) -> Dict[str, Any]:
    receipt = load_receipt(receipt_path)
    if receipt.get("state") in COMPLETED_STATES:
        receipt["replayed"] = True
        return receipt
    if receipt.get("state") != STATE_PUBLISHING:
        return _fail(
            receipt_path,
            receipt or _base_receipt("unknown"),
            STATE_PUBLICATION_FAILED,
            "not_publishing",
        )
    stopped = stop_dashboard_last(processes, killer=killer, own_pid=own_pid)
    if not stopped.get("ok"):
        return _fail(receipt_path, receipt, STATE_STREAMLIT_STOP_FAILED, str(stopped.get("reason")))
    runtime = dict(runtime_inspector(root))
    ports_closed = runtime.get("ports_closed") is True
    dashboard_gone = runtime.get("streamlit_stopped") is True
    loops_gone = runtime.get("runtime_stopped") is True and runtime.get("watchdog_stopped") is True
    if not (
        ports_closed
        and dashboard_gone
        and loops_gone
        and runtime.get("freshness_writer_idle") is True
        and not runtime.get("freshness_lock_present")
    ):
        return _fail(
            receipt_path, receipt, STATE_STREAMLIT_STOP_FAILED, "dashboard_or_ports_still_present"
        )
    git_identity = dict(git_inspector(root))
    if (
        git_identity.get("git_identity_valid") is not True
        or git_identity.get("index_empty") is not True
    ):
        return _fail(receipt_path, receipt, STATE_PUBLICATION_FAILED, "git_identity")
    events_path = Path(root) / "data" / "results" / "performance_accounting_events.jsonl"
    if ledger_has_session_close(events_path, str(receipt.get("session_date"))):
        _advance(receipt, STATE_ALREADY_CERTIFIED, reason="duplicate_session_close")
        receipt["writes_performed"] = False
        return _save(receipt_path, receipt)
    try:
        published = dict(publisher(root, receipt))
    except Exception as exc:
        receipt["publisher_error"] = type(exc).__name__
        return _fail(receipt_path, receipt, STATE_PUBLICATION_FAILED, "publisher_exception")
    blocked = published.get("blocked_reason")
    if blocked == "duplicate_session_close":
        _advance(receipt, STATE_ALREADY_CERTIFIED, reason="duplicate_session_close")
        receipt["writes_performed"] = False
        return _save(receipt_path, receipt)
    if blocked or published.get("writes_performed") is not True:
        return _fail(
            receipt_path,
            receipt,
            STATE_PUBLICATION_FAILED,
            str(blocked or published.get("reason") or "publication_failed"),
        )
    _advance(receipt, STATE_CERTIFIED, reason="published")
    receipt["writes_performed"] = True
    receipt["payload_hash"] = published.get("payload_hash")
    return _save(receipt_path, receipt)


def production_publisher(root: Path, receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Call the existing certified-close publisher with a GET_ONLY context."""
    from services.performance_evidence_collector import (
        InjectedSessionGate,
        MemoryEvidenceSource,
        PerformanceEvidenceCollector,
        PublicationContext,
    )
    from services.performance_publication_authority import (
        inspect_git_identity,
        inspect_runtime_state,
    )

    snap = dict(receipt.get("sealed_snapshot") or {})
    session_date = str(receipt.get("session_date"))
    benchmarks = {
        f"SPY:{session_date}": dict((receipt.get("benchmarks") or {}).get("SPY") or {}),
        f"QQQ:{session_date}": dict((receipt.get("benchmarks") or {}).get("QQQ") or {}),
    }
    source = MemoryEvidenceSource(
        snapshot=snap,
        benchmarks=benchmarks,
        hashes=dict(
            receipt.get("source_hashes")
            or {
                "account": "closeout-account",
                "positions": "closeout-positions",
                "orders": "closeout-orders",
                "activities": "closeout-activities",
                "SPY": "closeout-spy",
                "QQQ": "closeout-qqq",
            }
        ),
    )
    gate = InjectedSessionGate(
        {
            "session_state": "CLOSED",
            "reason": receipt.get("close_kind"),
            "session_date": session_date,
            "session_close_et": receipt.get("session_close_et"),
            "calendar": "XNYS",
        }
    )
    events_path = Path(root) / "data" / "results" / "performance_accounting_events.jsonl"
    lock_path = Path(root) / "data" / "results" / "performance_accounting_events.lock"
    collector = PerformanceEvidenceCollector(
        events_path=events_path,
        lock_path=lock_path,
        source=source,
        session_gate=gate,
        persist=False,
        allow_production_destination=True,
        production_root=Path(root),
    )
    identity = inspect_git_identity(Path(root))
    runtime = inspect_runtime_state(Path(root))
    context = PublicationContext(
        mode="PAPER_ONLY",
        live_execution_permitted=False,
        broker_adapter_mode="GET_ONLY",
        can_mutate_orders=False,
        writers_idle=True,
        freshness_lock_present=False,
        production_root=Path(root),
        expected_git_head=str(identity.get("head") or ""),
        git_identity=identity,
        runtime_state=runtime,
    )
    return collector.publish_session_close(confirm=CONFIRM_SESSION_CLOSE, context=context)


def _cli(argv: Sequence[str]) -> int:
    if len(argv) < 5 or argv[0] != "--worker":
        return 2
    root = Path(argv[argv.index("--root") + 1])
    path = Path(argv[argv.index("--receipt") + 1])
    from services.performance_publication_authority import (
        inspect_git_identity,
        inspect_runtime_state,
    )

    def _processes() -> List[Dict[str, Any]]:
        try:
            blob = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process | ForEach-Object { $_.ProcessId.ToString() + '`t' + $_.CommandLine }",
                ],
                text=True,
                timeout=20,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        rows = []
        for line in blob.splitlines():
            pid_text, _, cmd = line.partition("\t")
            if pid_text.strip().isdigit():
                rows.append({"pid": int(pid_text.strip()), "command_line": cmd})
        return rows

    def _killer(pid: int) -> None:
        subprocess.run(dashboard_stop_argv(pid), check=False, timeout=15)

    run_publishing_worker(
        root=root,
        receipt_path=path,
        runtime_inspector=inspect_runtime_state,
        git_inspector=inspect_git_identity,
        processes=_processes(),
        killer=_killer,
        publisher=production_publisher,
        own_pid=os.getpid(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
