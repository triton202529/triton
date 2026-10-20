"""Closeout defers only residual manual-exit sidecars until after Streamlit stops."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from services.basis_unproven_hold import enforce_basis_unproven_hold_on_order_post
from services.exit_source_reporting import _readonly_rows
from services.manual_portfolio_exit import (
    STORE_DB_NAME,
    build_manual_exit_review,
    classify_manual_exit_close_preflight,
    confirm_manual_paper_exit,
    evaluate_manual_exit_close_gate,
    immutable_readonly_sqlite_uri,
    note_proven_terminal_receipts,
)
from services.operator_closeout_control import (
    CONFIRMATION_PHRASE,
    ORDER_MUTATION_METHODS,
    STATE_ALREADY_CERTIFIED,
    STATE_CERTIFIED,
    STATE_PUBLICATION_FAILED,
    STATE_REFUSED,
    ReadOnlyBroker,
    execute_closeout,
    run_publishing_worker,
)
from services.paper_sell_arm_runtime_enforcement import enforce_paper_sell_broker_submission
from services.paper_sell_arm_store import close_manual_exit_sqlite
from services.performance_publication_authority import (
    MANUAL_EXIT_STORE_RELATIVE,
    REASON_ARTIFACT_UNEXPLAINED,
    evaluate_git_publication_identity,
    passing_runtime_state,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 17, 21, 0, tzinfo=timezone.utc)
AS_OF = datetime(2026, 10, 16, 20, 30, tzinfo=timezone.utc)
CLOSE = "2026-10-16T20:00:00+00:00"
HEAD = "abc"
PRODUCTION = (
    ROOT / "data" / "results" / "paper_order_events.jsonl",
    ROOT / "data" / "results" / "performance_accounting_events.jsonl",
    ROOT / "data" / "ledger" / "ledger.parquet",
    ROOT / "data" / "governance" / "basis_unproven_hold.json",
    ROOT / "data" / "governance" / "basis_unproven_hold_audit.jsonl",
)
PRODUCTION_STORE = ROOT / "data" / "governance" / "manual_portfolio_exit"


def _fingerprint(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _tree(root: Path) -> dict[str, tuple[str, int]]:
    if not root.exists():
        return {}
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            found[path.relative_to(root).as_posix()] = _fingerprint(path)
    return found


@pytest.fixture(scope="module", autouse=True)
def production_files_unchanged():
    before = {path: _fingerprint(path) for path in PRODUCTION if path.is_file()}
    before_store = _tree(PRODUCTION_STORE)
    yield
    after = {path: _fingerprint(path) for path in PRODUCTION if path.is_file()}
    assert after == before
    assert _tree(PRODUCTION_STORE) == before_store


class ExitBroker:
    def __init__(self):
        self.mode = "paper"
        self.base = "https://paper-api.alpaca.markets"
        self.orders = []

    def get_positions(self):
        return [{"symbol": "AAPL", "qty": "10", "current_price": "20"}]

    def submit_order(self, **kwargs):
        enforce_paper_sell_broker_submission(
            mode=self.mode,
            side=kwargs["side"],
            source="broker_alpaca.submit_order",
            symbol=kwargs["symbol"],
            qty=float(kwargs["qty"]),
            client_order_id=kwargs.get("client_order_id"),
        )
        enforce_basis_unproven_hold_on_order_post(
            {
                "symbol": kwargs["symbol"],
                "side": kwargs["side"],
                "qty": kwargs["qty"],
                "client_order_id": kwargs.get("client_order_id"),
            },
            source="broker_alpaca._post",
        )
        self.orders.append(kwargs)
        return {
            "id": "ord-1",
            "status": "accepted",
            "client_order_id": kwargs.get("client_order_id"),
        }


class AccountBroker:
    def __init__(self):
        self.mode = "paper"
        self.base = "https://paper-api.alpaca.markets"
        self.calls = []

    def get_account(self):
        self.calls.append("get_account")
        return {
            "id": "PAPER-ACCOUNT",
            "cash": "100.00",
            "equity": "150.00",
            "long_market_value": "50.00",
            "buying_power": "100.00",
        }

    def get_positions(self):
        self.calls.append("get_positions")
        return [
            {
                "symbol": "SPCX",
                "qty": "1",
                "market_value": "50.00",
                "avg_entry_price": "40",
                "side": "long",
            }
        ]

    def get_open_orders(self):
        self.calls.append("get_open_orders")
        return []

    def submit_order(self, *args, **kwargs):
        self.calls.append("submit_order")

    def cancel_order(self, *args, **kwargs):
        self.calls.append("cancel_order")

    def replace_order(self, *args, **kwargs):
        self.calls.append("replace_order")

    def close_position(self, *args, **kwargs):
        self.calls.append("close_position")

    def place_order(self, *args, **kwargs):
        self.calls.append("place_order")


def _session():
    return {
        "calendar": "XNYS",
        "session_state": "CLOSED",
        "reason": "regular_close",
        "session_date": "2026-10-16",
        "session_close_et": CLOSE,
    }


def _bars():
    return {
        "SPY": {"symbol": "SPY", "session_date": "2026-10-16", "close": "500"},
        "QQQ": {"symbol": "QQQ", "session_date": "2026-10-16", "close": "400"},
    }


def _classes():
    return {"continuous": {"state": "NOT_RUNNING"}, "watchdog": {"state": "NOT_RUNNING"}}


def _valid_git():
    return {
        "git_identity_valid": True,
        "index_empty": True,
        "tracked_worktree_safe": True,
        "failure_reason": None,
        "rejected_paths": [],
        "head": HEAD,
    }


def _recovered(tmp_path: Path):
    root = tmp_path / "repo"
    store = root / MANUAL_EXIT_STORE_RELATIVE
    review = build_manual_exit_review(
        {"symbol": "AAPL", "qty": "10", "current_price": "20", "avg_entry_price": "12"},
        blocked_symbols=[],
    )
    done = confirm_manual_paper_exit(
        review,
        broker=ExitBroker(),
        store_root=store,
        ledger_root=root,
        live_execution_permitted=False,
        blocked_symbols=[],
    )
    assert done["submitted"] is True
    aligned = note_proven_terminal_receipts(
        [
            {
                "event_type": "ORDER_TERMINAL_RECONCILED",
                "terminal_status_proven": True,
                "broker_terminal_status": "FILLED",
                "broker_order_id": done["broker_order_id"],
                "client_order_id": done["client_order_id"],
                "symbol": done["symbol"],
                "filled_qty": done["selected_qty"],
                "extra": {"raw_broker_status": "filled"},
            }
        ],
        root=root,
    )
    assert aligned["ok"] is True
    assert evaluate_manual_exit_close_gate(store)["reason"] == "manual_exit_store_consistent"
    return root, store


def _db(store: Path) -> Path:
    return store / STORE_DB_NAME


def _wal(store: Path) -> Path:
    return Path(str(_db(store)) + "-wal")


def _shm(store: Path) -> Path:
    return Path(str(_db(store)) + "-shm")


def _leave_residual_sidecars(store: Path) -> None:
    db = _db(store)
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    for path in (_wal(store), _shm(store)):
        if path.exists():
            path.unlink()
    _wal(store).write_bytes(b"")
    _shm(store).write_bytes(b"\x00" * 32768)


def _store_identity(root: Path):
    return evaluate_git_publication_identity(
        root=root,
        head=HEAD,
        branch="main",
        index_empty=True,
        porcelain_lines=[f"?? {MANUAL_EXIT_STORE_RELATIVE}/"],
        staged_paths=[],
        verify_filesystem=True,
    )


def _run(root, broker, publisher, git_inspector, **overrides):
    published = []
    holder_probe = overrides.pop("holder_probe", None)

    def default_publisher(_root, receipt):
        published.append(receipt["session_date"])
        return {"writes_performed": True, "payload_hash": "hash-1", "blocked_reason": None}

    chosen_publisher = publisher or default_publisher

    def worker(path):
        run_publishing_worker(
            root=root,
            receipt_path=path,
            runtime_inspector=lambda _root: passing_runtime_state(),
            git_inspector=git_inspector,
            processes=[{"pid": 42, "command_line": "python -m streamlit run view_results.py"}],
            killer=lambda pid: None,
            publisher=chosen_publisher,
            own_pid=999,
            holder_probe=holder_probe or (lambda _db: []),
        )

    def shutdown(**kwargs):
        return {
            "result": "ALREADY_STOPPED",
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
        }

    kwargs = dict(
        confirmation_phrase=CONFIRMATION_PHRASE,
        root=root,
        session=_session(),
        broker=broker,
        benchmarks=_bars(),
        classifications=_classes(),
        live_execution_blocked=True,
        daily_refresh={"target_session": "2026-10-16", "passed": True},
        duplicate_close=False,
        git_identity=overrides.pop("git_identity", _valid_git()),
        as_of=AS_OF,
        now=NOW,
        shutdown_fn=shutdown,
        worker_runner=worker,
    )
    kwargs.update(overrides)
    result = execute_closeout(**kwargs)
    return result, published, broker.calls


def _assert_reads_only(calls):
    assert not any(name in ORDER_MUTATION_METHODS for name in calls)
    wrapped = ReadOnlyBroker(AccountBroker())
    for name in ORDER_MUTATION_METHODS:
        assert getattr(wrapped, name) is None


def test_residual_sidecars_publish_only_after_checkpoint_and_recheck(tmp_path):
    root, store = _recovered(tmp_path)
    _leave_residual_sidecars(store)
    preflight = classify_manual_exit_close_preflight(store)
    assert preflight["deferred_sidecar_cleanup"] is True
    assert preflight["content_ok"] is True
    identity = _store_identity(root)
    assert identity["git_identity_valid"] is False
    assert identity["failure_reason"] == REASON_ARTIFACT_UNEXPLAINED
    seen = {"clean": False}

    def git_inspector(_root):
        if _wal(store).exists() or _shm(store).exists():
            return identity
        seen["clean"] = True
        return _valid_git()

    broker = AccountBroker()
    result, published, calls = _run(root, broker, None, git_inspector, git_identity=identity)
    assert result["state"] == STATE_CERTIFIED
    assert result["writes_performed"] is True
    assert result["deferred_sidecar_cleanup"] is True
    assert seen["clean"] is True
    assert published == ["2026-10-16"]
    assert not _wal(store).exists()
    assert not _shm(store).exists()
    assert evaluate_manual_exit_close_gate(store)["reason"] == "manual_exit_store_consistent"
    _assert_reads_only(calls)


def test_busy_checkpoint_and_open_handle_do_not_publish(tmp_path, monkeypatch):
    root, store = _recovered(tmp_path)
    _leave_residual_sidecars(store)
    identity = _store_identity(root)
    closed = []

    def busy(path):
        closed.append(path.name)
        return {"ok": False, "removed": [], "reason": "sqlite_checkpoint_busy", "busy": 1}

    monkeypatch.setattr("services.paper_sell_arm_store.close_manual_exit_sqlite", busy)
    broker = AccountBroker()
    result, published, calls = _run(
        root, broker, None, lambda _root: identity, git_identity=identity
    )
    assert result["state"] == STATE_PUBLICATION_FAILED
    assert result["reason"] == "sqlite_checkpoint_busy"
    assert result["writes_performed"] is False
    assert published == []
    assert closed == [STORE_DB_NAME]
    assert _wal(store).is_file() and _wal(store).stat().st_size == 0
    assert _shm(store).is_file()
    _assert_reads_only(calls)

    root, store = _recovered(tmp_path)
    _leave_residual_sidecars(store)
    identity = _store_identity(root)

    def refused(path):
        raise AssertionError(path)

    monkeypatch.setattr("services.paper_sell_arm_store.close_manual_exit_sqlite", refused)
    result, published, calls = _run(
        root,
        AccountBroker(),
        None,
        lambda _root: identity,
        git_identity=identity,
        holder_probe=lambda _db: [{"pid": 5, "app": "streamlit"}],
    )
    assert result["state"] == STATE_PUBLICATION_FAILED
    assert result["reason"] == "manual_exit_sidecar_handle_held"
    assert published == []
    assert _wal(store).is_file()
    assert _shm(store).is_file()
    _assert_reads_only(calls)


def test_unrelated_child_and_stale_evidence_still_block(tmp_path, monkeypatch):
    root, store = _recovered(tmp_path / "unrelated")
    (store / "notes.txt").write_text("unrelated", encoding="utf-8")
    called = []
    monkeypatch.setattr(
        "services.paper_sell_arm_store.close_manual_exit_sqlite",
        lambda path: called.append(path.name),
    )
    result, published, calls = _run(root, AccountBroker(), None, lambda _root: _valid_git())
    assert result["state"] == STATE_REFUSED
    assert result["reason"] == "manual_exit_gate"
    assert published == []
    assert called == []
    assert (store / "notes.txt").is_file()
    _assert_reads_only(calls)

    root, store = _recovered(tmp_path / "stale")
    identity_path = next((store / "order_identities").glob("*.json"))
    payload = identity_path.read_text(encoding="utf-8")
    identity_path.write_text(payload.replace("filled", "pending_new"), encoding="utf-8")
    result, published, calls = _run(root, AccountBroker(), None, lambda _root: _valid_git())
    assert result["state"] == STATE_REFUSED
    assert result["reason"] == "manual_exit_gate"
    assert classify_manual_exit_close_preflight(store)["reason"] == "stale_or_mismatched_state"
    assert published == []
    _assert_reads_only(calls)


def test_nonzero_wal_is_not_a_deferred_cleanup_candidate(tmp_path):
    root, store = _recovered(tmp_path)
    _wal(store).write_bytes(b"frame")
    preflight = classify_manual_exit_close_preflight(store)
    assert preflight["ok"] is False
    assert preflight["deferred_sidecar_cleanup"] is False
    assert preflight["reason"] == "sqlite_sidecar_unexplained"
    result, published, calls = _run(root, AccountBroker(), None, lambda _root: _valid_git())
    assert result["state"] == STATE_REFUSED
    assert result["reason"] == "manual_exit_gate"
    assert published == []
    assert _wal(store).read_bytes() == b"frame"
    _assert_reads_only(calls)


def test_report_read_uses_immutable_uri_and_creates_no_sidecar(tmp_path, monkeypatch):
    db = tmp_path / STORE_DB_NAME
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE sell_authorities (authority_id TEXT)")
    conn.execute("INSERT INTO sell_authorities VALUES ('auth-1')")
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    for path in (Path(str(db) + "-wal"), Path(str(db) + "-shm")):
        if path.exists():
            path.unlink()
    captured = {}
    real_connect = sqlite3.connect

    def spy(database, *args, **kwargs):
        captured["uri"] = database
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr("services.exit_source_reporting.sqlite3.connect", spy)
    rows = _readonly_rows(db, "SELECT authority_id FROM sell_authorities")
    assert rows == [{"authority_id": "auth-1"}]
    assert captured["uri"] == immutable_readonly_sqlite_uri(db)
    assert "immutable=1" in captured["uri"]
    assert not Path(str(db) + "-wal").exists()
    assert not Path(str(db) + "-shm").exists()


def test_clean_store_still_publishes_without_sidecar_cleanup(tmp_path, monkeypatch):
    root, store = _recovered(tmp_path)
    called = []
    monkeypatch.setattr(
        "services.paper_sell_arm_store.close_manual_exit_sqlite",
        lambda path: called.append(path.name) or {"ok": True, "removed": []},
    )
    result, published, calls = _run(root, AccountBroker(), None, lambda _root: _valid_git())
    assert result["state"] == STATE_CERTIFIED
    assert result["deferred_sidecar_cleanup"] is False
    assert published == ["2026-10-16"]
    assert called == []
    assert evaluate_manual_exit_close_gate(store)["ok"] is True
    _assert_reads_only(calls)


def test_duplicate_session_close_does_not_clean_or_publish(tmp_path, monkeypatch):
    root, store = _recovered(tmp_path)
    _leave_residual_sidecars(store)
    called = []
    monkeypatch.setattr(
        "services.paper_sell_arm_store.close_manual_exit_sqlite",
        lambda path: called.append(path.name),
    )
    result, published, calls = _run(
        root,
        AccountBroker(),
        None,
        lambda _root: _valid_git(),
        duplicate_close=True,
    )
    assert result["state"] == STATE_ALREADY_CERTIFIED
    assert result["writes_performed"] is False
    assert published == []
    assert called == []
    assert _wal(store).is_file()
    assert _shm(store).is_file()
    _assert_reads_only(calls)


def test_recheck_failure_refuses_publication(tmp_path, monkeypatch):
    root, store = _recovered(tmp_path)
    _leave_residual_sidecars(store)
    identity = _store_identity(root)
    real = evaluate_manual_exit_close_gate
    calls = {"n": 0}

    def wrapped(path):
        calls["n"] += 1
        result = real(path)
        if calls["n"] >= 2:
            return {
                "ok": False,
                "reason": "stale_or_mismatched_state",
                "failures": [{"path": STORE_DB_NAME, "reason": "stale_or_mismatched_state"}],
                "verified_broker_order_ids": [],
            }
        return result

    monkeypatch.setattr("services.manual_portfolio_exit.evaluate_manual_exit_close_gate", wrapped)
    result, published, broker_calls = _run(
        root, AccountBroker(), None, lambda _root: _valid_git(), git_identity=identity
    )
    assert result["state"] == STATE_PUBLICATION_FAILED
    assert result["reason"] == "manual_exit_gate"
    assert result["writes_performed"] is False
    assert published == []
    assert not _wal(store).exists()
    assert not _shm(store).exists()
    _assert_reads_only(broker_calls)
    assert close_manual_exit_sqlite.__name__ == "close_manual_exit_sqlite"
