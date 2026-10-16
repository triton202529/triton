"""Future manual-exit store agreement. These tests use temporary roots only."""

from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from services.basis_unproven_hold import enforce_basis_unproven_hold_on_order_post
from services.manual_portfolio_exit import (
    STORE_DB_NAME,
    confirm_manual_paper_exit,
    evaluate_manual_exit_close_gate,
    note_proven_terminal_receipts,
    build_manual_exit_review,
)
from services.paper_sell_arm_runtime_enforcement import enforce_paper_sell_broker_submission
from services.paper_sell_arm_store import (
    STORE_SCHEMA_VERSION,
    PaperSellArmStoreError,
    _audit,
    batch_hash_from_scope,
    canonical_json,
    close_manual_exit_sqlite,
    consume_authority_insert_dispatch,
    write_connection,
)

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = (
    ROOT / "data" / "results" / "paper_order_events.jsonl",
    ROOT / "data" / "results" / "performance_accounting_events.jsonl",
    ROOT / "data" / "governance" / "basis_unproven_hold_audit.jsonl",
    ROOT / "data" / "governance" / "basis_unproven_hold.json",
)
MANUAL_STORE = ROOT / "data" / "governance" / "manual_portfolio_exit"


def _fingerprint(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _tree_fingerprints(root: Path) -> dict[str, tuple[str, int]]:
    if not root.exists():
        return {}
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            found[path.relative_to(root).as_posix()] = _fingerprint(path)
    return found


@pytest.fixture(scope="module", autouse=True)
def production_files_unchanged():
    before = {path: _fingerprint(path) for path in PRODUCTION}
    before_store = _tree_fingerprints(MANUAL_STORE)
    yield
    after = {path: _fingerprint(path) for path in PRODUCTION}
    after_store = _tree_fingerprints(MANUAL_STORE)
    assert after == before
    assert after_store == before_store


class Broker:
    def __init__(self, symbol="AAPL", qty="10", price="20"):
        self.mode = "paper"
        self.base = "https://paper-api.alpaca.markets"
        self.positions = [{"symbol": symbol, "qty": qty, "current_price": price}]
        self.orders = []

    def get_positions(self):
        return list(self.positions)

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

    def get_order(self, order_id):
        raise AssertionError(order_id)


def _layout(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    store = root / "data" / "governance" / "manual_portfolio_exit"
    return root, store


def _confirm(root: Path, store: Path, broker: Broker):
    review = build_manual_exit_review(
        {"symbol": "AAPL", "qty": "10", "current_price": "20", "avg_entry_price": "12"},
        blocked_symbols=[],
    )
    result = confirm_manual_paper_exit(
        review,
        broker=broker,
        store_root=store,
        ledger_root=root,
        live_execution_permitted=False,
        blocked_symbols=[],
    )
    return review, result


def _event(result: dict) -> dict:
    return {
        "event_type": "ORDER_TERMINAL_RECONCILED",
        "terminal_status_proven": True,
        "broker_terminal_status": "FILLED",
        "broker_order_id": result["broker_order_id"],
        "client_order_id": result["client_order_id"],
        "symbol": result["symbol"],
        "filled_qty": result["selected_qty"],
        "extra": {"raw_broker_status": "filled"},
    }


def _challenge_row(store: Path):
    db_path = store / STORE_DB_NAME
    with write_connection(db_path) as conn:
        return conn.execute(
            "SELECT challenge_id, state, consuming_authority_id FROM sell_challenges"
        ).fetchone()


def test_successful_submit_consumes_challenge_with_authority(tmp_path):
    root, store = _layout(tmp_path)
    _review, done = _confirm(root, store, Broker())
    assert done["status"] == "submitted"
    assert done["submitted"] is True
    row = _challenge_row(store)
    assert row["state"] == "CONSUMED"
    assert row["consuming_authority_id"] == done["authority_id"]
    close_manual_exit_sqlite(store / STORE_DB_NAME)
    with write_connection(store / STORE_DB_NAME) as conn:
        authority = conn.execute(
            "SELECT state FROM sell_authorities WHERE authority_id=?",
            (done["authority_id"],),
        ).fetchone()
    assert authority["state"] == "CONSUMED"
    close_manual_exit_sqlite(store / STORE_DB_NAME)


def test_duplicate_confirm_returns_saved_result_without_second_order(tmp_path):
    root, store = _layout(tmp_path)
    broker = Broker()
    review, first = _confirm(root, store, broker)
    second = confirm_manual_paper_exit(
        review,
        broker=broker,
        store_root=store,
        ledger_root=root,
        live_execution_permitted=False,
        blocked_symbols=[],
    )
    assert first["broker_order_id"] == "ord-1"
    assert second["idempotent_replay"] is True
    assert second["broker_order_id"] == first["broker_order_id"]
    assert second["submitted"] is True
    assert len(broker.orders) == 1


def test_terminal_reconciliation_aligns_every_durable_record(tmp_path):
    root, store = _layout(tmp_path)
    _review, done = _confirm(root, store, Broker())
    before = evaluate_manual_exit_close_gate(store)
    assert before["ok"] is False
    result = note_proven_terminal_receipts([_event(done)], root=root)
    assert result["ok"] is True
    assert result["updated"] == 1
    receipt = json.loads((store / "requests" / f"{done['request_id']}.json").read_text())
    identity = json.loads(
        (store / "order_identities" / f"{done['broker_order_id']}.json").read_text()
    )
    evidence_dir = store / "dispatch_terminal"
    evidence = json.loads(next(evidence_dir.glob("*.json")).read_text())
    permit = json.loads((store / "paper_sell_arm_dispatch_permit.json").read_text())
    with write_connection(store / STORE_DB_NAME) as conn:
        dispatch = conn.execute("SELECT broker_ref, intent_json FROM sell_dispatches").fetchone()
        authority = conn.execute("SELECT payload_json FROM sell_authorities").fetchone()
    intent = json.loads(dispatch["intent_json"])
    payload = json.loads(authority["payload_json"])
    assert receipt["broker_status"] == "filled"
    assert receipt["reconciliation_state"] == "terminal_reconciled"
    assert identity["broker_status"] == "filled"
    assert identity["reconciliation_state"] == "terminal_reconciled"
    assert identity["broker_order_id"] == receipt["broker_order_id"]
    assert evidence["broker_status"] == "filled"
    assert evidence["reconciliation_state"] == "terminal_reconciled"
    assert evidence["broker_order_id"] == receipt["broker_order_id"]
    assert dispatch["broker_ref"] == receipt["broker_order_id"]
    assert intent["broker_status"] == "filled"
    assert intent["reconciliation_state"] == "terminal_reconciled"
    assert payload["broker_status"] == "filled"
    assert payload["reconciliation_state"] == "terminal_reconciled"
    assert permit["status"] == "NON_AUTHORITATIVE"
    assert permit["json_role"] == "NON_AUTHORITATIVE"
    assert permit["close_gate_evidence"] is False
    assert permit["authorizes_submission"] is False
    assert "permit_token" not in permit
    assert permit["broker_status"] == "filled"
    close_manual_exit_sqlite(store / STORE_DB_NAME)
    again = note_proven_terminal_receipts([_event(done)], root=root)
    assert again["ok"] is True
    assert again["updated"] == 0


def test_terminal_reconciliation_does_not_call_the_broker(tmp_path):
    root, store = _layout(tmp_path)
    broker = Broker()
    _review, done = _confirm(root, store, broker)
    calls = {"n": 0}

    def deny(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("broker call")

    broker.get_positions = deny
    broker.submit_order = deny
    broker.get_order = deny
    assert "broker" not in inspect.signature(note_proven_terminal_receipts).parameters
    result = note_proven_terminal_receipts([_event(done)], root=root)
    assert result["ok"] is True
    assert calls["n"] == 0
    assert len(broker.orders) == 1


def test_mismatched_identity_is_refused_without_writes(tmp_path):
    root, store = _layout(tmp_path)
    _review, done = _confirm(root, store, Broker())
    identity_path = store / "order_identities" / f"{done['broker_order_id']}.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["broker_order_id"] = "ord-other"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    receipt_path = store / "requests" / f"{done['request_id']}.json"
    before = receipt_path.read_text(encoding="utf-8")
    result = note_proven_terminal_receipts([_event(done)], root=root)
    assert result["ok"] is False
    assert result["reason"] == "manual_exit_terminal_mismatch"
    assert receipt_path.read_text(encoding="utf-8") == before
    assert json.loads(before)["broker_status"] == "accepted"
    assert json.loads(before)["reconciliation_state"] == "submission_recorded"


def test_open_challenge_blocks_terminal_alignment(tmp_path):
    root, store = _layout(tmp_path)
    _review, done = _confirm(root, store, Broker())
    db_path = store / STORE_DB_NAME
    with write_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE sell_challenges
            SET state='OPEN', consumed_at=NULL, consuming_authority_id=NULL
            """
        )
    close_manual_exit_sqlite(db_path)
    receipt_path = store / "requests" / f"{done['request_id']}.json"
    before = receipt_path.read_text(encoding="utf-8")
    result = note_proven_terminal_receipts([_event(done)], root=root)
    assert result["ok"] is False
    assert result["reason"] == "manual_exit_challenge_not_consumed"
    assert receipt_path.read_text(encoding="utf-8") == before
    with write_connection(db_path) as conn:
        state = conn.execute("SELECT state FROM sell_challenges").fetchone()["state"]
    assert state == "OPEN"
    close_manual_exit_sqlite(db_path)


def test_sqlite_sidecars_close_only_for_the_manual_exit_database(tmp_path, monkeypatch):
    root, store = _layout(tmp_path)
    _confirm(root, store, Broker())
    db_path = store / STORE_DB_NAME
    wal = Path(str(db_path) + "-wal")
    shm = Path(str(db_path) + "-shm")
    assert db_path.is_file()
    assert not wal.exists()
    assert not shm.exists()
    wal.write_bytes(b"stale")
    shm.write_bytes(b"stale")

    class _Checkpoint:
        def execute(self, _sql):
            class _Row:
                def fetchone(self):
                    return (0, 0, 0)

            return _Row()

    @contextmanager
    def _boundary(_db_path):
        yield _Checkpoint()

    monkeypatch.setattr("services.paper_sell_arm_store.write_connection", _boundary)
    closed = close_manual_exit_sqlite(db_path)
    assert closed["ok"] is True
    assert closed["removed"] == [
        "manual_portfolio_exit.sqlite3-wal",
        "manual_portfolio_exit.sqlite3-shm",
    ]
    assert not wal.exists()
    assert not shm.exists()
    other = tmp_path / "paper_sell_arm.sqlite3"
    other.write_bytes(b"")
    with pytest.raises(PaperSellArmStoreError) as exc:
        close_manual_exit_sqlite(other)
    assert exc.value.reason == "sqlite_sidecar_close_scope_refused"
    assert other.is_file()


def test_close_gate_rejects_stale_mutable_and_unrelated_children(tmp_path):
    root, store = _layout(tmp_path)
    _review, done = _confirm(root, store, Broker())
    refused = evaluate_manual_exit_close_gate(store)
    assert refused["ok"] is False
    assert "mutable_pre_submit_permit" in {item["reason"] for item in refused["failures"]}
    note_proven_terminal_receipts([_event(done)], root=root)
    accepted = evaluate_manual_exit_close_gate(store)
    assert accepted["ok"] is True
    assert accepted["verified_broker_order_ids"] == [done["broker_order_id"]]
    identity_path = store / "order_identities" / f"{done['broker_order_id']}.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["broker_status"] = "pending_new"
    identity["reconciliation_state"] = "submission_recorded"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    stale = evaluate_manual_exit_close_gate(store)
    assert stale["ok"] is False
    assert stale["reason"] == "stale_or_mismatched_state"
    note_proven_terminal_receipts([_event(done)], root=root)
    wal = Path(str(store / STORE_DB_NAME) + "-wal")
    wal.write_bytes(b"")
    sidecar = evaluate_manual_exit_close_gate(store)
    assert sidecar["ok"] is False
    assert sidecar["reason"] == "sqlite_sidecar_unexplained"
    wal.unlink()
    (store / "notes.txt").write_text("unrelated", encoding="utf-8")
    unrelated = evaluate_manual_exit_close_gate(store)
    assert unrelated["ok"] is False
    assert unrelated["reason"] == "unrelated_child"


def test_consume_rolls_back_an_open_challenge_with_its_authority(tmp_path, monkeypatch):
    db_path = tmp_path / STORE_DB_NAME
    scope = [{"symbol": "AAPL", "action": "EXIT", "max_qty": 10.0, "max_notional": 200.0}]
    batch_hash = batch_hash_from_scope(scope)
    plan_hash = "b" * 64
    now = datetime.now(timezone.utc)
    expires = (now + timedelta(minutes=10)).isoformat()
    payload = canonical_json({"lane": "manual_portfolio_exit"})
    with write_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO sell_challenges(
                challenge_id, plan_hash, batch_hash, batch_id, canonical_batch_json,
                payload_json, created_at, expires_at, state, operator_action_label,
                schema_version
            ) VALUES (?,?,?,?,?,?,?,?, 'OPEN', ?, ?)
            """,
            (
                "manual-chal-rollback",
                plan_hash,
                batch_hash,
                "manual-batch-rollback",
                canonical_json(scope),
                payload,
                now.isoformat(),
                expires,
                "MANUAL_PORTFOLIO_EXIT",
                STORE_SCHEMA_VERSION,
            ),
        )
        conn.execute(
            """
            INSERT INTO sell_authorities(
                authority_id, challenge_id, plan_hash, batch_hash, canonical_batch_json,
                payload_json, created_at, expires_at, single_use, state, schema_version
            ) VALUES (?,?,?,?,?,?,?,?,1,'UNUSED',?)
            """,
            (
                "manual-auth-rollback",
                "manual-chal-rollback",
                plan_hash,
                batch_hash,
                canonical_json(scope),
                payload,
                now.isoformat(),
                expires,
                STORE_SCHEMA_VERSION,
            ),
        )
        conn.execute("COMMIT")

    def boom(conn, event_type, **kwargs):
        if event_type == "challenge_consumed_with_authority":
            raise PaperSellArmStoreError("forced_rollback")
        _audit(conn, event_type, **kwargs)

    monkeypatch.setattr("services.paper_sell_arm_store._audit", boom)
    with pytest.raises(PaperSellArmStoreError) as exc:
        consume_authority_insert_dispatch(
            db_path,
            authority_id="manual-auth-rollback",
            plan_hash=plan_hash,
            batch_hash=batch_hash,
            expected_scope=scope,
            mode="paper",
            live_blocked=True,
            requested=[{"symbol": "AAPL", "action": "EXIT", "qty": 10.0, "side": "sell"}],
            now=now,
        )
    assert exc.value.reason == "forced_rollback"
    with write_connection(db_path) as conn:
        challenge = conn.execute(
            "SELECT state, consuming_authority_id FROM sell_challenges"
        ).fetchone()
        authority = conn.execute("SELECT state FROM sell_authorities").fetchone()
        dispatches = conn.execute("SELECT COUNT(*) FROM sell_dispatches").fetchone()[0]
    assert challenge["state"] == "OPEN"
    assert challenge["consuming_authority_id"] is None
    assert authority["state"] == "UNUSED"
    assert dispatches == 0
