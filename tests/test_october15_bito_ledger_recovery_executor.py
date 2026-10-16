"""Temporary-path tests for the disabled October 15 BITO recovery executor."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest
import requests

import services.manual_exit_ledger_recovery as recovery
from services.manual_exit_ledger_recovery import (
    AUTHORIZATION_RELATIVE,
    EVENT_ACTION_SUBMITTED,
    EXECUTOR_ENABLED,
    EXPECTED_ACTION_SUBMITTED_EVENT_ID,
    EXPECTED_FILL_SELL_EVENT_ID,
    EXPECTED_TERMINAL_EVENT_ID,
    GOOGL_BROKER_ORDER_ID,
    TARGET_AUTHORITY_ID,
    TARGET_BROKER_ORDER_ID,
    TARGET_CLIENT_ORDER_ID,
    TARGET_CREATED_AT,
    TARGET_DISPATCH_ID,
    TARGET_FILLED_AT,
    TARGET_FILL_TIMESTAMP,
    TARGET_REQUEST_ID,
    TARGET_SUBMITTED_AT,
    TARGET_UPDATED_AT,
    execute_october15_bito_ledger_recovery,
)
from services.paper_order_ledger import (
    EVENT_ACTION_SUBMITTED as PAPER_ACTION_SUBMITTED,
    append_events,
    as_utc_iso,
    load_events,
    manual_exit_submission_event,
)
from services.performance_accounting import append_events as append_accounting_events
from services.performance_accounting import build_event

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = (
    ROOT / "data" / "results" / "paper_order_events.jsonl",
    ROOT / "data" / "results" / "performance_accounting_events.jsonl",
    ROOT / "data" / "ledger" / "ledger.parquet",
    ROOT / "data" / "governance" / "basis_unproven_hold_audit.jsonl",
    ROOT / "data" / "governance" / "basis_unproven_hold.json",
)
PRODUCTION_AUTHORIZATION = ROOT / AUTHORIZATION_RELATIVE
DECOY_REQUEST_ID = "a" * 64
DECOY_DISPATCH_ID = "sell-dispatch-decoy"
PRIOR_BROKER_ORDER_ID = "prior-order-keep"
NONCE = "nonce-7f3c9a1e-one-time"


def _fingerprint(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


@pytest.fixture(scope="module", autouse=True)
def production_files_unchanged():
    assert EXECUTOR_ENABLED is False
    assert not PRODUCTION_AUTHORIZATION.exists()
    before = {path: _fingerprint(path) for path in PRODUCTION}
    yield
    after = {path: _fingerprint(path) for path in PRODUCTION}
    assert after == before
    assert not PRODUCTION_AUTHORIZATION.exists()
    assert "EXECUTOR_ENABLED = False" in (
        ROOT / "services" / "manual_exit_ledger_recovery.py"
    ).read_text(encoding="utf-8")


def _guard_external_calls(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("broker or parquet call")

    monkeypatch.setattr("services.ledger.save_ledger", boom)
    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setattr("services.broker_alpaca.AlpacaBroker.__init__", boom)
    monkeypatch.setattr("services.broker_alpaca.AlpacaBroker.submit_order", boom)
    monkeypatch.setattr("services.broker_alpaca.AlpacaBroker.get_order", boom)
    monkeypatch.setattr("services.broker_alpaca.AlpacaBroker.list_orders", boom)
    monkeypatch.setattr("services.broker_alpaca.AlpacaBroker._post", boom)
    monkeypatch.setattr(requests.sessions.Session, "request", boom)
    monkeypatch.setattr(
        "services.manual_exit_ledger_recovery._scan_repo_runtime_processes",
        boom,
    )


def _sha256(path: Path) -> str:
    data = path.read_bytes() if path.is_file() else b""
    return hashlib.sha256(data).hexdigest()


def _write_manifest(root: Path, nonce: str = NONCE, **overrides) -> None:
    payload = {
        "executor_baseline_commit": recovery.APPROVED_HEAD,
        "execution_commit": recovery.APPROVED_HEAD,
        "broker_order_id": recovery.TARGET_BROKER_ORDER_ID,
        "client_order_id": recovery.TARGET_CLIENT_ORDER_ID,
        "request_id": recovery.TARGET_REQUEST_ID,
        "authority_id": recovery.TARGET_AUTHORITY_ID,
        "dispatch_id": recovery.TARGET_DISPATCH_ID,
        "symbol": recovery.TARGET_SYMBOL,
        "quantity": format(recovery.TARGET_QTY, "f"),
        "fill_price": format(recovery.TARGET_FILL_PRICE, "f"),
        "fill_timestamp": recovery.TARGET_FILL_TIMESTAMP,
        "action_submitted_event_id": recovery.EXPECTED_ACTION_SUBMITTED_EVENT_ID,
        "order_terminal_reconciled_event_id": recovery.EXPECTED_TERMINAL_EVENT_ID,
        "fill_sell_event_id": recovery.EXPECTED_FILL_SELL_EVENT_ID,
        "paper_order_events_sha256": _sha256(
            root / "data" / "results" / "paper_order_events.jsonl"
        ),
        "performance_accounting_events_sha256": _sha256(
            root / "data" / "results" / "performance_accounting_events.jsonl"
        ),
        "nonce": nonce,
    }
    payload.update(overrides)
    _write_json(root / AUTHORIZATION_RELATIVE, payload)


def _target_row(**overrides) -> dict:
    row = {
        "snapshot_ts": "2026-10-15T23:39:00Z",
        "id": TARGET_BROKER_ORDER_ID,
        "client_order_id": TARGET_CLIENT_ORDER_ID,
        "created_at": TARGET_CREATED_AT,
        "updated_at": TARGET_UPDATED_AT,
        "submitted_at": TARGET_SUBMITTED_AT,
        "filled_at": TARGET_FILLED_AT,
        "symbol": "BITO",
        "qty": "450",
        "filled_qty": "450",
        "filled_avg_price": "11.01",
        "side": "sell",
        "status": "filled",
    }
    row.update(overrides)
    return row


def _write_snapshot(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _receipt(request_id: str = TARGET_REQUEST_ID, **overrides) -> dict:
    payload = {
        "broker_order_id": TARGET_BROKER_ORDER_ID,
        "broker_status": "pending_new",
        "idempotent_replay": False,
        "orders_submitted": True,
        "reason": "Manual operator exit / profit-taking",
        "request_id": request_id,
        "selected_qty": "450",
        "status": "submitted",
        "submitted": True,
        "symbol": "BITO",
    }
    payload.update(overrides)
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _seed_dispatch(root: Path) -> None:
    from services.paper_sell_arm_store import write_connection

    database = (
        root / "data" / "governance" / "manual_portfolio_exit" / "manual_portfolio_exit.sqlite3"
    )
    database.parent.mkdir(parents=True, exist_ok=True)
    now = "2026-10-15T16:14:33.044271+00:00"
    expires = "2026-10-15T16:24:33.044271+00:00"
    scope = '[{"symbol":"BITO","action":"EXIT","max_qty":450.0}]'
    with write_connection(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _insert_authority(
            connection,
            challenge_id="manual-chal-0f628c831b6e31d43583ceb6",
            authority_id=TARGET_AUTHORITY_ID,
            dispatch_id=TARGET_DISPATCH_ID,
            batch_hash="bito-batch",
            scope=scope,
            symbols='["BITO"]',
            now=now,
            expires=expires,
            challenge_state="OPEN",
        )
        _insert_authority(
            connection,
            challenge_id="manual-chal-decoy",
            authority_id="manual-auth-decoy",
            dispatch_id=DECOY_DISPATCH_ID,
            batch_hash="decoy-batch",
            scope='[{"symbol":"AAPL","action":"EXIT","max_qty":1.0}]',
            symbols='["AAPL"]',
            now=now,
            expires=expires,
            challenge_state="CONSUMED",
        )
        connection.execute("COMMIT")


def _insert_authority(
    connection,
    *,
    challenge_id: str,
    authority_id: str,
    dispatch_id: str,
    batch_hash: str,
    scope: str,
    symbols: str,
    now: str,
    expires: str,
    challenge_state: str,
) -> None:
    connection.execute(
        """
        INSERT INTO sell_challenges(
            challenge_id, plan_hash, batch_hash, batch_id, canonical_batch_json,
            payload_json, created_at, expires_at, state, operator_action_label, schema_version
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            challenge_id,
            "plan",
            batch_hash,
            challenge_id,
            scope,
            "{}",
            now,
            expires,
            challenge_state,
            "manual_portfolio_exit",
            "198I-YR-SQLITE-2",
        ),
    )
    connection.execute(
        """
        INSERT INTO sell_authorities(
            authority_id, challenge_id, plan_hash, batch_hash, canonical_batch_json,
            payload_json, created_at, expires_at, single_use, state, consumption_time,
            consuming_dispatch_id, final_result, schema_version
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            authority_id,
            challenge_id,
            "plan",
            batch_hash,
            scope,
            "{}",
            now,
            expires,
            1,
            "CONSUMED",
            now,
            dispatch_id,
            "DISPATCH_INTENT",
            "198I-YR-SQLITE-2",
        ),
    )
    connection.execute(
        """
        INSERT INTO sell_dispatches(
            dispatch_id, authority_id, plan_hash, batch_hash, canonical_batch_json,
            intent_json, created_at, state, broker_ref, symbols_submitted_json, schema_version
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            dispatch_id,
            authority_id,
            "plan",
            batch_hash,
            scope,
            "{}",
            now,
            "SUBMITTED",
            None,
            symbols,
            "198I-YR-SQLITE-2",
        ),
    )


def _prepare(root: Path, *, snapshot=True, receipt=True, dispatch=True, authorization=True):
    store = root / "data" / "governance" / "manual_portfolio_exit"
    parquet = root / "data" / "ledger" / "ledger.parquet"
    hold = root / "data" / "governance" / "basis_unproven_hold.json"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    hold.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"do-not-touch")
    hold.write_text('{"status":"ACTIVE"}', encoding="utf-8")
    (root / "data" / "live").mkdir(parents=True, exist_ok=True)
    (root / "data" / "live" / "continuous_loop.lock").write_text("0", encoding="utf-8")
    recent = root / "data" / "results" / "recent_orders.csv"
    if snapshot:
        _write_snapshot(recent, _target_row())
    receipt_path = store / "requests" / f"{TARGET_REQUEST_ID}.json"
    decoy_path = store / "requests" / f"{DECOY_REQUEST_ID}.json"
    if receipt:
        _write_json(receipt_path, _receipt())
    _write_json(
        decoy_path,
        _receipt(DECOY_REQUEST_ID, symbol="AAPL", broker_order_id="other-order", selected_qty="1"),
    )
    if dispatch:
        _seed_dispatch(root)
    del authorization
    return {
        "parquet": parquet,
        "hold": hold,
        "receipt": receipt_path,
        "decoy": decoy_path,
        "recent": recent,
        "orders": root / "data" / "results" / "paper_order_events.jsonl",
        "accounting": root / "data" / "results" / "performance_accounting_events.jsonl",
    }


def _call(
    root: Path,
    monkeypatch,
    *,
    authorize=True,
    head=None,
    execution_commit=None,
    manifest_overrides=None,
    **overrides,
):
    _guard_external_calls(monkeypatch)
    pinned = recovery.APPROVED_HEAD if execution_commit is None else execution_commit
    monkeypatch.setattr(
        "services.manual_exit_ledger_recovery._repository_head",
        lambda: pinned if head is None else head,
    )
    nonce = NONCE
    fields = dict(manifest_overrides or {})
    if manifest_overrides and manifest_overrides.get("nonce"):
        nonce = str(manifest_overrides["nonce"])
    fields.setdefault("execution_commit", pinned)
    if authorize:
        _write_manifest(root, nonce=nonce, **fields)
    arguments = {
        "broker_order_id": TARGET_BROKER_ORDER_ID,
        "client_order_id": TARGET_CLIENT_ORDER_ID,
        "request_id": TARGET_REQUEST_ID,
        "authority_id": TARGET_AUTHORITY_ID,
        "dispatch_id": TARGET_DISPATCH_ID,
        "symbol": "BITO",
        "quantity": "450",
        "fill_price": "11.01",
        "fill_timestamp": TARGET_FILL_TIMESTAMP,
        "authorization_token": nonce,
        "root": root,
    }
    arguments.update(overrides)
    return execute_october15_bito_ledger_recovery(**arguments)


def _dispatch_row(root: Path, dispatch_id: str):
    database = (
        root / "data" / "governance" / "manual_portfolio_exit" / "manual_portfolio_exit.sqlite3"
    )
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            """
            SELECT d.state, d.broker_ref, d.outcome_at, d.result_class, a.final_result, a.state AS authority_state
            FROM sell_dispatches d
            JOIN sell_authorities a ON a.authority_id = d.authority_id
            WHERE d.dispatch_id=?
            """,
            (dispatch_id,),
        ).fetchone()
    finally:
        connection.close()


def _audit_count(root: Path) -> int:
    database = (
        root / "data" / "governance" / "manual_portfolio_exit" / "manual_portfolio_exit.sqlite3"
    )
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM sell_audit_events").fetchone()[0])
    finally:
        connection.close()


def _plant_prior_events(root: Path) -> None:
    prior_order = manual_exit_submission_event(
        symbol="AAPL",
        qty=Decimal("1"),
        client_order_id="manual-AAPL-prior",
        broker_order_id=PRIOR_BROKER_ORDER_ID,
        authority_id="manual-auth-prior",
        dispatch_id="sell-dispatch-prior",
        event_timestamp="2026-10-01T15:00:00Z",
    )
    append_events([prior_order], root=root)
    prior_fill = build_event(
        "FILL_SELL",
        event_timestamp="2026-10-01T15:00:00+00:00",
        source="paper_order_ledger",
        body={
            "symbol": "AAPL",
            "qty": "1.000000000000",
            "price": "10.000000",
            "broker_order_id": PRIOR_BROKER_ORDER_ID,
            "broker_fill_id": "prior-fill",
            "client_order_id": "manual-AAPL-prior",
        },
    )
    append_accounting_events([prior_fill], root=root)


def test_simulated_execution_appends_only_the_three_target_events(tmp_path, monkeypatch):
    paths = _prepare(tmp_path)
    _plant_prior_events(tmp_path)
    before = {
        "parquet": paths["parquet"].read_bytes(),
        "hold": paths["hold"].read_bytes(),
        "decoy": paths["decoy"].read_bytes(),
        "audits": _audit_count(tmp_path),
    }
    report = _call(tmp_path, monkeypatch)
    assert report["ok"] is True
    assert report["wrote"] is True
    assert report["reason"] == "recovered"
    assert report["broker_called"] is False
    assert report["fee"] is None
    assert report["commission"] is None
    assert report["net_realized_pl"] is None
    assert report["realized_pl"] is None
    assert report["parquet_modified"] is False
    assert report["event_ids"] == {
        PAPER_ACTION_SUBMITTED: EXPECTED_ACTION_SUBMITTED_EVENT_ID,
        "ORDER_TERMINAL_RECONCILED": EXPECTED_TERMINAL_EVENT_ID,
        "FILL_SELL": EXPECTED_FILL_SELL_EVENT_ID,
    }
    orders = load_events(paths["orders"])
    assert [event["event_id"] for event in orders[1:]] == [
        EXPECTED_ACTION_SUBMITTED_EVENT_ID,
        EXPECTED_TERMINAL_EVENT_ID,
    ]
    assert orders[0]["broker_order_id"] == PRIOR_BROKER_ORDER_ID
    assert orders[1]["event_type"] == EVENT_ACTION_SUBMITTED
    assert orders[1]["logical_key"] == f"broker:{TARGET_BROKER_ORDER_ID}"
    assert orders[2]["broker_terminal_status"] == "FILLED"
    assert Decimal(str(orders[2]["filled_qty"])) == Decimal("450")
    assert Decimal(str(orders[2]["average_fill_price"])) == Decimal("11.01")
    accounting = [
        json.loads(line) for line in paths["accounting"].read_text(encoding="utf-8").splitlines()
    ]
    assert accounting[0]["body"]["broker_order_id"] == PRIOR_BROKER_ORDER_ID
    assert accounting[1]["event_id"] == EXPECTED_FILL_SELL_EVENT_ID
    assert accounting[1]["event_type"] == "FILL_SELL"
    assert "realized_pl" not in accounting[1]["body"]
    assert "fee" not in accounting[1]["body"]
    assert "commission" not in accounting[1]["body"]
    assert GOOGL_BROKER_ORDER_ID not in paths["accounting"].read_text(encoding="utf-8")
    saved = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert saved["broker_status"] == "filled"
    assert saved["status"] == "submitted"
    assert saved["reconciliation_state"] == "terminal_reconciled"
    assert saved["request_id"] == TARGET_REQUEST_ID
    assert saved["symbol"] == "BITO"
    assert "idempotent_replay" not in saved
    assert "realized_pl" not in saved
    assert paths["decoy"].read_bytes() == before["decoy"]
    dispatch = _dispatch_row(tmp_path, TARGET_DISPATCH_ID)
    assert dispatch["state"] == "SUBMITTED"
    assert dispatch["broker_ref"] == TARGET_BROKER_ORDER_ID
    assert dispatch["outcome_at"] is None
    assert dispatch["result_class"] is None
    assert dispatch["final_result"] == "DISPATCH_INTENT"
    assert dispatch["authority_state"] == "CONSUMED"
    decoy = _dispatch_row(tmp_path, DECOY_DISPATCH_ID)
    assert decoy["broker_ref"] is None
    assert decoy["final_result"] == "DISPATCH_INTENT"
    assert _audit_count(tmp_path) == before["audits"]
    assert paths["parquet"].read_bytes() == before["parquet"]
    assert paths["hold"].read_bytes() == before["hold"]
    assert report["manifest_consumed"] is True
    assert not (tmp_path / AUTHORIZATION_RELATIVE).exists()
    assert recovery.EXECUTOR_ENABLED is False


def test_rerun_does_not_append_again(tmp_path, monkeypatch):
    paths = _prepare(tmp_path)
    first = _call(tmp_path, monkeypatch)
    assert first["wrote"] is True
    assert not (tmp_path / AUTHORIZATION_RELATIVE).exists()
    order_bytes = paths["orders"].read_bytes()
    accounting_bytes = paths["accounting"].read_bytes()
    receipt_bytes = paths["receipt"].read_bytes()
    second = _call(tmp_path, monkeypatch)
    assert second["ok"] is False
    assert second["wrote"] is False
    assert second["reason"] == "target_event_exists"
    assert (tmp_path / AUTHORIZATION_RELATIVE).is_file()
    assert paths["orders"].read_bytes() == order_bytes
    assert paths["accounting"].read_bytes() == accounting_bytes
    assert paths["receipt"].read_bytes() == receipt_bytes
    assert _dispatch_row(tmp_path, TARGET_DISPATCH_ID)["broker_ref"] == TARGET_BROKER_ORDER_ID


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("broker_order_id", "other-order", "not_recovery_target"),
        ("broker_order_id", GOOGL_BROKER_ORDER_ID, "not_recovery_target"),
        ("client_order_id", "manual-BITO-other", "target_mismatch"),
        ("request_id", "f" * 64, "target_mismatch"),
        ("authority_id", "manual-auth-other", "target_mismatch"),
        ("dispatch_id", "sell-dispatch-other", "target_mismatch"),
        ("symbol", "GOOGL", "target_mismatch"),
        ("symbol", "VOO", "target_mismatch"),
        ("symbol", "XLK", "target_mismatch"),
        ("quantity", "449", "target_mismatch"),
        ("fill_price", "11.02", "target_mismatch"),
        ("fill_timestamp", "2026-10-15T16:14:34+00:00", "target_mismatch"),
        ("authorization_token", "wrong-token", "authorization_mismatch"),
    ],
)
def test_identity_mismatch_writes_nothing(tmp_path, monkeypatch, field, value, reason):
    paths = _prepare(tmp_path)
    receipt_before = paths["receipt"].read_bytes()
    report = _call(tmp_path, monkeypatch, **{field: value})
    assert report["ok"] is False
    assert report["wrote"] is False
    assert report["reason"] == reason
    assert paths["orders"].exists() is False
    assert paths["accounting"].exists() is False
    assert paths["receipt"].read_bytes() == receipt_before
    assert (tmp_path / AUTHORIZATION_RELATIVE).is_file()
    assert _dispatch_row(tmp_path, TARGET_DISPATCH_ID)["broker_ref"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"client_order_id": "manual-BITO-other"},
        {"symbol": "GOOGL"},
        {"symbol": "VOO"},
        {"symbol": "XLK"},
        {"qty": "449", "filled_qty": "449"},
        {"filled_avg_price": "10.00"},
        {"filled_at": "2026-10-16T16:14:33.914668051Z"},
        {"status": "canceled"},
        {"id": "other-order"},
        {"id": GOOGL_BROKER_ORDER_ID},
    ],
)
def test_snapshot_identity_mismatch_writes_nothing(tmp_path, monkeypatch, overrides):
    paths = _prepare(tmp_path, snapshot=False)
    row = _target_row(**overrides)
    _write_snapshot(paths["recent"], row)
    report = _call(tmp_path, monkeypatch)
    assert report["wrote"] is False
    assert report["reason"] in {"target_mismatch", "snapshot_row_missing", "not_recovery_target"}
    assert paths["orders"].exists() is False
    assert _dispatch_row(tmp_path, TARGET_DISPATCH_ID)["broker_ref"] is None


_OBSERVED_TIMESTAMPS = {
    "created_at": "2026-10-15T16:14:33.48799Z",
    "submitted_at": "2026-10-15T16:14:33.495871Z",
    "filled_at": "2026-10-15T16:14:33.914668Z",
    "updated_at": "2026-10-15T16:14:33.916512Z",
}


def test_observed_truncated_timestamps_preserve_approved_event_ids(tmp_path, monkeypatch):
    paths = _prepare(tmp_path, snapshot=False)
    _write_snapshot(paths["recent"], _target_row(**_OBSERVED_TIMESTAMPS))
    report = _call(tmp_path, monkeypatch)
    assert report["ok"] is True
    assert report["wrote"] is True
    assert report["reason"] == "recovered"
    orders = load_events(paths["orders"])
    terminal = orders[-1]
    assert [event["event_id"] for event in orders] == [
        EXPECTED_ACTION_SUBMITTED_EVENT_ID,
        EXPECTED_TERMINAL_EVENT_ID,
    ]
    assert terminal["event_type"] == "ORDER_TERMINAL_RECONCILED"
    assert terminal["event_id"] == EXPECTED_TERMINAL_EVENT_ID
    assert terminal["filled_timestamp"] == as_utc_iso(TARGET_FILLED_AT)
    accounting = [
        json.loads(line) for line in paths["accounting"].read_text(encoding="utf-8").splitlines()
    ]
    assert accounting[-1]["event_type"] == "FILL_SELL"
    assert accounting[-1]["event_id"] == EXPECTED_FILL_SELL_EVENT_ID
    assert report["event_ids"] == {
        PAPER_ACTION_SUBMITTED: EXPECTED_ACTION_SUBMITTED_EVENT_ID,
        "ORDER_TERMINAL_RECONCILED": EXPECTED_TERMINAL_EVENT_ID,
        "FILL_SELL": EXPECTED_FILL_SELL_EVENT_ID,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"created_at": "2026-10-15T16:14:33.487990Z"},
        {"submitted_at": "2026-10-15T16:14:33.4958710Z"},
        {"filled_at": "2026-10-15T16:14:33.9146680Z"},
        {"updated_at": "2026-10-15T16:14:33.9165120Z"},
        {"created_at": "2026-10-15T16:14:33.48798992Z"},
        {"submitted_at": "2026-10-15T16:14:33.495870609Z"},
        {"filled_at": "2026-10-15T16:14:33.914668051Z"},
        {"updated_at": "2026-10-15T16:14:33.916512203Z"},
        {"filled_at": "2026-10-15T16:14:33.914669Z"},
        {"filled_at": "2026-10-15T16:14:33.914667Z"},
        {"created_at": "2026-10-15T16:14:33.487991Z"},
        {"submitted_at": "2026-10-15T16:14:33.495872Z"},
        {"filled_at": "2026-10-15T16:14:33.9146685Z"},
        {"filled_at": "2026-10-15T16:14:33.9146681Z"},
        {"updated_at": "2026-10-15T16:14:33.916513Z"},
        {"filled_at": "2026-10-16T16:14:33.914668Z"},
        {"filled_at": "2026-10-15T16:14:34.914668Z"},
        {"filled_at": "2026-10-15T16:14:33.914668-04:00"},
        {"filled_at": "2026-10-15T12:14:33.914668-04:00"},
        {"id": "other-order"},
        {"id": GOOGL_BROKER_ORDER_ID},
    ],
)
def test_timestamp_precision_mismatch_writes_nothing(tmp_path, monkeypatch, overrides):
    paths = _prepare(tmp_path, snapshot=False)
    row = _target_row(**_OBSERVED_TIMESTAMPS)
    row.update(overrides)
    _write_snapshot(paths["recent"], row)
    report = _call(tmp_path, monkeypatch)
    assert report["ok"] is False
    assert report["wrote"] is False
    assert report["reason"] in {"target_mismatch", "snapshot_row_missing", "not_recovery_target"}
    assert paths["orders"].exists() is False
    assert paths["accounting"].exists() is False
    assert _dispatch_row(tmp_path, TARGET_DISPATCH_ID)["broker_ref"] is None


@pytest.mark.parametrize(
    "gap",
    ["snapshot", "snapshot_row", "receipt", "dispatch", "authorization"],
)
def test_missing_evidence_writes_nothing(tmp_path, monkeypatch, gap):
    paths = _prepare(
        tmp_path,
        snapshot=gap != "snapshot",
        receipt=gap != "receipt",
        dispatch=gap != "dispatch",
        authorization=gap != "authorization",
    )
    if gap == "snapshot_row":
        _write_snapshot(paths["recent"], _target_row(id="missing-target"))
    receipt_before = paths["receipt"].read_bytes() if paths["receipt"].is_file() else None
    report = _call(tmp_path, monkeypatch, authorize=gap != "authorization")
    assert report["ok"] is False
    assert report["wrote"] is False
    expected = {
        "snapshot": "snapshot_missing",
        "snapshot_row": "snapshot_row_missing",
        "receipt": "receipt_missing",
        "dispatch": "dispatch_missing",
        "authorization": "authorization_missing",
    }
    assert report["reason"] == expected[gap]
    assert paths["orders"].exists() is False
    assert paths["accounting"].exists() is False
    if receipt_before is not None:
        assert paths["receipt"].read_bytes() == receipt_before
    if gap != "dispatch":
        assert _dispatch_row(tmp_path, TARGET_DISPATCH_ID)["broker_ref"] is None
    assert not (tmp_path / AUTHORIZATION_RELATIVE).exists() or gap != "authorization"


def test_existing_submission_event_writes_nothing(tmp_path, monkeypatch):
    paths = _prepare(tmp_path)
    event = manual_exit_submission_event(
        symbol="BITO",
        qty=Decimal("450"),
        client_order_id=TARGET_CLIENT_ORDER_ID,
        broker_order_id=TARGET_BROKER_ORDER_ID,
        authority_id=TARGET_AUTHORITY_ID,
        dispatch_id=TARGET_DISPATCH_ID,
        event_timestamp=TARGET_SUBMITTED_AT,
    )
    append_events([event], root=tmp_path)
    before = paths["orders"].read_bytes()
    report = _call(tmp_path, monkeypatch)
    assert report["reason"] == "target_event_exists"
    assert report["wrote"] is False
    assert "ACTION_SUBMITTED" in report["details"]["events"]
    assert paths["orders"].read_bytes() == before
    assert (
        json.loads(paths["receipt"].read_text(encoding="utf-8"))["broker_status"] == "pending_new"
    )
    assert _dispatch_row(tmp_path, TARGET_DISPATCH_ID)["broker_ref"] is None


def test_existing_fill_event_writes_nothing(tmp_path, monkeypatch):
    paths = _prepare(tmp_path)
    paths["accounting"].parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "event_type": "FILL_SELL",
            "logical_key": f"fill:{TARGET_BROKER_ORDER_ID}:already",
        }
    )
    paths["accounting"].write_text(line + "\n", encoding="utf-8")
    before = paths["accounting"].read_bytes()
    report = _call(tmp_path, monkeypatch)
    assert report["reason"] == "target_event_exists"
    assert report["wrote"] is False
    assert paths["accounting"].read_bytes() == before
    assert paths["orders"].exists() is False


def test_runtime_lock_writes_nothing(tmp_path, monkeypatch):
    paths = _prepare(tmp_path)
    lock = tmp_path / "data" / "live" / "watchdog_loop.lock"
    lock.write_text("pid=4242\n", encoding="utf-8")
    report = _call(tmp_path, monkeypatch)
    assert report["reason"] == "runtime_lock"
    assert report["wrote"] is False
    assert paths["orders"].exists() is False
    assert (
        json.loads(paths["receipt"].read_text(encoding="utf-8"))["broker_status"] == "pending_new"
    )


def test_runtime_process_writes_nothing(tmp_path, monkeypatch):
    paths = _prepare(tmp_path)
    monkeypatch.setattr(
        "services.manual_exit_ledger_recovery._active_runtime_processes",
        lambda _root: {"ok": True, "pids": [4242]},
    )
    report = _call(tmp_path, monkeypatch)
    assert report["reason"] == "runtime_process_active"
    assert report["wrote"] is False
    assert paths["orders"].exists() is False
    assert paths["accounting"].exists() is False


def test_runtime_scan_failure_writes_nothing(tmp_path, monkeypatch):
    paths = _prepare(tmp_path)
    monkeypatch.setattr(
        "services.manual_exit_ledger_recovery._active_runtime_processes",
        lambda _root: {"ok": False, "reason": "runtime_process_scan_failed"},
    )
    report = _call(tmp_path, monkeypatch)
    assert report["reason"] == "runtime_process_scan_failed"
    assert report["wrote"] is False
    assert paths["orders"].exists() is False


def test_event_identity_mismatch_writes_nothing(tmp_path, monkeypatch):
    paths = _prepare(tmp_path)
    monkeypatch.setattr(
        "services.manual_exit_ledger_recovery.EXPECTED_ACTION_SUBMITTED_EVENT_ID",
        "0" * 64,
    )
    report = _call(tmp_path, monkeypatch)
    assert report["reason"] == "event_identity_mismatch"
    assert report["wrote"] is False
    assert paths["orders"].exists() is False
    assert paths["accounting"].exists() is False


def test_default_denial_without_manifest(tmp_path, monkeypatch):
    paths = _prepare(tmp_path)
    monkeypatch.setattr("services.manual_exit_ledger_recovery.EXECUTOR_ENABLED", True)
    report = _call(tmp_path, monkeypatch, authorize=False, authorization_token=NONCE)
    assert report["reason"] == "authorization_missing"
    assert report["wrote"] is False
    assert recovery.WRITE_MODE_ENABLED is False
    assert paths["orders"].exists() is False
    assert not (tmp_path / AUTHORIZATION_RELATIVE).exists()
    assert not PRODUCTION_AUTHORIZATION.exists()
    assert (
        json.loads(paths["receipt"].read_text(encoding="utf-8"))["broker_status"] == "pending_new"
    )


def test_missing_invocation_token_keeps_manifest(tmp_path, monkeypatch):
    paths = _prepare(tmp_path)
    report = _call(tmp_path, monkeypatch, authorization_token="")
    assert report["reason"] == "authorization_mismatch"
    assert report["wrote"] is False
    assert paths["orders"].exists() is False
    saved = json.loads((tmp_path / AUTHORIZATION_RELATIVE).read_text(encoding="utf-8"))
    assert saved["nonce"] == NONCE


def test_changed_head_keeps_manifest(tmp_path, monkeypatch):
    paths = _prepare(tmp_path)
    report = _call(tmp_path, monkeypatch, head="a" * 40)
    assert report["reason"] == "head_mismatch"
    assert report["wrote"] is False
    assert paths["orders"].exists() is False
    assert (tmp_path / AUTHORIZATION_RELATIVE).is_file()
    assert (
        json.loads(paths["receipt"].read_text(encoding="utf-8"))["broker_status"] == "pending_new"
    )


def test_changed_ledger_hash_keeps_manifest(tmp_path, monkeypatch):
    paths = _prepare(tmp_path)
    report = _call(
        tmp_path,
        monkeypatch,
        manifest_overrides={"paper_order_events_sha256": "0" * 64},
    )
    assert report["reason"] == "ledger_hash_mismatch"
    assert report["wrote"] is False
    assert paths["orders"].exists() is False
    saved = json.loads((tmp_path / AUTHORIZATION_RELATIVE).read_text(encoding="utf-8"))
    assert saved["paper_order_events_sha256"] == "0" * 64
    assert _dispatch_row(tmp_path, TARGET_DISPATCH_ID)["broker_ref"] is None


def test_changed_manifest_identity_keeps_manifest(tmp_path, monkeypatch):
    paths = _prepare(tmp_path)
    report = _call(tmp_path, monkeypatch, manifest_overrides={"symbol": "GOOGL"})
    assert report["reason"] == "manifest_rejected"
    assert report["wrote"] is False
    assert paths["orders"].exists() is False
    saved = json.loads((tmp_path / AUTHORIZATION_RELATIVE).read_text(encoding="utf-8"))
    assert saved["symbol"] == "GOOGL"
    assert json.loads(paths["receipt"].read_text(encoding="utf-8"))["symbol"] == "BITO"


def test_execution_commit_must_descend_from_baseline(tmp_path, monkeypatch):
    import subprocess

    parent = subprocess.check_output(
        ["git", "rev-parse", f"{recovery.APPROVED_HEAD}^"],
        cwd=ROOT,
        text=True,
    ).strip()
    paths = _prepare(tmp_path)
    report = _call(tmp_path, monkeypatch, execution_commit=parent, head=parent)
    assert report["reason"] == "execution_commit_not_descendant"
    assert report["wrote"] is False
    assert paths["orders"].exists() is False
    saved = json.loads((tmp_path / AUTHORIZATION_RELATIVE).read_text(encoding="utf-8"))
    assert saved["executor_baseline_commit"] == recovery.APPROVED_HEAD
    assert saved["execution_commit"] == parent
    assert _dispatch_row(tmp_path, TARGET_DISPATCH_ID)["broker_ref"] is None


def test_manifest_pin_refuses_until_executor_source_is_committed():
    report = recovery.execution_commit_for_manifest()
    assert report["ok"] is False
    assert report["wrote"] is False
    assert report["reason"] == "executor_not_committed"
    assert not PRODUCTION_AUTHORIZATION.exists()


def test_executor_source_has_no_broker_or_parquet_writer():
    text = (ROOT / "services" / "manual_exit_ledger_recovery.py").read_text(encoding="utf-8")
    for banned in (
        "submit_order(",
        "get_order(",
        "list_orders(",
        "save_ledger",
        "paper-api",
        "AlpacaBroker",
        "urlopen",
    ):
        assert banned not in text
    assert "EXECUTOR_ENABLED = False" in text
    assert "WRITE_MODE_ENABLED = False" in text
    assert "os.environ" not in text
    assert "getenv" not in text
    assert recovery.APPROVED_HEAD in text
