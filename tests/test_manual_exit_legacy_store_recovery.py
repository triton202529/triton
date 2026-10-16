"""Temporary-root tests for the disabled manual-exit store recovery."""

from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

import services.manual_exit_legacy_store_recovery as recovery
from services.manual_exit_legacy_store_recovery import (
    EXECUTOR_ENABLED,
    WRITE_MODE_ENABLED,
    dry_run_october15_16_manual_exit_store_recovery,
    execute_legacy_store_recovery,
    execute_october15_16_manual_exit_store_recovery,
)
from services.paper_sell_arm_store import STORE_SCHEMA_VERSION, canonical_json, write_connection

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = (
    ROOT / "data" / "results" / "paper_order_events.jsonl",
    ROOT / "data" / "results" / "performance_accounting_events.jsonl",
    ROOT / "data" / "governance" / "basis_unproven_hold.json",
    ROOT / "data" / "governance" / "basis_unproven_hold_audit.jsonl",
)
MANUAL_STORE = ROOT / "data" / "governance" / "manual_portfolio_exit"
NONCE = "one-time-test-nonce"
GOOGL = "924938e3-813d-4007-9384-b36fb639cd66"


def _fingerprint(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _tree(root: Path) -> dict[str, tuple[str, int]]:
    found = {}
    if not root.exists():
        return found
    for path in sorted(root.rglob("*")):
        if path.is_file():
            found[path.relative_to(root).as_posix()] = _fingerprint(path)
    return found


@pytest.fixture(scope="module", autouse=True)
def production_files_unchanged():
    before = {path: _fingerprint(path) for path in PRODUCTION}
    before_store = _tree(MANUAL_STORE)
    yield
    assert {path: _fingerprint(path) for path in PRODUCTION} == before
    assert _tree(MANUAL_STORE) == before_store


def _paper(target: dict, kind: str) -> dict:
    terminal = kind == "ORDER_TERMINAL_RECONCILED"
    return {
        "event_type": kind,
        "event_id": (
            target["terminal_event_id"] if terminal else target["action_submitted_event_id"]
        ),
        "broker_order_id": target["broker_order_id"],
        "client_order_id": target["client_order_id"],
        "authority_id": target["authority_id"],
        "dispatch_id": target["dispatch_id"],
        "symbol": "BITO",
        "selected_qty": float(target["qty"]),
        "filled_qty": float(target["qty"]) if terminal else None,
        "broker_terminal_status": "FILLED" if terminal else None,
        "terminal_status_proven": terminal,
    }


def _fill(target: dict) -> dict:
    return {
        "event_type": "FILL_SELL",
        "event_id": target["fill_sell_event_id"],
        "body": {
            "symbol": "BITO",
            "qty": format(target["qty"], "f"),
            "broker_order_id": target["broker_order_id"],
            "client_order_id": target["client_order_id"],
        },
    }


def _receipt(target: dict) -> dict:
    payload = {
        "broker_order_id": target["broker_order_id"],
        "broker_status": "filled",
        "orders_submitted": True,
        "reason": "Manual operator exit / profit-taking",
        "reconciliation_state": "terminal_reconciled",
        "request_id": target["request_id"],
        "selected_qty": format(target["qty"], "f"),
        "status": "submitted",
        "submitted": True,
        "symbol": "BITO",
    }
    if target["receipt_linked"]:
        payload.update(
            {
                "authority_id": target["authority_id"],
                "client_order_id": target["client_order_id"],
                "dispatch_id": target["dispatch_id"],
            }
        )
    return payload


def _identity(target: dict) -> dict:
    return {
        "authority_id": target["authority_id"],
        "broker_order_id": target["broker_order_id"],
        "broker_status": "pending_new",
        "client_order_id": target["client_order_id"],
        "dispatch_id": target["dispatch_id"],
        "idempotent_replay": False,
        "orders_submitted": True,
        "reason": "Manual operator exit / profit-taking",
        "reconciliation_state": "submission_recorded",
        "request_id": target["request_id"],
        "selected_qty": format(target["qty"], "f"),
        "status": "submitted",
        "submitted": True,
        "symbol": "BITO",
    }


def _plant(root: Path) -> None:
    store = root / "data" / "governance" / "manual_portfolio_exit"
    db_path = store / "manual_portfolio_exit.sqlite3"
    with write_connection(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for target in recovery.TARGETS:
            scope = canonical_json(
                [
                    {
                        "action": "EXIT",
                        "max_qty": float(target["qty"]),
                        "symbol": "BITO",
                    }
                ]
            )
            plan_hash = hashlib.sha256(target["request_id"].encode("utf-8")).hexdigest()
            connection.execute(
                """
                INSERT INTO sell_challenges(
                    challenge_id, plan_hash, batch_hash, batch_id, canonical_batch_json,
                    payload_json, created_at, expires_at, state, operator_action_label,
                    schema_version
                ) VALUES (?,?,?,?,?,?,?,?, 'OPEN', 'MANUAL_PORTFOLIO_EXIT', ?)
                """,
                (
                    target["challenge_id"],
                    plan_hash,
                    plan_hash,
                    f"batch-{target['request_id'][:16]}",
                    scope,
                    canonical_json({"request_id": target["request_id"]}),
                    "2026-10-16T00:00:00+00:00",
                    "2026-10-17T00:00:00+00:00",
                    STORE_SCHEMA_VERSION,
                ),
            )
            connection.execute(
                """
                INSERT INTO sell_authorities(
                    authority_id, challenge_id, plan_hash, batch_hash, canonical_batch_json,
                    payload_json, created_at, expires_at, single_use, state,
                    consumption_time, consuming_dispatch_id, final_result, schema_version
                ) VALUES (?,?,?,?,?,?,?,?,1,'CONSUMED',?,?,?,?)
                """,
                (
                    target["authority_id"],
                    target["challenge_id"],
                    plan_hash,
                    plan_hash,
                    scope,
                    canonical_json(
                        {
                            "lane": "manual_portfolio_exit",
                            "reason": "Manual operator exit / profit-taking",
                            "request_id": target["request_id"],
                            "single_use": True,
                        }
                    ),
                    "2026-10-16T00:00:00+00:00",
                    "2026-10-17T00:00:00+00:00",
                    "2026-10-16T12:00:00+00:00",
                    target["dispatch_id"],
                    target["final_result"],
                    STORE_SCHEMA_VERSION,
                ),
            )
            intent = {
                "authority_id": target["authority_id"],
                "clip_audit": None,
                "dispatch_id": target["dispatch_id"],
                "recorded_at": "2026-10-16T12:00:00+00:00",
                "requested": [
                    {
                        "action": "EXIT",
                        "qty": float(target["qty"]),
                        "side": "sell",
                        "symbol": "BITO",
                    }
                ],
                "status": "INTENT_COMMITTED",
            }
            connection.execute(
                """
                INSERT INTO sell_dispatches(
                    dispatch_id, authority_id, plan_hash, batch_hash, canonical_batch_json,
                    intent_json, created_at, state, broker_ref, result_class,
                    reconciliation_required, symbols_submitted_json, schema_version
                ) VALUES (?,?,?,?,?,?,?,'SUBMITTED',?,?,0,'[]',?)
                """,
                (
                    target["dispatch_id"],
                    target["authority_id"],
                    plan_hash,
                    plan_hash,
                    scope,
                    canonical_json(intent),
                    "2026-10-16T12:00:00+00:00",
                    target["broker_order_id"],
                    target["result_class"],
                    STORE_SCHEMA_VERSION,
                ),
            )
        connection.execute("COMMIT")
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if not sidecar.exists():
            sidecar.write_bytes(b"")
    requests = store / "requests"
    requests.mkdir(parents=True, exist_ok=True)
    identities = store / "order_identities"
    identities.mkdir(parents=True, exist_ok=True)
    for target in recovery.TARGETS:
        (requests / f"{target['request_id']}.json").write_text(
            json.dumps(_receipt(target), sort_keys=True),
            encoding="utf-8",
        )
        if target["identity_present"]:
            (identities / f"{target['broker_order_id']}.json").write_text(
                json.dumps(_identity(target), sort_keys=True),
                encoding="utf-8",
            )
    permit = {
        "actions": [{"action": "EXIT", "qty": 19.0, "symbol": "BITO"}],
        "additional_symbols_forbidden": True,
        "dispatch_intent_id": recovery.TARGETS[2]["dispatch_id"],
        "idempotency_key": recovery.TARGETS[2]["request_id"],
        "json_role": "AUDIT_EXPORT_ONLY",
        "permit_token": "test-permit-token",
        "replay_forbidden": True,
        "schema_version": "198H-PAPER-SELL-ARM-1",
        "status": "PERMIT_ISSUED_PRE_SUBMIT",
        "symbols": ["BITO"],
        "symbols_submitted": ["BITO"],
    }
    (store / "paper_sell_arm_dispatch_permit.json").write_text(
        json.dumps(permit, sort_keys=True),
        encoding="utf-8",
    )
    paper = root / "data" / "results" / "paper_order_events.jsonl"
    accounting = root / "data" / "results" / "performance_accounting_events.jsonl"
    paper.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "event_type": "ACTION_SUBMITTED",
            "event_id": "googl-submitted",
            "broker_order_id": GOOGL,
            "symbol": "GOOGL",
        }
    ]
    fills = [
        {
            "event_type": "FILL_SELL",
            "event_id": "googl-fill",
            "body": {"broker_order_id": GOOGL, "symbol": "GOOGL", "qty": "1"},
        }
    ]
    for target in recovery.TARGETS:
        rows.append(_paper(target, "ACTION_SUBMITTED"))
        rows.append(_paper(target, "ORDER_TERMINAL_RECONCILED"))
        fills.append(_fill(target))
    paper.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    accounting.write_text("".join(json.dumps(row) + "\n" for row in fills), encoding="utf-8")
    hold = root / "data" / "governance" / "basis_unproven_hold.json"
    hold.write_text('{"symbols":["VOO","XLK"]}', encoding="utf-8")


def _manifest(root: Path, nonce: str = NONCE) -> None:
    hashes = recovery.baseline_hashes(root)
    assert hashes["ok"] is True
    payload = dict(hashes["hashes"])
    payload["nonce"] = nonce
    path = root / recovery.MANIFEST_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _enable(monkeypatch) -> None:
    monkeypatch.setattr(recovery, "EXECUTOR_ENABLED", True)
    monkeypatch.setattr(recovery, "WRITE_MODE_ENABLED", True)


def _guard(monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("forbidden call")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setattr("services.paper_order_ledger.append_events", boom)
    monkeypatch.setattr("services.paper_order_ledger.record_manual_exit_submission", boom)
    monkeypatch.setattr("services.performance_accounting.record_proven_terminal_fills", boom)
    monkeypatch.setattr(
        "services.performance_evidence_collector.safe_publish_session_close",
        boom,
        raising=False,
    )
    monkeypatch.setattr("services.broker_alpaca.AlpacaBroker.submit_order", boom)
    monkeypatch.setattr("services.broker_alpaca.AlpacaBroker.get_order", boom)


def _execute(root: Path, monkeypatch, token: str = NONCE):
    _enable(monkeypatch)
    _guard(monkeypatch)
    return execute_october15_16_manual_exit_store_recovery(
        authorization_token=token,
        root=root,
    )


def test_flags_stay_disabled_and_cli_execute_refuses(tmp_path):
    source = inspect.getsource(recovery)
    assert "EXECUTOR_ENABLED = False" in source
    assert "WRITE_MODE_ENABLED = False" in source
    assert EXECUTOR_ENABLED is False
    assert WRITE_MODE_ENABLED is False
    for banned in (
        "append_events",
        "submit_order",
        "record_proven_terminal_fills",
        "safe_publish_session_close",
        "basis_unproven_hold",
    ):
        assert banned not in source
    assert execute_legacy_store_recovery()["reason"] == "executor_not_enabled"
    assert recovery.main(["--execute"]) == 2


def test_dry_run_reports_the_plan_and_writes_nothing(tmp_path):
    root = tmp_path / "repo"
    _plant(root)
    before = _tree(root)
    report = dry_run_october15_16_manual_exit_store_recovery(root=root)
    assert report["ok"] is True
    assert report["wrote"] is False
    assert report["broker_called"] is False
    assert report["ledger_appended"] is False
    assert report["changes"] == recovery._planned_changes()
    assert _tree(root) == before
    actions = {item["action"] for item in report["changes"]}
    assert "consume_challenge" in actions
    assert "replace_non_authoritative" in actions
    assert "remove" in actions


def test_disabled_execute_does_not_consume_the_manifest(tmp_path):
    root = tmp_path / "repo"
    _plant(root)
    _manifest(root)
    before = _tree(root)
    report = execute_october15_16_manual_exit_store_recovery(
        authorization_token=NONCE,
        root=root,
    )
    assert report["reason"] == "executor_not_enabled"
    assert report["wrote"] is False
    assert (root / recovery.MANIFEST_RELATIVE).is_file()
    assert _tree(root) == before


def test_success_aligns_the_three_rows_once(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _plant(root)
    _manifest(root)
    ledger = (root / "data" / "results" / "paper_order_events.jsonl").read_bytes()
    accounting = (root / "data" / "results" / "performance_accounting_events.jsonl").read_bytes()
    hold = (root / "data" / "governance" / "basis_unproven_hold.json").read_bytes()
    linked = root / "data" / "governance" / "manual_portfolio_exit" / "requests"
    unchanged = (linked / f"{recovery.TARGETS[1]['request_id']}.json").read_bytes()
    report = _execute(root, monkeypatch)
    assert report["ok"] is True
    assert report["wrote"] is True
    assert report["reason"] == "recovered"
    assert report["broker_called"] is False
    assert report["ledger_appended"] is False
    assert report["manifest_consumed"] is True
    assert report["exclusive_access"] is True
    assert not (root / recovery.MANIFEST_RELATIVE).exists()
    store = root / "data" / "governance" / "manual_portfolio_exit"
    assert not Path(str(store / "manual_portfolio_exit.sqlite3") + "-wal").exists()
    assert not Path(str(store / "manual_portfolio_exit.sqlite3") + "-shm").exists()
    permit = json.loads((store / "paper_sell_arm_dispatch_permit.json").read_text())
    assert permit["status"] == "NON_AUTHORITATIVE"
    assert permit["close_gate_evidence"] is False
    assert "permit_token" not in permit
    gate = recovery.evaluate_manual_exit_close_gate(store)
    assert gate["ok"] is True
    assert gate["verified_broker_order_ids"] == [
        item["broker_order_id"] for item in recovery.TARGETS
    ]
    connection = sqlite3.connect(store / "manual_portfolio_exit.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        rows = list(
            connection.execute(
                "SELECT c.state, a.final_result, d.result_class, d.broker_ref "
                "FROM sell_challenges c "
                "JOIN sell_authorities a ON a.challenge_id=c.challenge_id "
                "JOIN sell_dispatches d ON d.authority_id=a.authority_id "
                "ORDER BY d.broker_ref"
            )
        )
    finally:
        connection.close()
    assert [row["state"] for row in rows] == ["CONSUMED", "CONSUMED", "CONSUMED"]
    by_result = {row["broker_ref"]: row["final_result"] for row in rows}
    assert by_result[recovery.TARGETS[0]["broker_order_id"]] == "DISPATCH_INTENT"
    assert by_result[recovery.TARGETS[1]["broker_order_id"]] == "SUBMITTED"
    oct15 = next(row for row in rows if row["broker_ref"] == recovery.TARGETS[0]["broker_order_id"])
    assert oct15["result_class"] is None
    assert (linked / f"{recovery.TARGETS[1]['request_id']}.json").read_bytes() == unchanged
    assert (root / "data" / "results" / "paper_order_events.jsonl").read_bytes() == ledger
    assert (
        root / "data" / "results" / "performance_accounting_events.jsonl"
    ).read_bytes() == accounting
    assert (root / "data" / "governance" / "basis_unproven_hold.json").read_bytes() == hold
    assert b"924938e3-813d-4007-9384-b36fb639cd66" in ledger
    second = _execute(root, monkeypatch)
    assert second["ok"] is False
    assert second["wrote"] is False
    assert second["reason"] == "already_aligned"
    replacement = {field: "0" * 64 for field in recovery._HASH_FIELDS}
    replacement["nonce"] = NONCE
    manifest_path = root / recovery.MANIFEST_RELATIVE
    manifest_path.write_text(json.dumps(replacement, sort_keys=True), encoding="utf-8")
    third = _execute(root, monkeypatch)
    assert third["reason"] == "already_aligned"
    assert third["wrote"] is False
    assert manifest_path.is_file()
    assert len(list((store / "dispatch_terminal").glob("*.json"))) == 3


@pytest.mark.parametrize(
    "mutator, reason",
    [
        ("symbol", "target_mismatch"),
        ("quantity", "target_mismatch"),
        ("request_id", "target_mismatch"),
        ("authority_id", "target_mismatch"),
        ("dispatch_id", "target_mismatch"),
        ("broker_order_id", "target_mismatch"),
        ("client_order_id", "target_mismatch"),
        ("final_result", "target_mismatch"),
        ("stale_receipt", "stale_store_state"),
        ("missing_fill", "missing_evidence"),
        ("duplicate_terminal", "duplicated_evidence"),
        ("unexpected_child", "unexpected_child"),
        ("already_aligned", "already_aligned"),
    ],
)
def test_identity_and_evidence_refusals_write_nothing(tmp_path, monkeypatch, mutator, reason):
    root = tmp_path / "repo"
    _plant(root)
    store = root / "data" / "governance" / "manual_portfolio_exit"
    target = recovery.TARGETS[1]
    if mutator == "symbol":
        path = store / "requests" / f"{target['request_id']}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["symbol"] = "MSFT"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutator == "quantity":
        path = store / "requests" / f"{target['request_id']}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["selected_qty"] = "401"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutator == "request_id":
        path = store / "requests" / f"{recovery.TARGETS[2]['request_id']}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["request_id"] = "f" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutator == "authority_id":
        path = store / "requests" / f"{target['request_id']}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["authority_id"] = "manual-auth-other"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutator == "dispatch_id":
        path = store / "requests" / f"{target['request_id']}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["dispatch_id"] = "sell-dispatch-other"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutator == "broker_order_id":
        path = store / "order_identities" / f"{target['broker_order_id']}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["broker_order_id"] = "ord-other"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutator == "client_order_id":
        path = store / "requests" / f"{target['request_id']}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["client_order_id"] = "manual-BITO-other"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutator == "final_result":
        with write_connection(store / "manual_portfolio_exit.sqlite3") as connection:
            connection.execute(
                "UPDATE sell_authorities SET final_result='SUBMITTED' WHERE authority_id=?",
                (recovery.TARGETS[0]["authority_id"],),
            )
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(store / "manual_portfolio_exit.sqlite3") + suffix)
            if not sidecar.exists():
                sidecar.write_bytes(b"")
    elif mutator == "stale_receipt":
        path = store / "requests" / f"{target['request_id']}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["broker_status"] = "pending_new"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutator == "missing_fill":
        path = root / "data" / "results" / "performance_accounting_events.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if target["broker_order_id"] not in line or '"FILL_SELL"' not in line
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    elif mutator == "duplicate_terminal":
        path = root / "data" / "results" / "paper_order_events.jsonl"
        extra = _paper(target, "ORDER_TERMINAL_RECONCILED")
        extra["event_id"] = "b" * 64
        path.write_text(
            path.read_text(encoding="utf-8") + json.dumps(extra) + "\n", encoding="utf-8"
        )
    elif mutator == "unexpected_child":
        (store / "notes.txt").write_text("unrelated", encoding="utf-8")
    elif mutator == "already_aligned":
        with write_connection(store / "manual_portfolio_exit.sqlite3") as connection:
            connection.execute(
                """
                UPDATE sell_challenges
                SET state='CONSUMED', consuming_authority_id=?, consumed_at=?
                WHERE challenge_id=?
                """,
                (
                    target["authority_id"],
                    "2026-10-16T12:00:00+00:00",
                    target["challenge_id"],
                ),
            )
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(store / "manual_portfolio_exit.sqlite3") + suffix)
            if not sidecar.exists():
                sidecar.write_bytes(b"")
    before = _tree(root)
    _manifest(root)
    report = _execute(root, monkeypatch)
    assert report["ok"] is False
    assert report["wrote"] is False
    assert report["reason"] == reason
    assert (root / recovery.MANIFEST_RELATIVE).is_file()
    after = _tree(root)
    assert set(after) - {recovery.MANIFEST_RELATIVE.as_posix()} == set(before)


def test_baseline_hash_mismatch_writes_nothing(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _plant(root)
    _manifest(root)
    path = (
        root
        / "data"
        / "governance"
        / "manual_portfolio_exit"
        / "requests"
        / f"{recovery.TARGETS[0]['request_id']}.json"
    )
    path.write_bytes(path.read_bytes() + b" ")
    before = path.read_bytes()
    report = _execute(root, monkeypatch)
    assert report["reason"] == "baseline_hash_mismatch"
    assert report["wrote"] is False
    assert path.read_bytes() == before


def test_runtime_lock_and_streamlit_and_sqlite_handle_refuse(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _plant(root)
    _manifest(root)
    lock = root / "data" / "live" / "continuous_loop.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("4242", encoding="utf-8")
    before = _tree(root)
    locked = _execute(root, monkeypatch)
    assert locked["reason"] == "runtime_lock"
    assert locked["wrote"] is False
    assert _tree(root) == before
    lock.write_text("0", encoding="utf-8")
    real_root = recovery._repo_root
    monkeypatch.setattr(recovery, "_repo_root", lambda: root)
    monkeypatch.setattr(
        recovery,
        "inspect_runtime_state",
        lambda _root: {
            "inspect_failed": False,
            "streamlit_stopped": False,
            "runtime_stopped": True,
        },
    )
    streamlit = _execute(root, monkeypatch)
    assert streamlit["reason"] == "streamlit_active"
    assert streamlit["wrote"] is False
    monkeypatch.setattr(
        recovery,
        "inspect_runtime_state",
        lambda _root: {
            "inspect_failed": False,
            "streamlit_stopped": True,
            "runtime_stopped": False,
        },
    )
    runtime = _execute(root, monkeypatch)
    assert runtime["reason"] == "triton_runtime_active"
    monkeypatch.setattr(recovery, "_repo_root", real_root)
    db_path = (
        root / "data" / "governance" / "manual_portfolio_exit" / "manual_portfolio_exit.sqlite3"
    )
    holder = sqlite3.connect(str(db_path), timeout=0.2, isolation_level=None)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        busy = _execute(root, monkeypatch)
    finally:
        holder.close()
    assert busy["reason"] == "sqlite_handle_active"
    assert busy["wrote"] is False
    assert (root / recovery.MANIFEST_RELATIVE).is_file()


def test_failed_verification_restores_and_keeps_the_manifest(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _plant(root)
    _manifest(root)
    before = _tree(root)
    monkeypatch.setattr(
        recovery,
        "evaluate_manual_exit_close_gate",
        lambda _store: {"ok": False, "reason": "forced"},
    )
    report = _execute(root, monkeypatch)
    assert report["reason"] == "recovery_not_verified"
    assert report["wrote"] is False
    assert (root / recovery.MANIFEST_RELATIVE).is_file()
    assert _tree(root) == before


def test_reconciliation_does_not_import_a_broker_or_ledger_writer():
    source = Path(inspect.getfile(recovery)).read_text(encoding="utf-8")
    assert "AlpacaBroker" not in source
    assert "append_events(" not in source
    assert "record_manual_exit_submission(" not in source
    assert Decimal("450") == recovery.TARGETS[0]["qty"]
    assert Decimal("400") == recovery.TARGETS[1]["qty"]
    assert Decimal("19") == recovery.TARGETS[2]["qty"]
