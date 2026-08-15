"""Phase 198I-W — SQLite WAL+FULL paper SELL authority store.

PAPER ONLY. Fake broker. No real credentials. No repository data/live authority.
Does not add Operator Console buttons. Does not weaken JSON durability evaluator.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from services.paper_sell_arm_authority import (
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    confirm_and_create_paper_sell_arm,
    prepare_confirmation_challenge,
)
from services.paper_sell_arm_runtime_enforcement import (
    begin_paper_sell_dispatch,
    validate_runtime_sell_authority,
)
from services.paper_sell_arm_store import (
    PaperSellArmStoreError,
    batch_hash_from_scope,
    consume_authority_insert_dispatch,
    consume_challenge_and_create_authority,
    db_path_for_store_root,
    dump_schema_sql,
    health_check,
    open_connection,
    persist_open_challenge,
    record_dispatch_outcome,
    refuse_unsupported_db_path,
    sqlite_permit_grade_enabled,
    verify_pragmas_or_fail,
    write_connection,
)
from tests._phase_198i_sell_fakes import (
    SESSION_OPEN_NOW,
    NetworkFakeAlpacaBroker,
    block_real_network,
    isolate_sell_store,
    readiness_packet,
    seed_paper_sell_store,
)

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests" / "_phase_198i_w_crash_worker.py"
NOW = SESSION_OPEN_NOW


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    block_real_network(monkeypatch)
    monkeypatch.delenv("TRITON_PAPER_SELL_ARM_STORE", raising=False)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT

    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)


def _challenge_and_scope(tmp_path: Path):
    readiness = readiness_packet(now=NOW)
    (tmp_path / "paper_arm_confirm.json").write_text(
        json.dumps(
            {
                "timestamp": NOW.isoformat(),
                "mode": "ASSISTED",
                "expires_at": (NOW + timedelta(hours=4)).isoformat(),
                "allow_execute": True,
            }
        ),
        encoding="utf-8",
    )
    chal = prepare_confirmation_challenge(
        store_root=tmp_path,
        readiness_packet=readiness,
        account_raw="acct-198iw",
        trading_date=NOW.date().isoformat(),
        now=NOW,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    return readiness, chal


def test_a_wal_full_fk_active(tmp_path: Path):
    db = db_path_for_store_root(tmp_path)
    conn = open_connection(db)
    try:
        pragmas = verify_pragmas_or_fail(conn)
    finally:
        conn.close()
    assert pragmas["journal_mode"] == "wal"
    assert pragmas["synchronous"] == 2
    assert pragmas["foreign_keys"] == 1
    ok, reason, details = health_check(db)
    assert ok and reason == "ok"
    assert "SQLite format" in dump_schema_sql() or "sell_challenges" in dump_schema_sql()


def test_a_unsupported_pragma_fails_closed(tmp_path: Path):
    db = db_path_for_store_root(tmp_path)
    open_connection(db).close()
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
        with pytest.raises(PaperSellArmStoreError) as exc:
            verify_pragmas_or_fail(conn)
        assert exc.value.reason == "sqlite_synchronous_not_full"
        conn.execute("PRAGMA journal_mode=DELETE")
        with pytest.raises(PaperSellArmStoreError) as exc2:
            verify_pragmas_or_fail(conn)
        assert exc2.value.reason == "sqlite_journal_mode_not_wal"
    finally:
        conn.close()


def test_a_unc_path_refused():
    with pytest.raises(PaperSellArmStoreError) as exc:
        refuse_unsupported_db_path(Path(r"\\server\share\paper_sell_arm.sqlite3"))
    assert "unc" in exc.value.reason or "network" in exc.value.reason


def test_b_challenge_only_no_authority(tmp_path: Path, monkeypatch):
    readiness, chal = _challenge_and_scope(tmp_path)
    db = db_path_for_store_root(tmp_path)
    with write_connection(db) as conn:
        n_auth = conn.execute("SELECT COUNT(*) FROM sell_authorities").fetchone()[0]
        n_disp = conn.execute("SELECT COUNT(*) FROM sell_dispatches").fetchone()[0]
        row = conn.execute(
            "SELECT state FROM sell_challenges WHERE challenge_id=?", (chal["challenge_id"],)
        ).fetchone()
    assert n_auth == 0 and n_disp == 0
    assert row["state"] == "OPEN"
    assert chal["creates_no_authority"] is True
    fake = NetworkFakeAlpacaBroker()
    assert fake.fake_broker_call_count == 0


def test_b_challenge_exact_batch_and_expiry(tmp_path: Path, monkeypatch):
    readiness, chal = _challenge_and_scope(tmp_path)
    expired = NOW + timedelta(days=1)
    with pytest.raises(ValueError) as exc:
        confirm_and_create_paper_sell_arm(
            store_root=tmp_path,
            readiness_packet=readiness,
            challenge=chal,
            operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
            current_plan_hash=readiness["plan_hash"],
            now=expired,
        )
    assert "expired" in str(exc.value).lower() or "challenge" in str(exc.value)


def test_b_challenge_replay_and_race(tmp_path: Path, monkeypatch):
    readiness, chal = _challenge_and_scope(tmp_path)
    auth = confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
        now=NOW + timedelta(minutes=1),
    )
    assert auth["sqlite_state"] == "UNUSED"
    with pytest.raises(ValueError):
        confirm_and_create_paper_sell_arm(
            store_root=tmp_path,
            readiness_packet=readiness,
            challenge=chal,
            operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
            current_plan_hash=readiness["plan_hash"],
            now=NOW + timedelta(minutes=2),
        )
    db = db_path_for_store_root(tmp_path)
    # Fresh challenge for race: persist one OPEN challenge, two consume attempts.
    chal2 = dict(chal)
    chal2["challenge_id"] = "sell-chal-race-" + "x" * 8
    persist_open_challenge(db, challenge=chal2, now=NOW)
    results = []

    def _one(idx: int):
        try:
            consume_challenge_and_create_authority(
                db,
                challenge_id=chal2["challenge_id"],
                readiness_plan_hash=str(chal2["plan_hash"]),
                readiness_batch_id=str(chal2["batch_id"]),
                readiness_batch_hash=batch_hash_from_scope(chal2["authorized_scope"]),
                authority={
                    "audit_id": f"race-{idx}",
                    "plan_hash": chal2["plan_hash"],
                    "authorized_actions": chal2["authorized_scope"],
                    "issued_at": NOW.isoformat(),
                    "expires_at": (NOW + timedelta(minutes=30)).isoformat(),
                    "schema_version": "198I-W-SQLITE-1",
                },
                now=NOW + timedelta(minutes=1),
            )
            results.append("ok")
        except PaperSellArmStoreError:
            results.append("fail")

    t1 = threading.Thread(target=_one, args=(1,))
    t2 = threading.Thread(target=_one, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert results.count("ok") == 1
    assert results.count("fail") == 1
    with write_connection(db) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM sell_authorities WHERE challenge_id=?",
            (chal2["challenge_id"],),
        ).fetchone()[0]
    assert n == 1


def test_c_authority_single_use_exact_batch_mismatch(tmp_path: Path, monkeypatch):
    seed = seed_paper_sell_store(tmp_path, now=NOW)
    db = db_path_for_store_root(tmp_path)
    aid = seed["auth"]["authority_id"]
    scope = seed["auth"]["authorized_actions"]
    bhash = batch_hash_from_scope(scope)
    req_ok = [{"symbol": "TSLA", "action": "EXIT", "qty": 8.0, "side": "sell"}]
    consume_authority_insert_dispatch(
        db,
        authority_id=aid,
        plan_hash=seed["auth"]["plan_hash"],
        batch_hash=bhash,
        expected_scope=scope,
        mode="paper",
        live_blocked=True,
        requested=req_ok,
        now=NOW + timedelta(minutes=1),
    )
    with pytest.raises(PaperSellArmStoreError) as exc:
        consume_authority_insert_dispatch(
            db,
            authority_id=aid,
            plan_hash=seed["auth"]["plan_hash"],
            batch_hash=bhash,
            expected_scope=scope,
            mode="paper",
            live_blocked=True,
            requested=req_ok,
            now=NOW + timedelta(minutes=2),
        )
    assert exc.value.reason in {"sqlite_authority_not_unused", "sqlite_duplicate_dispatch"}


def test_c_symbol_qty_expired_legacy_json(tmp_path: Path, monkeypatch):
    seed = seed_paper_sell_store(tmp_path, now=NOW)
    db = db_path_for_store_root(tmp_path)
    aid = seed["auth"]["authority_id"]
    scope = seed["auth"]["authorized_actions"]
    bhash = batch_hash_from_scope(scope)
    with pytest.raises(PaperSellArmStoreError) as exc:
        consume_authority_insert_dispatch(
            db,
            authority_id=aid,
            plan_hash=seed["auth"]["plan_hash"],
            batch_hash=bhash,
            expected_scope=scope,
            mode="paper",
            live_blocked=True,
            requested=[{"symbol": "AAPL", "action": "EXIT", "qty": 8.0, "side": "sell"}],
            now=NOW + timedelta(minutes=1),
        )
    assert exc.value.reason == "sqlite_symbol_mismatch"
    with pytest.raises(PaperSellArmStoreError) as exc2:
        consume_authority_insert_dispatch(
            db,
            authority_id=aid,
            plan_hash=seed["auth"]["plan_hash"],
            batch_hash=bhash,
            expected_scope=scope,
            mode="paper",
            live_blocked=True,
            requested=[{"symbol": "TSLA", "action": "EXIT", "qty": 99.0, "side": "sell"}],
            now=NOW + timedelta(minutes=1),
        )
    assert exc2.value.reason == "sqlite_quantity_mismatch"
    with pytest.raises(PaperSellArmStoreError) as exc3:
        consume_authority_insert_dispatch(
            db,
            authority_id=aid,
            plan_hash=seed["auth"]["plan_hash"],
            batch_hash=bhash,
            expected_scope=scope,
            mode="paper",
            live_blocked=True,
            requested=[{"symbol": "TSLA", "action": "EXIT", "qty": 8.0, "side": "sell"}],
            now=NOW + timedelta(hours=5),
        )
    assert exc3.value.reason == "sqlite_authority_expired"
    json_only = tmp_path / "legacy_only"
    json_only.mkdir()
    isolate_sell_store(monkeypatch, json_only)
    (json_only / "paper_sell_arm_authority.json").write_text(
        json.dumps(seed["auth"]), encoding="utf-8"
    )
    ok, reason, _ = validate_runtime_sell_authority(
        contract_path=json_only / "paper_sell_arm_authority.json",
        now=NOW + timedelta(minutes=1),
        store_root=json_only,
    )
    assert not ok
    assert (
        reason in {"legacy_json_not_broker_usable", "approval_contract_missing"}
        or "legacy" in reason
    )


def test_d_consume_intent_atomic_permit_after_commit(tmp_path: Path, monkeypatch):
    seed = seed_paper_sell_store(tmp_path, now=NOW)
    from dataclasses import dataclass

    @dataclass
    class Planned:
        symbol: str = "TSLA"
        side: str = "sell"
        qty: float = 8.0
        limit_price: float = 332.5
        stance: str = "EXIT"
        client_order_id: str = "coid-w"
        force_exit_override: bool = False
        ref_price: float = 332.5
        management_action: str = "EXIT"

    from services.paper_sell_arm_authority import BrokerSnapshotView

    view = BrokerSnapshotView(
        positions={"TSLA": {"qty": 8.0, "qty_available": 8.0}},
        open_order_symbols=set(),
        mode="paper",
        live_blocked=True,
        trading_date=str(seed["auth"].get("trading_date")),
        account_fingerprint=str(seed["auth"].get("account_fingerprint")),
        current_plan_hash=str(seed["auth"].get("plan_hash")),
        paper_buy_arm_valid=True,
        paper_execution_authorized=True,
        cycle_id=str(seed["auth"].get("cycle_id")),
        run_id=str(seed["auth"].get("run_id")),
        batch_id=str(seed["auth"].get("batch_id")),
        plan_id=str(seed["auth"].get("plan_id")),
        readiness_packet_hash=str(seed["auth"].get("readiness_packet_hash")),
        confirmation_challenge_id=str(seed["auth"].get("confirmation_challenge_id")),
        confirmation_challenge_hash=str(seed["auth"].get("confirmation_challenge_hash")),
        evidence_independent=True,
        market_session="MARKET_OPEN",
    )
    g = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=[Planned()],
        broker_view=view,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
        live_override={"live_blocked": True, "live_explicitly_authorized": False},
    )
    assert g.permitted and g.permit is not None
    assert g.details.get("authority_backend") == "sqlite"
    db = db_path_for_store_root(tmp_path)
    with write_connection(db) as conn:
        a = conn.execute(
            "SELECT state FROM sell_authorities WHERE authority_id=?",
            (seed["auth"]["authority_id"],),
        ).fetchone()
        d = conn.execute(
            "SELECT state FROM sell_dispatches WHERE dispatch_id=?", (g.dispatch_intent_id,)
        ).fetchone()
    assert a["state"] == "CONSUMED"
    assert d["state"] == "INTENT_COMMITTED"
    g2 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=[Planned()],
        broker_view=view,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        live_override={"live_blocked": True, "live_explicitly_authorized": False},
    )
    assert not g2.permitted


def test_d_rollback_issues_no_permit(tmp_path: Path, monkeypatch):
    seed = seed_paper_sell_store(tmp_path, now=NOW)
    db = db_path_for_store_root(tmp_path)

    def _boom(*_a, **_k):
        raise PaperSellArmStoreError("sqlite_forced_rollback")

    monkeypatch.setattr(
        "services.paper_sell_arm_store.consume_authority_insert_dispatch",
        _boom,
    )
    from dataclasses import dataclass

    @dataclass
    class Planned:
        symbol: str = "TSLA"
        side: str = "sell"
        qty: float = 8.0
        limit_price: float = 332.5
        stance: str = "EXIT"
        client_order_id: str = "coid-w"
        force_exit_override: bool = False
        ref_price: float = 332.5
        management_action: str = "EXIT"

    from services.paper_sell_arm_authority import BrokerSnapshotView

    view = BrokerSnapshotView(
        positions={"TSLA": {"qty": 8.0, "qty_available": 8.0}},
        open_order_symbols=set(),
        mode="paper",
        live_blocked=True,
        trading_date=str(seed["auth"].get("trading_date")),
        account_fingerprint=str(seed["auth"].get("account_fingerprint")),
        current_plan_hash=str(seed["auth"].get("plan_hash")),
        paper_buy_arm_valid=True,
        paper_execution_authorized=True,
        cycle_id=str(seed["auth"].get("cycle_id")),
        run_id=str(seed["auth"].get("run_id")),
        batch_id=str(seed["auth"].get("batch_id")),
        plan_id=str(seed["auth"].get("plan_id")),
        readiness_packet_hash=str(seed["auth"].get("readiness_packet_hash")),
        confirmation_challenge_id=str(seed["auth"].get("confirmation_challenge_id")),
        confirmation_challenge_hash=str(seed["auth"].get("confirmation_challenge_hash")),
        evidence_independent=True,
        market_session="MARKET_OPEN",
    )
    g = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=[Planned()],
        broker_view=view,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
        live_override={"live_blocked": True, "live_explicitly_authorized": False},
    )
    assert not g.permitted
    assert g.permit is None
    with write_connection(db) as conn:
        st = conn.execute("SELECT state FROM sell_authorities").fetchone()["state"]
    assert st == "UNUSED"


def test_d_concurrent_dispatch_one_permit(tmp_path: Path, monkeypatch):
    seed = seed_paper_sell_store(tmp_path, now=NOW)
    db = db_path_for_store_root(tmp_path)
    aid = seed["auth"]["authority_id"]
    scope = seed["auth"]["authorized_actions"]
    bhash = batch_hash_from_scope(scope)
    req = [{"symbol": "TSLA", "action": "EXIT", "qty": 8.0, "side": "sell"}]
    results = []

    def _disp():
        try:
            consume_authority_insert_dispatch(
                db,
                authority_id=aid,
                plan_hash=seed["auth"]["plan_hash"],
                batch_hash=bhash,
                expected_scope=scope,
                mode="paper",
                live_blocked=True,
                requested=req,
                now=NOW + timedelta(minutes=1),
            )
            results.append("ok")
        except PaperSellArmStoreError:
            results.append("fail")

    threads = [threading.Thread(target=_disp) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 1
    with write_connection(db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM sell_dispatches").fetchone()[0]
    assert n == 1


def test_e_crash_recovery_matrix(tmp_path: Path):
    db = db_path_for_store_root(tmp_path)
    open_connection(db).close()
    readiness = readiness_packet(now=NOW)
    chal = {
        "challenge_id": "sell-chal-crash1",
        "plan_hash": readiness["plan_hash"],
        "batch_id": readiness["batch_id"],
        "authorized_scope": [
            {"symbol": "TSLA", "action": "EXIT", "max_qty": 8.0, "max_notional": 2660.0}
        ],
        "issued_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(minutes=30)).isoformat(),
        "operator_button": "ARM_REVIEWED_SELLS",
        "schema_version": "198I-W-SQLITE-1",
    }
    bhash = batch_hash_from_scope(chal["authorized_scope"])
    authority = {
        "audit_id": "auth-crash-1",
        "plan_hash": chal["plan_hash"],
        "authorized_actions": chal["authorized_scope"],
        "issued_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(minutes=30)).isoformat(),
        "schema_version": "198I-W-SQLITE-1",
    }
    py = sys.executable

    def _run(stage, extra=None):
        payload = {
            "stage": stage,
            "db_path": str(db),
            "now": NOW.isoformat(),
            "challenge": chal,
            "authority": authority,
            "batch_hash": bhash,
            "plan_hash": chal["plan_hash"],
            "live_blocked": True,
        }
        if extra:
            payload.update(extra)
        pfile = tmp_path / f"{stage}.json"
        pfile.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.run(
            [py, str(WORKER), str(pfile)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )

    r1 = _run("crash_before_challenge_commit")
    assert r1.returncode != 0
    with write_connection(db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM sell_challenges").fetchone()[0]
    assert n == 0

    r2 = _run("crash_after_challenge_commit")
    assert r2.returncode == 18
    with write_connection(db) as conn:
        st = conn.execute("SELECT state FROM sell_challenges").fetchone()["state"]
        n_auth = conn.execute("SELECT COUNT(*) FROM sell_authorities").fetchone()[0]
    assert st == "OPEN" and n_auth == 0

    db2 = tmp_path / "crash_auth.sqlite3"
    open_connection(db2).close()
    payload_auth = {
        "stage": "crash_before_authority_commit",
        "db_path": str(db2),
        "now": NOW.isoformat(),
        "challenge": {**chal, "challenge_id": "sell-chal-crash2"},
        "authority": authority,
        "batch_hash": bhash,
        "plan_hash": chal["plan_hash"],
        "live_blocked": True,
    }
    pfile = tmp_path / "auth_crash.json"
    pfile.write_text(json.dumps(payload_auth), encoding="utf-8")
    r3 = subprocess.run(
        [py, str(WORKER), str(pfile)], cwd=str(ROOT), capture_output=True, text=True, timeout=30
    )
    assert r3.returncode != 0
    with write_connection(db2) as conn:
        ch = conn.execute("SELECT state FROM sell_challenges").fetchone()["state"]
        n_auth = conn.execute("SELECT COUNT(*) FROM sell_authorities").fetchone()[0]
    assert ch == "OPEN" and n_auth == 0

    db3 = tmp_path / "after_auth.sqlite3"
    open_connection(db3).close()
    payload_ok = {
        "stage": "crash_after_authority_commit",
        "db_path": str(db3),
        "now": NOW.isoformat(),
        "challenge": {**chal, "challenge_id": "sell-chal-crash3"},
        "authority": {**authority, "audit_id": "auth-crash-3"},
        "batch_hash": bhash,
        "plan_hash": chal["plan_hash"],
        "live_blocked": True,
    }
    pfile.write_text(json.dumps(payload_ok), encoding="utf-8")
    r4 = subprocess.run(
        [py, str(WORKER), str(pfile)], cwd=str(ROOT), capture_output=True, text=True, timeout=30
    )
    assert r4.returncode == 19
    with write_connection(db3) as conn:
        a = conn.execute("SELECT state FROM sell_authorities").fetchone()
        n_d = conn.execute("SELECT COUNT(*) FROM sell_dispatches").fetchone()[0]
    assert a["state"] == "UNUSED" and n_d == 0

    committed = consume_authority_insert_dispatch(
        db3,
        authority_id="auth-crash-3",
        plan_hash=chal["plan_hash"],
        batch_hash=bhash,
        expected_scope=chal["authorized_scope"],
        mode="paper",
        live_blocked=True,
        requested=[{"symbol": "TSLA", "action": "EXIT", "qty": 8.0, "side": "sell"}],
        now=NOW + timedelta(minutes=1),
    )
    payload_ok["stage"] = "crash_after_broker_before_outcome"
    payload_ok["dispatch_id"] = committed["dispatch_id"]
    pfile.write_text(json.dumps(payload_ok), encoding="utf-8")
    r5 = subprocess.run(
        [py, str(WORKER), str(pfile)], cwd=str(ROOT), capture_output=True, text=True, timeout=30
    )
    assert r5.returncode == 21
    with write_connection(db3) as conn:
        a = conn.execute("SELECT state FROM sell_authorities").fetchone()["state"]
        d = conn.execute("SELECT state, reconciliation_required FROM sell_dispatches").fetchone()
    assert a == "CONSUMED"
    assert d["state"] == "INTENT_COMMITTED"
    # Do not restore UNUSED; classify reconciliation.
    record_dispatch_outcome(
        db3,
        dispatch_id=committed["dispatch_id"],
        outcome_state="OUTCOME_UNKNOWN",
        reconciliation_required=True,
        now=NOW + timedelta(minutes=2),
    )
    with write_connection(db3) as conn:
        a = conn.execute("SELECT state FROM sell_authorities").fetchone()["state"]
        d = conn.execute("SELECT state, reconciliation_required FROM sell_dispatches").fetchone()
    assert a == "CONSUMED"
    assert d["state"] == "OUTCOME_UNKNOWN"
    assert int(d["reconciliation_required"]) == 1


def test_f_corrupt_and_wrong_schema(tmp_path: Path):
    bad = tmp_path / "corrupt.sqlite3"
    bad.write_bytes(b"not a sqlite database!!!!")
    with pytest.raises(PaperSellArmStoreError) as exc:
        open_connection(bad)
    assert exc.value.reason == "sqlite_database_corrupt_or_incompatible"

    db = db_path_for_store_root(tmp_path / "schema")
    conn = open_connection(db)
    conn.execute("PRAGMA user_version=99")
    conn.close()
    with pytest.raises(PaperSellArmStoreError) as exc2:
        open_connection(db)
    assert exc2.value.reason == "sqlite_incompatible_schema_version"


def test_g_live_blocked_buy_add_no_broker_on_deny(tmp_path: Path, monkeypatch):
    seed = seed_paper_sell_store(tmp_path, now=NOW)
    db = db_path_for_store_root(tmp_path)
    fake = NetworkFakeAlpacaBroker()
    with pytest.raises(PaperSellArmStoreError):
        consume_authority_insert_dispatch(
            db,
            authority_id=seed["auth"]["authority_id"],
            plan_hash=seed["auth"]["plan_hash"],
            batch_hash=batch_hash_from_scope(seed["auth"]["authorized_actions"]),
            expected_scope=seed["auth"]["authorized_actions"],
            mode="paper",
            live_blocked=False,
            requested=[{"symbol": "TSLA", "action": "EXIT", "qty": 8.0, "side": "sell"}],
            now=NOW + timedelta(minutes=1),
        )
    with pytest.raises(PaperSellArmStoreError) as exc:
        consume_authority_insert_dispatch(
            db,
            authority_id=seed["auth"]["authority_id"],
            plan_hash=seed["auth"]["plan_hash"],
            batch_hash=batch_hash_from_scope(seed["auth"]["authorized_actions"]),
            expected_scope=seed["auth"]["authorized_actions"],
            mode="paper",
            live_blocked=True,
            requested=[{"symbol": "TSLA", "action": "BUY", "qty": 1.0, "side": "buy"}],
            now=NOW + timedelta(minutes=1),
        )
    assert exc.value.reason == "sqlite_buy_add_not_authorized_on_sell_store"
    assert fake.fake_broker_call_count == 0
    buy_before = (tmp_path / "paper_arm_confirm.json").read_bytes()
    assert buy_before
    # BUY/ADD file unchanged
    assert json.loads(buy_before.decode("utf-8"))["mode"] == "ASSISTED"


def test_json_cannot_restore_unused(tmp_path: Path, monkeypatch):
    seed = seed_paper_sell_store(tmp_path, consume=True, now=NOW)
    db = db_path_for_store_root(tmp_path)
    auth_path = tmp_path / "paper_sell_arm_authority.json"
    doc = json.loads(auth_path.read_text(encoding="utf-8"))
    doc["consumption_status"] = "UNUSED"
    doc["status"] = "ARMED"
    auth_path.write_text(json.dumps(doc), encoding="utf-8")
    ok, reason, _ = validate_runtime_sell_authority(
        contract_path=auth_path, now=NOW + timedelta(minutes=2), store_root=tmp_path
    )
    assert not ok
    with write_connection(db) as conn:
        st = conn.execute("SELECT state FROM sell_authorities").fetchone()["state"]
    assert st == "CONSUMED"


def test_sqlite_enabled_by_default():
    assert sqlite_permit_grade_enabled() is True
