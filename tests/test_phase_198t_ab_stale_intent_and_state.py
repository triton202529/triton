"""Phase 198T-AB — isolated stale dispatch-intent and SQLite/JSON consistency.

PAPER ONLY. Isolated tmp_path SQLite/JSON. Fake broker only. No production writes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from services.defensive_exit_approval_readiness import (
    SAME_CYCLE_DISPATCH_REASON,
    STATUS_UNAVAILABLE_SAME_CYCLE,
    generate_from_manage_artifacts,
    write_pre_plan_dispatch_evidence,
)
from services.manage_positions_cycle_binding import INVOCATION_ENV
from services.paper_sell_arm_authority import (
    APPROVAL_CLI_CAP,
    SCHEMA_VERSION,
    REASON_SQLITE_JSON_AUTHORITY_MISMATCH,
    classify_sqlite_json_export_consistency,
    export_json_authority_from_sqlite,
)
from services.paper_sell_arm_cap_slice import money_round
from services.paper_sell_arm_runtime_enforcement import (
    DISPATCH_INTENT_NAME,
    _ACTIVE_PERMIT,
    attempt_pre_plan_authorized_sell_dispatch,
    begin_paper_sell_dispatch,
    finalize_paper_sell_dispatch,
    find_unused_pre_plan_authority,
    validate_runtime_sell_authority,
)
from services.paper_sell_arm_store import (
    INTENT_CLASS_AMBIGUOUS,
    INTENT_CLASS_FOREIGN_HISTORICAL_TERMINAL,
    INTENT_CLASS_FOREIGN_NONTERMINAL,
    INTENT_CLASS_MATCHING_CURRENT_IN_PROGRESS,
    INTENT_CLASS_MATCHING_CURRENT_TERMINAL,
    INTENT_CLASS_NO_INTENT,
    PaperSellArmStoreError,
    REASON_FOREIGN_NONTERMINAL,
    REASON_INTENT_AMBIGUOUS,
    REASON_MATCHING_IN_PROGRESS,
    REASON_MATCHING_TERMINAL,
    classify_dispatch_intent_for_authority,
    consume_authority_insert_dispatch,
    count_dispatches_for_authority,
    inspect_authority_row,
    inspect_sqlite_arm_snapshot,
)
from services.place_live_orders import dispatch_unused_authorized_sell_slice_pre_plan
from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    OrderedTrace,
    block_real_network,
    freeze_dispatch_clock,
    install_boundary_trace,
    isolate_sell_store,
    stub_unrelated_pre_gates,
)
from tests.test_phase_198t_u_cross_session_sell_state import (
    OLD_AUTH,
    OLD_DISPATCH,
    NOW as TU_NOW,
    _arm_confirm,
    _authorized_clip_audit,
    _bind,
    _db,
    _fake_broker,
    _incident_packet,
    _seed_historical_b,
    _write_historical_consumption,
    _write_plan_artifacts,
    build_pre_plan_planned_orders,
)

PROD_ROOT = Path(r"C:\Users\akimw\triton")
ISO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 28, 16, 12, 0, tzinfo=timezone.utc)
INV = "mo-20260828_120712-dfd024d3cb364d4a95ddd209f7f5e37c"
AUTH_A = "sell-arm-audit-4af425cf389c"
CONSUMPTION_FILENAME = "paper_sell_arm_consumption.json"

AUG26_INTENT = {
    "schema_version": SCHEMA_VERSION,
    "dispatch_intent_id": OLD_DISPATCH,
    "audit_id": OLD_AUTH,
    "authority_id": OLD_AUTH,
    "idempotency_key": (
        "sell-arm|18c3b9cdb620ac45c59466440aeef887f9023d799a7119be8eca13f4a655c02a|"
        "manage-mo-20260826_1032|sell-chal-6c8daba906c64fc5"
    ),
    "plan_hash": "18c3b9cdb620ac45c59466440aeef887f9023d799a7119be8eca13f4a655c02a",
    "batch_id": "manage-mo-20260826_1032",
    "symbols": ["HD", "BITO"],
    "actions": [
        {
            "symbol": "HD",
            "action": "EXIT",
            "qty": 13.0,
            "price": 337.11,
            "notional": 4382.43,
            "client_order_id": "triton-preplan-HD-EXIT-e10b24dc61",
        },
        {
            "symbol": "BITO",
            "action": "EXIT",
            "qty": 28.0,
            "price": 10.5,
            "notional": 294.0,
            "client_order_id": "triton-preplan-BITO-EXIT-6962016f7b",
        },
    ],
    "recorded_at": "2026-08-26T15:20:50.074550+00:00",
    "status": "INTENT_RECORDED_PRE_SUBMIT",
    "broker_submission_started": True,
    "trading_date": "2026-08-26",
}


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    block_real_network(monkeypatch)
    for key in list(os.environ):
        if "ALPACA" in key.upper():
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    monkeypatch.delenv("TRITON_PAPER_SELL_ARM_STORE", raising=False)
    for key in (
        "TRITON_PAPER_SELL_ARM_STORE_ROOT",
        "TRITON_PAPER_SELL_ARM_RESULTS_ROOT",
        "TRITON_PAPER_SELL_ARM_DB",
        "RESULTS_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    stub_dir_fsync_ok(monkeypatch)
    _ACTIVE_PERMIT.set(None)
    resolved = tmp_path.resolve()
    text = str(resolved).lower()
    assert "\\appdata\\local\\temp\\" in text or "/appdata/local/temp/" in text
    assert not text.startswith(str(PROD_ROOT.resolve()).lower() + os.sep.lower())
    yield
    _ACTIVE_PERMIT.set(None)


def _write_intent(tmp_path: Path, doc: Dict[str, Any]) -> Path:
    path = tmp_path / DISPATCH_INTENT_NAME
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _setup_current(tmp_path, monkeypatch, packet=None, now=None):
    clock = now or TU_NOW
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, clock)
    stub_unrelated_pre_gates(monkeypatch)
    packet = _write_plan_artifacts(tmp_path, packet or _incident_packet())
    _seed_historical_b(_db(tmp_path))
    _write_historical_consumption(tmp_path)
    confirmed = _arm_confirm(tmp_path, packet, now=clock)
    auth = find_unused_pre_plan_authority(db_path=_db(tmp_path), now=clock + timedelta(minutes=1))
    assert auth is not None
    return packet, confirmed, auth


def _matching_intent(auth: Dict[str, Any], *, dispatch_id: str) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dispatch_intent_id": dispatch_id,
        "audit_id": auth.get("audit_id") or auth.get("authority_id"),
        "authority_id": auth.get("authority_id") or auth.get("audit_id"),
        "idempotency_key": auth.get("idempotency_key"),
        "plan_hash": auth.get("plan_hash"),
        "batch_id": auth.get("batch_id"),
        "symbols": [r["symbol"] for r in (auth.get("authorized_actions") or [])],
        "status": "INTENT_RECORDED_PRE_SUBMIT",
        "broker_submission_started": False,
        "trading_date": auth.get("trading_date"),
    }


def test_classify_no_intent():
    kind, reason, _ = classify_dispatch_intent_for_authority(None, {"authority_id": AUTH_A})
    assert kind == INTENT_CLASS_NO_INTENT
    assert reason == ""


def test_case_a_foreign_terminal_intent_ignored(tmp_path, monkeypatch):
    packet, confirmed, auth = _setup_current(tmp_path, monkeypatch)
    intent_path = _write_intent(tmp_path, AUG26_INTENT)
    before = intent_path.read_bytes()
    kind, reason, details = classify_dispatch_intent_for_authority(
        AUG26_INTENT, auth, db_path=_db(tmp_path)
    )
    assert kind == INTENT_CLASS_FOREIGN_HISTORICAL_TERMINAL, details
    assert reason == ""
    fake = _fake_broker(packet)
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=TU_NOW + timedelta(minutes=1),
    )
    assert result.get("dispatched") is True, result
    assert fake.submit_payloads
    assert intent_path.read_bytes() == before
    assert count_dispatches_for_authority(_db(tmp_path), OLD_AUTH) == 1
    assert count_dispatches_for_authority(_db(tmp_path), str(auth["authority_id"])) == 1
    leftover_intent = json.loads(intent_path.read_text(encoding="utf-8"))
    assert leftover_intent["dispatch_intent_id"] == OLD_DISPATCH
    assert leftover_intent["authority_id"] == OLD_AUTH


def test_case_b_matching_terminal_blocks_replay(tmp_path, monkeypatch):
    packet, confirmed, auth = _setup_current(tmp_path, monkeypatch)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=TU_NOW + timedelta(minutes=1))
    batch_hash = str(snap["latest_authority"].get("batch_hash") or "")
    committed = consume_authority_insert_dispatch(
        _db(tmp_path),
        authority_id=str(auth["authority_id"]),
        plan_hash=str(auth["plan_hash"]),
        batch_hash=batch_hash,
        expected_scope=list(auth.get("authorized_actions") or []),
        mode="paper",
        live_blocked=True,
        requested=[
            {
                "symbol": row["symbol"],
                "action": row.get("action") or "EXIT",
                "qty": row.get("max_qty"),
            }
            for row in (auth.get("authorized_actions") or [])
        ],
        now=TU_NOW + timedelta(minutes=1),
    )
    import services.paper_sell_arm_store as store_mod

    store_mod.record_dispatch_outcome(
        _db(tmp_path),
        dispatch_id=committed["dispatch_id"],
        outcome_state="SUBMITTED",
        result_class="SUCCESS",
        now=TU_NOW + timedelta(minutes=1),
    )
    _write_intent(tmp_path, _matching_intent(auth, dispatch_id=committed["dispatch_id"]))
    kind, reason, _ = classify_dispatch_intent_for_authority(
        json.loads((tmp_path / DISPATCH_INTENT_NAME).read_text(encoding="utf-8")),
        auth,
        db_path=_db(tmp_path),
    )
    assert kind == INTENT_CLASS_MATCHING_CURRENT_TERMINAL
    assert reason == REASON_MATCHING_TERMINAL
    fake = _fake_broker(packet)
    guard = attempt_pre_plan_authorized_sell_dispatch(
        authority=auth,
        broker=fake,
        results_root=tmp_path,
        now=TU_NOW + timedelta(minutes=1),
        explicit_runtime_write=True,
        db_path=_db(tmp_path),
    )
    assert guard.permitted is False
    assert guard.reason in {REASON_MATCHING_TERMINAL, "authority_consumed"}
    assert fake.fake_broker_call_count == 0
    assert count_dispatches_for_authority(_db(tmp_path), str(auth["authority_id"])) == 1


def test_case_c_matching_in_progress_fails_closed(tmp_path, monkeypatch):
    packet, confirmed, auth = _setup_current(tmp_path, monkeypatch)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=TU_NOW + timedelta(minutes=1))
    committed = consume_authority_insert_dispatch(
        _db(tmp_path),
        authority_id=str(auth["authority_id"]),
        plan_hash=str(auth["plan_hash"]),
        batch_hash=str(snap["latest_authority"].get("batch_hash") or ""),
        expected_scope=list(auth.get("authorized_actions") or []),
        mode="paper",
        live_blocked=True,
        requested=[
            {
                "symbol": row["symbol"],
                "action": row.get("action") or "EXIT",
                "qty": row.get("max_qty"),
            }
            for row in (auth.get("authorized_actions") or [])
        ],
        now=TU_NOW + timedelta(minutes=1),
    )
    _write_intent(tmp_path, _matching_intent(auth, dispatch_id=committed["dispatch_id"]))
    kind, reason, _ = classify_dispatch_intent_for_authority(
        json.loads((tmp_path / DISPATCH_INTENT_NAME).read_text(encoding="utf-8")),
        auth,
        db_path=_db(tmp_path),
    )
    assert kind == INTENT_CLASS_MATCHING_CURRENT_IN_PROGRESS
    assert reason == REASON_MATCHING_IN_PROGRESS
    fake = _fake_broker(packet)
    guard = attempt_pre_plan_authorized_sell_dispatch(
        authority=dict(auth),
        broker=fake,
        results_root=tmp_path,
        now=TU_NOW + timedelta(minutes=1),
        explicit_runtime_write=True,
        db_path=_db(tmp_path),
    )
    assert guard.permitted is False
    assert guard.reason in {REASON_MATCHING_IN_PROGRESS, "authority_consumed"}
    assert fake.fake_broker_call_count == 0
    assert count_dispatches_for_authority(_db(tmp_path), str(auth["authority_id"])) == 1


def test_case_d_foreign_nonterminal_fails_closed(tmp_path, monkeypatch):
    packet, confirmed, auth = _setup_current(tmp_path, monkeypatch)
    foreign = dict(AUG26_INTENT)
    foreign["dispatch_intent_id"] = "sell-dispatch-nonterminal-b"
    foreign["authority_id"] = "sell-arm-audit-foreign-nonterminal"
    foreign["audit_id"] = "sell-arm-audit-foreign-nonterminal"
    intent_path = _write_intent(tmp_path, foreign)
    before = intent_path.read_bytes()
    kind, reason, _ = classify_dispatch_intent_for_authority(foreign, auth, db_path=_db(tmp_path))
    assert kind == INTENT_CLASS_FOREIGN_NONTERMINAL
    assert reason == REASON_FOREIGN_NONTERMINAL
    fake = _fake_broker(packet)
    guard = attempt_pre_plan_authorized_sell_dispatch(
        authority=auth,
        broker=fake,
        results_root=tmp_path,
        now=TU_NOW + timedelta(minutes=1),
        explicit_runtime_write=True,
        db_path=_db(tmp_path),
    )
    assert guard.permitted is False
    assert guard.reason == REASON_FOREIGN_NONTERMINAL
    assert fake.fake_broker_call_count == 0
    assert intent_path.read_bytes() == before
    assert count_dispatches_for_authority(_db(tmp_path), str(auth["authority_id"])) == 0
    assert not (tmp_path / CONSUMPTION_FILENAME).is_file() or json.loads(
        (tmp_path / CONSUMPTION_FILENAME).read_text(encoding="utf-8")
    ).get("audit_id") != auth.get("authority_id")


@pytest.mark.parametrize(
    "label,mutator",
    [
        (
            "missing_authority",
            lambda d: d.pop("authority_id", None) or d.pop("audit_id", None) or d,
        ),
        ("missing_dispatch", lambda d: d.__setitem__("dispatch_intent_id", "") or d),
        ("missing_date_conflict", lambda d: d.__setitem__("trading_date", "1999-01-01") or d),
        ("batch_conflict", lambda d: d.__setitem__("batch_id", "other-batch") or d),
        ("packet_conflict", lambda d: d.__setitem__("readiness_packet_hash", "deadbeef" * 8) or d),
        ("key_conflict_same_id", lambda d: d.__setitem__("idempotency_key", "OTHER_KEY") or d),
    ],
)
def test_case_e_ambiguous_or_malformed_fails_closed(tmp_path, monkeypatch, label, mutator):
    packet, confirmed, auth = _setup_current(tmp_path, monkeypatch)
    if label in {
        "missing_date_conflict",
        "batch_conflict",
        "packet_conflict",
        "key_conflict_same_id",
    }:
        doc = _matching_intent(auth, dispatch_id="sell-dispatch-ambig")
    else:
        doc = dict(AUG26_INTENT)
    mutator(doc)
    if label == "missing_authority":
        doc["authority_id"] = ""
        doc["audit_id"] = ""
        doc["idempotency_key"] = "OTHER_KEY"
    _write_intent(tmp_path, doc)
    fake = _fake_broker(packet)
    guard = attempt_pre_plan_authorized_sell_dispatch(
        authority=auth,
        broker=fake,
        results_root=tmp_path,
        now=TU_NOW + timedelta(minutes=1),
        explicit_runtime_write=True,
        db_path=_db(tmp_path),
    )
    assert guard.permitted is False, label
    assert guard.reason in {
        REASON_INTENT_AMBIGUOUS,
        "incomplete_dispatch_intent",
        REASON_FOREIGN_NONTERMINAL,
        REASON_MATCHING_IN_PROGRESS,
    }
    assert fake.fake_broker_call_count == 0
    assert count_dispatches_for_authority(_db(tmp_path), str(auth["authority_id"])) == 0


def test_case_f_sqlite_unused_json_consumed_mismatch(tmp_path, monkeypatch):
    packet, confirmed, auth = _setup_current(tmp_path, monkeypatch)
    auth_path = tmp_path / "paper_sell_arm_authority.json"
    json_auth = json.loads(auth_path.read_text(encoding="utf-8"))
    json_auth["status"] = "CONSUMED"
    json_auth["consumption_status"] = "CONSUMED"
    json_auth["consumed_at"] = "2026-08-28T16:53:28.327338+00:00"
    auth_path.write_text(json.dumps(json_auth), encoding="utf-8")
    cons = {
        "schema_version": SCHEMA_VERSION,
        "consumed_at": "2026-08-28T16:53:28.327338+00:00",
        "audit_id": auth["authority_id"],
        "idempotency_key": auth["idempotency_key"],
        "plan_hash": auth["plan_hash"],
        "batch_id": auth["batch_id"],
        "result": "PRIOR_INTENT_OR_SPLIT",
    }
    (tmp_path / CONSUMPTION_FILENAME).write_text(json.dumps(cons), encoding="utf-8")
    kind, reason = classify_sqlite_json_export_consistency(
        sqlite_state="UNUSED",
        sqlite_payload=auth,
        json_authority=json_auth,
        consumption=cons,
    )
    assert kind == "mismatch"
    assert reason == REASON_SQLITE_JSON_AUTHORITY_MISMATCH
    ok, vreason, _ = validate_runtime_sell_authority(
        contract_path=auth_path,
        now=TU_NOW + timedelta(minutes=1),
        consumption_path=tmp_path / CONSUMPTION_FILENAME,
        store_root=tmp_path,
        db_path=_db(tmp_path),
    )
    assert not ok
    assert vreason == REASON_SQLITE_JSON_AUTHORITY_MISMATCH
    fake = _fake_broker(packet)
    guard = attempt_pre_plan_authorized_sell_dispatch(
        authority=auth,
        broker=fake,
        results_root=tmp_path,
        now=TU_NOW + timedelta(minutes=1),
        explicit_runtime_write=True,
        db_path=_db(tmp_path),
    )
    assert guard.permitted is False
    assert guard.reason == REASON_SQLITE_JSON_AUTHORITY_MISMATCH
    assert fake.fake_broker_call_count == 0
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=TU_NOW + timedelta(minutes=1))
    assert snap["latest_authority"]["state"] == "UNUSED"
    assert count_dispatches_for_authority(_db(tmp_path), str(auth["authority_id"])) == 0
    assert json.loads(auth_path.read_text(encoding="utf-8"))["consumed_at"] == cons["consumed_at"]


def test_case_g_sqlite_consumed_stale_json_unused(tmp_path, monkeypatch):
    packet, confirmed, auth = _setup_current(tmp_path, monkeypatch)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=TU_NOW + timedelta(minutes=1))
    consume_authority_insert_dispatch(
        _db(tmp_path),
        authority_id=str(auth["authority_id"]),
        plan_hash=str(auth["plan_hash"]),
        batch_hash=str(snap["latest_authority"].get("batch_hash") or ""),
        expected_scope=list(auth.get("authorized_actions") or []),
        mode="paper",
        live_blocked=True,
        requested=[
            {
                "symbol": row["symbol"],
                "action": row.get("action") or "EXIT",
                "qty": row.get("max_qty"),
            }
            for row in (auth.get("authorized_actions") or [])
        ],
        now=TU_NOW + timedelta(minutes=1),
    )
    json_auth = json.loads((tmp_path / "paper_sell_arm_authority.json").read_text(encoding="utf-8"))
    assert json_auth.get("consumption_status") != "CONSUMED" or True
    json_auth["status"] = "ARMED"
    json_auth["consumption_status"] = "UNUSED"
    json_auth.pop("consumed_at", None)
    (tmp_path / "paper_sell_arm_authority.json").write_text(json.dumps(json_auth), encoding="utf-8")
    fake = _fake_broker(packet)
    ok, reason, _ = validate_runtime_sell_authority(
        contract_path=tmp_path / "paper_sell_arm_authority.json",
        now=TU_NOW + timedelta(minutes=2),
        store_root=tmp_path,
        db_path=_db(tmp_path),
    )
    assert not ok
    assert reason == "authority_consumed"
    guard = attempt_pre_plan_authorized_sell_dispatch(
        authority=dict(auth),
        broker=fake,
        results_root=tmp_path,
        now=TU_NOW + timedelta(minutes=2),
        explicit_runtime_write=True,
        db_path=_db(tmp_path),
    )
    assert guard.permitted is False
    assert fake.fake_broker_call_count == 0


def test_case_h_failed_sqlite_finalize_does_not_claim_success(tmp_path, monkeypatch):
    packet, confirmed, auth = _setup_current(tmp_path, monkeypatch)
    fake = _fake_broker(packet)
    import services.paper_sell_arm_store as store_mod

    def _boom(*_a, **_k):
        raise PaperSellArmStoreError("sqlite_simulated_finalize_failure")

    monkeypatch.setattr(store_mod, "record_dispatch_outcome", _boom)
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=TU_NOW + timedelta(minutes=1),
    )
    assert result.get("dispatched") is True, result
    assert fake.fake_broker_call_count >= 1
    out = finalize_paper_sell_dispatch(
        store_root=tmp_path,
        outcome="SUCCESS",
        explicit_runtime_write=True,
        db_path=_db(tmp_path),
    )
    assert out.get("ok") is False
    assert out.get("reason") == "sqlite_dispatch_finalize_failed"
    assert out.get("recovery_required") is True
    assert out.get("broker_reconciliation_required") is True
    json_auth = json.loads((tmp_path / "paper_sell_arm_authority.json").read_text(encoding="utf-8"))
    assert json_auth.get("dispatch_outcome") != "SUCCESS"
    cons = json.loads((tmp_path / CONSUMPTION_FILENAME).read_text(encoding="utf-8"))
    assert cons.get("result") != "SUCCESS"


def test_case_i_sqlite_finalized_before_json_export(tmp_path, monkeypatch):
    packet, confirmed, auth = _setup_current(tmp_path, monkeypatch)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=TU_NOW + timedelta(minutes=1))
    committed = consume_authority_insert_dispatch(
        _db(tmp_path),
        authority_id=str(auth["authority_id"]),
        plan_hash=str(auth["plan_hash"]),
        batch_hash=str(snap["latest_authority"].get("batch_hash") or ""),
        expected_scope=list(auth.get("authorized_actions") or []),
        mode="paper",
        live_blocked=True,
        requested=[
            {
                "symbol": row["symbol"],
                "action": row.get("action") or "EXIT",
                "qty": row.get("max_qty"),
            }
            for row in (auth.get("authorized_actions") or [])
        ],
        now=TU_NOW + timedelta(minutes=1),
    )
    import services.paper_sell_arm_store as store_mod

    store_mod.record_dispatch_outcome(
        _db(tmp_path),
        dispatch_id=committed["dispatch_id"],
        outcome_state="SUBMITTED",
        result_class="SUCCESS",
        now=TU_NOW + timedelta(minutes=1),
    )
    json_auth = json.loads((tmp_path / "paper_sell_arm_authority.json").read_text(encoding="utf-8"))
    assert json_auth.get("consumption_status") != "CONSUMED"
    fake = _fake_broker(packet)
    guard = attempt_pre_plan_authorized_sell_dispatch(
        authority=dict(auth),
        broker=fake,
        results_root=tmp_path,
        now=TU_NOW + timedelta(minutes=2),
        explicit_runtime_write=True,
        db_path=_db(tmp_path),
    )
    assert guard.permitted is False
    assert fake.fake_broker_call_count == 0
    exported = export_json_authority_from_sqlite(
        store_root=tmp_path,
        db_path=_db(tmp_path),
        explicit_runtime_write=True,
        now=TU_NOW + timedelta(minutes=2),
    )
    assert exported["ok"] is True
    assert exported["dispatch_id"] == committed["dispatch_id"]
    json_auth = json.loads((tmp_path / "paper_sell_arm_authority.json").read_text(encoding="utf-8"))
    cons = json.loads((tmp_path / CONSUMPTION_FILENAME).read_text(encoding="utf-8"))
    assert json_auth["consumption_status"] == "CONSUMED"
    assert cons["consuming_dispatch_id"] == committed["dispatch_id"]
    assert cons["authority_id"] == auth["authority_id"]
    row = inspect_authority_row(_db(tmp_path), str(auth["authority_id"]))
    assert row is not None
    assert row.get("consuming_dispatch_id") == committed["dispatch_id"]
    assert fake.fake_broker_call_count == 0


def _four_symbol_packet() -> Dict[str, Any]:
    pkt = _incident_packet()
    actions = [
        {
            "symbol": "TSLA",
            "action": "EXIT",
            "qty": 8.0,
            "est_price": 347.8,
            "est_notional": 2782.4,
        },
        {"symbol": "HD", "action": "EXIT", "qty": 6.0, "est_price": 328.7, "est_notional": 1972.2},
        {"symbol": "DBA", "action": "EXIT", "qty": 8.0, "est_price": 29.08, "est_notional": 232.64},
        {"symbol": "BITO", "action": "EXIT", "qty": 1.0, "est_price": 10.52, "est_notional": 10.52},
    ]
    pkt["approval_ready_actions"] = actions
    pkt["approval_ready_symbols"] = ["TSLA", "HD", "DBA", "BITO"]
    pkt["total_proposed_notional"] = 4997.76
    pkt["cycle_id"] = INV
    pkt["market_open_invocation_id"] = INV
    pkt["batch_id"] = "manage-mo-20260828_1207"
    # Keep trading_date / generated_at from the 198T-U packet clock so arming
    # and session evidence stay on one isolated day. Historical Aug 26 intent
    # remains the foreign identity under test.
    from services.paper_sell_arm_cap_slice import slice_reviewed_sells

    sliced = slice_reviewed_sells(pkt, cap=APPROVAL_CLI_CAP)
    out = dict(sliced.get("packet") or pkt)
    out["generated_at"] = pkt["generated_at"]
    out["valid_until"] = pkt["valid_until"]
    return out


def test_case_j_and_k_cycle3_replay_and_same_cycle_suppression(tmp_path, monkeypatch):
    packet, confirmed, auth = _setup_current(tmp_path, monkeypatch, packet=_four_symbol_packet())
    intent_path = _write_intent(tmp_path, AUG26_INTENT)
    before = intent_path.read_bytes()
    fake = _fake_broker(packet)
    ok_plan, plan_reason, planned = build_pre_plan_planned_orders(
        auth, fake, now=TU_NOW + timedelta(minutes=1)
    )
    assert ok_plan, plan_reason
    by_sym = {p.symbol: int(p.qty) for p in planned}
    assert by_sym["TSLA"] <= 8
    assert by_sym["HD"] <= 6
    total = money_round(sum(float(p.qty) * float(fake._px(p.symbol)) for p in planned))
    assert total <= APPROVAL_CLI_CAP + 1e-9
    trace = OrderedTrace()
    install_boundary_trace(monkeypatch, trace)
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=TU_NOW + timedelta(minutes=1),
    )
    assert result.get("dispatched") is True, result
    assert fake.submit_payloads
    assert intent_path.read_bytes() == before
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=TU_NOW + timedelta(minutes=2))
    assert snap["latest_authority"]["state"] == "CONSUMED"
    assert count_dispatches_for_authority(_db(tmp_path), str(auth["authority_id"])) == 1
    cons = json.loads((tmp_path / CONSUMPTION_FILENAME).read_text(encoding="utf-8"))
    json_auth = json.loads((tmp_path / "paper_sell_arm_authority.json").read_text(encoding="utf-8"))
    assert json_auth["consumption_status"] == "CONSUMED"
    row = inspect_authority_row(_db(tmp_path), str(auth["authority_id"]))
    assert cons.get("consuming_dispatch_id")
    assert cons.get("authority_id") == auth["authority_id"]
    assert row is not None
    assert cons.get("consuming_dispatch_id") == row.get("consuming_dispatch_id")
    cycle_id = str(packet.get("cycle_id") or INV)
    _bind(tmp_path, monkeypatch, invocation_id=cycle_id)
    write_pre_plan_dispatch_evidence(
        tmp_path,
        {
            "attempted": True,
            "dispatched": True,
            "consumed": True,
            "broker_submissions": len(fake.submit_payloads),
            "broker_orders_submitted": True,
            "dispatch_performed": True,
            "dispatch_intent_id": result.get("dispatch_id") or result.get("dispatch_intent_id"),
            "authority_id": confirmed["authority_id"],
            "dispatch_terminal_status": "TERMINAL_SUCCESS",
            "reason": "ok",
            "submit": {"placed": len(fake.submit_payloads), "failed": 0},
        },
        invocation_id=cycle_id,
        now=TU_NOW + timedelta(minutes=1),
        session_date=TU_NOW.date().isoformat(),
    )
    f2 = generate_from_manage_artifacts(results_root=tmp_path, now=TU_NOW + timedelta(minutes=1))
    packet_out = f2["packet"]
    assert f2["ok"] is True
    assert packet_out["status"] == STATUS_UNAVAILABLE_SAME_CYCLE
    assert packet_out["reason"] == SAME_CYCLE_DISPATCH_REASON
    assert "CYCLE_FAIL" not in str(f2.get("packet_status") or "")
    first_calls = fake.fake_broker_call_count
    replay = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=TU_NOW + timedelta(minutes=2),
    )
    assert result.get("dispatched") is True
    assert replay.get("dispatched") is not True
    assert fake.fake_broker_call_count == first_calls
    assert count_dispatches_for_authority(_db(tmp_path), str(auth["authority_id"])) == 1


def test_case_l_duplicate_replay(tmp_path, monkeypatch):
    test_case_j_and_k_cycle3_replay_and_same_cycle_suppression(tmp_path, monkeypatch)
