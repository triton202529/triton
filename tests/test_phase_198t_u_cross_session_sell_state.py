"""Phase 198T-U — isolated cross-session SELL consumption and dispatch-count repair.

PAPER ONLY. Isolated tmp_path SQLite/JSON. Fake broker only. No production writes.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from services.defensive_exit_approval_readiness import (
    SAME_CYCLE_DISPATCH_REASON,
    STATUS_UNAVAILABLE_SAME_CYCLE,
    generate_from_manage_artifacts,
    write_pre_plan_dispatch_evidence,
)
from services.manage_positions_cycle_binding import (
    BLOCKER_MISSING_SELL_AUTHORITY,
    INVOCATION_ENV,
    write_cycle_binding,
)
from services.paper_sell_arm_authority import (
    APPROVAL_CLI_CAP,
    SCHEMA_VERSION,
    classify_consumption_record_for_authority,
    readiness_packet_hash,
    validate_paper_sell_arm_authority,
)
from services.paper_sell_arm_cap_slice import money_round, slice_reviewed_sells
from services.paper_sell_arm_console import (
    CONFIRM_PAPER_SELLS_PHRASE,
    CONFIRM_SUCCESS,
    arm_reviewed_sells,
    classify_post_create_confirm,
    confirm_and_arm_paper_sells,
    operator_fail_message,
)
from services.paper_sell_arm_runtime_enforcement import (
    _ACTIVE_PERMIT,
    _authorized_clip_audit,
    attempt_pre_plan_authorized_sell_dispatch,
    build_pre_plan_planned_orders,
    find_unused_pre_plan_authority,
    validate_runtime_sell_authority,
)
from services.paper_sell_arm_store import (
    STORE_SCHEMA_VERSION,
    canonical_json,
    consume_authority_insert_dispatch,
    count_dispatches_for_authority,
    health_check,
    inspect_sqlite_arm_snapshot,
    write_connection,
)
from services.place_live_orders import dispatch_unused_authorized_sell_slice_pre_plan
from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    LIVE_BLOCKED,
    OrderedTrace,
    block_real_network,
    freeze_dispatch_clock,
    install_boundary_trace,
    isolate_sell_store,
    stub_unrelated_pre_gates,
)
from tests.test_phase_198n_c_sell_cap_slice import (
    PacketQuotedFakeBroker,
    _db,
    _pres,
    _write_buy_arm,
    _write_market_session_evidence,
    _write_packet,
)

CONSUMPTION_FILENAME = "paper_sell_arm_consumption.json"
CHALLENGE_FILENAME = "paper_sell_confirmation_challenge.json"

ISO_ROOT = Path(__file__).resolve().parents[1]
PROD_ROOT = Path(r"C:\Users\akimw\triton")
SEED = ISO_ROOT / "_isolated_seed"
NOW = datetime(2026, 8, 27, 16, 52, 0, tzinfo=timezone.utc)
INV = "mo-20260827_125232-7a028b8dca724be196f0e5da3d3d5d8f"
INV_LATER = "mo-20260827_133902-a83e1742448649c5b8e3ccd76ac632d3"
OLD_AUTH = "sell-arm-audit-d55860b9ade2"
OLD_CHAL = "sell-chal-6c8daba906c64fc5"
OLD_DISPATCH = "sell-dispatch-7016ea27eadc41e1"
NEW_AUTH = "sell-arm-audit-db6e7e5e0cce"
GENERIC_RED = "Paper SELL arming failed closed. No orders were submitted."


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
    _assert_temp_only(tmp_path)
    yield
    _ACTIVE_PERMIT.set(None)


def _assert_temp_only(path: Path) -> None:
    resolved = path.resolve()
    text = str(resolved).lower()
    assert "\\appdata\\local\\temp\\" in text or "/appdata/local/temp/" in text
    prod = str(PROD_ROOT.resolve()).lower()
    assert not text.startswith(prod + os.sep.lower())
    assert resolved != PROD_ROOT.resolve()


def _incident_packet() -> Dict[str, Any]:
    actions = [
        {
            "symbol": "TSLA",
            "action": "EXIT",
            "qty": 8.0,
            "est_price": 351.98,
            "est_notional": 2815.84,
        },
        {
            "symbol": "HD",
            "action": "EXIT",
            "qty": 6.0,
            "est_price": 330.61,
            "est_notional": 1983.66,
        },
        {
            "symbol": "DBA",
            "action": "EXIT",
            "qty": 6.0,
            "est_price": 28.72,
            "est_notional": 172.32,
        },
    ]
    pkt = {
        "schema_version": "198I-ZA-SELL-READINESS-1",
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "generation_status": "ok",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "generated_at": NOW.isoformat(),
        "valid_until": (NOW + timedelta(minutes=40)).isoformat(),
        "cycle_id": INV,
        "run_id": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "batch_id": "manage-mo-20260827_1252",
        "plan_id": f"manage_positions_plan@{NOW.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "plan_hash": "planhash198tu0000000000000000000000000000000000000000000000000001",
        "approval_ready_symbols": ["TSLA", "HD", "DBA"],
        "approval_ready_actions": actions,
        "total_proposed_notional": 4971.82,
        "trading_date": NOW.date().isoformat(),
        "market_open_invocation_id": INV,
        "source_plan_sha256": "",
        "selected_row_hash": "",
        "packet_status": "AUTHORITY_READY",
        "arm_eligibility": "AUTHORITY_READY",
    }
    sliced = slice_reviewed_sells(pkt, cap=APPROVAL_CLI_CAP)
    out = dict(sliced.get("packet") or pkt)
    out["generated_at"] = pkt["generated_at"]
    out["valid_until"] = pkt["valid_until"]
    return out


def _write_plan_artifacts(tmp_path: Path, packet: Dict[str, Any]) -> Dict[str, Any]:
    from services.manage_positions_cycle_binding import selected_row_hash, sha256_file_bytes

    rows = []
    for a in packet.get("approval_ready_actions") or []:
        rows.append(
            {
                "symbol": a["symbol"],
                "final_action": a["action"],
                "qty": str(int(a["qty"])),
                "selected_for_execution": "true",
                "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
                "skip_reason": "",
                "current_position_qty": str(int(a["qty"])),
                "current_price": str(a.get("est_price") or 0),
                "planned_notional": str(a.get("est_notional") or 0),
            }
        )
    csv_path = tmp_path / "manage_positions_plan.csv"
    tmp_path.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    pkt = dict(packet)
    pkt["source_plan_sha256"] = sha256_file_bytes(csv_path)
    pkt["selected_row_hash"] = selected_row_hash(rows)
    pkt["plan_hash"] = readiness_packet_hash(pkt)
    _write_packet(tmp_path, pkt)
    (tmp_path / "manage_positions_plan.json").write_text(
        json.dumps({"plan_hash": pkt["plan_hash"], "timestamp": NOW.isoformat()}),
        encoding="utf-8",
    )
    pos_lines = ["symbol,qty,qty_available,current_price"]
    for a in pkt.get("approval_ready_actions") or []:
        pos_lines.append(f"{a['symbol']},{int(a['qty'])},{int(a['qty'])},{a.get('est_price', 100)}")
    (tmp_path / "positions_snapshot.csv").write_text("\n".join(pos_lines) + "\n", encoding="utf-8")
    _write_market_session_evidence(tmp_path, now=NOW)
    _write_buy_arm(tmp_path, now=NOW)
    return pkt


def _write_historical_consumption(tmp_path: Path) -> Dict[str, Any]:
    cons = json.loads((SEED / "paper_sell_arm_consumption.json").read_text(encoding="utf-8"))
    (tmp_path / CONSUMPTION_FILENAME).write_text(json.dumps(cons, indent=2), encoding="utf-8")
    return cons


def _seed_historical_b(db_path: Path) -> None:
    ok, reason, _ = health_check(db_path)
    assert ok, reason
    created = "2026-08-26T15:19:00+00:00"
    consumed = "2026-08-26T15:20:50.074550+00:00"
    plan = "18c3b9cdb620ac45c59466440aeef887f9023d799a7119be8eca13f4a655c02a"
    batch_id = "manage-mo-20260826_1032"
    batch_hash = "historical-batch-hash-aug26-b"
    payload = {
        "authority_id": OLD_AUTH,
        "audit_id": OLD_AUTH,
        "consumption_status": "CONSUMED",
        "status": "CONSUMED",
        "plan_hash": plan,
        "batch_id": batch_id,
    }
    with write_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO sell_challenges(
                challenge_id, plan_hash, batch_hash, batch_id, canonical_batch_json,
                payload_json, created_at, expires_at, state, consumed_at,
                consuming_authority_id, operator_action_label, schema_version
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                OLD_CHAL,
                plan,
                batch_hash,
                batch_id,
                canonical_json([]),
                canonical_json({"challenge_id": OLD_CHAL}),
                created,
                consumed,
                "CONSUMED",
                consumed,
                OLD_AUTH,
                "Arm Reviewed Sells",
                STORE_SCHEMA_VERSION,
            ),
        )
        conn.execute(
            """
            INSERT INTO sell_authorities(
                authority_id, challenge_id, plan_hash, batch_hash, canonical_batch_json,
                payload_json, created_at, expires_at, single_use, state,
                consumption_time, consuming_dispatch_id, final_result, schema_version
            ) VALUES (?,?,?,?,?,?,?,?,1,'CONSUMED',?,?,?,?)
            """,
            (
                OLD_AUTH,
                OLD_CHAL,
                plan,
                batch_hash,
                canonical_json([]),
                canonical_json(payload),
                created,
                consumed,
                consumed,
                OLD_DISPATCH,
                "DISPATCH_INTENT",
                STORE_SCHEMA_VERSION,
            ),
        )
        conn.execute(
            """
            INSERT INTO sell_dispatches(
                dispatch_id, authority_id, plan_hash, batch_hash, canonical_batch_json,
                intent_json, created_at, state, reconciliation_required,
                symbols_submitted_json, schema_version
            ) VALUES (?,?,?,?,?,?,?,'SUBMITTED',0,'["TSLA"]',?)
            """,
            (
                OLD_DISPATCH,
                OLD_AUTH,
                plan,
                batch_hash,
                canonical_json([]),
                canonical_json({"dispatch_id": OLD_DISPATCH, "authority_id": OLD_AUTH}),
                consumed,
                STORE_SCHEMA_VERSION,
            ),
        )
        conn.execute("COMMIT")


def _arm_confirm(tmp_path: Path, packet: Dict[str, Any], *, now: datetime = NOW) -> Dict[str, Any]:
    pres = _pres(tmp_path, packet, now=now)
    armed = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=now,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
        expected_presentation_token=pres["presentation_token"],
    )
    assert armed["ok"] is True, armed
    confirmed = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=armed["challenge_id"],
        now=now + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    return confirmed


def _fake_broker(packet: Dict[str, Any]) -> PacketQuotedFakeBroker:
    positions = []
    for row in packet.get("approval_ready_actions") or []:
        positions.append(
            {
                "symbol": row["symbol"],
                "qty": str(int(row["qty"])),
                "qty_available": str(int(row["qty"])),
            }
        )
    return PacketQuotedFakeBroker(packet, positions=positions)


def _bind(results: Path, monkeypatch, *, invocation_id: str, ts: str | None = None) -> None:
    monkeypatch.setenv(INVOCATION_ENV, invocation_id)
    plan = json.loads((results / "manage_positions_plan.json").read_text(encoding="utf-8"))
    write_cycle_binding(
        results_root=results,
        invocation_id=invocation_id,
        plan_generated_at=str(plan.get("timestamp") or ts or NOW.isoformat()),
        f_stage_status="COMPLETE",
        now=NOW,
        remaining_blocker=BLOCKER_MISSING_SELL_AUTHORITY,
        operator_execution_required=True,
        block_reasons=["DISPATCH_BLOCKED_APPROVAL_REQUIRED"],
        orders_executed=0,
    )


def test_case_a_historical_consumption_ignored(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW)
    packet = _write_plan_artifacts(tmp_path, _incident_packet())
    _seed_historical_b(_db(tmp_path))
    cons = _write_historical_consumption(tmp_path)
    confirmed = _arm_confirm(tmp_path, packet)
    assert confirmed["ok"] is True, confirmed
    auth = find_unused_pre_plan_authority(db_path=_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert auth is not None
    assert str(auth.get("authority_id")) != OLD_AUTH
    ok, reason, _ = validate_paper_sell_arm_authority(
        auth, now=NOW + timedelta(minutes=1), consumption=cons
    )
    assert ok, reason
    kind, cons_reason = classify_consumption_record_for_authority(auth, cons)
    assert kind == "ignore"
    assert cons_reason == ""
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert snap["dispatch_count"] == 1
    assert count_dispatches_for_authority(_db(tmp_path), OLD_AUTH) == 1
    assert count_dispatches_for_authority(_db(tmp_path), str(auth["authority_id"])) == 0


def test_case_b_matching_consumption_blocks_no_broker(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW)
    stub_unrelated_pre_gates(monkeypatch)
    packet = _write_plan_artifacts(tmp_path, _incident_packet())
    confirmed = _arm_confirm(tmp_path, packet)
    assert confirmed["ok"] is True, confirmed
    auth = find_unused_pre_plan_authority(db_path=_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert auth is not None
    matching = {
        "schema_version": SCHEMA_VERSION,
        "consumed_at": NOW.isoformat(),
        "audit_id": auth["authority_id"],
        "idempotency_key": auth["idempotency_key"],
        "plan_hash": auth["plan_hash"],
        "batch_id": auth["batch_id"],
        "result": "STARTED",
        "consumption_rule": "CONSUME_ON_FIRST_DISPATCH_ATTEMPT",
    }
    (tmp_path / CONSUMPTION_FILENAME).write_text(json.dumps(matching), encoding="utf-8")
    ok, reason, _ = validate_paper_sell_arm_authority(
        auth, now=NOW + timedelta(minutes=1), consumption=matching
    )
    assert not ok
    assert reason == "consumption_record_present"
    fake = _fake_broker(packet)
    guard = attempt_pre_plan_authorized_sell_dispatch(
        authority=auth,
        broker=fake,
        results_root=tmp_path,
        now=NOW + timedelta(minutes=1),
        explicit_runtime_write=True,
        db_path=_db(tmp_path),
    )
    assert guard.permitted is False
    assert guard.reason == "consumption_record_present"
    assert fake.fake_broker_call_count == 0
    assert not fake.submit_payloads


def test_case_c_sqlite_consumed_blocks(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW)
    packet = _write_plan_artifacts(tmp_path, _incident_packet())
    confirmed = _arm_confirm(tmp_path, packet)
    assert confirmed["ok"] is True, confirmed
    auth = find_unused_pre_plan_authority(db_path=_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert auth is not None
    snap0 = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    batch_hash = str(
        snap0.get("latest_authority", {}).get("batch_hash") or auth.get("batch_hash") or ""
    )
    consume_authority_insert_dispatch(
        _db(tmp_path),
        authority_id=str(auth["authority_id"]),
        plan_hash=str(auth["plan_hash"]),
        batch_hash=batch_hash,
        expected_scope=list(auth.get("authorized_scope") or auth.get("authorized_actions") or []),
        mode="paper",
        live_blocked=True,
        requested=[
            {
                "symbol": row["symbol"],
                "action": row.get("action") or "EXIT",
                "qty": row.get("max_qty") or row.get("qty"),
            }
            for row in (auth.get("authorized_scope") or auth.get("authorized_actions") or [])
        ],
        now=NOW + timedelta(minutes=1),
    )
    cons_path = tmp_path / CONSUMPTION_FILENAME
    if cons_path.is_file():
        cons_path.unlink()
    _write_historical_consumption(tmp_path)
    fake = _fake_broker(packet)
    ok, reason, _ = validate_runtime_sell_authority(
        contract_path=tmp_path / "paper_sell_arm_authority.json",
        now=NOW + timedelta(minutes=2),
        consumption_path=cons_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
    )
    assert not ok
    assert reason == "authority_consumed"
    guard = attempt_pre_plan_authorized_sell_dispatch(
        authority=dict(auth),
        broker=fake,
        results_root=tmp_path,
        now=NOW + timedelta(minutes=2),
        explicit_runtime_write=True,
        db_path=_db(tmp_path),
    )
    assert guard.permitted is False
    assert fake.fake_broker_call_count == 0


def test_case_d_malformed_current_record_fails_closed():
    auth = {
        "authority_id": NEW_AUTH,
        "audit_id": NEW_AUTH,
        "idempotency_key": (
            "sell-arm|2ee4db9de29650f8814171b0e9170e815225ad882c1c84cb1a9856077667164d|"
            "manage-mo-20260827_1205|sell-chal-faf7597b734e4695"
        ),
        "plan_hash": "2ee4db9de29650f8814171b0e9170e815225ad882c1c84cb1a9856077667164d",
        "batch_id": "manage-mo-20260827_1205",
        "confirmation_challenge_id": "sell-chal-faf7597b734e4695",
    }
    mixed = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": NEW_AUTH,
        "idempotency_key": (
            "sell-arm|18c3b9cdb620ac45c59466440aeef887f9023d799a7119be8eca13f4a655c02a|"
            "manage-mo-20260826_1032|sell-chal-6c8daba906c64fc5"
        ),
        "plan_hash": "18c3b9cdb620ac45c59466440aeef887f9023d799a7119be8eca13f4a655c02a",
        "batch_id": "manage-mo-20260826_1032",
    }
    kind, reason = classify_consumption_record_for_authority(auth, mixed)
    assert kind == "fail"
    assert reason == "consumption_identity_ambiguous"
    ok, vreason, _ = validate_paper_sell_arm_authority(
        {
            **json.loads((SEED / "paper_sell_arm_authority.json").read_text(encoding="utf-8")),
            "consumption_status": "UNUSED",
            "status": "ARMED",
        },
        now=NOW,
        consumption=mixed,
    )
    assert not ok
    assert vreason == "consumption_identity_ambiguous"
    empty_kind, empty_reason = classify_consumption_record_for_authority(auth, {})
    assert empty_kind == "fail"
    assert empty_reason == "consumption_record_malformed"


def test_case_e_historical_global_dispatch_does_not_false_deny(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW)
    packet = _write_plan_artifacts(tmp_path, _incident_packet())
    _seed_historical_b(_db(tmp_path))
    _write_historical_consumption(tmp_path)
    confirmed = _arm_confirm(tmp_path, packet)
    assert confirmed["ok"] is True, confirmed
    assert confirmed["reason"] is None
    assert confirmed["message"] == CONFIRM_SUCCESS
    assert confirmed["dispatch_count"] == 0
    assert GENERIC_RED not in str(confirmed.get("message") or "")
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert snap["dispatch_count"] == 1
    created = str(confirmed["authority_id"])
    assert created != OLD_AUTH
    assert count_dispatches_for_authority(_db(tmp_path), created) == 0
    assert count_dispatches_for_authority(_db(tmp_path), OLD_AUTH) == 1
    conn = sqlite3.connect(str(_db(tmp_path)))
    try:
        row = conn.execute(
            "SELECT state FROM sell_challenges WHERE challenge_id=?",
            (confirmed["challenge_id"],),
        ).fetchone()
        assert row[0] == "CONSUMED"
        n = conn.execute("SELECT COUNT(*) FROM sell_challenges WHERE state='CONSUMED'").fetchone()
        assert int(n[0]) >= 1
    finally:
        conn.close()


def test_case_f_same_authority_dispatch_prevents_unused_success(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW)
    packet = _write_plan_artifacts(tmp_path, _incident_packet())
    import services.paper_sell_arm_console as console_mod

    monkeypatch.setattr(console_mod, "count_dispatches_for_authority", lambda *_a, **_k: 1)
    confirmed = _arm_confirm(tmp_path, packet)
    assert confirmed["ok"] is False
    assert confirmed["reason"] == "authority_dispatch_already_present"
    assert confirmed["message"] != GENERIC_RED
    assert "dispatch" in confirmed["message"].lower()
    reason = classify_post_create_confirm(
        created_id=NEW_AUTH,
        latest_authority={"authority_id": NEW_AUTH, "state": "UNUSED"},
        challenge_consumed=True,
        unused=True,
        unexpired=True,
        current_unused=1,
        created_dispatch_count=1,
    )
    assert reason == "authority_dispatch_already_present"


def test_case_g_duplicate_confirmation(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW)
    packet = _write_plan_artifacts(tmp_path, _incident_packet())
    first = _arm_confirm(tmp_path, packet)
    assert first["ok"] is True, first
    before = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    second = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=first["challenge_id"],
        now=NOW + timedelta(minutes=2),
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    assert second["ok"] is False
    assert second["reason"] == "challenge_already_consumed"
    after = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=2))
    assert after["authority_count"] == before["authority_count"] == 1
    unused = count_dispatches_for_authority(_db(tmp_path), str(first["authority_id"]))
    assert unused == 0
    fake = _fake_broker(packet)
    assert fake.fake_broker_call_count == 0


def test_case_h_august27_incident_replay(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    packet = _write_plan_artifacts(tmp_path, _incident_packet())
    _seed_historical_b(_db(tmp_path))
    cons = _write_historical_consumption(tmp_path)
    confirmed = _arm_confirm(tmp_path, packet)
    assert confirmed["ok"] is True, confirmed
    auth = find_unused_pre_plan_authority(db_path=_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert auth is not None
    ok, reason, _ = validate_paper_sell_arm_authority(
        auth, now=NOW + timedelta(minutes=1), consumption=cons
    )
    assert ok, reason
    fake = _fake_broker(packet)
    ok_plan, plan_reason, planned = build_pre_plan_planned_orders(
        auth, fake, now=NOW + timedelta(minutes=1)
    )
    assert ok_plan, plan_reason
    by_sym = {p.symbol: int(p.qty) for p in planned}
    assert by_sym == {"TSLA": 8, "HD": 6, "DBA": 6}
    audit = {r["symbol"]: r for r in _authorized_clip_audit(auth, planned)}
    for sym in by_sym:
        assert audit[sym]["outcome"] in {"UNCHANGED", "CLIPPED_DOWN"}
        assert float(audit[sym].get("submitted_qty") or by_sym[sym]) <= float(
            auth["per_symbol_max_quantity"][sym]
        )
    total = money_round(sum(float(p.qty) * float(fake._px(p.symbol)) for p in planned))
    assert total <= APPROVAL_CLI_CAP + 1e-9
    trace = OrderedTrace()
    install_boundary_trace(monkeypatch, trace)
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
        verbose=True,
    )
    assert result.get("dispatched") is True, result
    submitted = {str(p.get("symbol") or "").upper() for p in fake.submit_payloads}
    assert submitted <= {"TSLA", "HD", "DBA"}
    assert submitted
    for payload in fake.submit_payloads:
        assert str(payload.get("side") or "").lower() == "sell"
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=2))
    assert snap["latest_authority"]["state"] == "CONSUMED"
    assert count_dispatches_for_authority(_db(tmp_path), OLD_AUTH) == 1
    assert count_dispatches_for_authority(_db(tmp_path), str(auth["authority_id"])) == 1
    leftover = json.loads((tmp_path / CONSUMPTION_FILENAME).read_text(encoding="utf-8"))
    assert leftover.get("audit_id") in {OLD_AUTH, auth["authority_id"], auth.get("audit_id")}
    conn = sqlite3.connect(str(_db(tmp_path)))
    try:
        old = conn.execute(
            "SELECT dispatch_id, authority_id FROM sell_dispatches WHERE authority_id=?",
            (OLD_AUTH,),
        ).fetchone()
        assert old[0] == OLD_DISPATCH
        n = conn.execute("SELECT COUNT(*) FROM sell_dispatches").fetchone()[0]
        assert int(n) == 2
    finally:
        conn.close()
    seed_auth = json.loads((SEED / "paper_sell_arm_authority.json").read_text(encoding="utf-8"))
    seed_cons = json.loads((SEED / "paper_sell_arm_consumption.json").read_text(encoding="utf-8"))
    ok_seed, seed_reason, _ = validate_paper_sell_arm_authority(
        seed_auth, now=NOW, consumption=seed_cons
    )
    assert ok_seed, seed_reason
    assert seed_auth["authority_id"] == NEW_AUTH
    assert seed_cons["audit_id"] == OLD_AUTH
    tmp_path.joinpath("_case_h_result.json").write_text(
        json.dumps({"result": result, "submitted": sorted(submitted)}, default=str),
        encoding="utf-8",
    )


def test_case_i_same_cycle_suppression_after_fake_dispatch(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    packet = _write_plan_artifacts(tmp_path, _incident_packet())
    _seed_historical_b(_db(tmp_path))
    _write_historical_consumption(tmp_path)
    confirmed = _arm_confirm(tmp_path, packet)
    assert confirmed["ok"] is True, confirmed
    fake = _fake_broker(packet)
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
    )
    assert result.get("dispatched") is True, result
    _bind(tmp_path, monkeypatch, invocation_id=INV)
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
        invocation_id=INV,
        now=NOW + timedelta(minutes=1),
        session_date=NOW.date().isoformat(),
    )
    f2 = generate_from_manage_artifacts(results_root=tmp_path, now=NOW + timedelta(minutes=1))
    packet_out = f2["packet"]
    assert f2["ok"] is True
    assert packet_out["status"] == STATUS_UNAVAILABLE_SAME_CYCLE
    assert packet_out["reason"] == SAME_CYCLE_DISPATCH_REASON
    assert packet_out["creates_no_authority"] is True
    assert packet_out.get("approval_ready_actions") == []
    assert f2.get("packet_status") in {SAME_CYCLE_DISPATCH_REASON, STATUS_UNAVAILABLE_SAME_CYCLE}
    assert "CYCLE_FAIL" not in str(f2.get("packet_status") or "")
    later_rows = [
        {
            "symbol": "TSLA",
            "final_action": "EXIT",
            "qty": "8",
            "selected_for_execution": "true",
            "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
            "skip_reason": "",
            "current_position_qty": "8",
            "current_price": "351.98",
            "planned_notional": "2815.84",
        }
    ]
    later = NOW + timedelta(hours=1)
    with (tmp_path / "manage_positions_plan.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(later_rows[0].keys()))
        writer.writeheader()
        writer.writerows(later_rows)
    (tmp_path / "manage_positions_plan.json").write_text(
        json.dumps({"plan_hash": "later", "timestamp": later.isoformat()}),
        encoding="utf-8",
    )
    monkeypatch.setenv(INVOCATION_ENV, INV_LATER)
    write_cycle_binding(
        results_root=tmp_path,
        invocation_id=INV_LATER,
        plan_generated_at=later.isoformat(),
        f_stage_status="COMPLETE",
        now=later,
        remaining_blocker=BLOCKER_MISSING_SELL_AUTHORITY,
        operator_execution_required=True,
        block_reasons=["DISPATCH_BLOCKED_APPROVAL_REQUIRED"],
        orders_executed=0,
    )
    write_pre_plan_dispatch_evidence(
        tmp_path,
        {
            "attempted": False,
            "dispatched": False,
            "consumed": False,
            "broker_submissions": 0,
            "broker_orders_submitted": False,
            "dispatch_performed": False,
            "reason": "no_unused_authority",
        },
        invocation_id=INV_LATER,
        now=later,
        session_date=later.date().isoformat(),
    )
    later_f2 = generate_from_manage_artifacts(results_root=tmp_path, now=later)
    assert (
        later_f2["packet"].get("status") != STATUS_UNAVAILABLE_SAME_CYCLE
        or later_f2["packet"].get("invocation_id") == INV_LATER
    )
    assert later_f2["packet"].get("creates_no_authority") is True


def test_writable_paths_stay_under_temp(tmp_path):
    _assert_temp_only(tmp_path)
    _assert_temp_only(ISO_ROOT)
    _assert_temp_only(_db(tmp_path).parent)
    assert operator_fail_message("unknown_exception") == GENERIC_RED
    assert "dispatch" in operator_fail_message("authority_dispatch_already_present").lower()
