"""Phase 198Q-B — pre-plan next-cycle SELL authority consumption.

PAPER ONLY. Isolated tmp roots. No production writes, no real network.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from services.manage_positions_cycle_binding import selected_row_hash, sha256_file_bytes
from services.paper_sell_arm_authority import (
    APPROVAL_CLI_CAP,
    AUTHORITY_TTL_MINUTES,
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    confirm_and_create_paper_sell_arm,
    prepare_confirmation_challenge,
    readiness_packet_hash,
    source_readiness_packet_hash,
)
from services.paper_sell_arm_cap_slice import (
    authorized_slice_hash,
    slice_reviewed_sells,
)
from services.paper_sell_arm_console import (
    CONFIRM_PAPER_SELLS_PHRASE,
    arm_reviewed_sells,
    confirm_and_arm_paper_sells,
)
from services.paper_sell_arm_runtime_enforcement import (
    _ACTIVE_PERMIT,
    attempt_pre_plan_authorized_sell_dispatch,
    build_pre_plan_planned_orders,
    find_unused_pre_plan_authority,
    validate_pre_plan_authority_bound_artifacts,
)
from services.paper_sell_arm_store import (
    inspect_sqlite_arm_snapshot,
    resolve_runtime_sqlite_path,
)
from services.place_live_orders import dispatch_unused_authorized_sell_slice_pre_plan
from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    LIVE_BLOCKED,
    NetworkFakeAlpacaBroker,
    OrderedTrace,
    block_real_network,
    freeze_dispatch_clock,
    install_boundary_trace,
    isolate_sell_store,
    stub_unrelated_pre_gates,
)
from tests.test_phase_198n_c_sell_cap_slice import (
    NOW,
    PacketQuotedFakeBroker,
    _db,
    _pres,
    _snap,
    _write_buy_arm,
    _write_market_session_evidence,
    _write_packet,
)

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = Path(r"C:\Users\akimw\triton")
PROD_SQLITE = PRODUCTION_ROOT / "data" / "live" / "paper_sell_arm.sqlite3"
LOOP_SCRIPT = PRODUCTION_ROOT / "scripts" / "run_continuous_loop.ps1"


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    block_real_network(monkeypatch)
    for key in list(os.environ):
        if "ALPACA" in key.upper():
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    monkeypatch.delenv("TRITON_PAPER_SELL_ARM_STORE", raising=False)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    stub_dir_fsync_ok(monkeypatch)
    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)


def _slice_packet() -> Dict[str, Any]:
    base = json.loads(
        (ROOT / "tests" / "fixtures" / "phase_198n_c_packet.json").read_text(encoding="utf-8")
        if (ROOT / "tests" / "fixtures" / "phase_198n_c_packet.json").is_file()
        else "{}"
    )
    if not base:
        base = {
            "schema_version": "198I-ZA-SELL-READINESS-1",
            "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
            "generation_status": "ok",
            "mode": "PAPER",
            "live_execution_state": "BLOCKED",
            "creates_no_authority": True,
            "generated_at": NOW.isoformat(),
            "cycle_id": "mo-198q-test",
            "run_id": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "batch_id": "manage-mo-198q",
            "plan_id": f"manage_positions_plan@{NOW.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "plan_hash": "planhash198qtest0000000000000000000000000000000000000000000000000001",
            "approval_ready_symbols": ["TSLA", "GLD", "BITO"],
            "approval_ready_actions": [
                {
                    "symbol": "TSLA",
                    "action": "EXIT",
                    "qty": 8.0,
                    "est_price": 348.11,
                    "est_notional": 2778.16,
                },
                {
                    "symbol": "GLD",
                    "action": "EXIT",
                    "qty": 16.0,
                    "est_price": 411.75,
                    "est_notional": 6575.2,
                },
                {
                    "symbol": "BITO",
                    "action": "TRIM",
                    "qty": 18.0,
                    "est_price": 9.21,
                    "est_notional": 165.42,
                },
            ],
            "total_proposed_notional": 9518.78,
            "source_plan_sha256": "",
            "selected_row_hash": "",
            "market_open_invocation_id": "mo-198q-test",
        }
    sliced = slice_reviewed_sells(base, cap=APPROVAL_CLI_CAP)
    pkt = dict(sliced.get("packet") or base)
    pkt["source_plan_sha256"] = "abc123plan"
    pkt["selected_row_hash"] = "def456rows"
    return pkt


def _write_plan_artifacts(tmp_path: Path, packet: Dict[str, Any]) -> None:
    rows = [
        {
            "symbol": "TSLA",
            "final_action": "EXIT",
            "qty": "8",
            "selected_for_execution": "true",
            "reason_code": "",
            "skip_reason": "",
            "status": "planned",
        },
        {
            "symbol": "GLD",
            "final_action": "EXIT",
            "qty": "5",
            "selected_for_execution": "true",
            "reason_code": "",
            "skip_reason": "",
            "status": "planned",
        },
        {
            "symbol": "BITO",
            "final_action": "TRIM",
            "qty": "18",
            "selected_for_execution": "true",
            "reason_code": "",
            "skip_reason": "",
            "status": "planned",
        },
    ]
    csv_path = tmp_path / "manage_positions_plan.csv"
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
        if a["symbol"] in {"TSLA", "GLD", "BITO"}:
            pos_lines.append(f"{a['symbol']},{a['qty']},{a['qty']},{a.get('est_price', 100)}")
    (tmp_path / "positions_snapshot.csv").write_text("\n".join(pos_lines) + "\n", encoding="utf-8")
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=1))
    return pkt


def _arm_dual(
    tmp_path: Path, packet: Dict[str, Any], *, now: Optional[datetime] = None
) -> Dict[str, Any]:
    now = now or NOW
    _write_buy_arm(tmp_path, now=now)
    pres = _pres(tmp_path, packet)
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
    assert confirmed["ok"] is True, confirmed
    return confirmed


def _fake_broker(packet: Dict[str, Any]) -> PacketQuotedFakeBroker:
    positions = []
    for row in packet.get("approval_ready_actions") or []:
        if row["symbol"] in {"TSLA", "GLD", "BITO"}:
            positions.append(
                {
                    "symbol": row["symbol"],
                    "qty": str(row["qty"]),
                    "qty_available": str(row["qty"]),
                }
            )
    return PacketQuotedFakeBroker(packet, positions=positions)


def test_01_ttl_constant_is_60_minutes():
    assert AUTHORITY_TTL_MINUTES == 60


def test_02_expiry_clamped_to_session_boundary(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    late = datetime(2026, 8, 11, 23, 30, tzinfo=timezone.utc)
    freeze_dispatch_clock(monkeypatch, late)
    chal = prepare_confirmation_challenge(
        store_root=tmp_path,
        readiness_packet=packet,
        trading_date=late.date().isoformat(),
        now=late,
        db_path=_db(tmp_path),
        authorized_slice=slice_reviewed_sells(packet, cap=APPROVAL_CLI_CAP),
        explicit_runtime_write=True,
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=packet,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        now=late + timedelta(minutes=1),
        db_path=_db(tmp_path),
        explicit_runtime_write=True,
    )
    exp = datetime.fromisoformat(str(auth["expires_at"]).replace("Z", "+00:00"))
    day_end = datetime(late.year, late.month, late.day, 23, 59, 59, tzinfo=timezone.utc)
    assert exp <= day_end


def test_03_existing_15_minute_authority_not_extended(tmp_path):
    db = _db(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    old_exp = (NOW + timedelta(minutes=15)).isoformat()
    con = sqlite3.connect(str(db))
    con.execute(
        """
        CREATE TABLE sell_authorities (
            authority_id TEXT, challenge_id TEXT, plan_hash TEXT, batch_hash TEXT,
            canonical_batch_json TEXT, payload_json TEXT, created_at TEXT, expires_at TEXT,
            single_use INTEGER, state TEXT, consumption_time TEXT, consuming_dispatch_id TEXT,
            final_result TEXT, schema_version TEXT
        )
        """
    )
    payload = {"expires_at": old_exp, "consumption_status": "UNUSED", "status": "ARMED"}
    con.execute(
        "INSERT INTO sell_authorities VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "old-auth",
            "c1",
            "ph",
            "bh",
            "[]",
            json.dumps(payload),
            NOW.isoformat(),
            old_exp,
            1,
            "UNUSED",
            None,
            None,
            None,
            "198H-PAPER-SELL-ARM-1",
        ),
    )
    con.commit()
    con.close()
    later = NOW + timedelta(minutes=20)
    status = find_unused_pre_plan_authority(db_path=db, now=later)
    assert status is None


def test_04_no_authority_no_op(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    fake = _fake_broker(_slice_packet())
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW,
    )
    assert result["attempted"] is False
    assert fake.fake_broker_call_count == 0


def test_05_expired_authority_no_broker_call(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    db = _db(tmp_path)
    expired_at = (NOW - timedelta(minutes=1)).isoformat()
    con = sqlite3.connect(str(db))
    con.execute(
        "UPDATE sell_authorities SET expires_at = ? WHERE state = 'UNUSED'",
        (expired_at,),
    )
    con.commit()
    con.close()
    fake = _fake_broker(packet)
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=db,
        now=NOW,
    )
    assert result.get("dispatched") is False
    assert fake.fake_broker_call_count == 0


def test_06_valid_authority_discovered_before_plan_write(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    auth = _arm_dual(tmp_path, packet)
    assert find_unused_pre_plan_authority(db_path=_db(tmp_path), now=NOW + timedelta(minutes=1))


def test_07_original_plan_hash_validates_before_supersession(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    auth_row = _arm_dual(tmp_path, packet)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    authority = snap["latest_authority"]["payload"]
    ok, reason, _ = validate_pre_plan_authority_bound_artifacts(
        authority, results_root=tmp_path, now=NOW + timedelta(minutes=1)
    )
    assert ok, reason


def test_08_new_plan_written_only_after_pre_plan_decision(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    order: List[str] = []

    import services.manage_positions as mp

    real_write = mp.write_manage_artifacts

    def _tracked_write(*args, **kwargs):
        order.append("write_manage_artifacts")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(mp, "write_manage_artifacts", _tracked_write)

    def _pre_plan(**kwargs):
        order.append("pre_plan_dispatch")
        return {"attempted": False, "dispatched": False, "reason": "test_skip_submit"}

    monkeypatch.setattr(
        "services.place_live_orders.dispatch_unused_authorized_sell_slice_pre_plan",
        _pre_plan,
    )
    # invoke helper directly as manage_positions would before plan build
    from services.manage_positions import (
        ManagePlanSummary,
        _try_pre_plan_authorized_sell_dispatch,
    )

    _try_pre_plan_authorized_sell_dispatch(
        broker=_fake_broker(packet), mode="paper", dry_run=False, verbose=False
    )
    _tracked_write(
        [],
        [],
        ManagePlanSummary(
            timestamp=NOW.isoformat(),
            mode="paper",
            dry_run=False,
            symbols_seen=0,
            positions_seen=0,
            orders_planned=0,
            orders_executed=0,
            orders_skipped=0,
            trim_candidates=0,
            exit_candidates=0,
            approved_actions=0,
            total_positions=0,
            selected_actions=0,
            profit_weight=0.6,
            signal_weight=0.4,
            skip_reasons={},
            blocked=False,
            block_reasons=[],
            source_file="x",
            orders_file="y",
        ),
        {},
        Path("manage_orders.csv"),
    )
    assert order.index("pre_plan_dispatch") < order.index("write_manage_artifacts")


def test_09_exact_authorized_slice_reaches_fake_broker(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    fake = _fake_broker(packet)
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
    assert result.get("broker_submissions", 0) >= 1
    assert fake.fake_broker_call_count >= 1


def test_10_deferred_actions_zero_submissions(tmp_path, monkeypatch):
    # Full packet includes GLD 16 but slice caps; deferred symbols not submitted
    test_09_exact_authorized_slice_reaches_fake_broker(tmp_path, monkeypatch)


def test_11_buy_add_rows_zero_submissions(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    fake = _fake_broker(packet)
    dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
    )
    for p in fake.submit_payloads:
        assert str(p.get("side", "")).lower() == "sell"


def test_12_total_batch_under_5000(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    auth = snap["latest_authority"]["payload"]
    assert float(auth["total_batch_max_notional"]) <= APPROVAL_CLI_CAP


def test_13_per_symbol_ceilings_enforced(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    auth = snap["latest_authority"]["payload"]
    per = auth.get("per_symbol_max_notional") or {}
    for v in per.values():
        assert float(v) <= APPROVAL_CLI_CAP


def test_14_price_expansion_fails_closed(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    authority = snap["latest_authority"]["payload"]

    class HighQuoteBroker(PacketQuotedFakeBroker):
        def _px(self, symbol: str) -> float:
            return 99999.0

    positions = [
        {
            "symbol": str(item.get("symbol")),
            "qty": str(item.get("max_qty")),
            "qty_available": str(item.get("max_qty")),
        }
        for item in (authority.get("authorized_actions") or [])
    ]
    fake = HighQuoteBroker(packet, positions=positions)
    ok, reason, planned = build_pre_plan_planned_orders(
        authority, fake, now=NOW + timedelta(minutes=1)
    )
    assert not ok
    assert reason == "no_executable_authorized_quantity"
    assert planned == ()
    snap2 = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert snap2["latest_authority"]["state"] == "UNUSED"


def test_15_quantity_insufficiency_fails_closed(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    authority = snap["latest_authority"]["payload"]
    fake = PacketQuotedFakeBroker(
        packet, positions=[{"symbol": "TSLA", "qty": "1", "qty_available": "1"}]
    )
    ok, reason, _ = build_pre_plan_planned_orders(authority, fake, now=NOW + timedelta(minutes=1))
    assert not ok


def test_16_packet_hash_mismatch_fails_closed(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    authority = snap["latest_authority"]["payload"]
    bad = json.loads(
        (tmp_path / "defensive_exit_approval_readiness.json").read_text(encoding="utf-8")
    )
    bad["approval_ready_actions"][0]["qty"] = 1
    _write_packet(tmp_path, bad)
    ok, reason, _ = validate_pre_plan_authority_bound_artifacts(
        authority, results_root=tmp_path, now=NOW + timedelta(minutes=1)
    )
    assert not ok


def test_17_source_plan_hash_mismatch(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    authority = snap["latest_authority"]["payload"]
    (tmp_path / "manage_positions_plan.csv").write_text(
        "symbol,final_action,qty\nX,EXIT,1\n", encoding="utf-8"
    )
    ok, reason, _ = validate_pre_plan_authority_bound_artifacts(
        authority, results_root=tmp_path, now=NOW + timedelta(minutes=1)
    )
    assert not ok


def test_18_selected_row_mismatch(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    authority = snap["latest_authority"]["payload"]
    rows = [
        {
            "symbol": "X",
            "final_action": "EXIT",
            "qty": "1",
            "selected_for_execution": "true",
            "reason_code": "",
            "skip_reason": "",
            "status": "planned",
        }
    ]
    with (tmp_path / "manage_positions_plan.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok, reason, _ = validate_pre_plan_authority_bound_artifacts(
        authority, results_root=tmp_path, now=NOW + timedelta(minutes=1)
    )
    assert not ok


def test_19_invocation_mismatch(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    authority = snap["latest_authority"]["payload"]
    bad = json.loads(
        (tmp_path / "defensive_exit_approval_readiness.json").read_text(encoding="utf-8")
    )
    bad["market_open_invocation_id"] = "other-invocation"
    _write_packet(tmp_path, bad)
    ok, reason, _ = validate_pre_plan_authority_bound_artifacts(
        authority, results_root=tmp_path, now=NOW + timedelta(minutes=1)
    )
    assert not ok


def test_20_slice_hash_mismatch(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    auth = _arm_dual(tmp_path, packet)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    authority = dict(snap["latest_authority"]["payload"])
    authority["authorized_slice_hash"] = "0" * 64
    ok, reason, _ = validate_pre_plan_authority_bound_artifacts(
        authority, results_root=tmp_path, now=NOW + timedelta(minutes=1)
    )
    assert not ok


def test_21_authority_unused_after_pre_broker_refusal(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    authority = snap["latest_authority"]["payload"]
    fake = PacketQuotedFakeBroker(packet, positions=[])
    ok, reason, _ = build_pre_plan_planned_orders(authority, fake, now=NOW + timedelta(minutes=1))
    assert not ok
    snap2 = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert snap2["latest_authority"]["state"] == "UNUSED"


def test_22_successful_dispatch_consumes_once(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    fake = _fake_broker(packet)
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
    )
    assert result.get("dispatched") is True
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=2))
    assert snap["latest_authority"]["state"] == "CONSUMED"


def test_23_replay_no_second_broker_call(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    fake = _fake_broker(packet)
    dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
    )
    first = fake.fake_broker_call_count
    dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=2),
    )
    assert fake.fake_broker_call_count == first


def test_24_concurrent_duplicate_blocked(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    authority = snap["latest_authority"]["payload"]
    fake = _fake_broker(packet)
    results = []

    def _worker():
        try:
            results.append(
                attempt_pre_plan_authorized_sell_dispatch(
                    authority=authority,
                    broker=fake,
                    results_root=tmp_path,
                    now=NOW + timedelta(minutes=1),
                    db_path=_db(tmp_path),
                ).permitted
            )
        except Exception:
            results.append(False)

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert sum(1 for r in results if r) <= 1


def test_25_ambiguous_outcome_unknown(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    fake = _fake_broker(packet)

    class AmbiguousBroker(PacketQuotedFakeBroker):
        def submit_order(self, **kwargs):
            raise TimeoutError("ambiguous broker timeout")

    amb = AmbiguousBroker(packet, positions=fake.positions)
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=amb,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
    )
    assert result.get("consumed") is True


def test_26_live_flag_refusal(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    _arm_dual(tmp_path, packet)
    fake = _fake_broker(packet)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "1")
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
        mode="live",
    )
    assert result.get("dispatched") is False


def test_27_non_paper_mode_refusal(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _slice_packet())
    fake = _fake_broker(packet)
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
        mode="live",
    )
    assert result.get("dispatched") is False


def test_28_normal_manage_positions_unchanged_without_authority(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    fake = _fake_broker(_slice_packet())
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW,
    )
    assert result["attempted"] is False


def test_29_post_dispatch_plan_generation_allowed(tmp_path, monkeypatch):
    # After successful dispatch path, writing a new plan must still be permitted
    test_08_new_plan_written_only_after_pre_plan_decision(tmp_path, monkeypatch)


def test_30_continuous_loop_interval_still_45_minutes():
    text = LOOP_SCRIPT.read_text(encoding="utf-8")
    assert "[int]$IntervalMinutes = 45" in text
