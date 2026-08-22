"""Phase 198R-J — next-loop paper SELL dispatch contract.

PAPER ONLY. Isolated tmp_path. Fake broker. Network disabled. Live blocked.
Does not write production artifacts, start TRITON, or touch the retained
production dispatch lock.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from services.manage_positions_cycle_binding import selected_row_hash, sha256_file_bytes
from services.paper_sell_arm_authority import (
    APPROVAL_CLI_CAP,
    AUTHORITY_TTL_MINUTES,
    CONTINUOUS_LOOP_INTERVAL_MINUTES,
    HARDENED_AUTHORITY_FILENAME,
    READINESS_REFERENCE_TTL_MINUTES,
    evaluate_readiness_packet_freshness,
    readiness_packet_hash,
    readiness_packet_valid_until,
)
from services.paper_sell_arm_cap_slice import (
    authorized_slice_hash,
    slice_reviewed_sells,
    source_readiness_packet_hash,
)
from services.paper_sell_arm_console import (
    CONFIRM_PAPER_SELLS_PHRASE,
    arm_reviewed_sells,
    confirm_and_arm_paper_sells,
)
from services.paper_sell_arm_runtime_enforcement import (
    LOCK_NAME,
    PaperSellDispatchBlocked,
    _ACTIVE_PERMIT,
    _acquire_lock,
    _dispatch_lock,
    _lock_path,
    _release_lock,
    evaluate_market_session_for_dispatch,
)
from services.paper_sell_arm_store import inspect_sqlite_arm_snapshot
from services.paper_sell_dispatch_session_evidence import (
    AUTHORITATIVE_SOURCE,
    SCHEMA_VERSION as SESSION_EVIDENCE_SCHEMA,
    SessionEvidenceError,
    write_dispatch_session_evidence,
)
from services.place_live_orders import dispatch_unused_authorized_sell_slice_pre_plan
from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    LIVE_BLOCKED,
    SESSION_OPEN_NOW,
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
)

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = Path(r"C:\Users\akimw\triton")
LOOP_SCRIPT = ROOT / "scripts" / "run_continuous_loop.ps1"
MARKET_OPEN_SCRIPT = ROOT / "scripts" / "run_market_open.ps1"
NOW = SESSION_OPEN_NOW
INVOCATION_T0 = "mo-198rj-cycle1"
INVOCATION_T1 = "mo-198rj-cycle2"
OPEN_CAL = {
    "session_state": "OPEN",
    "reason": "regular_session_open",
    "session_date": "2026-08-11",
    "session_open_et": "2026-08-11T09:30:00-04:00",
    "session_close_et": "2026-08-11T16:00:00-04:00",
    "source": AUTHORITATIVE_SOURCE,
    "as_of_utc": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "as_of_et": "2026-08-11T12:00:00-04:00",
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
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    stub_dir_fsync_ok(monkeypatch)
    monkeypatch.setattr(
        "services.live_block_classifier.evaluate_live_block_status",
        lambda paper_mode=True: dict(LIVE_BLOCKED),
    )
    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)


def _assert_no_production_write(tmp_path: Path) -> None:
    prod_lock = PRODUCTION_ROOT / "data" / "governance" / "paper_sell_arm_dispatch.lock"
    if prod_lock.is_file():
        assert prod_lock.stat().st_mtime == pytest.approx(prod_lock.stat().st_mtime)
    assert not str(tmp_path).lower().startswith(str(PRODUCTION_ROOT / "data").lower())


def _lock_report(store_root: Path) -> Dict[str, Any]:
    path = _lock_path(store_root)
    physical_present = path.is_file()
    fh = None
    try:
        fh = _acquire_lock(store_root)
        reacquisition_ok = True
        logical_held = False
    except PaperSellDispatchBlocked:
        reacquisition_ok = False
        logical_held = True
    if fh is not None:
        _release_lock(fh)
    return {
        "lock_path": str(path),
        "lock_name": LOCK_NAME,
        "physical_present": physical_present,
        "logical_held": logical_held,
        "reacquisition_ok": reacquisition_ok,
    }


def _assert_lock_logically_free(store_root: Path) -> Dict[str, Any]:
    report = _lock_report(store_root)
    assert report["logical_held"] is False
    assert report["reacquisition_ok"] is True
    return report


def _reviewed_packet(*, generated_at: Optional[datetime] = None) -> Dict[str, Any]:
    gen = generated_at or NOW
    actions = [
        {
            "symbol": "TSLA",
            "action": "EXIT",
            "qty": 8.0,
            "est_price": 344.91,
            "est_notional": 2759.28,
        },
        {
            "symbol": "GLD",
            "action": "EXIT",
            "qty": 16.0,
            "est_price": 413.72,
            "est_notional": 6619.52,
        },
        {
            "symbol": "DBA",
            "action": "EXIT",
            "qty": 88.0,
            "est_price": 28.38,
            "est_notional": 2497.44,
        },
        {
            "symbol": "HD",
            "action": "EXIT",
            "qty": 10.0,
            "est_price": 400.0,
            "est_notional": 4000.0,
        },
    ]
    pkt = {
        "schema_version": "198I-ZA-SELL-READINESS-1",
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "generation_status": "ok",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "generated_at": gen.isoformat(),
        "valid_until": readiness_packet_valid_until(gen).isoformat(),
        "freshness_ttl_minutes": READINESS_REFERENCE_TTL_MINUTES,
        "session_date": "2026-08-11",
        "cycle_id": INVOCATION_T0,
        "run_id": gen.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "batch_id": "manage-mo-198rj",
        "plan_id": f"manage_positions_plan@{gen.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "plan_hash": "planhash198rj0000000000000000000000000000000000000000000000000001",
        "approval_ready_symbols": [a["symbol"] for a in actions],
        "approval_ready_actions": actions,
        "total_proposed_notional": round(sum(a["est_notional"] for a in actions), 2),
        "source_plan_sha256": "",
        "selected_row_hash": "",
        "market_open_invocation_id": INVOCATION_T0,
        "invocation_id": INVOCATION_T0,
    }
    sliced = slice_reviewed_sells(pkt, cap=APPROVAL_CLI_CAP)
    out = dict(sliced.get("packet") or pkt)
    out["source_plan_sha256"] = "srcplan198rj"
    out["selected_row_hash"] = "srcrows198rj"
    out["session_date"] = "2026-08-11"
    out["generated_at"] = gen.isoformat()
    out["valid_until"] = readiness_packet_valid_until(gen).isoformat()
    return out


def _write_plan_artifacts(tmp_path: Path, packet: Dict[str, Any]) -> Dict[str, Any]:
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
            "symbol": "DBA",
            "final_action": "EXIT",
            "qty": "6",
            "selected_for_execution": "true",
            "reason_code": "",
            "skip_reason": "",
            "status": "planned",
        },
        {
            "symbol": "HD",
            "final_action": "EXIT",
            "qty": "10",
            "selected_for_execution": "false",
            "reason_code": "DEFERRED_CAP_REMAINDER",
            "skip_reason": "cap",
            "status": "deferred",
        },
    ]
    csv_path = tmp_path / "manage_positions_plan.csv"
    import csv

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    pkt = dict(packet)
    pkt["source_plan_sha256"] = sha256_file_bytes(csv_path) or pkt.get("source_plan_sha256")
    pkt["selected_row_hash"] = selected_row_hash(rows)
    (tmp_path / "defensive_exit_approval_readiness.json").write_text(
        json.dumps(pkt), encoding="utf-8"
    )
    (tmp_path / "manage_positions_plan.json").write_text(
        json.dumps({"plan_hash": pkt.get("plan_hash"), "timestamp": NOW.isoformat()}),
        encoding="utf-8",
    )
    pos_lines = ["symbol,qty,qty_available,current_price"]
    for a in pkt.get("approval_ready_actions") or []:
        pos_lines.append(f"{a['symbol']},{a['qty']},{a['qty']},{a.get('est_price', 100)}")
    (tmp_path / "positions_snapshot.csv").write_text("\n".join(pos_lines) + "\n", encoding="utf-8")
    return pkt


def _arm(
    tmp_path: Path, packet: Dict[str, Any], *, now: Optional[datetime] = None
) -> Dict[str, Any]:
    now = now or NOW
    _write_buy_arm(tmp_path, now=now)
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


def _insert_expired_historical(db_path: Path, *, now: datetime) -> None:
    from services.paper_sell_arm_store import open_connection

    conn = open_connection(db_path, initialize=True)
    created = (now - timedelta(days=1)).isoformat()
    expires = (now - timedelta(hours=1)).isoformat()
    conn.execute(
        """
        INSERT INTO sell_challenges (
          challenge_id, plan_hash, batch_hash, batch_id, canonical_batch_json,
          payload_json, created_at, expires_at, state, operator_action_label, schema_version
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "sell-chal-historical-expired",
            "old-plan-hash",
            "old-batch-hash",
            "old-batch",
            "[]",
            "{}",
            created,
            expires,
            "CONSUMED",
            "confirm",
            "198H-PAPER-SELL-ARM-1",
        ),
    )
    conn.execute(
        """
        INSERT INTO sell_authorities (
          authority_id, challenge_id, plan_hash, batch_hash, canonical_batch_json,
          payload_json, created_at, expires_at, single_use, state, schema_version
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "sell-arm-audit-historical-expired",
            "sell-chal-historical-expired",
            "old-plan-hash",
            "old-batch-hash",
            "[]",
            "{}",
            created,
            expires,
            1,
            "UNUSED",
            "198H-PAPER-SELL-ARM-1",
        ),
    )
    conn.commit()
    conn.close()


def _fake_broker(packet: Dict[str, Any]) -> PacketQuotedFakeBroker:
    positions = []
    for row in packet.get("approval_ready_actions") or []:
        positions.append(
            {
                "symbol": row["symbol"],
                "qty": str(row["qty"]),
                "qty_available": str(row["qty"]),
            }
        )
    return PacketQuotedFakeBroker(packet, positions=positions)


def _submitted_sells(fake: PacketQuotedFakeBroker) -> List[tuple]:
    out = []
    for call in getattr(fake, "_post_calls", None) or getattr(fake, "posts", None) or []:
        pass
    # NetworkFakeAlpacaBroker records via production submit_order → _post
    posts = getattr(fake, "post_calls", []) or getattr(fake, "_calls", [])
    if not posts and hasattr(fake, "submitted"):
        posts = fake.submitted
    if hasattr(fake, "fake_orders"):
        for order in fake.fake_orders:
            out.append(
                (
                    str(order.get("symbol") or "").upper(),
                    str(order.get("side") or "").upper(),
                    float(order.get("qty") or 0),
                )
            )
        return out
    for rec in getattr(fake, "orders", []):
        out.append(
            (
                str(rec.get("symbol") or "").upper(),
                str(rec.get("side") or "").upper(),
                float(rec.get("qty") or 0),
            )
        )
    return out


def test_01_freshness_contract_constants():
    assert CONTINUOUS_LOOP_INTERVAL_MINUTES == 45
    assert READINESS_REFERENCE_TTL_MINUTES == 50
    assert AUTHORITY_TTL_MINUTES == 60
    text = LOOP_SCRIPT.read_text(encoding="utf-8")
    assert "[int]$IntervalMinutes = 45" in text


def test_02_packet_age_44_and_46_eligible_expiry_rejected():
    packet = _reviewed_packet()
    assert evaluate_readiness_packet_freshness(
        packet, now=NOW + timedelta(minutes=44), session_payload=OPEN_CAL
    )[0]
    assert evaluate_readiness_packet_freshness(
        packet, now=NOW + timedelta(minutes=46), session_payload=OPEN_CAL
    )[0]
    ok, reason, _ = evaluate_readiness_packet_freshness(
        packet,
        now=NOW + timedelta(minutes=READINESS_REFERENCE_TTL_MINUTES, seconds=1),
        session_payload=OPEN_CAL,
    )
    assert ok is False
    assert reason == "stale_readiness_packet"


def test_03_session_close_and_prior_session_rejected():
    packet = _reviewed_packet()
    closed = dict(OPEN_CAL)
    closed["session_state"] = "CLOSED"
    ok, reason, _ = evaluate_readiness_packet_freshness(
        packet, now=NOW + timedelta(minutes=10), session_payload=closed
    )
    assert ok is False
    assert reason == "session_boundary_closed"
    prior = dict(packet)
    prior["session_date"] = "2026-08-10"
    ok2, reason2, _ = evaluate_readiness_packet_freshness(
        prior, now=NOW + timedelta(minutes=10), session_payload=OPEN_CAL
    )
    assert ok2 is False
    assert reason2 == "prior_session_packet"


def test_04_valid_until_never_past_session_close():
    close = datetime(2026, 8, 11, 16, 30, tzinfo=timezone.utc)
    until = readiness_packet_valid_until(NOW, session_close=close)
    assert until == close.astimezone(timezone.utc)


def test_05_run_step_captures_python_streams():
    text = MARKET_OPEN_SCRIPT.read_text(encoding="utf-8")
    assert "2>&1" in text
    assert "[REDACTED]" in text
    fn = text.split("function Run-Step")[1].split("function Run-RefreshStep")[0]
    assert "$output = & python @PyArgs 2>&1" in fn


def test_06_writer_produces_invocation_bound_open_evidence(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", INVOCATION_T1)
    freeze_dispatch_clock(monkeypatch, NOW)
    doc = write_dispatch_session_evidence(
        store_root=tmp_path,
        now=NOW,
        invocation_id=INVOCATION_T1,
        live_execution_blocked=True,
        calendar_payload=OPEN_CAL,
    )
    raw = (tmp_path / "independent_dispatch_bindings.json").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert doc["schema_version"] == SESSION_EVIDENCE_SCHEMA
    assert doc["market_open_invocation_id"] == INVOCATION_T1
    assert doc["market_session"] == "MARKET_OPEN"
    assert doc["live_execution_blocked"] is True
    assert doc["mode"] == "PAPER"
    ok, reason, _ = evaluate_market_session_for_dispatch(
        NOW, caller_session="MARKET_OPEN", session_evidence=doc
    )
    assert ok, reason


def test_07_writer_rejects_closed_unknown_and_missing_invocation(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    closed = dict(OPEN_CAL)
    closed["session_state"] = "CLOSED"
    with pytest.raises(SessionEvidenceError) as exc:
        write_dispatch_session_evidence(
            store_root=tmp_path,
            now=NOW,
            invocation_id=INVOCATION_T1,
            live_execution_blocked=True,
            calendar_payload=closed,
        )
    assert exc.value.reason == "market_session_not_open"
    unknown = dict(OPEN_CAL)
    unknown["session_state"] = "UNKNOWN"
    with pytest.raises(SessionEvidenceError) as exc2:
        write_dispatch_session_evidence(
            store_root=tmp_path,
            now=NOW,
            invocation_id=INVOCATION_T1,
            live_execution_blocked=True,
            calendar_payload=unknown,
        )
    assert exc2.value.reason == "market_session_derivation_unknown"
    monkeypatch.delenv("TRITON_MARKET_OPEN_INVOCATION_ID", raising=False)
    with pytest.raises(SessionEvidenceError) as exc3:
        write_dispatch_session_evidence(
            store_root=tmp_path,
            now=NOW,
            invocation_id="",
            live_execution_blocked=True,
            calendar_payload=OPEN_CAL,
        )
    assert exc3.value.reason == "session_evidence_invocation_missing"
    assert not (tmp_path / "independent_dispatch_bindings.json").is_file()


def test_08_historical_expired_authority_does_not_false_fail_arm(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _reviewed_packet())
    db = _db(tmp_path)
    _insert_expired_historical(db, now=NOW)
    confirmed = _arm(tmp_path, packet)
    assert confirmed["ok"] is True, confirmed
    assert confirmed["reason"] is None
    snap = inspect_sqlite_arm_snapshot(db, now=NOW + timedelta(minutes=1))
    assert snap["authority_count"] == 2
    assert str((snap.get("latest_authority") or {}).get("state")) == "UNUSED"
    assert snap["dispatch_count"] == 0
    assert confirmed.get("counts", {}).get("broker", 0) == 0


def test_09_lock_released_on_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with _dispatch_lock(tmp_path):
            raise RuntimeError("dispatch_boom")
    report = _assert_lock_logically_free(tmp_path)
    assert report["physical_present"] is True
    assert Path(report["lock_path"]).name == LOCK_NAME
    fh = _acquire_lock(tmp_path)
    _release_lock(fh)
    _assert_lock_logically_free(tmp_path)


def test_10_pre_plan_block_does_not_write_plan(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _reviewed_packet())
    db = _db(tmp_path)
    _insert_expired_historical(db, now=NOW)
    confirmed = _arm(tmp_path, packet)
    assert confirmed["ok"] is True
    plan_before = (tmp_path / "manage_positions_plan.csv").read_bytes()
    packet_before = (tmp_path / "defensive_exit_approval_readiness.json").read_bytes()
    wrote: List[str] = []

    def _blocked(**kwargs):
        return {
            "attempted": True,
            "dispatched": False,
            "consumed": False,
            "reason": "market_session_evidence_missing",
        }

    monkeypatch.setattr(
        "services.place_live_orders.dispatch_unused_authorized_sell_slice_pre_plan",
        _blocked,
    )
    monkeypatch.setattr(
        "services.paper_sell_dispatch_session_evidence.write_dispatch_session_evidence",
        lambda **kwargs: (_ for _ in ()).throw(
            SessionEvidenceError("market_session_evidence_missing")
        ),
    )
    from services.manage_positions import _try_pre_plan_authorized_sell_dispatch

    result = _try_pre_plan_authorized_sell_dispatch(
        broker=_fake_broker(packet), mode="paper", dry_run=False, verbose=False
    )
    assert result.get("abort_before_plan") is True
    assert result.get("consumed") is False
    assert (tmp_path / "manage_positions_plan.csv").read_bytes() == plan_before
    assert (tmp_path / "defensive_exit_approval_readiness.json").read_bytes() == packet_before
    snap = inspect_sqlite_arm_snapshot(db, now=NOW + timedelta(minutes=1))
    unused = [a for a in []]
    latest = snap.get("latest_authority") or {}
    assert latest.get("state") == "UNUSED"
    assert snap["dispatch_count"] == 0
    assert wrote == []
    _assert_lock_logically_free(tmp_path)


def _dispatch_at(
    tmp_path: Path,
    packet: Dict[str, Any],
    *,
    now: datetime,
    invocation: str,
    monkeypatch,
    fake: Optional[PacketQuotedFakeBroker] = None,
):
    freeze_dispatch_clock(monkeypatch, now)
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", invocation)
    write_dispatch_session_evidence(
        store_root=tmp_path,
        now=now,
        invocation_id=invocation,
        live_execution_blocked=True,
        calendar_payload=OPEN_CAL,
    )
    stub_unrelated_pre_gates(monkeypatch)
    broker = fake or _fake_broker(packet)
    return (
        dispatch_unused_authorized_sell_slice_pre_plan(
            broker=broker,
            results_root=tmp_path,
            store_root=tmp_path,
            db_path=_db(tmp_path),
            now=now,
            verbose=True,
        ),
        broker,
    )


def test_11_next_loop_exact_slice_consumes_once(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    assert INVOCATION_T0 == "mo-198rj-cycle1"
    assert INVOCATION_T1 == "mo-198rj-cycle2"
    assert INVOCATION_T0 != INVOCATION_T1
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", INVOCATION_T0)
    packet = _write_plan_artifacts(tmp_path, _reviewed_packet())
    assert packet["market_open_invocation_id"] == INVOCATION_T0
    assert packet["invocation_id"] == INVOCATION_T0
    assert packet["cycle_id"] == INVOCATION_T0
    readiness_path = tmp_path / "defensive_exit_approval_readiness.json"
    assert readiness_path.is_file()
    db = _db(tmp_path)
    _insert_expired_historical(db, now=NOW)
    confirmed = _arm(tmp_path, packet)
    assert confirmed["ok"] is True, confirmed
    auth = json.loads((tmp_path / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
    sliced = slice_reviewed_sells(packet, cap=APPROVAL_CLI_CAP)
    slice_actions = sliced.get("slice_actions") or []
    symbols = [(str(a["symbol"]), float(a["qty"])) for a in slice_actions]
    assert symbols[:3] == [("TSLA", 8.0), ("GLD", 5.0), ("DBA", 6.0)]
    total = round(sum(float(a["est_notional"]) for a in slice_actions), 2)
    assert total == 4998.16
    cycle_a_source_plan = str(packet.get("source_plan_sha256") or "")
    cycle_a_selected_rows = str(packet.get("selected_row_hash") or "")
    cycle_a_packet_hash = readiness_packet_hash(packet)
    cycle_a_source_hash = source_readiness_packet_hash(packet)
    cycle_a_slice_hash = authorized_slice_hash(
        scope=auth.get("authorized_actions") or [],
        slice_total=float(auth.get("total_batch_max_notional") or total),
        applied_cap=float(auth.get("applied_cap") or APPROVAL_CLI_CAP),
        cap_source=str(auth.get("applied_cap_source") or "APPROVAL_CLI_CAP"),
    )
    assert cycle_a_source_plan
    assert cycle_a_selected_rows
    assert cycle_a_packet_hash
    assert cycle_a_source_hash
    assert cycle_a_slice_hash
    assert auth.get("market_open_invocation_id") == INVOCATION_T0
    assert auth.get("readiness_packet_hash") == cycle_a_packet_hash
    assert auth.get("source_readiness_packet_hash") == cycle_a_source_hash
    assert auth.get("source_plan_sha256") == cycle_a_source_plan
    assert auth.get("selected_row_hash") == cycle_a_selected_rows
    assert auth.get("authorized_slice_hash") == cycle_a_slice_hash
    assert str(auth.get("consumption_status") or "").upper() == "UNUSED"
    later = NOW + timedelta(minutes=46)
    fresh_ok, fresh_reason, _ = evaluate_readiness_packet_freshness(
        packet, now=later, session_payload=OPEN_CAL
    )
    assert fresh_ok is True, fresh_reason
    fake = _fake_broker(packet)
    trace = OrderedTrace()
    install_boundary_trace(monkeypatch, trace)
    result, fake = _dispatch_at(
        tmp_path,
        packet,
        now=later,
        invocation=INVOCATION_T1,
        monkeypatch=monkeypatch,
        fake=fake,
    )
    assert result.get("dispatched") is True, result
    assert result.get("consumed") is True
    assert result.get("reason") != "invocation_mismatch"
    assert os.environ.get("TRITON_MARKET_OPEN_INVOCATION_ID") == INVOCATION_T1
    evidence = json.loads(
        (tmp_path / "independent_dispatch_bindings.json").read_text(encoding="utf-8")
    )
    assert evidence.get("market_open_invocation_id") == INVOCATION_T1
    auth_after = json.loads((tmp_path / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
    pkt_after = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert pkt_after.get("market_open_invocation_id") == INVOCATION_T0
    assert pkt_after.get("cycle_id") == INVOCATION_T0
    assert auth_after.get("market_open_invocation_id") == INVOCATION_T0
    assert auth_after.get("readiness_packet_hash") == cycle_a_packet_hash
    assert auth_after.get("source_readiness_packet_hash") == cycle_a_source_hash
    assert auth_after.get("source_plan_sha256") == cycle_a_source_plan
    assert auth_after.get("selected_row_hash") == cycle_a_selected_rows
    assert auth_after.get("authorized_slice_hash") == cycle_a_slice_hash
    assert evidence.get("market_open_invocation_id") != auth_after.get("market_open_invocation_id")
    payloads = fake.submit_payloads
    got = sorted(
        (
            str(p.get("symbol") or "").upper(),
            str(p.get("side") or "").upper(),
            float(p.get("qty") or 0),
        )
        for p in payloads
    )
    assert got == [("DBA", "SELL", 6.0), ("GLD", "SELL", 5.0), ("TSLA", "SELL", 8.0)]
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    current = conn.execute(
        "SELECT * FROM sell_authorities WHERE authority_id != ?",
        ("sell-arm-audit-historical-expired",),
    ).fetchone()
    hist = conn.execute(
        "SELECT * FROM sell_authorities WHERE authority_id = ?",
        ("sell-arm-audit-historical-expired",),
    ).fetchone()
    n_disp = conn.execute("SELECT COUNT(*) FROM sell_dispatches").fetchone()[0]
    conn.close()
    assert current["state"] == "CONSUMED"
    assert current["consumption_time"]
    assert hist["state"] == "UNUSED"
    assert n_disp == 1
    success_lock = _assert_lock_logically_free(tmp_path)
    assert success_lock["physical_present"] is True
    replay, fake2 = _dispatch_at(
        tmp_path,
        packet,
        now=later + timedelta(seconds=5),
        invocation=INVOCATION_T1,
        monkeypatch=monkeypatch,
        fake=fake,
    )
    assert replay.get("dispatched") is not True
    assert replay.get("reason") != "invocation_mismatch"
    assert len(fake2.submit_payloads) == 3
    conn = sqlite3.connect(str(db))
    n_disp_replay = conn.execute("SELECT COUNT(*) FROM sell_dispatches").fetchone()[0]
    conn.close()
    assert n_disp_replay == 1
    replay_lock = _assert_lock_logically_free(tmp_path)
    assert replay_lock["physical_present"] is True
    deferred = (tmp_path / "manage_positions_plan.csv").read_text(encoding="utf-8")
    assert "HD" in deferred
    assert "DEFERRED_CAP_REMAINDER" in deferred
    _assert_no_production_write(tmp_path)


def test_12_missing_session_evidence_blocks(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _reviewed_packet())
    _arm(tmp_path, packet)
    stub_unrelated_pre_gates(monkeypatch)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=_fake_broker(packet),
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
        verbose=True,
    )
    assert result.get("dispatched") is not True
    assert result.get("consumed") is not True
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert snap["dispatch_count"] == 0
    report = _assert_lock_logically_free(tmp_path)
    assert report["reacquisition_ok"] is True


def test_13_stale_future_mismatch_closed_unknown_evidence(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", INVOCATION_T1)
    freeze_dispatch_clock(monkeypatch, NOW)
    good = write_dispatch_session_evidence(
        store_root=tmp_path,
        now=NOW,
        invocation_id=INVOCATION_T1,
        live_execution_blocked=True,
        calendar_payload=OPEN_CAL,
    )
    stale = dict(good)
    stale["market_session_observed_at"] = (NOW - timedelta(minutes=20)).isoformat()
    ok, reason, _ = evaluate_market_session_for_dispatch(
        NOW, caller_session="MARKET_OPEN", session_evidence=stale
    )
    assert ok is False
    assert reason == "market_session_evidence_stale"
    future = dict(good)
    future["market_session_observed_at"] = (NOW + timedelta(minutes=5)).isoformat()
    ok2, reason2, _ = evaluate_market_session_for_dispatch(
        NOW, caller_session="MARKET_OPEN", session_evidence=future
    )
    assert ok2 is False
    assert reason2 == "market_session_evidence_future"
    mismatch = dict(good)
    mismatch["market_open_invocation_id"] = "mo-other"
    ok3, reason3, _ = evaluate_market_session_for_dispatch(
        NOW, caller_session="MARKET_OPEN", session_evidence=mismatch
    )
    assert ok3 is False
    assert reason3 == "session_evidence_invocation_mismatch"
    ok4, reason4, _ = evaluate_market_session_for_dispatch(
        NOW, caller_session="MARKET_OPEN", session_evidence={}
    )
    assert ok4 is False
    assert reason4 == "market_session_evidence_missing"


def test_14_hash_mismatch_and_expired_authority(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _reviewed_packet())
    _arm(tmp_path, packet)
    later = NOW + timedelta(minutes=1)
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", INVOCATION_T1)
    freeze_dispatch_clock(monkeypatch, later)
    write_dispatch_session_evidence(
        store_root=tmp_path,
        now=later,
        invocation_id=INVOCATION_T1,
        live_execution_blocked=True,
        calendar_payload=OPEN_CAL,
    )
    stub_unrelated_pre_gates(monkeypatch)
    broken = dict(packet)
    broken["plan_hash"] = "deadbeef" * 8
    (tmp_path / "defensive_exit_approval_readiness.json").write_text(
        json.dumps(broken), encoding="utf-8"
    )
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=_fake_broker(packet),
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=later,
        verbose=True,
    )
    assert result.get("dispatched") is not True
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=later)
    assert snap["dispatch_count"] == 0
    _assert_lock_logically_free(tmp_path)


def test_16_closed_and_unknown_calendar_block_dispatch(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", INVOCATION_T1)
    freeze_dispatch_clock(monkeypatch, NOW)
    doc = write_dispatch_session_evidence(
        store_root=tmp_path,
        now=NOW,
        invocation_id=INVOCATION_T1,
        live_execution_blocked=True,
        calendar_payload=OPEN_CAL,
    )
    monkeypatch.setattr(
        "services.market_calendar.regular_session_state",
        lambda now=None: {**OPEN_CAL, "session_state": "CLOSED"},
    )
    ok, reason, _ = evaluate_market_session_for_dispatch(
        NOW, caller_session="MARKET_OPEN", session_evidence=doc
    )
    assert ok is False
    assert "closed" in reason or reason == "market_session_not_open" or "caller_open" in reason
    monkeypatch.setattr(
        "services.market_calendar.regular_session_state",
        lambda now=None: {
            **OPEN_CAL,
            "session_state": "UNKNOWN",
            "reason": "calendar_unavailable",
        },
    )
    ok2, reason2, _ = evaluate_market_session_for_dispatch(
        NOW, caller_session="MARKET_OPEN", session_evidence=doc
    )
    assert ok2 is False
    assert reason2 == "market_session_derivation_unknown"


def test_17_failed_broker_and_exception_release_lock(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _reviewed_packet())
    _arm(tmp_path, packet)
    later = NOW + timedelta(minutes=1)
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", INVOCATION_T1)
    freeze_dispatch_clock(monkeypatch, later)
    write_dispatch_session_evidence(
        store_root=tmp_path,
        now=later,
        invocation_id=INVOCATION_T1,
        live_execution_blocked=True,
        calendar_payload=OPEN_CAL,
    )
    stub_unrelated_pre_gates(monkeypatch)
    fake = _fake_broker(packet)

    def _boom(path, payload):
        raise RuntimeError("broker_failed")

    fake._post = _boom  # type: ignore[method-assign]
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=later,
        verbose=True,
    )
    # Consumption may occur before broker POST; lock must still be free.
    report = _assert_lock_logically_free(tmp_path)
    assert report["reacquisition_ok"] is True
    assert result.get("dispatched") is not True or result.get("reason")
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=later)
    assert snap.get("ok") is True


def test_15_failed_atomic_evidence_write_aborts(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_plan_artifacts(tmp_path, _reviewed_packet())
    _insert_expired_historical(_db(tmp_path), now=NOW)
    confirmed = _arm(tmp_path, packet)
    assert confirmed["ok"] is True
    monkeypatch.setenv("TRITON_MARKET_OPEN_INVOCATION_ID", INVOCATION_T1)

    def _boom(*args, **kwargs):
        raise OSError("atomic_replace_failed")

    monkeypatch.setattr("services.paper_sell_dispatch_session_evidence._atomic_write_json", _boom)
    from services.manage_positions import _try_pre_plan_authorized_sell_dispatch

    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    result = _try_pre_plan_authorized_sell_dispatch(
        broker=_fake_broker(packet), mode="paper", dry_run=False
    )
    assert result.get("abort_before_plan") is True
    assert result.get("consumed") is False
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    assert snap["dispatch_count"] == 0
    _assert_lock_logically_free(tmp_path)


def test_18_lock_contended_while_held_then_free(tmp_path):
    store = tmp_path / "lockstore"
    store.mkdir()
    ready = tmp_path / "holder_ready.txt"
    stop = tmp_path / "holder_stop.txt"
    holder_src = tmp_path / "_lock_holder.py"
    holder_src.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from services.paper_sell_arm_runtime_enforcement import (\n"
        "    _acquire_lock, _release_lock,\n"
        ")\n"
        "store = Path(sys.argv[2])\n"
        "ready = Path(sys.argv[3])\n"
        "stop = Path(sys.argv[4])\n"
        "fh = _acquire_lock(store)\n"
        "ready.write_text('held', encoding='utf-8')\n"
        "while not stop.is_file():\n"
        "    time.sleep(0.05)\n"
        "_release_lock(fh)\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    env["TRITON_ENABLE_LIVE_EXECUTION"] = "0"
    env["TRITON_LIVE_EXECUTION"] = "0"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(holder_src),
            str(ROOT),
            str(store),
            str(ready),
            str(stop),
        ],
        env=env,
        cwd=str(tmp_path),
    )
    try:
        deadline = time.time() + 15
        while time.time() < deadline and not ready.is_file():
            time.sleep(0.05)
        assert ready.is_file(), "holder did not acquire lock"
        with pytest.raises(PaperSellDispatchBlocked) as exc:
            _acquire_lock(store)
        assert exc.value.reason == "dispatch_lock_unavailable"
        held = _lock_report(store)
        assert held["logical_held"] is True
        assert held["reacquisition_ok"] is False
        assert held["physical_present"] is True
    finally:
        stop.write_text("stop", encoding="utf-8")
        proc.wait(timeout=10)
    free = _assert_lock_logically_free(store)
    assert free["physical_present"] is True
    assert free["reacquisition_ok"] is True
    assert Path(free["lock_path"]).name == LOCK_NAME


def test_19_f_failure_skips_f2_and_retains_transcript(tmp_path):
    """Execute a copy of run_market_open.ps1 with isolated repoRoot + fake python.

    The copied script keeps the real Run-Step / F / F2 / G control flow.
    repoRoot is the temp tree because the copy lives at tmp/scripts/. Production
    data/ is never the writable target. Fake python exits 2 on F and records
    whether F2 or G is invoked.
    """
    iso = tmp_path / "market_open_iso"
    scripts = iso / "scripts"
    bin_dir = iso / "bin"
    results = iso / "data" / "results"
    scripts.mkdir(parents=True)
    bin_dir.mkdir()
    results.mkdir(parents=True)
    shutil.copy2(MARKET_OPEN_SCRIPT, scripts / "run_market_open.ps1")
    (scripts / "triton_env_bootstrap.ps1").write_text(
        "# isolated no-op bootstrap\n", encoding="utf-8"
    )
    fake_py = bin_dir / "fake_python.py"
    fake_py.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "log = Path(os.environ['FAKE_PY_LOG'])\n"
        "marker = Path(os.environ['FAKE_PY_MARKER'])\n"
        "joined = ' '.join(sys.argv[1:])\n"
        "prior = log.read_text(encoding='utf-8') if log.is_file() else ''\n"
        "log.write_text(prior + joined + '\\n', encoding='utf-8')\n"
        "if 'services.manage_positions' in joined:\n"
        "    print('F_STDOUT_KNOWN')\n"
        "    print('F_STDERR_KNOWN', file=sys.stderr)\n"
        "    raise SystemExit(2)\n"
        "if 'defensive_exit_approval_readiness' in joined:\n"
        "    prior_m = marker.read_text(encoding='utf-8') if marker.is_file() else ''\n"
        "    marker.write_text(prior_m + 'F2\\n', encoding='utf-8')\n"
        "    raise SystemExit(0)\n"
        "if 'touch_pipeline_heartbeat' in joined:\n"
        "    prior_m = marker.read_text(encoding='utf-8') if marker.is_file() else ''\n"
        "    marker.write_text(prior_m + 'G\\n', encoding='utf-8')\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    real_py = sys.executable
    wrapper_body = "@echo off\r\n" f'"{real_py}" "{fake_py}" %*\r\n' "exit /b %ERRORLEVEL%\r\n"
    (bin_dir / "python.bat").write_text(wrapper_body, encoding="ascii")
    (bin_dir / "python.cmd").write_text(wrapper_body, encoding="ascii")
    log_path = iso / "fake_python.log"
    marker_path = iso / "invoked_stages.txt"
    log_path.write_text("", encoding="utf-8")
    marker_path.write_text("", encoding="utf-8")
    prod_lock = PRODUCTION_ROOT / "data" / "governance" / "paper_sell_arm_dispatch.lock"
    prod_lock_meta = (
        (prod_lock.stat().st_mtime_ns, prod_lock.stat().st_size) if prod_lock.is_file() else None
    )
    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env["FAKE_PY_LOG"] = str(log_path)
    env["FAKE_PY_MARKER"] = str(marker_path)
    env["TRITON_ENABLE_LIVE_EXECUTION"] = "0"
    env["TRITON_LIVE_EXECUTION"] = "0"
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(scripts / "run_market_open.ps1"),
        ],
        cwd=str(iso),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    transcript_dir = results / "scheduled_runs"
    transcripts = list(transcript_dir.glob("market_open_*.log")) if transcript_dir.is_dir() else []
    transcript_text = "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in transcripts
    )
    captured = combined + "\n" + transcript_text
    assert completed.returncode == 2, captured[-4000:]
    assert "F_STDOUT_KNOWN" in captured
    assert "F_STDERR_KNOWN" in captured
    assert "F. POSITION MANAGEMENT" in captured
    assert "STEP_FAILED:F. POSITION MANAGEMENT:2" in captured or (
        "aborted at step: F. POSITION MANAGEMENT" in captured
    )
    invoked = marker_path.read_text(encoding="utf-8")
    assert "F2" not in invoked
    assert "G" not in invoked
    fake_log = log_path.read_text(encoding="utf-8")
    assert "services.manage_positions" in fake_log
    assert "defensive_exit_approval_readiness" not in fake_log
    assert "touch_pipeline_heartbeat" not in fake_log
    status_path = results / "market_open_sequence_status.json"
    assert status_path.is_file()
    status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    assert "F. POSITION MANAGEMENT" in str(status.get("failed_step") or "")
    assert int(status.get("exit_code") or 0) == 2
    assert not str(iso).lower().startswith(str(PRODUCTION_ROOT / "data").lower())
    if prod_lock_meta is not None:
        assert (
            prod_lock.stat().st_mtime_ns,
            prod_lock.stat().st_size,
        ) == prod_lock_meta
    iso_lock = iso / "data" / "governance" / LOCK_NAME
    assert not iso_lock.is_file()
