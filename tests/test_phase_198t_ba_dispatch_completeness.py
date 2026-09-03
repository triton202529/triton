"""Phase 198T-BA — dispatch completeness, remaining-cap, broker_ref, same-cycle jsonl.

PAPER ONLY. Isolated tmp_path SQLite and fake brokers. No production writes,
no real network, no TRITON start, no Streamlit.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from services.defensive_exit_approval_readiness import (
    SAME_CYCLE_DISPATCH_REASON,
    SAME_CYCLE_SUPPRESSION_JSONL,
    STATUS_UNAVAILABLE_SAME_CYCLE,
    generate_from_manage_artifacts,
    write_pre_plan_dispatch_evidence,
)
from services.manage_positions_cycle_binding import (
    BLOCKER_MISSING_SELL_AUTHORITY,
    INVOCATION_ENV,
    selected_row_hash,
    sha256_file_bytes,
    write_cycle_binding,
)
from services.paper_sell_arm_authority import APPROVAL_CLI_CAP, readiness_packet_hash
from services.paper_sell_arm_cap_slice import money_round
from services.paper_sell_arm_console import (
    CONFIRM_PAPER_SELLS_PHRASE,
    arm_reviewed_sells,
    build_presentation,
    confirm_and_arm_paper_sells,
)
from services.paper_sell_arm_runtime_enforcement import (
    _ACTIVE_PERMIT,
    _authorized_clip_audit,
    _downward_runtime_qty,
    build_pre_plan_planned_orders,
)
from services.paper_sell_arm_store import (
    STORE_SCHEMA_VERSION,
    canonical_json,
    count_dispatches_for_authority,
    health_check,
    inspect_sqlite_arm_snapshot,
    write_connection,
)
from services.place_live_orders import dispatch_unused_authorized_sell_slice_pre_plan
from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    LIVE_BLOCKED,
    block_real_network,
    freeze_dispatch_clock,
    isolate_sell_store,
    stub_unrelated_pre_gates,
)
from tests.test_phase_198n_c_sell_cap_slice import (
    NOW,
    _db,
    _pres,
    _write_buy_arm,
    _write_market_session_evidence,
    _write_packet,
)
from tests.test_phase_198t_ab_stale_intent_and_state import AUG26_INTENT, DISPATCH_INTENT_NAME
from tests.test_phase_198t_c_downward_sell_clipping import OverlayQuoteBroker
from tests.test_phase_198t_u_cross_session_sell_state import INV_LATER

SEP2_INV = "mo-20260902_141055-a6352064a97b424ba88446a42d3d7d42"
SEP2_FROZEN = [
    {"symbol": "HD", "action": "EXIT", "qty": 13.0, "est_price": 319.17, "est_notional": 4149.21},
    {"symbol": "GLD", "action": "EXIT", "qty": 2.0, "est_price": 400.41, "est_notional": 800.82},
    {"symbol": "DBA", "action": "EXIT", "qty": 1.0, "est_price": 29.23, "est_notional": 29.23},
]
SEP2_RUNTIME = {"HD": 318.72, "GLD": 400.79, "DBA": 29.23}
SEP2_HOLDINGS = {"HD": 13, "GLD": 16, "DBA": 86, "BRK.B": 4}


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


def _positions(extra: Optional[Dict[str, int]] = None) -> List[Dict[str, str]]:
    held = dict(SEP2_HOLDINGS)
    if extra:
        held.update(extra)
    return [
        {"symbol": sym, "qty": str(qty), "qty_available": str(qty)} for sym, qty in held.items()
    ]


def _seed_prior_terminal_intent(db_path: Path) -> None:
    """Historical terminal dispatch dated before SESSION_OPEN_NOW so arming still works."""
    ok, reason, _ = health_check(db_path)
    assert ok, reason
    created = "2026-08-10T15:19:00+00:00"
    consumed = "2026-08-10T15:20:50.074550+00:00"
    plan = "18c3b9cdb620ac45c59466440aeef887f9023d799a7119be8eca13f4a655c02a"
    batch_id = "manage-mo-20260810_1032"
    batch_hash = "historical-batch-hash-ba-prior"
    old_auth = str(AUG26_INTENT["authority_id"])
    old_dispatch = str(AUG26_INTENT["dispatch_intent_id"])
    old_chal = "sell-chal-ba-prior-historical"
    payload = {
        "authority_id": old_auth,
        "audit_id": old_auth,
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
                old_chal,
                plan,
                batch_hash,
                batch_id,
                canonical_json([]),
                canonical_json({"challenge_id": old_chal}),
                created,
                consumed,
                "CONSUMED",
                consumed,
                old_auth,
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
                old_auth,
                old_chal,
                plan,
                batch_hash,
                canonical_json([]),
                canonical_json(payload),
                created,
                consumed,
                consumed,
                old_dispatch,
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
            ) VALUES (?,?,?,?,?,?,?,'SUBMITTED',0,'["HD"]',?)
            """,
            (
                old_dispatch,
                old_auth,
                plan,
                batch_hash,
                canonical_json([]),
                canonical_json({"dispatch_id": old_dispatch, "authority_id": old_auth}),
                consumed,
                STORE_SCHEMA_VERSION,
            ),
        )
        conn.execute("COMMIT")


def _write_sep2_artifacts(tmp_path: Path, *, actions=None) -> Dict[str, Any]:
    actions = list(actions or SEP2_FROZEN)
    packet = {
        "schema_version": "198I-ZA-SELL-READINESS-1",
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "generation_status": "ok",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "generated_at": NOW.isoformat(),
        "cycle_id": SEP2_INV,
        "run_id": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "batch_id": "manage-mo-198t-ba",
        "plan_id": f"manage_positions_plan@{NOW.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "plan_hash": "planhash198tba0000000000000000000000000000000000000000000000000001",
        "approval_ready_symbols": [a["symbol"] for a in actions],
        "approval_ready_actions": [dict(a) for a in actions],
        "total_proposed_notional": round(sum(float(a["est_notional"]) for a in actions), 2),
        "market_open_invocation_id": SEP2_INV,
        "invocation_id": SEP2_INV,
        "source_plan_sha256": "",
        "selected_row_hash": "",
    }
    rows = [
        {
            "symbol": a["symbol"],
            "final_action": a["action"],
            "qty": str(int(a["qty"])),
            "selected_for_execution": "true",
            "reason_code": "",
            "skip_reason": "",
            "status": "planned",
        }
        for a in actions
    ]
    csv_path = tmp_path / "manage_positions_plan.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    packet["source_plan_sha256"] = sha256_file_bytes(csv_path)
    packet["selected_row_hash"] = selected_row_hash(rows)
    packet["plan_hash"] = readiness_packet_hash(packet)
    _write_packet(tmp_path, packet)
    (tmp_path / "manage_positions_plan.json").write_text(
        json.dumps({"plan_hash": packet["plan_hash"], "timestamp": NOW.isoformat()}),
        encoding="utf-8",
    )
    pos_lines = ["symbol,qty,qty_available,current_price"]
    for a in actions:
        held = SEP2_HOLDINGS.get(a["symbol"], int(a["qty"]))
        pos_lines.append(f"{a['symbol']},{held},{held},{a['est_price']}")
    pos_lines.append("BRK.B,4,4,500.00")
    (tmp_path / "positions_snapshot.csv").write_text("\n".join(pos_lines) + "\n", encoding="utf-8")
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=1))
    return packet


def _arm_sep2(tmp_path: Path, packet: Dict[str, Any]) -> Dict[str, Any]:
    _write_buy_arm(tmp_path, now=NOW)
    pres = _pres(tmp_path, packet)
    armed = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
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
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    assert confirmed["ok"] is True, confirmed
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=1))
    return snap["latest_authority"]["payload"]


def _latest_dispatch(db_path: Path) -> Tuple[Dict[str, Any], Optional[str], str]:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT intent_json, broker_ref, state FROM sell_dispatches ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return json.loads(row[0]), row[1], str(row[2])


def _submitted(fake) -> List[Tuple[str, int]]:
    return [
        (str(p.get("symbol") or "").upper(), int(float(p.get("qty") or 0)))
        for p in fake.submit_payloads
    ]


def _dispatch(tmp_path, monkeypatch, packet, overlay, *, seed_foreign=False, broker_cls=None):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=1))
    stub_unrelated_pre_gates(monkeypatch)
    if seed_foreign:
        _seed_prior_terminal_intent(_db(tmp_path))
        (tmp_path / DISPATCH_INTENT_NAME).write_text(json.dumps(AUG26_INTENT), encoding="utf-8")
    packet = packet or _write_sep2_artifacts(tmp_path)
    authority = _arm_sep2(tmp_path, packet)
    cls = broker_cls or OverlayQuoteBroker
    fake = cls(packet, overlay, positions=_positions())
    result = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=1),
    )
    return packet, authority, fake, result


class NoIdBroker(OverlayQuoteBroker):
    def _post(self, path, payload):
        out = super()._post(path, payload)
        if isinstance(out, dict):
            out.pop("id", None)
            out.pop("order_id", None)
        return out


class PartialFailBroker(OverlayQuoteBroker):
    def _post(self, path, payload):
        if path == "/v2/orders" and str(payload.get("symbol") or "").upper() == "GLD":
            raise TimeoutError("ambiguous broker timeout")
        return super()._post(path, payload)


def test_cap_math_three_clipped_quantities_fit():
    hd = money_round(13 * 318.72)
    gld = money_round(1 * 400.79)
    dba = money_round(1 * 29.23)
    assert hd == 4143.36
    assert gld == 400.79
    assert dba == 29.23
    total = money_round(hd + gld + dba)
    assert total == 4573.38
    assert total <= APPROVAL_CLI_CAP
    reserved_gld = money_round(2 * 400.79)
    assert reserved_gld > 800.82
    qty, reason = _downward_runtime_qty(2, 800.82, 400.79)
    assert qty == 1
    assert "downward_clip" in reason
    qty_dba, _ = _downward_runtime_qty(1, 29.23, 29.23)
    assert qty_dba == 1
    qty_zero, zreason = _downward_runtime_qty(1, 29.23, 29.24)
    assert qty_zero == 0
    assert zreason == "clipped_to_zero_vs_authorized_max_notional"


def test_no_intentional_two_order_sell_ceiling():
    root = Path(__file__).resolve().parents[1]
    hay = []
    for rel in (
        "services/paper_sell_arm_runtime_enforcement.py",
        "services/paper_sell_arm_authority.py",
        "services/paper_sell_arm_cap_slice.py",
        "services/paper_sell_arm_console.py",
        "services/place_live_orders.py",
    ):
        hay.append((root / rel).read_text(encoding="utf-8"))
    blob = "\n".join(hay)
    assert "MAX_SELL_ORDERS_PER_CYCLE" not in blob
    assert "max_sell_orders_per_run = 2" not in blob
    assert "orders[:2]" not in blob


def test_three_authorized_actions_all_fit_after_clip(tmp_path, monkeypatch):
    packet, authority, fake, result = _dispatch(tmp_path, monkeypatch, None, SEP2_RUNTIME)
    assert result.get("dispatched") is True, result
    assert _submitted(fake) == [("HD", 13), ("GLD", 1), ("DBA", 1)]
    intent, broker_ref, state = _latest_dispatch(_db(tmp_path))
    audit = {r["symbol"]: r for r in intent.get("clip_audit") or []}
    assert set(audit) == {"HD", "GLD", "DBA"}
    assert audit["HD"]["outcome"] == "UNCHANGED"
    assert audit["GLD"]["outcome"] == "CLIPPED_DOWN"
    assert audit["GLD"]["runtime_quantity"] == 1.0
    assert audit["DBA"]["outcome"] == "UNCHANGED"
    assert audit["DBA"]["runtime_quantity"] == 1.0
    requested = {r["symbol"] for r in intent["requested"]}
    assert requested == {"HD", "GLD", "DBA"}
    assert broker_ref
    assert "fake-ord-" in broker_ref
    assert state == "SUBMITTED"
    assert "FILLED" not in str(result.get("submit") or {})
    aid = str(authority.get("authority_id") or authority.get("audit_id"))
    assert count_dispatches_for_authority(_db(tmp_path), aid) == 1
    assert all(str(p.get("symbol")).upper() != "BRK.B" for p in fake.submit_payloads)
    assert os.environ.get("TRITON_ENABLE_LIVE_EXECUTION") == "0"


def test_earlier_clip_frees_cap_for_later_action(tmp_path, monkeypatch):
    _packet, _authority, fake, result = _dispatch(tmp_path, monkeypatch, None, SEP2_RUNTIME)
    assert result.get("dispatched") is True, result
    got = dict(_submitted(fake))
    assert got["GLD"] == 1
    assert got["DBA"] == 1
    submitted_notional = money_round(13 * 318.72 + 1 * 400.79 + 1 * 29.23)
    assert submitted_notional <= APPROVAL_CLI_CAP


def test_later_action_remaining_batch_cap_explicit(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = {
        "approval_ready_actions": [
            {
                "symbol": "HD",
                "action": "EXIT",
                "qty": 1.0,
                "est_price": 4900.0,
                "est_notional": 4900.0,
            },
            {
                "symbol": "GLD",
                "action": "EXIT",
                "qty": 1.0,
                "est_price": 200.0,
                "est_notional": 200.0,
            },
        ]
    }
    authority = {
        "authorized_actions": [
            {"symbol": "HD", "action": "EXIT", "max_qty": 1.0, "max_notional": 4900.0},
            {"symbol": "GLD", "action": "EXIT", "max_qty": 1.0, "max_notional": 200.0},
        ],
        "per_symbol_max_quantity": {"HD": 1.0, "GLD": 1.0},
        "per_symbol_max_notional": {"HD": 4900.0, "GLD": 200.0},
    }
    fake = OverlayQuoteBroker(
        packet,
        {"HD": 4900.0, "GLD": 200.0},
        positions=[
            {"symbol": "HD", "qty": "1", "qty_available": "1"},
            {"symbol": "GLD", "qty": "1", "qty_available": "1"},
        ],
    )
    ok, reason, planned = build_pre_plan_planned_orders(authority, fake, now=NOW)
    assert ok, reason
    by = {p.symbol: p for p in planned}
    assert int(by["HD"].qty) == 1
    assert int(by["GLD"].qty) == 0
    assert by["GLD"].clip_reason == "clipped_to_zero_vs_remaining_batch_cap"
    audit = {r["symbol"]: r for r in _authorized_clip_audit(authority, planned)}
    assert audit["GLD"]["outcome"] == "SKIPPED_ZERO"
    assert audit["GLD"]["reason"] == "clipped_to_zero_vs_remaining_batch_cap"


def test_zero_quantity_receives_explicit_disposition(tmp_path, monkeypatch):
    overlay = dict(SEP2_RUNTIME)
    overlay["DBA"] = 29.24
    _packet, _authority, fake, result = _dispatch(tmp_path, monkeypatch, None, overlay)
    assert result.get("dispatched") is True, result
    assert _submitted(fake) == [("HD", 13), ("GLD", 1)]
    intent, _ref, _state = _latest_dispatch(_db(tmp_path))
    audit = {r["symbol"]: r for r in intent.get("clip_audit") or []}
    assert audit["DBA"]["outcome"] == "SKIPPED_ZERO"
    assert audit["DBA"]["runtime_quantity"] == 0.0
    assert "clipped_to_zero" in str(audit["DBA"]["reason"])
    assert "DBA" not in {r["symbol"] for r in intent["requested"]}
    jsonl = (tmp_path / "paper_sell_arm_audit.jsonl").read_text(encoding="utf-8")
    assert "clip_audit" in jsonl
    assert "DBA" in jsonl


def test_foreign_historical_intent_still_persists_clip_audit(tmp_path, monkeypatch):
    overlay = dict(SEP2_RUNTIME)
    overlay["DBA"] = 29.24
    packet = _write_sep2_artifacts(tmp_path)
    _packet, authority, fake, result = _dispatch(
        tmp_path, monkeypatch, packet, overlay, seed_foreign=True
    )
    assert result.get("dispatched") is True, result
    leftover = json.loads((tmp_path / DISPATCH_INTENT_NAME).read_text(encoding="utf-8"))
    assert leftover["authority_id"] == AUG26_INTENT["authority_id"]
    intent, broker_ref, _state = _latest_dispatch(_db(tmp_path))
    audit = {r["symbol"]: r for r in intent.get("clip_audit") or []}
    assert audit["DBA"]["outcome"] == "SKIPPED_ZERO"
    assert broker_ref
    aid = str(authority.get("authority_id") or authority.get("audit_id"))
    assert count_dispatches_for_authority(_db(tmp_path), aid) == 1
    assert _submitted(fake) == [("HD", 13), ("GLD", 1)]


def test_partial_broker_success_plus_explicit_skip(tmp_path, monkeypatch):
    overlay = dict(SEP2_RUNTIME)
    overlay["DBA"] = 29.24
    _packet, authority, fake, result = _dispatch(
        tmp_path, monkeypatch, None, overlay, broker_cls=PartialFailBroker
    )
    assert result.get("consumed") is True
    assert _submitted(fake) == [("HD", 13)]
    intent, broker_ref, state = _latest_dispatch(_db(tmp_path))
    audit = {r["symbol"]: r for r in intent.get("clip_audit") or []}
    assert audit["DBA"]["outcome"] == "SKIPPED_ZERO"
    assert broker_ref
    assert state in {"SUBMITTED", "OUTCOME_UNKNOWN"}
    aid = str(authority.get("authority_id") or authority.get("audit_id"))
    second = OverlayQuoteBroker(
        json.loads(
            (tmp_path / "defensive_exit_approval_readiness.json").read_text(encoding="utf-8")
        ),
        overlay,
        positions=_positions(),
    )
    replay = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=second,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=2),
    )
    assert not replay.get("dispatched")
    assert second.submit_payloads == []
    assert count_dispatches_for_authority(_db(tmp_path), aid) == 1


def test_broker_response_without_order_id_stays_unconfirmed(tmp_path, monkeypatch):
    _packet, _authority, fake, result = _dispatch(
        tmp_path, monkeypatch, None, SEP2_RUNTIME, broker_cls=NoIdBroker
    )
    assert result.get("dispatched") is True, result
    _intent, broker_ref, state = _latest_dispatch(_db(tmp_path))
    assert broker_ref in {None, ""}
    assert state == "SUBMITTED"
    submit = result.get("submit") or {}
    for row in submit.get("broker_submissions") or []:
        assert not row.get("broker_order_id")
        assert str(row.get("broker_status") or "").upper() != "FILLED"


def test_single_use_replay_after_partial_success(tmp_path, monkeypatch):
    overlay = dict(SEP2_RUNTIME)
    overlay["DBA"] = 29.24
    packet, authority, fake, result = _dispatch(tmp_path, monkeypatch, None, overlay)
    assert result.get("dispatched") is True
    assert _submitted(fake) == [("HD", 13), ("GLD", 1)]
    aid = str(authority.get("authority_id") or authority.get("audit_id"))
    replay_fake = OverlayQuoteBroker(packet, overlay, positions=_positions())
    replay = dispatch_unused_authorized_sell_slice_pre_plan(
        broker=replay_fake,
        results_root=tmp_path,
        store_root=tmp_path,
        db_path=_db(tmp_path),
        now=NOW + timedelta(minutes=2),
    )
    assert not replay.get("dispatched")
    assert replay_fake.submit_payloads == []
    assert count_dispatches_for_authority(_db(tmp_path), aid) == 1
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=2))
    assert snap["latest_authority"]["state"] == "CONSUMED"


def test_same_cycle_suppression_jsonl_and_later_cycle_rebuild(tmp_path, monkeypatch):
    packet, authority, fake, result = _dispatch(tmp_path, monkeypatch, None, SEP2_RUNTIME)
    assert result.get("dispatched") is True
    monkeypatch.setenv(INVOCATION_ENV, SEP2_INV)
    write_cycle_binding(
        results_root=tmp_path,
        invocation_id=SEP2_INV,
        plan_generated_at=NOW.isoformat(),
        f_stage_status="COMPLETE",
        now=NOW + timedelta(minutes=1),
        remaining_blocker=BLOCKER_MISSING_SELL_AUTHORITY,
        operator_execution_required=True,
        block_reasons=["DISPATCH_BLOCKED_APPROVAL_REQUIRED"],
        orders_executed=0,
    )
    write_pre_plan_dispatch_evidence(
        tmp_path,
        {
            "attempted": True,
            "dispatched": True,
            "consumed": True,
            "broker_submissions": len(fake.submit_payloads),
            "broker_orders_submitted": True,
            "dispatch_performed": True,
            "dispatch_intent_id": result.get("dispatch_intent_id"),
            "authority_id": authority.get("authority_id") or authority.get("audit_id"),
            "dispatch_terminal_status": "TERMINAL_SUCCESS",
            "reason": "ok",
            "submit": {"placed": len(fake.submit_payloads), "failed": 0},
        },
        invocation_id=SEP2_INV,
        now=NOW + timedelta(minutes=1),
        session_date=NOW.date().isoformat(),
    )
    f2 = generate_from_manage_artifacts(results_root=tmp_path, now=NOW + timedelta(minutes=1))
    assert f2["packet"]["status"] == STATUS_UNAVAILABLE_SAME_CYCLE
    assert f2["packet"]["reason"] == SAME_CYCLE_DISPATCH_REASON
    jsonl_path = tmp_path / SAME_CYCLE_SUPPRESSION_JSONL
    assert jsonl_path.is_file()
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert rows[0]["classification"] == STATUS_UNAVAILABLE_SAME_CYCLE
    assert rows[0]["reason"] == SAME_CYCLE_DISPATCH_REASON
    assert rows[0]["invocation_id"] == SEP2_INV
    assert rows[0]["authority_id"]
    later = NOW + timedelta(hours=1)
    later_rows = [
        {
            "symbol": "DBA",
            "final_action": "EXIT",
            "qty": "1",
            "selected_for_execution": "true",
            "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
            "skip_reason": "",
            "current_position_qty": "86",
            "current_price": "29.24",
            "planned_notional": "29.24",
        }
    ]
    with (tmp_path / "manage_positions_plan.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(later_rows[0].keys()))
        writer.writeheader()
        writer.writerows(later_rows)
    (tmp_path / "manage_positions_plan.json").write_text(
        json.dumps({"plan_hash": "later-ba", "timestamp": later.isoformat()}),
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
    assert jsonl_path.is_file()
    assert later_f2["packet"].get("creates_no_authority") is True


def test_operator_console_shows_three_slice_actions(tmp_path, monkeypatch):
    isolate_sell_store(monkeypatch, tmp_path)
    packet = _write_sep2_artifacts(tmp_path)
    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    assert int(pres.get("slice_action_count") or pres.get("action_count") or 0) == 3
    assert int(pres.get("action_count") or 0) == 3
    names = {str(a.get("symbol") or "").upper() for a in pres.get("actions") or []}
    assert names == {"HD", "GLD", "DBA"}
    assert "BRK.B" not in names


def test_live_execution_stays_blocked(tmp_path, monkeypatch):
    _packet, _authority, fake, result = _dispatch(tmp_path, monkeypatch, None, SEP2_RUNTIME)
    assert result.get("dispatched") is True
    assert fake.mode == "paper"
    assert "alpaca.markets" not in str(fake.base)
    assert os.environ.get("TRITON_ENABLE_LIVE_EXECUTION") == "0"
    assert LIVE_BLOCKED["live_blocked"] is True
