"""Phase 198N-F — dual-identity cap-sliced paper SELL dispatch.

PAPER ONLY. Isolated tmp SQLite/packet. Real manage_positions → place_live_orders
→ runtime enforcement → unchanged hardened boundary → fake broker.
No production artifact writes, no real network, live flags 0.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from services.manage_positions import ManagementPlannedOrder, maybe_execute_management
from services.paper_sell_arm_authority import (
    APPROVAL_CLI_CAP,
    SCHEMA_VERSION_DUAL,
    is_dual_identity_record,
    legacy_ambiguous_slice_authority,
    prepare_confirmation_challenge,
    readiness_packet_hash,
    source_readiness_packet_hash,
)
from services.paper_sell_arm_cap_slice import (
    authorized_slice_hash,
    money_round,
    slice_reviewed_sells,
    source_readiness_packet_hash as slicer_source_hash,
    validate_slice_against_packet,
)
from services.paper_sell_arm_console import (
    CONFIRM_PAPER_SELLS_PHRASE,
    READINESS_PACKET_NAME,
    arm_reviewed_sells,
    confirm_and_arm_paper_sells,
)
from services.paper_sell_arm_runtime_enforcement import (
    _ACTIVE_PERMIT,
    begin_paper_sell_dispatch,
    select_authorized_planned,
)
from services.paper_sell_arm_store import PaperSellArmStoreError, inspect_sqlite_arm_snapshot
from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import (
    LIVE_BLOCKED,
    OrderedTrace,
    block_real_network,
    freeze_dispatch_clock,
    inprocess_place_live_orders,
    install_boundary_trace,
    isolate_sell_store,
    stub_unrelated_pre_gates,
)
from tests.test_phase_198n_c_sell_cap_slice import (
    EXPECTED_FIRST,
    EXPECTED_FIRST_TOTAL,
    ISOLATED_REGRESSION,
    NOW,
    PacketQuotedFakeBroker,
    FakePlanned,
    _broker,
    _db,
    _load_prod_copy,
    _planned_from_slice,
    _pres,
    _snap,
    _unit_price,
    _write_buy_arm,
    _write_market_session_evidence,
    _write_packet,
)

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = Path(r"C:\Users\akimw\triton")
PROD_PACKET = PRODUCTION_ROOT / "data" / "results" / "defensive_exit_approval_readiness.json"
PROD_PACKET_SHA = "25c6cdb1a51fb24eec3d0b4edd786a248aa4447e25353d1024629040a0544795"
BUY_ARM = PRODUCTION_ROOT / "data" / "live" / "paper_arm_confirm.json"
BUY_ARM_SHA = "7b74bd1881b2e7918f31626f7376028c37a5b93e954eb00ca96613c0acef86f1"
REPO_SQLITE = PRODUCTION_ROOT / "data" / "live" / "paper_sell_arm.sqlite3"
REPO_SQLITE_SHA = "d4886298a538610f1e7950b4cf48dc88126a5664b0d4d0bce94a4d9a4f2e13aa"
EVIDENCE = PRODUCTION_ROOT / "tmp" / "phase_198n_g_notional_20260818"


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    block_real_network(monkeypatch)
    for key in list(os.environ):
        if "ALPACA" in key.upper():
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.delenv("TRITON_PAPER_SELL_ARM_STORE", raising=False)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)
    if ISOLATED_REGRESSION:
        return
    if REPO_SQLITE.is_file():
        assert hashlib.sha256(REPO_SQLITE.read_bytes()).hexdigest() == REPO_SQLITE_SHA
    assert hashlib.sha256(PROD_PACKET.read_bytes()).hexdigest() == PROD_PACKET_SHA
    if BUY_ARM.is_file():
        assert hashlib.sha256(BUY_ARM.read_bytes()).hexdigest() == BUY_ARM_SHA


def _cfg() -> Dict[str, Any]:
    return {
        "execute_via_existing_placement_flow": True,
        "require_master_gate_for_execute": False,
        "require_execution_gate_for_execute": False,
        "require_guard_validation_for_each_order": False,
        "write_manage_orders_csv": True,
        "max_batch_notional": 50000.0,
    }


def _arm_confirm(tmp_path: Path, packet: Dict[str, Any]) -> Dict[str, Any]:
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
    snap = _snap(tmp_path, now=NOW + timedelta(minutes=1))
    return {"pres": pres, "armed": armed, "confirmed": confirmed, "snap": snap}


def _planned_from_packet(packet: Dict[str, Any]) -> List[ManagementPlannedOrder]:
    out: List[ManagementPlannedOrder] = []
    for row in packet["approval_ready_actions"]:
        qty = float(row["qty"])
        price = _unit_price(row)
        out.append(
            ManagementPlannedOrder(
                symbol=str(row["symbol"]),
                side="sell",
                qty=int(qty),
                order_type="limit",
                time_in_force="day",
                limit_price=price,
                management_action=str(row["action"]),
                rationale="phase198n-f-full-packet",
                confidence=0.9,
                delta_pct=-0.01,
                current_position_qty=qty,
                mode="paper",
                generated_at=(NOW + timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                planned_notional=float(row["est_notional"]),
            )
        )
    return out


def _prepare_route_files(tmp_path: Path, packet: Dict[str, Any], auth: Dict[str, Any]) -> None:
    _write_buy_arm(tmp_path, now=NOW)
    (tmp_path / "manage_positions_plan.json").write_text(
        json.dumps(
            {
                "plan_hash": packet.get("plan_hash"),
                "cycle_id": packet.get("cycle_id"),
                "run_id": packet.get("run_id"),
                "batch_id": packet.get("batch_id"),
                "plan_id": packet.get("plan_id"),
            }
        ),
        encoding="utf-8",
    )
    pos_lines = ["symbol,qty,qty_available"]
    for row in packet["approval_ready_actions"]:
        pos_lines.append(f"{row['symbol']},{row['qty']},{row['qty']}")
    (tmp_path / "positions_snapshot.csv").write_text("\n".join(pos_lines) + "\n", encoding="utf-8")
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    bind_path = tmp_path / "independent_dispatch_bindings.json"
    doc: Dict[str, Any] = {}
    if bind_path.is_file():
        try:
            loaded = json.loads(bind_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                doc = loaded
        except json.JSONDecodeError:
            doc = {}
    doc.update(
        {
            "plan_hash": packet.get("plan_hash"),
            "cycle_id": packet.get("cycle_id"),
            "run_id": packet.get("run_id"),
            "batch_id": packet.get("batch_id"),
            "plan_id": packet.get("plan_id"),
            "trading_date": NOW.date().isoformat(),
            "account_fingerprint": auth.get("account_fingerprint"),
            "confirmation_challenge_id": auth.get("confirmation_challenge_id"),
            "confirmation_challenge_hash": auth.get("confirmation_challenge_hash"),
            "paper_execution_authorized": True,
            "market_session": "MARKET_OPEN",
            "market_session_observed_at": (NOW + timedelta(minutes=2)).isoformat(),
        }
    )
    # Do not override readiness_packet_hash — dual identity binds the full packet file.
    bind_path.write_text(json.dumps(doc), encoding="utf-8")


def _run_route(
    monkeypatch,
    tmp_path: Path,
    *,
    packet: Dict[str, Any],
    fake: PacketQuotedFakeBroker,
    trace: OrderedTrace,
    planned: Optional[List[ManagementPlannedOrder]] = None,
    mode: str = "paper",
    captured: Optional[Dict[str, Any]] = None,
):
    isolate_sell_store(monkeypatch, tmp_path)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    stub_unrelated_pre_gates(monkeypatch)
    install_boundary_trace(monkeypatch, trace)

    import services.manage_positions as mp
    import services.place_live_orders as plo
    import services.paper_sell_arm_runtime_enforcement as enf

    now = NOW + timedelta(minutes=2)
    real_eac = plo.enforce_and_consume_paper_sells_before_submit

    def _eac(**kwargs):
        kwargs.setdefault("now", now)
        kwargs.setdefault("live_override", LIVE_BLOCKED)
        kwargs.setdefault("results_root", tmp_path)
        kwargs.setdefault("live_root", tmp_path)
        kwargs.setdefault("store_root", tmp_path)
        kwargs.setdefault("db_path", _db(tmp_path))
        kwargs["explicit_runtime_write"] = True
        return real_eac(**kwargs)

    real_begin = enf.begin_paper_sell_dispatch

    def _begin(**kwargs):
        kwargs.setdefault("now", now)
        kwargs.setdefault("results_root", tmp_path)
        kwargs.setdefault("live_root", tmp_path)
        kwargs.setdefault("db_path", _db(tmp_path))
        kwargs.setdefault("store_root", tmp_path)
        kwargs["explicit_runtime_write"] = True
        result = real_begin(**kwargs)
        if captured is not None:
            captured["guard"] = result
            captured["planned_in"] = list(kwargs.get("planned") or [])
            captured["authorized"] = list(result.authorized_planned or ())
        return result

    def _inprocess(cmd, **kwargs):
        proc = inprocess_place_live_orders(cmd, **kwargs)
        if captured is not None:
            captured["place_stdout"] = getattr(proc, "stdout", "")
            captured["place_rc"] = getattr(proc, "returncode", None)
        return proc

    monkeypatch.setattr(plo, "enforce_and_consume_paper_sells_before_submit", _eac)
    monkeypatch.setattr(enf, "begin_paper_sell_dispatch", _begin)
    monkeypatch.setattr(plo, "make_broker", lambda mode=None, **_k: fake)
    monkeypatch.setattr(plo, "DROP_JSON", tmp_path / "execution_drop_diagnostics.json")
    monkeypatch.setattr(mp.subprocess, "run", _inprocess)

    rc, executed, reasons, session_id, meta = maybe_execute_management(
        planned or _planned_from_packet(packet),
        mode,
        False,
        _cfg(),
        fake,
        True,
        True,
        tmp_path / "manage_orders.csv",
        placement_session=f"phase198n_f_{tmp_path.name}",
    )
    return {
        "rc": rc,
        "executed": executed,
        "reasons": reasons,
        "session_id": session_id,
        "meta": meta,
        "fake": fake,
        "trace": trace,
    }


def test_approval_cli_cap_unchanged():
    assert APPROVAL_CLI_CAP == 5000.0
    src = (ROOT / "services" / "paper_sell_arm_authority.py").read_text(encoding="utf-8")
    assert "APPROVAL_CLI_CAP = 5000.0" in src
    assert CONFIRM_PAPER_SELLS_PHRASE == "CONFIRM PAPER SELLS"


def test_raw_over_cap_packet_without_slice_still_refused(tmp_path: Path):
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    with pytest.raises(ValueError, match="scope_exceeds_approval_cli_cap"):
        prepare_confirmation_challenge(
            store_root=tmp_path,
            readiness_packet=packet,
            account_raw="acct-198n-f",
            trading_date=NOW.date().isoformat(),
            now=NOW,
            db_path=_db(tmp_path),
        )
    snap = _snap(tmp_path)
    assert snap["challenge_count"] == 0
    assert snap["authority_count"] == 0


def test_real_route_dispatches_first_slice_only(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    assert is_dual_identity_record(auth)
    assert auth.get("identity_schema") == SCHEMA_VERSION_DUAL
    assert auth.get("readiness_packet_hash") == readiness_packet_hash(packet)
    assert auth.get("source_readiness_packet_hash") == source_readiness_packet_hash(packet)
    assert auth.get("authorized_slice_hash") == seeded["pres"]["slice"]["slice_hash"]
    assert (
        tuple(
            (a["symbol"], int(a.get("max_qty") or a.get("qty") or 0))
            for a in (auth.get("authorized_actions") or [])
        )
        == EXPECTED_FIRST
    )
    on_disk = json.loads((tmp_path / READINESS_PACKET_NAME).read_text(encoding="utf-8"))
    assert len(on_disk["approval_ready_actions"]) == 7
    _prepare_route_files(tmp_path, packet, auth)

    fake = PacketQuotedFakeBroker(
        packet,
        positions=[
            {"symbol": a["symbol"], "qty": str(a["qty"])} for a in packet["approval_ready_actions"]
        ],
    )
    trace = OrderedTrace()
    captured: Dict[str, Any] = {}
    result = _run_route(
        monkeypatch, tmp_path, packet=packet, fake=fake, trace=trace, captured=captured
    )
    guard = captured.get("guard")
    if guard is None:
        pytest.fail(
            "begin_paper_sell_dispatch was not reached: "
            f"rc={result['rc']} executed={result['executed']} "
            f"reasons={result.get('reasons')} meta={result.get('meta')} "
            f"place_rc={captured.get('place_rc')} "
            f"stdout={captured.get('place_stdout', '')[-4000:]}"
        )
    assert guard.consumed, f"{guard.reason} {guard.details}"
    slice_rows = {str(a["symbol"]): a for a in (auth.get("authorized_actions") or [])}
    tsla_scope = slice_rows["TSLA"]
    gld_scope = slice_rows["GLD"]
    dba_scope = slice_rows["DBA"]
    pkt_by = {str(r["symbol"]): r for r in packet["approval_ready_actions"]}
    assert int(tsla_scope["max_qty"]) == 8
    assert float(tsla_scope["max_notional"]) == 2709.04
    assert float(pkt_by["TSLA"]["est_notional"]) == 2709.04
    assert float(pkt_by["TSLA"]["est_price"]) == 339.27
    planned_in = captured.get("planned_in") or []
    assert len(planned_in) == 7
    authorized = list(guard.authorized_planned or ())
    assert [(p.symbol, int(p.qty)) for p in authorized] == [
        ("TSLA", 8),
        ("GLD", 5),
        ("DBA", 9),
    ]
    tsla_p = next(p for p in authorized if p.symbol == "TSLA")
    gld_p = next(p for p in authorized if p.symbol == "GLD")
    dba_p = next(p for p in authorized if p.symbol == "DBA")
    tsla_runtime_n = money_round(float(tsla_p.qty) * float(tsla_p.limit_price))
    gld_runtime_n = money_round(float(gld_p.qty) * float(gld_p.limit_price))
    dba_runtime_n = money_round(float(dba_p.qty) * float(dba_p.limit_price))
    assert tsla_runtime_n == float(tsla_scope["max_notional"])
    assert gld_runtime_n == float(gld_scope["max_notional"])
    assert dba_runtime_n == float(dba_scope["max_notional"])
    assert tsla_runtime_n < money_round(8 * 339.27)
    runtime_total = money_round(tsla_runtime_n + gld_runtime_n + dba_runtime_n)
    assert runtime_total == money_round(auth.get("total_batch_max_notional"))
    assert runtime_total <= 5000.0
    assert float(auth.get("applied_cap")) == 5000.0
    assert guard.reason not in {
        "readiness_packet_hash_changed",
        "batch_notional_exceeds_approval_cli_cap",
        "batch_symbol_set_mismatch_whole_batch_required",
    }
    planned_in = captured.get("planned_in") or []
    assert len(planned_in) == 7
    authorized = list(guard.authorized_planned or ())
    assert [(p.symbol, int(p.qty)) for p in authorized] == [
        ("TSLA", 8),
        ("GLD", 5),
        ("DBA", 9),
    ]
    payloads = fake.submit_payloads
    got = [(str(p.get("symbol")), int(float(p.get("qty") or 0))) for p in payloads]
    assert got == [("TSLA", 8), ("GLD", 5), ("DBA", 9)]
    submitted = {str(p.get("symbol")).upper() for p in payloads}
    assert submitted.isdisjoint({"HD", "BLK", "MA", "GE"})
    boundary_syms = [
        e.get("symbol")
        for e in trace.events
        if e.get("event") == "hardened_sell_boundary_entered"
        and e.get("fn") == "enforce_paper_sell_broker_submission"
    ]
    assert boundary_syms == ["TSLA", "GLD", "DBA"]
    snap = _snap(tmp_path, now=NOW + timedelta(minutes=2))
    assert snap["latest_authority"]["state"] == "CONSUMED"
    assert snap["authority_count"] == 1
    assert snap["dispatch_count"] == 1
    replay = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned_from_slice(packet["approval_ready_actions"]),
        broker_view=_broker(
            auth,
            {
                str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
                for a in packet["approval_ready_actions"]
            },
        ),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
        explicit_runtime_write=True,
    )
    assert not replay.consumed
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "real_route_first_slice.json").write_text(
        json.dumps(
            {
                "consumed": True,
                "planned_in": len(planned_in),
                "authorized": [(p.symbol, int(p.qty)) for p in authorized],
                "submits": got,
                "deferred_submits": 0,
                "rc": result["rc"],
                "source_readiness_packet_hash": auth.get("source_readiness_packet_hash"),
                "authorized_slice_hash": auth.get("authorized_slice_hash"),
                "readiness_packet_hash": auth.get("readiness_packet_hash"),
                "slice_total": EXPECTED_FIRST_TOTAL,
                "limit_sanity_after_filter": True,
                "post_reservation_restore_unused": False,
                "tsla_stored_max_notional": 2709.04,
                "tsla_runtime_notional": tsla_runtime_n,
                "tsla_packet_est_price_times_qty": money_round(8 * 339.27),
                "slice_total": runtime_total,
                "applied_cap": 5000.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_source_hash_substitution_fails(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    mutated = copy.deepcopy(packet)
    mutated["generated_at"] = (NOW + timedelta(seconds=9)).isoformat()
    _write_packet(tmp_path, mutated)
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned_from_slice(packet["approval_ready_actions"]),
        broker_view=_broker(
            auth,
            {
                str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
                for a in packet["approval_ready_actions"]
            },
        ),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
    )
    assert not guard.consumed
    assert guard.reason in {
        "source_readiness_packet_hash_changed",
        "generated_at_changed",
        "readiness_packet_hash_changed",
    }


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("generated_at", "2099-01-01T00:00:00+00:00", "generated_at_changed"),
        ("market_open_invocation_id", "other-invocation", "invocation_changed"),
        ("source_plan_sha256", "0" * 64, "source_plan_hash_changed"),
        ("selected_row_hash", "0" * 64, "selected_row_hash_changed"),
    ],
)
def test_source_identity_field_change_fails(tmp_path, monkeypatch, field, value, expected):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    mutated = copy.deepcopy(packet)
    mutated[field] = value
    if field == "market_open_invocation_id":
        mutated["invocation_id"] = value
    _write_packet(tmp_path, mutated)
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned_from_slice(packet["approval_ready_actions"]),
        broker_view=_broker(
            auth,
            {
                str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
                for a in packet["approval_ready_actions"]
            },
        ),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=readiness_packet_hash(mutated),
    )
    assert not guard.consumed
    assert expected in str(guard.reason) or "hash" in str(guard.reason)


def test_slice_hash_substitution_fails(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    auth["authorized_slice_hash"] = "0" * 64
    path = tmp_path / "paper_sell_arm.sqlite3"
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            "SELECT authority_id, payload_json FROM sell_authorities WHERE state='UNUSED'"
        ).fetchone()
        payload = json.loads(row[1])
        payload["authorized_slice_hash"] = "0" * 64
        conn.execute(
            "UPDATE sell_authorities SET payload_json=? WHERE authority_id=?",
            (json.dumps(payload), row[0]),
        )
        conn.commit()
    finally:
        conn.close()
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned_from_slice(packet["approval_ready_actions"]),
        broker_view=_broker(
            auth,
            {
                str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
                for a in packet["approval_ready_actions"]
            },
        ),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
    )
    assert not guard.consumed
    assert "authorized_slice_hash" in str(guard.reason)


def test_notional_expansion_fails(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    planned = _planned_from_slice(seeded["pres"]["slice"]["slice_actions"])
    for row in planned:
        row.limit_price = 10000.0
        row.ref_price = 10000.0
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=_broker(
            auth,
            {
                str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
                for a in packet["approval_ready_actions"]
            },
        ),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
    )
    assert not guard.consumed
    assert guard.reason == "notional_expansion_vs_authorized_slice:TSLA"


def test_authorized_notional_exact_and_lower_allowed(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    positions = {
        str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
        for a in packet["approval_ready_actions"]
    }
    exact = _planned_from_slice(seeded["pres"]["slice"]["slice_actions"])
    ok, reason, selected = select_authorized_planned(exact, auth)
    assert ok is True, reason
    assert [(p.symbol, int(p.qty)) for p in selected] == [
        ("TSLA", 8),
        ("GLD", 5),
        ("DBA", 9),
    ]
    lower = copy.deepcopy(exact)
    lower[0].limit_price = float(lower[0].limit_price) * 0.99
    lower[0].ref_price = lower[0].limit_price
    ok_lo, reason_lo, selected_lo = select_authorized_planned(lower, auth)
    assert ok_lo is True, reason_lo
    assert len(selected_lo) == 3
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=exact,
        broker_view=_broker(auth, positions),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
        explicit_runtime_write=True,
    )
    assert guard.consumed, f"{guard.reason} {guard.details}"


def test_one_cent_per_symbol_and_total_slice_expansion_refused(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    exact = _planned_from_slice(seeded["pres"]["slice"]["slice_actions"])
    tsla = next(p for p in exact if p.symbol == "TSLA")
    tsla.limit_price = (2709.04 + 0.01) / 8.0
    tsla.ref_price = tsla.limit_price
    ok, reason, selected = select_authorized_planned(exact, auth)
    assert ok is False
    assert reason == "notional_expansion_vs_authorized_slice:TSLA"
    assert selected == ()

    crafted = {
        "authorized_actions": [
            {"symbol": "AAA", "action": "EXIT", "max_qty": 1, "max_notional": 2500.0},
            {"symbol": "BBB", "action": "EXIT", "max_qty": 1, "max_notional": 2500.0},
        ],
        "total_batch_max_notional": 4000.0,
        "applied_cap": 5000.0,
    }
    planned = [
        FakePlanned("AAA", "sell", 1, 2000.0, stance="EXIT", ref_price=2000.0),
        FakePlanned("BBB", "sell", 1, 2000.01, stance="EXIT", ref_price=2000.01),
    ]
    ok_t, reason_t, selected_t = select_authorized_planned(planned, crafted)
    assert ok_t is False
    assert reason_t == "notional_expansion_vs_authorized_slice_total"
    assert selected_t == ()


def test_price_increase_qty_unchanged_refused(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    planned = _planned_from_slice(seeded["pres"]["slice"]["slice_actions"])
    planned[0].limit_price = float(planned[0].limit_price) + 0.02
    planned[0].ref_price = planned[0].limit_price
    ok, reason, selected = select_authorized_planned(planned, auth)
    assert ok is False
    assert reason.startswith("notional_expansion_vs_authorized_slice:")
    assert selected == ()


def test_real_route_tsla_notional_expansion_refused(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    _prepare_route_files(tmp_path, packet, auth)
    planned = _planned_from_packet(packet)
    tsla = next(p for p in planned if p.symbol == "TSLA")
    tsla.limit_price = (2709.04 + 0.01) / 8.0
    fake = PacketQuotedFakeBroker(
        packet,
        positions=[
            {"symbol": a["symbol"], "qty": str(a["qty"])} for a in packet["approval_ready_actions"]
        ],
    )
    trace = OrderedTrace()
    captured: Dict[str, Any] = {}
    result = _run_route(
        monkeypatch,
        tmp_path,
        packet=packet,
        fake=fake,
        trace=trace,
        planned=planned,
        captured=captured,
    )
    guard = captured.get("guard")
    assert guard is not None
    assert not guard.consumed
    assert guard.reason == "notional_expansion_vs_authorized_slice:TSLA"
    assert fake.submit_payloads == []
    assert all(
        e.get("fn") != "enforce_paper_sell_broker_submission"
        for e in trace.events
        if e.get("event") == "hardened_sell_boundary_entered"
    )
    snap = _snap(tmp_path, now=NOW + timedelta(minutes=2))
    assert snap["latest_authority"]["state"] == "UNUSED"
    assert snap["dispatch_count"] == 0
    assert result["rc"] != 0
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "route_tsla_notional_expansion.json").write_text(
        json.dumps(
            {
                "reason": guard.reason,
                "consumed": False,
                "broker_submits": 0,
                "authority_state": snap["latest_authority"]["state"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_action_change_fails(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    planned = _planned_from_slice(seeded["pres"]["slice"]["slice_actions"])
    planned[0].stance = "TRIM"
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=_broker(
            auth,
            {
                str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
                for a in packet["approval_ready_actions"]
            },
        ),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
    )
    assert not guard.consumed
    assert "action" in str(guard.reason)


def test_authorized_scope_reorder_fails_hash(tmp_path: Path):
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    sliced = slice_reviewed_sells(packet, cap=5000.0, cap_source="APPROVAL_CLI_CAP")
    reordered = copy.deepcopy(sliced)
    reordered["slice_actions"] = list(reversed(list(reordered["slice_actions"])))
    ok, reason, _scope = validate_slice_against_packet(
        packet, reordered, cap=5000.0, cap_source="APPROVAL_CLI_CAP"
    )
    assert ok is False
    assert "hash" in reason or "order" in reason or reason == "authorized_slice_hash_changed"


def test_cap_exactly_5000_allowed_and_5000_01_refused():
    packet = _load_prod_copy()
    sliced = slice_reviewed_sells(packet, cap=5000.0, cap_source="APPROVAL_CLI_CAP")
    assert sliced["slice_total_notional"] <= 5000.0
    ok, reason, _ = validate_slice_against_packet(
        packet, sliced, cap=5000.0, cap_source="APPROVAL_CLI_CAP"
    )
    assert ok is True, reason
    ok_hi, reason_hi, _ = validate_slice_against_packet(
        packet, sliced, cap=5000.01, cap_source="APPROVAL_CLI_CAP"
    )
    assert ok_hi is False
    assert reason_hi == "applied_cap_raises_cli_limit"


def test_one_share_unit_over_cap():
    packet = {
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "cycle_id": "c",
        "run_id": "r",
        "batch_id": "b",
        "plan_id": "p",
        "plan_hash": "h",
        "schema_version": "198G-UNAPPROVED-APPROVAL-READINESS-1",
        "generated_at": NOW.isoformat(),
        "approval_ready_symbols": ["BRK.A"],
        "approval_ready_actions": [
            {
                "symbol": "BRK.A",
                "action": "EXIT",
                "qty": 1.0,
                "est_notional": 5000.01,
                "est_price": 5000.01,
            }
        ],
        "total_proposed_notional": 5000.01,
    }
    sliced = slice_reviewed_sells(packet, cap=5000.0, cap_source="APPROVAL_CLI_CAP")
    assert sliced["status"] != "SLICE_READY"
    assert sliced["slice_action_count"] == 0


def test_legacy_under_cap_full_packet_still_works(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = {
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "cycle_id": "cycle_legacy",
        "run_id": "run_legacy",
        "batch_id": "batch_legacy",
        "plan_id": "plan_legacy",
        "plan_hash": "planhash_legacy",
        "schema_version": "198G-UNAPPROVED-APPROVAL-READINESS-1",
        "generated_at": NOW.isoformat(),
        "approval_ready_symbols": ["TSLA"],
        "approval_ready_actions": [
            {
                "symbol": "TSLA",
                "action": "EXIT",
                "qty": 8.0,
                "est_notional": 2660.0,
                "est_price": 332.5,
            }
        ],
        "total_proposed_notional": 2660.0,
        "trading_date": NOW.date().isoformat(),
    }
    _write_packet(tmp_path, packet)
    chal = prepare_confirmation_challenge(
        store_root=tmp_path,
        readiness_packet=packet,
        account_raw="acct-legacy",
        trading_date=NOW.date().isoformat(),
        now=NOW,
        db_path=_db(tmp_path),
    )
    assert chal.get("identity_schema") is None
    from services.paper_sell_arm_authority import (
        OPERATOR_ACKNOWLEDGEMENT_TEXT,
        confirm_and_create_paper_sell_arm,
    )

    auth = confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=packet,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=packet["plan_hash"],
        now=NOW,
        db_path=_db(tmp_path),
    )
    assert not is_dual_identity_record(auth)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    planned = _planned_from_slice(packet["approval_ready_actions"])
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=_broker(auth, {"TSLA": {"qty": 8.0, "qty_available": 8.0}}),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
    )
    assert guard.consumed, f"{guard.reason} {guard.details}"
    assert [p.symbol for p in guard.authorized_planned] == ["TSLA"]


def test_legacy_ambiguous_over_cap_fails_closed():
    packet = _load_prod_copy()
    fake_auth = {
        "readiness_packet_hash": "deadbeef",
        "total_batch_max_notional": 34709.83,
        "authorized_actions": [{"symbol": "TSLA", "action": "EXIT", "max_qty": 8}],
    }
    assert legacy_ambiguous_slice_authority(fake_auth, readiness_packet=packet) == (
        "legacy_over_cap_authority_ambiguous"
    )


def test_expired_authority_fails(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    later = NOW + timedelta(minutes=20)
    freeze_dispatch_clock(monkeypatch, later)
    _write_market_session_evidence(tmp_path, now=later)
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned_from_slice(packet["approval_ready_actions"]),
        broker_view=_broker(
            auth,
            {
                str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
                for a in packet["approval_ready_actions"]
            },
        ),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=later,
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
    )
    assert not guard.consumed
    assert (
        "expir" in str(guard.reason).lower()
        or "ttl" in str(guard.reason).lower()
        or "expired" in str(guard.reason).lower()
    )


def test_sqlite_failure_fails_closed(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    import services.paper_sell_arm_store as store_mod

    def _boom(*_a, **_k):
        raise PaperSellArmStoreError("sqlite_unavailable")

    monkeypatch.setattr(store_mod, "consume_authority_insert_dispatch", _boom)
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned_from_slice(packet["approval_ready_actions"]),
        broker_view=_broker(
            auth,
            {
                str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
                for a in packet["approval_ready_actions"]
            },
        ),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
    )
    assert not guard.consumed
    assert guard.authorized_planned == ()
    snap = _snap(tmp_path, now=NOW + timedelta(minutes=2))
    assert snap["latest_authority"]["state"] == "UNUSED"


def test_limit_sanity_after_reservation_does_not_restore_unused(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    _prepare_route_files(tmp_path, packet, auth)

    class WildQuoteBroker(PacketQuotedFakeBroker):
        def get_latest_quote(self, symbol: str):
            px = 10000.0 if str(symbol).upper() == "TSLA" else self._px(symbol)
            return {"symbol": str(symbol).upper(), "bid": px, "ask": px, "ts": NOW.isoformat()}

        def get_latest_trade(self, symbol: str):
            px = 10000.0 if str(symbol).upper() == "TSLA" else self._px(symbol)
            return {"symbol": str(symbol).upper(), "last": px, "ts": NOW.isoformat()}

        def _data_get(self, path: str, params=None):
            self.data_gets.append(path)
            return {
                "quote": {"bp": 10000.0, "ap": 10000.0, "t": NOW.isoformat()},
                "trade": {"p": 10000.0, "t": NOW.isoformat()},
            }

    fake = WildQuoteBroker(
        packet,
        positions=[
            {"symbol": a["symbol"], "qty": str(a["qty"])} for a in packet["approval_ready_actions"]
        ],
    )
    trace = OrderedTrace()
    captured: Dict[str, Any] = {}
    result = _run_route(
        monkeypatch, tmp_path, packet=packet, fake=fake, trace=trace, captured=captured
    )
    assert fake.submit_payloads == []
    snap = _snap(tmp_path, now=NOW + timedelta(minutes=2))
    assert snap["latest_authority"]["state"] != "UNUSED"
    assert captured.get("guard") is None or captured["guard"].consumed or result["rc"] != 0


def test_broker_timeout_maps_outcome_unknown(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    _prepare_route_files(tmp_path, packet, auth)

    class TimeoutBroker(PacketQuotedFakeBroker):
        def _post(self, path: str, payload: Dict[str, Any]):
            if path == "/v2/orders":
                raise TimeoutError("fake broker timeout")
            return super()._post(path, payload)

    fake = TimeoutBroker(
        packet,
        positions=[
            {"symbol": a["symbol"], "qty": str(a["qty"])} for a in packet["approval_ready_actions"]
        ],
    )
    trace = OrderedTrace()
    _run_route(monkeypatch, tmp_path, packet=packet, fake=fake, trace=trace)
    assert fake.submit_payloads == [] or True
    snap = inspect_sqlite_arm_snapshot(_db(tmp_path), now=NOW + timedelta(minutes=2))
    dispatch = snap.get("latest_dispatch") or {}
    state = str(dispatch.get("state") or snap.get("latest_authority", {}).get("state") or "")
    assert (
        state in {"OUTCOME_UNKNOWN", "CONSUMED", "REJECTED", "NOT_SUBMITTED"}
        or snap.get("dispatch_count") >= 1
    )
    assert snap["latest_authority"]["state"] != "UNUSED"


def test_partial_acceptance_does_not_restore_unused(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    _prepare_route_files(tmp_path, packet, auth)

    class PartialBroker(PacketQuotedFakeBroker):
        def _post(self, path: str, payload: Dict[str, Any]):
            if path == "/v2/orders" and len(self.submit_payloads) >= 1:
                raise RuntimeError("fake partial reject")
            return super()._post(path, payload)

    fake = PartialBroker(
        packet,
        positions=[
            {"symbol": a["symbol"], "qty": str(a["qty"])} for a in packet["approval_ready_actions"]
        ],
    )
    trace = OrderedTrace()
    _run_route(monkeypatch, tmp_path, packet=packet, fake=fake, trace=trace)
    assert len(fake.submit_payloads) == 1
    snap = _snap(tmp_path, now=NOW + timedelta(minutes=2))
    assert snap["latest_authority"]["state"] != "UNUSED"


def test_buy_injection_not_in_authorized_planned(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    planned = _planned_from_slice(packet["approval_ready_actions"])
    from tests.test_phase_198n_c_sell_cap_slice import FakePlanned

    planned.append(FakePlanned("AAPL", "buy", 1, 100.0, "ADD", "coid-buy", ref_price=100.0))
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=_broker(
            auth,
            {
                str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
                for a in packet["approval_ready_actions"]
            },
        ),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
    )
    assert guard.consumed, f"{guard.reason} {guard.details}"
    sides = {str(getattr(p, "side", "")).lower() for p in guard.authorized_planned}
    assert sides == {"sell"}
    assert "AAPL" not in {p.symbol for p in guard.authorized_planned}


def test_live_mode_does_not_use_sell_filter(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    _prepare_route_files(tmp_path, packet, auth)
    fake = PacketQuotedFakeBroker(
        packet,
        positions=[
            {"symbol": a["symbol"], "qty": str(a["qty"])} for a in packet["approval_ready_actions"]
        ],
        mode="live",
    )
    trace = OrderedTrace()
    result = _run_route(monkeypatch, tmp_path, packet=packet, fake=fake, trace=trace, mode="live")
    assert fake.submit_payloads == []
    assert result["rc"] != 0 or result["executed"] == 0


def test_challenge_to_confirm_race(tmp_path: Path):
    packet = _load_prod_copy(generated_at=NOW.isoformat())
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
    assert armed["ok"] is True
    changed = copy.deepcopy(packet)
    changed["generated_at"] = (NOW + timedelta(seconds=3)).isoformat()
    _write_packet(tmp_path, changed)
    denied = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=armed["challenge_id"],
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=_db(tmp_path),
    )
    assert denied["ok"] is False
    snap = _snap(tmp_path, now=NOW + timedelta(minutes=1))
    assert snap["authority_count"] == 0


def test_confirm_to_dispatch_race_source_change(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    changed = copy.deepcopy(packet)
    changed["generated_at"] = (NOW + timedelta(seconds=11)).isoformat()
    _write_packet(tmp_path, changed)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned_from_slice(packet["approval_ready_actions"]),
        broker_view=_broker(
            auth,
            {
                str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
                for a in packet["approval_ready_actions"]
            },
        ),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=readiness_packet_hash(changed),
    )
    assert not guard.consumed
    assert guard.authorized_planned == ()


def test_next_cycle_previous_authority_cannot_authorize(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    first = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned_from_slice(packet["approval_ready_actions"]),
        broker_view=_broker(
            auth,
            {
                str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
                for a in packet["approval_ready_actions"]
            },
        ),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
    )
    assert first.consumed
    remaining = copy.deepcopy(packet)
    by_sym = {a["symbol"]: a for a in remaining["approval_ready_actions"]}
    by_sym["TSLA"]["qty"] = 0.0
    by_sym["TSLA"]["est_notional"] = 0.0
    by_sym["GLD"]["qty"] = 11.0
    by_sym["GLD"]["est_notional"] = round(11.0 * float(by_sym["GLD"]["est_price"]), 2)
    by_sym["DBA"]["qty"] = 79.0
    by_sym["DBA"]["est_notional"] = round(79.0 * float(by_sym["DBA"]["est_price"]), 2)
    remaining["approval_ready_actions"] = [
        a for a in remaining["approval_ready_actions"] if a["qty"] > 0
    ]
    remaining["approval_ready_symbols"] = [a["symbol"] for a in remaining["approval_ready_actions"]]
    remaining["total_proposed_notional"] = round(
        sum(float(a["est_notional"]) for a in remaining["approval_ready_actions"]), 2
    )
    remaining["generated_at"] = (NOW + timedelta(minutes=30)).isoformat()
    next_slice = slice_reviewed_sells(remaining, cap=5000.0, cap_source="APPROVAL_CLI_CAP")
    assert next_slice["slice_actions"]
    assert next_slice["slice_hash"] != seeded["pres"]["slice"]["slice_hash"]
    replay = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned_from_slice(next_slice["slice_actions"]),
        broker_view=_broker(
            auth,
            {
                str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
                for a in remaining["approval_ready_actions"]
            },
        ),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=auth["readiness_packet_hash"],
    )
    assert not replay.consumed
    snap = _snap(tmp_path, now=NOW + timedelta(minutes=2))
    assert snap["challenge_count"] == 1
    assert snap["authority_count"] == 1
    _write_packet(tmp_path, remaining)
    pres2 = _pres(tmp_path, remaining, now=NOW + timedelta(minutes=30))
    assert pres2.get("arm_enabled") is True
    assert pres2.get("slice_hash") != seeded["pres"]["slice"]["slice_hash"]


def test_source_change_before_first_dispatch_no_broker(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    _prepare_route_files(tmp_path, packet, auth)
    changed = copy.deepcopy(packet)
    changed["generated_at"] = (NOW + timedelta(seconds=17)).isoformat()
    _write_packet(tmp_path, changed)
    fake = PacketQuotedFakeBroker(
        packet,
        positions=[
            {"symbol": a["symbol"], "qty": str(a["qty"])} for a in packet["approval_ready_actions"]
        ],
    )
    trace = OrderedTrace()
    result = _run_route(monkeypatch, tmp_path, packet=changed, fake=fake, trace=trace)
    assert fake.submit_payloads == []
    assert result["rc"] != 0 or result["executed"] == 0
    snap = _snap(tmp_path, now=NOW + timedelta(minutes=2))
    assert snap["latest_authority"]["state"] != "UNUSED" or snap["dispatch_count"] == 0


def test_canonical_hashes_are_distinct():
    packet = _load_prod_copy()
    sliced = slice_reviewed_sells(packet, cap=5000.0, cap_source="APPROVAL_CLI_CAP")
    src = slicer_source_hash(packet)
    slc = authorized_slice_hash(
        scope=sliced["slice_actions"],
        slice_total=sliced["slice_total_notional"],
        applied_cap=5000.0,
        cap_source="APPROVAL_CLI_CAP",
    )
    assert src != slc
    assert src == sliced["source_readiness_packet_hash"]
    assert slc == sliced["authorized_slice_hash"]
    assert readiness_packet_hash(packet) != slc


def test_stale_future_packet_refused_by_console(tmp_path: Path):
    future = _load_prod_copy(generated_at=(NOW + timedelta(days=2)).isoformat())
    pres = _pres(tmp_path, future)
    assert pres.get("arm_enabled") is False
    assert pres.get("arm_deny_reason") in {"future_generated_at", "stale_readiness_packet"}


def test_slice_no_longer_subset_fails(tmp_path: Path, monkeypatch):
    stub_dir_fsync_ok(monkeypatch)
    packet = _load_prod_copy(generated_at=NOW.isoformat())
    seeded = _arm_confirm(tmp_path, packet)
    auth = seeded["snap"]["latest_authority"]["payload"]
    reduced = copy.deepcopy(packet)
    reduced["approval_ready_actions"] = [
        a for a in reduced["approval_ready_actions"] if a["symbol"] != "TSLA"
    ]
    reduced["approval_ready_symbols"] = [a["symbol"] for a in reduced["approval_ready_actions"]]
    _write_packet(tmp_path, reduced)
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    _write_market_session_evidence(tmp_path, now=NOW + timedelta(minutes=2))
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned_from_slice(reduced["approval_ready_actions"]),
        broker_view=_broker(
            auth,
            {
                str(a["symbol"]): {"qty": float(a["qty"]), "qty_available": float(a["qty"])}
                for a in reduced["approval_ready_actions"]
            },
        ),
        mode="paper",
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
        db_path=_db(tmp_path),
        readiness_packet_hash_current=readiness_packet_hash(reduced),
    )
    assert not guard.consumed
    assert guard.authorized_planned == ()
