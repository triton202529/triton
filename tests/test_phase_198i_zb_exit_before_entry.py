"""Phase 198I-ZB — cycle-bound SELL readiness and exit-before-entry interlock.

PAPER ONLY. Isolated tmp_path. Fake broker only in the two-cycle proof.
No repository SQLite, no production weekend packet writes, no real network.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from services.defensive_exit_approval_readiness import (
    PACKET_NAME,
    generate_from_manage_artifacts,
    run_market_open_readiness_stage,
)
from services.manage_positions_cycle_binding import (
    BLOCKER_MISSING_SELL_AUTHORITY,
    BLOCKER_NONE,
    BLOCKER_OTHER_GATE,
    CLASS_ALREADY_SUBMITTED,
    CLASS_AUTHORITY_READY,
    CLASS_BLOCKED_OTHER_GATE,
    CLASS_OUTCOME_UNKNOWN,
    DECISION_BLOCK,
    DECISION_CANDIDATE_AWARE,
    DECISION_MAY_CONTINUE,
    F2_GENERATION_FAILURE,
    F2_OK,
    F2_OTHER_GATE,
    F2_OUTCOME_UNKNOWN,
    F2_PENDING_SELL_ARM,
    F2_SOURCE_UNBOUND,
    INVOCATION_ENV,
    REASON_INVOCATION_MISMATCH,
    REASON_INVOCATION_MISSING,
    REASON_PENDING_SELL_ARM,
    REASON_SELL_OUTCOME_UNKNOWN,
    REASON_SELL_PRECONDITION,
    REASON_SELL_READINESS_UNKNOWN,
    evaluate_exit_before_entry,
    load_cycle_binding,
    write_cycle_binding,
    write_entry_interlock,
)
from services.paper_sell_arm_console import (
    CONFIRM_PAPER_SELLS_PHRASE,
    arm_reviewed_sells,
    build_presentation,
    confirm_and_arm_paper_sells,
)
from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT
from services.paper_sell_arm_store import db_path_for_store_root, inspect_sqlite_arm_snapshot
from tests._paper_sell_test_durability import stub_dir_fsync_ok
from tests._phase_198i_sell_fakes import SESSION_OPEN_NOW, block_real_network

ROOT = Path(__file__).resolve().parents[1]
NOW = SESSION_OPEN_NOW
REPO_SQLITE = ROOT / "data" / "live" / "paper_sell_arm.sqlite3"
PROD_PACKET = ROOT / "data" / "results" / "defensive_exit_approval_readiness.json"


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    stub_dir_fsync_ok(monkeypatch)
    block_real_network(monkeypatch)
    for key in list(os.environ):
        if "ALPACA" in key.upper():
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)
    assert not REPO_SQLITE.is_file()
    weekend = json.loads(PROD_PACKET.read_text(encoding="utf-8"))
    assert "generated_at" not in weekend


def _write_plan(
    results: Path,
    *,
    rows=None,
    ts=None,
):
    ts = ts or NOW.isoformat()
    results.mkdir(parents=True, exist_ok=True)
    rows = rows or [
        {
            "symbol": "TSLA",
            "final_action": "EXIT",
            "qty": 1.0,
            "selected_for_execution": "true",
            "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
            "skip_reason": "",
            "current_position_qty": 1.0,
            "current_price": 332.5,
            "planned_notional": 332.5,
        }
    ]
    (results / "manage_positions_plan.json").write_text(
        json.dumps({"timestamp": ts, "orders_executed": 0, "blocked": True}),
        encoding="utf-8",
    )
    fieldnames = [
        "symbol",
        "final_action",
        "qty",
        "selected_for_execution",
        "reason_code",
        "skip_reason",
        "current_position_qty",
        "current_price",
        "planned_notional",
    ]
    with (results / "manage_positions_plan.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return ts


def _bind(
    results: Path,
    monkeypatch,
    *,
    invocation_id: str = "zb-inv-1",
    remaining: str = BLOCKER_MISSING_SELL_AUTHORITY,
    op_req: bool = True,
    block_reasons=None,
    submitted_symbols=None,
    submitted_outcomes=None,
    outcome_unknown: bool = False,
    orders_executed: int = 0,
    now=NOW,
):
    monkeypatch.setenv(INVOCATION_ENV, invocation_id)
    plan = json.loads((results / "manage_positions_plan.json").read_text(encoding="utf-8"))
    write_cycle_binding(
        results_root=results,
        invocation_id=invocation_id,
        plan_generated_at=str(plan["timestamp"]),
        f_stage_status="COMPLETE",
        now=now,
        remaining_blocker=remaining,
        operator_execution_required=op_req,
        block_reasons=(
            block_reasons if block_reasons is not None else ["DISPATCH_BLOCKED_APPROVAL_REQUIRED"]
        ),
        submitted_symbols=submitted_symbols,
        submitted_outcomes=submitted_outcomes,
        outcome_unknown=outcome_unknown,
        orders_executed=orders_executed,
    )


def _entry_decision(monkeypatch, results: Path, now=NOW):
    clock = now
    real = evaluate_exit_before_entry

    def _wrapped(*, results_root, now=None, ttl_minutes=30, expected_invocation_id=None):
        return real(
            results_root=results_root,
            now=now or clock,
            ttl_minutes=ttl_minutes,
            expected_invocation_id=expected_invocation_id,
        )

    monkeypatch.setattr(
        "services.manage_positions_cycle_binding.evaluate_exit_before_entry",
        _wrapped,
    )
    monkeypatch.setattr("services.paper_entry_loop_stage.RESULTS", results)
    from services.paper_entry_loop_stage import evaluate_paper_entry_gates

    ok, reason, details = evaluate_paper_entry_gates(verbose=False)
    return ok, reason, details


def test_a_current_cycle_valid_empty_allows_existing_entry_gates(tmp_path, monkeypatch):
    _write_plan(
        tmp_path,
        rows=[
            {
                "symbol": "TSLA",
                "final_action": "HOLD",
                "qty": 1.0,
                "selected_for_execution": "false",
                "reason_code": "",
                "skip_reason": "",
                "current_position_qty": 1.0,
                "current_price": 332.5,
                "planned_notional": 0,
            }
        ],
    )
    _bind(
        tmp_path,
        monkeypatch,
        remaining=BLOCKER_NONE,
        op_req=False,
        block_reasons=[],
    )
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert result["ok"] is True
    assert result["exit_code"] == F2_OK
    assert result["packet_status"] == "VALID_EMPTY"
    assert result["packet"]["approval_ready_actions"] == []
    assert result["packet"]["cycle_id"] == "zb-inv-1"
    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    assert pres["freshness"] == "EMPTY"
    assert pres["arm_enabled"] is False
    interlock = evaluate_exit_before_entry(results_root=tmp_path, now=NOW)
    assert interlock["decision"] == DECISION_MAY_CONTINUE
    ok, reason, details = _entry_decision(monkeypatch, tmp_path)
    assert details.get("failed_check") != "exit_before_entry"
    assert reason != REASON_PENDING_SELL_ARM
    snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
    assert snap["authority_count"] == 0
    assert result["broker"] == 0


def test_b_authority_ready_blocks_entry_and_enables_arm(tmp_path, monkeypatch):
    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch)
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert result["exit_code"] == F2_PENDING_SELL_ARM
    assert result["packet"]["approval_ready_actions"][0]["symbol"] == "TSLA"
    assert result["packet"]["approval_ready_actions"][0]["qty"] == 1.0
    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    assert pres["freshness"] == "FRESH"
    assert pres["arm_enabled"] is True
    interlock = evaluate_exit_before_entry(results_root=tmp_path, now=NOW)
    binding = load_cycle_binding(tmp_path)
    packet = result["packet"]
    assert binding["invocation_id"] == packet["cycle_id"] == interlock["invocation_id"]
    assert (
        binding["market_open_invocation_id"]
        == packet["market_open_invocation_id"]
        == interlock["market_open_invocation_id"]
    )
    assert (
        binding["source_plan_sha256"]
        == packet["source_plan_sha256"]
        == interlock["source_plan_sha256"]
    )
    assert (
        binding["selected_row_hash"]
        == packet["selected_row_hash"]
        == interlock["selected_row_hash"]
    )
    assert binding["plan_generated_at"] == packet["plan_generated_at"]
    assert interlock["decision"] == DECISION_CANDIDATE_AWARE
    assert interlock["interlock_scope"] == "CANDIDATE_FILTER"
    assert interlock["proceeds_available_for_entry"] == 0
    ok, reason, details = _entry_decision(monkeypatch, tmp_path)
    assert details.get("failed_check") != "exit_before_entry"
    assert reason != REASON_PENDING_SELL_ARM
    assert result["broker"] == 0


def test_c_stale_plan_new_invocation_refuses_restamp(tmp_path, monkeypatch):
    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch, invocation_id="zb-old")
    first = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert first["ok"] is True
    before = (tmp_path / PACKET_NAME).read_bytes()
    monkeypatch.setenv(INVOCATION_ENV, "zb-new")
    second = generate_from_manage_artifacts(
        results_root=tmp_path,
        now=NOW + timedelta(minutes=5),
        expected_invocation_id="zb-new",
    )
    assert second["ok"] is False
    assert second["wrote"] is False
    assert second["exit_code"] == F2_SOURCE_UNBOUND
    assert (tmp_path / PACKET_NAME).read_bytes() == before
    assert json.loads(before)["generated_at"] == NOW.isoformat()
    rc = run_market_open_readiness_stage(
        results_root=tmp_path,
        now=NOW + timedelta(minutes=5),
        expected_invocation_id="zb-new",
    )
    assert rc == F2_SOURCE_UNBOUND
    ok, reason, _ = _entry_decision(monkeypatch, tmp_path, now=NOW + timedelta(minutes=5))
    assert ok is False
    assert reason == REASON_SELL_READINESS_UNKNOWN


def test_d_cycle_id_mismatch_not_fresh(tmp_path, monkeypatch):
    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch, invocation_id="cycle-a")
    generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    before = (tmp_path / PACKET_NAME).read_bytes()
    failed = generate_from_manage_artifacts(
        results_root=tmp_path,
        now=NOW + timedelta(minutes=1),
        expected_invocation_id="cycle-b",
    )
    assert failed["ok"] is False
    assert failed["reason"] == "cycle_id_mismatch"
    assert failed["wrote"] is False
    assert (tmp_path / PACKET_NAME).read_bytes() == before


def test_e_source_plan_hash_mismatch_not_fresh(tmp_path, monkeypatch):
    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch)
    first = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert first["ok"] is True
    before = (tmp_path / PACKET_NAME).read_bytes()
    _write_plan(
        tmp_path,
        rows=[
            {
                "symbol": "TSLA",
                "final_action": "EXIT",
                "qty": 9.0,
                "selected_for_execution": "true",
                "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
                "skip_reason": "",
                "current_position_qty": 9.0,
                "current_price": 332.5,
                "planned_notional": 2992.5,
            }
        ],
    )
    failed = generate_from_manage_artifacts(results_root=tmp_path, now=NOW + timedelta(minutes=1))
    assert failed["ok"] is False
    assert failed["reason"] in {"source_plan_hash_mismatch", "selected_row_hash_mismatch"}
    assert failed["wrote"] is False
    assert (tmp_path / PACKET_NAME).read_bytes() == before


def test_f_submitted_sell_excluded_entry_may_continue(tmp_path, monkeypatch):
    _write_plan(tmp_path)
    _bind(
        tmp_path,
        monkeypatch,
        remaining=BLOCKER_NONE,
        op_req=False,
        block_reasons=[],
        submitted_symbols=["TSLA"],
        submitted_outcomes={"TSLA": "SUBMITTED"},
        orders_executed=1,
    )
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert result["ok"] is True
    assert result["exit_code"] == F2_OK
    assert result["packet"]["approval_ready_actions"] == []
    classes = {c["symbol"]: c["classification"] for c in result["classifications"]}
    assert classes["TSLA"] == CLASS_ALREADY_SUBMITTED
    interlock = evaluate_exit_before_entry(results_root=tmp_path, now=NOW)
    assert interlock["decision"] == DECISION_MAY_CONTINUE


def test_g_outcome_unknown_not_rearmable_blocks_entry(tmp_path, monkeypatch):
    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch, outcome_unknown=True, remaining=BLOCKER_MISSING_SELL_AUTHORITY)
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert result["ok"] is False
    assert result["wrote"] is False
    assert result["exit_code"] == F2_OUTCOME_UNKNOWN
    assert all(c.get("classification") != CLASS_AUTHORITY_READY for c in result["classifications"])
    assert any(c.get("classification") == CLASS_OUTCOME_UNKNOWN for c in result["classifications"])
    rc = run_market_open_readiness_stage(results_root=tmp_path, now=NOW)
    assert rc == F2_OUTCOME_UNKNOWN
    ok, reason, details = _entry_decision(monkeypatch, tmp_path)
    assert ok is False
    assert reason == REASON_SELL_OUTCOME_UNKNOWN
    assert details["failed_check"] == "exit_before_entry"


def test_h_other_gate_not_authority_ready(tmp_path, monkeypatch):
    _write_plan(tmp_path)
    _bind(
        tmp_path,
        monkeypatch,
        remaining=BLOCKER_OTHER_GATE,
        op_req=False,
        block_reasons=["ORDER_GUARD_DENIED"],
    )
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert result["ok"] is True
    assert result["exit_code"] == F2_OTHER_GATE
    assert result["packet"]["approval_ready_actions"] == []
    assert result["classifications"][0]["classification"] == CLASS_BLOCKED_OTHER_GATE
    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    assert pres["freshness"] == "BLOCKED_OTHER_GATE"
    assert pres["arm_enabled"] is False
    ok, reason, _ = _entry_decision(monkeypatch, tmp_path)
    assert ok is False
    assert reason == REASON_SELL_PRECONDITION


def test_i_missing_artifacts_block_entry_no_restamp(tmp_path, monkeypatch):
    prior = {"schema_version": "198I-ZA-SELL-READINESS-1", "generated_at": NOW.isoformat()}
    (tmp_path / PACKET_NAME).write_text(json.dumps(prior), encoding="utf-8")
    before = (tmp_path / PACKET_NAME).read_bytes()
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW + timedelta(minutes=2))
    assert result["ok"] is False
    assert result["exit_code"] in {F2_SOURCE_UNBOUND, F2_GENERATION_FAILURE}
    assert (tmp_path / PACKET_NAME).read_bytes() == before
    rc = run_market_open_readiness_stage(results_root=tmp_path, now=NOW + timedelta(minutes=2))
    assert rc in {F2_SOURCE_UNBOUND, F2_GENERATION_FAILURE}
    monkeypatch.setenv(INVOCATION_ENV, "zb-inv-missing-artifacts")
    ok, reason, _ = _entry_decision(monkeypatch, tmp_path)
    assert ok is False
    assert reason in {REASON_SELL_READINESS_UNKNOWN, REASON_SELL_PRECONDITION}


def test_j_mixed_submitted_and_authority_ready(tmp_path, monkeypatch):
    _write_plan(
        tmp_path,
        rows=[
            {
                "symbol": "TSLA",
                "final_action": "EXIT",
                "qty": 1.0,
                "selected_for_execution": "true",
                "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
                "skip_reason": "",
                "current_position_qty": 1.0,
                "current_price": 332.5,
                "planned_notional": 332.5,
            },
            {
                "symbol": "MSFT",
                "final_action": "TRIM",
                "qty": 2.0,
                "selected_for_execution": "true",
                "reason_code": "TRIM",
                "skip_reason": "",
                "current_position_qty": 4.0,
                "current_price": 400.0,
                "planned_notional": 800.0,
            },
        ],
    )
    _bind(
        tmp_path,
        monkeypatch,
        submitted_symbols=["TSLA"],
        submitted_outcomes={"TSLA": "SUBMITTED"},
        remaining=BLOCKER_MISSING_SELL_AUTHORITY,
        orders_executed=1,
    )
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    symbols = [a["symbol"] for a in result["packet"]["approval_ready_actions"]]
    assert symbols == ["MSFT"]
    classes = {c["symbol"]: c["classification"] for c in result["classifications"]}
    assert classes["TSLA"] == CLASS_ALREADY_SUBMITTED
    assert classes["MSFT"] == CLASS_AUTHORITY_READY
    assert result["exit_code"] == F2_PENDING_SELL_ARM
    interlock = evaluate_exit_before_entry(results_root=tmp_path, now=NOW)
    assert interlock["decision"] == DECISION_CANDIDATE_AWARE
    ok, reason, details = _entry_decision(monkeypatch, tmp_path)
    assert details.get("failed_check") != "exit_before_entry"
    assert reason != REASON_PENDING_SELL_ARM


def test_ui_source_unbound_and_outcome_unknown_labels(tmp_path):
    from services.defensive_exit_approval_readiness import build_canonical_packet

    unbound = build_canonical_packet(
        now=NOW,
        actions=[{"symbol": "TSLA", "action": "EXIT", "qty": 1.0, "est_price": 1.0}],
        cycle_id="x",
        run_id="x",
        batch_id="x",
        plan_id="x",
        plan_hash="x",
        extra={"packet_status": "SOURCE_UNBOUND"},
    )
    (tmp_path / PACKET_NAME).write_text(json.dumps(unbound), encoding="utf-8")
    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    assert pres["freshness"] == "SOURCE_UNBOUND"
    assert pres["arm_enabled"] is False


def test_two_cycle_fake_broker_exit_before_entry(tmp_path, monkeypatch):
    from tests._phase_198i_sell_fakes import (
        LIVE_BLOCKED,
        NetworkFakeAlpacaBroker,
        OrderedTrace,
        freeze_dispatch_clock,
        inprocess_place_live_orders,
        install_boundary_trace,
        isolate_sell_store,
        stub_unrelated_pre_gates,
    )
    from services.manage_positions import ManagementPlannedOrder, maybe_execute_management
    from services.paper_sell_arm_authority import readiness_packet_hash

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    evidence = ROOT / "tmp" / f"phase_198i_zb_exit_before_entry_{stamp}"
    evidence.mkdir(parents=True, exist_ok=True)

    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch, invocation_id="zb-cycle-1")
    cycle1 = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert cycle1["exit_code"] == F2_PENDING_SELL_ARM
    interlock1 = evaluate_exit_before_entry(results_root=tmp_path, now=NOW)
    assert interlock1["decision"] == DECISION_CANDIDATE_AWARE
    assert interlock1["proceeds_available_for_entry"] == 0
    from services.paper_sell_entry_dependency import (
        ALLOW_EXISTING_GATES,
        BLOCK_SAME_SYMBOL_PENDING_SELL,
        filter_entry_candidates,
        context_from_interlock,
    )

    ctx = context_from_interlock(interlock1)
    allowed, decisions = filter_entry_candidates(
        [
            {
                "ticker": "MSFT",
                "allocation_action": "OPEN_NEW",
                "est_notional": 500.0,
            },
            {
                "ticker": "TSLA",
                "allocation_action": "ADD",
                "est_notional": 200.0,
            },
        ],
        pending_sells=ctx["pending_sells"],
        current_cash=10000.0,
    )
    by_sym = {d["symbol"]: d["decision"] for d in decisions}
    assert by_sym["MSFT"] == ALLOW_EXISTING_GATES
    assert by_sym["TSLA"] == BLOCK_SAME_SYMBOL_PENDING_SELL
    assert [c["ticker"] for c in allowed] == ["MSFT"]
    ok1, reason1, details1 = _entry_decision(monkeypatch, tmp_path)
    assert details1.get("failed_check") != "exit_before_entry"
    assert reason1 != REASON_PENDING_SELL_ARM

    db_path = tmp_path / "paper_sell_arm.sqlite3"
    armed = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=db_path,
    )
    assert armed["ok"] is True
    assert armed["counts"]["broker"] == 0
    snap1 = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
    assert snap1["open_challenges"][0]["state"] == "OPEN"
    assert snap1["authority_count"] == 0
    assert snap1["dispatch_count"] == 0

    confirmed = confirm_and_arm_paper_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        challenge_id=armed["challenge_id"],
        now=NOW + timedelta(minutes=1),
        paper_mode=True,
        live_blocked=True,
        db_path=db_path,
    )
    assert confirmed["ok"] is True
    snap2 = inspect_sqlite_arm_snapshot(
        db_path_for_store_root(tmp_path), now=NOW + timedelta(minutes=1)
    )
    assert snap2["latest_authority"]["state"] == "UNUSED"
    assert snap2["dispatch_count"] == 0
    assert _ACTIVE_PERMIT.get() is None

    packet = cycle1["packet"]
    auth = snap2["latest_authority"]
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
    (tmp_path / "independent_dispatch_bindings.json").write_text(
        json.dumps(
            {
                "plan_hash": packet["plan_hash"],
                "cycle_id": packet["cycle_id"],
                "run_id": packet["run_id"],
                "batch_id": packet["batch_id"],
                "plan_id": packet["plan_id"],
                "trading_date": NOW.date().isoformat(),
                "account_fingerprint": auth.get("account_fingerprint"),
                "readiness_packet_hash": readiness_packet_hash(packet),
                "confirmation_challenge_id": auth.get("confirmation_challenge_id"),
                "confirmation_challenge_hash": auth.get("confirmation_challenge_hash"),
                "paper_execution_authorized": True,
                "market_session": "MARKET_OPEN",
                "market_session_observed_at": NOW.isoformat(),
                "lifecycle_actions": {"TSLA": "EXIT"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "positions_snapshot.csv").write_text(
        "symbol,qty,qty_available\nTSLA,1,1\n", encoding="utf-8"
    )
    (tmp_path / "open_orders_snapshot.csv").write_text("symbol,side,qty\n", encoding="utf-8")

    fake = NetworkFakeAlpacaBroker(
        positions=[{"symbol": "TSLA", "qty": "1", "qty_available": "1", "avg_entry_price": "200"}]
    )
    trace = OrderedTrace()
    freeze_dispatch_clock(monkeypatch, NOW + timedelta(minutes=2))
    isolate_sell_store(monkeypatch, tmp_path)
    stub_unrelated_pre_gates(monkeypatch)
    install_boundary_trace(monkeypatch, trace)

    import services.place_live_orders as plo
    import services.manage_positions as mp

    dispatch_now = NOW + timedelta(minutes=2)
    real_eac = plo.enforce_and_consume_paper_sells_before_submit

    def _eac(**kwargs):
        kwargs.setdefault("now", dispatch_now)
        kwargs.setdefault("live_override", LIVE_BLOCKED)
        kwargs.setdefault("results_root", tmp_path)
        kwargs.setdefault("live_root", tmp_path)
        return real_eac(**kwargs)

    monkeypatch.setattr(plo, "enforce_and_consume_paper_sells_before_submit", _eac)
    monkeypatch.setattr(plo, "make_broker", lambda mode: fake)
    monkeypatch.setattr(mp.subprocess, "run", inprocess_place_live_orders)

    planned = [
        ManagementPlannedOrder(
            symbol="TSLA",
            side="sell",
            qty=1,
            order_type="limit",
            time_in_force="day",
            limit_price=332.5,
            management_action="EXIT",
            rationale="zb-two-cycle",
            confidence=0.9,
            delta_pct=-0.01,
            current_position_qty=1.0,
            mode="paper",
            generated_at=NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
            planned_notional=332.5,
        )
    ]
    rc, executed, reasons, session_id, meta = maybe_execute_management(
        planned,
        "paper",
        False,
        {
            "execute_via_existing_placement_flow": True,
            "require_master_gate_for_execute": False,
            "require_execution_gate_for_execute": False,
            "require_guard_validation_for_each_order": False,
            "write_manage_orders_csv": True,
            "max_batch_notional": 50000.0,
        },
        fake,
        True,
        True,
        tmp_path / "manage_orders.csv",
        placement_session=f"zb_{tmp_path.name}",
    )
    sell_payloads = [p for p in fake.submit_payloads if str(p.get("side") or "").lower() == "sell"]
    buy_payloads = [
        p for p in fake.submit_payloads if str(p.get("side") or "").lower() in {"buy", "add"}
    ]
    assert rc == 0
    assert len(sell_payloads) == 1
    assert float(sell_payloads[0]["qty"]) == 1
    assert buy_payloads == []
    assert fake.fake_broker_call_count >= 1

    rc2, executed2, _, _, _ = maybe_execute_management(
        planned,
        "paper",
        False,
        {
            "execute_via_existing_placement_flow": True,
            "require_master_gate_for_execute": False,
            "require_execution_gate_for_execute": False,
            "require_guard_validation_for_each_order": False,
            "write_manage_orders_csv": True,
            "max_batch_notional": 50000.0,
        },
        fake,
        True,
        True,
        tmp_path / "manage_orders.csv",
        placement_session=f"zb_replay_{tmp_path.name}",
    )
    sell_after = [p for p in fake.submit_payloads if str(p.get("side") or "").lower() == "sell"]
    assert len(sell_after) == 1

    _write_plan(tmp_path)
    _bind(
        tmp_path,
        monkeypatch,
        invocation_id="zb-cycle-2",
        remaining=BLOCKER_NONE,
        op_req=False,
        block_reasons=[],
        submitted_symbols=["TSLA"],
        submitted_outcomes={"TSLA": "SUBMITTED"},
        orders_executed=1,
        now=NOW + timedelta(minutes=3),
    )
    cycle2 = generate_from_manage_artifacts(results_root=tmp_path, now=NOW + timedelta(minutes=3))
    assert cycle2["exit_code"] == F2_OK
    assert cycle2["packet"]["approval_ready_actions"] == []
    ok2, reason2, details2 = _entry_decision(monkeypatch, tmp_path, now=NOW + timedelta(minutes=3))
    assert details2.get("failed_check") != "exit_before_entry" or ok2 is True
    assert reason2 != REASON_PENDING_SELL_ARM
    interlock2 = evaluate_exit_before_entry(results_root=tmp_path, now=NOW + timedelta(minutes=3))
    assert interlock2["decision"] == DECISION_MAY_CONTINUE

    snap_final = inspect_sqlite_arm_snapshot(
        db_path_for_store_root(tmp_path), now=NOW + timedelta(minutes=3)
    )
    assert snap_final["latest_authority"]["state"] == "CONSUMED"

    proof = {
        "accepted_fake_sell": len(sell_payloads),
        "duplicate_fake_sell": len(sell_after) - 1,
        "rejected_path_fake_sell": 0,
        "buy_add_before_resolution": 0,
        "real_broker": 0,
        "network": 0,
        "authority_final": snap_final["latest_authority"]["state"],
        "cycle1_entry_failed_check": details1.get("failed_check"),
        "cycle1_candidate_decisions": by_sym,
        "cycle1_allowed": [c["ticker"] for c in allowed],
        "cycle1_interlock_scope": interlock1.get("interlock_scope"),
        "cycle2_interlock": interlock2,
        "cycle2_entry_failed_check": details2.get("failed_check"),
        "sell_qty": sell_payloads[0].get("qty") if sell_payloads else None,
        "fake_broker_base": fake.base,
        "session": fake.session,
    }
    (evidence / "two_cycle_proof.json").write_text(
        json.dumps(proof, indent=2, default=str), encoding="utf-8"
    )
    assert proof["accepted_fake_sell"] == 1
    assert proof["duplicate_fake_sell"] == 0
    assert proof["buy_add_before_resolution"] == 0
    assert fake.session is None
    assert str(fake.base).startswith("http://phase198i-fake-broker.test")


def _entry_cand(symbol, **kwargs):
    row = {
        "ticker": symbol,
        "allocation_action": "OPEN_NEW",
        "est_notional": 500.0,
    }
    row.update(kwargs)
    return row


def _tsla_pending(**kwargs):
    row = {"symbol": "TSLA", "action": "EXIT", "qty": 1, "reason_code": "EXIT_HIGH_DRAG_PROTECTION"}
    row.update(kwargs)
    return row


def test_dep_a_independent_msft_with_ordinary_tsla_exit():
    from services.paper_sell_entry_dependency import (
        ALLOW_EXISTING_GATES,
        BLOCK_SAME_SYMBOL_PENDING_SELL,
        evaluate_entry_candidate,
    )

    pending = [_tsla_pending()]
    msft = evaluate_entry_candidate(
        _entry_cand("MSFT", est_notional=800.0),
        pending_sells=pending,
        current_cash=10000.0,
    )
    tsla_buy = evaluate_entry_candidate(
        _entry_cand("TSLA", allocation_action="ADD", est_notional=200.0),
        pending_sells=pending,
        current_cash=10000.0,
    )
    assert msft["decision"] == ALLOW_EXISTING_GATES
    assert msft["proceeds_credited"] == 0.0
    assert tsla_buy["decision"] == BLOCK_SAME_SYMBOL_PENDING_SELL


def test_dep_b_same_symbol_buy_and_add_blocked():
    from services.paper_sell_entry_dependency import (
        BLOCK_SAME_SYMBOL_PENDING_SELL,
        evaluate_entry_candidate,
    )

    pending = [_tsla_pending()]
    for action in ("OPEN_NEW", "ADD", "BUY"):
        rec = evaluate_entry_candidate(
            _entry_cand("TSLA", allocation_action=action),
            pending_sells=pending,
            current_cash=10000.0,
        )
        assert rec["decision"] == BLOCK_SAME_SYMBOL_PENDING_SELL


def test_dep_c_sale_funded_rotation_blocked():
    from services.paper_sell_entry_dependency import (
        BLOCK_DEPENDS_ON_SELL_PROCEEDS,
        evaluate_entry_candidate,
    )

    rec = evaluate_entry_candidate(
        _entry_cand(
            "MSFT", est_notional=5000.0, requires_sell_proceeds=True, funding_source="SELL_PROCEEDS"
        ),
        pending_sells=[_tsla_pending(planned_notional=4000.0)],
        current_cash=200.0,
    )
    assert rec["decision"] == BLOCK_DEPENDS_ON_SELL_PROCEEDS
    assert rec["proceeds_credited"] == 0.0


def test_dep_d_independent_cash_allows_unrelated_buy():
    from services.paper_sell_entry_dependency import ALLOW_EXISTING_GATES, evaluate_entry_candidate

    rec = evaluate_entry_candidate(
        _entry_cand("MSFT", est_notional=400.0),
        pending_sells=[_tsla_pending()],
        current_cash=5000.0,
    )
    assert rec["decision"] == ALLOW_EXISTING_GATES
    assert rec["proceeds_credited"] == 0.0


def test_dep_e_sector_concentration_conflict():
    from services.paper_sell_entry_dependency import (
        BLOCK_WORSENS_PENDING_RISK_REDUCTION,
        evaluate_entry_candidate,
    )

    rec = evaluate_entry_candidate(
        _entry_cand("NVDA", est_notional=300.0),
        pending_sells=[{"symbol": "MSFT", "action": "EXIT", "qty": 2, "reason_code": "TRIM"}],
        current_cash=10000.0,
    )
    assert rec["decision"] == BLOCK_WORSENS_PENDING_RISK_REDUCTION


def test_dep_f_unrelated_sector_healthcare_allowed():
    from services.paper_sell_entry_dependency import ALLOW_EXISTING_GATES, evaluate_entry_candidate

    rec = evaluate_entry_candidate(
        _entry_cand("UNH", est_notional=400.0),
        pending_sells=[{"symbol": "MSFT", "action": "EXIT", "qty": 2, "reason_code": "TRIM"}],
        current_cash=10000.0,
    )
    assert rec["decision"] == ALLOW_EXISTING_GATES


def test_dep_g_force_exit_global_block(tmp_path, monkeypatch):
    _write_plan(
        tmp_path,
        rows=[
            {
                "symbol": "TSLA",
                "final_action": "EXIT",
                "qty": 1.0,
                "selected_for_execution": "true",
                "reason_code": "FORCE_EXIT",
                "skip_reason": "",
                "current_position_qty": 1.0,
                "current_price": 332.5,
                "planned_notional": 332.5,
            }
        ],
    )
    _bind(tmp_path, monkeypatch)
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert result["exit_code"] == F2_PENDING_SELL_ARM
    interlock = evaluate_exit_before_entry(results_root=tmp_path, now=NOW)
    assert interlock["decision"] == DECISION_BLOCK
    assert interlock["interlock_scope"] == "GLOBAL_SAFETY_BLOCK"
    ok, reason, details = _entry_decision(monkeypatch, tmp_path)
    assert ok is False
    assert details["failed_check"] == "exit_before_entry"
    from services.paper_sell_entry_dependency import (
        BLOCK_CRITICAL_EXIT_PENDING,
        evaluate_entry_candidate,
        context_from_interlock,
    )

    ctx = context_from_interlock(interlock)
    rec = evaluate_entry_candidate(
        _entry_cand("UNH"),
        pending_sells=ctx["pending_sells"],
        current_cash=10000.0,
        global_scope=ctx["scope"],
        global_reason=ctx["global_reason"],
    )
    assert rec["decision"] == BLOCK_CRITICAL_EXIT_PENDING


def test_dep_h_outcome_unknown_global(tmp_path, monkeypatch):
    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch, outcome_unknown=True, remaining=BLOCKER_MISSING_SELL_AUTHORITY)
    run_market_open_readiness_stage(results_root=tmp_path, now=NOW)
    ok, reason, details = _entry_decision(monkeypatch, tmp_path)
    assert ok is False
    assert reason == REASON_SELL_OUTCOME_UNKNOWN
    assert details["failed_check"] == "exit_before_entry"


def test_dep_i_missing_readiness_global(tmp_path, monkeypatch):
    monkeypatch.setenv(INVOCATION_ENV, "zb-inv-missing-file")
    ok, reason, details = _entry_decision(monkeypatch, tmp_path)
    assert ok is False
    assert reason == REASON_SELL_READINESS_UNKNOWN
    assert details["failed_check"] == "exit_before_entry"


def test_dep_j_mixed_entry_batch():
    from services.paper_sell_entry_dependency import (
        ALLOW_EXISTING_GATES,
        BLOCK_DEPENDS_ON_SELL_PROCEEDS,
        BLOCK_SAME_SYMBOL_PENDING_SELL,
        filter_entry_candidates,
    )

    allowed, decisions = filter_entry_candidates(
        [
            _entry_cand("UNH", est_notional=400.0),
            _entry_cand("TSLA", allocation_action="ADD", est_notional=100.0),
            _entry_cand("NVDA", est_notional=300.0),
            _entry_cand("MSFT", est_notional=9000.0),
        ],
        pending_sells=[{"symbol": "TSLA", "action": "EXIT", "qty": 1, "reason_code": "TRIM"}],
        current_cash=1000.0,
    )
    by_sym = {d["symbol"]: d["decision"] for d in decisions}
    assert by_sym["UNH"] == ALLOW_EXISTING_GATES
    assert by_sym["NVDA"] == ALLOW_EXISTING_GATES
    assert by_sym["TSLA"] == BLOCK_SAME_SYMBOL_PENDING_SELL
    assert by_sym["MSFT"] == BLOCK_DEPENDS_ON_SELL_PROCEEDS
    assert [c["ticker"] for c in allowed] == ["UNH", "NVDA"]
    assert all(
        d["proceeds_credited"] == 0.0 or d.get("proceeds_credited") in (0, 0.0, None)
        for d in decisions
    )


def test_dep_k_after_sell_resolved_existing_gates(tmp_path, monkeypatch):
    from services.paper_sell_entry_dependency import ALLOW_EXISTING_GATES, evaluate_entry_candidate

    _write_plan(tmp_path)
    _bind(
        tmp_path,
        monkeypatch,
        remaining=BLOCKER_NONE,
        op_req=False,
        block_reasons=[],
        submitted_symbols=["TSLA"],
        submitted_outcomes={"TSLA": "SUBMITTED"},
        orders_executed=1,
    )
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert result["packet_status"] == "VALID_EMPTY"
    interlock = evaluate_exit_before_entry(results_root=tmp_path, now=NOW)
    assert interlock["decision"] == DECISION_MAY_CONTINUE
    rec = evaluate_entry_candidate(
        _entry_cand("TSLA", allocation_action="ADD"),
        pending_sells=interlock.get("pending_sell_actions") or [],
        current_cash=10000.0,
    )
    assert rec["decision"] == ALLOW_EXISTING_GATES
    ok, reason, details = _entry_decision(monkeypatch, tmp_path)
    assert details.get("failed_check") != "exit_before_entry"
    assert reason != REASON_PENDING_SELL_ARM


def test_inv_a_current_invocation_candidate_aware_filters(tmp_path, monkeypatch):
    from services.paper_sell_entry_dependency import (
        ALLOW_EXISTING_GATES,
        BLOCK_SAME_SYMBOL_PENDING_SELL,
        context_from_interlock,
        evaluate_entry_candidate,
    )

    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch, invocation_id="zb-inv-current")
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert result["interlock_decision"] == DECISION_CANDIDATE_AWARE
    ok, reason, details = _entry_decision(monkeypatch, tmp_path)
    assert details.get("failed_check") != "exit_before_entry"
    interlock = evaluate_exit_before_entry(results_root=tmp_path, now=NOW)
    ctx = context_from_interlock(interlock)
    msft = evaluate_entry_candidate(
        _entry_cand("MSFT", est_notional=800.0),
        pending_sells=ctx["pending_sells"],
        current_cash=10000.0,
    )
    tsla = evaluate_entry_candidate(
        _entry_cand("TSLA", allocation_action="ADD"),
        pending_sells=ctx["pending_sells"],
        current_cash=10000.0,
    )
    assert msft["decision"] == ALLOW_EXISTING_GATES
    assert tsla["decision"] == BLOCK_SAME_SYMBOL_PENDING_SELL
    assert msft["proceeds_credited"] == 0.0


def test_inv_b_prior_invocation_fresh_artifact_blocks_all(tmp_path, monkeypatch):
    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch, invocation_id="zb-inv-old")
    generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    path = tmp_path / "paper_sell_exit_before_entry.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["generated_at"] = NOW.isoformat()
    path.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setenv(INVOCATION_ENV, "zb-inv-new")
    ok, reason, details = _entry_decision(monkeypatch, tmp_path)
    assert ok is False
    assert reason == REASON_INVOCATION_MISMATCH
    assert details["failed_check"] == "exit_before_entry"


def test_inv_c_missing_process_invocation_fail_closed(tmp_path, monkeypatch):
    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch)
    generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    monkeypatch.delenv(INVOCATION_ENV, raising=False)
    ok, reason, details = _entry_decision(monkeypatch, tmp_path)
    assert ok is False
    assert reason == REASON_INVOCATION_MISSING
    assert details["failed_check"] == "exit_before_entry"


def test_inv_d_missing_interlock_invocation_fail_closed(tmp_path, monkeypatch):
    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch)
    generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    path = tmp_path / "paper_sell_exit_before_entry.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    for key in ("invocation_id", "cycle_id", "market_open_invocation_id"):
        doc.pop(key, None)
    path.write_text(json.dumps(doc), encoding="utf-8")
    ok, reason, details = _entry_decision(monkeypatch, tmp_path)
    assert ok is False
    assert reason == REASON_INVOCATION_MISSING
    assert details["failed_check"] == "exit_before_entry"


def test_inv_e_old_valid_empty_cannot_authorize_new_invocation(tmp_path, monkeypatch):
    _write_plan(
        tmp_path,
        rows=[
            {
                "symbol": "TSLA",
                "final_action": "HOLD",
                "qty": 1.0,
                "selected_for_execution": "false",
                "reason_code": "",
                "skip_reason": "",
                "current_position_qty": 1.0,
                "current_price": 332.5,
                "planned_notional": 0,
            }
        ],
    )
    _bind(
        tmp_path,
        monkeypatch,
        invocation_id="zb-inv-crash",
        remaining=BLOCKER_NONE,
        op_req=False,
        block_reasons=[],
    )
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert result["packet_status"] == "VALID_EMPTY"
    path = tmp_path / "paper_sell_exit_before_entry.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["generated_at"] = (NOW + timedelta(minutes=1)).isoformat()
    path.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setenv(INVOCATION_ENV, "zb-inv-after-crash")
    ok, reason, details = _entry_decision(monkeypatch, tmp_path, now=NOW + timedelta(minutes=1))
    assert ok is False
    assert reason == REASON_INVOCATION_MISMATCH
    assert details["failed_check"] == "exit_before_entry"


def test_inv_f_current_invocation_valid_empty_may_continue(tmp_path, monkeypatch):
    test_a_current_cycle_valid_empty_allows_existing_entry_gates(tmp_path, monkeypatch)


def test_sector_reduction_unmapped_candidate_blocked():
    from services.paper_sell_entry_dependency import (
        BLOCK_UNPROVEN_SECTOR_INDEPENDENCE,
        evaluate_entry_candidate,
    )

    rec = evaluate_entry_candidate(
        _entry_cand("ZZZZ", est_notional=300.0),
        pending_sells=[
            {
                "symbol": "TSLA",
                "action": "EXIT",
                "qty": 1,
                "reason_code": "SECTOR_CONCENTRATION_REDUCTION",
            }
        ],
        current_cash=10000.0,
    )
    assert rec["decision"] == BLOCK_UNPROVEN_SECTOR_INDEPENDENCE
    assert rec["proceeds_credited"] == 0.0


def test_ordinary_non_sector_sell_mapped_candidate_allowed():
    from services.paper_sell_entry_dependency import ALLOW_EXISTING_GATES, evaluate_entry_candidate

    rec = evaluate_entry_candidate(
        _entry_cand("UNH", est_notional=400.0),
        pending_sells=[_tsla_pending()],
        current_cash=10000.0,
    )
    assert rec["decision"] == ALLOW_EXISTING_GATES
    assert rec["proceeds_credited"] == 0.0


def test_explicit_rotation_pair_blocked():
    from services.paper_sell_entry_dependency import (
        BLOCK_ROTATION_DEPENDENCY,
        evaluate_entry_candidate,
    )

    rec = evaluate_entry_candidate(
        _entry_cand("MSFT", est_notional=400.0, rotation_id="ROT-77"),
        pending_sells=[_tsla_pending(rotation_id="ROT-77")],
        current_cash=10000.0,
    )
    assert rec["decision"] == BLOCK_ROTATION_DEPENDENCY
    assert rec["proceeds_credited"] == 0.0


def test_different_rotation_id_independent_cash_may_continue():
    from services.paper_sell_entry_dependency import ALLOW_EXISTING_GATES, evaluate_entry_candidate

    rec = evaluate_entry_candidate(
        _entry_cand("UNH", est_notional=400.0, rotation_id="ROT-B"),
        pending_sells=[_tsla_pending(rotation_id="ROT-A")],
        current_cash=10000.0,
    )
    assert rec["decision"] == ALLOW_EXISTING_GATES
    assert rec["proceeds_credited"] == 0.0


def test_missing_rotation_metadata_dependency_indicated_blocks_affected():
    from services.paper_sell_entry_dependency import (
        ALLOW_EXISTING_GATES,
        BLOCK_ROTATION_DEPENDENCY,
        evaluate_entry_candidate,
    )

    pending = [{"symbol": "TSLA", "action": "ROTATE_EXIT", "qty": 1, "reason_code": "ROTATION"}]
    blocked = evaluate_entry_candidate(
        _entry_cand("MSFT", est_notional=400.0, rotation_dependent=True),
        pending_sells=pending,
        current_cash=10000.0,
    )
    independent = evaluate_entry_candidate(
        _entry_cand("UNH", est_notional=400.0),
        pending_sells=pending,
        current_cash=10000.0,
    )
    assert blocked["decision"] == BLOCK_ROTATION_DEPENDENCY
    assert independent["decision"] == ALLOW_EXISTING_GATES
    assert independent["proceeds_credited"] == 0.0


def test_mixed_batch_filters_candidates_individually():
    from services.paper_sell_entry_dependency import (
        ALLOW_EXISTING_GATES,
        BLOCK_ROTATION_DEPENDENCY,
        BLOCK_SAME_SYMBOL_PENDING_SELL,
        BLOCK_UNPROVEN_SECTOR_INDEPENDENCE,
        filter_entry_candidates,
    )

    allowed, decisions = filter_entry_candidates(
        [
            _entry_cand("UNH", est_notional=400.0),
            _entry_cand("ZZZZ", est_notional=200.0),
            _entry_cand("MSFT", est_notional=300.0, rotation_id="BATCH-1"),
            _entry_cand("TSLA", allocation_action="ADD", est_notional=100.0),
        ],
        pending_sells=[
            {
                "symbol": "TSLA",
                "action": "EXIT",
                "qty": 1,
                "reason_code": "SECTOR_CONCENTRATION_REDUCTION",
                "rotation_id": "BATCH-1",
            }
        ],
        current_cash=10000.0,
    )
    by_sym = {d["symbol"]: d["decision"] for d in decisions}
    assert by_sym["UNH"] == ALLOW_EXISTING_GATES
    assert by_sym["ZZZZ"] == BLOCK_UNPROVEN_SECTOR_INDEPENDENCE
    assert by_sym["MSFT"] == BLOCK_ROTATION_DEPENDENCY
    assert by_sym["TSLA"] == BLOCK_SAME_SYMBOL_PENDING_SELL
    assert [c["ticker"] for c in allowed] == ["UNH"]
    assert all((d.get("proceeds_credited") or 0) == 0.0 for d in decisions)
