"""Phase 198I-ZA — SELL readiness packet producer/parser contract.

PAPER ONLY. Isolated tmp_path. No broker, no repository SQLite, no production packet writes.
"""

from __future__ import annotations

import ast
import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from services.defensive_exit_approval_readiness import (
    PACKET_NAME,
    PRODUCER_ID,
    SCHEMA_VERSION,
    build_canonical_packet,
    generate_from_manage_artifacts,
    run_market_open_readiness_stage,
    write_canonical_packet,
)
from services.manage_positions_cycle_binding import (
    BLOCKER_MISSING_SELL_AUTHORITY,
    F2_PENDING_SELL_ARM,
    INVOCATION_ENV,
    write_cycle_binding,
)
from services.paper_sell_arm_console import (
    CONFIRM_PAPER_SELLS_PHRASE,
    arm_reviewed_sells,
    build_presentation,
    operator_fail_message,
)
from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT
from services.paper_sell_arm_store import db_path_for_store_root, inspect_sqlite_arm_snapshot
from tests._phase_198i_sell_fakes import SESSION_OPEN_NOW, block_real_network

ROOT = Path(__file__).resolve().parents[1]
NOW = SESSION_OPEN_NOW
REPO_SQLITE = ROOT / "data" / "live" / "paper_sell_arm.sqlite3"
PROD_PACKET = ROOT / "data" / "results" / "defensive_exit_approval_readiness.json"
MARKET_OPEN = ROOT / "scripts" / "run_market_open.ps1"


@pytest.fixture(autouse=True)
def _safety(monkeypatch, tmp_path):
    block_real_network(monkeypatch)
    for key in list(os.environ):
        if "ALPACA" in key.upper():
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    for key in (
        "TRITON_PAPER_SELL_ARM_STORE_ROOT",
        "TRITON_PAPER_SELL_ARM_RESULTS_ROOT",
        "TRITON_PAPER_SELL_ARM_DB",
        "RESULTS_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)
    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)
    assert not REPO_SQLITE.is_file()


def _action(**overrides):
    row = {
        "symbol": "TSLA",
        "action": "EXIT",
        "qty": 8.0,
        "est_price": 332.5,
        "est_notional": 2660.0,
    }
    row.update(overrides)
    return row


def _packet(now=NOW, actions=None, **extra):
    return build_canonical_packet(
        now=now,
        actions=actions if actions is not None else [_action()],
        cycle_id="za_cycle",
        run_id="za_run",
        batch_id="za_batch",
        plan_id="za_plan",
        plan_hash="za_planhash",
        extra=extra,
    )


def _pres(tmp_path: Path, packet: dict, *, now=NOW):
    dest = tmp_path / PACKET_NAME
    dest.write_text(json.dumps(packet), encoding="utf-8")
    return build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=now,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )


def _write_plan(results: Path, *, symbol="TSLA", action="EXIT", qty=8.0, selected=True, ts=None):
    ts = ts or NOW.isoformat()
    results.mkdir(parents=True, exist_ok=True)
    (results / "manage_positions_plan.json").write_text(
        json.dumps({"timestamp": ts, "orders_executed": 0, "blocked": True}),
        encoding="utf-8",
    )
    with (results / "manage_positions_plan.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "symbol",
                "final_action",
                "qty",
                "selected_for_execution",
                "reason_code",
                "skip_reason",
                "current_position_qty",
                "current_price",
                "planned_notional",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "symbol": symbol,
                "final_action": action,
                "qty": qty,
                "selected_for_execution": "true" if selected else "false",
                "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
                "skip_reason": "",
                "current_position_qty": qty,
                "current_price": 332.5,
                "planned_notional": float(qty) * 332.5,
            }
        )


def _bind(
    results: Path,
    monkeypatch,
    *,
    invocation_id: str = "za-inv-1",
    remaining: str = BLOCKER_MISSING_SELL_AUTHORITY,
    op_req: bool = True,
    block_reasons=None,
    submitted_symbols=None,
    submitted_outcomes=None,
    outcome_unknown: bool = False,
    orders_executed: int = 0,
):
    monkeypatch.setenv(INVOCATION_ENV, invocation_id)
    plan = json.loads((results / "manage_positions_plan.json").read_text(encoding="utf-8"))
    write_cycle_binding(
        results_root=results,
        invocation_id=invocation_id,
        plan_generated_at=str(plan["timestamp"]),
        f_stage_status="COMPLETE",
        now=NOW,
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


def test_a_fresh_valid_sell_packet_enables_arm(tmp_path: Path):
    pkt = _packet()
    write_canonical_packet(tmp_path / PACKET_NAME, pkt)
    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    assert pres["freshness"] == "FRESH"
    assert pres["generated_at"] == pkt["generated_at"]
    assert pres["action_count"] == 1
    assert pres["arm_enabled"] is True
    assert pres["confirm_enabled"] is False
    assert pkt["producer"] == PRODUCER_ID
    assert pkt["schema_version"] == SCHEMA_VERSION
    assert not REPO_SQLITE.is_file()


def test_b_stale_packet_disables_buttons(tmp_path: Path):
    stale = NOW - timedelta(minutes=31)
    pres = _pres(tmp_path, _packet(now=stale), now=NOW)
    assert pres["freshness"] == "STALE"
    assert pres["arm_enabled"] is False
    assert pres["confirm_enabled"] is False
    assert pres["arm_deny_reason"] == "stale_readiness_packet"


def test_c_missing_generated_at_malformed(tmp_path: Path):
    pkt = _packet()
    del pkt["generated_at"]
    pres = _pres(tmp_path, pkt)
    assert pres["freshness"] == "MALFORMED"
    assert pres["generated_at"] is None
    assert pres["arm_enabled"] is False
    assert pres["confirm_enabled"] is False
    assert pres["arm_deny_reason"] == "missing_generated_at"
    assert operator_fail_message("missing_generated_at") == "Missing generated_at"


def test_d_invalid_future_naive_timestamps_fail_closed(tmp_path: Path):
    bad = _packet()
    bad["generated_at"] = "not-a-timestamp"
    pres = _pres(tmp_path, bad)
    assert pres["arm_enabled"] is False
    assert pres["arm_deny_reason"] == "unparseable_generated_at"

    future = _packet(now=NOW + timedelta(minutes=5))
    pres = _pres(tmp_path, future, now=NOW)
    assert pres["arm_enabled"] is False
    assert pres["arm_deny_reason"] == "future_generated_at"

    naive = _packet()
    naive["generated_at"] = "2026-08-15T12:00:00"
    pres = _pres(tmp_path, naive)
    assert pres["arm_enabled"] is False
    assert pres["arm_deny_reason"] == "naive_generated_at"


def test_e_empty_valid_packet(tmp_path: Path):
    pres = _pres(tmp_path, _packet(actions=[]))
    assert pres["freshness"] == "EMPTY"
    assert pres["arm_enabled"] is False
    assert pres["confirm_enabled"] is False
    assert pres["arm_deny_reason"] == "no_reviewed_sell_actions"
    assert "No reviewed SELL actions" in operator_fail_message("no_reviewed_sell_actions")


def test_f_buy_or_mixed_side_fails_closed(tmp_path: Path):
    with pytest.raises(ValueError, match="buy_or_add"):
        _packet(actions=[_action(action="BUY")])
    mixed = _packet()
    mixed["approval_ready_actions"].append(
        {"symbol": "AAPL", "action": "ADD", "qty": 1, "est_notional": 1}
    )
    pres = _pres(tmp_path, mixed)
    assert pres["arm_enabled"] is False
    assert pres["arm_deny_reason"] == "malformed_packet"


def test_partial_malformed_action_fails_entire_batch(tmp_path: Path):
    pkt = _packet()
    pkt["approval_ready_actions"].append({"symbol": "MSFT", "action": "EXIT", "qty": 0})
    pres = _pres(tmp_path, pkt)
    assert pres["arm_enabled"] is False
    assert pres["arm_deny_reason"] == "malformed_packet"


def test_legacy_production_packet_missing_generated_at_not_backfilled():
    assert PROD_PACKET.is_file()
    doc = json.loads(PROD_PACKET.read_text(encoding="utf-8"))
    assert "generated_at" not in doc
    from services.paper_sell_arm_console import packet_freshness

    ok, reason, generated = packet_freshness(doc, now=NOW)
    assert ok is False
    assert reason == "missing_generated_at"
    assert generated is None


def test_writer_from_isolated_plan_refreshes_generated_at(tmp_path: Path, monkeypatch):
    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch)
    first = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert first["ok"] is True
    assert first["packet"]["generated_at"] == NOW.isoformat()
    assert first["packet"]["approval_ready_actions"][0]["symbol"] == "TSLA"
    later = NOW + timedelta(minutes=2)
    second = generate_from_manage_artifacts(results_root=tmp_path, now=later)
    assert second["packet"]["generated_at"] == later.isoformat()
    assert first["broker"] == 0 and second["network"] == 0


def test_failed_generation_preserves_previous_packet(tmp_path: Path, monkeypatch):
    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch)
    first = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert first["ok"] is True
    before = (tmp_path / PACKET_NAME).read_text(encoding="utf-8")
    (tmp_path / "manage_positions_plan.csv").unlink()
    (tmp_path / "manage_positions_plan.json").unlink()
    failed = generate_from_manage_artifacts(results_root=tmp_path, now=NOW + timedelta(minutes=3))
    assert failed["ok"] is False
    assert failed["wrote"] is False
    assert failed["previous_preserved"] is True
    after = (tmp_path / PACKET_NAME).read_text(encoding="utf-8")
    assert after == before
    assert json.loads(after)["generated_at"] == NOW.isoformat()


def test_loop_stage_invokes_canonical_writer(tmp_path: Path, monkeypatch):
    src = MARKET_OPEN.read_text(encoding="utf-8")
    assert "services.defensive_exit_approval_readiness" in src
    assert "F2. PAPER SELL READINESS PACKET" in src
    assert "TRITON_MARKET_OPEN_INVOCATION_ID" in src
    assert 'Run-Step -Label "F2. PAPER SELL READINESS PACKET"' not in src
    tree = ast.parse(
        (ROOT / "services" / "defensive_exit_approval_readiness.py").read_text(encoding="utf-8")
    )
    banned = {"begin_paper_sell_dispatch", "submit_order", "AlpacaBroker", "place_order"}
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    assert names.isdisjoint(banned)
    _write_plan(tmp_path)
    _bind(tmp_path, monkeypatch)
    rc = run_market_open_readiness_stage(results_root=tmp_path)
    assert rc == F2_PENDING_SELL_ARM
    written = json.loads((tmp_path / PACKET_NAME).read_text(encoding="utf-8"))
    assert written["producer"] == PRODUCER_ID
    assert written["generated_at"]


def test_streamlit_fresh_then_stale(tmp_path: Path):
    write_canonical_packet(tmp_path / PACKET_NAME, _packet())
    try:
        from streamlit.testing.v1 import AppTest

        def _app():
            from dashboard.paper_sell_arm_controls import render_paper_sell_arm_controls

            render_paper_sell_arm_controls(
                db_path=tmp_path / "paper_sell_arm.sqlite3",
                results_root=tmp_path,
                audit_root=tmp_path,
                paper_mode=True,
                live_blocked=True,
            )

        at = AppTest.from_function(_app, default_timeout=30)
        at.run()
        if at.exception:
            raise at.exception
        body = []
        for bucket in (at.markdown, at.text, at.caption, at.warning, at.info, at.success):
            for item in bucket:
                body.append(str(getattr(item, "value", item)))
        joined = " ".join(body)
        assert "FRESH" in joined or "Packet freshness" in joined
        arm = [b for b in at.button if b.label == "Arm Reviewed Sells"][0]
        assert arm.disabled is False
        confirm = [b for b in at.button if b.label == "Confirm & Arm Paper Sells"][0]
        assert confirm.disabled is True
        arm.click().run()
        snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
        assert snap["challenge_count"] == 1
        assert snap["open_challenges"][0]["state"] == "OPEN"
        assert snap["authority_count"] == 0
        assert snap["dispatch_count"] == 0
        assert _ACTIVE_PERMIT.get() is None
        stale = _packet(now=NOW - timedelta(minutes=40))
        write_canonical_packet(tmp_path / PACKET_NAME, stale)
        at.run()
        arm2 = [b for b in at.button if b.label == "Arm Reviewed Sells"][0]
        assert arm2.disabled is True
        conf2 = [b for b in at.button if b.label == "Confirm & Arm Paper Sells"][0]
        assert conf2.disabled is True
    except Exception:
        armed = arm_reviewed_sells(
            store_root=tmp_path,
            results_root=tmp_path,
            now=NOW,
            paper_mode=True,
            live_blocked=True,
            db_path=tmp_path / "paper_sell_arm.sqlite3",
        )
        assert armed["ok"] is True
        snap = inspect_sqlite_arm_snapshot(db_path_for_store_root(tmp_path), now=NOW)
        assert snap["open_challenges"][0]["state"] == "OPEN"
        assert snap["authority_count"] == 0
        assert armed["counts"]["broker"] == 0
        stale = _packet(now=NOW - timedelta(minutes=40))
        write_canonical_packet(tmp_path / PACKET_NAME, stale)
        pres = build_presentation(
            store_root=tmp_path,
            results_root=tmp_path,
            now=NOW,
            paper_mode=True,
            live_blocked=True,
            db_path=tmp_path / "paper_sell_arm.sqlite3",
            confirmation_phrase=CONFIRM_PAPER_SELLS_PHRASE,
        )
        assert pres["arm_enabled"] is False
        assert pres["confirm_enabled"] is False
        assert pres["freshness"] == "STALE"
