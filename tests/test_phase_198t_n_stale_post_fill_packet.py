"""Phase 198T-O — same-cycle post-dispatch packet suppression.

Isolated tmp_path only. No production writes, no broker, no Streamlit.
"""

from __future__ import annotations

import ast
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from services.defensive_exit_approval_readiness import (
    PACKET_NAME,
    SAME_CYCLE_DISPATCH_REASON,
    STATUS_UNAVAILABLE_SAME_CYCLE,
    generate_from_manage_artifacts,
    write_canonical_packet,
    write_pre_plan_dispatch_evidence,
)
from services.manage_positions_cycle_binding import (
    BLOCKER_MISSING_SELL_AUTHORITY,
    INVOCATION_ENV,
    write_cycle_binding,
)
from services.paper_sell_arm_authority import (
    classify_readiness_packet,
    confirm_and_create_paper_sell_arm,
    prepare_confirmation_challenge,
)
from services.paper_sell_arm_console import build_presentation

NOW = datetime(2026, 8, 26, 15, 20, 56, tzinfo=timezone.utc)
INV = "mo-20260826_111929-7cf18c12763e436e8744f9b7f2637ade"
INV_LATER = "mo-20260826_151314-0a75eaec55a04e4c8d744baaf95edcd6"
ROOT = Path(__file__).resolve().parents[1]


def _plan_row(symbol: str, qty: float, *, selected: bool = True, price: float) -> dict:
    return {
        "symbol": symbol,
        "final_action": "EXIT",
        "qty": qty,
        "selected_for_execution": "true" if selected else "false",
        "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
        "skip_reason": "",
        "current_position_qty": qty,
        "current_price": price,
        "planned_notional": float(qty) * price,
    }


def _write_plan(results: Path, rows: list[dict], *, ts: str | None = None) -> None:
    ts = ts or NOW.isoformat()
    results.mkdir(parents=True, exist_ok=True)
    (results / "manage_positions_plan.json").write_text(
        json.dumps({"timestamp": ts, "orders_executed": 0, "blocked": True}),
        encoding="utf-8",
    )
    fields = [
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
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _bind(results: Path, monkeypatch, *, invocation_id: str) -> None:
    monkeypatch.setenv(INVOCATION_ENV, invocation_id)
    plan = json.loads((results / "manage_positions_plan.json").read_text(encoding="utf-8"))
    write_cycle_binding(
        results_root=results,
        invocation_id=invocation_id,
        plan_generated_at=str(plan["timestamp"]),
        f_stage_status="COMPLETE",
        now=NOW,
        remaining_blocker=BLOCKER_MISSING_SELL_AUTHORITY,
        operator_execution_required=True,
        block_reasons=["DISPATCH_BLOCKED_APPROVAL_REQUIRED"],
        orders_executed=0,
    )


def _evidence(
    results: Path,
    *,
    invocation_id: str,
    dispatch_performed=True,
    placed: int = 2,
    consumed: bool = True,
    dispatched: bool = True,
    attempted: bool = True,
    evidence_status: str = "OK",
    extra: dict | None = None,
) -> dict:
    payload = {
        "attempted": attempted,
        "dispatched": dispatched,
        "consumed": consumed,
        "broker_submissions": placed,
        "broker_orders_submitted": placed > 0 or dispatched,
        "dispatch_performed": dispatch_performed,
        "dispatch_intent_id": "sell-dispatch-7016ea27eadc41e1",
        "authority_id": "sell-arm-audit-d55860b9ade2",
        "dispatch_terminal_status": "TERMINAL_SUCCESS" if dispatched else "REJECTED",
        "reason": "ok" if dispatched else "broker_rejected",
        "submit": {"placed": placed, "failed": 0 if dispatched else 2},
    }
    if extra:
        payload.update(extra)
    ev = write_pre_plan_dispatch_evidence(
        results,
        payload,
        invocation_id=invocation_id,
        now=NOW,
        session_date="2026-08-26",
    )
    if evidence_status != "OK" or dispatch_performed not in (True, False):
        path = results / "manage_positions_pre_plan_dispatch.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["evidence_status"] = evidence_status
        if dispatch_performed not in (True, False):
            doc["dispatch_performed"] = dispatch_performed
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return doc
    return ev


def _hd_bito_rows(*, hd: float, bito: float) -> list[dict]:
    return [
        _plan_row("GLD", 16.0, price=420.98),
        _plan_row("HD", hd, price=337.02),
        _plan_row("BITO", bito, price=10.5),
    ]


def test_01_hd_bito_incident_replay_not_armable(tmp_path: Path, monkeypatch):
    rows = _hd_bito_rows(hd=40.0, bito=1365.0)
    _write_plan(tmp_path, rows)
    _bind(tmp_path, monkeypatch, invocation_id=INV)
    _evidence(tmp_path, invocation_id=INV, placed=2)
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    packet = result["packet"]
    assert result["ok"] is True
    assert packet["status"] == STATUS_UNAVAILABLE_SAME_CYCLE
    assert packet["reason"] == SAME_CYCLE_DISPATCH_REASON
    assert packet["creates_no_authority"] is True
    assert packet["orders_submitted"] is True
    assert packet["approval_ready_actions"] == []
    assert packet["arm_eligibility"] == "UNAVAILABLE"
    assert packet["invocation_id"] == INV
    assert packet["dispatch_id"] == "sell-dispatch-7016ea27eadc41e1"


def test_02_overwrites_older_ready_packet(tmp_path: Path, monkeypatch):
    prior = {
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "creates_no_authority": True,
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "cycle_id": "old",
        "run_id": "old",
        "batch_id": "old",
        "plan_id": "old",
        "plan_hash": "old",
        "packet_status": "AUTHORITY_READY",
        "approval_ready_symbols": ["HD"],
        "approval_ready_actions": [{"symbol": "HD", "action": "EXIT", "qty": 40.0}],
        "generated_at": NOW.isoformat(),
        "valid_until": NOW.isoformat(),
    }
    dest = tmp_path / PACKET_NAME
    write_canonical_packet(dest, prior)
    before = dest.read_text(encoding="utf-8")
    assert "AUTHORITY_READY" in before
    _write_plan(tmp_path, _hd_bito_rows(hd=40.0, bito=1365.0))
    _bind(tmp_path, monkeypatch, invocation_id=INV)
    _evidence(tmp_path, invocation_id=INV)
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert result["previous_preserved"] is False
    after = json.loads(dest.read_text(encoding="utf-8"))
    assert after["status"] == STATUS_UNAVAILABLE_SAME_CYCLE
    assert after["approval_ready_actions"] == []
    assert after["packet_status"] == SAME_CYCLE_DISPATCH_REASON


@pytest.mark.parametrize(
    "placed,consumed,dispatched",
    [
        (2, True, True),  # full fill / accepted
        (1, True, True),  # partial fill
        (2, True, True),  # accepted not yet filled still submitted
    ],
)
def test_03_to_05_fill_states_suppress(tmp_path: Path, monkeypatch, placed, consumed, dispatched):
    _write_plan(tmp_path, _hd_bito_rows(hd=40.0, bito=1365.0))
    _bind(tmp_path, monkeypatch, invocation_id=INV)
    _evidence(
        tmp_path,
        invocation_id=INV,
        placed=placed,
        consumed=consumed,
        dispatched=dispatched,
    )
    packet = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)["packet"]
    assert packet["status"] == STATUS_UNAVAILABLE_SAME_CYCLE
    assert packet["approval_ready_actions"] == []


def test_06_broker_rejection_no_accepted_order_suppresses(tmp_path: Path, monkeypatch):
    _write_plan(tmp_path, _hd_bito_rows(hd=40.0, bito=1365.0))
    _bind(tmp_path, monkeypatch, invocation_id=INV)
    _evidence(
        tmp_path,
        invocation_id=INV,
        dispatch_performed=True,
        placed=0,
        consumed=True,
        dispatched=False,
    )
    packet = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)["packet"]
    assert packet["status"] == STATUS_UNAVAILABLE_SAME_CYCLE
    assert packet["orders_submitted"] is False


def test_07_ambiguous_dispatch_fail_closed(tmp_path: Path, monkeypatch):
    _write_plan(tmp_path, _hd_bito_rows(hd=40.0, bito=1365.0))
    _bind(tmp_path, monkeypatch, invocation_id=INV)
    _evidence(
        tmp_path,
        invocation_id=INV,
        dispatch_performed="unknown",
        evidence_status="AMBIGUOUS",
    )
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert result["packet"]["status"] == STATUS_UNAVAILABLE_SAME_CYCLE
    assert result["dispatch_classification"] == "ambiguous"


def test_08_no_dispatch_cycle_may_remain_authority_ready(tmp_path: Path, monkeypatch):
    _write_plan(tmp_path, _hd_bito_rows(hd=40.0, bito=1365.0))
    _bind(tmp_path, monkeypatch, invocation_id=INV)
    _evidence(
        tmp_path,
        invocation_id=INV,
        dispatch_performed=False,
        placed=0,
        consumed=False,
        dispatched=False,
        attempted=False,
    )
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert result["packet_status"] == "AUTHORITY_READY"
    qtys = {a["symbol"]: a["qty"] for a in result["packet"]["approval_ready_actions"]}
    assert qtys["HD"] == 40.0
    assert qtys["BITO"] == 1365.0


def test_09_later_reconciled_cycle_hd27_bito1337(tmp_path: Path, monkeypatch):
    rows = _hd_bito_rows(hd=27.0, bito=1337.0)
    _write_plan(tmp_path, rows, ts="2026-08-26T19:14:25+00:00")
    _bind(tmp_path, monkeypatch, invocation_id=INV_LATER)
    _evidence(
        tmp_path,
        invocation_id=INV_LATER,
        dispatch_performed=False,
        placed=0,
        consumed=False,
        dispatched=False,
        attempted=False,
    )
    result = generate_from_manage_artifacts(
        results_root=tmp_path, now=datetime(2026, 8, 26, 19, 14, 29, tzinfo=timezone.utc)
    )
    assert result["packet_status"] == "AUTHORITY_READY"
    qtys = {a["symbol"]: a["qty"] for a in result["packet"]["approval_ready_actions"]}
    assert qtys["HD"] == 27.0
    assert qtys["BITO"] == 1337.0


def test_10_prior_invocation_dispatch_does_not_suppress(tmp_path: Path, monkeypatch):
    _write_plan(tmp_path, _hd_bito_rows(hd=27.0, bito=1337.0))
    _bind(tmp_path, monkeypatch, invocation_id=INV_LATER)
    _evidence(tmp_path, invocation_id=INV, dispatch_performed=True, placed=2)
    result = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    assert result["packet_status"] == "AUTHORITY_READY"


def test_11_challenge_rejects_non_armable(tmp_path: Path, monkeypatch):
    _write_plan(tmp_path, _hd_bito_rows(hd=40.0, bito=1365.0))
    _bind(tmp_path, monkeypatch, invocation_id=INV)
    _evidence(tmp_path, invocation_id=INV)
    packet = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)["packet"]
    ok, reason = classify_readiness_packet(packet)
    assert ok is False
    with pytest.raises(ValueError, match="challenge_blocked"):
        prepare_confirmation_challenge(
            store_root=tmp_path,
            readiness_packet=packet,
            explicit_runtime_write=True,
            db_path=tmp_path / "paper_sell_arm.sqlite3",
            now=NOW,
        )


def test_12_authority_rejects_non_armable(tmp_path: Path, monkeypatch):
    _write_plan(tmp_path, _hd_bito_rows(hd=40.0, bito=1365.0))
    _bind(tmp_path, monkeypatch, invocation_id=INV)
    _evidence(tmp_path, invocation_id=INV)
    packet = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)["packet"]
    from services.paper_sell_arm_authority import OPERATOR_ACKNOWLEDGEMENT_TEXT

    with pytest.raises(ValueError, match="arm_rejected:"):
        confirm_and_create_paper_sell_arm(
            store_root=tmp_path,
            readiness_packet=packet,
            challenge={"challenge_id": "x", "creates_no_authority": True},
            operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
            explicit_runtime_write=True,
            db_path=tmp_path / "paper_sell_arm.sqlite3",
            now=NOW,
        )


def test_13_dashboard_sell_arm_controls_disabled(tmp_path: Path, monkeypatch):
    _write_plan(tmp_path, _hd_bito_rows(hd=40.0, bito=1365.0))
    _bind(tmp_path, monkeypatch, invocation_id=INV)
    _evidence(tmp_path, invocation_id=INV)
    generate_from_manage_artifacts(results_root=tmp_path, now=NOW)
    pres = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=NOW,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    assert pres["arm_enabled"] is False
    assert pres["confirm_enabled"] is False
    from dashboard.paper_sell_arm_controls import map_sell_arm_visual_status

    visual = map_sell_arm_visual_status(pres)
    assert visual["code"] != "READY_TO_ARM"
    assert visual["label"] != "SELL ARM — READY TO ARM"


def test_14_audit_fields_preserved(tmp_path: Path, monkeypatch):
    _write_plan(tmp_path, _hd_bito_rows(hd=40.0, bito=1365.0))
    _bind(tmp_path, monkeypatch, invocation_id=INV)
    _evidence(tmp_path, invocation_id=INV)
    packet = generate_from_manage_artifacts(results_root=tmp_path, now=NOW)["packet"]
    assert packet["session_date"] == "2026-08-26"
    assert packet["invocation_id"] == INV
    assert packet["dispatch_id"]
    assert packet["reason"] == SAME_CYCLE_DISPATCH_REASON
    assert packet["authority_id"] == "sell-arm-audit-d55860b9ade2"


def test_15_no_broker_from_packet_or_dashboard():
    src = (ROOT / "services" / "defensive_exit_approval_readiness.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"AlpacaBroker", "list_positions", "submit_order", "place_order"}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert not (banned & names)
    assert "list_positions" not in calls
    dash = (ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    assert "AlpacaBroker" not in dash
    assert "list_positions" not in dash
    assert "Does not dispatch" in dash
    assert "contact a broker" in dash
