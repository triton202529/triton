"""Fixture checks for concentration-exit precedence. No production packet or ledger write."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORDER_LEDGER = ROOT / "data" / "results" / "paper_order_events.jsonl"
ACCOUNTING_LEDGER = ROOT / "data" / "results" / "performance_accounting_events.jsonl"
LEDGER_PARQUET = ROOT / "data" / "ledger" / "ledger.parquet"
HOLD_AUDIT = ROOT / "data" / "governance" / "basis_unproven_hold_audit.jsonl"
HOLD_POLICY = ROOT / "data" / "governance" / "basis_unproven_hold.json"
PRODUCTION_PACKET = ROOT / "data" / "results" / "defensive_exit_approval_readiness.json"
PRODUCTION_RECEIPTS = ROOT / "data" / "results" / "paper_cycle_receipts.jsonl"
NOW = datetime(2026, 8, 11, 16, 0, tzinfo=timezone.utc)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot() -> dict:
    return {
        "orders": _sha(ORDER_LEDGER),
        "accounting": _sha(ACCOUNTING_LEDGER),
        "parquet": _sha(LEDGER_PARQUET),
        "hold_audit_size": HOLD_AUDIT.stat().st_size,
        "hold_audit": _sha(HOLD_AUDIT),
        "hold_policy": _sha(HOLD_POLICY),
        "packet": _sha(PRODUCTION_PACKET),
        "receipts": _sha(PRODUCTION_RECEIPTS),
    }


@pytest.fixture(scope="module")
def production_snapshot():
    return _snapshot()


def _binding() -> dict:
    return {
        "invocation_id": "mo-concentration-fixture",
        "plan_generated_at": "2026-08-11T16:00:00+00:00",
        "selected_row_hash": "fixture-hash",
        "source_plan_sha256": "fixture-plan",
        "f_stage_status": "COMPLETE",
        "f_stage_completed_at": "2026-08-11T16:00:00+00:00",
        "remaining_blocker": "MISSING_PAPER_SELL_AUTHORITY",
    }


def _write_plan(results: Path, rows: list[dict]) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / "manage_positions_plan.json").write_text('{"present": true}', encoding="utf-8")
    header = (
        "symbol,final_action,qty,selected_for_execution,reason_code,"
        "current_position_qty,current_price,planned_notional\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row.get("symbol", "")),
                    str(row.get("final_action", "")),
                    str(row.get("qty", "")),
                    str(row.get("selected_for_execution", "")),
                    str(row.get("reason_code", "")),
                    str(row.get("current_position_qty", "")),
                    str(row.get("current_price", "")),
                    str(row.get("planned_notional", "")),
                ]
            )
            + "\n"
        )
    (results / "manage_positions_plan.csv").write_text("".join(lines), encoding="utf-8")


def _generate(tmp_path: Path, monkeypatch, rows: list[dict], classified: list[dict]):
    from services.defensive_exit_approval_readiness import (
        generate_from_manage_artifacts,
    )

    results = tmp_path / "results"
    _write_plan(results, rows)
    prior = results / "defensive_exit_approval_readiness.json"
    prior.write_text('{"generated_at":"prior","approval_ready_symbols":["KEEP"]}', encoding="utf-8")
    prior_bytes = prior.read_bytes()
    monkeypatch.setattr(
        "services.defensive_exit_approval_readiness._validate_cycle_binding",
        lambda **kwargs: {"ok": True, "binding": _binding(), "exit_code": 0},
    )
    monkeypatch.setattr(
        "services.defensive_exit_approval_readiness.classify_plan_rows",
        lambda plan_rows, binding: classified,
    )

    def _forbid_post(*args, **kwargs):
        raise AssertionError("broker POST")

    monkeypatch.setattr("requests.sessions.Session.post", _forbid_post)
    result = generate_from_manage_artifacts(results_root=results, write=True, now=NOW)
    return result, prior_bytes, prior


def test_hold_offender_becomes_capped_trim_without_clearing_the_breach(tmp_path, monkeypatch):
    hold = {
        "symbol": "CONC",
        "action": "NONE",
        "qty": "",
        "classification": "NOT_SELECTED",
        "reason_code": "NO_ACTION_HOLD",
        "selected_for_execution": False,
    }
    other = {
        "symbol": "BALLAST",
        "action": "EXIT",
        "qty": 100,
        "classification": "AUTHORITY_READY",
        "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
    }
    result, _, _ = _generate(
        tmp_path,
        monkeypatch,
        rows=[
            {
                "symbol": "CONC",
                "final_action": "NONE",
                "qty": "",
                "selected_for_execution": "false",
                "reason_code": "NO_ACTION_HOLD",
                "current_position_qty": "1000.4",
                "current_price": "10",
                "planned_notional": "",
            },
            {
                "symbol": "BALLAST",
                "final_action": "EXIT",
                "qty": "100",
                "selected_for_execution": "true",
                "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
                "current_position_qty": "100",
                "current_price": "10",
                "planned_notional": "1000",
            },
            {
                "symbol": "GOOGL",
                "final_action": "NONE",
                "qty": "",
                "selected_for_execution": "false",
                "reason_code": "HISTORICAL_SALE_UNAVAILABLE",
                "current_position_qty": "",
                "current_price": "",
                "planned_notional": "",
            },
        ],
        classified=[hold, other],
    )
    packet = result["packet"]
    assert result["ok"] is True
    assert result["broker"] == 0
    assert result["packet_status"] == "AUTHORITY_READY"
    assert hold["classification"] == "NOT_SELECTED"
    assert hold["reason_code"] == "NO_ACTION_HOLD"
    assert packet["approval_ready_symbols"] == ["CONC"]
    action = packet["approval_ready_actions"][0]
    assert action["action"] == "TRIM"
    assert action["qty"] == 500
    assert action["est_notional"] == 5000.0
    assert "GOOGL" not in packet["approval_ready_symbols"]
    assert "BALLAST" not in packet["approval_ready_symbols"]
    trim = packet["concentration_trims"][0]
    assert trim["threshold_pct"] == 25.0
    assert trim["before_weight_pct"] == 90.91
    assert trim["after_weight_pct"] == 83.34
    assert trim["after_weight_pct"] > trim["threshold_pct"]
    assert trim["after_weight_pct"] < trim["before_weight_pct"]
    assert packet["mode"] == "PAPER"
    assert packet["live_execution_state"] == "BLOCKED"
    assert packet["creates_no_authority"] is True
    assert packet["orders_submitted"] is False


def test_worsening_packet_is_refused_when_no_permitted_trim_exists(tmp_path, monkeypatch):
    result, prior_bytes, prior = _generate(
        tmp_path,
        monkeypatch,
        rows=[
            {
                "symbol": "HEAVY",
                "final_action": "NONE",
                "qty": "",
                "selected_for_execution": "false",
                "reason_code": "NO_ACTION_HOLD",
                "current_position_qty": "2",
                "current_price": "6000",
                "planned_notional": "",
            },
            {
                "symbol": "SMALL",
                "final_action": "EXIT",
                "qty": "10",
                "selected_for_execution": "true",
                "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
                "current_position_qty": "10",
                "current_price": "100",
                "planned_notional": "1000",
            },
        ],
        classified=[
            {
                "symbol": "HEAVY",
                "action": "NONE",
                "classification": "NOT_SELECTED",
                "reason_code": "NO_ACTION_HOLD",
            },
            {
                "symbol": "SMALL",
                "action": "EXIT",
                "qty": 10,
                "classification": "AUTHORITY_READY",
                "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
            },
        ],
    )
    assert result["ok"] is False
    assert result["wrote"] is False
    assert result["packet"] is None
    assert result["packet_status"] == "CONCENTRATION_EXIT_UNAVAILABLE"
    assert result["reason"] == "concentration_exit_unavailable"
    assert result["previous_preserved"] is True
    assert result["broker"] == 0
    assert prior.read_bytes() == prior_bytes


def test_voo_and_xlk_remain_excluded_when_they_are_the_offenders(tmp_path, monkeypatch):
    result, prior_bytes, prior = _generate(
        tmp_path,
        monkeypatch,
        rows=[
            {
                "symbol": "VOO",
                "final_action": "NONE",
                "qty": "",
                "selected_for_execution": "false",
                "reason_code": "NO_ACTION_HOLD",
                "current_position_qty": "50",
                "current_price": "100",
                "planned_notional": "",
            },
            {
                "symbol": "XLK",
                "final_action": "NONE",
                "qty": "",
                "selected_for_execution": "false",
                "reason_code": "NO_ACTION_HOLD",
                "current_position_qty": "30",
                "current_price": "100",
                "planned_notional": "",
            },
            {
                "symbol": "SMALL",
                "final_action": "EXIT",
                "qty": "5",
                "selected_for_execution": "true",
                "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
                "current_position_qty": "5",
                "current_price": "100",
                "planned_notional": "500",
            },
        ],
        classified=[
            {
                "symbol": "VOO",
                "action": "EXIT",
                "qty": 80,
                "classification": "AUTHORITY_READY",
                "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
            },
            {
                "symbol": "XLK",
                "action": "EXIT",
                "qty": 15,
                "classification": "AUTHORITY_READY",
                "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
            },
            {
                "symbol": "SMALL",
                "action": "EXIT",
                "qty": 5,
                "classification": "AUTHORITY_READY",
                "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
            },
        ],
    )
    assert result["ok"] is False
    assert result["wrote"] is False
    assert result["reason"] == "concentration_exit_unavailable"
    assert result["broker"] == 0
    assert prior.read_bytes() == prior_bytes
    assert b"VOO" not in prior_bytes or prior.read_bytes() == prior_bytes


def test_production_ledgers_hold_audit_and_packet_are_unchanged(production_snapshot):
    assert _snapshot() == production_snapshot
    assert production_snapshot["hold_audit_size"] == 0
    assert (
        b"0e99c1320742c1c1c9f750b10b53437e80cdb06013c77171f4ed218a56db09b1"
        in ACCOUNTING_LEDGER.read_bytes()
    )
