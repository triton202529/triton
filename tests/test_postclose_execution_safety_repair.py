"""Offline checks for post-close sell-packet, cycle-receipt, and operator-label repairs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORDER_LEDGER = ROOT / "data" / "results" / "paper_order_events.jsonl"
ACCOUNTING_LEDGER = ROOT / "data" / "results" / "performance_accounting_events.jsonl"
HOLD_AUDIT = ROOT / "data" / "governance" / "basis_unproven_hold_audit.jsonl"
# Historical pins from before the October 14 BITO fill. Do not retarget them
# to the current ledger. The live guard compares each file with itself.
ORDER_LEDGER_SHA = "f471049c339a3fc4569487f6654aa98e353c7e1bb76c8fff82194538f32ec7cf"
ACCOUNTING_LEDGER_SHA = "be24282f6ed8ab5ac9622639f8ee3c1cd75e05bf8c30c4c46d161ea10dd2c535"
PRODUCTION_GUARD_PATHS = (ORDER_LEDGER, ACCOUNTING_LEDGER, HOLD_AUDIT)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _production_snapshot() -> dict:
    snapshot = {}
    for path in PRODUCTION_GUARD_PATHS:
        data = path.read_bytes()
        snapshot[str(path)] = (hashlib.sha256(data).hexdigest(), len(data))
    return snapshot


@pytest.fixture(scope="module", autouse=True)
def production_ledgers_and_hold_audit_stable():
    before = _production_snapshot()
    assert before[str(HOLD_AUDIT)][1] == 0
    yield before
    assert _production_snapshot() == before
    assert HOLD_AUDIT.stat().st_size == 0


def _binding() -> dict:
    return {
        "invocation_id": "mo-test-hold",
        "plan_generated_at": "2026-10-13T20:00:00+00:00",
        "selected_row_hash": "abc",
        "source_plan_sha256": "def",
        "f_stage_status": "COMPLETE",
        "f_stage_completed_at": "2026-10-13T20:00:00+00:00",
        "remaining_blocker": "MISSING_PAPER_SELL_AUTHORITY",
    }


def _prepare_results(tmp_path: Path) -> tuple[Path, bytes]:
    results = tmp_path / "results"
    results.mkdir()
    (results / "manage_positions_plan.json").write_text('{"present": true}', encoding="utf-8")
    (results / "manage_positions_plan.csv").write_text("symbol,action,qty\n", encoding="utf-8")
    prior = results / "defensive_exit_approval_readiness.json"
    prior.write_text(
        '{"generated_at":"old","approval_ready_symbols":["XLK"]}',
        encoding="utf-8",
    )
    return results, prior.read_bytes()


def test_production_ledgers_and_hold_audit_remain_unchanged(
    production_ledgers_and_hold_audit_stable,
):
    before = production_ledgers_and_hold_audit_stable
    assert before[str(HOLD_AUDIT)][1] == 0
    assert HOLD_AUDIT.read_bytes() == b""
    assert _production_snapshot() == before


def test_packet_generation_excludes_voo_and_xlk(tmp_path: Path, monkeypatch):
    from services.defensive_exit_approval_readiness import generate_from_manage_artifacts

    results, prior_bytes = _prepare_results(tmp_path)
    monkeypatch.setattr(
        "services.defensive_exit_approval_readiness._validate_cycle_binding",
        lambda **kwargs: {"ok": True, "binding": _binding(), "exit_code": 0},
    )
    classified = [
        {
            "symbol": "XLK",
            "action": "EXIT",
            "qty": 7,
            "classification": "AUTHORITY_READY",
            "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
        },
        {
            "symbol": "VOO",
            "action": "EXIT",
            "qty": 2,
            "classification": "AUTHORITY_READY",
            "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
        },
        {
            "symbol": "SPCX",
            "action": "EXIT",
            "qty": 8,
            "classification": "AUTHORITY_READY",
            "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
        },
    ]
    monkeypatch.setattr(
        "services.defensive_exit_approval_readiness.classify_plan_rows",
        lambda rows, binding: classified,
    )
    result = generate_from_manage_artifacts(
        results_root=results,
        write=True,
        now=datetime(2026, 10, 13, 20, 0, tzinfo=timezone.utc),
    )
    packet = result["packet"]
    assert result["ok"] is True
    assert result["broker"] == 0
    assert packet["approval_ready_symbols"] == ["SPCX"]
    assert {row["symbol"] for row in packet["approval_ready_actions"]} == {"SPCX"}
    assert set(packet["excluded_basis_unproven_hold_symbols"]) == {"XLK", "VOO"}
    assert packet["generated_at"] != "old"
    assert classified[0]["classification"] == "AUTHORITY_READY"
    assert (results / "defensive_exit_approval_readiness.json").read_bytes() != prior_bytes
    assert HOLD_AUDIT.stat().st_size == 0


def test_hold_only_packet_is_not_authority_ready(tmp_path: Path, monkeypatch):
    from services.defensive_exit_approval_readiness import generate_from_manage_artifacts

    results, _prior = _prepare_results(tmp_path)
    monkeypatch.setattr(
        "services.defensive_exit_approval_readiness._validate_cycle_binding",
        lambda **kwargs: {"ok": True, "binding": _binding(), "exit_code": 0},
    )
    monkeypatch.setattr(
        "services.defensive_exit_approval_readiness.classify_plan_rows",
        lambda rows, binding: [
            {
                "symbol": "XLK",
                "action": "EXIT",
                "qty": 7,
                "classification": "AUTHORITY_READY",
            }
        ],
    )
    result = generate_from_manage_artifacts(
        results_root=results,
        write=True,
        now=datetime(2026, 10, 13, 20, 0, tzinfo=timezone.utc),
    )
    assert result["packet_status"] != "AUTHORITY_READY"
    assert result["packet"]["approval_ready_symbols"] == []
    assert result["packet"]["arm_eligibility"] == "EMPTY"


def test_policy_failure_preserves_existing_packet(tmp_path: Path, monkeypatch):
    from services.basis_unproven_hold import _policy_path
    from services.defensive_exit_approval_readiness import generate_from_manage_artifacts

    results, prior_bytes = _prepare_results(tmp_path)
    token = _policy_path.set(str(tmp_path / "missing-policy.json"))
    try:
        monkeypatch.setattr(
            "services.defensive_exit_approval_readiness._validate_cycle_binding",
            lambda **kwargs: {"ok": True, "binding": _binding(), "exit_code": 0},
        )
        monkeypatch.setattr(
            "services.defensive_exit_approval_readiness.classify_plan_rows",
            lambda rows, binding: [
                {
                    "symbol": "SPCX",
                    "action": "EXIT",
                    "qty": 8,
                    "classification": "AUTHORITY_READY",
                }
            ],
        )
        result = generate_from_manage_artifacts(
            results_root=results,
            write=True,
            now=datetime(2026, 10, 13, 20, 0, tzinfo=timezone.utc),
        )
    finally:
        _policy_path.reset(token)
    assert result["ok"] is False
    assert result["wrote"] is False
    assert result["packet"] is None
    assert result["packet_status"] == "HOLD_POLICY_UNAVAILABLE"
    assert result["previous_preserved"] is True
    assert (results / "defensive_exit_approval_readiness.json").read_bytes() == prior_bytes


def test_ready_to_arm_refuses_hold_symbols_without_editing_packet(tmp_path: Path):
    from services.paper_sell_arm_console import hold_arm_denial

    packet_path = tmp_path / "packet.json"
    packet = {
        "approval_ready_symbols": ["SPCX", "XLK"],
        "approval_ready_actions": [
            {"symbol": "SPCX", "action": "EXIT", "qty": 8},
            {"symbol": "XLK", "action": "EXIT", "qty": 7},
        ],
    }
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    before = packet_path.read_bytes()
    assert hold_arm_denial(packet) == "basis_unproven_hold_symbol_in_packet"
    assert (
        hold_arm_denial({"approval_ready_symbols": ["SPCX"], "approval_ready_actions": []}) is None
    )
    assert packet_path.read_bytes() == before


def test_ready_to_arm_fails_closed_when_policy_is_unavailable(tmp_path: Path):
    from services.basis_unproven_hold import _policy_path
    from services.paper_sell_arm_console import hold_arm_denial

    token = _policy_path.set(str(tmp_path / "missing-policy.json"))
    try:
        reason = hold_arm_denial(
            {
                "approval_ready_symbols": ["SPCX"],
                "approval_ready_actions": [{"symbol": "SPCX", "action": "EXIT", "qty": 8}],
            }
        )
    finally:
        _policy_path.reset(token)
    assert reason == "basis_unproven_hold_policy_unavailable"


def test_cycle_receipts_are_append_only(tmp_path: Path):
    from services.paper_entry_loop_stage import append_cycle_receipt

    def _audit(invocation: str, reason: str, submitted: bool = False) -> dict:
        return {
            "timestamp": "2026-10-13T15:50:04+00:00",
            "gate_pass": False,
            "gate_reason": reason,
            "executed": submitted,
            "broker_orders_submitted": submitted,
            "safe_block": True,
            "eligible_entry_symbols": ["ABBV", "AMZN"],
            "details": {
                "failed_check": "execution_readiness",
                "candidates": [
                    {
                        "ticker": "ABBV",
                        "allocation_action": "OPEN_NEW",
                        "eligible_for_auto_entry": True,
                        "block_reasons": "",
                    }
                ],
                "checks": {
                    "market_open_invocation_id_resolved": invocation,
                    "final_gate_classification": "ORANGE_DEGRADED_RISK_INCREASING_BLOCKED",
                    "execution_readiness_status": "PARTIALLY_READY",
                    "readiness_failed_checks": ["governance"],
                    "governance_mode": "ORANGE_DEGRADED",
                    "exit_before_entry": {
                        "decision": "CANDIDATE_AWARE",
                        "reason": "ENTRY_BLOCKED_PENDING_SELL_ARM",
                        "packet_status": "AUTHORITY_READY",
                        "pending_sell_symbols": ["SPCX"],
                    },
                },
            },
        }

    first = append_cycle_receipt(_audit("cycle-1", "ORANGE governance"), results_root=tmp_path)
    raw_first = (tmp_path / "paper_cycle_receipts.jsonl").read_bytes()
    again = append_cycle_receipt(_audit("cycle-1", "changed reason"), results_root=tmp_path)
    assert again["appended"] is False
    assert (tmp_path / "paper_cycle_receipts.jsonl").read_bytes() == raw_first
    append_cycle_receipt(_audit("cycle-2", "watchdog"), results_root=tmp_path)
    lines = (tmp_path / "paper_cycle_receipts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    saved = json.loads(lines[0])
    assert saved["invocation_id"] == "cycle-1"
    assert saved["no_order_reason"] == "ORANGE governance"
    assert saved["signal_candidates"] == ["ABBV", "AMZN"]
    assert saved["allocation_candidates"][0]["ticker"] == "ABBV"
    assert saved["final_gate_result"]["gate_reason"] == "ORANGE governance"
    assert saved["exit_gate"]["pending_sell_symbols"] == ["SPCX"]
    assert saved["order_attempted"] is False
    assert saved["order_submitted"] is False
    assert first["appended"] is True
    assert json.loads(lines[1])["invocation_id"] == "cycle-2"


def test_blank_invocation_id_cannot_append_a_receipt(tmp_path: Path):
    from services.paper_entry_loop_stage import append_cycle_receipt

    def _audit(invocation):
        return {
            "timestamp": "2026-10-13T15:50:04+00:00",
            "gate_pass": False,
            "gate_reason": "execution_readiness=NOT_READY",
            "executed": False,
            "broker_orders_submitted": False,
            "eligible_entry_symbols": ["ABBV"],
            "details": {
                "checks": {"market_open_invocation_id_resolved": invocation},
            },
        }

    receipt_path = tmp_path / "paper_cycle_receipts.jsonl"
    first = append_cycle_receipt(_audit("   "), results_root=tmp_path)
    second = append_cycle_receipt(_audit(""), results_root=tmp_path)
    missing = append_cycle_receipt(_audit(None), results_root=tmp_path)
    assert first["appended"] is False
    assert first["reason"] == "invocation_id_missing"
    assert second["appended"] is False
    assert missing["appended"] is False
    assert not receipt_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_preexisting_xlk_packet_is_refused_without_byte_change(tmp_path: Path, monkeypatch):
    from services.paper_sell_arm_console import (
        READINESS_PACKET_NAME,
        arm_reviewed_sells,
        build_presentation,
        hold_arm_denial,
    )
    from tests._phase_198i_sell_fakes import SESSION_OPEN_NOW, readiness_packet

    now = SESSION_OPEN_NOW
    packet = readiness_packet(symbol="XLK", qty=7.0, price=194.62, now=now)
    packet["generated_at"] = now.isoformat()
    packet["approval_ready_symbols"] = ["SPCX", "XLK"]
    packet["approval_ready_actions"] = [
        {
            "symbol": "SPCX",
            "action": "EXIT",
            "qty": 8.0,
            "est_notional": 1250.56,
            "est_price": 156.32,
        },
        {
            "symbol": "XLK",
            "action": "EXIT",
            "qty": 7.0,
            "est_notional": 1362.34,
            "est_price": 194.62,
        },
    ]
    packet["total_proposed_notional"] = 2612.90
    packet_path = tmp_path / READINESS_PACKET_NAME
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    before = packet_path.read_bytes()
    before_hash = hashlib.sha256(before).hexdigest()
    monkeypatch.setattr(
        "services.market_calendar.regular_session_state",
        lambda now=None: {"session_state": "OPEN", "session_date": "2026-08-11"},
    )

    denial = hold_arm_denial(json.loads(before.decode("utf-8")))
    presentation = build_presentation(
        store_root=tmp_path,
        results_root=tmp_path,
        now=now,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )
    armed = arm_reviewed_sells(
        store_root=tmp_path,
        results_root=tmp_path,
        now=now,
        paper_mode=True,
        live_blocked=True,
        db_path=tmp_path / "paper_sell_arm.sqlite3",
    )

    after = packet_path.read_bytes()
    assert denial == "basis_unproven_hold_symbol_in_packet"
    assert presentation["arm_enabled"] is False
    assert presentation["arm_deny_reason"] == "basis_unproven_hold_symbol_in_packet"
    assert armed.get("ok") is False
    assert armed.get("reason") == "basis_unproven_hold_symbol_in_packet"
    assert after == before
    assert hashlib.sha256(after).hexdigest() == before_hash


def test_operator_attention_names_the_active_entry_gate():
    from services.operator_console_status import _build_attention

    items = _build_attention(
        {
            "market": {"market_open": True, "session_phase": "MARKET_OPEN"},
            "governance": {
                "blocked": False,
                "execution_readiness_ready": False,
                "active_entry_blocker": {
                    "active": True,
                    "label": "ORANGE governance",
                    "detail": (
                        "Runtime entry gate: ORANGE governance. "
                        "The legacy certification banner is not this order gate."
                    ),
                    "certification_is_runtime_gate": False,
                },
            },
        }
    )
    titles = [item.title for item in items]
    entry = next(item for item in items if item.classification_code == "ACTIVE_ENTRY_BLOCKER")
    assert "Entry blocked: ORANGE governance" in titles
    assert "certification" not in entry.title.lower()
    assert "legacy certification banner is not this order gate" in entry.explanation
    assert entry.recommended_action.startswith("Use this runtime entry gate.")


def test_operator_labels_name_the_runtime_gate_not_certification():
    from services.governance_degraded_mode import describe_active_entry_blocker

    orange = describe_active_entry_blocker(
        {
            "readiness_status": "PARTIALLY_READY",
            "governance_mode": "ORANGE_DEGRADED",
            "failed_checks": ["governance"],
            "certification_status": "NOT_CERTIFIED",
        }
    )
    assert orange["label"] == "ORANGE governance"
    assert orange["certification_is_runtime_gate"] is False

    watchdog = describe_active_entry_blocker(
        {
            "readiness_status": "NOT_READY",
            "governance_mode": "ORANGE_DEGRADED",
            "failed_checks": ["governance", "watchdog"],
            "certification_status": "NOT_CERTIFIED",
        }
    )
    assert watchdog["label"] == "watchdog"

    capacity = describe_active_entry_blocker(
        {"readiness_status": "READY", "failed_checks": []},
        {"gate_reason": "MAX_POSITIONS", "details": {"failed_check": "capacity"}},
    )
    assert capacity["label"] == "capacity"

    current_orange = describe_active_entry_blocker(
        {
            "readiness_status": "PARTIALLY_READY",
            "governance_mode": "ORANGE_DEGRADED",
            "failed_checks": ["governance"],
        },
        {
            "gate_reason": "execution_readiness=NOT_READY",
            "details": {"checks": {"readiness_failed_checks": ["governance", "watchdog"]}},
        },
    )
    assert current_orange["label"] == "ORANGE governance"

    clear = describe_active_entry_blocker(
        {"readiness_status": "READY", "failed_checks": [], "certification_status": "NOT_CERTIFIED"}
    )
    assert clear["active"] is False
    assert clear["label"] == "none"
