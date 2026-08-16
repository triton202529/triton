"""Phase 198K — market-open invocation handoff (PENDING→READY, stale-replay hardened).

PAPER ONLY. Isolated tmp paths. No production arm/SELL/runtime writes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pytest

from services.manage_positions_cycle_binding import (
    BLOCKER_MISSING_SELL_AUTHORITY,
    BLOCKER_NONE,
    DECISION_BLOCK,
    DECISION_MAY_CONTINUE,
    INVOCATION_ENV,
    REASON_INVOCATION_MISMATCH,
    SCOPE_GLOBAL_SAFETY_BLOCK,
    SCOPE_NONE,
    evaluate_exit_before_entry,
    write_cycle_binding,
    write_entry_interlock,
)
from services.paper_entry_loop_stage import (
    HANDOFF_SCHEMA_VERSION,
    REASON_INVOCATION_HANDOFF_BINDING,
    REASON_INVOCATION_HANDOFF_NOT_READY,
    REASON_INVOCATION_HANDOFF_SESSION,
    REASON_INVOCATION_HANDOFF_STALE,
    REASON_INVOCATION_MALFORMED,
    REASON_INVOCATION_MISSING,
    REASON_INVOCATION_SOURCE_MISMATCH,
    _current_session_date_et,
    evaluate_paper_entry_gates,
    resolve_entry_invocation_id,
    run_paper_entry_stage,
    validate_market_open_invocation_handoff,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now(timezone.utc) - timedelta(minutes=5)
ET = ZoneInfo("America/New_York")
REPO_ARM = ROOT / "data" / "live" / "paper_arm_confirm.json"
REPO_SELL = ROOT / "data" / "live" / "paper_sell_arm.sqlite3"


@pytest.fixture(autouse=True)
def _safety(monkeypatch):
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    monkeypatch.delenv(INVOCATION_ENV, raising=False)
    for key in list(os.environ):
        if "ALPACA" in key.upper() or key.startswith("APCA_"):
            monkeypatch.delenv(key, raising=False)
    before = REPO_ARM.read_bytes() if REPO_ARM.is_file() else None
    yield
    after = REPO_ARM.read_bytes() if REPO_ARM.is_file() else None
    assert after == before
    assert not REPO_SELL.is_file()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_plan(results: Path) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / "manage_positions_plan.json").write_text(
        json.dumps({"timestamp": NOW.isoformat(), "mode": "paper", "rows": []}),
        encoding="utf-8",
    )
    (results / "manage_positions_plan.csv").write_text(
        "symbol,final_action,qty,selected_for_execution,reason_code,skip_reason,"
        "current_position_qty,current_price,planned_notional\n",
        encoding="utf-8",
    )


def _bind_cycle(
    results: Path,
    invocation_id: str,
    *,
    remaining: str = BLOCKER_NONE,
    decision: str = DECISION_MAY_CONTINUE,
    packet_status: str = "VALID_EMPTY",
    reason: str = "ok",
    outcome_unknown: bool = False,
    now: datetime = NOW,
) -> Path:
    _write_plan(results)
    write_cycle_binding(
        results_root=results,
        invocation_id=invocation_id,
        plan_generated_at=now.isoformat(),
        f_stage_status="COMPLETE",
        now=now,
        remaining_blocker=remaining,
        operator_execution_required=remaining != BLOCKER_NONE,
        block_reasons=[] if remaining == BLOCKER_NONE else ["DISPATCH_BLOCKED_APPROVAL_REQUIRED"],
        outcome_unknown=outcome_unknown,
    )
    write_entry_interlock(
        results_root=results,
        invocation_id=invocation_id,
        now=now,
        decision=decision,
        reason=reason,
        packet_status=packet_status,
        interlock_scope=SCOPE_GLOBAL_SAFETY_BLOCK if decision == DECISION_BLOCK else SCOPE_NONE,
        global_block_reason=reason if decision == DECISION_BLOCK else None,
    )
    return results / "paper_sell_exit_before_entry.json"


def _write_handoff(
    results: Path,
    invocation_id: str,
    *,
    status: str = "READY",
    session_date: Optional[str] = None,
    issued_at: Optional[datetime] = None,
    ready_at: Optional[datetime] = None,
    interlock_sha256: Optional[str] = None,
    schema_version: int = HANDOFF_SCHEMA_VERSION,
) -> Path:
    interlock = results / "paper_sell_exit_before_entry.json"
    issued = issued_at or NOW
    ready = ready_at or NOW
    payload = {
        "schema_version": schema_version,
        "status": status,
        "market_open_invocation_id": invocation_id,
        "session_date": session_date or _current_session_date_et(now=datetime.now(timezone.utc)),
        "issued_at": issued.isoformat(),
        "ready_at": ready.isoformat() if status == "READY" else None,
        "generated_at": (ready if status == "READY" else issued).isoformat(),
        "interlock_sha256": (
            interlock_sha256
            if interlock_sha256 is not None
            else (_sha(interlock) if interlock.is_file() and status == "READY" else None)
        ),
        "f_binding_sha256": (
            _sha(results / "manage_positions_cycle_binding.json")
            if (results / "manage_positions_cycle_binding.json").is_file() and status == "READY"
            else None
        ),
        "source": "scripts/run_market_open.ps1",
        "creates_no_authority": True,
        "live_execution_permitted": False,
        "broker_orders_submitted": False,
    }
    path = results / "market_open_invocation_handoff.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_1_exact_current_invocation_via_ready_handoff(tmp_path, monkeypatch):
    inv = "mo-20260817-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    _bind_cycle(tmp_path, inv)
    _write_handoff(tmp_path, inv, status="READY")
    monkeypatch.setattr("services.paper_entry_loop_stage.RESULTS", tmp_path)
    resolved, err = resolve_entry_invocation_id(
        results_root=tmp_path, now=datetime.now(timezone.utc)
    )
    assert err is None
    assert resolved == inv
    _ok, reason, details = evaluate_paper_entry_gates(verbose=False, results_root=tmp_path)
    assert reason != REASON_INVOCATION_MISSING
    assert details["checks"]["exit_before_entry"]["decision"] == DECISION_MAY_CONTINUE


def test_2_missing_entry_invocation_fail_closed(tmp_path, monkeypatch):
    inv = "mo-20260817-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    _bind_cycle(tmp_path, inv)
    monkeypatch.setattr("services.paper_entry_loop_stage.RESULTS", tmp_path)
    resolved, err = resolve_entry_invocation_id(results_root=tmp_path)
    assert resolved is None
    assert err == REASON_INVOCATION_MISSING
    ok, reason, details = evaluate_paper_entry_gates(verbose=False, results_root=tmp_path)
    assert ok is False
    assert reason == REASON_INVOCATION_MISSING
    assert details.get("safe_block") is True


def test_3_explicit_differs_from_env(tmp_path, monkeypatch):
    inv = "mo-20260817-cccccccccccccccccccccccccccccccc"
    _bind_cycle(tmp_path, inv)
    _write_handoff(tmp_path, inv, status="READY")
    monkeypatch.setenv(INVOCATION_ENV, inv)
    resolved, err = resolve_entry_invocation_id(
        explicit="mo-other-dddddddddddddddddddddddddddddddd",
        results_root=tmp_path,
    )
    assert resolved is None
    assert err == REASON_INVOCATION_SOURCE_MISMATCH


def test_4_entry_differs_from_ff2_interlock(tmp_path, monkeypatch):
    _bind_cycle(tmp_path, "mo-20260817-eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    other = "mo-20260817-ffffffffffffffffffffffffffffffff"
    # READY handoff bound to current interlock bytes but different invocation id field
    path = _write_handoff(tmp_path, other, status="READY")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["market_open_invocation_id"] = other
    path.write_text(json.dumps(doc), encoding="utf-8")
    ok, reason, details = evaluate_paper_entry_gates(verbose=False, results_root=tmp_path)
    assert ok is False
    inter = details["checks"].get("exit_before_entry") or {}
    assert (
        inter.get("reason") == REASON_INVOCATION_MISMATCH
        or reason == REASON_INVOCATION_MISMATCH
        or "MISMATCH" in reason
    )


def test_5_stale_replay_prior_ready_after_failed_new_cycle(tmp_path, monkeypatch):
    """Prior READY+F2 match must not authorize after new cycle writes PENDING then fails."""
    prior = "mo-20260817-oldoldoldoldoldoldoldoldoldoldol"
    newer = "mo-20260817-newnewnewnewnewnewnewnewnewnewne"
    _bind_cycle(tmp_path, prior)
    _write_handoff(tmp_path, prior, status="READY")
    # Prove prior pair would resolve before supersession.
    ok_prior, err_prior = resolve_entry_invocation_id(
        results_root=tmp_path, now=datetime.now(timezone.utc)
    )
    assert err_prior is None and ok_prior == prior
    # New market_open attempt supersedes with PENDING then fails before READY/F2.
    _write_handoff(tmp_path, newer, status="PENDING")
    resolved, err = resolve_entry_invocation_id(results_root=tmp_path)
    assert resolved is None
    assert err == REASON_INVOCATION_HANDOFF_NOT_READY
    ok, reason, _details = evaluate_paper_entry_gates(verbose=False, results_root=tmp_path)
    assert ok is False
    assert reason == REASON_INVOCATION_HANDOFF_NOT_READY


def test_5b_rapid_retry_minutes_later_still_blocks(tmp_path, monkeypatch):
    prior = "mo-20260817-rapidrapidrapidrapidrapidrapidra"
    newer = "mo-20260817-retryretryretryretryretryretryre"
    recent = datetime.now(timezone.utc) - timedelta(minutes=3)
    _bind_cycle(tmp_path, prior, now=recent)
    _write_handoff(
        tmp_path,
        prior,
        status="READY",
        issued_at=recent,
        ready_at=recent,
    )
    _write_handoff(tmp_path, newer, status="PENDING", issued_at=datetime.now(timezone.utc))
    resolved, err = resolve_entry_invocation_id(results_root=tmp_path)
    assert resolved is None
    assert err == REASON_INVOCATION_HANDOFF_NOT_READY


def test_6_malformed_and_pending_and_stale(tmp_path, monkeypatch):
    inv = "mo-20260817-gggggggggggggggggggggggggggggggg"
    _bind_cycle(tmp_path, inv)
    _write_handoff(tmp_path, inv, status="PENDING")
    assert (
        resolve_entry_invocation_id(results_root=tmp_path)[1] == REASON_INVOCATION_HANDOFF_NOT_READY
    )

    bad = tmp_path / "market_open_invocation_handoff.json"
    bad.write_text("{not-json", encoding="utf-8")
    assert resolve_entry_invocation_id(results_root=tmp_path)[1] == REASON_INVOCATION_MALFORMED

    _write_handoff(tmp_path, inv, status="READY")
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    _write_handoff(tmp_path, inv, status="READY", issued_at=stale, ready_at=stale)
    assert resolve_entry_invocation_id(results_root=tmp_path)[1] == REASON_INVOCATION_HANDOFF_STALE


def test_6b_binding_and_session_fail_closed(tmp_path, monkeypatch):
    inv = "mo-20260817-bindbindbindbindbindbindbindbind"
    _bind_cycle(tmp_path, inv)
    _write_handoff(tmp_path, inv, status="READY", interlock_sha256="0" * 64)
    assert (
        resolve_entry_invocation_id(results_root=tmp_path)[1] == REASON_INVOCATION_HANDOFF_BINDING
    )
    _write_handoff(tmp_path, inv, status="READY", session_date="1999-01-01")
    assert (
        resolve_entry_invocation_id(results_root=tmp_path)[1] == REASON_INVOCATION_HANDOFF_SESSION
    )


def test_7_duplicate_stage_no_execute(tmp_path, monkeypatch):
    inv = "mo-20260817-hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh"
    _bind_cycle(tmp_path, inv)
    _write_handoff(tmp_path, inv, status="READY")
    monkeypatch.setattr("services.paper_entry_loop_stage.RESULTS", tmp_path)
    monkeypatch.setattr("services.paper_entry_loop_stage.LIVE", tmp_path / "live")
    (tmp_path / "live").mkdir(parents=True, exist_ok=True)
    rc1 = run_paper_entry_stage(verbose=False, dry_run=True, market_open_invocation_id=inv)
    rc2 = run_paper_entry_stage(verbose=False, dry_run=True, market_open_invocation_id=inv)
    assert rc1 == 0 and rc2 == 0


def test_8_authority_ready_sell_blocks_entry(tmp_path, monkeypatch):
    inv = "mo-20260817-iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"
    _bind_cycle(
        tmp_path,
        inv,
        remaining=BLOCKER_MISSING_SELL_AUTHORITY,
        decision=DECISION_BLOCK,
        packet_status="AUTHORITY_READY",
        reason="ENTRY_BLOCKED_PENDING_SELL_ARM",
    )
    _write_handoff(tmp_path, inv, status="READY")
    ok, _reason, details = evaluate_paper_entry_gates(verbose=False, results_root=tmp_path)
    assert ok is False
    assert details["checks"]["exit_before_entry"]["decision"] == DECISION_BLOCK


def test_9_outcome_unknown_global_block(tmp_path, monkeypatch):
    inv = "mo-20260817-jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj"
    _bind_cycle(
        tmp_path,
        inv,
        decision=DECISION_BLOCK,
        packet_status="OUTCOME_UNKNOWN",
        reason="ENTRY_BLOCKED_OUTCOME_UNKNOWN",
        outcome_unknown=True,
    )
    _write_handoff(tmp_path, inv, status="READY")
    ok, _reason, details = evaluate_paper_entry_gates(verbose=False, results_root=tmp_path)
    assert ok is False
    assert details["checks"]["exit_before_entry"]["decision"] == DECISION_BLOCK


def test_10_static_ps1_two_phase_contract():
    src = (ROOT / "scripts" / "run_market_open.ps1").read_text(encoding="utf-8")
    assert 'Write-InvocationHandoff -Status "PENDING"' in src
    assert 'Write-InvocationHandoff -Status "READY"' in src
    assert "market_open_invocation_handoff.json" in src
    assert "interlock_sha256" in src or "InterlockSha256" in src
    pe = (ROOT / "services" / "paper_entry_loop_stage.py").read_text(encoding="utf-8")
    assert "REASON_INVOCATION_HANDOFF_NOT_READY" in pe
    assert "validate_market_open_invocation_handoff" in pe
    tree = ast.parse(pe)
    names = {getattr(n, "id", None) or getattr(n, "attr", None) for n in ast.walk(tree)}
    assert "AlpacaBroker" not in names
    assert "begin_paper_sell_dispatch" not in names


def test_11_buy_add_arm_contract_untouched():
    from services.set_paper_arm_confirm import CONFIRM_PATH, MAX_SUPPORTED_TTL_MINUTES

    assert CONFIRM_PATH.name == "paper_arm_confirm.json"
    assert MAX_SUPPORTED_TTL_MINUTES == 600
    assert not REPO_SELL.is_file()


def test_12_cli_env_ready_handoff_agree(tmp_path, monkeypatch):
    inv = "mo-20260817-kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk"
    _bind_cycle(tmp_path, inv)
    _write_handoff(tmp_path, inv, status="READY")
    monkeypatch.setenv(INVOCATION_ENV, inv)
    resolved, err = resolve_entry_invocation_id(explicit=inv, results_root=tmp_path)
    assert err is None
    assert resolved == inv


def test_13_never_adopts_id_from_interlock_alone(tmp_path, monkeypatch):
    inv = "mo-20260817-llllllllllllllllllllllllllllllll"
    _bind_cycle(tmp_path, inv)
    resolved, err = resolve_entry_invocation_id(results_root=tmp_path)
    assert resolved is None
    assert err == REASON_INVOCATION_MISSING


def test_14_schema_v1_legacy_not_accepted(tmp_path, monkeypatch):
    inv = "mo-20260817-mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"
    _bind_cycle(tmp_path, inv)
    path = tmp_path / "market_open_invocation_handoff.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "market_open_invocation_id": inv,
                "generated_at": NOW.isoformat(),
                "creates_no_authority": True,
            }
        ),
        encoding="utf-8",
    )
    assert resolve_entry_invocation_id(results_root=tmp_path)[1] == REASON_INVOCATION_MALFORMED


def test_15_live_permitted_flag_rejected(tmp_path, monkeypatch):
    inv = "mo-20260817-oooooooooooooooooooooooooooooooo"
    _bind_cycle(tmp_path, inv)
    path = _write_handoff(tmp_path, inv, status="READY")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["live_execution_permitted"] = True
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert resolve_entry_invocation_id(results_root=tmp_path)[1] == REASON_INVOCATION_MALFORMED
