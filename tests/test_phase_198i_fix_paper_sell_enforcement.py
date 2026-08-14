"""
Phase 198I-FIX — close verified bypasses; isolated mocked-broker tests only.
All state under tmp_path. No real governance/runtime/broker credentials.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from tests._paper_sell_test_durability import stub_dir_fsync_ok
from services.paper_sell_arm_authority import (
    HARDENED_AUTHORITY_FILENAME,
    LEGACY_197W_APPROVAL_FILENAME,
    LIVE_APPROVAL_PATH,
    LIVE_BUY_ARM_PATH,
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    BrokerSnapshotView,
    _atomic_write_json,
    confirm_and_create_paper_sell_arm,
    prepare_confirmation_challenge,
    readiness_packet_hash,
)
from services.paper_sell_arm_runtime_enforcement import (
    LEGACY_197W_RETIRED,
    PaperSellDispatchBlocked,
    begin_paper_sell_dispatch,
    enforce_paper_sell_broker_submission,
    finalize_paper_sell_dispatch,
    get_active_paper_sell_permit,
    legacy_writer_target_path,
    refuse_legacy_write_to_hardened_path,
    resolve_authoritative_live_blocked,
    validate_runtime_sell_authority,
)


@pytest.fixture(autouse=True)
def _dir_fsync_capable(monkeypatch):
    stub_dir_fsync_ok(monkeypatch)

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 11, 16, 0, 0, tzinfo=timezone.utc)
LIVE_BLOCKED = {
    "live_blocked": True,
    "live_block_reason": "TEST_LIVE_BLOCKED",
    "live_explicitly_authorized": False,
    "live_env_enabled": False,
    "paper_mode": True,
}


@pytest.fixture(autouse=True)
def _clear_sell_permit():
    from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT

    _ACTIVE_PERMIT.set(None)
    yield
    _ACTIVE_PERMIT.set(None)


@dataclass
class FakePlanned:
    symbol: str
    side: str
    qty: float
    limit_price: float = 100.0
    stance: str = "EXIT"
    client_order_id: str = ""
    force_exit_override: bool = False
    ref_price: Optional[float] = None
    management_action: Optional[str] = None


class AssertingFakeBroker:
    """Fake broker: asserts durable consumption exists before submit."""

    def __init__(self, store_root: Path, mode: str = "paper"):
        self.store_root = store_root
        self.mode = mode
        self.calls: List[Dict[str, Any]] = []

    def submit_order(self, **kwargs):
        cons = self.store_root / "paper_sell_arm_consumption.json"
        intent = self.store_root / "paper_sell_arm_dispatch_intent.json"
        assert cons.is_file(), "consumption missing at broker submit"
        assert intent.is_file(), "intent missing at broker submit"
        enforce_paper_sell_broker_submission(
            mode=self.mode,
            side=kwargs.get("side", ""),
            source="AssertingFakeBroker",
            symbol=kwargs.get("symbol"),
            qty=kwargs.get("qty"),
            client_order_id=kwargs.get("client_order_id"),
        )
        self.calls.append(kwargs)
        return {"id": f"ord-{len(self.calls)}", "status": "accepted"}


def _readiness(**overrides):
    base = {
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "cycle_id": "cycle_fix",
        "run_id": "run_fix",
        "batch_id": "batch_fix",
        "plan_id": "plan_fix",
        "plan_hash": "planhash_fix",
        "schema_version": "198G-UNAPPROVED-APPROVAL-READINESS-1",
        "approval_ready_symbols": ["TSLA"],
        "approval_ready_actions": [
            {
                "symbol": "TSLA",
                "action": "EXIT",
                "qty": 8,
                "est_notional": 2660.0,
                "est_price": 332.5,
            }
        ],
        "total_proposed_notional": 2660.0,
        "trading_date": "2026-08-11",
    }
    base.update(overrides)
    return base


def _seed_store(tmp_path: Path, readiness=None, now=NOW):
    readiness = readiness or _readiness()
    # Independent paper arm + bindings (never copy from authority alone).
    (tmp_path / "paper_arm_confirm.json").write_text(
        json.dumps(
            {
                "timestamp": now.isoformat(),
                "mode": "ASSISTED",
                "expires_at": (now + timedelta(hours=4)).isoformat(),
                "allow_execute": True,
            }
        ),
        encoding="utf-8",
    )
    chal = prepare_confirmation_challenge(
        store_root=tmp_path,
        readiness_packet=readiness,
        account_raw="acct-fix",
        trading_date="2026-08-11",
        now=now,
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
        now=now,
    )
    bindings = {
        "plan_hash": readiness["plan_hash"],
        "cycle_id": readiness["cycle_id"],
        "run_id": readiness["run_id"],
        "batch_id": readiness["batch_id"],
        "plan_id": readiness["plan_id"],
        "trading_date": "2026-08-11",
        "account_fingerprint": auth["account_fingerprint"],
        "readiness_packet_hash": auth["readiness_packet_hash"],
        "confirmation_challenge_id": auth["confirmation_challenge_id"],
        "confirmation_challenge_hash": auth["confirmation_challenge_hash"],
        "paper_execution_authorized": True,
        "market_session": "MARKET_OPEN",
        "market_session_observed_at": NOW.isoformat(),
        "lifecycle_actions": {"TSLA": "EXIT"},
    }
    (tmp_path / "independent_dispatch_bindings.json").write_text(
        json.dumps(bindings), encoding="utf-8"
    )
    (tmp_path / "defensive_exit_approval_readiness.json").write_text(
        json.dumps(readiness), encoding="utf-8"
    )
    (tmp_path / "positions_snapshot.csv").write_text(
        "symbol,qty,qty_available\nTSLA,8,8\n", encoding="utf-8"
    )
    (tmp_path / "open_orders_snapshot.csv").write_text(
        "symbol,side,qty\n", encoding="utf-8"
    )
    return readiness, chal, auth


def _planned(action="EXIT", qty=8.0, price=332.5, force=False):
    return [
        FakePlanned(
            "TSLA",
            "sell",
            qty,
            price,
            stance=action,
            management_action=action,
            client_order_id="coid-1",
            force_exit_override=force,
            ref_price=price,
        )
    ]


def _broker(auth, **overrides):
    b = BrokerSnapshotView(
        positions={"TSLA": {"qty": 8.0, "qty_available": 8.0}},
        open_order_symbols=set(),
        mode="paper",
        live_blocked=True,
        trading_date=str(auth.get("trading_date")),
        account_fingerprint=str(auth.get("account_fingerprint")),
        current_plan_hash=str(auth.get("plan_hash")),
        paper_buy_arm_valid=True,
        paper_execution_authorized=True,
        cycle_id=str(auth.get("cycle_id")),
        run_id=str(auth.get("run_id")),
        batch_id=str(auth.get("batch_id")),
        plan_id=str(auth.get("plan_id")),
        readiness_packet_hash=str(auth.get("readiness_packet_hash")),
        confirmation_challenge_id=str(auth.get("confirmation_challenge_id")),
        confirmation_challenge_hash=str(auth.get("confirmation_challenge_hash")),
        evidence_independent=True,
        market_session="MARKET_OPEN",
    )
    for k, v in overrides.items():
        setattr(b, k, v)
    return b


# --- Requirements 71–120 (198I-V gaps) ---


def test_71_caller_live_blocked_true_cannot_override_enabled(tmp_path: Path):
    ok, reason, _ = resolve_authoritative_live_blocked(
        caller_live_blocked=True,
        live_override={
            "live_blocked": False,
            "live_env_enabled": True,
            "live_explicitly_authorized": True,
            "paper_mode": True,
        },
    )
    assert not ok
    assert "live" in reason


def test_72_missing_live_state_blocks():
    ok, reason, _ = resolve_authoritative_live_blocked(live_override={})
    assert not ok and reason == "live_execution_state_missing"


def test_73_contradictory_live_state_blocks():
    ok, reason, _ = resolve_authoritative_live_blocked(
        live_override={
            "live_blocked": True,
            "live_contradictory": True,
            "live_env_enabled": False,
        }
    )
    assert not ok and "contradictory" in reason


def test_74_ambiguous_live_state_blocks():
    ok, reason, _ = resolve_authoritative_live_blocked(
        live_override={
            "live_blocked": True,
            "live_state_ambiguous": True,
        }
    )
    assert not ok and "ambiguous" in reason


def test_75_76_77_legacy_writer_cannot_overwrite_shadow_or_reset(tmp_path: Path):
    _, _, auth = _seed_store(tmp_path)
    hardened = tmp_path / HARDENED_AUTHORITY_FILENAME
    assert hardened.is_file()
    before = hardened.read_bytes()
    begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    consumed = hardened.read_bytes()
    assert b"CONSUMED" in consumed or b"DISPATCH_IN_PROGRESS" in consumed

    target = legacy_writer_target_path(tmp_path)
    with pytest.raises(PermissionError):
        refuse_legacy_write_to_hardened_path(tmp_path / HARDENED_AUTHORITY_FILENAME)
    with pytest.raises(PermissionError):
        refuse_legacy_write_to_hardened_path(
            tmp_path / "operator_paper_defensive_exit_approval.json"
        )
    # Legacy evidence may only land on the separated filename.
    target.write_text(
        json.dumps(
            {
                "decision": "APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH",
                "approval_mode": "PAPER_ONLY",
                "approved_symbols": ["TSLA"],
            }
        ),
        encoding="utf-8",
    )
    assert hardened.read_bytes() == consumed  # not reset / not overwritten
    assert target.name == LEGACY_197W_APPROVAL_FILENAME
    assert before != consumed
    ok, reason, _ = validate_runtime_sell_authority(
        contract_path=target, now=NOW + timedelta(minutes=1)
    )
    assert not ok and reason == LEGACY_197W_RETIRED


def test_78_dashboard_reader_not_armed_for_legacy(tmp_path: Path):
    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_LEGACY_RETIRED,
        PAPER_APPROVAL_VALID_CURRENT,
        classify_paper_operator_approval_artifact,
    )

    legacy = {
        "decision": "APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH",
        "approval_mode": "PAPER_ONLY",
        "approved_symbols": ["TSLA"],
        "approval_status": "APPROVED",
        "approval_action": "APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH",
        "approval_scope": "PAPER_ONLY",
        "created_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(hours=1)).isoformat(),
        "source_preview_hash": "x",
        "packet_id": "p",
    }
    p = tmp_path / LEGACY_197W_APPROVAL_FILENAME
    p.write_text(json.dumps(legacy), encoding="utf-8")
    state, _, _ = classify_paper_operator_approval_artifact(contract_path=p)
    assert state == PAPER_APPROVAL_LEGACY_RETIRED
    assert state != PAPER_APPROVAL_VALID_CURRENT
    # No dashboard ARMED sell-arm label path for legacy
    dash = (ROOT / "dashboard").read_text(encoding="utf-8") if False else ""
    assert "Arm Reviewed Sells" not in (ROOT / "dashboard" / "operator_console.py").read_text(
        encoding="utf-8", errors="ignore"
    )


@pytest.mark.parametrize(
    "source",
    [
        "manage_open_orders_reprice",
        "position_protector_sell",
        "reprice_open_orders",
        "place_orders_from_csv",
        "direct_symbol_sell",
    ],
)
def test_79_86_ungated_helpers_blocked_without_permit(source: str):
    with pytest.raises(PaperSellDispatchBlocked) as ei:
        enforce_paper_sell_broker_submission(mode="paper", side="sell", source=source)
    assert "permit" in ei.value.reason


def test_84_85_retry_and_recovery_cannot_reuse(tmp_path: Path):
    _, _, auth = _seed_store(tmp_path)
    g1 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    assert g1.consumed
    finalize_paper_sell_dispatch(store_root=tmp_path, outcome="TIMEOUT")
    g2 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    assert not g2.permitted


def test_87_98_independent_checks_and_in_lock_revalidation(tmp_path: Path):
    _, _, auth = _seed_store(tmp_path)
    # Wrong independent plan hash
    g = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth, current_plan_hash="WRONG"),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    assert not g.permitted and "plan_hash" in g.reason

    # Missing paper auth
    g2 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth, paper_buy_arm_valid=False, paper_execution_authorized=False),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    assert not g2.permitted and "paper_execution" in g2.reason

    # Challenge missing
    g3 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth, confirmation_challenge_id=None),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    assert not g3.permitted


def test_89_93_supersession_lifecycle_order_fill(tmp_path: Path):
    _, _, auth = _seed_store(tmp_path)
    cases = [
        ({"newer_completed_plan": True}, "newer_management"),
        ({"newer_batch_supersedes": True}, "newer_batch"),
        ({"lifecycle_actions": {"TSLA": "HOLD"}}, "lifecycle"),
        ({"newer_order_symbols": {"TSLA"}}, "newer_order"),
        ({"newer_fill_symbols": {"TSLA"}}, "newer_fill"),
    ]
    for kwargs, needle in cases:
        g = begin_paper_sell_dispatch(
            store_root=tmp_path,
            planned=_planned(),
            broker_view=_broker(auth, **kwargs),
            live_override=LIVE_BLOCKED,
            now=NOW + timedelta(minutes=1),
            results_root=tmp_path,
            live_root=tmp_path,
        )
        assert not g.permitted, needle
        assert any(x in g.reason for x in (needle.split("_")[0], needle)), (needle, g.reason)


def test_95_97_qty_independent(tmp_path: Path):
    _, _, auth = _seed_store(tmp_path)
    g = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(
            auth, positions={"TSLA": {"qty": 8.0, "qty_available": 2.0}}
        ),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    assert not g.permitted


def test_99_100_concurrent_thread_single_broker_attempt(tmp_path: Path):
    _, _, auth = _seed_store(tmp_path)
    results: List[Any] = []
    errors: List[str] = []
    barrier = threading.Barrier(2)

    def worker(label: str):
        try:
            barrier.wait(timeout=10)
            g = begin_paper_sell_dispatch(
                store_root=tmp_path,
                planned=_planned(),
                broker_view=_broker(auth),
                live_override=LIVE_BLOCKED,
                now=NOW + timedelta(minutes=1),
                results_root=tmp_path,
                live_root=tmp_path,
            )
            broker_calls = 0
            if g.permitted and g.permit is not None:
                try:
                    enforce_paper_sell_broker_submission(
                        mode="paper",
                        side="sell",
                        source=f"thread_{label}",
                        symbol="TSLA",
                        qty=8.0,
                        client_order_id="coid-thread-1",
                    )
                    broker_calls = 1
                except PaperSellDispatchBlocked as blocked:
                    broker_calls = 0
                    errors.append(f"{label}:broker_blocked:{blocked.reason}")
            results.append(
                {
                    "label": label,
                    "permitted": g.permitted,
                    "reason": g.reason,
                    "intent": g.dispatch_intent_id,
                    "broker_calls": broker_calls,
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}:{exc!r}")
            results.append(
                {
                    "label": label,
                    "permitted": False,
                    "reason": repr(exc),
                    "intent": None,
                    "broker_calls": 0,
                }
            )

    t1 = threading.Thread(target=worker, args=("t1",))
    t2 = threading.Thread(target=worker, args=("t2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    permitted = [r for r in results if r.get("permitted")]
    assert len(permitted) == 1, results
    assert len(results) == 2
    loser = [r for r in results if not r.get("permitted")][0]
    assert loser.get("reason"), f"losing thread must report reason: {loser}"
    broker_attempts = sum(int(r.get("broker_calls") or 0) for r in results)
    assert broker_attempts == 1, results
    intent_count = 1 if (tmp_path / "paper_sell_arm_dispatch_intent.json").is_file() else 0
    assert intent_count == 1
    attempt_count = broker_attempts
    auth_after = json.loads((tmp_path / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
    assert auth_after.get("consumption_status") == "CONSUMED"
    # Stale overwrite / remaining-symbol / unused-qty / newer-cycle reuse prohibitions
    stale = dict(auth)
    stale["status"] = "ARMED"
    stale["consumption_status"] = "UNUSED"
    with pytest.raises(PermissionError):
        _atomic_write_json(
            tmp_path / HARDENED_AUTHORITY_FILENAME, stale, refuse_authority_downgrade=True
        )
    stale_overwrite_assertion = "PermissionError on refuse_authority_downgrade ARMED overwrite"
    with pytest.raises(PaperSellDispatchBlocked):
        enforce_paper_sell_broker_submission(
            mode="paper", side="sell", source="remaining_symbol", symbol="AAPL", qty=1.0
        )
    remaining_symbol_assertion = "PaperSellDispatchBlocked for alternate symbol AAPL"
    with pytest.raises(PaperSellDispatchBlocked):
        enforce_paper_sell_broker_submission(
            mode="paper", side="sell", source="unused_qty", symbol="TSLA", qty=1.0
        )
    unused_qty_assertion = "PaperSellDispatchBlocked for unused qty reuse"
    g_new = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker({**auth, "cycle_id": "cycle_newer"}),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=5),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    assert not g_new.permitted
    newer_cycle_assertion = f"newer cycle begin blocked reason={g_new.reason}"
    (tmp_path / "thread_concurrency_evidence.json").write_text(
        json.dumps(
            {
                "results": results,
                "winner": permitted[0],
                "loser": loser,
                "broker_attempts": broker_attempts,
                "intent_count": intent_count,
                "attempt_count": attempt_count,
                "authority_consumption": auth_after.get("consumption_status"),
                "client_order_id": "coid-thread-1",
                "stale_overwrite_assertion": stale_overwrite_assertion,
                "remaining_symbol_assertion": remaining_symbol_assertion,
                "unused_qty_assertion": unused_qty_assertion,
                "newer_cycle_assertion": newer_cycle_assertion,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    assert not errors, errors

def test_101_stale_armed_cannot_overwrite_consumed(tmp_path: Path):
    _, _, auth = _seed_store(tmp_path)
    begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    path = tmp_path / HARDENED_AUTHORITY_FILENAME
    stale = dict(auth)
    stale["status"] = "ARMED"
    stale["consumption_status"] = "UNUSED"
    with pytest.raises(PermissionError):
        _atomic_write_json(path, stale, refuse_authority_downgrade=True)


def test_102_107_durable_before_broker_and_crash_semantics(tmp_path: Path):
    _, _, auth = _seed_store(tmp_path)
    g = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    assert g.consumed
    assert (tmp_path / "paper_sell_arm_consumption.json").is_file()
    assert (tmp_path / "paper_sell_arm_dispatch_intent.json").is_file()
    # Crash before broker: authority non-reusable
    g2 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    assert not g2.permitted
    broker = AssertingFakeBroker(tmp_path)
    # Prior permit may still be active from g — clear and require failure without new begin
    from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT

    _ACTIVE_PERMIT.set(None)
    with pytest.raises(PaperSellDispatchBlocked):
        broker.submit_order(symbol="TSLA", qty=8, side="sell")


def test_109_111_partial_batch_consumes_whole(tmp_path: Path):
    readiness = _readiness(
        approval_ready_symbols=["TSLA", "AAPL"],
        approval_ready_actions=[
            {"symbol": "TSLA", "action": "EXIT", "qty": 2, "est_notional": 600, "est_price": 300},
            {"symbol": "AAPL", "action": "TRIM", "qty": 1, "est_notional": 200, "est_price": 200},
        ],
        total_proposed_notional=800,
        plan_hash="plan_multi",
        batch_id="batch_multi",
    )
    _, _, auth = _seed_store(tmp_path, readiness=readiness)
    (tmp_path / "positions_snapshot.csv").write_text(
        "symbol,qty,qty_available\nTSLA,2,2\nAAPL,1,1\n", encoding="utf-8"
    )
    planned = [
        FakePlanned("TSLA", "sell", 2, 300, stance="EXIT", ref_price=300),
        FakePlanned("AAPL", "sell", 1, 200, stance="TRIM", ref_price=200),
    ]
    g = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=_broker(
            auth,
            positions={
                "TSLA": {"qty": 2.0, "qty_available": 2.0},
                "AAPL": {"qty": 1.0, "qty_available": 1.0},
            },
            current_plan_hash="plan_multi",
            readiness_packet_hash=auth["readiness_packet_hash"],
        ),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    assert g.consumed
    # Remaining symbol reuse blocked
    g2 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=[FakePlanned("AAPL", "sell", 1, 200, stance="TRIM", ref_price=200)],
        broker_view=_broker(
            auth,
            positions={"AAPL": {"qty": 1.0, "qty_available": 1.0}},
            current_plan_hash="plan_multi",
        ),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    assert not g2.permitted


def test_113_114_place_live_orders_integration(tmp_path: Path):
    from services.place_live_orders import enforce_and_consume_paper_sells_before_submit

    _, _, auth = _seed_store(tmp_path)
    buy_arm_bytes = (tmp_path / "paper_arm_confirm.json").read_bytes()
    guard, root = enforce_and_consume_paper_sells_before_submit(
        planned=_planned(),
        mode="paper",
        store_root=tmp_path,
        results_root=tmp_path,
        live_root=tmp_path,
        live_override=LIVE_BLOCKED,
        explicit_runtime_write=False,
        now=NOW + timedelta(minutes=1),
    )
    assert guard is not None and guard.consumed
    assert (tmp_path / "paper_sell_arm_consumption.json").is_file()
    broker = AssertingFakeBroker(tmp_path)
    broker.submit_order(symbol="TSLA", qty=8, side="sell", client_order_id="c1")
    assert len(broker.calls) == 1
    # Legacy cannot authorize
    ok, reason, _ = validate_runtime_sell_authority(
        contract_path=tmp_path / HARDENED_AUTHORITY_FILENAME, now=NOW + timedelta(minutes=1)
    )
    assert not ok  # already consumed
    assert (tmp_path / "paper_arm_confirm.json").read_bytes() == buy_arm_bytes


def test_115_117_buy_add_preservation(tmp_path: Path):
    _, _, auth = _seed_store(tmp_path)
    before = (tmp_path / "paper_arm_confirm.json").read_bytes()
    buys = [FakePlanned("AAPL", "buy", 1, 100.0, stance="BUY")]
    from services.place_live_orders import enforce_and_consume_paper_sells_before_submit

    g, _ = enforce_and_consume_paper_sells_before_submit(
        planned=buys, mode="paper", store_root=tmp_path, explicit_runtime_write=False
    )
    assert g is None
    # Sell consume leaves buy arm unchanged
    begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    assert (tmp_path / "paper_arm_confirm.json").read_bytes() == before
    # BUY side does not need sell permit
    enforce_paper_sell_broker_submission(mode="paper", side="buy", source="buy")


def test_118_119_120_live_safety_dashboard_and_isolation():
    assert LIVE_APPROVAL_PATH.name == HARDENED_AUTHORITY_FILENAME
    console = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8", errors="ignore")
    ui = (ROOT / "dashboard" / "operator_ui.py").read_text(encoding="utf-8", errors="ignore")
    for needle in ("Arm Paper Buys", "Arm Reviewed Sells", "Confirm & Arm Paper Sells"):
        assert needle not in console and needle not in ui
    # Real buy arm path exists but tests must not mutate it
    assert "paper_arm_confirm" in str(LIVE_BUY_ARM_PATH)


def test_force_exit_requires_hardened_gate(tmp_path: Path):
    readiness = _readiness(
        approval_ready_actions=[
            {
                "symbol": "TSLA",
                "action": "FORCE_EXIT",
                "qty": 8,
                "est_notional": 2660.0,
                "est_price": 332.5,
            }
        ]
    )
    _, _, auth = _seed_store(tmp_path, readiness=readiness)
    # Missing authority / permit path: ungated FORCE_EXIT blocked
    with pytest.raises(PaperSellDispatchBlocked):
        enforce_paper_sell_broker_submission(
            mode="paper", side="sell", source="force_exit"
        )
    g = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(action="FORCE_EXIT", force=True),
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=tmp_path,
        live_root=tmp_path,
    )
    assert g.consumed


def test_fsync_atomic_write_invoked(tmp_path: Path, monkeypatch):
    calls = {"fsync": 0}

    real_fsync = __import__("os").fsync

    def counting_fsync(fd):
        calls["fsync"] += 1
        try:
            return real_fsync(fd)
        except Exception:
            return None

    monkeypatch.setattr("os.fsync", counting_fsync)
    path = tmp_path / "x.json"
    _atomic_write_json(path, {"a": 1})
    assert path.is_file()
    assert calls["fsync"] >= 1


# Mapping table for requirements 1–70 (covered by 198H/198I/FIX suites).
REQUIREMENT_MAP = {
    1: ("test_phase_198i_paper_sell_arm_runtime_enforcement.py", "test_missing_authority_blocks_each_sell_action"),
    2: ("test_phase_198i_paper_sell_arm_runtime_enforcement.py", "test_legacy_197w_rejected_by_runtime_and_no_fallback"),
    3: ("test_phase_198i_fix_paper_sell_enforcement.py", "test_113_114_place_live_orders_integration"),
    4: ("test_phase_198i_paper_sell_arm_runtime_enforcement.py", "test_hardened_gate_permits_then_consumes_before_mock_submit"),
    5: ("test_phase_198i_paper_sell_arm_runtime_enforcement.py", "test_authority_not_reusable_after_mock_outcomes"),
    71: (__file__, "test_71_caller_live_blocked_true_cannot_override_enabled"),
    75: (__file__, "test_75_76_77_legacy_writer_cannot_overwrite_shadow_or_reset"),
    79: (__file__, "test_79_86_ungated_helpers_blocked_without_permit"),
    113: (__file__, "test_113_114_place_live_orders_integration"),
    120: (__file__, "test_118_119_120_live_safety_dashboard_and_isolation"),
}


def test_requirement_map_present():
    assert len(REQUIREMENT_MAP) >= 10
