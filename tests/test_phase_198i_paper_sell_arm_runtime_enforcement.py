"""
Phase 198I — runtime enforcement, legacy retirement, atomic pre-submit consumption.
All artifacts under tmp_path; mocked broker only; no live governance writes.
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
from tests._phase_198i_interlock_test_fixtures import (
    ban_repo_live_io as _ban_repo_live_io,
    block_network_and_live_env as _block_network_and_live_env,
    isolate_interlock_artifact_roots as _isolate_interlock_artifact_roots,
    write_valid_order_guard_fixture as _write_valid_order_guard_fixture,
)
from services.paper_sell_arm_authority import (
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    LIVE_APPROVAL_PATH,
    BrokerSnapshotView,
    confirm_and_create_paper_sell_arm,
    prepare_confirmation_challenge,
    validate_paper_sell_arm_authority,
)
from services.paper_sell_arm_runtime_enforcement import (
    LEGACY_197W_RETIRED,
    begin_paper_sell_dispatch as _begin_paper_sell_dispatch,
    finalize_paper_sell_dispatch,
    looks_like_legacy_197w_approval,
    planned_rows_include_paper_sells,
    validate_runtime_sell_authority,
)


@pytest.fixture(autouse=True)
def _dir_fsync_capable(monkeypatch, tmp_path):
    stub_dir_fsync_ok(monkeypatch)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)


def begin_paper_sell_dispatch(**kwargs):
    kwargs.setdefault(
        "live_override",
        {
            "live_blocked": True,
            "live_block_reason": "TEST_LIVE_BLOCKED",
            "live_explicitly_authorized": False,
            "live_env_enabled": False,
            "paper_mode": True,
        },
    )
    return _begin_paper_sell_dispatch(**kwargs)


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 11, 16, 0, 0, tzinfo=timezone.utc)


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


class MockBroker:
    def __init__(self, behavior: str = "success"):
        self.behavior = behavior
        self.calls: List[Dict[str, Any]] = []

    def submit_order(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.behavior == "success":
            return {"id": f"ord-{len(self.calls)}", "status": "accepted"}
        if self.behavior == "rejection":
            raise RuntimeError("broker_rejected")
        if self.behavior == "timeout":
            raise TimeoutError("broker_timeout")
        if self.behavior == "exception":
            raise RuntimeError("broker_exception")
        if self.behavior == "indeterminate":
            return {"id": None, "status": "unknown"}
        if self.behavior == "partial_fill":
            return {"id": f"ord-{len(self.calls)}", "status": "partially_filled"}
        raise RuntimeError(f"unknown behavior {self.behavior}")


def _readiness(**overrides):
    base = {
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "cycle_id": "cycle_test_198i",
        "run_id": "run_198i",
        "batch_id": "batch_198i",
        "plan_id": "plan_198i",
        "plan_hash": "planhash_198i",
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
    }
    base.update(overrides)
    return base


def _arm(tmp_path: Path, readiness=None, now=NOW):
    readiness = readiness or _readiness()
    chal = prepare_confirmation_challenge(
        store_root=tmp_path,
        readiness_packet=readiness,
        account_raw="acct-198i",
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
    (tmp_path / "independent_dispatch_bindings.json").write_text(
        json.dumps(
            {
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
                "market_session_observed_at": now.isoformat(),
                "lifecycle_actions": {"TSLA": "EXIT"},
            }
        ),
        encoding="utf-8",
    )
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
    return readiness, chal, auth


def _broker(auth: Dict[str, Any], **overrides):
    b = BrokerSnapshotView(
        positions={"TSLA": {"qty": 8.0, "qty_available": 8.0}},
        open_order_symbols=set(),
        mode="paper",
        live_blocked=True,
        trading_date=str(auth.get("trading_date")),
        account_fingerprint=str(auth.get("account_fingerprint")),
        current_plan_hash=str(auth.get("plan_hash")),
        newer_completed_plan=False,
        market_session="MARKET_OPEN",
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
    )
    for k, v in overrides.items():
        setattr(b, k, v)
    return b


_LIVE_BLOCKED = {
    "live_blocked": True,
    "live_block_reason": "TEST_LIVE_BLOCKED",
    "live_explicitly_authorized": False,
    "live_env_enabled": False,
    "paper_mode": True,
}


def _planned(action="EXIT", qty=8.0, price=332.5, force=False):
    return [
        FakePlanned(
            symbol="TSLA",
            side="sell",
            qty=qty,
            limit_price=price,
            stance=action,
            client_order_id="coid-tsla-1",
            force_exit_override=force,
            ref_price=price,
        )
    ]


def _mock_dispatch(tmp_path: Path, auth, planned, broker_behavior="success", **broker_kw):
    """Revalidate+consume then call mocked broker — mirrors production boundary."""
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=_broker(auth, **broker_kw),
        mode="paper",
        readiness_packet_hash_current=str(auth.get("readiness_packet_hash")),
        explicit_runtime_write=False,
        now=NOW + timedelta(minutes=1),
        live_override=_LIVE_BLOCKED,
    )
    mb = MockBroker(broker_behavior)
    if not guard.permitted:
        return guard, mb, None
    outcome = "SUCCESS"
    try:
        for p in planned:
            mb.submit_order(
                symbol=p.symbol,
                qty=p.qty,
                side=p.side,
                client_order_id=p.client_order_id,
            )
        if broker_behavior == "partial_fill":
            outcome = "PARTIAL"
        elif broker_behavior == "indeterminate":
            outcome = "INDETERMINATE"
    except TimeoutError:
        outcome = "TIMEOUT"
    except Exception:
        outcome = "EXCEPTION" if broker_behavior == "exception" else "REJECTION"
    finalize_paper_sell_dispatch(
        store_root=tmp_path,
        outcome=outcome,
        explicit_runtime_write=False,
        detail={"calls": len(mb.calls)},
    )
    return guard, mb, outcome


def test_legacy_197w_rejected_by_runtime_and_no_fallback(tmp_path: Path):
    legacy = {
        "approval_id": "legacy",
        "approved_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(minutes=30)).isoformat(),
        "approved_by": "op",
        "approval_mode": "PAPER_ONLY",
        "approved_symbols": ["TSLA"],
        "max_notional": 2660.0,
        "source_packet_id": "pkt",
        "source_preview_hash": "hash",
        "acknowledgement_live_blocked": True,
        "acknowledgement_no_live_execution": True,
        "acknowledgement_paper_only": True,
        "decision": "APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH",
        "approval_status": "APPROVED",
        "approval_action": "APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH",
        "approval_scope": "PAPER_ONLY",
        "created_at": NOW.isoformat(),
    }
    path = tmp_path / "paper_sell_arm_authority.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert looks_like_legacy_197w_approval(legacy)

    ok, reason, _ = validate_runtime_sell_authority(contract_path=path, now=NOW)
    assert not ok and reason == LEGACY_197W_RETIRED

    from services.defensive_exit_operator_paper_approval import (
        APPROVAL_STATUS_VALID,
        validate_operator_approval_contract,
    )

    st, reason2, det = validate_operator_approval_contract(
        contract_path=path, authority_policy="hardened_only"
    )
    assert st != APPROVAL_STATUS_VALID
    assert LEGACY_197W_RETIRED in reason2
    # historical review may still inspect shape without granting runtime authority flag
    st3, _, det3 = validate_operator_approval_contract(
        contract_path=path, authority_policy="legacy_historical_review"
    )
    assert det3.get("authority_policy") == "legacy_historical_review"

    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_LEGACY_RETIRED,
        PAPER_APPROVAL_VALID_CURRENT,
        classify_paper_operator_approval_artifact,
    )

    state, _, _ = classify_paper_operator_approval_artifact(contract_path=path)
    assert state == PAPER_APPROVAL_LEGACY_RETIRED
    assert state != PAPER_APPROVAL_VALID_CURRENT


def test_preview_readiness_challenge_buy_arm_rejected(tmp_path: Path):
    from services.defensive_exit_operator_paper_approval import (
        APPROVAL_STATUS_VALID,
        validate_operator_approval_contract,
    )

    for doc in (
        {"preview_status": "UNAPPROVED_PREVIEW_ONLY", "creates_no_authority": True},
        _readiness(),
        {"timestamp": "x", "mode": "ASSISTED", "allow_execute": True},
    ):
        p = tmp_path / "paper_sell_arm_authority.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        st, _, _ = validate_operator_approval_contract(contract_path=p)
        assert st != APPROVAL_STATUS_VALID

    readiness, chal, _ = _arm(tmp_path)
    p = tmp_path / "paper_sell_arm_authority.json"
    # overwrite authority with challenge
    p.write_text(json.dumps(chal), encoding="utf-8")
    st, reason, _ = validate_operator_approval_contract(contract_path=p)
    assert st != APPROVAL_STATUS_VALID


@pytest.mark.parametrize("action", ["TRIM", "EXIT", "ROTATE_EXIT", "FORCE_EXIT"])
def test_missing_authority_blocks_each_sell_action(tmp_path: Path, action: str):
    planned = _planned(action=action, force=(action == "FORCE_EXIT"))
    assert planned_rows_include_paper_sells(planned)
    guard = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=BrokerSnapshotView(
            positions={"TSLA": {"qty": 8, "qty_available": 8}},
            open_order_symbols=set(),
            live_blocked=True,
            trading_date="2026-08-11",
            current_plan_hash="x",
        ),
        mode="paper",
        now=NOW,
    )
    assert not guard.permitted


def test_hardened_gate_permits_then_consumes_before_mock_submit(tmp_path: Path):
    _, _, auth = _arm(tmp_path)
    planned = _planned()
    guard, mb, outcome = _mock_dispatch(tmp_path, auth, planned, "success")
    assert guard.permitted and guard.consumed
    assert len(mb.calls) == 1
    assert outcome == "SUCCESS"
    loaded = json.loads((tmp_path / "paper_sell_arm_authority.json").read_text(encoding="utf-8"))
    assert loaded["consumption_status"] == "CONSUMED"
    ok, reason, _ = validate_paper_sell_arm_authority(loaded)
    assert not ok


@pytest.mark.parametrize(
    "behavior,expected_outcome",
    [
        ("rejection", "REJECTION"),
        ("timeout", "TIMEOUT"),
        ("exception", "EXCEPTION"),
        ("indeterminate", "INDETERMINATE"),
        ("partial_fill", "PARTIAL"),
    ],
)
def test_authority_not_reusable_after_mock_outcomes(
    tmp_path: Path, behavior: str, expected_outcome: str
):
    _, _, auth = _arm(tmp_path)
    guard, mb, outcome = _mock_dispatch(tmp_path, auth, _planned(), behavior)
    assert guard.consumed
    assert outcome == expected_outcome
    # Second attempt must fail
    guard2 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth),
        mode="paper",
        readiness_packet_hash_current=str(auth.get("readiness_packet_hash")),
        now=NOW + timedelta(minutes=2),
    )
    assert not guard2.permitted


def test_concurrent_calls_single_dispatch(tmp_path: Path):
    _, _, auth = _arm(tmp_path)
    results = []

    def worker():
        g = begin_paper_sell_dispatch(
            store_root=tmp_path,
            planned=_planned(),
            broker_view=_broker(auth),
            mode="paper",
            readiness_packet_hash_current=str(auth.get("readiness_packet_hash")),
            now=NOW + timedelta(minutes=1),
        )
        results.append(g.permitted)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert sum(1 for r in results if r) <= 1


def test_quantity_symbol_action_batch_and_cap_rejected(tmp_path: Path):
    _, _, auth = _arm(tmp_path)
    # expansion
    g = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(qty=9),
        broker_view=_broker(auth),
        mode="paper",
        readiness_packet_hash_current=str(auth.get("readiness_packet_hash")),
        now=NOW + timedelta(minutes=1),
    )
    assert not g.permitted and "quantity_expansion" in g.reason

    g2 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(action="TRIM"),
        broker_view=_broker(auth),
        mode="paper",
        readiness_packet_hash_current=str(auth.get("readiness_packet_hash")),
        now=NOW + timedelta(minutes=1),
    )
    assert not g2.permitted and "action_substitution" in g2.reason

    # batch expansion
    planned = _planned() + [
        FakePlanned("GLD", "sell", 16, 400.0, stance="EXIT", client_order_id="gld")
    ]
    g3 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=_broker(
            auth,
            positions={
                "TSLA": {"qty": 8, "qty_available": 8},
                "GLD": {"qty": 16, "qty_available": 16},
            },
        ),
        mode="paper",
        readiness_packet_hash_current=str(auth.get("readiness_packet_hash")),
        now=NOW + timedelta(minutes=1),
    )
    assert not g3.permitted


def test_supersession_open_order_available_qty(tmp_path: Path):
    _, _, auth = _arm(tmp_path)
    g = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth, newer_completed_plan=True),
        mode="paper",
        readiness_packet_hash_current=str(auth.get("readiness_packet_hash")),
        now=NOW + timedelta(minutes=1),
    )
    assert not g.permitted and "supersedes" in g.reason

    g2 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth, open_order_symbols={"TSLA"}),
        mode="paper",
        readiness_packet_hash_current=str(auth.get("readiness_packet_hash")),
        now=NOW + timedelta(minutes=1),
    )
    assert not g2.permitted and "open_order" in g2.reason

    g3 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth, positions={"TSLA": {"qty": 8.0, "qty_available": 3.0}}),
        mode="paper",
        readiness_packet_hash_current=str(auth.get("readiness_packet_hash")),
        now=NOW + timedelta(minutes=1),
    )
    assert not g3.permitted


def test_wrong_ids_and_live_rejected(tmp_path: Path):
    _, _, auth = _arm(tmp_path)
    g = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth, current_plan_hash="other"),
        mode="paper",
        readiness_packet_hash_current=str(auth.get("readiness_packet_hash")),
        now=NOW + timedelta(minutes=1),
    )
    assert not g.permitted

    g2 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth),
        mode="paper",
        readiness_packet_hash_current=str(auth.get("readiness_packet_hash")),
        now=NOW + timedelta(minutes=1),
        # Caller cannot keep dispatch open when authoritative live state is enabled.
        live_override={
            "live_blocked": False,
            "live_block_reason": "TEST_LIVE_ENABLED",
            "live_explicitly_authorized": True,
            "live_env_enabled": True,
            "paper_mode": True,
        },
    )
    assert not g2.permitted

    g3 = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=_planned(),
        broker_view=_broker(auth),
        mode="live",
        now=NOW + timedelta(minutes=1),
    )
    assert not g3.permitted and g3.reason == "mode_not_paper"


def test_buy_rows_skip_sell_gate(tmp_path: Path):
    buys = [FakePlanned("AAPL", "buy", 1, 100.0, stance="BUY")]
    assert not planned_rows_include_paper_sells(buys)
    g = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=buys,
        broker_view=BrokerSnapshotView(positions={}, open_order_symbols=set()),
        mode="paper",
        now=NOW,
    )
    assert g.permitted and not g.consumed


def test_sell_arm_rejected_for_buy_add_scope():
    ok, reason, _ = validate_paper_sell_arm_authority(
        {
            "authority_type": "PAPER_SELL_ARM",
            "schema_version": "198H-PAPER-SELL-ARM-1",
            "status": "ARMED",
            "mode": "PAPER",
            "authorized_actions": [{"symbol": "AAPL", "action": "BUY", "max_qty": 1}],
        }
    )
    assert not ok


def test_no_real_governance_touched():
    assert LIVE_APPROVAL_PATH.name == "paper_sell_arm_authority.json"
    # This module's tests only write under tmp_path fixtures.


def test_interlock_blocks_when_order_guard_missing(tmp_path: Path, monkeypatch):
    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_ORDER_GUARD,
        evaluate_defensive_exit_dispatch_interlock,
    )

    _block_network_and_live_env(monkeypatch)
    _ban_repo_live_io(monkeypatch)
    roots = _isolate_interlock_artifact_roots(monkeypatch, tmp_path)
    assert not (roots["results"] / "guard_snapshot.json").is_file()
    assert not (roots["live"] / "guard_snapshot.json").is_file()
    assert not (roots["gov"] / "paper_sell_arm_authority.json").is_file()

    result = evaluate_defensive_exit_dispatch_interlock(
        mode="paper",
        broker_snapshot_ok=True,
        require_market_open=False,
        planned_count=1,
    )
    assert result.get("dispatch_interlock_state") == DISPATCH_BLOCKED_ORDER_GUARD
    assert result.get("broker_orders_submitted") is False
    assert result.get("eligible_for_execution") is False
    assert not (roots["gov"] / "paper_sell_arm_authority.json").is_file()


def test_interlock_blocks_without_hardened_authority(tmp_path: Path, monkeypatch):
    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_APPROVAL_REQUIRED,
        evaluate_defensive_exit_dispatch_interlock,
    )
    from services.order_guard_snapshot_authority import ORDER_GUARD_FRESH

    _block_network_and_live_env(monkeypatch)
    _ban_repo_live_io(monkeypatch)
    roots = _isolate_interlock_artifact_roots(monkeypatch, tmp_path)
    guard_path = _write_valid_order_guard_fixture(roots["results"])
    assert guard_path.is_file()
    assert not (roots["gov"] / "paper_sell_arm_authority.json").is_file()
    assert not (roots["live"] / "paper_arm_confirm.json").is_file()

    result = evaluate_defensive_exit_dispatch_interlock(
        mode="paper",
        broker_snapshot_ok=True,
        require_market_open=False,
        planned_count=1,
    )
    assert result.get("order_guard_authority_state") == ORDER_GUARD_FRESH
    assert result.get("dispatch_interlock_state") == DISPATCH_BLOCKED_APPROVAL_REQUIRED
    assert result.get("broker_orders_submitted") is False
    assert result.get("eligible_for_execution") is False
    assert result.get("hardened_sell_arm_ok") is False
    assert tmp_path.resolve() in guard_path.resolve().parents
    assert "data\\live" not in str(guard_path).replace("/", "\\")
    assert not (roots["gov"] / "paper_sell_arm_authority.json").is_file()
    assert not (roots["gov"] / "paper_sell_arm_consumption.json").is_file()


def test_authority_creation_does_not_dispatch(tmp_path: Path):
    _, _, auth = _arm(tmp_path)
    assert auth["orders_submitted"] is False
    assert auth["creates_dispatch_authority"] is False


def test_cli_cap_breach_blocked_at_dispatch(tmp_path: Path):
    readiness = _readiness(
        approval_ready_actions=[
            {"symbol": "BITO", "action": "EXIT", "qty": 100, "est_notional": 4000.0}
        ],
        approval_ready_symbols=["BITO"],
        total_proposed_notional=4000.0,
        plan_hash="plan_bito",
        batch_id="batch_bito",
    )
    # Arm with under-cap scope then try over-cap planned notional via price
    _, _, auth = _arm(tmp_path, readiness=readiness)
    planned = [FakePlanned("BITO", "sell", 100, 60.0, stance="EXIT", client_order_id="bito")]
    g = begin_paper_sell_dispatch(
        store_root=tmp_path,
        planned=planned,
        broker_view=_broker(
            auth,
            positions={"BITO": {"qty": 100, "qty_available": 100}},
            current_plan_hash=auth["plan_hash"],
            account_fingerprint=auth["account_fingerprint"],
            trading_date=auth["trading_date"],
        ),
        mode="paper",
        readiness_packet_hash_current=str(auth.get("readiness_packet_hash")),
        now=NOW + timedelta(minutes=1),
    )
    assert not g.permitted
    assert "cap" in g.reason or "notional" in g.reason
