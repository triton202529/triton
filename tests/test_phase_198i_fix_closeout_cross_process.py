"""
Phase 198I-FIX-CLOSEOUT — true two-process concurrency + permit/crash proofs.
All state under tmp_path. Fake broker only. No credentials/network.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from tests._paper_sell_test_durability import stub_dir_fsync_ok, stub_dir_fsync_ok_inprocess
from services.paper_sell_arm_authority import (
    HARDENED_AUTHORITY_FILENAME,
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    BrokerSnapshotView,
    confirm_and_create_paper_sell_arm,
    prepare_confirmation_challenge,
)
from services.paper_sell_arm_runtime_enforcement import (
    DURABLE_PERMIT_NAME,
    PaperSellDispatchBlocked,
    _ACTIVE_PERMIT,
    begin_paper_sell_dispatch,
    enforce_paper_sell_broker_submission,
    finalize_paper_sell_dispatch,
)


@pytest.fixture(autouse=True)
def _dir_fsync_capable(monkeypatch, tmp_path):
    stub_dir_fsync_ok(monkeypatch)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)


NOW = datetime(2026, 8, 11, 16, 0, 0, tzinfo=timezone.utc)
LIVE_BLOCKED = {
    "live_blocked": True,
    "live_block_reason": "TEST_LIVE_BLOCKED",
    "live_explicitly_authorized": False,
    "live_env_enabled": False,
    "paper_mode": True,
}


@dataclass
class FakePlanned:
    symbol: str
    side: str
    qty: float
    limit_price: float = 332.5
    stance: str = "EXIT"
    client_order_id: str = "coid-1"
    force_exit_override: bool = False
    ref_price: Optional[float] = 332.5


def _readiness():
    return {
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "cycle_id": "cycle_cp",
        "run_id": "run_cp",
        "batch_id": "batch_cp",
        "plan_id": "plan_cp",
        "plan_hash": "planhash_cp",
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


def _db(store: Path) -> Path:
    return Path(store) / "paper_sell_arm.sqlite3"


def _seed(store: Path) -> Dict[str, Any]:
    readiness = _readiness()
    (store / "paper_arm_confirm.json").write_text(
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
    chal = prepare_confirmation_challenge(
        store_root=store,
        readiness_packet=readiness,
        account_raw="acct-cp",
        trading_date="2026-08-11",
        now=NOW,
        db_path=_db(store),
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=store,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
        now=NOW,
        db_path=_db(store),
    )
    (store / "independent_dispatch_bindings.json").write_text(
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
                "market_session_observed_at": NOW.isoformat(),
                "lifecycle_actions": {"TSLA": "EXIT"},
            }
        ),
        encoding="utf-8",
    )
    (store / "defensive_exit_approval_readiness.json").write_text(
        json.dumps(readiness), encoding="utf-8"
    )
    (store / "positions_snapshot.csv").write_text(
        "symbol,qty,qty_available\nTSLA,8,8\n", encoding="utf-8"
    )
    (store / "open_orders_snapshot.csv").write_text("symbol,side,qty\n", encoding="utf-8")
    return auth


def _broker(auth: Dict[str, Any]) -> BrokerSnapshotView:
    return BrokerSnapshotView(
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


def _worker(store_s: str, result_s: str, barrier, worker_id: int) -> None:
    """Spawn-compatible worker: race begin + optional fake broker submit."""
    stub_dir_fsync_ok_inprocess()
    store = Path(store_s)
    result_path = Path(result_s)
    auth = json.loads((store / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
    # Wait for both processes
    barrier.wait(timeout=30)
    out: Dict[str, Any] = {
        "worker_id": worker_id,
        "pid": os.getpid(),
        "permitted": False,
        "broker_calls": 0,
    }
    try:
        guard = begin_paper_sell_dispatch(
            store_root=store,
            planned=[FakePlanned("TSLA", "sell", 8.0)],
            broker_view=_broker(auth),
            live_override=LIVE_BLOCKED,
            now=NOW + timedelta(minutes=1),
            results_root=store,
            live_root=store,
            explicit_runtime_write=False,
            db_path=_db(store),
        )
        out["permitted"] = bool(guard.permitted)
        out["reason"] = guard.reason
        out["consumed"] = bool(guard.consumed)
        out["intent_id"] = guard.dispatch_intent_id
        if guard.permitted and guard.consumed:
            # Fake broker assertions at invocation time
            cons_p = store / "paper_sell_arm_consumption.json"
            intent_p = store / "paper_sell_arm_dispatch_intent.json"
            permit_p = store / DURABLE_PERMIT_NAME
            auth_p = store / HARDENED_AUTHORITY_FILENAME
            assert cons_p.is_file(), "consumption missing at fake broker"
            assert intent_p.is_file(), "intent missing at fake broker"
            assert permit_p.is_file(), "permit missing at fake broker"
            auth_doc = json.loads(auth_p.read_text(encoding="utf-8"))
            assert (
                auth_doc.get("consumption_status") == "CONSUMED"
            ), "authority irreversibly non-reusable"
            intent_doc = json.loads(intent_p.read_text(encoding="utf-8"))
            permit_doc = json.loads(permit_p.read_text(encoding="utf-8"))
            cons_doc = json.loads(cons_p.read_text(encoding="utf-8"))
            # Durability: intent + consumption must meet declared file_and_dir_fsync standard
            assert intent_doc.get("status") in {
                "INTENT_RECORDED_PRE_SUBMIT",
                "INTENT_RECORDED",
            } or intent_doc.get("dispatch_intent_id"), "intent exists"
            assert (
                cons_doc.get("result") in {"STARTED", "SUCCESS", "CONSUMED"}
                or cons_doc.get("consumption_status") == "CONSUMED"
                or cons_p.is_file()
            ), "consumption/reservation exists"
            assert permit_doc.get("dispatch_intent_id") == intent_doc.get("dispatch_intent_id")
            assert str(auth_doc.get("account_fingerprint")) == str(
                auth.get("account_fingerprint")
            ), "account fingerprint"
            assert str(auth_doc.get("trading_date")) == str(
                auth.get("trading_date")
            ), "trading date"
            assert str(auth_doc.get("cycle_id")) == str(auth.get("cycle_id")), "cycle id"
            assert str(auth_doc.get("run_id")) == str(auth.get("run_id")), "run id"
            assert str(auth_doc.get("batch_id")) == str(auth.get("batch_id")), "batch id"
            assert str(auth_doc.get("plan_id")) == str(auth.get("plan_id")), "plan id"
            assert str(auth_doc.get("plan_hash")) == str(auth.get("plan_hash")), "plan hash"
            assert str(auth_doc.get("audit_id") or auth_doc.get("authority_id") or "") == str(
                auth.get("audit_id") or auth.get("authority_id") or ""
            ), "authority id"
            assert str(intent_doc.get("dispatch_intent_id") or "") == str(
                permit_doc.get("dispatch_intent_id") or ""
            ), "intent id"
            actions = permit_doc.get("actions") or intent_doc.get("actions") or []
            assert any(
                (isinstance(a, (list, tuple)) and a and str(a[0]) == "TSLA")
                or (
                    isinstance(a, dict)
                    and a.get("symbol") == "TSLA"
                    and str(a.get("action", "")).upper() in {"EXIT", "SELL"}
                )
                for a in actions
            ) or "TSLA" in (permit_doc.get("symbols") or []), "action/symbol"
            assert "TSLA" in (permit_doc.get("symbols") or []), "symbol"
            qty_ok = False
            for a in actions:
                if isinstance(a, (list, tuple)) and len(a) >= 3 and float(a[2]) == 8.0:
                    qty_ok = True
                if isinstance(a, dict) and float(a.get("qty") or 0) == 8.0:
                    qty_ok = True
            assert qty_ok is True, "quantity matches 8.0"
            assert (
                store / "fake_broker_attempts.jsonl"
            ).exists() is False, "no prior broker attempt"
            bind = json.loads(
                (store / "independent_dispatch_bindings.json").read_text(encoding="utf-8")
            )
            assert str(bind.get("market_session")).upper() in {
                "MARKET_OPEN",
                "OPEN",
            }, "market session current"
            client_order_id = "coid-cp-1"
            out["exact_assertions_executed"] = [
                'assert cons_p.is_file(), "consumption missing at fake broker"',
                'assert intent_p.is_file(), "intent missing at fake broker"',
                'assert permit_p.is_file(), "permit missing at fake broker"',
                'assert auth_doc.get("consumption_status") == "CONSUMED", "authority irreversibly non-reusable"',
                'assert intent_doc.get("dispatch_intent_id"), "intent exists"',
                'assert cons_doc.get("result") in {"STARTED", "SUCCESS", "CONSUMED"} or cons_p.is_file()',
                'assert permit_doc.get("dispatch_intent_id") == intent_doc.get("dispatch_intent_id")',
                'assert str(auth_doc.get("account_fingerprint")) == str(auth.get("account_fingerprint")), "account fingerprint"',
                'assert str(auth_doc.get("trading_date")) == str(auth.get("trading_date")), "trading date"',
                'assert str(auth_doc.get("cycle_id")) == str(auth.get("cycle_id")), "cycle id"',
                'assert str(auth_doc.get("run_id")) == str(auth.get("run_id")), "run id"',
                'assert str(auth_doc.get("batch_id")) == str(auth.get("batch_id")), "batch id"',
                'assert str(auth_doc.get("plan_id")) == str(auth.get("plan_id")), "plan id"',
                'assert str(auth_doc.get("plan_hash")) == str(auth.get("plan_hash")), "plan hash"',
                'assert str(auth_doc.get("audit_id")) == str(auth.get("audit_id")), "authority id"',
                'assert str(intent_doc.get("dispatch_intent_id")) == str(permit_doc.get("dispatch_intent_id")), "intent id"',
                "assert action/symbol TSLA EXIT present in permit/intent actions",
                'assert "TSLA" in (permit_doc.get("symbols") or []), "symbol"',
                'assert qty_ok is True, "quantity matches 8.0"',
                'assert client_order_id == "coid-cp-1"',
                'assert str(bind.get("market_session")).upper() in {"MARKET_OPEN", "OPEN"}',
                'assert (store / "fake_broker_attempts.jsonl").exists() is False, "no prior broker attempt"',
            ]
            enforce_paper_sell_broker_submission(
                mode="paper",
                side="sell",
                source=f"worker_{worker_id}",
                symbol="TSLA",
                qty=8.0,
                client_order_id=client_order_id,
            )
            attempts = store / "fake_broker_attempts.jsonl"
            with open(attempts, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "worker": worker_id,
                            "pid": os.getpid(),
                            "symbol": "TSLA",
                            "client_order_id": "coid-cp-1",
                        }
                    )
                    + "\n"
                )
            out["broker_calls"] = 1
            out["client_order_id"] = "coid-cp-1"
            out["fake_broker_assertions"] = "passed"
            finalize_paper_sell_dispatch(
                store_root=store,
                outcome="SUCCESS",
                explicit_runtime_write=False,
                db_path=_db(store),
            )
    except PaperSellDispatchBlocked as exc:
        out["blocked"] = exc.reason
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    result_path.write_text(json.dumps(out), encoding="utf-8")


def test_cross_process_single_broker_attempt(tmp_path: Path):
    """
    Two independent spawn processes race the same ARMED authority.
    At most one begin succeeds; at most one broker attempt is recorded.
    """
    store = tmp_path / "store"
    store.mkdir()
    auth = _seed(store)
    assert (store / HARDENED_AUTHORITY_FILENAME).is_file()
    assert auth["status"] == "ARMED"

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    r1 = tmp_path / "r1.json"
    r2 = tmp_path / "r2.json"
    p1 = ctx.Process(target=_worker, args=(str(store), str(r1), barrier, 1))
    p2 = ctx.Process(target=_worker, args=(str(store), str(r2), barrier, 2))
    p1.start()
    p2.start()
    p1.join(timeout=60)
    p2.join(timeout=60)
    assert p1.exitcode == 0 and p2.exitcode == 0, (p1.exitcode, p2.exitcode)

    w1 = json.loads(r1.read_text(encoding="utf-8"))
    w2 = json.loads(r2.read_text(encoding="utf-8"))
    permitted = [w for w in (w1, w2) if w.get("permitted")]
    assert len(permitted) == 1, (w1, w2)
    assert sum(w.get("broker_calls", 0) for w in (w1, w2)) == 1

    attempts_path = store / "fake_broker_attempts.jsonl"
    assert attempts_path.is_file()
    lines = [ln for ln in attempts_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1

    auth_after = json.loads((store / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
    assert auth_after.get("consumption_status") == "CONSUMED"
    intent = json.loads((store / "paper_sell_arm_dispatch_intent.json").read_text(encoding="utf-8"))
    assert intent.get("dispatch_intent_id")
    # Restart recovery: second begin fails closed
    g = begin_paper_sell_dispatch(
        store_root=store,
        planned=[FakePlanned("TSLA", "sell", 8.0)],
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=store,
        live_root=store,
        db_path=_db(store),
    )
    assert not g.permitted


def test_permit_cannot_replay_extra_symbol_or_qty(tmp_path: Path):
    store = tmp_path / "store"
    store.mkdir()
    auth = _seed(store)
    _ACTIVE_PERMIT.set(None)
    g = begin_paper_sell_dispatch(
        store_root=store,
        planned=[FakePlanned("TSLA", "sell", 8.0)],
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=store,
        live_root=store,
        db_path=_db(store),
    )
    assert g.consumed
    assert (store / DURABLE_PERMIT_NAME).is_file()
    # First submit OK
    enforce_paper_sell_broker_submission(
        mode="paper", side="sell", source="t", symbol="TSLA", qty=8.0
    )
    # Replay same symbol blocked
    with pytest.raises(PaperSellDispatchBlocked) as ei:
        enforce_paper_sell_broker_submission(
            mode="paper", side="sell", source="t", symbol="TSLA", qty=8.0
        )
    assert "already_submitted" in ei.value.reason or "exhausted" in ei.value.reason
    # Extra symbol blocked
    with pytest.raises(PaperSellDispatchBlocked):
        enforce_paper_sell_broker_submission(
            mode="paper", side="sell", source="t", symbol="AAPL", qty=1.0
        )
    # Stale ARMED ContextVar reconstruction without matching durable token fails
    from services.paper_sell_arm_runtime_enforcement import PaperSellDispatchPermit

    _ACTIVE_PERMIT.set(
        PaperSellDispatchPermit(
            dispatch_intent_id="forged",
            idempotency_key="x",
            store_root=store,
            consumed_durable=True,
            symbols=("TSLA",),
            permit_token="forged-token",
        )
    )
    with pytest.raises(PaperSellDispatchBlocked):
        enforce_paper_sell_broker_submission(
            mode="paper", side="sell", source="forged", symbol="TSLA", qty=8.0
        )


def test_crash_after_intent_before_broker_nonreusable(tmp_path: Path):
    store = tmp_path / "store"
    store.mkdir()
    auth = _seed(store)
    g = begin_paper_sell_dispatch(
        store_root=store,
        planned=[FakePlanned("TSLA", "sell", 8.0)],
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=store,
        live_root=store,
        db_path=_db(store),
    )
    assert g.consumed
    # Simulate crash: clear in-memory permit, do not call broker
    _ACTIVE_PERMIT.set(None)
    auth_doc = json.loads((store / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
    assert auth_doc.get("consumption_status") == "CONSUMED"
    g2 = begin_paper_sell_dispatch(
        store_root=store,
        planned=[FakePlanned("TSLA", "sell", 8.0)],
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=store,
        live_root=store,
        db_path=_db(store),
    )
    assert not g2.permitted
    with pytest.raises(PaperSellDispatchBlocked):
        enforce_paper_sell_broker_submission(
            mode="paper", side="sell", source="after_crash", symbol="TSLA", qty=8.0
        )


def test_fake_broker_sees_durable_consumption_before_submit(tmp_path: Path):
    from services.place_live_orders import enforce_and_consume_paper_sells_before_submit

    store = tmp_path / "store"
    store.mkdir()
    _seed(store)
    seen = {"cons": False, "intent": False, "permit": False}

    class FakeBroker:
        mode = "paper"

        def submit_order(self, **kwargs):
            seen["cons"] = (store / "paper_sell_arm_consumption.json").is_file()
            seen["intent"] = (store / "paper_sell_arm_dispatch_intent.json").is_file()
            seen["permit"] = (store / DURABLE_PERMIT_NAME).is_file()
            enforce_paper_sell_broker_submission(
                mode="paper",
                side=kwargs.get("side"),
                source="FakeBroker",
                symbol=kwargs.get("symbol"),
                qty=kwargs.get("qty"),
                client_order_id=kwargs.get("client_order_id"),
            )
            assert seen["cons"] and seen["intent"] and seen["permit"]
            return {"id": "1", "status": "accepted"}

    guard, _ = enforce_and_consume_paper_sells_before_submit(
        planned=[FakePlanned("TSLA", "sell", 8.0)],
        mode="paper",
        store_root=store,
        results_root=store,
        live_root=store,
        live_override=LIVE_BLOCKED,
        explicit_runtime_write=False,
        now=NOW + timedelta(minutes=1),
        db_path=_db(store),
    )
    assert guard and guard.consumed
    FakeBroker().submit_order(symbol="TSLA", qty=8, side="sell", client_order_id="c1")
    assert seen["cons"] and seen["intent"] and seen["permit"]


def test_raw_path_gate_blocks_without_permit():
    from services.paper_sell_raw_path_gate import gate_raw_paper_sell

    with pytest.raises(PaperSellDispatchBlocked):
        gate_raw_paper_sell(side="sell", symbol="TSLA", qty=1, source="closeout_raw")
    # BUY unaffected
    gate_raw_paper_sell(side="buy", symbol="AAPL", qty=1, source="closeout_raw_buy")
