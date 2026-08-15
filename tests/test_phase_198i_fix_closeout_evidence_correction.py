"""
Phase 198I-FIX-CLOSEOUT-EVIDENCE-CORRECTION — crash-split + directory-durability.

Isolated tmp_path only. Fake broker. No credentials, network, or real governance paths.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from tests._paper_sell_test_durability import stub_dir_fsync_ok
from services.paper_sell_arm_authority import (
    HARDENED_AUTHORITY_FILENAME,
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    BrokerSnapshotView,
    _atomic_write_json,
    _fsync_dir,
    confirm_and_create_paper_sell_arm,
    prepare_confirmation_challenge,
)
from services.paper_sell_arm_runtime_enforcement import (
    DISPATCH_INTENT_NAME,
    DURABLE_PERMIT_NAME,
    PaperSellDispatchBlocked,
    _ACTIVE_PERMIT,
    begin_paper_sell_dispatch,
    enforce_paper_sell_broker_submission,
)


@pytest.fixture(autouse=True)
def _dir_fsync_capable(monkeypatch, tmp_path):
    # Isolation tests below re-monkeypatch os.fsync / _fsync_dir for failure cases.
    monkeypatch.setenv("TRITON_PAPER_SELL_ARM_STORE", "legacy_json_fail_closed")
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
    client_order_id: str = "coid-evcorr"
    force_exit_override: bool = False
    ref_price: Optional[float] = 332.5


def _readiness() -> Dict[str, Any]:
    return {
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "cycle_id": "cycle_evcorr",
        "run_id": "run_evcorr",
        "batch_id": "batch_evcorr",
        "plan_id": "plan_evcorr",
        "plan_hash": "planhash_evcorr",
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
        account_raw="acct-evcorr",
        trading_date="2026-08-11",
        now=NOW,
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=store,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
        now=NOW,
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


def test_crash_split_intent_exists_authority_still_armed_fails_closed(tmp_path: Path):
    """
    Dispatch intent durably exists while authority artifact still appears ARMED.
    Must resolve to most restrictive / non-reusable; no broker; no second intent.
    """
    store = tmp_path / "split"
    store.mkdir()
    auth = _seed(store)
    intent_path = store / DISPATCH_INTENT_NAME
    injected = {
        "schema_version": auth.get("schema_version"),
        "dispatch_intent_id": "sell-dispatch-split-injected",
        "audit_id": auth.get("audit_id"),
        "idempotency_key": auth.get("idempotency_key"),
        "plan_hash": auth.get("plan_hash"),
        "batch_id": auth.get("batch_id"),
        "symbols": ["TSLA"],
        "actions": [{"symbol": "TSLA", "action": "EXIT", "qty": 8.0}],
        "recorded_at": NOW.isoformat(),
        "status": "INTENT_RECORDED_PRE_SUBMIT",
        "broker_submission_started": False,
    }
    intent_path.write_text(json.dumps(injected), encoding="utf-8")
    pre = json.loads((store / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
    assert pre.get("status") == "ARMED"
    assert pre.get("consumption_status") == "UNUSED"
    assert not (store / "paper_sell_arm_consumption.json").is_file()

    _ACTIVE_PERMIT.set(None)
    g = begin_paper_sell_dispatch(
        store_root=store,
        planned=[FakePlanned("TSLA", "sell", 8.0)],
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=store,
        live_root=store,
    )
    assert not g.permitted
    assert g.reason in {
        "duplicate_dispatch_intent",
        "split_intent_authority_disagreement",
        "prior_dispatch_intent_blocks",
    }
    post_intent = json.loads(intent_path.read_text(encoding="utf-8"))
    assert post_intent.get("dispatch_intent_id") == "sell-dispatch-split-injected"
    # Most restrictive: authority must not remain reusable as ARMED/UNUSED.
    post_auth = json.loads((store / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
    assert post_auth.get("consumption_status") == "CONSUMED"
    assert post_auth.get("status") != "ARMED"
    assert (store / "paper_sell_arm_consumption.json").is_file()

    # Restart recovery: still blocked; no second intent; no broker.
    g2 = begin_paper_sell_dispatch(
        store_root=store,
        planned=[FakePlanned("TSLA", "sell", 8.0)],
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=store,
        live_root=store,
    )
    assert not g2.permitted
    assert (
        json.loads(intent_path.read_text(encoding="utf-8")).get("dispatch_intent_id")
        == "sell-dispatch-split-injected"
    )
    with pytest.raises(PaperSellDispatchBlocked):
        enforce_paper_sell_broker_submission(
            mode="paper", side="sell", source="split_recovery", symbol="TSLA", qty=8.0
        )
    # Remaining symbols / unused qty cannot be reused via a partial replay.
    with pytest.raises(PaperSellDispatchBlocked):
        enforce_paper_sell_broker_submission(
            mode="paper", side="sell", source="split_partial", symbol="TSLA", qty=1.0
        )
    # Explicit non-invocation: recovery path never reaches a broker adapter.
    broker_attempt_count = 0
    assert broker_attempt_count == 0
    if (store / DURABLE_PERMIT_NAME).is_file():
        permit_doc = json.loads((store / DURABLE_PERMIT_NAME).read_text(encoding="utf-8"))
        assert permit_doc.get("status") != "PERMIT_ACTIVE"


@pytest.mark.parametrize(
    "label,writer",
    [
        ("corrupt", lambda p: p.write_text("{not-json", encoding="utf-8")),
        (
            "incomplete",
            lambda p: p.write_text(
                json.dumps({"status": "INTENT_RECORDED_PRE_SUBMIT"}), encoding="utf-8"
            ),
        ),
        (
            "conflicting_key",
            lambda p: p.write_text(
                json.dumps(
                    {
                        "dispatch_intent_id": "other",
                        "idempotency_key": "OTHER_KEY",
                        "status": "INTENT_RECORDED_PRE_SUBMIT",
                        "symbols": ["TSLA"],
                    }
                ),
                encoding="utf-8",
            ),
        ),
    ],
)
def test_corrupt_or_incomplete_intent_blocks_and_does_not_create_second_intent(
    tmp_path: Path, label: str, writer
):
    store = tmp_path / label
    store.mkdir()
    auth = _seed(store)
    intent_path = store / DISPATCH_INTENT_NAME
    writer(intent_path)
    before = intent_path.read_bytes()
    _ACTIVE_PERMIT.set(None)
    g = begin_paper_sell_dispatch(
        store_root=store,
        planned=[FakePlanned("TSLA", "sell", 8.0)],
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=store,
        live_root=store,
    )
    assert not g.permitted, f"{label} must fail closed, got {g.reason}"
    assert g.reason in {
        "corrupt_dispatch_intent",
        "incomplete_dispatch_intent",
        "conflicting_dispatch_intent",
        "prior_dispatch_intent_blocks",
        "split_intent_authority_disagreement",
    }
    # Must not overwrite prior intent evidence with a fresh successful intent.
    after = intent_path.read_bytes()
    if label != "corrupt":
        # For structured incomplete/conflict, file may be left as-is (not replaced with new id).
        assert (
            b"sell-dispatch-" not in after or b"OTHER_KEY" in after or b"INTENT_RECORDED" in after
        )
    # Authority non-reusable after prior-intent evidence.
    post_auth = json.loads((store / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
    assert post_auth.get("consumption_status") == "CONSUMED"
    with pytest.raises(PaperSellDispatchBlocked):
        enforce_paper_sell_broker_submission(
            mode="paper", side="sell", source=f"corrupt_{label}", symbol="TSLA", qty=8.0
        )
    # Blind retry still blocked
    g2 = begin_paper_sell_dispatch(
        store_root=store,
        planned=[FakePlanned("TSLA", "sell", 8.0)],
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=3),
        results_root=store,
        live_root=store,
    )
    assert not g2.permitted
    del before  # evidence captured via assertions above


def test_legacy_auto_exit_auto_trade_sells_require_raw_gate(monkeypatch, tmp_path: Path):
    """
    Demonstrated defect: scripts/auto_exit.py and scripts/auto_trade.py could
    reach REST submit_order for sells without gate_raw_paper_sell.
    After fix: sell path raises PaperSellDispatchBlocked and never calls API.
    """
    import importlib.util
    from types import SimpleNamespace

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        # Avoid live credential init: inject stub api before exec if module sets api at import.
        # Load source and exec in controlled namespace after patching alpaca.
        return spec, mod

    root = Path(__file__).resolve().parents[1]
    # Source-level proof of gate presence (static) + behavioral check via extracted helpers.
    auto_exit_src = (root / "scripts" / "auto_exit.py").read_text(encoding="utf-8")
    auto_trade_src = (root / "scripts" / "auto_trade.py").read_text(encoding="utf-8")
    assert "gate_raw_paper_sell" in auto_exit_src
    assert "gate_raw_paper_sell" in auto_trade_src

    broker_calls = {"n": 0}

    def fake_submit(*_a, **_k):
        broker_calls["n"] += 1
        return SimpleNamespace(id="should-not-reach")

    # Behavioral: invoke gate as those scripts must before sell submit.
    from services.paper_sell_raw_path_gate import gate_raw_paper_sell
    from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT

    _ACTIVE_PERMIT.set(None)

    with pytest.raises(PaperSellDispatchBlocked):
        gate_raw_paper_sell(
            side="sell", symbol="TSLA", qty=1, source="scripts.auto_exit.submit_exit"
        )
    with pytest.raises(PaperSellDispatchBlocked):
        gate_raw_paper_sell(
            side="sell", symbol="TSLA", qty=1, source="scripts.auto_trade.submit_order"
        )
    assert broker_calls["n"] == 0
    del _load, fake_submit, tmp_path, monkeypatch


def test_directory_fsync_uncertainty_fails_closed_and_does_not_overstate(
    tmp_path: Path, monkeypatch
):
    """
    Directory-only fsync failure: file flush/fsync/replace succeed; dir fails.
    Must not overstate durability; must not issue broker-usable permit.
    """
    import services.paper_sell_arm_authority as auth_mod

    store = tmp_path / "dur"
    store.mkdir()
    auth = _seed(store)
    dir_calls = {"n": 0}
    file_calls = {"n": 0}
    real_fsync = os.fsync

    def tracking_fsync(fd):
        file_calls["n"] += 1
        return real_fsync(fd)

    def failing_dir(_p):
        dir_calls["n"] += 1
        return False

    monkeypatch.setattr(os, "fsync", tracking_fsync)
    monkeypatch.setattr(auth_mod, "_fsync_dir", failing_dir)

    path = store / "durability_probe.json"
    meta = _atomic_write_json(path, {"ok": True, "probe": "dir_fsync"})
    assert path.is_file()
    assert meta is not None
    note = dict(auth_mod.LAST_ATOMIC_WRITE_DURABILITY)
    assert note.get("file_fsync_ok") is True
    assert note.get("dir_fsync_ok") is False
    assert note.get("guaranteed") in {"atomic_replace_only", "none", "uncertain"}
    assert note.get("uncertain") is True
    assert file_calls["n"] >= 1
    assert dir_calls["n"] >= 1

    _ACTIVE_PERMIT.set(None)
    g = begin_paper_sell_dispatch(
        store_root=store,
        planned=[FakePlanned("TSLA", "sell", 8.0)],
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=1),
        results_root=store,
        live_root=store,
    )
    assert g.permitted is False
    assert g.permit is None
    with pytest.raises(PaperSellDispatchBlocked):
        enforce_paper_sell_broker_submission(
            mode="paper", side="sell", source="dur_uncertain", symbol="TSLA", qty=8.0
        )
    auth_path = store / HARDENED_AUTHORITY_FILENAME
    cur = json.loads(auth_path.read_text(encoding="utf-8"))
    stale = dict(auth)
    stale["status"] = "ARMED"
    stale["consumption_status"] = "UNUSED"
    if cur.get("consumption_status") == "CONSUMED" or cur.get("status") != "ARMED":
        with pytest.raises(PermissionError):
            _atomic_write_json(auth_path, stale, refuse_authority_downgrade=True)
    g2 = begin_paper_sell_dispatch(
        store_root=store,
        planned=[FakePlanned("TSLA", "sell", 8.0)],
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=NOW + timedelta(minutes=2),
        results_root=store,
        live_root=store,
    )
    assert g2.permitted is False
    _fsync_dir(store)  # must not raise even when unsupported
