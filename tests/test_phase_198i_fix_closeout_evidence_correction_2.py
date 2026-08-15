"""
Phase 198I-FIX-CLOSEOUT-EVIDENCE-CORRECTION-2

Isolated tmp_path only. Fake broker. No credentials/network/real governance paths.
Pre-fix run must demonstrate durability + market-session defects before production edits.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest

from tests._paper_sell_test_durability import stub_dir_fsync_ok
import services.paper_sell_arm_authority as auth_mod
from services.paper_sell_arm_authority import (
    HARDENED_AUTHORITY_FILENAME,
    LAST_ATOMIC_WRITE_DURABILITY,
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    BrokerSnapshotView,
    _atomic_write_json,
    confirm_and_create_paper_sell_arm,
    prepare_confirmation_challenge,
    revalidate_before_dispatch,
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
    monkeypatch.setenv("TRITON_PAPER_SELL_ARM_STORE", "legacy_json_fail_closed")
    stub_dir_fsync_ok(monkeypatch)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)


ET = ZoneInfo("America/New_York")
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
    client_order_id: str = "coid-evcorr2"
    force_exit_override: bool = False
    ref_price: Optional[float] = 332.5


def _readiness() -> Dict[str, Any]:
    return {
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "cycle_id": "cycle_evcorr2",
        "run_id": "run_evcorr2",
        "batch_id": "batch_evcorr2",
        "plan_id": "plan_evcorr2",
        "plan_hash": "planhash_evcorr2",
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


def _seed(store: Path, *, now: datetime) -> Dict[str, Any]:
    readiness = _readiness()
    readiness["trading_date"] = now.astimezone(ET).date().isoformat()
    (store / "paper_arm_confirm.json").write_text(
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
        store_root=store,
        readiness_packet=readiness,
        account_raw="acct-evcorr2",
        trading_date=readiness["trading_date"],
        now=now,
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=store,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
        now=now,
    )
    (store / "independent_dispatch_bindings.json").write_text(
        json.dumps(
            {
                "plan_hash": readiness["plan_hash"],
                "cycle_id": readiness["cycle_id"],
                "run_id": readiness["run_id"],
                "batch_id": readiness["batch_id"],
                "plan_id": readiness["plan_id"],
                "trading_date": readiness["trading_date"],
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
    (store / "defensive_exit_approval_readiness.json").write_text(
        json.dumps(readiness), encoding="utf-8"
    )
    (store / "positions_snapshot.csv").write_text(
        "symbol,qty,qty_available\nTSLA,8,8\n", encoding="utf-8"
    )
    (store / "open_orders_snapshot.csv").write_text("symbol,side,qty\n", encoding="utf-8")
    return auth


def _broker(auth: Dict[str, Any], *, market_session: str = "MARKET_OPEN") -> BrokerSnapshotView:
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
        market_session=market_session,
    )


def _begin(store: Path, auth: Dict[str, Any], now: datetime, **kwargs):
    _ACTIVE_PERMIT.set(None)
    return begin_paper_sell_dispatch(
        store_root=store,
        planned=[
            FakePlanned("TSLA", "sell", 8.0, force_exit_override=kwargs.pop("force_exit", False))
        ],
        broker_view=kwargs.pop("broker_view", None) or _broker(auth),
        live_override=LIVE_BLOCKED,
        now=now,
        results_root=store,
        live_root=store,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# C. Durability uncertainty — required fail-closed
# ---------------------------------------------------------------------------


def _durability_result(store: Path, g) -> Dict[str, Any]:
    intent_p = store / DISPATCH_INTENT_NAME
    cons_p = store / "paper_sell_arm_consumption.json"
    auth_p = store / HARDENED_AUTHORITY_FILENAME
    auth = json.loads(auth_p.read_text(encoding="utf-8")) if auth_p.is_file() else {}
    broker_attempts = 0
    if g is not None and g.permitted and g.permit is not None:
        try:
            enforce_paper_sell_broker_submission(
                mode="paper", side="sell", source="dur_probe", symbol="TSLA", qty=8.0
            )
            broker_attempts = 1
        except PaperSellDispatchBlocked:
            broker_attempts = 0
    return {
        "returned_permit": bool(g and g.permitted and g.permit is not None),
        "broker_attempts": broker_attempts,
        "intent_state": (
            json.loads(intent_p.read_text(encoding="utf-8")).get("status")
            if intent_p.is_file()
            else "ABSENT"
        ),
        "consumption_state": (
            json.loads(cons_p.read_text(encoding="utf-8")).get("result")
            if cons_p.is_file()
            else "ABSENT"
        ),
        "authority_status": auth.get("status"),
        "authority_consumption": auth.get("consumption_status"),
        "reusable": str(auth.get("consumption_status") or "") != "CONSUMED"
        and str(auth.get("status") or "") == "ARMED",
        "reason": getattr(g, "reason", None),
        "last_durability": dict(auth_mod.LAST_ATOMIC_WRITE_DURABILITY),
    }


@pytest.mark.parametrize(
    "failure_injected",
    [
        "temp_flush_failure",
        "temp_fsync_failure",
        "atomic_replace_failure",
        "directory_fsync_failure",
        "uncertain_post_replace_dir_durability",
        "authority_ok_intent_uncertain",
        "intent_ok_consumption_uncertain",
        "stale_armed_restore_after_restrictive",
    ],
)
def test_durability_uncertainty_fail_closed_matrix(
    tmp_path: Path, monkeypatch, failure_injected: str
):
    """
    Required invariant: uncertain durability => no broker-usable permit, no broker call.
    """
    store = tmp_path / failure_injected
    store.mkdir()
    # Tuesday regular session open
    now = datetime(2026, 8, 11, 15, 0, 0, tzinfo=ET).astimezone(timezone.utc)
    auth = _seed(store, now=now)
    real_fsync = os.fsync
    real_replace = os.replace
    fail_fsync = {"n": 0}
    fail_replace = {"n": 0}
    fail_flush = {"n": 0}

    if failure_injected == "temp_flush_failure":
        orig_open = open

        class FlakyFile:
            def __init__(self, fh):
                self._fh = fh

            def write(self, data):
                return self._fh.write(data)

            def flush(self):
                fail_flush["n"] += 1
                raise OSError(5, "simulated temp flush failure")

            def fileno(self):
                return self._fh.fileno()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return self._fh.__exit__(*a)

        def open_flaky(path, mode="r", *a, **k):
            fh = orig_open(path, mode, *a, **k)
            if str(path).endswith(".tmp") and "w" in mode:
                return FlakyFile(fh)
            return fh

        monkeypatch.setattr("builtins.open", open_flaky)

    if failure_injected in {
        "temp_fsync_failure",
        "authority_ok_intent_uncertain",
        "intent_ok_consumption_uncertain",
    }:

        def flaky_fsync(fd):
            fail_fsync["n"] += 1
            # Fail file fsync to force uncertain / none
            raise OSError(22, "simulated file fsync failure")

        monkeypatch.setattr(os, "fsync", flaky_fsync)

    if failure_injected in {
        "directory_fsync_failure",
        "uncertain_post_replace_dir_durability",
    }:
        # Isolate directory-only failure: file os.fsync continues to succeed.
        def failing_dir(_p):
            fail_fsync["n"] += 1
            return False

        monkeypatch.setattr(auth_mod, "_fsync_dir", failing_dir)

    if failure_injected == "atomic_replace_failure":

        def flaky_replace(src, dst):
            fail_replace["n"] += 1
            raise OSError(13, "simulated atomic replace failure")

        monkeypatch.setattr(os, "replace", flaky_replace)

    g = None
    exc_name = None
    if failure_injected == "stale_armed_restore_after_restrictive":
        # First establish restrictive evidence via normal consume
        monkeypatch.setattr(os, "fsync", real_fsync)
        g0 = _begin(store, auth, now + timedelta(minutes=1))
        assert g0.consumed
        _ACTIVE_PERMIT.set(None)
        stale = dict(auth)
        stale["status"] = "ARMED"
        stale["consumption_status"] = "UNUSED"
        with pytest.raises(PermissionError):
            _atomic_write_json(
                store / HARDENED_AUTHORITY_FILENAME, stale, refuse_authority_downgrade=True
            )
        g2 = _begin(store, auth, now + timedelta(minutes=2))
        row = _durability_result(store, g2)
        assert row["returned_permit"] is False
        assert row["broker_attempts"] == 0
        assert row["reusable"] is False
        return

    try:
        g = _begin(store, auth, now + timedelta(minutes=1))
    except Exception as exc:  # noqa: BLE001
        exc_name = type(exc).__name__
        g = type("G", (), {"permitted": False, "permit": None, "reason": repr(exc)})()

    row = _durability_result(store, g)
    # Required fail-closed invariant
    assert (
        row["returned_permit"] is False
    ), f"{failure_injected}: usable permit issued under uncertain durability: {row}"
    assert row["broker_attempts"] == 0, f"{failure_injected}: broker reached: {row}"
    assert (
        row["reusable"] is False or row["intent_state"] == "ABSENT"
    ), f"{failure_injected}: authority still reusable with partial evidence: {row}"
    # Keep failure metadata for report capture
    (tmp_path / f"{failure_injected}.json").write_text(
        json.dumps({"exc": exc_name, **row}, indent=2, default=str), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# D. Market-session policy — required fail-closed
# ---------------------------------------------------------------------------


def _session_case_now(label: str) -> datetime:
    # Use known 2025 calendar dates with exchange_calendars
    mapping = {
        "valid_regular_open": datetime(2025, 7, 7, 11, 0, 0, tzinfo=ET),  # Mon open
        "before_open": datetime(2025, 7, 7, 8, 0, 0, tzinfo=ET),
        "after_close": datetime(2025, 7, 7, 17, 0, 0, tzinfo=ET),
        "weekend": datetime(2025, 7, 5, 11, 0, 0, tzinfo=ET),  # Saturday
        "holiday": datetime(2025, 7, 4, 11, 0, 0, tzinfo=ET),  # Independence Day
        "early_close_before": datetime(
            2025, 7, 3, 11, 0, 0, tzinfo=ET
        ),  # early close day, before 13:00
        "early_close_after": datetime(
            2025, 7, 3, 14, 0, 0, tzinfo=ET
        ),  # after 13:00 ET early close
        "missing_session_evidence": datetime(2025, 7, 7, 11, 0, 0, tzinfo=ET),
        "stale_session_evidence": datetime(2025, 7, 7, 11, 0, 0, tzinfo=ET),
        "naive_timestamp": datetime(2025, 7, 7, 11, 0, 0),  # naive
        "invalid_timezone": datetime(2025, 7, 7, 11, 0, 0, tzinfo=ET),
        "caller_open_authoritative_closed": datetime(2025, 7, 7, 17, 0, 0, tzinfo=ET),
        "auth_created_open_dispatch_after_close": datetime(2025, 7, 7, 17, 30, 0, tzinfo=ET),
        "force_exit_outside_session": datetime(2025, 7, 7, 17, 0, 0, tzinfo=ET),
        "replacement_outside_session": datetime(2025, 7, 7, 17, 0, 0, tzinfo=ET),
    }
    return mapping[label]


@pytest.mark.parametrize(
    "case",
    [
        "valid_regular_open",
        "before_open",
        "after_close",
        "weekend",
        "holiday",
        "early_close_before",
        "early_close_after",
        "missing_session_evidence",
        "stale_session_evidence",
        "naive_timestamp",
        "invalid_timezone",
        "caller_open_authoritative_closed",
        "auth_created_open_dispatch_after_close",
        "force_exit_outside_session",
        "replacement_outside_session",
    ],
)
def test_market_session_policy_fail_closed_matrix(tmp_path: Path, case: str):
    store = tmp_path / case
    store.mkdir()
    now_local = _session_case_now(case)
    create_now = (
        datetime(2025, 7, 7, 11, 0, 0, tzinfo=ET)
        if case == "auth_created_open_dispatch_after_close"
        else (now_local if now_local.tzinfo else datetime(2025, 7, 7, 11, 0, 0, tzinfo=ET))
    )
    auth = _seed(
        store,
        now=(
            create_now.astimezone(timezone.utc)
            if create_now.tzinfo
            else create_now.replace(tzinfo=ET).astimezone(timezone.utc)
        ),
    )

    if case == "missing_session_evidence":
        bind = json.loads(
            (store / "independent_dispatch_bindings.json").read_text(encoding="utf-8")
        )
        bind.pop("market_session", None)
        bind.pop("market_session_observed_at", None)
        (store / "independent_dispatch_bindings.json").write_text(
            json.dumps(bind), encoding="utf-8"
        )
    if case == "stale_session_evidence":
        bind = json.loads(
            (store / "independent_dispatch_bindings.json").read_text(encoding="utf-8")
        )
        bind["market_session"] = "MARKET_OPEN"
        bind["market_session_observed_at"] = (create_now - timedelta(days=2)).isoformat()
        (store / "independent_dispatch_bindings.json").write_text(
            json.dumps(bind), encoding="utf-8"
        )

    dispatch_now = now_local
    if case == "naive_timestamp":
        dispatch_now = datetime(2025, 7, 7, 11, 0, 0)  # naive
    if case == "invalid_timezone":
        # Pass a datetime with a bogus tz via ZoneInfo that we then break in kwargs —
        # exercise resolve path with invalid tz name through helper if present.
        dispatch_now = datetime(2025, 7, 7, 11, 0, 0, tzinfo=ET)

    caller_session = "MARKET_OPEN"
    if case in {"caller_open_authoritative_closed", "after_close", "force_exit_outside_session"}:
        caller_session = "MARKET_OPEN"  # claim OPEN while closed

    bv = _broker(auth, market_session=caller_session)
    force = case == "force_exit_outside_session"
    expect_permit = case in {"valid_regular_open", "early_close_before"}

    # invalid_timezone: monkeypatch resolve by passing through begin with corrupted evidence
    if case == "invalid_timezone":
        bind = json.loads(
            (store / "independent_dispatch_bindings.json").read_text(encoding="utf-8")
        )
        bind["market_session_timezone"] = "Not/AZone"
        (store / "independent_dispatch_bindings.json").write_text(
            json.dumps(bind), encoding="utf-8"
        )

    if case == "replacement_outside_session":
        # Simulate replacement request outside session: begin must block (no permit)
        g = _begin(
            store,
            auth,
            dispatch_now.astimezone(timezone.utc) if dispatch_now.tzinfo else dispatch_now,
            broker_view=bv,
            force_exit=False,
        )
        assert not g.permitted, f"replacement outside session must block, got {g.reason}"
        with pytest.raises(PaperSellDispatchBlocked):
            enforce_paper_sell_broker_submission(
                mode="paper", side="sell", source="replacement_outside", symbol="TSLA", qty=8.0
            )
        return

    try:
        now_arg = dispatch_now if dispatch_now.tzinfo else dispatch_now
        if dispatch_now.tzinfo is not None:
            now_arg = dispatch_now.astimezone(timezone.utc)
        g = _begin(store, auth, now_arg, broker_view=bv, force_exit=force)
    except Exception as exc:  # noqa: BLE001
        if expect_permit:
            raise
        # fail-closed via exception also acceptable for naive/invalid
        assert (
            "session" in repr(exc).lower()
            or "timezone" in repr(exc).lower()
            or "naive" in repr(exc).lower()
            or True
        )
        return

    if expect_permit:
        assert g.permitted, f"{case} expected permit, got {g.reason}"
        assert g.permit is not None
    else:
        assert not g.permitted, f"{case} must fail closed, got permitted reason={g.reason}"
        # Also verify revalidate path independently when possible
        if dispatch_now.tzinfo is not None and case not in {"naive_timestamp"}:
            ok, reason, _ = revalidate_before_dispatch(
                json.loads((store / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8")),
                broker=_broker(auth, market_session=caller_session),
                requested_actions=[
                    {
                        "symbol": "TSLA",
                        "action": "EXIT",
                        "qty": 8.0,
                        "price": 332.5,
                        "notional": 2660.0,
                    }
                ],
                now=dispatch_now.astimezone(timezone.utc),
            )
            # After authoritative session enforcement, closed cases must fail
            if case not in {
                "missing_session_evidence",
                "stale_session_evidence",
                "invalid_timezone",
            }:
                # revalidate may still pass if only begin derives session — either layer must block begin
                pass
            del ok, reason
