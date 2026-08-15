"""
Phase 198I-FIX-CLOSEOUT-EVIDENCE-CORRECTION-4/5

Same-test red-to-green directory-durability regression.

RED mode (isolated mutation fixture): restores ONLY the exact pre-fix
Windows durability exception that treated file_fsync_ok as sufficient even when
dir_fsync_ok is False — without modifying the worktree production file permanently.
Activated by env TRITON_198I_EVCORR5_RED=1 (or legacy TRITON_198I_EVCORR4_RED=1).

Observations (including fake-broker invocation and attempt count) are collected
before safety assertions so the red run actually executes the fake broker.

GREEN mode (default): current production durability_meets_dispatch_standard
fail-closes when directory durability fails, and on Windows never treats
FlushFileBuffers API success alone as proven OS-crash/power-loss metadata durability.

Invariant under directory-only durability failure (green):
- no broker-usable permit
- broker boundary not reached
- fake broker attempts == 0
- authority non-reusable after restart
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest

import services.paper_sell_arm_authority as auth_mod
from services.paper_sell_arm_authority import (
    HARDENED_AUTHORITY_FILENAME,
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    BrokerSnapshotView,
    _atomic_write_json,
    confirm_and_create_paper_sell_arm,
    durability_meets_dispatch_standard,
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
def _legacy_json_sell_arm_store(monkeypatch):
    monkeypatch.setenv("TRITON_PAPER_SELL_ARM_STORE", "legacy_json_fail_closed")


ET = ZoneInfo("America/New_York")
LIVE_BLOCKED = {
    "live_blocked": True,
    "live_block_reason": "TEST_LIVE_BLOCKED",
    "live_explicitly_authorized": False,
    "live_env_enabled": False,
    "paper_mode": True,
}

# Exact pre-fix lines reconstructed for isolated RED fixture only.
# Source: Correction-3 pre-change production behavior in durability_meets_dispatch_standard:
#   if sys.platform == "win32": return True  # after file_fsync_ok
PRE_FIX_DURABILITY_MEETS_SOURCE = """
def durability_meets_dispatch_standard(meta):
    if meta.get("flush_ok") is False:
        return False
    if not meta.get("file_fsync_ok"):
        return False
    if sys.platform == "win32":
        return True
    return bool(meta.get("dir_fsync_ok"))
""".strip()


@dataclass
class FakePlanned:
    symbol: str
    side: str
    qty: float
    limit_price: float = 332.5
    stance: str = "EXIT"
    client_order_id: str = "coid-evcorr4"
    force_exit_override: bool = False
    ref_price: Optional[float] = 332.5


class CountingFakeBroker:
    def __init__(self) -> None:
        self.attempts: List[Dict[str, Any]] = []

    def submit_order(self, **kwargs):
        self.attempts.append(dict(kwargs))
        return {"id": "fake-evcorr4", "status": "accepted"}


def _readiness() -> Dict[str, Any]:
    return {
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "cycle_id": "cycle_evcorr4",
        "run_id": "run_evcorr4",
        "batch_id": "batch_evcorr4",
        "plan_id": "plan_evcorr4",
        "plan_hash": "planhash_evcorr4",
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
        account_raw="acct-evcorr4",
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


def _last_durability() -> Dict[str, Any]:
    return dict(auth_mod.LAST_ATOMIC_WRITE_DURABILITY)


def _inject_directory_only_failure(monkeypatch, observed: Dict[str, Any]):
    real_fsync = os.fsync

    def tracking_fsync(fd: int) -> None:
        observed.setdefault("file_fsync_calls", []).append({"fd": int(fd), "via": "os.fsync"})
        return real_fsync(fd)

    def failing_dir(dir_path: Path) -> bool:
        observed.setdefault("dir_fsync_calls", []).append(
            {"path": str(dir_path), "result": False, "via": "_fsync_dir_injected"}
        )
        observed["dir_injection_active"] = True
        return False

    monkeypatch.setattr(os, "fsync", tracking_fsync)
    monkeypatch.setattr(auth_mod, "_fsync_dir", failing_dir)


def _install_pre_fix_windows_exception(monkeypatch):
    """
    Isolated mutation fixture: restore exact pre-fix durability predicate only.
    Does not write to the production source file on disk.
    """

    def pre_fix_durability_meets_dispatch_standard(meta: Dict[str, Any]) -> bool:
        if meta.get("flush_ok") is False:
            return False
        if not meta.get("file_fsync_ok"):
            return False
        if sys.platform == "win32":
            return True
        return bool(meta.get("dir_fsync_ok"))

    monkeypatch.setattr(
        auth_mod,
        "durability_meets_dispatch_standard",
        pre_fix_durability_meets_dispatch_standard,
    )
    # Also patch the name imported into this module if used locally
    monkeypatch.setattr(
        sys.modules[__name__],
        "durability_meets_dispatch_standard",
        pre_fix_durability_meets_dispatch_standard,
        raising=False,
    )


def _run_dispatch_under_dir_failure(store: Path, auth: Dict[str, Any], now: datetime):
    """
    Execute begin → (if permit) enforce → fake broker submit.
    Always attempts the fake-broker path when a permit is returned so red runs
    observe attempt_count == 1 before later safety assertions fail.
    """
    _ACTIVE_PERMIT.set(None)
    fake = CountingFakeBroker()
    g = None
    exc = None
    try:
        g = begin_paper_sell_dispatch(
            store_root=store,
            planned=[FakePlanned("TSLA", "sell", 8.0)],
            broker_view=_broker(auth),
            live_override=LIVE_BLOCKED,
            now=now,
            results_root=store,
            live_root=store,
        )
    except Exception as e:  # noqa: BLE001
        exc = repr(e)
        g = None
    returned_permit = bool(g and g.permitted and g.permit is not None)
    broker_attempts = 0
    boundary_reached = False
    broker_invoked = False
    enforce_exc = None
    if returned_permit:
        try:
            enforce_paper_sell_broker_submission(
                mode="paper",
                side="sell",
                source="evcorr5_boundary",
                symbol="TSLA",
                qty=8.0,
                client_order_id="coid-evcorr5",
            )
            boundary_reached = True
            fake.submit_order(symbol="TSLA", qty=8.0, side="sell", client_order_id="coid-evcorr5")
            broker_invoked = True
            broker_attempts = len(fake.attempts)
        except PaperSellDispatchBlocked as blocked:
            enforce_exc = blocked.reason
            boundary_reached = False
            broker_attempts = 0
            broker_invoked = False
    auth_p = store / HARDENED_AUTHORITY_FILENAME
    intent_p = store / DISPATCH_INTENT_NAME
    cons_p = store / "paper_sell_arm_consumption.json"
    auth_doc = json.loads(auth_p.read_text(encoding="utf-8")) if auth_p.is_file() else {}
    intent_doc = json.loads(intent_p.read_text(encoding="utf-8")) if intent_p.is_file() else None
    cons_doc = json.loads(cons_p.read_text(encoding="utf-8")) if cons_p.is_file() else None
    note = _last_durability()
    return {
        "g": g,
        "exc": exc,
        "enforce_exc": enforce_exc,
        "returned_permit": returned_permit,
        "boundary_reached": boundary_reached,
        "broker_invoked": broker_invoked,
        "broker_attempts": broker_attempts,
        "durability": note,
        "authority": auth_doc,
        "intent": intent_doc,
        "consumption": cons_doc,
        "reusable": auth_doc.get("status") == "ARMED"
        and auth_doc.get("consumption_status") != "CONSUMED",
    }


def test_directory_durability_uncertainty_blocks_permit_and_broker(tmp_path: Path, monkeypatch):
    """
    SAME TEST used for red (TRITON_198I_EVCORR5_RED=1) and green (default).

    Collect all observations (including fake-broker attempt count) first.
    Then assert the safety invariant (must fail under pre-fix red behavior).
    """
    store = tmp_path / "dur"
    store.mkdir()
    now = datetime(2026, 8, 11, 15, 0, 0, tzinfo=ET).astimezone(timezone.utc)
    auth = _seed(store, now=now)

    red_mode = (
        os.environ.get("TRITON_198I_EVCORR5_RED") == "1"
        or os.environ.get("TRITON_198I_EVCORR4_RED") == "1"
    )
    if red_mode:
        _install_pre_fix_windows_exception(monkeypatch)
        src = Path(auth_mod.__file__).read_text(encoding="utf-8")
        assert 'if sys.platform == "win32":\n        return False' in src.replace("\r\n", "\n")

    observed: Dict[str, Any] = {}
    _inject_directory_only_failure(monkeypatch, observed)

    probe = store / "probe.json"
    _atomic_write_json(probe, {"ok": True}, require_full_durability=False)
    probe_note = _last_durability()
    file_write_ok = probe.is_file()
    flush_ok = probe_note.get("flush_ok") is True
    file_fsync_ok = probe_note.get("file_fsync_ok") is True
    dir_fsync_ok = probe_note.get("dir_fsync_ok") is False
    atomic_replace_ok = bool(file_write_ok)

    assert file_fsync_ok is True
    assert dir_fsync_ok is True  # variable True means injected directory durability FAILED
    assert observed.get("dir_injection_active") is True
    assert len(observed.get("file_fsync_calls") or []) >= 1

    row = _run_dispatch_under_dir_failure(store, auth, now + timedelta(minutes=1))
    note = row["durability"]

    # Restart observation (collected before final green-only asserts)
    _ACTIVE_PERMIT.set(None)
    row2 = _run_dispatch_under_dir_failure(store, auth, now + timedelta(minutes=2))

    evidence = {
        "red_mode": red_mode,
        "pre_fix_source_snippet": PRE_FIX_DURABILITY_MEETS_SOURCE,
        "worktree_module": str(Path(auth_mod.__file__).resolve()),
        "file_write_ok": file_write_ok,
        "flush_ok": flush_ok,
        "file_fsync_ok": file_fsync_ok,
        "atomic_replace_ok": bool(atomic_replace_ok),
        "dir_fsync_ok": note.get("dir_fsync_ok"),
        "returned_permit": row["returned_permit"],
        "broker_invoked": row["broker_invoked"],
        "broker_attempts": row["broker_attempts"],
        "boundary_reached": row["boundary_reached"],
        "reason": getattr(row["g"], "reason", None) if row["g"] else None,
        "authority_status": row["authority"].get("status"),
        "authority_consumption": row["authority"].get("consumption_status"),
        "intent_present": row["intent"] is not None,
        "consumption_present": row["consumption"] is not None,
        "restart_returned_permit": row2["returned_permit"],
        "restart_broker_attempts": row2["broker_attempts"],
        "restart_authority_status": row2["authority"].get("status"),
        "restart_authority_consumption": row2["authority"].get("consumption_status"),
    }
    (tmp_path / "evcorr5_result.json").write_text(
        json.dumps(evidence, indent=2, default=str), encoding="utf-8"
    )
    # Print observations into pytest output for literal capture.
    print("EVCORR5_OBSERVATIONS " + json.dumps(evidence, default=str))

    assert note.get("file_fsync_ok") is True, f"file fsync must PASS: {note}"
    assert note.get("dir_fsync_ok") is False, f"dir fsync must FAIL: {note}"

    # Safety invariant — RED fails here AFTER fake broker was exercised when permit leaked.
    assert row["returned_permit"] is False, (
        f"directory durability uncertain must not return broker-usable permit; "
        f"got permit={row['returned_permit']} broker_invoked={row['broker_invoked']} "
        f"broker_attempts={row['broker_attempts']} reason={getattr(row['g'], 'reason', None)} "
        f"durability={note} red_mode={red_mode}"
    )
    assert row["broker_invoked"] is False
    assert row["boundary_reached"] is False
    assert row["broker_attempts"] == 0
    assert row2["returned_permit"] is False
    assert row2["broker_attempts"] == 0
    auth_after = row2["authority"]
    assert not (
        auth_after.get("status") == "ARMED"
        and auth_after.get("consumption_status") == "UNUSED"
        and (store / DISPATCH_INTENT_NAME).is_file()
    ), "restrictive recovery violated"


def test_windows_dir_flush_unit_mocked_boundaries(monkeypatch, tmp_path: Path):
    """
    Windows-specific unit test with mocked Win32 boundary helper outcomes.
    Proves _fsync_dir uses _fsync_dir_windows result and fail-closes when False
    if POSIX-style directory fsync also fails.
    """
    if sys.platform != "win32":
        pytest.skip("Windows-only unit test")

    calls = {"win": 0}

    def fake_ok(_p: Path) -> bool:
        calls["win"] += 1
        return True

    monkeypatch.setattr(auth_mod, "_fsync_dir_windows", fake_ok)
    assert auth_mod._fsync_dir(tmp_path) is True
    assert calls["win"] == 1

    def fake_fail(_p: Path) -> bool:
        calls["win"] += 1
        return False

    monkeypatch.setattr(auth_mod, "_fsync_dir_windows", fake_fail)

    def boom_open(*_a, **_k):
        raise OSError(13, "simulated directory open denied")

    monkeypatch.setattr(os, "open", boom_open)
    assert auth_mod._fsync_dir(tmp_path) is False
    assert calls["win"] >= 2


def test_windows_host_dir_flush_and_dispatch_without_stub(tmp_path: Path):
    """
    Actual-host capability: FlushFileBuffers on a temp directory may succeed, but
    Windows broker-usable permits remain fail-closed because metadata durability
    across OS crash/power loss is not proven from API success alone.
    """
    if sys.platform != "win32":
        pytest.skip("Windows-only host capability test")

    store = tmp_path / "host"
    store.mkdir()
    flush_ok = auth_mod._fsync_dir(store)
    assert flush_ok is True, "host directory FlushFileBuffers must succeed in temp dir"

    now = datetime(2026, 8, 11, 15, 0, 0, tzinfo=ET).astimezone(timezone.utc)
    auth = _seed(store, now=now)
    _ACTIVE_PERMIT.set(None)
    g = begin_paper_sell_dispatch(
        store_root=store,
        planned=[FakePlanned("TSLA", "sell", 8.0)],
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=now + timedelta(minutes=1),
        results_root=store,
        live_root=store,
    )
    note = _last_durability()
    assert note.get("file_fsync_ok") is True
    assert note.get("dir_fsync_ok") is True
    # API flush succeeded, but platform standard still fail-closes permits on Windows.
    assert durability_meets_dispatch_standard(note) is False
    assert g.permitted is False
    assert g.permit is None
    fake = CountingFakeBroker()
    assert len(fake.attempts) == 0
    (tmp_path / "host_capability.json").write_text(
        json.dumps(
            {
                "flush_api_ok": flush_ok,
                "durability": note,
                "meets_standard": False,
                "permitted": g.permitted,
                "reason": g.reason,
                "broker_attempts": 0,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
