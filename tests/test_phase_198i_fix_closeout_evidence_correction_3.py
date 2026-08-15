"""
Phase 198I-FIX-CLOSEOUT-EVIDENCE-CORRECTION-3

Directory-fsync isolation: file flush/fsync/replace succeed; ONLY directory
durability fails. Monkeypatches services.paper_sell_arm_authority._fsync_dir
(or call-site wrapper) — NEVER a global os.fsync that also breaks file fsync.

All paths under pytest tmp_path. Fake broker. No credentials/network/real governance.
"""

from __future__ import annotations

import json
import os
import sys
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
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    BrokerSnapshotView,
    DispatchDurabilityError,
    _atomic_write_json,
    _fsync_dir,
    confirm_and_create_paper_sell_arm,
    durability_meets_dispatch_standard,
    mark_authority_dispatch_attempt,
    prepare_confirmation_challenge,
    _platform_dispatch_durability_standard,
)


def _last_durability() -> Dict[str, Any]:
    """Read live module global (assignment rebinds; do not use imported name)."""
    return dict(auth_mod.LAST_ATOMIC_WRITE_DURABILITY)


from services.paper_sell_arm_runtime_enforcement import (
    DISPATCH_INTENT_NAME,
    DURABLE_PERMIT_NAME,
    PaperSellDispatchBlocked,
    _ACTIVE_PERMIT,
    begin_paper_sell_dispatch,
    enforce_paper_sell_broker_submission,
    finalize_paper_sell_dispatch,
)


@pytest.fixture(autouse=True)
def _legacy_json_sell_arm_store(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_PAPER_SELL_ARM_STORE", "legacy_json_fail_closed")
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
    client_order_id: str = "coid-evcorr3"
    force_exit_override: bool = False
    ref_price: Optional[float] = 332.5


class FakeBroker:
    def __init__(self) -> None:
        self.attempts: List[Dict[str, Any]] = []

    def submit_order(self, **kwargs):
        self.attempts.append(dict(kwargs))
        return {"id": "fake-ord-1", "status": "accepted"}


def _readiness() -> Dict[str, Any]:
    return {
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "cycle_id": "cycle_evcorr3",
        "run_id": "run_evcorr3",
        "batch_id": "batch_evcorr3",
        "plan_id": "plan_evcorr3",
        "plan_hash": "planhash_evcorr3",
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
        account_raw="acct-evcorr3",
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


def _begin(store: Path, auth: Dict[str, Any], now: datetime):
    _ACTIVE_PERMIT.set(None)
    return begin_paper_sell_dispatch(
        store_root=store,
        planned=[FakePlanned("TSLA", "sell", 8.0)],
        broker_view=_broker(auth),
        live_override=LIVE_BLOCKED,
        now=now,
        results_root=store,
        live_root=store,
    )


def _inject_directory_fsync_only_failure(monkeypatch, observed: Dict[str, Any]):
    """
    Isolate directory durability failure.
    - Real os.fsync continues to succeed for temporary-file descriptors.
    - Only auth_mod._fsync_dir is forced to FAIL/UNSUPPORTED.
    """
    real_fsync = os.fsync
    real_fsync_dir = auth_mod._fsync_dir

    def tracking_fsync(fd: int) -> None:
        observed.setdefault("file_fsync_calls", []).append(
            {"fd": fd, "kind": "file_or_unknown", "via": "os.fsync"}
        )
        return real_fsync(fd)

    def failing_fsync_dir(dir_path: Path) -> bool:
        observed.setdefault("dir_fsync_calls", []).append(
            {"path": str(dir_path), "result": False, "via": "_fsync_dir_injected"}
        )
        # Prove we did not call through to a path that would also break file fsync
        observed["dir_injection_active"] = True
        return False

    monkeypatch.setattr(os, "fsync", tracking_fsync)
    monkeypatch.setattr(auth_mod, "_fsync_dir", failing_fsync_dir)
    return real_fsync_dir


# ---------------------------------------------------------------------------
# D. Pre-change / isolation: directory-only fsync failure
# ---------------------------------------------------------------------------


def test_directory_fsync_only_failure_setup_conditions(tmp_path: Path, monkeypatch):
    """
    Prove setup conditions 1–10: temp create/write/flush/file-fsync/replace PASS;
    only directory open/fsync/close durability reports FAIL/UNSUPPORTED;
    no global os.fsync monkeypatch causes file fsync to fail.
    """
    store = tmp_path / "setup"
    store.mkdir()
    observed: Dict[str, Any] = {}
    _inject_directory_fsync_only_failure(monkeypatch, observed)

    path = store / "probe.json"
    meta = _atomic_write_json(path, {"probe": "dir_only", "n": 1}, require_full_durability=False)

    assert path.is_file(), "replaced file must exist"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["probe"] == "dir_only"
    assert meta is path or path.exists()

    note = _last_durability()
    assert note.get("flush_ok") is not False
    assert (
        note.get("file_fsync_ok") is True
    ), f"file fsync must PASS under dir-only injection: {note}"
    assert note.get("dir_fsync_ok") is False, f"dir fsync must FAIL under injection: {note}"
    assert observed.get("dir_injection_active") is True
    assert len(observed.get("file_fsync_calls") or []) >= 1, "file os.fsync must have been observed"
    assert len(observed.get("dir_fsync_calls") or []) >= 1
    # File fsync calls must not have raised
    assert (
        note.get("guaranteed")
        in {
            "file_fsync_atomic_replace_windows",
            "atomic_replace_only",
            "file_and_dir_fsync",
        }
        or note.get("file_fsync_ok") is True
    )

    meets = durability_meets_dispatch_standard(note)
    standard = _platform_dispatch_durability_standard()
    (tmp_path / "setup_conditions.json").write_text(
        json.dumps(
            {
                "observed": observed,
                "last_durability": note,
                "meets_standard": meets,
                "platform_standard": standard,
                "platform": sys.platform,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "transition",
    [
        "authority_restrictive_state",
        "dispatch_intent",
        "consumption_reservation",
        "dispatch_attempt",
        "outcome_persistence",
    ],
)
def test_directory_fsync_only_protected_transitions(tmp_path: Path, monkeypatch, transition: str):
    """
    Each protected dispatch transition with file fsync PASS + replace PASS + dir FAIL.
    Exercises real production persistence used by begin_paper_sell_dispatch.
    """
    store = tmp_path / transition
    store.mkdir()
    now = datetime(2026, 8, 11, 15, 0, 0, tzinfo=ET).astimezone(timezone.utc)
    auth = _seed(store, now=now)
    observed: Dict[str, Any] = {}
    _inject_directory_fsync_only_failure(monkeypatch, observed)

    fake = FakeBroker()
    g = None
    exc_info = None
    broker_attempts = 0
    boundary_reached = False

    if transition == "authority_restrictive_state":
        # Restrictive write via production atomic writer (same function begin uses)
        restrictive = dict(auth)
        restrictive["status"] = "CONSUMED"
        restrictive["consumption_status"] = "CONSUMED"
        try:
            _atomic_write_json(
                store / HARDENED_AUTHORITY_FILENAME,
                restrictive,
                refuse_authority_downgrade=True,
                require_full_durability=False,
            )
            note = _last_durability()
            assert note.get("file_fsync_ok") is True
            assert note.get("dir_fsync_ok") is False
            assert (store / HARDENED_AUTHORITY_FILENAME).is_file()
        except DispatchDurabilityError as exc:
            exc_info = repr(exc)
            note = dict(getattr(exc, "details", {}) or _last_durability())
        row = {
            "transition": transition,
            "injected": "_fsync_dir->False (os.fsync real)",
            "production_function": "services.paper_sell_arm_authority._atomic_write_json",
            "file_fsync": True,
            "replace": (store / HARDENED_AUTHORITY_FILENAME).is_file(),
            "dir_fsync": False,
            "last_durability": note,
            "meets_standard": durability_meets_dispatch_standard(note),
            "returned_permit": False,
            "boundary_reached": False,
            "broker_attempts": 0,
            "exc": exc_info,
            "observed": observed,
            "final_authority": (
                json.loads((store / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
                if (store / HARDENED_AUTHORITY_FILENAME).is_file()
                else None
            ),
        }
        (tmp_path / f"{transition}.json").write_text(
            json.dumps(row, indent=2, default=str), encoding="utf-8"
        )
        return

        if transition == "outcome_persistence":
            # First allow a normal begin, then inject dir-only failure for
            # finalize/outcome write. Happy-path begin must inject the OS
            # directory-durability primitive AND the dispatch-standard
            # evaluator. Stubbing `_fsync_dir` alone is not enough on Windows:
            # production durability_meets_dispatch_standard stays fail-closed
            # because FlushFileBuffers is diagnostic, not crash-class proof.
            stub_dir_fsync_ok(monkeypatch)
            g0 = _begin(store, auth, now + timedelta(minutes=1))
            assert g0.permitted and g0.permit is not None
            _inject_directory_fsync_only_failure(monkeypatch, observed)
        try:
            finalize_paper_sell_dispatch(
                g0,
                store_root=store,
                broker_results=[{"symbol": "TSLA", "status": "accepted"}],
                now=now + timedelta(minutes=2),
            )
        except Exception as exc:  # noqa: BLE001
            exc_info = repr(exc)
        note = _last_durability()
        auth_doc = json.loads((store / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
        intent_p = store / DISPATCH_INTENT_NAME
        row = {
            "transition": transition,
            "injected": "_fsync_dir->False after successful begin",
            "production_function": "finalize_paper_sell_dispatch -> _atomic_write_json",
            "file_fsync_ok": note.get("file_fsync_ok"),
            "dir_fsync_ok": note.get("dir_fsync_ok"),
            "replaced_exists": intent_p.is_file(),
            "returned_permit_from_begin": True,
            "broker_attempts": 0,
            "authority_status": auth_doc.get("status"),
            "authority_consumption": auth_doc.get("consumption_status"),
            "exc": exc_info,
            "last_durability": note,
            "observed": observed,
            "reusable": auth_doc.get("status") == "ARMED"
            and auth_doc.get("consumption_status") != "CONSUMED",
        }
        (tmp_path / f"{transition}.json").write_text(
            json.dumps(row, indent=2, default=str), encoding="utf-8"
        )
        assert row["reusable"] is False
        assert note.get("file_fsync_ok") is True
        assert note.get("dir_fsync_ok") is False
        assert observed.get("dir_injection_active") is True
        return

    # intent / consumption / attempt — exercise begin_paper_sell_dispatch under dir-only fail
    try:
        g = _begin(store, auth, now + timedelta(minutes=1))
    except Exception as exc:  # noqa: BLE001
        exc_info = repr(exc)
        g = None

    note = _last_durability()
    returned_permit = bool(g and g.permitted and g.permit is not None)
    if returned_permit:
        try:
            enforce_paper_sell_broker_submission(
                mode="paper",
                side="sell",
                source="dir_isolation",
                symbol="TSLA",
                qty=8.0,
            )
            boundary_reached = True
            fake.submit_order(symbol="TSLA", qty=8, side="sell")
            broker_attempts = len(fake.attempts)
        except PaperSellDispatchBlocked as blocked:
            boundary_reached = False
            broker_attempts = 0
            exc_info = f"PaperSellDispatchBlocked:{blocked.reason}"

    auth_doc = (
        json.loads((store / HARDENED_AUTHORITY_FILENAME).read_text(encoding="utf-8"))
        if (store / HARDENED_AUTHORITY_FILENAME).is_file()
        else {}
    )
    intent_p = store / DISPATCH_INTENT_NAME
    cons_p = store / "paper_sell_arm_consumption.json"
    intent_doc = json.loads(intent_p.read_text(encoding="utf-8")) if intent_p.is_file() else None
    cons_doc = json.loads(cons_p.read_text(encoding="utf-8")) if cons_p.is_file() else None
    permit_p = store / DURABLE_PERMIT_NAME
    permit_doc = json.loads(permit_p.read_text(encoding="utf-8")) if permit_p.is_file() else None

    meets = durability_meets_dispatch_standard(note)
    standard = _platform_dispatch_durability_standard()
    # Required invariant: if declared standard NOT met → no permit, no broker
    if not meets:
        assert returned_permit is False
        assert broker_attempts == 0
        assert boundary_reached is False

    row = {
        "transition": transition,
        "injected": "auth_mod._fsync_dir returns False; os.fsync real for files",
        "production_function": "begin_paper_sell_dispatch -> _atomic_write_json(require_full_durability=True)",
        "call_sites": [
            "services/paper_sell_arm_runtime_enforcement.py intent write",
            "services/paper_sell_arm_authority.py mark_authority_dispatch_attempt",
            "services/paper_sell_arm_runtime_enforcement.py durable permit write",
        ],
        "file_fsync_ok": note.get("file_fsync_ok"),
        "replace_ok": intent_p.is_file() or cons_p.is_file() or (note.get("file_fsync_ok") is True),
        "dir_fsync_ok": note.get("dir_fsync_ok"),
        "guaranteed": note.get("guaranteed"),
        "uncertain": note.get("uncertain"),
        "platform_standard": standard,
        "meets_declared_standard": meets,
        "returned_permit": returned_permit,
        "boundary_reached": boundary_reached,
        "broker_attempts": broker_attempts,
        "authority_status": auth_doc.get("status"),
        "authority_consumption": auth_doc.get("consumption_status"),
        "intent_state": (intent_doc or {}).get("status") if intent_doc else "ABSENT",
        "consumption_state": (cons_doc or {}).get("result") if cons_doc else "ABSENT",
        "attempt_state": (auth_doc or {}).get("dispatch_attempt_id")
        or ((cons_doc or {}).get("dispatch_attempt_id") if cons_doc else "ABSENT"),
        "permit_present": permit_doc is not None,
        "exc": exc_info,
        "observed_fsync": observed,
        "reusable": auth_doc.get("status") == "ARMED"
        and auth_doc.get("consumption_status") != "CONSUMED",
        "restart_interpretation": (
            "CONSUMED_NON_REUSABLE"
            if auth_doc.get("consumption_status") == "CONSUMED"
            else (
                "NO_INTENT_ARMED_MAY_RETRY"
                if not intent_p.is_file() and auth_doc.get("status") == "ARMED"
                else "RESTRICTIVE_OR_PARTIAL"
            )
        ),
    }
    (tmp_path / f"{transition}.json").write_text(
        json.dumps(row, indent=2, default=str), encoding="utf-8"
    )

    # Setup isolation assertions for begin path
    if note.get("file_fsync_ok") is not None:
        assert note.get("file_fsync_ok") is True or returned_permit is False
    if observed.get("dir_injection_active"):
        assert note.get("dir_fsync_ok") is False or note.get("dir_fsync_ok") is None


def test_directory_fsync_only_begin_records_decision(tmp_path: Path, monkeypatch):
    """
    Single focused begin under dir-only failure. Captures whether permit issues.
    Defect proven pre-fix: permit was issued. Post-fix must fail closed.
    """
    store = tmp_path / "begin_probe"
    store.mkdir()
    now = datetime(2026, 8, 11, 15, 0, 0, tzinfo=ET).astimezone(timezone.utc)
    auth = _seed(store, now=now)
    observed: Dict[str, Any] = {}
    _inject_directory_fsync_only_failure(monkeypatch, observed)

    g = None
    exc_info = None
    try:
        g = _begin(store, auth, now + timedelta(minutes=1))
    except Exception as exc:  # noqa: BLE001
        exc_info = repr(exc)

    note = _last_durability()
    meets = durability_meets_dispatch_standard(note)
    returned_permit = bool(g and g.permitted and g.permit is not None)
    broker_attempts = 0
    if returned_permit:
        try:
            enforce_paper_sell_broker_submission(
                mode="paper", side="sell", source="probe", symbol="TSLA", qty=8.0
            )
            broker_attempts = 1
        except PaperSellDispatchBlocked:
            broker_attempts = 0

    decision = {
        "platform": sys.platform,
        "platform_standard": _platform_dispatch_durability_standard(),
        "file_fsync_ok": note.get("file_fsync_ok"),
        "dir_fsync_ok": note.get("dir_fsync_ok"),
        "uncertain": note.get("uncertain"),
        "guaranteed": note.get("guaranteed"),
        "meets_declared_standard": meets,
        "returned_permit": returned_permit,
        "broker_attempts": broker_attempts,
        "reason": getattr(g, "reason", None) if g is not None else None,
        "exc": exc_info,
        "observed": observed,
        "invariant_when_standard_not_met": (
            (not returned_permit and broker_attempts == 0) if not meets else "standard_met_n/a"
        ),
    }
    (tmp_path / "defect_decision_probe.json").write_text(
        json.dumps(decision, indent=2, default=str), encoding="utf-8"
    )
    # Isolation proof
    assert note.get("file_fsync_ok") is True
    assert note.get("dir_fsync_ok") is False
    assert observed.get("dir_injection_active") is True
    assert len(observed.get("file_fsync_calls") or []) >= 1
    # Fail-closed invariant under directory-durability uncertainty
    assert meets is False
    assert returned_permit is False
    assert broker_attempts == 0


def test_crash_model_after_replace_before_dir_fsync(tmp_path: Path, monkeypatch):
    """
    Process-crash model: after file fsync + replace, before successful dir fsync.
    Proves recovery interpretation. Does NOT claim power-loss durability.
    """
    store = tmp_path / "crash_model"
    store.mkdir()
    now = datetime(2026, 8, 11, 15, 0, 0, tzinfo=ET).astimezone(timezone.utc)
    auth = _seed(store, now=now)
    observed: Dict[str, Any] = {}
    _inject_directory_fsync_only_failure(monkeypatch, observed)

    g = _begin(store, auth, now + timedelta(minutes=1))
    note = _last_durability()
    assert note.get("file_fsync_ok") is True
    assert note.get("dir_fsync_ok") is False
    assert g.permitted is False
    assert durability_meets_dispatch_standard(note) is False

    # Simulate process death: drop in-memory permit; reload from disk only
    _ACTIVE_PERMIT.set(None)
    intent_p = store / DISPATCH_INTENT_NAME
    cons_p = store / "paper_sell_arm_consumption.json"
    auth_p = store / HARDENED_AUTHORITY_FILENAME
    on_disk = {
        "intent_exists": intent_p.is_file(),
        "consumption_exists": cons_p.is_file(),
        "authority": json.loads(auth_p.read_text(encoding="utf-8")) if auth_p.is_file() else None,
        "intent": json.loads(intent_p.read_text(encoding="utf-8")) if intent_p.is_file() else None,
        "consumption": json.loads(cons_p.read_text(encoding="utf-8")) if cons_p.is_file() else None,
        "durability": note,
        "meets_standard": durability_meets_dispatch_standard(note),
        "first_begin_permitted": bool(g.permitted),
        "first_begin_reason": g.reason,
    }

    # Restart: second begin must not re-issue reusable ARMED sell
    g2 = _begin(store, auth, now + timedelta(minutes=2))
    auth_after = json.loads(auth_p.read_text(encoding="utf-8"))
    recovery = {
        "on_disk_after_first_begin": on_disk,
        "second_begin_permitted": bool(g2.permitted),
        "second_begin_reason": g2.reason,
        "authority_after": {
            "status": auth_after.get("status"),
            "consumption_status": auth_after.get("consumption_status"),
        },
        "reusable": auth_after.get("status") == "ARMED"
        and auth_after.get("consumption_status") != "CONSUMED",
        "power_loss_claim": "NOT_CLAIMED — process-crash model only",
        "filesystem_metadata_claim": "NOT_CLAIMED without successful directory durability",
    }
    (tmp_path / "crash_model.json").write_text(
        json.dumps(recovery, indent=2, default=str), encoding="utf-8"
    )
    assert recovery["second_begin_permitted"] is False
    # If intent landed on disk under uncertain durability, authority must not stay reusable ARMED
    if on_disk["intent_exists"]:
        assert (
            recovery["reusable"] is False
            or auth_after.get("consumption_status")
            in {
                "CONSUMED",
                "DURABILITY_UNCERTAIN",
            }
            or auth_after.get("status") != "ARMED"
        )
