"""
Phase 198H — isolated non-executing tests for paper sell-arm authority.
All writes use tmp_path only. Never touches live governance/runtime artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from services.paper_sell_arm_authority import (
    APPROVAL_CLI_CAP,
    AUTHORITY_TTL_MINUTES,
    AUTHORITY_TYPE,
    CHALLENGE_ARTIFACT_TYPE,
    CHALLENGE_TTL_MINUTES,
    LIVE_APPROVAL_PATH,
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    SCHEMA_VERSION,
    SellArmState,
    BrokerSnapshotView,
    allowed_transition,
    challenge_rejected_by_approval_validator,
    confirm_and_create_paper_sell_arm,
    inspect_paper_sell_arm,
    mark_authority_dispatch_attempt,
    paper_buy_arm_rejected_for_sells,
    prepare_confirmation_challenge,
    readiness_packet_hash,
    revalidate_before_dispatch,
    validate_paper_sell_arm_authority,
)


ROOT = Path(__file__).resolve().parents[1]

# Freeze creation/validation clock to regular-session hours so AFTER_CLOSE wall
# clock cannot clamp authority expires_at to an already-elapsed day_end.
FROZEN_NOW = datetime(2026, 8, 11, 15, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolate_canonical_sqlite(tmp_path, monkeypatch):
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", tmp_path / store_mod.DEFAULT_DB_NAME)


def _readiness(**overrides):
    base = {
        "status": "UNAPPROVED_APPROVAL_READINESS_ONLY",
        "mode": "PAPER",
        "live_execution_state": "BLOCKED",
        "creates_no_authority": True,
        "cycle_id": "cycle_test_1",
        "run_id": "2026-08-11T15:49:22Z",
        "batch_id": "batch_test_1",
        "plan_id": "plan_test_1",
        "plan_hash": "planhash_abc123",
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


def _broker(**overrides):
    b = BrokerSnapshotView(
        positions={"TSLA": {"qty": 8.0, "qty_available": 8.0}},
        open_order_symbols=set(),
        mode="paper",
        live_blocked=True,
        trading_date="2026-08-11",
        account_fingerprint=None,
        current_plan_hash="planhash_abc123",
        newer_completed_plan=False,
        market_session="MARKET_OPEN",
        paper_buy_arm_valid=True,
        paper_execution_authorized=True,
        cycle_id="cycle_test_1",
        run_id="2026-08-11T15:49:22Z",
        batch_id="batch_test_1",
        plan_id="plan_test_1",
        readiness_packet_hash=None,
        confirmation_challenge_id="chal-test",
        confirmation_challenge_hash="chalhash-test",
        evidence_independent=True,
    )
    for k, v in overrides.items():
        setattr(b, k, v)
    return b


def test_first_click_creates_no_authority(tmp_path: Path):
    readiness = _readiness()
    chal = prepare_confirmation_challenge(
        store_root=tmp_path, readiness_packet=readiness, account_raw="acct-1"
    )
    assert chal["creates_no_authority"] is True
    assert chal["artifact_type"] == CHALLENGE_ARTIFACT_TYPE
    assert not (tmp_path / "paper_sell_arm_authority.json").exists()
    assert not (tmp_path / "operator_paper_defensive_exit_approval.json").exists()
    insp = inspect_paper_sell_arm(store_root=tmp_path)
    assert insp["authority_present"] is False
    assert insp["state"] == SellArmState.CONFIRMATION_CHALLENGE_ISSUED.value


def test_challenge_non_authoritative_and_rejected_by_validators(tmp_path: Path):
    chal = prepare_confirmation_challenge(
        store_root=tmp_path, readiness_packet=_readiness(), account_raw="acct-1"
    )
    rejected, reason = challenge_rejected_by_approval_validator(chal)
    assert rejected
    assert "not_authority" in reason or "mismatch" in reason or "missing" in reason

    from services.defensive_exit_operator_paper_approval import (
        APPROVAL_STATUS_VALID,
        validate_operator_approval_contract,
    )

    # Place challenge at temp contract path — must not be VALID.
    path = tmp_path / "operator_paper_defensive_exit_approval.json"
    path.write_text(json.dumps(chal), encoding="utf-8")
    st, reason2, _ = validate_operator_approval_contract(contract_path=path)
    assert st != APPROVAL_STATUS_VALID
    assert "challenge" in reason2 or "invalid" in reason2.lower() or "hardened" in reason2


def test_second_step_requires_ack_and_exact_challenge(tmp_path: Path):
    readiness = _readiness()
    chal = prepare_confirmation_challenge(
        store_root=tmp_path, readiness_packet=readiness, account_raw="acct-1"
    )
    with pytest.raises(ValueError, match="acknowledgement_mismatch"):
        confirm_and_create_paper_sell_arm(
            store_root=tmp_path,
            readiness_packet=readiness,
            challenge=chal,
            operator_acknowledgement_text="wrong",
        )
    bad = dict(chal)
    bad["challenge_hash"] = "deadbeef"
    with pytest.raises(ValueError, match="challenge_hash_integrity_failed"):
        confirm_and_create_paper_sell_arm(
            store_root=tmp_path,
            readiness_packet=readiness,
            challenge=bad,
            operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        )
    auth = confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
    )
    assert auth["status"] == SellArmState.ARMED.value
    assert auth["orders_submitted"] is False
    assert auth["creates_dispatch_authority"] is False
    assert auth["authority_type"] == AUTHORITY_TYPE
    assert auth["schema_version"] == SCHEMA_VERSION


def test_challenge_replay_rejected(tmp_path: Path):
    readiness = _readiness()
    chal = prepare_confirmation_challenge(
        store_root=tmp_path, readiness_packet=readiness, account_raw="acct-1"
    )
    confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
    )
    with pytest.raises(ValueError, match="challenge_replay|existing_armed"):
        confirm_and_create_paper_sell_arm(
            store_root=tmp_path,
            readiness_packet=readiness,
            challenge=chal,
            operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
            current_plan_hash=readiness["plan_hash"],
        )


def test_changed_hashes_and_ids_rejected(tmp_path: Path):
    readiness = _readiness()
    chal = prepare_confirmation_challenge(
        store_root=tmp_path, readiness_packet=readiness, account_raw="acct-1"
    )
    # Changed readiness packet hash
    readiness2 = _readiness(total_proposed_notional=2661.0)
    with pytest.raises(ValueError, match="readiness_hash_mismatch"):
        confirm_and_create_paper_sell_arm(
            store_root=tmp_path,
            readiness_packet=readiness2,
            challenge=chal,
            operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        )
    with pytest.raises(ValueError, match="plan_hash_superseded"):
        confirm_and_create_paper_sell_arm(
            store_root=tmp_path,
            readiness_packet=readiness,
            challenge=chal,
            operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
            current_plan_hash="other_hash",
        )


def test_expired_challenge_and_authority_rejected(tmp_path: Path):
    """Deterministic TTL boundaries. Injected clock only — no wall-clock sleep."""
    readiness = _readiness()
    issued = datetime(2026, 8, 11, 12, 0, 0, tzinfo=timezone.utc)
    challenge_expires = issued + timedelta(minutes=CHALLENGE_TTL_MINUTES)
    db1 = tmp_path / "paper_sell_arm.sqlite3"

    chal = prepare_confirmation_challenge(
        store_root=tmp_path,
        readiness_packet=readiness,
        account_raw="acct-1",
        now=issued,
        trading_date="2026-08-11",
        db_path=db1,
    )
    assert chal["issued_at"].startswith("2026-08-11T12:00:00")
    assert chal["expires_at"].startswith(challenge_expires.isoformat())

    with pytest.raises(ValueError, match="challenge_expired"):
        confirm_and_create_paper_sell_arm(
            store_root=tmp_path,
            readiness_packet=readiness,
            challenge=chal,
            operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
            now=challenge_expires,
            db_path=db1,
        )
    with pytest.raises(ValueError, match="challenge_expired"):
        confirm_and_create_paper_sell_arm(
            store_root=tmp_path,
            readiness_packet=readiness,
            challenge=chal,
            operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
            now=challenge_expires + timedelta(seconds=1),
            db_path=db1,
        )

    store_before = tmp_path / "chal_before"
    db_before = store_before / "paper_sell_arm.sqlite3"
    chal_before = prepare_confirmation_challenge(
        store_root=store_before,
        readiness_packet=readiness,
        account_raw="acct-1",
        now=issued,
        trading_date="2026-08-11",
        db_path=db_before,
    )
    auth_before_challenge = confirm_and_create_paper_sell_arm(
        store_root=store_before,
        readiness_packet=readiness,
        challenge=chal_before,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        now=challenge_expires - timedelta(seconds=1),
        current_plan_hash=readiness["plan_hash"],
        db_path=db_before,
    )
    assert auth_before_challenge["status"] == SellArmState.ARMED.value

    store2 = tmp_path / "a2"
    db2 = store2 / "paper_sell_arm.sqlite3"
    chal2 = prepare_confirmation_challenge(
        store_root=store2,
        readiness_packet=readiness,
        account_raw="acct-1",
        now=issued,
        trading_date="2026-08-11",
        db_path=db2,
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=store2,
        readiness_packet=readiness,
        challenge=chal2,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        now=issued,
        current_plan_hash=readiness["plan_hash"],
        db_path=db2,
    )
    assert auth["status"] == SellArmState.ARMED.value
    assert auth["consumption_status"] == "UNUSED"
    authority_expires = issued + timedelta(minutes=AUTHORITY_TTL_MINUTES)
    assert auth["issued_at"].startswith("2026-08-11T12:00:00")
    assert auth["expires_at"].startswith(authority_expires.isoformat())

    one_second = timedelta(seconds=1)
    ok_before, reason_before, _ = validate_paper_sell_arm_authority(
        auth, now=authority_expires - one_second
    )
    assert ok_before and reason_before == "ok"

    ok_at, reason_at, _ = validate_paper_sell_arm_authority(auth, now=authority_expires)
    assert not ok_at
    assert reason_at == "authority_expired"

    ok_after, reason_after, _ = validate_paper_sell_arm_authority(
        auth, now=authority_expires + one_second
    )
    assert not ok_after
    assert reason_after == "authority_expired"

    ok_stale, reason_stale, _ = validate_paper_sell_arm_authority(
        json.loads((store2 / "paper_sell_arm_authority.json").read_text(encoding="utf-8")),
        now=authority_expires + timedelta(minutes=1),
    )
    assert not ok_stale
    assert reason_stale == "authority_expired"


def test_preview_and_readiness_rejected_as_authority():
    preview = {"preview_status": "UNAPPROVED_PREVIEW_ONLY", "creates_no_authority": True}
    ok, reason, _ = validate_paper_sell_arm_authority(preview)
    assert not ok
    readiness = _readiness()
    ok2, reason2, _ = validate_paper_sell_arm_authority(readiness)
    assert not ok2


def test_paper_buy_arm_rejected_for_sells():
    buy = {
        "timestamp": "2026-08-11T13:29:31Z",
        "mode": "ASSISTED",
        "expires_at": "2026-08-11T21:29:31Z",
        "allow_execute": True,
    }
    rejected, reason = paper_buy_arm_rejected_for_sells(buy)
    assert rejected


def test_sell_arm_rejects_buy_or_add_scope(tmp_path: Path):
    readiness = _readiness(
        approval_ready_actions=[
            {"symbol": "TSLA", "action": "BUY", "qty": 1, "est_notional": 100.0}
        ],
        approval_ready_symbols=["TSLA"],
    )
    # BUY filtered out of scope -> empty -> challenge blocked
    with pytest.raises(ValueError, match="empty_scope|challenge_blocked"):
        prepare_confirmation_challenge(
            store_root=tmp_path, readiness_packet=readiness, account_raw="acct-1"
        )


def test_live_mode_and_live_enabled_rejected(tmp_path: Path):
    readiness = _readiness()
    chal = prepare_confirmation_challenge(
        store_root=tmp_path, readiness_packet=readiness, account_raw="acct-1"
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
    )
    auth["mode"] = "LIVE"
    ok, reason, _ = validate_paper_sell_arm_authority(auth)
    assert not ok and reason == "mode_not_paper"
    auth["mode"] = "PAPER"
    auth["live_authorized"] = True
    ok2, reason2, _ = validate_paper_sell_arm_authority(auth)
    assert not ok2 and reason2 == "live_authorized_true"


def test_account_trading_date_cycle_batch_mismatches(tmp_path: Path):
    readiness = _readiness()
    chal = prepare_confirmation_challenge(
        store_root=tmp_path,
        readiness_packet=readiness,
        account_raw="acct-1",
        trading_date="2026-08-11",
        now=FROZEN_NOW,
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
        now=FROZEN_NOW,
    )
    ok, reason, _ = validate_paper_sell_arm_authority(
        auth, expected_account_fingerprint="acctfp_other", now=FROZEN_NOW
    )
    assert not ok and reason == "account_fingerprint_mismatch"
    ok2, reason2, _ = validate_paper_sell_arm_authority(
        auth, expected_trading_date="2026-08-12", now=FROZEN_NOW
    )
    assert not ok2 and reason2 == "trading_date_mismatch"
    ok3, reason3, _ = validate_paper_sell_arm_authority(
        auth, expected_cycle_id="other_cycle", now=FROZEN_NOW
    )
    assert not ok3 and reason3 == "cycle_id_mismatch"
    ok4, reason4, _ = validate_paper_sell_arm_authority(
        auth, expected_batch_id="other_batch", now=FROZEN_NOW
    )
    assert not ok4 and reason4 == "batch_id_mismatch"


def _broker_for_auth(auth, **overrides):
    return _broker(
        account_fingerprint=auth["account_fingerprint"],
        trading_date=auth["trading_date"],
        current_plan_hash=auth["plan_hash"],
        cycle_id=auth["cycle_id"],
        run_id=auth["run_id"],
        batch_id=auth["batch_id"],
        plan_id=auth["plan_id"],
        readiness_packet_hash=auth["readiness_packet_hash"],
        confirmation_challenge_id=auth["confirmation_challenge_id"],
        confirmation_challenge_hash=auth["confirmation_challenge_hash"],
        **overrides,
    )


def test_supersession_position_open_order_qty_caps(tmp_path: Path):
    readiness = _readiness()
    chal = prepare_confirmation_challenge(
        store_root=tmp_path,
        readiness_packet=readiness,
        account_raw="acct-1",
        trading_date="2026-08-11",
        now=FROZEN_NOW,
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
        now=FROZEN_NOW,
    )
    req = [{"symbol": "TSLA", "action": "EXIT", "qty": 8, "price": 332.5}]
    rhash = auth["readiness_packet_hash"]

    ok, reason, _ = revalidate_before_dispatch(
        auth,
        broker=_broker_for_auth(auth, newer_completed_plan=True),
        requested_actions=req,
        readiness_packet_hash_current=rhash,
        now=FROZEN_NOW,
    )
    assert not ok and reason == "newer_management_plan_supersedes"

    ok2, reason2, _ = revalidate_before_dispatch(
        auth,
        broker=_broker_for_auth(auth, positions={"TSLA": {"qty": 7.0, "qty_available": 7.0}}),
        requested_actions=req,
        readiness_packet_hash_current=rhash,
        now=FROZEN_NOW,
    )
    assert not ok2 and "held_qty_insufficient" in reason2

    ok3, reason3, _ = revalidate_before_dispatch(
        auth,
        broker=_broker_for_auth(auth, positions={"TSLA": {"qty": 8.0, "qty_available": 5.0}}),
        requested_actions=req,
        readiness_packet_hash_current=rhash,
        now=FROZEN_NOW,
    )
    assert not ok3 and ("available_qty_insufficient" in reason3 or "shares_reserved" in reason3)

    ok4, reason4, _ = revalidate_before_dispatch(
        auth,
        broker=_broker_for_auth(auth, open_order_symbols={"TSLA"}),
        requested_actions=req,
        readiness_packet_hash_current=rhash,
        now=FROZEN_NOW,
    )
    assert not ok4 and "open_order_conflict" in reason4

    # quantity expansion
    ok5, reason5, _ = revalidate_before_dispatch(
        auth,
        broker=_broker_for_auth(auth),
        requested_actions=[{"symbol": "TSLA", "action": "EXIT", "qty": 9, "price": 332.5}],
        readiness_packet_hash_current=rhash,
        now=FROZEN_NOW,
    )
    assert not ok5 and "quantity_expansion" in reason5

    # action substitution
    ok6, reason6, _ = revalidate_before_dispatch(
        auth,
        broker=_broker_for_auth(auth),
        requested_actions=[{"symbol": "TSLA", "action": "TRIM", "qty": 8, "price": 332.5}],
        readiness_packet_hash_current=rhash,
        now=FROZEN_NOW,
    )
    assert not ok6 and "action_substitution" in reason6

    # batch expansion / symbol substitution
    ok7, reason7, _ = revalidate_before_dispatch(
        auth,
        broker=_broker_for_auth(
            auth,
            positions={
                "TSLA": {"qty": 8.0, "qty_available": 8.0},
                "GLD": {"qty": 16.0, "qty_available": 16.0},
            },
        ),
        requested_actions=[
            {"symbol": "TSLA", "action": "EXIT", "qty": 8, "price": 332.5},
            {"symbol": "GLD", "action": "EXIT", "qty": 16, "price": 400.0},
        ],
        readiness_packet_hash_current=rhash,
        now=FROZEN_NOW,
    )
    assert not ok7 and "batch_symbol_set_mismatch" in reason7

    # happy path
    ok8, reason8, _ = revalidate_before_dispatch(
        auth,
        broker=_broker_for_auth(auth),
        requested_actions=req,
        readiness_packet_hash_current=rhash,
        now=FROZEN_NOW,
    )
    assert ok8 and reason8 == "ok"


def test_approval_cli_cap_and_max_qty_enforced(tmp_path: Path):
    readiness = _readiness(
        approval_ready_actions=[
            {
                "symbol": "BITO",
                "action": "EXIT",
                "qty": 1365,
                "est_notional": 12000.0,
            }
        ],
        approval_ready_symbols=["BITO"],
        total_proposed_notional=12000.0,
    )
    with pytest.raises(ValueError, match="exceeds_approval_cli_cap|exceeds_cap"):
        prepare_confirmation_challenge(
            store_root=tmp_path, readiness_packet=readiness, account_raw="acct-1"
        )
    assert APPROVAL_CLI_CAP == 5000.0


def test_consumed_authority_and_indeterminate_cannot_reuse(tmp_path: Path):
    readiness = _readiness()
    chal = prepare_confirmation_challenge(
        store_root=tmp_path,
        readiness_packet=readiness,
        account_raw="acct-1",
        trading_date="2026-08-11",
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
    )
    from services.paper_sell_arm_store import PaperSellArmStoreError

    with pytest.raises(PaperSellArmStoreError) as caught:
        mark_authority_dispatch_attempt(store_root=tmp_path, authority=auth, result="INDETERMINATE")
    assert caught.value.reason == "json_consume_rejected_sqlite_not_consumed"
    loaded = json.loads((tmp_path / "paper_sell_arm_authority.json").read_text(encoding="utf-8"))
    assert loaded.get("consumption_status") != "CONSUMED"
    assert loaded.get("status") == "ARMED"
    cons_path = tmp_path / "paper_sell_arm_consumption.json"
    assert (
        not cons_path.is_file()
        or json.loads(cons_path.read_text(encoding="utf-8")).get("result") != "INDETERMINATE"
    )


def test_malformed_timestamps_and_missing_fields_fail_closed():
    ok, reason, _ = validate_paper_sell_arm_authority(
        {
            "authority_type": AUTHORITY_TYPE,
            "schema_version": SCHEMA_VERSION,
            "status": "ARMED",
        }
    )
    assert not ok
    assert "missing_required_fields" in reason

    readiness = _readiness()
    # Build a valid one in memory via tmp then break timestamp
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        root = Path(td)
        chal = prepare_confirmation_challenge(
            store_root=root,
            readiness_packet=readiness,
            account_raw="acct-1",
            trading_date="2026-08-11",
        )
        auth = confirm_and_create_paper_sell_arm(
            store_root=root,
            readiness_packet=readiness,
            challenge=chal,
            operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
            current_plan_hash=readiness["plan_hash"],
        )
        auth["expires_at"] = "not-a-timestamp"
        ok2, reason2, _ = validate_paper_sell_arm_authority(auth)
        assert not ok2 and "timestamp_unparseable" in reason2


def test_historical_weaker_schema_fails_closed():
    ok, reason, _ = validate_paper_sell_arm_authority(
        {
            "authority_type": AUTHORITY_TYPE,
            "schema_version": "197W-WEAK",
            "status": "ARMED",
            "mode": "PAPER",
        }
    )
    assert not ok
    assert reason == "unsupported_or_historical_schema"


def test_authority_creation_does_not_dispatch_and_inspect_readonly(tmp_path: Path):
    readiness = _readiness()
    chal = prepare_confirmation_challenge(
        store_root=tmp_path, readiness_packet=readiness, account_raw="acct-1"
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
    )
    assert auth["orders_submitted"] is False
    assert auth["broker_execution_authorized"] is False
    before = list(tmp_path.iterdir())
    insp = inspect_paper_sell_arm(store_root=tmp_path)
    assert insp["read_only"] is True
    after = list(tmp_path.iterdir())
    # inspect must not create new files
    assert set(after) == set(before)


def test_no_runtime_artifact_touched_by_tests():
    # Live approval must remain absent or unchanged by this module's tests.
    # We only assert the path is not the tmp path and module refuses silent live writes.
    assert LIVE_APPROVAL_PATH.name == "paper_sell_arm_authority.json"
    assert "governance" in LIVE_APPROVAL_PATH.parts
    with pytest.raises(PermissionError):
        prepare_confirmation_challenge(
            store_root=LIVE_APPROVAL_PATH.parent,
            readiness_packet=_readiness(),
            account_raw="acct-1",
            explicit_runtime_write=False,
        )


def test_state_machine_no_direct_jump_to_armed():
    assert not allowed_transition(SellArmState.REVIEW_REQUIRED, SellArmState.ARMED)
    assert not allowed_transition(SellArmState.READY_FOR_CONFIRMATION, SellArmState.ARMED)
    assert allowed_transition(SellArmState.CONFIRMATION_CHALLENGE_ISSUED, SellArmState.ARMED)


def test_hardened_authority_can_pass_approval_gate_in_isolation(tmp_path: Path, monkeypatch):
    from services.defensive_exit_operator_paper_approval import (
        APPROVAL_STATUS_VALID,
        validate_operator_approval_contract,
    )
    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_APPROVAL_REQUIRED,
        DISPATCH_BLOCKED_ORDER_GUARD,
        evaluate_defensive_exit_dispatch_interlock,
    )
    import services.paper_sell_arm_authority as auth_mod
    import services.defensive_exit_operator_paper_approval as appr_mod
    from tests._phase_198i_interlock_test_fixtures import (
        install_hermetic_interlock,
        write_valid_order_guard_fixture,
    )

    monkeypatch.setattr(auth_mod, "_utc_now", lambda: FROZEN_NOW)
    monkeypatch.setattr(appr_mod, "_utc_now", lambda: FROZEN_NOW)

    readiness = _readiness()
    chal = prepare_confirmation_challenge(
        store_root=tmp_path,
        readiness_packet=readiness,
        account_raw="acct-1",
        trading_date="2026-08-11",
        now=FROZEN_NOW,
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=tmp_path,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
        now=FROZEN_NOW,
    )
    path = tmp_path / "paper_sell_arm_authority.json"
    st, reason, details = validate_operator_approval_contract(contract_path=path)
    assert st == APPROVAL_STATUS_VALID
    assert details.get("hardened_sell_arm") is True
    assert auth["orders_submitted"] is False

    roots = install_hermetic_interlock(monkeypatch, tmp_path)
    write_valid_order_guard_fixture(roots["results"])
    assert not roots["hardened"].is_file()
    interlock = evaluate_defensive_exit_dispatch_interlock(
        mode="paper",
        broker_snapshot_ok=True,
        require_market_open=False,
        planned_count=1,
    )
    assert interlock.get("dispatch_interlock_state") == DISPATCH_BLOCKED_APPROVAL_REQUIRED
    assert interlock.get("dispatch_interlock_state") != DISPATCH_BLOCKED_ORDER_GUARD
    assert interlock.get("broker_orders_submitted") is False
    assert auth["orders_submitted"] is False


def test_hardened_authority_interlock_blocks_when_order_guard_missing(tmp_path: Path, monkeypatch):
    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_ORDER_GUARD,
        evaluate_defensive_exit_dispatch_interlock,
    )
    from tests._phase_198i_interlock_test_fixtures import install_hermetic_interlock

    roots = install_hermetic_interlock(monkeypatch, tmp_path)
    assert not (roots["results"] / "guard_snapshot.json").is_file()
    result = evaluate_defensive_exit_dispatch_interlock(
        mode="paper",
        broker_snapshot_ok=True,
        require_market_open=False,
        planned_count=1,
    )
    assert result.get("dispatch_interlock_state") == DISPATCH_BLOCKED_ORDER_GUARD
    assert result.get("broker_orders_submitted") is False


def test_hardened_authority_at_isolated_path_passes_order_guard_gate(tmp_path: Path, monkeypatch):
    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_ORDER_GUARD,
        evaluate_defensive_exit_dispatch_interlock,
    )
    import services.paper_sell_arm_authority as auth_mod
    import services.paper_sell_arm_runtime_enforcement as enf_mod
    from tests._phase_198i_interlock_test_fixtures import (
        create_sqlite_sell_arm_via_public_apis,
        install_hermetic_interlock,
        write_valid_order_guard_fixture,
    )

    monkeypatch.setattr(auth_mod, "_utc_now", lambda: FROZEN_NOW)
    monkeypatch.setattr(enf_mod, "_utc_now", lambda: FROZEN_NOW)
    roots = install_hermetic_interlock(monkeypatch, tmp_path)
    write_valid_order_guard_fixture(roots["results"])
    create_sqlite_sell_arm_via_public_apis(
        roots["gov"], now=FROZEN_NOW, explicit_runtime_write=True
    )
    result = evaluate_defensive_exit_dispatch_interlock(
        mode="paper",
        broker_snapshot_ok=True,
        require_market_open=False,
        planned_count=1,
    )
    assert result.get("dispatch_interlock_state") != DISPATCH_BLOCKED_ORDER_GUARD
    assert result.get("hardened_sell_arm_ok") is True
    assert result.get("broker_orders_submitted") is False
    assert result.get("eligible_for_execution") is False


def test_readiness_hash_stable_for_material_fields():
    a = readiness_packet_hash(_readiness())
    b = readiness_packet_hash(_readiness())
    assert a == b
    c = readiness_packet_hash(_readiness(plan_hash="other"))
    assert a != c


def test_cli_refuses_explicit_runtime_write_flag():
    from services.paper_sell_arm_cli import main

    rc = main(
        [
            "prepare-challenge",
            "--store-root",
            str(ROOT / "tmp" / "should_not_matter"),
            "--readiness-json",
            str(ROOT / "data" / "results" / "defensive_exit_approval_readiness.json"),
            "--explicit-runtime-write",
        ]
    )
    assert rc == 2
