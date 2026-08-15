"""
Test-only hermetic fixtures for defensive-exit dispatch interlock evaluation.

Redirects order-guard, live-block, approval, and authority loaders to pytest
tmp_path. Never writes repository data/live/. Never contacts a broker.
Does not change production constants or evaluation order.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from tests._paper_sell_test_durability import stub_dir_fsync_ok

TEST_REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_LIVE = TEST_REPO_ROOT / "data" / "live"
GUARD_NOW = datetime(2026, 8, 11, 16, 0, 0, tzinfo=timezone.utc)


def sample_sell_readiness_packet(**overrides):
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


LIVE_BLOCKED_PAYLOAD = {
    "live_blocked": True,
    "live_block_reason": "TEST_LIVE_BLOCKED",
    "live_explicitly_authorized": False,
    "live_env_enabled": False,
    "paper_mode": True,
}


def ban_repo_live_io(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_read = Path.read_text
    orig_open = Path.open

    def _reject_if_repo_live(path: Path) -> None:
        try:
            resolved = path.resolve()
            live = REPO_LIVE.resolve()
        except OSError:
            return
        if resolved == live or live in resolved.parents:
            raise AssertionError("hermetic test read repository data/live/")

    def read_text(self, *args, **kwargs):
        _reject_if_repo_live(self)
        return orig_read(self, *args, **kwargs)

    def open_path(self, *args, **kwargs):
        _reject_if_repo_live(self)
        return orig_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "open", open_path)


def block_network_and_live_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "APCA_API_BASE_URL",
        "ALPACA_BASE_URL",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_FORCE_LIVE", "0")

    def _blocked(*_a, **_k):
        raise RuntimeError("network blocked in hermetic interlock tests")

    monkeypatch.setattr(socket, "create_connection", _blocked)


def isolate_interlock_artifact_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Dict[str, Path]:
    results = tmp_path / "data" / "results"
    live = tmp_path / "data" / "live"
    gov = tmp_path / "data" / "governance"
    results.mkdir(parents=True, exist_ok=True)
    live.mkdir(parents=True, exist_ok=True)
    gov.mkdir(parents=True, exist_ok=True)

    import services.defensive_exit_dispatch_interlock as interlock
    import services.defensive_exit_governance_review_criteria as criteria
    import services.defensive_exit_operator_paper_approval as approval_mod
    import services.live_block_classifier as live_block
    import services.order_guard_snapshot_authority as order_guard
    import services.paper_operator_approval_artifact_state as approval_state
    import services.paper_sell_arm_authority as authority
    import services.paper_sell_arm_runtime_enforcement as enforcement

    hardened = gov / authority.HARDENED_AUTHORITY_FILENAME

    monkeypatch.setattr(order_guard, "ROOT", tmp_path)
    monkeypatch.setattr(order_guard, "RESULTS", results)
    monkeypatch.setattr(order_guard, "LIVE", live)
    monkeypatch.setattr(order_guard, "GOVERNANCE", gov)
    monkeypatch.setattr(
        order_guard,
        "GUARD_PATHS",
        [results / "guard_snapshot.json", live / "guard_snapshot.json"],
    )
    monkeypatch.setattr(order_guard, "OPEN_ORDERS_JSON", results / "open_orders_snapshot.csv")
    monkeypatch.setattr(order_guard, "RECENT_ORDERS_JSON", results / "recent_orders.csv")
    monkeypatch.setattr(order_guard, "POSITIONS_JSON", results / "positions_snapshot.csv")
    monkeypatch.setattr(order_guard, "LIVE_ORDERS_RESULTS", results / "live_orders.csv")
    monkeypatch.setattr(order_guard, "LIVE_ORDERS_LIVE", live / "live_orders.csv")
    monkeypatch.setattr(order_guard, "REAL_APPROVAL_PATH", hardened)

    monkeypatch.setattr(authority, "LIVE_GOVERNANCE", gov)
    monkeypatch.setattr(authority, "LIVE_APPROVAL_PATH", hardened)
    monkeypatch.setattr(enforcement, "LIVE_GOVERNANCE", gov)
    import services.paper_sell_arm_store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_RUNTIME_DB", live / store_mod.DEFAULT_DB_NAME)

    monkeypatch.setattr(interlock, "ROOT", tmp_path)
    monkeypatch.setattr(interlock, "RESULTS", results)
    monkeypatch.setattr(interlock, "GOVERNANCE", gov)
    monkeypatch.setattr(interlock, "REAL_APPROVAL_PATH", hardened)
    monkeypatch.setattr(
        interlock, "PREVIEW_JSON", results / "defensive_exit_capped_batch_preview.json"
    )

    monkeypatch.setattr(live_block, "ROOT", tmp_path)
    monkeypatch.setattr(live_block, "RESULTS", results)
    monkeypatch.setattr(live_block, "LIVE", live)
    monkeypatch.setattr(live_block, "READINESS_JSON", results / "execution_readiness.json")
    monkeypatch.setattr(
        live_block,
        "LIVE_CONFIRM_PATHS",
        [live / "live_confirm.json", results / "live_confirm.json"],
    )
    monkeypatch.setattr(
        live_block, "evaluate_live_block_status", lambda **_k: dict(LIVE_BLOCKED_PAYLOAD)
    )

    monkeypatch.setattr(criteria, "ROOT", tmp_path)
    monkeypatch.setattr(criteria, "RESULTS", results)
    monkeypatch.setattr(criteria, "GOVERNANCE", gov)
    monkeypatch.setattr(
        criteria, "PREVIEW_JSON", results / "defensive_exit_capped_batch_preview.json"
    )
    monkeypatch.setattr(
        criteria,
        "OPERATOR_PACKET_JSON",
        results / "defensive_exit_operator_approval_packet.json",
    )
    monkeypatch.setattr(criteria, "READINESS_JSON", results / "execution_readiness.json")
    monkeypatch.setattr(criteria, "GOVERNANCE_AUTH_JSON", results / "governance_authorization.json")
    monkeypatch.setattr(criteria, "OPEN_ORDERS_JSON", results / "open_orders_snapshot.csv")
    monkeypatch.setattr(criteria, "POSITIONS_JSON", results / "positions_snapshot.csv")
    monkeypatch.setattr(criteria, "REAL_APPROVAL_PATH", hardened)

    monkeypatch.setattr(approval_mod, "ROOT", tmp_path)
    monkeypatch.setattr(approval_mod, "RESULTS", results)
    monkeypatch.setattr(approval_mod, "GOVERNANCE", gov)
    monkeypatch.setattr(approval_mod, "APPROVAL_CONTRACT_PATH", hardened)

    monkeypatch.setattr(approval_state, "ROOT", tmp_path)
    monkeypatch.setattr(approval_state, "GOVERNANCE", gov)
    monkeypatch.setattr(approval_state, "APPROVAL_PATH", hardened)

    return {"results": results, "live": live, "gov": gov, "hardened": hardened}


def write_valid_order_guard_fixture(results: Path, *, now: datetime = GUARD_NOW) -> Path:
    path = results / "guard_snapshot.json"
    path.write_text(
        json.dumps(
            {
                "mode": "NORMAL",
                "blocked": False,
                "kill_switch": False,
                "source": "phase_198i_hermetic_order_guard_fixture",
                "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def create_test_only_hardened_authority(
    tmp_path: Path,
    hardened_dest: Path,
    *,
    now: datetime,
) -> Dict[str, Any]:
    """Build authority under a non-live store, then copy onto the isolated hardened path."""
    from services.paper_sell_arm_authority import (
        HARDENED_AUTHORITY_FILENAME,
        OPERATOR_ACKNOWLEDGEMENT_TEXT,
        confirm_and_create_paper_sell_arm,
        prepare_confirmation_challenge,
    )

    store = tmp_path / "test_only_sell_arm_store"
    store.mkdir(parents=True, exist_ok=True)
    db = store / "paper_sell_arm.sqlite3"
    readiness = sample_sell_readiness_packet()
    chal = prepare_confirmation_challenge(
        store_root=store,
        readiness_packet=readiness,
        account_raw="acct-1",
        trading_date="2026-08-11",
        now=now,
        db_path=db,
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=store,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
        now=now,
        db_path=db,
    )
    src = store / HARDENED_AUTHORITY_FILENAME
    hardened_dest.parent.mkdir(parents=True, exist_ok=True)
    hardened_dest.write_bytes(src.read_bytes())
    return auth


def create_sqlite_sell_arm_via_public_apis(
    store_root: Path,
    *,
    now: datetime,
    explicit_runtime_write: bool = False,
) -> Dict[str, Any]:
    """Create UNUSED SQLite SELL authority through production challenge/confirm APIs.

    Does not begin dispatch, create a permit, insert SQL rows, or call a broker.
    JSON beside the database remains AUDIT_EXPORT_ONLY.
    """
    from services.paper_sell_arm_authority import (
        OPERATOR_ACKNOWLEDGEMENT_TEXT,
        confirm_and_create_paper_sell_arm,
        prepare_confirmation_challenge,
    )
    from services.paper_sell_arm_store import inspect_sqlite_authority, resolve_runtime_sqlite_path

    store_root = Path(store_root)
    store_root.mkdir(parents=True, exist_ok=True)
    db = resolve_runtime_sqlite_path()
    readiness = sample_sell_readiness_packet()
    chal = prepare_confirmation_challenge(
        store_root=store_root,
        readiness_packet=readiness,
        account_raw="acct-1",
        trading_date=now.date().isoformat(),
        now=now,
        explicit_runtime_write=explicit_runtime_write,
        db_path=db,
    )
    auth = confirm_and_create_paper_sell_arm(
        store_root=store_root,
        readiness_packet=readiness,
        challenge=chal,
        operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
        current_plan_hash=readiness["plan_hash"],
        now=now,
        explicit_runtime_write=explicit_runtime_write,
        db_path=db,
    )
    inspected = inspect_sqlite_authority(db, now=now)
    if str(inspected.get("state") or "") != "UNUSED":
        raise AssertionError(f"sqlite authority not UNUSED after confirm: {inspected}")
    return {
        "readiness": readiness,
        "challenge": chal,
        "authority": auth,
        "inspect": inspected,
    }


def install_hermetic_interlock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, inject_test_durability: bool = True
) -> Dict[str, Path]:
    block_network_and_live_env(monkeypatch)
    ban_repo_live_io(monkeypatch)
    if inject_test_durability:
        stub_dir_fsync_ok(monkeypatch)
    return isolate_interlock_artifact_roots(monkeypatch, tmp_path)
