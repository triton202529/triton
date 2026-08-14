"""
Phase 198E — read-only contracts for position-management approval gate diagnostics.

Does not create approvals, renew paper arm, run manage_positions, or submit orders.
"""

from __future__ import annotations

from pathlib import Path

from tests._phase_198i_interlock_test_fixtures import (
    install_hermetic_interlock,
    write_valid_order_guard_fixture,
)


def test_dispatch_interlock_requires_governance_defensive_exit_approval_path():
    src = Path("services/defensive_exit_dispatch_interlock.py").read_text(encoding="utf-8")
    assert 'REAL_APPROVAL_PATH = GOVERNANCE / "paper_sell_arm_authority.json"' in src
    assert "DISPATCH_BLOCKED_APPROVAL_REQUIRED" in src
    assert "DISPATCH_BLOCKED_ORDER_GUARD" in src
    assert "real_approval_artifact_absent" in src
    assert "DEFENSIVE_SELL_ACTIONS" in src
    assert '"TRIM"' in src and '"EXIT"' in src and '"FORCE_EXIT"' in src
    assert "legacy 197W" in src.lower() or "Legacy 197W" in src


def test_manage_positions_calls_dispatch_interlock_for_paper_defensive_handoff():
    src = Path("services/manage_positions.py").read_text(encoding="utf-8")
    assert "is_defensive_exit_manage_handoff" in src
    assert "evaluate_defensive_exit_dispatch_interlock" in src
    assert "write_handoff_audit" in src


def test_paper_arm_path_is_distinct_from_defensive_exit_approval_path():
    arm = Path("services/paper_entry_loop_stage.py").read_text(encoding="utf-8")
    interlock = Path("services/defensive_exit_dispatch_interlock.py").read_text(encoding="utf-8")
    assert "paper_arm" in arm.lower() or "classify_paper_arm" in arm
    assert "paper_arm_confirm" not in interlock
    assert "paper_sell_arm_authority.json" in interlock
    assert "REAL_APPROVAL_PATH = GOVERNANCE / \"paper_sell_arm_authority.json\"" in interlock


def test_operator_approval_packet_documents_explicit_artifact_requirement():
    src = Path("services/defensive_exit_operator_paper_approval.py").read_text(encoding="utf-8")
    assert "operator_paper_defensive_exit_approval.json" in src
    assert "approval_required" in src
    assert "Does not approve or submit orders" in src


def test_live_interlock_blocks_when_order_guard_missing(tmp_path, monkeypatch):
    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_ORDER_GUARD,
        evaluate_defensive_exit_dispatch_interlock,
    )

    roots = install_hermetic_interlock(monkeypatch, tmp_path)
    assert not (roots["results"] / "guard_snapshot.json").is_file()
    result = evaluate_defensive_exit_dispatch_interlock(
        mode="paper",
        broker_snapshot_ok=True,
        require_market_open=False,
        planned_count=5,
    )
    assert result.get("dispatch_interlock_state") == DISPATCH_BLOCKED_ORDER_GUARD
    assert result.get("dispatch_permitted") is False
    assert result.get("broker_orders_submitted") is False
    assert result.get("eligible_for_execution") is False


def test_live_interlock_evaluation_is_fail_closed_when_artifact_absent(tmp_path, monkeypatch):
    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_APPROVAL_REQUIRED,
        evaluate_defensive_exit_dispatch_interlock,
    )

    roots = install_hermetic_interlock(monkeypatch, tmp_path)
    write_valid_order_guard_fixture(roots["results"])
    assert not roots["hardened"].is_file()
    result = evaluate_defensive_exit_dispatch_interlock(
        mode="paper",
        broker_snapshot_ok=True,
        require_market_open=False,
        planned_count=5,
    )
    assert result.get("dispatch_interlock_state") == DISPATCH_BLOCKED_APPROVAL_REQUIRED
    assert result.get("dispatch_permitted") is False
    assert result.get("real_approval_artifact_exists") is False
    assert result.get("broker_orders_submitted") is False
    assert result.get("hardened_sell_arm_ok") is False
    reason = str(result.get("dispatch_interlock_reason") or "")
    assert "real_approval_artifact_absent" in reason or "approval_contract_missing" in reason


def test_live_interlock_with_guard_and_hardened_authority_reaches_later_gate(
    tmp_path, monkeypatch
):
    from datetime import datetime, timezone

    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_ORDER_GUARD,
        evaluate_defensive_exit_dispatch_interlock,
    )
    import services.paper_sell_arm_authority as auth_mod
    from tests._phase_198i_interlock_test_fixtures import create_test_only_hardened_authority

    frozen = datetime(2026, 8, 11, 15, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(auth_mod, "_utc_now", lambda: frozen)
    roots = install_hermetic_interlock(monkeypatch, tmp_path)
    write_valid_order_guard_fixture(roots["results"])
    create_test_only_hardened_authority(tmp_path, roots["hardened"], now=frozen)
    assert roots["hardened"].is_file()
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
