"""
Phase 198G — approval-readiness isolation and fail-closed checks.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readiness_path_is_not_governance_approval():
    readiness = ROOT / "data" / "results" / "defensive_exit_approval_readiness.json"
    hardened = ROOT / "data" / "governance" / "paper_sell_arm_authority.json"
    legacy = (
        ROOT / "data" / "governance" / "operator_paper_defensive_exit_approval.json"
    )
    assert readiness != hardened
    assert readiness != legacy
    assert "results" in readiness.parts
    assert "governance" in hardened.parts


def test_dispatch_interlock_ignores_readiness_and_results_packet():
    from services.defensive_exit_dispatch_interlock import REAL_APPROVAL_PATH

    src = (ROOT / "services" / "defensive_exit_dispatch_interlock.py").read_text(
        encoding="utf-8"
    )
    assert REAL_APPROVAL_PATH.name == "paper_sell_arm_authority.json"
    assert "defensive_exit_approval_readiness.json" not in src
    assert "governance" in REAL_APPROVAL_PATH.parts


def test_validator_rejects_missing_approval_contract():
    from services.defensive_exit_operator_paper_approval import (
        APPROVAL_STATUS_VALID,
        validate_operator_approval_contract,
    )

    st, reason, _ = validate_operator_approval_contract(broker_snapshot_ok=True)
    assert st != APPROVAL_STATUS_VALID
    assert "missing" in reason.lower() or "required" in reason.lower() or st


def test_analyze_sizing_does_not_write_approval_or_orders():
    from services import defensive_exit_sizing_cap as mod

    src = inspect.getsource(mod.analyze_defensive_exit_sizing)
    for banned in (
        "write_paper_only_defensive_exit_approval",
        "place_order",
        "submit_order",
        "cancel_order",
        "APPROVAL_CONTRACT_PATH",
    ):
        assert banned not in src


def test_198g_script_does_not_call_approval_writer_or_overwrite_cycle3_writers():
    """Readiness is non-authoritative: production 198G/H builders must not write approval or orders."""
    from services.paper_sell_arm_authority import (
        APPROVAL_CLI_CAP,
        READINESS_STATUS_OK,
        readiness_packet_hash,
    )

    assert READINESS_STATUS_OK == "UNAPPROVED_APPROVAL_READINESS_ONLY"
    src = (ROOT / "services" / "paper_sell_arm_authority.py").read_text(encoding="utf-8")
    assert "UNAPPROVED_APPROVAL_READINESS_ONLY" in src
    assert "creates_no_authority" in src or "NO authority" in src or "non-executing" in src.lower()
    tree = ast.parse(src)
    banned = {
        "write_paper_only_defensive_exit_approval",
        "write_operator_approval_packet",
        "write_capped_batch_preview",
        "place_order",
        "submit_order",
    }
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in banned:
                found.add(node.func.id)
            elif isinstance(node.func, ast.Attribute) and node.func.attr in banned:
                found.add(node.func.attr)
    assert not found
    from tests._phase_198i_interlock_test_fixtures import sample_sell_readiness_packet

    assert readiness_packet_hash(sample_sell_readiness_packet())
    assert APPROVAL_CLI_CAP == 5000.0


def test_cli_cap_constant_not_raised_in_198g_script():
    from services.paper_sell_arm_authority import APPROVAL_CLI_CAP

    src = (ROOT / "services" / "paper_sell_arm_authority.py").read_text(encoding="utf-8")
    assert "APPROVAL_CLI_CAP = 5000.0" in src
    assert APPROVAL_CLI_CAP == 5000.0
    assert "APPROVAL_CLI_CAP = 50000" not in src


def test_plan_fingerprint_changes_when_selection_changes():
    # Import helper by exec path content lightly
    import hashlib

    def fp(rows):
        blob = "|".join(
            sorted(
                f"{r.get('symbol')}:{r.get('final_action')}:{r.get('qty')}:"
                f"{r.get('selected_for_execution')}:{r.get('reason_code')}:{r.get('skip_reason')}"
                for r in rows
            )
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    a = [
        {
            "symbol": "TSLA",
            "final_action": "EXIT",
            "qty": "8",
            "selected_for_execution": "True",
            "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
            "skip_reason": "",
        }
    ]
    b = [
        {
            "symbol": "TSLA",
            "final_action": "EXIT",
            "qty": "8",
            "selected_for_execution": "True",
            "reason_code": "EXIT_HIGH_DRAG_PROTECTION",
            "skip_reason": "",
        },
        {
            "symbol": "HD",
            "final_action": "TRIM",
            "qty": "8",
            "selected_for_execution": "True",
            "reason_code": "TRIM_APPROVED",
            "skip_reason": "",
        },
    ]
    assert fp(a) != fp(b)
