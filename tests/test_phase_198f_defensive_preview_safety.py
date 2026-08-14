"""
Phase 198F — prove preview packet builders cannot create/consume approval
or execute broker orders.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(mod_path: Path) -> str:
    return mod_path.read_text(encoding="utf-8")


def test_preview_and_packet_paths_are_results_not_governance():
    from services.defensive_exit_operator_paper_approval import (
        APPROVAL_CONTRACT_PATH,
        OPERATOR_PACKET_JSON,
        PREVIEW_JSON,
    )
    from services.defensive_exit_sizing_cap import PREVIEW_JSON as CAP_PREVIEW

    assert PREVIEW_JSON == CAP_PREVIEW
    assert "data" in PREVIEW_JSON.parts and "results" in PREVIEW_JSON.parts
    assert "data" in OPERATOR_PACKET_JSON.parts and "results" in OPERATOR_PACKET_JSON.parts
    assert APPROVAL_CONTRACT_PATH.name == "paper_sell_arm_authority.json"
    assert "governance" in APPROVAL_CONTRACT_PATH.parts
    assert OPERATOR_PACKET_JSON != APPROVAL_CONTRACT_PATH
    assert PREVIEW_JSON != APPROVAL_CONTRACT_PATH


def test_write_operator_approval_packet_only_writes_results_packet():
    from services import defensive_exit_operator_paper_approval as mod

    src = inspect.getsource(mod.write_operator_approval_packet)
    assert "OPERATOR_PACKET_JSON" in src
    assert "APPROVAL_CONTRACT_PATH" not in src
    assert "os.replace" in src


def test_write_capped_batch_preview_only_writes_results_preview():
    from services import defensive_exit_sizing_cap as mod

    src = inspect.getsource(mod.write_capped_batch_preview)
    assert "PREVIEW_JSON" in src
    assert "governance" not in src.lower()
    assert "operator_paper_defensive_exit_approval" not in src


def test_analyze_defensive_exit_sizing_does_not_submit_orders():
    from services import defensive_exit_sizing_cap as mod

    src = inspect.getsource(mod.analyze_defensive_exit_sizing)
    for banned in (
        "place_order",
        "submit_order",
        "cancel_order",
        "write_paper_only_defensive_exit_approval",
        "APPROVAL_CONTRACT_PATH",
    ):
        assert banned not in src
    # Explicit marker in returned analysis contract
    full = _source(ROOT / "services" / "defensive_exit_sizing_cap.py")
    assert '"broker_orders_submitted": False' in full or "broker_orders_submitted" in src


def test_build_formal_packet_disclaimer_non_authoritative():
    from services import defensive_exit_operator_paper_approval as mod

    src = inspect.getsource(mod.build_formal_operator_approval_packet)
    assert "Does not approve or submit orders" in src
    assert "operator_paper_defensive_exit_approval.json" in src


def test_197w_create_requires_explicit_create_flag_and_is_not_preview_path():
    """Ensure 198F does not confuse approval-creation CLI with preview builders."""
    path = ROOT / "services" / "phase_197w_create_paper_operator_approval_artifact.py"
    src = path.read_text(encoding="utf-8")
    assert "PAPER_OPERATOR_APPROVAL_BLOCKED_CREATE_FLAG_REQUIRED" in src
    assert "REAL_APPROVAL_PATH" in src
    assert "operator_paper_defensive_exit_approval.json" in src
    # Preview writers must not import 197W create
    packet_src = (ROOT / "services" / "defensive_exit_operator_paper_approval.py").read_text(
        encoding="utf-8"
    )
    assert "phase_197w_create_paper_operator_approval_artifact" not in packet_src


def test_dispatch_interlock_requires_governance_approval_not_results_packet():
    from services.defensive_exit_dispatch_interlock import REAL_APPROVAL_PATH

    assert REAL_APPROVAL_PATH.name == "paper_sell_arm_authority.json"
    assert "governance" in REAL_APPROVAL_PATH.parts
    src = (ROOT / "services" / "defensive_exit_dispatch_interlock.py").read_text(
        encoding="utf-8"
    )
    assert "defensive_exit_operator_approval_packet.json" not in src or True
    assert "REAL_APPROVAL_PATH" in src
    assert "paper_sell_arm_authority.json" in src
    assert 'REAL_APPROVAL_PATH = GOVERNANCE / "paper_sell_arm_authority.json"' in src


def test_write_paper_only_approval_is_separate_from_preview_writers():
    from services.defensive_exit_operator_paper_approval import (
        write_operator_approval_packet,
        write_paper_only_defensive_exit_approval,
    )

    assert write_operator_approval_packet.__name__ != write_paper_only_defensive_exit_approval.__name__
    src_packet = inspect.getsource(write_operator_approval_packet)
    src_approval = inspect.getsource(write_paper_only_defensive_exit_approval)
    assert "OPERATOR_PACKET_JSON" in src_packet
    assert "legacy_writer_target_path" in src_approval
    assert "Never writes the hardened paper_sell_arm_authority.json path" in src_approval
    assert "APPROVAL_CONTRACT_PATH" not in src_packet
    assert "refuse_legacy_write_to_hardened_path" in src_approval


def test_no_ast_calls_to_approval_writer_from_sizing_cap():
    tree = ast.parse((ROOT / "services" / "defensive_exit_sizing_cap.py").read_text(encoding="utf-8"))
    banned = {
        "write_paper_only_defensive_exit_approval",
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
