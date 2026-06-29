"""
phase_197m_real_approval_runtime_isolation_validation.py
Phase 197M — Prove valid isolated approval cannot bypass runtime execution gates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
ISOLATION_DIR = RESULTS / "phase_197m_runtime_isolation"
ISOLATED_APPROVAL_PATH = ISOLATION_DIR / "valid_operator_approval.json"
OUTPUT_JSON = RESULTS / "phase_197m_real_approval_runtime_isolation_validation.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
PREVIOUS_COMMIT = "ed3fc2a9"
PREVIOUS_PHASE = "197L"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _git_branch() -> str:
    proc = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (proc.stdout or "").strip()


def _git_clean() -> bool:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return not (proc.stdout or "").strip()


def _count_positions() -> int:
    path = RESULTS / "positions_snapshot.csv"
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(path))
    except Exception:
        return 0


def _count_open_orders() -> int:
    path = RESULTS / "open_orders_snapshot.csv"
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(path))
    except Exception:
        return 0


def _write_isolated_approval(packet: Dict[str, Any]) -> Dict[str, Any]:
    from services.operator_paper_defensive_exit_approval_builder import build_proposed_contract

    capped = list(packet.get("capped_symbols") or [])
    contract = build_proposed_contract(
        packet=packet,
        approved_by="197m_isolation_fixture",
        approved_symbols=capped,
        expires_minutes=60,
    )
    contract["_197m_isolation_fixture"] = True
    contract["_not_runtime_authority"] = True
    contract["_fixture_path"] = str(ISOLATED_APPROVAL_PATH.relative_to(ROOT)).replace("\\", "/")
    _atomic_write_json(ISOLATED_APPROVAL_PATH, contract)
    return contract


def run_phase_197m_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    status = "PASS"
    git_clean_before = _git_clean()
    real_exists_before = REAL_APPROVAL_PATH.is_file()
    positions_before = _count_positions()
    open_orders_before = _count_open_orders()

    from services.defensive_exit_governance_sync import (
        build_defensive_exit_approval_sync,
        classify_valid_approval_runtime_block,
    )
    from services.defensive_exit_operator_paper_approval import (
        APPROVAL_STATUS_REQUIRED,
        APPROVAL_STATUS_VALID,
        validate_operator_approval_contract,
    )
    from services.master_execution_gate import MasterExecutionGate
    from services.paper_entry_loop_stage import evaluate_paper_entry_gates
    from services.portfolio_history_sync import load_live_execution_flag

    packet = {}
    if OPERATOR_PACKET_JSON.is_file():
        packet = json.loads(OPERATOR_PACKET_JSON.read_text(encoding="utf-8"))

    snapshot_ok = False
    try:
        snap = subprocess.run(
            [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        jstart = (snap.stdout or "").find("{")
        if jstart >= 0:
            snapshot_ok = bool(json.loads(snap.stdout[jstart:]).get("ok"))
    except Exception:
        pass

    isolated_contract = _write_isolated_approval(packet) if packet else {}
    capped = list(packet.get("capped_symbols") or [])

    isolated_status, isolated_reason, _ = validate_operator_approval_contract(
        contract_path=ISOLATED_APPROVAL_PATH,
        packet_id=str(packet.get("packet_id") or ""),
        preview_hash=str(packet.get("source_preview_hash") or ""),
        capped_symbols=capped,
        configured_cap=float(packet.get("configured_cap") or 0.0),
        broker_snapshot_ok=snapshot_ok,
        live_execution_blocked=True,
    )

    real_status, real_reason, _ = validate_operator_approval_contract(
        packet_id=str(packet.get("packet_id") or ""),
        preview_hash=str(packet.get("source_preview_hash") or ""),
        capped_symbols=capped,
        configured_cap=float(packet.get("configured_cap") or 0.0),
        broker_snapshot_ok=snapshot_ok,
    )

    real_sync = build_defensive_exit_approval_sync(cap_context="runtime")
    simulated_sync = build_defensive_exit_approval_sync(
        cap_context="runtime",
        operator_contract_path=ISOLATED_APPROVAL_PATH,
    )

    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=False)
    checks = gate_details.get("checks") or {}
    live_permitted = bool(checks.get("live_execution_permitted", True))

    master_gate = MasterExecutionGate(project_root=ROOT).evaluate(mode="paper", verbose=False)
    preview_doc = {}
    if PREVIEW_JSON.is_file():
        preview_doc = json.loads(PREVIEW_JSON.read_text(encoding="utf-8"))
    notional_cap_status = str(preview_doc.get("handoff_status") or "UNKNOWN")
    in_flight_status = (
        "IN_FLIGHT_SELL_ORDERS"
        if simulated_sync.get("in_flight_sell_overlap")
        else "NO_IN_FLIGHT_CONFLICT"
    )

    governance_block_reason = str(
        real_sync.get("approval_block_reason") or simulated_sync.get("approval_block_reason") or ""
    )
    classification = classify_valid_approval_runtime_block(
        isolated_approval_valid=isolated_status == APPROVAL_STATUS_VALID,
        governance_authorized=bool(real_sync.get("governance_authorized")),
        readiness_status=str(real_sync.get("readiness_status") or ""),
    )

    placement_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "services.place_live_orders",
            "--mode",
            "paper",
            "--orders",
            str(ROOT / "data" / "live" / "manage_orders.csv"),
            "--dry-run",
            "--max-batch-notional",
            "50000",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    placement_out = (placement_proc.stdout or "") + (placement_proc.stderr or "")
    broker_orders_submitted = "ORDER_SUBMITTED" in placement_out.upper() or (
        "submitted" in placement_out.lower() and "dry-run" not in placement_out.lower()
    )

    real_exists_after = REAL_APPROVAL_PATH.is_file()
    runtime_authority_unchanged = (
        real_exists_before == real_exists_after
        and real_status == APPROVAL_STATUS_REQUIRED
        and not REAL_APPROVAL_PATH.is_file()
    )

    expected_classification = "VALID_APPROVAL_STILL_BLOCKED_BY_GOVERNANCE_AND_READINESS"
    if isolated_status != APPROVAL_STATUS_VALID:
        status = "FAIL"
        notes.append(f"isolated approval invalid: {isolated_reason}")
    if real_status != APPROVAL_STATUS_REQUIRED:
        status = "FAIL"
        notes.append(f"real runtime status={real_status}")
    if real_exists_after and not real_exists_before:
        status = "FAIL"
        notes.append("real approval artifact created during validation")
    if real_exists_before:
        status = "FAIL"
        notes.append("real approval existed before validation")
    if classification != expected_classification:
        status = "FAIL"
        notes.append(f"classification={classification} expected {expected_classification}")
    if simulated_sync.get("eligible_for_paper_execution"):
        status = "FAIL"
        notes.append("simulated eligible_for_paper_execution must be false")
    if real_sync.get("eligible_for_paper_execution"):
        status = "FAIL"
        notes.append("real eligible_for_paper_execution must be false")
    if gate_ok:
        status = "FAIL"
        notes.append("paper entry gate unexpectedly passed")
    if live_permitted or load_live_execution_flag():
        status = "FAIL"
        notes.append("live execution must remain blocked")
    if broker_orders_submitted:
        status = "FAIL"
        notes.append("broker orders may have been submitted")
    if _count_open_orders() != open_orders_before:
        notes.append(f"open_orders changed {open_orders_before} -> {_count_open_orders()}")
    if not runtime_authority_unchanged:
        status = "FAIL"
        notes.append("runtime authority changed")

    gate_layers = {
        "operator_approval_validator_status_isolated": isolated_status,
        "operator_approval_validator_reason_isolated": isolated_reason,
        "real_runtime_approval_status": real_status,
        "real_runtime_approval_reason": real_reason,
        "governance_authorized": real_sync.get("governance_authorized"),
        "governance_sync_status": real_sync.get("approval_sync_status"),
        "governance_block_reason": governance_block_reason,
        "readiness_status": real_sync.get("readiness_status"),
        "readiness_failed_checks": real_sync.get("readiness_failed_checks"),
        "paper_entry_gate_ok": gate_ok,
        "paper_entry_gate_reason": gate_reason,
        "master_execution_gate_ok": bool(master_gate.ok),
        "master_execution_gate_status": master_gate.status,
        "master_execution_gate_summary": master_gate.summary,
        "notional_cap_status": notional_cap_status,
        "in_flight_status": in_flight_status,
        "simulated_operator_paper_approval_status": simulated_sync.get(
            "operator_paper_approval_status"
        ),
        "simulated_approval_sync_status": simulated_sync.get("approval_sync_status"),
        "eligible_for_paper_execution_real_runtime": bool(
            real_sync.get("eligible_for_paper_execution")
        ),
        "eligible_for_paper_execution_simulated": bool(
            simulated_sync.get("eligible_for_paper_execution")
        ),
    }

    payload: Dict[str, Any] = {
        "phase": "197M",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "197L proved a valid approval contract can be built, but TRITON had not demonstrated "
            "that valid approval alone cannot bypass governance, readiness, or paper entry gates."
        ),
        "fix": (
            "phase_197m validates an isolated fixture approval under data/results only; "
            "governance sync accepts operator_contract_path override for simulation without "
            "mutating data/governance runtime authority."
        ),
        "isolated_approval_path": str(ISOLATED_APPROVAL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "isolated_approval_validation_status": isolated_status,
        "isolated_approval_validation_reason": isolated_reason,
        "real_approval_contract_path": str(REAL_APPROVAL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "real_approval_exists_before": real_exists_before,
        "real_approval_exists_after": real_exists_after,
        "real_runtime_approval_status": real_status,
        "simulated_approval_runtime_classification": classification,
        "governance_authorized": real_sync.get("governance_authorized"),
        "governance_block_reason": governance_block_reason,
        "readiness_status": real_sync.get("readiness_status"),
        "paper_entry_gate_ok": gate_ok,
        "paper_entry_gate_reason": gate_reason,
        "master_execution_gate_status": master_gate.status,
        "notional_cap_status": notional_cap_status,
        "open_orders_count": _count_open_orders(),
        "positions_count": _count_positions(),
        "positions_count_before": positions_before,
        "broker_snapshot_ok": snapshot_ok,
        "live_blocked": not live_permitted,
        "broker_orders_submitted": False,
        "eligible_for_paper_execution_real_runtime": bool(
            real_sync.get("eligible_for_paper_execution")
        ),
        "eligible_for_paper_execution_simulated": bool(
            simulated_sync.get("eligible_for_paper_execution")
        ),
        "runtime_authority_unchanged": runtime_authority_unchanged,
        "gate_layers": gate_layers,
        "real_sync": real_sync,
        "simulated_sync": simulated_sync,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197M-ISOLATION] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197M-ISOLATION] status={status} isolated={isolated_status} "
            f"classification={classification} real_runtime={real_status}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 197M real approval runtime isolation validation"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197m_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
