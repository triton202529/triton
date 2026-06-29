"""
phase_197j_operator_paper_approval_packet_validation.py
Phase 197J — Validate formal operator paper approval packet and contract (read-only).
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
OUTPUT_JSON = RESULTS / "phase_197j_operator_paper_approval_packet_validation.json"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
APPROVAL_CONTRACT_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "4ff9fd7f"
PREVIOUS_PHASE = "197I"


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


def run_phase_197j_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    git_clean_before = _git_clean()

    from services.defensive_exit_governance_sync import build_defensive_exit_approval_sync
    from services.defensive_exit_operator_paper_approval import (
        APPROVAL_CONTRACT_PATH,
        APPROVAL_STATUS_REQUIRED,
        APPROVAL_STATUS_VALID,
        build_formal_operator_approval_packet,
        validate_operator_approval_contract,
        write_operator_approval_packet,
    )
    from services.paper_entry_loop_stage import evaluate_paper_entry_gates
    from services.portfolio_history_sync import load_live_execution_flag

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
            snap_doc = json.loads(snap.stdout[jstart:])
            snapshot_ok = bool(snap_doc.get("ok"))
    except Exception:
        pass

    sync = build_defensive_exit_approval_sync(cap_context="runtime")
    packet = build_formal_operator_approval_packet(
        sync=sync,
        broker_snapshot_ok=snapshot_ok,
        broker_orders_submitted=False,
    )
    write_operator_approval_packet(packet)

    approval_status, approval_reason, approval_details = validate_operator_approval_contract(
        packet_id=str(packet.get("packet_id") or ""),
        preview_hash=str(packet.get("source_preview_hash") or ""),
        capped_symbols=list(packet.get("capped_symbols") or []),
        configured_cap=float(packet.get("configured_cap") or 0.0),
        broker_snapshot_ok=snapshot_ok,
    )

    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=False)
    checks = gate_details.get("checks") or {}
    live_permitted = bool(checks.get("live_execution_permitted", True))

    contract_exists = APPROVAL_CONTRACT_PATH.is_file()
    packet_created = OPERATOR_PACKET_JSON.is_file()

    required_packet_fields = {
        "packet_id",
        "generated_at",
        "expires_at",
        "paper_only",
        "live_blocked",
        "approval_required",
        "approval_status",
        "capped_symbols",
        "deferred_symbols",
        "operator_decision_options",
        "risk_reduction_intent",
    }

    status = "PASS"
    if not packet_created:
        status = "FAIL"
        notes.append("operator approval packet not created")
    missing_fields = sorted(required_packet_fields - set(packet.keys()))
    if missing_fields:
        status = "FAIL"
        notes.append(f"packet missing fields: {','.join(missing_fields)}")
    if contract_exists:
        status = "FAIL"
        notes.append("approval contract must not be auto-created")
    if approval_status != APPROVAL_STATUS_REQUIRED:
        status = "FAIL"
        notes.append(f"expected approval_status={APPROVAL_STATUS_REQUIRED} got {approval_status}")
    if not packet.get("paper_only"):
        status = "FAIL"
        notes.append("packet paper_only must be true")
    if not packet.get("live_blocked"):
        status = "FAIL"
        notes.append("packet live_blocked must be true")
    if live_permitted or load_live_execution_flag():
        status = "FAIL"
        notes.append("live execution must remain blocked")
    if gate_ok:
        status = "FAIL"
        notes.append("paper entry gate unexpectedly passed")
    if sync.get("eligible_for_paper_execution"):
        status = "FAIL"
        notes.append("eligible_for_paper_execution must remain false")
    if _count_open_orders() != 0:
        notes.append(f"open_orders_count={_count_open_orders()} (expected 0)")

    payload: Dict[str, Any] = {
        "phase": "197J",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "197I synced capped preview with governance state but lacked a formal operator "
            "paper approval packet and explicit decision contract for capped defensive exits."
        ),
        "fix": (
            "defensive_exit_operator_paper_approval defines read-only packet, decision options, "
            "and validator for operator_paper_defensive_exit_approval.json without auto-approval."
        ),
        "packet_created": packet_created,
        "packet_id": packet.get("packet_id"),
        "packet_expires_at": packet.get("expires_at"),
        "approval_contract_path": str(APPROVAL_CONTRACT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "approval_exists": contract_exists,
        "approval_status": approval_status,
        "approval_validation_status": approval_status,
        "approval_validation_reason": approval_reason,
        "capped_symbols": packet.get("capped_symbols") or [],
        "deferred_symbols": packet.get("deferred_symbols") or [],
        "capped_notional": packet.get("capped_notional"),
        "configured_cap": packet.get("configured_cap"),
        "readiness_status": sync.get("readiness_status"),
        "governance_authorized": sync.get("governance_authorized"),
        "governance_sync_status": sync.get("approval_sync_status"),
        "eligible_for_paper_execution": bool(sync.get("eligible_for_paper_execution")),
        "broker_snapshot_ok": snapshot_ok,
        "open_orders_count": _count_open_orders(),
        "positions_count": _count_positions(),
        "live_blocked": not live_permitted,
        "paper_entry_gate_ok": gate_ok,
        "paper_entry_gate_reason": gate_reason,
        "broker_orders_submitted": False,
        "operator_packet": packet,
        "approval_validation_details": approval_details,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197J-OP-PACKET] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197J-OP-PACKET] status={status} approval_status={approval_status} "
            f"packet_id={packet.get('packet_id')}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197J operator paper approval packet validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197j_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
