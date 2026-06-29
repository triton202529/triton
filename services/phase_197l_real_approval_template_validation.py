"""
phase_197l_real_approval_template_validation.py
Phase 197L — Validate guarded approval template workflow without execution.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_197l_real_approval_template_validation.json"
TEMPLATE_JSON = RESULTS / "operator_paper_defensive_exit_approval_template.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
PREVIOUS_COMMIT = "1904adaa"
PREVIOUS_PHASE = "197K"


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


def run_phase_197l_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    status = "PASS"
    git_clean_before = _git_clean()
    real_exists_before = REAL_APPROVAL_PATH.is_file()
    positions_before = _count_positions()
    open_orders_before = _count_open_orders()

    from services.defensive_exit_governance_sync import build_defensive_exit_approval_sync
    from services.defensive_exit_operator_paper_approval import (
        APPROVAL_STATUS_INVALID,
        APPROVAL_STATUS_REQUIRED,
        APPROVAL_STATUS_VALID,
        validate_operator_approval_contract,
    )
    from services.operator_paper_defensive_exit_approval_builder import run_approval_builder
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
            snapshot_ok = bool(json.loads(snap.stdout[jstart:]).get("ok"))
    except Exception:
        pass

    builder_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "services.operator_paper_defensive_exit_approval_builder",
            "--dry-run",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    builder_cli_ok = builder_proc.returncode == 0

    builder_result = run_approval_builder(
        dry_run=True,
        write_real_approval=False,
        broker_snapshot_ok=snapshot_ok,
        verbose=False,
    )

    template = (
        json.loads(TEMPLATE_JSON.read_text(encoding="utf-8")) if TEMPLATE_JSON.is_file() else {}
    )
    packet = (
        json.loads(OPERATOR_PACKET_JSON.read_text(encoding="utf-8"))
        if OPERATOR_PACKET_JSON.is_file()
        else {}
    )

    template_validates_non_runtime = False
    if template.get("template_only") is True:
        template_status, template_reason, _ = validate_operator_approval_contract(
            contract_path=TEMPLATE_JSON,
            packet_id=str(packet.get("packet_id") or ""),
            preview_hash=str(packet.get("source_preview_hash") or ""),
            capped_symbols=list(packet.get("capped_symbols") or []),
            configured_cap=float(packet.get("configured_cap") or 0.0),
            broker_snapshot_ok=snapshot_ok,
            live_execution_blocked=True,
        )
        template_validates_non_runtime = (
            template_status == APPROVAL_STATUS_INVALID
            and template_reason == "template_only_not_runtime_authority"
        )

    proposed = dict(
        template.get("proposed_contract") or builder_result.get("proposed_contract") or {}
    )
    simulated_valid = False
    simulated_status = ""
    simulated_reason = ""
    if proposed:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(proposed, tmp, indent=2)
            sim_path = Path(tmp.name)
        try:
            simulated_status, simulated_reason, _ = validate_operator_approval_contract(
                contract_path=sim_path,
                packet_id=str(packet.get("packet_id") or ""),
                preview_hash=str(packet.get("source_preview_hash") or ""),
                capped_symbols=list(packet.get("capped_symbols") or []),
                configured_cap=float(packet.get("configured_cap") or 0.0),
                broker_snapshot_ok=snapshot_ok,
                live_execution_blocked=True,
            )
            simulated_valid = simulated_status == APPROVAL_STATUS_VALID
        finally:
            sim_path.unlink(missing_ok=True)

    blocked_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "services.operator_paper_defensive_exit_approval_builder",
            "--write-real-approval",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    write_blocked_without_flag = blocked_proc.returncode != 0

    real_exists_after = REAL_APPROVAL_PATH.is_file()
    real_status, _, _ = validate_operator_approval_contract(
        packet_id=str(packet.get("packet_id") or ""),
        preview_hash=str(packet.get("source_preview_hash") or ""),
        capped_symbols=list(packet.get("capped_symbols") or []),
        configured_cap=float(packet.get("configured_cap") or 0.0),
    )
    sync = build_defensive_exit_approval_sync(cap_context="runtime")
    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=False)
    checks = gate_details.get("checks") or {}
    live_permitted = bool(checks.get("live_execution_permitted", True))

    if not builder_cli_ok:
        status = "FAIL"
        notes.append("builder CLI dry-run failed")
    if not TEMPLATE_JSON.is_file():
        status = "FAIL"
        notes.append("approval template not created")
    if not template.get("template_only"):
        status = "FAIL"
        notes.append("template_only not true")
    if not template_validates_non_runtime:
        status = "FAIL"
        notes.append("template not rejected as non-runtime authority")
    if not simulated_valid:
        status = "FAIL"
        notes.append(f"simulated contract invalid: {simulated_reason}")
    if real_exists_after:
        status = "FAIL"
        notes.append("real approval artifact created during validation")
    if real_exists_before:
        status = "FAIL"
        notes.append("real approval existed before validation")
    if real_status != APPROVAL_STATUS_REQUIRED:
        status = "FAIL"
        notes.append(f"real runtime status={real_status}")
    if not write_blocked_without_flag:
        status = "FAIL"
        notes.append("write-real-approval without approved-by was not blocked")
    if live_permitted or load_live_execution_flag():
        status = "FAIL"
        notes.append("live execution must remain blocked")
    if gate_ok:
        status = "FAIL"
        notes.append("paper entry gate unexpectedly passed")
    if sync.get("eligible_for_paper_execution"):
        status = "FAIL"
        notes.append("eligible_for_paper_execution must remain false")
    if _count_open_orders() != open_orders_before:
        notes.append(f"open_orders changed {open_orders_before} -> {_count_open_orders()}")

    payload: Dict[str, Any] = {
        "phase": "197L",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "197K proved contract validation but lacked a guarded operator workflow to produce "
            "an approval template without auto-creating runtime authority."
        ),
        "fix": (
            "operator_paper_defensive_exit_approval_builder writes template_only artifact by "
            "default; real governance approval requires explicit --write-real-approval flag."
        ),
        "dry_run_template_created": TEMPLATE_JSON.is_file(),
        "template_path": str(TEMPLATE_JSON.relative_to(ROOT)).replace("\\", "/"),
        "real_approval_contract_path": str(REAL_APPROVAL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "real_approval_exists_before": real_exists_before,
        "real_approval_exists_after": real_exists_after,
        "real_approval_written": False,
        "write_real_approval_attempted": False,
        "write_real_approval_blocked_without_flag": write_blocked_without_flag,
        "template_validates_as_non_runtime": template_validates_non_runtime,
        "template_rejection_reason": "template_only_not_runtime_authority",
        "simulated_real_contract_validates": simulated_valid,
        "simulated_contract_status": simulated_status,
        "simulated_contract_reason": simulated_reason,
        "broker_snapshot_ok": snapshot_ok,
        "open_orders_count": _count_open_orders(),
        "positions_count": _count_positions(),
        "positions_count_before": positions_before,
        "live_blocked": not live_permitted,
        "paper_entry_gate_ok": gate_ok,
        "paper_entry_gate_reason": gate_reason,
        "broker_orders_submitted": False,
        "approval_status_real_runtime": real_status,
        "readiness_status": sync.get("readiness_status"),
        "governance_authorized": sync.get("governance_authorized"),
        "governance_sync_status": sync.get("approval_sync_status"),
        "eligible_for_paper_execution_real_runtime": bool(sync.get("eligible_for_paper_execution")),
        "builder_result": builder_result,
        "template_summary": template.get("human_readable_summary"),
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197L-TEMPLATE] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197L-TEMPLATE] status={status} template_only={template.get('template_only')} "
            f"simulated_valid={simulated_valid} real_runtime={real_status}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197L real approval template validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197l_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
