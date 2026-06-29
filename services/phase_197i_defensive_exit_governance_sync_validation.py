"""
phase_197i_defensive_exit_governance_sync_validation.py
Phase 197I — Validate capped defensive exit governance/readiness approval sync.
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
LIVE = ROOT / "data" / "live"
SCRIPTS = ROOT / "scripts"
OUTPUT_JSON = RESULTS / "phase_197i_defensive_exit_governance_sync_validation.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
HANDOFF_CSV = LIVE / "manage_orders.csv"
PREVIOUS_COMMIT = "fe668944"
PREVIOUS_PHASE = "197H"


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


def _handoff_exit_rows() -> int:
    if not HANDOFF_CSV.is_file():
        return 0
    try:
        import pandas as pd

        df = pd.read_csv(HANDOFF_CSV, low_memory=False)
        if df.empty:
            return 0
        opp = df.get("opportunity_type", df.get("effective_stance", ""))
        return int((opp.astype(str).str.upper() == "EXIT").sum())
    except Exception:
        return 0


def run_phase_197i_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    git_clean_before = _git_clean()

    from services.defensive_exit_governance_sync import (
        build_defensive_exit_approval_sync,
        build_operator_approval_packet,
        write_operator_approval_packet,
    )
    from services.live_block_classifier import (
        apply_defensive_exit_validator_gate_checks,
        defensive_exit_phase_validator_gates,
    )

    sync = build_defensive_exit_approval_sync(cap_context="runtime")
    packet = build_operator_approval_packet(sync)
    write_operator_approval_packet(packet)

    placement_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "services.place_live_orders",
            "--mode",
            "paper",
            "--orders",
            str(HANDOFF_CSV),
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
    placement_maps_exit_rows = (
        "| EXIT |" in placement_out and "close_full_position" in placement_out
    )
    placement_batch_empty = "BATCH_EMPTY" in placement_out and not placement_maps_exit_rows

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

    gates = defensive_exit_phase_validator_gates(verbose=False)
    gate_ok = gates["gate_ok"]
    gate_reason = gates["gate_reason"]
    gate_details = gates["gate_details"]

    approval_status = str(sync.get("approval_sync_status") or "")
    explicit_statuses = {
        "CAPPED_DEFENSIVE_BATCH_AVAILABLE",
        "CAPPED_DEFENSIVE_BATCH_APPROVED",
        "CAPPED_DEFENSIVE_BATCH_APPROVED_FOR_PAPER",
        "CAPPED_DEFENSIVE_BATCH_REJECTED_BY_OPERATOR",
        "CAPPED_DEFENSIVE_BATCH_AWAITING_OPERATOR_APPROVAL",
        "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_READINESS",
        "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_GOVERNANCE",
        "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_NOTIONAL_CAP",
        "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_IN_FLIGHT_ORDERS",
    }

    status = "PASS"
    if not PREVIEW_JSON.is_file():
        status = "FAIL"
        notes.append("defensive_exit_capped_batch_preview.json missing")
    if _handoff_exit_rows() == 0:
        status = "FAIL"
        notes.append("197G regression: no handoff EXIT rows")
    if placement_batch_empty:
        status = "FAIL"
        notes.append("197G regression: BATCH_EMPTY with handoff rows")
    if not placement_maps_exit_rows:
        status = "FAIL"
        notes.append("197G regression: placement does not map EXIT rows")
    if approval_status not in explicit_statuses:
        status = "FAIL"
        notes.append(f"unknown approval_sync_status {approval_status}")
    if not sync.get("approval_block_reason") and approval_status not in (
        "CAPPED_DEFENSIVE_BATCH_APPROVED",
        "CAPPED_DEFENSIVE_BATCH_AVAILABLE",
    ):
        notes.append("approval_block_reason empty for blocked state")
    if sync.get("eligible_for_paper_execution"):
        notes.append("unexpected eligible_for_paper_execution=true during validation")
    status = apply_defensive_exit_validator_gate_checks(
        status=status, notes=notes, gates=gates, fail_on_gate_open=True
    )

    payload: Dict[str, Any] = {
        "phase": "197I",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "197H capped defensive exit preview existed but was not connected to "
            "governance_authorization, execution_readiness, or human_approval_queue — "
            "runtime could not state whether a capped batch was approved or blocked."
        ),
        "fix": (
            "defensive_exit_governance_sync classifies capped batch against governance, "
            "readiness, operator queue, in-flight orders, and notional cap without executing."
        ),
        "broker_snapshot_ok": snapshot_ok,
        "live_blocked": gates["live_blocked"],
        "live_block_reason": gates["live_block_reason"],
        "paper_mode": True,
        "positions_count": _count_positions(),
        "open_orders_count": _count_open_orders(),
        "preview_exists": bool(sync.get("preview_exists")),
        "partial_batch_authorized": bool(sync.get("partial_batch_authorized")),
        "capped_preview_symbols": sync.get("capped_preview_symbols") or [],
        "deferred_symbols": sync.get("deferred_symbols") or [],
        "capped_preview_notional": sync.get("capped_preview_notional"),
        "configured_cap": sync.get("configured_cap"),
        "readiness_status": sync.get("readiness_status"),
        "governance_authorized": sync.get("governance_authorized"),
        "operator_authorized": sync.get("operator_authorized_global"),
        "operator_authorized_for_capped_batch": sync.get("operator_authorized_for_capped_batch"),
        "execution_authorized": sync.get("execution_authorized_policy"),
        "approval_sync_status": approval_status,
        "approval_block_reason": sync.get("approval_block_reason"),
        "approval_source": sync.get("approval_source"),
        "eligible_for_paper_execution": bool(sync.get("eligible_for_paper_execution")),
        "placement_maps_exit_rows": placement_maps_exit_rows,
        "paper_entry_gate_ok": gate_ok,
        "paper_entry_gate_reason": gate_reason,
        "paper_entry_gate_classification": gates["gate_classification"],
        "broker_orders_submitted": False,
        "operator_approval_packet_path": "data/results/defensive_exit_operator_approval_packet.json",
        "approval_sync": sync,
        "operator_packet": packet,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197I-GOV-SYNC] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197I-GOV-SYNC] status={status} approval_sync={approval_status} "
            f"reason={sync.get('approval_block_reason')}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197I defensive exit governance sync validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197i_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
