"""
phase_197g_defensive_exit_handoff_validation.py
Phase 197G — Validate defensive exit handoff from plan to manage_orders placement.
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
OUTPUT_JSON = RESULTS / "phase_197g_defensive_exit_handoff_validation.json"
PLAN_CSV = RESULTS / "manage_positions_plan.csv"
HANDOFF_CSV = LIVE / "manage_orders.csv"
READINESS_JSON = RESULTS / "execution_readiness.json"
PREVIOUS_COMMIT = "46e0a54a"
PREVIOUS_PHASE = "197F"


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

        df = pd.read_csv(path)
        return len(df)
    except Exception:
        return 0


def run_phase_197g_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    git_clean_before = _git_clean()

    from services.defensive_exit_preview_handoff import (
        ensure_defensive_exit_preview_handoff,
        resolve_defensive_handoff_orders_path,
    )
    from services.manage_positions_handoff_diagnostic import build_handoff_diagnostic
    from services.paper_entry_loop_stage import evaluate_paper_entry_gates
    from services.portfolio_history_sync import load_live_execution_flag

    preview_meta = ensure_defensive_exit_preview_handoff(mode="paper")
    handoff_path, handoff_source = resolve_defensive_handoff_orders_path()

    placement_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "services.place_live_orders",
            "--mode",
            "paper",
            "--orders",
            str(handoff_path),
            "--dry-run",
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
    placement_contract_mismatch = "HANDOFF_CONTRACT_MISMATCH" in placement_out

    diag = build_handoff_diagnostic(placement_output=placement_out, mode="paper")

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

    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=False)
    checks = gate_details.get("checks") or {}

    readiness = {}
    try:
        readiness = json.loads(READINESS_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    readiness_status = str(
        readiness.get("readiness_status") or diag.get("readiness_status") or "UNKNOWN"
    )
    live_permitted = bool(readiness.get("live_execution_permitted", True))

    planned_count = int(diag.get("planned_exit_count") or 0)
    handoff_rows = int(diag.get("manage_orders_handoff_rows") or 0)
    handoff_status = str(diag.get("handoff_status") or "")
    batch_explained = bool(diag.get("batch_empty_explained"))

    status = "PASS"
    outcome_a = handoff_rows > 0 and placement_maps_exit_rows and not placement_contract_mismatch
    outcome_b = batch_explained and (
        handoff_status.startswith("HANDOFF_BLOCKED")
        or handoff_status == "HANDOFF_PREVIEW_ROWS_PRESENT"
    )
    preview_ok = (
        handoff_status == "HANDOFF_PREVIEW_ROWS_PRESENT"
        and bool(diag.get("preview_generation_allowed"))
        and not bool(diag.get("execution_allowed"))
    )
    if preview_ok and handoff_rows > 0:
        outcome_a = True
    if planned_count == 0:
        notes.append("no planned defensive exits in plan csv (informational)")
    if not outcome_a and not outcome_b:
        if planned_count > 0 and handoff_rows > 0 and placement_batch_empty:
            status = "FAIL"
            notes.append("handoff rows exist but placement still reports BATCH_EMPTY")
        elif planned_count > 0 and handoff_rows > 0 and placement_contract_mismatch:
            status = "FAIL"
            notes.append("HANDOFF_CONTRACT_MISMATCH still present")
        elif planned_count > 0 and handoff_rows > 0 and not placement_maps_exit_rows:
            status = "FAIL"
            notes.append("handoff rows exist but placement dry-run maps zero EXIT rows")
        elif planned_count > 0 and handoff_rows == 0 and not outcome_b:
            status = "FAIL"
            notes.append("planned exits without handoff rows or explicit block")
    if live_permitted or load_live_execution_flag():
        status = "FAIL"
        notes.append("live execution must remain blocked")

    payload: Dict[str, Any] = {
        "phase": "197G",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "place_live_orders._placement_stance_from_row ignored effective_stance=EXIT and "
            "opportunity_type=EXIT in manage_positions handoff CSV, mapping zero lifecycle rows "
            "and reporting BATCH_EMPTY while manage_positions_plan.csv had planned defensive exits."
        ),
        "fix": (
            "Map actionable stances from effective_stance/opportunity_type; classify manage_handoff "
            "source; emit manage_positions_handoff_audit.json with explicit block reasons."
        ),
        "broker_snapshot_ok": snapshot_ok,
        "live_blocked": not live_permitted,
        "paper_mode": True,
        "positions_count": _count_positions(),
        "open_orders_count": _count_open_orders(),
        "planned_exit_count": planned_count,
        "planned_exit_symbols": diag.get("planned_exit_symbols") or [],
        "manage_orders_rows": handoff_rows,
        "manage_orders_handoff_path": str(HANDOFF_CSV.relative_to(ROOT)).replace("\\", "/"),
        "handoff_status": handoff_status,
        "handoff_block_reason": diag.get("handoff_block_reason") or "",
        "handoff_source": diag.get("handoff_source") or handoff_source,
        "preview_generation_allowed": diag.get("preview_generation_allowed"),
        "execution_allowed": diag.get("execution_allowed"),
        "preview_meta": preview_meta,
        "handoff_orders_path": str(handoff_path.relative_to(ROOT)).replace("\\", "/"),
        "batch_empty_explained": batch_explained,
        "readiness_status": readiness_status,
        "governance_authorized": readiness.get("governance_authorized"),
        "execution_authorized": gate_ok,
        "paper_entry_gate_reason": gate_reason,
        "placement_dry_run_ok": placement_proc.returncode == 0,
        "placement_maps_exit_rows": placement_maps_exit_rows,
        "placement_batch_empty": placement_batch_empty,
        "broker_orders_submitted": False,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "handoff_diagnostic": diag,
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197G-HANDOFF] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197G-HANDOFF] status={status} planned={planned_count} handoff={handoff_rows} "
            f"placement_maps_exit={placement_maps_exit_rows}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197G defensive exit handoff validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197g_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
