"""
phase_197h_defensive_exit_sizing_cap_validation.py
Phase 197H — Validate defensive exit batch notional cap classification and capped preview.
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
OUTPUT_JSON = RESULTS / "phase_197h_defensive_exit_sizing_cap_validation.json"
HANDOFF_CSV = LIVE / "manage_orders.csv"
READINESS_JSON = RESULTS / "execution_readiness.json"
PREVIOUS_COMMIT = "fd9c898b"
PREVIOUS_PHASE = "197G"


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


def run_phase_197h_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    git_clean_before = _git_clean()

    from services.defensive_exit_sizing_cap import (
        analyze_defensive_exit_sizing,
        load_defensive_exit_handoff_orders,
        write_capped_batch_preview,
    )
    from services.paper_entry_loop_stage import evaluate_paper_entry_gates
    from services.portfolio_history_sync import load_live_execution_flag

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
    placement_notional_block = "BATCH_NOTIONAL_BLOCK" in placement_out
    placement_handoff_notional = "HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED" in placement_out

    sizing = analyze_defensive_exit_sizing(placement_output=placement_out, use_cli_default_cap=True)
    write_capped_batch_preview(sizing)

    handoff_orders = load_defensive_exit_handoff_orders()
    handoff_rows = len(handoff_orders)

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
    readiness = {}
    try:
        readiness = json.loads(READINESS_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    readiness_status = str(readiness.get("readiness_status") or "UNKNOWN")
    live_permitted = bool(readiness.get("live_execution_permitted", True))

    preview = sizing.get("capped_preview_cli_cap") or sizing.get("capped_preview") or {}
    capped_symbols = list(preview.get("capped_preview_symbols") or [])
    deferred_symbols = list(preview.get("deferred_symbols") or [])
    partial_authorized = bool(sizing.get("partial_batch_authorized"))

    status = "PASS"
    if handoff_rows == 0:
        status = "FAIL"
        notes.append("no defensive handoff rows in data/live/manage_orders.csv")
    if not placement_maps_exit_rows:
        status = "FAIL"
        notes.append("197G handoff regression: placement no longer maps EXIT rows")
    if placement_batch_empty:
        status = "FAIL"
        notes.append("vague BATCH_EMPTY with handoff rows present")
    if not placement_notional_block and not placement_handoff_notional:
        status = "FAIL"
        notes.append("expected explicit BATCH_NOTIONAL_BLOCK classification")
    if not sizing.get("cap_excess_amount"):
        status = "FAIL"
        notes.append("cap_excess_amount not computed")
    if partial_authorized and not capped_symbols:
        notes.append(
            "partial batch authorized but capped preview empty (cap may be too low for any single row)"
        )
    if live_permitted or load_live_execution_flag():
        status = "FAIL"
        notes.append("live execution must remain blocked")

    payload: Dict[str, Any] = {
        "phase": "197H",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "Defensive exit handoff maps correctly after 197G but full batch notional (~95k) "
            "exceeds place_live_orders default max_batch_notional (5000) or manage_positions "
            "configured cap (50000), causing placement dry-run failure without explicit classification."
        ),
        "fix": (
            "Classify HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED, emit capped batch preview by "
            "plan priority, and preserve partial-batch policy from max_management_orders_per_run."
        ),
        "broker_snapshot_ok": snapshot_ok,
        "live_blocked": not live_permitted,
        "paper_mode": True,
        "positions_count": _count_positions(),
        "open_orders_count": _count_open_orders(),
        "handoff_rows": handoff_rows,
        "planned_exit_count": handoff_rows,
        "placement_maps_exit_rows": placement_maps_exit_rows,
        "placement_dry_run_ok": placement_proc.returncode == 0,
        "placement_block_reason": sizing.get("placement_block_reason") or "BATCH_NOTIONAL_BLOCK",
        "placement_notional_block_explicit": placement_handoff_notional or placement_notional_block,
        "batch_notional_estimate": sizing.get("batch_notional_estimate"),
        "configured_batch_cap": sizing.get("configured_batch_cap_effective"),
        "configured_batch_cap_cli_default": sizing.get("configured_batch_cap_cli_default"),
        "configured_batch_cap_manage_positions": sizing.get(
            "configured_batch_cap_manage_positions"
        ),
        "cap_excess_amount": sizing.get("cap_excess_amount"),
        "capped_preview_available": bool(capped_symbols),
        "capped_preview_symbols": capped_symbols,
        "deferred_symbols": deferred_symbols,
        "partial_batch_authorized": partial_authorized,
        "partial_batch_status": sizing.get("partial_batch_status"),
        "broker_orders_submitted": False,
        "readiness_status": readiness_status,
        "governance_authorized": readiness.get("governance_authorized"),
        "execution_authorized": gate_ok,
        "paper_entry_gate_reason": gate_reason,
        "handoff_status": sizing.get("handoff_status"),
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "sizing_analysis": sizing,
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197H-SIZING] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197H-SIZING] status={status} notional={sizing.get('batch_notional_estimate')} "
            f"cap={sizing.get('configured_batch_cap_effective')} "
            f"preview={capped_symbols}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197H defensive exit sizing cap validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197h_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
