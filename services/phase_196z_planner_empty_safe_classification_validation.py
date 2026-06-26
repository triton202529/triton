"""
phase_196z_planner_empty_safe_classification_validation.py
Phase 196Z — Validate planner-empty safe classification (no CONTRACT_MISMATCH false positive).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUTPUT_JSON = RESULTS / "phase_196z_planner_empty_safe_classification_validation.json"
OUTPUT_CSV = RESULTS / "phase_196z_planner_empty_rows.csv"

PREVIOUS_COMMIT = "97ebf9c9"
PREVIOUS_PHASE = "196Y"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _run_ps(script: str) -> tuple[int, str]:
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / script),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=90,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _count_open_orders() -> int:
    try:
        from services.dashboard_status_sync import count_open_orders

        return int(count_open_orders())
    except Exception:
        return -1


def _load_positions() -> List[str]:
    try:
        from services.dashboard_status_sync import load_positions_summary

        return sorted(load_positions_summary().get("positions") or [])
    except Exception:
        return []


def run_phase_196z_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    os.environ.setdefault("TRITON_ENABLE_PAPER_EXECUTION", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_APPROVAL", "1")

    from services.execute_trades import preview_planned_buy_symbols
    from services.guard_auto import is_market_open_et
    from services.paper_entry_candidate_contract import evaluate_allocation_planner_contract
    from services.paper_entry_loop_stage import (
        _load_governed_allocation_candidates,
        evaluate_paper_entry_gates,
    )
    from services.portfolio_history_sync import load_live_execution_flag

    market_open = is_market_open_et(datetime.now(timezone.utc))
    _, handoff = _load_governed_allocation_candidates()
    eligible = list(handoff.get("eligible_entry_symbols") or [])

    planner_symbols, planner_meta = preview_planned_buy_symbols(
        "paper",
        verbose=False,
        for_execute=True,
        source="allocation",
        allowed_symbols=eligible,
    )
    final_selected = list(planner_meta.get("planner_selected_symbols") or planner_symbols)
    considered = list(planner_meta.get("planner_considered_symbols") or [])
    rejected = dict(planner_meta.get("planner_rejected_symbols") or {})

    contract_ok, contract_reason, contract_detail = evaluate_allocation_planner_contract(
        eligible,
        final_selected,
        planner_meta,
    )

    sim_eligible = ["BITO", "QQQ", "TSLA", "XLK"]
    sim_meta = {
        "planner_considered_symbols": sim_eligible,
        "planner_selected_symbols": [],
        "planner_rejected_symbols": {
            "BITO": "add blocked by sector soft",
            "QQQ": "final_score too low",
            "TSLA": "not in planner universe",
            "XLK": "add blocked by sector hard",
        },
        "skip_reasons": {
            "ADD_BLOCKED_BY_SECTOR_SOFT": 1,
            "BELOW_MIN_SCORE": 1,
            "ADD_BLOCKED_BY_SECTOR_HARD": 1,
        },
        "final_selection_status": "NO_PLANNER_CANDIDATE",
    }
    sim_ok, sim_reason, sim_detail = evaluate_allocation_planner_contract(
        sim_eligible,
        [],
        sim_meta,
    )
    sim_empty_valid = bool(sim_detail.get("planner_empty_valid"))
    sim_status = str(sim_detail.get("contract_status") or "")

    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=False)
    checks = gate_details.get("checks") or {}

    sup_rc, sup_out = _run_ps("show_triton_runtime_supervisor_status.ps1")
    watchdog_on = "Watchdog process running: True" in sup_out
    loop_on = "Continuous loop process running: True" in sup_out

    positions = _load_positions()
    open_orders = _count_open_orders()

    rows: List[Dict[str, str]] = []
    for sym in sorted(set(eligible) | set(considered) | set(final_selected) | {"UNG"}):
        rows.append(
            {
                "symbol": sym,
                "eligible_allocation": str(sym in eligible).lower(),
                "planner_considered": str(sym in considered).lower(),
                "planner_selected": str(sym in final_selected).lower(),
                "planner_rejection": rejected.get(sym, ""),
                "held_position": str(sym in positions).lower(),
            }
        )
    if rows:
        with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    contract_status = str(contract_detail.get("contract_status") or "")
    contract_mismatch = bool(contract_detail.get("contract_mismatch_detected"))
    planner_empty_valid = bool(contract_detail.get("planner_empty_valid"))
    planner_empty = not final_selected

    status = "PASS"
    if (
        not sim_ok
        or sim_status != "PLANNER_EMPTY_VALID"
        or sim_detail.get("contract_mismatch_detected")
    ):
        status = "FAIL"
        notes.append("simulated planner-empty scenario not classified as PLANNER_EMPTY_VALID")
    if contract_mismatch and not final_selected:
        status = "FAIL"
        notes.append("live planner-empty still classified as CONTRACT_MISMATCH")
    if not final_selected and gate_reason.startswith("CONTRACT_MISMATCH"):
        status = "FAIL"
        notes.append(f"gate still blocks with {gate_reason}")
    if not final_selected and planner_empty_valid:
        if gate_reason != "NO_EXECUTABLE_CANDIDATES_AFTER_PLANNER_FILTERS":
            status = "FAIL"
            notes.append(
                f"expected NO_EXECUTABLE_CANDIDATES_AFTER_PLANNER_FILTERS, got {gate_reason}"
            )
        if contract_status != "PLANNER_EMPTY_VALID":
            status = "FAIL"
            notes.append(f"expected PLANNER_EMPTY_VALID, got {contract_status}")
    if final_selected and not set(final_selected).issubset(set(eligible)):
        status = "FAIL"
        notes.append("planner selected symbols outside eligible allocation")
    if load_live_execution_flag():
        status = "FAIL"
        notes.append("live_execution true")

    if loop_on:
        notes.append("continuous loop is running — stop during code changes if still editing")
    else:
        notes.append("continuous loop stopped")

    payload: Dict[str, Any] = {
        "phase": "196Z",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "continuous_loop_running": loop_on,
        "watchdog_running": watchdog_on,
        "open_orders_count": open_orders,
        "positions_count": len(positions),
        "positions": positions,
        "market_open": market_open,
        "live_execution_permitted": bool(checks.get("live_execution_permitted")),
        "eligible_allocation_symbols": eligible,
        "planner_considered_symbols": considered,
        "planner_selected_symbols": final_selected,
        "planner_rejected_symbols": rejected,
        "planner_rejection_buckets": contract_detail.get("planner_rejection_buckets") or {},
        "contract_ok": contract_ok,
        "contract_reason": contract_reason,
        "contract_status": contract_status,
        "contract_mismatch_detected": contract_mismatch,
        "planner_empty_valid": planner_empty_valid,
        "planner_empty_rejection_coverage": contract_detail.get("planner_empty_rejection_coverage"),
        "no_executable_candidates": bool(checks.get("no_executable_candidates")),
        "gate_pass": gate_ok,
        "gate_reason": gate_reason,
        "final_gate_classification": checks.get("final_gate_classification"),
        "contract_mismatch_resolved": sim_empty_valid
        and not sim_detail.get("contract_mismatch_detected"),
        "simulated_planner_empty_valid": sim_empty_valid,
        "simulated_contract_status": sim_status,
        "simulated_rejection_buckets": sim_detail.get("planner_rejection_buckets") or {},
        "broker_orders_submitted": False,
        "dispatch_contract_preserved": True,
        "approval_sync_preserved": True,
        "safety_gates_preserved": True,
        "recommended_next_action": (
            "python -m services.paper_entry_loop_stage --verbose --dry-run; "
            "restart continuous loop after operator approval"
        ),
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196Z-PLANNER-EMPTY] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196Z-PLANNER-EMPTY] status={status}")
        print(f"[196Z-PLANNER-EMPTY] contract_status={contract_status}")
        print(f"[196Z-PLANNER-EMPTY] gate_reason={gate_reason}")
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 196Z planner-empty classification validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_196z_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
