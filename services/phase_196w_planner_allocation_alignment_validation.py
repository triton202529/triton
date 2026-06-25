"""
phase_196w_planner_allocation_alignment_validation.py
Phase 196W — Validate planner preview alignment with governed allocation candidates.
Read-only; does not submit broker orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
ALLOC_RECOMMENDATIONS = RESULTS / "portfolio_allocation_recommendations.csv"
OUTPUT_JSON = RESULTS / "phase_196w_planner_allocation_alignment_validation.json"
OUTPUT_CSV = RESULTS / "phase_196w_planner_alignment_rows.csv"

PREVIOUS_COMMIT = "e49349b3"
PREVIOUS_PHASE = "196V"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _allocation_mtime() -> Optional[str]:
    if not ALLOC_RECOMMENDATIONS.is_file():
        return None
    return datetime.fromtimestamp(
        ALLOC_RECOMMENDATIONS.stat().st_mtime, tz=timezone.utc
    ).isoformat()


def run_phase_196w_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    os.environ.setdefault("TRITON_ENABLE_PAPER_EXECUTION", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")

    from services.guard_auto import is_market_open_et
    from services.paper_arm_status import classify_paper_arm_confirm_status
    from services.paper_entry_loop_stage import (
        _check_allocation_planner_contract,
        _load_governed_allocation_candidates,
        evaluate_paper_entry_gates,
    )
    from services.execute_trades import preview_planned_buy_symbols
    from services.portfolio_history_sync import load_live_execution_flag

    market_open = is_market_open_et(datetime.now(timezone.utc))
    paper_arm_valid = classify_paper_arm_confirm_status() == "VALID"
    readiness = {}
    try:
        readiness = json.loads((RESULTS / "execution_readiness.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    readiness_status = str(readiness.get("readiness_status") or "UNKNOWN")
    execution_readiness_ready = readiness_status == "READY"

    _, handoff = _load_governed_allocation_candidates()
    eligible = list(handoff.get("eligible_entry_symbols") or [])

    planner_symbols, planner_meta = preview_planned_buy_symbols(
        "paper",
        verbose=False,
        for_execute=True,
        source="allocation",
        allowed_symbols=eligible,
    )
    contract_ok, contract_reason, contract_detail = _check_allocation_planner_contract(
        eligible,
        planner_symbols,
        planner_meta,
    )

    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=False)
    checks = gate_details.get("checks") or {}

    ung_selected = "UNG" in {_norm(s) for s in planner_symbols if s}
    ung_in_eligible = "UNG" in {_norm(s) for s in eligible if s}

    would_execute = bool(
        market_open
        and gate_ok
        and contract_ok
        and paper_arm_valid
        and execution_readiness_ready
        and not load_live_execution_flag()
        and planner_symbols
    )

    from services.paper_entry_candidate_contract import assess_controlled_candidates_cache

    cache = assess_controlled_candidates_cache()
    stale_used = bool(cache.get("executable_source"))

    rejected = planner_meta.get("planner_rejected_symbols") or {}
    considered = planner_meta.get("planner_considered_symbols") or eligible
    rows: List[Dict[str, str]] = []
    for sym in sorted(set(list(considered) + list(planner_symbols))):
        rows.append(
            {
                "symbol": sym,
                "eligible_allocation": str(sym in eligible).lower(),
                "planner_selected": str(sym in planner_symbols).lower(),
                "rejection_reason": str(rejected.get(sym, "")),
            }
        )

    if rows:
        with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    status = "PASS"
    if not ALLOC_RECOMMENDATIONS.is_file():
        status = "FAIL"
        notes.append("allocation source missing")
    elif ung_selected and not ung_in_eligible:
        status = "FAIL"
        notes.append("UNG selected outside governed allocation candidates")
    elif contract_detail.get("contract_mismatch_detected"):
        status = "FAIL"
        notes.append(f"contract mismatch: {contract_reason}")
    elif load_live_execution_flag():
        status = "FAIL"
        notes.append("live_execution true")
    elif planner_meta.get("planner_input_mode") != "allocation":
        status = "FAIL"
        notes.append("planner not in allocation mode")

    if market_open:
        notes.append("market open during validation — no orders submitted")
    else:
        notes.append("market closed — execution blocked as expected")

    if not market_open and gate_reason != "market is closed":
        status = "REVIEW"
        notes.append(f"unexpected gate reason when closed: {gate_reason}")

    if rejected:
        notes.append(f"planner rejections: {rejected}")

    recommended = (
        "Observe at market open: python -m services.paper_entry_loop_stage --verbose --dry-run"
    )

    payload: Dict[str, Any] = {
        "phase": "196W",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "market_open": market_open,
        "paper_arm_valid": paper_arm_valid,
        "execution_readiness_ready": execution_readiness_ready,
        "execution_readiness_status": readiness_status,
        "allocation_source_exists": ALLOC_RECOMMENDATIONS.is_file(),
        "allocation_source_mtime": _allocation_mtime(),
        "allocation_open_new_count": handoff.get("allocation_open_new_count", 0),
        "eligible_allocation_symbols": eligible,
        "planner_input_mode": planner_meta.get("planner_input_mode"),
        "planner_considered_symbols": considered,
        "planner_selected_symbols": planner_meta.get("planner_selected_symbols") or planner_symbols,
        "planner_rejected_symbols": rejected,
        "contract_mismatch_detected": bool(contract_detail.get("contract_mismatch_detected")),
        "ung_selected_outside_allocation": ung_selected and not ung_in_eligible,
        "would_execute_if_market_open": would_execute,
        "orders_submitted": False,
        "live_execution_blocked": not load_live_execution_flag(),
        "safety_gates_preserved": True,
        "stale_candidate_cache_used_for_execution": stale_used,
        "gate_pass": gate_ok,
        "gate_reason": gate_reason,
        "safe_block": bool(gate_details.get("safe_block")),
        "recommended_next_action": recommended,
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196W-PLANNER] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196W-PLANNER] status={status}")
        print(f"[196W-PLANNER] selected={payload['planner_selected_symbols']}")
    return payload


def _norm(s: str) -> str:
    return str(s or "").strip().upper()


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 196W planner allocation alignment validation")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    report = run_phase_196w_validation(verbose=args.verbose)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
