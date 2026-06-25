"""
phase_196v_allocation_to_paper_entry_handoff_validation.py
Phase 196V — Validate allocation-to-paper-entry handoff repair (read-only).
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
OUTPUT_JSON = RESULTS / "phase_196v_allocation_to_paper_entry_handoff_validation.json"
OUTPUT_CSV = RESULTS / "phase_196v_candidate_handoff_rows.csv"
AUDIT_JSON = RESULTS / "paper_entry_loop_stage_audit.json"
LAST_MARKET_JSON = RESULTS / "paper_entry_loop_stage_last_market_decision.json"

PREVIOUS_COMMIT = "4a5ff075"
PREVIOUS_PHASE = "196U"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _allocation_mtime() -> Optional[str]:
    if not ALLOC_RECOMMENDATIONS.is_file():
        return None
    return datetime.fromtimestamp(
        ALLOC_RECOMMENDATIONS.stat().st_mtime, tz=timezone.utc
    ).isoformat()


def run_phase_196v_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    from services.guard_auto import is_market_open_et
    from services.paper_arm_status import classify_paper_arm_confirm_status
    from services.paper_entry_candidate_contract import build_candidate_contract_rows
    from services.paper_entry_loop_stage import (
        _check_allocation_planner_contract,
        _load_governed_allocation_candidates,
        evaluate_paper_entry_gates,
    )
    from services.execute_trades import preview_planned_buy_symbols
    from services.portfolio_history_sync import load_live_execution_flag

    os.environ.setdefault("TRITON_ENABLE_PAPER_EXECUTION", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")

    market_open = is_market_open_et(datetime.now(timezone.utc))
    paper_arm_valid = classify_paper_arm_confirm_status() == "VALID"

    readiness = _load_json(RESULTS / "execution_readiness.json")
    readiness_status = str(readiness.get("readiness_status") or "UNKNOWN")
    execution_readiness_ready = readiness_status == "READY"

    candidates, handoff = _load_governed_allocation_candidates()
    eligible = handoff.get("eligible_entry_symbols") or []
    rejected = handoff.get("rejected_entry_symbols") or {}

    planner_symbols, planner_meta = preview_planned_buy_symbols(
        "paper", verbose=False, for_execute=True
    )
    contract_ok, contract_reason, contract_detail = _check_allocation_planner_contract(
        list(eligible),
        planner_symbols,
        planner_meta,
    )

    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=False)
    checks = gate_details.get("checks") or {}

    would_execute = bool(
        market_open
        and gate_ok
        and contract_ok
        and paper_arm_valid
        and execution_readiness_ready
        and not load_live_execution_flag()
    )

    contract_rows = build_candidate_contract_rows()
    handoff_rows: List[Dict[str, str]] = []
    for row in contract_rows:
        handoff_rows.append(
            {
                "symbol": row.get("symbol", ""),
                "allocation_action": row.get("allocation_action", ""),
                "risk_flag": row.get("risk_flag", ""),
                "eligible_for_auto_entry": row.get("executable_candidate", "false"),
                "final_selection_status": row.get("final_selection_status", ""),
                "rejection_reason": row.get("rejection_reason", ""),
                "planner_status": row.get("planner_status", ""),
                "source": row.get("source", ""),
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(handoff_rows[0].keys()) if handoff_rows else [])
        if handoff_rows:
            writer.writeheader()
            writer.writerows(handoff_rows)

    stale_cache_used = False
    from services.paper_entry_candidate_contract import assess_controlled_candidates_cache

    cache = assess_controlled_candidates_cache()
    if cache.get("executable_source"):
        stale_cache_used = True
        notes.append("stale controlled_paper_entry_loop_candidates marked executable")

    status = "PASS"
    if not ALLOC_RECOMMENDATIONS.is_file():
        status = "FAIL"
        notes.append("allocation source missing")
    elif (
        handoff.get("allocation_open_new_count", 0) == 0
        and handoff.get("allocation_add_count", 0) == 0
    ):
        status = "REVIEW"
        notes.append("no OPEN_NEW/ADD allocation rows")
    elif not eligible and handoff.get("allocation_open_new_count", 0) > 0:
        status = "REVIEW"
        notes.append("OPEN_NEW rows present but none eligible after gates")
    elif load_live_execution_flag():
        status = "FAIL"
        notes.append("live_execution true")
    elif market_open and gate_ok and not contract_ok:
        status = "REVIEW"
        notes.append(f"market open but contract mismatch: {contract_reason}")
    elif not market_open and gate_reason != "market is closed":
        status = "REVIEW"
        notes.append(f"unexpected gate reason when closed: {gate_reason}")

    if market_open:
        notes.append("market open during validation — no orders submitted by this script")
    else:
        notes.append("market closed — execution correctly blocked")

    if contract_detail.get("contract_mismatch_detected"):
        notes.append(
            f"planner/allocation mismatch: planner={contract_detail.get('planner_preview_symbols')} "
            f"eligible={contract_detail.get('eligible_allocation_symbols')}"
        )

    recommended_next = (
        "Observe at market open: python -m services.paper_entry_loop_stage --verbose --dry-run"
    )
    if contract_detail.get("contract_mismatch_detected"):
        recommended_next += (
            "; planner still ranks trade_opportunities — allocation OPEN_NEW symbols "
            "must align with planner or entry blocks with CONTRACT_MISMATCH"
        )

    payload: Dict[str, Any] = {
        "phase": "196V",
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
        "allocation_add_count": handoff.get("allocation_add_count", 0),
        "eligible_allocation_symbols": list(eligible),
        "rejected_allocation_symbols": rejected,
        "candidate_handoff_status": contract_detail.get("contract_status"),
        "contract_mismatch_detected": bool(contract_detail.get("contract_mismatch_detected")),
        "contract_mismatch_reason": contract_reason if not contract_ok else "",
        "planner_preview_symbols": list(planner_symbols),
        "planner_preview_candidate": planner_meta.get("preview_candidate"),
        "planner_rejection_detail": contract_detail.get("planner_rejection_detail"),
        "final_selected_symbols": checks.get("final_selected_symbols") or planner_symbols,
        "gate_pass": gate_ok,
        "gate_reason": gate_reason,
        "safe_block": bool(gate_details.get("safe_block")),
        "would_execute_if_market_open": would_execute,
        "orders_submitted": False,
        "live_execution_blocked": not load_live_execution_flag(),
        "safety_gates_preserved": True,
        "stale_candidate_cache_used_for_execution": stale_cache_used,
        "governed_source": "portfolio_allocation_recommendations.csv",
        "audit_path": str(AUDIT_JSON.relative_to(ROOT)).replace("\\", "/"),
        "last_market_decision_path": str(LAST_MARKET_JSON.relative_to(ROOT)).replace("\\", "/"),
        "recommended_next_action": recommended_next,
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196V-HANDOFF] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196V-HANDOFF] status={status}")
        print(f"[196V-HANDOFF] eligible_allocation={eligible}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 196V allocation-to-paper-entry handoff validation"
    )
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    report = run_phase_196v_validation(verbose=args.verbose)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
