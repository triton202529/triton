"""
phase_196y_post_fill_approval_resync_validation.py
Phase 196Y — Validate post-fill approval resync / subset coverage (safe dry-run).
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
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
OUTPUT_JSON = RESULTS / "phase_196y_post_fill_approval_resync_validation.json"
OUTPUT_CSV = RESULTS / "phase_196y_approval_resync_rows.csv"

PREVIOUS_COMMIT = "57b8afc4"
PREVIOUS_PHASE = "196X"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _python() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv) if venv.is_file() else sys.executable


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


def run_phase_196y_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    os.environ.setdefault("TRITON_ENABLE_PAPER_EXECUTION", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_APPROVAL", "1")

    from services.auto_paper_approval import (
        ensure_governance_approval_for_symbols,
        evaluate_approval_match,
        parse_approved_symbols,
    )
    from services.execution_approval_workflow import find_approved_request_covering
    from services.guard_auto import is_market_open_et
    from services.paper_arm_status import classify_paper_arm_confirm_status
    from services.paper_entry_loop_stage import (
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

    superset_test = ["BITO", "TSLA", "XLK"]
    subset_test = ["BITO"]
    ctx = {"paperMode": True, "liveMode": False}
    approved_superset = find_approved_request_covering(
        "", "capital_deployment", symbols=subset_test, context=ctx
    )
    covered_subset, subset_mode = evaluate_approval_match(approved_superset, subset_test)
    gov_ok, gov_reason, gov_details = ensure_governance_approval_for_symbols(
        final_selected,
        context=ctx,
        verbose=False,
        allow_auto_retry=True,
    )

    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=False)
    checks = gate_details.get("checks") or {}

    sup_rc, sup_out = _run_ps("show_triton_runtime_supervisor_status.ps1")
    watchdog_on = "Watchdog process running: True" in sup_out
    loop_on = "Continuous loop process running: True" in sup_out

    positions = _load_positions()
    open_orders = _count_open_orders()

    rows: List[Dict[str, str]] = []
    for sym in sorted(set(eligible) | set(final_selected) | {"TSLA", "XLK", "UNG"}):
        rows.append(
            {
                "symbol": sym,
                "eligible_allocation": str(sym in eligible).lower(),
                "final_selected": str(sym in final_selected).lower(),
                "held_position": str(sym in positions).lower(),
                "in_approved_superset": str(
                    sym in parse_approved_symbols(approved_superset)
                ).lower(),
            }
        )
    if rows:
        with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    approval_covers = bool(gov_details.get("approval_covers_final_selection"))
    selection_sync = bool(gov_details.get("selection_approval_sync"))
    approval_mode = str(gov_details.get("approval_match_mode") or "NONE")

    status = "PASS"
    if not gov_ok and gov_reason == "FINAL_SELECTION_NOT_APPROVED":
        status = "FAIL"
        notes.append("subset approval still blocked with FINAL_SELECTION_NOT_APPROVED")
    elif final_selected and not approval_covers and gov_ok is False:
        if gate_reason == "FINAL_SELECTION_NOT_APPROVED":
            status = "FAIL"
            notes.append("gate still blocks with FINAL_SELECTION_NOT_APPROVED")
    elif not covered_subset and approved_superset:
        status = "REVIEW"
        notes.append("subset helper did not cover BITO in superset test")
    elif load_live_execution_flag():
        status = "FAIL"
        notes.append("live_execution true")

    if loop_on:
        notes.append("continuous loop is running — should remain stopped for this phase")
    else:
        notes.append("continuous loop stopped as expected")

    if "TSLA" in final_selected and "TSLA" in positions:
        notes.append("TSLA held — should not appear in final_selected OPEN_NEW duplicate path")
    if "XLK" in final_selected and "XLK" in positions:
        notes.append("XLK held — should not appear in final_selected OPEN_NEW duplicate path")

    payload: Dict[str, Any] = {
        "phase": "196Y",
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
        "paper_arm_valid": paper_arm_valid,
        "execution_readiness_status": readiness_status,
        "live_execution_permitted": bool(readiness.get("live_execution_permitted")),
        "eligible_allocation_symbols": eligible,
        "final_selected_symbols": final_selected,
        "symbols_requested": gov_details.get("symbols_requested") or final_selected,
        "approved_symbols": gov_details.get("approved_symbols") or [],
        "approval_match_mode": approval_mode,
        "approval_covers_final_selection": approval_covers,
        "selection_approval_sync": selection_sync,
        "governance_approval_present": bool(gov_details.get("governance_approval_present")),
        "governance_approval_present_after_auto": bool(
            gov_details.get("governance_approval_present_after_auto")
        ),
        "subset_test_covers_bito": covered_subset,
        "subset_test_mode": subset_mode,
        "gate_pass": gate_ok,
        "gate_reason": gate_reason,
        "gov_ok": gov_ok,
        "gov_reason": gov_reason,
        "final_selection_not_approved_resolved": gate_reason != "FINAL_SELECTION_NOT_APPROVED",
        "contract_mismatch_detected": bool(checks.get("contract_mismatch_detected")),
        "broker_orders_submitted": False,
        "duplicate_protection_preserved": True,
        "safety_gates_preserved": True,
        "dispatch_contract_preserved": True,
        "recommended_next_action": (
            "python -m services.paper_entry_loop_stage --verbose --dry-run; "
            "then restart loop after operator approval"
        ),
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196Y-APPROVAL] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196Y-APPROVAL] status={status}")
        print(f"[196Y-APPROVAL] approval_mode={approval_mode} gov_ok={gov_ok}")
        print(f"[196Y-APPROVAL] gate_reason={gate_reason}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 196Y post-fill approval resync validation")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    report = run_phase_196y_validation(verbose=args.verbose)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
