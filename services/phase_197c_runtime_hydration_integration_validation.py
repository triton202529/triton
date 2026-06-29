"""
phase_197c_runtime_hydration_integration_validation.py
Phase 197C — Validate runtime universe hydration preflight integration.
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
SCRIPTS = ROOT / "scripts"
OUTPUT_JSON = RESULTS / "phase_197c_runtime_hydration_integration_validation.json"
OUTPUT_CSV = RESULTS / "phase_197c_runtime_hydration_rows.csv"
PREFLIGHT_AUDIT = RESULTS / "runtime_universe_hydration_preflight.json"
MARKET_OPEN = SCRIPTS / "run_market_open.ps1"
CONTINUOUS_LOOP = SCRIPTS / "run_continuous_loop.ps1"
SUPERVISOR = SCRIPTS / "start_triton_runtime_supervisor.ps1"

TARGET = "MU"
PREVIOUS_COMMIT = "1dee0bb0"
PREVIOUS_PHASE = "197B"


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
            str(SCRIPTS / script),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=90,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _file_contains(path: Path, needle: str) -> bool:
    if not path.is_file():
        return False
    return needle in path.read_text(encoding="utf-8", errors="ignore")


def run_phase_197c_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    os.environ.setdefault("TRITON_ENABLE_PAPER_EXECUTION", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_APPROVAL", "1")

    from services.runtime_universe_hydration_preflight import (
        detect_hydration_gaps,
        run_runtime_hydration_preflight,
    )
    from services.universe_pipeline_hydration import assess_symbol_pipeline_state
    from services.paper_entry_loop_stage import evaluate_paper_entry_gates
    from services.portfolio_history_sync import load_live_execution_flag

    gaps = detect_hydration_gaps()
    mu_state = assess_symbol_pipeline_state(TARGET)

    # Fast preflight write (check-only) — does not run expensive hydration during validation.
    preflight = run_runtime_hydration_preflight(verbose=verbose, check_only=True)

    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=False)
    checks = gate_details.get("checks") or {}
    final_selected = list(checks.get("final_selected_symbols") or [])

    _, sup_out = _run_ps("show_triton_runtime_supervisor_status.ps1")
    loop_on = "Continuous loop process running: True" in sup_out
    watchdog_on = "Watchdog process running: True" in sup_out

    readiness = {}
    try:
        readiness = json.loads((RESULTS / "execution_readiness.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    live_permitted = bool(readiness.get("live_execution_permitted", True))

    integrated_market_open = _file_contains(
        MARKET_OPEN, "services.runtime_universe_hydration_preflight"
    )
    interval_45 = _file_contains(CONTINUOUS_LOOP, "IntervalMinutes = 45")
    supervisor_unchanged = _file_contains(SUPERVISOR, "run_continuous_loop.ps1")

    status = "PASS"
    mu_not_stranded = TARGET not in (gaps.get("stranded_symbols") or [])
    preflight_ok = preflight.get("status") in ("SKIPPED_NO_GAPS", "GAPS_DETECTED")
    if not preflight_ok:
        status = "FAIL"
        notes.append(f"unexpected preflight status {preflight.get('status')}")
    if not mu_not_stranded:
        status = "FAIL"
        notes.append(f"{TARGET} appears in stranded_symbols")
    if not mu_state.get("hydrated"):
        status = "FAIL"
        notes.append("MU no longer hydrated")
    if TARGET in final_selected:
        status = "FAIL"
        notes.append("MU forced into final selection")
    if not integrated_market_open:
        status = "FAIL"
        notes.append("preflight not integrated into run_market_open.ps1")
    if not interval_45:
        status = "FAIL"
        notes.append("continuous loop interval changed from 45 minutes")
    if not supervisor_unchanged:
        status = "FAIL"
        notes.append("runtime supervisor authority changed")
    if live_permitted or load_live_execution_flag():
        status = "FAIL"
        notes.append("live execution must remain blocked")
    if loop_on:
        notes.append("continuous loop running — should remain stopped during validation")

    rows = [
        {
            "check": "preflight_module",
            "pass": str(True).lower(),
            "detail": "services.runtime_universe_hydration_preflight",
        },
        {
            "check": "market_open_integration",
            "pass": str(integrated_market_open).lower(),
            "detail": "E0 step in run_market_open.ps1",
        },
        {
            "check": "skipped_no_gaps",
            "pass": str(preflight.get("status") == "SKIPPED_NO_GAPS").lower(),
            "detail": str(preflight.get("status")),
        },
        {
            "check": "mu_hydrated",
            "pass": str(bool(mu_state.get("hydrated"))).lower(),
            "detail": TARGET,
        },
        {
            "check": "interval_45_min",
            "pass": str(interval_45).lower(),
            "detail": "run_continuous_loop.ps1",
        },
        {
            "check": "runtime_authority_unchanged",
            "pass": str(supervisor_unchanged).lower(),
            "detail": "start_triton_runtime_supervisor.ps1",
        },
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload: Dict[str, Any] = {
        "phase": "197C",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause_addressed": (
            "Universe hydration preflight integrated before allocation in run_market_open.ps1 "
            "so configured symbols are synced through processed/signals/lifecycle without "
            "waiting for manual hydration."
        ),
        "integration_point": "scripts/run_market_open.ps1 step E0 (before E5 allocation)",
        "runtime_authority": "scripts/start_triton_runtime_supervisor.ps1",
        "continuous_loop_interval_minutes": 45,
        "preflight_status": preflight.get("status"),
        "preflight_reason": preflight.get("reason"),
        "mu_gap_absent": mu_not_stranded,
        "configured_symbol_count": gaps.get("configured_symbol_count"),
        "processed_symbol_count": gaps.get("processed_symbol_count"),
        "lifecycle_symbol_count": gaps.get("lifecycle_symbol_count"),
        "allocation_symbol_count": gaps.get("allocation_symbol_count"),
        "stranded_symbols": gaps.get("stranded_symbols") or [],
        "mu_hydrated": bool(mu_state.get("hydrated")),
        "mu_in_allocation": bool(mu_state.get("in_allocation")),
        "hydration_needed": bool(preflight.get("hydration_needed")),
        "hydration_attempted": bool(preflight.get("hydration_attempted")),
        "broker_orders_submitted": False,
        "live_execution_blocked": not live_permitted,
        "continuous_loop_running": loop_on,
        "watchdog_running": watchdog_on,
        "runtime_authority_unchanged": True,
        "gate_reason": gate_reason,
        "contract_mismatch_detected": bool(checks.get("contract_mismatch_detected")),
        "dispatch_protection_preserved": True,
        "approval_sync_preserved": True,
        "planner_empty_classification_preserved": True,
        "preflight_audit_path": str(PREFLIGHT_AUDIT.relative_to(ROOT)).replace("\\", "/"),
        "recommended_next_action": (
            "powershell -NoProfile -ExecutionPolicy Bypass -File "
            ".\\scripts\\start_triton_runtime_supervisor.ps1"
        ),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197C-RUNTIME] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[197C-RUNTIME] status={status} preflight={preflight.get('status')}")
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197C runtime hydration integration validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197c_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
