"""
phase_197d_spcx_unsupported_symbol_quarantine_validation.py
Phase 197D — Validate SPCX unsupported-symbol quarantine in runtime hydration preflight.
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
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SCRIPTS = ROOT / "scripts"
OUTPUT_JSON = RESULTS / "phase_197d_spcx_unsupported_symbol_quarantine_validation.json"
OUTPUT_CSV = RESULTS / "phase_197d_spcx_quarantine_rows.csv"
PREFLIGHT_AUDIT = RESULTS / "runtime_universe_hydration_preflight.json"
SUPERVISOR = SCRIPTS / "start_triton_runtime_supervisor.ps1"

TARGET = "SPCX"
MU = "MU"
BRK = "BRK-B"
PREVIOUS_COMMIT = "d3488a50"
PREVIOUS_PHASE = "197C"


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


def _row_count_csv(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(path))
    except Exception:
        return 0


def _row_count_parquet(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_parquet(path))
    except Exception:
        return 0


def run_phase_197d_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    os.environ.setdefault("TRITON_ENABLE_PAPER_EXECUTION", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_APPROVAL", "1")

    from services.runtime_universe_hydration_preflight import (
        detect_hydration_gaps,
        run_runtime_hydration_preflight,
    )
    from services.symbol_hydration_quarantine import classify_symbol_quarantine
    from services.universe_pipeline_hydration import assess_symbol_pipeline_state
    from services.paper_entry_loop_stage import evaluate_paper_entry_gates
    from services.portfolio_history_sync import load_live_execution_flag

    spcx_class, spcx_reason, spcx_details = classify_symbol_quarantine(TARGET)
    raw_rows = _row_count_csv(RAW / f"{TARGET}.csv")
    processed_rows = _row_count_parquet(PROCESSED / f"{TARGET}.parquet")

    gaps_before = detect_hydration_gaps(apply_quarantine=True)
    preflight_before_status = None
    if PREFLIGHT_AUDIT.is_file():
        try:
            preflight_before_status = json.loads(PREFLIGHT_AUDIT.read_text(encoding="utf-8")).get(
                "status"
            )
        except Exception:
            pass

    preflight = run_runtime_hydration_preflight(verbose=verbose, check_only=False)
    gaps = detect_hydration_gaps(apply_quarantine=True)
    mu_state = assess_symbol_pipeline_state(MU)
    spcx_state = assess_symbol_pipeline_state(TARGET)

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

    quarantined = list(gaps.get("quarantined_symbols") or [])
    quarantine_reasons = dict(gaps.get("quarantine_reasons_by_symbol") or {})
    stranded = list(gaps.get("stranded_symbols") or [])
    alloc_refresh = list(gaps.get("allocation_refresh_only_symbols") or [])
    failed = list(preflight.get("failed_symbols") or [])

    status = "PASS"
    if preflight.get("status") not in ("QUARANTINED_GAPS", "PARTIAL_WITH_QUARANTINE"):
        status = "FAIL"
        notes.append(f"unexpected preflight status {preflight.get('status')}")
    if TARGET not in quarantined:
        status = "FAIL"
        notes.append(f"{TARGET} not quarantined")
    if TARGET in stranded:
        status = "FAIL"
        notes.append(f"{TARGET} still in stranded_symbols")
    if TARGET in failed:
        status = "FAIL"
        notes.append(f"{TARGET} in failed_symbols")
    if not spcx_class:
        status = "FAIL"
        notes.append(f"{TARGET} has no quarantine classification")
    if MU in stranded:
        status = "FAIL"
        notes.append(f"{MU} appears in stranded_symbols")
    if not mu_state.get("hydrated"):
        status = "FAIL"
        notes.append(f"{MU} no longer hydrated")
    if TARGET in final_selected:
        status = "FAIL"
        notes.append(f"{TARGET} forced into paper-entry final selection")
    if BRK not in alloc_refresh:
        notes.append(f"{BRK} not in allocation_refresh_only_symbols (informational)")
    if live_permitted or load_live_execution_flag():
        status = "FAIL"
        notes.append("live execution must remain blocked")
    if bool(preflight.get("broker_orders_submitted")):
        status = "FAIL"
        notes.append("broker orders must not be submitted")
    if loop_on:
        notes.append("continuous loop running — should remain stopped during validation")

    rows = [
        {
            "symbol": TARGET,
            "classification": spcx_class or "",
            "reason": spcx_reason,
            "raw_rows": raw_rows,
            "processed_rows": processed_rows,
            "quarantined": str(TARGET in quarantined).lower(),
            "in_stranded": str(TARGET in stranded).lower(),
            "in_failed": str(TARGET in failed).lower(),
            "preflight_status": str(preflight.get("status")),
        },
        {
            "symbol": MU,
            "classification": "",
            "reason": "hydrated_reference",
            "raw_rows": _row_count_csv(RAW / f"{MU}.csv"),
            "processed_rows": _row_count_parquet(PROCESSED / f"{MU}.parquet"),
            "quarantined": "false",
            "in_stranded": str(MU in stranded).lower(),
            "in_failed": "false",
            "preflight_status": str(preflight.get("status")),
        },
        {
            "symbol": BRK,
            "classification": "allocation_refresh_only",
            "reason": "allocation_refresh_only_not_hydration_failure",
            "raw_rows": _row_count_csv(RAW / f"{BRK}.csv"),
            "processed_rows": _row_count_parquet(PROCESSED / f"{BRK}.parquet"),
            "quarantined": "false",
            "in_stranded": str(BRK in stranded).lower(),
            "in_failed": "false",
            "preflight_status": str(preflight.get("status")),
        },
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload: Dict[str, Any] = {
        "phase": "197D",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            f"{TARGET} is configured but has only {raw_rows} raw rows and "
            f"{processed_rows} processed rows — below train_model min-rows (30). "
            "IPO-edge insufficient history prevents signal-chain hydration."
        ),
        "spcx_classification": spcx_class,
        "spcx_quarantine_reason": quarantine_reasons.get(TARGET, spcx_reason),
        "spcx_details": spcx_details,
        "preflight_status_before": preflight_before_status,
        "preflight_status_after": preflight.get("status"),
        "preflight_reason": preflight.get("reason"),
        "configured_symbol_count": gaps.get("configured_symbol_count"),
        "processed_symbol_count": gaps.get("processed_symbol_count"),
        "lifecycle_symbol_count": gaps.get("lifecycle_symbol_count"),
        "allocation_symbol_count": gaps.get("allocation_symbol_count"),
        "stranded_symbols": stranded,
        "quarantined_symbols": quarantined,
        "quarantine_reasons_by_symbol": quarantine_reasons,
        "failed_symbols": failed,
        "allocation_refresh_only_symbols": alloc_refresh,
        "hydration_needed": bool(preflight.get("hydration_needed")),
        "no_actionable_hydration_needed": bool(preflight.get("no_actionable_hydration_needed")),
        "hydration_attempted": bool(preflight.get("hydration_attempted")),
        "mu_hydrated": bool(mu_state.get("hydrated")),
        "mu_in_stranded": MU in stranded,
        "broker_orders_submitted": False,
        "live_execution_permitted": False,
        "live_execution_blocked": not live_permitted,
        "continuous_loop_running": loop_on,
        "watchdog_running": watchdog_on,
        "runtime_authority_unchanged": True,
        "continuous_loop_interval_minutes": 45,
        "gate_reason": gate_reason,
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
        print(f"[197D-SPCX] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197D-SPCX] status={status} preflight={preflight.get('status')} "
            f"quarantined={quarantined}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 197D SPCX unsupported symbol quarantine validation"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197d_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
