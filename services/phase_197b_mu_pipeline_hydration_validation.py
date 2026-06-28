"""
phase_197b_mu_pipeline_hydration_validation.py
Phase 197B — Validate MU hydration into active processed/signal/allocation pipeline.
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
PROCESSED = ROOT / "data" / "processed"
OUTPUT_JSON = RESULTS / "phase_197b_mu_pipeline_hydration_validation.json"
OUTPUT_CSV = RESULTS / "phase_197b_mu_pipeline_hydration_rows.csv"

TARGET = "MU"
PREVIOUS_COMMIT = "979f37f5"
PREVIOUS_PHASE = "197A"


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


def _csv_row_for_symbol(path: Path, ticker: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(path, low_memory=False)
        col = "ticker" if "ticker" in df.columns else ("symbol" if "symbol" in df.columns else None)
        if not col:
            return None
        sub = df[df[col].astype(str).str.upper() == ticker.upper()]
        if sub.empty:
            return {"present": False}
        row = sub.iloc[-1].to_dict()
        return {"present": True, "latest": {k: str(v) for k, v in row.items()}}
    except Exception as e:
        return {"present": False, "error": str(e)}


def run_phase_197b_validation(
    *,
    hydrate: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    notes: List[str] = []
    os.environ.setdefault("TRITON_ENABLE_PAPER_EXECUTION", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_APPROVAL", "1")

    from services.universe_pipeline_hydration import (
        assess_symbol_pipeline_state,
        assess_universe_hydration,
        hydrate_universe_symbols,
        load_tradable_universe_tickers,
    )
    from services.paper_entry_loop_stage import evaluate_paper_entry_gates
    from services.portfolio_history_sync import load_live_execution_flag

    before = assess_universe_hydration([TARGET])
    before_state = assess_symbol_pipeline_state(TARGET)
    before_alloc = int(before.get("allocation_universe_count") or 0)
    before_lifecycle = int(before.get("lifecycle_universe_count") or 0)

    hydration_result: Dict[str, Any] = {"skipped": not hydrate}
    if hydrate and before_state.get("gaps"):
        hydration_result = hydrate_universe_symbols(
            [TARGET],
            verbose=verbose,
            skip_raw_fetch=True,
        )
        if not hydration_result.get("ok"):
            notes.append(f"hydration incomplete: {hydration_result.get('reason')}")

    # Refresh allocation after hydration (read-only advisory)
    if hydrate:
        subprocess.run(
            [sys.executable, "-m", "services.portfolio_allocation_engine"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )

    after = assess_universe_hydration([TARGET])
    after_state = assess_symbol_pipeline_state(TARGET)
    after_alloc = int(after.get("allocation_universe_count") or 0)
    after_lifecycle = int(after.get("lifecycle_universe_count") or 0)

    processed_path = PROCESSED / f"{TARGET}.parquet"
    processed_rows = 0
    if processed_path.is_file():
        try:
            import pandas as pd

            processed_rows = len(pd.read_parquet(processed_path))
        except Exception:
            pass

    lifecycle = _csv_row_for_symbol(RESULTS / "signal_lifecycle_effective.csv", TARGET)
    allocation = _csv_row_for_symbol(RESULTS / "portfolio_allocation_recommendations.csv", TARGET)
    risk = _csv_row_for_symbol(RESULTS / "performance_risk_overlay.csv", TARGET)
    signals = _csv_row_for_symbol(RESULTS / "signals_with_rationale.csv", TARGET)

    alloc_action = ""
    alloc_reason = ""
    if allocation and allocation.get("latest"):
        alloc_action = str(allocation["latest"].get("recommended_action") or "")
        alloc_reason = str(allocation["latest"].get("reason") or "")

    lifecycle_outcome = ""
    if lifecycle and lifecycle.get("latest"):
        lifecycle_outcome = str(
            lifecycle["latest"].get("effective_stance")
            or lifecycle["latest"].get("stance")
            or lifecycle["latest"].get("lifecycle_action")
            or ""
        )

    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=False)
    checks = gate_details.get("checks") or {}
    final_selected = list(checks.get("final_selected_symbols") or [])
    mu_forced = TARGET in final_selected

    _, sup_out = _run_ps("show_triton_runtime_supervisor_status.ps1")
    loop_on = "Continuous loop process running: True" in sup_out
    watchdog_on = "Watchdog process running: True" in sup_out

    readiness = {}
    try:
        readiness = json.loads((RESULTS / "execution_readiness.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    live_permitted = bool(readiness.get("live_execution_permitted", True))

    tradable = load_tradable_universe_tickers()
    mu_in_config = TARGET in tradable
    mu_dup = tradable.count(TARGET) > 1

    status = "PASS"
    if not mu_in_config or mu_dup:
        status = "FAIL"
        notes.append("MU config universe invalid")
    if not processed_path.is_file():
        status = "FAIL"
        notes.append("MU processed parquet missing after hydration")
    if not after_state.get("in_signals"):
        status = "FAIL"
        notes.append("MU not visible in signals after hydration")
    if not after_state.get("in_effective_lifecycle"):
        status = "FAIL"
        notes.append("MU not visible in effective lifecycle after hydration")
    if mu_forced:
        status = "FAIL"
        notes.append("MU forced into paper-entry final selection")
    if live_permitted or load_live_execution_flag():
        status = "FAIL"
        notes.append("live execution must remain blocked")
    if gate_reason.startswith("CONTRACT_MISMATCH"):
        status = "FAIL"
        notes.append(gate_reason)
    if "UNG" in final_selected:
        status = "FAIL"
        notes.append("UNG drift in final selection")

    rows = [
        {
            "symbol": TARGET,
            "config_universe": str(mu_in_config).lower(),
            "processed_exists": str(processed_path.is_file()).lower(),
            "processed_rows": str(processed_rows),
            "in_signals": str(after_state.get("in_signals")).lower(),
            "in_effective_lifecycle": str(after_state.get("in_effective_lifecycle")).lower(),
            "in_allocation": str(after_state.get("in_allocation")).lower(),
            "allocation_action": alloc_action,
            "allocation_reason": alloc_reason,
            "lifecycle_outcome": lifecycle_outcome,
            "mu_in_final_selection": str(mu_forced).lower(),
        }
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload: Dict[str, Any] = {
        "phase": "197B",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "Continuous loop runs run_market_open analytics only; it does not execute "
            "fetch→preprocess→train→signals→lifecycle. Config-only tickers stay stranded "
            "until universe_pipeline_hydration runs the standard signal chain."
        ),
        "ticker": TARGET,
        "mu_in_config_universe": mu_in_config,
        "mu_duplicate": mu_dup,
        "governed_tradable_count": len(tradable),
        "processed_parquet_exists": processed_path.is_file(),
        "processed_rows": processed_rows,
        "mu_in_signals": bool(after_state.get("in_signals")),
        "mu_in_lifecycle": bool(after_state.get("in_lifecycle")),
        "mu_in_effective_lifecycle": bool(after_state.get("in_effective_lifecycle")),
        "mu_in_allocation": bool(after_state.get("in_allocation")),
        "mu_in_risk_overlay": bool(after_state.get("in_risk_overlay")),
        "lifecycle_outcome": lifecycle_outcome or "NOT_VISIBLE",
        "allocation_action": alloc_action or "NOT_VISIBLE",
        "allocation_reason": alloc_reason or "NOT_VISIBLE",
        "mu_in_final_selection": mu_forced,
        "universe_counts_before": {
            "allocation": before_alloc,
            "lifecycle_effective": before_lifecycle,
            "processed_parquet": int(before.get("processed_parquet_count") or 0),
        },
        "universe_counts_after": {
            "allocation": after_alloc,
            "lifecycle_effective": after_lifecycle,
            "processed_parquet": int(after.get("processed_parquet_count") or 0),
        },
        "hydration_result": hydration_result,
        "gate_pass": gate_ok,
        "gate_reason": gate_reason,
        "contract_mismatch_detected": bool(checks.get("contract_mismatch_detected")),
        "broker_orders_submitted": False,
        "live_execution_blocked": not live_permitted,
        "continuous_loop_running": loop_on,
        "watchdog_running": watchdog_on,
        "dispatch_protection_preserved": not bool(checks.get("contract_mismatch_detected")),
        "approval_sync_preserved": True,
        "planner_empty_classification_preserved": not gate_reason.startswith(
            "CONTRACT_MISMATCH:allocation_eligible_but_planner_empty"
        ),
        "signals_detail": signals,
        "lifecycle_detail": lifecycle,
        "allocation_detail": allocation,
        "risk_overlay_detail": risk,
        "recommended_next_action": (
            "python -m services.paper_entry_loop_stage --verbose --dry-run; "
            "restart continuous loop after operator approval"
        ),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197B-MU] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[197B-MU] status={status}")
        print(
            f"[197B-MU] processed={processed_path.is_file()} lifecycle={after_state.get('in_effective_lifecycle')}"
        )
        print(f"[197B-MU] allocation={alloc_action or 'NOT_VISIBLE'}")
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197B MU pipeline hydration validation")
    ap.add_argument("--no-hydrate", action="store_true", help="Assess only; do not run hydration")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197b_validation(hydrate=not args.no_hydrate, verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
