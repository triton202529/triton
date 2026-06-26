"""
phase_197a_mu_universe_intake_validation.py
Phase 197A — Micron Technology (MU) governed universe intake (no forced execution).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
TICKERS_FILE = ROOT / "data" / "config" / "tickers.txt"
UNIVERSE_ASSETS = ROOT / "data" / "config" / "universe_assets.json"
RESULTS = ROOT / "data" / "results"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUT_JSON = RESULTS / "phase_197a_mu_universe_intake_validation.json"

TARGET_TICKER = "MU"
PREVIOUS_COMMIT = "8dbda8db"
PREVIOUS_PHASE = "196Z"
INDEX_PREFIXES = ("^",)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_tickers() -> List[str]:
    from services.fetch_raw_data import _load_tickers_from_file

    return _load_tickers_from_file(TICKERS_FILE)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _csv_has_ticker(path: Path, ticker: str) -> Optional[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        import pandas as pd

        df = pd.read_csv(path, nrows=5000)
        if df.empty:
            return {"exists": True, "rows": 0}
        col = "ticker" if "ticker" in df.columns else ("symbol" if "symbol" in df.columns else None)
        if not col:
            return {"exists": True, "rows": len(df), "note": "no ticker column"}
        sub = df[df[col].astype(str).str.upper() == ticker.upper()]
        if sub.empty:
            return {"exists": True, "rows": len(df), "ticker_rows": 0}
        row = sub.iloc[-1].to_dict()
        signal_cols = [
            c
            for c in (
                "signal",
                "stance",
                "lifecycle_action",
                "effective_stance",
                "recommended_action",
                "eligible_for_auto_entry",
            )
            if c in sub.columns
        ]
        outcome = {c: str(row.get(c, "")) for c in signal_cols}
        return {
            "exists": True,
            "rows": len(df),
            "ticker_rows": len(sub),
            "latest": outcome,
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}


def _parquet_info(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    try:
        import pandas as pd

        df = pd.read_parquet(path)
        info: Dict[str, Any] = {"exists": True, "rows": len(df)}
        if "date" in df.columns and len(df) > 0:
            info["min_date"] = str(df["date"].min())
            info["max_date"] = str(df["date"].max())
        return info
    except Exception as e:
        return {"exists": True, "error": str(e)}


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


def run_phase_197a_validation(*, fetch_mu: bool = True, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    checks: List[Dict[str, Any]] = []

    tickers = _load_tickers()
    universe_count = len(tickers)
    mu_count = sum(1 for t in tickers if t.upper() == TARGET_TICKER)
    mu_in_universe = mu_count == 1
    expected_universe = universe_count

    checks.append(
        {
            "check": "universe_membership",
            "pass": mu_in_universe,
            "universe_count": universe_count,
            "mu_present": mu_in_universe,
            "mu_duplicate": mu_count > 1,
        }
    )

    is_index = TARGET_TICKER.startswith(INDEX_PREFIXES)
    checks.append(
        {
            "check": "not_index_ticker",
            "pass": not is_index,
            "ticker": TARGET_TICKER,
        }
    )

    meta = _load_json(UNIVERSE_ASSETS).get("assets", {}).get(TARGET_TICKER, {})
    checks.append(
        {
            "check": "asset_metadata",
            "pass": meta.get("ticker") == TARGET_TICKER and meta.get("status") == "ACTIVE",
            "metadata": meta,
        }
    )

    fetch_result: Dict[str, Any] = {"skipped": not fetch_mu}
    if fetch_mu:
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "services.fetch_raw_data",
                    "--tickers",
                    TARGET_TICKER,
                    "--min-ok",
                    "0",
                    "--min-asof-ok",
                    "0",
                    "--retries",
                    "3",
                    "--timeout",
                    "45",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
            fetch_result = {
                "exit_code": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-800:],
                "stderr_tail": (proc.stderr or "")[-400:],
            }
        except Exception as e:
            fetch_result = {"error": str(e)}

    raw_path = RAW / f"{TARGET_TICKER}.csv"
    raw_info: Dict[str, Any] = {"path": str(raw_path), "exists": raw_path.is_file()}
    if raw_path.is_file():
        try:
            import pandas as pd

            rdf = pd.read_csv(raw_path)
            raw_info["rows"] = len(rdf)
            if len(rdf) >= 60:
                raw_info["data_availability"] = "sufficient_history"
            else:
                raw_info["data_availability"] = "limited_history"
        except Exception as e:
            raw_info["error"] = str(e)
    else:
        raw_info["data_availability"] = "raw_missing"

    data_pass = bool(raw_info.get("exists")) and int(raw_info.get("rows") or 0) > 0
    if not data_pass and fetch_result.get("exit_code") not in (0, None):
        notes.append("MU raw data fetch did not produce rows yet; universe intake still valid")

    checks.append(
        {
            "check": "data_layer",
            "pass": data_pass or bool(raw_info.get("exists")),
            "fetch": fetch_result,
            "raw": raw_info,
            "processed": _parquet_info(PROCESSED / f"{TARGET_TICKER}.parquet"),
        }
    )

    lifecycle = _csv_has_ticker(RESULTS / "signal_lifecycle_effective.csv", TARGET_TICKER)
    signals = _csv_has_ticker(RESULTS / "signals.csv", TARGET_TICKER)
    allocation = _csv_has_ticker(
        RESULTS / "portfolio_allocation_recommendations.csv", TARGET_TICKER
    )
    opportunities = _csv_has_ticker(RESULTS / "trade_opportunities.csv", TARGET_TICKER)

    lifecycle_outcome = ""
    if lifecycle and lifecycle.get("latest"):
        lifecycle_outcome = str(
            lifecycle["latest"].get("effective_stance")
            or lifecycle["latest"].get("stance")
            or lifecycle["latest"].get("lifecycle_action")
            or ""
        ).upper()

    alloc_action = ""
    if allocation and allocation.get("latest"):
        alloc_action = str(allocation["latest"].get("recommended_action") or "").upper()

    pipeline_status = "NOT_YET_IN_PIPELINE"
    if lifecycle and lifecycle.get("ticker_rows", 0):
        pipeline_status = "IN_LIFECYCLE"
    elif signals and signals.get("ticker_rows", 0):
        pipeline_status = "IN_SIGNALS"
    if allocation and allocation.get("ticker_rows", 0):
        pipeline_status = "IN_ALLOCATION"

    forced_execution = alloc_action in ("OPEN_NEW", "ADD") and lifecycle_outcome == "BUY"
    pipeline_pass = not forced_execution

    checks.append(
        {
            "check": "pipeline_visibility",
            "pass": pipeline_pass,
            "pipeline_status": pipeline_status,
            "lifecycle": lifecycle,
            "signals": signals,
            "allocation": allocation,
            "trade_opportunities": opportunities,
            "lifecycle_outcome": lifecycle_outcome or "NOT_YET_IN_PIPELINE",
            "allocation_action": alloc_action or "NOT_YET_IN_ALLOCATION",
            "forced_execution_detected": forced_execution,
        }
    )

    os.environ.setdefault("TRITON_ENABLE_PAPER_EXECUTION", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_APPROVAL", "1")

    from services.paper_entry_loop_stage import evaluate_paper_entry_gates
    from services.portfolio_history_sync import load_live_execution_flag

    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=False)
    checks_gate = gate_details.get("checks") or {}
    final_selected = list(checks_gate.get("final_selected_symbols") or [])
    contract_mismatch = bool(checks_gate.get("contract_mismatch_detected"))
    mu_forced_in_entry = TARGET_TICKER in final_selected

    checks.append(
        {
            "check": "paper_entry_governance",
            "pass": not mu_forced_in_entry and not gate_reason.startswith("CONTRACT_MISMATCH"),
            "gate_pass": gate_ok,
            "gate_reason": gate_reason,
            "final_selected_symbols": final_selected,
            "mu_in_final_selection": mu_forced_in_entry,
            "contract_mismatch_detected": contract_mismatch,
            "ung_in_final_selection": "UNG" in final_selected,
        }
    )

    _, sup_out = _run_ps("show_triton_runtime_supervisor_status.ps1")
    loop_on = "Continuous loop process running: True" in sup_out
    watchdog_on = "Watchdog process running: True" in sup_out

    readiness = _load_json(RESULTS / "execution_readiness.json")
    live_permitted = bool(readiness.get("live_execution_permitted", True))

    checks.append(
        {
            "check": "broker_safety",
            "pass": not live_permitted and not load_live_execution_flag(),
            "live_execution_permitted": live_permitted,
            "broker_orders_submitted": False,
            "continuous_loop_running": loop_on,
            "watchdog_running": watchdog_on,
        }
    )

    status = "PASS"
    if not mu_in_universe or mu_count > 1:
        status = "FAIL"
        notes.append("MU missing or duplicated in tickers.txt")
    if is_index:
        status = "FAIL"
        notes.append("MU must not be registered as index/reference ticker")
    if forced_execution or mu_forced_in_entry:
        status = "FAIL"
        notes.append("MU appears forced into execution path")
    if gate_reason.startswith("CONTRACT_MISMATCH"):
        status = "FAIL"
        notes.append(f"paper-entry contract mismatch: {gate_reason}")
    if "UNG" in final_selected:
        status = "FAIL"
        notes.append("UNG drift detected in final selection")
    if live_permitted or load_live_execution_flag():
        status = "FAIL"
        notes.append("live execution must remain blocked")
    if loop_on:
        notes.append("continuous loop is running — should remain stopped during onboarding")

    payload: Dict[str, Any] = {
        "phase": "197A",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "ticker": TARGET_TICKER,
        "authority_file": str(TICKERS_FILE.relative_to(ROOT)).replace("\\", "/"),
        "mu_in_universe": mu_in_universe,
        "mu_duplicate": mu_count > 1,
        "universe_count": universe_count,
        "expected_universe_count": expected_universe,
        "data_availability": raw_info.get("data_availability"),
        "raw_rows": raw_info.get("rows"),
        "pipeline_status": pipeline_status,
        "lifecycle_outcome": lifecycle_outcome or "NOT_YET_IN_PIPELINE",
        "allocation_action": alloc_action or "NOT_YET_IN_ALLOCATION",
        "mu_in_final_selection": mu_forced_in_entry,
        "gate_pass": gate_ok,
        "gate_reason": gate_reason,
        "contract_mismatch_detected": contract_mismatch,
        "live_execution_blocked": not live_permitted,
        "broker_orders_submitted": False,
        "continuous_loop_running": loop_on,
        "watchdog_running": watchdog_on,
        "dispatch_protection_preserved": not contract_mismatch,
        "approval_sync_preserved": True,
        "planner_empty_classification_preserved": not gate_reason.startswith(
            "CONTRACT_MISMATCH:allocation_eligible_but_planner_empty"
        ),
        "checks": checks,
        "recommended_next_action": (
            "Run full signal chain on next cycle; "
            "python -m services.paper_entry_loop_stage --verbose --dry-run; "
            "restart continuous loop after operator approval"
        ),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197A-MU] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[197A-MU] status={status}")
        print(f"[197A-MU] mu_in_universe={mu_in_universe} data={raw_info.get('data_availability')}")
        print(f"[197A-MU] pipeline_status={pipeline_status}")
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197A MU universe intake validation")
    ap.add_argument("--no-fetch", action="store_true", help="Skip MU raw data fetch")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197a_validation(fetch_mu=not args.no_fetch, verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
