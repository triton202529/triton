"""
phase_196l_pipeline_health_execution_status_split_validation.py
Phase 196L — Split runtime pipeline health from execution certification status.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUTPUT_JSON = RESULTS / "phase_196l_pipeline_health_execution_status_split.json"

PREVIOUS_COMMIT = "e09393f8"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _portfolio_state() -> Dict[str, Any]:
    positions: List[str] = []
    open_orders = 0
    live_execution = False
    try:
        from services.dashboard_status_sync import (
            build_dashboard_health_classification,
            count_open_orders,
            detect_loop_processes,
            load_positions_summary,
        )
        from services.portfolio_history_sync import load_live_execution_flag

        pos = load_positions_summary()
        positions = list(pos.get("positions") or [])
        open_orders = int(count_open_orders())
        live_execution = load_live_execution_flag()
        loops = detect_loop_processes()
        health = build_dashboard_health_classification()
    except Exception as exc:
        return {
            "positions": positions,
            "positions_count": len(positions),
            "open_orders": open_orders,
            "live_execution": live_execution,
            "error": str(exc),
            "loops": {},
            "health": {},
        }
    return {
        "positions": positions,
        "positions_count": len(positions),
        "open_orders": open_orders,
        "live_execution": live_execution,
        "loops": loops,
        "health": health,
    }


def _view_results_uses_split() -> bool:
    try:
        src = (ROOT / "view_results.py").read_text(encoding="utf-8")
        sync = (ROOT / "services" / "dashboard_status_sync.py").read_text(encoding="utf-8")
    except Exception:
        return False
    return (
        "build_dashboard_health_classification" in sync
        and "runtime_pipeline_health" in sync
        and "EXECUTION LOCKED: PIPELINE HEALTH" not in sync
        and "Pipeline health looks degraded/missing" not in src
        and "build_dashboard_health_classification" in src
    )


def run_phase_196l_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    state = _portfolio_state()
    health = state.get("health") or {}
    loops = state.get("loops") or {}

    runtime_pipeline_health = str(health.get("runtime_pipeline_health") or "FAIL")
    data_freshness = str(health.get("data_freshness") or "STALE")
    execution_status = str(health.get("execution_status") or "BLOCKED")
    block_reasons = list(health.get("execution_block_reasons") or [])
    market_status = str(health.get("market_status") or "CLOSED")

    pipeline_fail_only_runtime = runtime_pipeline_health not in ("HEALTHY", "OFF_HOURS_HEALTHY")
    cert_not_pipeline = "NOT_CERTIFIED" in block_reasons and "PIPELINE_HEALTH" not in block_reasons
    pipeline_health_not_for_cert = (
        "PIPELINE_HEALTH" not in block_reasons
        or runtime_pipeline_health in ("DEGRADED", "FAIL")
        or data_freshness == "STALE"
    )

    status = "PASS"
    if not _view_results_uses_split():
        status = "FAIL"
        notes.append("Dashboard split helpers or wording not applied.")
    elif runtime_pipeline_health in ("FAIL", "DEGRADED") and data_freshness == "FRESH":
        status = "REVIEW"
        notes.append("Runtime pipeline degraded while data freshness is FRESH — verify thresholds.")
    elif state.get("positions") != ["JPM", "WFC"]:
        status = "FAIL"
        notes.append(f"Unexpected positions: {state.get('positions')}")
    elif int(state.get("open_orders") or 0) != 0:
        status = "FAIL"
        notes.append("Open orders not zero.")
    elif state.get("live_execution"):
        status = "FAIL"
        notes.append("live_execution is true.")
    elif (
        runtime_pipeline_health in ("HEALTHY", "OFF_HOURS_HEALTHY")
        and "PIPELINE_HEALTH" in block_reasons
    ):
        status = "FAIL"
        notes.append("PIPELINE_HEALTH block reason shown while runtime pipeline is healthy.")
    elif market_status == "CLOSED" and "MARKET_CLOSED" not in block_reasons:
        status = "REVIEW"
        notes.append("Market closed but MARKET_CLOSED not in block reasons.")
    elif "NOT_CERTIFIED" not in block_reasons:
        status = "REVIEW"
        notes.append("Expected NOT_CERTIFIED in block reasons for current certification state.")
    else:
        notes.append("Pipeline health separated from execution/certification block display.")

    payload = {
        "phase": "196L",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_commit": PREVIOUS_COMMIT,
        "runtime_pipeline_health": runtime_pipeline_health,
        "data_freshness": data_freshness,
        "pipeline_fail_shown_only_for_real_runtime_failure": pipeline_fail_only_runtime is False
        or runtime_pipeline_health in ("DEGRADED", "FAIL"),
        "pipeline_health_not_used_for_certification_block": cert_not_pipeline,
        "execution_status": execution_status,
        "execution_block_reasons": block_reasons,
        "market_status": market_status,
        "paper_authorized": bool(health.get("paper_authorized")),
        "live_execution": state.get("live_execution"),
        "watchdog_running": bool(loops.get("watchdog_process_running")),
        "continuous_loop_running": bool(loops.get("continuous_process_running")),
        "heartbeat_source": "data/results/heartbeat.json",
        "heartbeat_fresh": bool(health.get("heartbeat_fresh")),
        "open_orders": state.get("open_orders"),
        "positions_count": state.get("positions_count"),
        "positions": state.get("positions"),
        "portfolio_history_current": bool(health.get("portfolio_history_current")),
        "manual_execute_trades_run": False,
        "manual_approval_used": False,
        "broker_orders_submitted": False,
        "execution_logic_changed": False,
        "governance_logic_changed": False,
        "health_classification": health,
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196L] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196L] status={status}")
        print(f"[196L] runtime_pipeline_health={runtime_pipeline_health}")
        print(f"[196L] execution_block_reasons={block_reasons}")
    return payload


def main() -> int:
    report = run_phase_196l_validation(verbose=True)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
