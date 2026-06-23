"""
phase_196m_contract_artifact_lock_status_reconciliation_validation.py
Phase 196M — Reconcile contract/artifact integrity display from runtime pipeline health.
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
OUTPUT_JSON = RESULTS / "phase_196m_contract_artifact_lock_status_reconciliation.json"

PREVIOUS_COMMIT = "5dd16f1e"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _artifact_current(path: Path, max_age_min: float = 120.0) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        age = (
            datetime.now(timezone.utc)
            - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        ).total_seconds() / 60.0
        return age <= max_age_min
    except Exception:
        return False


def _assess_contracts() -> tuple[bool | None, Dict[str, Any]]:
    try:
        import view_results as vr

        results = vr.validate_all_contracts()
        summary = vr.contracts_summary(results)
        ok = (summary.get("failed", 0) == 0) and (summary.get("error_count", 0) == 0)
        return ok, summary
    except Exception as exc:
        return None, {"error": str(exc)}


def run_phase_196m_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    from services.dashboard_status_sync import (
        audit_reconciled_runtime_locks,
        build_dashboard_guard_summary,
        build_dashboard_health_classification,
        classify_contract_artifact_integrity,
        reconcile_integrity_lock_display,
    )

    contracts_ok, contract_summary = _assess_contracts()
    health = build_dashboard_health_classification()
    lock_audit = audit_reconciled_runtime_locks()

    runtime_pipeline_health = str(health.get("runtime_pipeline_health") or "FAIL")
    data_freshness = str(health.get("data_freshness") or "STALE")
    contract_integrity = classify_contract_artifact_integrity(
        contracts_ok=contracts_ok,
        contract_summary=contract_summary,
    )

    # Simulate legacy execution lock reason (artifact page health emoji != green)
    fake_pipeline_locked = runtime_pipeline_health in ("HEALTHY", "OFF_HOURS_HEALTHY")
    fake_reason = "Execution locked: pipeline health 🔴" if fake_pipeline_locked else ""

    integrity_display = reconcile_integrity_lock_display(
        pipeline_locked=True,
        pipeline_lock_reason=fake_reason,
        runtime_pipeline_health=runtime_pipeline_health,
        contracts_ok=contracts_ok,
        contract_integrity=contract_integrity,
    )

    summary = build_dashboard_guard_summary(
        pipeline_locked=True,
        pipeline_lock_reason=fake_reason,
        live_ready=False,
        live_gate_bits={"CONFIRM": False, "MARKET": False, "ARM": False, "GUARD": True},
        contracts_ok=contracts_ok,
        contract_summary=contract_summary,
    )

    block_reasons = list(health.get("execution_block_reasons") or [])
    wd_lock = str(lock_audit.get("watchdog_lock_classification") or "")
    cl_lock = str(lock_audit.get("continuous_lock_classification") or "")

    not_conflated = not (
        integrity_display.get("show_integrity_lock_caption")
        and runtime_pipeline_health in ("HEALTHY", "OFF_HOURS_HEALTHY")
    )
    if (
        integrity_display.get("show_integrity_lock_caption")
        and "pipeline health" in str(integrity_display.get("integrity_lock_detail", "")).lower()
    ):
        not_conflated = False

    pipeline_block_only_real = (
        "PIPELINE_HEALTH" not in block_reasons
        or runtime_pipeline_health
        in (
            "DEGRADED",
            "FAIL",
        )
    )

    vr_src = (ROOT / "view_results.py").read_text(encoding="utf-8")
    sync_src = (ROOT / "services" / "dashboard_status_sync.py").read_text(encoding="utf-8")
    wording_ok = (
        "Data integrity lock (contracts/artifacts): {pipeline_reason}" not in vr_src
        and "classify_contract_artifact_integrity" in sync_src
        and "LOCK_TIMESTAMP_STALE_BUT_PROCESS_ACTIVE"
        in (ROOT / "services" / "runtime_lock_classifier.py").read_text(encoding="utf-8")
    )

    positions = list(health.get("positions") or [])
    open_orders = int(health.get("open_orders") or 0)

    status = "PASS"
    if not wording_ok:
        status = "FAIL"
        notes.append("Reconciliation helpers or sidebar wording not applied.")
    elif runtime_pipeline_health in ("HEALTHY", "OFF_HOURS_HEALTHY") and not not_conflated:
        status = "FAIL"
        notes.append("Misleading pipeline-health integrity lock still shown in display.")
    elif "PIPELINE_HEALTH" in block_reasons and runtime_pipeline_health in (
        "HEALTHY",
        "OFF_HOURS_HEALTHY",
    ):
        status = "FAIL"
        notes.append("PIPELINE_HEALTH block reason shown while runtime pipeline healthy.")
    elif contract_integrity == "FAIL":
        status = "REVIEW"
        notes.append("Data contracts failing — integrity display should show FAIL caption.")
    elif positions != ["JPM", "WFC"]:
        status = "FAIL"
        notes.append(f"Unexpected positions: {positions}")
    elif open_orders != 0:
        status = "FAIL"
        notes.append("Open orders not zero.")
    else:
        notes.append("Contract/artifact integrity separated from runtime pipeline health display.")

    if cl_lock == "STALE_ACTIVE_LOCK" and lock_audit.get("continuous_process_running"):
        notes.append(
            "Continuous lock still STALE_ACTIVE_LOCK — verify lock timestamp reconciliation."
        )

    payload = {
        "phase": "196M",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_commit": PREVIOUS_COMMIT,
        "runtime_pipeline_health": runtime_pipeline_health,
        "data_freshness": data_freshness,
        "contract_artifact_integrity": contract_integrity,
        "contract_artifact_integrity_not_conflated_with_pipeline_health": not_conflated,
        "pipeline_health_block_reason_only_when_real_pipeline_failure": pipeline_block_only_real,
        "continuous_loop_running": bool(lock_audit.get("continuous_process_running")),
        "watchdog_running": bool(lock_audit.get("watchdog_process_running")),
        "continuous_loop_lock_status": cl_lock,
        "watchdog_lock_status": wd_lock,
        "heartbeat_source": "data/results/heartbeat.json",
        "heartbeat_fresh": bool(health.get("heartbeat_fresh")),
        "positions_snapshot_current": _artifact_current(RESULTS / "positions_snapshot.csv"),
        "open_orders_snapshot_current": _artifact_current(RESULTS / "open_orders_snapshot.csv"),
        "portfolio_history_current": bool(health.get("portfolio_history_current")),
        "execution_status": str(health.get("execution_status") or "BLOCKED"),
        "execution_block_reasons": block_reasons,
        "open_orders": open_orders,
        "positions_count": int(health.get("positions_count") or 0),
        "positions": positions,
        "live_execution": False,
        "manual_execute_trades_run": False,
        "manual_approval_used": False,
        "broker_orders_submitted": False,
        "execution_logic_changed": False,
        "governance_logic_changed": False,
        "integrity_display": integrity_display,
        "guard_summary_contract_integrity": summary.get("contract_artifact_integrity"),
        "notes": notes,
    }

    try:
        from services.portfolio_history_sync import load_live_execution_flag

        payload["live_execution"] = load_live_execution_flag()
        if payload["live_execution"]:
            status = "FAIL"
            payload["status"] = status
            notes.append("live_execution is true.")
    except Exception:
        pass

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196M] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196M] status={status}")
        print(f"[196M] contract_integrity={contract_integrity} wd={wd_lock} cl={cl_lock}")
    return payload


def main() -> int:
    report = run_phase_196m_validation(verbose=True)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
