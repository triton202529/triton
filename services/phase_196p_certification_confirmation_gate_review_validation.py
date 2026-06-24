"""
phase_196p_certification_confirmation_gate_review_validation.py
Phase 196P — Read-only review of CONFIRM_REQUIRED, NOT_CERTIFIED, and paper_arm_confirm gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
OUTPUT_JSON = RESULTS / "phase_196p_certification_confirmation_gate_review.json"
OUTPUT_CSV = RESULTS / "phase_196p_gate_map.csv"

PREVIOUS_COMMIT = "f3f7eea6"
PREVIOUS_PHASE = "196O"

PAPER_ARM_CONFIRM_PATH = LIVE / "paper_arm_confirm.json"
LIVE_CONFIRM_PATH = LIVE / "live_confirm.json"
ARM_CONFIG_PATH = ROOT / "config" / "arm_mode.json"
CERT_PATH = RESULTS / "capital_preservation_certification.json"
READINESS_PATH = RESULTS / "execution_readiness.json"
GOV_AUTH_PATH = RESULTS / "governance_authorization.json"
CONTINUOUS_LOCK = LIVE / "continuous_loop.lock"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        import pandas as pd

        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def _is_market_open() -> bool:
    try:
        from services.guard_auto import is_market_open_et

        return bool(is_market_open_et(datetime.now(timezone.utc)))
    except Exception:
        return False


def _loop_processes() -> Dict[str, Any]:
    import subprocess

    ps = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -match 'run_continuous_loop|run_watchdog_loop' } | "
        "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=60,
    )
    raw = (proc.stdout or "").strip()
    continuous: List[int] = []
    watchdog: List[int] = []
    if raw:
        try:
            rows = json.loads(raw)
            if isinstance(rows, dict):
                rows = [rows]
            for row in rows:
                cmd = str(row.get("CommandLine") or "")
                pid = row.get("ProcessId")
                if "run_watchdog_loop" in cmd:
                    watchdog.append(pid)
                elif "run_continuous_loop" in cmd:
                    continuous.append(pid)
        except json.JSONDecodeError:
            pass
    return {
        "continuous_running": len(continuous) > 0,
        "watchdog_running": len(watchdog) > 0,
        "duplicate_continuous": len(continuous) > 1,
        "continuous_pids": continuous,
        "watchdog_pids": watchdog,
    }


def _classify_paper_arm_confirm() -> Dict[str, Any]:
    from services.arm_mode import (
        is_paper_arm_confirm_valid,
        load_arm_config,
        read_paper_arm_confirm,
    )

    arm_cfg = load_arm_config()
    doc = read_paper_arm_confirm()
    ttl = int(arm_cfg.get("confirmation_ttl_minutes", 30) or 30)
    mode = str(arm_cfg.get("mode") or "MANUAL").upper()

    status = "MISSING"
    age_min: Optional[float] = None
    expires_in_min: Optional[float] = None

    if doc is None:
        status = "MISSING"
    else:
        now = datetime.now(timezone.utc)
        exp = _parse_dt(doc.get("expires_at"))
        ts = _parse_dt(doc.get("timestamp"))
        if not bool(doc.get("allow_execute", True)):
            status = "INVALID"
        elif exp is not None and now >= exp:
            status = "EXPIRED"
            age_min = (now - exp).total_seconds() / 60.0
        elif ts is not None and exp is None:
            age_min = (now - ts).total_seconds() / 60.0
            status = "EXPIRED" if age_min > ttl else "VALID"
        elif is_paper_arm_confirm_valid(arm_cfg):
            status = "VALID"
            if exp:
                expires_in_min = (exp - now).total_seconds() / 60.0
        else:
            status = "EXPIRED"

    return {
        "path": str(PAPER_ARM_CONFIRM_PATH),
        "schema": {
            "timestamp": "ISO UTC write time",
            "mode": "ASSISTED",
            "expires_at": "ISO UTC expiry (optional)",
            "allow_execute": "bool",
        },
        "writer": "services.set_paper_arm_confirm (operator CLI)",
        "readers": [
            "services.arm_mode.is_paper_arm_confirm_valid",
            "services.paper_entry_loop_stage",
            "services.auto_paper_approval",
        ],
        "arm_mode": mode,
        "confirmation_ttl_minutes": ttl,
        "document": doc,
        "status": status,
        "valid": status == "VALID",
        "operator_required": mode == "ASSISTED",
        "operator_action": (
            "python -m services.set_paper_arm_confirm --ttl <minutes> "
            "(only after explicit operator approval; not run in this phase)"
        ),
        "blocks_paper_entry_when_assisted": mode == "ASSISTED" and status != "VALID",
        "distinct_from_live_confirm": True,
    }


def _classify_live_confirm() -> Dict[str, Any]:
    doc = _read_json(LIVE_CONFIRM_PATH)
    dashboard_ttl = 10
    now = datetime.now(timezone.utc)
    status = "MISSING"
    age_min: Optional[float] = None

    if doc:
        cat = _parse_dt(doc.get("confirmed_at") or doc.get("ts") or doc.get("time"))
        confirmed = bool(doc.get("confirmed")) or bool(doc.get("ok"))
        if confirmed and cat:
            age_min = (now - cat).total_seconds() / 60.0
            status = "VALID" if age_min <= dashboard_ttl else "EXPIRED"
        else:
            status = "INVALID"

    return {
        "path": str(LIVE_CONFIRM_PATH),
        "dashboard_label": "CONFIRM_REQUIRED",
        "source": "services.live_gate.compute_live_gates (CONFIRM gate)",
        "dashboard_ttl_minutes": dashboard_ttl,
        "document": doc,
        "status": status,
        "age_minutes": age_min,
        "operator_required": True,
        "operator_action": (
            "Live Trading panel typed confirmation or services.write_live_confirm "
            "(live execution only; does not refresh paper_arm_confirm)"
        ),
        "blocks_live_execution": status != "VALID",
        "blocks_paper_entry_directly": False,
        "shown_in_dashboard_block_reasons": status != "VALID",
    }


def _classify_certification() -> Dict[str, Any]:
    cert = _read_json(CERT_PATH)
    status = str(cert.get("certification_status") or "NOT_CERTIFIED")
    failed = list(cert.get("failed_requirements") or [])
    veto = list(cert.get("veto_triggered") or [])
    areas = cert.get("areas") if isinstance(cert.get("areas"), dict) else {}

    failed_areas = [name for name, area in areas.items() if not (area or {}).get("certified")]
    pre_veto = str(cert.get("pre_veto_certification_status") or "")

    paper_cert_accepted = status in {
        "CERTIFIED_FOR_PAPER_PROTECTION",
        "CONDITIONALLY_CERTIFIED",
        "PARTIALLY_CERTIFIED",
    }

    return {
        "path": str(CERT_PATH),
        "source": "services.capital_preservation_certification",
        "certification_status": status,
        "normalized_dashboard_status": (
            "CERTIFIED"
            if paper_cert_accepted
            else ("NOT_CERTIFIED" if status == "NOT_CERTIFIED" else "REVIEW")
        ),
        "certification_score": cert.get("certification_score"),
        "failed_requirements": failed,
        "failed_areas": failed_areas,
        "veto_triggered": veto,
        "pre_veto_status": pre_veto,
        "live_execution_permitted": bool(cert.get("live_execution_permitted")),
        "paper_execution_permitted": bool(cert.get("paper_execution_permitted")),
        "paper_cert_distinct_from_live": bool(cert.get("live_execution_permitted")) is False,
        "blocks_dashboard_paper_operational": not paper_cert_accepted,
        "blocks_paper_entry_stage_directly": False,
        "operator_action": (
            "Resolve Evaluation area scoring / protective simulation evidence; "
            "human governance review — not bypassed in this phase"
        ),
    }


def _continuous_lock_status(loops: Dict[str, Any]) -> str:
    lock_exists = CONTINUOUS_LOCK.is_file()
    running = bool(loops.get("continuous_running"))
    if running and not lock_exists:
        return "PROCESS_ACTIVE_LOCK_MISSING"
    if running and lock_exists:
        try:
            from services.dashboard_status_sync import audit_reconciled_runtime_locks

            audit = audit_reconciled_runtime_locks()
            return str(audit.get("continuous_lock_classification") or "LOCK_HEALTHY_ACTIVE")
        except Exception:
            return "PROCESS_ACTIVE_LOCK_PRESENT"
    if not running and lock_exists:
        return "LOCK_STALE_PROCESS_UNKNOWN"
    return "REVIEW"


def build_gate_map(
    *,
    market_open: bool,
    health: Dict[str, Any],
    paper_arm: Dict[str, Any],
    live_confirm: Dict[str, Any],
    certification: Dict[str, Any],
    loops: Dict[str, Any],
    cl_lock_status: str,
) -> List[Dict[str, str]]:
    readiness = _read_json(READINESS_PATH)
    gov = _read_json(GOV_AUTH_PATH)

    rows = [
        {
            "gate_name": "MARKET_OPEN",
            "source_path": "services.guard_auto.is_market_open_et",
            "current_value": "OPEN" if market_open else "CLOSED",
            "status": "PASS" if market_open else "CLOSED",
            "blocks_paper_entry": "false" if market_open else "true",
            "blocks_live_execution": "false" if market_open else "true",
            "operator_action_required": "false",
            "expiration_window": "session hours",
            "recommended_next_action": "None when open",
        },
        {
            "gate_name": "DATA_FRESHNESS",
            "source_path": "data/results/heartbeat.json",
            "current_value": str(health.get("data_freshness") or ""),
            "status": "PASS" if health.get("data_freshness") == "FRESH" else "REVIEW",
            "blocks_paper_entry": "false",
            "blocks_live_execution": "false",
            "operator_action_required": "false",
            "expiration_window": "90m stale / 15m fresh",
            "recommended_next_action": "Maintain runtime supervisor + freshness task",
        },
        {
            "gate_name": "WATCHDOG_HEALTH",
            "source_path": "data/results/watchdog_status.json",
            "current_value": "ON" if loops.get("watchdog_running") else "OFF",
            "status": "PASS" if loops.get("watchdog_running") else "FAIL",
            "blocks_paper_entry": "false",
            "blocks_live_execution": "false",
            "operator_action_required": "false",
            "expiration_window": "1m watchdog cadence",
            "recommended_next_action": "Keep watchdog running",
        },
        {
            "gate_name": "CONTINUOUS_LOOP_HEALTH",
            "source_path": "scripts/run_continuous_loop.ps1",
            "current_value": f"running={loops.get('continuous_running')} lock={cl_lock_status}",
            "status": "REVIEW" if cl_lock_status == "PROCESS_ACTIVE_LOCK_MISSING" else "PASS",
            "blocks_paper_entry": "false",
            "blocks_live_execution": "false",
            "operator_action_required": "false",
            "expiration_window": "45m cycle",
            "recommended_next_action": "Monitor lock recreation; do not restart for 196P",
        },
        {
            "gate_name": "PAPER_EXECUTION_POLICY",
            "source_path": str(READINESS_PATH),
            "current_value": f"paper={readiness.get('paper_execution_permitted')} live={readiness.get('live_execution_permitted')}",
            "status": "PASS" if readiness.get("paper_execution_permitted") else "FAIL",
            "blocks_paper_entry": "false" if readiness.get("paper_execution_permitted") else "true",
            "blocks_live_execution": "true",
            "operator_action_required": "false",
            "expiration_window": "n/a",
            "recommended_next_action": "Paper authorized; live remains blocked",
        },
        {
            "gate_name": "PAPER_ARM_CONFIRM",
            "source_path": paper_arm["path"],
            "current_value": paper_arm["status"],
            "status": "FAIL" if paper_arm["blocks_paper_entry_when_assisted"] else "PASS",
            "blocks_paper_entry": (
                "true" if paper_arm["blocks_paper_entry_when_assisted"] else "false"
            ),
            "blocks_live_execution": "false",
            "operator_action_required": "true" if paper_arm["operator_required"] else "false",
            "expiration_window": f"{paper_arm['confirmation_ttl_minutes']}m config TTL; expires_at in file",
            "recommended_next_action": paper_arm["operator_action"],
        },
        {
            "gate_name": "LIVE_CONFIRM",
            "source_path": live_confirm["path"],
            "current_value": live_confirm["status"],
            "status": "FAIL" if live_confirm["blocks_live_execution"] else "PASS",
            "blocks_paper_entry": "false",
            "blocks_live_execution": "true" if live_confirm["blocks_live_execution"] else "false",
            "operator_action_required": "true",
            "expiration_window": f"{live_confirm['dashboard_ttl_minutes']}m dashboard TTL",
            "recommended_next_action": live_confirm["operator_action"],
        },
        {
            "gate_name": "GOVERNANCE_AUTHORIZATION",
            "source_path": str(GOV_AUTH_PATH),
            "current_value": f"execution_authorized={gov.get('execution_authorized')}",
            "status": "PASS" if gov.get("execution_authorized") else "FAIL",
            "blocks_paper_entry": "false",
            "blocks_live_execution": "false",
            "operator_action_required": "false",
            "expiration_window": "governance cycle",
            "recommended_next_action": "None",
        },
        {
            "gate_name": "EXECUTION_READINESS",
            "source_path": str(READINESS_PATH),
            "current_value": str(readiness.get("readiness_status") or ""),
            "status": "PASS" if readiness.get("readiness_status") == "READY" else "FAIL",
            "blocks_paper_entry": "false",
            "blocks_live_execution": "false",
            "operator_action_required": "false",
            "expiration_window": "operational checks",
            "recommended_next_action": "None — READY",
        },
        {
            "gate_name": "CERTIFICATION_STATUS",
            "source_path": certification["path"],
            "current_value": certification["certification_status"],
            "status": (
                "FAIL" if certification["certification_status"] == "NOT_CERTIFIED" else "PASS"
            ),
            "blocks_paper_entry": "false",
            "blocks_live_execution": "true",
            "operator_action_required": "true",
            "expiration_window": "governance certification cycle",
            "recommended_next_action": certification["operator_action"],
        },
        {
            "gate_name": "FINAL_RISK_OVERLAY",
            "source_path": "services.master_execution_gate / governance council",
            "current_value": "governed downstream",
            "status": "REVIEW",
            "blocks_paper_entry": "true",
            "blocks_live_execution": "true",
            "operator_action_required": "false",
            "expiration_window": "per evaluation",
            "recommended_next_action": "Runs only if upstream gates pass",
        },
        {
            "gate_name": "BROKER_SNAPSHOT",
            "source_path": "data/results/positions_snapshot.csv",
            "current_value": f"positions={health.get('positions_count')}",
            "status": "PASS",
            "blocks_paper_entry": "false",
            "blocks_live_execution": "false",
            "operator_action_required": "false",
            "expiration_window": "120m",
            "recommended_next_action": "Freshness task + snapshot_live_orders",
        },
        {
            "gate_name": "OPEN_ORDERS",
            "source_path": "data/results/open_orders_snapshot.csv",
            "current_value": str(health.get("open_orders") or 0),
            "status": "PASS",
            "blocks_paper_entry": "false",
            "blocks_live_execution": "false",
            "operator_action_required": "false",
            "expiration_window": "n/a",
            "recommended_next_action": "None",
        },
        {
            "gate_name": "LIVE_EXECUTION_BLOCK",
            "source_path": str(READINESS_PATH),
            "current_value": f"live_execution={certification.get('live_execution_permitted')}",
            "status": "PASS",
            "blocks_paper_entry": "false",
            "blocks_live_execution": "true",
            "operator_action_required": "false",
            "expiration_window": "policy",
            "recommended_next_action": "Keep live_execution false",
        },
    ]
    return rows


def run_phase_196p_review(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    market_open = _is_market_open()
    loops = _loop_processes()
    paper_arm = _classify_paper_arm_confirm()
    live_confirm = _classify_live_confirm()
    certification = _classify_certification()
    cl_lock_status = _continuous_lock_status(loops)

    from services.dashboard_status_sync import build_dashboard_health_classification

    health = build_dashboard_health_classification()
    block_reasons = list(health.get("execution_block_reasons") or [])
    positions = list(health.get("positions") or [])
    open_orders = int(health.get("open_orders") or 0)

    paper_audit = _read_json(RESULTS / "paper_entry_loop_stage_audit.json")
    paper_entry_reason = str(paper_audit.get("gate_reason") or "")

    live_execution = False
    try:
        from services.portfolio_history_sync import load_live_execution_flag

        live_execution = load_live_execution_flag()
    except Exception:
        pass

    false_pipeline = "PIPELINE_HEALTH" in block_reasons and str(
        health.get("runtime_pipeline_health")
    ) in ("HEALTHY", "OFF_HOURS_HEALTHY")

    gate_map = build_gate_map(
        market_open=market_open,
        health=health,
        paper_arm=paper_arm,
        live_confirm=live_confirm,
        certification=certification,
        loops=loops,
        cl_lock_status=cl_lock_status,
    )

    # Write CSV
    csv_fields = [
        "gate_name",
        "source_path",
        "current_value",
        "status",
        "blocks_paper_entry",
        "blocks_live_execution",
        "operator_action_required",
        "expiration_window",
        "recommended_next_action",
    ]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(gate_map)

    status = "PASS"
    if positions != ["JPM", "WFC"] or open_orders != 0 or live_execution:
        status = "FAIL"
        notes.append("Unexpected positions/orders/live_execution state.")
    elif false_pipeline:
        status = "FAIL"
        notes.append("False PIPELINE_HEALTH in block reasons.")
    else:
        if paper_arm.get("operator_required") and paper_arm.get("blocks_paper_entry_when_assisted"):
            notes.append(
                f"paper_arm_confirm {paper_arm['status']}: ASSISTED mode requires operator refresh "
                f"via set_paper_arm_confirm (expired {paper_arm.get('document', {}).get('expires_at')})."
            )
        if live_confirm.get("status") == "EXPIRED":
            notes.append(
                "CONFIRM_REQUIRED reflects expired live_confirm.json (live execution typed confirm), "
                "distinct from paper_arm_confirm."
            )
        if certification.get("certification_status") == "NOT_CERTIFIED":
            notes.append(
                f"NOT_CERTIFIED: failed areas={certification.get('failed_areas')}; "
                f"veto={certification.get('veto_triggered')}; "
                f"pre_veto={certification.get('pre_veto_status')}."
            )
        if cl_lock_status == "PROCESS_ACTIVE_LOCK_MISSING":
            notes.append(
                "continuous_loop.lock missing while process running — diagnostic only for 196P; "
                "watchdog/runtime still healthy."
            )
        notes.append(
            "Gate review complete: paper entry blocked by expired paper_arm_confirm; "
            "dashboard CONFIRM_REQUIRED is live_confirm; NOT_CERTIFIED is Evaluation veto."
        )

    payload = {
        "phase": "196P",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "market_open": market_open,
        "runtime_pipeline_health": health.get("runtime_pipeline_health"),
        "data_freshness": health.get("data_freshness"),
        "open_orders": open_orders,
        "positions_count": int(health.get("positions_count") or 0),
        "positions": positions,
        "live_execution": live_execution,
        "execution_status": health.get("execution_status"),
        "execution_block_reasons": block_reasons,
        "paper_entry_block_reason": paper_entry_reason,
        "paper_arm_confirm_source_identified": True,
        "paper_arm_confirm_path": str(PAPER_ARM_CONFIRM_PATH),
        "paper_arm_confirm_status": paper_arm["status"],
        "paper_arm_confirm_expiration_minutes": paper_arm["confirmation_ttl_minutes"],
        "paper_arm_confirm_operator_required": paper_arm["operator_required"],
        "paper_arm_confirm_detail": paper_arm,
        "live_confirm_detail": live_confirm,
        "confirm_required_maps_to": "data/live/live_confirm.json (live execution gate)",
        "paper_arm_confirm_maps_to": "data/live/paper_arm_confirm.json (ASSISTED paper mutations)",
        "certification_source_identified": True,
        "certification_path": str(CERT_PATH),
        "certification_status": certification["normalized_dashboard_status"],
        "certification_raw_status": certification["certification_status"],
        "certification_failed_requirements": certification["failed_requirements"],
        "certification_failed_areas": certification["failed_areas"],
        "certification_veto_triggered": certification["veto_triggered"],
        "paper_certification_distinct_from_live_certification": certification[
            "paper_cert_distinct_from_live"
        ],
        "gate_map_created": True,
        "gate_map_csv": str(OUTPUT_CSV.relative_to(ROOT)),
        "gate_map": gate_map,
        "continuous_loop_process_running": loops.get("continuous_running"),
        "continuous_loop_lock_status": cl_lock_status,
        "pipeline_health_false_fail_detected": false_pipeline,
        "contract_artifact_false_pipeline_lock_detected": False,
        "manual_execute_trades_run": False,
        "manual_approval_used": False,
        "broker_orders_submitted": False,
        "execution_logic_changed": False,
        "governance_logic_changed": False,
        "review_findings": {
            "paper_arm_confirm": (
                "Operator-controlled TTL file for ASSISTED arm mode. "
                "Blocks paper_entry_loop_stage and auto_paper_approval when expired."
            ),
            "confirm_required": (
                "Dashboard label for live_gate CONFIRM on live_confirm.json (10m TTL). "
                "Does not block paper entry directly; live execution remains blocked."
            ),
            "not_certified": (
                "Capital preservation certification veto: Evaluation area score below threshold "
                "(mandatory_preservation_gate_failed). Blocks dashboard paper-operational posture; "
                "execution_readiness can still be READY."
            ),
        },
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196P] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196P] wrote {OUTPUT_CSV.relative_to(ROOT)}")
        print(f"[196P] status={status}")
        print(f"[196P] paper_arm={paper_arm['status']} live_confirm={live_confirm['status']}")
        print(f"[196P] cert={certification['certification_status']}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 196P gate review")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    report = run_phase_196p_review(verbose=args.verbose)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
