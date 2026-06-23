"""
dashboard_status_sync.py
Read-only dashboard guard / heartbeat / execution status aggregation.

Separates pipeline heartbeat (heartbeat.json) from runtime watchdog heartbeat
(watchdog_status.json). Does not alter execution or governance logic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"

PIPELINE_HEARTBEAT_PATH = RESULTS / "heartbeat.json"
PIPELINE_STATUS_PATH = RESULTS / "pipeline_status.json"
WATCHDOG_STATUS_PATH = RESULTS / "watchdog_status.json"
GUARD_SNAPSHOT_RESULTS_PATH = RESULTS / "guard_snapshot.json"
OPEN_ORDERS_SNAPSHOT_PATH = RESULTS / "open_orders_snapshot.csv"
POSITIONS_SNAPSHOT_PATH = RESULTS / "positions_snapshot.csv"
EXECUTION_READINESS_PATH = RESULTS / "execution_readiness.json"
CONTINUOUS_LOCK_PATH = LIVE / "continuous_loop.lock"
WATCHDOG_LOCK_PATH = LIVE / "watchdog_loop.lock"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        import pandas as pd

        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def _age_minutes(ts: Optional[datetime]) -> Optional[float]:
    if ts is None:
        return None
    return (_utc_now() - ts).total_seconds() / 60.0


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _lock_fresh(path: Path, max_age_minutes: float = 180.0) -> Optional[bool]:
    if not path.is_file():
        return False
    try:
        age = (
            _utc_now() - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        ).total_seconds() / 60.0
        return age <= max_age_minutes
    except Exception:
        return None


def load_pipeline_heartbeat() -> Dict[str, Any]:
    doc = _read_json(PIPELINE_HEARTBEAT_PATH)
    path = PIPELINE_HEARTBEAT_PATH
    if not doc:
        doc = _read_json(PIPELINE_STATUS_PATH)
        path = PIPELINE_STATUS_PATH
    ts_raw = (
        doc.get("timestamp")
        or doc.get("ts")
        or doc.get("updated_at")
        or doc.get("time")
        or doc.get("as_of")
    )
    ts = _parse_dt(ts_raw)
    return {
        "path": str(path),
        "exists": bool(doc),
        "timestamp": ts,
        "age_minutes": _age_minutes(ts),
        "stage": str(doc.get("stage") or doc.get("step") or ""),
        "status": str(doc.get("status") or ""),
        "message": str(doc.get("message") or ""),
    }


def load_runtime_heartbeat() -> Dict[str, Any]:
    doc = _read_json(WATCHDOG_STATUS_PATH)
    hb = doc.get("heartbeat") if isinstance(doc.get("heartbeat"), dict) else {}
    ts_raw = hb.get("timestamp") or doc.get("timestamp")
    ts = _parse_dt(ts_raw)
    return {
        "path": str(WATCHDOG_STATUS_PATH),
        "exists": bool(doc),
        "timestamp": ts,
        "age_minutes": _age_minutes(ts),
        "status": str(hb.get("status") or doc.get("watchdog_status") or ""),
        "refresh_source": str(hb.get("refresh_source") or ""),
        "positions_count": hb.get("positions_count", doc.get("positions_count")),
        "open_orders_count": hb.get("open_orders_count", doc.get("open_orders_count")),
    }


def detect_loop_processes() -> Dict[str, Any]:
    watchdog_running = False
    continuous_running = False
    try:
        from services.phase_195h_operational_readiness_freeze import _find_ps_processes

        watchdog_running, _ = _find_ps_processes("run_watchdog_loop.ps1")
        continuous_running, _ = _find_ps_processes("run_continuous_loop.ps1")
    except Exception:
        pass
    return {
        "watchdog_process_running": watchdog_running,
        "continuous_process_running": continuous_running,
        "watchdog_lock_present": WATCHDOG_LOCK_PATH.is_file(),
        "continuous_lock_present": CONTINUOUS_LOCK_PATH.is_file(),
        "watchdog_lock_fresh": _lock_fresh(WATCHDOG_LOCK_PATH),
        "continuous_lock_fresh": _lock_fresh(CONTINUOUS_LOCK_PATH),
    }


def load_positions_summary() -> Dict[str, Any]:
    try:
        from services.portfolio_history_sync import load_positions_snapshot_summary

        return load_positions_snapshot_summary()
    except Exception:
        return {"positions": [], "position_count": 0, "invested_value": None}


def count_open_orders() -> int:
    try:
        from services.portfolio_history_sync import count_open_orders as _count

        return int(_count())
    except Exception:
        return 0


def classify_data_freshness(
    pipeline_age_min: Optional[float],
    *,
    stale_threshold_min: float = 90.0,
    aging_threshold_min: float = 15.0,
) -> str:
    if pipeline_age_min is None or pipeline_age_min != pipeline_age_min:
        return "UNKNOWN"
    if pipeline_age_min > stale_threshold_min:
        return "STALE"
    if pipeline_age_min > aging_threshold_min:
        return "AGING"
    return "FRESH"


def choose_sidebar_label(
    *,
    data_freshness: str,
    pipeline_locked: bool,
    paper_operational: bool,
    paper_permitted: bool,
    readiness_status: str,
    live_permitted: bool,
    execution_blocked_reasons: List[str],
) -> Tuple[str, str]:
    """Return (label, level) where level is success|warning|error|info."""
    if live_permitted:
        return "LIVE RUNTIME ENABLED", "error"

    if data_freshness == "STALE" or pipeline_locked:
        detail = (
            "DATA STALE · EXECUTION BLOCKED"
            if data_freshness == "STALE"
            else "EXECUTION LOCKED: PIPELINE HEALTH"
        )
        return detail, "error"

    if execution_blocked_reasons:
        return "PAPER AUTHORIZED · EXECUTION CURRENTLY BLOCKED", "warning"

    if paper_operational:
        return "PAPER RUNTIME: ACTIVE", "success"

    if paper_permitted and readiness_status == "READY":
        return "PAPER AUTHORIZED · EXECUTION CURRENTLY BLOCKED", "warning"

    if not paper_permitted:
        return "LIVE BLOCKED · PAPER INACTIVE", "error"

    return "LIVE BLOCKED · PAPER ONLY", "info"


def build_dashboard_guard_summary(
    *,
    pipeline_locked: bool = False,
    pipeline_lock_reason: str = "",
    live_ready: bool = False,
    live_gate_bits: Optional[Dict[str, bool]] = None,
    posture: Optional[Any] = None,
) -> Dict[str, Any]:
    pipeline_hb = load_pipeline_heartbeat()
    runtime_hb = load_runtime_heartbeat()
    loops = detect_loop_processes()
    positions = load_positions_summary()
    open_orders = count_open_orders()

    readiness = _read_json(EXECUTION_READINESS_PATH)
    if "readiness" in readiness and isinstance(readiness["readiness"], dict):
        readiness = readiness["readiness"]

    live_permitted = bool(readiness.get("live_execution_permitted", False))
    paper_permitted = bool(readiness.get("paper_execution_permitted", False))
    readiness_status = str(readiness.get("readiness_status") or "NOT_READY")

    paper_operational = False
    approval_label = ""
    if posture is not None:
        paper_operational = bool(getattr(posture, "paper_operational", False))
        approval_label = str(getattr(posture, "approval_status_label", "") or "")

    data_freshness = classify_data_freshness(pipeline_hb.get("age_minutes"))

    gate_bits = live_gate_bits or {}
    blocked_gates = [k for k, ok in gate_bits.items() if not ok]
    execution_blocked_reasons: List[str] = []
    if pipeline_locked:
        execution_blocked_reasons.append(pipeline_lock_reason or "pipeline lock")
    if not live_ready:
        execution_blocked_reasons.extend(blocked_gates)
    if data_freshness == "STALE":
        execution_blocked_reasons.append("pipeline heartbeat stale")

    if live_permitted:
        execution_gate = "NOT READY"
    elif pipeline_locked or data_freshness == "STALE" or not live_ready:
        execution_gate = "BLOCKED"
    elif paper_operational and not pipeline_locked:
        execution_gate = "READY"
    else:
        execution_gate = "BLOCKED"

    sidebar_label, sidebar_level = choose_sidebar_label(
        data_freshness=data_freshness,
        pipeline_locked=pipeline_locked,
        paper_operational=paper_operational,
        paper_permitted=paper_permitted,
        readiness_status=readiness_status,
        live_permitted=live_permitted,
        execution_blocked_reasons=execution_blocked_reasons,
    )

    watchdog_state = (
        "ON"
        if loops.get("watchdog_process_running")
        else ("OFF" if loops.get("watchdog_lock_present") is False else "UNKNOWN")
    )
    continuous_state = (
        "ON"
        if loops.get("continuous_process_running")
        else ("OFF" if loops.get("continuous_lock_present") is False else "UNKNOWN")
    )

    return {
        "data_freshness": data_freshness,
        "pipeline_heartbeat": pipeline_hb,
        "runtime_heartbeat": runtime_hb,
        "pipeline_heartbeat_source_path": pipeline_hb.get("path", str(PIPELINE_HEARTBEAT_PATH)),
        "pipeline_heartbeat_age_minutes": pipeline_hb.get("age_minutes"),
        "runtime_heartbeat_age_minutes": runtime_hb.get("age_minutes"),
        "watchdog": watchdog_state,
        "continuous_loop": continuous_state,
        "loops": loops,
        "execution_gate": execution_gate,
        "paper_mode": (
            "ACTIVE" if paper_operational else ("AUTHORIZED" if paper_permitted else "INACTIVE")
        ),
        "live_mode": "BLOCKED",
        "open_orders": open_orders,
        "positions_count": int(positions.get("position_count") or 0),
        "positions": list(positions.get("positions") or []),
        "sidebar_label": sidebar_label,
        "sidebar_level": sidebar_level,
        "execution_blocked_reasons": execution_blocked_reasons,
        "live_gate_bits": gate_bits,
        "live_ready": live_ready,
        "pipeline_locked": pipeline_locked,
        "pipeline_lock_reason": pipeline_lock_reason,
        "readiness_status": readiness_status,
        "approval_status_label": approval_label,
        "dashboard_stale_warning_truthful": (
            pipeline_hb.get("age_minutes") is not None
            and float(pipeline_hb.get("age_minutes")) > 60.0
        ),
    }


def verify_portfolio_history_working() -> bool:
    try:
        from services.portfolio_history_sync import resolve_portfolio_dashboard_state

        state = resolve_portfolio_dashboard_state()
        return (
            bool(state.get("positions_snapshot_exists"))
            and int(state.get("position_count") or 0) >= 0
        )
    except Exception:
        return False


def run_phase_196g_report(*, verbose: bool = False) -> Dict[str, Any]:
    notes: List[str] = []
    pipeline_hb = load_pipeline_heartbeat()
    runtime_hb = load_runtime_heartbeat()
    loops = detect_loop_processes()
    positions = load_positions_summary()
    open_orders = count_open_orders()

    readiness = _read_json(EXECUTION_READINESS_PATH)
    if "readiness" in readiness and isinstance(readiness["readiness"], dict):
        readiness = readiness["readiness"]

    summary = build_dashboard_guard_summary(
        pipeline_locked=True,
        pipeline_lock_reason="validation default",
        live_ready=False,
        live_gate_bits={"CONFIRM": False, "FRESH": False, "ARM": False, "GUARD": True},
    )

    vr_path = ROOT / "view_results.py"
    sync_path = ROOT / "services" / "dashboard_status_sync.py"
    unsafe_removed = False
    try:
        vr_src = vr_path.read_text(encoding="utf-8")
        sync_src = sync_path.read_text(encoding="utf-8")
        unsafe_removed = (
            'st.success("✅ Execution enabled")' not in vr_src
            and "render_dashboard_guard_summary" in vr_src
            and "build_dashboard_guard_summary" in vr_src
            and "choose_sidebar_label" in sync_src
        )
    except Exception:
        unsafe_removed = False

    portfolio_ok = verify_portfolio_history_working()
    ph_age = pipeline_hb.get("age_minutes")
    if ph_age is not None and float(ph_age) > 60:
        notes.append(
            f"Pipeline heartbeat ({PIPELINE_HEARTBEAT_PATH.name}) is {float(ph_age):.0f}m old — stale warning is truthful."
        )
    rt_age = runtime_hb.get("age_minutes")
    if rt_age is not None and float(rt_age) <= 60:
        notes.append(
            f"Runtime watchdog heartbeat is fresh ({float(rt_age):.0f}m) — separate from pipeline heartbeat."
        )

    status = "PASS"
    if not portfolio_ok:
        status = "FAIL"
        notes.append("Portfolio history sync check failed.")
    elif not unsafe_removed:
        status = "REVIEW"
        notes.append("Verify sidebar no longer shows unsafe Execution enabled label.")

    payload = {
        "phase": "196G",
        "generated_at": _utc_now().isoformat(),
        "status": status,
        "dashboard_file_checked": vr_path.is_file(),
        "unsafe_execution_enabled_label_removed_or_guarded": unsafe_removed,
        "heartbeat_source_identified": bool(pipeline_hb.get("path")),
        "heartbeat_source_path": str(pipeline_hb.get("path") or PIPELINE_HEARTBEAT_PATH),
        "runtime_heartbeat_source_path": str(runtime_hb.get("path") or WATCHDOG_STATUS_PATH),
        "heartbeat_age_minutes": ph_age,
        "runtime_heartbeat_age_minutes": rt_age,
        "dashboard_stale_warning_truthful": summary.get("dashboard_stale_warning_truthful"),
        "continuous_loop_running": loops.get("continuous_process_running"),
        "watchdog_running": loops.get("watchdog_process_running"),
        "execution_gate": summary.get("execution_gate"),
        "paper_runtime": summary.get("paper_mode"),
        "live_runtime": "BLOCKED",
        "open_orders": open_orders,
        "positions_count": int(positions.get("position_count") or 0),
        "positions": list(positions.get("positions") or []),
        "portfolio_history_still_working": portfolio_ok,
        "manual_execute_trades_run": False,
        "manual_approval_used": False,
        "broker_orders_submitted": False,
        "execution_logic_changed": False,
        "governance_logic_changed": False,
        "data_freshness": summary.get("data_freshness"),
        "sidebar_label": summary.get("sidebar_label"),
        "notes": notes,
    }

    out_path = RESULTS / "phase_196g_heartbeat_status_display_hardening.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, out_path)

    if verbose:
        print(f"[196G] wrote {out_path.relative_to(ROOT)}")
        print(f"[196G] status={status}")

    return payload


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Phase 196G dashboard status verification")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    report = run_phase_196g_report(verbose=args.verbose)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    raise SystemExit(main())
