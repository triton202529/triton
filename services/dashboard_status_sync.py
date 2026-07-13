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
PORTFOLIO_HISTORY_PATH = RESULTS / "portfolio_history.csv"
EXECUTION_READINESS_PATH = RESULTS / "execution_readiness.json"
CERTIFICATION_PATH = RESULTS / "capital_preservation_certification.json"
CONTINUOUS_LOCK_PATH = LIVE / "continuous_loop.lock"
WATCHDOG_LOCK_PATH = LIVE / "watchdog_loop.lock"

PAPER_CERTIFICATION_STATUSES = frozenset(
    {
        "CERTIFIED_FOR_PAPER_PROTECTION",
        "CONDITIONALLY_CERTIFIED",
        "PARTIALLY_CERTIFIED",
    }
)

RUNTIME_PIPELINE_FAIL_MINUTES = 360.0
RUNTIME_PIPELINE_DEGRADED_MINUTES = 90.0
BROKER_SNAPSHOT_FRESH_MINUTES = 120.0


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
    from services.runtime_process_detector import detect_runtime_loop_processes

    detected = detect_runtime_loop_processes()
    return {
        "watchdog_process_running": detected.get("watchdog_loop_running"),
        "continuous_process_running": detected.get("continuous_loop_running"),
        "watchdog_lock_present": WATCHDOG_LOCK_PATH.is_file(),
        "continuous_lock_present": CONTINUOUS_LOCK_PATH.is_file(),
        "watchdog_lock_fresh": _lock_fresh(WATCHDOG_LOCK_PATH),
        "continuous_lock_fresh": _lock_fresh(CONTINUOUS_LOCK_PATH),
        "detected_watchdog_process_ids": detected.get("detected_watchdog_process_ids"),
        "detected_continuous_process_ids": detected.get("detected_continuous_process_ids"),
        "process_detection_confidence": detected.get("process_detection_confidence"),
        "process_detection_warnings": detected.get("process_detection_warnings"),
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
    market_open: bool = True,
) -> str:
    if pipeline_age_min is None or pipeline_age_min != pipeline_age_min:
        return "STALE"
    if pipeline_age_min > stale_threshold_min:
        return "STALE"
    if pipeline_age_min > aging_threshold_min:
        return "AGING_AFTER_CLOSE" if not market_open else "FRESH"
    return "FRESH"


def _broker_snapshots_ok() -> bool:
    now = _utc_now()
    for path in (POSITIONS_SNAPSHOT_PATH, OPEN_ORDERS_SNAPSHOT_PATH):
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            age = (
                now - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            ).total_seconds() / 60.0
            if age > BROKER_SNAPSHOT_FRESH_MINUTES:
                return False
        except Exception:
            return False
    return True


def _portfolio_history_ok() -> bool:
    if not PORTFOLIO_HISTORY_PATH.is_file() or PORTFOLIO_HISTORY_PATH.stat().st_size == 0:
        return False
    try:
        age = (
            _utc_now()
            - datetime.fromtimestamp(PORTFOLIO_HISTORY_PATH.stat().st_mtime, tz=timezone.utc)
        ).total_seconds() / 60.0
        return age <= RUNTIME_PIPELINE_DEGRADED_MINUTES * 4
    except Exception:
        return False


def _is_market_open() -> bool:
    try:
        from services.guard_auto import is_market_open_et

        return bool(is_market_open_et(_utc_now()))
    except Exception:
        return False


def classify_runtime_pipeline_health(
    *,
    pipeline_hb: Dict[str, Any],
    runtime_hb: Dict[str, Any],
    loops: Dict[str, Any],
    market_open: bool,
    broker_snapshots_ok: Optional[bool] = None,
) -> str:
    """Runtime/data pipeline health — independent of certification or execution gates."""
    if broker_snapshots_ok is None:
        broker_snapshots_ok = _broker_snapshots_ok()

    ph_age = pipeline_hb.get("age_minutes")
    rt_age = runtime_hb.get("age_minutes")

    if not pipeline_hb.get("exists"):
        return "FAIL"

    if ph_age is None or float(ph_age) > RUNTIME_PIPELINE_FAIL_MINUTES:
        return "FAIL"

    if rt_age is not None and float(rt_age) > RUNTIME_PIPELINE_FAIL_MINUTES:
        return "FAIL"

    wd_alive = (
        bool(loops.get("watchdog_process_running")) or loops.get("watchdog_lock_fresh") is True
    )
    cl_alive = (
        bool(loops.get("continuous_process_running")) or loops.get("continuous_lock_fresh") is True
    )
    if not wd_alive and not cl_alive:
        return "FAIL"

    degraded = False
    if ph_age is not None and float(ph_age) > RUNTIME_PIPELINE_DEGRADED_MINUTES:
        degraded = True
    if rt_age is not None and float(rt_age) > RUNTIME_PIPELINE_DEGRADED_MINUTES:
        degraded = True
    if not loops.get("watchdog_process_running"):
        degraded = True
    if not loops.get("continuous_process_running"):
        degraded = True
    if not broker_snapshots_ok:
        degraded = True
    if not _portfolio_history_ok():
        degraded = True

    hb_status = str(pipeline_hb.get("status") or "").lower()
    if hb_status in ("fail", "failed", "error", "crash"):
        return "FAIL"
    if hb_status in ("warn", "warning", "degraded"):
        degraded = True

    if degraded:
        return "DEGRADED"

    if not market_open:
        return "OFF_HOURS_HEALTHY"
    return "HEALTHY"


def _normalize_certification_status(raw: str) -> str:
    status = str(raw or "").strip()
    if status in PAPER_CERTIFICATION_STATUSES:
        return "CERTIFIED"
    if status == "NOT_CERTIFIED":
        return "NOT_CERTIFIED"
    if status:
        return "REVIEW"
    return "NOT_CERTIFIED"


def derive_execution_block_reasons(
    *,
    market_open: bool,
    gate_bits: Dict[str, bool],
    certification_status: str,
    runtime_pipeline_health: str,
    data_freshness: str,
    live_permitted: bool,
) -> List[str]:
    """Display-only execution block reasons — not conflated with artifact contract failures."""
    reasons: List[str] = []
    if not market_open:
        reasons.append("MARKET_CLOSED")
    if not gate_bits.get("CONFIRM", False):
        reasons.append("CONFIRM_REQUIRED")
    cert_norm = _normalize_certification_status(certification_status)
    if cert_norm != "CERTIFIED":
        reasons.append("NOT_CERTIFIED")
    if not live_permitted:
        pass  # live_blocked tracked separately; paper mode expects live blocked
    if runtime_pipeline_health in ("DEGRADED", "FAIL"):
        reasons.append("PIPELINE_HEALTH")
    elif data_freshness == "STALE":
        reasons.append("PIPELINE_HEALTH")
    return reasons


def build_top_guard_message(
    *,
    runtime_pipeline_health: str,
    execution_block_reasons: List[str],
    market_open: bool,
) -> str:
    if runtime_pipeline_health in ("FAIL", "DEGRADED"):
        return "Pipeline runtime looks degraded. Open **System → Pipeline Health / Heartbeat**."
    if not execution_block_reasons:
        return "Runtime pipeline healthy. Execution gates are clear."
    if market_open is False and "NOT_CERTIFIED" in execution_block_reasons:
        return "Runtime pipeline healthy. Execution remains blocked by market/certification/confirmation."
    if market_open is False:
        return "Market closed. Paper execution safely blocked."
    if "NOT_CERTIFIED" in execution_block_reasons:
        return "Certification review required before execution can proceed."
    if "CONFIRM_REQUIRED" in execution_block_reasons:
        return "Execution blocked pending paper_arm_confirm (paper operator latch)."
    return "Runtime pipeline healthy. Execution remains blocked by policy gates."


def classify_contract_artifact_integrity(
    *,
    contracts_ok: Optional[bool] = None,
    contract_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Contract/artifact integrity — separate from runtime pipeline health emoji."""
    if contracts_ok is True:
        if contract_summary and int(contract_summary.get("warn_count") or 0) > 0:
            return "REVIEW"
        return "OK"
    if contracts_ok is False:
        return "FAIL"
    if (
        _broker_snapshots_ok()
        and _portfolio_history_ok()
        and load_pipeline_heartbeat().get("exists")
    ):
        return "OK"
    return "REVIEW"


def reconcile_integrity_lock_display(
    *,
    pipeline_locked: bool,
    pipeline_lock_reason: str,
    runtime_pipeline_health: str,
    contracts_ok: Optional[bool],
    contract_integrity: str,
) -> Dict[str, Any]:
    """
    Display-only reconciliation: do not show 'pipeline health' integrity lock
    when runtime pipeline is healthy and contracts pass.
    """
    contracts_failed = contracts_ok is False or contract_integrity == "FAIL"
    pipeline_runtime_bad = runtime_pipeline_health in ("DEGRADED", "FAIL")
    reason_lower = (pipeline_lock_reason or "").lower()

    show_caption = False
    detail = ""
    conflated = False

    if contracts_failed:
        show_caption = True
        if "data contracts" in reason_lower:
            detail = pipeline_lock_reason or "Data contracts FAIL"
        else:
            detail = "Data contracts FAIL — open System → Data Contracts"
    elif pipeline_runtime_bad and pipeline_locked:
        show_caption = True
        detail = pipeline_lock_reason or "Pipeline runtime degraded"
    elif pipeline_locked and not pipeline_runtime_bad and not contracts_failed:
        conflated = "pipeline health" in reason_lower

    return {
        "contract_artifact_integrity": contract_integrity,
        "show_integrity_lock_caption": show_caption,
        "integrity_lock_detail": detail,
        "integrity_conflated_with_pipeline_health": conflated,
    }


def audit_reconciled_runtime_locks() -> Dict[str, Any]:
    """Process-aware lock audit for dashboard display (Phase 196M)."""
    from services.phase_195h_operational_readiness_freeze import _find_ps_processes
    from services.runtime_lock_classifier import audit_runtime_locks

    wd_running, wd_pids = _find_ps_processes("run_watchdog_loop.ps1")
    cl_running, cl_pids = _find_ps_processes("run_continuous_loop.ps1")
    return audit_runtime_locks(
        watchdog_running=wd_running,
        continuous_running=cl_running,
        watchdog_pids=wd_pids,
        continuous_pids=cl_pids,
    )


def build_dashboard_health_classification(
    *,
    posture: Optional[Any] = None,
    live_ready: bool = False,
    live_gate_bits: Optional[Dict[str, bool]] = None,
    pipeline_locked: bool = False,
    pipeline_lock_reason: str = "",
) -> Dict[str, Any]:
    """
    Read-only dashboard health taxonomy (Phase 196L).
    Separates runtime pipeline health from execution/certification blocks.
    """
    pipeline_hb = load_pipeline_heartbeat()
    runtime_hb = load_runtime_heartbeat()
    loops = detect_loop_processes()
    positions = load_positions_summary()
    open_orders = count_open_orders()

    readiness = _read_json(EXECUTION_READINESS_PATH)
    if "readiness" in readiness and isinstance(readiness["readiness"], dict):
        readiness = readiness["readiness"]
    cert_doc = _read_json(CERTIFICATION_PATH)

    live_permitted = bool(readiness.get("live_execution_permitted", False))
    paper_permitted = bool(readiness.get("paper_execution_permitted", False))
    readiness_status = str(readiness.get("readiness_status") or "NOT_READY")
    certification_raw = str(cert_doc.get("certification_status") or "")
    if posture is not None:
        certification_raw = str(getattr(posture, "certification_status", "") or certification_raw)
        paper_permitted = bool(
            getattr(posture, "paper_execution_permitted", paper_permitted) or paper_permitted
        )
        live_permitted = bool(
            getattr(posture, "live_execution_permitted", live_permitted) or live_permitted
        )
        readiness_status = str(getattr(posture, "readiness_status", "") or readiness_status)

    market_open = _is_market_open()
    if posture is not None:
        market_open = bool(getattr(posture, "market_open", market_open))

    gate_bits = dict(live_gate_bits or {})
    data_freshness = classify_data_freshness(
        pipeline_hb.get("age_minutes"),
        market_open=market_open,
    )
    runtime_pipeline_health = classify_runtime_pipeline_health(
        pipeline_hb=pipeline_hb,
        runtime_hb=runtime_hb,
        loops=loops,
        market_open=market_open,
    )
    certification_status = _normalize_certification_status(certification_raw)
    execution_block_reasons = derive_execution_block_reasons(
        market_open=market_open,
        gate_bits=gate_bits,
        certification_status=certification_raw,
        runtime_pipeline_health=runtime_pipeline_health,
        data_freshness=data_freshness,
        live_permitted=live_permitted,
    )

    paper_operational = False
    if posture is not None:
        paper_operational = bool(getattr(posture, "paper_operational", False))

    if live_permitted:
        execution_status = "READY"
    elif paper_operational and not execution_block_reasons:
        execution_status = "READY"
    elif execution_block_reasons or not live_ready:
        execution_status = "BLOCKED"
    else:
        execution_status = "NOT_READY"

    return {
        "runtime_pipeline_health": runtime_pipeline_health,
        "data_freshness": data_freshness,
        "execution_status": execution_status,
        "execution_block_reasons": execution_block_reasons,
        "certification_status": certification_status,
        "market_status": "OPEN" if market_open else "CLOSED",
        "paper_authorized": paper_permitted,
        "live_blocked": not live_permitted,
        "readiness_status": readiness_status,
        "pipeline_integrity_locked": pipeline_locked,
        "pipeline_integrity_lock_reason": pipeline_lock_reason,
        "top_guard_message": build_top_guard_message(
            runtime_pipeline_health=runtime_pipeline_health,
            execution_block_reasons=execution_block_reasons,
            market_open=market_open,
        ),
        "pipeline_heartbeat": pipeline_hb,
        "runtime_heartbeat": runtime_hb,
        "loops": loops,
        "open_orders": open_orders,
        "positions_count": int(positions.get("position_count") or 0),
        "positions": list(positions.get("positions") or []),
        "heartbeat_fresh": (
            pipeline_hb.get("age_minutes") is not None
            and float(pipeline_hb.get("age_minutes")) <= RUNTIME_PIPELINE_DEGRADED_MINUTES
        ),
        "portfolio_history_current": _portfolio_history_ok(),
    }


def choose_sidebar_label(
    *,
    data_freshness: str,
    runtime_pipeline_health: str,
    market_open: bool,
    paper_operational: bool,
    paper_permitted: bool,
    readiness_status: str,
    live_permitted: bool,
    execution_blocked_reasons: List[str],
) -> Tuple[str, str]:
    """Return (label, level) where level is success|warning|error|info."""
    if live_permitted:
        return "LIVE RUNTIME ENABLED", "error"

    if runtime_pipeline_health in ("FAIL", "DEGRADED"):
        return "PIPELINE DEGRADED · EXECUTION BLOCKED", "error"

    if data_freshness == "STALE":
        return "DATA STALE · EXECUTION BLOCKED", "error"

    if not market_open:
        return "MARKET CLOSED · EXECUTION BLOCKED", "warning"

    if execution_blocked_reasons:
        return "PAPER AUTHORIZED · EXECUTION BLOCKED", "warning"

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
    contracts_ok: Optional[bool] = None,
    contract_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    health_cls = build_dashboard_health_classification(
        posture=posture,
        live_ready=live_ready,
        live_gate_bits=live_gate_bits,
        pipeline_locked=pipeline_locked,
        pipeline_lock_reason=pipeline_lock_reason,
    )

    pipeline_hb = health_cls["pipeline_heartbeat"]
    runtime_hb = health_cls["runtime_heartbeat"]
    loops = health_cls["loops"]
    open_orders = int(health_cls.get("open_orders") or 0)

    live_permitted = not health_cls.get("live_blocked", True)
    paper_permitted = bool(health_cls.get("paper_authorized"))
    readiness_status = str(health_cls.get("readiness_status") or "NOT_READY")
    data_freshness = str(health_cls.get("data_freshness") or "STALE")
    runtime_pipeline_health = str(health_cls.get("runtime_pipeline_health") or "FAIL")
    execution_block_reasons = list(health_cls.get("execution_block_reasons") or [])
    market_open = health_cls.get("market_status") == "OPEN"

    contract_integrity = classify_contract_artifact_integrity(
        contracts_ok=contracts_ok,
        contract_summary=contract_summary,
    )
    integrity_display = reconcile_integrity_lock_display(
        pipeline_locked=pipeline_locked,
        pipeline_lock_reason=pipeline_lock_reason,
        runtime_pipeline_health=runtime_pipeline_health,
        contracts_ok=contracts_ok,
        contract_integrity=contract_integrity,
    )

    lock_audit: Dict[str, Any] = {}
    try:
        lock_audit = audit_reconciled_runtime_locks()
    except Exception:
        pass

    paper_operational = False
    approval_label = ""
    if posture is not None:
        paper_operational = bool(getattr(posture, "paper_operational", False))
        approval_label = str(getattr(posture, "approval_status_label", "") or "")

    gate_bits = live_gate_bits or {}
    execution_gate = str(health_cls.get("execution_status") or "BLOCKED")

    sidebar_label, sidebar_level = choose_sidebar_label(
        data_freshness=data_freshness,
        runtime_pipeline_health=runtime_pipeline_health,
        market_open=market_open,
        paper_operational=paper_operational,
        paper_permitted=paper_permitted,
        readiness_status=readiness_status,
        live_permitted=live_permitted,
        execution_blocked_reasons=execution_block_reasons,
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

    wd_lock = str(lock_audit.get("watchdog_lock_classification") or "UNKNOWN")
    cl_lock = str(lock_audit.get("continuous_lock_classification") or "UNKNOWN")
    if posture is not None and wd_lock == "UNKNOWN":
        wd_lock = str(getattr(posture, "watchdog_lock_classification", "") or wd_lock)
    if posture is not None and cl_lock == "UNKNOWN":
        cl_lock = str(getattr(posture, "continuous_lock_classification", "") or cl_lock)

    return {
        "health_classification": health_cls,
        "runtime_pipeline_health": runtime_pipeline_health,
        "data_freshness": data_freshness,
        "contract_artifact_integrity": contract_integrity,
        "integrity_display": integrity_display,
        "watchdog_lock_classification": wd_lock,
        "continuous_lock_classification": cl_lock,
        "pipeline_heartbeat": pipeline_hb,
        "runtime_heartbeat": runtime_hb,
        "pipeline_heartbeat_source_path": pipeline_hb.get("path", str(PIPELINE_HEARTBEAT_PATH)),
        "pipeline_heartbeat_age_minutes": pipeline_hb.get("age_minutes"),
        "runtime_heartbeat_age_minutes": runtime_hb.get("age_minutes"),
        "watchdog": watchdog_state,
        "continuous_loop": continuous_state,
        "loops": loops,
        "execution_gate": execution_gate,
        "execution_status": execution_gate,
        "paper_mode": (
            "ACTIVE" if paper_operational else ("AUTHORIZED" if paper_permitted else "INACTIVE")
        ),
        "live_mode": "BLOCKED",
        "market_status": health_cls.get("market_status"),
        "certification_status": health_cls.get("certification_status"),
        "open_orders": open_orders,
        "positions_count": int(health_cls.get("positions_count") or 0),
        "positions": list(health_cls.get("positions") or []),
        "sidebar_label": sidebar_label,
        "sidebar_level": sidebar_level,
        "execution_blocked_reasons": execution_block_reasons,
        "top_guard_message": health_cls.get("top_guard_message"),
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
