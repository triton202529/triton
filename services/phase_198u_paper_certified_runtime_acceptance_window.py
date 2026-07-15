"""
phase_198u_paper_certified_runtime_acceptance_window.py
Phase 198U — Paper-certified runtime acceptance window audit.

Read-only acceptance observation. No broker orders, approvals, or live enablement.
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
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"

OUTPUT_JSON = RESULTS / "phase_198u_paper_certified_runtime_acceptance_window.json"
OUTPUT_MD = RESULTS / "phase_198u_paper_certified_runtime_acceptance_window.md"

APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
PAPER_ARM_PATH = LIVE / "paper_arm_confirm.json"
LIVE_CONFIRM_PATH = LIVE / "live_confirm.json"

CONTINUOUS_LOG_DIR = RESULTS / "continuous_loop"
WATCHDOG_LOG_DIR = RESULTS / "watchdog_logs"

PAPER_CERT_STATUSES = frozenset(
    {
        "CERTIFIED_FOR_PAPER_PROTECTION",
        "CONDITIONALLY_CERTIFIED",
        "PARTIALLY_CERTIFIED",
    }
)

ARTIFACT_PATHS = {
    "certification": RESULTS / "capital_preservation_certification.json",
    "evaluation": RESULTS / "protective_action_evaluation.json",
    "cert_fp_elimination": RESULTS / "certification_false_positive_elimination.json",
    "readiness": RESULTS / "execution_readiness.json",
    "governance_authorization": RESULTS / "governance_authorization.json",
    "executive_risk_summary": RESULTS / "executive_risk_summary.json",
    "watchdog_alerts": RESULTS / "watchdog_alerts.json",
    "watchdog_status": RESULTS / "watchdog_status.json",
    "positions_snapshot": RESULTS / "positions_snapshot.csv",
    "open_orders_snapshot": RESULTS / "open_orders_snapshot.csv",
    "recent_orders": RESULTS / "recent_orders.csv",
    "allocation_summary": RESULTS / "portfolio_allocation_summary.json",
    "allocation_recommendations": RESULTS / "portfolio_allocation_recommendations.csv",
    "phase_198t": RESULTS
    / "phase_198t_post_certification_paper_readiness_dashboard_sync_audit.json",
    "phase_198l": RESULTS / "phase_198l_gld_20pct_target_reduction_analysis.json",
    "phase_197y": RESULTS / "phase_197y_loop_mediated_paper_execution_gate.json",
    "phase_197z": RESULTS / "phase_197z_readiness_bridge_validation.json",
    "heartbeat": RESULTS / "heartbeat.json",
}

MARKER = "PAPER_CERTIFIED_RUNTIME_ACCEPTANCE_WINDOW_AUDIT_ONLY"
GLD_SYMBOL = "GLD"
GLD_TARGET_WEIGHT = 0.20
EXPECTED_GLD_QTY = 16


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return max(0, len(list(csv.reader(f))) - 1)
    except Exception:
        return 0


def _latest_log(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.is_dir():
        return None
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _minutes_since(ts: Optional[datetime]) -> Optional[float]:
    if ts is None:
        return None
    return (datetime.now(timezone.utc) - ts).total_seconds() / 60.0


def _parse_ts(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _gld_from_positions(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"qty": 0, "weight": 0.0}
    total = 0.0
    gld: Dict[str, Any] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mv = abs(_safe_float(row.get("market_value") or row.get("value")))
            total += mv
            sym = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
            if sym == GLD_SYMBOL:
                gld = row
    if not gld:
        return {"qty": 0, "weight": 0.0, "total_invested": total}
    qty = _safe_float(gld.get("qty") or gld.get("qty_available"))
    mv = abs(_safe_float(gld.get("market_value") or gld.get("value")))
    weight = mv / total if total > 0 else 0.0
    return {
        "qty": int(qty),
        "weight": round(weight, 6),
        "market_value": mv,
        "total_invested": total,
        "unrealized_plpc": _safe_float(gld.get("unrealized_plpc")),
    }


def _allocation_counts(summary: Dict[str, Any], rec_csv: Path) -> Dict[str, int]:
    open_new = int(summary.get("open_new_candidates") or 0)
    add = int(summary.get("add_candidates") or 0)
    exit_n = int(summary.get("exit_candidates") or 0)
    trim = int(summary.get("trim_candidates") or 0)
    if rec_csv.is_file():
        try:
            with rec_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not open_new:
                open_new = sum(1 for r in rows if r.get("recommended_action") == "OPEN_NEW")
            if not add:
                add = sum(1 for r in rows if r.get("recommended_action") == "ADD")
        except Exception:
            pass
    return {
        "open_new_count": open_new,
        "add_count": add,
        "exit_count": exit_n,
        "trim_count": trim,
    }


def _drawdown_monitoring(watchdog: Dict[str, Any], gld: Dict[str, Any]) -> Tuple[bool, bool]:
    alert = next(
        (
            a
            for a in (watchdog.get("active_alerts") or [])
            if str(a.get("alert_type") or "") == "EXCESS_POSITION_DRAWDOWN"
            and str(a.get("subject") or "").upper() == GLD_SYMBOL
        ),
        None,
    )
    if not alert:
        return False, False
    alert_pct = _safe_float((alert.get("details") or {}).get("unrealized_pl_pct"), 0.0)
    broker_pct = _safe_float(gld.get("unrealized_plpc"), 0.0) * 100.0
    supported = abs(alert_pct - broker_pct) <= 1.5 and broker_pct <= -10.0
    return True, supported


def _stale_gld_preview(phase_197y: Dict[str, Any], gld_qty: int) -> bool:
    if phase_197y.get("target_already_completed") and not phase_197y.get("eligible_for_execution"):
        return False
    if (
        phase_197y.get("defensive_preview_invalidated")
        and _safe_int(phase_197y.get("eligible_qty")) == 0
    ):
        return False
    preview = phase_197y.get("defensive_preview_broker_state") or {}
    preview_remaining = _safe_int(preview.get("preview_remaining_qty"))
    broker_qty = _safe_int(preview.get("broker_qty") or gld_qty)
    if (
        preview_remaining > broker_qty
        and broker_qty > 0
        and phase_197y.get("eligible_for_execution")
    ):
        return True
    if _safe_int(phase_197y.get("stale_preview_original_qty")) and phase_197y.get(
        "eligible_for_execution"
    ):
        return True
    return False


def _stale_concentration(evaluation: Dict[str, Any]) -> bool:
    conc = next(
        (
            e
            for e in (evaluation.get("evaluations") or [])
            if e.get("trial_id") == "paper_concentration_reduction"
        ),
        {},
    )
    if not conc:
        return False
    if str(conc.get("evaluation") or "").upper() == "NEGATIVE":
        return True
    if conc.get("evaluation_mode") not in ("COMPLETED_BROKER_VERIFIED",):
        if not evaluation.get("concentration_reduction_completed"):
            return True
    return False


def _stale_certification(
    cert_status: str,
    failed: List[str],
    cert_fp: Dict[str, Any],
    dash_blockers: List[str],
) -> bool:
    if cert_status not in PAPER_CERT_STATUSES or failed:
        return False
    fp_status = str(cert_fp.get("certification_status") or "")
    if fp_status == "NOT_CERTIFIED":
        return True
    wd = _load_json(ARTIFACT_PATHS["watchdog_status"])
    if wd.get("certification_status") == "NOT_CERTIFIED":
        return True
    if "NOT_CERTIFIED" in dash_blockers:
        return True
    return False


def _dashboard_sync_status(cert_status: str, failed: List[str]) -> str:
    try:
        from services.dashboard_status_sync import build_dashboard_health_classification
        from services.live_gate import compute_display_gates

        live_ready, _, gate_bits, _ = compute_display_gates()
        dash = build_dashboard_health_classification(
            live_gate_bits=gate_bits,
            live_ready=live_ready,
        )
        cert_norm = str(dash.get("certification_status") or "")
        blockers = list(dash.get("execution_block_reasons") or [])
        cert_ok = cert_status in PAPER_CERT_STATUSES and len(failed) == 0
        if cert_ok and ("NOT_CERTIFIED" in blockers or cert_norm == "NOT_CERTIFIED"):
            return "STALE_CERTIFICATION_DISPLAY"
        if cert_ok:
            return "SYNCED"
        return f"REVIEW:{cert_norm}"
    except Exception as exc:
        return f"ERROR:{exc}"


def _runtime_loop_status() -> Dict[str, Any]:
    from services.dashboard_status_sync import detect_loop_processes, load_runtime_heartbeat

    loops = detect_loop_processes()
    hb = load_runtime_heartbeat()
    heartbeat_doc = _load_json(ARTIFACT_PATHS["heartbeat"])
    hb_ts = _parse_ts(heartbeat_doc.get("timestamp"))
    hb_age = _minutes_since(hb_ts)

    cl_log = _latest_log(CONTINUOUS_LOG_DIR, "continuous_loop_*.log")
    wd_log = _latest_log(WATCHDOG_LOG_DIR, "watchdog_*.log")
    cl_log_age = (
        _minutes_since(datetime.fromtimestamp(cl_log.stat().st_mtime, tz=timezone.utc))
        if cl_log
        else None
    )
    wd_log_age = (
        _minutes_since(datetime.fromtimestamp(wd_log.stat().st_mtime, tz=timezone.utc))
        if wd_log
        else None
    )

    cl_running = bool(loops.get("continuous_process_running"))
    wd_running = bool(loops.get("watchdog_process_running"))

    if cl_running:
        cl_status = "RUNNING"
    elif loops.get("continuous_lock_present") and not cl_running:
        cl_status = "STOPPED_LOCK_PRESENT"
    else:
        cl_status = "STOPPED"

    if wd_running:
        wd_status = "RUNNING"
    elif loops.get("watchdog_lock_present") and not wd_running:
        wd_status = "STOPPED_LOCK_PRESENT"
    else:
        wd_status = "STOPPED"

    return {
        "continuous_loop_status": cl_status,
        "watchdog_status": wd_status,
        "continuous_process_running": cl_running,
        "watchdog_process_running": wd_running,
        "runtime_heartbeat_age_minutes": hb.get("age_minutes"),
        "heartbeat_age_minutes": round(hb_age, 1) if hb_age is not None else None,
        "latest_continuous_log": (
            str(cl_log.relative_to(ROOT)).replace("\\", "/") if cl_log else None
        ),
        "latest_continuous_log_age_minutes": (
            round(cl_log_age, 1) if cl_log_age is not None else None
        ),
        "latest_watchdog_log": (
            str(wd_log.relative_to(ROOT)).replace("\\", "/") if wd_log else None
        ),
        "latest_watchdog_log_age_minutes": (
            round(wd_log_age, 1) if wd_log_age is not None else None
        ),
        "process_detection_confidence": loops.get("process_detection_confidence"),
    }


def _paper_entry_assessment(
    *,
    cert_status: str,
    failed: List[str],
    arm: Dict[str, Any],
    alloc: Dict[str, int],
    dash_blockers: List[str],
    live_allowed: bool,
) -> Tuple[bool, str]:
    cert_ok = cert_status in PAPER_CERT_STATUSES and len(failed) == 0
    paper_arm_valid = str(arm.get("status") or "") == "VALID"
    paper_entry_armed = bool(arm.get("paper_entry_armed"))
    open_new = int(alloc.get("open_new_count") or 0)
    add = int(alloc.get("add_count") or 0)
    alloc_entries = open_new + add

    if live_allowed:
        return False, "live_execution_not_blocked"
    if not cert_ok:
        return False, "certification_not_ready"
    if "NOT_CERTIFIED" in dash_blockers:
        return False, "dashboard_still_shows_not_certified_block"
    if alloc_entries == 0:
        return False, "allocation_blocked_no_open_new_or_add_candidates"
    if not paper_arm_valid:
        return False, "paper_arm_confirm_not_valid"
    if not paper_entry_armed:
        return False, "paper_entry_not_armed"
    return True, (
        f"eligible_via_controlled_paper_entry_lane "
        f"(open_new={open_new}, add={add}); allocation is authority; no manual bypass"
    )


def _bridge_qty(phase_197z: Dict[str, Any]) -> int:
    if "bridge_qty" in phase_197z:
        return _safe_int(phase_197z.get("bridge_qty"))
    bridge = phase_197z.get("bridge") or {}
    return _safe_int(bridge.get("bridge_qty"))


def _198l_sell_qty(phase_198l: Dict[str, Any]) -> int:
    return _safe_int(
        phase_198l.get("recommended_sell_qty")
        or (phase_198l.get("analysis") or {}).get("recommended_sell_qty")
        or phase_198l.get("sell_qty")
    )


def _certification_forced(cert: Dict[str, Any], phase_198t: Dict[str, Any]) -> bool:
    if "certification_forced" in cert:
        return bool(cert.get("certification_forced"))
    audit = phase_198t.get("audit") or {}
    return bool(audit.get("certification_forced", False))


def _render_markdown(payload: Dict[str, Any]) -> str:
    a = payload.get("acceptance") or {}
    rt = payload.get("runtime") or {}
    safety = payload.get("safety") or {}
    return f"""# TRITON Phase 198U — Paper-Certified Runtime Acceptance Window

**Generated:** {payload.get("generated_at")}
**Status:** {payload.get("status")}
**Acceptance state:** {a.get("acceptance_state")}

## Executive summary

{payload.get("executive_summary")}

## Certification

| Field | Value |
|-------|-------|
| Status | {a.get("certification_status")} |
| Forced | {a.get("certification_forced")} |
| Failed requirements | {a.get("failed_requirements")} |
| Dashboard sync | {a.get("dashboard_sync_status")} |
| Stale NOT_CERTIFIED | {a.get("stale_certification_detected")} |

## Runtime posture

| Field | Value |
|-------|-------|
| Watchdog | {a.get("watchdog_status")} |
| Continuous loop | {a.get("continuous_loop_status")} |
| Paper arm valid | {a.get("paper_arm_valid")} |
| Paper entry armed | {a.get("paper_entry_armed")} |
| Live blocked | {a.get("live_execution_blocked")} |
| Recommended action | {a.get("recommended_next_runtime_action")} |

## Paper entry & allocation

| Field | Value |
|-------|-------|
| Normal entry possible | {a.get("normal_paper_entry_possible_now")} |
| Reason | {a.get("normal_paper_entry_reason")} |
| open_new / add | {a.get("allocation_open_new_count")} / {a.get("allocation_add_count")} |
| exit / trim | {a.get("allocation_exit_count")} / {a.get("allocation_trim_count")} |

## GLD & drawdown monitoring

| Field | Value |
|-------|-------|
| GLD qty / weight | {a.get("gld_qty")} / {a.get("gld_weight_pct")}% |
| Target satisfied | {a.get("gld_target_satisfied")} |
| Residual drawdown monitoring | {a.get("residual_drawdown_monitoring_active")} |

## Safety

| Check | Value |
|-------|-------|
| broker_orders_submitted | {safety.get("broker_orders_submitted")} |
| approval artifact | {a.get("approval_artifact_present")} |
| broker auth artifact | {a.get("broker_auth_artifact_present")} |
| open orders | {a.get("open_orders_count")} |

## Runtime logs

| Log | Path | Age (min) |
|-----|------|-----------|
| Continuous loop | {rt.get("latest_continuous_log") or "n/a"} | {rt.get("latest_continuous_log_age_minutes")} |
| Watchdog | {rt.get("latest_watchdog_log") or "n/a"} | {rt.get("latest_watchdog_log_age_minutes")} |

---

*{MARKER}*
"""


def run_phase_198u_audit(*, verbose: bool = True) -> Dict[str, Any]:
    from services.paper_arm_status import build_paper_arm_status
    from services.portfolio_history_sync import load_live_execution_flag

    artifacts = {k: _load_json(p) for k, p in ARTIFACT_PATHS.items() if p.suffix == ".json"}
    cert = artifacts.get("certification") or {}
    evaluation = artifacts.get("evaluation") or {}
    cert_fp = artifacts.get("cert_fp_elimination") or {}
    readiness = artifacts.get("readiness") or {}
    gov_auth = artifacts.get("governance_authorization") or {}
    watchdog = artifacts.get("watchdog_alerts") or {}
    phase_198t = artifacts.get("phase_198t") or {}
    phase_198l = artifacts.get("phase_198l") or {}
    phase_197y = artifacts.get("phase_197y") or {}
    phase_197z = artifacts.get("phase_197z") or {}

    gld = _gld_from_positions(ARTIFACT_PATHS["positions_snapshot"])
    alloc = _allocation_counts(
        artifacts.get("allocation_summary") or {},
        ARTIFACT_PATHS["allocation_recommendations"],
    )
    open_orders = _count_csv_rows(ARTIFACT_PATHS["open_orders_snapshot"])
    positions_count = _count_csv_rows(ARTIFACT_PATHS["positions_snapshot"])

    arm = build_paper_arm_status(verbose_notes=False)
    approval_present = APPROVAL_PATH.is_file()
    broker_auth_present = BROKER_AUTH_PATH.is_file()
    live_flag = bool(load_live_execution_flag())
    live_readiness = bool(readiness.get("live_execution_permitted"))
    live_allowed = live_flag or live_readiness or bool(gov_auth.get("live_execution_permitted"))

    cert_status = str(cert.get("certification_status") or "")
    failed = cert.get("failed_requirements") or []
    cert_forced = _certification_forced(cert, phase_198t)

    dd_active, dd_supported = _drawdown_monitoring(watchdog, gld)
    stale_preview = _stale_gld_preview(phase_197y, int(gld.get("qty") or 0))
    stale_conc = _stale_concentration(evaluation)

    dash_sync = _dashboard_sync_status(cert_status, failed)
    runtime = _runtime_loop_status()

    try:
        from services.dashboard_status_sync import build_dashboard_health_classification
        from services.live_gate import compute_display_gates

        live_ready, _, gate_bits, _ = compute_display_gates()
        dash_blockers = list(
            (
                build_dashboard_health_classification(
                    live_gate_bits=gate_bits,
                    live_ready=live_ready,
                )
            ).get("execution_block_reasons")
            or []
        )
    except Exception:
        dash_blockers = []

    stale_cert = _stale_certification(cert_status, failed, cert_fp, dash_blockers)
    normal_possible, entry_reason = _paper_entry_assessment(
        cert_status=cert_status,
        failed=failed,
        arm=arm,
        alloc=alloc,
        dash_blockers=dash_blockers,
        live_allowed=live_allowed,
    )

    gld_weight = float(gld.get("weight") or 0)
    gld_qty = int(gld.get("qty") or 0)
    gld_target = bool(
        phase_198l.get("target_already_satisfied") or gld_weight <= GLD_TARGET_WEIGHT + 0.005
    )
    sell_qty_198l = _198l_sell_qty(phase_198l)
    eligible_qty_197y = _safe_int(phase_197y.get("eligible_qty"))
    bridge_qty_197z = _bridge_qty(phase_197z)

    alloc_entries = int(alloc.get("open_new_count") or 0) + int(alloc.get("add_count") or 0)
    open_orders_allowed = open_orders == 0 or (alloc_entries > 0 and open_orders > 0)

    checks: Dict[str, bool] = {
        "certification_certified_for_paper": cert_status == "CERTIFIED_FOR_PAPER_PROTECTION",
        "certification_not_forced": not cert_forced,
        "failed_requirements_empty": len(failed) == 0,
        "live_blocked": not live_allowed,
        "approval_absent": not approval_present,
        "broker_auth_absent": not broker_auth_present,
        "open_orders_acceptable": open_orders_allowed,
        "gld_qty_16": gld_qty == EXPECTED_GLD_QTY,
        "gld_weight_at_target": gld_weight <= GLD_TARGET_WEIGHT + 0.005,
        "198l_sell_qty_zero": sell_qty_198l == 0,
        "197y_eligible_qty_zero": eligible_qty_197y == 0,
        "197z_bridge_qty_zero": bridge_qty_197z == 0,
        "stale_certification_absent": not stale_cert,
        "stale_gld_preview_absent": not stale_preview,
        "stale_concentration_absent": not stale_conc,
        "paper_arm_valid": str(arm.get("status") or "") == "VALID",
        "paper_entry_armed": bool(arm.get("paper_entry_armed")),
        "dashboard_synced": dash_sync == "SYNCED",
        "no_broker_orders": True,
    }

    if live_allowed or approval_present or broker_auth_present:
        status = "FAIL"
        acceptance_state = "UNSAFE_RUNTIME"
    elif (
        not checks["certification_certified_for_paper"] or stale_cert or stale_preview or stale_conc
    ):
        status = "REVIEW_REQUIRED"
        acceptance_state = "CERTIFICATION_OR_SYNC_DRIFT"
    elif (
        not open_orders_allowed
        or gld_qty != EXPECTED_GLD_QTY
        or gld_weight > GLD_TARGET_WEIGHT + 0.005
    ):
        status = "REVIEW_REQUIRED"
        acceptance_state = "BROKER_OR_GLD_DRIFT"
    elif dd_active and dd_supported:
        status = "MONITORING_PASS"
        acceptance_state = "PAPER_CERTIFIED_MONITOR_DRAWDOWN"
    elif all(checks.values()):
        status = "PASS"
        acceptance_state = "STABLE_PAPER_CERTIFIED"
    else:
        status = "REVIEW_REQUIRED"
        acceptance_state = "ACCEPTANCE_CHECK_INCOMPLETE"

    if status in ("PASS", "MONITORING_PASS"):
        if alloc_entries == 0:
            recommended = (
                "MAINTAIN_PAPER_CERTIFIED_POSTURE; monitor GLD drawdown; "
                "allocation controls entry (open_new/add=0)"
            )
        else:
            recommended = (
                "MAINTAIN_PAPER_CERTIFIED_POSTURE; controlled paper entry lane only "
                "when allocation signals open_new/add"
            )
    elif status == "FAIL":
        recommended = "HALT_AND_REVIEW_SAFETY_VIOLATION"
    else:
        recommended = "REVIEW_DRIFT_AND_REFRESH_RUNTIME_ARTIFACTS"

    acceptance_fields = {
        "acceptance_state": acceptance_state,
        "certification_status": cert_status,
        "certification_forced": cert_forced,
        "failed_requirements": failed,
        "watchdog_status": runtime.get("watchdog_status"),
        "dashboard_sync_status": dash_sync,
        "continuous_loop_status": runtime.get("continuous_loop_status"),
        "paper_arm_valid": str(arm.get("status") or "") == "VALID",
        "paper_entry_armed": bool(arm.get("paper_entry_armed")),
        "live_execution_blocked": not live_allowed,
        "live_execution_allowed": live_allowed,
        "approval_artifact_present": approval_present,
        "broker_auth_artifact_present": broker_auth_present,
        "open_orders_count": open_orders,
        "positions_count": positions_count,
        "gld_qty": gld_qty,
        "gld_weight": gld_weight,
        "gld_weight_pct": round(gld_weight * 100.0, 4),
        "gld_target_satisfied": gld_target,
        "residual_drawdown_monitoring_active": dd_active,
        "residual_drawdown_monitoring_supported": dd_supported,
        "allocation_open_new_count": alloc.get("open_new_count"),
        "allocation_add_count": alloc.get("add_count"),
        "allocation_exit_count": alloc.get("exit_count"),
        "allocation_trim_count": alloc.get("trim_count"),
        "normal_paper_entry_possible_now": normal_possible,
        "normal_paper_entry_reason": entry_reason,
        "stale_certification_detected": stale_cert,
        "stale_gld_preview_detected": stale_preview,
        "stale_concentration_detected": stale_conc,
        "198l_sell_qty": sell_qty_198l,
        "197y_eligible_qty": eligible_qty_197y,
        "197z_bridge_qty": bridge_qty_197z,
        "recommended_next_runtime_action": recommended,
    }

    baseline_198t = phase_198t.get("audit") or {}
    drift_notes: List[str] = []
    if baseline_198t:
        if baseline_198t.get("certification_status") != cert_status:
            drift_notes.append(
                f"cert_status drift: 198T={baseline_198t.get('certification_status')} now={cert_status}"
            )
        if baseline_198t.get("gld_qty") != gld_qty:
            drift_notes.append(f"gld_qty drift: 198T={baseline_198t.get('gld_qty')} now={gld_qty}")

    executive_summary = (
        f"Acceptance {acceptance_state} ({status}). "
        f"Cert {cert_status} forced={cert_forced}; dashboard={dash_sync}; "
        f"watchdog={runtime.get('watchdog_status')} loop={runtime.get('continuous_loop_status')}. "
        f"Paper entry={normal_possible} ({entry_reason}). "
        f"GLD {gld_qty}@{acceptance_fields['gld_weight_pct']}%; drawdown monitor={dd_active}."
    )

    payload: Dict[str, Any] = {
        "phase": "198U",
        "generated_at": _utc_now_iso(),
        "status": status,
        "checks": checks,
        "executive_summary": executive_summary,
        "acceptance": acceptance_fields,
        "runtime": runtime,
        "baseline_198t_comparison": {
            "phase_198t_status": phase_198t.get("status"),
            "phase_198t_generated_at": phase_198t.get("generated_at"),
            "drift_notes": drift_notes,
        },
        "paper_arm": {
            "status": arm.get("status"),
            "paper_entry_armed": arm.get("paper_entry_armed"),
            "paper_arm_confirm_present": PAPER_ARM_PATH.is_file(),
            "live_confirm_present": LIVE_CONFIRM_PATH.is_file(),
        },
        "safety": {
            "broker_orders_submitted": False,
            "approval_artifact_created": False,
            "broker_auth_artifact_created": False,
            "live_execution_modified": False,
        },
        "artifact_snapshot": {
            k: {"path": str(p.relative_to(ROOT)).replace("\\", "/"), "present": p.is_file()}
            for k, p in ARTIFACT_PATHS.items()
        },
        "git_status_short": "",
        "git_diff_stat": "",
        "marker": MARKER,
    }

    try:
        payload["git_status_short"] = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        payload["git_diff_stat"] = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception:
        pass

    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(OUTPUT_MD, _render_markdown(payload))

    if verbose:
        print(
            f"[198U] status={status} acceptance={acceptance_state} cert={cert_status} "
            f"loop={runtime.get('continuous_loop_status')} wd={runtime.get('watchdog_status')} "
            f"gld_qty={gld_qty}"
        )

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 198U paper-certified runtime acceptance")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run_phase_198u_audit(verbose=not args.quiet)
    return 0 if result.get("status") in ("PASS", "MONITORING_PASS", "REVIEW_REQUIRED") else 1


if __name__ == "__main__":
    sys.exit(main())
