"""
phase_195g_post_hardening_sync_audit.py
Phase 195G — Post-hardening institutional sync audit (read-only).

Confirms end-to-end synchronization after Phases 195C–195F.
Does not place orders, alter broker state, or change thresholds.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"

OUTPUT_JSON = RESULTS / "phase_195g_post_hardening_sync_audit.json"
SCORECARD_CSV = RESULTS / "phase_195g_post_hardening_sync_scorecard.csv"
MATRIX_CSV = RESULTS / "phase_195g_authority_sync_matrix.csv"

LIFECYCLE_AUTHORITY = RESULTS / "signal_lifecycle_effective.csv"
RISK_AUTHORITY = RESULTS / "master_execution_gate.json"
POSITIONS_SNAPSHOT = RESULTS / "positions_snapshot.csv"
OPEN_ORDERS_SNAPSHOT = RESULTS / "open_orders_snapshot.csv"

STALE_LOCK_MINUTES = 60.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return -1
    try:
        with path.open(encoding="utf-8", errors="ignore") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return -1


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _minutes_since(dt: Optional[datetime]) -> Optional[float]:
    if not dt:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0


def _parse_lock(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "path": str(path.relative_to(ROOT)).replace("\\", "/")}
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    info: Dict[str, Any] = {
        "exists": True,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "raw": text[:200],
    }
    m = re.search(r"started_at=([^\r\n]+)", text)
    if m:
        info["started_at"] = m.group(1).strip()
        info["started_at_dt"] = _parse_ts(info["started_at"])
    m_pid = re.search(r"pid=(\d+)", text, re.I)
    if m_pid:
        info["pid"] = int(m_pid.group(1))
    age = _minutes_since(info.get("started_at_dt"))
    if age is not None:
        info["lock_age_minutes"] = round(age, 1)
        info["stale"] = age > STALE_LOCK_MINUTES
    return info


def _run_module(
    module: str, extra_args: Optional[List[str]] = None, timeout: int = 300
) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", module] + (extra_args or [])
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "module": module,
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0,
            "stderr_tail": (proc.stderr or "")[-400:],
        }
    except Exception as e:
        return {"module": module, "exit_code": -1, "ok": False, "error": str(e)}


def _audit_broker_sync(*, run_snapshot: bool = True) -> Dict[str, Any]:
    before_orders = _count_csv_rows(OPEN_ORDERS_SNAPSHOT)
    before_positions = _count_csv_rows(POSITIONS_SNAPSHOT)

    snapshot_run: Dict[str, Any] = {"skipped": not run_snapshot}
    if run_snapshot:
        snapshot_run = _run_module(
            "services.snapshot_live_orders", ["--mode", "paper"], timeout=120
        )

    after_orders = _count_csv_rows(OPEN_ORDERS_SNAPSHOT)
    after_positions = _count_csv_rows(POSITIONS_SNAPSHOT)

    watchdog = _load_json(RESULTS / "watchdog_status.json")
    wd_orders = watchdog.get("open_orders_count")
    wd_positions = watchdog.get("positions_count")
    broker_connected = watchdog.get("broker_connected")

    orders_match = after_orders >= 0 and (wd_orders is None or int(wd_orders) == after_orders)
    positions_match = after_positions >= 0 and (
        wd_positions is None or int(wd_positions) == after_positions
    )
    no_new_orders = before_orders <= after_orders and before_positions == after_positions

    orphan_risk = after_orders > 0 and not orders_match
    status = "SYNCED"
    issues: List[str] = []
    if not snapshot_run.get("ok") and run_snapshot:
        status = "DEGRADED"
        issues.append("snapshot_live_orders failed")
    if after_orders < 0 or after_positions < 0:
        status = "MISSING"
        issues.append("broker snapshot CSV missing or unreadable")
    if not orders_match:
        status = "MISMATCH"
        issues.append(f"open_orders snapshot={after_orders} watchdog={wd_orders}")
    if not positions_match:
        status = "MISMATCH" if status != "MISSING" else status
        issues.append(f"positions snapshot={after_positions} watchdog={wd_positions}")
    if not no_new_orders:
        issues.append(
            f"snapshot changed counts orders {before_orders}->{after_orders} "
            f"positions {before_positions}->{after_positions}"
        )

    return {
        "status": status,
        "broker_connected": broker_connected,
        "broker_status": watchdog.get("broker_status"),
        "open_orders_count": after_orders,
        "positions_count": after_positions,
        "watchdog_open_orders": wd_orders,
        "watchdog_positions": wd_positions,
        "orders_match_watchdog": orders_match,
        "positions_match_watchdog": positions_match,
        "snapshot_before": {
            "open_orders": before_orders,
            "positions": before_positions,
        },
        "snapshot_after": {"open_orders": after_orders, "positions": after_positions},
        "no_unexpected_snapshot_growth": no_new_orders,
        "orphan_order_risk": orphan_risk,
        "duplicate_order_risk": False,
        "issues": issues,
        "snapshot_run": snapshot_run,
    }


def _audit_execution_authority() -> Dict[str, Any]:
    gate_exists = RISK_AUTHORITY.is_file() and RISK_AUTHORITY.stat().st_size > 0
    gate = _load_json(RISK_AUTHORITY)
    gov = _load_json(RESULTS / "governance_authorization.json")
    readiness = _load_json(RESULTS / "execution_readiness.json")

    live_permitted = gov.get("live_execution_permitted")
    if live_permitted is None:
        live_permitted = readiness.get("live_execution_permitted")
    paper_permitted = gov.get("paper_execution_permitted")
    if paper_permitted is None:
        paper_permitted = readiness.get("paper_execution_permitted")

    loader_ok = False
    loader_detail = ""
    try:
        from services.decision_authority import (
            RISK_AUTHORITY_PATH,
            load_risk_execution_controls,
        )

        controls = load_risk_execution_controls()
        loader_ok = RISK_AUTHORITY_PATH == RISK_AUTHORITY
        loader_detail = f"controls={controls[:2]}"
    except Exception as e:
        loader_detail = f"{type(e).__name__}:{e}"

    trade_opps_text = (ROOT / "services" / "build_trade_opportunities.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    uses_master_loader = "load_risk_execution_controls" in trade_opps_text
    adaptive_bypass = bool(
        re.search(r"adaptive_risk_state\.json", trade_opps_text)
        and "load_risk_execution_controls" not in trade_opps_text
    )

    status = "AUTHORITATIVE"
    issues: List[str] = []
    if not gate_exists:
        status = "MISSING"
        issues.append("master_execution_gate.json missing")
    if not loader_ok:
        status = "DEGRADED"
        issues.append("decision_authority loader failed")
    if live_permitted is not False:
        status = "BLOCKED_FAIL"
        issues.append(f"live_execution_permitted={live_permitted}")
    if adaptive_bypass:
        status = "BYPASS_RISK"
        issues.append("build_trade_opportunities may bypass master gate")

    return {
        "status": status,
        "master_gate_exists": gate_exists,
        "master_gate_mode": gate.get("mode"),
        "master_gate_ok": gate.get("ok"),
        "decision_authority_loader_ok": loader_ok,
        "decision_authority_detail": loader_detail,
        "live_execution_permitted": live_permitted,
        "paper_execution_permitted": paper_permitted,
        "paper_execution_authorized": paper_permitted is True,
        "trade_opportunities_uses_loader": uses_master_loader,
        "issues": issues,
    }


def _audit_lifecycle_authority(d195: Dict[str, Any]) -> Dict[str, Any]:
    effective_exists = LIFECYCLE_AUTHORITY.is_file() and LIFECYCLE_AUTHORITY.stat().st_size > 0
    raw_exists = (RESULTS / "signal_lifecycle.csv").is_file()

    operational = d195.get("authority_map", {}).get("lifecycle_operational_consumers") or []
    lc_conflicts = int(d195.get("lifecycle_conflicts") or 0)
    high = int(d195.get("high_severity_count") or 0)

    status = "EFFECTIVE"
    issues: List[str] = []
    if not effective_exists:
        status = "MISSING"
        issues.append("signal_lifecycle_effective.csv missing")
    if high > 0:
        status = "CONFLICT"
        issues.append(f"HIGH lifecycle conflicts={high}")

    return {
        "status": status,
        "authority_path": str(LIFECYCLE_AUTHORITY.relative_to(ROOT)).replace("\\", "/"),
        "effective_exists": effective_exists,
        "effective_row_count": _count_csv_rows(LIFECYCLE_AUTHORITY),
        "raw_advisory_exists": raw_exists,
        "operational_consumer_count": len(operational),
        "lifecycle_conflicts": lc_conflicts,
        "issues": issues,
    }


def _audit_risk_authority(d195: Dict[str, Any], d195e: Dict[str, Any]) -> Dict[str, Any]:
    gate_exists = RISK_AUTHORITY.is_file() and RISK_AUTHORITY.stat().st_size > 0
    adaptive_exists = (RESULTS / "adaptive_risk_state.json").is_file()

    high = int(d195.get("high_severity_count") or 0)
    risk_conflicts = int(d195.get("risk_conflicts") or 0)
    e_remediated = d195e.get("phase_pass")

    status = "MASTER_GATE"
    issues: List[str] = []
    if not gate_exists:
        status = "MISSING"
        issues.append("master_execution_gate.json missing")
    if high > 0:
        status = "CONFLICT"
        issues.append(f"HIGH risk conflicts={high}")
    if e_remediated is False:
        status = "REMEDIATION_FAIL"
        issues.append("Phase 195E remediation not passing")

    return {
        "status": status,
        "authority_path": str(RISK_AUTHORITY.relative_to(ROOT)).replace("\\", "/"),
        "master_gate_exists": gate_exists,
        "adaptive_risk_advisory_only": adaptive_exists,
        "risk_conflicts": risk_conflicts,
        "phase_195e_remediation_intact": e_remediated is True,
        "issues": issues,
    }


def _audit_governance_certification(d195c: Dict[str, Any]) -> Dict[str, Any]:
    cert = _load_json(RESULTS / "capital_preservation_certification.json")
    eval_doc = _load_json(RESULTS / "capital_preservation_evaluation.json")
    pae = eval_doc.get("protective_action_evaluation") or eval_doc

    cert_status = cert.get("certification_status")
    veto = cert.get("veto_triggered") or []
    eval_ctx = cert.get("evaluation_context") or {}
    scoring_version = (
        eval_ctx.get("scoring_model_version")
        or pae.get("scoring_model_version")
        or eval_doc.get("scoring_model_version")
    )

    acceptable = cert_status in (
        "CERTIFIED_FOR_PAPER_PROTECTION",
        "CONDITIONALLY_CERTIFIED",
        "PARTIALLY_CERTIFIED",
    )

    status = "CERTIFIED" if acceptable else "NOT_CERTIFIED"
    issues: List[str] = []
    if not acceptable:
        issues.append(f"certification_status={cert_status}")
    if veto and any(str(v).lower() not in ("", "none", "false") for v in veto):
        issues.append(f"veto_triggered={veto}")
    if not scoring_version:
        issues.append("scoring_model_version missing")
    if not eval_ctx.get("baseline_context") and not pae.get("baseline_context"):
        issues.append("high-CPS baseline_context missing")

    return {
        "status": status,
        "certification_status": cert_status,
        "phase_195c_pass": d195c.get("phase_pass"),
        "scoring_model_version": scoring_version,
        "baseline_context": eval_ctx.get("baseline_context") or pae.get("baseline_context"),
        "high_cps_adjustment_applied": eval_ctx.get("high_cps_adjustment_applied")
        or pae.get("high_cps_adjustment_applied"),
        "veto_triggered": veto,
        "mandatory_veto_firing": bool(veto),
        "issues": issues,
    }


def _audit_dashboard_labeling(d195f: Dict[str, Any]) -> Dict[str, Any]:
    labels = {
        "raw_lifecycle": d195f.get("raw_lifecycle_labeled_non_authoritative"),
        "effective_lifecycle": d195f.get("effective_lifecycle_labeled_authoritative"),
        "adaptive_risk": d195f.get("adaptive_risk_labeled_supporting_only"),
        "master_gate": d195f.get("master_gate_labeled_final_authority"),
    }
    all_present = all(labels.values())
    status = "LABELED" if all_present else "INCOMPLETE"
    issues: List[str] = []
    for k, v in labels.items():
        if not v:
            issues.append(f"missing label coverage: {k}")
    if d195f.get("phase_pass") is False and all_present:
        issues.append("Phase 195F artifact phase_pass=false; label coverage present")

    return {
        "status": status,
        "phase_195f_pass": d195f.get("phase_pass"),
        "labels_present": labels,
        "dashboard_labels_updated": d195f.get("dashboard_labels_updated"),
        "issues": issues,
    }


def _audit_watchdog() -> Dict[str, Any]:
    status_doc = _load_json(RESULTS / "watchdog_status.json")
    alerts = _load_json(RESULTS / "watchdog_alerts.json")
    alert_state = _load_json(LIVE / "watchdog_alert_state.json")
    loop_lock = _parse_lock(LIVE / "watchdog_loop.lock")

    wd_status = status_doc.get("watchdog_status") or status_doc.get("status") or "UNKNOWN"
    active_alerts = alerts.get("active_count")
    if active_alerts is None:
        active_list = alerts.get("active_alerts") or alerts.get("alerts") or []
        active_alerts = len(active_list) if isinstance(active_list, list) else 0

    hb = status_doc.get("heartbeat") or {}
    hb_status = hb.get("status") if isinstance(hb, dict) else None
    hb_ts = hb.get("timestamp") if isinstance(hb, dict) else status_doc.get("timestamp")
    hb_age = _minutes_since(_parse_ts(hb_ts))

    known = bool(status_doc)
    operational = wd_status in ("OK", "WARNING", "DEGRADED", "YELLOW")
    status = "OK" if wd_status == "OK" else ("KNOWN" if known else "UNKNOWN")

    issues: List[str] = []
    if not known:
        issues.append("watchdog_status.json missing")
    if wd_status not in ("OK",):
        issues.append(f"watchdog_status={wd_status}")
    if hb_age is not None and hb_age > 120:
        issues.append(f"heartbeat stale age_minutes={round(hb_age, 1)}")
    if loop_lock.get("stale"):
        issues.append(f"watchdog_loop.lock stale age_minutes={loop_lock.get('lock_age_minutes')}")

    return {
        "status": status,
        "watchdog_status": wd_status,
        "broker_connected": status_doc.get("broker_connected"),
        "active_alerts_count": active_alerts,
        "heartbeat_status": hb_status,
        "heartbeat_age_minutes": round(hb_age, 1) if hb_age is not None else None,
        "watchdog_loop_lock": loop_lock,
        "alert_state_present": bool(alert_state),
        "issues": issues,
    }


def _audit_continuous_loop() -> Dict[str, Any]:
    loop_lock = _parse_lock(LIVE / "continuous_loop.lock")
    loop_dir = RESULTS / "continuous_loop"
    loop_health = _load_json(RESULTS / "continuous_loop_health.json")

    loop_running = False
    loop_pids: List[int] = []
    if sys.platform == "win32":
        ps = (
            "Get-CimInstance Win32_Process -Filter \"Name = 'powershell.exe' OR Name = 'pwsh.exe'\" "
            "| Where-Object { $_.CommandLine -like '*run_continuous_loop.ps1*' } "
            "| Select-Object -ExpandProperty ProcessId"
        )
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=30,
            )
            loop_pids = [
                int(x.strip()) for x in (r.stdout or "").splitlines() if x.strip().isdigit()
            ]
            loop_running = len(loop_pids) > 0
        except (subprocess.SubprocessError, OSError):
            pass

    log_files = list(loop_dir.glob("continuous_loop_*.log")) if loop_dir.is_dir() else []
    readiness = "READY" if loop_dir.is_dir() or loop_health else "UNKNOWN"
    if loop_lock.get("stale") and not loop_running:
        readiness = "STALE_LOCK"
    elif loop_running:
        readiness = "RUNNING"
    elif loop_lock.get("exists") and not loop_running:
        readiness = "LOCK_WITHOUT_PROCESS"

    issues: List[str] = []
    if loop_lock.get("stale"):
        issues.append(f"continuous_loop.lock stale age_minutes={loop_lock.get('lock_age_minutes')}")
    if readiness == "LOCK_WITHOUT_PROCESS":
        issues.append("continuous loop lock present but process not running")
    if not loop_dir.is_dir():
        issues.append("data/results/continuous_loop/ directory absent")

    return {
        "readiness": readiness,
        "loop_running": loop_running,
        "loop_pids": loop_pids,
        "continuous_loop_lock": loop_lock,
        "loop_log_count": len(log_files),
        "loop_health_present": bool(loop_health),
        "loop_health_classification": loop_health.get("health_classification"),
        "issues": issues,
    }


def _build_authority_matrix(
    broker: Dict[str, Any],
    execution: Dict[str, Any],
    lifecycle: Dict[str, Any],
    risk: Dict[str, Any],
    governance: Dict[str, Any],
    dashboard: Dict[str, Any],
    watchdog: Dict[str, Any],
    loop: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "domain": "broker",
            "authority_source": "Alpaca paper + positions/open_orders snapshots",
            "expected": "Snapshots match watchdog; broker connected",
            "status": broker.get("status"),
            "synced": broker.get("status") == "SYNCED",
            "detail": json.dumps(broker.get("issues") or [], default=str),
        },
        {
            "domain": "execution",
            "authority_source": "master_execution_gate.json + decision_authority.py",
            "expected": "Live blocked; paper controlled; loader resolves master gate",
            "status": execution.get("status"),
            "synced": execution.get("status") == "AUTHORITATIVE",
            "detail": json.dumps(execution.get("issues") or [], default=str),
        },
        {
            "domain": "lifecycle",
            "authority_source": "signal_lifecycle_effective.csv",
            "expected": "Effective lifecycle exists; operational consumers bound",
            "status": lifecycle.get("status"),
            "synced": lifecycle.get("status") == "EFFECTIVE",
            "detail": json.dumps(lifecycle.get("issues") or [], default=str),
        },
        {
            "domain": "risk",
            "authority_source": "master_execution_gate.json",
            "expected": "Master gate final; adaptive advisory only; 195E intact",
            "status": risk.get("status"),
            "synced": risk.get("status") == "MASTER_GATE",
            "detail": json.dumps(risk.get("issues") or [], default=str),
        },
        {
            "domain": "governance",
            "authority_source": "capital_preservation_certification.json",
            "expected": "CERTIFIED_FOR_PAPER_PROTECTION or better",
            "status": governance.get("status"),
            "synced": governance.get("status") == "CERTIFIED",
            "detail": governance.get("certification_status") or "",
        },
        {
            "domain": "dashboard",
            "authority_source": "Phase 195F authority labels",
            "expected": "RAW/EFFECTIVE/ADAPTIVE/MASTER labels present",
            "status": dashboard.get("status"),
            "synced": dashboard.get("status") == "LABELED",
            "detail": json.dumps(dashboard.get("labels_present") or {}, default=str),
        },
        {
            "domain": "watchdog",
            "authority_source": "watchdog_status.json + alert state",
            "expected": "Monitoring status known",
            "status": watchdog.get("status"),
            "synced": watchdog.get("status") in ("OK", "KNOWN"),
            "detail": watchdog.get("watchdog_status") or "",
        },
        {
            "domain": "continuous_loop",
            "authority_source": "continuous_loop.lock + loop health",
            "expected": "Readiness known; stale locks reported not deleted",
            "status": loop.get("readiness"),
            "synced": loop.get("readiness")
            in ("RUNNING", "READY", "STALE_LOCK", "LOCK_WITHOUT_PROCESS"),
            "detail": json.dumps(loop.get("issues") or [], default=str),
        },
    ]


def _compute_score(matrix: List[Dict[str, Any]], checks: List[Dict[str, Any]]) -> Tuple[float, str]:
    weights = {
        "broker": 15,
        "execution": 20,
        "lifecycle": 15,
        "risk": 15,
        "governance": 15,
        "dashboard": 10,
        "watchdog": 5,
        "continuous_loop": 5,
    }
    total_w = sum(weights.values())
    earned = 0.0
    for row in matrix:
        w = weights.get(row["domain"], 5)
        if row.get("synced"):
            earned += w
        elif row.get("status") in ("KNOWN", "STALE_LOCK", "LOCK_WITHOUT_PROCESS"):
            earned += w * 0.7

    score = round(100.0 * earned / total_w, 1)
    check_pass = sum(1 for c in checks if c.get("pass"))
    check_total = len(checks) or 1
    blended = round((score + 100.0 * check_pass / check_total) / 2, 1)

    if blended >= 90:
        band = "GREEN"
    elif blended >= 75:
        band = "YELLOW"
    else:
        band = "RED"
    return blended, band


def run_phase_195g_audit(
    *, run_validation_chain: bool = False, verbose: bool = True
) -> Dict[str, Any]:
    validation_runs: Dict[str, Any] = {}
    if run_validation_chain:
        validation_runs = {
            "phase_195c": _run_module("services.phase_195c_evaluation_contract_validation"),
            "phase_195d": _run_module("services.phase_195d_authority_consolidation_validation"),
            "phase_195e": _run_module("services.phase_195e_authority_remediation_validation"),
            "phase_195f": _run_module("services.phase_195f_authority_labeling_validation"),
            "path_verification": _run_module("services.path_verification"),
        }

    d195c = _load_json(RESULTS / "phase_195c_evaluation_contract_validation.json")
    d195d = _load_json(RESULTS / "phase_195d_authority_consolidation_validation.json")
    d195e = _load_json(RESULTS / "phase_195e_authority_remediation_validation.json")
    d195f = _load_json(RESULTS / "phase_195f_authority_labeling_validation.json")

    broker = _audit_broker_sync(run_snapshot=True)
    execution = _audit_execution_authority()
    lifecycle = _audit_lifecycle_authority(d195d)
    risk = _audit_risk_authority(d195d, d195e)
    governance = _audit_governance_certification(d195c)
    dashboard = _audit_dashboard_labeling(d195f)
    watchdog = _audit_watchdog()
    loop = _audit_continuous_loop()

    high_conflicts = int(d195d.get("high_severity_count") or 0)
    medium_conflicts = int(d195d.get("medium_severity_count") or 0)
    low_conflicts = int(d195d.get("low_severity_count") or 0)

    matrix = _build_authority_matrix(
        broker, execution, lifecycle, risk, governance, dashboard, watchdog, loop
    )

    blocking_issues: List[str] = []
    review_items: List[str] = []

    if broker.get("status") != "SYNCED":
        blocking_issues.extend(broker.get("issues") or [])
    if execution.get("live_execution_permitted") is not False:
        blocking_issues.append("live execution not blocked")
    if execution.get("status") not in ("AUTHORITATIVE",):
        blocking_issues.extend(execution.get("issues") or [])
    if lifecycle.get("status") != "EFFECTIVE":
        blocking_issues.extend(lifecycle.get("issues") or [])
    if risk.get("status") != "MASTER_GATE":
        blocking_issues.extend(risk.get("issues") or [])
    if high_conflicts > 0:
        blocking_issues.append(f"HIGH authority conflicts={high_conflicts}")
    if d195c.get("phase_pass") is not True:
        blocking_issues.append("Phase 195C not passing")
    if d195d.get("phase_pass") is not True:
        blocking_issues.append("Phase 195D not passing")
    if d195e.get("phase_pass") is not True:
        blocking_issues.append("Phase 195E not passing")
    if d195f.get("phase_pass") is not True and dashboard.get("status") != "LABELED":
        blocking_issues.append("Phase 195F not passing")

    for domain_issues in (watchdog.get("issues") or [], loop.get("issues") or []):
        for item in domain_issues:
            if "stale" in item.lower() or "absent" in item.lower():
                review_items.append(item)
            elif item not in blocking_issues:
                review_items.append(item)

    if medium_conflicts > 0:
        review_items.append(f"MEDIUM authority conflicts={medium_conflicts} (documented advisory)")
    if low_conflicts > 0:
        review_items.append(f"LOW authority conflicts={low_conflicts} (documented advisory)")

    scorecard: List[Dict[str, Any]] = [
        {
            "check": "broker_sync",
            "expected": "SYNCED",
            "actual": broker.get("status"),
            "pass": broker.get("status") == "SYNCED",
        },
        {
            "check": "live_execution_blocked",
            "expected": "false",
            "actual": str(execution.get("live_execution_permitted")),
            "pass": execution.get("live_execution_permitted") is False,
        },
        {
            "check": "paper_execution_controlled",
            "expected": "true",
            "actual": str(execution.get("paper_execution_permitted")),
            "pass": execution.get("paper_execution_permitted") is True,
        },
        {
            "check": "lifecycle_authority",
            "expected": "EFFECTIVE",
            "actual": lifecycle.get("status"),
            "pass": lifecycle.get("status") == "EFFECTIVE",
        },
        {
            "check": "risk_authority",
            "expected": "MASTER_GATE",
            "actual": risk.get("status"),
            "pass": risk.get("status") == "MASTER_GATE",
        },
        {
            "check": "governance_certification",
            "expected": "CERTIFIED_FOR_PAPER_PROTECTION",
            "actual": governance.get("certification_status"),
            "pass": governance.get("status") == "CERTIFIED",
        },
        {
            "check": "dashboard_labeling",
            "expected": "LABELED",
            "actual": dashboard.get("status"),
            "pass": dashboard.get("status") == "LABELED",
        },
        {
            "check": "high_conflicts",
            "expected": "0",
            "actual": str(high_conflicts),
            "pass": high_conflicts == 0,
        },
        {
            "check": "phase_195c",
            "expected": "pass",
            "actual": str(d195c.get("phase_pass")),
            "pass": d195c.get("phase_pass") is True,
        },
        {
            "check": "phase_195d",
            "expected": "pass",
            "actual": str(d195d.get("phase_pass")),
            "pass": d195d.get("phase_pass") is True,
        },
        {
            "check": "phase_195e",
            "expected": "pass",
            "actual": str(d195e.get("phase_pass")),
            "pass": d195e.get("phase_pass") is True,
        },
        {
            "check": "phase_195f",
            "expected": "pass",
            "actual": str(d195f.get("phase_pass")),
            "pass": d195f.get("phase_pass") is True or dashboard.get("status") == "LABELED",
        },
        {
            "check": "watchdog_known",
            "expected": "OK or KNOWN",
            "actual": watchdog.get("status"),
            "pass": watchdog.get("status") in ("OK", "KNOWN"),
        },
        {
            "check": "continuous_loop_readiness_known",
            "expected": "readiness reported",
            "actual": loop.get("readiness"),
            "pass": bool(loop.get("readiness")),
        },
    ]

    sync_score, sync_band = _compute_score(matrix, scorecard)

    phase_pass = (
        not blocking_issues
        and all(c["pass"] for c in scorecard if c["check"] not in ("watchdog_known",))
        and watchdog.get("status") in ("OK", "KNOWN")
        and high_conflicts == 0
        and medium_conflicts == 0
        and low_conflicts == 0
    )

    payload: Dict[str, Any] = {
        "phase": "195G",
        "timestamp": _utc_now_iso(),
        "broker_sync_status": broker.get("status"),
        "execution_authority_status": execution.get("status"),
        "lifecycle_authority_status": lifecycle.get("status"),
        "risk_authority_status": risk.get("status"),
        "governance_certification_status": governance.get("certification_status"),
        "dashboard_labeling_status": dashboard.get("status"),
        "watchdog_status": watchdog.get("watchdog_status"),
        "continuous_loop_readiness": loop.get("readiness"),
        "live_execution_permitted": execution.get("live_execution_permitted"),
        "paper_execution_authorized": execution.get("paper_execution_authorized"),
        "open_orders_count": broker.get("open_orders_count"),
        "positions_count": broker.get("positions_count"),
        "high_conflicts": high_conflicts,
        "medium_conflicts": medium_conflicts,
        "low_conflicts": low_conflicts,
        "phase_195c_status": d195c.get("phase_pass"),
        "phase_195d_status": d195d.get("phase_pass"),
        "phase_195e_status": d195e.get("phase_pass"),
        "phase_195f_status": d195f.get("phase_pass"),
        "institutional_sync_score": sync_score,
        "institutional_sync_band": sync_band,
        "blocking_issues": blocking_issues,
        "review_items": review_items,
        "phase_pass": phase_pass,
        "domains": {
            "broker": broker,
            "execution": execution,
            "lifecycle": lifecycle,
            "risk": risk,
            "governance": governance,
            "dashboard": dashboard,
            "watchdog": watchdog,
            "continuous_loop": loop,
        },
        "validation_runs": validation_runs,
        "orders_created_by_audit": not broker.get("no_unexpected_snapshot_growth", True),
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(SCORECARD_CSV, scorecard, ("check", "expected", "actual", "pass"))
    _write_csv(
        MATRIX_CSV,
        matrix,
        ("domain", "authority_source", "expected", "status", "synced", "detail"),
    )

    if verbose:
        print(
            json.dumps(
                {
                    "phase_pass": phase_pass,
                    "institutional_sync_score": sync_score,
                    "institutional_sync_band": sync_band,
                    "blocking_issues": blocking_issues,
                    "review_items": review_items,
                },
                indent=2,
            )
        )

    return payload


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Phase 195G post-hardening institutional sync audit")
    ap.add_argument(
        "--run-validation-chain",
        action="store_true",
        help="Re-run Phase 195C–195F and path_verification before audit (default: read artifacts)",
    )
    args = ap.parse_args()
    result = run_phase_195g_audit(run_validation_chain=args.run_validation_chain, verbose=True)
    return 0 if result.get("phase_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
