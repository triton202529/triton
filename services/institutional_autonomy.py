"""
TRITON Institutional Autonomy — Phases 22–24.

Capital preservation audit, stress test lab, and certification engine.
Paper-mode / simulation only. NO live trading, orders, or portfolio changes.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

CERTIFICATION_AREAS = (
    "Monitoring",
    "Alerting",
    "Governance",
    "Authorization",
    "Readiness",
    "Simulation",
    "Evaluation",
    "Governor",
)

GOVERNANCE_BLOCK_LEVELS = frozenset(
    {
        "MANAGEMENT_REVIEW_REQUIRED",
        "BOARD_REVIEW_REQUIRED",
        "CRITICAL_INTERVENTION",
    }
)

STRESS_SCENARIOS = (
    "Extreme concentration",
    "Extreme drawdown",
    "Broker outage",
    "Data freshness failure",
    "Governance failure",
    "Multiple simultaneous incidents",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: Optional[datetime] = None) -> str:
    t = dt or _utc_now()
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _stress_result(score: float) -> str:
    if score >= 70:
        return "PASS"
    if score >= 45:
        return "WARN"
    return "FAIL"


def _audit_event(
    *,
    audit_timestamp: str,
    event_type: str,
    event_source: str,
    event_result: str,
    details: Optional[str] = None,
) -> Dict[str, Any]:
    ev: Dict[str, Any] = {
        "audit_timestamp": audit_timestamp,
        "event_type": event_type,
        "event_source": event_source,
        "event_result": event_result,
    }
    if details:
        ev["details"] = details
    return ev


def compute_capital_preservation_audit(
    *,
    alerts_doc: Dict[str, Any],
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpa_doc: Dict[str, Any],
    cpd_doc: Dict[str, Any],
    sim_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    auth_doc: Optional[Dict[str, Any]] = None,
    readiness_doc: Optional[Dict[str, Any]] = None,
    trials_doc: Optional[Dict[str, Any]] = None,
    adaptive_doc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 22: aggregate audit trail from preservation stack artifacts."""
    ts = _iso_utc()
    auth_doc = auth_doc or {}
    readiness_doc = readiness_doc or {}
    trials_doc = trials_doc or {}
    adaptive_doc = adaptive_doc or {}
    events: List[Dict[str, Any]] = []

    alert_ts = str(alerts_doc.get("generated_at") or ts)
    for alert in alerts_doc.get("active_alerts") or []:
        if not isinstance(alert, dict):
            continue
        events.append(
            _audit_event(
                audit_timestamp=alert_ts,
                event_type="ALERT",
                event_source="Watchdog Alert Engine",
                event_result=str(alert.get("severity") or "UNKNOWN"),
                details=str(alert.get("alert_type") or alert.get("alert_id") or ""),
            )
        )

    cpe_ts = str(cpe_doc.get("generated_at") or ts)
    events.append(
        _audit_event(
            audit_timestamp=cpe_ts,
            event_type="ESCALATION",
            event_source="Capital Preservation Escalation",
            event_result=str(cpe_doc.get("escalation_state") or "UNKNOWN"),
            details=", ".join(cpe_doc.get("escalation_reason_labels") or []) or None,
        )
    )

    cpa_ts = str(cpa_doc.get("generated_at") or ts)
    for adv in cpa_doc.get("advisories") or []:
        if not isinstance(adv, dict):
            continue
        events.append(
            _audit_event(
                audit_timestamp=cpa_ts,
                event_type="ADVISORY",
                event_source="Capital Preservation Advisory",
                event_result=str(adv.get("priority") or "UNKNOWN"),
                details=str(adv.get("title") or adv.get("reason") or ""),
            )
        )

    cpd_ts = str(cpd_doc.get("generated_at") or ts)
    for item in cpd_doc.get("decision_support_items") or []:
        if not isinstance(item, dict):
            continue
        events.append(
            _audit_event(
                audit_timestamp=cpd_ts,
                event_type="DECISION_SUPPORT",
                event_source="Capital Preservation Decision Support",
                event_result=str(item.get("priority") or "UNKNOWN"),
                details=str(item.get("issue") or ""),
            )
        )

    sim_ts = str(sim_doc.get("generated_at") or ts)
    for sim in sim_doc.get("simulations") or []:
        if not isinstance(sim, dict):
            continue
        risk_delta = _safe_float(sim.get("risk_score_delta"), 0.0) or 0.0
        result = (
            "BENEFICIAL" if risk_delta >= 10 else ("NEUTRAL" if risk_delta >= 0 else "NEGATIVE")
        )
        events.append(
            _audit_event(
                audit_timestamp=sim_ts,
                event_type="SIMULATION",
                event_source="Defensive Simulation Lab",
                event_result=result,
                details=str(sim.get("simulation_name") or sim.get("simulation_type") or ""),
            )
        )

    eval_ts = str(evaluation_doc.get("generated_at") or ts)
    for ev in evaluation_doc.get("evaluations") or []:
        if not isinstance(ev, dict):
            continue
        events.append(
            _audit_event(
                audit_timestamp=eval_ts,
                event_type="EVALUATION",
                event_source="Protective Action Evaluation",
                event_result=str(ev.get("evaluation") or "UNKNOWN"),
                details=str(ev.get("trial_name") or ev.get("trial_id") or ""),
            )
        )

    if trials_doc.get("trial_count", 0):
        events.append(
            _audit_event(
                audit_timestamp=str(trials_doc.get("generated_at") or ts),
                event_type="TRIAL",
                event_source="Protective Action Trials",
                event_result="SIMULATION_ONLY",
                details=f"{trials_doc.get('trial_count', 0)} trials completed",
            )
        )

    if adaptive_doc.get("best_protection"):
        events.append(
            _audit_event(
                audit_timestamp=str(adaptive_doc.get("generated_at") or ts),
                event_type="ADAPTIVE_LEARNING",
                event_source="Adaptive Capital Preservation",
                event_result=str(adaptive_doc.get("best_protection") or "UNKNOWN"),
                details=str(adaptive_doc.get("learning_summary") or "")[:120] or None,
            )
        )

    gov_ts = str(governor_doc.get("generated_at") or ts)
    events.append(
        _audit_event(
            audit_timestamp=gov_ts,
            event_type="GOVERNOR",
            event_source="Capital Preservation Governor",
            event_result=str(governor_doc.get("preservation_posture") or "UNKNOWN"),
            details=", ".join(governor_doc.get("top_drivers") or [])[:120] or None,
        )
    )

    cpi_ts = str(cpi_doc.get("generated_at") or ts)
    events.append(
        _audit_event(
            audit_timestamp=cpi_ts,
            event_type="MONITORING",
            event_source="Capital Preservation Intelligence",
            event_result=str(cpi_doc.get("health_band") or "UNKNOWN"),
            details=f"CPS={cpi_doc.get('capital_preservation_score')} trend={cpi_doc.get('risk_trend')}",
        )
    )

    if auth_doc.get("generated_at"):
        events.append(
            _audit_event(
                audit_timestamp=str(auth_doc.get("generated_at")),
                event_type="AUTHORIZATION",
                event_source="Governance Authorization",
                event_result="AUTHORIZED" if auth_doc.get("overall_authorization") else "DENIED",
                details=", ".join(k for k, v in (auth_doc.get("gate_reasons") or {}).items() if v)[
                    :120
                ]
                or None,
            )
        )

    if readiness_doc.get("generated_at"):
        events.append(
            _audit_event(
                audit_timestamp=str(readiness_doc.get("generated_at")),
                event_type="READINESS",
                event_source="Execution Readiness",
                event_result=str(readiness_doc.get("readiness_status") or "UNKNOWN"),
                details=", ".join(readiness_doc.get("failed_checks") or [])[:120] or None,
            )
        )

    events.sort(key=lambda e: e.get("audit_timestamp") or "", reverse=True)

    summary_by_type: Dict[str, int] = {}
    for ev in events:
        et = str(ev.get("event_type") or "UNKNOWN")
        summary_by_type[et] = summary_by_type.get(et, 0) + 1

    latest = events[0] if events else None

    return {
        "generated_at": ts,
        "event_count": len(events),
        "events": events,
        "summary_by_event_type": summary_by_type,
        "latest_event_type": latest.get("event_type") if latest else None,
        "latest_event_result": latest.get("event_result") if latest else None,
        "latest_event_source": latest.get("event_source") if latest else None,
        "disclaimer": "Audit trail only. No trades, orders, or portfolio modifications.",
    }


def _largest_concentration_pct(positions: List[Dict[str, Any]]) -> float:
    if not positions:
        return 0.0
    total_mv = sum(_safe_float(p.get("market_value"), 0.0) or 0.0 for p in positions)
    if total_mv <= 0:
        return 0.0
    largest = max(_safe_float(p.get("market_value"), 0.0) or 0.0 for p in positions)
    return round((largest / total_mv) * 100.0, 2)


def _worst_drawdown_pct(positions: List[Dict[str, Any]]) -> float:
    worst = 0.0
    for p in positions:
        pl = _safe_float(p.get("unrealized_pl_pct"))
        if pl is not None and pl < worst:
            worst = pl
    return round(worst, 2)


def _alert_types(active_alerts: List[Dict[str, Any]]) -> set:
    return {str(a.get("alert_type") or "") for a in active_alerts if isinstance(a, dict)}


def compute_stress_test_results(
    *,
    positions: List[Dict[str, Any]],
    active_alerts: List[Dict[str, Any]],
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    sim_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    broker_connected: bool,
    watchdog_status: str,
) -> Dict[str, Any]:
    """Phase 23: counterfactual stress scenarios (simulation only)."""
    ts = _iso_utc()
    alert_types = _alert_types(active_alerts)
    conc_pct = _largest_concentration_pct(positions)
    worst_dd = _worst_drawdown_pct(positions)
    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    escalation = str(cpe_doc.get("escalation_state") or "GREEN").upper()
    gov_level = str(gov_doc.get("governance_awareness_level") or "").upper()
    readiness_status = str(readiness_doc.get("readiness_status") or "NOT_READY")
    sim_count = len(sim_doc.get("simulations") or [])
    eval_count = evaluation_doc.get("evaluation_count", 0)
    beneficial = evaluation_doc.get("beneficial_count", 0)
    posture = str(governor_doc.get("preservation_posture") or "UNKNOWN")
    incident_count = len(active_alerts)

    scenarios: List[Dict[str, Any]] = []

    # 1. Extreme concentration
    conc_detected = "EXCESS_CONCENTRATION" in alert_types or conc_pct >= 50
    conc_escalated = escalation in ("RED", "ORANGE", "CRITICAL") and conc_pct >= 40
    conc_sim = any(
        isinstance(s, dict) and s.get("simulation_type") == "concentration_cap"
        for s in (sim_doc.get("simulations") or [])
    )
    conc_score = 0
    conc_score += 35 if conc_detected else 0
    conc_score += 25 if conc_escalated else 10
    conc_score += 20 if conc_sim else 0
    conc_score += min(20, max(0, 100 - int(conc_pct)))
    conc_score = int(min(100, conc_score))
    scenarios.append(
        {
            "scenario_name": "Extreme concentration",
            "survivability_score": conc_score,
            "result": _stress_result(conc_score),
            "details": (
                f"Largest position {conc_pct}% of portfolio. "
                f"Alert detected={conc_detected}, escalation={escalation}, "
                f"simulations_available={conc_sim}."
            ),
        }
    )

    # 2. Extreme drawdown
    dd_detected = "EXCESS_POSITION_DRAWDOWN" in alert_types or worst_dd <= -10
    dd_eval = eval_count > 0 and beneficial >= 1
    dd_score = 0
    dd_score += 40 if dd_detected else 0
    dd_score += 25 if escalation in ("RED", "ORANGE", "YELLOW") and dd_detected else 10
    dd_score += 20 if dd_eval else 0
    dd_score += min(15, max(0, 15 + int(worst_dd)))
    dd_score = int(min(100, dd_score))
    scenarios.append(
        {
            "scenario_name": "Extreme drawdown",
            "survivability_score": dd_score,
            "result": _stress_result(dd_score),
            "details": (
                f"Worst unrealized P/L {worst_dd}%. "
                f"Alert detected={dd_detected}, beneficial_trials={beneficial}."
            ),
        }
    )

    # 3. Broker outage
    outage_alert = "BROKER_DISCONNECT" in alert_types
    if not broker_connected or outage_alert:
        outage_score = 70 if outage_alert else 45
        outage_score += 15 if cpe_doc.get("escalation_state") in ("RED", "ORANGE") else 5
        outage_score += 10 if readiness_doc.get("checks", {}).get("broker") is False else 0
    else:
        outage_score = 85
    outage_score = int(min(100, outage_score))
    scenarios.append(
        {
            "scenario_name": "Broker outage",
            "survivability_score": outage_score,
            "result": _stress_result(outage_score),
            "details": (
                f"broker_connected={broker_connected}, outage_alert={outage_alert}, "
                f"watchdog_status={watchdog_status}."
            ),
        }
    )

    # 4. Data freshness failure
    stale_alert = "STALE_HEARTBEAT" in alert_types
    data_fresh = readiness_doc.get("checks", {}).get("data_freshness", False)
    fresh_score = 0
    fresh_score += 40 if stale_alert else 20
    fresh_score += 30 if not data_fresh else 10
    fresh_score += 20 if cpe_doc.get("escalation_state") in ("RED", "ORANGE", "YELLOW") else 5
    fresh_score += 10 if cpi_doc.get("generated_at") else 0
    fresh_score = int(min(100, fresh_score))
    scenarios.append(
        {
            "scenario_name": "Data freshness failure",
            "survivability_score": fresh_score,
            "result": _stress_result(fresh_score),
            "details": (
                f"stale_heartbeat_alert={stale_alert}, "
                f"data_freshness_check={data_fresh}, "
                f"failed_checks={readiness_doc.get('failed_checks') or []}."
            ),
        }
    )

    # 5. Governance failure
    gov_blocked = gov_level in GOVERNANCE_BLOCK_LEVELS
    auth_blocked = not auth_doc.get("overall_authorization", False)
    gov_score = 0
    gov_score += 35 if gov_blocked else 10
    gov_score += 25 if gov_doc.get("governance_status") else 5
    gov_score += 20 if auth_blocked else 0
    gov_score += 20 if posture in ("RED", "ORANGE", "CRITICAL") else 10
    gov_score = int(min(100, gov_score))
    scenarios.append(
        {
            "scenario_name": "Governance failure",
            "survivability_score": gov_score,
            "result": _stress_result(gov_score),
            "details": (
                f"governance_level={gov_level}, authorization_blocked={auth_blocked}, "
                f"preservation_posture={posture}."
            ),
        }
    )

    # 6. Multiple simultaneous incidents
    multi_score = 0
    multi_score += min(40, incident_count * 12)
    multi_score += 20 if escalation in ("RED", "CRITICAL") else 10
    multi_score += 15 if sim_count >= 3 else 5
    multi_score += 15 if eval_count >= 2 else 5
    multi_score += 10 if governor_doc.get("governor_confidence", 0) >= 60 else 0
    multi_score = int(min(100, multi_score))
    scenarios.append(
        {
            "scenario_name": "Multiple simultaneous incidents",
            "survivability_score": multi_score,
            "result": _stress_result(multi_score),
            "details": (
                f"active_incidents={incident_count}, escalation={escalation}, "
                f"simulations={sim_count}, evaluations={eval_count}."
            ),
        }
    )

    pass_count = sum(1 for s in scenarios if s["result"] == "PASS")
    warn_count = sum(1 for s in scenarios if s["result"] == "WARN")
    fail_count = sum(1 for s in scenarios if s["result"] == "FAIL")
    avg_score = (
        round(sum(s["survivability_score"] for s in scenarios) / len(scenarios), 1)
        if scenarios
        else 0.0
    )

    return {
        "generated_at": ts,
        "scenario_count": len(scenarios),
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "average_survivability": avg_score,
        "scenarios": scenarios,
        "baseline_cps": cps,
        "baseline_escalation": escalation,
        "disclaimer": "Stress tests are counterfactual simulations only. No broker actions.",
    }


def _area_result(certified: bool, score: int, notes: str) -> Dict[str, Any]:
    return {"certified": certified, "score": score, "notes": notes}


def compute_preservation_certification(
    *,
    cpi_doc: Dict[str, Any],
    alerts_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    sim_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 24: score certification areas from existing artifacts."""
    ts = _iso_utc()
    areas: Dict[str, Dict[str, Any]] = {}
    failed: List[str] = []

    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    monitoring_score = int(min(100, max(0, cps)))
    monitoring_cert = bool(cpi_doc.get("generated_at")) and cps >= 30
    areas["Monitoring"] = _area_result(
        monitoring_cert,
        monitoring_score,
        f"CPS={cps}, band={cpi_doc.get('health_band')}, trend={cpi_doc.get('risk_trend')}",
    )

    active_count = len(alerts_doc.get("active_alerts") or [])
    alert_score = 70 if alerts_doc.get("generated_at") else 20
    if active_count > 0:
        alert_score = min(100, alert_score + 10)
    alert_cert = bool(alerts_doc.get("generated_at"))
    areas["Alerting"] = _area_result(
        alert_cert,
        alert_score,
        f"{active_count} active alerts tracked",
    )

    gov_level = str(gov_doc.get("governance_awareness_level") or "").upper()
    gov_score = 80 if gov_level == "GREEN" else (60 if gov_level == "YELLOW" else 35)
    gov_cert = gov_level not in GOVERNANCE_BLOCK_LEVELS and bool(gov_doc.get("generated_at"))
    areas["Governance"] = _area_result(
        gov_cert,
        gov_score,
        f"awareness={gov_level}, status={gov_doc.get('governance_status')}",
    )

    auth_score = 90 if auth_doc.get("overall_authorization") else 25
    auth_cert = bool(
        auth_doc.get("overall_authorization")
        and auth_doc.get("execution_authorized")
        and auth_doc.get("governance_authorized")
    )
    areas["Authorization"] = _area_result(
        auth_cert,
        auth_score,
        f"overall={auth_doc.get('overall_authorization')}, execution={auth_doc.get('execution_authorized')}",
    )

    passing = readiness_doc.get("checks_passing_count", 0)
    total = readiness_doc.get("checks_total", 8) or 8
    readiness_score = int(round((passing / total) * 100)) if total else 0
    readiness_cert = (
        readiness_doc.get("readiness_status") == "READY"
        and not readiness_doc.get("live_execution_permitted")
        and readiness_score >= 75
    )
    areas["Readiness"] = _area_result(
        readiness_cert,
        readiness_score,
        f"status={readiness_doc.get('readiness_status')}, {passing}/{total} checks passing",
    )

    sim_count = len(sim_doc.get("simulations") or [])
    sim_score = min(100, 40 + sim_count * 8) if sim_doc.get("generated_at") else 0
    sim_cert = sim_count >= 3 and bool(sim_doc.get("generated_at"))
    areas["Simulation"] = _area_result(
        sim_cert,
        sim_score,
        f"{sim_count} defensive simulations on file",
    )

    eval_count = evaluation_doc.get("evaluation_count", 0)
    avg_eff = _safe_float(evaluation_doc.get("average_effectiveness"), 0.0) or 0.0
    eval_score = int(min(100, avg_eff)) if eval_count else 0
    eval_cert = eval_count >= 2 and avg_eff >= 50
    areas["Evaluation"] = _area_result(
        eval_cert,
        eval_score,
        f"{eval_count} trials evaluated, avg effectiveness={avg_eff}",
    )

    gov_conf = int(governor_doc.get("governor_confidence") or 0)
    posture = str(governor_doc.get("preservation_posture") or "UNKNOWN")
    governor_score = gov_conf
    governor_cert = bool(governor_doc.get("generated_at")) and posture not in ("CRITICAL",)
    areas["Governor"] = _area_result(
        governor_cert,
        governor_score,
        f"posture={posture}, confidence={gov_conf}%",
    )

    for name, area in areas.items():
        if not area["certified"]:
            failed.append(name)

    certification_score = (
        round(sum(a["score"] for a in areas.values()) / len(areas), 1) if areas else 0.0
    )
    certified_count = sum(1 for a in areas.values() if a["certified"])

    if (
        auth_doc.get("execution_authorized")
        or readiness_doc.get("live_execution_permitted")
        or auth_doc.get("overall_authorization")
    ):
        certification_status = "NOT_CERTIFIED"
    elif certified_count == len(CERTIFICATION_AREAS) and certification_score >= 80:
        certification_status = "CERTIFIED_FOR_PAPER_PROTECTION"
    elif certified_count >= 4 and certification_score >= 55:
        certification_status = "PARTIALLY_CERTIFIED"
    else:
        certification_status = "NOT_CERTIFIED"

    if auth_doc.get("execution_authorized") or readiness_doc.get("live_execution_permitted"):
        if "Live execution gates must remain blocked" not in failed:
            failed.insert(0, "Live execution gates must remain blocked")

    return {
        "generated_at": ts,
        "certification_status": certification_status,
        "certification_score": certification_score,
        "certified_area_count": certified_count,
        "total_areas": len(CERTIFICATION_AREAS),
        "areas": areas,
        "failed_requirements": failed,
        "escalation_state": cpe_doc.get("escalation_state"),
        "disclaimer": "Certification assesses paper-mode protection only. No live execution.",
    }


def _append_audit_history(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "event_count",
        "latest_event_type",
        "latest_event_result",
        "alert_events",
        "escalation_events",
        "governor_events",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def _append_certification_history(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "certification_status",
        "certification_score",
        "certified_area_count",
        "failed_requirement_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def persist_institutional_autonomy(
    *,
    results_dir: Path,
    positions: List[Dict[str, Any]],
    active_alerts: List[Dict[str, Any]],
    alerts_doc: Dict[str, Any],
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpa_doc: Dict[str, Any],
    cpd_doc: Dict[str, Any],
    sim_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    evaluation_doc: Dict[str, Any],
    adaptive_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    trials_doc: Dict[str, Any],
    broker_connected: bool,
    watchdog_status: str,
) -> Dict[str, Any]:
    """Run phases 22–24 and write JSON artifacts."""
    results_dir = Path(results_dir)

    audit_doc = compute_capital_preservation_audit(
        alerts_doc=alerts_doc,
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        cpa_doc=cpa_doc,
        cpd_doc=cpd_doc,
        sim_doc=sim_doc,
        evaluation_doc=evaluation_doc,
        governor_doc=governor_doc,
        auth_doc=auth_doc,
        readiness_doc=readiness_doc,
        trials_doc=trials_doc,
        adaptive_doc=adaptive_doc,
    )
    _atomic_write_json(audit_doc, results_dir / "capital_preservation_audit.json")

    stress_doc = compute_stress_test_results(
        positions=positions,
        active_alerts=active_alerts,
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        gov_doc=gov_doc,
        readiness_doc=readiness_doc,
        auth_doc=auth_doc,
        sim_doc=sim_doc,
        evaluation_doc=evaluation_doc,
        governor_doc=governor_doc,
        broker_connected=broker_connected,
        watchdog_status=watchdog_status,
    )
    _atomic_write_json(stress_doc, results_dir / "stress_test_results.json")

    cert_doc = compute_preservation_certification(
        cpi_doc=cpi_doc,
        alerts_doc=alerts_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        readiness_doc=readiness_doc,
        sim_doc=sim_doc,
        evaluation_doc=evaluation_doc,
        governor_doc=governor_doc,
        cpe_doc=cpe_doc,
    )
    _atomic_write_json(cert_doc, results_dir / "capital_preservation_certification.json")

    summary = audit_doc.get("summary_by_event_type") or {}
    _append_audit_history(
        results_dir / "capital_preservation_audit_history.csv",
        {
            "timestamp": audit_doc.get("generated_at"),
            "event_count": audit_doc.get("event_count"),
            "latest_event_type": audit_doc.get("latest_event_type"),
            "latest_event_result": audit_doc.get("latest_event_result"),
            "alert_events": summary.get("ALERT", 0),
            "escalation_events": summary.get("ESCALATION", 0),
            "governor_events": summary.get("GOVERNOR", 0),
        },
    )

    _append_certification_history(
        results_dir / "capital_preservation_certification_history.csv",
        {
            "timestamp": cert_doc.get("generated_at"),
            "certification_status": cert_doc.get("certification_status"),
            "certification_score": cert_doc.get("certification_score"),
            "certified_area_count": cert_doc.get("certified_area_count"),
            "failed_requirement_count": len(cert_doc.get("failed_requirements") or []),
        },
    )

    return {
        "capital_preservation_audit": audit_doc,
        "stress_test_results": stress_doc,
        "capital_preservation_certification": cert_doc,
    }
