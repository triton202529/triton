"""
TRITON Capital Preservation Certification — Phase 24 / 150C / 158 alignment.

Scores certification areas from existing artifacts. Paper-mode protection only.
Phase 158: certification veto framework eliminates false-positive pathways.
NO live trading, orders, or portfolio changes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Phase 158 — certification veto thresholds
CPS_CRITICAL_THRESHOLD = 50
ESCALATION_VETO_STATES = frozenset({"RED", "ORANGE", "CRITICAL"})
PORTFOLIO_CRITICAL_HEALTH = frozenset({"CRITICAL", "HIGH_RISK"})
MANDATORY_PRESERVATION_GATES = frozenset(
    {"Monitoring", "Governor", "Authorization", "Readiness", "Evaluation"}
)

CERTIFICATION_VETO_RULES = (
    "portfolio_health_critical",
    "preservation_status_fail",
    "risk_trend_severely_deteriorating",
    "escalation_state_red",
    "cps_below_critical_threshold",
    "mandatory_preservation_gate_failed",
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

OUTPUT_JSON = RESULTS / "certification_false_positive_elimination.json"
OUTPUT_CSV = RESULTS / "certification_false_positive_scorecard.csv"

SCORECARD_COLUMNS = [
    "scenario",
    "test_type",
    "status",
    "certification_status",
    "detail",
    "generated_at",
]

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: Optional[datetime] = None) -> str:
    t = dt or _utc_now()
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_scorecard_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _area_result(certified: bool, score: int, notes: str) -> Dict[str, Any]:
    return {"certified": certified, "score": score, "notes": notes}


def _resolve_baseline_context_label(evaluation_doc: Dict[str, Any]) -> str:
    """Resolve Phase 150B baseline label from string or structured baseline_context."""
    raw = evaluation_doc.get("baseline_context")
    if isinstance(raw, dict):
        label = raw.get("baseline_context_label") or raw.get("label")
        if label:
            return str(label)
    if isinstance(raw, str) and raw:
        return raw
    label = evaluation_doc.get("baseline_context_label")
    return str(label or "")


def _log_tag(tag: str, fields: Dict[str, Any], *, verbose: bool = True) -> None:
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"[{tag}] {parts}"
    logger.info(line)
    if verbose:
        print(line, flush=True)


def _log_veto(*, rule: str, triggered: bool, reason: str, verbose: bool = True) -> None:
    _log_tag(
        "CERTIFICATION_VETO",
        {"rule": rule, "triggered": triggered, "reason": reason},
        verbose=verbose,
    )


def _live_execution_permitted(
    auth_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
) -> bool:
    """True only when live execution is explicitly permitted (paper auth does not count)."""
    if auth_doc.get("live_execution_permitted") is True:
        return True
    if readiness_doc.get("live_execution_permitted") is True:
        return True
    return False


def _paper_execution_authorized(auth_doc: Dict[str, Any]) -> bool:
    if auth_doc.get("paper_execution_permitted") is True:
        return True
    gate_reasons = auth_doc.get("gate_reasons") or {}
    execution_reason = str(gate_reasons.get("execution") or "").lower()
    if "paper" in execution_reason and auth_doc.get("execution_authorized") is True:
        return True
    return bool(
        auth_doc.get("execution_authorized")
        and auth_doc.get("governance_authorized")
        and auth_doc.get("overall_authorization")
    )


def _resolve_portfolio_health(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    exec_doc: Optional[Dict[str, Any]] = None,
    watchdog_doc: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str]]:
    """Authoritative portfolio health from CPI, escalation, executive, and watchdog artifacts."""
    sources: List[str] = []
    bands: List[str] = []

    for label, raw in (
        ("cpi.health_band", cpi_doc.get("health_band")),
        ("cpe.health_band", cpe_doc.get("health_band")),
    ):
        band = str(raw or "").upper()
        if band:
            bands.append(band)
            sources.append(f"{label}={band}")

    exec_summary = (exec_doc or {}).get("executive_summary") or {}
    exec_health = str(exec_summary.get("portfolio_health") or "").upper()
    if exec_health:
        bands.append(exec_health)
        sources.append(f"executive.portfolio_health={exec_health}")

    if exec_doc and exec_doc.get("portfolio_health"):
        band = str(exec_doc.get("portfolio_health")).upper()
        bands.append(band)
        sources.append(f"executive_risk.portfolio_health={band}")

    cp = (watchdog_doc or {}).get("capital_preservation") or {}
    wd_band = str(cp.get("health_band") or "").upper()
    if wd_band:
        bands.append(wd_band)
        sources.append(f"watchdog.health_band={wd_band}")

    if any(b in PORTFOLIO_CRITICAL_HEALTH for b in bands):
        return "CRITICAL", sources
    if any(b in ("ELEVATED_RISK", "CAUTION") for b in bands):
        return "ELEVATED", sources
    if bands:
        return bands[0], sources
    return "UNKNOWN", sources


def _derive_preservation_status(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
) -> str:
    """Derive capital preservation status (PASS/FAIL) from preservation artifacts."""
    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    band = str(cpi_doc.get("health_band") or "").upper()
    escalation = str(cpe_doc.get("escalation_state") or "").upper()
    posture = str(governor_doc.get("preservation_posture") or "").upper()

    if band in PORTFOLIO_CRITICAL_HEALTH or cps < CPS_CRITICAL_THRESHOLD:
        return "FAIL"
    if escalation in ESCALATION_VETO_STATES:
        return "FAIL"
    if posture in ("RED", "CRITICAL"):
        return "FAIL"
    return "PASS"


def _is_severely_deteriorating(cpi_doc: Dict[str, Any]) -> bool:
    trend = str(cpi_doc.get("risk_trend") or "").upper()
    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    if trend == "SEVERELY_DETERIORATING":
        return True
    return trend == "DETERIORATING" and cps < CPS_CRITICAL_THRESHOLD


def _evaluation_gate_requires_veto(
    *,
    evaluation_doc: Dict[str, Any],
    cpi_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    areas: Dict[str, Dict[str, Any]],
) -> bool:
    """True when Evaluation failure should trigger mandatory preservation veto."""
    eval_area = areas.get("Evaluation") or {}
    if eval_area.get("certified"):
        return False
    if not evaluation_doc.get("generated_at"):
        return True
    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    band = str(cpi_doc.get("health_band") or "").upper()
    readiness = str(readiness_doc.get("readiness_status") or "")
    if (
        evaluation_doc.get("high_cps_adjustment_applied")
        and evaluation_doc.get("preservation_stable_context")
        and cps >= 75
        and band in ("HEALTHY", "EXCELLENT", "STABLE")
        and readiness == "READY"
    ):
        return False
    return True


def _evaluate_certification_vetoes(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    areas: Dict[str, Dict[str, Any]],
    live_open: bool,
    exec_doc: Optional[Dict[str, Any]] = None,
    watchdog_doc: Optional[Dict[str, Any]] = None,
    evaluation_doc: Optional[Dict[str, Any]] = None,
    readiness_doc: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Evaluate Phase 158 veto rules; return audit rows and triggered rule names."""
    portfolio_health, health_sources = _resolve_portfolio_health(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        exec_doc=exec_doc,
        watchdog_doc=watchdog_doc,
    )
    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    escalation = str(cpe_doc.get("escalation_state") or "").upper()
    preservation_status = _derive_preservation_status(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        governor_doc=governor_doc,
    )
    severely_deteriorating = _is_severely_deteriorating(cpi_doc)

    failed_gates = [
        name
        for name in MANDATORY_PRESERVATION_GATES
        if not (areas.get(name) or {}).get("certified")
    ]
    if (
        "Evaluation" in failed_gates
        and evaluation_doc
        and readiness_doc
        and not _evaluation_gate_requires_veto(
            evaluation_doc=evaluation_doc,
            cpi_doc=cpi_doc,
            readiness_doc=readiness_doc,
            areas=areas,
        )
    ):
        failed_gates = [g for g in failed_gates if g != "Evaluation"]
    if live_open:
        failed_gates.append("live_execution_blocked")

    rule_checks: List[Tuple[str, bool, str]] = [
        (
            "portfolio_health_critical",
            portfolio_health == "CRITICAL",
            f"portfolio_health={portfolio_health} sources={';'.join(health_sources) or 'none'}",
        ),
        (
            "preservation_status_fail",
            preservation_status == "FAIL",
            f"preservation_status={preservation_status} cps={cps} escalation={escalation}",
        ),
        (
            "risk_trend_severely_deteriorating",
            severely_deteriorating,
            (
                f"risk_trend={cpi_doc.get('risk_trend')} cps={cps} "
                f"threshold={CPS_CRITICAL_THRESHOLD}"
            ),
        ),
        (
            "escalation_state_red",
            escalation in ESCALATION_VETO_STATES,
            f"escalation_state={escalation}",
        ),
        (
            "cps_below_critical_threshold",
            cps < CPS_CRITICAL_THRESHOLD,
            f"cps={cps} threshold={CPS_CRITICAL_THRESHOLD}",
        ),
        (
            "mandatory_preservation_gate_failed",
            bool(failed_gates),
            f"failed_gates={','.join(failed_gates) or 'none'}",
        ),
    ]

    audits: List[Dict[str, Any]] = []
    triggered: List[str] = []
    for rule, is_triggered, reason in rule_checks:
        _log_veto(rule=rule, triggered=is_triggered, reason=reason, verbose=verbose)
        audits.append(
            {
                "rule": rule,
                "triggered": is_triggered,
                "reason": reason,
            }
        )
        if is_triggered:
            triggered.append(rule)

    return audits, triggered


def _apply_portfolio_health_override(
    *,
    certification_status: str,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    exec_doc: Optional[Dict[str, Any]] = None,
    watchdog_doc: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[str, bool, str]:
    """Portfolio health is authoritative — CRITICAL forces NOT_CERTIFIED."""
    portfolio_health, sources = _resolve_portfolio_health(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        exec_doc=exec_doc,
        watchdog_doc=watchdog_doc,
    )
    triggered = portfolio_health == "CRITICAL"
    if triggered and certification_status == "CERTIFIED_FOR_PAPER_PROTECTION":
        certification_status = "NOT_CERTIFIED"
    reason = f"portfolio_health={portfolio_health} sources={';'.join(sources) or 'none'}"
    _log_tag(
        "PORTFOLIO_HEALTH_OVERRIDE",
        {"triggered": triggered, "reason": reason},
        verbose=verbose,
    )
    return certification_status, triggered, reason


def _log_alignment_audit(
    *,
    requirement: str,
    expected: str,
    actual: str,
    status: str,
    reason: str,
) -> None:
    logger.info(
        "[CERTIFICATION_ALIGNMENT_AUDIT] requirement=%s expected=%s actual=%s status=%s reason=%s",
        requirement,
        expected,
        actual,
        status,
        reason,
    )
    print(
        f"[CERTIFICATION_ALIGNMENT_AUDIT] requirement={requirement} "
        f"expected={expected} actual={actual} status={status} reason={reason}"
    )


def _audit_requirement(
    *,
    requirement: str,
    expected: str,
    passed: bool,
    actual: str,
    reason: str,
) -> Tuple[bool, Dict[str, Any]]:
    status = "PASS" if passed else "FAIL"
    _log_alignment_audit(
        requirement=requirement,
        expected=expected,
        actual=actual,
        status=status,
        reason=reason,
    )
    return passed, {
        "requirement": requirement,
        "expected": expected,
        "actual": actual,
        "status": status,
        "reason": reason,
    }


def _evaluation_certification(
    evaluation_doc: Dict[str, Any],
) -> Tuple[bool, int, str, Dict[str, Any]]:
    eval_count = int(evaluation_doc.get("evaluation_count") or 0)
    avg_eff = _safe_float(evaluation_doc.get("average_effectiveness"), 0.0) or 0.0
    eval_score = int(min(100, avg_eff)) if eval_count else 0

    baseline_context = _resolve_baseline_context_label(evaluation_doc)
    beneficial = int(evaluation_doc.get("beneficial_count") or 0)
    negative = int(evaluation_doc.get("negative_count") or 0)
    neutral = int(evaluation_doc.get("neutral_count") or 0)
    high_cps_adj = bool(evaluation_doc.get("high_cps_adjustment_applied"))
    scoring_model = str(evaluation_doc.get("scoring_model_version") or "")
    preservation_stable = bool(evaluation_doc.get("preservation_stable_context"))

    if eval_count < 2:
        notes = f"{eval_count} trials evaluated (minimum 2 required)"
        return (
            False,
            eval_score,
            notes,
            {
                "baseline_context": baseline_context,
                "threshold": "eval_count>=2",
            },
        )

    if baseline_context == "HIGH_CPS_PRESERVATION" and high_cps_adj:
        threshold = 40.0
        eval_cert = avg_eff >= threshold and (beneficial >= 1 or neutral >= 2) and negative <= 1
        if preservation_stable and not eval_cert and avg_eff >= threshold and negative <= 1:
            eval_cert = beneficial + neutral >= 2
        threshold_label = "high_cps_baseline_aware>=40"
    elif baseline_context == "MODERATE_CPS_IMPROVEMENT":
        threshold = 45.0
        eval_cert = avg_eff >= threshold and beneficial >= 1
        threshold_label = "moderate_cps>=45"
    else:
        threshold = 50.0
        eval_cert = avg_eff >= threshold
        threshold_label = "standard>=50"

    top_reason = ""
    evaluations = evaluation_doc.get("evaluations") or []
    if evaluations and isinstance(evaluations[0], dict):
        top_reason = str(evaluations[0].get("evaluation_reason") or "")

    notes = (
        f"{eval_count} trials, avg={avg_eff}, baseline={baseline_context}, "
        f"beneficial={beneficial}, neutral={neutral}, negative={negative}, "
        f"model={scoring_model or 'unknown'}, threshold={threshold_label}"
    )
    if top_reason:
        notes += f", top_reason={top_reason[:80]}"

    return (
        eval_cert,
        eval_score,
        notes,
        {
            "baseline_context": baseline_context,
            "high_cps_adjustment_applied": high_cps_adj,
            "scoring_model_version": scoring_model,
            "beneficial_count": beneficial,
            "neutral_count": neutral,
            "negative_count": negative,
            "average_effectiveness": avg_eff,
            "threshold": threshold_label,
            "threshold_value": threshold,
        },
    )


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
    exec_doc: Optional[Dict[str, Any]] = None,
    watchdog_doc: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Score certification areas from existing artifacts with Phase 158 veto enforcement."""
    ts = _iso_utc()
    areas: Dict[str, Dict[str, Any]] = {}
    failed: List[str] = []
    alignment_audits: List[Dict[str, Any]] = []

    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    health_band = str(cpi_doc.get("health_band") or "").upper()
    monitoring_score = int(min(100, max(0, cps)))
    monitoring_cert = (
        bool(cpi_doc.get("generated_at"))
        and cps >= CPS_CRITICAL_THRESHOLD
        and health_band not in PORTFOLIO_CRITICAL_HEALTH
    )
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
    gov_score = (
        80
        if gov_level in ("GREEN", "ROUTINE_OVERSIGHT")
        else (60 if gov_level in ("YELLOW", "INCREASED_MONITORING") else 35)
    )
    gov_cert = gov_level not in GOVERNANCE_BLOCK_LEVELS and bool(gov_doc.get("generated_at"))
    areas["Governance"] = _area_result(
        gov_cert,
        gov_score,
        f"awareness={gov_level}, status={gov_doc.get('governance_status')}",
    )

    paper_ok = _paper_execution_authorized(auth_doc)
    live_open = _live_execution_permitted(auth_doc, readiness_doc)
    auth_score = 90 if paper_ok and not live_open else (25 if live_open else 55)
    auth_cert = bool(paper_ok and auth_doc.get("governance_authorized") and not live_open)
    areas["Authorization"] = _area_result(
        auth_cert,
        auth_score,
        (
            f"paper={auth_doc.get('paper_execution_permitted')}, "
            f"live={auth_doc.get('live_execution_permitted')}, "
            f"execution={auth_doc.get('execution_authorized')}"
        ),
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

    eval_cert, eval_score, eval_notes, eval_context = _evaluation_certification(evaluation_doc)
    areas["Evaluation"] = _area_result(eval_cert, eval_score, eval_notes)

    gov_conf = int(governor_doc.get("governor_confidence") or 0)
    posture = str(governor_doc.get("preservation_posture") or "UNKNOWN")
    governor_score = gov_conf
    governor_cert = bool(governor_doc.get("generated_at")) and posture not in ("CRITICAL", "RED")
    areas["Governor"] = _area_result(
        governor_cert,
        governor_score,
        f"posture={posture}, confidence={gov_conf}%",
    )

    live_actual = (
        f"auth.live={auth_doc.get('live_execution_permitted')}, "
        f"readiness.live={readiness_doc.get('live_execution_permitted')}, "
        f"paper={auth_doc.get('paper_execution_permitted')}"
    )
    live_passed, live_audit = _audit_requirement(
        requirement="Live execution gates must remain blocked",
        expected="live_execution_permitted=false (paper authorization allowed)",
        passed=not live_open,
        actual=live_actual,
        reason=(
            "Live gates blocked; paper execution authorization is distinct"
            if not live_open
            else "Live execution gate open — certification blocked"
        ),
    )
    alignment_audits.append(live_audit)
    if not live_passed:
        failed.append("Live execution gates must remain blocked")

    for name, area in areas.items():
        if not area["certified"]:
            failed.append(name)

    certification_score = (
        round(sum(a["score"] for a in areas.values()) / len(areas), 1) if areas else 0.0
    )
    certified_count = sum(1 for a in areas.values() if a["certified"])

    portfolio_health, health_sources = _resolve_portfolio_health(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        exec_doc=exec_doc,
        watchdog_doc=watchdog_doc,
    )
    preservation_status = _derive_preservation_status(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        governor_doc=governor_doc,
    )

    if live_open:
        certification_status = "NOT_CERTIFIED"
    elif certified_count == len(CERTIFICATION_AREAS) and certification_score >= 80:
        certification_status = "CERTIFIED_FOR_PAPER_PROTECTION"
    elif (
        certified_count == len(CERTIFICATION_AREAS)
        and certification_score >= 70
        and preservation_status == "PASS"
        and portfolio_health == "HEALTHY"
    ):
        certification_status = "CONDITIONALLY_CERTIFIED"
    elif certified_count >= 4 and certification_score >= 55:
        certification_status = "PARTIALLY_CERTIFIED"
    else:
        certification_status = "NOT_CERTIFIED"

    veto_audits, veto_triggered = _evaluate_certification_vetoes(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        governor_doc=governor_doc,
        areas=areas,
        live_open=live_open,
        exec_doc=exec_doc,
        watchdog_doc=watchdog_doc,
        evaluation_doc=evaluation_doc,
        readiness_doc=readiness_doc,
        verbose=verbose,
    )
    pre_override_status = certification_status
    if veto_triggered:
        certification_status = "NOT_CERTIFIED"
        failed.extend(f"veto:{rule}" for rule in veto_triggered)

    certification_status, health_override_triggered, health_override_reason = (
        _apply_portfolio_health_override(
            certification_status=certification_status,
            cpi_doc=cpi_doc,
            cpe_doc=cpe_doc,
            exec_doc=exec_doc,
            watchdog_doc=watchdog_doc,
            verbose=verbose,
        )
    )
    if health_override_triggered and "veto:portfolio_health_critical" not in failed:
        failed.append("portfolio_health_override")

    return {
        "generated_at": ts,
        "certification_status": certification_status,
        "certification_score": certification_score,
        "certified_area_count": certified_count,
        "total_areas": len(CERTIFICATION_AREAS),
        "areas": areas,
        "failed_requirements": failed,
        "alignment_audits": alignment_audits,
        "evaluation_context": eval_context,
        "escalation_state": cpe_doc.get("escalation_state"),
        "live_execution_permitted": live_open,
        "paper_execution_permitted": auth_doc.get("paper_execution_permitted"),
        "portfolio_health": portfolio_health,
        "portfolio_health_sources": health_sources,
        "preservation_status": preservation_status,
        "cps_critical_threshold": CPS_CRITICAL_THRESHOLD,
        "veto_rules_active": True,
        "veto_audits": veto_audits,
        "veto_triggered": veto_triggered,
        "portfolio_health_override_triggered": health_override_triggered,
        "portfolio_health_override_reason": health_override_reason,
        "pre_veto_certification_status": pre_override_status,
        "disclaimer": "Certification assesses paper-mode protection only. No live execution.",
    }


def _merge_docs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_docs(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _strong_paper_inputs() -> Dict[str, Dict[str, Any]]:
    return {
        "cpi": {
            "generated_at": _iso_utc(),
            "capital_preservation_score": 88,
            "health_band": "HEALTHY",
            "risk_trend": "IMPROVING",
        },
        "cpe": {
            "generated_at": _iso_utc(),
            "escalation_state": "GREEN",
            "health_band": "HEALTHY",
        },
        "gov": {
            "generated_at": _iso_utc(),
            "governance_awareness_level": "GREEN",
            "governance_status": "Normal Operations",
        },
        "auth": {
            "generated_at": _iso_utc(),
            "overall_authorization": True,
            "governance_authorized": True,
            "paper_execution_permitted": True,
            "execution_authorized": True,
            "live_execution_permitted": False,
        },
        "readiness": {
            "generated_at": _iso_utc(),
            "readiness_status": "READY",
            "checks_passing_count": 8,
            "checks_total": 8,
            "failed_checks": [],
            "live_execution_permitted": False,
        },
        "sim": {
            "generated_at": _iso_utc(),
            "simulations": [{}, {}, {}, {}],
        },
        "evaluation": {
            "generated_at": _iso_utc(),
            "average_effectiveness": 75.0,
            "evaluation_count": 5,
            "beneficial_count": 4,
            "negative_count": 0,
            "neutral_count": 1,
            "baseline_context": "HIGH_CPS_PRESERVATION",
            "high_cps_adjustment_applied": True,
        },
        "governor": {
            "generated_at": _iso_utc(),
            "preservation_posture": "GREEN",
            "governor_confidence": 92,
        },
        "alerts": {"generated_at": _iso_utc(), "active_alerts": [{}]},
    }


def run_false_positive_elimination_tests(
    *,
    baseline_docs: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = False,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Phase 158 Part C — adversarial scenarios must not earn full certification."""
    base = baseline_docs or _strong_paper_inputs()
    rows: List[Dict[str, Any]] = []
    failures = 0
    ts = _iso_utc()

    scenarios = {
        "A_weak_portfolio_strong_governance": {
            "cpi": {
                "capital_preservation_score": 32,
                "health_band": "CRITICAL",
                "risk_trend": "DETERIORATING",
            },
            "cpe": {"escalation_state": "GREEN", "health_band": "CRITICAL"},
        },
        "B_failing_preservation_strong_committees": {
            "cpi": {
                "capital_preservation_score": 38,
                "health_band": "HIGH_RISK",
                "risk_trend": "DETERIORATING",
            },
            "cpe": {"escalation_state": "ORANGE", "health_band": "HIGH_RISK"},
            "governor": {"preservation_posture": "ORANGE", "governor_confidence": 90},
        },
        "C_red_escalation_perfect_evaluation": {
            "cpi": {
                "capital_preservation_score": 32,
                "health_band": "CRITICAL",
                "risk_trend": "DETERIORATING",
            },
            "cpe": {"escalation_state": "RED", "health_band": "CRITICAL"},
            "evaluation": {
                "average_effectiveness": 95.0,
                "evaluation_count": 6,
                "beneficial_count": 6,
                "negative_count": 0,
                "neutral_count": 0,
                "baseline_context": "HIGH_CPS_PRESERVATION",
                "high_cps_adjustment_applied": True,
            },
        },
    }

    for scenario, overrides in scenarios.items():
        docs = {k: _merge_docs(base[k], overrides.get(k, {})) for k in base}
        cert = compute_preservation_certification(
            cpi_doc=docs["cpi"],
            alerts_doc=docs["alerts"],
            gov_doc=docs["gov"],
            auth_doc=docs["auth"],
            readiness_doc=docs["readiness"],
            sim_doc=docs["sim"],
            evaluation_doc=docs["evaluation"],
            governor_doc=docs["governor"],
            cpe_doc=docs["cpe"],
            verbose=False,
        )
        status = str(cert.get("certification_status") or "")
        passed = status != "CERTIFIED_FOR_PAPER_PROTECTION"
        if not passed:
            failures += 1
        result = "PASS" if passed else "FAIL"
        detail = f"certification_status={status} vetoes={cert.get('veto_triggered')}"
        _log_tag(
            "CERTIFICATION_FALSE_POSITIVE_TEST",
            {"scenario": scenario, "result": result},
            verbose=verbose,
        )
        rows.append(
            {
                "scenario": scenario,
                "test_type": "false_positive",
                "status": result,
                "certification_status": status,
                "detail": detail,
                "generated_at": ts,
            }
        )

    return failures, rows


def run_false_negative_protection_test(
    *,
    baseline_docs: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Phase 158 Part D — healthy portfolio with temporary governance weakness still certifies."""
    base = baseline_docs or _strong_paper_inputs()
    overrides = {
        "gov": {
            "governance_awareness_level": "YELLOW",
            "governance_status": "Temporary governance review — increased monitoring",
        },
    }
    docs = {k: _merge_docs(base[k], overrides.get(k, {})) for k in base}
    cert = compute_preservation_certification(
        cpi_doc=docs["cpi"],
        alerts_doc=docs["alerts"],
        gov_doc=docs["gov"],
        auth_doc=docs["auth"],
        readiness_doc=docs["readiness"],
        sim_doc=docs["sim"],
        evaluation_doc=docs["evaluation"],
        governor_doc=docs["governor"],
        cpe_doc=docs["cpe"],
        verbose=False,
    )
    status = str(cert.get("certification_status") or "")
    passed = status == "CERTIFIED_FOR_PAPER_PROTECTION"
    result = "PASS" if passed else "FAIL"
    scenario = "healthy_portfolio_temporary_governance_weakness"
    detail = f"certification_status={status} vetoes={cert.get('veto_triggered')}"
    _log_tag(
        "CERTIFICATION_FALSE_NEGATIVE_TEST",
        {"scenario": scenario, "result": result},
        verbose=verbose,
    )
    row = {
        "scenario": scenario,
        "test_type": "false_negative",
        "status": result,
        "certification_status": status,
        "detail": detail,
        "generated_at": _iso_utc(),
    }
    return passed, row


def _load_runtime_docs(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    rd = Path(results_dir)
    return {
        "cpi": _read_json(rd / "capital_preservation_intelligence.json"),
        "cpe": _read_json(rd / "capital_preservation_escalation.json"),
        "gov": _read_json(rd / "governance_risk_summary.json"),
        "auth": _read_json(rd / "governance_authorization.json"),
        "readiness": _read_json(rd / "execution_readiness.json"),
        "sim": _read_json(rd / "defensive_simulation_results.json"),
        "evaluation": _read_json(rd / "protective_action_evaluation.json"),
        "governor": _read_json(rd / "capital_preservation_governor.json"),
        "alerts": _read_json(rd / "watchdog_alerts.json"),
        "exec": _read_json(rd / "executive_risk_report.json"),
        "watchdog": _read_json(rd / "watchdog_status.json"),
    }


def run_certification_false_positive_elimination(
    *,
    results_dir: Optional[Path] = None,
    verbose: bool = False,
    run_stress_revalidation: bool = True,
) -> Dict[str, Any]:
    """Phase 158 — run certification, embedded tests, optional stress revalidation, write artifacts."""
    rd = Path(results_dir) if results_dir else RESULTS
    ts = _iso_utc()
    docs = _load_runtime_docs(rd)

    before_stress = _read_json(rd / "institutional_intelligence_stress_test.json")
    before_phase_158 = str(before_stress.get("overall_result") or "UNKNOWN")

    cert_doc = compute_preservation_certification(
        cpi_doc=docs["cpi"],
        alerts_doc=docs["alerts"],
        gov_doc=docs["gov"],
        auth_doc=docs["auth"],
        readiness_doc=docs["readiness"],
        sim_doc=docs["sim"],
        evaluation_doc=docs["evaluation"],
        governor_doc=docs["governor"],
        cpe_doc=docs["cpe"],
        exec_doc=docs["exec"],
        watchdog_doc=docs["watchdog"],
        verbose=verbose,
    )
    _atomic_write_json(cert_doc, rd / "capital_preservation_certification.json")

    fp_failures, fp_rows = run_false_positive_elimination_tests(verbose=verbose)
    fn_passed, fn_row = run_false_negative_protection_test(verbose=verbose)
    csv_rows = fp_rows + [fn_row]

    revalidation_result = "SKIP"
    stress_payload: Dict[str, Any] = {}
    if run_stress_revalidation:
        from services.institutional_intelligence_stress_test import (
            run_institutional_intelligence_stress_test,
        )

        stress_payload = run_institutional_intelligence_stress_test(
            results_dir=rd,
            verbose=verbose,
        )
        revalidation_result = str(stress_payload.get("overall_result") or "FAIL")
        _log_tag(
            "INSTITUTIONAL_REVALIDATION",
            {"result": revalidation_result},
            verbose=verbose,
        )

    after_phase_158 = revalidation_result if run_stress_revalidation else "NOT_RUN"
    _log_tag(
        "CERTIFICATION_INTEGRITY_REVIEW",
        {
            "before_phase_158": before_phase_158,
            "after_phase_158": after_phase_158,
        },
        verbose=verbose,
    )

    false_negative_failures = 0 if fn_passed else 1
    embedded_pass = fp_failures == 0 and fn_passed
    overall_result = "PASS"
    if fp_failures > 0 or false_negative_failures > 0:
        overall_result = "FAIL"
    elif revalidation_result == "FAIL":
        overall_result = "FAIL"
    elif revalidation_result == "WARN":
        overall_result = "WARN"

    payload = {
        "generated_at": ts,
        "phase": "158",
        "false_positive_pathways": fp_failures,
        "false_negative_pathways": false_negative_failures,
        "veto_rules_active": True,
        "portfolio_health_override": True,
        "overall_result": overall_result,
        "embedded_tests_pass": embedded_pass,
        "institutional_revalidation": revalidation_result,
        "cps_critical_threshold": CPS_CRITICAL_THRESHOLD,
        "veto_rules": list(CERTIFICATION_VETO_RULES),
        "certification_status": cert_doc.get("certification_status"),
        "veto_triggered": cert_doc.get("veto_triggered"),
        "portfolio_health": cert_doc.get("portfolio_health"),
    }

    _atomic_write_json(payload, OUTPUT_JSON)

    for row in csv_rows:
        row.setdefault("generated_at", ts)
    csv_rows.append(
        {
            "scenario": "overall",
            "test_type": "summary",
            "status": overall_result,
            "certification_status": cert_doc.get("certification_status", ""),
            "detail": (
                f"false_positive_pathways={fp_failures} "
                f"false_negative_pathways={false_negative_failures} "
                f"revalidation={revalidation_result}"
            ),
            "generated_at": ts,
        }
    )
    _write_scorecard_csv(csv_rows, OUTPUT_CSV)

    if (
        run_stress_revalidation
        and revalidation_result in ("PASS", "WARN")
        and stress_payload.get("institutional_intelligence_certification", {}).get(
            "promotion_recommended"
        )
    ):
        strengths = "; ".join(
            (stress_payload.get("part_i_final_decision") or {}).get("strengths") or []
        )[:200]
        _log_tag(
            "INSTITUTIONAL_INTELLIGENCE_CERTIFIED",
            {
                "status": "STRESS_TESTED_CERTIFIED",
                "strengths": strengths or "adversarial_stress_pass",
                "remaining_gaps": "None",
            },
            verbose=verbose,
        )

    print(
        f"[CAPITAL_PRESERVATION_CERTIFICATION] phase=158 "
        f"status={cert_doc.get('certification_status')} "
        f"false_positive_pathways={fp_failures} "
        f"false_negative_pathways={false_negative_failures} "
        f"overall={overall_result} "
        f"json={OUTPUT_JSON} csv={OUTPUT_CSV}",
        flush=True,
    )

    return {
        "certification": cert_doc,
        "phase_158": payload,
        "stress_test": stress_payload,
        "csv_rows": csv_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 158 — Capital Preservation Certification with veto framework"
    )
    parser.add_argument("--verbose", action="store_true", help="Emit detailed certification logs")
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS),
        help="Path to data/results directory",
    )
    parser.add_argument(
        "--skip-stress-revalidation",
        action="store_true",
        help="Skip institutional intelligence stress test revalidation",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    payload = run_certification_false_positive_elimination(
        results_dir=Path(args.results_dir),
        verbose=args.verbose,
        run_stress_revalidation=not args.skip_stress_revalidation,
    )
    overall = payload.get("phase_158", {}).get("overall_result", "FAIL")
    return 0 if overall in ("PASS", "WARN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
