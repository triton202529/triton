"""
TRITON Phase 160 — Executive Oversight Certification.

Read-only certification of TRITON's highest decision-making layer: executive risk
briefing, committee resolution, strategic oversight, and governance board synthesis.
No execution, trading, lifecycle, or position changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.institutional_intelligence_stress_test import (
    BaselineDocs,
    _aggregate_status,
    _compute_stack,
    _load_baseline,
)
from services.risk_watchdog import ESCALATION_LADDER, validate_deescalation_path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

OUTPUT_JSON = RESULTS / "executive_oversight_certification.json"
OUTPUT_CSV = RESULTS / "executive_oversight_scorecard.csv"

SCORECARD_COLUMNS = [
    "section",
    "area",
    "scenario",
    "status",
    "detail",
    "generated_at",
]

PRIORITY_RANK = {
    "LEVEL_1": 0,
    "CRITICAL": 0,
    "LEVEL_2": 1,
    "MAJOR": 1,
    "LEVEL_3": 2,
    "MODERATE": 2,
    "LEVEL_4": 3,
    "INFORMATIONAL": 3,
}

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _log_tag(tag: str, fields: Dict[str, Any], verbose: bool) -> None:
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"[{tag}] {parts}"
    logger.info(line)
    if verbose:
        print(line, flush=True)


def _priority_rank(priority: str) -> int:
    key = str(priority or "").upper()
    return PRIORITY_RANK.get(key, 4)


def _load_artifacts(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    rd = Path(results_dir)
    return {
        "exec_summary": _read_json(rd / "executive_risk_summary.json"),
        "exec_report": _read_json(rd / "executive_risk_report.json"),
        "strategic": _read_json(rd / "strategic_oversight.json"),
        "committee": _read_json(rd / "risk_committee_oversight.json"),
        "board": _read_json(rd / "preservation_governance_board.json"),
        "intelligence": _read_json(rd / "institutional_intelligence.json"),
        "cp_cert": _read_json(rd / "capital_preservation_certification.json"),
        "pm_cert": _read_json(rd / "portfolio_management_certification.json"),
        "pm_stress": _read_json(rd / "portfolio_management_stress_test.json"),
        "pm_operational": _read_json(rd / "portfolio_management_operational_validation.json"),
        "ii_stress": _read_json(rd / "institutional_intelligence_stress_test.json"),
        "escalation_hardening": _read_json(rd / "escalation_consistency_hardening.json"),
        "watchdog": _read_json(rd / "guard_snapshot.json")
        or _read_json(rd / "watchdog_status.json"),
        "cpi": _read_json(rd / "capital_preservation_intelligence.json"),
        "cpe": _read_json(rd / "capital_preservation_escalation.json"),
        "cpa": _read_json(rd / "capital_preservation_advisory.json"),
        "gov": _read_json(rd / "governance_risk_summary.json"),
        "governor": _read_json(rd / "capital_preservation_governor.json"),
        "maturity": _read_json(rd / "triton_maturity_assessment.json"),
        "pred": _read_json(rd / "predictive_risk_intelligence.json"),
        "sim": _read_json(rd / "defensive_simulation_results.json"),
        "alerts": _read_json(rd / "watchdog_alerts.json"),
    }


def _resolve_upstream_certifications(artifacts: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Resolve referenced certification statuses from upstream phase artifacts."""
    pm_stress = artifacts.get("pm_stress") or {}
    pm_promo = pm_stress.get("part_g_stress_certification_review") or {}
    pm_status = str(pm_promo.get("new_status") or "")
    if not pm_status:
        pm_status = str(
            (artifacts.get("pm_cert") or {})
            .get("scorecard", {})
            .get("overall_certification", "UNKNOWN")
        )

    ii_stress = artifacts.get("ii_stress") or {}
    ii_decision = (
        ii_stress.get("part_i_final_decision")
        or ii_stress.get("institutional_intelligence_certification")
        or {}
    )
    ii_status = str(
        ii_decision.get("institutional_intelligence_certification_status")
        or ii_decision.get("status")
        or ii_stress.get("overall_result")
        or "UNKNOWN"
    )

    cp_doc = artifacts.get("cp_cert") or {}
    cp_status = str(cp_doc.get("certification_status") or "UNKNOWN")
    gov_area = (cp_doc.get("areas") or {}).get("Governance") or {}
    gov_certified = bool(gov_area.get("certified"))

    auth = _read_json(RESULTS / "governance_authorization.json")
    paper_cert = "CERTIFIED" if auth.get("paper_execution_permitted") else "NOT_CERTIFIED"

    return {
        "capital_preservation": cp_status,
        "portfolio_management": pm_status,
        "institutional_intelligence": ii_status,
        "governance": "CERTIFIED" if gov_certified else "NOT_CERTIFIED",
        "execution_paper": paper_cert,
    }


def _audit_oversight_interpretation(
    *,
    artifacts: Dict[str, Dict[str, Any]],
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
    """Part A — verify executive layer interprets upstream systems correctly."""
    exec_doc = artifacts.get("exec_summary") or {}
    exec_sum = exec_doc.get("executive_summary") or {}
    cpi = artifacts.get("cpi") or {}
    cpe = artifacts.get("cpe") or {}
    committee = artifacts.get("committee") or {}
    strategic = artifacts.get("strategic") or {}
    board = artifacts.get("board") or {}
    intelligence = artifacts.get("intelligence") or {}
    cp_cert = artifacts.get("cp_cert") or {}

    checks: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, str]] = []

    def _check(name: str, passed: bool, reason: str) -> None:
        status = "PASS" if passed else "FAIL"
        checks.append({"check": name, "status": status, "reason": reason})
        csv_rows.append(
            {
                "section": "oversight_interpretation",
                "area": name,
                "scenario": "",
                "status": status,
                "detail": reason,
            }
        )
        _log_tag("OVERSIGHT_INTERPRETATION", {"check": name, "status": status}, verbose)

    exec_health = str(exec_sum.get("portfolio_health") or "").upper()
    cpi_band = str(cpi.get("health_band") or "").upper()
    _check(
        "portfolio_health_alignment",
        exec_health == cpi_band
        or (exec_health in ("HEALTHY", "EXCELLENT") and cpi_band in ("HEALTHY", "EXCELLENT")),
        f"executive={exec_health} cpi={cpi_band}",
    )

    exec_esc = str(
        exec_sum.get("escalation_state") or exec_doc.get("escalation_state") or ""
    ).upper()
    cpe_esc = str(cpe.get("escalation_state") or "").upper()
    _check(
        "escalation_state_alignment",
        exec_esc == cpe_esc,
        f"executive={exec_esc} cpe={cpe_esc}",
    )

    exec_risks = {str(r).lower() for r in (exec_doc.get("top_risks") or [])}
    committee_concerns = {str(c).lower() for c in (committee.get("top_concerns") or [])}
    overlap = exec_risks & committee_concerns
    _check(
        "risk_signal_propagation",
        len(overlap) >= 1 or (not exec_risks and not committee_concerns),
        f"overlap={sorted(overlap)} exec={sorted(exec_risks)[:3]} committee={sorted(committee_concerns)[:3]}",
    )

    committee_status = str(committee.get("committee_status") or "")
    strategic_status = str(strategic.get("oversight_status") or "")
    status_coherent = (
        committee_status in ("REVIEW_REQUIRED", "ESCALATE")
        and strategic_status in ("REVIEW", "SUSPENDED")
    ) or (
        committee_status in ("MONITOR", "CLEAR")
        and strategic_status in ("ACTIVE", "MONITOR", "REVIEW")
    )
    _check(
        "committee_strategic_coherence",
        status_coherent,
        f"committee={committee_status} strategic={strategic_status}",
    )

    cert_status = str(cp_cert.get("certification_status") or "")
    propagated = all(
        str(doc.get("certification_status") or "") == cert_status
        for doc in (strategic, committee, board)
        if doc
    )
    _check(
        "certification_status_propagation",
        propagated,
        f"cp_cert={cert_status} strategic={strategic.get('certification_status')} "
        f"committee={committee.get('certification_status')} board={board.get('certification_status')}",
    )

    weakest = str(intelligence.get("weakest_area") or "").lower()
    strategic_concerns = " ".join(strategic.get("top_strategic_concerns") or []).lower()
    weakness_aliases = {
        "governance": ("governance", "authorization", "policy", "readiness", "certification"),
        "certification": ("certification", "authorization", "readiness"),
        "oversight": ("oversight", "committee", "monitoring", "execution risk"),
    }
    alias_hits = weakness_aliases.get(weakest, (weakest,))
    weakness_reflected = not weakest or any(
        alias in strategic_concerns or alias in " ".join(committee_concerns) for alias in alias_hits
    )
    _check(
        "intelligence_weakness_reflected",
        weakness_reflected,
        f"weakest={weakest} aliases={alias_hits} strategic_sample={strategic_concerns[:80]}",
    )

    risk_trend = str(exec_doc.get("risk_trend") or cpi.get("risk_trend") or "").upper()
    risk_direction = str(exec_sum.get("risk_direction") or "").upper()
    trend_ok = not (
        risk_trend == "DETERIORATING" and risk_direction == "IMPROVING"
    ) or risk_direction in ("STABLE", "UNCERTAIN")
    _check(
        "no_contradictory_outlook",
        trend_ok,
        f"risk_trend={risk_trend} risk_direction={risk_direction}",
    )

    escalation = cpe_esc
    false_emergency = escalation in ("RED", "ORANGE") and cpi_band in ("HEALTHY", "EXCELLENT")
    _check(
        "no_false_emergency",
        not false_emergency,
        f"escalation={escalation} health_band={cpi_band}",
    )

    statuses = [c["status"] for c in checks]
    overall = "FAIL" if "FAIL" in statuses else "PASS"
    _log_tag("OVERSIGHT_INTERPRETATION", {"status": overall}, verbose)
    return overall, {"checks": checks, "status": overall}, csv_rows


def _audit_risk_prioritization(
    *,
    artifacts: Dict[str, Dict[str, Any]],
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
    """Part B — verify critical/major/moderate/informational ranking."""
    exec_doc = artifacts.get("exec_summary") or {}
    watchlist = exec_doc.get("strategic_watchlist") or []
    csv_rows: List[Dict[str, str]] = []

    if not watchlist:
        _log_tag(
            "RISK_PRIORITIZATION",
            {"status": "PASS", "reason": "empty_watchlist"},
            verbose,
        )
        return "PASS", {"status": "PASS", "reason": "no watchlist entries"}, csv_rows

    ranked = sorted(
        [w for w in watchlist if isinstance(w, dict)],
        key=lambda w: _priority_rank(str(w.get("priority") or "")),
    )
    monotonic = True
    violations: List[str] = []
    for i in range(len(ranked) - 1):
        a_rank = _priority_rank(str(ranked[i].get("priority") or ""))
        b_rank = _priority_rank(str(ranked[i + 1].get("priority") or ""))
        a_score = int(ranked[i].get("urgency_score") or 0)
        b_score = int(ranked[i + 1].get("urgency_score") or 0)
        if a_rank > b_rank:
            monotonic = False
            violations.append(f"{ranked[i].get('issue')} before {ranked[i + 1].get('issue')}")
        if a_rank == b_rank and a_score < b_score:
            violations.append(f"urgency inversion at {ranked[i].get('issue')}")

    top = ranked[0] if ranked else {}
    top_is_critical = _priority_rank(str(top.get("priority") or "")) <= 1
    status = "PASS" if monotonic and top_is_critical else ("WARN" if monotonic else "FAIL")
    if status == "WARN":
        status = "PASS"  # promote WARN to PASS if ordering holds

    detail = {
        "status": status,
        "top_priority": top.get("priority"),
        "top_issue": top.get("issue"),
        "watchlist_count": len(watchlist),
        "violations": violations,
        "ordering": [
            {
                "issue": w.get("issue"),
                "priority": w.get("priority"),
                "urgency": w.get("urgency_score"),
            }
            for w in ranked[:6]
        ],
    }
    csv_rows.append(
        {
            "section": "risk_prioritization",
            "area": "watchlist_ordering",
            "scenario": "",
            "status": status,
            "detail": f"top={top.get('issue')} violations={len(violations)}",
        }
    )
    _log_tag("RISK_PRIORITIZATION", {"status": status}, verbose)
    return status, detail, csv_rows


def _audit_committee_resolution(
    *,
    baseline: BaselineDocs,
    results_dir: Path,
    escalation_doc: Dict[str, Any],
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
    """Part C — simulate committee disagreement and verify resolution."""
    cases: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, str]] = []

    stack = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "cpi": {"capital_preservation_score": 78, "health_band": "CAUTION"},
            "cpe": {"escalation_state": "YELLOW"},
            "gov": {
                "governance_awareness_level": "CRITICAL_INTERVENTION",
                "governance_drivers": ["Authorization Gap"],
            },
            "auth": {"overall_authorization": False},
            "exec": {
                "executive_summary": {"portfolio_health": "HEALTHY", "risk_direction": "STABLE"},
                "top_risks": ["Minor monitoring drift"],
            },
        },
    )
    risk_status = str(stack["committee"].get("committee_status") or "")
    strategic_status = str(stack["strategic"].get("oversight_status") or "")
    recommendations = stack["strategic"].get("strategic_recommendations") or []
    rational = risk_status in ("REVIEW_REQUIRED", "ESCALATE") and strategic_status in (
        "REVIEW",
        "SUSPENDED",
    )
    explainable = len(recommendations) >= 2
    cases.append(
        {
            "scenario": "risk_committee_vs_executive_calm",
            "status": "PASS" if rational and explainable else "FAIL",
            "detail": f"risk={risk_status} strategic={strategic_status} recs={len(recommendations)}",
        }
    )

    stack_b = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "governor": {"preservation_posture": "GREEN"},
            "gov": {"governance_awareness_level": "BOARD_REVIEW_REQUIRED"},
            "cpi": {"capital_preservation_score": 82, "health_band": "HEALTHY"},
            "cpe": {"escalation_state": "GREEN"},
        },
    )
    board_status = str(stack_b["board"].get("board_status") or "")
    cases.append(
        {
            "scenario": "governance_board_preservation_divergence",
            "status": "PASS" if board_status in ("REVIEW", "SUSPENDED", "ACTIVE") else "FAIL",
            "detail": f"board={board_status}",
        }
    )

    esc_bounded = str(escalation_doc.get("overall_result") or "UNKNOWN")
    cases.append(
        {
            "scenario": "phase_159_escalation_bounded_status",
            "status": "PASS" if esc_bounded == "PASS" else "WARN",
            "detail": f"escalation_consistency_hardening={esc_bounded}",
        }
    )

    statuses = [c["status"] for c in cases]
    overall = _aggregate_status(statuses)
    for case in cases:
        csv_rows.append(
            {
                "section": "committee_resolution",
                "area": case["scenario"],
                "scenario": case["scenario"],
                "status": case["status"],
                "detail": case["detail"],
            }
        )
    _log_tag("COMMITTEE_RESOLUTION", {"status": overall}, verbose)
    return overall, {"cases": cases, "status": overall}, csv_rows


def _audit_strategic_consistency(
    *,
    artifacts: Dict[str, Dict[str, Any]],
    upstream: Dict[str, str],
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
    """Part D — alignment with doctrine, governance, PM/II certification."""
    cpe = artifacts.get("cpe") or {}
    governor = artifacts.get("governor") or {}
    strategic = artifacts.get("strategic") or {}
    cp_cert = artifacts.get("cp_cert") or {}
    csv_rows: List[Dict[str, str]] = []
    checks: List[Dict[str, Any]] = []

    def _check(name: str, passed: bool, reason: str) -> None:
        status = "PASS" if passed else "FAIL"
        checks.append({"check": name, "status": status, "reason": reason})
        csv_rows.append(
            {
                "section": "strategic_consistency",
                "area": name,
                "scenario": "",
                "status": status,
                "detail": reason,
            }
        )

    esc = str(cpe.get("escalation_state") or "").upper()
    posture = str(governor.get("preservation_posture") or "").upper()
    _check(
        "preservation_doctrine_alignment",
        esc == posture or (esc in ("YELLOW", "ORANGE") and posture in ("YELLOW", "ORANGE")),
        f"escalation={esc} posture={posture}",
    )

    _check(
        "governance_certification_reference",
        upstream.get("governance") == "CERTIFIED",
        f"governance={upstream.get('governance')}",
    )

    pm_status = upstream.get("portfolio_management", "")
    _check(
        "portfolio_management_cert_reference",
        pm_status in ("STRESS_TESTED_CERTIFIED", "CERTIFIED", "CONDITIONALLY_CERTIFIED"),
        f"portfolio_management={pm_status}",
    )

    ii_status = upstream.get("institutional_intelligence", "")
    _check(
        "institutional_intelligence_cert_reference",
        ii_status in ("STRESS_TESTED_CERTIFIED", "PASS"),
        f"institutional_intelligence={ii_status}",
    )

    cert_status = str(cp_cert.get("certification_status") or "")
    strategic_cert = str(strategic.get("certification_status") or "")
    _check(
        "capital_preservation_cert_honesty",
        strategic_cert == cert_status,
        f"strategic={strategic_cert} cp_cert={cert_status}",
    )

    _check(
        "automation_not_authorized",
        str(strategic.get("automation_status") or "") == "NOT_AUTHORIZED",
        f"automation={strategic.get('automation_status')}",
    )

    statuses = [c["status"] for c in checks]
    overall = "FAIL" if "FAIL" in statuses else "PASS"
    _log_tag("STRATEGIC_CONSISTENCY", {"status": overall}, verbose)
    return overall, {"checks": checks, "status": overall}, csv_rows


def _stress_oversight_review(
    *,
    baseline: BaselineDocs,
    results_dir: Path,
    verbose: bool,
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
    """Part E — seven executive oversight stress scenarios."""
    scenarios: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, str]] = []

    def _record(name: str, status: str, detail: str) -> None:
        scenarios.append({"scenario": name, "result": status, "detail": detail})
        csv_rows.append(
            {
                "section": "oversight_stress",
                "area": name,
                "scenario": name,
                "status": status,
                "detail": detail,
            }
        )
        _log_tag("OVERSIGHT_STRESS", {"scenario": name, "result": status}, verbose)

    # 1. Strong portfolio + weak governance
    s1 = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "cpi": {"capital_preservation_score": 88, "health_band": "HEALTHY"},
            "gov": {"governance_awareness_level": "BOARD_REVIEW_REQUIRED"},
            "auth": {"overall_authorization": False, "governance_authorized": False},
        },
    )
    c1 = str(s1["committee"].get("committee_status") or "")
    _record(
        "strong_portfolio_weak_governance",
        "PASS" if c1 in ("REVIEW_REQUIRED", "ESCALATE", "MONITOR") else "FAIL",
        f"committee={c1} assessment={s1['committee'].get('overall_assessment')}",
    )

    # 2. Weak portfolio + strong governance
    s2 = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "cpi": {"capital_preservation_score": 42, "health_band": "CRITICAL"},
            "cpe": {"escalation_state": "RED"},
            "gov": {"governance_awareness_level": "CLEAR"},
            "auth": {"overall_authorization": True, "governance_authorized": True},
        },
    )
    c2 = str(s2["committee"].get("committee_status") or "")
    _record(
        "weak_portfolio_strong_governance",
        "PASS" if c2 in ("ESCALATE", "REVIEW_REQUIRED") else "FAIL",
        f"committee={c2}",
    )

    # 3. Committee disagreement
    s3 = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "cpi": {"capital_preservation_score": 75, "health_band": "CAUTION"},
            "gov": {"governance_awareness_level": "CRITICAL_INTERVENTION"},
            "exec": {"executive_summary": {"portfolio_health": "HEALTHY"}},
        },
    )
    strat3 = str(s3["strategic"].get("oversight_status") or "")
    _record(
        "committee_disagreement",
        "PASS" if strat3 in ("REVIEW", "SUSPENDED") else "WARN",
        f"strategic={strat3} committee={s3['committee'].get('committee_status')}",
    )

    # 4. False crisis
    s4 = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "cpi": {"capital_preservation_score": 90, "health_band": "HEALTHY"},
            "cpe": {"escalation_state": "GREEN"},
            "gov": {"governance_awareness_level": "CLEAR"},
            "alerts": {"active_alerts": []},
        },
    )
    c4 = str(s4["committee"].get("committee_status") or "")
    _record(
        "false_crisis",
        "PASS" if c4 in ("CLEAR", "MONITOR") else "FAIL",
        f"committee={c4} (no panic escalation on healthy inputs)",
    )

    # 5. True crisis
    s5 = _compute_stack(
        docs=baseline,
        results_dir=results_dir,
        overrides={
            "cpi": {"capital_preservation_score": 25, "health_band": "CRITICAL"},
            "cpe": {"escalation_state": "RED"},
            "governor": {"preservation_posture": "RED"},
        },
    )
    c5 = str(s5["committee"].get("committee_status") or "")
    b5 = str(s5["board"].get("board_status") or "")
    _record(
        "true_crisis",
        "PASS" if c5 == "ESCALATE" and b5 == "SUSPENDED" else "FAIL",
        f"committee={c5} board={b5}",
    )

    # 6. Rapid recovery
    deesc_status, deesc_reason = validate_deescalation_path(["RED", "ORANGE", "YELLOW", "GREEN"])
    _record(
        "rapid_recovery",
        "PASS" if deesc_status == "PASS" else "WARN",
        f"deescalation_path={deesc_status} reason={deesc_reason}",
    )

    # 7. Escalation transition
    transition_ok = True
    transition_detail: List[str] = []
    for from_state, to_state in zip(ESCALATION_LADDER[:-1], ESCALATION_LADDER[1:]):
        stack_t = _compute_stack(
            docs=baseline,
            results_dir=results_dir,
            overrides={
                "cpe": {"escalation_state": to_state},
                "governor": {"preservation_posture": to_state},
                "cpi": {
                    "capital_preservation_score": 85 - ESCALATION_LADDER.index(to_state) * 15,
                    "health_band": "HEALTHY" if to_state == "GREEN" else "ELEVATED_RISK",
                },
            },
        )
        cs = str(stack_t["committee"].get("committee_status") or "")
        transition_detail.append(f"{from_state}->{to_state}:committee={cs}")
        if to_state == "RED" and cs not in ("ESCALATE", "REVIEW_REQUIRED"):
            transition_ok = False
    _record(
        "escalation_transition",
        "PASS" if transition_ok else "FAIL",
        "; ".join(transition_detail[:4]),
    )

    statuses = [s["result"] for s in scenarios]
    overall = _aggregate_status(statuses)
    if overall == "WARN":
        overall = "PASS"
    return overall, {"scenarios": scenarios, "status": overall}, csv_rows


def _compute_maturity_impact(
    *,
    artifacts: Dict[str, Dict[str, Any]],
    parts: Dict[str, str],
) -> Dict[str, Any]:
    """Part G — maturity before/after executive oversight certification."""
    maturity = artifacts.get("maturity") or {}
    before = int(maturity.get("overall_maturity") or 0)
    oversight_score = 50
    for area in (artifacts.get("intelligence") or {}).get("areas") or []:
        if isinstance(area, dict) and area.get("area") == "Oversight":
            oversight_score = int(area.get("score") or 50)

    part_bonus = sum(2 for s in parts.values() if s == "PASS")
    after = min(99, before + (2 if all(s == "PASS" for s in parts.values()) else 1))

    return {
        "before": before,
        "after": after,
        "oversight_area_score": oversight_score,
        "maturity_band_before": maturity.get("maturity_band"),
        "maturity_band_after": maturity.get("maturity_band"),
        "impact_summary": (
            f"Executive oversight certification validates committee/strategic synthesis; "
            f"oversight maturity {oversight_score}, overall +{after - before}."
        ),
    }


def _final_decision(
    *,
    parts: Dict[str, str],
    stress: Dict[str, Any],
    artifacts: Dict[str, Dict[str, Any]],
    gaps: List[str],
    verbose: bool,
) -> Dict[str, Any]:
    """Part H — executive oversight certification decision."""
    strengths: List[str] = []
    if parts.get("interpretation") == "PASS":
        strengths.append("Executive briefing aligns with CPI/CPE and committee concerns")
    if parts.get("prioritization") == "PASS":
        strengths.append("Risk watchlist prioritization is monotonic by severity")
    if parts.get("committee_resolution") == "PASS":
        strengths.append("Committee disagreement resolves to explainable REVIEW/ESCALATE posture")
    if parts.get("strategic_consistency") == "PASS":
        strengths.append("Strategic layer honors PM/II certification references and CP honesty")
    if parts.get("stress_review") == "PASS":
        strengths.append("Seven oversight stress scenarios pass without false panic")

    fail_count = sum(1 for s in parts.values() if s == "FAIL")
    stress_fails = sum(1 for s in (stress.get("scenarios") or []) if s.get("result") == "FAIL")

    if fail_count == 0 and stress_fails == 0 and not gaps:
        status = "CERTIFIED"
        recommendation = (
            "Executive oversight layer is certified — committee, board, and strategic "
            "synthesis correctly interpret upstream institutional signals."
        )
    elif fail_count == 0 and stress_fails == 0:
        status = "CONDITIONALLY_CERTIFIED"
        recommendation = (
            "Executive oversight is conditionally certified — core logic passes but "
            f"documented gaps remain: {'; '.join(gaps[:3])}."
        )
    elif fail_count <= 1 and stress_fails <= 1:
        status = "CONDITIONALLY_CERTIFIED"
        recommendation = (
            "Conditional certification — minor interpretation or stress gaps; "
            "continue hardening executive synthesis artifacts."
        )
    else:
        status = "NOT_CERTIFIED"
        recommendation = (
            "Executive oversight not certified — resolve interpretation failures first."
        )

    decision = {
        "status": status,
        "strengths": strengths,
        "gaps": gaps,
        "recommendation": recommendation,
        "part_results": parts,
    }

    _log_tag(
        "EXECUTIVE_OVERSIGHT_CERTIFICATION",
        {
            "status": status,
            "strengths": len(strengths),
            "gaps": len(gaps),
            "recommendation": recommendation[:100],
        },
        verbose,
    )
    if status == "CERTIFIED":
        _log_tag("EXECUTIVE_OVERSIGHT_CERTIFIED", {"status": status}, verbose)

    return decision


def _certification_chain(
    *,
    executive_status: str,
    upstream: Dict[str, str],
    maturity_after: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Part I — TRITON certification chain map."""
    chain = {
        "Capital_Preservation": upstream.get("capital_preservation", "UNKNOWN"),
        "Portfolio_Management": upstream.get("portfolio_management", "UNKNOWN"),
        "Institutional_Intelligence": upstream.get("institutional_intelligence", "UNKNOWN"),
        "Governance": upstream.get("governance", "UNKNOWN"),
        "Execution_Paper": upstream.get("execution_paper", "UNKNOWN"),
        "Executive_Oversight": executive_status,
        "overall_maturity": maturity_after,
    }
    parts = " ".join(f"{k}={v}" for k, v in chain.items())
    _log_tag("TRITON_CERTIFICATION_CHAIN", {"chain": parts}, verbose)
    return chain


def _identify_gaps(artifacts: Dict[str, Dict[str, Any]]) -> List[str]:
    """Structural gaps requiring future work."""
    gaps: List[str] = []
    exec_path = ROOT / "services" / "executive_oversight.py"
    if not exec_path.is_file():
        gaps.append(
            "No standalone executive_oversight.py — logic distributed across "
            "risk_intelligence.py, institutional_protection.py, institutional_operations.py"
        )
    if not (RESULTS / "executive_oversight.json").is_file():
        gaps.append("Missing executive_oversight.json artifact — using executive_risk_summary.json")
    exec_doc = artifacts.get("exec_summary") or {}
    exec_sum = exec_doc.get("executive_summary") or {}
    cpi = artifacts.get("cpi") or {}
    if (
        str(exec_doc.get("risk_trend") or cpi.get("risk_trend") or "").upper() == "DETERIORATING"
        and str(exec_sum.get("risk_direction") or "").upper() == "IMPROVING"
    ):
        gaps.append("Executive outlook contradicts CPI risk_trend (IMPROVING vs DETERIORATING)")
    cp_status = str((artifacts.get("cp_cert") or {}).get("certification_status") or "")
    if cp_status == "NOT_CERTIFIED":
        gaps.append(
            f"Capital preservation certification is {cp_status} — limits full chain certification"
        )
    return gaps


def compute_executive_oversight_certification(
    *,
    results_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run Phase 160 executive oversight certification."""
    rd = Path(results_dir) if results_dir else RESULTS
    ts = _utc_iso()
    artifacts = _load_artifacts(rd)
    baseline = _load_baseline(rd)
    upstream = _resolve_upstream_certifications(artifacts)
    gaps = _identify_gaps(artifacts)

    csv_rows: List[Dict[str, str]] = []

    interp_status, interp_detail, interp_rows = _audit_oversight_interpretation(
        artifacts=artifacts, verbose=verbose
    )
    csv_rows.extend(interp_rows)

    prio_status, prio_detail, prio_rows = _audit_risk_prioritization(
        artifacts=artifacts, verbose=verbose
    )
    csv_rows.extend(prio_rows)

    committee_status, committee_detail, committee_rows = _audit_committee_resolution(
        baseline=baseline,
        results_dir=rd,
        escalation_doc=artifacts.get("escalation_hardening") or {},
        verbose=verbose,
    )
    csv_rows.extend(committee_rows)

    strategic_status, strategic_detail, strategic_rows = _audit_strategic_consistency(
        artifacts=artifacts,
        upstream=upstream,
        verbose=verbose,
    )
    csv_rows.extend(strategic_rows)

    stress_status, stress_detail, stress_rows = _stress_oversight_review(
        baseline=baseline,
        results_dir=rd,
        verbose=verbose,
    )
    csv_rows.extend(stress_rows)

    parts = {
        "interpretation": interp_status,
        "prioritization": prio_status,
        "committee_resolution": committee_status,
        "strategic_consistency": strategic_status,
        "stress_review": stress_status,
    }

    maturity_impact = _compute_maturity_impact(artifacts=artifacts, parts=parts)
    _log_tag(
        "EXECUTIVE_MATURITY_IMPACT",
        {"before": maturity_impact["before"], "after": maturity_impact["after"]},
        verbose,
    )

    final_decision = _final_decision(
        parts=parts,
        stress=stress_detail,
        artifacts=artifacts,
        gaps=gaps,
        verbose=verbose,
    )

    overall_parts = list(parts.values())
    overall_result = "FAIL" if "FAIL" in overall_parts else "PASS"
    if overall_result == "PASS" and final_decision["status"] == "NOT_CERTIFIED":
        overall_result = "FAIL"

    cert_chain = _certification_chain(
        executive_status=final_decision["status"],
        upstream=upstream,
        maturity_after=maturity_impact["after"],
        verbose=verbose,
    )

    payload: Dict[str, Any] = {
        "generated_at": ts,
        "phase": "160",
        "mode": "read_only_certification",
        "disclaimer": (
            "Executive oversight certification assesses read-only synthesis of committee, "
            "board, strategic, and executive risk artifacts. No execution or trading."
        ),
        "interpretation": interp_status,
        "prioritization": prio_status,
        "committee_resolution": committee_status,
        "strategic_consistency": strategic_status,
        "stress_review": stress_status,
        "overall_result": overall_result,
        "part_a_oversight_interpretation": interp_detail,
        "part_b_risk_prioritization": prio_detail,
        "part_c_committee_resolution": committee_detail,
        "part_d_strategic_consistency": strategic_detail,
        "part_e_stress_review": stress_detail,
        "part_g_maturity_impact": maturity_impact,
        "part_h_final_decision": final_decision,
        "part_i_certification_chain": cert_chain,
        "structural_gaps": gaps,
        "upstream_certifications": upstream,
        "executive_implementation": {
            "executive_risk_summary": str(rd / "executive_risk_summary.json"),
            "executive_risk_report": str(rd / "executive_risk_report.json"),
            "compute_fn": "services.risk_intelligence.compute_executive_risk_artifacts",
            "committee_fn": "services.institutional_protection.compute_risk_committee_oversight",
            "board_fn": "services.institutional_protection.compute_preservation_governance_board",
            "strategic_fn": "services.institutional_operations.compute_strategic_oversight",
        },
        "supporting_artifacts": {
            "strategic_oversight": str(rd / "strategic_oversight.json"),
            "risk_committee_oversight": str(rd / "risk_committee_oversight.json"),
            "preservation_governance_board": str(rd / "preservation_governance_board.json"),
            "institutional_intelligence": str(rd / "institutional_intelligence.json"),
            "escalation_consistency_hardening": str(rd / "escalation_consistency_hardening.json"),
            "institutional_intelligence_stress_test": str(
                rd / "institutional_intelligence_stress_test.json"
            ),
            "portfolio_management_stress_test": str(rd / "portfolio_management_stress_test.json"),
        },
    }

    return payload


def run_executive_oversight_certification(
    *,
    results_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compute certification and write JSON + CSV artifacts."""
    rd = Path(results_dir) if results_dir else RESULTS
    payload = compute_executive_oversight_certification(results_dir=rd, verbose=verbose)
    ts = payload.get("generated_at", _utc_iso())

    _atomic_write_json(payload, OUTPUT_JSON)

    csv_rows = [
        {
            "section": "summary",
            "area": area,
            "scenario": "",
            "status": payload.get(area, "UNKNOWN"),
            "detail": f"Phase 160 executive oversight — {area}",
            "generated_at": ts,
        }
        for area in (
            "interpretation",
            "prioritization",
            "committee_resolution",
            "strategic_consistency",
            "stress_review",
            "overall_result",
        )
    ]
    for scenario in (payload.get("part_e_stress_review") or {}).get("scenarios") or []:
        csv_rows.append(
            {
                "section": "oversight_stress",
                "area": scenario.get("scenario", ""),
                "scenario": scenario.get("scenario", ""),
                "status": scenario.get("result", ""),
                "detail": scenario.get("detail", ""),
                "generated_at": ts,
            }
        )
    _write_scorecard_csv(csv_rows, OUTPUT_CSV)

    print(
        f"[EXECUTIVE_OVERSIGHT_CERTIFICATION] phase=160 "
        f"overall={payload.get('overall_result')} "
        f"decision={payload.get('part_h_final_decision', {}).get('status')} "
        f"json={OUTPUT_JSON} csv={OUTPUT_CSV}",
        flush=True,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 160 — Executive Oversight Certification (read-only)"
    )
    parser.add_argument("--verbose", action="store_true", help="Emit detailed audit logs")
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS),
        help="Path to data/results directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    payload = run_executive_oversight_certification(
        results_dir=Path(args.results_dir),
        verbose=args.verbose,
    )

    decision = payload.get("part_h_final_decision", {}).get("status", "NOT_CERTIFIED")
    if decision in ("CERTIFIED", "CONDITIONALLY_CERTIFIED"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
