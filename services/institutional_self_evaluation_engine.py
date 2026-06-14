"""
TRITON Phase 208 — Institutional Self-Evaluation & Strategic Evolution Engine.

Comprehensive institutional self-assessment using historical experience, current
architecture, governance intelligence, operational performance, and institutional
memory to determine strategic evolution over 1, 3, and 5 years.

Read-only — does NOT modify governance, trading, risk controls, or settings.

Outputs (data/results/):
    institutional_self_evaluation.json
    institutional_strengths.csv
    institutional_weaknesses.csv
    institutional_growth_roadmap.csv
    strategic_recommendations.csv

Run:
    python -m services.institutional_self_evaluation_engine
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from services.concentration_governance_analysis import _f, _read_json

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "institutional_self_evaluation.json"
OUT_STRENGTHS = RESULTS / "institutional_strengths.csv"
OUT_WEAKNESSES = RESULTS / "institutional_weaknesses.csv"
OUT_ROADMAP = RESULTS / "institutional_growth_roadmap.csv"
OUT_RECOMMENDATIONS = RESULTS / "strategic_recommendations.csv"

PHASE_ARTIFACTS = [
    "institutional_memory_report.json",
    "post_green_certification_status.json",
    "green_authorization_strategy.json",
    "governance_recovery_simulation.json",
    "paper_execution_certification.json",
    "lifecycle_readiness_diagnostics.json",
    "pipeline_status.json",
    "executive_oversight_certification.json",
    "portfolio_management_certification.json",
    "adaptive_risk_state.json",
]


def _log(msg: str) -> None:
    print(f"[institutional_self_evaluation_engine] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _grade(score: float) -> str:
    if score >= 95:
        return "A+"
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    return "D"


def _evolution_band(score: float) -> str:
    if score >= 95:
        return "AUTONOMOUS_INSTITUTION"
    if score >= 85:
        return "INSTITUTIONAL_GRADE"
    if score >= 70:
        return "ADVANCED_PLATFORM"
    if score >= 50:
        return "DEVELOPING_SYSTEM"
    return "EARLY_STAGE"


def _priority(importance: str, difficulty: str) -> str:
    if importance == "CRITICAL":
        return "CRITICAL"
    if importance == "HIGH" and difficulty in ("LOW", "MODERATE"):
        return "HIGH_PRIORITY"
    if importance == "HIGH":
        return "HIGH_PRIORITY"
    if importance == "MODERATE":
        return "MEDIUM_PRIORITY"
    return "LOW_PRIORITY"


def _artifact_coverage() -> float:
    return sum(1 for f in PHASE_ARTIFACTS if (RESULTS / f).is_file()) / len(PHASE_ARTIFACTS)


def _load_baseline() -> Dict[str, Any]:
    memory = _read_json(RESULTS / "institutional_memory_report.json")
    post_green = _read_json(RESULTS / "post_green_certification_status.json")
    green_strat = _read_json(RESULTS / "green_authorization_strategy.json")
    paper = _read_json(RESULTS / "paper_execution_certification.json")
    lifecycle = _read_json(RESULTS / "lifecycle_readiness_diagnostics.json")
    pipeline = _read_json(RESULTS / "pipeline_status.json")
    sim = _read_json(RESULTS / "governance_recovery_simulation.json")

    cert = post_green.get("certification_status") or {}
    mon = post_green.get("continuous_monitoring") or {}
    port = mon.get("portfolio_integrity") or {}
    ops = mon.get("operational_integrity") or {}
    gov = mon.get("governance_integrity") or {}

    wisdom = (memory.get("institutional_wisdom") or {}).get("institutional_wisdom_score", 0)
    cps = cert.get("cps", 66)
    qqq_pct = cert.get("qqq_concentration_pct", 100.0)
    escalation = cert.get("escalation_state", "ORANGE")
    watchdog = ops.get("watchdog_stability_score", 50.0)
    exec_stability = ops.get("execution_readiness_stability", 50.0)
    evidence_pct = gov.get("evidence_quality_pct", 0)
    lifecycle_ok = lifecycle.get("overall_pass", False)
    pipeline_ok = pipeline.get("status") == "ok"
    paper_ready = (paper.get("prep") or {}).get("after", {}).get("readiness_status") == "READY"

    return {
        "wisdom_score": _f(wisdom or 0, 1) or 0.0,
        "cps": cps,
        "qqq_concentration_pct": qqq_pct,
        "escalation_state": escalation,
        "watchdog_stability": watchdog,
        "execution_stability": exec_stability,
        "evidence_quality_pct": evidence_pct,
        "lifecycle_ok": lifecycle_ok,
        "pipeline_ok": pipeline_ok,
        "paper_ready": paper_ready,
        "diversification_score": port.get("diversification_quality_score", 25.0),
        "position_count": port.get("position_count", 1),
        "projected_green": cert.get("projected_green_date"),
        "projected_yellow": cert.get("projected_yellow_date"),
        "green_paths": len(
            (green_strat.get("authorization_paths") or green_strat.get("paths") or [])
        ),
        "sim_available": bool(sim),
        "artifact_coverage_pct": _f(_artifact_coverage() * 100, 1),
    }


def _maturity_divisions(baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Four institutional divisions with numeric scores and letter grades."""

    signal_score = float(
        np.clip(
            55
            + (10 if baseline["lifecycle_ok"] else 0)
            + (10 if baseline["pipeline_ok"] else 0)
            + (5 if baseline["paper_ready"] else 0)
            - (baseline["qqq_concentration_pct"] - 35) * 0.3,
            40,
            95,
        )
    )
    trading = {
        "division": "Trading Intelligence",
        "measures": {
            "signal_consistency": _f(signal_score * 0.95, 1),
            "prediction_quality": _f(signal_score * 0.90, 1),
            "model_diversity": _f(max(50, 100 - baseline["qqq_concentration_pct"] * 0.5), 1),
            "adaptability": _f(signal_score * 0.85, 1),
            "historical_performance": _f(75.0, 1),
        },
        "composite_score": _f(signal_score, 1),
        "grade": _grade(signal_score),
    }

    gov_base = float(
        np.clip(
            70
            + (baseline["wisdom_score"] * 0.15)
            + (10 if baseline["escalation_state"] in ("ORANGE", "RED", "YELLOW") else 0)
            + (baseline["evidence_quality_pct"] * 0.1)
            - max(0, 90 - baseline["cps"]) * 0.2,
            50,
            98,
        )
    )
    risk_gov = {
        "division": "Risk & Governance",
        "measures": {
            "risk_prevention": _f(gov_base * 1.02, 1),
            "capital_protection": _f(gov_base * 1.05, 1),
            "authorization_discipline": _f(gov_base * 0.98, 1),
            "recovery_capability": _f(max(55, baseline["cps"] * 0.85), 1),
            "certification_maturity": _f(gov_base * 0.92, 1),
        },
        "composite_score": _f(gov_base, 1),
        "grade": _grade(gov_base),
    }

    ops_base = float(
        np.clip(
            baseline["watchdog_stability"] * 0.35
            + baseline["execution_stability"] * 0.35
            + (25 if baseline["pipeline_ok"] else 5)
            + (15 if baseline["paper_ready"] else 0),
            30,
            95,
        )
    )
    operations = {
        "division": "Operations & Reliability",
        "measures": {
            "automation_stability": _f(ops_base * 0.95, 1),
            "self_healing_capability": _f(baseline["watchdog_stability"], 1),
            "monitoring_coverage": _f(baseline["execution_stability"], 1),
            "failure_response": _f(ops_base * 0.88, 1),
            "system_uptime": _f(90.0 if baseline["pipeline_ok"] else 60.0, 1),
        },
        "composite_score": _f(ops_base, 1),
        "grade": _grade(ops_base),
    }

    intel_base = float(
        np.clip(
            baseline["wisdom_score"] * 0.85
            + baseline["artifact_coverage_pct"] * 0.15
            + (5 if baseline["sim_available"] else 0),
            60,
            100,
        )
    )
    intelligence = {
        "division": "Institutional Intelligence",
        "measures": {
            "learning_ability": _f(baseline["wisdom_score"], 1),
            "historical_knowledge": _f(intel_base * 1.02, 1),
            "decision_transparency": _f(intel_base * 0.95, 1),
            "strategic_awareness": _f(intel_base * 0.90, 1),
        },
        "composite_score": _f(intel_base, 1),
        "grade": _grade(intel_base),
    }

    return [trading, risk_gov, operations, intelligence]


def _current_institutional_level(divisions: List[Dict[str, Any]]) -> str:
    avg = np.mean([d["composite_score"] for d in divisions])
    if avg >= 90:
        return "Digital investment committee"
    if avg >= 80:
        return "Institutional trading platform"
    if avg >= 65:
        return "Advanced trading system"
    return "Developing trading system"


def _strategic_gaps(baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
    gaps = [
        {
            "gap_id": "GAP-001",
            "capability": "Portfolio diversification execution",
            "current_state": f"QQQ {baseline['qqq_concentration_pct']}% concentration",
            "target_1yr": "≤35% max single-name weight; CPS ≥75",
            "target_3yr": "Multi-sector balanced portfolio with dynamic rebalancing",
            "target_5yr": "Autonomous multi-asset allocation across regimes",
            "importance": "CRITICAL",
            "difficulty": "HIGH",
            "required_investment": "Recovery execution + governance clearance",
            "expected_benefit": "Eliminate concentration systemic risk",
            "strategic_priority": "CRITICAL",
        },
        {
            "gap_id": "GAP-002",
            "capability": "Operational watchdog reliability",
            "current_state": f"Watchdog stability {baseline['watchdog_stability']}",
            "target_1yr": "Watchdog stability ≥80; false positive rate <20%",
            "target_3yr": "Self-healing loop with zero manual intervention",
            "target_5yr": "Predictive failure prevention before anomalies",
            "importance": "HIGH",
            "difficulty": "MODERATE",
            "required_investment": "Watchdog tuning + loop hardening",
            "expected_benefit": "Continuous institutional monitoring without blind spots",
            "strategic_priority": "HIGH_PRIORITY",
        },
        {
            "gap_id": "GAP-003",
            "capability": "GREEN certification achievement",
            "current_state": f"CPS {baseline['cps']}; ORANGE escalation",
            "target_1yr": "GREEN certification with post-GREEN monitor active",
            "target_3yr": "Sustained GREEN with regression prevention",
            "target_5yr": "Self-certifying institution with audit trail",
            "importance": "CRITICAL",
            "difficulty": "HIGH",
            "required_investment": "Evidence accumulation + diversification paths",
            "expected_benefit": "Full institutional authorization restored",
            "strategic_priority": "CRITICAL",
        },
        {
            "gap_id": "GAP-004",
            "capability": "Investment committee simulation",
            "current_state": "Executive oversight exists; committee simulation partial",
            "target_1yr": "Full committee deliberation simulation before major decisions",
            "target_3yr": "Multi-stakeholder scenario planning with stress overlays",
            "target_5yr": "Autonomous investment committee with human veto only",
            "importance": "HIGH",
            "difficulty": "HIGH",
            "required_investment": "Phase 209+ committee architecture",
            "expected_benefit": "Institutional-grade decision quality at scale",
            "strategic_priority": "HIGH_PRIORITY",
        },
        {
            "gap_id": "GAP-005",
            "capability": "International & multi-asset expansion",
            "current_state": "US equities and ETFs only",
            "target_1yr": "Expanded ETF universe with sector rotation",
            "target_3yr": "International equity and fixed income integration",
            "target_5yr": "Global multi-asset institutional portfolio",
            "importance": "MODERATE",
            "difficulty": "HIGH",
            "required_investment": "Data infrastructure + model expansion",
            "expected_benefit": "True diversification beyond US tech concentration",
            "strategic_priority": "MEDIUM_PRIORITY",
        },
        {
            "gap_id": "GAP-006",
            "capability": "Market regime intelligence",
            "current_state": "Regime detection present; limited macro integration",
            "target_1yr": "Regime-aware allocation adjustments",
            "target_3yr": "Macro intelligence layer with alternative data",
            "target_5yr": "AI-driven regime forecasting with institutional validation",
            "importance": "MODERATE",
            "difficulty": "MODERATE",
            "required_investment": "Research pipeline + validation framework",
            "expected_benefit": "Adaptive strategy across market cycles",
            "strategic_priority": "MEDIUM_PRIORITY",
        },
        {
            "gap_id": "GAP-007",
            "capability": "Regulatory & audit readiness",
            "current_state": "Internal audit systems strong; external readiness partial",
            "target_1yr": "Complete audit trail for all governance decisions",
            "target_3yr": "Regulatory reporting templates and compliance checks",
            "target_5yr": "Institutional-grade external audit certification",
            "importance": "MODERATE",
            "difficulty": "MODERATE",
            "required_investment": "Reporting infrastructure + documentation",
            "expected_benefit": "External institutional credibility",
            "strategic_priority": "MEDIUM_PRIORITY",
        },
    ]
    for g in gaps:
        g["strategic_priority"] = _priority(g["importance"], g["difficulty"])
    return gaps


def _evolution_roadmap(baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    research = [
        ("New AI model integration", "1yr", "Evaluate ensemble models for signal generation"),
        ("Market regime intelligence", "3yr", "Macro overlay with regime-aware allocation"),
        ("Alternative data integration", "3yr", "Sentiment, flows, and macro indicators"),
        (
            "Macro intelligence layer",
            "5yr",
            "Institutional macro forecasting validated by committee",
        ),
    ]
    portfolio = [
        (
            "Dynamic diversification engine",
            "1yr",
            "Execute PATH_B recovery basket without BITO dependency",
        ),
        ("Multi-asset management", "3yr", "Equities, bonds, commodities, crypto allocation"),
        ("International markets", "3yr", "Developed and emerging market ETF integration"),
        ("Advanced optimization", "5yr", "Risk-parity and factor-aware portfolio construction"),
    ]
    institutional = [
        ("Investment committee simulation", "1yr", "Deliberation engine before major transitions"),
        ("Advanced scenario planning", "3yr", "Stress-test all certification pathways"),
        ("Regulatory readiness", "3yr", "Audit trail and compliance reporting"),
        ("Institutional reporting", "5yr", "External-grade quarterly institutional reports"),
    ]
    operational = [
        ("Watchdog self-repair", "1yr", "Reduce false positives; stabilize continuous loop"),
        ("Continuous validation", "1yr", "Daily contract validation with auto-remediation"),
        (
            "Automated improvement recommendations",
            "3yr",
            "Self-evaluation engine drives phase roadmap",
        ),
        (
            "Greater operational autonomy",
            "5yr",
            "Self-healing with human oversight on capital decisions",
        ),
    ]

    idx = 1
    for category, entries in [
        ("Research Evolution", research),
        ("Portfolio Evolution", portfolio),
        ("Institutional Evolution", institutional),
        ("Operational Evolution", operational),
    ]:
        for capability, horizon, description in entries:
            items.append(
                {
                    "roadmap_id": f"ROAD-{idx:03d}",
                    "category": category,
                    "capability": capability,
                    "horizon": horizon,
                    "description": description,
                    "current_readiness_pct": _f(
                        baseline["artifact_coverage_pct"] if horizon == "1yr" else 40.0, 1
                    ),
                    "dependency": (
                        "GREEN certification"
                        if "diversification" in capability.lower()
                        else "Operational stability"
                    ),
                }
            )
            idx += 1
    return items


def _strengths(divisions: List[Dict[str, Any]], baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = [
        {
            "strength_id": "STR-001",
            "category": "Governance",
            "title": "Capital preservation discipline",
            "description": "ORANGE escalation correctly blocked authorization during 100% QQQ concentration",
            "evidence": "Phase 199 protective block; Phase 207 institutional memory",
            "impact": "CRITICAL — capital preserved during concentration crisis",
            "score": _f((divisions[1]["composite_score"] or 0) * 1.02, 1),
        },
        {
            "strength_id": "STR-002",
            "category": "Institutional Intelligence",
            "title": "Comprehensive recovery intelligence stack",
            "description": "Phases 199–207 provide end-to-end governance learning and recovery planning",
            "evidence": f"Wisdom score {baseline['wisdom_score']}; 8 recovery phase artifacts",
            "impact": "HIGH — operator can answer every recovery question",
            "score": baseline["wisdom_score"],
        },
        {
            "strength_id": "STR-003",
            "category": "Certification Architecture",
            "title": "Multi-layer certification framework",
            "description": "Daily evidence, simulation, GREEN strategy, and post-GREEN monitoring",
            "evidence": "Phases 201, 204, 205, 206 artifacts present",
            "impact": "HIGH — certification is measurable, not aspirational",
            "score": _f(divisions[1]["composite_score"], 1),
        },
        {
            "strength_id": "STR-004",
            "category": "Operations",
            "title": "Paper execution certification",
            "description": "Execution layer verified READY with full audit trail and duplicate protection",
            "evidence": "Phase 163 paper_execution_certification.json",
            "impact": "MODERATE — operational capability exists when governance permits",
            "score": _f(85.0 if baseline["paper_ready"] else 60.0, 1),
        },
        {
            "strength_id": "STR-005",
            "category": "Transparency",
            "title": "Decision transparency and audit systems",
            "description": "Executive reporting, evidence tracking, and institutional memory preservation",
            "evidence": "Phase 207 knowledge base; executive oversight certification",
            "impact": "HIGH — every major decision has documented rationale",
            "score": _f(divisions[3]["composite_score"], 1),
        },
        {
            "strength_id": "STR-006",
            "category": "Recovery Planning",
            "title": "Alternative diversification pathways",
            "description": "PATH_B GREEN achievable without BITO via XLE+PEP alternatives",
            "evidence": "Phase 205 green_authorization_strategy.json",
            "impact": "HIGH — reduces single-asset dependency risk",
            "score": _f(80.0, 1),
        },
    ]
    return sorted(raw, key=lambda x: x.get("score") or 0, reverse=True)


def _weaknesses(baseline: Dict[str, Any], divisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw = [
        {
            "weakness_id": "WKN-001",
            "category": "Portfolio",
            "title": "Extreme concentration risk",
            "description": f"QQQ at {baseline['qqq_concentration_pct']}% — single point of portfolio failure",
            "evidence": "post_green_certification_status.json",
            "severity": "CRITICAL",
            "impact": "Systemic — CPS at 66; ORANGE escalation",
            "score": _f(100 - baseline["qqq_concentration_pct"], 1),
        },
        {
            "weakness_id": "WKN-002",
            "category": "Operations",
            "title": "Watchdog instability",
            "description": f"Watchdog stability {baseline['watchdog_stability']} with elevated false positives",
            "evidence": "Phase 206 operational_integrity metrics",
            "severity": "HIGH",
            "impact": "Monitoring blind spots during critical recovery period",
            "score": baseline["watchdog_stability"],
        },
        {
            "weakness_id": "WKN-003",
            "category": "Governance",
            "title": "Governance paradox unresolved",
            "description": "Diversification required but authorization blocked — partial resolution only",
            "evidence": "Phase 203 execution feasibility; Phase 207 EVT-002",
            "severity": "HIGH",
            "impact": "Recovery delay; operator confusion risk",
            "score": _f(45.0, 1),
        },
        {
            "weakness_id": "WKN-004",
            "category": "Evidence",
            "title": "Low certification evidence quality",
            "description": f"Evidence quality at {baseline['evidence_quality_pct']}% during recovery",
            "evidence": "Phase 206 governance_integrity",
            "severity": "HIGH",
            "impact": "GREEN certification delayed; conditional approval only",
            "score": baseline["evidence_quality_pct"],
        },
        {
            "weakness_id": "WKN-005",
            "category": "Trading",
            "title": "Limited model diversity in live portfolio",
            "description": f"Only {baseline['position_count']} position(s); diversification score {baseline['diversification_score']}",
            "evidence": "Portfolio integrity metrics",
            "severity": "CRITICAL",
            "impact": "Signal intelligence not reflected in portfolio construction",
            "score": baseline["diversification_score"],
        },
        {
            "weakness_id": "WKN-006",
            "category": "Strategic",
            "title": "No live institutional authorization",
            "description": f"Escalation {baseline['escalation_state']}; governance_authorized=false",
            "evidence": "Current governance state",
            "severity": "HIGH",
            "impact": "Institution cannot act on its own intelligence",
            "score": _f(baseline["cps"], 1),
        },
    ]
    return sorted(raw, key=lambda x: x.get("score") or 0)


def _self_critique(baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues = [
        {
            "issue_id": "CRIT-001",
            "category": "Single Point of Failure",
            "issue": "100% QQQ concentration",
            "severity": "CRITICAL",
            "probability_pct": 95.0,
            "institutional_impact": "Total portfolio exposure to single tech ETF",
            "recommended_mitigation": "Execute PATH_B diversification; enforce 35% max weight post-GREEN",
        },
        {
            "issue_id": "CRIT-002",
            "category": "Dangerous Assumption",
            "issue": "Operational READY implies governance authorization",
            "severity": "HIGH",
            "probability_pct": 60.0,
            "institutional_impact": "False confidence during ORANGE escalation",
            "recommended_mitigation": "Maintain separate ops vs governance readiness scores (Phase 198)",
        },
        {
            "issue_id": "CRIT-003",
            "category": "Over-Engineering Risk",
            "issue": "200+ phase intelligence modules without unified orchestration",
            "severity": "MODERATE",
            "probability_pct": 40.0,
            "institutional_impact": "Operator cognitive load; maintenance burden",
            "recommended_mitigation": "Executive dashboard consolidation; phase dependency map",
        },
        {
            "issue_id": "CRIT-004",
            "category": "Missing Capability",
            "issue": "No autonomous recovery execution",
            "severity": "HIGH",
            "probability_pct": 80.0,
            "institutional_impact": "Recovery depends on manual operator intervention",
            "recommended_mitigation": "Pre-authorized recovery playbooks with governance clearance",
        },
        {
            "issue_id": "CRIT-005",
            "category": "Human Oversight Required",
            "issue": "Capital allocation decisions",
            "severity": "CRITICAL",
            "probability_pct": 100.0,
            "institutional_impact": "Irreversible capital commitment requires human judgment",
            "recommended_mitigation": "Never automate capital deployment without explicit authorization",
        },
        {
            "issue_id": "CRIT-006",
            "category": "Human Oversight Required",
            "issue": "Governance threshold changes",
            "severity": "CRITICAL",
            "probability_pct": 100.0,
            "institutional_impact": "Threshold changes alter institutional risk appetite",
            "recommended_mitigation": "All threshold changes require documented human approval",
        },
        {
            "issue_id": "CRIT-007",
            "category": "Architectural Strength",
            "issue": "Layered governance with healthy separation",
            "severity": "LOW",
            "probability_pct": 90.0,
            "institutional_impact": "Positive — prevents ops stack from bypassing governance",
            "recommended_mitigation": "Preserve and extend; do not collapse layers",
        },
        {
            "issue_id": "CRIT-008",
            "category": "Stagnation Risk",
            "issue": "Recovery stalled at CPS 66 for extended period",
            "severity": "HIGH",
            "probability_pct": 70.0,
            "institutional_impact": "Institution learns but cannot act on lessons",
            "recommended_mitigation": "Accelerate evidence accumulation; resolve governance paradox",
        },
    ]
    return issues


def _strategic_recommendations(
    gaps: List[Dict[str, Any]],
    strengths: List[Dict[str, Any]],
    weaknesses: List[Dict[str, Any]],
    baseline: Dict[str, Any],
) -> List[Dict[str, Any]]:
    recs = [
        {
            "recommendation_id": "REC-001",
            "priority": "CRITICAL",
            "horizon": "1yr",
            "title": "Complete concentration recovery",
            "action": "Execute PATH_B diversification basket; target CPS ≥75 by projected YELLOW date",
            "rationale": "Concentration is the dominant systemic risk (Phase 207 KB-002)",
            "never_change": False,
        },
        {
            "recommendation_id": "REC-002",
            "priority": "CRITICAL",
            "horizon": "1yr",
            "title": "Achieve GREEN certification",
            "action": "Accumulate daily evidence; run Phase 204 simulation before live transition",
            "rationale": "Institutional authorization requires measurable certification evidence",
            "never_change": False,
        },
        {
            "recommendation_id": "REC-003",
            "priority": "HIGH_PRIORITY",
            "horizon": "1yr",
            "title": "Stabilize watchdog and continuous loop",
            "action": "Reduce false positive rate; target watchdog stability ≥80",
            "rationale": f"Current watchdog stability {baseline['watchdog_stability']} creates monitoring blind spots",
            "never_change": False,
        },
        {
            "recommendation_id": "REC-004",
            "priority": "HIGH_PRIORITY",
            "horizon": "3yr",
            "title": "Build investment committee simulation",
            "action": "Multi-stakeholder deliberation engine for major portfolio transitions",
            "rationale": "Bridge from intelligent platform to digital investment committee",
            "never_change": False,
        },
        {
            "recommendation_id": "REC-005",
            "priority": "MEDIUM_PRIORITY",
            "horizon": "3yr",
            "title": "Expand to multi-asset institutional portfolio",
            "action": "International equities, fixed income, and regime-aware allocation",
            "rationale": "True diversification requires asset class expansion beyond US ETFs",
            "never_change": False,
        },
        {
            "recommendation_id": "REC-006",
            "priority": "MEDIUM_PRIORITY",
            "horizon": "5yr",
            "title": "Autonomous institution with human veto",
            "action": "Self-evaluating, self-improving institution; human approval on capital only",
            "rationale": "Phase 208 vision — institution that designs its own evolution",
            "never_change": False,
        },
        {
            "recommendation_id": "REC-007",
            "priority": "CRITICAL",
            "horizon": "Permanent",
            "title": "Never remove capital preservation governance",
            "action": "Preserve ORANGE/RED escalation, CPS thresholds, and authorization blocks",
            "rationale": "Governance protected capital during concentration crisis — never override",
            "never_change": True,
        },
        {
            "recommendation_id": "REC-008",
            "priority": "CRITICAL",
            "horizon": "Permanent",
            "title": "Never automate threshold or permission changes",
            "action": "All governance rule changes require documented human approval",
            "rationale": "Self-modifying risk controls are the highest institutional failure mode",
            "never_change": True,
        },
        {
            "recommendation_id": "REC-009",
            "priority": "HIGH_PRIORITY",
            "horizon": "1yr",
            "title": "Resolve governance paradox",
            "action": "Allocation-layer execution with post-hoc authorization evidence tracking",
            "rationale": "Paradox blocks recovery execution despite operational readiness",
            "never_change": False,
        },
        {
            "recommendation_id": "REC-010",
            "priority": "HIGH_PRIORITY",
            "horizon": "3yr",
            "title": "Institutional memory as permanent operating system",
            "action": "Auto-update knowledge base after every certification milestone",
            "rationale": f"Wisdom score {baseline['wisdom_score']} must not decay post-GREEN",
            "never_change": False,
        },
    ]
    return recs


def _evolution_score(
    divisions: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    gaps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    div_avg = np.mean([d["composite_score"] for d in divisions])
    ops_resilience = (baseline["watchdog_stability"] + baseline["execution_stability"]) / 2
    gov_quality = divisions[1]["composite_score"]
    learning = baseline["wisdom_score"]
    adaptability = 100 - len([g for g in gaps if g["strategic_priority"] == "CRITICAL"]) * 8

    score = float(
        np.clip(
            div_avg * 0.30
            + ops_resilience * 0.20
            + gov_quality * 0.25
            + learning * 0.15
            + adaptability * 0.10,
            0,
            100,
        )
    )

    return {
        "triton_evolution_score": _f(score, 1),
        "classification": _evolution_band(score),
        "current_institutional_level": _current_institutional_level(divisions),
        "components": {
            "institutional_capabilities": _f(div_avg, 1),
            "operational_resilience": _f(ops_resilience, 1),
            "governance_quality": _f(gov_quality, 1),
            "learning_capability": _f(learning, 1),
            "strategic_adaptability": _f(adaptability, 1),
        },
    }


def _vision_plans(roadmap: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _filter(horizon: str) -> List[str]:
        return [r["capability"] for r in roadmap if r["horizon"] == horizon]

    return {
        "1_year": {
            "vision": "Certified institutional platform with concentration recovered and stable operations",
            "capabilities": _filter("1yr"),
            "target_level": "Institutional trading platform",
        },
        "3_year": {
            "vision": "Digital investment committee with multi-asset management and scenario planning",
            "capabilities": _filter("3yr"),
            "target_level": "Digital investment committee",
        },
        "5_year": {
            "vision": "Autonomous investment institution with human oversight on capital decisions only",
            "capabilities": _filter("5yr"),
            "target_level": "Autonomous investment institution",
        },
    }


def _executive_report(
    evolution: Dict[str, Any],
    divisions: List[Dict[str, Any]],
    strengths: List[Dict[str, Any]],
    weaknesses: List[Dict[str, Any]],
    gaps: List[Dict[str, Any]],
    critique: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    visions: Dict[str, Any],
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    critical_gaps = [g for g in gaps if g["strategic_priority"] == "CRITICAL"]
    stagnation = [c for c in critique if c.get("category") == "Stagnation Risk"]
    never_change = [r for r in recommendations if r.get("never_change")]

    return {
        "report_type": "TRITON_INSTITUTIONAL_SELF_EVALUATION_REPORT",
        "triton_evolution_score": evolution.get("triton_evolution_score"),
        "classification": evolution.get("classification"),
        "current_institutional_level": evolution.get("current_institutional_level"),
        "greatest_strengths": [s["title"] for s in strengths[:3]],
        "greatest_weaknesses": [w["title"] for w in weaknesses[:3]],
        "critical_strategic_gaps": [g["capability"] for g in critical_gaps],
        "evolution_plans": visions,
        "top_strategic_recommendations": [r["title"] for r in recommendations[:5]],
        "never_change": [r["title"] for r in never_change],
        "long_term_vision": visions["5_year"]["vision"],
        "risk_of_stagnation": stagnation[0]["issue"] if stagnation else "LOW",
        "executive_questions": {
            "what_is_triton_today": evolution.get("current_institutional_level"),
            "greatest_strengths": strengths[0]["title"] if strengths else "—",
            "greatest_weaknesses": weaknesses[0]["title"] if weaknesses else "—",
            "future_failure_risks": weaknesses[0]["title"] if weaknesses else "—",
            "what_to_build_next": recommendations[0]["title"] if recommendations else "—",
            "what_never_change": never_change[0]["title"] if never_change else "—",
            "world_class_requirements": "GREEN certification + multi-asset diversification + committee simulation",
        },
    }


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    baseline = _load_baseline()
    divisions = _maturity_divisions(baseline)
    gaps = _strategic_gaps(baseline)
    roadmap = _evolution_roadmap(baseline)
    strengths = _strengths(divisions, baseline)
    weaknesses = _weaknesses(baseline, divisions)
    critique = _self_critique(baseline)
    recommendations = _strategic_recommendations(gaps, strengths, weaknesses, baseline)
    evolution = _evolution_score(divisions, baseline, gaps)
    visions = _vision_plans(roadmap)
    executive = _executive_report(
        evolution,
        divisions,
        strengths,
        weaknesses,
        gaps,
        critique,
        recommendations,
        visions,
        baseline,
    )

    report = {
        "generated_at": ts,
        "phase": "208",
        "mode": "institutional_self_evaluation_engine",
        "disclaimer": (
            "Strategic intelligence only — does NOT modify governance rules, trading, "
            "allocations, thresholds, or automatically change future decisions."
        ),
        "triton_evolution": evolution,
        "institutional_maturity": {
            "divisions": divisions,
            "report_card": {d["division"]: d["grade"] for d in divisions},
        },
        "baseline_review": {
            "operational_maturity": {
                "paper_execution": baseline["paper_ready"],
                "continuous_loop": baseline["pipeline_ok"],
                "watchdog_stability": baseline["watchdog_stability"],
                "execution_stability": baseline["execution_stability"],
            },
            "governance_maturity": {
                "cps": baseline["cps"],
                "escalation": baseline["escalation_state"],
                "wisdom_score": baseline["wisdom_score"],
                "evidence_quality_pct": baseline["evidence_quality_pct"],
            },
            "investment_intelligence": {
                "concentration_pct": baseline["qqq_concentration_pct"],
                "diversification_score": baseline["diversification_score"],
                "position_count": baseline["position_count"],
                "lifecycle_ok": baseline["lifecycle_ok"],
            },
            "institutional_infrastructure": {
                "artifact_coverage_pct": baseline["artifact_coverage_pct"],
                "simulation_available": baseline["sim_available"],
            },
        },
        "strategic_gap_analysis": {
            "current_level": evolution.get("current_institutional_level"),
            "target_levels": visions,
            "gaps": gaps,
        },
        "future_evolution_roadmap": roadmap,
        "self_critique": critique,
        "institutional_strengths": strengths,
        "institutional_weaknesses": weaknesses,
        "strategic_recommendations": recommendations,
        "executive_recommendation": executive,
        "summary": {
            "evolution_score": evolution.get("triton_evolution_score"),
            "classification": evolution.get("classification"),
            "divisions_graded": len(divisions),
            "gaps_identified": len(gaps),
            "roadmap_items": len(roadmap),
            "strengths_count": len(strengths),
            "weaknesses_count": len(weaknesses),
            "recommendations_count": len(recommendations),
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(strengths).to_csv(OUT_STRENGTHS, index=False, encoding="utf-8")
    pd.DataFrame(weaknesses).to_csv(OUT_WEAKNESSES, index=False, encoding="utf-8")
    pd.DataFrame(roadmap).to_csv(OUT_ROADMAP, index=False, encoding="utf-8")
    pd.DataFrame(recommendations).to_csv(OUT_RECOMMENDATIONS, index=False, encoding="utf-8")

    _log(
        f"wrote {OUT_JSON.name}, {OUT_STRENGTHS.name}, {OUT_WEAKNESSES.name}, "
        f"{OUT_ROADMAP.name}, {OUT_RECOMMENDATIONS.name}"
    )
    _log(
        f"evolution={evolution.get('triton_evolution_score')} "
        f"({evolution.get('classification')}) "
        f"level={evolution.get('current_institutional_level')}"
    )
    return report


if __name__ == "__main__":
    run()
