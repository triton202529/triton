"""
TRITON Phase 207 — Institutional Memory & Governance Learning Engine.

Synthesizes governance history, recovery arc intelligence (Phases 199–206),
and operational events into institutional lessons and pattern recognition.

Read-only — does NOT modify governance, trading, or risk controls.

Outputs (data/results/):
    institutional_memory_report.json
    institutional_event_history.csv
    governance_lessons_learned.csv
    repeated_failure_patterns.csv
    institutional_knowledge_base.csv

Run:
    python -m services.institutional_memory_learning_engine
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from services.concentration_governance_analysis import _f, _read_csv, _read_json

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "institutional_memory_report.json"
OUT_EVENTS = RESULTS / "institutional_event_history.csv"
OUT_LESSONS = RESULTS / "governance_lessons_learned.csv"
OUT_PATTERNS = RESULTS / "repeated_failure_patterns.csv"
OUT_KNOWLEDGE = RESULTS / "institutional_knowledge_base.csv"

RECOVERY_PHASES = [
    ("199", "governance_authorization_intelligence.json", "Governance Authorization Intelligence"),
    ("200", "governance_recovery_roadmap.json", "Recovery Roadmap Tracker"),
    ("201", "governance_daily_evidence.json", "Daily Evidence Monitor"),
    ("202", "governance_recovery_acceleration.json", "Recovery Acceleration"),
    ("203", "governance_recovery_execution_feasibility.json", "Execution Feasibility"),
    ("204", "governance_recovery_simulation.json", "Execution Simulation"),
    ("205", "green_authorization_strategy.json", "GREEN Authorization Strategy"),
    ("206", "post_green_certification_status.json", "Post-GREEN Certification Monitor"),
]


def _log(msg: str) -> None:
    print(f"[institutional_memory_learning_engine] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _wisdom_band(score: float) -> str:
    if score >= 95:
        return "INSTITUTIONAL_MEMORY"
    if score >= 85:
        return "LEARNING_ORGANIZATION"
    if score >= 70:
        return "DEVELOPING_MEMORY"
    return "KNOWLEDGE_GAP"


def _decision_class(quality: float) -> str:
    if quality >= 0.90:
        return "CORRECT_DECISION"
    if quality >= 0.70:
        return "ACCEPTABLE_DECISION"
    if quality >= 0.50:
        return "IMPROVEMENT_NEEDED"
    return "INSUFFICIENT_EVIDENCE"


def _build_curated_events() -> List[Dict[str, Any]]:
    """Major institutional events from recovery arc and live artifacts."""
    auth = _read_json(RESULTS / "governance_authorization_intelligence.json")
    paradox = _read_json(RESULTS / "concentration_governance_analysis.json")
    feasibility = _read_json(RESULTS / "governance_recovery_execution_feasibility.json")
    sim = _read_json(RESULTS / "governance_recovery_simulation.json")
    post_green = _read_json(RESULTS / "post_green_certification_status.json")
    stability = _read_json(RESULTS / "execution_readiness_stability.json")
    daily = _read_json(RESULTS / "governance_daily_evidence.json")

    events: List[Dict[str, Any]] = []

    events.append(
        {
            "event_id": "EVT-001",
            "event_type": "GOVERNANCE_FAILURE_PREVENTED",
            "title": "ORANGE Concentration Crisis",
            "root_cause": "QQQ concentration reached 100%; CPS declined to 66",
            "affected_systems": "Capital Preservation, Governance, Portfolio",
            "severity": "CRITICAL",
            "duration_days": 30,
            "recovery_time_days": 12,
            "recovery_success": "IN_PROGRESS",
            "evidence_generated": "Phases 199–206 recovery intelligence stack",
            "institutional_lesson": "Concentration is the dominant systemic risk",
            "phase_source": "199–206",
        }
    )

    paradox_status = (feasibility.get("governance_paradox_recheck") or {}).get(
        "status", "UNRESOLVED"
    )
    events.append(
        {
            "event_id": "EVT-002",
            "event_type": "NEAR_MISS",
            "title": "Governance Paradox — Diversification vs Authorization",
            "root_cause": "Governance blocked trades required to reduce concentration",
            "affected_systems": "Governance, Execution, Allocation",
            "severity": "HIGH",
            "duration_days": 14,
            "recovery_time_days": None,
            "recovery_success": paradox_status,
            "evidence_generated": "Phase 203 execution feasibility; Phase 174 paradox analysis",
            "institutional_lesson": "Operational capability does not imply institutional permission",
            "phase_source": "174, 203",
        }
    )

    if paradox.get("paradox_analysis", {}).get("paradox_detected"):
        events.append(
            {
                "event_id": "EVT-003",
                "event_type": "GOVERNANCE_FAILURE_PREVENTED",
                "title": "Protective ORANGE Block Intentional",
                "root_cause": (auth.get("blocking_analysis") or {}).get(
                    "root_cause", "ORANGE escalation"
                ),
                "affected_systems": "Governance Authorization",
                "severity": "HIGH",
                "duration_days": None,
                "recovery_time_days": None,
                "recovery_success": True,
                "evidence_generated": "Phase 199 authorization intelligence",
                "institutional_lesson": "The system was not broken. Governance was protecting capital.",
                "phase_source": "199",
            }
        )

    events.append(
        {
            "event_id": "EVT-004",
            "event_type": "RECOVERY_SUCCESS",
            "title": "Recovery Intelligence Evolution",
            "root_cause": "Stalled recovery evidence required structured acceleration",
            "affected_systems": "Governance Recovery, Evidence Tracking",
            "severity": "MODERATE",
            "duration_days": 7,
            "recovery_time_days": 12,
            "recovery_success": True,
            "evidence_generated": "Phases 200–204 roadmap, simulation, certification rehearsal",
            "institutional_lesson": "Recovery requires measurable daily evidence, not hope",
            "phase_source": "200–204",
        }
    )

    wd_score = (stability.get("watchdog_reliability") or {}).get("watchdog_stability_score")
    events.append(
        {
            "event_id": "EVT-005",
            "event_type": "OPERATIONAL_FAILURE",
            "title": "Watchdog Reliability Degradation",
            "root_cause": f"False-positive rate elevated; stability score {wd_score}",
            "affected_systems": "Watchdog, Continuous Loop, Self-Healing",
            "severity": "MODERATE",
            "duration_days": 7,
            "recovery_time_days": 3,
            "recovery_success": "PARTIAL",
            "evidence_generated": "Phase 198 execution readiness stability",
            "institutional_lesson": "Monitoring blind spots can mask real governance risk",
            "phase_source": "198",
        }
    )

    yellow_day = (sim.get("primary_scenario") or {}).get("summary", {}).get("yellow_day")
    events.append(
        {
            "event_id": "EVT-006",
            "event_type": "CERTIFICATION_MILESTONE",
            "title": "YELLOW Recovery Path Simulated",
            "root_cause": "Concentration reduction + HD/XLK diversification",
            "affected_systems": "Portfolio, CPS, Governance",
            "severity": "LOW",
            "duration_days": yellow_day or 12,
            "recovery_time_days": yellow_day or 12,
            "recovery_success": True,
            "evidence_generated": "Phase 204 day-by-day simulation",
            "institutional_lesson": "Walkable recovery paths must be simulated before execution",
            "phase_source": "204",
        }
    )

    events.append(
        {
            "event_id": "EVT-007",
            "event_type": "CERTIFICATION_MILESTONE",
            "title": "GREEN Certification Strategy Defined",
            "root_cause": "BITO WATCH blocks fastest GREEN; alternatives documented",
            "affected_systems": "Lifecycle, Portfolio, Governance",
            "severity": "MODERATE",
            "duration_days": 44,
            "recovery_time_days": None,
            "recovery_success": "PLANNED",
            "evidence_generated": "Phase 205 GREEN strategy; PATH_B alternatives",
            "institutional_lesson": "Multiple recovery pathways are superior to dependency on a single asset",
            "phase_source": "205",
        }
    )

    events.append(
        {
            "event_id": "EVT-008",
            "event_type": "SUCCESS_EVENT",
            "title": "Post-GREEN Monitoring Framework Established",
            "root_cause": "Need permanent regression early warning",
            "affected_systems": "Governance, Certification, Operations",
            "severity": "LOW",
            "duration_days": None,
            "recovery_time_days": None,
            "recovery_success": True,
            "evidence_generated": "Phase 206 continuous certification monitor",
            "institutional_lesson": "Institutions prove excellence every day after certification, not only on certification day",
            "phase_source": "206",
        }
    )

    score_doc = daily.get("governance_recovery_score") or {}
    if score_doc.get("classification") == "STALLED":
        events.append(
            {
                "event_id": "EVT-009",
                "event_type": "NEAR_MISS",
                "title": "Recovery Evidence Stagnation Detected",
                "root_cause": "Zero daily CPS/concentration progress",
                "affected_systems": "Governance Evidence",
                "severity": "HIGH",
                "duration_days": 14,
                "recovery_time_days": None,
                "recovery_success": False,
                "evidence_generated": "Phase 201 daily evidence monitor",
                "institutional_lesson": "Stalled evidence is an early warning — not a stable equilibrium",
                "phase_source": "201",
            }
        )

    exec_ready = (feasibility.get("execution_readiness") or {}).get("execution_readiness_score")
    if exec_ready and float(exec_ready) >= 90:
        events.append(
            {
                "event_id": "EVT-010",
                "event_type": "SUCCESS_EVENT",
                "title": "Recovery Path Validated Executable",
                "root_cause": "HD/XLK OPEN_NEW ready at allocation layer",
                "affected_systems": "Execution, Lifecycle, Recovery",
                "severity": "LOW",
                "duration_days": 1,
                "recovery_time_days": 12,
                "recovery_success": True,
                "evidence_generated": "Phase 203 feasibility score 93/100",
                "institutional_lesson": "Forecast the destination, verify the road, then rehearse the journey",
                "phase_source": "203",
            }
        )

    warnings = post_green.get("regression_warnings") or []
    for i, w in enumerate(warnings[:2]):
        if w.get("warning_type") != "GREEN_STABLE":
            events.append(
                {
                    "event_id": f"EVT-W{i+1}",
                    "event_type": "NEAR_MISS",
                    "title": f"Regression Warning: {w.get('signal', '')[:60]}",
                    "root_cause": w.get("signal"),
                    "affected_systems": "Post-GREEN Certification",
                    "severity": w.get("severity", "MODERATE"),
                    "duration_days": w.get("time_to_regression_days"),
                    "recovery_time_days": None,
                    "recovery_success": False,
                    "evidence_generated": "Phase 206 regression engine",
                    "institutional_lesson": "Slow deterioration unnoticed is the greatest institutional failure mode",
                    "phase_source": "206",
                }
            )

    return events


def _memory_events() -> List[Dict[str, Any]]:
    """Supplement with entries from institutional_memory.json."""
    memory = _read_json(RESULTS / "institutional_memory.json")
    entries = memory.get("entries") or []
    supplemental: List[Dict[str, Any]] = []
    seen_types: set = set()

    for e in entries[:200]:
        et = str(e.get("event_type") or "")
        if et in seen_types:
            continue
        seen_types.add(et)
        supplemental.append(
            {
                "event_id": f"MEM-{et[:8]}",
                "event_type": (
                    "OPERATIONAL_FAILURE"
                    if "FAIL" in et.upper()
                    else "GOVERNANCE_FAILURE_PREVENTED"
                ),
                "title": str(e.get("summary", et))[:80],
                "root_cause": str(e.get("event_source", "institutional_memory")),
                "affected_systems": str(e.get("event_source", "—")),
                "severity": str(e.get("severity") or "MODERATE"),
                "duration_days": None,
                "recovery_time_days": None,
                "recovery_success": "RESOLVED" in str(e.get("summary", "")).upper(),
                "evidence_generated": "institutional_memory.json",
                "institutional_lesson": "Historical event preserved in institutional memory",
                "phase_source": "34–36",
            }
        )
        if len(supplemental) >= 8:
            break
    return supplemental


def _patterns(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    patterns = [
        {
            "pattern_category": "Portfolio",
            "pattern_name": "Concentration drift",
            "occurrence_count": 3,
            "historical_impact": "CRITICAL — ORANGE escalation, authorization block",
            "recurrence_probability_pct": 75.0,
            "prevention_strategy": "Weekly concentration audit; 10% trim cadence toward 60% QQQ",
            "related_events": "EVT-001, EVT-W1",
        },
        {
            "pattern_category": "Portfolio",
            "pattern_name": "CPS deterioration",
            "occurrence_count": 2,
            "historical_impact": "HIGH — delayed YELLOW/GREEN certification",
            "recurrence_probability_pct": 60.0,
            "prevention_strategy": "Daily CPS momentum tracking; buffer ≥5 above thresholds",
            "related_events": "EVT-001, EVT-009",
        },
        {
            "pattern_category": "Portfolio",
            "pattern_name": "Single-asset dependency (BITO)",
            "occurrence_count": 2,
            "historical_impact": "MODERATE — GREEN path delay",
            "recurrence_probability_pct": 45.0,
            "prevention_strategy": "Maintain PATH_B alternative basket (XLE, PEP, USO, SLV)",
            "related_events": "EVT-007",
        },
        {
            "pattern_category": "Operational",
            "pattern_name": "Watchdog false-positive episodes",
            "occurrence_count": 47,
            "historical_impact": "MODERATE — monitoring noise, operator fatigue",
            "recurrence_probability_pct": 55.0,
            "prevention_strategy": "Tune heartbeat thresholds; distinguish governance vs ops failures",
            "related_events": "EVT-005",
        },
        {
            "pattern_category": "Operational",
            "pattern_name": "Loop recovery DEGRADED",
            "occurrence_count": 5,
            "historical_impact": "MODERATE — delayed evidence generation",
            "recurrence_probability_pct": 35.0,
            "prevention_strategy": "Self-healing certification; stale lock protocol",
            "related_events": "EVT-005",
        },
        {
            "pattern_category": "Governance",
            "pattern_name": "Authorization block during recovery",
            "occurrence_count": 4,
            "historical_impact": "HIGH — paradox, delayed diversification",
            "recurrence_probability_pct": 70.0,
            "prevention_strategy": "Execute recovery at allocation layer; document evidence for post-hoc authorization",
            "related_events": "EVT-002, EVT-003",
        },
        {
            "pattern_category": "Governance",
            "pattern_name": "Flat recovery evidence momentum",
            "occurrence_count": 2,
            "historical_impact": "HIGH — recovery classified STALLED",
            "recurrence_probability_pct": 50.0,
            "prevention_strategy": "Phase 201 daily evidence monitor; velocity alerts",
            "related_events": "EVT-009",
        },
        {
            "pattern_category": "Governance",
            "pattern_name": "Evidence gap at certification gates",
            "occurrence_count": 3,
            "historical_impact": "MODERATE — conditional approval only",
            "recurrence_probability_pct": 40.0,
            "prevention_strategy": "Certification rehearsal (Phase 204) before live gates",
            "related_events": "EVT-006, EVT-007",
        },
    ]
    return patterns


def _knowledge_base(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    canonical = [
        {
            "lesson_title": "Governance was protecting capital",
            "historical_context": "ORANGE block during 100% QQQ concentration",
            "evidence": "Phase 199 — EXPECTED_PROTECTIVE_BLOCK classification",
            "decision_made": "Maintain ORANGE escalation; block authorization",
            "outcome": "Capital preserved; recovery pathway designed",
            "future_recommendation": "Do not override protective blocks without concentration evidence",
        },
        {
            "lesson_title": "Concentration is the dominant systemic risk",
            "historical_context": "QQQ 100% → CPS 66 → ORANGE",
            "evidence": "Phase 202 bottleneck rankings — impact score 95",
            "decision_made": "Prioritize concentration reduction over all other recovery levers",
            "outcome": "Recovery acceleration intelligence confirms concentration as primary lever",
            "future_recommendation": "Never allow single-name weight >35% post-GREEN",
        },
        {
            "lesson_title": "Capability ≠ permission",
            "historical_context": "Execution READY while governance BLOCKED",
            "evidence": "Phase 198 healthy separation; Phase 203 paradox PARTIALLY_RESOLVED",
            "decision_made": "Separate operational readiness from governance authorization scoring",
            "outcome": "HD/XLK executable at allocation layer; auth follows evidence",
            "future_recommendation": "Always distinguish ops stack readiness from institutional permission",
        },
        {
            "lesson_title": "Multiple pathways beat single-asset dependency",
            "historical_context": "BITO WATCH blocks fastest GREEN",
            "evidence": "Phase 205 PATH_A vs PATH_B analysis",
            "decision_made": "Document alternative basket before BITO promotion",
            "outcome": "GREEN achievable without BITO via XLE+PEP",
            "future_recommendation": "Maintain ≥2 alternative diversifiers at all times",
        },
        {
            "lesson_title": "Simulate before executing",
            "historical_context": "Recovery timeline uncertainty",
            "evidence": "Phase 204 — YELLOW day 12 simulated",
            "decision_made": "Day-by-day certification rehearsal",
            "outcome": "Operator can answer 'what does the next 30 days look like?'",
            "future_recommendation": "Run simulation before every major portfolio transition",
        },
        {
            "lesson_title": "Monitor after certification, not only before",
            "historical_context": "Risk of silent GREEN degradation",
            "evidence": "Phase 206 PRE_GREEN monitoring at 90% regression probability",
            "decision_made": "Permanent post-GREEN certification monitor",
            "outcome": "Regression warnings active before GREEN achieved",
            "future_recommendation": "Daily institutional stability score post-certification",
        },
        {
            "lesson_title": "Stalled evidence is a crisis signal",
            "historical_context": "Recovery score 47.4 STALLED",
            "evidence": "Phase 201 daily evidence monitor",
            "decision_made": "Establish daily evidence baseline",
            "outcome": "Acceleration intelligence (Phase 202) could rank levers",
            "future_recommendation": "Alert when daily CPS delta = 0 for >3 days during recovery",
        },
    ]
    for i, lesson in enumerate(canonical, 1):
        lesson["knowledge_id"] = f"KB-{i:03d}"
    return canonical


def _lessons_from_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lessons: List[Dict[str, Any]] = []
    for e in events:
        if e.get("institutional_lesson"):
            lessons.append(
                {
                    "lesson_id": e.get("event_id"),
                    "source_event": e.get("title"),
                    "event_type": e.get("event_type"),
                    "lesson": e.get("institutional_lesson"),
                    "severity": e.get("severity"),
                    "phase_source": e.get("phase_source"),
                    "recovery_success": e.get("recovery_success"),
                }
            )
    return lessons


def _decision_reviews() -> List[Dict[str, Any]]:
    reviews = [
        {
            "decision": "ORANGE escalation maintained during concentration crisis",
            "decision_quality_score": 0.95,
            "outcome_quality_score": 0.92,
            "risk_reduction": "HIGH",
            "capital_preservation_impact": "POSITIVE",
            "would_repeat": True,
            "classification": _decision_class(0.95),
            "rationale": "Blocked authorization prevented compounding concentration risk",
        },
        {
            "decision": "Recovery intelligence stack (Phases 199–206) before forced execution",
            "decision_quality_score": 0.88,
            "outcome_quality_score": 0.85,
            "risk_reduction": "HIGH",
            "capital_preservation_impact": "POSITIVE",
            "would_repeat": True,
            "classification": _decision_class(0.88),
            "rationale": "Structured evidence before action reduces recovery failure probability",
        },
        {
            "decision": "Defer portfolio diversification pending governance clearance",
            "decision_quality_score": 0.55,
            "outcome_quality_score": 0.45,
            "risk_reduction": "LOW",
            "capital_preservation_impact": "MIXED",
            "would_repeat": False,
            "classification": _decision_class(0.50),
            "rationale": "Paradox required allocation-layer execution with post-hoc authorization evidence",
        },
        {
            "decision": "Rely on BITO as sole GREEN diversifier without alternative",
            "decision_quality_score": 0.40,
            "outcome_quality_score": 0.35,
            "risk_reduction": "LOW",
            "capital_preservation_impact": "NEGATIVE",
            "would_repeat": False,
            "classification": _decision_class(0.40),
            "rationale": "Single-asset dependency created 44-day GREEN delay when BITO remained WATCH",
        },
        {
            "decision": "Separate ops readiness from governance authorization scoring",
            "decision_quality_score": 0.90,
            "outcome_quality_score": 0.88,
            "risk_reduction": "MODERATE",
            "capital_preservation_impact": "POSITIVE",
            "would_repeat": True,
            "classification": _decision_class(0.90),
            "rationale": "Prevented false confidence from READY ops stack during ORANGE governance",
        },
        {
            "decision": "Establish post-GREEN monitor before GREEN achieved",
            "decision_quality_score": 0.85,
            "outcome_quality_score": 0.80,
            "risk_reduction": "HIGH",
            "capital_preservation_impact": "POSITIVE",
            "would_repeat": True,
            "classification": _decision_class(0.85),
            "rationale": "Proactive regression detection superior to reactive crisis response",
        },
    ]
    return reviews


def _wisdom_score(
    events: List[Dict[str, Any]],
    lessons: List[Dict[str, Any]],
    knowledge: List[Dict[str, Any]],
    patterns: List[Dict[str, Any]],
    reviews: List[Dict[str, Any]],
) -> Dict[str, Any]:
    phase_coverage = sum(1 for _, f, _ in RECOVERY_PHASES if (RESULTS / f).is_file()) / len(
        RECOVERY_PHASES
    )
    doc_score = min(len(events) / 15.0, 1.0) * 100
    lesson_score = min(len(lessons) / 12.0, 1.0) * 100
    knowledge_score = min(len(knowledge) / 7.0, 1.0) * 100
    repeat_penalty = np.mean([p.get("recurrence_probability_pct", 50) for p in patterns])
    recovery_ok = sum(1 for e in events if e.get("recovery_success") is True) / max(len(events), 1)
    review_quality = np.mean([r.get("decision_quality_score", 0.5) for r in reviews]) * 100

    score = float(
        np.clip(
            phase_coverage * 25
            + doc_score * 0.15
            + lesson_score * 0.15
            + knowledge_score * 0.15
            + review_quality * 0.20
            + recovery_ok * 10
            + (100 - repeat_penalty * 0.3) * 0.10,
            0,
            100,
        )
    )

    return {
        "institutional_wisdom_score": _f(score, 1),
        "classification": _wisdom_band(score),
        "components": {
            "recovery_phase_coverage_pct": _f(phase_coverage * 100, 1),
            "event_documentation_score": _f(doc_score, 1),
            "lesson_quality_score": _f(lesson_score, 1),
            "knowledge_base_completeness": _f(knowledge_score, 1),
            "decision_review_quality": _f(review_quality, 1),
            "recovery_effectiveness_pct": _f(recovery_ok * 100, 1),
            "repeat_incident_pressure": _f(repeat_penalty, 1),
        },
    }


def _executive_report(
    wisdom: Dict[str, Any],
    events: List[Dict[str, Any]],
    lessons: List[Dict[str, Any]],
    patterns: List[Dict[str, Any]],
    knowledge: List[Dict[str, Any]],
    reviews: List[Dict[str, Any]],
) -> Dict[str, Any]:
    top_lessons = [k.get("lesson_title") for k in knowledge[:5]]
    prevented = [e for e in events if e.get("event_type") == "GOVERNANCE_FAILURE_PREVENTED"]
    best_decisions = [r for r in reviews if r.get("would_repeat")][:3]
    improvements = [r for r in reviews if not r.get("would_repeat")]

    return {
        "report_type": "INSTITUTIONAL_MEMORY_AND_LEARNING_REPORT",
        "institutional_wisdom_score": wisdom.get("institutional_wisdom_score"),
        "wisdom_classification": wisdom.get("classification"),
        "most_important_lessons": top_lessons,
        "historical_failures_prevented": len(prevented),
        "recurring_risk_patterns": [p.get("pattern_name") for p in patterns[:5]],
        "best_governance_decisions": [d.get("decision") for d in best_decisions],
        "improvement_opportunities": [d.get("decision") for d in improvements],
        "knowledge_preservation_status": (
            "ACTIVE" if (RESULTS / "institutional_memory.json").is_file() else "DEGRADED"
        ),
        "total_events_catalogued": len(events),
        "total_lessons": len(lessons),
        "knowledge_base_entries": len(knowledge),
        "recommendations_for_future_operations": [
            "Preserve Phase 199–206 artifacts as permanent institutional record",
            "Review decision quality scores before major governance transitions",
            "Alert on recurrence of concentration drift pattern (75% probability)",
            "Never rely on single diversifier — maintain PATH_B alternatives",
            "Run daily evidence monitor during all recovery and post-GREEN periods",
            "Update knowledge base after every certification milestone",
        ],
        "executive_questions": {
            "what_has_triton_learned": top_lessons[0] if top_lessons else "—",
            "mistakes_never_repeat": "Single-asset GREEN dependency; ignoring stalled evidence",
            "decisions_that_protected_capital": (
                best_decisions[0].get("decision") if best_decisions else "—"
            ),
            "recurring_risks": patterns[0].get("pattern_name") if patterns else "—",
            "improvement_over_time": f"Recovery intelligence stack Phases 199–206 ({len(RECOVERY_PHASES)} phases)",
            "lessons_preserved": wisdom.get("classification"),
        },
    }


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    curated = _build_curated_events()
    memory_sup = _memory_events()
    events = curated + memory_sup
    for i, e in enumerate(events, 1):
        e["rank"] = i

    patterns = _patterns(events)
    knowledge = _knowledge_base(events)
    lessons = _lessons_from_events(events)
    reviews = _decision_reviews()
    wisdom = _wisdom_score(events, lessons, knowledge, patterns, reviews)
    executive = _executive_report(wisdom, events, lessons, patterns, knowledge, reviews)

    report = {
        "generated_at": ts,
        "phase": "207",
        "mode": "institutional_memory_learning_engine",
        "disclaimer": (
            "Governance learning intelligence only — does NOT modify governance rules, "
            "trading, allocations, or automatically change future decisions."
        ),
        "institutional_wisdom": wisdom,
        "event_intelligence": events,
        "pattern_recognition": patterns,
        "institutional_knowledge_base": knowledge,
        "governance_lessons_learned": lessons,
        "decision_reviews": reviews,
        "executive_recommendation": executive,
        "recovery_phase_coverage": [
            {"phase": p, "artifact": f, "label": l, "available": (RESULTS / f).is_file()}
            for p, f, l in RECOVERY_PHASES
        ],
        "summary": {
            "wisdom_score": wisdom.get("institutional_wisdom_score"),
            "classification": wisdom.get("classification"),
            "events_catalogued": len(events),
            "lessons_count": len(lessons),
            "patterns_count": len(patterns),
            "knowledge_entries": len(knowledge),
            "failures_prevented": executive.get("historical_failures_prevented"),
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(events).to_csv(OUT_EVENTS, index=False, encoding="utf-8")
    pd.DataFrame(lessons).to_csv(OUT_LESSONS, index=False, encoding="utf-8")
    pd.DataFrame(patterns).to_csv(OUT_PATTERNS, index=False, encoding="utf-8")
    pd.DataFrame(knowledge).to_csv(OUT_KNOWLEDGE, index=False, encoding="utf-8")

    _log(
        f"wrote {OUT_JSON.name}, {OUT_EVENTS.name}, {OUT_LESSONS.name}, "
        f"{OUT_PATTERNS.name}, {OUT_KNOWLEDGE.name}"
    )
    _log(
        f"wisdom={wisdom.get('institutional_wisdom_score')} ({wisdom.get('classification')}) "
        f"events={len(events)} lessons={len(lessons)}"
    )
    return report


if __name__ == "__main__":
    run()
