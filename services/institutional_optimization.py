"""
TRITON Institutional Optimization — Phases 46–48.

Attention allocation, system coordination, and institutional optimization engines.
Paper-mode / simulation only. NO live trading, orders, or portfolio changes.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from services.institutional_protection import _atomic_write_json, _iso_utc

ATTENTION_AREAS = (
    "Governance",
    "Readiness",
    "Certification",
    "Risk",
    "Oversight",
    "Preservation",
)

AREA_DISPLAY = {
    "Governance": "Governance Alignment",
    "Readiness": "Execution Readiness",
    "Certification": "Certification",
    "Risk": "Risk Management",
    "Oversight": "Strategic Oversight",
    "Preservation": "Capital Preservation",
}

GRAPH_AREA_MAP = {
    "Capital Preservation": "Preservation",
    "Monitoring": "Risk",
    "Certification": "Certification",
    "Governance": "Governance",
    "Readiness": "Readiness",
    "Oversight": "Oversight",
    "Authorization": "Governance",
    "Policy": "Governance",
    "Evaluation": "Certification",
}

OPTIMIZATION_HISTORY_FIELDNAMES = [
    "timestamp",
    "top_optimization",
    "optimization_score",
    "expected_system_benefit",
    "coordination_score",
    "highest_attention_area",
]


def _clamp_score(value: float, lo: int = 0, hi: int = 100) -> int:
    return int(min(hi, max(lo, round(value))))


def _readiness_score(readiness_doc: Dict[str, Any]) -> int:
    status = str(readiness_doc.get("readiness_status") or "NOT_READY").upper()
    passing = int(readiness_doc.get("checks_passing_count") or 0)
    total = max(1, int(readiness_doc.get("checks_total") or 8))
    ratio = passing / total * 100
    if status == "READY":
        return _clamp_score(max(ratio, 85))
    if status == "PARTIALLY_READY":
        return _clamp_score(ratio * 0.85 + 10)
    return _clamp_score(ratio * 0.6)


def _cert_score(cert_doc: Dict[str, Any]) -> int:
    return _clamp_score(float(cert_doc.get("certification_score") or 0))


def _normalize_focus_percent(scores: Dict[str, int]) -> Dict[str, int]:
    total = sum(max(0, s) for s in scores.values()) or 1
    raw = {k: max(0, s) / total * 100 for k, s in scores.items()}
    rounded = {k: int(round(v)) for k, v in raw.items()}
    drift = 100 - sum(rounded.values())
    if drift and rounded:
        top_key = max(rounded, key=rounded.get)
        rounded[top_key] = max(0, rounded[top_key] + drift)
    return rounded


def _priority_boost(priorities_doc: Dict[str, Any], keywords: Tuple[str, ...]) -> int:
    boost = 0
    for p in priorities_doc.get("priorities") or []:
        focus = str(p.get("focus_area") or "").lower()
        impact = int(p.get("expected_impact") or 0)
        if any(kw in focus for kw in keywords):
            boost = max(boost, _clamp_score(impact * 0.35))
    return boost


def compute_attention_allocation(
    *,
    priorities_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    strategic_reasoning_doc: Dict[str, Any],
    consequence_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    strategic_doc: Dict[str, Any],
    insights_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 46 — score attention need across six institutional areas."""
    ts = _iso_utc()
    readiness = _readiness_score(readiness_doc)
    cert = _cert_score(cert_doc)
    failed = readiness_doc.get("failed_checks") or []
    posture = str(governor_doc.get("preservation_posture") or "YELLOW").upper()
    escalation = str(cpe_doc.get("escalation_state") or "GREEN").upper()
    high_severity = sum(
        1 for f in (consequence_doc.get("forecasts") or []) if f.get("severity") == "HIGH"
    )
    top_issue = str(strategic_reasoning_doc.get("top_strategic_issue") or "")
    gov_concern = bool(insights_doc.get("most_important_governance_concern"))
    oversight_conf = int(strategic_doc.get("strategic_confidence") or 55)

    uncertified = sum(
        1
        for a in (cert_doc.get("areas") or {}).values()
        if isinstance(a, dict) and not a.get("certified")
    )

    raw_scores = {
        "Governance": _clamp_score(
            (45 if gov_concern else 25)
            + (30 if "governance" in failed else 0)
            + _priority_boost(priorities_doc, ("governance", "authorization"))
            + (15 if "governance" in top_issue.lower() else 0)
        ),
        "Readiness": _clamp_score(
            100
            - readiness
            + len(failed) * 8
            + (20 if "readiness" in top_issue.lower() or top_issue == "Execution Readiness" else 0)
            + _priority_boost(priorities_doc, ("readiness", "execution"))
        ),
        "Certification": _clamp_score(
            100
            - cert
            + uncertified * 6
            + _priority_boost(priorities_doc, ("certification", "certified"))
        ),
        "Risk": _clamp_score(
            {"GREEN": 20, "YELLOW": 40, "ORANGE": 60, "RED": 80, "CRITICAL": 92}.get(escalation, 35)
            + high_severity * 5
            + (15 if posture in {"RED", "CRITICAL"} else 8 if posture == "ORANGE" else 0)
        ),
        "Oversight": _clamp_score(
            100
            - oversight_conf
            + (25 if strategic_doc.get("oversight_status") != "ALIGNED" else 8)
            + _priority_boost(priorities_doc, ("oversight", "committee"))
        ),
        "Preservation": _clamp_score(
            100
            - int(governor_doc.get("capital_preservation_score") or 50)
            + {"GREEN": 5, "YELLOW": 15, "ORANGE": 25, "RED": 35, "CRITICAL": 45}.get(posture, 20)
        ),
    }

    focus = _normalize_focus_percent(raw_scores)
    allocations: List[Dict[str, Any]] = []

    rationales = {
        "Governance": (
            f"Governance awareness and failed checks ({', '.join(failed) or 'none'}) "
            f"drive alignment attention; top issue context={top_issue[:40]}."
        ),
        "Readiness": (
            f"Readiness {readiness_doc.get('readiness_status')} with "
            f"{readiness_doc.get('checks_passing_count', 0)}/"
            f"{readiness_doc.get('checks_total', 8)} checks passing."
        ),
        "Certification": (
            f"Certification score {cert} with {uncertified} uncertified areas "
            f"({cert_doc.get('certification_status', 'unknown')})."
        ),
        "Risk": (
            f"Escalation {escalation}, governor posture {posture}, "
            f"{high_severity} high-severity consequence forecasts."
        ),
        "Oversight": (
            f"Strategic oversight confidence {oversight_conf}%; "
            f"status={strategic_doc.get('oversight_status', 'unknown')}."
        ),
        "Preservation": (
            f"CPI {governor_doc.get('capital_preservation_score')} with "
            f"preservation posture {posture}."
        ),
    }

    for area in ATTENTION_AREAS:
        allocations.append(
            {
                "area": area,
                "attention_score": raw_scores[area],
                "recommended_focus_percent": focus[area],
                "rationale": rationales[area],
            }
        )

    allocations.sort(key=lambda a: (-a["attention_score"], a["area"]))
    top = allocations[0]
    top_area = top["area"]

    return {
        "generated_at": ts,
        "highest_attention_area": AREA_DISPLAY.get(top_area, top_area),
        "attention_score": top["attention_score"],
        "recommended_focus_percent": top["recommended_focus_percent"],
        "allocations": allocations,
        "disclaimer": (
            "Attention allocation is advisory focus guidance only. "
            "No trades, orders, or automated intervention."
        ),
    }


def _edge_strength(edge_type: str) -> int:
    return {
        "SUPPORTS": 85,
        "INFORMS": 75,
        "TRIGGERS": 65,
        "ESCALATES_TO": 70,
        "BLOCKS": 35,
        "CONSTRAINS": 40,
    }.get(edge_type, 60)


def _graph_connections(graph_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    graph = graph_doc.get("graph") or {}
    nodes = {n["id"]: n for n in (graph.get("nodes") or []) if n.get("id")}
    pair_weights: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for edge in graph.get("edges") or []:
        src = nodes.get(edge.get("source"), {})
        tgt = nodes.get(edge.get("target"), {})
        src_area = GRAPH_AREA_MAP.get(str(src.get("area") or ""), "")
        tgt_area = GRAPH_AREA_MAP.get(str(tgt.get("area") or ""), "")
        if not src_area or not tgt_area or src_area == tgt_area:
            continue
        key = tuple(sorted((src_area, tgt_area)))
        pair_weights[key].append(_edge_strength(str(edge.get("type") or "")))

    connections: List[Dict[str, Any]] = []
    for (a, b), strengths in pair_weights.items():
        avg = _clamp_score(sum(strengths) / len(strengths))
        connections.append(
            {
                "from": a,
                "to": b,
                "strength": avg,
                "edge_count": len(strengths),
            }
        )

    connections.sort(key=lambda c: (-c["strength"], c["from"], c["to"]))
    return connections


def _alignment_score(*values: int) -> int:
    if not values:
        return 50
    spread = max(values) - min(values)
    avg = sum(values) / len(values)
    return _clamp_score(avg - spread * 0.35)


def compute_system_coordination(
    *,
    graph_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    strategic_doc: Dict[str, Any],
    priorities_doc: Dict[str, Any],
    future_paths_doc: Dict[str, Any],
    intelligence_doc: Dict[str, Any],
    maturity_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 47 — measure coordination across institutional systems."""
    ts = _iso_utc()
    readiness = _readiness_score(readiness_doc)
    cert = _cert_score(cert_doc)
    cpi = int(governor_doc.get("capital_preservation_score") or 50)
    intel_coord = int(intelligence_doc.get("coordination_score") or 55)
    maturity = int(maturity_doc.get("overall_maturity") or 50)

    cat_scores = {
        c["category"]: int(c["score"])
        for c in (maturity_doc.get("categories") or [])
        if c.get("category")
    }

    dimensions = {
        "governance_coordination": _alignment_score(
            cat_scores.get("Governance", 40),
            100 - len(readiness_doc.get("failed_checks") or []) * 12,
            intel_coord,
        ),
        "oversight_coordination": _alignment_score(
            int(strategic_doc.get("strategic_confidence") or 55),
            cat_scores.get("Oversight", 35),
            maturity,
        ),
        "preservation_coordination": _alignment_score(
            cpi,
            100
            - {"GREEN": 5, "YELLOW": 15, "ORANGE": 30, "RED": 45, "CRITICAL": 55}.get(
                str(governor_doc.get("preservation_posture") or "YELLOW").upper(), 20
            ),
            cat_scores.get("Preservation", 50),
        ),
        "certification_coordination": _alignment_score(cert, readiness, intel_coord),
        "planning_coordination": _alignment_score(
            len(priorities_doc.get("priorities") or []) * 8 + 20,
            (
                (future_paths_doc.get("paths") or [{}])[0].get("expected_benefit", 50)
                if future_paths_doc.get("paths")
                else 50
            ),
            maturity,
        ),
    }

    connections = _graph_connections(graph_doc)
    if not connections:
        connections = [
            {"from": "Governance", "to": "Preservation", "strength": 72, "edge_count": 0},
            {"from": "Readiness", "to": "Certification", "strength": 48, "edge_count": 0},
        ]

    strongest = connections[0]
    weakest = connections[-1]
    overall = _clamp_score(
        sum(dimensions.values()) / len(dimensions) * 0.65
        + (sum(c["strength"] for c in connections[:5]) / max(1, min(5, len(connections)))) * 0.35
    )

    return {
        "generated_at": ts,
        "coordination_score": overall,
        "strongest_connection": f"{strongest['from']} → {strongest['to']}",
        "weakest_connection": f"{weakest['from']} → {weakest['to']}",
        "dimensions": dimensions,
        "connections": connections,
        "disclaimer": (
            "System coordination is observational alignment analysis only. "
            "No execution, orders, or portfolio changes."
        ),
    }


def compute_institutional_optimization(
    *,
    priorities_doc: Dict[str, Any],
    improvement_doc: Dict[str, Any],
    maturity_doc: Dict[str, Any],
    future_paths_doc: Dict[str, Any],
    attention_doc: Dict[str, Any],
    coordination_doc: Dict[str, Any],
    strategic_reasoning_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 48 — rank optimization opportunities across institutional systems."""
    ts = _iso_utc()
    top_priority = str(priorities_doc.get("top_priority") or "Execution Readiness")
    leverage = str(
        priorities_doc.get("highest_leverage_improvement")
        or (improvement_doc.get("highest_leverage_enhancements") or ["Governance"])[0]
    )
    maturity = int(maturity_doc.get("overall_maturity") or 50)
    improvement_score = int(improvement_doc.get("improvement_score") or 70)
    top_attention = attention_doc.get("highest_attention_area") or top_priority
    attention_score = int(attention_doc.get("attention_score") or 80)
    coord_score = int(coordination_doc.get("coordination_score") or 70)

    paths = future_paths_doc.get("paths") or []
    accelerated = next((p for p in paths if p.get("path") == "ACCELERATED_IMPROVEMENT"), {})
    current = next((p for p in paths if p.get("path") == "CURRENT_PATH"), {})
    governance_path = next((p for p in paths if p.get("path") == "GOVERNANCE_IMPROVEMENT"), {})

    optimizations: List[Dict[str, Any]] = [
        {
            "type": "highest_roi",
            "focus": top_priority,
            "score": _clamp_score(
                attention_score * 0.4 + improvement_score * 0.35 + coord_score * 0.25
            ),
            "expected_benefit": _clamp_score(
                (priorities_doc.get("priorities") or [{}])[0].get("expected_impact", 75)
            ),
            "timeframe": "90d",
            "rationale": (
                "Top strategic priority with highest attention score and cross-system benefit."
            ),
        },
        {
            "type": "highest_leverage",
            "focus": leverage[:120],
            "score": _clamp_score(improvement_score),
            "expected_benefit": _clamp_score(improvement_score * 0.92),
            "timeframe": "60d",
            "rationale": "Highest-leverage enhancement from self-improvement synthesis.",
        },
        {
            "type": "fastest",
            "focus": accelerated.get("path", "ACCELERATED_IMPROVEMENT"),
            "score": _clamp_score(accelerated.get("expected_benefit", 65)),
            "expected_benefit": _clamp_score(accelerated.get("confidence", 60)),
            "timeframe": "30d",
            "rationale": (
                "Accelerated improvement path with shortest milestone horizon in paper mode."
            ),
        },
        {
            "type": "longest_term",
            "focus": governance_path.get("path", "GOVERNANCE_IMPROVEMENT"),
            "score": _clamp_score(governance_path.get("expected_benefit", maturity)),
            "expected_benefit": _clamp_score(governance_path.get("confidence", maturity)),
            "timeframe": "180d",
            "rationale": ("Governance-first trajectory for durable institutional maturity gains."),
        },
    ]

    optimizations.sort(key=lambda o: (-o["score"], o["type"]))
    top = optimizations[0]
    expected_benefit = _clamp_score(sum(o["expected_benefit"] for o in optimizations[:2]) / 2)

    return {
        "generated_at": ts,
        "top_optimization": top_attention if top["type"] == "highest_roi" else top["focus"],
        "optimization_score": top["score"],
        "expected_system_benefit": expected_benefit,
        "highest_roi_improvement": top_priority,
        "highest_leverage_improvement": leverage[:120],
        "fastest_improvement": accelerated.get("path", "ACCELERATED_IMPROVEMENT"),
        "longest_term_improvement": governance_path.get("path", "GOVERNANCE_IMPROVEMENT"),
        "recommended_path": future_paths_doc.get("recommended_path"),
        "strategic_importance": strategic_reasoning_doc.get("strategic_importance"),
        "current_path_benefit": current.get("expected_benefit"),
        "optimizations": optimizations,
        "disclaimer": (
            "Institutional optimization is advisory prioritization only. "
            "No trades, orders, or automated intervention."
        ),
    }


def _append_optimization_history(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OPTIMIZATION_HISTORY_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in OPTIMIZATION_HISTORY_FIELDNAMES})


def persist_institutional_optimization(
    *,
    results_dir: Path,
    priorities_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    strategic_reasoning_doc: Dict[str, Any],
    consequence_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    strategic_doc: Dict[str, Any],
    insights_doc: Dict[str, Any],
    graph_doc: Dict[str, Any],
    future_paths_doc: Dict[str, Any],
    intelligence_doc: Dict[str, Any],
    maturity_doc: Dict[str, Any],
    improvement_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Run phases 46–48 and write JSON artifacts."""
    results_dir = Path(results_dir)

    attention_doc = compute_attention_allocation(
        priorities_doc=priorities_doc,
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
        governor_doc=governor_doc,
        strategic_reasoning_doc=strategic_reasoning_doc,
        consequence_doc=consequence_doc,
        cpe_doc=cpe_doc,
        strategic_doc=strategic_doc,
        insights_doc=insights_doc,
    )
    _atomic_write_json(attention_doc, results_dir / "attention_allocation.json")

    coordination_doc = compute_system_coordination(
        graph_doc=graph_doc,
        readiness_doc=readiness_doc,
        cert_doc=cert_doc,
        governor_doc=governor_doc,
        strategic_doc=strategic_doc,
        priorities_doc=priorities_doc,
        future_paths_doc=future_paths_doc,
        intelligence_doc=intelligence_doc,
        maturity_doc=maturity_doc,
    )
    _atomic_write_json(coordination_doc, results_dir / "system_coordination.json")

    optimization_doc = compute_institutional_optimization(
        priorities_doc=priorities_doc,
        improvement_doc=improvement_doc,
        maturity_doc=maturity_doc,
        future_paths_doc=future_paths_doc,
        attention_doc=attention_doc,
        coordination_doc=coordination_doc,
        strategic_reasoning_doc=strategic_reasoning_doc,
    )
    _atomic_write_json(optimization_doc, results_dir / "institutional_optimization.json")

    _append_optimization_history(
        results_dir / "institutional_optimization_history.csv",
        {
            "timestamp": optimization_doc["generated_at"],
            "top_optimization": optimization_doc.get("top_optimization"),
            "optimization_score": optimization_doc.get("optimization_score"),
            "expected_system_benefit": optimization_doc.get("expected_system_benefit"),
            "coordination_score": coordination_doc.get("coordination_score"),
            "highest_attention_area": attention_doc.get("highest_attention_area"),
        },
    )

    return {
        "attention_allocation": attention_doc,
        "system_coordination": coordination_doc,
        "institutional_optimization": optimization_doc,
    }
