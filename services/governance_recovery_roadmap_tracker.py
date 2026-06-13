"""
TRITON Phase 200 — Governance Recovery Execution Roadmap & Evidence Tracker.

Converts governance recovery into a measurable institutional roadmap with
evidence tracking. Read-only — does NOT modify governance, execution, trading,
or risk controls.

Outputs (data/results/):
    governance_recovery_roadmap.json
    governance_recovery_milestones.csv
    governance_recovery_evidence_log.csv
    governance_recovery_progress_history.csv

Run:
    python -m services.governance_recovery_roadmap_tracker
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from services.concentration_recovery_simulator import CPS_GREEN, CPS_YELLOW

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "governance_recovery_roadmap.json"
OUT_MILESTONES = RESULTS / "governance_recovery_milestones.csv"
OUT_EVIDENCE = RESULTS / "governance_recovery_evidence_log.csv"
OUT_PROGRESS = RESULTS / "governance_recovery_progress_history.csv"


def _log(msg: str) -> None:
    print(f"[governance_recovery_roadmap_tracker] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _confidence_label(score: float) -> str:
    if score >= 85:
        return "VERY_HIGH"
    if score >= 70:
        return "HIGH"
    if score >= 50:
        return "MODERATE"
    return "LOW"


def _readiness_band(score: float) -> str:
    if score >= 90:
        return "GOVERNANCE_READY"
    if score >= 75:
        return "NEAR_AUTHORIZATION"
    if score >= 50:
        return "RECOVERING"
    return "NOT_READY"


def _load_context() -> Dict[str, Any]:
    return {
        "intel": _read_json(RESULTS / "governance_authorization_intelligence.json"),
        "auth": _read_json(RESULTS / "governance_authorization.json"),
        "cpe": _read_json(RESULTS / "capital_preservation_escalation.json"),
        "cpi": _read_json(RESULTS / "capital_preservation_intelligence.json"),
        "timeline": _read_json(RESULTS / "governance_recovery_timeline.json"),
        "thresholds": _read_json(RESULTS / "governance_recovery_thresholds.json"),
        "basket": _read_json(RESULTS / "diversification_recovery_basket.json"),
        "stability": _read_json(RESULTS / "execution_readiness_stability.json"),
        "recovery_stab": _read_json(RESULTS / "recovery_stabilization_status.json"),
        "readiness": _read_json(RESULTS / "execution_readiness.json"),
        "watchdog": _read_json(RESULTS / "watchdog_status.json"),
    }


def _metrics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    cpe = ctx["cpe"]
    timeline = ctx["timeline"].get("executive_report") or {}
    current = ctx["timeline"].get("current_state") or {}
    intel_cps = ctx["intel"].get("cps_alignment") or {}

    cps = float(
        cpe.get("capital_preservation_score")
        or current.get("current_cps")
        or intel_cps.get("cps_score")
        or 0
    )
    escalation = str(
        cpe.get("escalation_state") or current.get("current_escalation") or "ORANGE"
    ).upper()
    qqq_pct = float(current.get("qqq_concentration_pct") or 100.0)
    target_yellow = current.get("target_yellow_weights") or {"QQQ": 60.0, "HD": 15.0, "XLK": 25.0}
    target_qqq = float(target_yellow.get("QQQ", 60.0))

    dist_yellow = max(0.0, CPS_YELLOW - cps)
    dist_green = max(0.0, CPS_GREEN - cps)
    conc_gap = max(0.0, qqq_pct - target_qqq)

    cps_momentum = cpe.get("risk_trend") or "STABLE"
    momentum_score = {"IMPROVING": 1.0, "STABLE": 0.5, "DETERIORATING": 0.0}.get(
        str(cps_momentum).upper(), 0.5
    )

    return {
        "cps": cps,
        "escalation": escalation,
        "qqq_concentration_pct": qqq_pct,
        "target_qqq_pct": target_qqq,
        "concentration_gap_pct": conc_gap,
        "distance_to_yellow": dist_yellow,
        "distance_to_green": dist_green,
        "cps_momentum": cps_momentum,
        "momentum_score": momentum_score,
        "days_to_yellow": timeline.get("days_to_yellow") or intel_cps.get("days_to_yellow"),
        "days_to_green": timeline.get("days_to_green") or intel_cps.get("days_to_green"),
        "expected_yellow_date": timeline.get("expected_yellow_date")
        or intel_cps.get("expected_yellow_date"),
        "expected_green_date": timeline.get("expected_green_date")
        or intel_cps.get("expected_green_date"),
        "recovery_probability_score": timeline.get("recovery_probability_score") or 75,
        "hd_status": current.get("hd_status"),
        "xlk_status": current.get("xlk_status"),
        "bito_status": current.get("bito_status"),
    }


def _stage(
    name: str,
    *,
    current_status: str,
    completion_pct: float,
    required_evidence: List[str],
    remaining_gap: str,
    eta_days: Optional[int],
    probability: float,
) -> Dict[str, Any]:
    return {
        "stage": name,
        "current_status": current_status,
        "completion_percentage": round(min(100.0, max(0.0, completion_pct)), 1),
        "required_evidence": required_evidence,
        "remaining_gap": remaining_gap,
        "estimated_time_to_completion_days": eta_days,
        "probability_of_success": round(probability, 1),
    }


def _build_roadmap(m: Dict[str, Any], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    cps = m["cps"]
    esc = m["escalation"]
    base_prob = float(m.get("recovery_probability_score") or 75)
    days_y = m.get("days_to_yellow")
    days_g = m.get("days_to_green")

    cps_yellow_pct = min(100.0, (cps / CPS_YELLOW) * 100) if CPS_YELLOW else 0
    cps_green_pct = min(100.0, (cps / CPS_GREEN) * 100) if CPS_GREEN else 0
    conc_reduction_pct = (
        min(
            100.0,
            ((100.0 - m["qqq_concentration_pct"]) / max(100.0 - m["target_qqq_pct"], 1)) * 100,
        )
        if m["qqq_concentration_pct"] > m["target_qqq_pct"]
        else 100.0
    )

    yellow_done = esc in ("YELLOW", "GREEN") or cps >= CPS_YELLOW
    green_done = esc == "GREEN" or cps >= CPS_GREEN
    gov_auth = bool(ctx["auth"].get("governance_authorized"))

    return [
        _stage(
            "Current State (ORANGE)",
            current_status=esc,
            completion_pct=100.0 if esc == "ORANGE" else 100.0,
            required_evidence=["Baseline CPS recorded", "Escalation state documented"],
            remaining_gap=(
                "None — baseline established" if esc == "ORANGE" else f"Already past ORANGE ({esc})"
            ),
            eta_days=0,
            probability=100.0,
        ),
        _stage(
            "Concentration Reduction",
            current_status="IN_PROGRESS" if conc_reduction_pct < 100 else "COMPLETE",
            completion_pct=conc_reduction_pct,
            required_evidence=[
                f"QQQ weight ≤ {m['target_qqq_pct']}%",
                "HD/XLK allocation evidence",
                "Diversification basket execution plan",
            ],
            remaining_gap=f"{m['concentration_gap_pct']:.1f}% QQQ above YELLOW target",
            eta_days=days_y,
            probability=base_prob * 0.95,
        ),
        _stage(
            "CPS Improvement",
            current_status="IN_PROGRESS" if cps < CPS_YELLOW else "COMPLETE",
            completion_pct=cps_yellow_pct,
            required_evidence=[
                f"CPS ≥ {CPS_YELLOW}",
                "Stable or improving risk trend",
                "Concentration drivers cleared",
            ],
            remaining_gap=f"{m['distance_to_yellow']:.0f} CPS points to YELLOW",
            eta_days=days_y,
            probability=base_prob,
        ),
        _stage(
            "YELLOW Eligibility",
            current_status="COMPLETE" if yellow_done else "BLOCKED",
            completion_pct=100.0 if yellow_done else cps_yellow_pct,
            required_evidence=[
                "CPS ≥ 75",
                "Escalation downgrade to YELLOW",
                "Risk review documented",
            ],
            remaining_gap="None" if yellow_done else f"CPS {cps} < {CPS_YELLOW}",
            eta_days=0 if yellow_done else days_y,
            probability=95.0 if yellow_done else base_prob,
        ),
        _stage(
            "Governance Reassessment",
            current_status="COMPLETE" if gov_auth else "BLOCKED",
            completion_pct=100.0 if gov_auth else max(0.0, cps_yellow_pct - 10),
            required_evidence=[
                "Governance authorization intelligence report",
                "Capital preservation escalation review",
                "Operator risk committee sign-off",
            ],
            remaining_gap="None" if gov_auth else "Governance gate awaiting YELLOW evidence",
            eta_days=0 if gov_auth else days_y,
            probability=90.0 if gov_auth else base_prob * 0.85,
        ),
        _stage(
            "GREEN Eligibility",
            current_status="COMPLETE" if green_done else "BLOCKED",
            completion_pct=100.0 if green_done else cps_green_pct,
            required_evidence=[
                f"CPS ≥ {CPS_GREEN}",
                "BITO lifecycle promotion",
                "Full diversification evidence",
            ],
            remaining_gap="None" if green_done else f"CPS {cps} < {CPS_GREEN}",
            eta_days=0 if green_done else days_g,
            probability=85.0 if green_done else base_prob * 0.7,
        ),
        _stage(
            "Full Governance Restoration",
            current_status="COMPLETE" if gov_auth and green_done else "BLOCKED",
            completion_pct=100.0 if (gov_auth and green_done) else cps_green_pct * 0.9,
            required_evidence=[
                "GREEN escalation",
                "governance_authorized=true",
                "Overall authorization=true",
            ],
            remaining_gap=(
                "None" if (gov_auth and green_done) else "Awaiting GREEN eligibility + reassessment"
            ),
            eta_days=0 if (gov_auth and green_done) else days_g,
            probability=100.0 if (gov_auth and green_done) else base_prob * 0.65,
        ),
    ]


def _milestone_status(completion: float, blocked: bool) -> str:
    if completion >= 100:
        return "COMPLETE"
    if blocked:
        return "BLOCKED"
    if completion >= 60:
        return "IN_PROGRESS"
    if completion >= 30:
        return "AT_RISK"
    if completion <= 0:
        return "FAILED"
    return "IN_PROGRESS"


def _build_milestones(
    m: Dict[str, Any], ctx: Dict[str, Any], roadmap: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    intel_reqs = (ctx["intel"].get("authorization_status") or {}).get("required_evidence") or []
    timeline = ctx["timeline"].get("executive_report") or {}
    milestones: List[Dict[str, Any]] = []

    stage_map = {s["stage"]: s for s in roadmap}

    defs = [
        (
            "M1",
            "Document ORANGE baseline",
            "Current State (ORANGE)",
            "Operator",
            None,
            "Establishes recovery starting point",
        ),
        (
            "M2",
            "Reduce QQQ concentration to YELLOW target",
            "Concentration Reduction",
            "Portfolio / Allocation",
            "M1",
            "Primary concentration driver clearance",
        ),
        (
            "M3",
            f"Achieve CPS ≥ {CPS_YELLOW}",
            "CPS Improvement",
            "Capital Preservation",
            "M2",
            "Enables YELLOW escalation eligibility",
        ),
        (
            "M4",
            "YELLOW escalation eligibility confirmed",
            "YELLOW Eligibility",
            "Risk Committee",
            "M3",
            "Partial governance authorization path",
        ),
        (
            "M5",
            "Governance reassessment completed",
            "Governance Reassessment",
            "Governance Board",
            "M4",
            "Institutional authorization review",
        ),
        (
            "M6",
            f"Achieve CPS ≥ {CPS_GREEN}",
            "GREEN Eligibility",
            "Portfolio / Lifecycle",
            "M4",
            "Full restoration threshold",
        ),
        (
            "M7",
            "Full governance authorization restored",
            "Full Governance Restoration",
            "Investment Committee",
            "M5,M6",
            "End-to-end authorization",
        ),
    ]

    for mid, title, stage_name, owner, dep, impact in defs:
        stage = stage_map.get(stage_name, {})
        comp = stage.get("completion_percentage", 0)
        blocked = stage.get("current_status") == "BLOCKED" and comp < 100
        status = _milestone_status(comp, blocked)
        milestones.append(
            {
                "milestone_id": mid,
                "title": title,
                "roadmap_stage": stage_name,
                "status": status,
                "priority": "CRITICAL" if mid in ("M2", "M3", "M4") else "HIGH",
                "owner": owner,
                "dependency": dep,
                "completion_percentage": comp,
                "expected_completion_date": (
                    m.get("expected_yellow_date")
                    if mid in ("M2", "M3", "M4", "M5")
                    else m.get("expected_green_date")
                ),
                "impact_on_authorization": impact,
                "estimated_days_remaining": stage.get("estimated_time_to_completion_days"),
            }
        )

    for i, req in enumerate(intel_reqs):
        milestones.append(
            {
                "milestone_id": f"EV{i + 1}",
                "title": req.get("requirement"),
                "roadmap_stage": "Evidence",
                "status": "BLOCKED" if m["escalation"] == "ORANGE" else "IN_PROGRESS",
                "priority": req.get("priority"),
                "owner": req.get("owner"),
                "dependency": "M2",
                "completion_percentage": 0.0,
                "expected_completion_date": (
                    m.get("expected_yellow_date")
                    if req.get("estimated_resolution_days", 99)
                    and req.get("estimated_resolution_days") <= 30
                    else m.get("expected_green_date")
                ),
                "impact_on_authorization": req.get("impact_on_authorization"),
                "estimated_days_remaining": req.get("estimated_resolution_days"),
            }
        )

    return milestones


def _build_evidence_log(m: Dict[str, Any], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    ts = _now_iso()
    stability = ctx["stability"].get("stability_score") or {}
    recovery = ctx["recovery_stab"].get("summary") or {}
    readiness = ctx["readiness"].get("checks") or {}
    basket = ctx["basket"].get("executive_report") or {}

    entries: List[Dict[str, Any]] = []

    def _ev(category: str, item: str, collected: bool, value: Any, source: str, gap: str) -> None:
        entries.append(
            {
                "timestamp": ts,
                "category": category,
                "evidence_item": item,
                "status": "COLLECTED" if collected else "PENDING",
                "current_value": value,
                "source_artifact": source,
                "remaining_gap": gap,
            }
        )

    _ev(
        "Capital Preservation",
        "CPS trajectory",
        m["cps"] > 0,
        m["cps"],
        "capital_preservation_escalation.json",
        f"{m['distance_to_yellow']:.0f} points to YELLOW",
    )
    _ev(
        "Capital Preservation",
        "Distance to YELLOW",
        m["cps"] >= CPS_YELLOW,
        m["distance_to_yellow"],
        "governance_recovery_timeline.json",
        "Achieve CPS ≥ 75",
    )
    _ev(
        "Capital Preservation",
        "Distance to GREEN",
        m["cps"] >= CPS_GREEN,
        m["distance_to_green"],
        "governance_recovery_thresholds.json",
        "Achieve CPS ≥ 90",
    )
    _ev(
        "Capital Preservation",
        "CPS momentum",
        m["momentum_score"] >= 0.5,
        m["cps_momentum"],
        "capital_preservation_escalation.json",
        "Risk trend must be STABLE or IMPROVING",
    )
    _ev(
        "Capital Preservation",
        "Risk trend",
        str(m["cps_momentum"]).upper() != "DETERIORATING",
        m["cps_momentum"],
        "capital_preservation_escalation.json",
        "Avoid deteriorating trend",
    )

    _ev(
        "Portfolio Recovery",
        "Concentration reduction",
        m["qqq_concentration_pct"] <= m["target_qqq_pct"],
        f"{m['qqq_concentration_pct']:.1f}% QQQ",
        "governance_recovery_timeline.json",
        f"Reduce QQQ by {m['concentration_gap_pct']:.1f}%",
    )
    _ev(
        "Portfolio Recovery",
        "Diversification progress",
        m["qqq_concentration_pct"] < 100,
        basket.get("best_recovery_basket"),
        "diversification_recovery_basket.json",
        "Execute HD/XLK additions per recovery basket",
    )
    _ev(
        "Portfolio Recovery",
        "Recovery candidate availability",
        bool(m.get("hd_status") == "OPEN_NEW" and m.get("xlk_status") == "OPEN_NEW"),
        f"HD={m.get('hd_status')} XLK={m.get('xlk_status')} BITO={m.get('bito_status')}",
        "governance_recovery_timeline.json",
        "BITO promotion for GREEN path",
    )
    _ev(
        "Portfolio Recovery",
        "Lifecycle eligibility",
        m.get("bito_status") == "OPEN_NEW",
        m.get("bito_status"),
        "diversification_recovery_basket.json",
        "Promote BITO from WATCH to OPEN_NEW",
    )
    _ev(
        "Portfolio Recovery",
        "Allocation readiness",
        readiness.get("lifecycle") and readiness.get("signals"),
        readiness.get("lifecycle"),
        "execution_readiness.json",
        "Maintain lifecycle and signal freshness",
    )

    wd_score = (ctx["stability"].get("watchdog_reliability") or {}).get("watchdog_stability_score")
    _ev(
        "Operational Stability",
        "Broker reliability",
        readiness.get("broker"),
        ctx["watchdog"].get("broker_connected"),
        "watchdog_status.json",
        "Maintain broker connectivity",
    )
    _ev(
        "Operational Stability",
        "Watchdog reliability",
        (wd_score or 0) >= 50,
        wd_score,
        "execution_readiness_stability.json",
        "Reduce false-positive watchdog failures",
    )
    _ev(
        "Operational Stability",
        "Loop self-healing health",
        recovery.get("loop_stable") or recovery.get("current_state") != "FAILED",
        recovery.get("current_state"),
        "recovery_stabilization_status.json",
        "Stabilize continuous loop recovery certification",
    )
    _ev(
        "Operational Stability",
        "Execution readiness stability",
        (stability.get("stability_score") or 0) >= 50,
        stability.get("stability_score"),
        "execution_readiness_stability.json",
        "Operational readiness without governance conflation",
    )

    _ev(
        "Governance Requirements",
        "Risk review status",
        False,
        "Pending",
        "governance_recovery_requirements.csv",
        "Complete concentration and stale order reviews",
    )
    _ev(
        "Governance Requirements",
        "Authorization history",
        True,
        "4100+ readiness samples",
        "execution_readiness_history.csv",
        "Persistent ORANGE block documented",
    )
    _ev(
        "Governance Requirements",
        "Previous blocking decisions",
        True,
        "EXPECTED_PROTECTIVE_BLOCK",
        "governance_authorization_intelligence.json",
        "Intentional block — not misconfiguration",
    )

    collected_count = sum(1 for e in entries if e.get("status") == "COLLECTED")
    entries.append(
        {
            "timestamp": ts,
            "category": "Governance Requirements",
            "evidence_item": "Evidence completeness",
            "status": "COLLECTED" if collected_count >= len(entries) * 0.5 else "PENDING",
            "current_value": f"{collected_count}/{len(entries)}",
            "source_artifact": "governance_recovery_roadmap.json",
            "remaining_gap": "Collect pending portfolio and review evidence",
        }
    )

    return entries


def _probability_forecast(m: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    base = float(m.get("recovery_probability_score") or 75)
    timeline = ctx["timeline"].get("executive_report") or {}
    conf = timeline.get("forecast_confidence") or "HIGH"

    yellow_prob = min(95.0, base + m["momentum_score"] * 10)
    green_prob = min(90.0, base * 0.85)
    restore_prob = min(85.0, base * 0.8) if m["escalation"] == "ORANGE" else 95.0
    delay_prob = max(5.0, 100.0 - yellow_prob)
    failure_prob = max(5.0, 100.0 - green_prob * 0.9)

    return {
        "probability_yellow_pct": round(yellow_prob, 1),
        "probability_green_pct": round(green_prob, 1),
        "probability_governance_restoration_pct": round(restore_prob, 1),
        "probability_delayed_recovery_pct": round(delay_prob, 1),
        "probability_recovery_failure_pct": round(failure_prob, 1),
        "confidence_level": conf if isinstance(conf, str) else _confidence_label(base),
        "expected_authorization_yellow_date": m.get("expected_yellow_date"),
        "expected_authorization_green_date": m.get("expected_green_date"),
    }


def _readiness_score(
    m: Dict[str, Any], evidence: List[Dict[str, Any]], prob: Dict[str, Any]
) -> Dict[str, Any]:
    cps_progress = min(100.0, (m["cps"] / CPS_GREEN) * 100)
    collected = sum(1 for e in evidence if e.get("status") == "COLLECTED")
    evidence_pct = (collected / max(len(evidence), 1)) * 100
    div_pct = max(
        0.0,
        min(
            100.0,
            (100 - m["qqq_concentration_pct"]) / max(100 - m["target_qqq_pct"], 1) * 100,
        ),
    )
    ops = prob.get("probability_yellow_pct", 50) * 0.4 + 60
    timeline_conf = prob.get("probability_green_pct", 50)

    score = round(
        cps_progress * 0.30
        + div_pct * 0.25
        + ops * 0.15
        + evidence_pct * 0.15
        + timeline_conf * 0.15,
        1,
    )
    return {
        "governance_readiness_score": score,
        "classification": _readiness_band(score),
        "factors": {
            "cps_progress": round(cps_progress, 1),
            "diversification_evidence": round(div_pct, 1),
            "operational_reliability": round(ops, 1),
            "governance_evidence_completion": round(evidence_pct, 1),
            "recovery_timeline_confidence": round(timeline_conf, 1),
        },
    }


def _executive_report(
    m: Dict[str, Any],
    roadmap: List[Dict[str, Any]],
    milestones: List[Dict[str, Any]],
    prob: Dict[str, Any],
    readiness: Dict[str, Any],
    ctx: Dict[str, Any],
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    timeline = ctx["timeline"].get("executive_report") or {}
    completed = [ms for ms in milestones if ms.get("status") == "COMPLETE"]
    blocked = [ms for ms in milestones if ms.get("status") == "BLOCKED"]
    remaining = [ms for ms in milestones if ms.get("status") not in ("COMPLETE",)]

    primary_block = blocked[0] if blocked else None

    return {
        "report_type": "GOVERNANCE_RECOVERY_ROADMAP_REPORT",
        "current_governance_state": m["escalation"],
        "current_cps": m["cps"],
        "distance_to_yellow": m["distance_to_yellow"],
        "distance_to_green": m["distance_to_green"],
        "completed_milestones_count": len(completed),
        "remaining_requirements": [r.get("title") for r in remaining[:6]],
        "recovery_probability_yellow": prob.get("probability_yellow_pct"),
        "recovery_probability_green": prob.get("probability_green_pct"),
        "estimated_authorization_yellow_date": m.get("expected_yellow_date"),
        "estimated_authorization_green_date": m.get("expected_green_date"),
        "top_risks": timeline.get("major_risks") or [],
        "top_accelerators": timeline.get("major_accelerators") or [],
        "recommended_next_actions": [
            "Execute balanced concentration reduction toward YELLOW weights (QQQ 60 / HD 15 / XLK 25)",
            "Complete operator risk reviews (concentration + stale orders)",
            "Advance BITO lifecycle promotion for GREEN path",
            "Maintain operational stability scores while accumulating governance evidence",
        ],
        "primary_blocking_milestone": primary_block.get("title") if primary_block else None,
        "governance_readiness_classification": readiness.get("classification"),
        "executive_questions": {
            "how_close_to_yellow": f"{m['distance_to_yellow']:.0f} CPS points ({m['cps']}/{CPS_YELLOW})",
            "how_close_to_green": f"{m['distance_to_green']:.0f} CPS points ({m['cps']}/{CPS_GREEN})",
            "missing_evidence": [e["evidence_item"] for e in evidence if e["status"] == "PENDING"][
                :5
            ],
            "blocking_milestone": (
                primary_block.get("title")
                if primary_block
                else "Concentration reduction / CPS improvement"
            ),
            "recovery_probability": prob.get("probability_yellow_pct"),
            "restoration_timeline": timeline.get("projected_governance_recovery"),
            "highest_impact_actions": [
                "Reduce QQQ concentration",
                f"Achieve CPS {CPS_YELLOW}+",
                "Document risk committee reviews",
            ],
        },
    }


def _append_progress(row: Dict[str, Any]) -> None:
    if OUT_PROGRESS.is_file():
        df = pd.read_csv(OUT_PROGRESS)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(OUT_PROGRESS, index=False, encoding="utf-8")


def run_analysis() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    ctx = _load_context()
    m = _metrics(ctx)
    roadmap = _build_roadmap(m, ctx)
    milestones = _build_milestones(m, ctx, roadmap)
    evidence = _build_evidence_log(m, ctx)
    prob = _probability_forecast(m, ctx)
    readiness = _readiness_score(m, evidence, prob)
    executive = _executive_report(m, roadmap, milestones, prob, readiness, ctx, evidence)

    overall_completion = round(
        sum(s["completion_percentage"] for s in roadmap) / max(len(roadmap), 1), 1
    )

    report = {
        "generated_at": ts,
        "phase": "200",
        "mode": "governance_recovery_roadmap_tracker",
        "disclaimer": (
            "Governance recovery roadmap intelligence only — does NOT modify governance rules, "
            "execution permissions, trading, allocation, or risk controls."
        ),
        "recovery_roadmap": roadmap,
        "recovery_progress_pct": overall_completion,
        "metrics": m,
        "milestones": milestones,
        "evidence_summary": {
            "collected": sum(1 for e in evidence if e["status"] == "COLLECTED"),
            "pending": sum(1 for e in evidence if e["status"] == "PENDING"),
            "total": len(evidence),
        },
        "authorization_probability": prob,
        "governance_readiness_score": readiness,
        "executive_recommendation": executive,
        "summary": {
            "escalation_state": m["escalation"],
            "cps": m["cps"],
            "distance_to_yellow": m["distance_to_yellow"],
            "distance_to_green": m["distance_to_green"],
            "recovery_progress_pct": overall_completion,
            "readiness_classification": readiness.get("classification"),
            "probability_yellow_pct": prob.get("probability_yellow_pct"),
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(milestones).to_csv(OUT_MILESTONES, index=False, encoding="utf-8")
    pd.DataFrame(evidence).to_csv(OUT_EVIDENCE, index=False, encoding="utf-8")

    _append_progress(
        {
            "timestamp": ts,
            "cps": m["cps"],
            "escalation_state": m["escalation"],
            "qqq_concentration_pct": m["qqq_concentration_pct"],
            "distance_to_yellow": m["distance_to_yellow"],
            "distance_to_green": m["distance_to_green"],
            "recovery_progress_pct": overall_completion,
            "readiness_score": readiness.get("governance_readiness_score"),
            "probability_yellow_pct": prob.get("probability_yellow_pct"),
        }
    )

    _log(
        f"cps={m['cps']} progress={overall_completion}% readiness={readiness.get('classification')} "
        f"p_yellow={prob.get('probability_yellow_pct')}"
    )
    return report


def main() -> int:
    run_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
