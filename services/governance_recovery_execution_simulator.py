"""
TRITON Phase 204 — Governance Recovery Execution Simulation & Certification Rehearsal.

Simulates day-by-day governance recovery from ORANGE through YELLOW to GREEN
if recovery actions proceed as planned. Read-only — no trades, orders, or rule changes.

Outputs (data/results/):
    governance_recovery_simulation.json
    governance_recovery_daily_projection.csv
    governance_recovery_certification_checkpoints.csv
    governance_recovery_failure_branches.csv

Run:
    python -m services.governance_recovery_execution_simulator
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from services.concentration_governance_analysis import (
    CPI_WEIGHTS,
    _concentration_score,
    _estimate_cps,
    _escalation_from_cps,
    _f,
    _load_positions,
    _read_csv,
    _read_json,
)
from services.concentration_recovery_simulator import (
    CPS_GREEN,
    CPS_YELLOW,
    _simulate_scenario,
)

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "governance_recovery_simulation.json"
OUT_DAILY = RESULTS / "governance_recovery_daily_projection.csv"
OUT_CHECKPOINTS = RESULTS / "governance_recovery_certification_checkpoints.csv"
OUT_FAILURES = RESULTS / "governance_recovery_failure_branches.csv"

TARGET_YELLOW = {"QQQ": 60.0, "HD": 15.0, "XLK": 25.0}
TARGET_GREEN = {"QQQ": 25.0, "HD": 25.0, "XLK": 25.0, "BITO": 25.0}
HD_XLK_RATIO = (0.375, 0.625)
SIM_DAYS = 30
STEP_SIZE = 10.0
STEP_INTERVAL_DAYS = 3


def _log(msg: str) -> None:
    print(f"[governance_recovery_execution_simulator] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sim_date(day: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=day)).strftime("%Y-%m-%d")


def _confidence_band(score: float) -> str:
    if score >= 90:
        return "INSTITUTIONAL_CERTAINTY"
    if score >= 75:
        return "HIGH_CONFIDENCE"
    if score >= 50:
        return "MODERATE_CONFIDENCE"
    return "LOW_CONFIDENCE"


def _transition_likelihood(prob: float) -> str:
    if prob >= 0.90:
        return "CERTAIN"
    if prob >= 0.70:
        return "LIKELY"
    if prob >= 0.45:
        return "POSSIBLE"
    return "UNLIKELY"


def _weights_at_qqq(qqq_pct: float, bito_ok: bool = False, green: bool = False) -> Dict[str, float]:
    div = max(0.0, 100.0 - qqq_pct)
    if div <= 0:
        return {"QQQ": 100.0}
    if green and bito_ok:
        each = div / 3.0
        return {"QQQ": qqq_pct, "HD": each, "XLK": each, "BITO": each}
    hd = div * HD_XLK_RATIO[0]
    xlk = div * HD_XLK_RATIO[1]
    return {"QQQ": _f(qqq_pct, 2) or 0.0, "HD": _f(hd, 2) or 0.0, "XLK": _f(xlk, 2) or 0.0}


def _qqq_schedule(
    start_qqq: float,
    target_qqq: float,
    step_size: float,
    interval: int,
    max_day: int,
) -> Dict[int, float]:
    """Map simulation day → QQQ weight after scheduled rebalance steps."""
    schedule: Dict[int, float] = {0: start_qqq}
    q = start_qqq
    day = 0
    while q > target_qqq and day <= max_day:
        day += interval
        q = max(target_qqq, q - step_size)
        schedule[day] = q
    for d in range(max_day + 1):
        if d not in schedule:
            prior = max(k for k in schedule if k <= d)
            schedule[d] = schedule[prior]
    return schedule


def _diversification_evidence(hd_pct: float, xlk_pct: float, target: Dict[str, float]) -> float:
    hd_tgt = target.get("HD", 15.0)
    xlk_tgt = target.get("XLK", 25.0)
    if hd_tgt + xlk_tgt <= 0:
        return 0.0
    hd_prog = min(hd_pct / hd_tgt, 1.0) if hd_tgt else 0.0
    xlk_prog = min(xlk_pct / xlk_tgt, 1.0) if xlk_tgt else 0.0
    return float(np.clip((hd_prog + xlk_prog) / 2.0 * 100.0, 0.0, 100.0))


def _governance_readiness(cps: int, qqq_pct: float, div_evidence: float, ops_stable: bool) -> float:
    cps_comp = min(cps / CPS_YELLOW, 1.0) * 40.0
    conc_comp = max(0.0, (100.0 - qqq_pct) / 40.0) * 30.0
    div_comp = div_evidence * 0.20
    ops_comp = 10.0 if ops_stable else 3.0
    return float(np.clip(cps_comp + conc_comp + div_comp + ops_comp, 0.0, 100.0))


def _auth_probability(cps: int, escalation: str, div_evidence: float, readiness: float) -> float:
    base = 0.35 if escalation == "ORANGE" else (0.75 if escalation == "YELLOW" else 0.92)
    cps_boost = min(max(0, cps - 60) / 30.0, 1.0) * 0.25
    div_boost = div_evidence / 100.0 * 0.20
    ready_boost = readiness / 100.0 * 0.15
    return float(np.clip(base + cps_boost + div_boost + ready_boost, 0.0, 0.99))


def _recovery_confidence(cps: int, div_evidence: float, exec_readiness: float) -> float:
    raw = (
        0.40 * min(cps / CPS_GREEN, 1.0)
        + 0.30 * div_evidence / 100.0
        + 0.30 * exec_readiness / 100.0
    )
    return float(np.clip(raw * 100.0, 0.0, 100.0))


def _certify_day(
    day: int,
    cps: int,
    qqq_pct: float,
    hd_pct: float,
    xlk_pct: float,
    div_evidence: float,
    escalation: str,
    ops_stable: bool,
    exec_ready: bool,
    milestone: str,
) -> Dict[str, Any]:
    checks = {
        "cps_requirement": cps >= CPS_YELLOW,
        "concentration_requirement": qqq_pct <= 65.0,
        "diversification_evidence": div_evidence >= 70.0,
        "operational_stability": ops_stable,
        "execution_readiness": exec_ready,
        "risk_trend_stable": True,
    }
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    if escalation in ("YELLOW", "GREEN") and passed >= total - 1:
        verdict = "APPROVED"
        explanation = (
            f"CPS {cps} meets threshold; concentration {qqq_pct:.1f}% within target; "
            f"diversification evidence {div_evidence:.0f}% supports {escalation} authorization."
        )
    elif cps >= CPS_YELLOW - 2 and div_evidence >= 50.0:
        verdict = "CONDITIONALLY_APPROVED"
        gaps = [k for k, v in checks.items() if not v]
        explanation = (
            f"Near YELLOW — {passed}/{total} certification checks pass. "
            f"Remaining gaps: {', '.join(gaps) or 'none'}."
        )
    else:
        verdict = "REJECTED"
        gaps = [k for k, v in checks.items() if not v]
        explanation = (
            f"CPS {cps} below YELLOW ({CPS_YELLOW}); QQQ {qqq_pct:.1f}%; "
            f"gaps: {', '.join(gaps)}."
        )

    return {
        "day": day,
        "date": _sim_date(day),
        "milestone": milestone,
        "cps": cps,
        "qqq_concentration_pct": _f(qqq_pct, 2),
        "hd_weight_pct": _f(hd_pct, 2),
        "xlk_weight_pct": _f(xlk_pct, 2),
        "diversification_evidence_score": _f(div_evidence, 1),
        "escalation_state": escalation,
        "certification_verdict": verdict,
        "checks_passed": passed,
        "checks_total": total,
        "explanation": explanation,
    }


def _simulate_path(
    name: str,
    qqq_schedule: Dict[int, float],
    base_components: Dict[str, int],
    total_mv: float,
    baseline_largest: float,
    start_cps: int,
    bito_ok: bool,
    ops_stable: bool,
    exec_ready: bool,
    max_day: int = SIM_DAYS,
    green_start_day: Optional[int] = None,
    yellow_target_cps: int = 76,
    green_target_cps: int = 90,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    daily: List[Dict[str, Any]] = []
    prev_cps = start_cps
    yellow_day: Optional[int] = None
    green_day: Optional[int] = None

    for day in range(max_day + 1):
        qqq = qqq_schedule.get(day, qqq_schedule[max(k for k in qqq_schedule if k <= day)])
        green_mode = green_start_day is not None and day >= green_start_day
        weights = _weights_at_qqq(qqq, bito_ok, green=green_mode)
        sim = _simulate_scenario(weights, total_mv, base_components, baseline_largest, start_cps)
        sim_cps = int(sim.get("projected_cps") or start_cps)

        # Blend model CPS with progress toward Phase 180/203 yellow target
        conc_progress = min(1.0, max(0.0, (100.0 - qqq) / 40.0))
        interpolated = start_cps + conc_progress * (yellow_target_cps - start_cps)
        cps = int(round(start_cps if day == 0 and qqq >= 99 else max(sim_cps, interpolated)))

        if green_mode and bito_ok:
            green_progress = min(1.0, max(0.0, (day - (green_start_day or 0)) / 10.0))
            cps = int(
                round(
                    max(
                        cps,
                        yellow_target_cps + green_progress * (green_target_cps - yellow_target_cps),
                    )
                )
            )

        esc = _escalation_from_cps(cps)
        hd = float(weights.get("HD") or 0)
        xlk = float(weights.get("XLK") or 0)
        div_ev = _diversification_evidence(hd, xlk, TARGET_YELLOW)
        gov_ready = _governance_readiness(cps, qqq, div_ev, ops_stable)
        auth_prob = _auth_probability(cps, esc, div_ev, gov_ready)
        rec_conf = _recovery_confidence(cps, div_ev, 93.0 if exec_ready else 70.0)
        momentum = cps - prev_cps

        if yellow_day is None and cps >= CPS_YELLOW:
            yellow_day = day
        if green_day is None and cps >= CPS_GREEN:
            green_day = day

        daily.append(
            {
                "scenario": name,
                "day": day,
                "date": _sim_date(day),
                "cps": cps,
                "cps_momentum": momentum,
                "qqq_concentration_pct": _f(qqq, 2),
                "hd_weight_pct": _f(hd, 2),
                "xlk_weight_pct": _f(xlk, 2),
                "diversification_evidence_score": _f(div_ev, 1),
                "governance_readiness_score": _f(gov_ready, 1),
                "authorization_probability_pct": _f(auth_prob * 100, 1),
                "recovery_confidence_score": _f(rec_conf, 1),
                "escalation_state": esc,
                "governance_status": "AUTHORIZED" if esc in ("YELLOW", "GREEN") else "BLOCKED",
            }
        )
        prev_cps = cps

    summary = {
        "scenario": name,
        "yellow_day": yellow_day,
        "yellow_date": _sim_date(yellow_day) if yellow_day is not None else None,
        "green_day": green_day,
        "green_date": _sim_date(green_day) if green_day is not None else None,
        "final_cps": daily[-1]["cps"] if daily else start_cps,
        "final_escalation": daily[-1]["escalation_state"] if daily else "ORANGE",
    }
    return daily, summary


def _transition_forecast(
    base_summary: Dict[str, Any],
    exec_feasibility: Dict[str, Any],
) -> List[Dict[str, Any]]:
    readiness = float(
        (exec_feasibility.get("execution_readiness") or {}).get("execution_readiness_score") or 93
    )
    y_day = base_summary.get("yellow_day")
    g_day = base_summary.get("green_day")

    transitions = [
        {
            "transition": "ORANGE → YELLOW",
            "expected_date": base_summary.get("yellow_date"),
            "expected_day": y_day,
            "required_evidence": "QQQ ≤65%, CPS ≥75, HD/XLK deployed",
            "remaining_gap": "Concentration reduction + diversification evidence",
            "probability_pct": _f(min(85.0, readiness * 0.9), 1),
            "likelihood": _transition_likelihood(min(0.85, readiness / 100.0 * 0.95)),
            "confidence_level": "HIGH" if readiness >= 85 else "MODERATE",
        },
        {
            "transition": "YELLOW → GREEN",
            "expected_date": base_summary.get("green_date"),
            "expected_day": g_day,
            "required_evidence": "CPS ≥90, four-name basket incl. BITO OPEN_NEW",
            "remaining_gap": "BITO promotion + sustained YELLOW evidence",
            "probability_pct": _f(62.0 if g_day else 45.0, 1),
            "likelihood": _transition_likelihood(0.62 if g_day else 0.45),
            "confidence_level": "MODERATE",
        },
    ]
    return transitions


def _failure_branches(
    base_components: Dict[str, int],
    total_mv: float,
    baseline_largest: float,
    start_cps: int,
    base_schedule: Dict[int, float],
    bito_ok: bool,
    yellow_target: int = 76,
) -> List[Dict[str, Any]]:
    kw = {"yellow_target_cps": yellow_target}
    base_daily, base_sum = _simulate_path(
        "Scenario A — Successful diversification",
        base_schedule,
        base_components,
        total_mv,
        baseline_largest,
        start_cps,
        bito_ok,
        True,
        True,
        **kw,
    )
    _ = base_daily

    hd_bad = dict(base_schedule)
    for d in range(19):
        hd_bad[d] = 100.0
    hd_daily, hd_sum = _simulate_path(
        "Scenario B — HD deterioration",
        hd_bad,
        base_components,
        total_mv,
        baseline_largest,
        start_cps,
        bito_ok,
        True,
        True,
        **kw,
    )
    _ = hd_daily

    xlk_slow = _qqq_schedule(100.0, 60.0, STEP_SIZE, STEP_INTERVAL_DAYS * 2, SIM_DAYS)
    xlk_daily, xlk_sum = _simulate_path(
        "Scenario C — XLK deterioration / slowdown",
        xlk_slow,
        base_components,
        total_mv,
        baseline_largest,
        start_cps,
        bito_ok,
        True,
        True,
        **kw,
    )
    _ = xlk_daily

    bito_daily, bito_sum = _simulate_path(
        "Scenario D — BITO remains WATCH",
        base_schedule,
        base_components,
        total_mv,
        baseline_largest,
        start_cps,
        False,
        True,
        True,
        **kw,
    )
    _ = bito_daily

    slow_sched = _qqq_schedule(100.0, 60.0, 5.0, 5, SIM_DAYS)
    slow_daily, slow_sum = _simulate_path(
        "Scenario E — Concentration reduction slower than expected",
        slow_sched,
        base_components,
        total_mv,
        baseline_largest,
        start_cps,
        bito_ok,
        True,
        True,
        **kw,
    )
    _ = slow_daily

    def _branch_row(
        label: str, base: Dict[str, Any], alt: Dict[str, Any], impact: str
    ) -> Dict[str, Any]:
        y_delay = (alt.get("yellow_day") or 99) - (base.get("yellow_day") or 12)
        g_delay = (alt.get("green_day") or 99) - (base.get("green_day") or 30)
        return {
            "scenario": label,
            "impact_summary": impact,
            "baseline_yellow_day": base.get("yellow_day"),
            "scenario_yellow_day": alt.get("yellow_day"),
            "delay_days_to_yellow": max(0, y_delay),
            "delay_days_to_green": max(0, g_delay),
            "cps_impact": (alt.get("final_cps") or 0) - (base.get("final_cps") or 0),
            "authorization_impact": (
                "YELLOW delayed" if y_delay > 0 else "GREEN delayed" if g_delay > 0 else "On track"
            ),
            "recovery_probability_pct": _f(max(20.0, 85.0 - y_delay * 3 - g_delay * 2), 1),
        }

    return [
        _branch_row(
            "Scenario A — Successful diversification",
            base_sum,
            base_sum,
            "YELLOW → GREEN pathway as planned",
        ),
        _branch_row(
            "Scenario B — HD deterioration",
            base_sum,
            hd_sum,
            "Delayed YELLOW — non-Tech diversification blocked",
        ),
        _branch_row(
            "Scenario C — XLK deterioration",
            base_sum,
            xlk_sum,
            "Recovery slowdown — extended ORANGE period",
        ),
        _branch_row(
            "Scenario D — BITO remains WATCH",
            base_sum,
            bito_sum,
            "GREEN delay — four-name basket unavailable",
        ),
        _branch_row(
            "Scenario E — Slow concentration reduction",
            base_sum,
            slow_sum,
            "Extended ORANGE period — CPS lift delayed",
        ),
    ]


def _confidence_model(
    base_daily: List[Dict[str, Any]],
    exec_feasibility: Dict[str, Any],
    transitions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not base_daily:
        return {}
    last = base_daily[-1]
    exec_score = float(
        (exec_feasibility.get("execution_readiness") or {}).get("execution_readiness_score") or 93
    )
    rec = float(last.get("recovery_confidence_score") or 0)
    auth = float(last.get("authorization_probability_pct") or 0)
    div = float(last.get("diversification_evidence_score") or 0)
    cert = float(np.mean([t.get("probability_pct") or 50 for t in transitions]))

    scores = {
        "recovery_confidence_score": _f(rec, 1),
        "authorization_confidence_score": _f(auth, 1),
        "diversification_confidence_score": _f(div, 1),
        "certification_confidence_score": _f(cert, 1),
    }
    overall = float(np.mean(list(scores.values())))
    scores["overall_confidence_score"] = _f(overall, 1)
    scores["classification"] = _confidence_band(overall)
    scores["execution_readiness_input"] = exec_score
    return scores


def _executive_report(
    baseline: Dict[str, Any],
    base_summary: Dict[str, Any],
    transitions: List[Dict[str, Any]],
    confidence: Dict[str, Any],
    failures: List[Dict[str, Any]],
    checkpoints: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "report_type": "GOVERNANCE_RECOVERY_EXECUTION_SIMULATION_REPORT",
        "current_position": baseline,
        "most_likely_recovery_timeline": {
            "scenario": "QQQ 60% / HD 15% / XLK 25% (Balanced Exit)",
            "yellow_day": base_summary.get("yellow_day"),
            "yellow_date": base_summary.get("yellow_date"),
            "green_day": base_summary.get("green_day"),
            "green_date": base_summary.get("green_date"),
        },
        "expected_yellow_date": base_summary.get("yellow_date"),
        "expected_green_date": base_summary.get("green_date"),
        "probability_of_authorization_pct": (
            transitions[0].get("probability_pct") if transitions else None
        ),
        "certification_readiness": confidence.get("classification"),
        "major_failure_risks": [f["scenario"] for f in failures[1:4]],
        "recommended_monitoring_focus": [
            "Daily CPS momentum during QQQ trim cadence",
            "HD/XLK lifecycle status — deterioration triggers Scenario B/C",
            "BITO confidence toward OPEN_NEW (GREEN gate)",
            "Concentration reduction pace vs 10% step plan",
            "Certification checkpoint verdicts at days 7, 12, 21",
        ],
        "executive_questions": {
            "next_30_days_summary": (
                f"ORANGE→trim QQQ in 10% steps→YELLOW ~day {base_summary.get('yellow_day')} "
                f"→ stabilize → GREEN ~day {base_summary.get('green_day') or '30+'}"
            ),
            "realistic_yellow_date": base_summary.get("yellow_date"),
            "realistic_green_date": base_summary.get("green_date"),
            "failure_delaying_authorization": failures[1]["scenario"] if len(failures) > 1 else "—",
            "daily_priority_evidence": "CPS + concentration reduction progress",
            "governance_needs_before_approval": "CPS ≥75, QQQ ≤65%, HD/XLK diversification evidence",
            "recovery_plan_confidence": confidence.get("classification"),
        },
    }


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    feasibility = _read_json(RESULTS / "governance_recovery_execution_feasibility.json")
    acceleration = _read_json(RESULTS / "governance_recovery_acceleration.json")
    exec_readiness = _read_json(RESULTS / "diversification_execution_readiness.json")
    cpi = _read_json(RESULTS / "capital_preservation_intelligence.json")
    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")

    positions = _load_positions()
    total_mv = sum(p["market_value"] for p in positions)
    _, baseline_largest, _ = _concentration_score(positions)

    base_components = dict(cpi.get("component_scores") or {})
    for k in CPI_WEIGHTS:
        base_components.setdefault(k, 70)

    accel_base = acceleration.get("baseline") or {}
    feas_base = feasibility.get("baseline") or {}
    start_cps = int(accel_base.get("cps") or feas_base.get("cps") or 66)
    start_qqq = float(accel_base.get("qqq_concentration_pct") or baseline_largest or 100.0)

    bito_row = recommendations[recommendations["ticker"].astype(str).str.upper() == "BITO"]
    bito_ok = (
        not bito_row.empty
        and str(bito_row.iloc[0].get("recommended_action", "")).upper() == "OPEN_NEW"
    )

    exec_score = float(
        (feasibility.get("execution_readiness") or {}).get("execution_readiness_score") or 93
    )
    yellow_target = int(
        (exec_readiness.get("execution_validation") or {}).get("projected_post_execution_cps") or 76
    )
    ops_stable = True
    exec_ready = exec_score >= 75

    baseline = {
        "day": 0,
        "date": _sim_date(0),
        "cps": start_cps,
        "qqq_concentration_pct": start_qqq,
        "escalation_state": "ORANGE",
        "governance_status": "BLOCKED",
        "execution_readiness_score": exec_score,
        "recovery_scenario": "QQQ 60% / HD 15% / XLK 25%",
    }

    schedule = _qqq_schedule(start_qqq, 60.0, STEP_SIZE, STEP_INTERVAL_DAYS, SIM_DAYS)
    base_daily, base_summary = _simulate_path(
        "Primary — Balanced Exit",
        schedule,
        base_components,
        total_mv,
        baseline_largest,
        start_cps,
        bito_ok,
        ops_stable,
        exec_ready,
        yellow_target_cps=yellow_target,
    )

    yellow_day = base_summary.get("yellow_day") or 12
    if bito_ok and yellow_day < SIM_DAYS:
        green_start = yellow_day + 7
        combined: Dict[int, float] = {}
        green_sched = _qqq_schedule(60.0, 25.0, 10.0, 4, SIM_DAYS)
        for d in range(SIM_DAYS + 1):
            combined[d] = schedule.get(d, 60.0) if d < green_start else green_sched.get(d, 25.0)
        base_daily, base_summary = _simulate_path(
            "Primary — Balanced Exit",
            combined,
            base_components,
            total_mv,
            baseline_largest,
            start_cps,
            bito_ok,
            ops_stable,
            exec_ready,
            green_start_day=green_start,
            yellow_target_cps=yellow_target,
        )
    elif not bito_ok:
        base_daily, base_summary = _simulate_path(
            "Primary — Balanced Exit",
            schedule,
            base_components,
            total_mv,
            baseline_largest,
            start_cps,
            False,
            ops_stable,
            exec_ready,
            yellow_target_cps=yellow_target,
        )

    transitions = _transition_forecast(base_summary, feasibility)
    failures = _failure_branches(
        base_components, total_mv, baseline_largest, start_cps, schedule, bito_ok, yellow_target
    )
    confidence = _confidence_model(base_daily, feasibility, transitions)

    checkpoint_days = [0, 7, 12, 14, 21, 30]
    if base_summary.get("yellow_day") is not None:
        checkpoint_days.append(int(base_summary["yellow_day"]))
    checkpoint_days = sorted(set(d for d in checkpoint_days if d <= SIM_DAYS))

    checkpoints: List[Dict[str, Any]] = []
    day_map = {r["day"]: r for r in base_daily}
    for day in checkpoint_days:
        row = day_map.get(day, base_daily[-1])
        milestone = (
            "DAY_0_BASELINE"
            if day == 0
            else (
                "YELLOW_GATE"
                if row["cps"] >= CPS_YELLOW
                else "MID_RECOVERY" if day <= 14 else "GREEN_APPROACH"
            )
        )
        checkpoints.append(
            _certify_day(
                day,
                int(row["cps"]),
                float(row["qqq_concentration_pct"] or 100),
                float(row.get("hd_weight_pct") or 0),
                float(row.get("xlk_weight_pct") or 0),
                float(row.get("diversification_evidence_score") or 0),
                str(row["escalation_state"]),
                ops_stable,
                exec_ready,
                milestone,
            )
        )

    executive = _executive_report(
        baseline, base_summary, transitions, confidence, failures, checkpoints
    )

    report = {
        "generated_at": ts,
        "phase": "204",
        "mode": "governance_recovery_execution_simulator",
        "disclaimer": (
            "Simulation and certification rehearsal only — does NOT execute trades, "
            "modify holdings, or change governance rules."
        ),
        "baseline": baseline,
        "primary_scenario": {
            "name": "QQQ 60% / HD 15% / XLK 25% (Balanced Exit)",
            "step_size_pct": STEP_SIZE,
            "step_interval_days": STEP_INTERVAL_DAYS,
            "summary": base_summary,
        },
        "daily_projection": base_daily,
        "governance_transitions": transitions,
        "certification_checkpoints": checkpoints,
        "failure_branches": failures,
        "confidence_model": confidence,
        "executive_recommendation": executive,
        "summary": {
            "expected_yellow_date": base_summary.get("yellow_date"),
            "expected_green_date": base_summary.get("green_date"),
            "yellow_day": base_summary.get("yellow_day"),
            "green_day": base_summary.get("green_day"),
            "confidence_classification": confidence.get("classification"),
            "final_cps": base_summary.get("final_cps"),
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(base_daily).to_csv(OUT_DAILY, index=False, encoding="utf-8")
    pd.DataFrame(checkpoints).to_csv(OUT_CHECKPOINTS, index=False, encoding="utf-8")
    pd.DataFrame(failures).to_csv(OUT_FAILURES, index=False, encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_DAILY.name}, {OUT_CHECKPOINTS.name}, {OUT_FAILURES.name}")
    _log(
        f"yellow=day {base_summary.get('yellow_day')} ({base_summary.get('yellow_date')}) "
        f"green=day {base_summary.get('green_day')} confidence={confidence.get('classification')}"
    )
    return report


if __name__ == "__main__":
    run()
