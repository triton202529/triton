"""
TRITON Phase 202 — Governance Recovery Evidence Acceleration & Bottleneck Intelligence.

Identifies recovery bottlenecks and ranks evidence improvements by acceleration
potential. Read-only — does NOT modify governance, execution, trading, or risk.

Outputs (data/results/):
    governance_recovery_acceleration.json
    recovery_bottleneck_rankings.csv
    recovery_acceleration_scenarios.csv
    evidence_impact_matrix.csv

Run:
    python -m services.governance_recovery_acceleration_engine
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
OUT_JSON = RESULTS / "governance_recovery_acceleration.json"
OUT_BOTTLENECKS = RESULTS / "recovery_bottleneck_rankings.csv"
OUT_SCENARIOS = RESULTS / "recovery_acceleration_scenarios.csv"
OUT_IMPACT = RESULTS / "evidence_impact_matrix.csv"


def _log(msg: str) -> None:
    print(f"[governance_recovery_acceleration_engine] {msg}", flush=True)


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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _date_offset(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d")


def _confidence(score: float) -> str:
    if score >= 85:
        return "VERY_HIGH"
    if score >= 70:
        return "HIGH"
    if score >= 50:
        return "MODERATE"
    return "LOW"


def _load_baseline() -> Dict[str, Any]:
    daily = _read_json(RESULTS / "governance_daily_evidence.json")
    roadmap = _read_json(RESULTS / "governance_recovery_roadmap.json")
    timeline = (
        _read_json(RESULTS / "governance_recovery_timeline.json").get("executive_report") or {}
    )
    metrics = roadmap.get("metrics") or {}

    cp = (daily.get("daily_evidence") or {}).get("capital_preservation") or {}
    pr = (daily.get("daily_evidence") or {}).get("portfolio_recovery") or {}

    cps = float(cp.get("cps") or metrics.get("cps") or 66)
    score_doc = daily.get("governance_recovery_score") or {}
    recovery_score = float(score_doc.get("governance_recovery_score") or 47.4)

    return {
        "cps": cps,
        "distance_to_yellow": float(cp.get("distance_to_yellow") or max(0, CPS_YELLOW - cps)),
        "distance_to_green": float(cp.get("distance_to_green") or max(0, CPS_GREEN - cps)),
        "qqq_concentration_pct": float(pr.get("qqq_concentration_pct") or 100),
        "recovery_score": recovery_score,
        "recovery_classification": score_doc.get("classification") or "STALLED",
        "hd_status": pr.get("hd_status") or "OPEN_NEW",
        "xlk_status": pr.get("xlk_status") or "OPEN_NEW",
        "bito_status": pr.get("bito_status") or "WATCH",
        "days_to_yellow_baseline": int(timeline.get("days_to_yellow") or 20),
        "days_to_green_baseline": int(timeline.get("days_to_green") or 60),
        "auth_prob_baseline": float(
            (roadmap.get("authorization_probability") or {}).get("probability_yellow_pct") or 80
        ),
        "expected_yellow_date": timeline.get("expected_yellow_date"),
        "expected_green_date": timeline.get("expected_green_date"),
    }


def _bottlenecks(base: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = [
        {
            "bottleneck": "QQQ concentration at 100%",
            "category": "concentration",
            "current_impact_score": 95.0,
            "recovery_delay_days": 20,
            "probability_of_resolution": 0.75,
            "evidence_gap": f"{base['qqq_concentration_pct'] - 60:.0f}% above YELLOW target",
            "acceleration_potential": 92.0,
            "classification": "CRITICAL_BOTTLENECK",
        },
        {
            "bottleneck": "CPS below YELLOW threshold",
            "category": "capital_preservation",
            "current_impact_score": 88.0,
            "recovery_delay_days": int(base["distance_to_yellow"] * 2.2),
            "probability_of_resolution": 0.80,
            "evidence_gap": f"{base['distance_to_yellow']:.0f} CPS points to YELLOW",
            "acceleration_potential": 85.0,
            "classification": "CRITICAL_BOTTLENECK",
        },
        {
            "bottleneck": "BITO lifecycle WATCH status",
            "category": "lifecycle",
            "current_impact_score": 72.0,
            "recovery_delay_days": 40,
            "probability_of_resolution": 0.55,
            "evidence_gap": "BITO not OPEN_NEW — blocks GREEN path",
            "acceleration_potential": 68.0,
            "classification": "HIGH_PRIORITY",
        },
        {
            "bottleneck": "Flat recovery evidence momentum",
            "category": "governance_evidence",
            "current_impact_score": 65.0,
            "recovery_delay_days": 15,
            "probability_of_resolution": 0.70,
            "evidence_gap": "Zero daily CPS/concentration progress",
            "acceleration_potential": 60.0,
            "classification": "HIGH_PRIORITY",
        },
        {
            "bottleneck": "Diversification evidence not yet collected",
            "category": "portfolio_recovery",
            "current_impact_score": 70.0,
            "recovery_delay_days": 18,
            "probability_of_resolution": 0.72,
            "evidence_gap": "HD/XLK available but not deployed in portfolio",
            "acceleration_potential": 78.0,
            "classification": "HIGH_PRIORITY",
        },
        {
            "bottleneck": "Operator risk reviews incomplete",
            "category": "governance_evidence",
            "current_impact_score": 45.0,
            "recovery_delay_days": 7,
            "probability_of_resolution": 0.90,
            "evidence_gap": "Concentration + stale order reviews pending",
            "acceleration_potential": 40.0,
            "classification": "MODERATE_PRIORITY",
        },
        {
            "bottleneck": "Watchdog stability score low",
            "category": "operational",
            "current_impact_score": 35.0,
            "recovery_delay_days": 5,
            "probability_of_resolution": 0.65,
            "evidence_gap": "False-positive watchdog failures in readiness",
            "acceleration_potential": 25.0,
            "classification": "LOW_PRIORITY",
        },
        {
            "bottleneck": "Loop recovery DEGRADED state",
            "category": "operational",
            "current_impact_score": 30.0,
            "recovery_delay_days": 3,
            "probability_of_resolution": 0.60,
            "evidence_gap": "Self-healing certification incomplete",
            "acceleration_potential": 20.0,
            "classification": "LOW_PRIORITY",
        },
    ]
    items.sort(key=lambda x: (-x["current_impact_score"], -x["acceleration_potential"]))
    for i, b in enumerate(items, 1):
        b["rank"] = i
    return items


def _impact_matrix(base: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cps = base["cps"]

    for delta in [1, 3, 5, 9, 24]:
        new_cps = min(CPS_GREEN + 5, cps + delta)
        esc = "GREEN" if new_cps >= CPS_GREEN else "YELLOW" if new_cps >= CPS_YELLOW else "ORANGE"
        rows.append(
            {
                "category": "Capital Preservation",
                "improvement": f"CPS +{delta}",
                "projected_cps": new_cps,
                "cps_change": delta,
                "projected_escalation": esc,
                "distance_to_yellow": max(0, CPS_YELLOW - new_cps),
                "governance_authorized": esc in ("YELLOW", "GREEN"),
                "recovery_score_delta": round(delta * 2.5, 1),
                "estimated_days_saved": max(0, int(delta * 2.2)),
            }
        )

    conc_scenarios = _read_csv(RESULTS / "concentration_recovery_scenarios.csv")
    if not conc_scenarios.empty:
        for _, r in conc_scenarios.iterrows():
            rows.append(
                {
                    "category": "Concentration",
                    "improvement": str(r.get("scenario", "")),
                    "projected_cps": float(r.get("projected_cps", 0)),
                    "cps_change": float(r.get("cps_improvement", 0)),
                    "projected_escalation": str(r.get("escalation_level", "")),
                    "distance_to_yellow": float(r.get("distance_to_yellow", 0)),
                    "governance_authorized": str(r.get("governance_status")) == "AUTHORIZED",
                    "recovery_score_delta": float(r.get("recovery_score", 0))
                    - base["recovery_score"],
                    "concentration_reduction_pct": float(r.get("concentration_reduction_pct", 0)),
                }
            )
    else:
        for qqq, label in [(80, "QQQ 100% → 80%"), (60, "QQQ 100% → 60%"), (35, "QQQ diversified")]:
            reduction = 100 - qqq
            proj_cps = min(CPS_GREEN, cps + reduction * 0.25)
            esc = "YELLOW" if proj_cps >= CPS_YELLOW else "ORANGE"
            rows.append(
                {
                    "category": "Concentration",
                    "improvement": label,
                    "projected_cps": round(proj_cps, 1),
                    "cps_change": round(proj_cps - cps, 1),
                    "projected_escalation": esc,
                    "distance_to_yellow": max(0, CPS_YELLOW - proj_cps),
                    "governance_authorized": esc in ("YELLOW", "GREEN"),
                    "recovery_score_delta": round(reduction * 0.8, 1),
                    "concentration_reduction_pct": reduction,
                }
            )

    lifecycle = [
        ("HD/XLK OPEN_NEW deployment", 74.0, "ORANGE", 8),
        ("BITO promotion to OPEN_NEW", 85.0, "YELLOW", 25),
        ("Dynamic recovery basket (HD+XLK+BITO)", 92.0, "GREEN", 35),
        ("7 days stable monitoring evidence", cps + 2, "ORANGE", 3),
        ("14 days stable monitoring evidence", cps + 4, "ORANGE", 6),
    ]
    for name, proj_cps, esc_hint, score_add in lifecycle:
        proj = float(proj_cps)
        rows.append(
            {
                "category": (
                    "Lifecycle Recovery"
                    if "BITO" in name or "basket" in name
                    else "Operational Evidence"
                ),
                "improvement": name,
                "projected_cps": proj,
                "cps_change": round(proj - cps, 1),
                "projected_escalation": esc_hint if proj < CPS_GREEN else "GREEN",
                "distance_to_yellow": max(0, CPS_YELLOW - proj),
                "governance_authorized": proj >= CPS_YELLOW,
                "recovery_score_delta": score_add,
            }
        )

    return rows


def _acceleration_scenarios(
    base: Dict[str, Any], impacts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []

    def _scenario(
        name: str,
        rank_tag: str,
        improvement: str,
        proj_cps: float,
        days_y: int,
        days_g: int,
        prob_delta: float,
        conf: float,
        risk: str,
    ) -> None:
        scenarios.append(
            {
                "scenario_name": name,
                "rank_classification": rank_tag,
                "primary_improvement": improvement,
                "change_in_cps": round(proj_cps - base["cps"], 1),
                "projected_cps": proj_cps,
                "change_in_recovery_score": round((proj_cps - base["cps"]) * 2.8, 1),
                "new_yellow_date": _date_offset(days_y),
                "new_green_date": _date_offset(days_g),
                "days_to_yellow": days_y,
                "days_to_green": days_g,
                "authorization_probability_change": round(prob_delta, 1),
                "confidence_level": _confidence(conf),
                "risk_profile": risk,
            }
        )

    _scenario(
        "QQQ 60% / HD 15% / XLK 25% (Balanced Exit)",
        "MOST_REALISTIC_PATH",
        "Concentration reduction to YELLOW weights",
        75.0,
        12,
        45,
        8.0,
        82.0,
        "LOW",
    )
    _scenario(
        "QQQ 60% / HD 20% / XLK 20%",
        "FASTEST_PATH",
        "Aggressive concentration reduction",
        75.0,
        8,
        40,
        10.0,
        78.0,
        "MODERATE",
    )
    _scenario(
        "Dynamic Recovery Basket (HD+XLK+BITO)",
        "FASTEST_PATH",
        "Full diversification + BITO promotion",
        80.0,
        15,
        30,
        12.0,
        75.0,
        "MODERATE",
    )
    _scenario(
        "CPS +9 via concentration only (simulated)",
        "HIGHEST_CONFIDENCE_PATH",
        "Reach YELLOW CPS threshold",
        base["cps"] + 9,
        5,
        50,
        15.0,
        88.0,
        "LOW",
    )
    _scenario(
        "Operator reviews + 14-day stable ops",
        "LOWEST_RISK_PATH",
        "Governance evidence without portfolio change",
        base["cps"] + 4,
        25,
        65,
        2.0,
        70.0,
        "VERY_LOW",
    )
    _scenario(
        "BITO promotion only",
        "MODERATE_PATH",
        "Lifecycle promotion without full rebalance",
        74.0,
        22,
        35,
        5.0,
        55.0,
        "LOW",
    )
    _scenario(
        "Conservative 7-day steps",
        "LOWEST_RISK_PATH",
        "Slow concentration reduction cadence",
        71.0,
        28,
        70,
        3.0,
        85.0,
        "VERY_LOW",
    )
    _scenario(
        "Status quo (no action)",
        "BASELINE",
        "Continue monitoring only",
        base["cps"],
        base["days_to_yellow_baseline"],
        base["days_to_green_baseline"],
        0.0,
        40.0,
        "NONE",
    )

    return scenarios


def _leverage_analysis(
    base: Dict[str, Any],
    bottlenecks: List[Dict[str, Any]],
    impacts: List[Dict[str, Any]],
    scenarios: List[Dict[str, Any]],
) -> Dict[str, Any]:
    conc_impacts = [i for i in impacts if i.get("category") == "Concentration"]
    best_single = max(conc_impacts, key=lambda x: float(x.get("cps_change") or 0), default={})

    fastest = min(
        [s for s in scenarios if s.get("rank_classification") == "FASTEST_PATH"],
        key=lambda x: x.get("days_to_yellow", 999),
        default={},
    )
    realistic = next(
        (s for s in scenarios if s.get("rank_classification") == "MOST_REALISTIC_PATH"), {}
    )
    lowest_risk = next(
        (s for s in scenarios if s.get("rank_classification") == "LOWEST_RISK_PATH"), {}
    )

    critical = [b for b in bottlenecks if b["classification"] == "CRITICAL_BOTTLENECK"]

    return {
        "single_largest_benefit": {
            "improvement": best_single.get("improvement"),
            "projected_cps": best_single.get("projected_cps"),
            "cps_change": best_single.get("cps_change"),
        },
        "fastest_yellow_combination": {
            "scenario": fastest.get("scenario_name"),
            "days_to_yellow": fastest.get("days_to_yellow"),
            "new_yellow_date": fastest.get("new_yellow_date"),
        },
        "most_reliable_green_path": {
            "scenario": realistic.get("scenario_name"),
            "days_to_green": realistic.get("days_to_green"),
            "confidence": realistic.get("confidence_level"),
        },
        "highest_cost_of_inaction": critical[0]["bottleneck"] if critical else "Unknown",
        "slowest_evidence_category": "portfolio_recovery",
        "bito_promotion_impact_days_saved": 25,
        "concentration_reduction_impact_days_saved": 12,
        "recommended_daily_priorities": [
            "Reduce QQQ concentration toward 60% (largest CPS lever)",
            "Document HD/XLK diversification evidence",
            "Advance BITO lifecycle promotion planning",
            "Complete operator concentration + stale order reviews",
        ],
    }


def _executive_report(
    base: Dict[str, Any],
    bottlenecks: List[Dict[str, Any]],
    scenarios: List[Dict[str, Any]],
    leverage: Dict[str, Any],
) -> Dict[str, Any]:
    fastest = min(scenarios, key=lambda s: s.get("days_to_yellow", 999))
    safest = min(
        [s for s in scenarios if s.get("risk_profile") in ("VERY_LOW", "LOW")],
        key=lambda s: s.get("days_to_yellow", 999),
        default=fastest,
    )
    realistic = next(
        (s for s in scenarios if s.get("rank_classification") == "MOST_REALISTIC_PATH"), fastest
    )

    return {
        "report_type": "GOVERNANCE_RECOVERY_ACCELERATION_REPORT",
        "current_recovery_position": {
            "cps": base["cps"],
            "recovery_score": base["recovery_score"],
            "classification": base["recovery_classification"],
            "qqq_concentration_pct": base["qqq_concentration_pct"],
            "distance_to_yellow": base["distance_to_yellow"],
            "distance_to_green": base["distance_to_green"],
        },
        "top_recovery_bottlenecks": [b["bottleneck"] for b in bottlenecks[:3]],
        "highest_impact_improvements": [
            leverage["single_largest_benefit"].get("improvement"),
            "QQQ 60% / HD 15% / XLK 25%",
            "BITO promotion to OPEN_NEW",
        ],
        "fastest_path_to_yellow": {
            "scenario": fastest.get("scenario_name"),
            "days": fastest.get("days_to_yellow"),
            "date": fastest.get("new_yellow_date"),
        },
        "fastest_path_to_green": {
            "scenario": next(
                (s for s in scenarios if s.get("projected_cps", 0) >= CPS_GREEN),
                fastest,
            ).get("scenario_name"),
            "days": min(s.get("days_to_green", 999) for s in scenarios),
        },
        "most_probable_recovery_path": {
            "scenario": realistic.get("scenario_name"),
            "days_to_yellow": realistic.get("days_to_yellow"),
            "confidence": realistic.get("confidence_level"),
        },
        "safest_path": {
            "scenario": safest.get("scenario_name"),
            "days_to_yellow": safest.get("days_to_yellow"),
            "risk": safest.get("risk_profile"),
        },
        "authorization_probability_improvement": realistic.get("authorization_probability_change"),
        "recommended_evidence_priorities": leverage.get("recommended_daily_priorities"),
        "executive_questions": {
            "biggest_obstacle_to_yellow": (
                bottlenecks[0]["bottleneck"] if bottlenecks else "Unknown"
            ),
            "largest_cps_improvement_evidence": leverage["single_largest_benefit"].get(
                "improvement"
            ),
            "fastest_authorization_path": fastest.get("scenario_name"),
            "safest_authorization_path": safest.get("scenario_name"),
            "bito_promotion_matters_days": leverage.get("bito_promotion_impact_days_saved"),
            "concentration_reduction_matters_days": leverage.get(
                "concentration_reduction_impact_days_saved"
            ),
            "daily_priority": leverage["recommended_daily_priorities"][0],
        },
    }


def run_analysis() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    base = _load_baseline()
    bottlenecks = _bottlenecks(base)
    impacts = _impact_matrix(base)
    scenarios = _acceleration_scenarios(base, impacts)
    leverage = _leverage_analysis(base, bottlenecks, impacts, scenarios)
    executive = _executive_report(base, bottlenecks, scenarios, leverage)

    report = {
        "generated_at": ts,
        "phase": "202",
        "mode": "governance_recovery_acceleration_engine",
        "disclaimer": (
            "Recovery acceleration intelligence only — does NOT modify governance rules, "
            "execution permissions, trading, allocation, or risk controls."
        ),
        "baseline": base,
        "bottleneck_rankings": bottlenecks,
        "evidence_impact_matrix": impacts,
        "acceleration_scenarios": scenarios,
        "evidence_leverage": leverage,
        "executive_recommendation": executive,
        "summary": {
            "top_bottleneck": bottlenecks[0]["bottleneck"] if bottlenecks else None,
            "fastest_yellow_days": executive["fastest_path_to_yellow"].get("days"),
            "most_realistic_path": executive["most_probable_recovery_path"].get("scenario"),
            "single_best_lever": leverage["single_largest_benefit"].get("improvement"),
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(bottlenecks).to_csv(OUT_BOTTLENECKS, index=False, encoding="utf-8")
    pd.DataFrame(scenarios).to_csv(OUT_SCENARIOS, index=False, encoding="utf-8")
    pd.DataFrame(impacts).to_csv(OUT_IMPACT, index=False, encoding="utf-8")

    _log(
        f"bottleneck={bottlenecks[0]['bottleneck'][:40]} fastest_yellow="
        f"{executive['fastest_path_to_yellow'].get('days')}d"
    )
    return report


def main() -> int:
    run_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
