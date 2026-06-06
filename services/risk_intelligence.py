"""
TRITON Advanced Risk Intelligence — Phases 10–12 (read-only).

Defensive simulation, predictive risk, and executive summaries.
No trading, orders, or portfolio modifications.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

CONCENTRATION_CAPS = (10.0, 15.0, 20.0, 25.0)
EXPOSURE_REDUCTION_LEVELS = (0.10, 0.20, 0.30)
ESCALATION_ORDER = ["GREEN", "YELLOW", "ORANGE", "RED", "CRITICAL"]
TREND_DELTA = 3.0


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


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _largest_concentration_pct(positions: List[Dict[str, Any]]) -> float:
    mvs = [abs(_safe_float(p.get("market_value"), 0.0) or 0.0) for p in positions]
    total = sum(mvs)
    if total <= 0:
        return 0.0
    return (max(mvs) / total) * 100.0


def _escalation_from_cps(cps: float) -> str:
    if cps >= 90:
        return "GREEN"
    if cps >= 75:
        return "YELLOW"
    if cps >= 60:
        return "ORANGE"
    if cps >= 40:
        return "RED"
    return "CRITICAL"


def _simulation_deltas(
    *,
    concentration_excess: float = 0.0,
    drawdown_positions: int = 0,
    exposure_reduction: float = 0.0,
    elevated_risk: bool = False,
) -> Dict[str, float]:
    """Heuristic counterfactual deltas (simulation only, not executed)."""
    conc_factor = max(0.0, concentration_excess)
    dd_factor = float(drawdown_positions)
    exp_factor = exposure_reduction

    return {
        "portfolio_return_delta": round(-min(4.0, conc_factor * 0.08 + exp_factor * 2.5), 2),
        "max_drawdown_delta": round(
            -min(12.0, conc_factor * 0.35 + dd_factor * 1.5 + exp_factor * 8.0), 2
        ),
        "volatility_delta": round(-min(8.0, conc_factor * 0.2 + exp_factor * 5.0), 2),
        "concentration_delta": round(-conc_factor if conc_factor > 0 else 0.0, 2),
        "risk_score_delta": round(
            min(
                20.0,
                conc_factor * 0.45
                + dd_factor * 2.0
                + exp_factor * 12.0
                + (5.0 if elevated_risk else 0.0),
            ),
            2,
        ),
    }


def compute_defensive_simulation_results(
    *,
    positions: List[Dict[str, Any]],
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    drawdown_threshold_pct: float = -10.0,
) -> Dict[str, Any]:
    """Phase 10: counterfactual defensive control simulations (read-only)."""
    ts = _iso_utc()
    largest_pct = _largest_concentration_pct(positions)
    current_cps = int(cpi_doc.get("capital_preservation_score") or 0)
    escalation = str(cpe_doc.get("escalation_state") or "GREEN")
    elevated = escalation in ("ORANGE", "RED", "CRITICAL")

    simulations: List[Dict[str, Any]] = []

    for cap in CONCENTRATION_CAPS:
        excess = max(0.0, largest_pct - cap)
        deltas = _simulation_deltas(concentration_excess=excess)
        simulations.append(
            {
                "simulation_type": "concentration_cap",
                "simulation_name": f"Concentration Cap {int(cap)}%",
                "parameters": {
                    "max_position_pct": cap,
                    "current_largest_pct": round(largest_pct, 2),
                },
                **deltas,
                "note": "Counterfactual estimate only — no positions modified",
            }
        )

    drawdown_positions = [
        p
        for p in positions
        if (_safe_float(p.get("unrealized_pl_pct")) or 0.0) <= drawdown_threshold_pct
    ]
    dd_deltas = _simulation_deltas(drawdown_positions=len(drawdown_positions))
    simulations.append(
        {
            "simulation_type": "drawdown_protection",
            "simulation_name": "Drawdown Protection Review",
            "parameters": {
                "drawdown_threshold_pct": drawdown_threshold_pct,
                "positions_below_threshold": len(drawdown_positions),
                "symbols": [p.get("symbol") for p in drawdown_positions],
            },
            **dd_deltas,
            "note": "Simulates earlier review of positions exceeding drawdown threshold",
        }
    )

    for reduction in EXPOSURE_REDUCTION_LEVELS:
        exp_deltas = _simulation_deltas(
            exposure_reduction=reduction,
            elevated_risk=elevated,
        )
        simulations.append(
            {
                "simulation_type": "exposure_reduction",
                "simulation_name": f"Exposure Reduction {int(reduction * 100)}%",
                "parameters": {
                    "exposure_reduction_pct": int(reduction * 100),
                    "during_escalation": escalation,
                },
                **exp_deltas,
                "note": "Simulates reduced exposure during elevated risk periods",
            }
        )

    return {
        "generated_at": ts,
        "baseline": {
            "capital_preservation_score": current_cps,
            "escalation_state": escalation,
            "largest_concentration_pct": round(largest_pct, 2),
        },
        "simulations": simulations,
        "disclaimer": "Simulation only. No trades, orders, or portfolio changes executed.",
    }


def _parse_history_scores(rows: List[Dict[str, str]]) -> List[Tuple[datetime, float, int]]:
    out: List[Tuple[datetime, float, int]] = []
    for row in rows:
        raw_ts = row.get("timestamp")
        score = _safe_float(row.get("capital_preservation_score"))
        if not raw_ts or score is None:
            continue
        try:
            dt = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            alerts = int(float(row.get("active_alerts") or 0))
            out.append((dt, score, alerts))
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda x: x[0])
    return out


def _alert_events_from_advisory_history(rows: List[Dict[str, str]]) -> List[datetime]:
    events: List[datetime] = []
    for row in rows:
        raw_ts = row.get("timestamp")
        if not raw_ts:
            continue
        try:
            dt = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            events.append(dt)
        except ValueError:
            continue
    return sorted(events)


def compute_predictive_risk_intelligence(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpi_history_path: Path,
    cpa_history_path: Path,
) -> Dict[str, Any]:
    """Phase 11: predictive risk metrics from historical watchdog artifacts."""
    ts = _iso_utc()
    current_cps = float(cpi_doc.get("capital_preservation_score") or 0)
    current_esc = str(cpe_doc.get("escalation_state") or _escalation_from_cps(current_cps))

    history = _parse_history_scores(_read_csv_rows(cpi_history_path))
    alert_events = _alert_events_from_advisory_history(_read_csv_rows(cpa_history_path))

    if len(history) >= 2:
        first_score = history[0][1]
        last_score = history[-1][1]
        if last_score >= first_score + TREND_DELTA:
            risk_momentum = "IMPROVING"
        elif last_score <= first_score - TREND_DELTA:
            risk_momentum = "DETERIORATING"
        else:
            risk_momentum = "STABLE"
        slope = (history[-1][1] - history[0][1]) / max(1, len(history) - 1)
    else:
        risk_momentum = str(cpi_doc.get("risk_trend") or "STABLE")
        slope = 0.0

    now = _utc_now()
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    alerts_24h = sum(1 for e in alert_events if e >= day_ago)
    alerts_7d = sum(1 for e in alert_events if e >= week_ago)
    alert_velocity = round(alerts_24h / 1.0, 2)
    prior_velocity = round(max(0, alerts_7d - alerts_24h) / 6.0, 2) if alerts_7d else 0.0
    if alert_velocity > prior_velocity + 0.5:
        alert_acceleration = "INCREASING"
    elif alert_velocity + 0.5 < prior_velocity:
        alert_acceleration = "DECREASING"
    else:
        alert_acceleration = "STABLE"

    projected_cps = round(max(0.0, min(100.0, current_cps + slope * 5)), 1)
    projected_esc = _escalation_from_cps(projected_cps)
    risk_direction = risk_momentum

    days_to_threshold = None
    if slope < -0.01 and current_cps > 40:
        days_to_threshold = int(max(1, (current_cps - 40) / abs(slope)))
    elif slope < -0.01 and current_cps > 60:
        days_to_threshold = int(max(1, (current_cps - 60) / abs(slope)))

    confidence = min(95, max(45, 40 + len(history) * 8))

    return {
        "generated_at": ts,
        "risk_momentum": risk_momentum,
        "risk_direction": risk_direction,
        "alert_velocity": alert_velocity,
        "alert_velocity_7d_avg": round(alerts_7d / 7.0, 2) if alerts_7d else 0.0,
        "alert_acceleration": alert_acceleration,
        "portfolio_health_forecast": {
            "current_cps": current_cps,
            "projected_cps": projected_cps,
            "cps_slope_per_cycle": round(slope, 3),
        },
        "escalation_forecast": {
            "current_escalation": current_esc,
            "projected_escalation": projected_esc,
        },
        "forecast_confidence": confidence,
        "estimated_days_to_threshold": days_to_threshold,
        "disclaimer": "Predictive estimates only — not trade signals or actions",
    }


def compute_executive_risk_artifacts(
    *,
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpa_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    sim_doc: Dict[str, Any],
    alerts_doc: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Phase 12: executive risk summary and full report (read-only)."""
    ts = _iso_utc()
    cps = int(cpi_doc.get("capital_preservation_score") or 0)
    escalation = str(cpe_doc.get("escalation_state") or "GREEN")
    health_band = str(cpi_doc.get("health_band") or "—")
    risk_trend = str(cpi_doc.get("risk_trend") or "STABLE")

    top_risks: List[str] = []
    for adv in cpa_doc.get("advisories") or []:
        if isinstance(adv, dict):
            title = str(adv.get("title") or "").strip()
            if title and title not in top_risks:
                top_risks.append(title)
    if not top_risks:
        top_risks = list(gov_doc.get("governance_summary") or [])

    watchlist = [
        {
            "issue": adv.get("title"),
            "priority": adv.get("priority"),
            "urgency_score": adv.get("urgency_score"),
            "subject": adv.get("subject"),
        }
        for adv in (cpa_doc.get("advisories") or [])[:10]
        if isinstance(adv, dict)
    ]

    incident = alerts_doc.get("incident_intelligence") or {}
    cpa_summary = cpa_doc.get("summary") or {}

    summary = {
        "generated_at": ts,
        "executive_summary": {
            "portfolio_health": health_band,
            "capital_preservation_score": cps,
            "escalation_state": escalation,
            "governance_awareness": gov_doc.get("governance_awareness_label"),
            "top_risks": top_risks[:5],
            "risk_direction": pred_doc.get("risk_direction"),
            "projected_escalation": pred_doc.get("escalation_forecast", {}).get(
                "projected_escalation"
            ),
        },
        "capital_preservation_score": cps,
        "escalation_state": escalation,
        "governance_awareness_level": gov_doc.get("governance_awareness_level"),
        "governance_awareness_label": gov_doc.get("governance_awareness_label"),
        "top_risks": top_risks,
        "predictive_outlook": {
            "risk_direction": pred_doc.get("risk_direction"),
            "forecast_confidence": pred_doc.get("forecast_confidence"),
            "projected_cps": pred_doc.get("portfolio_health_forecast", {}).get("projected_cps"),
            "estimated_days_to_threshold": pred_doc.get("estimated_days_to_threshold"),
        },
        "incident_intelligence": incident,
        "risk_trend": risk_trend,
        "strategic_watchlist": watchlist,
        "advisory_count": cpa_summary.get("advisory_count", 0),
    }

    report = {
        **summary,
        "full_context": {
            "cpi": {
                "health_band": health_band,
                "component_scores": cpi_doc.get("component_scores"),
                "risk_trend": risk_trend,
            },
            "escalation": {
                "reasons": cpe_doc.get("escalation_reason"),
                "recommended_actions": cpe_doc.get("recommended_review_actions"),
            },
            "predictive": pred_doc,
            "simulation_highlights": (sim_doc.get("simulations") or [])[:4],
            "active_alert_count": len(alerts_doc.get("active_alerts") or []),
        },
        "disclaimer": "Executive read-only briefing. No trading or governance actions executed.",
    }
    return summary, report


def persist_advanced_risk_intelligence(
    *,
    results_dir: Path,
    positions: List[Dict[str, Any]],
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpa_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    alerts_doc: Dict[str, Any],
    drawdown_threshold_pct: float = -10.0,
) -> Dict[str, Any]:
    """Run phases 10–12 and write all JSON artifacts."""
    results_dir = Path(results_dir)
    sim_doc = compute_defensive_simulation_results(
        positions=positions,
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        drawdown_threshold_pct=drawdown_threshold_pct,
    )
    pred_doc = compute_predictive_risk_intelligence(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        cpi_history_path=results_dir / "capital_preservation_history.csv",
        cpa_history_path=results_dir / "capital_preservation_advisory_history.csv",
    )
    exec_summary, exec_report = compute_executive_risk_artifacts(
        cpi_doc=cpi_doc,
        cpe_doc=cpe_doc,
        cpa_doc=cpa_doc,
        gov_doc=gov_doc,
        pred_doc=pred_doc,
        sim_doc=sim_doc,
        alerts_doc=alerts_doc,
    )

    _atomic_write_json(sim_doc, results_dir / "defensive_simulation_results.json")
    _atomic_write_json(pred_doc, results_dir / "predictive_risk_intelligence.json")
    _atomic_write_json(exec_summary, results_dir / "executive_risk_summary.json")
    _atomic_write_json(exec_report, results_dir / "executive_risk_report.json")

    return {
        "defensive_simulation": sim_doc,
        "predictive_risk_intelligence": pred_doc,
        "executive_risk_summary": exec_summary,
        "executive_risk_report": exec_report,
    }
