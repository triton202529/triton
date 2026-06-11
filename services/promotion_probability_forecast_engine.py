"""
TRITON Phase 183 — Promotion Probability Forecast Engine (forecasting only).

Forecasts future lifecycle promotions and estimates the probability that recovery
candidates transition into executable opportunities.

SAFETY: forecasting only — no governance, lifecycle, execution, broker, allocation,
or risk control changes.

Outputs (data/results/):
    promotion_probability_forecast.json
    promotion_probability_forecast.csv
    promotion_probability_history.csv

Run:
    python -m services.promotion_probability_forecast_engine
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

from services.concentration_governance_analysis import _f, _read_csv, _read_json
from services.diversification_feasibility_analysis import MIN_OPEN_NEW_CONFIDENCE, STRONG_EDGE_SCORE
from services.pnl_ledger_builder import _norm_sym, _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "promotion_probability_forecast.json"
OUT_CSV = RESULTS / "promotion_probability_forecast.csv"
OUT_HISTORY = RESULTS / "promotion_probability_history.csv"

FORECAST_HORIZONS = [7, 14, 30, 60, 90]
RECOVERY_FOCUS = ["HD", "XLK", "BITO"]
PROMOTION_PAIRS = [("WATCH", "OPEN_NEW"), ("OPEN_NEW", "WATCH"), ("BUY", "HOLD"), ("HOLD", "EXIT")]


def _log(msg: str) -> None:
    print(f"[promotion_probability_forecast_engine] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _date_offset(base: datetime, days: int) -> str:
    return (base + timedelta(days=days)).strftime("%Y-%m-%d")


def _classify_prob(p: float) -> str:
    if p >= 0.80:
        return "VERY_LIKELY"
    if p >= 0.60:
        return "LIKELY"
    if p >= 0.40:
        return "POSSIBLE"
    if p >= 0.20:
        return "UNLIKELY"
    return "VERY_UNLIKELY"


def _trend(series: pd.Series) -> str:
    if len(series) < 2:
        return "STABLE"
    delta = float(series.iloc[-1] - series.iloc[0])
    if delta > 0.01:
        return "IMPROVING"
    if delta < -0.01:
        return "DETERIORATING"
    return "STABLE"


def _history_trends(status_hist: pd.DataFrame, sym: str) -> Tuple[str, str, float]:
    sub = status_hist[status_hist["ticker"].map(_norm_sym) == sym].copy()
    if sub.empty:
        return "STABLE", "STABLE", 0.0
    sub = sub.sort_values("generated_at")
    conf = pd.to_numeric(sub["confidence"], errors="coerce").dropna()
    edge = pd.to_numeric(sub["edge_score"], errors="coerce").dropna()
    conf_tr = _trend(conf) if len(conf) >= 2 else "STABLE"
    edge_tr = _trend(edge) if len(edge) >= 2 else "STABLE"
    conf_vel = float(conf.diff().mean()) if len(conf) >= 2 else 0.0
    return conf_tr, edge_tr, conf_vel


def _signal_strength(conf: float, edge: float, delta: float) -> float:
    return float(
        np.clip(
            0.40 * min(conf / MIN_OPEN_NEW_CONFIDENCE, 1.2)
            + 0.35 * min(edge / STRONG_EDGE_SCORE, 1.2)
            + 0.25 * min(max(delta, 0) / 0.02, 1.0),
            0.0,
            1.0,
        )
    )


def _promotion_probs(
    action: str,
    conf: float,
    edge: float,
    conf_tr: str,
    edge_tr: str,
    conf_vel: float,
    horizon: int,
) -> Tuple[float, float, float]:
    """Return (promotion_prob, demotion_prob, stability_prob) for horizon days."""
    action = action.upper()
    if action == "OPEN_NEW":
        demote = 0.05 + 0.02 * (horizon / 90.0)
        if conf_tr == "DETERIORATING":
            demote += 0.10
        promote = 0.02
        stability = max(0.0, 1.0 - demote - promote)
        return promote, demote, stability

    if action == "WATCH":
        gap = max(0.0, MIN_OPEN_NEW_CONFIDENCE - conf)
        base = max(0.0, 1.0 - gap / MIN_OPEN_NEW_CONFIDENCE)
        if edge >= STRONG_EDGE_SCORE:
            base += 0.15
        if conf_tr == "IMPROVING":
            base += 0.10
        elif conf_tr == "DETERIORATING":
            base -= 0.15
        horizon_factor = min(horizon / 30.0, 2.0)
        promote = float(np.clip(base * 0.35 * horizon_factor, 0.05, 0.95))
        demote = float(np.clip(0.08 * (horizon / 60.0), 0.02, 0.20))
        if conf_vel > 0:
            promote = min(0.95, promote + conf_vel * horizon * 5)
        stability = max(0.0, 1.0 - promote - demote)
        return promote, demote, stability

    if action in ("ADD", "HOLD"):
        promote = 0.05
        demote = 0.10 + 0.05 * (horizon / 90.0)
        stability = max(0.0, 1.0 - promote - demote)
        return promote, demote, stability

    return 0.10, 0.15, 0.75


def _expected_promotion_date(
    action: str,
    conf: float,
    conf_vel: float,
    base_dt: datetime,
    horizon_probs: Dict[int, float],
) -> Optional[str]:
    if action.upper() == "OPEN_NEW":
        return None
    if action.upper() != "WATCH":
        return None
    gap = max(0.0, MIN_OPEN_NEW_CONFIDENCE - conf)
    if gap <= 0:
        return _date_offset(base_dt, 1)
    daily = max(conf_vel, gap / 45.0) if conf_vel > 0 else gap / 30.0
    days = int(np.ceil(gap / daily)) if daily > 0 else 90
    for h in FORECAST_HORIZONS:
        if horizon_probs.get(h, 0) >= 0.50:
            days = min(days, h)
            break
    return _date_offset(base_dt, min(days, 90))


def _forecast_candidate(
    sym: str,
    rec: Dict[str, Any],
    status_hist: pd.DataFrame,
    base_dt: datetime,
    in_recovery: bool,
) -> Dict[str, Any]:
    conf = _to_float(rec.get("confidence"), 0.0)
    edge = _to_float(rec.get("edge_score"), 0.0)
    delta = _to_float(rec.get("delta_pct"), 0.0)
    action = str(rec.get("recommended_action") or rec.get("allocation_status") or "—")
    lc = str(rec.get("lifecycle_action") or rec.get("lifecycle_status") or "—")
    conf_tr, edge_tr, conf_vel = _history_trends(status_hist, sym)
    signal = _signal_strength(conf, edge, delta)

    horizon_forecasts: Dict[str, Any] = {}
    horizon_probs: Dict[int, float] = {}
    for h in FORECAST_HORIZONS:
        prom, dem, stab = _promotion_probs(action, conf, edge, conf_tr, edge_tr, conf_vel, h)
        horizon_probs[h] = prom
        horizon_forecasts[str(h)] = {
            "horizon_days": h,
            "promotion_probability": _f(prom * 100, 1),
            "demotion_probability": _f(dem * 100, 1),
            "stability_probability": _f(stab * 100, 1),
            "promotion_classification": _classify_prob(prom),
        }

    h30 = horizon_probs.get(30, 0.0)
    expected_date = _expected_promotion_date(action, conf, conf_vel, base_dt, horizon_probs)

    recovery_impact = "LOW"
    if sym in ("HD", "XLK"):
        recovery_impact = "CRITICAL_YELLOW" if action == "OPEN_NEW" else "HIGH_YELLOW"
    elif sym == "BITO":
        recovery_impact = "CRITICAL_GREEN"

    return {
        "ticker": sym,
        "current_status": action,
        "lifecycle_status": lc,
        "confidence": _f(conf, 4),
        "edge_score": _f(edge, 4),
        "confidence_trend": conf_tr,
        "edge_trend": edge_tr,
        "signal_strength": _f(signal * 100, 1),
        "in_recovery_basket": in_recovery,
        "horizon_forecasts": horizon_forecasts,
        "promotion_probability_30d": _f(h30 * 100, 1),
        "promotion_classification_30d": _classify_prob(h30),
        "expected_promotion_date": expected_date,
        "confidence_gap_to_open_new": _f(max(0, MIN_OPEN_NEW_CONFIDENCE - conf), 4),
        "recovery_impact": recovery_impact,
        "forecasted_events": _forecast_events(action, h30),
    }


def _forecast_events(action: str, prom_30: float) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    action = action.upper()
    if action == "WATCH" and prom_30 > 0.2:
        events.append(
            {
                "transition": "WATCH → OPEN_NEW",
                "probability_30d": _f(prom_30 * 100, 1),
                "classification": _classify_prob(prom_30),
            }
        )
    if action == "OPEN_NEW":
        events.append(
            {
                "transition": "OPEN_NEW → WATCH",
                "probability_30d": _f(0.08 * 100, 1),
                "classification": "UNLIKELY",
            }
        )
    return events


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    base_dt = datetime.now(timezone.utc)

    monitor = _read_json(RESULTS / "recovery_basket_monitor.json")
    timeline = _read_json(RESULTS / "governance_recovery_timeline.json")
    readiness = _read_json(RESULTS / "diversification_execution_readiness.json")
    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    status_hist = _read_csv(RESULTS / "recovery_basket_status.csv")
    promo_hist = _read_csv(RESULTS / "recovery_basket_promotion_history.csv")

    if recommendations.empty:
        recommendations = pd.DataFrame()

    open_new = (
        recommendations[recommendations["recommended_action"].astype(str).str.upper() == "OPEN_NEW"]
        if "recommended_action" in recommendations.columns
        else pd.DataFrame()
    )
    watch = (
        recommendations[recommendations["recommended_action"].astype(str).str.upper() == "WATCH"]
        if "recommended_action" in recommendations.columns
        else pd.DataFrame()
    )

    eval_syms = list(
        dict.fromkeys(
            RECOVERY_FOCUS
            + [_norm_sym(t) for t in open_new.get("ticker", pd.Series()).tolist()]
            + [_norm_sym(t) for t in watch.get("ticker", pd.Series()).tolist()]
        )
    )

    basket_status = {r.get("ticker"): r for r in (monitor.get("recovery_basket_status") or [])}

    candidate_forecasts: List[Dict[str, Any]] = []
    for sym in eval_syms:
        rec_row = recommendations[recommendations["ticker"].map(_norm_sym) == sym]
        rec = rec_row.iloc[0].to_dict() if not rec_row.empty else {"ticker": sym}
        if sym in basket_status:
            bs = basket_status[sym]
            rec.setdefault("recommended_action", bs.get("allocation_status"))
            rec.setdefault("confidence", bs.get("confidence"))
            rec.setdefault("edge_score", bs.get("edge_score"))
        candidate_forecasts.append(
            _forecast_candidate(sym, rec, status_hist, base_dt, sym in RECOVERY_FOCUS)
        )

    candidate_forecasts.sort(
        key=lambda c: _to_float(c.get("promotion_probability_30d"), 0), reverse=True
    )

    bito = next((c for c in candidate_forecasts if c["ticker"] == "BITO"), {})
    hd = next((c for c in candidate_forecasts if c["ticker"] == "HD"), {})
    xlk = next((c for c in candidate_forecasts if c["ticker"] == "XLK"), {})

    days_to_green = _to_float((timeline.get("executive_report") or {}).get("days_to_green"), 60)
    days_to_yellow = _to_float((timeline.get("executive_report") or {}).get("days_to_yellow"), 20)

    promotion_dependencies = {
        "most_important_for_recovery": (
            "BITO" if bito.get("current_status") == "WATCH" else "HD+XLK"
        ),
        "closest_to_promotion": next(
            (
                {"ticker": c["ticker"], "prob_30d": c.get("promotion_probability_30d")}
                for c in candidate_forecasts
                if c.get("current_status") == "WATCH"
            ),
            {"ticker": "—", "note": "All OPEN_NEW candidates already promoted"},
        ),
        "green_blocker": "BITO",
        "bito_open_new_probability_30d": bito.get("promotion_probability_30d"),
        "bito_open_new_probability_60d": (bito.get("horizon_forecasts") or {})
        .get("60", {})
        .get("promotion_probability"),
        "bito_days_to_threshold": (
            int(np.ceil(_to_float(bito.get("confidence_gap_to_open_new"), 0.025) / 0.001))
            if bito.get("confidence_gap_to_open_new")
            else None
        ),
        "bito_expected_promotion_date": bito.get("expected_promotion_date"),
        "hd_xlk_status": "OPEN_NEW — YELLOW path ready",
    }

    bito_prob_30 = _to_float(bito.get("promotion_probability_30d"), 45) / 100.0
    recovery_impact = {
        "impact_on_yellow": {
            "level": "HIGH",
            "reason": "HD+XLK already OPEN_NEW — YELLOW executable",
            "probability": 95.0,
        },
        "impact_on_green": {
            "level": "CRITICAL",
            "reason": "BITO WATCH blocks GREEN until OPEN_NEW promotion",
            "probability": _f(bito_prob_30 * 100, 1),
        },
        "impact_on_recovery_probability": _f((0.95 * 0.5 + bito_prob_30 * 0.5) * 100, 1),
        "bito_promotion_accelerates_green_by_days": (
            max(0, int(days_to_green - 30)) if bito_prob_30 > 0.6 else None
        ),
    }

    timeline_impact = {
        "days_to_yellow": int(days_to_yellow),
        "days_to_green_baseline": int(days_to_green),
        "days_to_green_if_bito_promotes_30d": max(30, int(days_to_green - 15)),
        "days_to_green_if_bito_delayed_90d": int(days_to_green) + 30,
        "yellow_blocked_by": None,
        "green_blocked_by": "BITO WATCH → OPEN_NEW promotion required",
        "candidate_readiness_for_yellow": "HD+XLK READY",
    }

    promotion_probabilities = {
        c["ticker"]: {
            h: (c.get("horizon_forecasts") or {}).get(str(h), {}) for h in FORECAST_HORIZONS
        }
        for c in candidate_forecasts
    }

    executive_rows = [
        {
            "candidate": c["ticker"],
            "current_status": c["current_status"],
            "promotion_probability_30d": c.get("promotion_probability_30d"),
            "expected_promotion_date": c.get("expected_promotion_date"),
            "recovery_impact": c.get("recovery_impact"),
            "timeline_impact": (
                f"GREEN gate"
                if c["ticker"] == "BITO"
                else "YELLOW support" if c["ticker"] in ("HD", "XLK") else "LOW"
            ),
            "confidence_level": c.get("promotion_classification_30d"),
        }
        for c in candidate_forecasts
        if c["ticker"] in RECOVERY_FOCUS or c.get("current_status") in ("OPEN_NEW", "WATCH")
    ]

    executive = {
        "report_type": "PROMOTION_PROBABILITY_FORECAST_REPORT",
        "candidates": executive_rows,
        "next_promotion_candidate": promotion_dependencies.get("closest_to_promotion"),
        "bito_forecast": {
            "probability_30d": bito.get("promotion_probability_30d"),
            "probability_60d": (bito.get("horizon_forecasts") or {})
            .get("60", {})
            .get("promotion_probability"),
            "expected_date": bito.get("expected_promotion_date"),
            "classification": bito.get("promotion_classification_30d"),
        },
        "executive_recommendation": (
            f"Monitor BITO for WATCH→OPEN_NEW (30d prob {bito.get('promotion_probability_30d')}%). "
            f"HD/XLK remain OPEN_NEW — proceed YELLOW recovery on timeline (~{int(days_to_yellow)} days). "
            f"GREEN depends on BITO promotion before day {int(days_to_green)}."
        ),
    }

    csv_rows: List[Dict[str, Any]] = []
    for c in candidate_forecasts:
        for h in FORECAST_HORIZONS:
            hf = (c.get("horizon_forecasts") or {}).get(str(h), {})
            csv_rows.append(
                {
                    "generated_at": ts,
                    "ticker": c["ticker"],
                    "current_status": c["current_status"],
                    "horizon_days": h,
                    "promotion_probability": hf.get("promotion_probability"),
                    "demotion_probability": hf.get("demotion_probability"),
                    "stability_probability": hf.get("stability_probability"),
                    "promotion_classification": hf.get("promotion_classification"),
                    "confidence_trend": c.get("confidence_trend"),
                    "edge_trend": c.get("edge_trend"),
                    "signal_strength": c.get("signal_strength"),
                    "recovery_impact": c.get("recovery_impact"),
                }
            )
    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False, encoding="utf-8")

    hist_row = {
        "generated_at": ts,
        "bito_promotion_prob_30d": bito.get("promotion_probability_30d"),
        "hd_status": hd.get("current_status"),
        "xlk_status": xlk.get("current_status"),
        "closest_promotion_ticker": (promotion_dependencies.get("closest_to_promotion") or {}).get(
            "ticker"
        ),
        "days_to_green": int(days_to_green),
    }
    prev_hist = _read_csv(OUT_HISTORY)
    combined = pd.concat([prev_hist, pd.DataFrame([hist_row])], ignore_index=True)
    combined.to_csv(OUT_HISTORY, index=False, encoding="utf-8")

    summary = {
        "candidates_forecast": len(candidate_forecasts),
        "open_new_count": len(
            [c for c in candidate_forecasts if c.get("current_status") == "OPEN_NEW"]
        ),
        "watch_count": len([c for c in candidate_forecasts if c.get("current_status") == "WATCH"]),
        "next_promotion": (promotion_dependencies.get("closest_to_promotion") or {}).get("ticker"),
        "bito_prob_30d": bito.get("promotion_probability_30d"),
        "most_critical": "BITO",
    }

    report = {
        "generated_at": ts,
        "executive_report": executive,
        "candidate_forecasts": candidate_forecasts,
        "promotion_probabilities": promotion_probabilities,
        "promotion_rankings": sorted(
            [
                {
                    "rank": i + 1,
                    "ticker": c["ticker"],
                    "promotion_probability_30d": c.get("promotion_probability_30d"),
                    "classification": c.get("promotion_classification_30d"),
                }
                for i, c in enumerate(candidate_forecasts)
            ],
            key=lambda x: _to_float(x.get("promotion_probability_30d"), 0),
            reverse=True,
        ),
        "promotion_dependencies": promotion_dependencies,
        "recovery_impact": recovery_impact,
        "timeline_impact": timeline_impact,
        "promotion_history_count": len(combined),
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "forecasting-only — no trades, orders, governance, lifecycle, or authorization modified",
        },
        "disclaimer": (
            "Promotion probability forecasts are planning intelligence only — does not modify "
            "governance, lifecycle, execution, broker, authorization, or allocation rules."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_CSV.name}, {OUT_HISTORY.name}")
    _log(f"candidates={len(candidate_forecasts)} bito_30d={bito.get('promotion_probability_30d')}%")
    return report


if __name__ == "__main__":
    run()
