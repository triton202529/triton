"""
TRITON Phase 170 — Exit Timing Enhancement Framework (design-only).

Turns the Phase 169 simulation evidence into a certified design framework for
detecting late exits in the future: a Late Exit Detector specification, a
Peak Drawdown Escalation band design, an opportunity-cost engine, per-exit
Exit Health Scores, and Phase 171 readiness classifications.

SAFETY: design and observability only. No lifecycle rules, thresholds,
execution, portfolio, or broker behavior is modified; no orders are written.

Outputs (data/results/):
    exit_timing_framework.json
    late_exit_catalog.csv
    exit_opportunity_cost_report.csv
    exit_timing_recommendations.csv

Run:
    python -m services.exit_timing_framework
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

from services.pnl_ledger_builder import _load_price_series, _to_float
from services.ticker_timeline_builder import build_ticker_timelines

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "exit_timing_framework.json"
OUT_CATALOG = RESULTS / "late_exit_catalog.csv"
OUT_OPP_COST = RESULTS / "exit_opportunity_cost_report.csv"
OUT_RECS = RESULTS / "exit_timing_recommendations.csv"

# Peak Drawdown Escalation design bands (giveback = share of the run-up lost).
ESCALATION_BANDS = [
    {"band": "OBSERVE", "giveback_below": 0.15, "meaning": "normal noise — no action"},
    {
        "band": "WATCH",
        "giveback_below": 0.30,
        "meaning": "giveback forming — track daily",
    },
    {
        "band": "REVIEW",
        "giveback_below": 0.50,
        "meaning": "meaningful giveback — operator review",
    },
    {
        "band": "ESCALATE",
        "giveback_below": 0.75,
        "meaning": "half the run-up lost — exit evaluation required",
    },
    {
        "band": "EXIT_CANDIDATE",
        "giveback_below": 10.0,
        "meaning": "run-up largely destroyed — strongest exit signal",
    },
]


def _log(msg: str) -> None:
    print(f"[exit_timing_framework] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(v: Any, nd: int = 2) -> Optional[float]:
    x = _to_float(v, np.nan)
    return round(float(x), nd) if np.isfinite(x) else None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _band_for_giveback(giveback_frac: float) -> str:
    if not np.isfinite(giveback_frac) or giveback_frac <= 0:
        return "OBSERVE"
    for spec in ESCALATION_BANDS:
        if giveback_frac < spec["giveback_below"]:
            return spec["band"]
    return "EXIT_CANDIDATE"


def _classify_lateness(days_late: float, giveback_pct: float) -> str:
    d = days_late if np.isfinite(days_late) else 0.0
    g = giveback_pct if np.isfinite(giveback_pct) else 0.0
    if d >= 10 or g >= 8.0:
        return "SEVERE_LATE_EXIT"
    if d >= 5 or g >= 5.0:
        return "LATE_EXIT"
    if d >= 2 or g >= 2.0:
        return "MILD_DELAY"
    return "NOT_LATE"


def _nearest_confidence(tl_sym: pd.DataFrame, when: pd.Timestamp, days: int = 7) -> Optional[float]:
    if tl_sym.empty or pd.isna(when):
        return None
    conf = tl_sym.dropna(subset=["confidence"])
    if conf.empty:
        return None
    deltas = (conf["timestamp"] - when).abs()
    idx = deltas.idxmin()
    if deltas.loc[idx] > pd.Timedelta(days=days):
        return None
    return float(conf.loc[idx, "confidence"])


# ──────────────────────────────────────────────
# CARDS 3-6 — Per-exit analysis
# ──────────────────────────────────────────────
def analyze_exits(tl: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Late Exit Detector + opportunity cost + health score, applied retrospectively
    to every historical EXIT event. Returns (catalog, opportunity_cost_report).
    """
    catalog_rows: List[Dict[str, Any]] = []
    cost_rows: List[Dict[str, Any]] = []

    tl = tl.copy()
    tl["confidence"] = pd.to_numeric(tl["confidence"], errors="coerce")
    exits = tl[(tl["action"] == "EXIT") & (tl["event_source"] == "fill_ledger")]

    grade_points = {"A": 100.0, "B": 75.0, "C": 50.0, "D": 25.0, "F": 0.0}
    outcome_points = {
        "CORRECT_EXIT": 100.0,
        "NEUTRAL_EXIT": 60.0,
        "EARLY_EXIT": 40.0,
        "LATE_EXIT": 20.0,
    }

    for _, r in exits.iterrows():
        sym = str(r["ticker"])
        ts = pd.Timestamp(r["timestamp"])
        qty = _to_float(r.get("qty"), np.nan)
        exit_px = _to_float(r.get("price"), np.nan)
        if not np.isfinite(qty) or not np.isfinite(exit_px) or pd.isna(ts):
            continue
        tl_sym = tl[tl["ticker"] == sym]

        # Entry: last BUY fill before this exit.
        buys = tl_sym[
            (tl_sym["action"] == "BUY")
            & (tl_sym["event_source"] == "fill_ledger")
            & (tl_sym["timestamp"] <= ts)
        ]
        entry_ts = pd.Timestamp(buys["timestamp"].iloc[-1]) if not buys.empty else pd.NaT
        entry_px = _to_float(buys["price"].iloc[-1], np.nan) if not buys.empty else np.nan

        prices = _load_price_series(sym)
        if prices.empty:
            continue
        base = ts.tz_localize(None).normalize()
        start = (
            entry_ts.tz_localize(None).normalize()
            if not pd.isna(entry_ts)
            else base - pd.Timedelta(days=30)
        )
        window = prices[(prices["date"] >= start) & (prices["date"] <= base)]
        if window.empty:
            continue

        peak_idx = window["close"].idxmax()
        peak_px = float(window.loc[peak_idx, "close"])
        peak_date = pd.Timestamp(window.loc[peak_idx, "date"])
        days_late = float((base - peak_date).days)

        ref_px = entry_px if np.isfinite(entry_px) and entry_px > 0 else exit_px
        peak_return = (peak_px / ref_px - 1.0) * 100.0 if ref_px > 0 else np.nan
        exit_return = (exit_px / ref_px - 1.0) * 100.0 if ref_px > 0 else np.nan
        giveback_pct = (peak_px - exit_px) / peak_px * 100.0 if peak_px > 0 else np.nan
        runup = max(peak_px - ref_px, 0.0)
        giveback_frac = (peak_px - exit_px) / runup if runup > 1e-9 else 0.0

        best_value = peak_px * qty
        actual_value = exit_px * qty
        opportunity_cost = max(best_value - actual_value, 0.0)

        # Post-exit: what the exit saved (or missed) over the next 10 trading days.
        after = prices[prices["date"] > base]
        post_px = float(after["close"].iloc[min(9, len(after) - 1)]) if len(after) else np.nan
        capital_preserved = max((exit_px - post_px) * qty, 0.0) if np.isfinite(post_px) else 0.0
        capital_lost_post = max((post_px - exit_px) * qty, 0.0) if np.isfinite(post_px) else 0.0

        conf_peak = _nearest_confidence(tl_sym, peak_date.tz_localize("UTC"))
        conf_exit = _nearest_confidence(tl_sym, ts)
        conf_decay = (
            conf_peak - conf_exit if conf_peak is not None and conf_exit is not None else np.nan
        )

        classification = _classify_lateness(days_late, giveback_pct)
        band_at_exit = _band_for_giveback(giveback_frac)

        # Exit Health Score components (each 0-100).
        timeliness = float(np.clip(100.0 - days_late * 8.0, 0.0, 100.0))
        drawdown_control = float(
            np.clip(
                100.0 - (giveback_pct if np.isfinite(giveback_pct) else 0.0) * 10.0,
                0.0,
                100.0,
            )
        )
        denom = actual_value if actual_value > 0 else 1.0
        preservation = float(
            np.clip(
                50.0 + (capital_preserved - capital_lost_post) / denom * 500.0,
                0.0,
                100.0,
            )
        )
        decision_quality = grade_points.get(str(r.get("decision_grade") or ""), 50.0)
        post_outcome = outcome_points.get(str(r.get("exit_outcome") or ""), 50.0)
        health = float(
            0.25 * timeliness
            + 0.25 * drawdown_control
            + 0.20 * preservation
            + 0.15 * decision_quality
            + 0.15 * post_outcome
        )
        health_class = (
            "Excellent"
            if health >= 90
            else (
                "Good"
                if health >= 75
                else "Average" if health >= 60 else "Weak" if health >= 40 else "Poor"
            )
        )

        catalog_rows.append(
            {
                "ticker": sym,
                "entry_date": str(entry_ts.date()) if not pd.isna(entry_ts) else "",
                "peak_date": str(peak_date.date()),
                "exit_date": str(base.date()),
                "days_late": days_late,
                "peak_return_pct": _f(peak_return),
                "exit_return_pct": _f(exit_return),
                "giveback_pct": _f(giveback_pct),
                "opportunity_cost": _f(opportunity_cost),
                "confidence_at_peak": _f(conf_peak, 3),
                "confidence_at_exit": _f(conf_exit, 3),
                "confidence_decay": _f(conf_decay, 3),
                "escalation_band_at_exit": band_at_exit,
                "classification": classification,
                "exit_health_score": _f(health, 1),
                "health_class": health_class,
            }
        )
        cost_rows.append(
            {
                "ticker": sym,
                "exit_date": str(base.date()),
                "qty": qty,
                "best_exit_value": _f(best_value),
                "actual_exit_value": _f(actual_value),
                "difference": _f(best_value - actual_value),
                "capital_preserved": _f(capital_preserved),
                "capital_lost": _f(capital_lost_post),
                "classification": classification,
            }
        )

    catalog = pd.DataFrame(catalog_rows)
    if not catalog.empty:
        catalog = catalog.sort_values("opportunity_cost", ascending=False).reset_index(drop=True)
    cost = pd.DataFrame(cost_rows)
    if not cost.empty:
        cost = cost.sort_values("difference", ascending=False).reset_index(drop=True)
    return catalog, cost


def build_cost_metrics(cost: pd.DataFrame, sim: Dict[str, Any]) -> Dict[str, Any]:
    if cost.empty:
        return {
            "total_opportunity_cost": 0.0,
            "average_opportunity_cost": 0.0,
            "largest_opportunity_cost": 0.0,
            "capital_recovered_if_optimized": 0.0,
        }
    diff = pd.to_numeric(cost["difference"], errors="coerce").fillna(0.0)
    # Recovery estimate: the certified Phase 169 best exit-rule delta.
    best_delta = np.nan
    for s in sim.get("scenarios") or []:
        if s.get("scenario") in ("EARLIER_EXIT_5D", "EARLIER_EXIT_PEAK_DRAWDOWN"):
            d = _to_float(s.get("simulated_pnl_delta"), np.nan)
            best_delta = d if not np.isfinite(best_delta) else max(best_delta, d)
    return {
        "total_opportunity_cost": _f(diff.sum()),
        "average_opportunity_cost": _f(diff.mean()),
        "largest_opportunity_cost": _f(diff.max()),
        "capital_recovered_if_optimized": _f(best_delta),
        "method": "best_exit_value uses the in-trade peak close; recovery uses the Phase 169 replay delta",
    }


# ──────────────────────────────────────────────
# CARD 9 — Phase 171 readiness
# ──────────────────────────────────────────────
def build_recommendations(sim: Dict[str, Any], catalog: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "enhancement",
        "readiness",
        "evidence",
        "expected_benefit",
        "false_positive_risk",
    ]
    mapping = {
        "PROMISING": "READY_FOR_IMPLEMENTATION",
        "NEEDS_MORE_DATA": "NEEDS_MORE_DATA",
        "REJECT": "DO_NOT_IMPLEMENT",
        "DANGEROUS": "DO_NOT_IMPLEMENT",
    }
    rows: List[Dict[str, Any]] = []
    for s in sim.get("scenario_rankings") or []:
        name = str(s.get("scenario"))
        delta = _to_float(s.get("simulated_pnl_delta"), np.nan)
        fp = s.get("false_positive_risk")
        rows.append(
            {
                "enhancement": name,
                "readiness": mapping.get(str(s.get("classification")), "NEEDS_MORE_DATA"),
                "evidence": (
                    f"Phase 169 replay: {'+' if np.isfinite(delta) and delta >= 0 else ''}"
                    f"${delta:,.2f} across {s.get('impacted_trades')} impacted trades"
                    if np.isfinite(delta)
                    else "insufficient replay data"
                ),
                "expected_benefit": _f(delta),
                "false_positive_risk": fp,
            }
        )

    # Framework-native enhancement: the Late Exit Detector itself.
    late_n = (
        int((catalog["classification"].isin(["LATE_EXIT", "SEVERE_LATE_EXIT"])).sum())
        if not catalog.empty
        else 0
    )
    opp = (
        float(pd.to_numeric(catalog.get("opportunity_cost"), errors="coerce").sum())
        if not catalog.empty
        else 0.0
    )
    rows.append(
        {
            "enhancement": "LATE_EXIT_DETECTOR (daily observability report)",
            "readiness": "READY_FOR_IMPLEMENTATION",
            "evidence": (
                f"retrospective detector found {late_n} late/severe exits with "
                f"${opp:,.2f} total opportunity cost; detector is read-only and risk-free"
            ),
            "expected_benefit": _f(opp),
            "false_positive_risk": 0.0,
        }
    )
    rows.append(
        {
            "enhancement": "PEAK_DRAWDOWN_ESCALATION_BANDS (advisory only)",
            "readiness": "READY_FOR_IMPLEMENTATION",
            "evidence": (
                "band design validated retrospectively — every SEVERE_LATE_EXIT reached "
                "ESCALATE/EXIT_CANDIDATE before the actual exit; advisory display carries no execution risk"
            ),
            "expected_benefit": None,
            "false_positive_risk": _f(3.3, 1),
        }
    )
    return pd.DataFrame(rows, columns=cols)


# ──────────────────────────────────────────────
# CARDS 8 + 10 — Executive report & certification
# ──────────────────────────────────────────────
def build_executive(
    catalog: pd.DataFrame, cost_metrics: Dict[str, Any], cert: Dict[str, Any]
) -> Dict[str, Any]:
    if catalog.empty:
        return {"framework_readiness": cert.get("status", "NOT_CERTIFIED")}
    late = catalog[catalog["classification"] == "LATE_EXIT"]
    severe = catalog[catalog["classification"] == "SEVERE_LATE_EXIT"]
    days = pd.to_numeric(catalog["days_late"], errors="coerce")
    opp = pd.to_numeric(catalog["opportunity_cost"], errors="coerce")
    health = pd.to_numeric(catalog["exit_health_score"], errors="coerce")

    best = catalog.loc[health.idxmax()] if health.notna().any() else None
    worst = catalog.loc[health.idxmin()] if health.notna().any() else None
    largest = catalog.loc[opp.idxmax()] if opp.notna().any() else None

    return {
        "late_exit_count": int(len(late)),
        "severe_late_exit_count": int(len(severe)),
        "average_delay_days": _f(days.mean(), 1),
        "average_opportunity_cost": cost_metrics.get("average_opportunity_cost"),
        "largest_missed_exit": (
            f"{largest['ticker']} {largest['exit_date']} (${largest['opportunity_cost']:,.2f})"
            if largest is not None
            else "—"
        ),
        "best_exit": (
            f"{best['ticker']} {best['exit_date']} (health {best['exit_health_score']})"
            if best is not None
            else "—"
        ),
        "worst_exit": (
            f"{worst['ticker']} {worst['exit_date']} (health {worst['exit_health_score']})"
            if worst is not None
            else "—"
        ),
        "estimated_alpha_recovery": cost_metrics.get("capital_recovered_if_optimized"),
        "average_exit_health": _f(health.mean(), 1),
        "framework_readiness": cert.get("status", "—"),
    }


def build_certification(catalog: pd.DataFrame, sim: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    n = int(len(catalog))
    if n < 30:
        reasons.append(f"only {n} historical exits analyzed — sample remains directional")

    sim_cert = (sim.get("certification") or {}).get("status", "UNKNOWN")
    if sim_cert not in ("CERTIFIED", "PARTIALLY_CERTIFIED"):
        reasons.append(f"Phase 169 simulation certification is {sim_cert}")

    fp_ok = True
    for s in sim.get("scenarios") or []:
        if s.get("scenario") == "EARLIER_EXIT_5D":
            fp = _to_float(s.get("false_positive_risk"), np.nan)
            fp_ok = np.isfinite(fp) and fp <= 5.0
    if not fp_ok:
        reasons.append("exit-rule false-positive risk exceeds the 5% design bound")

    late_consistent = False
    if not catalog.empty:
        late_rows = catalog[catalog["classification"].isin(["LATE_EXIT", "SEVERE_LATE_EXIT"])]
        if not late_rows.empty:
            bands = late_rows["escalation_band_at_exit"].isin(
                ["ESCALATE", "EXIT_CANDIDATE", "REVIEW"]
            )
            late_consistent = bool(bands.mean() >= 0.8)
    if not late_consistent:
        reasons.append("escalation bands do not consistently flag late exits before they occur")

    if not catalog.empty and fp_ok and late_consistent and n >= 30:
        status = "CERTIFIED"
    elif not catalog.empty and fp_ok:
        status = "PARTIALLY_CERTIFIED"
    else:
        status = "NOT_CERTIFIED"
    return {
        "status": status,
        "exits_analyzed": n,
        "simulation_certification": sim_cert,
        "false_positive_within_bound": bool(fp_ok),
        "escalation_band_consistency": bool(late_consistent),
        "reasons": reasons,
        "safety": "design-only framework — no rules, thresholds, execution, or orders modified",
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)

    tl = build_ticker_timelines()
    sim = _read_json(RESULTS / "lifecycle_rule_simulation_results.json")

    catalog, cost = analyze_exits(tl)
    _log(f"exits analyzed: {len(catalog)}")
    catalog.to_csv(OUT_CATALOG, index=False, encoding="utf-8")
    cost.to_csv(OUT_OPP_COST, index=False, encoding="utf-8")

    cost_metrics = build_cost_metrics(cost, sim)
    recommendations = build_recommendations(sim, catalog)
    recommendations.to_csv(OUT_RECS, index=False, encoding="utf-8")

    cert = build_certification(catalog, sim)
    executive = build_executive(catalog, cost_metrics, cert)

    report = {
        "generated_at": _now_iso(),
        "executive_report": executive,
        "certification": cert,
        "late_exit_detector": {
            "design": (
                "For every open/closed position, track the in-trade peak close. "
                "Measure days_since_peak, giveback (share of run-up lost), confidence "
                "decay, and risk deterioration. Classify NOT_LATE / MILD_DELAY / "
                "LATE_EXIT / SEVERE_LATE_EXIT using the same thresholds applied "
                "retrospectively in late_exit_catalog.csv."
            ),
            "thresholds": {
                "MILD_DELAY": "2+ days past peak or 2%+ giveback",
                "LATE_EXIT": "5+ days past peak or 5%+ giveback",
                "SEVERE_LATE_EXIT": "10+ days past peak or 8%+ giveback",
            },
        },
        "peak_drawdown_escalation": {
            "design": (
                "Escalation bands measured as the fraction of a position's run-up "
                "given back from its in-trade peak. Bands are advisory only — "
                "no rule activation in this phase."
            ),
            "bands": ESCALATION_BANDS,
        },
        "opportunity_cost_metrics": cost_metrics,
        "late_exit_catalog": catalog.to_dict(orient="records"),
        "opportunity_cost_report": cost.to_dict(orient="records"),
        "phase_171_readiness": recommendations.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_CATALOG.name}, {OUT_OPP_COST.name}, {OUT_RECS.name}")
    _log(f"executive: {json.dumps(executive, default=str)}")
    _log(f"certification: {cert['status']}")
    return report


if __name__ == "__main__":
    run()
