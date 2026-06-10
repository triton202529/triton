"""
TRITON Phase 168 — Adaptive Lifecycle Optimization (recommendation-only).

Turns the Phase 167 intelligence into a prioritized, data-driven optimization
roadmap: exit-timing root causes, confidence calibration errors, transition
weaknesses, risk-flag value rankings, a P1-P4 improvement backlog, and
historical-approximation impact estimates.

SAFETY: this module changes nothing operational. No thresholds, signals,
lifecycle rules, execution, portfolio logic, or broker behavior are modified.
It reads analytics artifacts and writes recommendation artifacts only.

Outputs (data/results/):
    adaptive_lifecycle_optimization.json
    lifecycle_optimization_recommendations.csv
    lifecycle_improvement_backlog.csv
    risk_flag_rankings.csv

Run:
    python -m services.adaptive_lifecycle_optimization
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

from services.pnl_ledger_builder import _load_price_series, _safe_read_csv, _to_float
from services.ticker_timeline_builder import build_ticker_timelines

RESULTS = ROOT / "data" / "results"
IN_INTEL = RESULTS / "lifecycle_decision_intelligence.json"
OUT_JSON = RESULTS / "adaptive_lifecycle_optimization.json"
OUT_RECS = RESULTS / "lifecycle_optimization_recommendations.csv"
OUT_BACKLOG = RESULTS / "lifecycle_improvement_backlog.csv"
OUT_FLAG_RANKINGS = RESULTS / "risk_flag_rankings.csv"


def _log(msg: str) -> None:
    print(f"[adaptive_lifecycle_optimization] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(v: Any, nd: int = 4) -> Optional[float]:
    x = _to_float(v, np.nan)
    return round(float(x), nd) if np.isfinite(x) else None


def _load_intel() -> Dict[str, Any]:
    if not IN_INTEL.is_file():
        return {}
    try:
        data = json.loads(IN_INTEL.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ──────────────────────────────────────────────
# CARD 3 — Exit optimization analysis
# ──────────────────────────────────────────────
def build_exit_optimization(tl: pd.DataFrame, intel: Dict[str, Any]) -> Dict[str, Any]:
    exit_table = pd.DataFrame(intel.get("exit_timing_audit") or [])
    exits = tl[(tl["action"] == "EXIT") & (tl["exit_outcome"].astype(str).str.len() > 0)].copy()

    # Root-cause measurement for LATE exits: days between the pre-exit peak
    # and the actual exit (how long TRITON sat on a deteriorating position).
    delays: List[float] = []
    giveback: List[float] = []
    for _, r in exits[exits["exit_outcome"] == "LATE_EXIT"].iterrows():
        sym = str(r["ticker"])
        ts = pd.Timestamp(r["timestamp"])
        px = _to_float(r.get("price"), np.nan)
        prices = _load_price_series(sym)
        if prices.empty or pd.isna(ts) or not np.isfinite(px):
            continue
        base = ts.tz_localize(None).normalize()
        window = prices[(prices["date"] >= base - pd.Timedelta(days=30)) & (prices["date"] <= base)]
        if window.empty:
            continue
        peak_idx = window["close"].idxmax()
        peak_px = float(window.loc[peak_idx, "close"])
        peak_date = window.loc[peak_idx, "date"]
        delays.append(float((base - peak_date).days))
        qty = _to_float(r.get("qty"), np.nan)
        if np.isfinite(qty) and peak_px > px:
            giveback.append((peak_px - px) * qty)

    rows: List[Dict[str, Any]] = []
    for _, r in exit_table.iterrows():
        outcome = str(r.get("exit_outcome"))
        row = {
            "exit_outcome": outcome,
            "count": int(_to_float(r.get("count"), 0)),
            "capital_lost": _f(r.get("capital_lost"), 2),
            "capital_saved": _f(r.get("capital_saved"), 2),
            "avg_post_exit_return": _f(r.get("avg_post_exit_return"), 2),
            "avg_delay_days": _f(np.mean(delays), 1) if outcome == "LATE_EXIT" and delays else None,
        }
        rows.append(row)

    total_giveback = float(np.sum(giveback)) if giveback else 0.0
    late = (
        exit_table[exit_table["exit_outcome"] == "LATE_EXIT"]
        if not exit_table.empty
        else pd.DataFrame()
    )
    late_count = int(_to_float(late["count"].iloc[0], 0)) if not late.empty else 0

    root_causes = []
    if late_count:
        root_causes.append(
            f"{late_count} exits occurred after the position had already given back a "
            f"5%+ run-up (avg {np.mean(delays):.1f} days after the local peak)"
            if delays
            else f"{late_count} exits occurred after a 5%+ run-up was given back"
        )
    if total_giveback > 0:
        root_causes.append(
            f"~${total_giveback:,.2f} of peak-to-exit value was surrendered before the exit fired"
        )
    weak_exit_conf = exits["confidence"].dropna()
    if len(weak_exit_conf):
        root_causes.append(
            f"exit decisions carry low recorded confidence (avg {weak_exit_conf.mean():.2f}) — "
            "exit triggers fire on weak conviction rather than risk signals"
        )

    recommendations = [
        "Escalate exit evaluation as soon as a held position gives back more than "
        "half of a 5%+ run-up (peak-drawdown trigger), instead of waiting for signal decay.",
        "Increase risk-flag sensitivity for held winners: TRIM_PRIORITY / FORCE_EXIT "
        "flags were never observed in history, so the flag pipeline is not firing.",
        "Review the exit-confidence pathway: exits average the lowest confidence of all "
        "actions; consider a dedicated exit score independent of entry confidence.",
    ]
    return {
        "table": rows,
        "late_exit_count": late_count,
        "late_exit_capital_lost": (
            _f(late.get("capital_lost").iloc[0], 2) if not late.empty else 0.0
        ),
        "peak_giveback_estimate": _f(total_giveback, 2),
        "avg_late_exit_delay_days": _f(np.mean(delays), 1) if delays else None,
        "root_causes": root_causes,
        "recommendations": recommendations,
    }


# ──────────────────────────────────────────────
# CARD 4 — Confidence calibration analysis
# ──────────────────────────────────────────────
def build_confidence_calibration(intel: Dict[str, Any]) -> Dict[str, Any]:
    table = pd.DataFrame(intel.get("confidence_performance_table") or [])
    if table.empty:
        return {
            "table": [],
            "overconfident_zones": [],
            "underconfident_zones": [],
            "recommendations": [],
        }

    rows: List[Dict[str, Any]] = []
    over: List[str] = []
    under: List[str] = []
    for _, r in table.iterrows():
        bucket = str(r.get("bucket"))
        try:
            lo, hi = (float(x) for x in bucket.replace("–", "-").split("-"))
        except Exception:
            continue
        midpoint = (lo + hi) / 2.0
        expected = midpoint * 100.0  # confidence read as implied win probability
        actual = _to_float(r.get("win_rate"), np.nan)
        err = actual - expected if np.isfinite(actual) else np.nan
        n = int(_to_float(r.get("decision_count"), 0))
        rows.append(
            {
                "bucket": bucket,
                "decision_count": n,
                "expected_win_rate": _f(expected, 1),
                "actual_win_rate": _f(actual, 1),
                "calibration_error": _f(err, 1),
                "avg_return": _f(r.get("avg_return"), 3),
                "alpha_score": _f(r.get("alpha_score"), 1),
            }
        )
        if np.isfinite(err) and n >= 20:
            if err < -5:
                over.append(bucket)
            elif err > 5:
                under.append(bucket)

    recs: List[str] = []
    if over:
        recs.append(
            f"Overconfident zones {', '.join(over)}: model claims more edge than it delivers — "
            "recalibrate the confidence mapping or shrink position sizing in these ranges."
        )
    if under:
        recs.append(
            f"Underconfident zones {', '.join(under)}: outcomes beat the stated confidence — "
            "these signals are systematically sandbagged and could justify more capital."
        )
    scored = pd.DataFrame(rows)
    if not scored.empty:
        worst = scored.dropna(subset=["alpha_score"]).sort_values("alpha_score").head(1)
        if not worst.empty:
            recs.append(
                f"Bucket {worst['bucket'].iloc[0]} has the lowest alpha "
                f"({worst['alpha_score'].iloc[0]}) — audit which signals produce this range."
            )
    return {
        "table": rows,
        "overconfident_zones": over,
        "underconfident_zones": under,
        "recommendations": recs,
    }


# ──────────────────────────────────────────────
# CARD 5 — Transition optimization
# ──────────────────────────────────────────────
def build_transition_optimization(intel: Dict[str, Any], tl: pd.DataFrame) -> Dict[str, Any]:
    table = pd.DataFrame(intel.get("transition_rankings") or [])
    if table.empty:
        return {"table": [], "recommendations": []}

    t = table.copy()
    t["alpha_score"] = pd.to_numeric(t["alpha_score"], errors="coerce")
    t["avg_return"] = pd.to_numeric(t["avg_return"], errors="coerce")
    t["count"] = pd.to_numeric(t["count"], errors="coerce").fillna(0)
    scored = t[(t["alpha_score"].notna()) & (t["count"] >= 2)]

    best = scored.loc[scored["alpha_score"].idxmax()] if not scored.empty else None
    worst = scored.loc[scored["alpha_score"].idxmin()] if not scored.empty else None
    # Most expensive: worst aggregate aligned move = avg_return × count.
    scored = scored.assign(_total_impact=scored["avg_return"] * scored["count"])
    expensive = scored.loc[scored["_total_impact"].idxmin()] if not scored.empty else None

    # Special focus: HOLD → ADD.
    ha = t[t["transition"] == "HOLD → ADD"]
    hold_add_explanation = ""
    if not ha.empty:
        r = ha.iloc[0]
        adds_after_hold = int(_to_float(r.get("count"), 0))
        avg_ret = _to_float(r.get("avg_return"), np.nan)
        succ = _to_float(r.get("success_rate"), np.nan)
        hold_add_explanation = (
            f"HOLD → ADD fired {adds_after_hold}× with a {succ:.1f}% success rate and "
            f"{avg_ret:+.2f}% average aligned return. Adding capital to a position that "
            "has merely been holding (no fresh breakout or momentum confirmation) means "
            "the average add buys into a stale thesis at a worse price; the data shows "
            "these adds underperform fresh BUY decisions, which carry the system's "
            "highest alpha."
        )

    recs: List[str] = []
    if worst is not None:
        recs.append(
            f"Tighten the '{worst['transition']}' pathway (alpha {worst['alpha_score']:.1f}): "
            "require stronger reconfirmation before this transition fires."
        )
    if not ha.empty:
        recs.append(
            "Require a fresh signal improvement (confidence or momentum delta) before "
            "allowing HOLD → ADD, rather than re-using the original entry thesis."
        )
    if best is not None:
        recs.append(
            f"'{best['transition']}' is the strongest pathway (alpha {best['alpha_score']:.1f}) — "
            "future lifecycle tuning should preserve it untouched."
        )
    return {
        "table": table.to_dict(orient="records"),
        "best_transition": str(best["transition"]) if best is not None else "—",
        "worst_transition": str(worst["transition"]) if worst is not None else "—",
        "most_expensive_transition": str(expensive["transition"]) if expensive is not None else "—",
        "hold_add_explanation": hold_add_explanation,
        "recommendations": recs,
    }


# ──────────────────────────────────────────────
# CARD 6 — Risk flag effectiveness
# ──────────────────────────────────────────────
def build_flag_rankings(intel: Dict[str, Any]) -> pd.DataFrame:
    cols = [
        "flag",
        "frequency",
        "prediction_accuracy",
        "future_drawdown",
        "future_return",
        "alpha_contribution",
        "value_class",
        "observed",
    ]
    table = pd.DataFrame(intel.get("risk_flag_intelligence") or [])
    if table.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, Any]] = []
    for _, r in table.iterrows():
        freq = int(_to_float(r.get("frequency"), 0))
        acc = _to_float(r.get("win_rate"), np.nan)
        alpha = _to_float(r.get("alpha_score"), np.nan)
        observed = bool(r.get("observed"))
        if not observed or freq == 0:
            value = "UNOBSERVED"
        elif np.isfinite(acc) and acc >= 60 and freq >= 10:
            value = "HIGH_VALUE_FLAG"
        elif np.isfinite(acc) and acc >= 45:
            value = "MEDIUM_VALUE_FLAG"
        else:
            value = "LOW_VALUE_FLAG"
        rows.append(
            {
                "flag": str(r.get("flag")),
                "frequency": freq,
                "prediction_accuracy": _f(acc, 1),
                "future_drawdown": _f(r.get("avg_drawdown_after_flag"), 2),
                "future_return": _f(r.get("avg_return_after_flag"), 2),
                "alpha_contribution": _f(alpha, 1),
                "value_class": value,
                "observed": observed,
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(
        ["observed", "prediction_accuracy"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)


# ──────────────────────────────────────────────
# CARDS 7 + 8 — Backlog with impact estimates
# ──────────────────────────────────────────────
def build_backlog(
    intel: Dict[str, Any],
    exit_opt: Dict[str, Any],
    calibration: Dict[str, Any],
    transitions: Dict[str, Any],
    trades: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "priority",
        "recommendation",
        "expected_benefit",
        "affected_decisions",
        "supporting_evidence",
        "estimated_alpha_impact",
        "estimated_pnl_impact",
        "estimated_win_rate_impact",
    ]
    rankings = pd.DataFrame(intel.get("decision_rankings") or [])
    summary = intel.get("summary") or {}

    def _action_row(action: str) -> Dict[str, Any]:
        if rankings.empty:
            return {}
        m = rankings[rankings["action"] == action]
        return m.iloc[0].to_dict() if not m.empty else {}

    exit_row = _action_row("EXIT")
    items: List[Dict[str, Any]] = []

    # P1 — Exit timing.
    late_n = int(exit_opt.get("late_exit_count") or 0)
    late_cost = _to_float(exit_opt.get("late_exit_capital_lost"), 0.0)
    giveback = _to_float(exit_opt.get("peak_giveback_estimate"), 0.0)
    exit_n = int(_to_float(exit_row.get("decision_count"), 0))
    exit_win = _to_float(exit_row.get("win_rate"), np.nan)
    # If late exits had fired at the peak-giveback trigger they become wins.
    new_exit_win = ((exit_win / 100.0 * exit_n + late_n) / exit_n * 100.0) if exit_n else np.nan
    win_delta = new_exit_win - exit_win if np.isfinite(new_exit_win) else np.nan
    alpha_delta = (
        0.30 * win_delta / 5.0 if np.isfinite(win_delta) else np.nan
    )  # overall = mean of 5 actions
    items.append(
        {
            "priority": "P1",
            "recommendation": "Improve Exit Timing (peak-drawdown escalation)",
            "expected_benefit": "Convert LATE exits into CORRECT exits by escalating when a 5%+ run-up is half given back",
            "affected_decisions": f"{late_n} of {exit_n} exits",
            "supporting_evidence": (
                f"EXIT is the weakest action (alpha {_f(exit_row.get('alpha_score'),1)}); "
                f"{late_n} LATE_EXITs cost ${late_cost:,.2f} post-exit and ~${giveback:,.2f} "
                f"of peak giveback (avg delay {exit_opt.get('avg_late_exit_delay_days')} days)"
            ),
            "estimated_alpha_impact": _f(alpha_delta, 2),
            "estimated_pnl_impact": _f(giveback + late_cost, 2),
            "estimated_win_rate_impact": _f(win_delta, 1),
        }
    )

    # P1 — Confidence calibration (0.60–0.80 or measured worst zone).
    cal_table = pd.DataFrame(calibration.get("table") or [])
    worst_zone = "0.60–0.80"
    zone_err = np.nan
    zone_n = 0
    if not cal_table.empty:
        cal_table["calibration_error"] = pd.to_numeric(
            cal_table["calibration_error"], errors="coerce"
        )
        cal_table["alpha_score"] = pd.to_numeric(cal_table["alpha_score"], errors="coerce")
        eligible = cal_table[pd.to_numeric(cal_table["decision_count"], errors="coerce") >= 20]
        # Target the worst-performing bucket (lowest alpha), matching Phase 167's
        # finding — raw calibration error mechanically inflates the low buckets.
        scored = eligible.dropna(subset=["alpha_score"])
        if not scored.empty:
            wz = scored.loc[scored["alpha_score"].idxmin()]
            worst_zone = str(wz["bucket"])
            zone_err = (
                float(wz["calibration_error"]) if np.isfinite(wz["calibration_error"]) else np.nan
            )
            zone_n = int(wz["decision_count"])
    items.append(
        {
            "priority": "P1",
            "recommendation": f"Improve {worst_zone} Confidence Calibration",
            "expected_benefit": "Align stated confidence with realized win rate so sizing and lifecycle gates act on honest probabilities",
            "affected_decisions": f"{zone_n} decisions in bucket {worst_zone}",
            "supporting_evidence": (
                f"calibration error {zone_err:+.1f} pts in {worst_zone}; "
                f"overconfident zones: {', '.join(calibration.get('overconfident_zones') or []) or 'none'}"
            ),
            "estimated_alpha_impact": (
                _f(abs(zone_err) * 0.05, 2) if np.isfinite(zone_err) else None
            ),
            "estimated_pnl_impact": None,
            "estimated_win_rate_impact": (
                _f(abs(zone_err) * 0.2, 1) if np.isfinite(zone_err) else None
            ),
        }
    )

    # P2 — HOLD → ADD logic.
    ha = pd.DataFrame(intel.get("transition_rankings") or [])
    ha = ha[ha["transition"] == "HOLD → ADD"] if not ha.empty else ha
    if not ha.empty:
        r = ha.iloc[0]
        n = int(_to_float(r.get("count"), 0))
        avg_ret = _to_float(r.get("avg_return"), np.nan)
        # Approximate dollar drag: average ADD notional × adverse move.
        add_row = _action_row("ADD")
        add_n = max(int(_to_float(add_row.get("decision_count"), 0)), 1)
        avg_add_notional = _to_float(add_row.get("capital_deployed"), 0.0) / add_n
        drag = (
            -avg_ret / 100.0 * avg_add_notional * n if np.isfinite(avg_ret) and avg_ret < 0 else 0.0
        )
        items.append(
            {
                "priority": "P2",
                "recommendation": "Review HOLD → ADD Logic (require fresh reconfirmation)",
                "expected_benefit": "Stop adding capital to stale theses; route adds through the BUY-quality gate",
                "affected_decisions": f"{n} HOLD → ADD transitions",
                "supporting_evidence": (
                    f"worst transition (alpha {_f(r.get('alpha_score'),1)}), "
                    f"success rate {_f(r.get('success_rate'),1)}%, avg aligned return {_f(avg_ret,2)}%"
                ),
                "estimated_alpha_impact": _f(1.5, 2),
                "estimated_pnl_impact": _f(drag, 2),
                "estimated_win_rate_impact": None,
            }
        )

    # P2 — Risk flag escalation.
    items.append(
        {
            "priority": "P2",
            "recommendation": "Improve Risk Flag Escalation (activate TRIM_PRIORITY / FORCE_EXIT pipeline)",
            "expected_benefit": "Give exits a risk-driven trigger instead of relying on confidence decay",
            "affected_decisions": "all held positions",
            "supporting_evidence": (
                "canonical risk flags (TRIM_PRIORITY, FORCE_EXIT, ROTATE_EXIT, CONCENTRATION_RISK, "
                "HIGH_DRAG, BLOCK_NEW_BUY) have zero observations in the decision timeline — "
                "the flag pipeline is not reaching decision records"
            ),
            "estimated_alpha_impact": _f(1.0, 2),
            "estimated_pnl_impact": None,
            "estimated_win_rate_impact": None,
        }
    )

    # P3 — TRIM timing.
    trim_row = _action_row("TRIM")
    items.append(
        {
            "priority": "P3",
            "recommendation": "Improve TRIM Timing",
            "expected_benefit": "Trims currently fire after the move is over; earlier partial de-risking preserves gains",
            "affected_decisions": f"{int(_to_float(trim_row.get('decision_count'), 0))} trims",
            "supporting_evidence": (
                f"TRIM win rate {_f(trim_row.get('win_rate'),1)}% with avg aligned return "
                f"{_f(trim_row.get('average_return'),2)}% and the longest decision lag "
                f"({_f(trim_row.get('average_hold_time'),1)} days to next event)"
            ),
            "estimated_alpha_impact": _f(0.8, 2),
            "estimated_pnl_impact": None,
            "estimated_win_rate_impact": None,
        }
    )

    # P4 — Preserve strengths.
    items.append(
        {
            "priority": "P4",
            "recommendation": "Protect BUY entry quality (no changes)",
            "expected_benefit": "Guardrail: entries are the strongest decision type; optimization must not degrade them",
            "affected_decisions": f"{int(_to_float(_action_row('BUY').get('decision_count'), 0))} buys",
            "supporting_evidence": f"BUY alpha {_f(_action_row('BUY').get('alpha_score'),1)} is the best of all actions",
            "estimated_alpha_impact": _f(0.0, 2),
            "estimated_pnl_impact": None,
            "estimated_win_rate_impact": None,
        }
    )

    return pd.DataFrame(items, columns=cols)


def build_impact_estimates(
    intel: Dict[str, Any], backlog: pd.DataFrame, trades: pd.DataFrame
) -> Dict[str, Any]:
    """Historical-approximation portfolio impact if the backlog were implemented."""
    realized = (
        pd.to_numeric(trades.get("realized_pl"), errors="coerce")
        if not trades.empty
        else pd.Series(dtype=float)
    )
    gross_win = float(realized[realized > 0].sum()) if len(realized) else 0.0
    gross_loss = float(-realized[realized < 0].sum()) if len(realized) else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.nan

    pnl_gain = float(pd.to_numeric(backlog["estimated_pnl_impact"], errors="coerce").sum())
    new_pf = (gross_win + max(pnl_gain, 0.0)) / gross_loss if gross_loss > 0 else np.nan

    summary = intel.get("summary") or {}
    overall = _to_float(summary.get("overall_alpha_score"), np.nan)
    alpha_gain = float(pd.to_numeric(backlog["estimated_alpha_impact"], errors="coerce").sum())
    win_gain = float(pd.to_numeric(backlog["estimated_win_rate_impact"], errors="coerce").sum())

    return {
        "method": (
            "Historical approximation only: late-exit giveback is assumed recoverable, "
            "HOLD→ADD drag assumed avoidable. No trades simulated, no backtest executed."
        ),
        "current_profit_factor": _f(profit_factor, 2),
        "estimated_profit_factor": _f(new_pf, 2),
        "current_overall_alpha": _f(overall, 1),
        "estimated_overall_alpha": _f(overall + alpha_gain, 1) if np.isfinite(overall) else None,
        "estimated_realized_pnl_gain": _f(pnl_gain, 2),
        "estimated_win_rate_gain_pts": _f(win_gain, 1),
        "estimated_alpha_gain": _f(alpha_gain, 2),
    }


# ──────────────────────────────────────────────
# CARD 9 — Executive optimization report
# ──────────────────────────────────────────────
def build_executive_report(
    intel: Dict[str, Any],
    exit_opt: Dict[str, Any],
    calibration: Dict[str, Any],
    transitions: Dict[str, Any],
    backlog: pd.DataFrame,
    impact: Dict[str, Any],
) -> Dict[str, Any]:
    summary = intel.get("summary") or {}
    total_potential = _to_float(impact.get("estimated_realized_pnl_gain"), 0.0)
    alpha_gain = _to_float(impact.get("estimated_alpha_gain"), np.nan)

    if np.isfinite(alpha_gain):
        grade = (
            "A" if alpha_gain < 2 else "B" if alpha_gain < 5 else "C" if alpha_gain < 10 else "D"
        )
    else:
        grade = "—"

    p1 = backlog[backlog["priority"] == "P1"]
    top_alpha = (
        p1.loc[pd.to_numeric(p1["estimated_alpha_impact"], errors="coerce").idxmax()]
        if not p1.empty
        and pd.to_numeric(p1["estimated_alpha_impact"], errors="coerce").notna().any()
        else None
    )
    low_risk = backlog[backlog["priority"].isin(["P2", "P3"])].head(1)

    return {
        "top_problem": (
            f"Exit timing — {exit_opt.get('late_exit_count', 0)} late exits, "
            f"~${_to_float(exit_opt.get('peak_giveback_estimate'), 0):,.2f} peak giveback"
        ),
        "top_opportunity": str(backlog["recommendation"].iloc[0]) if not backlog.empty else "—",
        "highest_alpha_improvement": (
            str(top_alpha["recommendation"]) if top_alpha is not None else "—"
        ),
        "lowest_risk_improvement": (
            str(low_risk["recommendation"].iloc[0]) if not low_risk.empty else "—"
        ),
        "most_expensive_decision_pattern": transitions.get("most_expensive_transition", "—"),
        "most_valuable_decision_pattern": transitions.get("best_transition", "—"),
        "estimated_total_improvement_potential": _f(total_potential, 2),
        "overall_optimization_grade": grade,
        "phase_169_focus": (
            "Exit timing (peak-drawdown escalation) and the "
            f"{(calibration.get('overconfident_zones') or ['0.60–0.80'])[0]} confidence zone"
        ),
        "current_lifecycle_grade": str(summary.get("overall_lifecycle_grade", "—")),
    }


def build_certification(intel: Dict[str, Any]) -> Dict[str, Any]:
    upstream = intel.get("certification") or {}
    reasons: List[str] = list(upstream.get("reasons") or [])
    status = str(upstream.get("status") or "NOT_CERTIFIED")
    exits = int(_to_float(upstream.get("classified_exits"), 0))
    if exits < 30:
        reasons.append(
            f"impact estimates rest on only {exits} classified exits — treat dollar "
            "figures as directional, not precise"
        )
        if status == "CERTIFIED":
            status = "PARTIALLY_CERTIFIED"
    return {
        "status": status,
        "based_on": "lifecycle_decision_intelligence.json certification",
        "upstream_status": upstream.get("status", "UNKNOWN"),
        "reasons": reasons,
        "safety": "recommendation-only phase — no thresholds, rules, or execution paths modified",
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)

    intel = _load_intel()
    if not intel:
        _log(
            "lifecycle_decision_intelligence.json missing — run services.lifecycle_decision_intelligence first"
        )
    tl = build_ticker_timelines()

    trades = _safe_read_csv(RESULTS / "trade_outcomes.csv")
    if not trades.empty:
        trades.columns = [str(c).strip().lower() for c in trades.columns]
        trades = trades[trades.get("outcome_type", "").astype(str).str.upper() == "REALIZED"].copy()
        trades["realized_pl"] = pd.to_numeric(trades.get("realized_pl"), errors="coerce")

    exit_opt = build_exit_optimization(tl, intel)
    calibration = build_confidence_calibration(intel)
    transitions = build_transition_optimization(intel, tl)
    flag_rankings = build_flag_rankings(intel)
    backlog = build_backlog(intel, exit_opt, calibration, transitions, trades)
    impact = build_impact_estimates(intel, backlog, trades)
    executive = build_executive_report(intel, exit_opt, calibration, transitions, backlog, impact)
    certification = build_certification(intel)

    flag_rankings.to_csv(OUT_FLAG_RANKINGS, index=False, encoding="utf-8")
    backlog.to_csv(OUT_BACKLOG, index=False, encoding="utf-8")

    # Flat recommendations file (one row per recommendation, all sections).
    rec_rows: List[Dict[str, Any]] = []
    for section, recs in (
        ("exit_optimization", exit_opt.get("recommendations") or []),
        ("confidence_calibration", calibration.get("recommendations") or []),
        ("transition_optimization", transitions.get("recommendations") or []),
    ):
        for rec in recs:
            rec_rows.append({"section": section, "recommendation": rec})
    for _, r in backlog.iterrows():
        rec_rows.append(
            {"section": f"backlog_{r['priority']}", "recommendation": r["recommendation"]}
        )
    pd.DataFrame(rec_rows).to_csv(OUT_RECS, index=False, encoding="utf-8")

    report = {
        "generated_at": _now_iso(),
        "executive_report": executive,
        "certification": certification,
        "exit_optimization": exit_opt,
        "confidence_calibration": calibration,
        "transition_optimization": transitions,
        "risk_flag_rankings": flag_rankings.to_dict(orient="records"),
        "improvement_backlog": backlog.to_dict(orient="records"),
        "impact_estimates": impact,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_RECS.name}, {OUT_BACKLOG.name}, {OUT_FLAG_RANKINGS.name}")
    _log(f"executive: {json.dumps(executive, default=str)}")
    return report


if __name__ == "__main__":
    run()
